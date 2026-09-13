"""Drive VIC with forecast precipitation and route to Hirakud inflow.

The hydrologic half of the Dong et al. (2025) framework, with VIC in place of
XAJ.  For every initialisation, every ensemble member and each precipitation
product (EC, EC-QM, EC-CNN), VIC runs 30 days forward from the saved state and
its runoff is routed with the RVIC unit hydrographs.

WHAT IS A FORECAST HERE AND WHAT IS NOT.  Only the forcing changes at the
initialisation date; the catchment state does not.  All three products and all
ten members branch from the SAME state (hydrology/vic/vic_states.py), so any
difference in simulated inflow is caused by the precipitation forecast and
nothing else.

FORCING VARIABLES.  VIC needs seven; a precipitation forecast supplies one.
    PREC       the forecast product under test
    AIR_TEMP   ECMWF t2m from the same reforecast
    SWDOWN     ECMWF ssrd
    PRESSURE   ECMWF msl
    WIND       sqrt(u10^2 + v10^2)
    LWDOWN     day-of-year climatology of the observed forcing
    VP         day-of-year climatology of the observed forcing
The last two have no forecast equivalent in the S2S archive.  Climatology is
the honest choice: inventing them would put unearned skill into the physics.
Dong handle the analogous gap by delta-correcting forecast temperature and are
silent on the rest.

ROUTING NEEDS THE PAST.  The RVIC convolution has 10 lags, so inflow on
forecast day 1 depends on runoff from the 10 days BEFORE initialisation.  Those
come from the observed-forcing run -- they are history at forecast time, not a
peek at the future.

Run:  python hydrology/vic/vic_forecast.py --product ec [--members 10] [--workers 4]
Out:  data/processed/vic_forecast_<product>.npz
"""

import argparse
import glob
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs import config as C

VIC_HOME = Path.home() / ".vic_mahanadi"
VIC_EXE = C.ROOT / "hydrology/vic/VIC/vic/drivers/classic/vic_classic.exe"
STATES = VIC_HOME / "states"
FORCING_SRC = VIC_HOME / "forcings"
LEAD = C.V2_LEAD_MAX
STEPS = 4                      # MODEL_STEPS_PER_DAY in global_param.txt
RECORD_START = pd.Timestamp("2003-01-01")
N_FORCE = 7                    # PREC AIR_TEMP SWDOWN LWDOWN PRESSURE VP WIND


def vic_cells() -> np.ndarray:
    return np.array([[float(x) for x in p.name.replace("data_", "").split("_")]
                     for p in sorted(FORCING_SRC.glob("data_*"))])


def load_observed_forcing(cells) -> np.ndarray:
    """(n_cell, n_step, 7) of the ERA5-driven forcing VIC was calibrated on."""
    out = []
    for la, lo in cells:
        out.append(np.loadtxt(FORCING_SRC / f"data_{la:.4f}_{lo:.4f}", dtype=np.float32))
    return np.stack(out)


def climatology(obs, dates) -> np.ndarray:
    """Day-of-year mean per cell per variable, for the fields with no forecast."""
    doy = dates.dayofyear.values
    clim = np.zeros((obs.shape[0], 367, obs.shape[2]), np.float32)
    for d in range(1, 367):
        m = doy == d
        if m.any():
            clim[:, d] = obs[:, m].mean(axis=1)
    return clim


def fine_index(cells):
    """Map each VIC cell to a fine-grid (row, col); nearest cell if outside."""
    flats = np.arange(C.FINE_LAT_MIN, C.FINE_LAT_MAX + C.FINE_RES / 2, C.FINE_RES)
    flons = np.arange(C.FINE_LON_MIN, C.FINE_LON_MAX + C.FINE_RES / 2, C.FINE_RES)
    return np.array([[int(np.abs(flats - la).argmin()), int(np.abs(flons - lo).argmin())]
                     for la, lo in cells])


def write_gp(path: Path, start, end, state: Path, forcing_prefix: Path, out_dir: Path):
    # FORCING1 and RESULT_DIR are REPLACED IN PLACE, never moved: VIC attributes
    # each FORCE_TYPE line to the forcing file declared above it, so appending
    # FORCING1 at the end makes the parser fail with "too many variables".
    # FORCEYEAR/MONTH/DAY say when the forcing FILE begins, which is separate
    # from the simulation window.  A forecast forcing file starts at the first
    # forecast day, not 2003-01-01, and VIC errors with "No data for the
    # specified time period" if this is left pointing at the observed record.
    replace = {"FORCING1": f"FORCING1    {forcing_prefix}",
               "RESULT_DIR": f"RESULT_DIR  {out_dir}",
               "FORCEYEAR": f"FORCEYEAR   {start.year}",
               "FORCEMONTH": f"FORCEMONTH  {start.month}",
               "FORCEDAY": f"FORCEDAY    {start.day}",
               "FORCESEC": "FORCESEC    0"}
    drop = {"STARTYEAR", "STARTMONTH", "STARTDAY", "STARTSEC", "ENDYEAR",
            "ENDMONTH", "ENDDAY", "INIT_STATE", "STATENAME", "STATEYEAR",
            "STATEMONTH", "STATEDAY", "STATESEC", "STATE_FORMAT", "LOG_DIR"}
    out = []
    for line in (VIC_HOME / "global_param.txt").read_text().splitlines():
        key = line.split()[0] if line.split() else ""
        if key in replace:
            out.append(replace.pop(key))
        elif key not in drop:
            out.append(line)
    out += list(replace.values())
    out += [f"STARTYEAR   {start.year}", f"STARTMONTH  {start.month}",
            f"STARTDAY    {start.day}", "STARTSEC    0",
            f"ENDYEAR     {end.year}", f"ENDMONTH    {end.month}",
            f"ENDDAY      {end.day}", f"INIT_STATE  {state}"]
    path.write_text("\n".join(out) + "\n")


def run_one(job):
    """One (init, member) VIC forecast -> daily runoff+baseflow per cell."""
    (init, prec, met, cells, clim_doy) = job
    init = pd.Timestamp(init)
    # VIC occasionally fails to open a file under parallel load -- observed once
    # in 5,320 runs, in make_in_and_outfiles, and not reproducible on a rerun of
    # the same (init, member).  It is a transient, so retry before giving up;
    # aborting the whole product for one of these threw away 1,800 good runs.
    for attempt in range(3):
        init_, flux, err = _run_once(init, prec, met, cells, clim_doy)
        if err is None:
            return init_, flux, None
        if attempt < 2:
            time.sleep(1.0 + attempt)
    return init_, None, f"after 3 attempts: {err}"


def _run_once(init, prec, met, cells, clim_doy):
    tmp = Path(tempfile.mkdtemp(prefix="vicfc_"))
    try:
        fdir = tmp / "forcing"; fdir.mkdir()
        odir = tmp / "out"; odir.mkdir()
        days = pd.date_range(init + pd.Timedelta(days=1), periods=LEAD)
        for ci, (la, lo) in enumerate(cells):
            block = np.repeat(clim_doy[ci], STEPS, axis=0).copy()   # (LEAD*STEPS, 7)
            block[:, 0] = np.repeat(prec[ci] / STEPS, STEPS)        # PREC split evenly
            for col, vals in met.items():
                block[:, col] = np.repeat(vals[ci], STEPS)
            np.savetxt(fdir / f"data_{la:.4f}_{lo:.4f}", block, fmt="%.4f")
        gp = tmp / "gp.txt"
        state = STATES / f"state_{init:%Y%m%d}_00000"
        write_gp(gp, days[0], days[-1], state, fdir / "data_", odir)
        r = subprocess.run([str(VIC_EXE), "-g", str(gp)], capture_output=True, text=True)
        if r.returncode != 0:
            # VIC prints its actual ERROR line FIRST and then a long stack
            # trace, so keeping only the tail throws away the diagnosis.
            out = r.stdout + r.stderr
            err_lines = [l for l in out.splitlines()
                         if "ERROR" in l or "error" in l.lower()]
            return init, None, (" | ".join(err_lines[:4]) or out[:600])
        flux = np.zeros((len(cells), LEAD), np.float32)
        for ci, (la, lo) in enumerate(cells):
            f = odir / f"fluxes_{la:.4f}_{lo:.4f}.txt"
            a = np.loadtxt(f, skiprows=3)
            flux[ci] = (a[:LEAD, 3] + a[:LEAD, 4])       # OUT_RUNOFF + OUT_BASEFLOW
        return init, flux, None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def met_at_cells(coarse, names, idx, want):
    """Forecast met variables interpolated to the fine grid, sampled at VIC cells."""
    import torch
    hw = int((C.FINE_LAT_MAX - C.FINE_LAT_MIN) / C.FINE_RES) + 1
    out = {}
    for col, var, fn in want:
        a = coarse[:, :, names.index(var)] if var in names else None
        if a is None:
            continue
        n, m = a.shape[:2]
        f = torch.nn.functional.interpolate(
            torch.from_numpy(a.reshape(-1, 1, *a.shape[2:])), size=(hw, hw),
            mode="bilinear", align_corners=False).reshape(n, m, hw, hw).numpy()
        out[col] = fn(f[:, :, idx[:, 0], idx[:, 1]])          # (n_key, n_mem, n_cell)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", choices=["ec", "ec_qm", "ec_cnn"], required=True)
    ap.add_argument("--members", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N inits (testing)")
    args = ap.parse_args()

    from cnn.loyo_percell_v2 import load_members
    cells = vic_cells()
    idx = fine_index(cells)
    print(f"{len(cells)} VIC cells; loading archive ...", flush=True)
    coarse, inits, leads, names = load_members(args.members)

    # THIRD PLACE THIS BITES.  load_members walks sorted(glob) -- month/day
    # filename outer, year inner -- but `prec` below is read from files built in
    # dataset order (year outer).  `keys` is derived from load_members, so the
    # boolean mask `sel` would select glob-ordered rows out of a dataset-ordered
    # array: every forecast driven by another date's rainfall.  The two orders
    # coincided on the 2004-2014 build and diverge on 2004-2022.  Reorder the
    # member arrays into dataset order once, here, so everything downstream
    # shares one indexing convention.
    _dset = next((q for q in (C.PROCESSED / "downscaling_v2_full.npz",
                              C.PROCESSED / "downscaling_v2.npz") if q.exists()),
                 C.PROCESSED / "downscaling_v2.npz")
    _d = np.load(_dset, allow_pickle=True)
    _want = list(zip(_d["init"].astype("datetime64[D]").astype(str).tolist(),
                     _d["lead"].tolist()))
    _have = {k: i for i, k in enumerate(
        zip(inits.values.astype("datetime64[D]").astype(str).tolist(),
            leads.tolist()))}
    _missing = [k for k in _want if k not in _have]
    if _missing:
        sys.exit(f"{len(_missing)} of {len(_want)} (init, lead) keys are in "
                 f"{_dset.name} but not in the member archive")
    _order = np.array([_have[k] for k in _want], np.int64)
    coarse = coarse[_order]
    inits = pd.DatetimeIndex(inits.values[_order])
    leads = leads[_order]
    print(f"  aligned to {_dset.name} on (init, lead): {len(_order)} rows, "
          f"{int((_order != np.arange(len(_order))).sum())} reordered", flush=True)

    keys = pd.DataFrame({"init": inits, "lead": leads})

    # ---- precipitation under test, at the VIC cells
    if args.product == "ec_cnn":
        z = np.load(C.PROCESSED / "percell_v2_oof.npz", allow_pickle=True)
        cat_cells = z["cells"]
        lookup = {(int(r), int(c)): j for j, (r, c) in enumerate(cat_cells)}
        cols = [lookup.get((r, c)) for r, c in idx]
        src = z["oof_members"]                                  # (n_key, mem, 224)
        prec = np.stack([src[:, :, j] if j is not None
                         else np.zeros(src.shape[:2], np.float32) for j in cols], -1)
    else:
        z = np.load(next((q for q in (C.PROCESSED / "ec_qm_v2_full.npz",
                                      C.PROCESSED / "ec_qm_v2.npz") if q.exists()),
                         C.PROCESSED / "ec_qm_v2.npz"), allow_pickle=True)
        cat_cells = z["cells"]
        lookup = {(int(r), int(c)): j for j, (r, c) in enumerate(cat_cells)}
        cols = [lookup.get((r, c)) for r, c in idx]
        src = z["ec" if args.product == "ec" else "ec_qm"]
        prec = np.stack([src[:, :, j] if j is not None
                         else np.zeros(src.shape[:2], np.float32) for j in cols], -1)
    miss = sum(1 for c in cols if c is None)
    print(f"  precipitation {prec.shape}; {miss} VIC cells outside the scored grid "
          f"(western headwaters) filled from the nearest column", flush=True)

    # ---- other forcing: forecast where available, climatology otherwise
    met = met_at_cells(coarse, names, idx, [
        (1, "t2m", lambda x: x - 273.15),                       # K -> degC
        (2, "ssrd", lambda x: x),                               # already W/m2
        (4, "msl", lambda x: x / 1000.0),                       # Pa -> kPa
    ])
    u = met_at_cells(coarse, names, idx, [(6, "u10", lambda x: x)]).get(6)
    v = met_at_cells(coarse, names, idx, [(6, "v10", lambda x: x)]).get(6)
    if u is not None and v is not None:
        met[6] = np.sqrt(u ** 2 + v ** 2)

    # Length comes from the forcing itself.  This was hard-coded as 17528 steps
    # (4,382 days, the 2003-2014 record); the forcing now runs to 2022 and a
    # fixed length silently mismatches the climatology mask.
    obs = load_observed_forcing(cells)                          # (n_cell, n_step, 7)
    obs_daily = obs.reshape(len(cells), -1, STEPS, N_FORCE).mean(2)
    obs_dates = pd.date_range(RECORD_START, periods=obs_daily.shape[1], freq="D")
    clim = climatology(obs_daily, obs_dates)                    # (n_cell, 367, 7)

    uniq = sorted(set(inits))
    if args.limit:
        uniq = uniq[:args.limit]
    print(f"{len(uniq)} inits x {args.members} members = "
          f"{len(uniq)*args.members} VIC runs", flush=True)

    flux = np.zeros((len(uniq), args.members, len(cells), LEAD), np.float32)
    t0 = time.time()
    jobs = []
    for ki, init in enumerate(uniq):
        sel = (keys["init"] == init).values
        order = np.argsort(keys["lead"].values[sel])
        doy = pd.date_range(init + pd.Timedelta(days=1), periods=LEAD).dayofyear.values
        for mi in range(args.members):
            p = prec[sel][order][:, mi, :].T                    # (n_cell, LEAD)
            m = {c: met[c][sel][order][:, mi, :].T for c in met}
            jobs.append((ki, mi, (init, p, m, cells, clim[:, doy].transpose(0, 1, 2))))

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, j[2]): (j[0], j[1]) for j in jobs}
        done = 0
        for f in as_completed(futs):
            ki, mi = futs[f]
            _, fl, err = f.result()
            if err:
                print(f"  FAILED init#{ki} ({uniq[ki]:%Y-%m-%d}) member {mi}: {err}")
                sys.exit(1)
            flux[ki, mi] = fl
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)

    out = C.PROCESSED / f"vic_forecast_{args.product}.npz"
    np.savez_compressed(out, flux=flux, inits=np.array([str(u) for u in uniq]),
                        cells=cells, lead=np.arange(1, LEAD + 1))
    print(f"\nwrote {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
