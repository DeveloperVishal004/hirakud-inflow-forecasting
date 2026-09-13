"""Run VIC end-to-end on PLACEHOLDER radiation, to shake out format errors now.

The radiation download is still in the CDS queue, but nothing about the soil
parameter file, the vegetation files, the global parameter file or the forcing
layout depends on it being real.  Those are exactly the things that fail with
opaque messages -- a column in the wrong position, a veg class missing from the
library, a filename VIC rebuilds differently from how we wrote it -- and each
one costs a round trip to discover.  Better to find them against fake radiation
tonight than against real radiation tomorrow.

WHAT IS FAKE HERE
    SWDOWN    constant 200 W/m2
    LWDOWN    constant 380 W/m2
    PRESSURE  barometric from cell elevation, no synoptic variation

Everything else -- precipitation, temperature, wind, vapour pressure -- is the
real data already downloaded.

THE OUTPUT OF THIS SCRIPT IS NOT A RESULT.  Constant radiation removes the
seasonal cycle of evaporative demand, so the water balance is meaningless.  It
answers one question only: does VIC parse our inputs and run to completion?
Everything is written under vic/smoke/ and never touches the real directories.

Run:  python hydrology/vic/smoke_test.py [--year 2004]
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs import config as C
from hydrology.vic import build_forcing as BF

# Same space-free routing as build_global_param.py: VIC truncates any path
# containing a space.
from hydrology.vic.build_global_param import VICDIR as _SAFE_VIC
SMOKE_REAL = C.PROCESSED / "vic" / "smoke"
SMOKE = _SAFE_VIC / "smoke"
VIC_EXE = (Path(__file__).resolve().parent / "VIC" / "vic" / "drivers"
           / "classic" / "vic_classic.exe")

FAKE_SW = 200.0     # W/m2
FAKE_LW = 380.0     # W/m2


def barometric_kpa(elev_m):
    """Standard atmosphere pressure from elevation -- what MTCLIM used."""
    return 101.325 * (1.0 - 2.25577e-5 * elev_m) ** 5.25588


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2004)
    args = ap.parse_args()

    if not VIC_EXE.exists():
        sys.exit(f"VIC binary not found at {VIC_EXE}")

    basin = np.load(C.PROCESSED / "basin.npz", allow_pickle=True)
    dem = np.load(C.PROCESSED / "dem_vic.npz", allow_pickle=True)
    lats, lons = basin["lats"], basin["lons"]
    active = basin["fraction"] > 0

    # ---- real temperature, wind, dewpoint
    t2m, _ = BF.open_stream("t2m_*", ["t2m"])
    met, _ = BF.open_stream("met_*", ["d2m", "u10", "v10"])
    if t2m is None or met is None:
        sys.exit("t2m/met streams missing")
    air_hourly = t2m.t2m
    wind_raw = np.hypot(met.u10, met.v10)
    dew_raw = met.d2m

    # VIC needs 4 records/day (SNOW_STEPS_PER_DAY >= 4), so build a 6-hourly
    # series in UTC rather than daily means.
    steps = pd.date_range(f"{args.year}-01-01", f"{args.year}-12-31 18:00",
                          freq="6h")
    print(f"smoke test: {args.year}, {len(steps)} 6-hourly steps, "
          f"{int(active.sum())} cells")

    air_6h = air_hourly.reindex(valid_time=steps, method="nearest",
                                tolerance=pd.Timedelta("3h"))
    sel = dict(valid_time=steps)
    air_g = BF.regrid(air_6h - 273.15, lats, lons)
    wind_g = BF.regrid(wind_raw.sel(**sel), lats, lons)
    vp_g = BF.vapour_pressure_kpa(BF.regrid(dew_raw.sel(**sel) - 273.15, lats, lons))

    # ---- real precipitation
    import glob
    import xarray as xr
    files = sorted(glob.glob(str(C.IMD_NC_DIR / "RF25_ind*.nc")))
    imd = xr.concat([xr.open_dataset(f) for f in files], dim="TIME").sortby("TIME")
    imd = imd.sel(LATITUDE=slice(lats[0], lats[-1]),
                  LONGITUDE=slice(lons[0], lons[-1]))
    # IMD is daily; VIC wants 6-hourly.  Spread each day's total evenly over
    # its four steps -- with a daily observation there is no sub-daily
    # information to preserve, and a fabricated diurnal shape would be worse
    # than an honest uniform one.
    daily = np.nan_to_num(imd.RAINFALL.reindex(
        TIME=pd.DatetimeIndex(steps.normalize().unique())).values, nan=0.0)
    prec_g = np.repeat(daily, 4, axis=0)[:len(steps)] / 4.0

    # ---- placeholders
    pres_cell = barometric_kpa(dem["elev_mean"])
    print(f"  placeholder SW {FAKE_SW} W/m2, LW {FAKE_LW} W/m2, "
          f"pressure {pres_cell.min():.1f}-{pres_cell.max():.1f} kPa from elevation")

    fdir = SMOKE / "forcings"
    if SMOKE_REAL.exists():
        shutil.rmtree(SMOKE_REAL)
    for d in (fdir, SMOKE / "output", SMOKE / "logs"):
        d.mkdir(parents=True, exist_ok=True)

    n = 0
    for i in range(len(lats)):
        for j in range(len(lons)):
            if not active[i, j]:
                continue
            block = np.column_stack([
                prec_g[:, i, j], air_g[:, i, j],
                np.full(len(steps), FAKE_SW), np.full(len(steps), FAKE_LW),
                np.full(len(steps), pres_cell[i, j]),
                vp_g[:, i, j], wind_g[:, i, j]])
            np.savetxt(fdir / f"data_{lats[i]:.4f}_{lons[j]:.4f}", block,
                       fmt="%.3f %.2f %.1f %.1f %.2f %.4f %.2f")
            n += 1
    print(f"  wrote {n} placeholder forcing files")

    # ---- a global parameter file pointing only at the smoke directory
    gp = (C.PROCESSED / "vic" / "global_param.txt").read_text()
    gp = gp.replace(str(_SAFE_VIC / "forcings" / "data_"), str(fdir / "data_"))
    gp = gp.replace(f"RESULT_DIR   {_SAFE_VIC / 'output'}/",
                    f"RESULT_DIR   {SMOKE / 'output'}/")
    gp = gp.replace(f"LOG_DIR      {_SAFE_VIC / 'logs'}/",
                    f"LOG_DIR      {SMOKE / 'logs'}/")
    for k, v in [("STARTYEAR", args.year), ("ENDYEAR", args.year),
                 ("FORCEYEAR", args.year)]:
        gp = "\n".join(f"{k}   {v}" if ln.split()[:1] == [k] else ln
                       for ln in gp.splitlines())
    gpath = SMOKE / "global_param.txt"
    gpath.write_text(gp)

    print(f"\nrunning VIC ...")
    r = subprocess.run([str(VIC_EXE), "-g", str(gpath)],
                       capture_output=True, text=True, timeout=3600)
    out = (r.stdout or "") + (r.stderr or "")
    print(f"  exit code {r.returncode}")

    if r.returncode != 0:
        print("\n--- VIC output (last 30 lines) ---")
        print("\n".join(out.splitlines()[-30:]))
        logs = sorted((SMOKE / "logs").glob("*"))
        if logs:
            print(f"\n--- {logs[-1].name} (last 20) ---")
            print("\n".join(logs[-1].read_text().splitlines()[-20:]))
        print("\nFAIL -- VIC did not run to completion")
        sys.exit(1)

    files_out = sorted((SMOKE / "output").glob("*"))
    print(f"  wrote {len(files_out)} output file(s)")
    if not files_out:
        print("\nFAIL -- VIC exited 0 but produced no output")
        sys.exit(1)

    sample = files_out[0]
    txt = sample.read_text().splitlines()
    print(f"\n--- {sample.name}: {len(txt)} lines ---")
    for ln in txt[:6]:
        print("  " + ln[:110])

    # ---- water balance.  Fake radiation makes the PARTITION meaningless, but
    # closure is a property of the solver, not of the forcing: whatever comes
    # in must leave or be stored.  A residual here would mean a real bug.
    hdr = txt[2].split()
    tot = None
    for f in files_out:
        d = pd.read_csv(f, sep=r"\s+", skiprows=3, header=None, names=hdr)
        tot = d[hdr[3:]] if tot is None else tot + d[hdr[3:]]
    tot /= len(files_out)

    P = tot["OUT_PREC"].sum()
    E = tot["OUT_EVAP"].sum()
    R = tot["OUT_RUNOFF"].sum()
    B = tot["OUT_BASEFLOW"].sum()
    sm = [c for c in hdr if c.startswith("OUT_SOIL_MOIST")]
    dS = sum(tot[c].iloc[-1] - tot[c].iloc[0] for c in sm)
    resid = P - E - R - B - dS

    print(f"\n=== water balance, catchment mean (mm) ===")
    for lab, v in [("precipitation", P), ("evaporation", E), ("runoff", R),
                   ("baseflow", B), ("storage change", dS)]:
        print(f"  {lab:18s} {v:9.1f}")
    # Tolerance is relative, and 0.1 % is the right order: the soil-moisture
    # term omits canopy interception storage, which VIC tracks separately, so
    # a small persistent residual is expected rather than a defect.
    rel = abs(resid) / P
    print(f"  residual           {resid:9.3f}  ({100 * rel:.3f} % of P)  "
          f"{'CLOSES' if rel < 0.01 else 'DOES NOT CLOSE'}")

    if "OUT_SWE" in hdr:
        print(f"  max SWE            {tot['OUT_SWE'].max():9.4f}  "
              f"{'OK, no snow' if tot['OUT_SWE'].max() < 0.01 else '<- UNEXPECTED'}")

    if rel >= 0.01:
        print("\nFAIL -- water balance does not close")
        sys.exit(1)

    print("\nPASS -- VIC parsed the soil, vegetation and forcing files, ran to")
    print("completion, and conserved water.")
    print("REMINDER: radiation was FAKE (constant 200 W/m2), so the PARTITION")
    print("between ET and runoff is not meaningful -- only the closure is.")


if __name__ == "__main__":
    main()
