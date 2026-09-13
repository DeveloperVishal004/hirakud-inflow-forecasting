"""Route VIC runoff to Hirakud with the RVIC unit hydrographs, and score it.

RVIC's own `convolution` step expects VIC image-driver NetCDF output; the
classic driver writes ASCII per cell.  Rather than convert formats, this
applies the convolution directly -- it is exactly what RVIC does, and doing it
here keeps the arithmetic inspectable:

    Q(t) = sum over source cells s of  sum over lag k of
               UH[k, s] * runoff_volume[s, t - k]

The unit hydrographs come from the generated RVIC parameter file, so the
network, travel times and mass conservation are RVIC's, not a reimplementation.

Volume conversion: VIC reports runoff and baseflow as depths (mm) over the
cell.  Only the part of a cell inside the basin contributes, so depth is
multiplied by area x basin_fraction -- the same fractions build_basin.py
derived at 15 arc-sec.  Using whole cells would inflate discharge by the
55 % of the bounding box that drains elsewhere.

Run:  python hydrology/vic/route_and_evaluate.py
Out:  results/metrics/vic_inflow.json, results/figures/vic_hydrograph.png
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs import config as C
from hydrology.calibrate_xaj import load_series, nse, kge

VICOUT = C.PROCESSED / "vic" / "output"
PRM = glob.glob(str(Path.home() / ".vic_mahanadi" / "rvic" / "case" / "params"
                    / "*.prm.*.nc"))
R_EARTH = 6371.0072


def cell_area_km2(lat, res=C.FINE_RES):
    return (R_EARTH ** 2 * np.radians(res)
            * (np.sin(np.radians(lat + res / 2)) - np.sin(np.radians(lat - res / 2))))


def main() -> None:
    import xarray as xr

    if not PRM:
        sys.exit("no RVIC parameter file -- run build_rvic_inputs.py --run")
    prm = xr.open_dataset(sorted(PRM)[-1])
    uh = prm["unit_hydrograph"].values[:, :, 0]        # (lags, sources)
    slat = prm["source_lat"].values
    slon = prm["source_lon"].values
    n_lag, n_src = uh.shape

    basin = np.load(C.PROCESSED / "basin.npz", allow_pickle=True)
    frac = basin["fraction"]
    blats, blons = basin["lats"], basin["lons"]

    # ---- load VIC output for every source cell, in the parameter file's order
    files = {}
    for f in glob.glob(str(VICOUT / "fluxes_*.txt")):
        parts = Path(f).stem.split("_")
        files[(round(float(parts[1]), 4), round(float(parts[2]), 4))] = f
    if not files:
        sys.exit(f"no VIC output in {VICOUT} -- run vic_classic.exe first")

    hdr = open(next(iter(files.values()))).read().splitlines()[2].split()
    vol = None
    missing = 0
    for s in range(n_src):
        key = (round(float(slat[s]), 4), round(float(slon[s]), 4))
        if key not in files:
            missing += 1
            continue
        d = pd.read_csv(files[key], sep=r"\s+", skiprows=3, header=None, names=hdr)
        if vol is None:
            dates = pd.to_datetime(dict(year=d.YEAR, month=d.MONTH, day=d.DAY))
            vol = np.zeros((len(d), n_src))
        i = int(np.argmin(np.abs(blats - slat[s])))
        j = int(np.argmin(np.abs(blons - slon[s])))
        # mm/day over the contributing area -> m3/s
        a_m2 = cell_area_km2(blats[i]) * 1e6 * frac[i, j]
        vol[:, s] = (d.OUT_RUNOFF + d.OUT_BASEFLOW).values * 1e-3 * a_m2 / 86400.0
    if missing:
        sys.exit(f"{missing} source cells had no VIC output")

    print(f"routing {n_src} source cells, {len(vol):,} days, {n_lag} lags")

    # ---- convolve
    q = np.zeros(len(vol))
    for k in range(n_lag):
        q[k:] += (vol[:len(vol) - k] * uh[k][None, :]).sum(axis=1)
    sim = pd.Series(q, index=dates, name="q_vic")

    # Persist the routed series and the catchment-mean runoff depth: the plan's
    # Module 7 feeds "past VIC simulated runoff" and "past VIC simulated inflow"
    # into FutureTST, which needs them as a daily time series rather than as a
    # summary metric.
    runoff_mm = vol.sum(axis=1) * 86400.0 / (cell_area_km2(blats.mean()) * 1e6 *
                                             frac.sum()) * 1e3
    pd.DataFrame({"date": dates, "q_vic": q, "runoff_vic_mm": runoff_mm}).to_parquet(
        C.PROCESSED / "vic_daily.parquet", index=False)
    print(f"wrote {C.PROCESSED / 'vic_daily.parquet'}")

    # ---- compare with observed inflow
    # HONOUR `inflow_valid`.  load_series() stores missing gauge days as 0.0 and
    # flags them separately; scoring against those zeros marks a day the gauge
    # never reported as a day the model over-predicted.  Measured on the
    # calibration years alone: 213 zero-valued invalid days, all in Jan-Jun.
    _s = load_series()
    obs = _s["inflow"].where(_s["valid"])
    df = pd.DataFrame({"sim": sim}).join(obs.rename("obs")).dropna()
    mon = df[df.index.month.isin(C.INIT_MONTHS)]
    print(f"\n=== VIC + RVIC vs observed inflow ===")
    print(f"  overlapping days {len(df):,}   monsoon days {len(mon):,}")

    # The project's split, reported explicitly rather than left implicit.  VIC's
    # soil parameters are fitted on 2004-2011 ONLY, so those years are partly
    # in-sample; 2012-2014 is validation; 2015-2022 was never seen by the
    # calibration in any form and is the honest out-of-sample test.  Note
    # 2015-2022 is monsoon-only in the inflow record, so its "all" and "monsoon"
    # rows are nearly the same sample.
    SPLITS = [("cal 2004-2011", range(2004, 2012)),
              ("val 2012-2014", range(2012, 2015)),
              ("heldout 2015-2022", range(2015, 2023))]
    periods = [("all", df), ("monsoon", mon)]
    for lab, yrs in SPLITS:
        ys = set(yrs)
        periods.append((lab, mon[mon.index.year.isin(ys)]))

    out = {}
    for label, d_ in periods:
        if len(d_) < 30:
            print(f"  {label:18s} skipped -- only {len(d_)} days")
            continue
        r = {"NSE": float(nse(d_.sim.values, d_.obs.values)),
             "KGE": float(kge(d_.sim.values, d_.obs.values)),
             "corr": float(np.corrcoef(d_.sim, d_.obs)[0, 1]),
             "bias_pct": float(100 * (d_.sim.mean() - d_.obs.mean()) / d_.obs.mean()),
             "sim_mean": float(d_.sim.mean()), "obs_mean": float(d_.obs.mean())}
        r["n_days"] = int(len(d_))
        out[label] = r
        print(f"  {label:18s} NSE {r['NSE']:+7.3f}  KGE {r['KGE']:+7.3f}  "
              f"corr {r['corr']:+.3f}  bias {r['bias_pct']:+6.1f} %  "
              f"(sim {r['sim_mean']:6.0f} vs obs {r['obs_mean']:6.0f} m3/s, "
              f"n={r['n_days']:,})")

    # This used to assert the run was uncalibrated.  Say what is actually in
    # the soil file instead of hard-coding a claim that goes stale.
    cal = Path.home() / ".vic_mahanadi/soil_param_calibrated.txt"
    soil = Path.home() / ".vic_mahanadi/soil_param.txt"
    is_cal = cal.exists() and soil.exists() and cal.read_bytes() == soil.read_bytes()
    print(f"\n  soil parameters: {'CALIBRATED' if is_cal else 'UNCALIBRATED (defaults)'}")
    if not is_cal:
        print("  Correlation is the number to read here -- it says whether the")
        print("  TIMING is right; bias and NSE are what calibration then fixes.")

    C.METRICS.mkdir(parents=True, exist_ok=True)
    (C.METRICS / "vic_inflow.json").write_text(json.dumps(out, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    yr = df[df.index.year == C.TEST_YEARS[0]]
    fig, ax = plt.subplots(2, 1, figsize=(13, 8))
    ax[0].plot(df.index, df.obs, lw=0.6, label="observed")
    ax[0].plot(df.index, df.sim, lw=0.6, label="VIC + RVIC (uncalibrated)")
    ax[0].set_title(f"Hirakud inflow, {df.index[0]:%Y}-{df.index[-1]:%Y}  "
                    f"NSE {out['all']['NSE']:+.2f}")
    ax[1].plot(yr.index, yr.obs, lw=1.2, label="observed")
    ax[1].plot(yr.index, yr.sim, lw=1.2, label="VIC + RVIC")
    ax[1].set_title(f"{C.TEST_YEARS[0]} detail")
    for a in ax:
        a.set_ylabel("m3/s"); a.legend(); a.grid(alpha=0.3)
    fig.tight_layout()
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / "vic_hydrograph.png", dpi=110)
    print(f"\nwrote {C.METRICS / 'vic_inflow.json'}")
    print(f"wrote {C.FIGURES / 'vic_hydrograph.png'}")


if __name__ == "__main__":
    main()
