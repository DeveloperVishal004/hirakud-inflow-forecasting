"""Catchment-mean daily meteorology, 2003-2014, for the FutureTST encoder.

Ambika et al. (2025) train FutureTST on *observed* meteorology (Daymet:
precipitation, shortwave radiation, max/min air temperature, vapour pressure)
and bring forecast meteorology in only at real-time evaluation.  The equivalent
observed record here is the ERA5-Land forcing already assembled for VIC, which
carries the same variables at 6-hourly resolution over the same grid the basin
mask is defined on.

Aggregation follows the physics rather than a single rule: precipitation is
summed over the day, radiation and state variables are averaged, and air
temperature additionally yields a daily max and min because the diurnal range
carries evaporative-demand information a daily mean destroys.

Cells are weighted by basin fraction, so the 55 % of the bounding box that
drains elsewhere does not dilute the catchment signal -- the same weighting
route_and_evaluate.py uses for runoff volume.

Run:  python preprocessing/build_daily_met.py
Out:  data/processed/daily_catchment_met.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

SUM_VARS = ["prec"]
MEAN_VARS = ["air_temp", "wind", "swdown", "lwdown", "pressure", "vp"]


def main() -> None:
    src = C.PROCESSED / "vic" / "forcing_daily.nc"
    if not src.exists():
        sys.exit(f"missing {src} -- run hydrology/vic/build_forcing.py")
    ds = xr.open_dataset(src)

    basin = np.load(C.PROCESSED / "basin.npz", allow_pickle=True)
    frac = basin["fraction"]
    if not (np.allclose(basin["lats"], ds.lat.values)
            and np.allclose(basin["lons"], ds.lon.values)):
        sys.exit("basin mask and forcing grid disagree -- refusing to average")
    w = xr.DataArray(frac, coords={"lat": ds.lat, "lon": ds.lon}, dims=["lat", "lon"])
    wsum = float(w.sum())
    print(f"weighting {int((frac > 0).sum())} cells, total basin fraction {wsum:.2f}")

    # Catchment mean first, then resample: both are linear, and doing the
    # spatial reduction first keeps the daily resample cheap.
    cat = (ds[SUM_VARS + MEAN_VARS] * w).sum(dim=["lat", "lon"]) / wsum
    steps_per_day = int(round(len(ds.time) / len(np.unique(ds.time.dt.floor("D")))))
    print(f"forcing is {24 // steps_per_day}-hourly ({steps_per_day} steps/day)")

    out = pd.DataFrame(index=pd.DatetimeIndex(
        np.unique(cat.time.dt.floor("D").values), name="date"))
    day = cat.time.dt.floor("D")
    for v in SUM_VARS:
        # prec is a rate (mm per step) in the VIC forcing, so a day is the sum.
        out[v] = cat[v].groupby(day).sum().to_series()
    for v in MEAN_VARS:
        out[v] = cat[v].groupby(day).mean().to_series()
    # Diurnal range: the paper's tmax/tmin, recovered from the sub-daily series.
    out["air_temp_max"] = cat["air_temp"].groupby(day).max().to_series()
    out["air_temp_min"] = cat["air_temp"].groupby(day).min().to_series()

    out = out.reset_index()
    n_bad = int(out.isna().sum().sum())
    if n_bad:
        sys.exit(f"{n_bad} NaNs in the aggregated met -- investigate before use")

    out.to_parquet(C.PROCESSED / "daily_catchment_met.parquet", index=False)
    print(f"\n{len(out):,} days  {out.date.min().date()} -> {out.date.max().date()}")
    print(out.describe().T[["mean", "min", "max"]].round(2).to_string())
    print(f"\nwrote {C.PROCESSED / 'daily_catchment_met.parquet'}")


if __name__ == "__main__":
    main()
