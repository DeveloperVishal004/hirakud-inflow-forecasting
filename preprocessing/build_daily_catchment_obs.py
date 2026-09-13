"""Daily observed catchment-mean rainfall from IMD, for EVERY day of the year.

`catchment_rainfall.parquet` only covers monsoon valid dates, because that is
all the ECMWF reforecast reaches.  That leaves the antecedent-rainfall windows
undefined for the first initialisation of each season (3 June): the 3-30 day
lookback falls in May, before the record starts.

Those are exactly the rows where antecedent wetness matters most.  A 3 June
forecast is issued into a catchment that has been dry for months, so the same
rainfall produces far less inflow than in August -- the Day A / Day B contrast
the whole project rationale rests on.  Imputing it away would erase the signal.

IMD is full-year, so the fix is to read the observed field directly rather than
inherit the reforecast's seasonal window.

Run:  python preprocessing/build_daily_catchment_obs.py
Out:  data/processed/daily_catchment_obs.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import re
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

RENAME = {"LATITUDE": "lat", "LONGITUDE": "lon", "TIME": "time", "RAINFALL": "rain"}


def main() -> None:
    t = np.load(C.PROCESSED / "fine_target.npz")
    flats, flons, catchment = t["fine_lats"], t["fine_lons"], t["catchment"]

    files = sorted(C.IMD_NC_DIR.glob("*.nc"))
    if not files:
        sys.exit(f"no IMD NetCDFs in {C.IMD_NC_DIR}")

    # One year either side of the inflow record, so a 30-day lookback from the
    # first usable init date is still fully covered.
    # Cover every IMD year on disk, not just the S2S v1 window.  The inflow
    # record now runs to 2022 and the S2S archive to 2022, so the observed
    # rainfall must too -- otherwise the extended years have a target but no
    # antecedent-rainfall predictors.  --years overrides.
    # FIXED 2026-09-06: the default upper bound was C.YEAR_MAX, which is frozen
    # at 2014 for the v1 archive.  That contradicted the comment directly above
    # and silently truncated this file to 2004-2014 on any rebuild -- leaving
    # 2015-2022 with an inflow target but no observed rainfall.  The default is
    # now the last year actually present on disk, which is what the comment
    # always said.  Both bounds remain overridable by environment variable.
    import os
    _disk = sorted({int(m.group()) for f in files
                    for m in [re.search(r"(19|20)\d{2}", f.name)] if m})
    if not _disk:
        sys.exit(f"cannot read years from the IMD filenames in {C.IMD_NC_DIR}")
    lo = int(os.environ.get("IMD_YEAR_MIN", C.YEAR_MIN - 1))
    hi = int(os.environ.get("IMD_YEAR_MAX", _disk[-1]))
    years = set(range(lo, hi + 1))
    print(f"  IMD years requested: {lo}-{hi}")

    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            ds = ds.rename({k: v for k, v in RENAME.items() if k in ds.variables or k in ds.dims})
            if not pd.to_datetime(ds["time"].values).year.isin(years).any():
                continue
            parts.append(ds["rain"].sel(lat=flats, lon=flons, method="nearest").load())

    rain = xr.concat(parts, dim="time").sortby("time")
    rain = rain.where(~(rain < 0.0))

    vals = rain.transpose("time", "lat", "lon").values.astype(np.float32)
    times = pd.to_datetime(rain["time"].values)

    # Mean over observed catchment cells only; a cell with no observation must
    # not be counted as zero rainfall.
    sel = vals[:, catchment]
    observed = np.isfinite(sel)
    total = np.where(observed, sel, 0.0).sum(axis=1)
    count = observed.sum(axis=1)
    mean_rain = np.where(count > 0, total / np.maximum(count, 1), np.nan)

    df = pd.DataFrame({"date": times, "rain_obs_daily": mean_rain,
                       "n_cells": count}).dropna(subset=["rain_obs_daily"])
    out = C.PROCESSED / "daily_catchment_obs.parquet"
    df.to_parquet(out, index=False)

    print(f"wrote {out}   {len(df):,} days  "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  catchment cells averaged: {int(df['n_cells'].max())}")
    m = df.set_index("date")["rain_obs_daily"]
    print("\n  monthly mean rainfall (mm/d) -- monsoon onset should be sharp:")
    print("   ", m.groupby(m.index.month).mean().round(2).to_dict())


if __name__ == "__main__":
    main()
