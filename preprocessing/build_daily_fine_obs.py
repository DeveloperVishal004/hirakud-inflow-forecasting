"""Daily per-cell IMD rainfall for every calendar day -- the lag-feature source.

The original notebook fed the model `rain_lag_1..5`, but computed them relative
to VALID time: at lead 17, "lag 1" was rain observed 16 days after the forecast
was issued -- information that does not exist at issue time.  That leak is why
those features were dropped when the pipeline was rebuilt.

The legal version is lags relative to INITIALISATION time: rainfall observed at
the target cell on the days up to the issue date.  It is real signal, not just
persistence -- monsoon rainfall is autocorrelated over 1-4 weeks through
active/break spells, so "how wet has this cell been lately" is informative even
at long leads.  This script materialises the daily 0.25 deg IMD field the lag
lookup needs, full-year, one year before YEAR_MIN so early-June inits have a
complete lookback.

Run:  python preprocessing/build_daily_fine_obs.py
Out:  data/processed/daily_fine_obs.npz
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

RENAME = {"LATITUDE": "lat", "LONGITUDE": "lon", "TIME": "time", "RAINFALL": "rain"}


def main() -> None:
    t = np.load(C.PROCESSED / "fine_target.npz")
    flats, flons = t["fine_lats"], t["fine_lons"]

    years = set(range(C.YEAR_MIN - 1, C.YEAR_MAX + 1))
    parts = []
    for f in sorted(C.IMD_NC_DIR.glob("*.nc")):
        with xr.open_dataset(f) as ds:
            ds = ds.rename({k: v for k, v in RENAME.items() if k in ds.variables or k in ds.dims})
            if not pd.to_datetime(ds["time"].values).year.isin(years).any():
                continue
            parts.append(ds["rain"].sel(lat=flats, lon=flons, method="nearest").load())

    rain = xr.concat(parts, dim="time").sortby("time")
    rain = rain.where(~(rain < 0.0))
    vals = rain.transpose("time", "lat", "lon").values.astype(np.float32)
    dates = pd.to_datetime(rain["time"].values).normalize()

    # Contiguous daily axis so date -> row index is pure arithmetic.
    full = pd.date_range(dates.min(), dates.max(), freq="D")
    arr = np.full((len(full),) + vals.shape[1:], np.nan, np.float32)
    arr[full.get_indexer(dates)] = vals

    np.savez_compressed(
        C.PROCESSED / "daily_fine_obs.npz",
        obs=np.nan_to_num(arr, nan=0.0),   # paired with `observed`; never used alone
        observed=np.isfinite(arr),
        start_date=np.datetime64(full[0], "D"),
        fine_lats=flats, fine_lons=flons,
    )
    print(f"wrote {C.PROCESSED / 'daily_fine_obs.npz'}")
    print(f"  {arr.shape[0]:,} days  {full[0].date()} -> {full[-1].date()}")
    print(f"  observed fraction over land: "
          f"{np.isfinite(arr)[:, t['land']].mean():.3f}")


if __name__ == "__main__":
    main()
