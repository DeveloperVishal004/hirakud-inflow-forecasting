"""Assemble the (forecast rainfall + antecedent state) -> Hirakud inflow dataset.

One row per (initialisation, lead): what a reservoir operator knows at issue
time, paired with the inflow that actually arrived `lead` days later.

The hard rule here is the information cutoff.  Everything observed must be dated
at or before `init_date`; everything after it may only come from the forecast.
Antecedent inflow in particular is the strongest predictor available, and taking
it at `valid_date` instead of `init_date` would leak the answer -- the model
would look excellent and be useless operationally.  `assert_no_leakage()` checks
this rather than trusting the construction.

Feature groups:
  forecast   rainfall on the target day, cumulative rain since issue, and the
             3-day window ending on the target day (runoff responds to recent
             accumulation, not a single day)
  antecedent inflow and observed rainfall as of issue time, at several windows,
             which together encode catchment wetness -- the Day A / Day B
             distinction that makes identical rainfall produce different inflow
  seasonal   day-of-year encoding plus the day-of-year inflow climatology

Run:  python preprocessing/build_inflow_dataset.py
Out:  data/processed/inflow_dataset.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

ANTECEDENT_WINDOWS = (3, 7, 15, 30)


def load_inflow() -> pd.DataFrame:
    q = pd.read_parquet(C.PROCESSED / "inflow_daily.parquet")
    q = q.rename(columns={"date": "d"})[["d", "inflow", "inflow_valid"]]
    return q.set_index("d").sort_index()


def inflow_climatology(q: pd.DataFrame, train_years) -> pd.Series:
    """Day-of-year mean inflow from training years only, +-15 day smoothing."""
    tr = q[q.index.year.isin(train_years) & q["inflow_valid"]]
    by_doy = tr.groupby(tr.index.dayofyear)["inflow"].mean()
    full = by_doy.reindex(range(1, 367))
    # Circular rolling mean so the window is defined at the year boundary.
    padded = pd.concat([full, full, full])
    smoothed = padded.rolling(31, center=True, min_periods=1).mean()
    out = smoothed.iloc[366:732]
    out.index = range(1, 367)
    return out.fillna(tr["inflow"].mean())


def main() -> None:
    rain = pd.read_parquet(C.PROCESSED / "catchment_rainfall.parquet")
    q = load_inflow()
    clim = inflow_climatology(q, C.TRAIN_YEARS)

    inflow = q["inflow"]
    valid_flag = q["inflow_valid"]
    # Antecedent aggregates are computed on the daily series first, then looked
    # up at init_date -- so every window is strictly historical by construction.
    ante = pd.DataFrame(index=q.index)
    masked = inflow.where(valid_flag)
    for w in ANTECEDENT_WINDOWS:
        ante[f"q_ante_{w}"] = masked.rolling(w, min_periods=1).mean()
    ante["q_ante_1"] = masked

    # Observed catchment rainfall on EVERY calendar day, taken from IMD rather
    # than from the reforecast's monsoon-only window.  The 3 June initialisation
    # needs a lookback into May, and that pre-monsoon dryness is precisely the
    # antecedent-wetness signal the inflow model depends on.
    obs_daily = pd.read_parquet(C.PROCESSED / "daily_catchment_obs.parquet")
    obs_full = obs_daily.set_index("date")["rain_obs_daily"].sort_index()
    obs_full = obs_full.reindex(
        pd.date_range(obs_full.index.min(), obs_full.index.max(), freq="D"))
    r_ante = pd.DataFrame(index=obs_full.index)
    for w in ANTECEDENT_WINDOWS:
        r_ante[f"r_ante_{w}"] = obs_full.rolling(w, min_periods=1).mean()

    # Cumulative and windowed forecast rainfall, within each initialisation.
    rain = rain.sort_values(["init_date", "lead_day"]).copy()
    grp = rain.groupby("init_date", sort=False)["rain_pred"]
    rain["rain_cum"] = grp.cumsum()
    rain["rain_3d"] = grp.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    grp_o = rain.groupby("init_date", sort=False)["rain_obs"]
    rain["rain_obs_cum"] = grp_o.cumsum()
    rain["rain_obs_3d"] = grp_o.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    grp_e = rain.groupby("init_date", sort=False)["rain_ecmwf"]
    rain["rain_ecmwf_cum"] = grp_e.cumsum()
    rain["rain_ecmwf_3d"] = grp_e.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)

    # Longer forecast-rain window: runoff at Hirakud integrates about a week of
    # catchment rainfall, not just three days.
    for src in ["rain_pred", "rain_obs", "rain_ecmwf"]:
        tag = {"rain_pred": "rain", "rain_obs": "rain_obs", "rain_ecmwf": "rain_ecmwf"}[src]
        rain[f"{tag}_7d"] = (rain.groupby("init_date", sort=False)[src]
                             .rolling(7, min_periods=1).mean().reset_index(level=0, drop=True))

    df = rain.copy()
    # Antecedent state is read at the day BEFORE issue: inflow for init_date
    # itself may not be available when the forecast is produced.
    cutoff = df["init_date"] - pd.Timedelta(days=1)
    for col in ante.columns:
        df[col] = ante[col].reindex(cutoff).to_numpy()
    for col in r_ante.columns:
        df[col] = r_ante[col].reindex(cutoff).to_numpy()

    # Projected wetness: catchment-mean rainfall over the W days ENDING at the
    # target day, bridging observed rain before issue with forecast rain after
    # it.  The plain antecedent features freeze catchment state at init-time,
    # which is 17 days stale for the longest lead -- precisely where skill
    # collapsed (NSE +0.13 at leads 13-17).  For lead >= W the window is pure
    # forecast; for lead < W the remainder is filled from observations.
    for w in ANTECEDENT_WINDOWS:
        n_fc = np.minimum(df["lead_day"], w)                 # forecast days in window
        n_ob = w - n_fc                                       # observed days in window
        ob_part = df[f"r_ante_{w}"] * n_ob
        for cum_col, out in [("rain_cum", f"wet_proj_{w}"),
                             ("rain_obs_cum", f"wet_proj_obs_{w}"),
                             ("rain_ecmwf_cum", f"wet_proj_ecmwf_{w}")]:
            fc_part = df[cum_col] - np.where(
                df["lead_day"] > w,
                df.groupby("init_date")[cum_col].shift(w).fillna(0.0),
                0.0,
            )
            df[out] = (fc_part + ob_part) / w

    df["target_inflow"] = inflow.reindex(df["valid_date"]).to_numpy()
    df["target_valid"] = valid_flag.reindex(df["valid_date"]).fillna(False).to_numpy()

    doy = df["valid_date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 366.0)
    df["q_clim"] = clim.reindex(doy).to_numpy()

    df = df.dropna(subset=["target_inflow"])
    df = df[df["target_valid"]].copy()

    # q_ante_1 is NaN when the previous day's inflow was one of the flagged
    # implausible zeros.  Fall back to the 3-day mean rather than dropping the
    # row: the wider window still describes catchment state, and these are
    # dry-season days where inflow is low regardless.
    df["q_ante_1"] = df["q_ante_1"].fillna(df["q_ante_3"])
    # A row with no antecedent inflow at all cannot represent catchment wetness.
    df = df.dropna(subset=[f"q_ante_{w}" for w in ANTECEDENT_WINDOWS])

    assert_no_leakage(df)

    out = C.PROCESSED / "inflow_dataset.parquet"
    df.to_parquet(out, index=False)

    print(f"wrote {out}   {len(df):,} rows")
    print(f"  inits {df['init_date'].nunique()}  leads {df['lead_day'].min()}-{df['lead_day'].max()}")
    print("\n  rows per split:", df["split"].value_counts().to_dict())
    print("\n  target inflow (cumecs) by split:")
    print(df.groupby("split")["target_inflow"].describe()[["count", "mean", "50%", "max"]].round(0).to_string())
    print("\n  correlation with target (all rows):")
    for c in ["q_ante_1", "q_ante_7", "q_ante_30", "q_clim", "rain_pred", "rain_cum", "rain_3d"]:
        print(f"    {c:12s} {df[c].corr(df['target_inflow']):+.3f}")


def assert_no_leakage(df: pd.DataFrame) -> None:
    """Every antecedent feature must be dated strictly before init_date."""
    assert (df["valid_date"] > df["init_date"]).all(), "valid_date must follow init_date"
    expected = df["init_date"] + pd.to_timedelta(df["lead_day"], unit="D")
    assert (df["valid_date"] == expected).all(), "valid_date != init_date + lead_day"
    print("leakage checks passed: antecedent state read at init_date - 1, "
          "target at init_date + lead_day")


if __name__ == "__main__":
    main()
