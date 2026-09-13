"""Module 3: empirical quantile mapping of forecast meteorology onto observed.

§4.7 localised the largest single loss in the chain: FutureTST scores pooled NSE
+0.577 on observed meteorology and +0.103 on forecast meteorology.  Some of that
0.47 is irreducible forecast error, but part is a *distribution* mismatch that
does not need a better forecast to fix -- only a mapping.

Three of the four non-precipitation channels are not even the same physical
quantity in the two sources:

    swdown    ERA5-Land gives DOWNWARD shortwave; S2S `ssr` is NET shortwave
    pressure  ERA5-Land gives SURFACE pressure; S2S `msl` is MEAN SEA LEVEL
    air_temp  same quantity, but 1.5 deg grid-box mean vs 0.1 deg catchment mean

Standardising each source by its own mean and variance -- what futuretst_v2 did
as a stopgap -- corrects the first two moments and nothing else.  Quantile
mapping corrects the whole distribution, which matters because the model's skill
lives in the tails: a forecast that gets the monsoon mean right but compresses
heavy-rain quantiles will systematically under-predict the events that fill the
reservoir.

Method (Panofsky & Brier, the standard hydrological form):

    F_obs^-1( F_fc(x) )

estimated empirically from matched quantiles, with two departures that matter in
practice:

  * The mapping is fitted on TRAINING YEARS ONLY and applied to the held-out
    year.  A mapping fitted on all years leaks the test monsoon's distribution
    into its own correction -- the same error as fitting a scaler before
    splitting, which §3 lists as one of the six inherited defects.
  * The upper tail is extrapolated by the last quantile *delta* rather than
    clamped.  np.interp would flatten every forecast above the training maximum
    onto one value, which is precisely the wrong behaviour for extremes.

Precipitation additionally gets a dry-day frequency correction: quantile mapping
alone leaves a wet-bias in drizzle, because a forecast that rains 0.2 mm on
every day of the monsoon has the wrong *frequency* even after its magnitudes are
mapped.

Run:  python preprocessing/bias_correction.py     (self-test on the S2S met)
Used by: inflow/futuretst_v2.py, inside the fold loop
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

N_QUANTILES = 100
WET_THRESHOLD = 0.1        # mm/day below which a day is "dry"


def fit(fc: np.ndarray, obs: np.ndarray, precip: bool = False) -> dict:
    """Fit an empirical quantile map from forecast -> observed.

    `fc` and `obs` need not be paired or even the same length: quantile mapping
    matches marginal distributions, not individual days.  They should however
    cover the same seasons, or the mapping absorbs a seasonal cycle it should
    not.
    """
    fc = np.asarray(fc, float)
    obs = np.asarray(obs, float)
    fc = fc[np.isfinite(fc)]
    obs = obs[np.isfinite(obs)]
    q = np.linspace(0.0, 1.0, N_QUANTILES)
    m = {"fq": np.quantile(fc, q), "oq": np.quantile(obs, q), "precip": precip}
    if precip:
        # Match the dry-day frequency: find the forecast threshold whose
        # exceedance rate equals the observed wet-day rate.  Anything below it
        # is set to zero rather than mapped, which is what removes drizzle bias.
        wet_obs = float((obs >= WET_THRESHOLD).mean())
        m["dry_cut"] = float(np.quantile(fc, 1.0 - wet_obs)) if wet_obs < 1.0 else -np.inf
        wet = obs[obs >= WET_THRESHOLD]
        m["fq"] = np.quantile(fc[fc >= m["dry_cut"]], q) if (fc >= m["dry_cut"]).sum() > 10 else m["fq"]
        m["oq"] = np.quantile(wet, q) if len(wet) > 10 else m["oq"]
    return m


def apply(x: np.ndarray, m: dict) -> np.ndarray:
    """Apply a fitted map, extrapolating the upper tail rather than clamping."""
    x = np.asarray(x, float)
    fq, oq = m["fq"], m["oq"]
    out = np.interp(x, fq, oq)

    # Upper tail: preserve the increment above the highest training quantile,
    # scaled by the ratio of the two distributions' top spreads.  Clamping here
    # would cap every future extreme at the largest one seen in training.
    hi = x > fq[-1]
    if hi.any():
        df = fq[-1] - fq[-5] if len(fq) >= 5 else 1.0
        do = oq[-1] - oq[-5] if len(oq) >= 5 else 1.0
        scale = (do / df) if df > 1e-9 else 1.0
        out[hi] = oq[-1] + (x[hi] - fq[-1]) * scale
    lo = x < fq[0]
    if lo.any():
        out[lo] = oq[0] + (x[lo] - fq[0])

    if m.get("precip"):
        out = np.where(x < m["dry_cut"], 0.0, np.maximum(out, 0.0))
    return out


def _self_test() -> None:
    """Fit on 2004-2011, verify on 2012-2014: does the mapping generalise?"""
    fm = pd.read_parquet(C.PROCESSED / "forecast_catchment_met.parquet")
    om = pd.read_parquet(C.PROCESSED / "daily_catchment_met.parquet")
    rain = pd.read_parquet(C.PROCESSED / "loyo_oof_rainfall.parquet")
    imd = pd.read_parquet(C.PROCESSED / "daily_catchment_obs.parquet")

    fm["valid_date"] = pd.to_datetime(fm["valid_date"])
    rain["valid_date"] = pd.to_datetime(rain["valid_date"])
    fm = fm.merge(rain[["init_date", "lead_day", "rain_pred"]].assign(
        init_date=lambda d: pd.to_datetime(d.init_date)),
        left_on=["init_date", "lead_day"], right_on=["init_date", "lead_day"], how="inner")
    fm = fm.rename(columns={"rain_pred": "prec"})
    om = om.set_index("date")
    om["prec"] = imd.set_index("date")["rain_obs_daily"].reindex(om.index).fillna(om["prec"])

    print(f"{'variable':10s} {'split':6s} {'raw bias':>10s} {'QM bias':>9s} "
          f"{'raw p95':>9s} {'QM p95':>8s} {'obs p95':>8s}")
    for v in ["prec", "air_temp", "swdown", "wind", "pressure"]:
        yr = fm["valid_date"].dt.year
        tr, te = yr <= 2011, yr >= 2012
        # Sample the observed series on the SAME valid dates as the forecast
        # rows.  Comparing JJAS forecasts against an all-year observed
        # distribution would make the mapping absorb the seasonal cycle and
        # report a seasonal offset as if it were forecast bias.
        obs_tr = om[v].reindex(fm.loc[tr, "valid_date"]).dropna().to_numpy()
        obs_te = om[v].reindex(fm.loc[te, "valid_date"]).dropna().to_numpy()
        m = fit(fm.loc[tr, v].to_numpy(), obs_tr, precip=(v == "prec"))

        raw = fm.loc[te, v].to_numpy()
        cor = apply(raw, m)
        print(f"{v:10s} {'test':6s} {raw.mean() - obs_te.mean():+10.2f} "
              f"{cor.mean() - obs_te.mean():+9.2f} "
              f"{np.percentile(raw, 95):9.2f} {np.percentile(cor, 95):8.2f} "
              f"{np.percentile(obs_te, 95):8.2f}")


if __name__ == "__main__":
    _self_test()
