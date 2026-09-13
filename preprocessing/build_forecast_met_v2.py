"""Catchment-mean forecast meteorology at leads 1-30, for the FutureTST stage.

The v1 file (forecast_catchment_met.parquet) stops at lead 17 because the old
S2S download did.  This rebuilds it from the v2 archive so the transformer can
run to the 30-day horizon Ambika et al. (2025) report.

    prec          the CNN-downscaled rainfall (percell_v2_oof.npz), averaged
                  over the 224 catchment cells -- the stage-1 handoff
    swdown        ECMWF ssrd, catchment mean
    air_temp_max  no forecast equivalent: the S2S request did not include daily
    air_temp_min  max/min 2 m temperature, and the archive carries only a daily
    vp            mean.  Vapour pressure likewise needs 2 m dewpoint, also not
                  requested.  These three are left ABSENT here so the consumer
                  fills them from climatology and says so, rather than this
                  script inventing values that look like forecasts.

Closing that gap properly means one more small download: the ECDS schema does
offer maximum_2_m_temperature_in_the_last_6_hours,
minimum_2_m_temperature_in_the_last_6_hours and 2_m_dewpoint_temperature.

Run:  python preprocessing/build_forecast_met_v2.py
Out:  data/processed/forecast_catchment_met_v2.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

OUT = C.PROCESSED / "forecast_catchment_met_v2.parquet"


def _dset():
    """The archive percell_v2_oof.npz was built from -- 19 monsoons when present.

    percell_v2_oof.npz now holds 15,960 rows (2004-2022) while downscaling_v2.npz
    still holds 9,240 (2004-2014); pairing them would misalign every row.
    """
    return next((q for q in (C.PROCESSED / "downscaling_v2_full.npz",
                             C.PROCESSED / "downscaling_v2.npz") if q.exists()),
                C.PROCESSED / "downscaling_v2.npz")


def main():
    oof = np.load(C.PROCESSED / "percell_v2_oof.npz", allow_pickle=True)
    d = np.load(_dset(), allow_pickle=True)
    ch = [str(c) for c in d["channels"]]
    lead = d["lead"].astype(int)
    valid = pd.to_datetime(d["valid"])
    init = valid - pd.to_timedelta(lead, unit="D")

    # stage-1 rainfall: mean over the catchment cells the CNN predicts
    prec = oof["oof"].mean(axis=1)

    # ssrd is stored as a mean flux (W m-2) on the coarse grid; the catchment
    # mean of the 3x3 centre is close enough for a lumped forcing series and
    # avoids re-interpolating the whole field.
    ssrd = d["coarse"][:, ch.index("ssrd")][:, 2:5, 2:5].mean(axis=(1, 2))

    out = pd.DataFrame({"init_date": init, "lead_day": lead,
                        "valid_date": valid, "prec": prec, "swdown": ssrd})
    out = out.sort_values(["init_date", "lead_day"]).reset_index(drop=True)
    out.to_parquet(OUT, index=False)
    print(f"  {OUT.name}: {len(out):,} rows, "
          f"{out.init_date.min().date()}..{out.init_date.max().date()}")
    print(f"  leads {out.lead_day.min()}-{out.lead_day.max()}, "
          f"{out.init_date.nunique()} initialisations")
    print(f"  prec mean {out.prec.mean():.2f} mm/d   swdown mean {out.swdown.mean():.1f} W/m2")
    print(f"  absent by design (filled from climatology downstream): "
          f"air_temp_max, air_temp_min, vp")





def rainfall_handoff():
    """Stage-1 rainfall as (init_date, lead_day, rain_pred, rain_obs), leads 1-30.

    Same schema as loyo_oof_rainfall_cnn.parquet so the inflow stage needs no
    special case, but derived from the v2 per-cell CNN out-of-fold predictions
    and therefore covering the full 30-day horizon rather than 17.
    """
    z = np.load(C.PROCESSED / "percell_v2_oof.npz", allow_pickle=True)
    d = np.load(_dset(), allow_pickle=True)
    lead = d["lead"].astype(int)
    valid = pd.to_datetime(d["valid"])
    init = valid - pd.to_timedelta(lead, unit="D")
    m = z["mask"]
    pred = np.where(m, z["oof"], np.nan)
    obs = np.where(m, z["target"], np.nan)
    out = pd.DataFrame({
        "init_date": init, "lead_day": lead, "valid_date": valid,
        "rain_pred": np.nanmean(pred, axis=1),
        "rain_obs": np.nanmean(obs, axis=1)}).dropna()
    f = C.PROCESSED / "loyo_oof_rainfall_v2.parquet"
    out.sort_values(["init_date", "lead_day"]).to_parquet(f, index=False)
    print(f"\n  {f.name}: {len(out):,} rows, leads {out.lead_day.min()}-{out.lead_day.max()}")
    print(f"  corr(pred, obs) overall {out.rain_pred.corr(out.rain_obs):+.3f}")
    for lo, hi in ((1, 5), (14, 17), (26, 30)):
        s = out[(out.lead_day >= lo) & (out.lead_day <= hi)]
        print(f"    leads {lo:2}-{hi:2}: corr {s.rain_pred.corr(s.rain_obs):+.3f}")


if __name__ == "__main__":
    main()
    rainfall_handoff()
