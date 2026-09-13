"""Extend the Hirakud inflow record with the 2015-2022 monsoon observations.

The original record (data/raw/source/Vishal_data.xlsx -> inflow_daily.parquet)
covers 2003-2014 in cumec.  Inflow.xlsx adds June-October of 2015-2022 in
CUSEC, which is converted here.

WHY THE UNIT MATTERS ENOUGH TO CHECK RATHER THAN TRUST THE COLUMN HEADING.
Read as cumec the file implies a JJASO mean of 63,254 m3/s and a peak of
868,135 m3/s -- 10,026 mm of runoff over an 83,400 km2 catchment that receives
about 1,200 mm of monsoon rainfall, and a peak twenty times Hirakud's spillway
capacity.  Converted from cusec it gives 1,791 and 24,583 m3/s, against 2,267
and 29,204 for the existing 2004-2014 record.  The heading is correct; the
assertion below keeps it that way if the file is ever replaced.

The new record is monsoon-only (June-October), so the extended series is NOT
continuous: 2015-2022 has no November-May data.  `has_full_year` marks which
years can support a continuous hydrologic simulation and which are usable only
for forecast verification within the monsoon.

Run:  python preprocessing/merge_inflow_2015_2022.py
Out:  data/processed/inflow_daily_extended.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

SRC = C.RAW / "source" / "Inflow.xlsx"
OUT = C.PROCESSED / "inflow_daily_extended.parquet"
CUSEC_TO_CUMEC = 0.0283168466


def main():
    old = pd.read_parquet(C.PROCESSED / "inflow_daily.parquet")
    new = pd.read_excel(SRC)
    new.columns = ["date", "inflow_cusec"]
    new["date"] = pd.to_datetime(new["date"])
    new["inflow"] = new["inflow_cusec"] * CUSEC_TO_CUMEC

    # Physical guard: seasonal runoff depth over the catchment must be within a
    # plausible fraction of monsoon rainfall.  Reading cusec as cumec fails this
    # by a factor of ~35.
    area_m2 = 83_400e6
    depth_mm = new["inflow"].mean() * (153 * 86400) / area_m2 * 1000
    assert 100 < depth_mm < 900, (
        f"implied seasonal runoff {depth_mm:.0f} mm is not physical -- check "
        f"whether Inflow.xlsx is really in cusec")
    print(f"  unit check: implied seasonal runoff {depth_mm:.0f} mm  (expected 100-900)")

    keep = ["date", "inflow"]
    new = new[keep].copy()
    new["inflow_valid"] = new["inflow"].notna() & (new["inflow"] >= 0)
    new["is_zero"] = new["inflow"] == 0
    new["is_monsoon"] = new["date"].dt.month.isin(C.INIT_MONTHS)
    new["role"] = "extension"
    new["split"] = None

    both = pd.concat([old, new], ignore_index=True).sort_values("date")
    both = both.drop_duplicates("date", keep="first").reset_index(drop=True)

    # Two different questions, so two flags.  A continuous simulation needs the
    # whole year; scoring a monsoon forecast only needs the monsoon.  Every year
    # in the original record has scattered missing days (2013 has 292 of 365),
    # so an annual-completeness threshold misclassifies years whose monsoon is
    # in fact intact -- 2013 has 115 of 122 JJAS days.
    v = both[both["inflow_valid"]]
    ann = v.groupby(v["date"].dt.year).size()
    mon = v[v["date"].dt.month.isin(C.INIT_MONTHS)].groupby(
        v[v["date"].dt.month.isin(C.INIT_MONTHS)]["date"].dt.year).size()
    full = set(ann[ann > 280].index)
    monsoon_ok = set(mon[mon > 110].index)          # of 122 JJAS days
    both["has_full_year"] = both["date"].dt.year.isin(full)
    both["monsoon_complete"] = both["date"].dt.year.isin(monsoon_ok)

    both.to_parquet(OUT, index=False)
    v = both[both["inflow_valid"]]
    print(f"\n  {OUT.name}: {len(both)} rows, {v['date'].min().date()}..{v['date'].max().date()}")
    print(f"  full-year record:   {sorted(int(y) for y in full)}")
    print(f"  monsoon complete:   {sorted(int(y) for y in monsoon_ok)}")
    print(f"  monsoon incomplete: {sorted(int(y) for y in set(both['date'].dt.year.unique()) - monsoon_ok)}")
    j = v[v["date"].dt.month.isin((6, 7, 8, 9, 10))]
    for lo, hi in ((2003, 2014), (2015, 2022)):
        s = j[(j["date"].dt.year >= lo) & (j["date"].dt.year <= hi)]
        print(f"  JJASO {lo}-{hi}: n={len(s):5}  mean {s['inflow'].mean():7,.0f}  "
              f"peak {s['inflow'].max():8,.0f} m3/s")


if __name__ == "__main__":
    main()
