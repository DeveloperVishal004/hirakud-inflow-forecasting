"""Prepare the Hirakud reservoir inflow series used as the final target.

The raw workbook is daily 2003-2014 with no gaps or duplicates.  Two things
still need care:

  * It is trimmed to 2004-2014, the ECMWF overlap.  The 2003 season is kept
    separately as warm-up history: FutureTST conditions on past inflow, so the
    2004 season needs a lead-in that is not itself a training target.
  * 352 exact zeros sit almost entirely in the dry season (May 71, Jun 60,
    Mar 54, Feb 51, Jan 49; longest run 9 days).  Zero inflow at a reservoir of
    this size is not physical -- it is a back-computed water balance clipped at
    zero.  They are flagged, not silently kept, so no log transform or relative
    error metric ever divides by them.

Run:  python preprocessing/build_inflow.py
Out:  data/processed/inflow_daily.parquet
"""

import sys

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from configs import config as C


def load_inflow() -> pd.DataFrame:
    df = pd.read_excel(C.INFLOW_XLSX)
    df = df.rename(columns={"Date": "date", "Inflow (cumecs)": "inflow"})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def audit(df: pd.DataFrame) -> None:
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    missing = len(set(full) - set(df["date"]))
    assert missing == 0, f"{missing} missing days"
    assert not df["date"].duplicated().any(), "duplicate dates"
    assert (df["inflow"] >= 0).all(), "negative inflow"


def main() -> None:
    df = load_inflow()
    audit(df)
    print(f"raw  {len(df):,} days  {df['date'].min().date()} -> {df['date'].max().date()}")

    df["is_zero"] = df["inflow"] == 0.0
    # A zero is implausible; treat it as unobserved rather than as a real value.
    df["inflow_valid"] = ~df["is_zero"]

    # Keep 2003 as warm-up: available as lagged input, never as a target.
    df["role"] = "warmup"
    in_range = df["date"].dt.year.between(C.YEAR_MIN, C.YEAR_MAX)
    df.loc[in_range, "role"] = "usable"

    year = df["date"].dt.year
    df.loc[in_range & year.isin(C.TRAIN_YEARS), "split"] = "train"
    df.loc[in_range & year.isin(C.VAL_YEARS), "split"] = "val"
    df.loc[in_range & year.isin(C.TEST_YEARS), "split"] = "test"

    df["is_monsoon"] = df["date"].dt.month.isin(C.INIT_MONTHS)

    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    out = C.PROCESSED / "inflow_daily.parquet"
    df.to_parquet(out, index=False)

    usable = df[df["role"] == "usable"]
    monsoon = usable[usable["is_monsoon"]]
    print(f"\nwrote {out}")
    print(f"  usable {C.YEAR_MIN}-{C.YEAR_MAX}   {len(usable):,} days")
    print(f"  monsoon (JJAS)      {len(monsoon):,} days")
    print(f"  zeros flagged       {int(df['is_zero'].sum())} total, "
          f"{int(monsoon['is_zero'].sum())} in usable JJAS")
    print(f"  JJAS inflow  mean {monsoon['inflow'].mean():,.0f}  "
          f"median {monsoon['inflow'].median():,.0f}  max {monsoon['inflow'].max():,.0f} cumecs")
    print("\n  split day counts (JJAS):")
    print("   ", monsoon["split"].value_counts().to_dict())
    print("\n  peak inflow per split year:")
    print(monsoon.groupby([monsoon["date"].dt.year, "split"])["inflow"].max().round(0).to_string())


if __name__ == "__main__":
    main()
