"""VIC's annual water balance -- is the model partitioning rain correctly?

WHY THIS EXISTS.  `results/metrics/vic_inflow_calibrated.json` carries a
`water_balance` block (runoff coefficient 0.5587, ET/P 0.4772) that is quoted in
several documents as proof that VIC misses its physical targets.  NO COMMITTED
SCRIPT PRODUCES THOSE NUMBERS.  They cannot be reproduced, checked, or updated,
which is the same defect this project already found in its VIC skill figures.
This script is the missing one.

WHAT THE RATIOS MEAN, and the trap in comparing them.

    runoff coefficient  Q / P     the share of rain that leaves as streamflow
    evaporative ratio   ET / P    the share returned to the atmosphere
    closure            (Q+ET)/P   1.0 if catchment storage is unchanged

The published Mahanadi bands -- runoff coefficient 0.35-0.40 and ET/P 0.55-0.65
-- are ANNUAL figures.  Computing either over monsoon days only and comparing it
against an annual band is not a like-for-like test: inside the monsoon the soil
is already wet so a larger share of rain runs off, while ET is small next to a
very large P.  Both ratios therefore move in exactly the direction that makes a
sound model look broken.  This script reports BOTH bases and compares only the
annual one against the published band.

The spin-up year is excluded: it starts from an assumed soil moisture state and
its storage change is an artefact of that assumption, not of the model.

Run:  python hydrology/vic/water_balance.py
      python hydrology/vic/water_balance.py --outdir <dir> --years 2004-2011
Out:  results/metrics/vic_water_balance.json
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs import config as C

# Columns of the classic driver's flux file, read off its own header line
# (YEAR MONTH DAY OUT_RUNOFF OUT_BASEFLOW OUT_EVAP OUT_PREC ...).
COL_RUNOFF, COL_BASEFLOW, COL_EVAP, COL_PREC = 3, 4, 5, 6

# Published annual bands for the Mahanadi above Hirakud.  These are ANNUAL.
RC_BAND = (0.35, 0.40)
ET_BAND = (0.55, 0.65)

SPIN_UP_YEAR = 2003


def load_fluxes(outdir: Path) -> pd.DataFrame:
    """Catchment-mean daily runoff, baseflow, evaporation and precipitation, mm."""
    files = sorted(glob.glob(str(outdir / "fluxes_*.txt")))
    if not files:
        sys.exit(f"no fluxes_*.txt in {outdir}")
    acc, dates, n = None, None, 0
    for f in files:
        a = np.loadtxt(f, skiprows=3)
        if acc is None:
            acc = np.zeros((len(a), 4))
            dates = pd.to_datetime(dict(year=a[:, 0].astype(int),
                                        month=a[:, 1].astype(int),
                                        day=a[:, 2].astype(int)))
        elif len(a) != len(acc):
            sys.exit(f"{Path(f).name} has {len(a)} rows, expected {len(acc)} -- "
                     f"the output directory mixes runs of different lengths")
        acc += a[:, [COL_RUNOFF, COL_BASEFLOW, COL_EVAP, COL_PREC]]
        n += 1
    # An unweighted cell mean: VIC cells are equal-area to within 1 % over this
    # 5.5 deg latitude span, and the routing model -- not this diagnostic -- is
    # where area actually enters the numbers that get scored.
    df = pd.DataFrame(acc / n, columns=["runoff", "baseflow", "evap", "prec"])
    df["date"] = dates.values
    print(f"  {n} cells x {len(df):,} days  "
          f"({df.date.min():%Y-%m-%d} to {df.date.max():%Y-%m-%d})")
    return df.set_index("date")


def ratios(d: pd.DataFrame) -> dict:
    P = d["prec"].sum()
    if P <= 0:
        return {}
    Q = d["runoff"].sum() + d["baseflow"].sum()
    E = d["evap"].sum()
    n_years = len(d) / 365.25
    return {"P_mm_per_year": P / n_years, "ET_mm_per_year": E / n_years,
            "Q_mm_per_year": Q / n_years,
            "runoff_coefficient": Q / P, "ET_over_P": E / P,
            "closure": (Q + E) / P, "n_days": int(len(d))}


def band(name, v, lo, hi):
    if v is None:
        return "        n/a"
    if v < lo:
        return f"{v:.3f}  BELOW {lo:.2f}-{hi:.2f} by {lo - v:.3f}"
    if v > hi:
        return f"{v:.3f}  ABOVE {lo:.2f}-{hi:.2f} by {v - hi:.3f}"
    return f"{v:.3f}  within {lo:.2f}-{hi:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(C.PROCESSED / "vic" / "output"),
                    help="directory of VIC fluxes_*.txt")
    ap.add_argument("--years", default="",
                    help="restrict to YYYY-YYYY (default: all but the spin-up)")
    args = ap.parse_args()

    print("VIC water balance")
    d = load_fluxes(Path(args.outdir))

    if args.years:
        lo, hi = (int(x) for x in args.years.split("-"))
    else:
        lo, hi = SPIN_UP_YEAR + 1, int(d.index.year.max())
    d = d[(d.index.year >= lo) & (d.index.year <= hi)]
    print(f"  scoring {lo}-{hi}  (spin-up {SPIN_UP_YEAR} excluded)\n")

    annual = ratios(d)
    monsoon = ratios(d[d.index.month.isin(C.INIT_MONTHS)])

    print("  ANNUAL -- the basis the published bands are stated on")
    print(f"    P    {annual['P_mm_per_year']:7.1f} mm/yr")
    print(f"    ET   {annual['ET_mm_per_year']:7.1f} mm/yr")
    print(f"    Q    {annual['Q_mm_per_year']:7.1f} mm/yr")
    print(f"    runoff coefficient  {band('rc', annual['runoff_coefficient'], *RC_BAND)}")
    print(f"    ET / P              {band('et', annual['ET_over_P'], *ET_BAND)}")
    print(f"    closure (Q+ET)/P    {annual['closure']:.3f}   "
          f"(1.0 = no net storage change)")

    print("\n  MONSOON ONLY (Jun-Sep) -- NOT comparable to the annual bands")
    print(f"    runoff coefficient  {monsoon['runoff_coefficient']:.3f}")
    print(f"    ET / P              {monsoon['ET_over_P']:.3f}")
    print("    Inside the monsoon the soil is already wet, so more rain runs off,")
    print("    and ET is small beside a very large P.  Both ratios move away from")
    print("    the annual band for reasons that are seasonal, not structural.")

    per_year = {}
    for y, g in d.groupby(d.index.year):
        r = ratios(g)
        per_year[int(y)] = {k: float(r[k]) for k in
                            ("runoff_coefficient", "ET_over_P", "closure",
                             "P_mm_per_year")}

    out = {"outdir": str(args.outdir), "years": [lo, hi],
           "spin_up_excluded": SPIN_UP_YEAR,
           "published_bands_annual": {"runoff_coefficient": list(RC_BAND),
                                      "ET_over_P": list(ET_BAND)},
           "annual": {k: float(v) for k, v in annual.items()},
           "monsoon_only": {k: float(v) for k, v in monsoon.items()},
           "per_year": per_year,
           "note": "Published bands are ANNUAL. Compare them against `annual` "
                   "only; `monsoon_only` is reported to show how large the "
                   "difference is, not as a target to hit."}
    C.METRICS.mkdir(parents=True, exist_ok=True)
    p = C.METRICS / "vic_water_balance.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
