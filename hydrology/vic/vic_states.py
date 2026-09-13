"""Save a VIC model state at every forecast initialisation date.

A forecast must start from the catchment as it actually was -- soil moisture,
baseflow storage, snow -- not from a cold start.  Dong et al. drive XAJ the same
way: the hydrologic model carries antecedent conditions, and only the *forcing*
becomes a forecast at the initialisation date.

WHY A CHAIN RATHER THAN ONE RUN PER INITIALISATION.  (The counts below say 308
and 2003-2014; the archive now holds 532 initialisations over 2003-2022.  The
argument is unchanged -- only the numbers grew.)  VIC classic saves state at one date
per run, so producing 308 states naively means 308 runs of the full 2003-2014
record -- about 12 years of simulation each, over an hour in total.  Instead
this runs the record ONCE, in 308 consecutive segments: each segment starts from
the previous segment's saved state and ends at the next initialisation date.
Total simulated time is unchanged (12 years), so the cost is one continuous run
plus per-run startup.

The states are the shared starting point for every forecast: 3 precipitation
products x 10 members all branch from the same state at a given init date, so
any difference between them is caused by the forcing, never by the spin-up.

Run:  python hydrology/vic/vic_states.py [--limit N]
Out:  data/processed/vic/states/state_YYYYMMDD  (one per initialisation)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs import config as C

VIC_HOME = Path.home() / ".vic_mahanadi"
VIC_EXE = C.ROOT / "hydrology/vic/VIC/vic/drivers/classic/vic_classic.exe"
STATES = VIC_HOME / "states"
GP_TEMPLATE = VIC_HOME / "global_param.txt"
RECORD_START = pd.Timestamp("2003-01-01")


def init_dates() -> pd.DatetimeIndex:
    """Every initialisation date of the v2 archive, in order.

    Follows whichever archive the CNN used: 308 inits on the 2004-2014 build,
    532 on the 2004-2022 one.  Reading the 11-year file here while the rest of
    the chain runs on 19 monsoons would silently leave 2015-2022 with no state
    to start from.
    """
    dset = next((q for q in (C.PROCESSED / "downscaling_v2_full.npz",
                             C.PROCESSED / "downscaling_v2.npz") if q.exists()),
                C.PROCESSED / "downscaling_v2.npz")
    z = np.load(dset, allow_pickle=True)
    valid = pd.to_datetime(z["valid"])
    init = valid - pd.to_timedelta(z["lead"], unit="D")
    return pd.DatetimeIndex(sorted(set(init)))


def write_gp(path: Path, start, end, init_state: Path | None, save_at) -> None:
    """A global-parameter file for one segment of the chain."""
    txt = GP_TEMPLATE.read_text().splitlines()
    out, skip = [], {"STARTYEAR", "STARTMONTH", "STARTDAY", "STARTSEC",
                     "ENDYEAR", "ENDMONTH", "ENDDAY", "INIT_STATE", "STATENAME",
                     "STATEYEAR", "STATEMONTH", "STATEDAY", "STATESEC",
                     "STATE_FORMAT", "RESULT_DIR"}
    for line in txt:
        if line.split() and line.split()[0] in skip:
            continue
        out.append(line)
    out += [
        f"STARTYEAR   {start.year}", f"STARTMONTH  {start.month}",
        f"STARTDAY    {start.day}", "STARTSEC    0",
        f"ENDYEAR     {end.year}", f"ENDMONTH    {end.month}",
        f"ENDDAY      {end.day}",
        f"RESULT_DIR  {VIC_HOME / 'chain_out'}",
        f"STATENAME   {STATES / 'state'}",
        "STATE_FORMAT ASCII",
        f"STATEYEAR   {save_at.year}", f"STATEMONTH  {save_at.month}",
        f"STATEDAY    {save_at.day}", "STATESEC    0",
    ]
    if init_state is not None:
        out.append(f"INIT_STATE  {init_state}")
    path.write_text("\n".join(out) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N inits only (testing)")
    args = ap.parse_args()

    if not VIC_EXE.exists():
        sys.exit(f"no VIC binary at {VIC_EXE}")
    STATES.mkdir(parents=True, exist_ok=True)
    (VIC_HOME / "chain_out").mkdir(exist_ok=True)
    (VIC_HOME / "logs").mkdir(exist_ok=True)

    dates = init_dates()
    if args.limit:
        dates = dates[:args.limit]
    print(f"building {len(dates)} VIC states, {dates[0].date()} .. {dates[-1].date()}")

    gp = VIC_HOME / "gp_chain.txt"
    prev_state, prev_date, t0 = None, RECORD_START, time.time()
    for i, d in enumerate(dates):
        # VIC needs at least one step in the segment
        start = prev_date if prev_state is None else prev_date + pd.Timedelta(days=1)
        if start > d:
            continue
        write_gp(gp, start, d, prev_state, d)
        r = subprocess.run([str(VIC_EXE), "-g", str(gp)],
                           capture_output=True, text=True)
        want = STATES / f"state_{d:%Y%m%d}_00000"
        if r.returncode != 0 or not want.exists():
            print(f"  FAILED at {d.date()} (rc={r.returncode})")
            print((r.stdout + r.stderr)[-600:])
            sys.exit(1)
        prev_state, prev_date = want, d
        if i % 25 == 0 or i == len(dates) - 1:
            print(f"  [{i+1}/{len(dates)}] {d.date()}  "
                  f"{time.time()-t0:.0f}s elapsed", flush=True)

    n = len(list(STATES.glob("state_*")))
    print(f"\n{n} states in {STATES}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
