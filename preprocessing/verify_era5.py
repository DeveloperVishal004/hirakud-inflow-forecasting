"""Check an ERA5-Land month before it is trusted as VIC forcing.

A download that is present but wrong is worse than one that is absent, because
VIC will run straight through it.  Every check here failed at least plausibly
during the June 2003 smoke test, so none of them is ceremonial.

Checks:
  1. grid      0.1 deg spacing, and the catchment sits strictly inside with
               margin on all four sides (edge cells have no neighbours to
               regrid from, and on the west those are the headwaters)
  2. time      exactly 24 steps per day, regular, no gaps
  3. coverage  NaN fraction per variable
  4. physics   plausible ranges; dewpoint <= temperature
  5. tp        confirms the accumulation convention rather than assuming it

Run:  python preprocessing/verify_era5.py [path/to/dir_or_file]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

DEFAULT = C.RAW / "vic" / "forcing" / "era5"

# (name, unit, offset, scale, plausible min, plausible max), FULL YEAR.
#
# These bounds must cover January as well as the monsoon.  An earlier 5 C floor
# on t2m flagged 24 of 31 files: the true minima run to 1.93 C, every one of
# them at 01Z -- 06:30 IST, precisely when Tmin occurs -- during mid-January
# cold waves in the northern catchment.  Correct data, wrong threshold.  The
# bounds below are set where a value would be genuinely impossible for this
# domain, not merely unusual.
RANGES = {
    "t2m":   ("C",      -273.15, 1.0,    0.0,  52.0),
    "d2m":   ("C",      -273.15, 1.0,  -15.0,  32.0),
    "sp":    ("hPa",       0.0,  0.01, 850.0, 1020.0),
    "u10":   ("m/s",       0.0,  1.0,  -30.0,  30.0),
    "v10":   ("m/s",       0.0,  1.0,  -30.0,  30.0),
    "swvl1": ("m3/m3",     0.0,  1.0,    0.0,   0.7),
    "swvl4": ("m3/m3",     0.0,  1.0,    0.0,   0.7),
}


def check(path: Path) -> bool:
    ds = xr.open_dataset(path)
    ok = True
    print(f"\n=== {path.parent.name}/{path.name} ===")

    # ---- 1. grid
    la, lo = ds.latitude.values, ds.longitude.values
    dla, dlo = abs(np.diff(la)).mean(), abs(np.diff(lo)).mean()
    margins = {"N": la.max() - C.CATCHMENT_LAT_MAX, "S": C.CATCHMENT_LAT_MIN - la.min(),
               "W": C.CATCHMENT_LON_MIN - lo.min(), "E": lo.max() - C.CATCHMENT_LON_MAX}
    bad = [k for k, v in margins.items() if v < 0]
    print(f"  grid     {len(la)}x{len(lo)} @ {dla:.3f}/{dlo:.3f} deg   "
          f"margins " + "  ".join(f"{k}{v:+.2f}" for k, v in margins.items()))
    if bad:
        print(f"    FAIL: catchment extends past the domain on {', '.join(bad)}")
        ok = False
    if not (0.09 < dla < 0.11 and 0.09 < dlo < 0.11):
        print("    FAIL: not the expected 0.1 deg ERA5-Land grid")
        ok = False

    # ---- 2. time
    #
    # Two sampling rates are expected, not one.  t2m is hourly because Tmax and
    # Tmin are diurnal EXTREMES; d2m/u10/v10 are 6-hourly because VIC uses only
    # their daily MEANS.  Asserting 60-minute steps everywhere would fail the
    # met stream for being exactly what it was asked to be.
    t = pd.DatetimeIndex(ds.valid_time.values)
    steps = np.unique(np.diff(t.values).astype("timedelta64[m]").astype(int))
    step = int(steps[0]) if len(steps) == 1 else None
    per_day = 24 * 60 // step if step else 0
    n_days = len(t) // per_day if per_day else 0
    print(f"  time     {t[0]:%Y-%m-%d %HZ} .. {t[-1]:%Y-%m-%d %HZ}   "
          f"n={len(t)} ({n_days} days)  step={steps} min")
    if step not in (60, 360):
        print(f"    FAIL: irregular or unexpected time step {steps}")
        ok = False
    elif len(t) % per_day:
        print("    FAIL: not a whole number of days -- steps are missing")
        ok = False
    if "t2m" in ds.data_vars and step != 60:
        print("    FAIL: t2m must be hourly or Tmax/Tmin are clipped")
        ok = False

    # ---- 3/4. coverage and physics
    for v in ds.data_vars:
        a = ds[v].values
        nan = 100 * np.isnan(a).mean()
        line = f"  {v:6s}   NaN {nan:5.2f}%"
        if v in RANGES:
            unit, off, sc, lo_ok, hi_ok = RANGES[v]
            b = (a + off) * sc
            line += (f"   {np.nanmin(b):8.2f} .. {np.nanmax(b):8.2f} {unit}"
                     f"   mean {np.nanmean(b):7.2f}")
            if np.nanmin(b) < lo_ok or np.nanmax(b) > hi_ok:
                line += f"   OUT OF RANGE [{lo_ok}, {hi_ok}]"
                ok = False
        print(line)
        if nan > 50:
            print(f"    FAIL: {v} is mostly missing")
            ok = False

    if {"t2m", "d2m"} <= set(ds.data_vars):
        ex = (ds.d2m.values - ds.t2m.values).max()
        # ERA5 permits marginal supersaturation; only a real excess matters.
        verdict = "numerical noise, ignore" if ex < 0.05 else "REAL -- clip d2m to t2m"
        print(f"  d2m-t2m  max excess {ex:+.4f} C   ({verdict})")

    # ---- 5. tp accumulation convention
    if "tp" in ds.data_vars:
        c = ds.tp.values.mean(axis=(1, 2)) * 1000.0
        hrs = t.hour
        naive = c.sum()
        at23 = c[hrs == 23].sum()
        # Value at 00Z carries the whole preceding day; 01Z is the first hour
        # of the new one.  Verified on June 2003: 23Z 15.68 -> 00Z 16.09 -> 01Z 0.37.
        resets = (c[1:][hrs[1:] == 1] < c[:-1][hrs[:-1] == 0]).mean()
        print(f"  tp       naive sum of all hours {naive:8.1f} mm  <- WRONG, "
              f"accumulated series")
        print(f"           sum of 23Z values      {at23:8.1f} mm  <- 23 h/day, "
              f"undercounts")
        print(f"           01Z < preceding 00Z in {100*resets:5.1f}% of days "
              f"-> {'accumulates from 00Z, resets daily' if resets > 0.9 else 'CONVENTION UNCLEAR'}")
        print(f"           correct daily total for day D = tp at 00Z on day D+1")
        if resets <= 0.9:
            ok = False

    print(f"  RESULT   {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    files = [root] if root.is_file() else sorted(root.rglob("*.nc"))
    if not files:
        print(f"no .nc under {root}")
        sys.exit(1)
    results = [check(f) for f in files]
    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
