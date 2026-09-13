"""One request: do 2t and tcc come back per-lead if asked as 24 h periods?

WHY.  The main download (download_s2s_dong.py) asks for leadtime_hour
"24","48",..."720" -- instantaneous hours.  Nine surface fields honour that and
return 330 GRIB messages across 30 steps.  `2t` and `tcc` return 11 messages at
a single step labelled "0-24", i.e. one 24 h mean on day 1 and nothing after.

The ECDS schema explains it: leadtime_hour accepts instantaneous hours AND 24 h
period windows ("0_24", "24_48", ... "696_720"), all 30 of which exist.  2 m
temperature and total cloud cover are period-processed in the S2S reforecast, so
an instantaneous request matches only the one window that happens to align.

This probes a single initialisation.  If it returns 30 steps, the same change
applied to the main download recovers both variables lead-wise -- and a daily
MEAN is arguably the better predictor for a daily rainfall model than an
instantaneous 00Z snapshot.

Writes to a scratch path; it cannot touch data/raw/s2s.
Run:  python preprocessing/probe_2t_tcc.py
"""

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

OUT = C.ROOT / ".probe_log" / "probe_2t_tcc.grib"
PERIODS = [f"{24*d}_{24*(d+1)}" for d in range(30)]


def main():
    import cdsapi
    import eccodes as ec

    rc = Path.home() / ".ecdsapirc"
    cfg = {k.strip(): v.strip()
           for k, v in (l.split(":", 1) for l in rc.read_text().splitlines() if ":" in l)}

    body = {
        "origin": "ecmwf", "year": "2024", "month": "06", "day": "03",
        "time": "00:00",
        "hyear": ["2004"],                    # one hindcast year: smallest useful probe
        "hmonth": ["06"], "hday": ["03"],
        "forecast_type": "control_forecast",
        "level_type": "single_level",
        "variable": ["2_m_temperature", "total_cloud_cover"],
        "leadtime_hour": PERIODS,             # <- the change under test
        "area": [25.5, 79.5, 16.5, 88.5],
        "data_format": "grib",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"requesting 2t + tcc, {len(PERIODS)} x 24h windows, 1 hindcast year")
    cdsapi.Client(url=cfg.get("url"), key=cfg["key"], quiet=True).retrieve(
        "s2s-reforecasts", body, str(OUT))

    cnt, steps = collections.Counter(), collections.defaultdict(set)
    with open(OUT, "rb") as fh:
        while (gid := ec.codes_grib_new_from_file(fh)) is not None:
            cnt[ec.codes_get(gid, "shortName")] += 1
            steps[ec.codes_get(gid, "shortName")].add(ec.codes_get(gid, "step"))
            ec.codes_release(gid)

    print(f"\n{OUT.stat().st_size/1e3:.0f} kB")
    print(f"{'var':6} {'msgs':>5} {'steps':>6}  range")
    for k in sorted(cnt):
        s = sorted(steps[k])
        print(f"{k:6} {cnt[k]:5} {len(s):6}  {s[0]} .. {s[-1]}")

    ok = all(len(steps[k]) == 30 for k in cnt) and len(cnt) == 2
    print("\nVERDICT:", "period request works -- patch the main download"
          if ok else "still not per-lead; 2t/tcc stay lead-invariant")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
