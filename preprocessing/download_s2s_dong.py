"""Download the ECMWF S2S reforecast to Dong et al. (2025)'s specification.

Same product and the same predictors as the paper, re-pointed at this project:

    Dong et al. (Yangtze)          ->   here (Mahanadi above Hirakud)
    2002-2019, May-Aug, 35 inits   ->   2004-2014, JJAS, 31 inits
    30-day lead                    ->   30-day lead   (was 17 -- a download
                                        choice, not an S2S limit)
    10 ensemble members            ->   10 ensemble members
    19 predictors                  ->   the same 19
    CN05.1 rainfall target         ->   IMD rainfall target

WHERE THIS DATA LIVES, because it took three wrong turns to find:

  * NOT the Copernicus CDS (cds.climate.copernicus.eu).  164 datasets, no
    sub-seasonal reforecast among them -- only ERA5 and seasonal products.
  * NOT the legacy MARS archive (api.ecmwf.int/v1) reached by `ecmwfapi`.
    That path returns 403 for datasets/s2s unless the LEGACY licence is
    accepted, which is a different registry from the new one.
  * IT IS the ECMWF Data Store, ECDS: https://ecds.ecmwf.int/api
    A CADS-style API, so the `cdsapi` client works against it once pointed at
    that URL with an ECDS token.

Every field name below was read from the live schema at
`ecds.ecmwf.int/api/retrieve/v1/processes/s2s-reforecasts`, not guessed.  Three
of them differ from the obvious guess and would each have failed silently:
`level_type` is "pressure" (not "pressure_level"), `level_value` is "200_hpa"
(not "200"), and `data_format` accepts grib only (no netcdf).

WHY THE UPPER-AIR FIELDS ARE THE POINT.  Weather is three-dimensional: 850 hPa
(~1.5 km) carries the low-level moisture that becomes rain, 500 hPa (~5.5 km) is
the steering level showing where a system is heading, 200 hPa (~12 km) is the
jet stream organising everything below.  A surface-only forecast cannot say
which 25 km cell a storm lands in -- which is what heavy-rain R2 = -1.25 in all
eleven folds has been telling us (results/metrics/tier1_sweep.json), after
eleven independent attempts to fix it from the model side.

SETUP.  Get an ECDS API token from https://ecds.ecmwf.int -> your profile, and
accept the S2S licence there.  Put it in ~/.ecdsapirc:

    url: https://ecds.ecmwf.int/api
    key: <your ECDS token>

Run:  python preprocessing/download_s2s_dong.py --pilot 1     # verify first
      python preprocessing/download_s2s_dong.py --pilot 0 --workers 3
Out:  data/raw/s2s/{cf,pf}_{sfc,pl}_MM_DD.grib
"""

import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

ECDS_URL = "https://ecds.ecmwf.int/api"
SUFFIX = ""                      # set from --hyears; keeps year ranges separate
RC = Path.home() / ".ecdsapirc"
OUT_DIR = C.RAW / "s2s"

# The S2S reforecast fires Monday/Thursday; these are the JJAS dates the
# existing archive already uses, so old and new align initialisation for
# initialisation.  JJAS is the Indian monsoon -- Dong's May-Aug is the Chinese one.
SCHEDULE = {
    "06": ["03", "10", "17", "24"],
    "07": ["01", "04", "08", "12", "15", "19", "22", "26", "29"],
    "08": ["01", "05", "08", "12", "15", "19", "22", "26", "29"],
    "09": ["02", "05", "09", "12", "16", "19", "23", "26", "30"],
}

# Hindcast years.  Overridable so the record can be extended without touching
# the 2004-2014 archive already on disk: a different range writes files with a
# year suffix, so the two sets sit side by side and build_s2s_archive.py can
# consume either.  The ECDS archive offers 1981-2025.
HYEARS = [str(y) for y in range(C.YEAR_MIN, C.YEAR_MAX + 1)]      # 2004-2014
MODEL_VERSION_YEAR = "2024"          # which model generated the reforecast
LEAD_DAYS = 30                       # was 17
LEADTIME_HOUR = [str(24 * d) for d in range(1, LEAD_DAYS + 1)]

# Dong's surface set, plus the fields already in use so nothing is lost.
SFC_VARIABLES = [
    "total_precipitation",
    "convective_precipitation",      # Dong has it; the current archive does not
    "2_m_temperature",
    "10_m_u_component_of_wind",
    "10_m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_cloud_cover",
    "surface_solar_radiation_downwards",
    "surface_sensible_heat_flux",
    "surface_latent_heat_flux",
    "orography",
]
# 2t and tcc are PERIOD-PROCESSED in the S2S reforecast, not instantaneous.  Asked
# for as hours ("24","48",...) they return a single field at step 0-24 and nothing
# after -- verified with preprocessing/probe_2t_tcc.py, which gets all 30 daily
# windows back once they are requested as periods.  They therefore go out as their
# own small request (kind "sfcp"); the other nine surface fields are unaffected, so
# the sfc files already on disk stay valid.
SFCP_VARIABLES = ["2_m_temperature", "total_cloud_cover"]
PERIOD_LEADTIME = [f"{24*d}_{24*(d+1)}" for d in range(LEAD_DAYS)]

# The nine missing predictors: 3 levels x these 5, minus nothing.
PL_VARIABLES = [
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "temperature",
    "geopotential_height",
]
PL_LEVELS = ["200_hpa", "500_hpa", "850_hpa"]

# Wide box: fixes the 48 % edge-padded 3x3 patches and brings the catchment's
# western headwaters (80.5 E) inside the domain.  Both are current limitations.
AREA = [25.5, 79.5, 16.5, 88.5]      # N, W, S, E


def read_rc():
    if not RC.exists():
        sys.exit(f"no {RC}\n\n"
                 "Get an ECDS token from https://ecds.ecmwf.int (profile page),\n"
                 "accept the S2S licence there, then write:\n\n"
                 f"    url: {ECDS_URL}\n    key: <your ECDS token>\n")
    cfg = {}
    for line in RC.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            cfg[k.strip()] = v.strip()
    return cfg.get("url", ECDS_URL), cfg["key"]


# The reforecast fires Monday and Thursday.  These three came from the old
# archive's schedule, are Fridays in the 2024 model cycle, and ECDS answers them
# with HTTP 400 -- they are impossible, not flaky, so do not spend requests.
UNAVAILABLE = {("07", "12"), ("07", "19"), ("07", "26")}


def build_requests(args):
    dates = [(m, d) for m, days in SCHEDULE.items() for d in days
             if (m, d) not in UNAVAILABLE]
    kinds = ["sfc", "sfcp", "pl"] if args.levels == "both" else [args.levels]
    types = {"control": ["control_forecast"], "perturbed": ["perturbed_forecast"],
             "both": ["control_forecast", "perturbed_forecast"]}[args.members]
    tag = {"control_forecast": "cf", "perturbed_forecast": "pf"}
    return [{"month": m, "day": d, "kind": k, "ftype": ty,
             "name": f"{tag[ty]}_{k}_{m}_{d}{SUFFIX}.grib"}
            for ty in types for m, d in dates for k in kinds]


def build_body(r):
    body = {
        "origin": "ecmwf",
        "year": MODEL_VERSION_YEAR,
        "month": r["month"],
        "day": r["day"],
        "time": "00:00",
        "hyear": HYEARS,
        "hmonth": [r["month"]],
        "hday": [r["day"]],
        "forecast_type": r["ftype"],
        "leadtime_hour": LEADTIME_HOUR,
        "area": AREA,
        "data_format": "grib",
    }
    if r["kind"] == "sfcp":
        body["level_type"] = "single_level"
        body["variable"] = SFCP_VARIABLES
        body["leadtime_hour"] = PERIOD_LEADTIME     # periods, not hours
    elif r["kind"] == "sfc":
        body["level_type"] = "single_level"
        body["variable"] = SFC_VARIABLES
    else:
        body["level_type"] = "pressure"           # NOT "pressure_level"
        body["variable"] = PL_VARIABLES
        body["level_value"] = PL_LEVELS           # NOT bare "200"
    return body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--members", choices=["control", "perturbed", "both"], default="both")
    ap.add_argument("--levels", choices=["sfc", "sfcp", "pl", "both"], default="both",
                    help='"sfcp" is the 2t/tcc period-window pass on its own')
    ap.add_argument("--workers", type=int, default=3,
                    help="ECDS queues per user; a few concurrent requests help, "
                         "many do not")
    ap.add_argument("--pilot", type=int, default=1,
                    help="fetch this many, report, then STOP.  0 runs everything")
    ap.add_argument("--hyears", default=None,
                    help="hindcast year range, e.g. 2015-2022.  Default is the "
                         "config range (2004-2014).  A non-default range suffixes "
                         "every filename so nothing already downloaded is overwritten.")
    args = ap.parse_args()

    global HYEARS, SUFFIX
    SUFFIX = ""
    if args.hyears:
        lo, hi = (int(x) for x in args.hyears.split("-"))
        HYEARS = [str(y) for y in range(lo, hi + 1)]
        SUFFIX = f"_{lo}-{hi}"

    reqs = build_requests(args)
    n_pred = len(SFC_VARIABLES) + len(PL_VARIABLES) * len(PL_LEVELS)
    print(f"{len(reqs)} ECDS requests")
    n_init = sum(len(v) for v in SCHEDULE.values()) - len(UNAVAILABLE)
    print(f"  {n_init} inits/year x {len(HYEARS)} "
          f"hindcast years ({HYEARS[0]}-{HYEARS[-1]})"
          f"   [{len(UNAVAILABLE)} non-Mon/Thu dates excluded]")
    print(f"  leads 1-{LEAD_DAYS} d   members: {args.members}   levels: {args.levels}")
    print(f"  {n_pred} predictor fields ({len(SFC_VARIABLES)} surface + "
          f"{len(PL_VARIABLES)} x {len(PL_LEVELS)} pressure)")
    print(f"  area {AREA}   -> {OUT_DIR}")

    if args.dry_run:
        for r in reqs[:4]:
            print(f"    {r['name']}")
        print(f"    ... and {len(reqs) - 4} more\n  sample body:")
        for k, v in build_body(reqs[1]).items():
            print(f"    {k:16} {str(v)[:84]}")
        return

    url, key = read_rc()
    try:
        import cdsapi
    except ImportError:
        sys.exit("pip install cdsapi")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = [r for r in reqs
            if not ((OUT_DIR / r["name"]).exists()
                    and (OUT_DIR / r["name"]).stat().st_size > 0)]
    print(f"\n{len(reqs) - len(todo)} on disk, {len(todo)} to fetch")
    if not todo:
        return
    if args.pilot:
        todo = todo[:args.pilot]
        print(f"PILOT: {len(todo)} only, then stopping for inspection.")

    lock, done, failed = threading.Lock(), [], []

    def fetch(r):
        target = OUT_DIR / r["name"]
        try:
            cdsapi.Client(url=url, key=key, quiet=True).retrieve(
                "s2s-reforecasts", build_body(r), str(target))
            with lock:
                done.append(r["name"])
                print(f"  [{len(done)}/{len(todo)}] {r['name']}  "
                      f"{target.stat().st_size/1e6:.1f} MB", flush=True)
        except Exception as e:
            with lock:
                failed.append((r["name"], f"{type(e).__name__}: {e}"))
                print(f"  FAILED {r['name']}: {type(e).__name__}: {str(e)[:200]}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(as_completed([ex.submit(fetch, r) for r in todo]))

    print(f"\n{len(done)} ok, {len(failed)} failed")
    for name, err in failed:
        print(f"  {name}: {err[:300]}")
    if args.pilot and not failed:
        print(f"\nPilot OK.  Rerun with --pilot 0 for the remaining "
              f"{len(reqs) - len(done)}.")


if __name__ == "__main__":
    main()
