"""Download the 10-member ECMWF S2S perturbed-forecast ensemble.

The current reforecast is `control_forecast` -- one deterministic member.  Dong
et al. (2025) use all 10 perturbed members, and the ensemble matters for two
independent reasons:

  1. The ensemble MEAN is a better forcing than the control.  Averaging cancels
     unpredictable small-scale detail while retaining the predictable signal,
     which is exactly the regime that collapsed beyond lead 7 here (ECMWF
     control correlation 0.06-0.22 at leads 8-17).
  2. The ensemble SPREAD is a physically-grounded uncertainty estimate.  The
     current intervals are widened by conformal calibration on past residuals,
     which cannot know that a particular forecast is unusually uncertain.
     Spread can, and it is the standard fix for bands that are too narrow ahead
     of a large event.

Requires a CDS account and `~/.cdsapirc`:  https://cds.climate.copernicus.eu
    pip install cdsapi

The initialisation dates match the existing control download exactly, so the
members align row-for-row with `ecmwf_s2s_reforecast_final.csv`.

Run:  python preprocessing/download_ecmwf_ensemble.py [--dry-run]
Out:  data/raw/ensemble/*.grib
"""

import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

OUT_DIR = C.RAW / "ensemble"

# Same 31 initialisations per year as the control download.
SCHEDULE = {
    "06": ["03", "10", "17", "24"],
    "07": ["01", "04", "08", "12", "15", "19", "22", "26", "29"],
    "08": ["01", "05", "08", "12", "15", "19", "22", "26", "29"],
    "09": ["02", "05", "09", "12", "16", "19", "23", "26", "30"],
}

# Only the overlap years are useful: Hirakud inflow ends in 2014.
HYEARS = [str(y) for y in range(C.YEAR_MIN, C.YEAR_MAX + 1)]

# Surface fields.  `convective_precipitation` is new: Dong et al. use it and we
# did not have it -- convective rain (thunderstorm updrafts) behaves differently
# from broad frontal rain, and the distinction matters for heavy events.
SINGLE_VARIABLES = [
    "total_precipitation",
    "convective_precipitation",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_cloud_cover",
    "surface_solar_radiation_downwards",
    "surface_sensible_heat_flux",
    "surface_latent_heat_flux",
]

# The nine predictors we are missing, and the reason the downscaler is stuck.
# Weather is three-dimensional: 850 hPa (~1.5 km) carries the low-level moisture
# that becomes rain, 500 hPa (~5.5 km) is the steering level that shows where a
# system is heading, 200 hPa (~12 km) is the jet stream organising everything
# below it.  A surface-only forecast cannot say which 25 km cell a storm lands
# in -- which is exactly what heavy-rain R2 = -1.25 in all 11 folds is telling
# us, after eleven attempts to fix it from the model side.
PRESSURE_VARIABLES = [
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "temperature",
    "geopotential",
]
PRESSURE_LEVELS = ["200", "500", "850"]

EXT = {"netcdf": "nc", "grib": "grib"}

LEADTIME_HOURS = [str(24 * d) for d in range(C.LEAD_MIN, C.LEAD_MAX + 1)]

# A margin ring beyond the current 18-24N / 81-87E box.  48 % of fine cells
# currently get an edge-padded 3x3 patch, and the catchment's western edge
# (80.5 E) falls outside the domain entirely -- worth fixing while re-downloading.
# Two boxes, and the choice has consequences well beyond the download.
#
#   CURRENT  matches the 5x5 grid already on disk.  Pressure-level fields
#            downloaded on this box drop into the existing pipeline as extra
#            CHANNELS -- no config change, no re-indexing, and the single-level
#            data already held stays valid.  Strictly additive.
#
#   WIDE     a margin ring that fixes the 48 % edge-padded patches and brings
#            the western headwaters (80.5 E) inside the domain.  But it makes
#            the coarse grid 7x7, which invalidates COARSE_LATS/COARSE_LONS,
#            OCEAN_CELLS, the patch extraction and every cached array -- AND it
#            requires re-downloading the single-level fields for the new ring,
#            because what is on disk only covers the current box.
AREA_CURRENT = [24.0, 81.0, 18.0, 87.0]   # N, W, S, E -- the 5x5 grid on disk
AREA_WIDE = [25.5, 79.5, 16.5, 88.5]


def build_requests(args):
    """Every (member, level-type, date) combination we intend to fetch."""
    dates = [(m, d) for m, days in SCHEDULE.items() for d in days]
    kinds = ["single", "pressure"] if args.levels == "both" else [args.levels]
    members = (["control", "perturbed"] if args.members == "both" else [args.members])
    out = []
    for mem in members:
        tag = "ctl" if mem == "control" else "ens"
        ftype = "control_forecast" if mem == "control" else "perturbed_forecast"
        for month, day in dates:
            for kind in kinds:
                out.append({
                    "name": f"{tag}_{kind}_{month}_{day}.{EXT[args.format]}",
                    "month": month, "day": day, "kind": kind, "ftype": ftype,
                })
    return out


def build_body(r, args):
    body = {
        "origin": "ecmwf",
        "year": "2024",
        "month": r["month"],
        "day": r["day"],
        "time": "00:00",
        "hyear": HYEARS,
        "hmonth": [r["month"]],
        "hday": [r["day"]],
        "forecast_type": r["ftype"],
        "leadtime_hour": LEADTIME_HOURS,
        "area": AREA,
        "format": args.format,
    }
    if r["kind"] == "single":
        body["level_type"] = "single_level"
        body["variable"] = SINGLE_VARIABLES
    else:
        body["level_type"] = "pressure_level"
        body["variable"] = PRESSURE_VARIABLES
        body["pressure_level"] = PRESSURE_LEVELS
    return body


def inspect(path: Path) -> str:
    """One-line description of what actually landed, so a bad request is obvious."""
    try:
        import xarray as xr
        ds = xr.open_dataset(path)
        dims = ", ".join(f"{k}={v}" for k, v in ds.sizes.items())
        return f"vars={list(ds.data_vars)} dims=({dims})"
    except Exception as e:                       # netcdf reader may not apply
        return f"[{path.stat().st_size/1e6:.1f} MB, not readable as netcdf: {type(e).__name__}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="list the requests without downloading")
    ap.add_argument("--members", choices=["control", "perturbed", "both"],
                    default="both",
                    help="control = 1 member; perturbed = the 10-member ensemble; "
                         "both = all 11")
    ap.add_argument("--levels", choices=["single", "pressure", "both"], default="both",
                    help="pressure = the 9 predictors we are missing")
    ap.add_argument("--area", choices=["current", "wide"], default="wide",
                    help="current = the 5x5 grid already on disk (additive, no "
                         "rework); wide = fixes edge padding but forces a re-grid")
    ap.add_argument("--hyears", choices=["overlap", "all"], default="all",
                    help="overlap = 2004-2014; all = 2004-2023")
    ap.add_argument("--format", choices=["netcdf", "grib"], default="netcdf",
                    help="netcdf reads natively with xarray; grib needs eccodes")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent CDS requests.  The queue is per-user, so "
                         "beyond a handful this stops helping and risks throttling")
    ap.add_argument("--pilot", type=int, default=2,
                    help="fetch this many first, print what landed, then STOP. "
                         "0 runs everything.  Never commit a long queue to a "
                         "request shape nobody has verified")
    args = ap.parse_args()

    global AREA, HYEARS
    AREA = AREA_CURRENT if args.area == "current" else AREA_WIDE
    if args.hyears == "all":
        HYEARS = [str(y) for y in range(C.YEAR_MIN, 2024)]

    reqs = build_requests(args)
    print(f"{len(reqs)} CDS requests  ({len(HYEARS)} hindcast years "
          f"{HYEARS[0]}-{HYEARS[-1]}, leads {C.LEAD_MIN}-{C.LEAD_MAX})")
    print(f"members: {args.members}   levels: {args.levels}   "
          f"area: {args.area} {AREA}   format: {args.format}")
    print(f"output -> {OUT_DIR}")

    if args.dry_run:
        for r in reqs[:5]:
            print(f"  would request {r['name']}")
        print(f"  ... and {len(reqs) - 5} more")
        return

    try:
        import cdsapi
    except ImportError:
        sys.exit("pip install cdsapi, and configure ~/.cdsapirc first")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = [r for r in reqs
            if not ((OUT_DIR / r["name"]).exists()
                    and (OUT_DIR / r["name"]).stat().st_size > 0)]
    print(f"{len(reqs) - len(todo)} already on disk, {len(todo)} to fetch")
    if not todo:
        print("nothing to do")
        return

    if args.pilot:
        todo = todo[:args.pilot]
        print(f"\nPILOT: fetching {len(todo)} only, then stopping for inspection.")

    lock = threading.Lock()
    done, failed = [], []

    def fetch(r):
        target = OUT_DIR / r["name"]
        # One client per thread: the CDS client holds per-request state and is
        # not documented as thread-safe.
        client = cdsapi.Client(quiet=True)
        try:
            client.retrieve("s2s-reforecasts", build_body(r, args), str(target))
            with lock:
                done.append(r["name"])
                print(f"  [{len(done)}/{len(todo)}] {r['name']}  "
                      f"{target.stat().st_size/1e6:.1f} MB", flush=True)
        except Exception as e:
            with lock:
                failed.append((r["name"], f"{type(e).__name__}: {e}"))
                print(f"  FAILED {r['name']}: {type(e).__name__}: {e}", flush=True)

    print(f"\nrunning {min(args.workers, len(todo))} concurrent requests ...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(as_completed([ex.submit(fetch, r) for r in todo]))

    print(f"\n{len(done)} ok, {len(failed)} failed")
    for name, err in failed:
        print(f"  {name}: {err}")
    for name in done:
        print(f"\n{name}\n  {inspect(OUT_DIR / name)}")

    if args.pilot and not failed:
        print(f"\nPilot OK.  If the variables and dimensions above look right, "
              f"rerun with --pilot 0 to fetch the remaining {len(reqs) - len(done)}.")


if __name__ == "__main__":
    main()
