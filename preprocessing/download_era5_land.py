"""Download ERA5-Land forcing for the VIC setup, in as few requests as CDS allows.

The naive shape -- 8 variables, hourly, one month per request -- needs 144
downloads, and CDS permits only ONE concurrent request per user, so those 144
queue waits run strictly end to end.  That is a multi-day job.

Both limits were probed rather than assumed (see REQUEST PLANNING below), and
neither can be argued with: ~5,952 fields per request, one request at a time.
Since requests = total-fields / 5,952, the only lever is total fields -- and the
saving comes from noticing that the variables do not all need hourly data.
Splitting them by required time resolution brings the job to 36 requests.

Coverage rationale:

  years    2003-2014.  2003 is spin-up: VIC's soil moisture at the start of the
           2004 monsoon has to be produced by the model, not assumed.
  months   all 12, not just the monsoon.  VIC is a continuous water balance;
           the pre-monsoon drying is what sets the state the monsoon starts
           from.  Same reason the IMD series are built full-year.

Radiation and pressure ARE included, in the `rad` stream.  An earlier version
of this file omitted them on the grounds that MTCLIM derives radiation from the
diurnal temperature range -- that is VIC 4 behaviour.  VIC 5 removed MTCLIM
(release notes, GH#288) and vic_force.c now aborts unless shortwave, longwave
and pressure are all supplied.  They are ACCUMULATED fields, so they need the
00Z-reset handling described under STREAMS.

Area is deliberately wider than the catchment (19.75-23.60 N, 80.50-84.25 E):
regridding 0.1 deg -> 0.25 deg needs neighbours on every side, and the
headwater cells on the western edge are where runoff is generated.

Setup:
    pip install cdsapi
    # then put your CDS API key in ~/.cdsapirc (see the "How to use" tab)

Run:
    python preprocessing/download_era5_land.py                    # 36 requests
    python preprocessing/download_era5_land.py --streams t2m met extra
    python preprocessing/download_era5_land.py --status
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

OUT_DIR = C.RAW / "vic" / "forcing" / "era5"

# North / West / South / East, as CDS expects.  Catchment is 19.75-23.60 N,
# 80.50-84.25 E; the 0.25 deg margin on every side gives the regridding step
# neighbours for the edge cells, which on the west are the headwaters.
AREA = [23.75, 80.25, 19.5, 84.5]

# Spin-up year plus the inflow record.
# Default is the v1 window (2003 spin-up + 2004-2014).  --years extends it:
# the inflow record and the S2S archive now reach 2022, so the observed
# meteorology has to as well or the extended monsoons have a target and no
# predictors.  Filenames carry the year, so ranges never collide.
YEARS = list(range(C.YEAR_MIN - 1, C.YEAR_MAX + 1))

# VIC forcing proper.  All instantaneous -- no de-accumulation needed.
VARIABLES = [
    "2m_temperature",           # -> daily Tmax / Tmin (VIC cannot run without it)
    "2m_dewpoint_temperature",  # -> humidity; beats MTCLIM's Tmin ~= dewpoint
    "10m_u_component_of_wind",  # -> wind speed, with v
    "10m_v_component_of_wind",
    "surface_pressure",         # instantaneous, small; MTCLIM would otherwise
                                # infer it from elevation alone
]

# ---------------------------------------------------------------------------
# REQUEST PLANNING
#
# Two hard CDS limits, both probed directly rather than assumed:
#
#   cost        ~5,952 fields per request, where fields = variables x months
#               x days x hours.  8 vars x 1 month x 31 x 24 = 5,952 passes;
#               5 vars x 2 months = 7,440 is refused with "cost limits
#               exceeded".
#   concurrency ONE.  "The maximum number of per-user requests that access the
#               CDS-MARS data is 1" -- submitting 6 in parallel got all of them
#               rejected.  Threads cannot help; only fewer requests can.
#
# Requests therefore equal total-fields / 5,952, and the only lever is total
# fields.  The saving comes from noticing that the variables do not all need
# the same time resolution:
#
#   t2m          HOURLY.  Tmax/Tmin are diurnal EXTREMES; subsampling clips
#                both ends, and MTCLIM derives radiation and humidity from the
#                range.  Irreducible.
#   d2m, u, v    6-HOURLY.  VIC consumes these only as DAILY MEANS, and four
#                samples a day give an essentially unbiased mean of a roughly
#                sinusoidal diurnal cycle.
#
# 24 + 12 = 36 requests, against 144 for a flat hourly 8-variable fetch.
#
# A request may span multiple YEARS as well as months -- the CDS web form's
# single-select year box is a form limitation, not an API one.  Splitting by
# years as well as months lets each stream fill the 5,952-field budget more
# completely: radiation at 3 variables x 1 year is only 4,464, wasting a
# quarter of the allowance, so it is split into a 2-variable stream at 2 years
# per request (5,952, exactly full) and pressure alone at 4 years (5,952).
# That is 9 requests where the naive one-year-per-request shape needs 12.
#
# name -> (variables, hours, months per request, years per request)
STREAMS = {
    "t2m": (["2m_temperature"], "hourly", 6, 1),
    "met": (["2m_dewpoint_temperature", "10m_u_component_of_wind",
             "10m_v_component_of_wind"], "6hourly", 12, 1),
    # REQUIRED, contrary to what this file previously assumed.  VIC 5 removed
    # MTCLIM (release notes, GH#288): "VIC forcings are now required to be
    # provided at the same time frequency as the model will be run at."  So
    # radiation and pressure are no longer derived internally -- vic_force.c
    # aborts with "Downward shortwave radiation must be supplied as a forcing"
    # and likewise for longwave and pressure.  That was a VIC 4 behaviour.
    #
    # Both radiation fields are ACCUMULATED from 00 UTC and reset daily, so the
    # daily total is the value at 00Z of the FOLLOWING day -- which 6-hourly
    # sampling retains, since it includes 00Z.  Same convention already
    # verified for tp.
    "rad": (["surface_solar_radiation_downwards",
             "surface_thermal_radiation_downwards"], "6hourly", 12, 2),
    "sp":  (["surface_pressure"], "6hourly", 12, 4),
    # Not VIC forcing.  Soil moisture is a candidate Ridge feature and tp an
    # independent cross-check on IMD; both are optional, so they are a separate
    # stream that can be skipped or added later without redoing the rest.
    # tp is ACCUMULATED and its daily total sits at 00Z, which 6-hourly retains.
    "extra": (["volumetric_soil_water_layer_1",
               "volumetric_soil_water_layer_4", "total_precipitation"],
              "6hourly", 12, 1),
}

# Not forcing.  These are ERA5-Land's OWN land-surface output, so feeding them
# to VIC would be circular -- VIC exists to compute soil moisture, not to be
# handed another model's answer.  They earn their place for two other jobs:
#
#   volumetric_soil_water_layer_*  deep-layer storage has multi-week memory,
#       the timescale the 8-17 day leads need, so it is a candidate Ridge
#       feature -- and later a reference to validate VIC's own soil moisture
#       against.  If used as a feature it MUST be sampled at the init date,
#       never the valid date (same discipline as LagSource in train_gbm.py);
#       valid-date sampling is future information.
#   total_precipitation  ACCUMULATED from 00 UTC, resetting daily -- the same
#       class as the ECMWF tp/ssr bug fixed in build_forcing.py.  Raw hourly
#       values are running totals, not hourly rain.  Useful only as an
#       independent cross-check on IMD, which stays the observational target.
DAYS = [f"{d:02d}" for d in range(1, 32)]
HOURS = {
    "hourly": [f"{h:02d}:00" for h in range(24)],
    "6hourly": [f"{h:02d}:00" for h in range(0, 24, 6)],
}
COST_LIMIT = 5952


def plan(streams):
    """Every (stream, years, months) request, with its field cost checked."""
    jobs = []
    for name in streams:
        vars_, hkey, mo_per, yr_per = STREAMS[name]
        hours = HOURS[hkey]
        cost = len(vars_) * mo_per * yr_per * 31 * len(hours)
        if cost > COST_LIMIT:
            raise ValueError(f"stream {name} costs {cost} > {COST_LIMIT} fields")
        months = list(range(1, 13))
        for y0 in range(0, len(YEARS), yr_per):
            years = YEARS[y0:y0 + yr_per]
            for k in range(0, 12, mo_per):
                jobs.append((name, years, months[k:k + mo_per]))
    return jobs




def request_for(name: str, years: list[int], months: list[int]) -> dict:
    vars_, hkey, _, _ = STREAMS[name]
    # Days 29-31 are accepted for every month; CDS ignores those that do not
    # exist, so no per-month calendar logic is needed.
    return {
        "variable": vars_,
        "year": [str(y) for y in years],
        "month": [f"{m:02d}" for m in months],
        "day": DAYS,
        "time": HOURS[hkey],
        "area": AREA,
        "data_format": "netcdf",
        # zip, not unarchived: CDS splits the response whenever the requested
        # variables do not share one grid/step layout.  Asking for unarchived
        # then fails at the server rather than here.
        "download_format": "zip",
    }


def dest_for(name: str, years: list[int], months: list[int]) -> Path:
    span = f"{years[0]}" if len(years) == 1 else f"{years[0]}-{years[-1]}"
    return OUT_DIR / f"{name}_{span}_{months[0]:02d}-{months[-1]:02d}"


def done(dest: Path) -> bool:
    # An interrupted download leaves an empty or truncated file behind, and
    # counting that as present would put a silent gap in the forcing that VIC
    # would run straight through.  Require real NetCDF bytes.
    if not dest.is_dir():
        return False
    return any(f.stat().st_size > 100_000 for f in dest.glob("*.nc"))


def fetch(client, name: str, years: list[int], months: list[int]) -> Path:
    """Retrieve one request and extract the archive into its own directory."""
    import zipfile

    dest = dest_for(name, years, months)
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / "_download.zip"
    client.retrieve("reanalysis-era5-land",
                    request_for(name, years, months)).download(str(tmp))

    if zipfile.is_zipfile(tmp):
        with zipfile.ZipFile(tmp) as z:
            for member in z.namelist():
                if member.endswith(".nc"):
                    # Flatten: CDS nests under a generated directory name that
                    # differs between requests.
                    (dest / Path(member).name).write_bytes(z.read(member))
        tmp.unlink()
    else:
        tmp.rename(dest / f"{name}_{years[0]}.nc")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="+", default=["t2m", "met", "rad", "sp"],
                    choices=list(STREAMS),
                    help="t2m and met are the VIC forcing; extra adds soil "
                         "moisture and precipitation, which are optional and "
                         "can be fetched later without redoing the rest")
    ap.add_argument("--years", default=None,
                    help="year range to fetch, e.g. 2015-2022.  Default is the "
                         "config window (2003-2014).")
    ap.add_argument("--status", action="store_true",
                    help="report what is present and what is missing, then exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan without contacting CDS")
    args = ap.parse_args()
    if args.years:
        lo, hi = (int(x) for x in args.years.split("-"))
        global YEARS
        YEARS = list(range(lo, hi + 1))
        print(f"  year override: {YEARS[0]}-{YEARS[-1]}")

    jobs = plan(args.streams)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"ERA5-Land -> {OUT_DIR}")
    print(f"  area     {AREA[0]} N  {AREA[2]} S  {AREA[1]} W  {AREA[3]} E")
    for name in args.streams:
        vars_, hkey, mo, yr = STREAMS[name]
        n = sum(1 for j in jobs if j[0] == name)
        cost = len(vars_) * mo * yr * 31 * len(HOURS[hkey])
        print(f"  {name:6s}  {len(vars_)} var(s)  {hkey:8s}  {yr}y x {mo}mo  "
              f"{cost:5,}/{COST_LIMIT} fields  -> {n:2d} requests")
    print(f"  TOTAL    {len(jobs)} requests\n")

    if args.dry_run:
        print(request_for(*jobs[0]))
        return

    missing = [j for j in jobs if not done(dest_for(*j))]
    have = len(jobs) - len(missing)
    if args.status:
        size = sum(f.stat().st_size for f in OUT_DIR.rglob("*.nc")) / 1e9
        print(f"  present {have}/{len(jobs)}  ({size:.1f} GB)")
        for name in args.streams:
            js = [j for j in jobs if j[0] == name]
            row = "".join("." if done(dest_for(*j)) else "x" for j in js)
            print(f"    {name:6s} {row}")
        print("\n  '.' present   'x' missing")
        return

    print(f"  {have} already present, {len(missing)} to fetch")
    # Strictly sequential: CDS permits ONE concurrent per-user MARS request,
    # so a thread pool gets every request rejected rather than speeding it up.
    print("  (sequential -- CDS allows only 1 concurrent request per user)\n")

    import cdsapi
    client = cdsapi.Client()

    failed = []
    for i, job in enumerate(missing, 1):
        name, years, months = job
        span = f"{years[0]}" if len(years) == 1 else f"{years[0]}-{years[-1]}"
        label = f"{name} {span} {months[0]:02d}-{months[-1]:02d}"
        print(f"  [{i}/{len(missing)}] {label} queued ...", flush=True)
        try:
            dest = fetch(client, *job)
        except Exception as exc:  # noqa: BLE001 -- one bad request must not end the run
            # CDS times out and rate-limits under load.  Record and continue;
            # re-running the script picks up whatever is still missing.
            print(f"      FAILED: {type(exc).__name__}: {str(exc)[:150]}")
            failed.append(job)
            continue
        mb = sum(f.stat().st_size for f in dest.glob("*.nc")) / 1e6
        print(f"      done ({mb:.0f} MB)")

    if failed:
        print(f"\n{len(failed)} request(s) failed: "
              + ", ".join(f"{n} {y[0]} {m[0]:02d}" for n, y, m in failed))
        print("Re-run the script to retry only those.")
        # Non-zero so an unattended retry loop can tell "finished" from
        # "stopped with gaps" -- a silent 0 here would let a watchdog conclude
        # the job was done while months were still missing.
        sys.exit(1)
    print("\nAll requests complete.")
    print("Next: python hydrology/vic/check_inputs.py")


if __name__ == "__main__":
    main()
