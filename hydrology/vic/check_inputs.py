"""Report which VIC inputs are present and which are still outstanding.

Run this after each data drop rather than discovering a missing file part-way
through a calibration run.  Checks are on content, not just existence: a file
that is present but covers the wrong period or grid is worse than an absent
one, because it fails silently.

Run:  python hydrology/vic/check_inputs.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs import config as C

VIC_DIR = C.RAW / "vic"
DOMAIN = dict(lat=(C.CATCHMENT_LAT_MIN, C.CATCHMENT_LAT_MAX),
              lon=(C.CATCHMENT_LON_MIN, C.CATCHMENT_LON_MAX))

# One ERA5-Land request supplies temperature, humidity, wind and pressure
# together, so they are tracked as a single item rather than the separate
# temperature/ and wind/ directories the manifest originally assumed.
ITEMS = [
    ("forcing/era5", "ERA5-Land hourly forcing (T, Td, wind, p)",
     "python preprocessing/download_era5_land.py --extras"),
    ("veg/mcd12q1", "MODIS land-cover classification",
     "MCD12Q1 IGBP (LP DAAC, needs a free Earthdata account -- "
     "python preprocessing/build_landcover.py after logging in)"),
    ("routing", "HydroSHEDS DIR/ACC 15 arc-sec rasters",
     "https://data.hydrosheds.org/file/hydrosheds-v1-dir/hyd_as_dir_15s.zip"),
]

# 12 years x 12 months.  A partial ERA5 download is the likeliest failure mode
# here, and a gap would be invisible once the months are concatenated.
ERA5_EXPECTED = 45   # 24 t2m + 12 met + 6 rad + 3 sp (see download_era5_land.STREAMS)


def status(sub: str) -> tuple[bool, str]:
    d = VIC_DIR / sub
    if not d.exists():
        return False, "directory absent"
    files = [f for f in d.rglob("*") if f.is_file()]
    if not files:
        return False, "directory empty"
    size_mb = sum(f.stat().st_size for f in files) / 1e6
    detail = f"{len(files)} file(s), {size_mb:.0f} MB"

    if sub == "forcing/era5":
        # Count months, not files: an in-progress download looks "present"
        # otherwise, and the missing months would only surface much later as a
        # silent gap in the concatenated forcing.
        got = sum(1 for m in d.iterdir()
                  if m.is_dir() and any(f.stat().st_size > 100_000
                                        for f in m.glob("*.nc")))
        detail = f"{got}/{ERA5_EXPECTED} requests, {size_mb:.0f} MB"
        if got < ERA5_EXPECTED:
            return False, detail + "  (incomplete -- re-run the download)"
    return True, detail


def main() -> None:
    print("VIC input check")
    print(f"  domain  {DOMAIN['lat'][0]}-{DOMAIN['lat'][1]} N, "
          f"{DOMAIN['lon'][0]}-{DOMAIN['lon'][1]} E")
    print(f"  period  {C.YEAR_MIN}-{C.YEAR_MAX}")
    print(f"  root    {VIC_DIR}\n")

    # These three are already derived products under data/processed/, not raw
    # drops under data/raw/vic/, so they are checked by file rather than by
    # directory.  Precipitation is checked for the spin-up year specifically:
    # the archive originally began at 2004, and a missing 2003 would only
    # surface once VIC had nothing to spin up on.
    years = {int(p.stem[8:12]) for p in C.IMD_NC_DIR.glob("RF25_ind*")} \
        if C.IMD_NC_DIR.exists() else set()
    need = set(range(C.YEAR_MIN - 1, C.YEAR_MAX + 1))
    gaps = sorted(need - years)
    print(f"  [{'OK ' if not gaps else '-- '}] precipitation      IMD 0.25 deg  "
          + ("all years incl. spin-up" if not gaps
             else f"MISSING {gaps} -- run preprocessing/fetch_imd_2003.py"))

    for label, name, how in [
        # dem_vic.npz, not dem_fine.npz: the latter is on the CNN's 25x25 grid,
        # which starts at 81.0 E and therefore has no elevation for the
        # catchment's 80.50/80.75 headwater columns.
        ("elevation   ", "dem_vic.npz", "preprocessing/build_dem.py --grid vic"),
        ("soil texture", "soilgrids.npz", "preprocessing/build_soilgrids.py"),
        ("basin mask  ", "basin.npz", "preprocessing/build_basin.py"),
        ("routing grid", "routing.npz", "preprocessing/build_routing.py"),
        ("land cover  ", "landcover.npz", "preprocessing/build_landcover.py"),
    ]:
        f = C.PROCESSED / name
        print(f"  [{'OK ' if f.exists() else '-- '}] {label}       "
              f"{name}  ({f'{f.stat().st_size / 1e3:.0f} kB' if f.exists() else f'MISSING -- run {how}'})")

    missing = []
    for sub, what, source in ITEMS:
        ok, detail = status(sub)
        print(f"  [{'OK ' if ok else '-- '}] {what:35s} {detail}")
        if not ok:
            missing.append((sub, what, source))

    if not missing:
        print("\nAll inputs present. Next: build the forcing NetCDF, then the "
              "soil and vegetation parameter files.")
        return

    print(f"\n{len(missing)} item(s) outstanding:\n")
    for sub, what, source in missing:
        print(f"  {what}")
        print(f"    -> {source}")
        print(f"    -> place under {VIC_DIR / sub}/\n")

    print("Full detail, including which parameters are calibrated rather than "
          "measured: hydrology/vic/DATA_REQUIREMENTS.md")


if __name__ == "__main__":
    main()
