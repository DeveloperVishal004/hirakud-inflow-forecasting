"""MODIS MCD12Q1 IGBP land cover -> per-cell class fractions on the VIC grid.

VIC's vegetation parameter file lists, for every grid cell, the fraction of
each land-cover class present and its rooting depths.  The veg library is
conventionally keyed to the IGBP 17-class legend, which is exactly what
MCD12Q1's LC_Type1 provides -- so no crosswalk is needed, unlike ESA WorldCover
or CGLS-LC100.

MCD12Q1 is also the only open land-cover product covering 2001-present.
WorldCover is 2020/2021 and CGLS-LC100 is 2015-2019; both post-date this
project's 2004-2014 record by a decade, over which irrigation in the Mahanadi
basin expanded materially.  Using them would describe a different catchment.

Access.  MCD12Q1 ships as HDF-EOS2 (HDF4), which the installed GDAL cannot
read, so this uses pyhdf directly rather than rasterio.  Rather than warping
the sinusoidal grid, every MODIS pixel centre is inverted to lat/lon and binned
into the target cell -- the same approach as build_soilgrids.py, and exact
where a warp would resample.

Requires a free Earthdata account (https://urs.earthdata.nasa.gov/users/new).
`earthaccess` will prompt on first run and can persist to ~/.netrc.

Run:  python preprocessing/build_landcover.py [--year 2010]
Out:  data/processed/landcover.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

OUT = C.PROCESSED / "landcover.npz"
CACHE = C.RAW / "vic" / "veg" / "mcd12q1"

# IGBP, as VIC's standard vegetation library indexes them.
IGBP = {
    1: "evergreen needleleaf", 2: "evergreen broadleaf", 3: "deciduous needleleaf",
    4: "deciduous broadleaf", 5: "mixed forest", 6: "closed shrubland",
    7: "open shrubland", 8: "woody savanna", 9: "savanna", 10: "grassland",
    11: "permanent wetland", 12: "cropland", 13: "urban",
    14: "cropland/natural mosaic", 15: "snow and ice", 16: "barren", 17: "water",
}
# VIC treats water and urban as non-vegetated; they become bare fraction.
NONVEG = {13, 15, 16, 17}

# MODIS sinusoidal grid constants (500 m product, 2400x2400 per tile).
R_MODIS = 6371007.181
TILE_M = 1111950.5196666666
X0, Y0 = -20015109.354, 10007554.677
NPIX = 2400
PIXEL_M = TILE_M / NPIX


def pixel_lonlat(h: int, v: int):
    """lat/lon of every pixel centre in tile (h, v), by inverting sinusoidal."""
    x = X0 + h * TILE_M + (np.arange(NPIX) + 0.5) * PIXEL_M          # (2400,)
    y = Y0 - v * TILE_M - (np.arange(NPIX) + 0.5) * PIXEL_M          # (2400,)
    lat = np.degrees(y / R_MODIS)                                     # row-wise
    # Longitude depends on latitude in a sinusoidal projection, so it cannot be
    # collapsed to a 1-D axis the way a plate-carree grid can.
    coslat = np.cos(np.radians(lat))
    lon = np.degrees(x[None, :] / (R_MODIS * np.where(coslat < 1e-9, np.nan,
                                                      coslat)[:, None]))
    return np.broadcast_to(lat[:, None], (NPIX, NPIX)), lon


def download(year: int):
    import earthaccess

    CACHE.mkdir(parents=True, exist_ok=True)
    have = sorted(CACHE.glob("MCD12Q1*.hdf"))
    if have:
        print(f"  using {len(have)} cached granule(s)")
        return have

    try:
        earthaccess.login(persist=True)
    except Exception as exc:  # noqa: BLE001
        sys.exit(
            f"Earthdata login failed ({exc}).\n"
            "  1. create a free account at https://urs.earthdata.nasa.gov/users/new\n"
            "  2. re-run; earthaccess will prompt and save to ~/.netrc")

    # Search by bounding box rather than hardcoded tile IDs: the basin spans a
    # tile seam and the delineated divide reaches past the project box.
    results = earthaccess.search_data(
        short_name="MCD12Q1", version="061",
        temporal=(f"{year}-01-01", f"{year}-12-31"),
        bounding_box=(C.CATCHMENT_LON_MIN - 0.5, C.CATCHMENT_LAT_MIN - 0.5,
                      C.CATCHMENT_LON_MAX + 0.5, C.CATCHMENT_LAT_MAX + 0.5),
    )
    if not results:
        sys.exit(f"no MCD12Q1 granules found for {year}")
    print(f"  {len(results)} granule(s) found; downloading")
    earthaccess.download(results, str(CACHE))
    return sorted(CACHE.glob("MCD12Q1*.hdf"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2010,
                    help="a single representative year; land-cover change is "
                         "not being studied here, so one mid-record year is "
                         "the right choice")
    args = ap.parse_args()

    lats = np.arange(C.CATCHMENT_LAT_MIN, C.CATCHMENT_LAT_MAX, C.FINE_RES)
    lons = np.arange(C.CATCHMENT_LON_MIN, C.CATCHMENT_LON_MAX + 1e-9, C.FINE_RES)
    half = C.FINE_RES / 2
    print(f"MCD12Q1 {args.year} -> {len(lats)}x{len(lons)} cells at {C.FINE_RES} deg")

    files = download(args.year)
    from pyhdf.SD import SD, SDC

    counts = np.zeros((len(lats), len(lons), 18), dtype=np.int64)
    for f in files:
        # Tile id is encoded in the filename as .hHHvVV.
        tag = next(p for p in f.name.split(".") if p.startswith("h") and len(p) == 6)
        h, v = int(tag[1:3]), int(tag[4:6])
        sd = SD(str(f), SDC.READ)
        lc = sd.select("LC_Type1")[:]
        sd.end()

        plat, plon = pixel_lonlat(h, v)
        keep = ((plat >= lats[0] - half) & (plat < lats[-1] + half)
                & (plon >= lons[0] - half) & (plon < lons[-1] + half)
                & np.isfinite(plon))
        if not keep.any():
            print(f"  {f.name}  h{h:02d}v{v:02d}  no overlap, skipped")
            continue

        iy = ((plat[keep] - (lats[0] - half)) / C.FINE_RES).astype(int)
        ix = ((plon[keep] - (lons[0] - half)) / C.FINE_RES).astype(int)
        cls = lc[keep].astype(int).clip(0, 17)
        flat = (iy * len(lons) + ix) * 18 + cls
        counts += np.bincount(flat, minlength=counts.size).reshape(counts.shape)
        print(f"  {f.name}  h{h:02d}v{v:02d}  {keep.sum():,} pixels binned")

    total = counts[:, :, 1:].sum(axis=2)
    if total.min() == 0:
        n = int((total == 0).sum())
        sys.exit(f"{n} cell(s) received no MODIS pixels -- a tile is missing")
    frac = counts[:, :, 1:] / total[:, :, None]        # classes 1..17

    print(f"\n=== catchment composition ({args.year}) ===")
    order = np.argsort(-frac.mean(axis=(0, 1)))
    for k in order[:8]:
        m = frac[:, :, k].mean()
        if m > 0.002:
            print(f"  {IGBP[k + 1]:26s} {100 * m:5.1f} %")
    veg = 1.0 - sum(frac[:, :, c - 1] for c in NONVEG)
    print(f"\n  vegetated fraction   mean {100 * veg.mean():.1f} %  "
          f"range {100 * veg.min():.1f}-{100 * veg.max():.1f} %")
    # ~3,345 expected: a 0.25 deg cell at this latitude is ~718 km2 and the
    # MODIS "500 m" pixel is really 463.31 m (0.2147 km2).  Using the nominal
    # 500 m would predict ~2,900 and make a correct count look wrong.
    print(f"  pixels per 0.25 deg cell: {total.min():,} - {total.max():,} "
          f"(expected ~3,345 at the true 463.31 m pixel)")

    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT, lats=lats, lons=lons, fraction=frac,
        classes=np.array([IGBP[i] for i in range(1, 18)]),
        veg_fraction=veg, year=args.year,
        source=f"MODIS MCD12Q1.061 LC_Type1 (IGBP), {args.year}",
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
