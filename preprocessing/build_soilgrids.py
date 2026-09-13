"""Fetch SoilGrids 250 m texture and bulk density, aggregate to the VIC grid.

VIC's soil parameter file wants sand/silt/clay fractions and bulk density per
layer per cell.  SoilGrids (ISRIC) is machine-learned from ~240k profiles and
resolves the basin's texture contrasts; HWSD's India coverage derives from
older 1:5M FAO mapping and would smear them.

Grid: 0.25 deg over the catchment only (19.75-23.50 N, 80.50-84.25 E, 16x16).
Deliberately NOT the 25x25 domain of dem_fine.npz -- that grid exists because
the DEM is a CNN predictor over the whole downscaling domain, whereas soil is a
VIC parameter and VIC runs only on the catchment.  Fetching 6x the area would
cost 6x the download for cells that are never used.

Depth mapping.  SoilGrids reports six standard intervals; VIC 5 conventionally
runs three layers.  They are combined by thickness weighting:

    VIC layer 1   0-15 cm     <- SoilGrids 0-5, 5-15        (fast response)
    VIC layer 2   15-60 cm    <- SoilGrids 15-30, 30-60     (root zone)
    VIC layer 3   60-200 cm   <- SoilGrids 60-100, 100-200  (baseflow store)

Units.  SoilGrids stores integers to save space: texture in g/kg (divide by 10
for percent) and bulk density in cg/cm3 (divide by 100 for g/cm3).  Applying
these is not optional -- raw values would put sand at ~230 "percent".

Nodata.  The WCS returns 0 rather than a nodata flag.  Zero is implausible for
all four properties (no soil is 0 % sand AND 0 % silt AND 0 % clay), so 0 is
treated as missing.  Cells that are entirely missing -- open water, chiefly the
Hirakud reservoir itself -- are filled from the catchment median, since VIC
still needs a value there and the alternative is a NaN that propagates.

Run:  python preprocessing/build_soilgrids.py
Out:  data/processed/soilgrids.npz
"""

import io
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

WCS = "https://maps.isric.org/mapserv?map=/map/{prop}.map"

# property -> (divisor to reach the unit VIC wants, unit label)
# SoilGrids stores integers: texture in g/kg, bulk density in cg/cm3, and
# organic carbon in dg/kg.  soc is needed because the Saxton & Rawls (2006)
# pedotransfer functions take organic matter as an explicit term -- omitting it
# biases water retention low in the topsoil, where OM is highest.
PROPERTIES = {
    "sand": (10.0, "%"),
    "silt": (10.0, "%"),
    "clay": (10.0, "%"),
    "bdod": (100.0, "g/cm3"),
    "soc": (100.0, "%"),
}

DEPTHS = ["0-5", "5-15", "15-30", "30-60", "60-100", "100-200"]
THICKNESS = {"0-5": 5, "5-15": 10, "15-30": 15,
             "30-60": 30, "60-100": 40, "100-200": 100}

VIC_LAYERS = [
    ("layer1_0-15cm",   ["0-5", "5-15"]),
    ("layer2_15-60cm",  ["15-30", "30-60"]),
    ("layer3_60-200cm", ["60-100", "100-200"]),
]

RES = C.FINE_RES
LATS = np.arange(C.CATCHMENT_LAT_MIN, C.CATCHMENT_LAT_MAX, RES)
LONS = np.arange(C.CATCHMENT_LON_MIN, C.CATCHMENT_LON_MAX + 1e-9, RES)
BBOX = (LONS[0] - RES / 2, LATS[0] - RES / 2,
        LONS[-1] + RES / 2, LATS[-1] + RES / 2)   # W, S, E, N

OUT = C.PROCESSED / "soilgrids.npz"


def fetch(prop: str, depth: str, retries: int = 4):
    """One coverage over the catchment bbox, returned in EPSG:4326."""
    import rasterio

    params = {
        "SERVICE": "WCS", "VERSION": "2.0.1", "REQUEST": "GetCoverage",
        "COVERAGEID": f"{prop}_{depth}cm_mean", "FORMAT": "image/tiff",
        "SUBSET": [f"X({BBOX[0]},{BBOX[2]})", f"Y({BBOX[1]},{BBOX[3]})"],
        # SoilGrids is stored in Interrupted Goode Homolosine; asking the
        # server to subset and deliver in 4326 avoids reprojecting here.
        "SUBSETTINGCRS": "http://www.opengis.net/def/crs/EPSG/0/4326",
        "OUTPUTCRS": "http://www.opengis.net/def/crs/EPSG/0/4326",
    }
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(WCS.format(prop=prop), params=params, timeout=600)
            r.raise_for_status()
            if "tiff" not in (r.headers.get("Content-Type") or ""):
                raise RuntimeError(f"expected tiff, got {r.headers.get('Content-Type')}: "
                                   f"{r.content[:200]!r}")
            with rasterio.io.MemoryFile(r.content) as m, m.open() as src:
                return src.read(1).astype("float64"), src.transform, src.shape
        except Exception as exc:  # noqa: BLE001 -- ISRIC rate-limits under load
            last = exc
            wait = 5 * (attempt + 1)
            print(f"      retry {attempt + 1}/{retries} in {wait}s ({type(exc).__name__})")
            time.sleep(wait)
    raise RuntimeError(f"{prop} {depth} failed after {retries} attempts") from last


def aggregate(arr, transform, shape):
    """Mean of the 250 m pixels falling in each 0.25 deg cell."""
    ny, nx = shape
    # Pixel centres from the affine transform, then bin by cell edges.
    px_lon = transform.c + transform.a * (np.arange(nx) + 0.5)
    px_lat = transform.f + transform.e * (np.arange(ny) + 0.5)
    ix = np.digitize(px_lon, LONS + RES / 2)
    iy = np.digitize(px_lat, LATS + RES / 2)

    valid = arr > 0                       # 0 is the WCS's stand-in for nodata
    out = np.full((len(LATS), len(LONS)), np.nan)
    cover = np.zeros_like(out)
    IY, IX = np.meshgrid(iy, ix, indexing="ij")
    inside = (IY < len(LATS)) & (IX < len(LONS))
    flat_cell = (IY * len(LONS) + IX)[inside]

    tot = np.bincount(flat_cell, weights=(arr * valid)[inside],
                      minlength=out.size)
    cnt = np.bincount(flat_cell, weights=valid[inside].astype(float),
                      minlength=out.size)
    n = np.bincount(flat_cell, minlength=out.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(cnt > 0, tot / cnt, np.nan).reshape(out.shape)
        cover = np.where(n > 0, cnt / n, 0.0).reshape(out.shape)
    return out, cover


def main() -> None:
    print(f"SoilGrids -> {len(LATS)}x{len(LONS)} cells at {RES} deg")
    print(f"  bbox   W{BBOX[0]:.3f} S{BBOX[1]:.3f} E{BBOX[2]:.3f} N{BBOX[3]:.3f}")
    print(f"  fetch  {len(PROPERTIES)} properties x {len(DEPTHS)} depths "
          f"= {len(PROPERTIES) * len(DEPTHS)} requests\n")

    raw, cover = {}, None
    for prop, (div, unit) in PROPERTIES.items():
        for depth in DEPTHS:
            t0 = time.time()
            arr, transform, shape = fetch(prop, depth)
            grid, cov = aggregate(arr, transform, shape)
            grid /= div
            raw[(prop, depth)] = grid
            cover = cov if cover is None else np.minimum(cover, cov)
            print(f"  {prop:5s} {depth:8s}cm  {shape[0]}x{shape[1]} px -> "
                  f"{np.nanmean(grid):7.2f} {unit:7s} "
                  f"({np.nanmin(grid):.1f}-{np.nanmax(grid):.1f})  "
                  f"{time.time() - t0:.0f}s")

    # ---- collapse six SoilGrids depths into three VIC layers
    out = {"lats": LATS, "lons": LONS, "valid_fraction": cover,
           "source": "SoilGrids 250m (ISRIC) via WCS, EPSG:4326"}
    for name, depths in VIC_LAYERS:
        w = np.array([THICKNESS[d] for d in depths], dtype=float)
        w /= w.sum()
        for prop in PROPERTIES:
            stack = np.stack([raw[(prop, d)] for d in depths])
            out[f"{prop}_{name}"] = np.tensordot(w, stack, axes=1)
        out[f"thickness_{name}"] = sum(THICKNESS[d] for d in depths) / 100.0  # m

    # ---- validation: texture must close to 100 %
    print("\n=== checks ===")
    ok = True
    for name, _ in VIC_LAYERS:
        soc = out[f"soc_{name}"]
        print(f"  {name:16s} organic carbon  {np.nanmin(soc):.2f}-{np.nanmax(soc):.2f} %")
        if np.nanmax(soc) > 20.0:
            print("    FAIL: organic carbon implausible for mineral soil")
            ok = False
        tot = sum(out[f"{p}_{name}"] for p in ("sand", "silt", "clay"))
        dev = np.nanmax(np.abs(tot - 100.0))
        print(f"  {name:16s} sand+silt+clay  mean {np.nanmean(tot):6.2f} %  "
              f"max deviation {dev:5.2f} %")
        if dev > 3.0:
            print("    FAIL: texture does not close -- check the unit divisor")
            ok = False
        b = out[f"bdod_{name}"]
        print(f"  {name:16s} bulk density    {np.nanmin(b):.2f}-{np.nanmax(b):.2f} g/cm3")
        if np.nanmin(b) < 0.7 or np.nanmax(b) > 2.0:
            print("    FAIL: bulk density outside a physical range for mineral soil")
            ok = False

    nan_cells = int(np.isnan(out["sand_layer1_0-15cm"]).sum())
    print(f"  cells with no soil data: {nan_cells}/{LATS.size * LONS.size} "
          f"(open water -- filled from the catchment median)")
    for k in list(out):
        if isinstance(out[k], np.ndarray) and out[k].shape == (len(LATS), len(LONS)):
            m = np.isnan(out[k])
            if m.any():
                out[k] = np.where(m, np.nanmedian(out[k]), out[k])

    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **out)
    print(f"\n{'PASS' if ok else 'FAIL'} -- wrote {OUT}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
