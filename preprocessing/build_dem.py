"""Fetch a DEM and aggregate it to the 0.25 deg fine grid as a predictor.

The only terrain information the model currently sees is ECMWF `orog` at 1.5 deg
-- one value per 165 x 165 km cell.  Over a basin that falls from the Chhattisgarh
uplands to the Mahanadi valley, that is far too coarse to explain why one 0.25 deg
cell rains more than its neighbour.  Dong et al. (2025) include surface elevation
among their 19 predictors for this reason.

Aggregation matters as much as the source.  A single sampled elevation per cell
throws away the sub-grid relief that actually forces orographic rainfall, so each
fine cell gets four statistics: mean, standard deviation (roughness), min and max.

A local DEM in data/raw/dem/ is used if present (.nc or .tif).  Otherwise the
script tries open OPeNDAP endpoints that need no API key.  THREDDS endpoints move
between server versions, so if both fail, drop a DEM covering the basin window
into data/raw/dem/ and re-run -- any of these work:

  * OpenTopography  https://portal.opentopography.org/raster?opentopoID=OTSRTM.042013.4326.1
    SRTM GL3 (90 m), bbox 17.75-24.25 N, 80.75-87.25 E, GeoTIFF
  * GEBCO           https://download.gebco.net  (grid subset, NetCDF)
  * USGS EarthExplorer  GTOPO30 tile W100N40

Run:  python preprocessing/build_dem.py
Out:  data/processed/dem_fine.npz
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

# A margin so every fine cell has full coverage out to its edges, not just its
# centre, when the source grid is aggregated into it.
MARGIN = C.FINE_RES

OPENDAP_SOURCES = [
    (
        "GEBCO 2024",
        "https://thredds.socib.es/thredds/dodsC/ancillary_data/bathymetry/GEBCO_2024.nc",
        "elevation",
        "lat",
        "lon",
    ),
    (
        "ETOPO 2022 (30 arc-sec)",
        "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/30s/30s_surface_elev_netcdf/ETOPO_2022_v1_30s_N90W180_surface.nc",
        "z",
        "lat",
        "lon",
    ),
]


def _subset(url: str, var: str, latname: str, lonname: str):
    """Open a remote DEM and pull only the basin window over OPeNDAP."""
    import xarray as xr

    ds = xr.open_dataset(url, decode_times=False)
    da = ds[var]

    # Span BOTH grids.  The catchment's western edge (80.50) sits outside the
    # CNN domain (81.0), so a window cut to FINE_LON_MIN alone leaves the
    # headwater columns with no pixels at all.
    lats = da[latname].values
    lo = min(C.FINE_LAT_MIN, C.CATCHMENT_LAT_MIN) - MARGIN
    hi = max(C.FINE_LAT_MAX, C.CATCHMENT_LAT_MAX) + MARGIN
    lat_slice = slice(lo, hi) if lats[0] < lats[-1] else slice(hi, lo)

    lons = da[lonname].values
    lo = min(C.FINE_LON_MIN, C.CATCHMENT_LON_MIN) - MARGIN
    hi = max(C.FINE_LON_MAX, C.CATCHMENT_LON_MAX) + MARGIN
    lon_slice = slice(lo, hi) if lons[0] < lons[-1] else slice(hi, lo)

    return da.sel({latname: lat_slice, lonname: lon_slice}).load(), latname, lonname


def fetch_dem():
    """Return (elevation DataArray, lat name, lon name, source label)."""
    errors = []
    for label, url, var, latname, lonname in OPENDAP_SOURCES:
        try:
            print(f"trying {label} ...")
            da, la, lo = _subset(url, var, latname, lonname)
            if da.size == 0:
                raise ValueError("empty subset")
            print(f"  got {da.shape} from {label}")
            return da, la, lo, label
        except Exception as exc:  # noqa: BLE001 - report and fall through
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
            print(f"  failed: {type(exc).__name__}")

    raise RuntimeError(
        "no DEM source reachable:\n  " + "\n  ".join(errors)
        + "\n\nOffline fallback: download a GeoTIFF/NetCDF DEM covering "
        f"{C.FINE_LAT_MIN}-{C.FINE_LAT_MAX}N, {C.FINE_LON_MIN}-{C.FINE_LON_MAX}E "
        "(SRTM via OpenTopography, or GEBCO) into data/raw/dem/ and re-run."
    )


def local_dem():
    """Use a DEM already present in data/raw/dem/, if the user supplied one."""
    import xarray as xr

    d = C.RAW / "dem"
    files = sorted(list(d.glob("*.nc")) + list(d.glob("*.tif")))
    if not files:
        return None

    f = files[0]
    print(f"using local DEM {f.name}")
    if f.suffix == ".tif":
        da = xr.open_dataarray(f, engine="rasterio").squeeze()
        return da, "y", "x", f.name

    ds = xr.open_dataset(f)
    var = next(v for v in ds.data_vars if ds[v].ndim >= 2)
    latname = next(c for c in ds[var].dims if str(c).lower().startswith(("lat", "y")))
    lonname = next(c for c in ds[var].dims if str(c).lower().startswith(("lon", "x")))
    return ds[var], latname, lonname, f.name


def aggregate_to_fine(da, latname: str, lonname: str, grid: str = "cnn"):
    """Reduce the native DEM into per-cell statistics on the requested grid.

    groupby_bins over both axes assigns every native pixel to the cell whose
    footprint contains it, so the statistics describe the cell's area rather than
    a point sample at its centre.

    Two grids, because they answer different questions:

      cnn  18-24 N, 81-87 E (25x25).  Set by the ECMWF downscaling domain; this
           is the grid the trained model expects and must not change.
      vic  the catchment (19.75-23.50 N, 80.50-84.25 E, 16x16).  The catchment's
           western edge at 80.50 lies OUTSIDE the CNN domain, so dem_fine.npz has
           no elevation for the 80.50 and 80.75 columns -- the headwaters.  VIC
           needs elevation in every cell it runs, hence a separate grid rather
           than a redefinition of the first.
    """
    import xarray as xr  # noqa: F401  (needed for the accessor)

    if grid == "vic":
        flats = np.arange(C.CATCHMENT_LAT_MIN, C.CATCHMENT_LAT_MAX, C.FINE_RES)
        flons = np.arange(C.CATCHMENT_LON_MIN, C.CATCHMENT_LON_MAX + 1e-9, C.FINE_RES)
    else:
        flats = np.arange(C.FINE_LAT_MIN, C.FINE_LAT_MAX + C.FINE_RES / 2, C.FINE_RES)
        flons = np.arange(C.FINE_LON_MIN, C.FINE_LON_MAX + C.FINE_RES / 2, C.FINE_RES)

    half = C.FINE_RES / 2
    lat_edges = np.append(flats - half, flats[-1] + half)
    lon_edges = np.append(flons - half, flons[-1] + half)

    # Elevation only; ocean/bathymetry is clipped so sea cells read as 0 m rather
    # than dragging a cell's mean down to -2000 m.
    da = da.clip(min=0)

    def reduce_with(op: str) -> np.ndarray:
        g = getattr(da.groupby_bins(latname, lat_edges), op)()
        g = getattr(g.groupby_bins(lonname, lon_edges), op)()
        return g.transpose(f"{latname}_bins", f"{lonname}_bins").values.astype(np.float32)

    stats = {
        "elev_mean": reduce_with("mean"),
        "elev_std": reduce_with("std"),
        "elev_min": reduce_with("min"),
        "elev_max": reduce_with("max"),
    }
    # std is undefined where a bin holds one pixel; flat is the right reading.
    stats["elev_std"] = np.nan_to_num(stats["elev_std"], nan=0.0)
    return stats, flats, flons


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["cnn", "vic"], default="cnn",
                    help="cnn -> dem_fine.npz (25x25, do not change: the "
                         "trained model depends on it); vic -> dem_vic.npz "
                         "(16x16 catchment, includes the 80.50/80.75 "
                         "headwater columns the CNN domain omits)")
    args = ap.parse_args()

    got = local_dem()
    if got is None:
        da, latname, lonname, label = fetch_dem()
    else:
        da, latname, lonname, label = got

    stats, flats, flons = aggregate_to_fine(da, latname, lonname, args.grid)

    holes = {k: int(np.isnan(v).sum()) for k, v in stats.items()}
    assert holes["elev_mean"] == 0, f"cells with no DEM pixels: {holes}"

    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    if args.grid == "vic":
        out = C.PROCESSED / "dem_vic.npz"
        np.savez_compressed(out, source=np.array(label),
                            lats=flats, lons=flons, **stats)
        print(f"\nwrote {out}   source: {label}")
        print(f"  grid {len(flats)}x{len(flons)}  "
              f"{flats[0]}-{flats[-1]} N, {flons[0]}-{flons[-1]} E")
        for k, v in stats.items():
            print(f"  {k:10s} {v.min():7.1f} - {v.max():7.1f} m   mean {v.mean():7.1f} m")
        soil = C.PROCESSED / "soilgrids.npz"
        if soil.exists():
            s = np.load(soil, allow_pickle=True)
            assert np.allclose(s["lats"], flats) and np.allclose(s["lons"], flons), \
                "VIC DEM grid does not match the SoilGrids grid"
            print("  grid matches soilgrids.npz exactly")
        return

    out = C.PROCESSED / "dem_fine.npz"
    np.savez_compressed(
        out,
        source=np.array(label),
        fine_lats=flats,
        fine_lons=flons,
        **stats,
    )

    t = np.load(C.PROCESSED / "fine_target.npz")
    cat = t["catchment"]
    print(f"\nwrote {out}   source: {label}")
    for k, v in stats.items():
        print(f"  {k:10s} domain {v.min():7.1f} - {v.max():7.1f} m   "
              f"catchment mean {v[cat].mean():7.1f} m")
    print(f"\n  relief within a single 0.25 deg cell (max-min), catchment: "
          f"mean {(stats['elev_max'] - stats['elev_min'])[cat].mean():.0f} m, "
          f"max {(stats['elev_max'] - stats['elev_min'])[cat].max():.0f} m")
    print("  ECMWF orog resolves none of this: it carries one value per 1.5 deg cell.")


if __name__ == "__main__":
    main()
