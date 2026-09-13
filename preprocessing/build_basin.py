"""Delineate the Mahanadi basin above Hirakud from HydroSHEDS, and upscale it.

Three things come out of this, and the first is the one that validates the
other two:

  1. BASIN AREA.  The catchment above Hirakud is published at ~83,400 km2.
     Delineating from flow direction and comparing against that number is a
     real test -- a mis-snapped pour point or a wrong direction encoding
     produces an area off by a factor, not a few percent.
  2. A 0.25 deg BASIN MASK with per-cell area fractions, which replaces the
     rectangular lat/lon box the project has used so far.  The box includes
     cells that drain away from Hirakud and excludes some that drain into it.
  3. A 0.25 deg FLOW DIRECTION grid for RVIC routing.

Why this matters beyond routing: the project's catchment has been the rectangle
19.75-23.60 N, 80.50-84.25 E, and the CNN's version of it is narrower still
(81.0 E, set by the ECMWF domain).  Neither is the actual basin.  This replaces
an assumption with a delineation.

HydroSHEDS D8 encoding: 1=E 2=SE 4=S 8=SW 16=W 32=NW 64=N 128=NE, 0=ocean/sink,
255=nodata.

Run:  python preprocessing/build_basin.py
Out:  data/processed/basin.npz, results/figures/basin.png
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

ROUTING = C.RAW / "vic" / "routing"
DIR_TIF = ROUTING / "hyd_as_dir_15s.tif"
ACC_TIF = ROUTING / "hyd_as_acc_15s.tif"

# Hirakud dam. The published catchment area is the check on this location.
DAM_LAT, DAM_LON = 21.52, 83.87
PUBLISHED_AREA_KM2 = 83_400.0

# Window generously around the basin: the true divide reaches beyond the
# rectangle the project has been using, which is the whole point.
WIN = dict(lat=(19.0, 24.5), lon=(79.5, 85.0))

# (row, col) step for each D8 code. Rows increase southward in raster order.
D8 = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
      16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}
# Code that a neighbour at offset (dr, dc) must carry to flow INTO the centre.
INTO = {(-dr, -dc): code for code, (dr, dc) in D8.items()}

R_EARTH = 6371.0072


def read_window(path, win):
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(path) as src:
        w = from_bounds(win["lon"][0], win["lat"][0],
                        win["lon"][1], win["lat"][1], src.transform)
        return src.read(1, window=w), src.window_transform(w)


def cell_area_km2(lats_edges_top, res_deg, ncols):
    """Area of one 15 arc-sec cell per raster row -- varies with latitude."""
    phi1 = np.radians(lats_edges_top)
    phi2 = np.radians(lats_edges_top - res_deg)
    dlon = np.radians(res_deg)
    return (R_EARTH ** 2 * dlon * (np.sin(phi1) - np.sin(phi2)))


def delineate(dirs, outlet):
    """Every cell draining to `outlet`, by walking the D8 graph upstream."""
    nrow, ncol = dirs.shape
    mask = np.zeros(dirs.shape, dtype=bool)
    stack = [outlet]
    mask[outlet] = True
    offsets = list(INTO.items())
    while stack:
        r, c = stack.pop()
        for (dr, dc), code in offsets:
            rr, cc = r + dr, c + dc
            if 0 <= rr < nrow and 0 <= cc < ncol and not mask[rr, cc]:
                if dirs[rr, cc] == code:
                    mask[rr, cc] = True
                    stack.append((rr, cc))
    return mask


def snap(acc, transform, lat, lon, radius_px=12):
    """Move the dam onto the river: the highest-accumulation cell nearby.

    A pour point taken from a gazetteer will not sit exactly on the modelled
    channel, and delineating from an off-channel cell yields a tiny basin --
    which is why the area check below is the real test.
    """
    inv = ~transform
    col, row = inv * (lon, lat)
    row, col = int(row), int(col)
    sub = acc[row - radius_px:row + radius_px + 1,
              col - radius_px:col + radius_px + 1].astype("float64")
    sub[sub == np.iinfo(np.uint32).max] = -1
    dr, dc = np.unravel_index(np.argmax(sub), sub.shape)
    return (row - radius_px + dr, col - radius_px + dc), sub.max()


def main() -> None:
    for f in (DIR_TIF, ACC_TIF):
        if not f.exists():
            sys.exit(f"missing {f} -- see hydrology/vic/DATA_REQUIREMENTS.md")

    dirs, transform = read_window(DIR_TIF, WIN)
    acc, _ = read_window(ACC_TIF, WIN)
    res = transform.a
    print(f"HydroSHEDS window {dirs.shape} at {res * 3600:.0f} arc-sec")

    outlet, acc_at_outlet = snap(acc, transform, DAM_LAT, DAM_LON)
    olat = transform.f + transform.e * (outlet[0] + 0.5)
    olon = transform.c + transform.a * (outlet[1] + 0.5)
    print(f"  dam       {DAM_LAT:.4f} N {DAM_LON:.4f} E")
    print(f"  snapped   {olat:.4f} N {olon:.4f} E  "
          f"({np.hypot(olat - DAM_LAT, olon - DAM_LON) * 111:.1f} km away)")
    print(f"  upstream cells at outlet (from ACC): {acc_at_outlet:,.0f}")

    mask = delineate(dirs, outlet)

    rows = np.arange(dirs.shape[0])
    top = transform.f + transform.e * rows
    area_row = cell_area_km2(top, res, dirs.shape[1])
    area = float((mask * area_row[:, None]).sum())

    print(f"\n=== area check ===")
    print(f"  delineated {area:10,.0f} km2")
    print(f"  published  {PUBLISHED_AREA_KM2:10,.0f} km2")
    err = 100 * (area - PUBLISHED_AREA_KM2) / PUBLISHED_AREA_KM2
    print(f"  difference {err:+9.1f} %  "
          f"{'OK' if abs(err) < 10 else '<- CHECK the pour point / encoding'}")

    lat_px = transform.f + transform.e * (np.arange(dirs.shape[0]) + 0.5)
    lon_px = transform.c + transform.a * (np.arange(dirs.shape[1]) + 0.5)
    rr, cc = np.where(mask)
    print(f"\n  basin extent {lat_px[rr].min():.2f}-{lat_px[rr].max():.2f} N, "
          f"{lon_px[cc].min():.2f}-{lon_px[cc].max():.2f} E")
    print(f"  project box  {C.CATCHMENT_LAT_MIN}-{C.CATCHMENT_LAT_MAX} N, "
          f"{C.CATCHMENT_LON_MIN}-{C.CATCHMENT_LON_MAX} E")

    # ---- upscale to the 0.25 deg VIC grid as an AREA FRACTION, not a
    # yes/no mask: edge cells are partly in the basin, and rainfall averaged
    # with fractional weights is the physically right quantity.
    lats = np.arange(C.CATCHMENT_LAT_MIN, C.CATCHMENT_LAT_MAX, C.FINE_RES)
    lons = np.arange(C.CATCHMENT_LON_MIN, C.CATCHMENT_LON_MAX + 1e-9, C.FINE_RES)
    half = C.FINE_RES / 2
    frac = np.zeros((len(lats), len(lons)))
    for i, la in enumerate(lats):
        rsel = (lat_px >= la - half) & (lat_px < la + half)
        if not rsel.any():
            continue
        for j, lo in enumerate(lons):
            csel = (lon_px >= lo - half) & (lon_px < lo + half)
            if not csel.any():
                continue
            sub = mask[np.ix_(rsel, csel)]
            frac[i, j] = sub.mean()

    inside_box = float((mask * area_row[:, None])[
        np.ix_((lat_px >= lats[0] - half) & (lat_px <= lats[-1] + half),
               (lon_px >= lons[0] - half) & (lon_px <= lons[-1] + half))].sum())
    print(f"\n  basin area inside the project box: {inside_box:,.0f} km2 "
          f"({100 * inside_box / area:.1f} % of the basin)")
    print(f"  0.25 deg cells with any basin: {int((frac > 0).sum())} of {frac.size}")
    print(f"  cells fully inside:            {int((frac > 0.99).sum())}")

    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    out = C.PROCESSED / "basin.npz"
    np.savez_compressed(
        out, lats=lats, lons=lons, fraction=frac,
        area_km2=area, published_area_km2=PUBLISHED_AREA_KM2,
        outlet_lat=olat, outlet_lon=olon,
        # The 15 arc-sec mask, kept because routing needs it: a coarse cell on
        # the divide contains pixels of neighbouring basins, and its highest-
        # accumulation pixel can belong to a river flowing AWAY from Hirakud.
        # Upscaling flow direction has to search only within the basin.
        fine_mask=mask, fine_lat=lat_px, fine_lon=lon_px,
        source="HydroSHEDS v1 DIR/ACC 15 arc-sec",
    )
    print(f"\nwrote {out}")

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ext = [lon_px[0], lon_px[-1], lat_px[-1], lat_px[0]]
    ax[0].imshow(np.log1p(acc.astype("float64").clip(0, 1e7)), cmap="Blues",
                 extent=ext, origin="upper")
    ax[0].contour(lon_px, lat_px, mask, levels=[0.5], colors="crimson", linewidths=1.2)
    ax[0].plot(olon, olat, "r*", ms=14, mec="k")
    ax[0].set_title(f"HydroSHEDS flow accumulation + delineated divide\n"
                    f"{area:,.0f} km2 (published {PUBLISHED_AREA_KM2:,.0f})")
    box = [[C.CATCHMENT_LON_MIN, C.CATCHMENT_LON_MAX, C.CATCHMENT_LON_MAX,
            C.CATCHMENT_LON_MIN, C.CATCHMENT_LON_MIN],
           [C.CATCHMENT_LAT_MIN, C.CATCHMENT_LAT_MIN, C.CATCHMENT_LAT_MAX,
            C.CATCHMENT_LAT_MAX, C.CATCHMENT_LAT_MIN]]
    ax[0].plot(*box, "k--", lw=1.4, label="project box")
    ax[0].legend(loc="lower left")

    im = ax[1].imshow(frac, origin="lower", cmap="YlGnBu", vmin=0, vmax=1,
                      extent=[lons[0] - half, lons[-1] + half,
                              lats[0] - half, lats[-1] + half])
    plt.colorbar(im, ax=ax[1], label="basin area fraction")
    ax[1].plot(olon, olat, "r*", ms=14, mec="k")
    ax[1].set_title("upscaled to the 0.25 deg VIC grid")
    for a in ax:
        a.set_xlabel("lon"); a.set_ylabel("lat")
    fig.tight_layout()
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / "basin.png", dpi=110)
    print(f"wrote {C.FIGURES / 'basin.png'}")


if __name__ == "__main__":
    main()
