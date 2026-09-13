"""Verify the DEM output and its alignment with the IMD 0.25 deg grid.

Checks, in order of how badly a failure would corrupt training:

  1. Shape and coordinate identity against fine_target.npz -- the DEM must sit on
     exactly the same 25 x 25 grid as the rainfall target, not merely a similar one.
  2. Physical plausibility of the elevation statistics.
  3. Ordering invariants (min <= mean <= max, std >= 0) that catch a transposed
     or mis-binned aggregation.
  4. Independent spot-checks against known terrain, which catch a lat/lon swap
     that all the shape assertions above would happily pass.
  5. Correlation of elevation with mean observed rainfall -- a physical sanity
     check that the DEM carries real signal for downscaling.

Also writes figures/dem_verification.png: DEM maps beside the IMD rainfall
climatology, so misalignment is visible rather than merely asserted.

Run:  python preprocessing/verify_dem.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

# Landmarks inside the domain, for the lat/lon-swap check.  A transposed grid
# puts the Chhattisgarh uplands where the coastal plain should be.
LANDMARKS = [
    ("Bay of Bengal (18.25N, 86.75E)", 18.25, 86.75, 0, 60),
    ("Mahanadi delta / coast (20.25N, 86.25E)", 20.25, 86.25, 0, 120),
    ("Chhattisgarh uplands (22.00N, 82.00E)", 22.00, 82.00, 200, 900),
    ("Hirakud dam (21.50N, 83.75E)", 21.50, 83.75, 100, 600),
]

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def main() -> None:
    dem_path = C.PROCESSED / "dem_fine.npz"
    if not dem_path.exists():
        sys.exit(f"{dem_path} not found -- run preprocessing/build_dem.py first")

    d = np.load(dem_path, allow_pickle=True)
    t = np.load(C.PROCESSED / "fine_target.npz")
    print(f"DEM source: {d['source']}\n")

    mean, std = d["elev_mean"], d["elev_std"]
    emin, emax = d["elev_min"], d["elev_max"]
    land, catchment = t["land"], t["catchment"]

    # ---------------------------------------------------------------- 1. grid
    print("1. Grid alignment with IMD target")
    check("DEM shape == target shape", mean.shape == land.shape, f"{mean.shape} vs {land.shape}")
    check("latitudes identical", np.array_equal(d["fine_lats"], t["fine_lats"]))
    check("longitudes identical", np.array_equal(d["fine_lons"], t["fine_lons"]))
    check("grid is 0.25 deg", np.allclose(np.diff(d["fine_lats"]), C.FINE_RES))
    check("all four stats same shape", {mean.shape, std.shape, emin.shape, emax.shape} == {mean.shape})

    # -------------------------------------------------------- 2. plausibility
    print("\n2. Physical plausibility")
    check("no NaNs in mean", not np.isnan(mean).any())
    check("no negative elevation (bathymetry clipped)", (mean >= 0).all(), f"min {mean.min():.1f} m")
    check("max below Indian peninsula ceiling", emax.max() < 2000, f"max {emax.max():.1f} m")
    check("catchment is upland, not sea level",
          100 < mean[catchment].mean() < 900, f"mean {mean[catchment].mean():.1f} m")

    # ----------------------------------------------------------- 3. orderings
    print("\n3. Aggregation invariants")
    check("min <= mean everywhere", (emin <= mean + 1e-3).all())
    check("mean <= max everywhere", (mean <= emax + 1e-3).all())
    check("std >= 0", (std >= 0).all())
    check("std > 0 somewhere (relief resolved)", (std > 0).sum() > 0.5 * std.size,
          f"{(std > 0).sum()}/{std.size} cells")

    # ----------------------------------------------------------- 4. landmarks
    print("\n4. Landmark spot-checks (catches a lat/lon transpose)")
    for name, lat, lon, lo, hi in LANDMARKS:
        i = int(np.abs(d["fine_lats"] - lat).argmin())
        j = int(np.abs(d["fine_lons"] - lon).argmin())
        v = float(mean[i, j])
        check(name, lo <= v <= hi, f"{v:.0f} m (expected {lo}-{hi})")

    # Elevation must rise westward (inland) across the catchment.
    west = mean[:, d["fine_lons"] < 83.0][catchment[:, d["fine_lons"] < 83.0]].mean()
    east = mean[:, d["fine_lons"] > 85.0][land[:, d["fine_lons"] > 85.0]].mean()
    check("west (inland) higher than east (coast)", west > east,
          f"west {west:.0f} m vs east {east:.0f} m")

    # ------------------------------------------------------- 5. rainfall link
    print("\n5. Does the DEM carry signal for downscaling?")
    rain = np.where(t["mask"], t["target"], np.nan)
    clim = np.nanmean(rain, axis=0)
    ok = np.isfinite(clim) & land
    r_mean = np.corrcoef(mean[ok], clim[ok])[0, 1]
    r_std = np.corrcoef(std[ok], clim[ok])[0, 1]
    print(f"  corr(elev_mean, mean rainfall) = {r_mean:+.3f}")
    print(f"  corr(elev_std,  mean rainfall) = {r_std:+.3f}")
    check("elevation informative about rainfall", max(abs(r_mean), abs(r_std)) > 0.10)

    # ------------------------------------------------------------- statistics
    print("\nStatistics (catchment cells only)")
    for k, v in [("elev_mean", mean), ("elev_std", std), ("elev_min", emin), ("elev_max", emax)]:
        c = v[catchment]
        print(f"  {k:10s} min {c.min():7.1f}  mean {c.mean():7.1f}  max {c.max():7.1f} m")
    relief = (emax - emin)[catchment]
    print(f"  {'relief':10s} min {relief.min():7.1f}  mean {relief.mean():7.1f}  max {relief.max():7.1f} m")
    print("  ^ sub-grid relief ECMWF orog cannot see: it holds one value per 1.5 deg cell.")

    plot(d, t, clim)

    print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"{len(FAILURES)} FAILED: {FAILURES}"))
    sys.exit(1 if FAILURES else 0)


def plot(d, t, clim) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed -- skipping figures")
        return

    lats, lons = d["fine_lats"], d["fine_lons"]
    # origin="lower" because latitude ascends down the array axis; without it the
    # maps would render flipped and alignment would be impossible to judge.
    ext = [lons.min() - 0.125, lons.max() + 0.125, lats.min() - 0.125, lats.max() + 0.125]
    kw = dict(origin="lower", extent=ext, aspect="auto")

    panels = [
        ("DEM mean elevation (m)", d["elev_mean"], "terrain"),
        ("DEM relief, max-min (m)", d["elev_max"] - d["elev_min"], "magma"),
        ("IMD mean rainfall (mm/d)", clim, "Blues"),
        ("IMD land mask", t["land"].astype(float), "gray"),
        ("Hirakud catchment mask", t["catchment"].astype(float), "gray"),
        ("DEM masked to catchment (m)", np.where(t["catchment"], d["elev_mean"], np.nan), "terrain"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (title, data, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(data, cmap=cmap, **kw)
        ax.set_title(title, fontsize=10)
        ax.plot(83.87, 21.52, "r*", ms=14, mec="k", label="Hirakud dam")
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        ax.legend(loc="upper right", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("DEM vs IMD 0.25 deg grid alignment  (dam marker must land on the "
                 "same pixel in every panel)", fontsize=12)
    fig.tight_layout()

    out_dir = C.FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "dem_verification.png", dpi=130)
    print(f"\nwrote {out_dir / 'dem_verification.png'}")


if __name__ == "__main__":
    main()
