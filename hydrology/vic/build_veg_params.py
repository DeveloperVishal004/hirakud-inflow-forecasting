"""VIC 5 vegetation library and vegetation parameter file, from MCD12Q1.

Two files come out of this:

  veg_lib.txt    one row per IGBP class: monthly LAI, albedo, roughness and
                 displacement, plus the resistance and radiation terms VIC's
                 Penman-Monteith needs.
  veg_param.txt  one block per grid cell: which classes are present, their
                 area fractions, and root distributions.

The class fractions come from MODIS; everything in the library is literature
values, and the one place they have been changed deliberately from the stock
VIC library is documented below because it materially affects the water
balance here.

WHY THE STOCK LIBRARY IS WRONG FOR THIS BASIN
---------------------------------------------
The distributed VIC vegetation library was assembled for North American and
European land cover.  Its cropland LAI peaks in June-July and has fallen away
by September -- the temperate growing season.

This catchment is 69 % cropland and that cropland is monsoon KHARIF, mostly
rice: sown with the June onset, peaking late August to September, harvested
October-November, with bare or stubble ground through the hot pre-monsoon.  A
temperate curve puts peak transpiration a month and a half early and has the
canopy senescing while the monsoon is still delivering its heaviest rain.  That
is a systematic error in evapotranspiration during exactly the season the
forecast is about, so the cropland, grassland and deciduous curves below follow
the Indian phenological calendar instead.

The deciduous broadleaf curve is likewise dry-season-deciduous (teak and sal
drop their leaves February-May), not winter-deciduous.

Run:  python hydrology/vic/build_veg_params.py
Out:  data/processed/vic/veg_lib.txt, veg_param.txt
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs import config as C

OUTDIR = C.PROCESSED / "vic"

# Root zones matched to the three soil layers, so the mapping is inspectable.
ROOT_DEPTH = [0.15, 0.45, 1.40]

# Classes VIC treats as non-vegetated; their area becomes bare soil.
NONVEG = {13, 15, 16, 17}

# Below the noise floor of a 3,345-pixel sample; kept out to avoid a long tail
# of classes contributing nothing but file size.
MIN_CV = 0.005

#                      J    F    M    A    M    J    J    A    S    O    N    D
LAI = {
    1:  [4.0, 4.0, 4.0, 4.0, 4.2, 4.5, 4.8, 4.8, 4.6, 4.3, 4.1, 4.0],
    2:  [4.5, 4.5, 4.4, 4.4, 4.5, 5.0, 5.5, 5.6, 5.5, 5.2, 4.8, 4.6],
    3:  [1.0, 0.8, 0.6, 0.8, 1.5, 3.0, 4.0, 4.2, 3.8, 2.5, 1.5, 1.2],
    # dry-season deciduous: bare Feb-May, full canopy through the monsoon
    4:  [2.0, 1.0, 0.5, 0.5, 1.0, 3.0, 4.5, 5.0, 5.0, 4.0, 3.0, 2.5],
    5:  [2.5, 2.0, 1.5, 1.5, 2.0, 3.2, 4.2, 4.6, 4.5, 3.8, 3.0, 2.7],
    6:  [1.0, 0.9, 0.7, 0.7, 0.9, 1.6, 2.4, 2.6, 2.5, 1.9, 1.4, 1.1],
    7:  [0.6, 0.5, 0.4, 0.4, 0.5, 1.0, 1.6, 1.8, 1.7, 1.2, 0.9, 0.7],
    8:  [1.8, 1.4, 1.0, 1.0, 1.3, 2.6, 3.8, 4.2, 4.1, 3.2, 2.4, 2.0],
    9:  [1.2, 0.9, 0.7, 0.7, 0.9, 1.9, 3.0, 3.4, 3.3, 2.4, 1.7, 1.4],
    10: [0.5, 0.4, 0.3, 0.3, 0.4, 1.0, 2.0, 2.5, 2.5, 1.5, 1.0, 0.7],
    11: [1.5, 1.3, 1.0, 1.0, 1.2, 2.2, 3.4, 3.8, 3.7, 2.8, 2.0, 1.7],
    # kharif rice: sown at the June onset, peak Aug-Sep, harvested Oct-Nov,
    # small rabi crop over winter, bare through the pre-monsoon
    12: [0.5, 0.5, 0.3, 0.2, 0.2, 0.8, 2.5, 4.0, 4.2, 2.5, 1.0, 0.8],
    14: [0.9, 0.8, 0.5, 0.4, 0.5, 1.4, 3.0, 4.1, 4.2, 2.9, 1.5, 1.1],
}

# class -> (overstory, rarc, rmin, albedo, height_m, RGL, rad_atten,
#           wind_atten, trunk_ratio)
TRAITS = {
    1:  (1, 60, 100, 0.12, 17.0, 30, 0.5, 0.5, 0.2),
    2:  (1, 60, 150, 0.12, 20.0, 30, 0.5, 0.5, 0.2),
    3:  (1, 60, 100, 0.14, 14.0, 30, 0.5, 0.5, 0.2),
    4:  (1, 60, 100, 0.16, 15.0, 30, 0.5, 0.5, 0.2),
    5:  (1, 60, 125, 0.14, 15.0, 30, 0.5, 0.5, 0.2),
    6:  (0, 60, 135, 0.18,  1.5, 100, 0.5, 0.5, 0.2),
    7:  (0, 60, 135, 0.20,  1.0, 100, 0.5, 0.5, 0.2),
    8:  (1, 60, 125, 0.15, 10.0, 65, 0.5, 0.5, 0.2),
    9:  (0, 60, 125, 0.17,  3.0, 65, 0.5, 0.5, 0.2),
    10: (0, 60, 120, 0.19,  0.5, 100, 0.5, 0.5, 0.2),
    11: (0, 60, 120, 0.14,  1.0, 100, 0.5, 0.5, 0.2),
    12: (0, 60, 120, 0.18,  1.0, 100, 0.5, 0.5, 0.2),
    14: (0, 60, 120, 0.18,  2.0, 100, 0.5, 0.5, 0.2),
}

# Fraction of roots in each of the three zones.  Trees reach deeper; the
# shallow-rooted rice that dominates here does not.
ROOT_FRACT = {
    1: [0.20, 0.40, 0.40], 2: [0.20, 0.40, 0.40], 3: [0.25, 0.45, 0.30],
    4: [0.25, 0.45, 0.30], 5: [0.25, 0.45, 0.30], 6: [0.35, 0.45, 0.20],
    7: [0.40, 0.45, 0.15], 8: [0.25, 0.45, 0.30], 9: [0.35, 0.45, 0.20],
    10: [0.50, 0.40, 0.10], 11: [0.50, 0.40, 0.10], 12: [0.45, 0.45, 0.10],
    14: [0.40, 0.45, 0.15],
}


def write_library(path: Path, names) -> None:
    """One row per class. Roughness and displacement from canopy height."""
    with open(path, "w") as f:
        f.write("#class overstory rarc rmin LAI(12) albedo(12) rough(12) "
                "disp(12) wind_h RGL rad_atn wind_atn trunk_ratio comment\n")
        for k in sorted(TRAITS):
            over, rarc, rmin, alb, h, rgl, ratt, watt, trunk = TRAITS[k]
            # Standard aerodynamic ratios (Brutsaert 1982).
            rough, disp = 0.123 * h, 0.67 * h
            # Wind reference must clear the canopy; ERA5 supplies 10 m and VIC
            # applies a log profile between the two.
            wind_h = max(10.0, h + 10.0) if over else 10.0
            f.write(f"{k} {over} {rarc} {rmin} "
                    + " ".join(f"{v:.2f}" for v in LAI[k]) + " "
                    + " ".join(f"{alb:.2f}" for _ in range(12)) + " "
                    + " ".join(f"{rough:.3f}" for _ in range(12)) + " "
                    + " ".join(f"{disp:.3f}" for _ in range(12)) + " "
                    + f"{wind_h:.1f} {rgl} {ratt} {watt} {trunk} "
                    + f"{names[k - 1]}\n")


def main() -> None:
    lc = np.load(C.PROCESSED / "landcover.npz", allow_pickle=True)
    basin = np.load(C.PROCESSED / "basin.npz", allow_pickle=True)
    lats, lons = lc["lats"], lc["lons"]
    assert np.allclose(basin["lats"], lats), "landcover/basin grids differ"
    frac, names = lc["fraction"], lc["classes"]
    ny, nx = len(lats), len(lons)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    write_library(OUTDIR / "veg_lib.txt", names)
    print(f"wrote {OUTDIR / 'veg_lib.txt'}  ({len(TRAITS)} classes)")

    blocks, stats = [], []
    dropped_total = 0.0
    for i in range(ny):
        for j in range(nx):
            gid = i * nx + j + 1
            present = []
            for k in range(17):
                cls = k + 1
                cv = float(frac[i, j, k])
                if cls in NONVEG or cv < MIN_CV:
                    dropped_total += cv if cls not in NONVEG else 0.0
                    continue
                present.append((cls, cv))
            lines = [f"{gid} {len(present)}"]
            for cls, cv in present:
                rf = ROOT_FRACT[cls]
                root = " ".join(f"{ROOT_DEPTH[z]:.2f} {rf[z]:.2f}" for z in range(3))
                lines.append(f"  {cls} {cv:.4f} {root}")
            blocks.append("\n".join(lines))
            stats.append((len(present), sum(cv for _, cv in present)))

    with open(OUTDIR / "veg_param.txt", "w") as f:
        f.write("\n".join(blocks) + "\n")

    nveg = np.array([s[0] for s in stats])
    cvsum = np.array([s[1] for s in stats])
    in_basin = (basin["fraction"] > 0).ravel()

    print(f"wrote {OUTDIR / 'veg_param.txt'}  ({len(blocks)} cells)")
    print("\n=== checks ===")
    ok = True
    print(f"  classes per cell     {nveg.min()} - {nveg.max()}  "
          f"(mean {nveg.mean():.1f})")
    print(f"  sum(Cv) per cell     {cvsum.min():.4f} - {cvsum.max():.4f}")
    if cvsum.max() > 1.0 + 1e-9:
        print("    FAIL: vegetated fractions exceed 1 -- VIC requires sum(Cv) <= 1")
        ok = False
    bare = 1.0 - cvsum
    print(f"  bare-soil fraction   {bare.min():.4f} - {bare.max():.4f}  "
          f"(mean {bare.mean():.4f})")
    if bare.min() < -1e-9:
        print("    FAIL: negative bare fraction")
        ok = False
    empty = int((nveg[in_basin] == 0).sum())
    print(f"  basin cells with no vegetation: {empty}  "
          f"{'OK' if empty == 0 else '<- VIC needs at least bare soil defined'}")
    print(f"  area lost to the {MIN_CV:.3f} threshold: "
          f"{100 * dropped_total / (ny * nx):.2f} % of domain "
          f"(becomes bare soil)")

    for k in sorted(TRAITS):
        if len(LAI[k]) != 12:
            print(f"    FAIL: class {k} LAI has {len(LAI[k])} months")
            ok = False
        if sum(ROOT_FRACT[k]) < 0.999 or sum(ROOT_FRACT[k]) > 1.001:
            print(f"    FAIL: class {k} root fractions sum to {sum(ROOT_FRACT[k])}")
            ok = False
    print(f"  LAI 12 months, root fractions sum to 1: "
          f"{'OK' if ok else 'see failures above'}")

    # The phenology change is the point of this file; show it.
    print("\n  cropland LAI by month (kharif calendar, not the stock temperate curve):")
    mn = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    print("    " + "  ".join(f"{m}" for m in mn))
    print("    " + "  ".join(f"{v:3.1f}" for v in LAI[12]))
    print(f"    peak in {mn[int(np.argmax(LAI[12]))]}; the stock VIC library peaks in Jul")

    print(f"\n{'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
