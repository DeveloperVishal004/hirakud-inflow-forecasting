"""Map the aggregated SoilGrids fields and check them against the DEM.

Numbers closing to 100 % prove the units are right; they do not prove the
fields are in the correct place.  A transposed or flipped aggregation would
still sum to 100 %.  These maps, and the comparison against an independent
grid (the DEM), are what catch that.

Run:  python preprocessing/verify_soilgrids.py
Out:  results/figures/soilgrids.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

SOIL = C.PROCESSED / "soilgrids.npz"
LAYERS = ["layer1_0-15cm", "layer2_15-60cm", "layer3_60-200cm"]


def main() -> None:
    d = np.load(SOIL, allow_pickle=True)
    la, lo = d["lats"], d["lons"]
    ext = [lo[0] - C.FINE_RES / 2, lo[-1] + C.FINE_RES / 2,
           la[0] - C.FINE_RES / 2, la[-1] + C.FINE_RES / 2]

    fig, ax = plt.subplots(3, 4, figsize=(17, 11))
    for r, layer in enumerate(LAYERS):
        for c, (prop, cmap, unit) in enumerate([
                ("sand", "YlOrBr", "%"), ("silt", "YlGn", "%"),
                ("clay", "RdPu", "%"), ("bdod", "cividis", "g/cm3")]):
            a = d[f"{prop}_{layer}"]
            im = ax[r, c].imshow(a, origin="lower", extent=ext, cmap=cmap)
            ax[r, c].set_title(f"{prop} {layer.split('_')[1]}", fontsize=10)
            plt.colorbar(im, ax=ax[r, c], fraction=0.046, label=unit)
            # Hirakud dam, as an orientation landmark.
            ax[r, c].plot(83.87, 21.52, "w*", ms=11, mec="k")
            if c == 0:
                ax[r, c].set_ylabel(f"{layer}\nlat", fontsize=9)
            if r == 2:
                ax[r, c].set_xlabel("lon")
    fig.suptitle("SoilGrids 250 m aggregated to the 0.25 deg VIC grid "
                 "(star = Hirakud dam)", fontsize=13)
    fig.tight_layout()
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    out = C.FIGURES / "soilgrids.png"
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")

    # ---- independent placement check against the DEM.
    # dem_vic.npz, not dem_fine.npz: the CNN grid begins at 81.0 E and has no
    # cells for the catchment's 80.50/80.75 columns, so matching against it
    # would silently pair those with cells 0.5 deg away.
    dem = np.load(C.PROCESSED / "dem_vic.npz", allow_pickle=True)
    ii = [int(np.argmin(np.abs(dem["lats"] - x))) for x in la]
    jj = [int(np.argmin(np.abs(dem["lons"] - x))) for x in lo]
    elev = dem["elev_mean"][np.ix_(ii, jj)]
    clay = d["clay_layer1_0-15cm"]
    off = max(abs(dem["lats"][ii] - la).max(), abs(dem["lons"][jj] - lo).max())
    print(f"\n  grid alignment: DEM cells matched to within {off:.4f} deg"
          f"   {'OK' if off < 1e-6 else '<- MISMATCH'}")
    print(f"  corr(clay, elevation) = {np.corrcoef(clay.ravel(), elev.ravel())[0, 1]:+.3f}"
          "   (weak by design -- soil carries information the DEM does not)")

    # A north-south gradient is the expected signature here; a flipped
    # aggregation would reverse its sign.
    north = clay[len(la) // 2:].mean()
    south = clay[:len(la) // 2].mean()
    print(f"  clay northern half {north:.1f} %  vs southern half {south:.1f} %  "
          f"-> {'increases northward, as expected' if north > south else 'DECREASES northward -- suspect a flip'}")


if __name__ == "__main__":
    main()
