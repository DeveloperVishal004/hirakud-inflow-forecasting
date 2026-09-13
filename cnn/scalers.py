"""Feature and coordinate normalisation, fit from the TRAIN split only.

The previous notebook fit its StandardScaler on the full dataset before
splitting (cell 85), leaking test statistics into training.  Both scalers here
are fit strictly from `split == "train"` rows, then applied unchanged to val
and test -- and, at inference time, to whatever new data comes in.

Two separate scalers because the two inputs have unrelated scales:
  * patch features -- ECMWF fields (msl ~1e5 Pa, tp ~0-500 mm, ssr ~1e2-1e3
    W/m^2) vary by five orders of magnitude across channels.
  * coordinates -- lat/lon/lead_day/elevation, which never appear inside a
    convolution, so they get their own embedding-friendly scaling.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C


@dataclass
class ArrayScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> "ArrayScaler":
        d = np.load(path)
        return cls(d["mean"], d["std"])


def _safe_std(std: np.ndarray) -> np.ndarray:
    """A constant channel would divide by zero; leave it un-rescaled instead."""
    return np.where(std < 1e-6, 1.0, std)


def fit_feature_scaler() -> ArrayScaler:
    """Per-channel mean/std over the coarse grid, train rows only.

    Computed over all 5x5 cells (ocean included): those cells are real ECMWF
    physics, not missing data, and the patch gather can pull from any of them.
    """
    g = np.load(C.PROCESSED / "coarse_grid.npz", allow_pickle=True)
    train = g["split"].astype(str) == "train"
    grid = g["grid"][train]  # (n, n_feat, 5, 5)

    mean = grid.mean(axis=(0, 2, 3))
    std = _safe_std(grid.std(axis=(0, 2, 3)))
    return ArrayScaler(mean.reshape(-1, 1, 1).astype(np.float32),
                        std.reshape(-1, 1, 1).astype(np.float32))


def fit_coord_scaler(n_coords: int) -> ArrayScaler:
    """Scale for [lat, lon, lead_day, (DEM stats...)].

    Lat/lon/lead_day are scaled by their known physical range rather than a
    sampled mean/std: the range is a fixed property of the domain and the S2S
    product, not a statistic that could leak test information by being
    estimated from training data.  DEM stats are static (do not vary by
    split), so their mean/std over land cells carries no split leakage either.
    """
    means = [
        (C.FINE_LAT_MIN + C.FINE_LAT_MAX) / 2,
        (C.FINE_LON_MIN + C.FINE_LON_MAX) / 2,
        (C.LEAD_MIN + C.LEAD_MAX) / 2,
    ]
    stds = [
        (C.FINE_LAT_MAX - C.FINE_LAT_MIN) / 2,
        (C.FINE_LON_MAX - C.FINE_LON_MIN) / 2,
        (C.LEAD_MAX - C.LEAD_MIN) / 2,
    ]

    land = np.load(C.PROCESSED / "fine_target.npz")["land"]

    dem_path = C.PROCESSED / "dem_fine.npz"
    if n_coords > 3 and dem_path.exists():
        dem = np.load(dem_path, allow_pickle=True)
        for key in ["elev_mean", "elev_std", "elev_min", "elev_max"]:
            v = dem[key][land]
            means.append(float(v.mean()))
            stds.append(float(v.std()) if v.std() > 1e-6 else 1.0)

    clim_path = C.PROCESSED / "climatology.npz"
    if clim_path.exists():
        # Climatology is itself derived from train rows only (see
        # preprocessing/build_climatology.py), so scaling by its own moments
        # introduces no split leakage.
        clim = np.load(clim_path)["clim"][:, land]
        means.append(float(clim.mean()))
        stds.append(float(clim.std()) if clim.std() > 1e-6 else 1.0)
        # sin/cos of day-of-year are already bounded in [-1, 1] with known
        # moments; hard-code rather than estimate, for the same reason lat/lon
        # use their physical range.
        means.extend([0.0, 0.0])
        stds.extend([0.7071, 0.7071])

    return ArrayScaler(np.array(means, np.float32), np.array(stds, np.float32))
