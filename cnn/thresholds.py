"""Per-fine-cell heavy-rain threshold for the differentiable threat score.

Dong et al. (2025) eq. 3-7 use the 90th percentile of observed precipitation,
per grid cell, computed over the training period only -- using val/test rain to
set the threshold would leak label information into the loss the model is
scored against.

Cells with too few observed rain days (`HEAVY_RAIN_MIN_OBS`) get the domain-wide
train p90 instead of an unstable per-cell estimate.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C


def compute_thresholds() -> np.ndarray:
    g = np.load(C.PROCESSED / "coarse_grid.npz", allow_pickle=True)
    t = np.load(C.PROCESSED / "fine_target.npz")

    train = g["split"].astype(str) == "train"
    target = t["target"][train]
    mask = t["mask"][train]
    land = t["land"]

    thr = np.full(land.shape, np.nan, np.float32)
    global_vals = target[mask & land[None, :, :]]
    global_thr = np.percentile(global_vals, C.HEAVY_RAIN_PERCENTILE) if global_vals.size else 0.0

    n_fallback = 0
    for a, b in np.argwhere(land):
        vals = target[:, a, b][mask[:, a, b]]
        if vals.size >= C.HEAVY_RAIN_MIN_OBS:
            thr[a, b] = np.percentile(vals, C.HEAVY_RAIN_PERCENTILE)
        else:
            thr[a, b] = global_thr
            n_fallback += 1

    print(f"heavy-rain threshold (train p{C.HEAVY_RAIN_PERCENTILE}): "
          f"domain-wide {global_thr:.2f} mm, {n_fallback} cells fell back to it "
          f"(< {C.HEAVY_RAIN_MIN_OBS} observed rain days)")
    return thr


def load_or_compute_thresholds() -> np.ndarray:
    path = C.PROCESSED / "rain_thresholds.npz"
    if path.exists():
        return np.load(path)["threshold"]
    thr = compute_thresholds()
    np.savez(path, threshold=thr)
    return thr


if __name__ == "__main__":
    thr = compute_thresholds()
    np.savez(C.PROCESSED / "rain_thresholds.npz", threshold=thr)
    land = np.load(C.PROCESSED / "fine_target.npz")["land"]
    print(f"wrote {C.PROCESSED / 'rain_thresholds.npz'}")
    print(f"  land-cell thresholds: min {thr[land].min():.2f}  "
          f"mean {thr[land].mean():.2f}  max {thr[land].max():.2f} mm")
