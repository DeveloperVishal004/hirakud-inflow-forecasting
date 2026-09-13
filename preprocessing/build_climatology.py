"""Per-cell seasonal rainfall climatology, fit from the TRAIN split only.

Why this exists.  A skill diagnostic on the test split showed ECMWF's own
forecast correlation with observed rainfall collapses beyond lead 7:

    lead  1-7 : corr 0.30 - 0.48
    lead 8-17 : corr 0.06 - 0.22  (negative at lead 17)

So for more than half the sample there is essentially no forecast signal to
extract, and the best available prediction is climatology.  Measured on the test
split, per-cell seasonal climatology alone scores R^2 = +0.055 against a
global-mean baseline of 0.000 -- free skill the model currently cannot use,
because its static input vector carries no notion of date at all.

Handing the model this climatology (plus a day-of-year encoding) lets it learn
the thing the lead-skill table implies it should do: lean on the forecast at
short lead, and fall back toward climatology as lead grows.

Leakage: the climatology is built strictly from `split == "train"` rows.  Using
val/test rainfall to define it would leak the answer into the input.

Run:  python preprocessing/build_climatology.py
Out:  data/processed/climatology.npz
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

# Half-width of the day-of-year smoothing window.  A single calendar day has
# only ~8 training years x a few cells of support, which is far too noisy; +-15
# days pools ~31x that while staying well inside the monsoon's seasonal cycle.
DOY_HALF_WINDOW = 15


def build() -> None:
    g = np.load(C.PROCESSED / "coarse_grid.npz", allow_pickle=True)
    t = np.load(C.PROCESSED / "fine_target.npz")

    split = g["split"].astype(str)
    lead = g["lead_day"]
    valid = pd.to_datetime(g["time"]) + pd.to_timedelta(lead, unit="D")
    doy = valid.dayofyear.values

    train = split == "train"
    target, mask = t["target"][train], t["mask"][train]
    doy_train = doy[train]
    land = t["land"]

    shape = land.shape
    total = np.zeros((367,) + shape)
    count = np.zeros((367,) + shape)
    for i, d in enumerate(doy_train):
        total[d] += np.where(mask[i], target[i], 0.0)
        count[d] += mask[i]

    # Circular +-window sum, so late-September and early-October share support.
    sm_total = np.zeros_like(total)
    sm_count = np.zeros_like(count)
    for d in range(1, 367):
        window = [(d + k - 1) % 366 + 1 for k in range(-DOY_HALF_WINDOW, DOY_HALF_WINDOW + 1)]
        sm_total[d] = total[window].sum(0)
        sm_count[d] = count[window].sum(0)

    global_mean = target[mask].mean()
    clim = np.where(sm_count > 0, sm_total / np.maximum(sm_count, 1), np.nan)
    # Cells/days with no training support (e.g. non-land) fall back to the
    # global train mean rather than NaN, so the feature is always defined.
    clim = np.where(np.isfinite(clim), clim, global_mean).astype(np.float32)

    np.savez_compressed(
        C.PROCESSED / "climatology.npz",
        clim=clim,                       # (367, n_lat, n_lon), indexed by day-of-year
        global_mean=np.float32(global_mean),
        doy_half_window=np.int16(DOY_HALF_WINDOW),
    )

    covered = clim[:, land]
    print(f"wrote {C.PROCESSED / 'climatology.npz'}")
    print(f"  shape            {clim.shape}  (day-of-year x fine grid)")
    print(f"  train global mean {global_mean:.3f} mm")
    print(f"  land-cell clim    min {covered.min():.2f}  mean {covered.mean():.2f}  max {covered.max():.2f} mm")
    peak = np.nanmean(clim[:, land], axis=1)
    order = np.argsort(-peak)[:3]
    print(f"  wettest days-of-year: {sorted(order.tolist())} "
          f"(peak {peak[order[0]]:.1f} mm) -- expect mid-monsoon, ~Jul-Aug")


if __name__ == "__main__":
    build()
