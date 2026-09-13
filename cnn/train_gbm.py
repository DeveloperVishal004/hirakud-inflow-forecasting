"""Rainfall downscaler v2: GBM with init-relative observation lags.

The GBM replaced the CNN as the production downscaler (test R^2 0.1399 vs
0.1178 on identical features).  This script adds the one untried feature family
with real headroom: rainfall OBSERVED at the target cell up to the issue date.

Legality: every lag is dated init_date - k with k >= 1, so it exists when the
forecast is issued.  This is the correct version of the old notebook's leaky
`rain_lag_*` (which were relative to valid time -- at lead 17 they read rain 16
days into the future).  An assertion enforces the dating rule.

Why it should help: monsoon rainfall is autocorrelated over 1-4 weeks through
active/break spells, so recent observed wetness at the cell is predictive even
when the ECMWF signal has decayed -- exactly the long-lead regime where skill
was weakest.

Runs an ablation (with vs without lags) so the contribution is measured, not
assumed.  The winner is saved as checkpoints/gbm.pkl for the catchment stage.

Run:  python cnn/train_gbm.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C
from preprocessing.build_downscaling_dataset import PatchDataset, patch_index_map

LAG_DAYS = (1, 2, 3)          # single recent days before issue
LAG_WINDOWS = (7, 15, 30)     # means over windows ending the day before issue

GBM_PARAMS = dict(
    max_iter=1200, learning_rate=0.06, max_leaf_nodes=63,
    min_samples_leaf=200, l2_regularization=1.0,
    early_stopping=True, validation_fraction=0.12,
    n_iter_no_change=40, random_state=C.SEED,
)


class LagSource:
    """Per-cell observed-rain lags, indexed by (init_date, cell)."""

    def __init__(self):
        d = np.load(C.PROCESSED / "daily_fine_obs.npz")
        self.obs = d["obs"]
        self.observed = d["observed"]
        self.start = pd.Timestamp(d["start_date"].item())
        # Cumulative sums along time make any window mean O(1).
        self.cum = np.concatenate([np.zeros((1,) + self.obs.shape[1:], np.float32),
                                   np.cumsum(self.obs, axis=0)])

    def day_index(self, dates: pd.DatetimeIndex) -> np.ndarray:
        return (dates - self.start).days.to_numpy()

    def features(self, init_dates: pd.DatetimeIndex, a: np.ndarray, b: np.ndarray):
        """(n_samples, n_lag_features) for sample cells (a, b) at given inits."""
        idx = self.day_index(init_dates)
        cols, names = [], []
        for k in LAG_DAYS:
            day = idx - k
            assert (day >= 0).all(), "lag reaches before the observation record"
            cols.append(self.obs[day, a, b])
            names.append(f"obs_lag_{k}")
        for w in LAG_WINDOWS:
            hi = idx            # cum[hi] - cum[hi-w] = sum over [init-w .. init-1]
            lo = idx - w
            assert (lo >= 0).all(), "window reaches before the observation record"
            cols.append((self.cum[hi, a, b] - self.cum[lo, a, b]) / w)
            names.append(f"obs_ante_{w}")
        return np.column_stack(cols).astype(np.float32), names


def design_matrix(split: str, region: str, lags: LagSource | None):
    ds = PatchDataset(split, region=region)
    idx, _ = patch_index_map()
    a, b = ds.cells[:, 0], ds.cells[:, 1]
    n_time, n_cells = ds.grid.shape[0], len(ds.cells)

    rows, cols = idx[a, b, :, :, 0], idx[a, b, :, :, 1]
    patches = ds.grid[:, :, rows, cols].transpose(0, 2, 1, 3, 4).reshape(n_time * n_cells, -1)

    lead = np.repeat(ds.lead_day, n_cells)
    doy = np.repeat(ds.doy, n_cells)
    angle = 2 * np.pi * doy / 366.0
    clim = ds.clim[doy, np.tile(a, n_time), np.tile(b, n_time)]
    statics = np.column_stack([
        np.tile(ds.fine_lats[a], n_time), np.tile(ds.fine_lons[b], n_time), lead,
        np.tile(ds.dem[a, b], (n_time, 1)), clim, np.sin(angle), np.cos(angle),
    ])
    X = np.hstack([patches, statics]).astype(np.float32)

    if lags is not None:
        g = np.load(C.PROCESSED / "coarse_grid.npz", allow_pickle=True)
        init = pd.to_datetime(g["time"][g["split"].astype(str) == split])
        init_rep = init.repeat(n_cells)
        lag_X, _ = lags.features(init_rep, np.tile(a, n_time), np.tile(b, n_time))
        X = np.hstack([X, lag_X])

    y = ds.target[:, a, b].ravel()
    m = ds.mask[:, a, b].ravel()
    lead_all = np.repeat(ds.lead_day, n_cells)
    return X[m], y[m], lead_all[m]


def r2(p, o):
    return float(1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum())


def main() -> None:
    lags = LagSource()
    results = {}

    for tag, lag_src in [("no-lags", None), ("with-lags", lags)]:
        Xtr, ytr, _ = design_matrix("train", "land", lag_src)
        Xva, yva, _ = design_matrix("val", "catchment", lag_src)
        Xte, yte, lte = design_matrix("test", "catchment", lag_src)

        gbm = HistGradientBoostingRegressor(**GBM_PARAMS)
        gbm.fit(Xtr, ytr)

        pva = np.clip(gbm.predict(Xva), 0, None)
        pte = np.clip(gbm.predict(Xte), 0, None)
        results[tag] = {
            "val_R2": r2(pva, yva),
            "test_R2": r2(pte, yte),
            "test_by_lead": {f"{lo}-{hi}": r2(pte[(lte >= lo) & (lte <= hi)],
                                              yte[(lte >= lo) & (lte <= hi)])
                             for lo, hi in [(1, 3), (4, 7), (8, 12), (13, 17)]},
            "n_iter": int(gbm.n_iter_),
            "model": gbm,
        }
        print(f"{tag:10s} val R2 {results[tag]['val_R2']:+.4f}   "
              f"test R2 {results[tag]['test_R2']:+.4f}   "
              f"by lead {  {k: round(v, 3) for k, v in results[tag]['test_by_lead'].items()} }")

    # Select on validation, report on test.
    winner = max(results, key=lambda k: results[k]["val_R2"])
    print(f"\nselected on val: {winner}")

    out = {k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in results.items()}
    out["selected"] = winner
    C.METRICS.mkdir(parents=True, exist_ok=True)
    (C.METRICS / "downscaler_gbm.json").write_text(json.dumps(out, indent=2))

    C.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    with open(C.CHECKPOINTS / "gbm.pkl", "wb") as f:
        pickle.dump(results[winner]["model"], f)
    with open(C.CHECKPOINTS / "gbm_meta.json", "w") as f:
        json.dump({"variant": winner, "uses_lags": winner == "with-lags",
                   "lag_days": LAG_DAYS, "lag_windows": LAG_WINDOWS}, f, indent=2)
    print(f"saved checkpoints/gbm.pkl ({winner})")


if __name__ == "__main__":
    main()
