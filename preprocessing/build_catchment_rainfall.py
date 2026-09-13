"""Aggregate downscaled 0.25 deg rainfall into a catchment-average series.

VIC (and, in the leaner path, the inflow model directly) consumes basin-average
forcing, not a grid.  This collapses the 224 catchment fine cells into one
number per (initialisation, lead) -- the mean rainfall over the Mahanadi
catchment upstream of Hirakud.

Three series are produced for every (init, lead):

    rain_pred   downscaled forecast (GBM), the operational input
    rain_ecmwf  raw ECMWF tp mapped to fine cells, the no-downscaling baseline
    rain_obs    IMD observed, the verification target and the "perfect forecast"
                upper bound for the inflow model

Keeping all three means the inflow stage can be scored against a perfect-rainfall
ceiling, separating "the inflow model is weak" from "the rainfall forcing is
weak" -- the decomposition Dong et al. (2025) sect. 5.3 found essential.

Run:  python preprocessing/build_catchment_rainfall.py
Out:  data/processed/catchment_rainfall.parquet
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C
from preprocessing.build_downscaling_dataset import PatchDataset, patch_index_map

GBM_PATH = C.CHECKPOINTS / "gbm.pkl"
GBM_META = C.CHECKPOINTS / "gbm_meta.json"


def design_matrix(ds: PatchDataset, split: str, uses_lags: bool):
    """Flatten patches + statics into the tabular layout the GBM was fit on.

    Column order must match cnn/train_gbm.py exactly: 10 vars x 3 x 3 patch
    cells, then [lat, lon, lead, 4 DEM stats, clim, sin_doy, cos_doy], then --
    if the saved model was trained with them (gbm_meta.json) -- the
    init-relative observation lags in the same order as training.
    """
    idx, _ = patch_index_map()
    a, b = ds.cells[:, 0], ds.cells[:, 1]
    n_time = ds.grid.shape[0]
    rows, cols = idx[a, b, :, :, 0], idx[a, b, :, :, 1]

    patches = ds.grid[:, :, rows, cols].transpose(0, 2, 1, 3, 4)
    patches = patches.reshape(n_time * len(ds.cells), -1)

    lead = np.repeat(ds.lead_day, len(ds.cells))
    doy = np.repeat(ds.doy, len(ds.cells))
    angle = 2 * np.pi * doy / 366.0
    clim = ds.clim[doy, np.tile(a, n_time), np.tile(b, n_time)]

    statics = np.column_stack([
        np.tile(ds.fine_lats[a], n_time),
        np.tile(ds.fine_lons[b], n_time),
        lead,
        np.tile(ds.dem[a, b], (n_time, 1)),
        clim,
        np.sin(angle),
        np.cos(angle),
    ])
    X = np.hstack([patches, statics]).astype(np.float32)

    if uses_lags:
        from cnn.train_gbm import LagSource
        g = np.load(C.PROCESSED / "coarse_grid.npz", allow_pickle=True)
        init = pd.to_datetime(g["time"][g["split"].astype(str) == split])
        lag_X, _ = LagSource().features(
            init.repeat(len(ds.cells)), np.tile(a, n_time), np.tile(b, n_time))
        X = np.hstack([X, lag_X])
    return X, a, b, n_time


def catchment_mean(values, mask, n_time, n_cells):
    """Mean over observed catchment cells, per time step.

    Averaging only where IMD reports data keeps the observed series honest; a
    cell with no observation must not be silently counted as zero rainfall.
    """
    v = values.reshape(n_time, n_cells)
    m = mask.reshape(n_time, n_cells)
    total = np.where(m, v, 0.0).sum(axis=1)
    count = m.sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan), count


def main() -> None:
    if not GBM_PATH.exists():
        sys.exit(f"{GBM_PATH} not found -- train the downscaler first")
    gbm = pickle.load(open(GBM_PATH, "rb"))
    uses_lags = GBM_META.exists() and json.load(open(GBM_META))["uses_lags"]
    print(f"downscaler: gbm.pkl (uses_lags={uses_lags})")

    g = np.load(C.PROCESSED / "coarse_grid.npz", allow_pickle=True)
    feats = list(g["features"])
    tp_i = feats.index("tp")
    clats, clons = np.array(C.COARSE_LATS), np.array(C.COARSE_LONS)

    frames = []
    for split in ["train", "val", "test"]:
        ds = PatchDataset(split, region="catchment")
        X, a, b, n_time = design_matrix(ds, split, uses_lags)
        n_cells = len(ds.cells)

        pred = np.clip(gbm.predict(X), 0.0, None)
        obs = ds.target[:, a, b].ravel()
        mask = ds.mask[:, a, b].ravel()

        # Raw ECMWF at each fine cell = value of its parent coarse cell.
        pi = np.abs(ds.fine_lats[a][:, None] - clats[None, :]).argmin(1)
        pj = np.abs(ds.fine_lons[b][:, None] - clons[None, :]).argmin(1)
        ecmwf = ds.grid[:, tp_i][:, pi, pj].ravel()

        r_pred, count = catchment_mean(pred, mask, n_time, n_cells)
        r_obs, _ = catchment_mean(obs, mask, n_time, n_cells)
        r_ec, _ = catchment_mean(ecmwf, mask, n_time, n_cells)

        init = pd.to_datetime(g["time"][g["split"].astype(str) == split])
        frames.append(pd.DataFrame({
            "init_date": init,
            "lead_day": ds.lead_day.astype(int),
            "valid_date": init + pd.to_timedelta(ds.lead_day, unit="D"),
            "split": split,
            "rain_pred": r_pred,
            "rain_ecmwf": r_ec,
            "rain_obs": r_obs,
            "n_cells_observed": count,
        }))

    df = pd.concat(frames, ignore_index=True).sort_values(["init_date", "lead_day"])
    out = C.PROCESSED / "catchment_rainfall.parquet"
    df.to_parquet(out, index=False)

    print(f"wrote {out}   {len(df):,} rows "
          f"({df['init_date'].nunique()} inits x {df['lead_day'].nunique()} leads)")
    print(f"  cells averaged   {int(df['n_cells_observed'].max())} catchment cells")
    print(f"  any NaN          {int(df[['rain_pred', 'rain_obs', 'rain_ecmwf']].isna().sum().sum())}")
    print("\n  catchment-mean rainfall (mm/d) by split:")
    print(df.groupby("split")[["rain_obs", "rain_pred", "rain_ecmwf"]].mean().round(2).to_string())

    # Skill of the aggregated series -- spatial averaging cancels independent
    # per-cell errors, so this is normally well above the per-cell R^2 of 0.14.
    print("\n  catchment-mean skill vs observed (correlation):")
    for split in ["train", "val", "test"]:
        s = df[df["split"] == split].dropna(subset=["rain_obs"])
        print(f"    {split:5s}  pred {s['rain_pred'].corr(s['rain_obs']):+.3f}   "
              f"ecmwf {s['rain_ecmwf'].corr(s['rain_obs']):+.3f}")


if __name__ == "__main__":
    main()
