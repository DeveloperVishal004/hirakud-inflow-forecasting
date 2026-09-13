"""Leave-one-year-out cross-validation of the full chain, end to end.

Every number reported so far rests on one test monsoon (2014).  With 11 monsoons
available and validation-vs-test differing by ~0.1 NSE, a single held-out year
cannot distinguish "the model works" from "2014 was an easy year".  This runs
each of the 11 years as the held-out test set and reports the distribution.

Two passes, so the rainfall forcing is never in-sample:

  Pass 1  for each year Y, refit the downscaler WITHOUT Y and predict Y.  Every
          year therefore ends up with out-of-fold rainfall.  Climatology is
          refit per fold too -- it is derived from observed rainfall, so reusing
          a climatology built on all years would leak Y into its own forcing.
  Pass 2  for each year Y, fit the inflow model on the other years' out-of-fold
          rainfall and score it on Y.

Training the inflow stage on out-of-fold (rather than in-sample) rainfall
matters: in-sample forcing is unrealistically accurate, so a model trained on it
learns to trust rainfall more than it should and degrades when deployed.

Runtime is dominated by pass 1 (11 downscaler fits on ~2.2M rows).

Run:  python inflow/loyo_cv.py
Out:  results/metrics/loyo.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C
from preprocessing.build_downscaling_dataset import patch_index_map
from cnn.train_gbm import LagSource, LAG_DAYS, LAG_WINDOWS
from inflow.train_inflow import nse, kge, score, ANTE_Q, ANTE_R, STATIC

# Years come from the inflow record, not from config: C.YEAR_MAX is still 2014
# and is read by 22 modules, so widen here rather than move it under all of them.
def _years():
    import pandas as _pd
    f = next((q for q in (C.PROCESSED / "inflow_daily_extended.parquet",
                          C.PROCESSED / "inflow_daily.parquet") if q.exists()), None)
    if f is None:
        return list(range(C.YEAR_MIN, C.YEAR_MAX + 1))
    d = _pd.read_parquet(f)
    d = d[d["inflow_valid"]] if "inflow_valid" in d.columns else d
    ys = sorted(set(_pd.to_datetime(d["date"]).dt.year))
    return [y for y in ys if y >= C.YEAR_MIN]


YEARS = _years()
DOY_HALF_WINDOW = 15

# Fewer iterations than the production fit (1200): 11 folds x 2 passes, and the
# fold-to-fold spread we are measuring is far larger than the small accuracy
# given up here.
DOWNSCALER_PARAMS = dict(
    max_iter=500, learning_rate=0.08, max_leaf_nodes=63,
    min_samples_leaf=200, l2_regularization=1.0,
    early_stopping=True, validation_fraction=0.12,
    n_iter_no_change=30, random_state=C.SEED,
)
INFLOW_PARAMS = dict(
    max_iter=600, learning_rate=0.05, max_leaf_nodes=15,
    min_samples_leaf=60, l2_regularization=1.0,
    early_stopping=True, validation_fraction=0.15,
    n_iter_no_change=30, random_state=C.SEED,
)


def load_all():
    g = np.load(C.PROCESSED / "coarse_grid.npz", allow_pickle=True)
    t = np.load(C.PROCESSED / "fine_target.npz")
    d = np.load(C.PROCESSED / "dem_fine.npz", allow_pickle=True)
    dem = np.stack([d[k] for k in ["elev_mean", "elev_std", "elev_min", "elev_max"]], -1)
    init = pd.to_datetime(g["time"])
    lead = g["lead_day"].astype(int)
    return {
        "grid": g["grid"], "init": init, "lead": lead,
        "valid": init + pd.to_timedelta(lead, unit="D"),
        "target": t["target"], "mask": t["mask"],
        "land": t["land"], "catchment": t["catchment"],
        "lats": t["fine_lats"], "lons": t["fine_lons"], "dem": dem,
        "features": list(g["features"]),
    }


def fold_climatology(D, fit_rows) -> np.ndarray:
    """Day-of-year x cell rainfall climatology from the fold's training rows."""
    doy = D["valid"].dayofyear.values
    shape = D["land"].shape
    total = np.zeros((367,) + shape)
    count = np.zeros((367,) + shape)
    tgt, msk = D["target"][fit_rows], D["mask"][fit_rows]
    for i, dd in enumerate(doy[fit_rows]):
        total[dd] += np.where(msk[i], tgt[i], 0.0)
        count[dd] += msk[i]
    sm_t = np.zeros_like(total)
    sm_c = np.zeros_like(count)
    for dd in range(1, 367):
        w = [(dd + k - 1) % 366 + 1 for k in range(-DOY_HALF_WINDOW, DOY_HALF_WINDOW + 1)]
        sm_t[dd] = total[w].sum(0)
        sm_c[dd] = count[w].sum(0)
    gm = tgt[msk].mean()
    clim = np.where(sm_c > 0, sm_t / np.maximum(sm_c, 1), np.nan)
    return np.where(np.isfinite(clim), clim, gm).astype(np.float32)


def design(D, rows, cells, clim, lags: LagSource):
    """Tabular design matrix for the given time rows and fine cells."""
    idx, _ = patch_index_map()
    a, b = cells[:, 0], cells[:, 1]
    n_t, n_c = int(rows.sum()), len(cells)
    prow, pcol = idx[a, b, :, :, 0], idx[a, b, :, :, 1]

    patches = D["grid"][rows][:, :, prow, pcol].transpose(0, 2, 1, 3, 4).reshape(n_t * n_c, -1)
    lead = np.repeat(D["lead"][rows], n_c)
    doy = np.repeat(D["valid"][rows].dayofyear.values, n_c)
    ang = 2 * np.pi * doy / 366.0
    a_t, b_t = np.tile(a, n_t), np.tile(b, n_t)
    statics = np.column_stack([
        np.tile(D["lats"][a], n_t), np.tile(D["lons"][b], n_t), lead,
        np.tile(D["dem"][a, b], (n_t, 1)), clim[doy, a_t, b_t], np.sin(ang), np.cos(ang),
    ])
    lag_X, _ = lags.features(D["init"][rows].repeat(n_c), a_t, b_t)
    X = np.hstack([patches, statics, lag_X]).astype(np.float32)

    y = D["target"][rows][:, a, b].ravel()
    m = D["mask"][rows][:, a, b].ravel()
    return X, y, m, n_t, n_c


def pass1_out_of_fold_rainfall(D, lags) -> pd.DataFrame:
    """Refit the downscaler per fold; return out-of-fold catchment rainfall."""
    land_cells = np.argwhere(D["land"])
    cat_cells = np.argwhere(D["catchment"])
    init_year = D["init"].year.values
    frames = []

    for yi, Y in enumerate(YEARS, 1):
        t0 = time.time()
        fit_rows = init_year != Y
        pred_rows = init_year == Y
        clim = fold_climatology(D, fit_rows)

        Xtr, ytr, mtr, _, _ = design(D, fit_rows, land_cells, clim, lags)
        gbm = HistGradientBoostingRegressor(**DOWNSCALER_PARAMS).fit(Xtr[mtr], ytr[mtr])

        Xte, yte, mte, n_t, n_c = design(D, pred_rows, cat_cells, clim, lags)
        pred = np.clip(gbm.predict(Xte), 0, None)

        # Catchment mean over observed cells only.
        pr, ob, mk = (v.reshape(n_t, n_c) for v in (pred, yte, mte))
        cnt = mk.sum(1)
        r_pred = np.where(cnt > 0, np.where(mk, pr, 0).sum(1) / np.maximum(cnt, 1), np.nan)
        r_obs = np.where(cnt > 0, np.where(mk, ob, 0).sum(1) / np.maximum(cnt, 1), np.nan)

        cell_r2 = 1 - ((pred[mte] - yte[mte]) ** 2).sum() / ((yte[mte] - yte[mte].mean()) ** 2).sum()
        frames.append(pd.DataFrame({
            "init_date": D["init"][pred_rows], "lead_day": D["lead"][pred_rows],
            "valid_date": D["valid"][pred_rows], "fold_year": Y,
            "rain_pred": r_pred, "rain_obs": r_obs,
        }))
        print(f"  [{yi:2d}/11] hold out {Y}: per-cell R2 {cell_r2:+.4f}  ({time.time() - t0:.0f}s)")

    return pd.concat(frames, ignore_index=True)


def build_inflow_frame(rain: pd.DataFrame) -> pd.DataFrame:
    """Attach antecedent state, projected wetness and targets to out-of-fold rainfall."""
    q = pd.read_parquet(next(
        (f for f in (C.PROCESSED / "inflow_daily_extended.parquet",
                     C.PROCESSED / "inflow_daily.parquet") if f.exists()),
        C.PROCESSED / "inflow_daily.parquet")).rename(columns={"date": "d"})
    q = q.set_index("d").sort_index()
    inflow, ok = q["inflow"], q["inflow_valid"]
    masked = inflow.where(ok)

    obs_daily = pd.read_parquet(C.PROCESSED / "daily_catchment_obs.parquet")
    obs_full = obs_daily.set_index("date")["rain_obs_daily"].sort_index()
    obs_full = obs_full.reindex(pd.date_range(obs_full.index.min(), obs_full.index.max(), freq="D"))

    windows = (3, 7, 15, 30)
    ante = pd.DataFrame({f"q_ante_{w}": masked.rolling(w, min_periods=1).mean() for w in windows})
    ante["q_ante_1"] = masked
    r_ante = pd.DataFrame({f"r_ante_{w}": obs_full.rolling(w, min_periods=1).mean() for w in windows})

    df = rain.sort_values(["init_date", "lead_day"]).copy()
    for src, tag in [("rain_pred", "rain"), ("rain_obs", "rain_obs")]:
        g = df.groupby("init_date", sort=False)[src]
        df[f"{tag}_cum"] = g.cumsum()
        df[f"{tag}_3d"] = g.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"{tag}_7d"] = g.rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)

    cutoff = df["init_date"] - pd.Timedelta(days=1)
    for col in ante.columns:
        df[col] = ante[col].reindex(cutoff).to_numpy()
    for col in r_ante.columns:
        df[col] = r_ante[col].reindex(cutoff).to_numpy()

    for w in windows:
        n_ob = w - np.minimum(df["lead_day"], w)
        ob_part = df[f"r_ante_{w}"] * n_ob
        for cum, out in [("rain_cum", f"wet_proj_{w}"), ("rain_obs_cum", f"wet_proj_obs_{w}")]:
            fc = df[cum] - np.where(df["lead_day"] > w,
                                    df.groupby("init_date")[cum].shift(w).fillna(0.0), 0.0)
            df[out] = (fc + ob_part) / w

    df["target_inflow"] = inflow.reindex(df["valid_date"]).to_numpy()
    df["target_valid"] = ok.reindex(df["valid_date"]).fillna(False).to_numpy()
    doy = df["valid_date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 366.0)
    df["_doy"] = doy

    df = df.dropna(subset=["target_inflow"])
    df = df[df["target_valid"]].copy()
    df["q_ante_1"] = df["q_ante_1"].fillna(df["q_ante_3"])
    return df.dropna(subset=[f"q_ante_{w}" for w in windows])


def main() -> None:
    # Pass 1 is ~45 min of downscaler refits and its output does not depend on
    # anything in pass 2, so it is cached: iterating on the inflow stage should
    # not require recomputing the rainfall folds.
    oof_path = C.PROCESSED / "loyo_oof_rainfall.parquet"
    if oof_path.exists():
        print(f"pass 1: reusing cached out-of-fold rainfall ({oof_path.name})")
        rain = pd.read_parquet(oof_path)
    else:
        D = load_all()
        lags = LagSource()
        print("pass 1: out-of-fold rainfall (refitting downscaler per held-out year)")
        rain = pass1_out_of_fold_rainfall(D, lags)
        rain.to_parquet(oof_path, index=False)
        print(f"  cached -> {oof_path.name}")

    df = build_inflow_frame(rain)
    cols = (["rain_pred", "rain_cum", "rain_3d", "rain_7d"]
            + [f"wet_proj_{w}" for w in (3, 7, 15, 30)] + ANTE_Q + ANTE_R + STATIC)

    print("\npass 2: inflow model per held-out year")
    year = df["valid_date"].dt.year
    folds, preds = [], []
    for Y in YEARS:
        tr, te = df[year != Y], df[year == Y]
        if len(te) < 20:
            continue
        # Inflow climatology, like everything else, is refit per fold.
        cl = tr.groupby("_doy")["target_inflow"].mean().reindex(range(1, 367))
        cl = pd.concat([cl, cl, cl]).rolling(31, center=True, min_periods=1).mean().iloc[366:732]
        cl.index = range(1, 367)
        cl = cl.fillna(tr["target_inflow"].mean())
        tr, te = tr.copy(), te.copy()
        tr["q_clim"] = cl.reindex(tr["_doy"]).to_numpy()
        te["q_clim"] = cl.reindex(te["_doy"]).to_numpy()

        y = "target_inflow"
        gbm = HistGradientBoostingRegressor(**INFLOW_PARAMS).fit(tr[cols], tr[y])
        gq = HistGradientBoostingRegressor(loss="quantile", quantile=0.8,
                                           **INFLOW_PARAMS).fit(tr[cols], tr[y])
        sc = StandardScaler().fit(tr[cols])
        ridge = Ridge(alpha=10.0).fit(sc.transform(tr[cols]), tr[y])

        obs = te[y].to_numpy()
        p_gbm = np.clip(gbm.predict(te[cols]), 0, None)
        p_bl = 0.5 * p_gbm + 0.5 * np.clip(gq.predict(te[cols]), 0, None)
        p_rid = np.clip(ridge.predict(sc.transform(te[cols])), 0, None)

        row = {"year": Y, "n": len(te)}
        for nm, p in [("GBM", p_gbm), ("Blend", p_bl), ("Ridge", p_rid),
                      ("Climatology", te["q_clim"].to_numpy())]:
            s = score(p, obs)
            row[f"{nm}_NSE"], row[f"{nm}_KGE"] = s["NSE"], s["KGE"]
        for lo, hi in [(1, 3), (4, 7), (8, 12), (13, 17)]:
            m = ((te["lead_day"] >= lo) & (te["lead_day"] <= hi)).to_numpy()
            for nm, p in [("Blend", p_bl), ("Ridge", p_rid), ("Climatology", te["q_clim"].to_numpy())]:
                row[f"{nm}_NSE_{lo}-{hi}"] = nse(p[m], obs[m]) if m.sum() > 5 else np.nan
        folds.append(row)
        preds.append(pd.DataFrame({"year": Y, "valid_date": te["valid_date"],
                                   "lead_day": te["lead_day"], "obs": obs, "pred": p_bl}))
        print(f"  {Y}: GBM {row['GBM_NSE']:+.3f}  Blend {row['Blend_NSE']:+.3f}  "
              f"Ridge {row['Ridge_NSE']:+.3f}  Clim {row['Climatology_NSE']:+.3f}")

    res = pd.DataFrame(folds)
    print("\n=== LOYO summary over %d monsoons ===" % len(res))
    for m in ["GBM", "Blend", "Ridge", "Climatology"]:
        v = res[f"{m}_NSE"]
        print(f"  {m:12s} NSE  median {v.median():+.3f}  mean {v.mean():+.3f}  "
              f"IQR [{v.quantile(.25):+.3f}, {v.quantile(.75):+.3f}]  "
              f"min {v.min():+.3f}  max {v.max():+.3f}")
    print("\n  NSE by lead band (median over folds) -- this is the honest skill horizon:")
    print(f"    {'band':>8}  {'Ridge':>8}  {'Blend':>8}  {'Clim':>8}")
    for lo, hi in [(1, 3), (4, 7), (8, 12), (13, 17)]:
        print(f"    {lo:3d}-{hi:<4d}"
              f"  {res[f'Ridge_NSE_{lo}-{hi}'].median():+8.3f}"
              f"  {res[f'Blend_NSE_{lo}-{hi}'].median():+8.3f}"
              f"  {res[f'Climatology_NSE_{lo}-{hi}'].median():+8.3f}")

    best = max(["GBM", "Blend", "Ridge"], key=lambda m: res[f"{m}_NSE"].median())
    beats = int((res[f"{best}_NSE"] > res["Climatology_NSE"]).sum())
    print(f"\n  Selected by median LOYO NSE: {best}  "
          f"(beats climatology in {beats}/{len(res)} folds, "
          f"negative in {int((res[f'{best}_NSE'] < 0).sum())})")

    C.METRICS.mkdir(parents=True, exist_ok=True)
    res.to_json(C.METRICS / "loyo.json", orient="records", indent=2)
    pd.concat(preds).to_parquet(C.PROCESSED / "loyo_predictions.parquet", index=False)
    print("\nwrote results/metrics/loyo.json and data/processed/loyo_predictions.parquet")


if __name__ == "__main__":
    main()
