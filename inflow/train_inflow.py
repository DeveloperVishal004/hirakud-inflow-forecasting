"""Direct rainfall -> Hirakud inflow model, with baselines and a skill ceiling.

This is the lean path through the chain: forecast catchment rainfall plus
antecedent catchment state, straight to reservoir inflow, with no VIC in
between.  Its purpose is twofold -- deliver a working 1-17 day inflow forecast
now, and establish the number a physically-based hydrological model would have
to beat to justify its cost.

Every model is scored against four references, because an inflow number in
isolation says nothing:

  climatology   day-of-year mean inflow; the "no forecast at all" floor
  persistence   last observed inflow carried forward
  ECMWF-driven  same model, raw ECMWF rainfall instead of downscaled -- isolates
                what the downscaler contributes
  perfect-rain  same model, OBSERVED rainfall instead of forecast -- the ceiling
                the inflow stage could reach if rainfall forecasting were solved

That last one is the important diagnostic (Dong et al. 2025, sect. 5.3): it
separates "the inflow model is weak" from "the rainfall forcing is weak", which
determines where further effort should go.

Run:  python inflow/train_inflow.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

ANTE_Q = [f"q_ante_{w}" for w in (1, 3, 7, 15, 30)]
ANTE_R = [f"r_ante_{w}" for w in (3, 7, 15, 30)]
STATIC = ["lead_day", "doy_sin", "doy_cos", "q_clim"]


def feature_cols(rain_prefix: str) -> list[str]:
    """Feature list, parameterised by which rainfall series drives the model.

    Each forcing gets its own wet_proj_* family so the downscaled-vs-ECMWF
    comparison stays a comparison of forcing quality, not of feature sets.
    """
    wp = {"pred": "wet_proj", "obs": "wet_proj_obs", "ecmwf": "wet_proj_ecmwf"}[rain_prefix]
    proj = [f"{wp}_{w}" for w in (3, 7, 15, 30)]
    if rain_prefix == "pred":
        rain = ["rain_pred", "rain_cum", "rain_3d", "rain_7d"]
    elif rain_prefix == "obs":
        rain = ["rain_obs", "rain_obs_cum", "rain_obs_3d", "rain_obs_7d"]
    elif rain_prefix == "ecmwf":
        rain = ["rain_ecmwf", "rain_ecmwf_cum", "rain_ecmwf_3d", "rain_ecmwf_7d"]
    else:
        raise ValueError(rain_prefix)
    return rain + proj + ANTE_Q + ANTE_R + STATIC


def nse(pred, obs):
    """Nash-Sutcliffe efficiency -- the standard hydrological skill score.

    Algebraically identical to R^2 against the observed mean; reported under the
    hydrology name because that is how Papers 1 and 2 state their results.
    """
    return 1 - ((pred - obs) ** 2).sum() / ((obs - obs.mean()) ** 2).sum()


def kge(pred, obs):
    """Modified Kling-Gupta efficiency (Gupta et al. 2009).

    Decomposes skill into correlation, variability and bias, so a model that
    gets the timing right but the magnitude wrong is not credited the same as
    one that gets both -- the failure mode NSE alone tends to hide.
    """
    r = np.corrcoef(pred, obs)[0, 1]
    alpha = pred.std() / obs.std()
    beta = pred.mean() / obs.mean()
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def score(pred, obs) -> dict:
    pred = np.asarray(pred, float)
    obs = np.asarray(obs, float)
    return {
        "NSE": float(nse(pred, obs)),
        "KGE": float(kge(pred, obs)),
        "RMSE": float(np.sqrt(((pred - obs) ** 2).mean())),
        "MAE": float(np.abs(pred - obs).mean()),
        "corr": float(np.corrcoef(pred, obs)[0, 1]),
        "bias": float(pred.mean() - obs.mean()),
    }


def fit_eval(df, cols, tag, results):
    tr = df[df["split"] == "train"]
    va = df[df["split"] == "val"]
    te = df[df["split"] == "test"]
    y = "target_inflow"

    # Only ~4.2k training rows, so the GBM is kept deliberately small: shallow
    # trees, high min_samples_leaf, early stopping on a held-out slice.
    gbm = HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=60, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=C.SEED,
    )
    gbm.fit(tr[cols], tr[y])

    sc = StandardScaler().fit(tr[cols])
    ridge = Ridge(alpha=10.0).fit(sc.transform(tr[cols]), tr[y])

    # Peak fix: a squared-error model predicts the conditional mean, which
    # flattens flood peaks whenever the inputs are uncertain.  A second GBM fit
    # to the 0.8 quantile deliberately over-predicts; the blend weight between
    # the two is chosen on VALIDATION NSE only, never on test.
    gbm_q = HistGradientBoostingRegressor(
        loss="quantile", quantile=0.8,
        max_iter=600, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=60, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=C.SEED,
    )
    gbm_q.fit(tr[cols], tr[y])

    pv_mean = np.clip(gbm.predict(va[cols]), 0, None)
    pv_q = np.clip(gbm_q.predict(va[cols]), 0, None)
    yv = va[y].to_numpy()
    w_best = max(np.arange(0.0, 0.51, 0.05),
                 key=lambda w: nse((1 - w) * pv_mean + w * pv_q, yv))

    def blend(d):
        return (1 - w_best) * np.clip(gbm.predict(d[cols]), 0, None) \
            + w_best * np.clip(gbm_q.predict(d[cols]), 0, None)

    def ridge_predict(d):
        return ridge.predict(sc.transform(d[cols]))

    for name, fn in [("GBM", lambda d: gbm.predict(d[cols])),
                     ("Ridge [PRODUCTION]", ridge_predict),
                     (f"GBM+q80(w={w_best:.2f})", blend)]:
        for split, d in [("val", va), ("test", te)]:
            p = np.clip(fn(d), 0, None)
            results[f"{tag}|{name}|{split}"] = score(p, d[y].to_numpy())

    # Ridge is the production model, decided by 11-fold leave-one-year-out
    # (inflow/loyo_cv.py), not by the single 2014 split -- on which the GBM
    # happened to look better:
    #
    #     model            median NSE   min      beats clim   negative folds
    #     Ridge              +0.338    +0.028      10/11           0/11
    #     GBM                +0.131    -0.868       5/11           2/11
    #     GBM + q80 blend    +0.123    -0.917       5/11           2/11
    #
    # With ~4.2k rows and ~25 features the boosted trees overfit: they fail to
    # beat climatology in 6 of 11 monsoons and collapse to NSE -0.87 in 2009.
    # Ridge never goes negative.  The q80 blend was added to lift flood peaks
    # and does help on 2014, but across all years it is neutral-to-harmful, so
    # it is reported as a baseline rather than shipped.
    return ridge_predict, cols


def main() -> None:
    df = pd.read_parquet(C.PROCESSED / "inflow_dataset.parquet")
    results: dict = {}

    # ---- reference baselines (no fitting beyond the training climatology)
    for split in ["val", "test"]:
        d = df[df["split"] == split]
        obs = d["target_inflow"].to_numpy()
        results[f"-|Climatology|{split}"] = score(d["q_clim"].to_numpy(), obs)
        results[f"-|Persistence|{split}"] = score(d["q_ante_1"].to_numpy(), obs)

    predict_fn, cols_pred = fit_eval(df, feature_cols("pred"), "downscaled", results)
    fit_eval(df, feature_cols("ecmwf"), "raw-ECMWF", results)
    fit_eval(df, feature_cols("obs"), "perfect-rain", results)

    rows = []
    for key, m in results.items():
        forcing, model, split = key.split("|")
        rows.append({"forcing": forcing, "model": model, "split": split, **m})
    tab = pd.DataFrame(rows)

    print("=== Hirakud inflow, leads 1-17 pooled ===\n")
    for split in ["val", "test"]:
        print(f"--- {split.upper()} ---")
        s = tab[tab["split"] == split].drop(columns="split")
        print(s.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
        print()

    # ---- lead-band breakdown for the operational model
    te = df[df["split"] == "test"]
    p = np.clip(predict_fn(te), 0, None)
    print("--- TEST, downscaled-rainfall Ridge (production), by lead band ---")
    print("  band   n     NSE      KGE     corr    RMSE")
    for lo, hi in [(1, 3), (4, 7), (8, 12), (13, 17)]:
        m = (te["lead_day"] >= lo) & (te["lead_day"] <= hi)
        if m.sum() < 10:
            continue
        sc_ = score(p[m.to_numpy()], te.loc[m, "target_inflow"].to_numpy())
        print(f"  {lo:2d}-{hi:2d} {int(m.sum()):5d} {sc_['NSE']:+8.3f} {sc_['KGE']:+8.3f} "
              f"{sc_['corr']:+7.3f} {sc_['RMSE']:8.0f}")

    C.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    C.METRICS.mkdir(parents=True, exist_ok=True)
    out = C.METRICS / "inflow.json"
    out.write_text(json.dumps(
        {k: v for k, v in results.items()}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
