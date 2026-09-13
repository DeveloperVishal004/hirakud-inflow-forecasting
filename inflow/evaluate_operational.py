"""Operational scoring of the forecast chain: ensemble mean, identical windows.

WHY THIS EXISTS.  inflow/lstm_postproc.py scores every ensemble MEMBER
separately.  That measures how good a single member is, and raw ECMWF members
are wild, so any post-processor that tames them looks excellent:

    median per-year NSE      per-member      ensemble mean
      EC                        -0.466           +0.096
      EC-CNN                    +0.029           +0.039

Averaging the ensemble already removes most of that member noise, so the CNN's
apparent +0.50 advantage collapses to roughly nothing -- and slightly reverses.
An operational forecast uses the ensemble mean, so that is what is reported
here, with the per-member figure kept only as a secondary diagnostic.

THE SHRINKAGE CONTROL.  A forecast whose timing is wrong scores better in
squared error if it is simply damped toward the mean.  So `EC x a`, a single
constant fitted leave-one-year-out on raw EC, is included as a baseline: any
claim that the CNN helps downstream has to beat multiplying by one number.

PEAKS ARE REPORTED SEPARATELY.  NSE rewards the damping that makes a forecast
worse at flood peaks, which is what a dam operator actually needs.  Peak
magnitude ratio and peak timing error are therefore reported alongside.

Run:  python inflow/evaluate_operational.py
Out:  results/metrics/operational_comparison.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C
from inflow.lstm_postproc import LEAD

# EVAL_PAST defines the COMMON EVALUATION SAMPLE and is frozen at 60.
# It used to be imported from lstm_postproc.PAST, which made the set of scored
# windows a side effect of an LSTM hyperparameter: tuning n_seq would silently
# change which windows every OTHER model was scored on, and no comparison would
# be valid across runs.  The LSTM may now train on whatever history length suits
# the basin; every model is still judged on the same 309 windows.
EVAL_PAST = 60

PRODUCTS = ["ec", "ec_qm", "ec_cnn"]


def nse(p, o):
    return float(1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum())


def load_windows():
    """Ensemble-mean forecasts on the windows every product shares.

    A window is kept only when EVAL_PAST (=60) days of antecedent inflow
    are present AND all 30 forecast days are observed, so NSE, peak timing and
    peak magnitude are all computed on exactly the same sample.
    """
    obs_df = pd.read_parquet(C.PROCESSED / "inflow_daily_extended.parquet")
    obs = obs_df[obs_df["inflow_valid"]].set_index("date")["inflow"]

    keep, per_prod, rowmap = None, {}, {}
    for prod in PRODUCTS:
        z = np.load(C.PROCESSED / f"vic_inflow_{prod}.npz", allow_pickle=True)
        q, inits = z["q"], pd.to_datetime([str(x) for x in z["inits"]])
        ok, rows, r = [], [], 0
        for ki, init in enumerate(inits):
            hist = pd.date_range(init - pd.Timedelta(days=EVAL_PAST - 1), init)
            if not all(d in obs.index for d in hist):
                continue
            days = pd.date_range(init + pd.Timedelta(days=1), periods=LEAD)
            y = obs.reindex(days).values.astype(np.float32)
            rows.append((init, np.arange(r, r + q.shape[1])))
            r += q.shape[1]
            ok.append((init, ki, y))
        good = {i for i, _, y in ok if np.isfinite(y).all()}
        keep = good if keep is None else (keep & good)
        per_prod[prod] = (q, ok)
        rowmap[prod] = rows
    return obs, keep, per_prod, rowmap


def main():
    obs, keep, per_prod, rowmap = load_windows()
    inits = sorted(keep)
    years = np.array([i.year for i in inits])
    Y = None
    sims = {}
    for prod in PRODUCTS:
        q, ok = per_prod[prod]
        sel = [(i, ki, y) for i, ki, y in ok if i in keep]
        sel.sort(key=lambda t: t[0])
        sims[prod] = np.array([q[ki].mean(0) for _, ki, _ in sel])
        if Y is None:
            Y = np.array([y for _, _, y in sel])

    # LSTM-corrected ensemble mean, on the same windows
    for prod in PRODUCTS:
        f = C.PROCESSED / f"lstm_inflow_{prod}.npz"
        if not f.exists():
            continue
        pred = np.load(f, allow_pickle=True)["pred"]
        idx = {i: ix for i, ix in rowmap[prod]}
        sims[f"{prod}+lstm"] = np.array([pred[idx[i]].mean(0) for i in inits])

    # shrinkage control: one constant per held-out year, fitted on the others
    grid = np.arange(0.30, 1.51, 0.01)
    shrunk = np.zeros_like(sims["ec"])
    alphas = {}
    for y in sorted(set(years)):
        tr, te = years != y, years == y
        o, p = Y[tr].ravel(), sims["ec"][tr].ravel()
        a = grid[np.argmax([nse(g * p, o) for g in grid])]
        alphas[int(y)] = float(a)
        shrunk[te] = a * sims["ec"][te]
    sims["ec x a (LOYO)"] = shrunk

    out = {"n_windows": len(inits), "years": sorted(set(int(y) for y in years)),
           "shrinkage_alpha_by_year": alphas, "products": {}}
    print(f"{len(inits)} shared windows, {len(set(years))} years, ensemble mean\n")
    hdr = (f"{'product':<16}{'NSE med':>9}{'NSE mean':>10}{'RMSE':>9}"
           f"{'corr':>7}{'bias%':>8}{'peak':>7}{'|dt| d':>8}")
    print(hdr); print("-" * len(hdr))
    for name, S in sims.items():
        per_year = [nse(S[years == y].ravel(), Y[years == y].ravel())
                    for y in sorted(set(years))]
        rmse = float(np.sqrt(((S - Y) ** 2).mean()))
        corr = float(np.corrcoef(S.ravel(), Y.ravel())[0, 1])
        bias = float(100 * (S.mean() - Y.mean()) / Y.mean())
        pk = float(np.median(S.max(1) / Y.max(1)))
        dt = float(np.abs(S.argmax(1) - Y.argmax(1)).mean())
        out["products"][name] = {
            "NSE_median": float(np.median(per_year)),
            "NSE_mean": float(np.mean(per_year)), "RMSE": rmse, "corr": corr,
            "bias_pct": bias, "peak_ratio": pk, "peak_timing_err_days": dt,
            "NSE_by_year": {int(y): float(v)
                            for y, v in zip(sorted(set(years)), per_year)}}
        print(f"{name:<16}{np.median(per_year):>9.3f}{np.mean(per_year):>10.3f}"
              f"{rmse:>9.0f}{corr:>7.3f}{bias:>8.1f}{pk:>7.2f}{dt:>8.2f}")

    C.METRICS.mkdir(parents=True, exist_ok=True)
    p = C.METRICS / "operational_comparison.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
