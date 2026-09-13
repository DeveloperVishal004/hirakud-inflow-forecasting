"""Decompose forecast MSE into bias, amplitude and phase -- with the aggregation stated.

WHY THIS EXISTS.  docs/FINDINGS.md finding 4 has published these numbers since the
19-monsoon rebuild:

    EC       bias^2 2.7 %  + variance  5.2 % + phase 92.1 %
    EC-CNN   bias^2 2.3 %  + variance 27.1 % + phase 70.6 %

No script in the repository computed them and no results file stored them, so they
could not be reproduced or checked.  This script computes them.

TWO NUMBERS, NOT ONE.  The Murphy/Theil identity

    MSE = (mu_f - mu_o)^2  +  (sd_f - sd_o)^2  +  2 sd_f sd_o (1 - r)
          \___ bias^2 ___/     \_ amplitude _/     \____ phase ____/

is exact for whatever sample you feed it -- but the SHARES depend entirely on how
you aggregate, and here they differ by 37 percentage points:

    pooled      one decomposition over all 309 windows x 30 leads at once
    per-window  decompose each 30-day hydrograph alone, then average the shares

Pooled counts the between-window spread (which monsoon windows are wet) as forecast
variance; per-window removes it.  Neither is wrong.  Quoting either without saying
which is.  Both are written to the output.

Run:  python tools/mse_decomposition.py
Out:  results/metrics/mse_decomposition.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C
from inflow.evaluate_operational import load_windows, PRODUCTS

LABEL = {"ec": "EC -> VIC -> RVIC", "ec_qm": "EC-QM -> VIC -> RVIC",
         "ec_cnn": "EC-CNN -> VIC -> RVIC", "ec_lstm": "EC -> VIC -> LSTM"}


def split(f, o):
    """Murphy/Theil terms for one matched pair of series."""
    f, o = np.asarray(f, float).ravel(), np.asarray(o, float).ravel()
    m = np.isfinite(f) & np.isfinite(o)
    f, o = f[m], o[m]
    sf, so = f.std(), o.std()
    r = np.corrcoef(f, o)[0, 1] if f.size > 1 and sf > 0 and so > 0 else np.nan
    bias2 = (f.mean() - o.mean()) ** 2
    amp = (sf - so) ** 2
    phase = 2 * sf * so * (1 - r)
    return bias2, amp, phase, float(r), float(((f - o) ** 2).mean())


def main():
    obs_df = pd.read_parquet(C.PROCESSED / "inflow_daily_extended.parquet")
    _, keep, per_prod, rowmap = load_windows()
    inits = sorted(keep)

    Y, det = None, {}
    for prod in PRODUCTS:
        q, ok = per_prod[prod]
        sel = sorted([(i, ki, y) for i, ki, y in ok if i in keep], key=lambda t: t[0])
        det[prod] = np.array([q[ki].mean(0) for _, ki, _ in sel])
        if Y is None:
            Y = np.array([y for _, _, y in sel])

    f = C.PROCESSED / "lstm_inflow_ec.npz"
    if f.exists():
        idx = {i: ix for i, ix in rowmap["ec"]}
        det["ec_lstm"] = np.array([np.load(f, allow_pickle=True)["pred"][idx[i]].mean(0)
                                   for i in inits])

    out = {"n_windows": len(inits), "n_leads": int(Y.shape[1]),
           "n_years": len(set(i.year for i in inits)),
           "note": ("Shares depend on aggregation; both are given. Quote the "
                    "aggregation with the number."),
           "pooled": {}, "per_window": {}}

    print(f"{len(inits)} windows x {Y.shape[1]} leads, ensemble mean\n")
    print("POOLED  (one decomposition over every window and lead together)")
    print(f"  {'product':24s}{'MSE':>12}{'bias^2':>9}{'amp':>8}{'phase':>9}{'r':>8}")
    for k in ("ec", "ec_qm", "ec_cnn", "ec_lstm"):
        if k not in det:
            continue
        b, a, p, r, mse = split(det[k], Y)
        t = b + a + p
        out["pooled"][LABEL[k]] = {"MSE": mse, "bias2_pct": 100 * b / t,
                                   "amplitude_pct": 100 * a / t,
                                   "phase_pct": 100 * p / t, "corr": r}
        print(f"  {LABEL[k]:24s}{mse:>12.0f}{100*b/t:>8.1f}%{100*a/t:>7.1f}%"
              f"{100*p/t:>8.1f}%{r:>8.3f}")

    print("\nPER-WINDOW  (each 30-day hydrograph decomposed alone, shares averaged)")
    print(f"  {'product':24s}{'n':>12}{'bias^2':>9}{'amp':>8}{'phase':>9}")
    for k in ("ec", "ec_qm", "ec_cnn", "ec_lstm"):
        if k not in det:
            continue
        B, A, P = [], [], []
        for fw, ow in zip(det[k], Y):
            b, a, p, r, _ = split(fw, ow)
            t = b + a + p
            if not np.isfinite(t) or t <= 0:
                continue
            B.append(b / t); A.append(a / t); P.append(p / t)
        out["per_window"][LABEL[k]] = {"n": len(B), "bias2_pct": 100 * float(np.mean(B)),
                                       "amplitude_pct": 100 * float(np.mean(A)),
                                       "phase_pct": 100 * float(np.mean(P))}
        print(f"  {LABEL[k]:24s}{len(B):>12}{100*np.mean(B):>8.1f}%"
              f"{100*np.mean(A):>7.1f}%{100*np.mean(P):>8.1f}%")

    pp = out["pooled"]["EC -> VIC -> RVIC"]["phase_pct"]
    pw = out["per_window"]["EC -> VIC -> RVIC"]["phase_pct"]
    out["aggregation_sensitivity_points"] = abs(pp - pw)
    print(f"\n  EC phase share: {pp:.1f} % pooled vs {pw:.1f} % per-window "
          f"-> {abs(pp-pw):.0f} points apart.")
    print("  Phase is the largest single term under BOTH aggregations, but the")
    print("  exact percentage is not a stable project result.  State which one.")

    C.METRICS.mkdir(parents=True, exist_ok=True)
    dst = C.METRICS / "mse_decomposition.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
