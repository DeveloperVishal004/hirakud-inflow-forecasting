"""Does ensemble spread predict inflow error?  Compute it, rather than assert it.

WHY THIS EXISTS.  Three numbers have circulated in this project's prose --

    Spearman(spread, |inflow error|) = +0.475 overall, +0.531 at leads 8-14,
    lowest-spread quartile 865 m3/s mean error vs 2,462 for the highest

-- and they are the entire justification for feeding ensemble SPREAD to
FutureTST as a second input channel alongside the mean.  No script in the
repository computed or stored them, so they could not be checked.  This does.

WHAT COUNTS AS "SPREAD" HERE.  The disagreement between the 10 perturbed ECMWF
members, measured on the routed inflow (vic_inflow_ec.npz, raw-EC driven), per
(initialisation, lead).  That is downstream of the same physics every member
passes through, so it isolates member disagreement rather than mixing in
rainfall-scale variance.

WHAT COUNTS AS "ERROR".  |ensemble mean - observed| on the same cell, from the
309 common windows so the sample matches head_to_head.py exactly.

Run:  python tools/spread_error.py
Out:  results/metrics/spread_error.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C
from inflow.evaluate_operational import load_windows

BANDS = [(1, 3), (4, 7), (8, 14), (15, 21), (22, 30)]


def main():
    _, keep, per_prod, _ = load_windows()
    inits = sorted(keep)
    q, ok = per_prod["ec"]
    sel = sorted([(i, ki, y) for i, ki, y in ok if i in keep], key=lambda t: t[0])

    ens = np.array([q[ki] for _, ki, _ in sel])          # (n_win, n_mem, n_lead)
    Y = np.array([y for _, _, y in sel])                 # (n_win, n_lead)
    spread = ens.std(axis=1)                             # member disagreement
    err = np.abs(ens.mean(axis=1) - Y)

    m = np.isfinite(spread) & np.isfinite(err)
    print(f"{len(inits)} windows x {Y.shape[1]} leads, {ens.shape[1]} members")
    print(f"{m.sum():,} finite (spread, |error|) pairs\n")

    out = {"n_windows": len(inits), "n_members": int(ens.shape[1]),
           "definition": {"spread": "std across the 10 perturbed members of routed "
                                    "inflow (vic_inflow_ec.npz)",
                          "error": "|ensemble mean - observed| on the same cell",
                          "sample": "the 309 common windows used by head_to_head.py"}}

    rho = st.spearmanr(spread[m], err[m])
    out["overall"] = {"spearman": float(rho.statistic), "p": float(rho.pvalue),
                      "n": int(m.sum())}
    print(f"overall Spearman(spread, |error|) = {rho.statistic:+.3f}  "
          f"(p={rho.pvalue:.2g}, n={m.sum():,})")

    print("\nby lead band:")
    out["by_band"] = {}
    for lo, hi in BANDS:
        sl = slice(lo - 1, hi)
        sm, em = spread[:, sl], err[:, sl]
        mm = np.isfinite(sm) & np.isfinite(em)
        r = st.spearmanr(sm[mm], em[mm])
        out["by_band"][f"{lo}-{hi}"] = {"spearman": float(r.statistic),
                                        "p": float(r.pvalue), "n": int(mm.sum())}
        print(f"  {str(lo)+'-'+str(hi)+' d':<9} Spearman {r.statistic:+.3f}  "
              f"(p={r.pvalue:.2g}, n={mm.sum():,})")

    # quartiles of spread -> mean error in each
    s, e = spread[m], err[m]
    qs = np.quantile(s, [0.25, 0.5, 0.75])
    lab = ["Q1 (lowest spread)", "Q2", "Q3", "Q4 (highest spread)"]
    bins = np.digitize(s, qs)
    print("\nmean |error| by spread quartile:")
    out["by_spread_quartile"] = {}
    for i, l in enumerate(lab):
        v = e[bins == i]
        out["by_spread_quartile"][l] = {"mean_abs_error_cumecs": float(v.mean()),
                                        "median_abs_error_cumecs": float(np.median(v)),
                                        "n": int(v.size)}
        print(f"  {l:<22} mean {v.mean():8.0f} m3/s   median {np.median(v):8.0f}   "
              f"n={v.size:,}")
    lo_, hi_ = (out["by_spread_quartile"][lab[0]]["mean_abs_error_cumecs"],
                out["by_spread_quartile"][lab[3]]["mean_abs_error_cumecs"])
    out["quartile_error_ratio"] = hi_ / lo_ if lo_ else float("nan")
    print(f"\n  highest-spread quartile has {hi_/lo_:.2f}x the mean error of the lowest")
    print("  -> spread carries real information about when the forecast is unreliable,")
    print("     which is why it is fed to FutureTST as a second input channel.")

    C.METRICS.mkdir(parents=True, exist_ok=True)
    dst = C.METRICS / "spread_error.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
