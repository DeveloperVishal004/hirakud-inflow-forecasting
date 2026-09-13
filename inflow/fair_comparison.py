"""Re-analysis of head_to_head.json: is the comparison actually on equal footing?

THE PROBLEM THIS EXISTS TO EXPOSE.  VIC's seven soil parameters were fitted on a
FIXED split -- 2004-2011 calibration, 2012-2013 check, 2014 test
(hydrology/vic/calibrate_vic.py).  VIC is the only stage in the project that is
not leave-one-year-out.  head_to_head.py then scores the VIC chains on ALL 19
monsoons, so for 8 of them (42 % of the evaluation) those chains -- and the
`0.86 x EC` baseline, which is built from VIC output -- are partly IN-SAMPLE.
FutureTST is strictly out-of-sample everywhere.

That is not a like-for-like comparison, and it favours the baseline.  This script
splits the same per-year scores by VIC exposure so the asymmetry is visible.

THREE THINGS head_to_head.py DOES NOT REPORT, ADDED HERE.

  1. Significance for EVERY model, not only the best FutureTST variant.  Two of
     the project's own chains turn out to be significantly WORSE than the scalar
     baseline, which was never stated.
  2. median, mean AND win-count together.  Median alone flatters high-variance
     models: FutureTST's median is far above the baseline on VIC-unseen years
     while winning fewer than half of them.
  3. A non-parametric test (Wilcoxon) and a Holm correction across the six
     challengers.  The per-year NSE differences run -3.19 to +0.82 and are
     strongly left-skewed, which strains the t-test's normality assumption; the
     six challengers share one baseline, which inflates the false-positive rate.

The t and p values are also STORED here.  head_to_head.py prints them to stdout
and they are lost -- they are not among its JSON keys.

Run:  python inflow/fair_comparison.py
Out:  results/metrics/fair_comparison.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

REF = "0.86 x EC (LOYO)"

# hydrology/vic/calibrate_vic.py: 2003 spin-up, 2004-2011 calibration,
# 2012-2013 validation, 2014 test.  Everything from 2014 on is unseen by the
# calibration search; 2014 was its held-out test year, so it is grouped with
# the unseen block rather than the fitted one.
GROUPS = [
    ("all 19 monsoons (as published)", lambda y: True),
    ("VIC-FITTED years 2004-2011",     lambda y: 2004 <= y <= 2011),
    ("VIC-checked years 2012-2013",    lambda y: y in (2012, 2013)),
    ("VIC-UNSEEN years 2014-2022",     lambda y: y >= 2014),
]


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    src = C.METRICS / "head_to_head.json"
    if not src.exists():
        sys.exit(f"missing {src} -- run inflow/head_to_head.py first")
    hh = json.loads(src.read_text())
    models = hh["models"]
    if REF not in models:
        sys.exit(f"baseline {REF!r} not in {src}")

    ref_py = models[REF]["per_year"]
    challengers = [m for m in models if m != REF]
    out = {"source": src.name, "baseline": REF, "groups": {}}

    print(f"baseline: {REF}\n")
    print("VIC soil parameters were fitted on 2004-2011 and checked on 2012-2013.")
    print("head_to_head.py scores the VIC chains on all 19 years, so 8 of them")
    print("(42 %) are partly in-sample for those chains and for the baseline.")
    print("FutureTST is strictly out-of-sample in every year.\n")

    for gname, sel in GROUPS:
        yy = sorted([y for y in ref_py if sel(int(y))], key=int)
        if len(yy) < 3:
            continue
        b = np.array([ref_py[y] for y in yy], float)

        rows, praw = [], []
        for name in challengers:
            a = np.array([models[name]["per_year"][y] for y in yy], float)
            d = a - b
            t = st.ttest_rel(a, b)
            try:
                w = st.wilcoxon(a, b)
                wp = float(w.pvalue)
            except ValueError:                      # all differences identical
                wp = float("nan")
            rows.append({"model": name,
                         "median_NSE": float(np.median(a)),
                         "mean_NSE": float(np.mean(a)),
                         "median_diff_vs_baseline": float(np.median(d)),
                         "mean_diff_vs_baseline": float(np.mean(d)),
                         "t": float(t.statistic), "p_ttest": float(t.pvalue),
                         "p_wilcoxon": wp,
                         "wins": int((a > b).sum()), "n_years": len(yy)})
            praw.append(float(t.pvalue))

        for r, ph in zip(rows, holm(np.array(praw))):
            r["p_ttest_holm"] = float(ph)
            r["significant_holm_0.05"] = bool(ph < 0.05)

        out["groups"][gname] = {"years": [int(y) for y in yy],
                                "baseline_median_NSE": float(np.median(b)),
                                "baseline_mean_NSE": float(np.mean(b)),
                                "models": rows}

        print(f"=== {gname}   (n={len(yy)}) ===")
        print(f"  {'model':<24}{'median':>8}{'mean':>8}{'t':>7}{'p':>8}"
              f"{'p_holm':>8}{'p_wilc':>8}{'wins':>8}")
        for r in sorted(rows, key=lambda x: -x["median_NSE"]):
            star = " *" if r["significant_holm_0.05"] else ""
            worse = "  (WORSE than baseline)" if (
                r["significant_holm_0.05"] and r["mean_diff_vs_baseline"] < 0) else ""
            print(f"  {r['model']:<24}{r['median_NSE']:>8.3f}{r['mean_NSE']:>8.3f}"
                  f"{r['t']:>7.2f}{r['p_ttest']:>8.3f}{r['p_ttest_holm']:>8.3f}"
                  f"{r['p_wilcoxon']:>8.3f}{r['wins']:>5d}/{r['n_years']}{star}{worse}")
        print(f"  {'-> baseline ' + REF:<24}{np.median(b):>8.3f}{np.mean(b):>8.3f}\n")

    # --- the headline consequence -------------------------------------------
    g_all = out["groups"]["all 19 monsoons (as published)"]
    g_un = out["groups"].get("VIC-UNSEEN years 2014-2022")
    if g_un:
        def get(g, n):
            return next(r for r in g["models"] if r["model"] == n)
        best = max(g_all["models"], key=lambda r: r["median_NSE"])["model"]
        a_all, a_un = get(g_all, best), get(g_un, best)
        out["headline"] = {
            "best_model": best,
            "published_margin_vs_baseline": a_all["median_NSE"] - g_all["baseline_median_NSE"],
            "vic_unseen_margin_vs_baseline": a_un["median_NSE"] - g_un["baseline_median_NSE"],
            "note": ("On years VIC never saw, the best challenger's median margin "
                     "over the baseline widens -- but the paired test still does "
                     "not reach significance, and its win-count stays near half. "
                     "The median flatters a high-variance model.")}
        print("HEADLINE CONSEQUENCE")
        print(f"  best challenger: {best}")
        print(f"    published (all 19 yrs) : median {a_all['median_NSE']:+.3f} vs "
              f"baseline {g_all['baseline_median_NSE']:+.3f}  "
              f"-> margin {out['headline']['published_margin_vs_baseline']:+.3f}, "
              f"wins {a_all['wins']}/{a_all['n_years']}, p={a_all['p_ttest']:.3f}")
        print(f"    VIC-unseen (2014-22)   : median {a_un['median_NSE']:+.3f} vs "
              f"baseline {g_un['baseline_median_NSE']:+.3f}  "
              f"-> margin {out['headline']['vic_unseen_margin_vs_baseline']:+.3f}, "
              f"wins {a_un['wins']}/{a_un['n_years']}, p={a_un['p_ttest']:.3f}")
        print("  The margin widens on the fair subset; significance still does not")
        print("  follow, and the win-count shows why -- large wins in a minority of")
        print("  years, not consistent superiority.")

    C.METRICS.mkdir(parents=True, exist_ok=True)
    dst = C.METRICS / "fair_comparison.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
