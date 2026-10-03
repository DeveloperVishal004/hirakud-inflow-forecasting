"""The Dong et al. (2025) precipitation metrics, sect. 3.5 and Fig. 4.

The paper scores the AREAL-AVERAGED precipitation -- the basin mean, which is
also what a lumped hydrological model is driven with:

    RMSE, RE   of the ensemble-mean daily basin rainfall, over all leads and
               for each lead window (1-7, 8-15, 16-23, 24-30 d)
    CRPS       of the daily basin rainfall over the 10 members, so the spread
               is scored and not just the mean
    heavy      5-day basin totals at or above the 90th percentile of all
               observed 5-day totals are heavy events, the rest light; RMSE is
               reported for each.  5-day totals label events -- they are not the
               headline daily RMSE.

The basin mean of each row uses only the cells IMD observed on that day, for
the forecast and the observation alike.  Only test rows are scored.

Run:  python cnn/paper_metrics.py [--split fixed] [--tag _x]
Out:  results/metrics/paper_metrics_<split><tag>.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

PENTAD = 5


def crps_ensemble(ens, obs):
    """CRPS by the energy form: E|X-y| - 0.5 E|X-X'|, exact for a finite sample.

    ens (n, m), obs (n,).  Averaged over samples.
    """
    n, m = ens.shape
    term1 = np.abs(ens - obs[:, None]).mean(1)
    s = np.sort(ens, 1)
    # E|X - X'| for a sorted sample, without forming the m x m matrix
    w = (2 * np.arange(1, m + 1) - m - 1)
    term2 = 2.0 * (s * w).sum(1) / (m * m)
    return float((term1 - 0.5 * term2).mean())


def basin_mean(x, ok):
    """(n_key, [member,] cell) -> (n_key, [member]) mean over observed cells."""
    if x.ndim == 3:
        ok = ok[:, None, :]
    w = ok.astype(np.float64)
    n = w.sum(-1)
    with np.errstate(invalid="ignore"):
        return np.where(n > 0, (np.nan_to_num(x) * w).sum(-1) / n, np.nan)


def scores(f, o):
    """RMSE and RE (%) of forecast f against observed o, NaNs dropped."""
    ok = np.isfinite(f) & np.isfinite(o)
    f, o = f[ok], o[ok]
    return {"RMSE": float(np.sqrt(((f - o) ** 2).mean())),
            "RE_pct": float(100 * (f.sum() - o.sum()) / o.sum()), "n": int(ok.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["fixed", "loyo"], default="fixed")
    ap.add_argument("--tag", default="", help="suffix of the CNN run to score")
    args = ap.parse_args()

    base = np.load(C.PROCESSED / f"ec_qm_v2_{args.split}.npz", allow_pickle=True)
    lead, valid, obs, ok = base["lead"], base["valid"], base["obs"], base["obs_mask"]
    test = base["tested"].copy()
    prods = {"EC": base["ec"], "EC-QM": base["ec_qm"]}
    cnn_path = C.PROCESSED / f"percell_v2_{args.split}{args.tag}.npz"
    if cnn_path.exists():
        cnn = np.load(cnn_path, allow_pickle=True)
        for k in ("lead", "init_year", "valid"):
            if not np.array_equal(cnn[k], base[k]):
                raise SystemExit(f"{cnn_path.name} and the baseline disagree on "
                                 f"`{k}` -- they are not on the same rows")
        if not np.array_equal(cnn["cells"], base["cells"]):
            raise SystemExit("CNN and baseline use different catchment cells")
        test &= cnn["tested"]
        prods["EC-CNN"] = cnn["pred_members"]
        print(f"  scoring {cnn_path.name} against EC and EC-QM")
    else:
        print(f"  {cnn_path.name} not found -- scoring EC and EC-QM only")

    # Arrange rows as (init, lead) so 5-day totals can be summed.
    inits = pd.to_datetime(valid) - pd.to_timedelta(lead, unit="D")
    L = C.V2_LEAD_MAX
    order = np.lexsort((lead, inits.values.astype("int64")))
    n_init = len(order) // L
    assert n_init * L == len(order) and (lead[order].reshape(n_init, L)
                                         == np.arange(1, L + 1)).all(), \
        "every initialisation must carry leads 1-30 exactly once"
    def cube(a):
        return a[order].reshape(n_init, L, *a.shape[1:])

    obs_b = cube(basin_mean(obs, ok))                     # (n_init, L)
    test_c = cube(test)[:, 0]
    assert (cube(test) == test_c[:, None]).all(), "test split cuts through an init"

    # Heavy threshold: p90 of all observed 5-day basin totals, every year, as
    # the paper does ("all historic 5 d precipitation during 2002-2019").
    o5_all = obs_b.reshape(n_init, L // PENTAD, PENTAD).sum(-1)
    thr5 = float(np.nanpercentile(o5_all, C.HEAVY_RAIN_PERCENTILE))
    o5 = o5_all[test_c]
    heavy = o5 >= thr5

    win_idx = [(lead_lo, lead_hi) for lead_lo, lead_hi in C.DONG_LEAD_WINDOWS]
    out = {"split": args.split, "n_test_inits": int(test_c.sum()),
           "test_years": sorted(set(base["init_year"][test].tolist())),
           "heavy_threshold_mm_per_5d": thr5, "products": {}}
    for name, arr in prods.items():
        ens = cube(basin_mean(arr, ok))[test_c]           # (n, L, member)
        mean = ens.mean(-1)
        o = obs_b[test_c]
        r = {"daily_all": scores(mean, o)}
        good = np.isfinite(o) & np.isfinite(mean)
        r["daily_all"]["CRPS"] = crps_ensemble(ens[good], o[good])
        for lo, hi in win_idx:
            s = slice(lo - 1, hi)
            r[f"daily_{lo}-{hi}"] = scores(mean[:, s], o[:, s])
            g = good[:, s]
            r[f"daily_{lo}-{hi}"]["CRPS"] = crps_ensemble(ens[:, s][g], o[:, s][g])
        f5 = mean.reshape(len(mean), L // PENTAD, PENTAD).sum(-1)
        r["5d_all"] = scores(f5, o5)
        r["5d_heavy"] = scores(f5[heavy], o5[heavy])
        r["5d_light"] = scores(f5[~heavy], o5[~heavy])
        out["products"][name] = r

    ec = out["products"]["EC"]
    for name, r in out["products"].items():
        if name != "EC":
            r["RMSE_reduction_vs_EC_pct"] = {
                k: float(100 * (1 - r[k]["RMSE"] / ec[k]["RMSE"])) for k in r
                if isinstance(r[k], dict) and "RMSE" in r[k]}

    C.METRICS.mkdir(parents=True, exist_ok=True)
    f = C.METRICS / f"paper_metrics_{args.split}{args.tag}.json"
    f.write_text(json.dumps(out, indent=2))

    print(f"\n=== basin-mean precipitation, test years {out['test_years']} "
          f"({out['n_test_inits']} inits) ===")
    cols = ["daily_all"] + [f"daily_{lo}-{hi}" for lo, hi in win_idx] + ["5d_heavy"]
    print(f"{'RMSE (mm)':10}" + "".join(f"{c.replace('daily_', 'd '):>11}" for c in cols)
          + f"{'RE %':>8}{'CRPS':>7}")
    for name, r in out["products"].items():
        print(f"{name:10}" + "".join(f"{r[c]['RMSE']:11.2f}" for c in cols)
              + f"{r['daily_all']['RE_pct']:8.1f}{r['daily_all']['CRPS']:7.2f}")
    print(f"(heavy = 5-day basin total >= {thr5:.1f} mm; 5d_heavy RMSE is per 5 days)")
    print(f"\nwrote {f}")


if __name__ == "__main__":
    main()
