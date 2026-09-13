"""The Dong et al. (2025) metric suite, sect. 3.5, for both halves of the chain.

PRECIPITATION, evaluated on 5-DAY totals, not daily.  The paper is explicit:
"we classify the 5 d daily precipitation less than and greater than the 90th
percentile of all historic 5 d precipitation ... as light rain events and heavy
rain events".  Aggregating to pentads is not cosmetic -- it removes day-to-day
timing error and is a large part of why their reported skill exceeds anything
computed daily.  Scoring daily against their pentad numbers understates our
result; this makes the comparison honest in both directions.

    RMSE    all events, and heavy events separately (>= 90th percentile)
    RE      relative error of the mean
    CRPS    over the 10 ensemble members, so spread is scored, not just the mean

STREAMFLOW
    RMSE, RE, NSE, and REF -- relative error of the maximum daily flow, the
    paper's extreme-event metric.

Run:  python cnn/paper_metrics.py
Out:  results/metrics/paper_metrics.json
"""

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


def to_pentad(x, lead):
    """Sum consecutive 5-lead blocks: (..., 30) -> (..., 6)."""
    idx = [(lead >= lo) & (lead < lo + PENTAD) for lo in range(1, 31, PENTAD)]
    return np.stack([x[..., i].sum(-1) for i in idx], -1)


def precip_metrics():
    # Prefer the 19-monsoon baseline when it exists; the 11-monsoon file is kept
    # so the original paper table stays reproducible.  Both must be the same
    # length as percell_v2_oof.npz or the products are not on the same rows.
    ecqm = next((q for q in (C.PROCESSED / "ec_qm_v2_full.npz",
                             C.PROCESSED / "ec_qm_v2.npz") if q.exists()),
                C.PROCESSED / "ec_qm_v2.npz")
    z = np.load(ecqm, allow_pickle=True)
    print(f"  baseline: {ecqm.name}  ({len(z['lead']):,} rows)")
    lead = z["lead"]
    obs, ok = z["obs"], z["obs_mask"]
    prods = {"EC": z["ec"], "EC-QM": z["ec_qm"]}
    p = C.PROCESSED / "percell_v2_oof.npz"
    if p.exists():
        cnn = np.load(p, allow_pickle=True)
        if len(cnn["lead"]) != len(z["lead"]):
            raise SystemExit(
                f"{p.name} has {len(cnn['lead'])} rows but {ecqm.name} has "
                f"{len(z['lead'])} -- rebuild the baseline with "
                f"cnn/quantile_mapping.py before scoring")
        prods["EC-CNN"] = cnn["oof_members"]

    # reshape to (n_init, lead, member, cell) so pentads can be summed
    inits = pd.to_datetime(z["valid"]) - pd.to_timedelta(lead, unit="D")
    ui = pd.DatetimeIndex(sorted(set(inits)))
    order = np.lexsort((lead, inits.astype("int64")))
    L = C.V2_LEAD_MAX
    def cube(a):
        return a[order].reshape(len(ui), L, *a.shape[1:])
    o5 = cube(obs).transpose(0, 2, 1).reshape(-1, L)
    m5 = cube(ok).transpose(0, 2, 1).reshape(-1, L)
    leads = np.arange(1, L + 1)
    o_p = to_pentad(o5, leads)
    keep = to_pentad(m5.astype(int), leads) == PENTAD      # complete pentads only
    thr = np.percentile(o_p[keep], C.HEAVY_RAIN_PERCENTILE)

    out = {"heavy_threshold_mm_per_pentad": float(thr),
           "n_pentads": int(keep.sum())}
    for name, arr in prods.items():
        c = cube(arr)                                       # (n_init, L, mem, cell)
        mean5 = c.mean(2).transpose(0, 2, 1).reshape(-1, L)
        f_p = to_pentad(mean5, leads)
        hv = keep & (o_p >= thr)
        out[name] = {
            "RMSE_all": float(np.sqrt(((f_p[keep] - o_p[keep]) ** 2).mean())),
            "RMSE_heavy": float(np.sqrt(((f_p[hv] - o_p[hv]) ** 2).mean())),
            "RE_pct": float(100 * (f_p[keep].mean() - o_p[keep].mean()) / o_p[keep].mean()),
        }
        ens = c.transpose(0, 3, 1, 2).reshape(-1, L, c.shape[2])
        ens_p = to_pentad(ens.transpose(0, 2, 1), leads).transpose(0, 2, 1)
        out[name]["CRPS"] = crps_ensemble(
            ens_p.reshape(-1, ens_p.shape[-1])[keep.ravel()], o_p[keep])
    for name in prods:
        if name != "EC":
            r = out[name]["RMSE_all"] / out["EC"]["RMSE_all"]
            out[name]["RMSE_reduction_vs_EC_pct"] = float(100 * (1 - r))
    return out


def streamflow_metrics():
    obs = pd.read_parquet(C.PROCESSED / "inflow_daily.parquet")
    obs = obs[obs["inflow_valid"]].set_index("date")["inflow"]
    res = {}
    for prod in ("ec", "ec_qm", "ec_cnn"):
        f = C.PROCESSED / f"lstm_inflow_{prod}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        p, t, m = z["pred"], z["target"], z["mask"]
        vic = np.load(C.PROCESSED / f"vic_inflow_{prod}.npz", allow_pickle=True)["q"]
        vic = vic.reshape(-1, vic.shape[-1])[:len(p)]
        def block(pred):
            pm, tm = pred[m], t[m]
            return {"RMSE": float(np.sqrt(((pm - tm) ** 2).mean())),
                    "RE_pct": float(100 * (pm.mean() - tm.mean()) / tm.mean()),
                    "NSE": float(1 - ((pm - tm) ** 2).sum() / ((tm - tm.mean()) ** 2).sum()),
                    "REF_pct": float(100 * (pred.max(1).mean() - t.max(1).mean())
                                     / t.max(1).mean())}
        res[prod.upper().replace("_", "-")] = {"VIC": block(vic), "VIC-LSTM": block(p)}
    return res


def main():
    out = {"precipitation_pentad": precip_metrics(),
           "streamflow": streamflow_metrics()}
    C.METRICS.mkdir(parents=True, exist_ok=True)
    f = C.METRICS / "paper_metrics.json"
    f.write_text(json.dumps(out, indent=2))
    pm = out["precipitation_pentad"]
    print(f"=== precipitation, 5-day totals ({pm['n_pentads']:,} pentads, "
          f"heavy >= {pm['heavy_threshold_mm_per_pentad']:.1f} mm) ===")
    print(f"{'product':10}{'RMSE all':>10}{'RMSE heavy':>12}{'RE %':>8}{'CRPS':>9}{'vs EC':>9}")
    for k, v in pm.items():
        if isinstance(v, dict):
            print(f"{k:10}{v['RMSE_all']:10.2f}{v['RMSE_heavy']:12.2f}"
                  f"{v['RE_pct']:8.1f}{v['CRPS']:9.2f}"
                  f"{v.get('RMSE_reduction_vs_EC_pct', 0):8.1f}%")
    if out["streamflow"]:
        print(f"\n=== streamflow ===")
        print(f"{'product':10}{'model':10}{'RMSE':>10}{'RE %':>8}{'NSE':>9}{'REF %':>9}")
        for prod, d in out["streamflow"].items():
            for mdl, v in d.items():
                print(f"{prod:10}{mdl:10}{v['RMSE']:10.0f}{v['RE_pct']:8.1f}"
                      f"{v['NSE']:9.3f}{v['REF_pct']:9.1f}")
    print(f"\nwrote {f}")


if __name__ == "__main__":
    main()
