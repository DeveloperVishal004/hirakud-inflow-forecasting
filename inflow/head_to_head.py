"""One table: both pipelines, identical windows, identical metrics.

Every earlier comparison in this project was wrong in the same way -- the two
sides were scored on different rows, different lead ranges, or per-member versus
ensemble-mean, and the difference was read as a modelling result.  Twice.  So
this script derives ONE window set and scores everything on it:

    309 initialisations x 30 leads, 19 monsoons, ensemble mean, same target,
    same NSE, windows where all 30 forecast days are observed AND the 60 days
    of antecedent inflow the LSTM needs exist (inflow/lstm_postproc.py PAST=60;
    this is the binding constraint that takes 532 initialisations down to 309).

Models compared
    persistence           inflow observed on the init date, held flat
    0.86 x EC             one scalar on raw ECMWF, fitted leave-one-year-out
    EC -> VIC -> RVIC     the Dong chain on raw ECMWF
    EC-CNN -> VIC -> RVIC the Dong chain on downscaled rainfall
    EC -> VIC -> RVIC -> LSTM      the error-correction LSTM on the raw chain
    EC-CNN -> VIC -> RVIC -> LSTM  Dong et al.'s proposed system, VIC for XAJ
    FutureTST + FT        Ambika, fine-tuned on forecast forcing
    FutureTST + QM + FT   Ambika, quantile-mapped and fine-tuned

The probabilistic rows carry CRPS, 80/90 % coverage and interval width; the
deterministic ones cannot -- they emit a point, and that asymmetry is itself a
result, not a gap in the table.

Run:  python inflow/head_to_head.py
Out:  results/metrics/head_to_head.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C
from inflow.evaluate_operational import load_windows, nse, PRODUCTS

BANDS = [(1, 3), (4, 7), (8, 14), (15, 21), (22, 30)]
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def main():
    obs_df = pd.read_parquet(C.PROCESSED / "inflow_daily_extended.parquet")
    obs = obs_df[obs_df["inflow_valid"]].set_index("date")["inflow"]
    _, keep, per_prod, rowmap = load_windows()
    inits = sorted(keep)
    years = np.array([i.year for i in inits])

    Y, det = None, {}
    for prod in PRODUCTS:
        q, ok = per_prod[prod]
        sel = sorted([(i, ki, y) for i, ki, y in ok if i in keep], key=lambda t: t[0])
        det[prod] = np.array([q[ki].mean(0) for _, ki, _ in sel])
        if Y is None:
            Y = np.array([y for _, _, y in sel])

    # LSTM-corrected VIC, ensemble mean, same windows.  BOTH arms: on raw EC and
    # on the downscaled chain.  The second is Dong et al.'s actual proposed
    # system -- coupled EC-CNN + XAJ-LSTM, with VIC+RVIC standing in for XAJ --
    # and until 2026-09-06 it was computed but never tabulated here, so no
    # conclusion in this project was ever drawn from the paper's headline chain.
    for tag, prod in (("ec_lstm", "ec"), ("ec_cnn_lstm", "ec_cnn")):
        f = C.PROCESSED / f"lstm_inflow_{prod}.npz"
        if not f.exists():
            print(f"  [skip] {tag}: {f.name} missing -- run "
                  f"inflow/lstm_postproc.py --product {prod}")
            continue
        lz = np.load(f, allow_pickle=True)
        if "init" in lz.files:
            # Match by DATE.  Row position depends on the LSTM's n_seq, so
            # positional indexing misaligns silently whenever it is retuned.
            ri = pd.to_datetime([str(x) for x in lz["init"]])
            by_date = {}
            for r, d in enumerate(ri):
                by_date.setdefault(d, []).append(r)
            missing = [i for i in inits if i not in by_date]
            if missing:
                sys.exit(f"{tag}: {len(missing)} of {len(inits)} windows have no "
                         f"{prod} row -- refusing to tabulate a partial column")
            det[tag] = np.array([lz["pred"][by_date[i]].mean(0) for i in inits])
            if "n_seq" in lz.files:
                print(f"  {tag}: n_seq = {int(lz['n_seq'])} d, "
                      f"{len(set(ri))} initialisations trained, "
                      f"{len(inits)} scored")
        else:
            idx = {i: ix for i, ix in rowmap[prod]}
            missing = [i for i in inits if i not in idx]
            if missing:
                sys.exit(f"{tag}: {len(missing)} of {len(inits)} windows have no "
                         f"{prod} row -- refusing to tabulate a partial column")
            det[tag] = np.array([lz["pred"][idx[i]].mean(0) for i in inits])

    # persistence: the inflow observed on the initialisation date, held flat
    det["persistence"] = np.repeat(
        np.array([obs.loc[i] for i in inits])[:, None], Y.shape[1], 1)

    # scalar shrinkage on raw EC, fitted leave-one-year-out (no in-sample edge)
    grid = np.arange(0.30, 1.51, 0.01)
    shrunk = np.zeros_like(det["ec"])
    for y_ in sorted(set(years)):
        tr, te = years != y_, years == y_
        o, p = Y[tr].ravel(), det["ec"][tr].ravel()
        shrunk[te] = grid[np.argmax([nse(g * p, o) for g in grid])] * det["ec"][te]
    det["ec_shrunk"] = shrunk

    # FutureTST: pivot the quantile columns onto the same (init, lead) grid
    ft = pd.read_parquet(C.PROCESSED / "futuretst_prob_predictions.parquet")
    ft["init_date"] = pd.to_datetime(ft["init_date"])
    prob = {}
    for tag, col in [("ftst_ft", "p_ft"), ("ftst_qft", "p_qft")]:
        qs = {}
        for prob_lvl in QUANTILES:
            c = f"{col}_q{int(prob_lvl * 100):02d}"
            piv = ft.pivot_table(index="init_date", columns="lead_day", values=c)
            piv = piv.reindex(index=pd.DatetimeIndex(inits),
                              columns=range(1, Y.shape[1] + 1))
            qs[prob_lvl] = piv.to_numpy(np.float32)
        prob[tag] = qs
        det[tag] = qs[0.50]

    LABEL = [("persistence", "persistence"), ("ec_shrunk", "0.86 x EC (LOYO)"),
             ("ec", "EC -> VIC -> RVIC"), ("ec_cnn", "EC-CNN -> VIC -> RVIC"),
             ("ec_lstm", "EC -> VIC -> RVIC -> LSTM"),
             ("ec_cnn_lstm", "EC-CNN -> VIC -> RVIC -> LSTM"),
             ("ftst_ft", "FutureTST + FT"), ("ftst_qft", "FutureTST + QM + FT")]
    LABEL = [(t, l) for t, l in LABEL if t in det]

    def metrics(S, sl=slice(None), tag=None):
        y_, s_ = Y[:, sl], S[:, sl]
        ok = np.isfinite(s_).all(1)
        y_, s_, yy = y_[ok], s_[ok], years[ok]
        per_year = [nse(s_[yy == v].ravel(), y_[yy == v].ravel())
                    for v in sorted(set(yy))]
        # Pooled correlation mixes two abilities: knowing WHICH windows are wet
        # (between-window, largely the seasonal cycle) and getting the day-to-day
        # shape right within a window (the actual forecast skill the phase
        # argument rests on).  Report all three -- pooled alone overstates skill.
        sa, oa = s_ - s_.mean(1, keepdims=True), y_ - y_.mean(1, keepdims=True)
        corr_within = (float(np.corrcoef(sa.ravel(), oa.ravel())[0, 1])
                       if sa.std() > 0 and oa.std() > 0 else float("nan"))
        corr_between = (float(np.corrcoef(s_.mean(1), y_.mean(1))[0, 1])
                        if len(s_) > 1 else float("nan"))

        # A FLAT forecast (persistence) has argmax == 0 in every window, so the
        # "timing error" below collapses to the mean position of the OBSERVED
        # peak -- a property of the observations, not of the forecast.  Report it
        # only where the forecast actually has a peak to place.
        varies = s_.max(1) > s_.min(1)
        pk_t = (float(np.abs(s_.argmax(1)[varies] - y_.argmax(1)[varies]).mean())
                if varies.any() else float("nan"))

        m = {"NSE_median": float(np.median(per_year)),
             "NSE_mean": float(np.mean(per_year)),
             "RMSE": float(np.sqrt(((s_ - y_) ** 2).mean())),
             "corr": float(np.corrcoef(s_.ravel(), y_.ravel())[0, 1]),
             "corr_within_window": corr_within,
             "corr_between_window": corr_between,
             "bias_pct": float(100 * (s_.mean() - y_.mean()) / y_.mean()),
             "peak_ratio": float(np.median(s_.max(1) / y_.max(1))),
             "peak_timing_err_days": pk_t,
             "peak_timing_n_windows_with_a_peak": int(varies.sum()),
             "peak_timing_flat_forecast": bool(not varies.all()),
             "per_year": {int(v): float(x) for v, x in zip(sorted(set(yy)), per_year)}}
        if tag in prob:                                   # probabilistic extras
            qs = {k: v[:, sl][ok] for k, v in prob[tag].items()}
            tot = np.zeros_like(y_)
            for lvl in QUANTILES:
                e = y_ - qs[lvl]
                tot += np.maximum(lvl * e, (lvl - 1.0) * e)
            m["CRPS"] = float(2 * tot.mean() / len(QUANTILES))
            for name, (lo, hi) in [("80", (0.10, 0.90)), ("90", (0.05, 0.95))]:
                a, b = qs[lo], qs[hi]
                m[f"coverage_{name}"] = float(((y_ >= a) & (y_ <= b)).mean())
                m[f"width_{name}"] = float((b - a).mean())
        return m

    out = {"n_windows": len(inits), "n_years": len(set(years)),
           "horizon": int(Y.shape[1]), "models": {}, "lead_bands": {}}
    print(f"{len(inits)} windows x {Y.shape[1]} leads, {len(set(years))} monsoons, "
          f"ensemble mean\n")
    hdr = (f"  {'model':<24}{'NSE med':>9}{'RMSE':>8}{'corr':>7}{'bias%':>8}"
           f"{'peak':>7}{'|dt|':>7}{'CRPS':>8}{'cov80':>7}{'cov90':>7}{'width80':>9}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for tag, label in LABEL:
        m = metrics(det[tag], tag=tag)
        out["models"][label] = m
        ex = (f"{m['CRPS']:>8.0f}{m['coverage_80']:>7.2f}{m['coverage_90']:>7.2f}"
              f"{m['width_80']:>9.0f}") if "CRPS" in m else f"{'-':>8}{'-':>7}{'-':>7}{'-':>9}"
        print(f"  {label:<24}{m['NSE_median']:>9.3f}{m['RMSE']:>8.0f}{m['corr']:>7.3f}"
              f"{m['bias_pct']:>8.1f}{m['peak_ratio']:>7.2f}"
              f"{m['peak_timing_err_days']:>7.2f}{ex}")

    print("\n  median NSE by lead band:")
    print("    " + "band".ljust(9) + "".join(l[:13].rjust(15) for _, l in LABEL))
    for lo, hi in BANDS:
        sl = slice(lo - 1, hi)
        row, band = "", {}
        for tag, label in LABEL:
            m = metrics(det[tag], sl, tag=tag)
            band[label] = m
            row += f"{m['NSE_median']:>15.3f}"
        out["lead_bands"][f"{lo}-{hi}"] = band
        print(f"    {str(lo) + '-' + str(hi) + ' d':<9}{row}")

    print("\n  FutureTST + QM + FT, calibration by band:")
    for lo, hi in BANDS:
        b = out["lead_bands"][f"{lo}-{hi}"]["FutureTST + QM + FT"]
        print(f"    {str(lo) + '-' + str(hi) + ' d':<9} CRPS {b['CRPS']:6.0f}   "
              f"cov80 {b['coverage_80']:.2f}   cov90 {b['coverage_90']:.2f}   "
              f"width80 {b['width_80']:6.0f}   peak {b['peak_ratio']:.2f}")

    # is the best FutureTST significantly better than the best VIC-chain model?
    print("\n  paired per-year tests (n=19):")
    ref = "0.86 x EC (LOYO)"
    for tag, label in LABEL:
        if label == ref:
            continue
        a = np.array([out["models"][label]["per_year"][y_] for y_ in sorted(set(years))])
        b = np.array([out["models"][ref]["per_year"][y_] for y_ in sorted(set(years))])
        t = st.ttest_rel(a, b)
        out.setdefault("significance_vs_baseline", {})[label] = {
            "baseline": ref, "t": float(t.statistic), "p": float(t.pvalue),
            "wins": int((a > b).sum()), "n_years": int(len(a)),
            "note": ("Paired t-test on per-year NSE. See inflow/fair_comparison.py "
                     "for Wilcoxon, Holm correction, and the split by VIC "
                     "calibration exposure -- VIC was fitted on 2004-2011, which "
                     "is 42 % of these years.")}
        print(f"    {label:<24} vs {ref}: mean {a.mean() - b.mean():+.3f}  "
              f"t {t.statistic:+5.2f}  p {t.pvalue:.3f}  wins {int((a > b).sum())}/19")

    C.METRICS.mkdir(parents=True, exist_ok=True)
    p = C.METRICS / "head_to_head.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
