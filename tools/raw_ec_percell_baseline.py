#!/usr/bin/env python3
"""The raw-ECMWF per-cell R2 baseline that the CNN's +0.057 is compared against.

WHY THIS EXISTS.  The figure "raw ECMWF per-cell R2 = -0.136" was quoted in the
documentation for a long time with no script behind it and no entry in
results/metrics/.  It traces only to archive/historical_docs/, and it is wrong
for the comparison it was used in: it is a PER-MEMBER score, while the CNN's
+0.057 is an ENSEMBLE-MEAN score (loyo_percell_v2.py averages over members
before scoring: `out[b] = acc / n_mem`).  Mixing those two regimes is a mistake
this project has now made three times; docs/FINDINGS.md records the first two.

Scored on exactly the CNN's basis: the same 224 catchment cells, the same
observation mask, the same 19 leave-one-year-out test folds, ensemble mean.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnn.loyo_percell_v2 import patch_index          # noqa: E402
from cnn.train_field import r2                        # noqa: E402

ARCHIVE = Path("data/processed/percell_v2_full.npz")
OUT = Path("results/metrics/raw_ec_percell_baseline.json")


def main():
    d = np.load(ARCHIVE, allow_pickle=True)
    ch = [str(x) for x in d["channels"]]
    j = ch.index("tp")
    cells = np.argwhere(d["catchment"])
    rows, cols = patch_index(cells, d["coarse"].shape[3], d["coarse"].shape[4])
    # rows/cols are the 3x3 patch; column 1 is the centre, i.e. the coarse cell
    # the fine cell actually sits in.
    ci, cj = rows[:, 1], cols[:, 1]

    mu = float(d["mu"][0, 0, j, 0, 0])
    sd = float(d["sd"][0, 0, j, 0, 0])
    cs = d["coarse"][:, :, j].astype(np.float32)
    y = d["target"][:, cells[:, 0], cells[:, 1]]
    m = d["mask"][:, cells[:, 0], cells[:, 1]]
    iy = d["init_year"]
    years = sorted(set(iy.tolist()))

    def per_fold(pred):
        return np.array([r2(pred[iy == Y][m[iy == Y]], y[iy == Y][m[iy == Y]])
                         for Y in years])

    variants = {}

    # (1) the like-for-like baseline: ensemble mean, nearest coarse cell
    nearest = np.clip(cs[:, :, ci, cj].mean(1) * sd + mu, 0, None)
    variants["ensemble_mean_nearest_coarse_cell"] = per_fold(nearest)

    # (2) 3x3 coarse mean -- a crude smoothing, i.e. what interpolation buys
    p3 = cs[:, :, rows[:, :, None], cols[:, None, :]].mean(axis=(1, 3, 4))
    variants["ensemble_mean_3x3_smoothed"] = per_fold(np.clip(p3 * sd + mu, 0, None))

    # (3) the per-member regime -- the one the -0.136 figure belongs to
    per_mem = []
    for Y in years:
        k = iy == Y
        sel = m[k]
        per_mem.append(np.mean([
            r2(np.clip(cs[k][:, mi][:, ci, cj] * sd + mu, 0, None)[sel], y[k][sel])
            for mi in range(cs.shape[1])]))
    variants["per_member_then_averaged"] = np.array(per_mem)

    # (4) sanity: a constant must score ~0 by construction
    variants["constant_mean_sanity"] = per_fold(np.full_like(nearest, y[m].mean()))

    out = {
        "source": str(ARCHIVE),
        "n_cells": int(len(cells)),
        "n_keys": int(len(y)),
        "n_folds": len(years),
        "scoring": "ensemble mean over 10 perturbed members, masked, per test year",
        "variants": {},
    }
    for name, sc in variants.items():
        out["variants"][name] = {
            "mean": float(sc.mean()),
            "median": float(np.median(sc)),
            "folds_positive": int((sc > 0).sum()),
            "per_year": {str(Y): float(v) for Y, v in zip(years, sc)},
        }
        print(f"  {name:36s} mean {sc.mean():+.4f}  median {np.median(sc):+.4f}  "
              f"pos {(sc > 0).sum()}/{len(years)}")

    out["note"] = (
        "The headline comparison is ensemble_mean_nearest_coarse_cell "
        f"({variants['ensemble_mean_nearest_coarse_cell'].mean():+.4f}) against the "
        "CNN's +0.0571 from loyo_percell_v2.json. The long-quoted -0.136 is not "
        "reproducible here in any ensemble-mean variant; it sits in the range of "
        "the per-member regime, which is not the regime the CNN is scored in.")

    # ---- what the aggregate number hides: heavy rain, and lead time ------
    oof_path = Path("data/processed/percell_v2_oof.npz")
    if oof_path.exists():
        o = np.load(oof_path, allow_pickle=True)
        cnn, lead = o["oof"], o["lead"]
        p3f = np.clip(p3 * sd + mu, 0, None)
        preds = {"raw_ec": nearest, "smoothed_3x3": p3f, "cnn": cnn}

        thr = float(np.percentile(y[m], 90))
        heavy = m & (y > thr)
        out["heavy_rain"] = {
            "threshold_mm": thr, "n": int(heavy.sum()),
            "observed_mean_mm": float(y[heavy].mean()),
            "models": {k: {
                "R2_within_heavy": r2(v[heavy], y[heavy]),
                "bias_mm": float((v[heavy] - y[heavy]).mean()),
                "captured_pct_of_observed": float(100 * v[heavy].mean() / y[heavy].mean()),
            } for k, v in preds.items()},
            "note": ("R2 is computed within the heavy subset, so it is negative for "
                     "every model by construction; the comparison between models and "
                     "the capture percentage are the meaningful parts."),
        }

        out["by_lead_band"] = {}
        for lo, hi in [(1, 3), (4, 7), (8, 14), (15, 21), (22, 30)]:
            k = m & ((lead >= lo) & (lead <= hi))[:, None]
            out["by_lead_band"][f"{lo}-{hi}"] = {
                kk: r2(v[k], y[k]) for kk, v in preds.items()}

        out["headline"] = (
            "The CNN's aggregate gain over raw ECMWF comes entirely from leads 8-30. "
            "At leads 1-7 -- the only leads with usable skill -- it is WORSE than raw "
            "ECMWF, and on heavy rain it is the worst of the three.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(OUT)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
