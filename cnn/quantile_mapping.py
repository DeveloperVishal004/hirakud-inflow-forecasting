"""EC and EC-QM at the catchment cells -- the paper's two precipitation benchmarks.

Dong et al. (2025) sect. 3.2.2, followed as written:

    EC      raw ECMWF; each 0.25 deg cell takes its nearest 1.5 deg cell, the
            "corresponding forecast grid cell" the paper maps from
    EC-QM   non-parametric quantile mapping, one empirical CDF pair per fine
            cell AND per lead time, with dry days (< 0.1 mm) excluded from both
            CDFs.  Forecasts below 0.1 mm stay dry (0 mm); wet forecasts are
            mapped from the wet-day forecast CDF onto the wet-day observed CDF.
            Beyond the calibration range the map holds the extreme value.

The CDFs pool the 10 perturbed members.  They are fitted on the calibration
years and applied to the test years only -- never fitted on the year being
scored.  --split fixed (default) calibrates on 2004-2017 and tests 2018-2022,
the same test years as the CNN; --split loyo refits per monsoon.

Run:  python cnn/quantile_mapping.py [--split fixed]
Out:  data/processed/ec_qm_v2_<split>.npz   (ec_qm is NaN outside test rows)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C
from cnn.loyo_percell_v2 import load_members, patch_index, split_folds, DSET


def wet_quantile_map(f_cal, o_cal, f_new, wet=C.QM_WET_DAY_MM, min_wet=30):
    """Map f_new through wet-day CDFs of f_cal (forecast) and o_cal (observed).

    Returns None when either calibration sample has fewer than `min_wet` wet days.
    """
    f_wet = np.sort(f_cal[f_cal >= wet])
    o_wet = np.sort(o_cal[o_cal >= wet])
    if len(f_wet) < min_wet or len(o_wet) < min_wet:
        return None
    pct = np.searchsorted(f_wet, f_new, "right") / len(f_wet)
    mapped = np.quantile(o_wet, np.clip(pct, 0.0, 1.0))
    return np.where(f_new < wet, 0.0, mapped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["fixed", "loyo"], default="fixed")
    args = ap.parse_args()

    coarse, inits, leads, names = load_members(10)      # (n_key, 10, 26, 7, 7)
    d = np.load(DSET, allow_pickle=True)

    # load_members walks sorted(glob) -- month/day filename outer, year inner --
    # while the dataset is ordered year first.  Join on (init, lead) and fail
    # loudly rather than pair each forecast with another row's observation.
    want = list(zip(d["init"].astype("datetime64[D]").astype(str).tolist(),
                    d["lead"].tolist()))
    have = {k: i for i, k in enumerate(
        zip(inits.values.astype("datetime64[D]").astype(str).tolist(),
            leads.tolist()))}
    missing = [k for k in want if k not in have]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(want)} (init, lead) keys are "
                         f"in {DSET.name} but not in the member archive")
    order = np.array([have[k] for k in want], np.int64)
    tp = coarse[order][:, :, names.index("tp")]         # (n_key, 10, 7, 7) mm
    del coarse
    print(f"  aligned to {DSET.name} on (init, lead): {len(order)} rows")

    # Every label below comes from `d`, the source the data was aligned to.
    # Saving load_members' own inits/leads here is what scrambled the forecast
    # dates in the old ec_qm_v2_full.npz (see docs/FINDINGS.md).
    target, mask, catchment = d["target"], d["mask"], d["catchment"]
    init_year, lead = d["init_year"], d["lead"]
    cells = np.argwhere(catchment)
    rows, cols = patch_index(cells)
    ec = tp[:, :, rows[:, 1], cols[:, 1]].astype(np.float32)   # (n_key, 10, 224)
    obs = target[:, cells[:, 0], cells[:, 1]]
    obs_ok = mask[:, cells[:, 0], cells[:, 1]]
    print(f"EC raw at cells (nearest coarse cell): {ec.shape}")

    qm = np.full_like(ec, np.nan)
    tested = np.zeros(len(ec), bool)
    n_fallback = 0
    for name, test_k, _ in split_folds(init_year, args.split):
        cal_k = ~test_k                    # QM has no early stopping to feed
        for L in range(C.V2_LEAD_MIN, C.V2_LEAD_MAX + 1):
            cal = cal_k & (lead == L)
            te = np.where(test_k & (lead == L))[0]
            for j in range(len(cells)):
                o_cal = obs[cal, j][obs_ok[cal, j]]
                mapped = wet_quantile_map(ec[cal, :, j].ravel(), o_cal,
                                          ec[te, :, j].ravel())
                if mapped is None:
                    n_fallback += 1
                    mapped = ec[te, :, j].ravel()
                qm[te, :, j] = mapped.reshape(len(te), -1)
        tested |= test_k
        print(f"  [{name}] calibrated on {sorted(set(init_year[cal_k].tolist()))[0]}-"
              f"{sorted(set(init_year[cal_k].tolist()))[-1]}, "
              f"mapped {int(test_k.sum())} keys", flush=True)
    if n_fallback:
        print(f"  {n_fallback} (fold, lead, cell) maps had < 30 wet days -> raw EC")

    out = C.PROCESSED / f"ec_qm_v2_{args.split}.npz"
    np.savez_compressed(out, ec=ec, ec_qm=qm, tested=tested, obs=obs,
                        obs_mask=obs_ok, cells=cells, lead=lead,
                        init_year=init_year, init=d["init"], valid=d["valid"])
    print(f"\nwrote {out}")
    m = obs_ok & tested[:, None]
    for nm, arr in (("EC", ec.mean(1)), ("EC-QM", qm.mean(1))):
        r = np.sqrt(((arr[m] - obs[m]) ** 2).mean())
        print(f"  {nm:6} test per-cell ensemble-mean RMSE {r:6.3f} mm/day   "
              f"bias {arr[m].mean() - obs[m].mean():+.2f}")


if __name__ == "__main__":
    main()
