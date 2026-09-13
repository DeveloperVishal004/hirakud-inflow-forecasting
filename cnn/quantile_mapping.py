"""EC and EC-QM at the catchment cells -- the paper's two precipitation benchmarks.

Dong et al. (2025) sect. 3.2.2 uses quantile mapping as the benchmark the CNN
must beat.  This produces both baselines on exactly the keys, cells, members and
folds the CNN was scored on, so the three products are directly comparable:

    EC      raw ECMWF, bilinearly interpolated from 1.5 deg to 0.25 deg
    EC-QM   EC with an empirical quantile map fitted per fine cell
    EC-CNN  cnn/loyo_percell_v2.py            (already computed)

LEAVE-ONE-YEAR-OUT, NOT A GLOBAL FIT.  The quantile map is built from the ten
training monsoons of each fold and applied to the held-out one.  Fitting it on
all eleven would let the test year's own distribution set the mapping -- the
same leak the split rules elsewhere in this project exist to prevent, and an
easy one to miss because QM feels like "just preprocessing".

Run:  python cnn/quantile_mapping.py
Out:  data/processed/ec_qm_v2.npz
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C
from cnn.loyo_percell_v2 import load_members, fine_grid, DSET

# Follow whichever archive loyo_percell_v2 selected (19 monsoons when the full
# build exists), so this baseline is always scored on the same rows as the CNN.
OUT = C.PROCESSED / ("ec_qm_v2_full.npz" if "full" in DSET.name
                     else "ec_qm_v2.npz")


def main():
    coarse, inits, leads, names = load_members(10)      # (n_key, 10, 26, 7, 7)
    d = np.load(DSET, allow_pickle=True)
    YEARS = sorted(set(int(y) for y in d["init_year"]))

    # SAME TRAP AS THE PACKER.  load_members walks sorted(glob) -- month/day
    # filename outer, year inner -- while the dataset is ordered year first.
    # The two coincided on the 2004-2014 build and diverge completely once the
    # 2015-2022 files join the glob, which would pair every forecast with
    # another row's observation.  Join on (init, lead) and fail loudly instead.
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
    coarse = coarse[order]
    print(f"  aligned to {DSET.name} on (init, lead): {len(order)} rows, "
          f"{int((order != np.arange(len(order))).sum())} reordered")
    target, mask, catchment = d["target"], d["mask"], d["catchment"]
    init_year = d["init_year"]
    hw = target.shape[1]
    cells = np.argwhere(catchment)

    # Raw forecast on the fine grid: bilinear from 1.5 deg, the same "no
    # downscaling" baseline the paper compares against.
    tp = coarse[:, :, names.index("tp")]                # (n_key, 10, 7, 7)
    n_key, n_mem = tp.shape[:2]
    flat = torch.from_numpy(tp.reshape(-1, 1, *tp.shape[2:]))
    fine = torch.nn.functional.interpolate(flat, size=(hw, hw), mode="bilinear",
                                           align_corners=False)
    fine = fine.reshape(n_key, n_mem, hw, hw).numpy().astype(np.float32)
    ec = fine[:, :, cells[:, 0], cells[:, 1]]           # (n_key, 10, 224)
    obs = target[:, cells[:, 0], cells[:, 1]]
    obs_ok = mask[:, cells[:, 0], cells[:, 1]]
    print(f"EC raw at cells: {ec.shape}")

    qm = np.empty_like(ec)
    for Y in YEARS:
        te = init_year == Y
        tr = ~te
        for j in range(len(cells)):
            f_tr = np.sort(ec[tr, :, j].ravel())
            o_tr = np.sort(obs[tr, j][obs_ok[tr, j]])
            if len(o_tr) < 30 or len(f_tr) < 30:
                qm[te, :, j] = ec[te, :, j]
                continue
            # percentile of each test forecast within the training forecast
            # distribution, then the matching percentile of training observations
            pct = np.searchsorted(f_tr, ec[te, :, j].ravel(), "left") / len(f_tr)
            qm[te, :, j] = np.quantile(o_tr, np.clip(pct, 0, 1)).reshape(-1, n_mem)
        print(f"  {Y}: mapped {int(te.sum())} keys", flush=True)

    # ROW-ORDER BUG, FIXED.  `coarse` was reordered into the dataset's order
    # above, and ec/ec_qm/obs/init_year all follow it -- but `leads` and `inits`
    # come from load_members() and were NEVER reordered.  Saving them raw wrote
    # a `valid` array in filename-outer order alongside data in dataset order,
    # so `valid - lead` gave the true initialisation on only 300 of 15,960 rows.
    # `lead` survived by luck (1..30 repeats identically under both orders),
    # which is why the obvious consistency check passed.
    #
    # Everything that keyed off `valid` was silently reading another forecast's
    # rainfall: inflow/lstm_postproc.py, inflow/futuretst_prob.py, and
    # data/processed/ec_ensemble_catchment.parquet (which is derived from this
    # file and inherited the fault exactly, r = 1.0000 against the broken
    # labelling).  Measured severity: 522 of 532 initialisations received the
    # wrong 30-day series, correlation +0.03 with the correct one, and
    # lead-1 forecast-vs-observed correlation collapsed from +0.68 to -0.02.
    #
    # Take the labels from `d`, which is the same source the data was aligned
    # to, rather than reordering load_members' copies -- one source of truth.
    assert len(d["init"]) == len(ec), "label/data length mismatch"
    np.savez_compressed(OUT, ec=ec, ec_qm=qm, obs=obs, obs_mask=obs_ok,
                        cells=cells, lead=d["lead"], init_year=init_year,
                        init=d["init"], valid=d["valid"])
    print(f"\nwrote {OUT}")
    m = obs_ok
    for nm, arr in (("EC", ec.mean(1)), ("EC-QM", qm.mean(1))):
        r = np.sqrt(((arr[m] - obs[m]) ** 2).mean())
        print(f"  {nm:6} ensemble-mean RMSE {r:6.3f} mm/day   bias {arr[m].mean()-obs[m].mean():+.2f}")


if __name__ == "__main__":
    main()
