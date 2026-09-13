"""Concatenate the 2004-2014 and 2015-2022 archives into one 19-monsoon dataset.

The two ranges were built separately so neither could corrupt the other, and
because their GRIBs carry different hindcast-year axes.  Everything downstream
wants a single array, so this joins them once, in initialisation order.

    downscaling_v2.npz            9,240 rows   2004-2014
    downscaling_v2_2015-2022.npz  6,720 rows   2015-2022
    -> downscaling_v2_full.npz   15,960 rows   19 monsoons

The per-cell packed file is rebuilt from the same source so the CNN and the
field model see identical samples.  Fold years are NOT hard-coded anywhere
downstream after this: they are read from `init_year`, so adding a year range
never requires editing a constant.

Run:  python preprocessing/merge_v2_ranges.py
Out:  data/processed/downscaling_v2_full.npz
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

A = C.PROCESSED / "downscaling_v2.npz"
B = C.PROCESSED / "downscaling_v2_2015-2022.npz"
OUT = C.PROCESSED / "downscaling_v2_full.npz"

# Arrays that vary per sample get concatenated; the rest must be identical in
# both files and are carried through unchanged.
PER_SAMPLE = {"coarse", "init", "lead", "member", "valid", "target", "mask", "init_year"}


def main():
    a = np.load(A, allow_pickle=True)
    b = np.load(B, allow_pickle=True)
    out = {}
    for k in a.files:
        if k in PER_SAMPLE:
            out[k] = np.concatenate([a[k], b[k]], axis=0)
        else:
            if k in b.files and a[k].shape == b[k].shape and not np.array_equal(a[k], b[k]):
                sys.exit(f"static array '{k}' differs between the two ranges -- refusing "
                         f"to merge, since downstream code assumes one grid and one "
                         f"channel order")
            out[k] = a[k]

    order = np.argsort(out["init"].astype("datetime64[ns]"), kind="stable")
    for k in PER_SAMPLE:
        if k in out:
            out[k] = out[k][order]

    np.savez_compressed(OUT, **out)
    yrs = sorted(set(int(y) for y in out["init_year"]))
    print(f"  {OUT.name}: {out['coarse'].shape[0]:,} rows  ({a['coarse'].shape[0]:,} + "
          f"{b['coarse'].shape[0]:,})")
    print(f"  years {yrs[0]}-{yrs[-1]}  ({len(yrs)} monsoons)")
    print(f"  coarse {out['coarse'].shape}   target {out['target'].shape}")
    print(f"  {OUT.stat().st_size/1e6:.0f} MB")
    # sanity: the two halves must agree on climatology, or something is misaligned
    m = out["mask"] & out["catchment"][None]
    t = out["target"]
    for lo, hi in ((2004, 2014), (2015, 2022)):
        s = (out["init_year"] >= lo) & (out["init_year"] <= hi)
        print(f"    {lo}-{hi}: IMD target mean {t[s][m[s]].mean():.2f} mm/d")


if __name__ == "__main__":
    main()
