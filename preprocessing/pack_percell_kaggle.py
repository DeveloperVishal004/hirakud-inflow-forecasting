"""Pack the per-cell training inputs into one upload-sized file for Kaggle.

cnn/loyo_percell_v2.py normally reads data/processed/s2s/*.npz -- 28 files,
385 MB, a directory the Kaggle payload builder does not carry.  This flattens
exactly what the per-cell run needs into a single file, three ways smaller:

  1. CROP TO THE PATCHES THAT EXIST.  Only 224 catchment cells are scored, and
     their 3x3 patches touch coarse rows 0-5 and columns 0-4 -- 30 of the 49
     cells.  Both ranges start at 0, so cropping needs no index remapping.
        471 MB -> 288 MB
  2. STANDARDISE ONCE, GLOBALLY.  Per-fold standardisation still happens in the
     trainer; composing two affine transforms is the same transform, so nothing
     leaks and nothing changes numerically.
  3. STORE float16.  Only safe AFTER standardising: msl is ~1e5 Pa and float16
     tops out at 65504, so raw fields would overflow silently.  Standardised
     fields are ~N(0, 1), where float16's ~1e-3 relative precision is far below
     the noise in a 30-day precipitation forecast.
        288 MB -> 144 MB

Run:  python preprocessing/pack_percell_kaggle.py
Out:  data/processed/percell_v2.npz
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C
from cnn.loyo_percell_v2 import load_members, DSET

OUT = C.PROCESSED / ("percell_v2_full.npz" if "full" in DSET.name
                     else "percell_v2.npz")
N_ROW, N_COL = 6, 5          # coarse subgrid the catchment patches need


def main():
    print("packing per-cell inputs for Kaggle")
    coarse, inits, leads, names = load_members(10)
    print(f"  loaded {coarse.shape}  {coarse.nbytes/1e6:.0f} MB")

    # ALIGN, NEVER ASSUME.  coarse comes from sorted(glob) over the member
    # archive -- month/day filename first, year on the inner axis -- while the
    # dataset below is ordered year first.  Those two orders coincided on the
    # 2004-2014 build and stopped coinciding the moment the 2015-2022 files
    # joined the glob: 0 of 15,960 rows lined up, so every field was packed
    # against another row's target and the CNN trained on noise.  Join on
    # (init, lead) explicitly and refuse to write anything that does not match.
    d = np.load(DSET, allow_pickle=True)
    want = list(zip(d["init"].astype("datetime64[D]").astype(str).tolist(),
                    d["lead"].tolist()))
    have = {k: i for i, k in enumerate(
        zip(inits.values.astype("datetime64[D]").astype(str).tolist(),
            leads.tolist()))}
    missing = [k for k in want if k not in have]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(want)} (init, lead) keys are in {DSET.name} "
            f"but not in the member archive; first: {missing[:3]}")
    order = np.array([have[k] for k in want], np.int64)
    n_moved = int((order != np.arange(len(order))).sum())
    coarse = coarse[order]
    print(f"  aligned to {DSET.name} on (init, lead): "
          f"{len(order)} rows, {n_moved} reordered")

    coarse = coarse[:, :, :, :N_ROW, :N_COL]
    print(f"  cropped to {coarse.shape}  {coarse.nbytes/1e6:.0f} MB")

    mu = coarse.mean(axis=(0, 1, 3, 4), keepdims=True).astype(np.float32)
    sd = coarse.std(axis=(0, 1, 3, 4), keepdims=True).astype(np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd)
    z = ((coarse - mu) / sd).astype(np.float16)
    print(f"  standardised + float16  {z.nbytes/1e6:.0f} MB")
    lost = np.abs(z.astype(np.float32) - (coarse - mu) / sd).max()
    print(f"  max float16 round-trip error: {lost:.2e} (in standard deviations)")

    np.savez_compressed(
        OUT, coarse=z, mu=mu, sd=sd, channels=np.array(names),
        init_year=d["init_year"], lead=d["lead"], valid=d["valid"],
        target=d["target"], mask=d["mask"], catchment=d["catchment"],
        fine_lats=d["fine_lats"], fine_lons=d["fine_lons"],
        n_row=N_ROW, n_col=N_COL,
    )
    print(f"\nwrote {OUT}  ({OUT.stat().st_size/1e6:.0f} MB on disk)")


if __name__ == "__main__":
    main()
