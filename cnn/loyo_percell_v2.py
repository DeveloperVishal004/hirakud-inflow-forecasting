"""Per-cell ResNet downscaler on the v2 archive -- Dong et al. (2025) sect. 3.2.1.

This is the paper's architecture and the paper's data path, not an adaptation:

    input      26 predictors on a 3 x 3 coarse patch centred on the target cell
               (the paper uses 19; ours is a superset -- same 15 upper-air
                fields, more surface fields)
    model      3 ResNet blocks, 64 -> 32 -> 16 feature maps, 3 x 3 kernels, ELU
               coordinate embedding merged with the flattened conv features,
               then two fully connected layers        (cnn/model.py)
    loss       b(1 - TS) + MSE with a differentiable threat score, eqs. 3-7
    optimiser  Adam with early stopping
    ensemble   each member downscaled separately, exactly as the paper does

WHY THE PATCH MATTERS NOW.  The paper chose 3 x 3 after testing 1x1, 5x5 and
7x7, so the patch size is a finding, not an arbitrary default.  On the old 5 x 5
download 48 % of these patches ran off the grid and were edge-clamped, feeding
the model a repeated row of cells as if it were data.  On the 7 x 7 v2 domain
that figure is 0 % -- every catchment cell has a genuine neighbourhood.

COST.  9,240 (init, lead) keys x 10 members x 224 catchment cells = 20.7 M
per-cell samples per epoch.  Cells are not drawn independently: all 224 patches
of one field are gathered and predicted together, which is both faster and
exactly equivalent, since they share the same coarse field.

Run:  python cnn/loyo_percell_v2.py [--epochs 60] [--members 10]
Out:  results/metrics/loyo_percell_v2.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C
from cnn.model import ResNetDownscaler, masked_hybrid_loss
from cnn.train_field import device, set_seed, r2
from cnn.loyo_field_v2 import V1_FOLDS

# Folds come from the DATA, not from config: the record now spans 2004-2022
# and a hard-coded YEAR_MAX silently drops every season after 2014.  Set
# STREAMFLOW_FULL=0 to fall back to the original 11-monsoon dataset.
import os
_FULL = os.environ.get("STREAMFLOW_FULL", "1") != "0"
DSET = (C.PROCESSED / "downscaling_v2_full.npz") if _FULL and (
    C.PROCESSED / "downscaling_v2_full.npz").exists() else (
    C.PROCESSED / "downscaling_v2.npz")
YEARS = sorted(set(int(y) for y in np.load(DSET, allow_pickle=True)["init_year"]))


def fine_grid():
    return (np.arange(C.FINE_LAT_MIN, C.FINE_LAT_MAX + C.FINE_RES / 2, C.FINE_RES),
            np.arange(C.FINE_LON_MIN, C.FINE_LON_MAX + C.FINE_RES / 2, C.FINE_RES))


def load_members(n_members: int):
    """-> coarse (n_key, n_member, 26, 7, 7), init dates, leads, channel names.

    Members are kept separate: the paper downscales each ensemble member, so the
    member is part of the sample, not something to average away first.
    """
    files = sorted(C.S2S_V2_DIR.glob("*.npz"))
    acc_idx = None
    inits, leads, blocks = [], [], []
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        f = z["forcing"]
        V = [str(x) for x in z["variables"]]
        if acc_idx is None:
            acc_idx = [V.index(v) for v in C.V2_ACCUMULATED_VARS]
            names = V
        f = f.copy()
        a = f[:, :, :, acc_idx]
        a[:, 1:] = np.diff(a, axis=1)          # undo accumulation from init
        f[:, :, :, acc_idx] = a
        # member 0 is the control; the paper's "10 ensemble members" are the
        # perturbed ones, so take 1..10 unless asked for all 11.
        f = f[:, :, 1:1 + n_members] if n_members <= 10 else f
        # Read the year axis from the file, never from config.  The archive now
        # holds two hindcast ranges; "C.YEAR_MIN + yi" would label the 2015-2022
        # files as 2004-2011 and every downstream fold would be wrong, silently.
        month, day = fp.stem.split("_")[:2]
        yrs = ([int(y) for y in z["years"]] if "years" in z.files
               else list(range(C.YEAR_MIN, C.YEAR_MIN + f.shape[0])))
        for yi in range(f.shape[0]):
            for li in range(f.shape[1]):
                inits.append(pd.Timestamp(f"{yrs[yi]}-{month}-{day}"))
                leads.append(li + 1)
        blocks.append(f)

    coarse = np.concatenate([b.reshape(-1, *b.shape[2:]) for b in blocks], axis=0)
    for v in ("ssrd", "sshf", "slhf"):        # J/m^2 per day -> mean W/m^2
        coarse[:, :, names.index(v)] /= C.SECONDS_PER_DAY
    return (coarse.astype(np.float32), pd.DatetimeIndex(inits),
            np.array(leads, np.int16), names)


def patch_index(cells, n_row=7, n_col=7):
    """For each target fine cell, the 3 x 3 coarse indices centred on it.

    n_row/n_col are the grid actually loaded: the Kaggle payload is cropped to
    the 6 x 5 subgrid these patches touch.  Both ranges start at 0, so the
    indices are identical either way -- only the bounds check changes.
    """
    flats, flons = fine_grid()
    clats, clons = np.array(C.V2_COARSE_LATS), np.array(C.V2_COARSE_LONS)
    rows, cols = [], []
    for a, b in cells:
        ci = int(np.abs(clats - flats[a]).argmin())
        cj = int(np.abs(clons - flons[b]).argmin())
        r = np.arange(ci - 1, ci + 2)
        c = np.arange(cj - 1, cj + 2)
        assert r.min() >= 0 and r.max() < n_row and c.min() >= 0 and c.max() < n_col, \
            f"patch ran off the {n_row}x{n_col} grid"
        rows.append(r); cols.append(c)
    return np.array(rows), np.array(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--members", type=int, default=10, help="paper uses 10 perturbed")
    ap.add_argument("--fields-per-batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--folds", type=int, default=len(YEARS))
    ap.add_argument("--sigmoid-sharpness", type=float,
                    default=C.LOSS_SIGMOID_SHARPNESS,
                    help="a in the differentiable threat score, "
                         "sigmoid(a*(pred - threshold)).  MEASURED 2026-09-08: "
                         "at a=1.0 the threshold is 24.46 mm while the median "
                         "sigmoid input is -15.95, so 99.74%% of cells sit in the "
                         "flat tail with mean gradient 1.2e-4 -- the TS term is "
                         "SWITCHED OFF, not merely under-weighted, which is why "
                         "a b sweep to 66 changed nothing.  Live-gradient share: "
                         "a=1.0 -> 0.19%%, a=0.3 -> 44%%, a=0.1 -> 100%%.  Tune THIS "
                         "before tuning b")
    ap.add_argument("--ts-weight", type=float, default=C.LOSS_TS_WEIGHT,
                    help="b in Dong eq. 3, loss = b*(1-TS) + MSE. MEASURED "
                         "2026-09-08: at b=1 the TS term is 0.375%% of the loss "
                         "(MSE 265 mm^2 vs b*(1-TS) 1.00), so the extreme-rain "
                         "term is inert and the model trains on plain MSE. The "
                         "paper does not specify b. b~66 makes TS ~20%% of the "
                         "loss. Raising it should RAISE peak ratio and threat "
                         "score while LOWERING overall R2 -- judge it on the "
                         "former, not the latter")
    ap.add_argument("--tag", default="", help="suffix for the output filenames, "
                                              "so a sweep does not overwrite")
    ap.add_argument("--val-rotate", action="store_true",
                    help="rotate the inner validation year instead of always "
                         "using the chronologically last one. OFF by default so "
                         "published results stay reproducible -- turning it on "
                         "requires a full re-run.")
    args = ap.parse_args()

    set_seed(C.SEED)
    dev = device()
    t0 = time.time()
    # Prefer the 19-monsoon packed file when it exists; percell_v2.npz is the
    # original 11-monsoon build, kept so v1 results stay reproducible.
    packed = next((q for q in (C.PROCESSED / "percell_v2_full.npz",
                               C.PROCESSED / "percell_v2.npz") if q.exists()),
                  C.PROCESSED / "percell_v2.npz")
    if packed.exists():
        # Already cropped, standardised and float16 -- see pack_percell_kaggle.py
        z = np.load(packed, allow_pickle=True)
        coarse = z["coarse"][:, :args.members].astype(np.float32)
        pre_standardised = True
        print(f"device={dev}  coarse {coarse.shape} from packed file "
              f"loaded in {time.time()-t0:.0f}s")
    else:
        coarse, inits, leads, names = load_members(args.members)
        z = np.load(DSET, allow_pickle=True)
        pre_standardised = False
        print(f"device={dev}  coarse {coarse.shape}  ({coarse.nbytes/1e6:.0f} MB) "
              f"loaded in {time.time()-t0:.0f}s")
    target, mask, catchment = z["target"], z["mask"], z["catchment"]
    dem = np.load(C.PROCESSED / "dem_fine.npz", allow_pickle=True)
    flats, flons = fine_grid()

    cells = np.argwhere(catchment)                       # (224, 2)
    rows, cols = patch_index(cells, coarse.shape[3], coarse.shape[4])
    # patch_index() asserts every 3x3 patch is in bounds, so reaching here
    # means none were clamped -- but say it from the data, not from a literal.
    print(f"  {len(cells)} catchment cells, "
          f"{len(cells)} in-bounds 3x3 patches on the "
          f"{coarse.shape[3]}x{coarse.shape[4]} coarse grid")

    # Coordinates for the embedding: latitude, longitude, elevation of the cell.
    coords = np.stack([flats[cells[:, 0]], flons[cells[:, 1]],
                       dem["elev_mean"][cells[:, 0], cells[:, 1]]], 1).astype(np.float32)

    y_cells = target[:, cells[:, 0], cells[:, 1]]        # (n_key, 224)
    m_cells = mask[:, cells[:, 0], cells[:, 1]]
    init_year = z["init_year"]
    n_key, n_mem = len(y_cells), coarse.shape[1]
    print(f"  {n_key} keys x {n_mem} members x {len(cells)} cells "
          f"= {n_key*n_mem*len(cells)/1e6:.1f}M per-cell samples/epoch\n")

    rows_t = torch.from_numpy(rows).to(dev)
    cols_t = torch.from_numpy(cols).to(dev)
    coords_t = torch.from_numpy(coords).to(dev)

    def gather(cb):
        """(B, C, 7, 7) -> (B*224, C, 3, 3): every cell's patch from every field."""
        p = cb[:, :, rows_t[:, :, None], cols_t[:, None, :]]   # (B, C, 224, 3, 3)
        return p.permute(0, 2, 1, 3, 4).reshape(-1, cb.shape[1], 3, 3)

    rows_out = []
    oof = np.zeros((n_key, len(cells)), np.float32)
    # Every member's downscaled field, kept because CRPS and the ensemble
    # streamflow forecast both need the spread, not just the mean.
    oof_mem = np.zeros((n_key, n_mem, len(cells)), np.float32)
    for Y in YEARS[:args.folds]:
        tf = time.time()
        test_k = init_year == Y
        # Inner validation year for early stopping.  DEFAULT: the chronologically
        # last non-test year -- which is 2022 for 18 of the 19 folds, so 2022
        # never enters training and every fold's early stopping is tuned on the
        # same monsoon.  That wastes a year of data and correlates model
        # selection across folds.  It is not a leak: the validation year is
        # always excluded from training.  --val-rotate gives each year one turn
        # as validation instead.  The documented requirement -- a WHOLE year,
        # never a random split, because fields inside one monsoon correlate --
        # holds either way.
        if args.val_rotate:
            inner = YEARS[(YEARS.index(Y) + 1) % len(YEARS)]
        else:
            inner = [y for y in YEARS if y != Y][-1]
        val_k = init_year == inner
        train_k = ~test_k & ~val_k

        mu = coarse[train_k].mean(axis=(0, 1, 3, 4), keepdims=True)
        sd = coarse[train_k].std(axis=(0, 1, 3, 4), keepdims=True)
        sd = np.where(sd < 1e-6, 1.0, sd)
        cmu = coords[  # coordinate scaler, train cells are all cells
            :].mean(0, keepdims=True)
        csd = np.where(coords.std(0, keepdims=True) < 1e-6, 1.0, coords.std(0, keepdims=True))
        cz = torch.from_numpy(((coords - cmu) / csd).astype(np.float32)).to(dev)

        # Domain-wide fallback, matching configs/config.py.  A cell with too
        # few observations previously fell back to 0.0, which makes every
        # prediction a "hit" for that cell and silently removes it from the
        # threat-score term.  Use the domain p90 over training rows instead.
        _dom = y_cells[train_k][m_cells[train_k]]
        dom_p90 = float(np.percentile(_dom, C.HEAVY_RAIN_PERCENTILE)) if _dom.size else 0.0
        thr = np.zeros(len(cells), np.float32)
        n_fallback = 0
        for j in range(len(cells)):
            v = y_cells[train_k, j][m_cells[train_k, j]]
            if len(v) >= C.HEAVY_RAIN_MIN_OBS:
                thr[j] = np.percentile(v, C.HEAVY_RAIN_PERCENTILE)
            else:
                thr[j] = dom_p90
                n_fallback += 1
        if n_fallback:
            print(f"    {n_fallback}/{len(cells)} cells below "
                  f"{C.HEAVY_RAIN_MIN_OBS} obs -> domain p90 {dom_p90:.2f} mm")
        thr_t = torch.from_numpy(thr).to(dev)

        model = ResNetDownscaler(n_features=coarse.shape[2], n_coords=3,
                                 dropout=C.DROPOUT).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=C.WEIGHT_DECAY)

        tr_idx = np.repeat(np.where(train_k)[0], n_mem)
        tr_mem = np.tile(np.arange(n_mem), train_k.sum())

        def run(sel_k, train: bool, members_out=None):
            idx = np.where(sel_k)[0]
            out = np.zeros((len(idx), len(cells)), np.float32)
            model.train(train)
            order = np.arange(len(idx))
            for s in range(0, len(idx), args.fields_per_batch):
                b = order[s:s + args.fields_per_batch]
                k = idx[b]
                # inference uses the ensemble MEAN of member predictions, which
                # is what the paper scores; training sees members individually
                mem = range(n_mem) if not train else [np.random.randint(n_mem)]
                acc, per_mem = 0, []
                for mi in mem:
                    cb = torch.from_numpy(coarse[k, mi]).to(dev)
                    cb = (cb - torch.from_numpy(mu[0, 0]).to(dev)) / torch.from_numpy(sd[0, 0]).to(dev)
                    p = model(gather(cb), cz.repeat(len(k), 1)).view(len(k), -1)
                    if train:
                        yb = torch.from_numpy(y_cells[k]).to(dev)
                        mb = torch.from_numpy(m_cells[k]).to(dev)
                        opt.zero_grad()
                        loss, _ = masked_hybrid_loss(
                            p, yb, mb, thr_t.expand_as(p),
                            a=args.sigmoid_sharpness, b=args.ts_weight,
                            log_space=C.LOSS_LOG_SPACE)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP_NORM)
                        opt.step()
                    else:
                        acc = acc + p.detach()
                        per_mem.append(p.detach().cpu().numpy())
                if not train:
                    out[b] = (acc / n_mem).cpu().numpy()
                    if members_out is not None:
                        members_out[b] = np.stack(per_mem, 1)
            return out

        best, best_state, bad = -np.inf, None, 0
        for ep in range(args.epochs):
            run(train_k, True)
            p = run(val_k, False)
            sel = m_cells[val_k]
            score = r2(p[sel], y_cells[val_k][sel])
            if score > best + C.EARLY_STOP_MIN_DELTA:
                best, bad = score, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break

        model.load_state_dict(best_state)
        mem_buf = np.zeros((int(test_k.sum()), n_mem, len(cells)), np.float32)
        p = run(test_k, False, members_out=mem_buf)
        oof[test_k] = p
        oof_mem[test_k] = mem_buf
        sel = m_cells[test_k]
        score = r2(p[sel], y_cells[test_k][sel])
        rows_out.append({"year": Y, "percell_v2": score, "field_v1": V1_FOLDS.get(Y, float('nan')),
                         "delta": score - V1_FOLDS.get(Y, float('nan')), "epochs": ep + 1,
                         "seconds": round(time.time() - tf, 1)})
        print(f"  {Y}: per-cell v2 {score:+.4f}   field v1 {V1_FOLDS.get(Y, float('nan')):+.4f}   "
              f"delta {score - V1_FOLDS.get(Y, float('nan')):+.4f}   ({ep+1} ep, {time.time()-tf:.0f}s)")

    v2 = np.array([r["percell_v2"] for r in rows_out])
    # V1_FOLDS stops at 2014, so every year after it has delta = nan.  Averaging
    # those in poisons delta_mean/delta_se/t to nan and makes "wins" count a
    # missing baseline as a loss.  Compare only where a baseline exists, and say
    # how many years that was -- the v2 scores themselves still use all folds.
    d_all = np.array([r["delta"] for r in rows_out])
    d = d_all[np.isfinite(d_all)]
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    summary = {"median": float(np.median(v2)), "mean": float(v2.mean()),
               "_v1_note": "field_v1/delta compare against the SUPERSEDED 11-monsoon "
                           "v1 run (5x5, 10 vars, leads 1-17, control member). Legacy "
                           "provenance only -- not a like-for-like baseline. Use raw EC "
                           "/ EC-QM from cnn/paper_metrics.py.",
               "delta_median": float(np.median(d)), "delta_mean": float(d.mean()),
               "delta_se": float(se), "t": float(d.mean() / se) if se else None,
               "wins": int((d > 0).sum()), "n_folds": len(v2),
               "n_compared": len(d)}
    print(f"\n=== per-cell v2 (Dong et al. architecture), {len(v2)} monsoons ===")
    print(f"  per-cell v2  median {summary['median']:+.4f}  mean {summary['mean']:+.4f}")
    # The v1 delta is LEGACY and deliberately demoted below the headline.  v1 is
    # 5x5 cells / 10 predictors / leads 1-17 / control member / 11 monsoons; this
    # is 7x7 / 26 / 1-30 / 10 members / 19.  The difference is not a measurement
    # of anything, and it exists for only the 11 years the two share.  The
    # baselines that apply to this model are raw EC and EC-QM on all 19 monsoons
    # (cnn/paper_metrics.py).  Kept so v1 stays reproducible, not to be quoted.
    print(f"  [legacy] vs field v1 on the {len(d)} shared monsoons only: "
          f"delta {summary['delta_mean']:+.4f} +/- {se:.4f} (SE), "
          f"wins {summary['wins']}/{len(d)} -- NOT a like-for-like baseline")

    C.METRICS.mkdir(parents=True, exist_ok=True)
    out = C.METRICS / f"loyo_percell_v2{args.tag}.json"
    out.write_text(json.dumps({"config": vars(args), "folds": rows_out,
                               "summary": summary}, indent=2, default=str))
    print(f"\nwrote {out}")

    np.savez_compressed(
        C.PROCESSED / f"percell_v2_oof{args.tag}.npz",
        oof=oof, oof_members=oof_mem, cells=cells,
        init_year=init_year, lead=z["lead"], valid=z["valid"],
        target=y_cells, mask=m_cells,
    )
    print(f"wrote {C.PROCESSED/('percell_v2_oof'+args.tag+'.npz')}  "
          f"<- per-member fields for CRPS and the LSTM stage")


if __name__ == "__main__":
    main()
