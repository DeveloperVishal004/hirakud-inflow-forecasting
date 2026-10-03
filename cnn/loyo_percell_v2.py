"""Per-cell ResNet downscaler on the v2 archive -- Dong et al. (2025) sect. 3.2.1.

The paper's architecture, loss, predictors and split:

    input      18 predictors on a 3 x 3 coarse patch centred on the target cell:
               orography, tp, cp and u, v, q, t, gh at 200/500/850 hPa
               (C.DONG_PREDICTORS; --all-predictors uses all 26 archive fields)
    model      3 ResNet blocks, 64 -> 32 -> 16 feature maps, 3 x 3 kernels, ELU
               coordinate embedding merged with the flattened conv features,
               then two fully connected layers        (cnn/model.py)
    loss       b(1 - TS) + MSE, eqs. 3-7, with Supplement Table S1's a = 2 and
               b = 0.4 / 0.8 / 1.5 / 2 for leads 1-7 / 8-15 / 16-23 / 24-30,
               on rainfall divided by its training-year standard deviation
    optimiser  Adam, early stopping on the same loss over a held-out year
    ensemble   one independently trained model per perturbed member
    split      --split fixed (default): train 2004-2016, stop on 2017, test
               2018-2022, mirroring the paper's 2002-2015 / 2016-2019.
               --split loyo: one fold per monsoon, for the final robustness check.

WHY THE PATCH MATTERS.  The paper chose 3 x 3 after testing 1x1, 5x5 and 7x7.
On the 7 x 7 v2 domain every catchment cell has a genuine neighbourhood; none
of the patches is edge-clamped.

COST.  All 224 patches of one field are gathered and predicted together, which
is faster than drawing cells independently and exactly equivalent, since they
share the same coarse field.

Run:  python cnn/loyo_percell_v2.py [--split fixed] [--epochs 60] [--members 10]
Out:  results/metrics/percell_v2_<split><tag>.json
      data/processed/percell_v2_<split><tag>.npz   (test-row predictions, mm)
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C
from cnn.model import ResNetDownscaler, masked_hybrid_loss

# The dataset the member archive is aligned to (used by quantile_mapping.py and
# pack_percell_kaggle.py).  STREAMFLOW_FULL=0 falls back to the 11-monsoon build.
_FULL = os.environ.get("STREAMFLOW_FULL", "1") != "0"
DSET = (C.PROCESSED / "downscaling_v2_full.npz") if _FULL and (
    C.PROCESSED / "downscaling_v2_full.npz").exists() else (
    C.PROCESSED / "downscaling_v2.npz")


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def r2(p, o):
    return float(1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum())


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




def lead_window(lead):
    """Lead (days) -> index into C.DONG_LEAD_WINDOWS."""
    lead = np.asarray(lead)
    w = np.full(lead.shape, -1, np.int64)
    for i, (lo, hi) in enumerate(C.DONG_LEAD_WINDOWS):
        w[(lead >= lo) & (lead <= hi)] = i
    assert (w >= 0).all(), "lead outside every Table S1 window"
    return w


def split_folds(init_year, split: str, n_folds=None):
    """-> [(name, test_rows, val_rows)].  Training rows are everything else.

    Validation is always a WHOLE year, never a random split: fields inside one
    monsoon are correlated, so a random split would leak.
    """
    years = sorted(set(int(y) for y in init_year))
    if split == "fixed":
        test = np.isin(init_year, C.DONG_TEST_YEARS)
        val = init_year == C.DONG_VAL_YEAR
        assert test.any() and val.any(), "fixed split years missing from the data"
        return [("fixed", test, val)]
    folds = []
    for Y in years[:n_folds]:
        # The chronologically last non-test year stops early (2022 for 18 of 19
        # folds); it never enters training.
        inner = [y for y in years if y != Y][-1]
        folds.append((str(Y), init_year == Y, init_year == inner))
    return folds


def load_inputs(n_members: int, all_predictors: bool):
    """-> coarse (n_key, member, channel, row, col) float32, channels, dataset z.

    Prefers the packed file (cropped, standardised, float16 -- see
    preprocessing/pack_percell_kaggle.py), which is all a Kaggle run ships.
    """
    packed = next((q for q in (C.PROCESSED / "percell_v2_full.npz",
                               C.PROCESSED / "percell_v2.npz") if q.exists()), None)
    if packed is not None:
        z = np.load(packed, allow_pickle=True)
        names = [str(c) for c in z["channels"]]
        keep = list(range(len(names))) if all_predictors else \
            [names.index(v) for v in C.DONG_PREDICTORS]
        coarse = z["coarse"][:, :n_members][:, :, keep].astype(np.float32)
        print(f"  inputs from {packed.name}")
    else:
        coarse, inits, leads, names = load_members(n_members)
        z = np.load(DSET, allow_pickle=True)
        keep = list(range(len(names))) if all_predictors else \
            [names.index(v) for v in C.DONG_PREDICTORS]
        coarse = coarse[:, :, keep]
        print(f"  inputs from {C.S2S_V2_DIR} (unaligned path: check row order)")
    return coarse, [names[i] for i in keep], z


def parse_members(spec: str, n_mem: int):
    """'' -> all; '0-4' -> [0..4]; '5,7' -> [5, 7]."""
    if not spec:
        return list(range(n_mem))
    ids = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        ids += list(range(int(lo), int(hi or lo) + 1))
    assert ids and min(ids) >= 0 and max(ids) < n_mem, f"--member-ids {spec} outside 0-{n_mem - 1}"
    return sorted(set(ids))


def fold_readout(pred, obs, ok):
    """Quick ensemble-mean scores; cnn/paper_metrics.py does the paper's scoring."""
    p = pred.mean(axis=1)
    basin_p = np.nanmean(np.where(ok, p, np.nan), 1)
    basin_o = np.nanmean(np.where(ok, obs, np.nan), 1)
    return {"percell_r2": r2(p[ok], obs[ok]),
            "basin_rmse_mm": float(np.sqrt(np.nanmean((basin_p - basin_o) ** 2)))}


def merge_parts(split: str, tag: str):
    """Combine per-GPU _part outputs into the single run they add up to."""
    stem = f"percell_v2_{split}{tag}"
    parts = sorted(C.PROCESSED.glob(f"{stem}_part*.npz"))
    if not parts:
        raise SystemExit(f"no {stem}_part*.npz to merge")
    zs = [np.load(p, allow_pickle=True) for p in parts]
    js = [json.loads((C.METRICS / f"{p.stem}.json").read_text()) for p in parts]
    base = zs[0]
    for z in zs[1:]:
        for k in ("tested", "init_year", "lead", "valid", "cells"):
            assert np.array_equal(z[k], base[k]), f"parts disagree on {k}"
    pred = base["pred_members"].copy()
    for z in zs[1:]:
        both = np.isfinite(pred) & np.isfinite(z["pred_members"])
        assert not both.any(), "two parts trained the same member"
        pred = np.where(np.isfinite(pred), pred, z["pred_members"])
    tested = base["tested"]
    missing = [m for m in range(pred.shape[1]) if not np.isfinite(pred[tested, m]).all()]
    assert not missing, f"members {missing} have no predictions"

    folds = []
    for fi, f0 in enumerate(js[0]["folds"]):
        members = sorted((m for j in js for m in j["folds"][fi]["members"]),
                         key=lambda m: (m["member"], m.get("model", "")))
        test_k = np.isin(base["init_year"], C.DONG_TEST_YEARS) if split == "fixed" \
            else base["init_year"] == int(f0["fold"])
        folds.append({"fold": f0["fold"], "rain_scale_mm": f0["rain_scale_mm"],
                      "members": members,
                      "seconds": max(j["folds"][fi]["seconds"] for j in js),
                      **fold_readout(pred[test_k], base["target"][test_k],
                                     base["mask"][test_k])})
        print(f"  [{f0['fold']}] {len(members)} models  ensemble-mean per-cell R^2 "
              f"{folds[-1]['percell_r2']:+.4f}   basin-mean RMSE "
              f"{folds[-1]['basin_rmse_mm']:.2f} mm/day")
    cfg = dict(js[0]["config"], member_ids="", merged_from=[p.name for p in parts])
    (C.METRICS / f"{stem}.json").write_text(json.dumps(
        {"config": cfg, "predictors": js[0]["predictors"], "folds": folds},
        indent=2, default=str))
    np.savez_compressed(C.PROCESSED / f"{stem}.npz", pred_members=pred, tested=tested,
                        cells=base["cells"], init_year=base["init_year"],
                        lead=base["lead"], valid=base["valid"],
                        target=base["target"], mask=base["mask"])
    print(f"wrote {stem}.npz and {stem}.json from {len(parts)} parts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["fixed", "loyo"], default="fixed")
    ap.add_argument("--epochs", type=int, default=C.MAX_EPOCHS)
    ap.add_argument("--members", type=int, default=10, help="paper uses 10 perturbed")
    ap.add_argument("--fields-per-batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=C.EARLY_STOP_PATIENCE)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--folds", type=int, default=None, help="loyo only: first N folds")
    ap.add_argument("--sigmoid-sharpness", type=float, default=C.DONG_SIGMOID_SHARPNESS,
                    help="a in sigmoid(a*(pred - threshold)), in scaled units")
    ap.add_argument("--ts-weights", default=",".join(str(b) for b in C.DONG_TS_WEIGHTS),
                    help="b per lead window, comma-separated (Table S1)")
    ap.add_argument("--no-rain-scale", action="store_true",
                    help="train on rainfall in mm (the TS term is then inert)")
    ap.add_argument("--all-predictors", action="store_true",
                    help="use all 26 archive fields instead of the paper's 18")
    ap.add_argument("--tag", default="", help="suffix for the output filenames")
    ap.add_argument("--member-ids", default="",
                    help="train only these members, e.g. 0-4 or 5,6 (one process "
                         "per GPU); outputs get a _part suffix for --merge")
    ap.add_argument("--per-window", action="store_true",
                    help="train a separate model per lead window (1-7, 8-15, 16-23, "
                         "24-30) for each member, instead of one model for all leads")
    ap.add_argument("--merge", action="store_true",
                    help="combine the _part outputs of this --split/--tag and exit")
    args = ap.parse_args()
    if args.merge:
        merge_parts(args.split, args.tag)
        return
    ts_weights = [float(b) for b in args.ts_weights.split(",")]
    assert len(ts_weights) == len(C.DONG_LEAD_WINDOWS), "one b per lead window"

    set_seed(C.SEED)
    dev = device()
    t0 = time.time()
    coarse, channels, z = load_inputs(args.members, args.all_predictors)
    print(f"device={dev}  coarse {coarse.shape}  ({len(channels)} predictors) "
          f"loaded in {time.time()-t0:.0f}s")
    target, mask, catchment = z["target"], z["mask"], z["catchment"]
    init_year, lead = z["init_year"], z["lead"]
    flats, flons = fine_grid()

    cells = np.argwhere(catchment)                       # (224, 2)
    rows, cols = patch_index(cells, coarse.shape[3], coarse.shape[4])
    print(f"  {len(cells)} catchment cells, all 3x3 patches in bounds on the "
          f"{coarse.shape[3]}x{coarse.shape[4]} coarse grid")

    # The paper embeds latitude and longitude. Elevation stays in the coarse
    # predictor channels rather than the coordinate embedding.
    coords = np.stack([flats[cells[:, 0]], flons[cells[:, 1]]], 1).astype(np.float32)
    csd = np.where(coords.std(0, keepdims=True) < 1e-6, 1.0, coords.std(0, keepdims=True))
    cz = torch.from_numpy(((coords - coords.mean(0, keepdims=True)) / csd)
                          .astype(np.float32)).to(dev)

    y_cells = target[:, cells[:, 0], cells[:, 1]]        # (n_key, 224) mm
    m_cells = mask[:, cells[:, 0], cells[:, 1]]
    win = lead_window(lead)
    n_key, n_mem = len(y_cells), coarse.shape[1]

    rows_t = torch.from_numpy(rows).to(dev)
    cols_t = torch.from_numpy(cols).to(dev)
    def gather(cb):
        """(B, C, H, W) -> (B*224, C, 3, 3): every cell's patch from every field."""
        p = cb[:, :, rows_t[:, :, None], cols_t[:, None, :]]   # (B, C, 224, 3, 3)
        return p.permute(0, 2, 1, 3, 4).reshape(-1, cb.shape[1], 3, 3)

    member_ids = parse_members(args.member_ids, n_mem)
    stem = f"percell_v2_{args.split}{args.tag}"
    if len(member_ids) < n_mem:
        stem += f"_part{member_ids[0]}-{member_ids[-1]}"
    ckpt_dir = C.CHECKPOINTS / f"percell_v2_{args.split}{args.tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    folds = split_folds(init_year, args.split, args.folds)
    rows_out = []
    pred_mem = np.full((n_key, n_mem, len(cells)), np.nan, np.float32)   # mm
    tested = np.zeros(n_key, bool)
    for name, test_k, val_k in folds:
        tf = time.time()
        train_k = ~test_k & ~val_k
        tr_years = sorted(set(int(y) for y in init_year[train_k]))
        print(f"\n[{name}] train {tr_years[0]}-{tr_years[-1]} ({int(train_k.sum())} keys)  "
              f"val {sorted(set(init_year[val_k].tolist()))}  "
              f"test {sorted(set(init_year[test_k].tolist()))}  "
              f"x {n_mem} members x {len(cells)} cells")

        obs_tr = y_cells[train_k][m_cells[train_k]]
        scale = 1.0 if args.no_rain_scale else float(obs_tr.std())

        # Heavy-rain threshold: per-cell p90 over training rows.  A cell with too
        # few observations takes the domain p90 rather than 0.0, which would make
        # every prediction a hit and drop that cell from the TS term.
        dom_p90 = float(np.percentile(obs_tr, C.HEAVY_RAIN_PERCENTILE))
        thr = np.full(len(cells), dom_p90, np.float32)
        for j in range(len(cells)):
            v = y_cells[train_k, j][m_cells[train_k, j]]
            if len(v) >= C.HEAVY_RAIN_MIN_OBS:
                thr[j] = np.percentile(v, C.HEAVY_RAIN_PERCENTILE)
        print(f"  rain scale {scale:.2f} mm   p90 threshold median "
              f"{np.median(thr):.1f} mm ({np.median(thr)/scale:.2f} scaled)")
        thr_t = torch.from_numpy(thr / scale).to(dev)

        def batch(k):
            return (torch.from_numpy(y_cells[k] / scale).float().to(dev),
                    torch.from_numpy(m_cells[k]).to(dev),
                    torch.from_numpy(win[k])[:, None].expand(-1, len(cells)).to(dev))

        def train_model(mi, train_k, val_k, test_k, label):
            """Train one model for member mi on the given rows.

            -> predictions for test_k rows in mm, info.  `label` names the
            checkpoint: a member, or a member and lead window.
            """
            mu = coarse[train_k, mi].mean(axis=(0, 2, 3), keepdims=True)
            sd = coarse[train_k, mi].std(axis=(0, 2, 3), keepdims=True)
            sd = np.where(sd < 1e-6, 1.0, sd)
            mu_t = torch.from_numpy(mu[0]).to(dev)
            sd_t = torch.from_numpy(sd[0]).to(dev)
            # Seed per member, so a member's model does not depend on which
            # members trained before it in the same process.
            rng = np.random.default_rng(C.SEED + mi)
            torch.manual_seed(C.SEED + mi)

            model = ResNetDownscaler(n_features=coarse.shape[2], n_coords=2,
                                     dropout=C.DROPOUT).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                                   weight_decay=C.WEIGHT_DECAY)

            def forward(k):
                cb = (torch.from_numpy(coarse[k, mi]).to(dev) - mu_t) / sd_t
                return model(gather(cb), cz.repeat(len(k), 1)).view(len(k), -1)

            def loss_on(p, k):
                yb, mb, wb = batch(k)
                return masked_hybrid_loss(p, yb, mb, thr_t.expand_as(p),
                                          a=args.sigmoid_sharpness, b=ts_weights,
                                          log_space=C.LOSS_LOG_SPACE, window=wb)

            def predict(sel_k):
                """Scaled predictions for every row of sel_k, eval mode."""
                idx = np.where(sel_k)[0]
                model.eval()
                with torch.no_grad():
                    return idx, torch.cat([forward(idx[s:s + 64])
                                           for s in range(0, len(idx), 64)])

            tr_idx = np.where(train_k)[0]
            best, best_state, bad, history = np.inf, None, 0, []
            for ep in range(args.epochs):
                model.train()
                perm = rng.permutation(tr_idx)
                for s in range(0, len(perm), args.fields_per_batch):
                    k = perm[s:s + args.fields_per_batch]
                    opt.zero_grad()
                    loss, _ = loss_on(forward(k), k)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP_NORM)
                    opt.step()
                # Early stopping on the training loss itself, over the whole
                # validation year: stopping on R^2 would pick the most MSE-like
                # epoch and undo what the TS term is there for.
                idx, p = predict(val_k)
                with torch.no_grad():
                    vloss, info = loss_on(p, idx)
                vloss = float(vloss)
                history.append({"epoch": ep + 1, "val_loss": vloss,
                                "val_mse": float(info["mse"]),
                                "val_ts": info["ts_by_window"]})
                if not np.isfinite(vloss):
                    raise RuntimeError(f"member {mi}: validation loss {vloss} at epoch {ep + 1}")
                improved = vloss < best - C.EARLY_STOP_MIN_DELTA
                if improved:
                    best, bad, best_ep = vloss, 0, ep + 1
                    best_state = {n: v.detach().clone() for n, v in model.state_dict().items()}
                else:
                    bad += 1
                print(f"      {label} ep {ep + 1:2d}  val loss {vloss:.4f}  mse {history[-1]['val_mse']:.4f}"
                      f"  TS {[round(t, 3) for t in info['ts_by_window']]}"
                      f"{'  *' if improved else ''}", flush=True)
                if bad >= args.patience:
                    break
            model.load_state_dict(best_state)
            torch.save({"state_dict": best_state, "mu": mu[0], "sd": sd[0],
                        "rain_scale_mm": scale, "threshold_mm": thr, "channels": channels,
                        "cells": cells, "member": mi, "fold": name, "best_epoch": best_ep,
                        "config": vars(args)}, ckpt_dir / f"{name}_{label}.pt")
            idx, p = predict(test_k)
            return p.cpu().numpy() * scale, {"member": mi, "model": label, "epochs": ep + 1,
                                             "best_epoch": best_ep, "best_val_loss": best,
                                             "history": history}

        member_info = []
        for mi in member_ids:
            # Per-window: Table S1 sets b by lead window, and one model for all
            # leads was measured to shrink every lead alike -- discarding the
            # skill ECMWF has at leads 1-7.  Each window model sees only its leads.
            windows = range(len(C.DONG_LEAD_WINDOWS)) if args.per_window else [None]
            for w in windows:
                tm = time.time()
                in_w = np.ones(n_key, bool) if w is None else win == w
                label = f"member{mi:02d}" + ("" if w is None else
                                             "_lead{}-{}".format(*C.DONG_LEAD_WINDOWS[w]))
                pm, info = train_model(mi, train_k & in_w, val_k & in_w,
                                       test_k & in_w, label)
                pred_mem[test_k & in_w, mi] = pm
                member_info.append(info)
                print(f"    {label}: {info['epochs']} epochs, best {info['best_epoch']}  "
                      f"val loss {info['best_val_loss']:.4f}  ({time.time()-tm:.0f}s)",
                      flush=True)
        tested |= test_k
        rows_out.append({"fold": name, "rain_scale_mm": scale, "members": member_info,
                         "seconds": round(time.time() - tf, 1),
                         **fold_readout(pred_mem[test_k][:, member_ids],
                                        y_cells[test_k], m_cells[test_k])})
        print(f"  [{name}] ensemble-mean per-cell R^2 {rows_out[-1]['percell_r2']:+.4f}   "
              f"basin-mean RMSE {rows_out[-1]['basin_rmse_mm']:.2f} mm/day   "
              f"({time.time()-tf:.0f}s)")

    C.METRICS.mkdir(parents=True, exist_ok=True)
    out = C.METRICS / f"{stem}.json"
    out.write_text(json.dumps({"config": vars(args), "predictors": channels,
                               "folds": rows_out}, indent=2, default=str))
    print(f"\nwrote {out}")
    np.savez_compressed(
        C.PROCESSED / f"{stem}.npz",
        pred_members=pred_mem, tested=tested, cells=cells,
        init_year=init_year, lead=lead, valid=z["valid"],
        target=y_cells, mask=m_cells,
    )
    print(f"wrote {C.PROCESSED / (stem + '.npz')}  <- per-member test predictions, mm")


if __name__ == "__main__":
    main()
