"""Train the field-level downscaler and compare it against the per-cell model.

Fields are assembled once into memory (5,797 samples x ~26 channels x 25 x 25
is only ~150 MB) rather than gathered per sample, because the whole point of
this model is that the sample IS the field.

Scored identically to the per-cell GBM -- per-cell R^2 over catchment cells on
the same splits -- so the architecture comparison is like-for-like.

Run:  python cnn/train_field.py [--epochs 80]
Out:  checkpoints/field_model.pt, results/metrics/field_downscaler.json
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C
from cnn.field_model import FieldDownscaler, masked_field_loss
from cnn.train_gbm import LagSource


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def build_fields():
    """Assemble coarse inputs, fine static/antecedent inputs, targets and masks."""
    g = np.load(C.PROCESSED / "coarse_grid.npz", allow_pickle=True)
    t = np.load(C.PROCESSED / "fine_target.npz")
    dem = np.load(C.PROCESSED / "dem_fine.npz", allow_pickle=True)
    clim = np.load(C.PROCESSED / "climatology.npz")["clim"]

    coarse = g["grid"].astype(np.float32)                       # (N, 10, 5, 5)
    split = g["split"].astype(str)
    lead = g["lead_day"].astype(np.int16)
    init = pd.to_datetime(g["time"])
    valid = init + pd.to_timedelta(lead, unit="D")
    doy = valid.dayofyear.values

    n, hw = len(coarse), t["land"].shape[0]
    lats, lons = t["fine_lats"], t["fine_lons"]

    # ---- static fine-resolution channels (identical for every sample)
    static = [dem[k] for k in ["elev_mean", "elev_std", "elev_min", "elev_max"]]
    static.append(np.repeat(lats[:, None], hw, 1))
    static.append(np.repeat(lons[None, :], hw, 0))
    static = np.stack(static).astype(np.float32)                # (6, 25, 25)

    # ---- per-sample fine channels: climatology, season, lead, observed lags
    lags = LagSource()
    idx = lags.day_index(init)
    lag_fields = []
    for k in (1, 2, 3):
        lag_fields.append(lags.obs[idx - k])
    for w in (7, 15, 30):
        lag_fields.append((lags.cum[idx] - lags.cum[idx - w]) / w)
    lag_fields = np.stack(lag_fields, axis=1).astype(np.float32)  # (N, 6, 25, 25)

    ang = 2 * np.pi * doy / 366.0
    ones = np.ones((n, 1, hw, hw), np.float32)
    per_sample = np.concatenate([
        clim[doy][:, None].astype(np.float32),                    # climatology map
        ones * np.sin(ang)[:, None, None, None],
        ones * np.cos(ang)[:, None, None, None],
        ones * (lead[:, None, None, None] / C.LEAD_MAX),
        lag_fields,
    ], axis=1)

    fine_static = np.concatenate(
        [np.broadcast_to(static, (n,) + static.shape), per_sample], axis=1
    ).astype(np.float32)                                          # (N, 16, 25, 25)

    # Raw (unstandardised) ECMWF rainfall bilinearly upsampled to the fine grid:
    # the baseline the residual model corrects.  Must stay in mm, since it is
    # added to the network output in physical units.
    feats = list(g["features"])
    tp_coarse = coarse[:, feats.index("tp")]
    tp_interp = torch.nn.functional.interpolate(
        torch.from_numpy(tp_coarse[:, None]), size=(hw, hw),
        mode="bilinear", align_corners=False).squeeze(1).numpy().astype(np.float32)

    target = t["target"].astype(np.float32)
    mask = t["mask"] & t["land"][None]
    return coarse, fine_static, target, mask, split, lead, t["catchment"], tp_interp


def standardise(x, fit_mask, axes):
    """Channel-wise standardisation using TRAIN samples only."""
    mu = x[fit_mask].mean(axis=axes, keepdims=True)
    sd = x[fit_mask].std(axis=axes, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return ((x - mu) / sd).astype(np.float32)


def r2(p, o):
    return float(1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum())


def evaluate(model, coarse, fine, tp, target, dev, bs=128):
    model.eval()
    out = np.empty_like(target)
    with torch.no_grad():
        for i in range(0, len(coarse), bs):
            c = torch.from_numpy(coarse[i:i + bs]).to(dev)
            f = torch.from_numpy(fine[i:i + bs]).to(dev)
            t = torch.from_numpy(tp[i:i + bs]).to(dev)
            out[i:i + bs] = model(c, f, t).cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--residual", action="store_true", default=True)
    ap.add_argument("--no-residual", dest="residual", action="store_false")
    ap.add_argument("--tag", type=str, default="field")
    args = ap.parse_args()

    set_seed(C.SEED)
    dev = device()
    print(f"device={dev}")

    coarse, fine, target, mask, split, lead, catchment, tp_interp = build_fields()
    tr, va, te = (split == "train"), (split == "val"), (split == "test")

    coarse = standardise(coarse, tr, axes=(0, 2, 3))
    fine = standardise(fine, tr, axes=(0, 2, 3))
    print(f"coarse {coarse.shape}  fine_static {fine.shape}  target {target.shape}")
    print(f"train {tr.sum()}  val {va.sum()}  test {te.sum()} fields")

    # Per-cell heavy-rain threshold, train rows only (matches the per-cell model).
    thr = np.zeros(target.shape[1:], np.float32)
    for a in range(thr.shape[0]):
        for b in range(thr.shape[1]):
            v = target[tr, a, b][mask[tr, a, b]]
            thr[a, b] = np.percentile(v, C.HEAVY_RAIN_PERCENTILE) if len(v) >= 30 else 0.0
    thr_t = torch.from_numpy(thr).to(dev)

    ds = TensorDataset(torch.from_numpy(coarse[tr]), torch.from_numpy(fine[tr]),
                       torch.from_numpy(tp_interp[tr]),
                       torch.from_numpy(target[tr]), torch.from_numpy(mask[tr]))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    generator=torch.Generator().manual_seed(C.SEED))

    model = FieldDownscaler(n_coarse=coarse.shape[1],
                            n_fine_static=fine.shape[1],
                            fine_hw=target.shape[1],
                            width=args.width, dropout=args.dropout,
                            residual=args.residual).to(dev)
    print(f"variant: {args.tag}  width={args.width} dropout={args.dropout} "
          f"residual={args.residual}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", factor=0.5, patience=5)
    print(f"params {sum(p.numel() for p in model.parameters()):,}")

    cat = catchment[None]
    best, best_state, bad = -np.inf, None, 0
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for c, f, tp, y, m in dl:
            c, f, tp = c.to(dev), f.to(dev), tp.to(dev)
            y, m = y.to(dev), m.to(dev)
            opt.zero_grad(set_to_none=True)
            pred = model(c, f, tp)
            loss, _ = masked_field_loss(pred, y, m, thr_t,
                                        a=C.LOSS_SIGMOID_SHARPNESS, b=C.LOSS_TS_WEIGHT)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP_NORM)
            opt.step()
            tot += loss.item()

        pv = evaluate(model, coarse[va], fine[va], tp_interp[va], target[va], dev)
        mv = mask[va] & cat
        v_r2 = r2(pv[mv], target[va][mv])
        sched.step(v_r2)

        flag = ""
        if v_r2 > best + 1e-5:
            best, best_state, bad = v_r2, {k: v.detach().cpu().clone()
                                           for k, v in model.state_dict().items()}, 0
            flag = "  <- best"
        else:
            bad += 1
        print(f"epoch {ep:3d} ({time.time()-t0:5.1f}s)  train loss {tot/len(dl):8.3f}   "
              f"val R2 {v_r2:+.4f}{flag}")
        if bad >= args.patience:
            print(f"early stopping after {args.patience} epochs without improvement")
            break

    model.load_state_dict(best_state)
    pt = evaluate(model, coarse[te], fine[te], tp_interp[te], target[te], dev)
    mt = mask[te] & cat
    res = {"val_R2_catchment": best, "test_R2_catchment": r2(pt[mt], target[te][mt])}
    res["test_by_lead"] = {}
    lt = lead[te]
    for lo, hi in [(1, 3), (4, 7), (8, 12), (13, 17)]:
        s = (lt >= lo) & (lt <= hi)
        mm = mask[te][s] & cat
        res["test_by_lead"][f"{lo}-{hi}"] = r2(pt[s][mm], target[te][s][mm])

    print(f"\nFIELD model   val R2 {res['val_R2_catchment']:+.4f}   "
          f"test R2 {res['test_R2_catchment']:+.4f}")
    print(f"  by lead {json.dumps({k: round(v, 3) for k, v in res['test_by_lead'].items()})}")
    print("\nper-cell reference (cnn/train_gbm.py, same splits):")
    ref = C.METRICS / "downscaler_gbm.json"
    if ref.exists():
        d = json.load(open(ref))
        sel = d.get("selected", "with-lags")
        print(f"  GBM ({sel})  val R2 {d[sel]['val_R2']:+.4f}   test R2 {d[sel]['test_R2']:+.4f}")
        print(f"  by lead {json.dumps({k: round(v, 3) for k, v in d[sel]['test_by_lead'].items()})}")
        res["gbm_reference"] = d[sel]

    C.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    res["variant"] = {"tag": args.tag, "width": args.width,
                      "dropout": args.dropout, "residual": args.residual,
                      "params": sum(p.numel() for p in model.parameters())}
    torch.save({"model_state": best_state, "n_coarse": coarse.shape[1],
                "n_fine_static": fine.shape[1], "variant": res["variant"]},
               C.CHECKPOINTS / f"{args.tag}.pt")
    C.METRICS.mkdir(parents=True, exist_ok=True)
    (C.METRICS / f"{args.tag}.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {C.CHECKPOINTS / (args.tag + '.pt')} and "
          f"{C.METRICS / (args.tag + '.json')}")


if __name__ == "__main__":
    main()
