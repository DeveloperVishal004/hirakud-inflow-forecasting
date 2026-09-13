"""Leave-one-year-out field downscaler on the v2 S2S archive.

Same folds, same masks, same metric as cnn/loyo_field.py, so the only thing that
changed is the input data:

    v1  5x5 grid, 10 vars, leads 1-17, control member    -> median R^2 +0.1047
    v2  7x7 grid, 26 vars + 26 ensemble-spread channels,
        leads 1-30, 11-member ensemble                   -> this run

Eleven folds, each holding out one monsoon by INITIALISATION year.  Never split
at random: one valid date is reached by several initialisations at different
leads, so a random split puts the same observed rainfall field in train and test.

The inner validation year for early stopping is the chronologically last
training year, again a whole year rather than random fields -- fields from one
monsoon are strongly correlated and a random inner split leaks.

Run:  python cnn/loyo_field_v2.py [--epochs 60] [--width 32]
Out:  results/metrics/loyo_field_v2.json
"""

import argparse
import json
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
from cnn.train_field import standardise, r2, device, set_seed
from cnn.train_gbm import LagSource

# Folds come from the DATA, not from config: the record now spans 2004-2022
# and a hard-coded YEAR_MAX silently drops every season after 2014.  Set
# STREAMFLOW_FULL=0 to fall back to the original 11-monsoon dataset.
import os
_FULL = os.environ.get("STREAMFLOW_FULL", "1") != "0"
DSET = (C.PROCESSED / "downscaling_v2_full.npz") if _FULL and (
    C.PROCESSED / "downscaling_v2_full.npz").exists() else (
    C.PROCESSED / "downscaling_v2.npz")
YEARS = sorted(set(int(y) for y in np.load(DSET, allow_pickle=True)["init_year"]))

# v1 field-downscaler folds (results/metrics/loyo_field.json) -- the baseline
# this run has to beat, fold for fold.
V1_FOLDS = {2004: 0.0510, 2005: 0.1133, 2006: 0.1047, 2007: 0.1089,
            2008: 0.1105, 2009: 0.1390, 2010: 0.0905, 2011: 0.0877,
            2012: 0.0653, 2013: 0.0612, 2014: 0.1313}


def build_fine_static(valid, lead, hw):
    """Fine-resolution channels: terrain, position, season, lead, antecedent rain.

    These do not come from the forecast -- they are what the model knows about
    the ground and the recent past regardless of what ECMWF said.
    """
    dem = np.load(C.PROCESSED / "dem_fine.npz", allow_pickle=True)
    clim = np.load(C.PROCESSED / "climatology.npz")["clim"]
    z = np.load(DSET, allow_pickle=True)
    flats, flons = z["fine_lats"], z["fine_lons"]

    static = [dem[k] for k in ("elev_mean", "elev_std", "elev_min", "elev_max")]
    static.append(np.repeat(flats[:, None], hw, 1))
    static.append(np.repeat(flons[None, :], hw, 0))
    static = np.stack(static).astype(np.float32)

    n = len(valid)
    doy = valid.dayofyear.values
    ang = 2 * np.pi * doy / 366.0
    ones = np.ones((n, 1, hw, hw), np.float32)

    # Antecedent observed rain is indexed by INITIALISATION date: at forecast
    # time that is the last observation available.  Indexing it by valid date
    # would leak the future into the predictors.
    init = valid - pd.to_timedelta(lead, unit="D")
    lags = LagSource()
    idx = lags.day_index(init)
    lag_fields = [lags.obs[idx - k] for k in (1, 2, 3)]
    lag_fields += [(lags.cum[idx] - lags.cum[idx - w]) / w for w in (7, 15, 30)]
    lag_fields = np.stack(lag_fields, axis=1).astype(np.float32)

    per_sample = np.concatenate([
        clim[doy][:, None].astype(np.float32),
        ones * np.sin(ang)[:, None, None, None],
        ones * np.cos(ang)[:, None, None, None],
        ones * (lead[:, None, None, None] / C.V2_LEAD_MAX),
        lag_fields,
    ], axis=1)
    return np.concatenate(
        [np.broadcast_to(static, (n,) + static.shape), per_sample], axis=1
    ).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--val-rotate", action="store_true",
                    help="rotate the inner validation year instead of always "
                         "using the chronologically last one. OFF by default so "
                         "published results stay reproducible -- turning it on "
                         "requires a full re-run.")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-lead", type=int, default=C.V2_LEAD_MAX,
                    help="restrict to leads <= this, for a like-for-like v1 run")
    args = ap.parse_args()

    set_seed(C.SEED)
    dev = device()
    z = np.load(DSET, allow_pickle=True)
    coarse = z["coarse"]; target = z["target"]; mask = z["mask"]
    lead = z["lead"].astype(np.int16); init_year = z["init_year"]
    catchment = z["catchment"]; valid = pd.to_datetime(z["valid"])
    hw = target.shape[1]

    keep = lead <= args.max_lead
    coarse, target, mask = coarse[keep], target[keep], mask[keep]
    lead, init_year, valid = lead[keep], init_year[keep], valid[keep]

    print(f"device={dev}  coarse {coarse.shape}  target {target.shape}")
    print(f"leads 1-{args.max_lead}  width={args.width}  epochs={args.epochs}\n")

    fine = build_fine_static(valid, lead, hw)

    # Raw ensemble-mean rainfall on the fine grid: the baseline the residual
    # head corrects.  Kept in mm, since it is added to the output in mm.
    ch = [str(c) for c in z["channels"]]
    tp = torch.nn.functional.interpolate(
        torch.from_numpy(coarse[:, ch.index("tp")][:, None]), size=(hw, hw),
        mode="bilinear", align_corners=False).squeeze(1).numpy().astype(np.float32)

    cat = catchment[None]
    rows, oof = [], np.zeros_like(target)
    for Y in YEARS:
        t0 = time.time()
        test_m = init_year == Y
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
        val_m = init_year == inner
        train_m = ~test_m & ~val_m

        cs = standardise(coarse, train_m, axes=(0, 2, 3))
        fs = standardise(fine, train_m, axes=(0, 2, 3))

        thr = np.zeros(target.shape[1:], np.float32)
        for a in range(hw):
            for b in range(hw):
                v = target[train_m, a, b][mask[train_m, a, b]]
                thr[a, b] = (np.percentile(v, C.HEAVY_RAIN_PERCENTILE)
                             if len(v) >= C.HEAVY_RAIN_MIN_OBS else 0.0)
        thr_t = torch.from_numpy(thr).to(dev)

        dl = DataLoader(
            TensorDataset(torch.from_numpy(cs[train_m]), torch.from_numpy(fs[train_m]),
                          torch.from_numpy(tp[train_m]), torch.from_numpy(target[train_m]),
                          torch.from_numpy(mask[train_m])),
            batch_size=args.batch, shuffle=True,
            generator=torch.Generator().manual_seed(C.SEED))

        model = FieldDownscaler(n_coarse=cs.shape[1], n_fine_static=fs.shape[1],
                                fine_hw=hw, width=args.width,
                                dropout=C.DROPOUT, residual=True).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=C.WEIGHT_DECAY)

        def infer(sel):
            model.eval()
            out = np.empty((sel.sum(), hw, hw), np.float32)
            with torch.no_grad():
                cc, ff, tt = cs[sel], fs[sel], tp[sel]
                for i in range(0, len(out), 128):
                    out[i:i+128] = model(
                        torch.from_numpy(cc[i:i+128]).to(dev),
                        torch.from_numpy(ff[i:i+128]).to(dev),
                        torch.from_numpy(tt[i:i+128]).to(dev)).cpu().numpy()
            return out

        best, best_state, bad = -np.inf, None, 0
        for ep in range(args.epochs):
            model.train()
            for cb, fb, tb, yb, mb in dl:
                cb, fb, tb, yb, mb = (x.to(dev) for x in (cb, fb, tb, yb, mb))
                opt.zero_grad()
                loss, _ = masked_field_loss(model(cb, fb, tb), yb, mb, thr_t,
                                            a=C.LOSS_SIGMOID_SHARPNESS,
                                            b=C.LOSS_TS_WEIGHT)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP_NORM)
                opt.step()
            p = infer(val_m)
            sel = mask[val_m] & cat
            score = r2(p[sel], target[val_m][sel])
            if score > best + C.EARLY_STOP_MIN_DELTA:
                best, bad = score, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break

        model.load_state_dict(best_state)
        p = infer(test_m)
        oof[test_m] = p
        sel = mask[test_m] & cat
        score = r2(p[sel], target[test_m][sel])
        rows.append({"year": Y, "v2": score, "v1": V1_FOLDS.get(Y, float('nan')),
                     "delta": score - V1_FOLDS.get(Y, float('nan')), "epochs": ep + 1,
                     "seconds": round(time.time() - t0, 1)})
        print(f"  {Y}: v2 {score:+.4f}   v1 {V1_FOLDS.get(Y, float('nan')):+.4f}   "
              f"delta {score - V1_FOLDS.get(Y, float('nan')):+.4f}   ({ep+1} ep, {time.time()-t0:.0f}s)")

    d = np.array([r["delta"] for r in rows])
    v2 = np.array([r["v2"] for r in rows])
    v1 = np.array([r["v1"] for r in rows])
    se = d.std(ddof=1) / np.sqrt(len(d))
    summary = {
        "v2_median": float(np.median(v2)), "v2_mean": float(v2.mean()),
        "v1_median": float(np.median(v1)), "v1_mean": float(v1.mean()),
        "delta_median": float(np.median(d)), "delta_mean": float(d.mean()),
        "delta_se": float(se), "t": float(d.mean() / se) if se > 0 else float("nan"),
        "v2_wins": int((d > 0).sum()), "n_folds": len(d),
    }
    print(f"\n=== v2 vs v1 field downscaler, {len(d)} monsoons, leads 1-{args.max_lead} ===")
    print(f"  v2     median {summary['v2_median']:+.4f}  mean {summary['v2_mean']:+.4f}")
    print(f"  v1     median {summary['v1_median']:+.4f}  mean {summary['v1_mean']:+.4f}")
    print(f"  delta  median {summary['delta_median']:+.4f}  mean {summary['delta_mean']:+.4f}"
          f"  v2 wins {summary['v2_wins']}/{len(d)}")
    print(f"  paired mean delta {d.mean():+.4f} +/- {se:.4f} (SE), t = {summary['t']:+.2f}")

    C.METRICS.mkdir(parents=True, exist_ok=True)
    out = C.METRICS / "loyo_field_v2.json"
    out.write_text(json.dumps({"config": vars(args), "folds": rows,
                               "summary": summary}, indent=2, default=str))
    print(f"\nwrote {out}")
    np.savez_compressed(C.PROCESSED / "loyo_oof_rainfall_v2.npz",
                        oof=oof, valid=valid.values, lead=lead,
                        init_year=init_year, catchment=catchment, mask=mask)
    print(f"wrote {C.PROCESSED/'loyo_oof_rainfall_v2.npz'}  <- rainfall handoff to the inflow stage")


if __name__ == "__main__":
    main()
