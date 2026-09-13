"""LSTM post-processing of VIC forecast inflow -- the XAJ-LSTM role, with VIC.

Dong et al. (2025) describe their LSTM as "a post-processing model of the XAJ
model, similar to the CNN model as a post-processing model of the ECMWF forecast
model."  This is that, with VIC as the physical model.

WHY POST-PROCESSING RATHER THAN REPLACEMENT.  Measured on this basin, VIC driven
by observed ERA5 forcing has excellent timing and a badly biased volume:

    JJAS 2003-2014:  corr +0.883   NSE +0.192   bias +57.6 %

Dynamics that good with a bias that large is exactly the case a learned
correction fixes.  The LSTM never has to discover when the basin responds -- VIC
already knows -- only how much water actually arrives.

LEAKAGE RULE.  Antecedent observed inflow is indexed by INITIALISATION date, so
the model sees the hydrograph up to the moment the forecast is issued and
nothing after.  Indexing by valid date would let it read the answer, and would
produce a spectacular NSE that means nothing.  Folds are by initialisation year,
matching every other stage of this project.

RAINFALL FEATURE, decided 2026-09-06.  The LSTM's future-rainfall channel takes
the rainfall the forecast was ACTUALLY driven by -- raw EC for --product ec,
quantile-mapped for ec_qm, the CNN's out-of-fold downscaled field for ec_cnn.

It used to take raw EC for every product.  That made `EC-CNN -> VIC -> RVIC ->
LSTM` a control contrast rather than the Dong-style chain its name claimed: the
LSTM was correcting a VIC run driven by CNN rainfall while being shown rainfall
that run had never seen.  The two feature series are genuinely different
(catchment-mean r = 0.765; 8.79 vs 8.33 mm/day), so this is not cosmetic.

Rows are matched on (init, lead), never by position -- percell_v2_oof.npz and
ec_qm_v2_full.npz come from different scripts and share no row-order guarantee.

Run:  python inflow/lstm_postproc.py --product ec [--epochs 60]
Out:  results/metrics/lstm_postproc_<product>.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C
from cnn.train_field import device, set_seed

# Folds come from the data, not from config: C.YEAR_MAX is still 2014 and is
# read by 22 modules, so widen here rather than move it under all of them.
_FCF = next((f for f in (C.PROCESSED / "vic_inflow_ec_cnn.npz",
                         C.PROCESSED / "vic_inflow_ec.npz") if f.exists()), None)
YEARS = (sorted(set(pd.to_datetime(np.load(_FCF, allow_pickle=True)["inits"]).year))
         if _FCF is not None else list(range(C.YEAR_MIN, C.YEAR_MAX + 1)))
# n_seq -- how many days of observed hydrograph the encoder reads.
#
# Dong et al. call this "an optimized hyperparameter" and note it is "typically
# set to a larger value" in SNOW-AFFECTED basins, where accumulation and melt
# "can span hundreds of days".  The Mahanadi is monsoon-driven, not snow-fed,
# so 60 days was borrowed from a setting that does not apply here.
#
# Measured on this basin, observed inflow autocorrelation is +0.484 at 7 days,
# +0.217 at 30 days and -0.013 at 60.  The last 30 days of a 60-day window carry
# no information at all -- and they cost 121 of 532 initialisations, because a
# forecast needs that much continuous history to be usable.  From 2015 the
# record starts in June, so the discarded windows are disproportionately
# early-monsoon: the onset, which operators most need.
#
#   n_seq   usable inits      autocorrelation at that lag
#      7    495 (93.0 %)      +0.484
#     15    476 (89.5 %)      ~+0.35
#     30    432 (81.2 %)      +0.217
#     60    311 (58.5 %)      -0.013
#
# SWEPT 2026-09-07 on the common 309 windows, ensemble mean:
#
#   n_seq    median NSE   mean    worst yr   trained on
#      7        0.192     0.036     -1.10    495 inits
#     15        0.200     0.040     -1.05    476 inits   <- default
#     30        0.146    -0.026     -2.24    432 inits
#     45        0.177    -0.119     -2.65    376 inits
#     60        0.140     0.034        --    311 inits   <- previous default
#
# HONEST READING.  Short history clearly beats long: 7 and 15 both comfortably
# exceed 45 and 60.  Choosing 15 over 7 is NOT significant (mean diff +0.004,
# p = 0.59, 9/19 wins) -- the spread among short windows is driven by a couple of
# bad years, chiefly 2009 and 2017.  Report this as "short history, ~7-15 days",
# not as an optimum at 15.
#
# Two effects, both favourable: 53 % more training initialisations (311 -> 476),
# and the removal of inputs carrying no information.
PAST = 15
LEAD = C.V2_LEAD_MAX


class Postproc(nn.Module):
    """Encoder over the past hydrograph, decoder over the forecast horizon.

    The decoder emits a multiplicative correction to VIC rather than the inflow
    itself: VIC's shape is already right, so learning log(obs) - log(vic) is a
    smaller, better-conditioned target than learning the hydrograph from
    scratch, and it keeps the physics visible in the output.
    """

    def __init__(self, n_past_feat, n_fut_feat, hidden=64, layers=1, dropout=0.2):
        super().__init__()
        self.enc = nn.LSTM(n_past_feat, hidden, layers, batch_first=True)
        self.dec = nn.LSTM(n_fut_feat, hidden, layers, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 32),
                                  nn.ELU(), nn.Linear(32, 1))

    def forward(self, past, fut, vic_log):
        _, (h, c) = self.enc(past)
        out, _ = self.dec(fut, (h, c))
        delta = self.head(out).squeeze(-1)          # log-space correction
        return vic_log + delta


def nse(p, o):
    return float(1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum())


def build(product, n_seq=PAST):
    z = np.load(C.PROCESSED / f"vic_inflow_{product}.npz", allow_pickle=True)
    q = z["q"]                                       # (n_init, n_mem, 30) m3/s
    inits = pd.to_datetime([str(x) for x in z["inits"]])

    obs = pd.read_parquet(next(
        (f for f in (C.PROCESSED / "inflow_daily_extended.parquet",
                     C.PROCESSED / "inflow_daily.parquet") if f.exists()),
        C.PROCESSED / "inflow_daily.parquet"))
    obs = obs[obs["inflow_valid"]].set_index("date")["inflow"]

    ec = np.load(next((f for f in (C.PROCESSED / "ec_qm_v2_full.npz",
                                   C.PROCESSED / "ec_qm_v2.npz") if f.exists()),
                      C.PROCESSED / "ec_qm_v2.npz"), allow_pickle=True)
    # Prefer the STORED init over re-deriving it from `valid - lead`.  This line
    # is where the 2026-09 date-label bug entered: quantile_mapping.py was saving
    # `valid` from un-reordered arrays, so every row here got another date's
    # rainfall and the LSTM trained on scrambled weather.  `init` is written by
    # the same statement that writes the data, so it cannot drift from it.
    # The fallback keeps older archives readable; the assertion catches the bug
    # class rather than this one instance of it.
    _init = (pd.to_datetime([str(x) for x in ec["init"]]) if "init" in ec.files
             else pd.to_datetime(ec["valid"]) -
                  pd.to_timedelta(ec["lead"], unit="D"))
    if "init" in ec.files and "valid" in ec.files:
        _derived = pd.to_datetime(ec["valid"]) - pd.to_timedelta(ec["lead"], unit="D")
        if not (_derived == _init).all():
            sys.exit("ec archive labels are inconsistent: `init` disagrees with "
                     "`valid - lead`.  Run tools/repair_ec_qm_labels.py.")
    key = pd.DataFrame({"init": _init, "lead": ec["lead"]})
    # RESOLVED 2026-09-06.  This used to be raw EC catchment-mean precipitation
    # for EVERY product, including --product ec_cnn, so the "EC-CNN -> VIC ->
    # RVIC -> LSTM" arm was corrected using rainfall its VIC run had never seen.
    # That made it a control contrast, not the Dong-style chain it was labelled
    # as.  The LSTM now receives the rainfall the forecast was ACTUALLY driven
    # by, so the arm means what its name says.
    #
    # Rows are matched on (init, lead), never by position: percell_v2_oof.npz
    # and ec_qm_v2_full.npz are written by different scripts and there is no
    # guarantee they share a row order.  Assuming they did is precisely the
    # class of bug that scrambled this project's dates once already.
    if product == "ec_cnn":
        _o = np.load(C.PROCESSED / "percell_v2_oof.npz", allow_pickle=True)
        _ok = pd.MultiIndex.from_arrays(
            [pd.to_datetime(_o["valid"]) - pd.to_timedelta(_o["lead"], unit="D"),
             _o["lead"].astype(int)])
        # (rows, 224 cells) -- already the member mean, matching how the routed
        # ensemble mean is scored downstream.
        _src = pd.Series(_o["oof"].mean(1), index=_ok)
    else:
        _src = pd.Series(ec["ec" if product == "ec" else "ec_qm"].mean(1).mean(1),
                         index=pd.MultiIndex.from_arrays([_init, ec["lead"].astype(int)]))
    _want = pd.MultiIndex.from_arrays([key["init"], key["lead"].astype(int)])
    if _src.index.duplicated().any():
        sys.exit("rainfall source has duplicate (init, lead) keys")
    _missing = (~_want.isin(_src.index)).sum()
    if _missing:
        sys.exit(f"{_missing} of {len(_want)} (init, lead) keys have no "
                 f"{product} rainfall -- refusing to train on a misaligned feature")
    prec_all = _src.reindex(_want).to_numpy()
    print(f"  rainfall feature: {product}  "
          f"(mean {np.nanmean(prec_all):.2f} mm/day over {len(prec_all):,} rows)")

    past, fut, vic, targ, ok, iy, row_init = [], [], [], [], [], [], []
    for ki, init in enumerate(inits):
        hist = pd.date_range(init - pd.Timedelta(days=n_seq - 1), init)
        if not all(d in obs.index for d in hist):
            continue
        h = np.log1p(obs.reindex(hist).values).astype(np.float32)
        days = pd.date_range(init + pd.Timedelta(days=1), periods=LEAD)
        y = obs.reindex(days).values.astype(np.float32)
        sel = (key["init"] == init).values
        order = np.argsort(key["lead"].values[sel])
        p = prec_all[sel][order].astype(np.float32)
        doy = days.dayofyear.values
        f = np.stack([p / 20.0,
                      np.arange(1, LEAD + 1) / LEAD,
                      np.sin(2 * np.pi * doy / 366), np.cos(2 * np.pi * doy / 366)], 1)
        for mi in range(q.shape[1]):
            past.append(h[:, None]); vic.append(np.log1p(q[ki, mi]))
            fut.append(f); targ.append(y); ok.append(np.isfinite(y))
            iy.append(init.year); row_init.append(init)
    print(f"  n_seq {n_seq} d -> {len(set(row_init))} of {len(inits)} initialisations usable")
    return (np.array(past, np.float32), np.array(fut, np.float32),
            np.array(vic, np.float32), np.nan_to_num(np.array(targ, np.float32)),
            np.array(ok), np.array(iy), q.shape[1],
            np.array([str(d.date()) for d in row_init]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", choices=["ec", "ec_qm", "ec_cnn"], default="ec")
    ap.add_argument("--past", type=int, default=PAST,
                    help="n_seq: days of observed hydrograph the encoder reads. "
                         "Dong et al. treat this as an optimised hyperparameter; "
                         "see the sweep table above the PAST constant")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--val-rotate", action="store_true",
                    help="rotate the inner validation year instead of always "
                         "using the chronologically last one. OFF by default so "
                         "published results stay reproducible -- turning it on "
                         "requires a full re-run.")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    set_seed(C.SEED); dev = device()
    past, fut, vic, targ, ok, iy, n_mem, row_init = build(args.product, n_seq=args.past)
    print(f"device={dev}  samples {past.shape[0]} "
          f"({past.shape[0]//n_mem} inits x {n_mem} members)")
    print(f"  past {past.shape}  fut {fut.shape}  target {targ.shape}\n")

    rows, pred_all = [], np.zeros_like(targ)
    for Y in YEARS:
        te = iy == Y
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
        va = iy == inner
        tr = ~te & ~va
        t = lambda a, m: torch.from_numpy(a[m]).to(dev)
        model = Postproc(past.shape[2], fut.shape[2]).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=C.WEIGHT_DECAY)
        best, best_state, bad = -np.inf, None, 0
        idx = np.where(tr)[0]
        for ep in range(args.epochs):
            model.train()
            np.random.shuffle(idx)
            for s in range(0, len(idx), args.batch):
                b = idx[s:s + args.batch]
                p = model(t(past, b), t(fut, b), t(vic, b))
                m = torch.from_numpy(ok[b]).to(dev)
                loss = (((p - torch.from_numpy(np.log1p(targ[b])).to(dev)) ** 2)
                        * m).sum() / m.sum().clamp(min=1)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP_NORM)
                opt.step()
            model.eval()
            with torch.no_grad():
                pv = np.expm1(model(t(past, va), t(fut, va), t(vic, va)).cpu().numpy())
            s_ = nse(pv[ok[va]], targ[va][ok[va]])
            if s_ > best + C.EARLY_STOP_MIN_DELTA:
                best, bad = s_, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            pt = np.expm1(model(t(past, te), t(fut, te), t(vic, te)).cpu().numpy())
        pred_all[te] = pt
        m = ok[te]
        raw = np.expm1(vic[te])
        rows.append({"year": Y,
                     "vic_nse": nse(raw[m], targ[te][m]),
                     "lstm_nse": nse(pt[m], targ[te][m]),
                     "vic_bias_pct": float(100 * (raw[m].mean() - targ[te][m].mean())
                                           / targ[te][m].mean()),
                     "lstm_bias_pct": float(100 * (pt[m].mean() - targ[te][m].mean())
                                            / targ[te][m].mean()),
                     "epochs": ep + 1})
        print(f"  {Y}: VIC NSE {rows[-1]['vic_nse']:+.3f} -> VIC-LSTM {rows[-1]['lstm_nse']:+.3f}"
              f"   bias {rows[-1]['vic_bias_pct']:+.0f}% -> {rows[-1]['lstm_bias_pct']:+.0f}%"
              f"   ({ep+1} ep)", flush=True)

    v = np.array([r["vic_nse"] for r in rows]); l = np.array([r["lstm_nse"] for r in rows])
    d = l - v; se = d.std(ddof=1) / np.sqrt(len(d))
    summ = {"vic_median": float(np.median(v)), "lstm_median": float(np.median(l)),
            "delta_mean": float(d.mean()), "delta_se": float(se),
            "t": float(d.mean() / se), "wins": int((d > 0).sum()), "n": len(d)}
    print(f"\n=== VIC vs VIC-LSTM ({args.product}), {len(rows)} monsoons ===")
    print(f"  VIC       median NSE {summ['vic_median']:+.4f}")
    print(f"  VIC-LSTM  median NSE {summ['lstm_median']:+.4f}")
    print(f"  delta {d.mean():+.4f} +/- {se:.4f} (SE)  t={summ['t']:+.2f}  "
          f"wins {summ['wins']}/{len(d)}")
    out = C.METRICS / f"lstm_postproc_{args.product}.json"
    out.write_text(json.dumps({"config": vars(args), "folds": rows,
                               "summary": summ}, indent=2, default=str))
    print(f"\nwrote {out}")
    # `init` is saved so consumers match rows by DATE.  Row POSITION depends on
    # n_seq -- a shorter history admits more initialisations -- so positional
    # indexing across files silently misaligns the moment n_seq changes.  That
    # is the same class of defect as the 2026-09 date-label bug.
    np.savez_compressed(C.PROCESSED / f"lstm_inflow_{args.product}.npz",
                        pred=pred_all, target=targ, mask=ok, init_year=iy,
                        init=row_init, n_seq=args.past)


if __name__ == "__main__":
    main()
