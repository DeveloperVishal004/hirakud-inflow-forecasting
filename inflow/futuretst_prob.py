"""Ensemble-aware, probabilistic FutureTST.

    ECMWF ensemble -> QM -> FutureTST(mean, spread) -> P10/P25/P50/P75/P90

Three changes from the deterministic futuretst_v3.py (archived 2026-09-07 to
archive/superseded_20260907/), and nothing else, so the comparison against
that run's QM + fine-tuned variant (median NSE 0.241) isolates their effect:

  ENSEMBLE   the future precipitation channel becomes (mean, spread) instead of
             the mean alone.  Spread is not decoration: over the 309 shared
             windows, Spearman(spread, |inflow error|) is +0.475 overall and
             +0.531 at leads 8-14, and windows in the lowest spread quartile
             average 865 m3/s error against 2,462 for the highest.  The
             ensemble already knows when it does not know.

  QUANTILES  five outputs per lead, trained with pinball loss, instead of one
             point.  Beyond day 3 the error is 70-90 % phase -- a point forecast
             cannot be right, and NSE actively rewards damping the forecast
             toward the mean (the CNN chain reached NSE parity while
             under-forecasting peaks by 36 %).  A proper scoring rule does not:
             a wide interval when genuinely uncertain is correct, not hedging.

  SCORING    CRPS, 80/90 % coverage and mean interval width alongside NSE and
             RMSE, all reported per lead band.

PAST_LEN stays at 64 deliberately.  The point of this run is to isolate
ensemble information and probabilistic output; changing the history length at
the same time would confound the two.  (A separate diagnostic shows 7 days of
inflow history predicts as well as 64 -- autocorrelation is 0.48 at lag 7 and
-0.01 at lag 60 -- so shortening it is a compute optimisation for later, not a
skill question.)

Run:  python inflow/futuretst_prob.py
Out:  results/metrics/futuretst_prob.json
"""

import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C
from inflow.train_inflow import nse, kge
from inflow.loyo_cv import build_inflow_frame, YEARS
from preprocessing import bias_correction as BC

# Stage 2 trained on CPU even with a GPU attached until this was added: nothing
# was ever moved onto the device.  On Kaggle's T4 that cost ~20-40 min per fold.
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAST_LEN = 64
PAST_PATCH = 8         # past halves are patched: 64 days -> 8 tokens
N_PAST_TOK = PAST_LEN // PAST_PATCH
HORIZON = 30           # was 17: the archive now reaches lead 30, which is the
                       # horizon Ambika et al. (2025) report (1, 7, 14, 30 d)
# Ambika et al. (2025) Sect. 2.3: "precipitation, shortwave radiation, maximum
# air temperature, minimum air temperature, and water vapor pressure".  v2 used
# a single mean air temperature plus wind and surface pressure -- neither of
# which the paper feeds -- and dropped vapour pressure entirely.  The diurnal
# range (tmax - tmin) is the part that matters most: it proxies cloudiness and
# evaporative demand, both of which a daily mean discards.
MET = ["prec", "swdown", "air_temp_max", "air_temp_min", "vp"]

# Ablation switch: see daily_frames().  Declared here with the other
# constants because daily_frames reads it.
DEGRADE_MET = "--degrade-met" in sys.argv

# WHICH RAINFALL FORECAST FEEDS THE FUTURE BRANCH.
#
# Ambika et al. is implemented INDEPENDENTLY of the Dong et al. pipeline, so the
# default is RAW ECMWF.  Their case is precisely that FutureTST "reduces the
# cumbersome need for bias correction and high-resolution downscaling", and they
# drive it with raw ECMWF/TIGGE forecasts.  Handing it the CNN-downscaled
# rainfall would make pipeline 2 inherit pipeline 1's contribution and then be
# scored against it -- the two would no longer be independent implementations.
#
#   --rainfall raw   (default) catchment-mean raw ECMWF, ec_qm_v2*.npz
#   --rainfall cnn             the stage-1 CNN handoff; a deliberate HYBRID,
#                              reported separately, never as "paper 2"
RAIN_SOURCE = "cnn" if "--rainfall" in sys.argv and \
    sys.argv[sys.argv.index("--rainfall") + 1] == "cnn" else "raw"
RAINFALL = next(p for p in (C.PROCESSED / "loyo_oof_rainfall_v2.parquet",
                            C.PROCESSED / "loyo_oof_rainfall_cnn.parquet",
                            C.PROCESSED / "loyo_oof_rainfall.parquet") if p.exists())


def raw_ec_catchment_prec():
    """Catchment-mean raw ECMWF precipitation per (init, lead), mm/day."""
    # The catchment mean is 100 kB; ec_qm_v2_full.npz is 200 MB.  Ship the
    # parquet so a Kaggle payload does not carry the whole per-cell array.
    cached = C.PROCESSED / "raw_ec_catchment_prec.parquet"
    if cached.exists():
        df = pd.read_parquet(cached)
        df["init_date"] = pd.to_datetime(df["init_date"])
        return df
    f = next((q for q in (C.PROCESSED / "ec_qm_v2_full.npz",
                          C.PROCESSED / "ec_qm_v2.npz") if q.exists()))
    z = np.load(f, allow_pickle=True)
    prec = z["ec"].mean(1).mean(1)                    # members, then cells
    valid = pd.to_datetime(z["valid"])
    df = pd.DataFrame({"init_date": valid - pd.to_timedelta(z["lead"], unit="D"),
                       "lead_day": z["lead"].astype(int), "prec": prec})
    df.to_parquet(cached, index=False)
    return df
D_MODEL = 64
N_HEADS = 4
N_ENC = 2
N_DEC = 2
DROPOUT = 0.15
LR = 1e-3
WEIGHT_DECAY = 1e-2
MAX_EPOCHS = 120
PATIENCE = 15
BATCH = 128
N_SEEDS = 3
# P05/P95 are added to the requested set so 90 % coverage is measurable
# directly; P10-P90 alone gives 80 % and nothing wider.  50 % (P25-P75) comes
# free and is the most sensitive calibration check of the three.
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
N_Q = len(QUANTILES)
MEDIAN_Q = QUANTILES.index(0.50)
Q_IDX = {q: i for i, q in enumerate(QUANTILES)}
INTERVALS = {"50%": (0.25, 0.75), "80%": (0.10, 0.90), "90%": (0.05, 0.95)}
# Fine-tuning on forecast meteorology: low learning rate and few epochs, because
# the point is to recalibrate how far the model trusts its forcing, not to
# relearn the rainfall-runoff response on 279 sequences.
FT_LR = 1e-4
FT_EPOCHS = 60
FT_PATIENCE = 10
FT_BATCH = 32

# Fast end-to-end check: FTST_FAST=1 shrinks epochs and seeds so every branch --
# QM, noise-aug, fine-tuning and the entire scoring block -- executes in a few
# minutes.  Unset, it changes nothing, so real runs are unaffected.  The point
# is to exercise the code paths, not to produce meaningful skill.
FAST_RUN = bool(os.environ.get("FTST_FAST"))
if FAST_RUN:
    MAX_EPOCHS, PATIENCE, N_SEEDS = 2, 1, 1
    FT_EPOCHS, FT_PATIENCE = 2, 1
    print("FTST_FAST: 2 epochs, 1 seed -- code-path check only, numbers are junk")
    print("           output suffixed '_FAST' so it cannot overwrite real results")

torch.set_num_threads(8)


# --------------------------------------------------------------------------- data


def daily_frames():
    q = pd.read_parquet(next(
        (f for f in (C.PROCESSED / "inflow_daily_extended.parquet",
                     C.PROCESSED / "inflow_daily.parquet") if f.exists()),
        C.PROCESSED / "inflow_daily.parquet")).set_index("date").sort_index()
    inflow = q["inflow"].where(q["inflow_valid"])
    full = pd.date_range(inflow.index.min(), inflow.index.max(), freq="D")
    inflow = inflow.reindex(full)

    met = pd.read_parquet(C.PROCESSED / "daily_catchment_met.parquet").set_index("date")
    met = met.reindex(full).interpolate(limit_direction="both")
    # IMD is the rainfall of record; ERA5-Land precipitation is only a fallback
    # where IMD has no value, so the observed-met branch and the rest of the
    # project agree on what "observed rainfall" means.
    imd = pd.read_parquet(C.PROCESSED / "daily_catchment_obs.parquet")
    imd = imd.set_index("date")["rain_obs_daily"].reindex(full)
    met["prec"] = imd.fillna(met["prec"])

    # --degrade-met: replace tmax, tmin and vapour pressure with their
    # day-of-year climatology, reproducing on the OBSERVED branch exactly the
    # handicap the forecast branch operates under (those three were never
    # downloaded as forecasts).  The gap between this run and the normal one is
    # what a supplementary ECDS download would actually buy.
    if DEGRADE_MET:
        clim_vars = [v for v in ("air_temp_max", "air_temp_min", "vp") if v in met.columns]
        doy = met.index.dayofyear
        for v in clim_vars:
            met[v] = doy.map(met.groupby(doy)[v].mean())
        print(f"  DEGRADED: {clim_vars} replaced by day-of-year climatology")
    # Sixth channel to match the forecast branch.  Observed weather has no
    # ensemble, so its spread is exactly zero -- which is also the honest
    # signal: on the past half there is nothing to be uncertain about.
    out = met[MET].copy()
    out["prec_spread"] = 0.0
    return inflow, out, full


def build_daily_sequences():
    """Every day that has a full past window and a full horizon: ~3,900."""
    inflow, met, full = daily_frames()
    q_ok = inflow.notna().to_numpy(np.float32)
    q_fill = inflow.interpolate(limit_direction="both").to_numpy(np.float32)
    M = met.to_numpy(np.float32)
    n = len(full)

    idx = np.arange(PAST_LEN, n - HORIZON)
    past_off = np.arange(-PAST_LEN, 0)
    fut_off = np.arange(0, HORIZON)
    p_ix = idx[:, None] + past_off[None, :]
    f_ix = idx[:, None] + fut_off[None, :]

    past_flow = np.stack([q_fill[p_ix], q_ok[p_ix]], -1)      # (n, L, 2)
    met_seq = np.concatenate([M[p_ix], M[f_ix]], 1)           # (n, L+H, 5)
    y = q_fill[f_ix]
    mask = q_ok[f_ix].astype(bool)
    init = full[idx]                                          # forecast issued on this day
    return past_flow, met_seq, y, mask, init, full.to_numpy()[f_ix]


def forecast_met_seq(inits_needed, past_met_lookup):
    """Replace the future half of the met sequence with the S2S forecast."""
    # v2 forecast met: leads 1-30, with stage-1 CNN rainfall already merged in as
    # `prec`.  The v1 file stopped at lead 17 and cannot serve HORIZON = 30.
    v2 = C.PROCESSED / "forecast_catchment_met_v2.parquet"
    if v2.exists():
        fm = pd.read_parquet(v2)
        fm["init_date"] = pd.to_datetime(fm["init_date"])
        fm = fm.sort_values(["init_date", "lead_day"])
    else:
        fm = pd.read_parquet(C.PROCESSED / "forecast_catchment_met.parquet")
        rain = pd.read_parquet(RAINFALL)
        fm["init_date"] = pd.to_datetime(fm["init_date"])
        rain["init_date"] = pd.to_datetime(rain["init_date"])
        fm = fm.merge(rain[["init_date", "lead_day", "rain_pred"]],
                      on=["init_date", "lead_day"], how="inner")
        fm = fm.rename(columns={"rain_pred": "prec"}).sort_values(
            ["init_date", "lead_day"])
    fm = fm[fm["lead_day"] <= HORIZON]

    # Independent implementation: swap the CNN prec for raw ECMWF unless the
    # hybrid was explicitly asked for.  Everything else in the row is unchanged.
    if RAIN_SOURCE == "raw":
        ens = pd.read_parquet(C.PROCESSED / "ec_ensemble_catchment.parquet")
        ens["init_date"] = pd.to_datetime(ens["init_date"])
        # QM'd ensemble: the mean is the forcing, the spread is the model's
        # cue for how far to trust it.  Both go in as channels.
        # RAW ensemble mean, not the QM'd one.  The QM branch below applies
        # quantile mapping itself, exactly as the archived futuretst_v3 did; feeding it a
        # pre-mapped series would double-correct and make the headline branch
        # incomparable with that run's 0.241.  Spread comes from the raw
        # members for the same reason -- it must describe the forcing the model
        # is actually given.
        ens = ens[["init_date", "lead_day", "prec_raw_mean", "prec_raw_spread"]]
        fm = fm.drop(columns=["prec"]).merge(ens, on=["init_date", "lead_day"],
                                             how="inner")
        fm = fm.rename(columns={"prec_raw_mean": "prec",
                                "prec_raw_spread": "prec_spread"})
        print(f"  precipitation: RAW ECMWF ENSEMBLE mean + spread "
              f"({len(fm):,} rows) -- QM applied downstream, as in v3")
    else:
        print("  precipitation: stage-1 CNN handoff (HYBRID, not paper 2)")

    # The S2S archive supplies daily-mean 2 m temperature but no daily max/min,
    # and no vapour pressure -- neither was requested in the download.  Rather
    # than silently substituting a different variable, fill those channels with
    # day-of-year climatology from the observed record and say so.  The model
    # therefore sees a real precipitation and radiation FORECAST with
    # climatological temperature range and humidity, which is a weaker forcing
    # than the paper's but an honest one.
    obs_met = pd.read_parquet(C.PROCESSED / "daily_catchment_met.parquet")
    obs_met["date"] = pd.to_datetime(obs_met["date"])
    missing = [c for c in MET if c not in fm.columns]
    if missing:
        print(f"  forecast meteorology has no {missing}; filling from day-of-year "
              f"climatology of the observed record", flush=True)
        clim = obs_met.assign(doy=obs_met["date"].dt.dayofyear).groupby("doy")[
            [c for c in missing]].mean()
        fm["_doy"] = pd.to_datetime(fm["valid_date"]).dt.dayofyear
        for c in missing:
            fm[c] = fm["_doy"].map(clim[c])
    cols = MET + ["prec_spread"]
    if "prec_spread" not in fm.columns:      # CNN hybrid path has no ensemble
        fm["prec_spread"] = 0.0
    piv = {c: fm.pivot(index="init_date", columns="lead_day", values=c) for c in cols}
    inits = pd.DatetimeIndex(piv["prec"].index)
    arr = np.stack([piv[c][sorted(piv[c].columns)].to_numpy(np.float32) for c in cols], -1)
    return inits, arr                                          # (n_init, H, 6)


# -------------------------------------------------------------------------- model


class GME(nn.Module):
    """Global Meteorological Embedding: the encoder half of Figure 1."""

    def __init__(self, n_met: int, n_tok: int):
        super().__init__()
        # Past meteorology is patched (8 days per token), future meteorology is
        # not: the horizon is only HORIZON (=30) steps and each lead needs its
        # own token to be queried individually by the decoder.
        self.past_proj = nn.Linear(PAST_PATCH * n_met, D_MODEL)
        self.fut_proj = nn.Linear(n_met, D_MODEL)
        self.pos = nn.Parameter(torch.zeros(1, n_tok, D_MODEL))
        # A past/future type embedding: forecast meteorology carries error that
        # observed meteorology does not, and the encoder should be able to
        # discount it rather than treat the two as one homogeneous series.
        self.typ = nn.Parameter(torch.zeros(1, 2, D_MODEL))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.typ, std=0.02)
        layer = nn.TransformerEncoderLayer(
            D_MODEL, N_HEADS, dim_feedforward=2 * D_MODEL, dropout=DROPOUT,
            batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, N_ENC, enable_nested_tensor=False)

    def forward(self, met):
        b = met.shape[0]
        p = self.past_proj(met[:, :PAST_LEN].reshape(b, N_PAST_TOK, -1)) + self.typ[:, 0:1]
        f = self.fut_proj(met[:, PAST_LEN:]) + self.typ[:, 1:2]
        return self.enc(torch.cat([p, f], 1) + self.pos)


class FutureTSTv2(nn.Module):
    def __init__(self, n_met: int, n_flow: int):
        super().__init__()
        self.gme = GME(n_met, N_PAST_TOK + HORIZON)
        self.flow_proj = nn.Linear(PAST_PATCH * n_flow, D_MODEL)
        self.flow_pos = nn.Parameter(torch.zeros(1, N_PAST_TOK, D_MODEL))
        # Learned horizon queries: one per lead day.  These are what the decoder
        # uses to interrogate the meteorological memory -- lead k asks its own
        # question rather than sharing a pooled representation.
        self.query = nn.Parameter(torch.zeros(1, HORIZON, D_MODEL))
        nn.init.normal_(self.flow_pos, std=0.02)
        nn.init.normal_(self.query, std=0.02)
        layer = nn.TransformerDecoderLayer(
            D_MODEL, N_HEADS, dim_feedforward=2 * D_MODEL, dropout=DROPOUT,
            batch_first=True, norm_first=True, activation="gelu")
        self.dec = nn.TransformerDecoder(layer, N_DEC)
        self.norm = nn.LayerNorm(D_MODEL)
        # One output per quantile.  Sorting is enforced at inference, not here:
        # the pinball loss is minimised by ordered quantiles anyway, and forcing
        # monotonicity through a cumulative-softplus during training slows
        # convergence for no gain at these horizons.
        self.head = nn.Linear(D_MODEL, N_Q)

    def forward(self, past_flow, met):
        memory = self.gme(met)
        f = self.flow_proj(past_flow.reshape(past_flow.shape[0], N_PAST_TOK, -1)) + self.flow_pos
        tgt = torch.cat([f, self.query.expand(f.shape[0], -1, -1)], 1)
        out = self.norm(self.dec(tgt, memory))
        return self.head(out[:, N_PAST_TOK:])          # (B, HORIZON, N_Q)


def degrade(d: dict, rho, seed: int = 0) -> dict:
    """A copy of `d` with the future rainfall channel degraded to correlation `rho`.

    Used for the validation set so early stopping selects a model that is good
    on noisy forcing -- which is what deployment supplies -- rather than on the
    clean observed forcing it is trained from.  Fixed seed: the validation
    target must not move between epochs.
    """
    out = dict(d)
    met = d["met"].clone()
    fut = met[:, PAST_LEN:, 0]
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(fut.shape, generator=g).to(met.device)
    met[:, PAST_LEN:, 0] = rho * fut + torch.sqrt(1 - rho ** 2) * z
    out["met"] = met
    return out


_QT = None


def pinball(pred, target, mask):
    """Mean pinball (quantile) loss over QUANTILES, masked.

    pred (B, H, N_Q), target (B, H), mask (B, H).  Averaging over quantiles of
    the pinball loss is a proper scoring rule -- unlike MSE it cannot be
    improved by shrinking the forecast toward the mean, which is exactly the
    failure mode NSE rewarded in the deterministic runs.
    """
    global _QT
    if _QT is None or _QT.device != pred.device:
        _QT = torch.tensor(QUANTILES, device=pred.device, dtype=pred.dtype)
    e = target.unsqueeze(-1) - pred                     # (B, H, N_Q)
    l = torch.maximum(_QT * e, (_QT - 1.0) * e).mean(-1)
    return (l * mask).sum() / mask.sum().clamp(min=1)


def fit_one(tr, va, seed: int, model=None, lr=LR, epochs=MAX_EPOCHS,
            patience=PATIENCE, batch=BATCH, rho=None):
    """Train, or continue training a model that is passed in.

    Passing `model` turns this into the fine-tuning pass: the weights carry the
    rainfall-runoff response learned from 3,500 observed-meteorology sequences,
    and the low learning rate lets the forecast sequences adjust how much the
    model leans on its forcing without destroying that response.

    `rho` (length H) turns on lead-dependent noise augmentation.  Trained on
    observed meteorology, the model learns that future rainfall is TRUE and
    reads it at full confidence; at deployment day-15 rainfall correlates only
    ~0.35 with reality, so that confidence converts forcing error into inflow
    error almost one for one -- which is why observed-met skill of +0.615 at
    13-17 d becomes -0.229 on forecast met.  Degrading the future rainfall
    channel to the correlation the real forecast actually achieves at each lead
    teaches the model to discount long leads, and it learns this from all ~3,500
    observed sequences rather than the 341 forecast-forced ones fine-tuning uses.

    For standardised x, `rho * x + sqrt(1 - rho^2) * z` with z ~ N(0, 1) has
    correlation exactly rho with x and unit variance, so only the signal-to-noise
    ratio changes -- not the scale the model has already calibrated to.  Fresh
    noise every epoch, so it is augmentation rather than a fixed corruption.
    """
    torch.manual_seed(seed)
    if model is None:
        model = FutureTSTv2(tr["met"].shape[-1], tr["flow"].shape[-1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(tr["flow"])
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad = np.inf, None, 0

    for ep in range(epochs):
        model.train()
        for idx in torch.randperm(n, generator=g).split(batch):
            opt.zero_grad()
            met_b = tr["met"][idx]
            if rho is not None:
                met_b = met_b.clone()
                fut = met_b[:, PAST_LEN:, 0]          # future precipitation only
                z = torch.randn(fut.shape, generator=g).to(met_b.device)
                met_b[:, PAST_LEN:, 0] = rho * fut + torch.sqrt(1 - rho ** 2) * z
            p = model(tr["flow"][idx], met_b)
            m = tr["mask"][idx]
            loss = pinball(p, tr["y"][idx], m)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            # Early stopping on the SAME loss the model is trained with: MSE on
            # the median would select a model with a well-placed centre and
            # badly calibrated tails, which is the opposite of the point here.
            pv = model(va["flow"], va["met"])
            vl = float(pinball(pv, va["y"], va["mask"]))
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


# --------------------------------------------------------------------------- main


def main() -> None:
    for f in ["daily_catchment_met.parquet", "forecast_catchment_met.parquet"]:
        if not (C.PROCESSED / f).exists():
            sys.exit(f"missing {f}")
    if not RAINFALL.exists():
        sys.exit(f"missing {RAINFALL.name} -- run cnn/loyo_field.py first")
    print(f"rainfall source: {RAIN_SOURCE}"
          + ("  (raw ECMWF -- Ambika et al., independent of the CNN pipeline)"
             if RAIN_SOURCE == "raw" else f"  (CNN hybrid: {RAINFALL.name})"))
    print(f"device: {DEV}" + (f" ({torch.cuda.get_device_name(0)})" if DEV.type == "cuda" else ""), flush=True)

    past_flow, met_seq, y, mask, inits, vdates = build_daily_sequences()
    print(f"daily sequences: {len(inits):,}  "
          f"({inits[0].date()} -> {inits[-1].date()})")
    f_inits, f_met = forecast_met_seq(None, None)
    print(f"S2S initialisations with forecast met: {len(f_inits):,}")

    # Evaluation rows: identical to every other model in the project.
    rain = pd.read_parquet(RAINFALL)
    rain["valid_date"] = pd.to_datetime(rain["valid_date"])
    rain["init_date"] = pd.to_datetime(rain["init_date"])
    ref = build_inflow_frame(rain)[["init_date", "lead_day", "valid_date", "target_inflow"]]
    ref["init_date"] = pd.to_datetime(ref["init_date"])
    print(f"evaluation rows: {len(ref):,}")

    pos = {d: i for i, d in enumerate(inits)}
    keep = np.array([d in pos for d in f_inits])
    f_inits, f_met = f_inits[keep], f_met[keep]
    f_rows = np.array([pos[d] for d in f_inits])

    iyear = inits.year.to_numpy()
    fyear = f_inits.year.to_numpy()
    rows = []

    _smoke = "--smoke" in sys.argv
    for Y in (YEARS[:2] if _smoke else YEARS):
        t0 = time.time()
        tr_years = [v for v in YEARS if v != Y]
        va_year = tr_years[(YEARS.index(Y) + 1) % len(tr_years)]
        tr = (iyear != Y) & (iyear != va_year)
        va = iyear == va_year
        te = fyear == Y
        if te.sum() == 0:
            continue

        fm = met_seq[tr].reshape(-1, met_seq.shape[-1]).mean(0)
        fs = met_seq[tr].reshape(-1, met_seq.shape[-1]).std(0) + 1e-6
        pm = past_flow[tr].reshape(-1, past_flow.shape[-1]).mean(0)
        ps = past_flow[tr].reshape(-1, past_flow.shape[-1]).std(0) + 1e-6
        ym, ys = y[tr][mask[tr]].mean(), y[tr][mask[tr]].std() + 1e-6

        # Module 3, fitted per fold: map each forecast variable's distribution
        # onto the observed one, using training-year rows only.  The observed
        # sample is met_seq[f_rows, PAST_LEN:], i.e. the observed meteorology on
        # exactly the forecast valid dates -- so the mapping cannot absorb a
        # seasonal offset and call it forecast bias.
        f_tr = (fyear != Y) & (fyear != va_year)
        # The ensemble-spread channel has NO observed counterpart -- it is
        # identically zero on the observed record, so the line above gives it
        # mean 0 and std 1e-6.  The QM branch writes raw forecast values into
        # that channel and then divides by fs, which turned a spread of ~5 mm/d
        # into ~5e6 and destroyed the branch (median NSE 0.241 -> -0.087 while
        # every other branch reproduced).  Standardise it with the FORECAST
        # statistics instead, from training-year initialisations only.
        if met_seq.shape[-1] > len(MET):
            sp = f_met[f_tr][..., len(MET)].ravel()
            fm[len(MET)] = float(sp.mean())
            fs[len(MET)] = float(sp.std() + 1e-6)

        # Per-lead correlation the downscaled forecast ACTUALLY achieves, from
        # this fold's training years only -- estimating it on the held-out year
        # would leak the very thing being scored.
        r_tr = rain[(rain["init_date"].dt.year != Y)
                    & (rain["init_date"].dt.year != va_year)]
        rho_np = np.array([np.corrcoef(gg["rain_pred"], gg["rain_obs"])[0, 1]
                           for _, gg in r_tr.groupby("lead_day")], np.float32)
        rho_t = torch.tensor(np.clip(rho_np, 0.05, 0.99)).to(DEV)

        qmaps = []
        for vi, vname in enumerate(MET):
            qmaps.append(BC.fit(f_met[f_tr, :, vi].ravel(),
                                met_seq[f_rows[f_tr], PAST_LEN:, vi].ravel(),
                                precip=(vname == "prec")))

        def qm(o):
            # Only the MET channels have quantile maps.  The 6th channel is the
            # ensemble SPREAD, which must not be mapped onto the observed
            # distribution: it is a measure of forecast disagreement, not a
            # weather variable, and the observed record has no counterpart to
            # map it to (its past-half value is identically zero).  Mapping it
            # would also destroy exactly the signal it was added for.
            o = o.copy()
            for vi in range(min(len(qmaps), o.shape[-1])):
                o[..., vi] = BC.apply(o[..., vi], qmaps[vi])
            return o

        def pack(sel, met_override=None, use_qm=False):
            m_ = met_seq[sel].copy()
            if met_override is not None:
                o = met_override
                if use_qm:
                    # Module 3 proper: after quantile mapping the forecast is on
                    # the OBSERVED distribution, so it is standardised with the
                    # observed statistics like everything else -- no
                    # source-specific rescaling needed.
                    m_[:, PAST_LEN:] = qm(o)
                else:
                    # Stopgap kept for comparison: match the first two moments
                    # only, by standardising with the forecast source's own
                    # statistics.  Isolates what the full distribution mapping
                    # adds over a mean/variance correction.
                    m_[:, PAST_LEN:] = ((o - o.reshape(-1, o.shape[-1]).mean(0))
                                        / (o.reshape(-1, o.shape[-1]).std(0) + 1e-6)) * fs + fm
            return {"flow": torch.tensor((past_flow[sel] - pm) / ps).to(DEV),
                    "met": torch.tensor((m_ - fm) / fs).to(DEV),
                    "y": torch.tensor((y[sel] - ym) / ys).to(DEV),
                    "mask": torch.tensor(mask[sel], dtype=torch.float32).to(DEV)}

        d_tr, d_va = pack(tr), pack(va)
        models = [fit_one(d_tr, d_va, s) for s in range(N_SEEDS)]

        # Lead-dependent noise augmentation: same 3,500 observed sequences, but
        # the future rainfall is degraded each epoch to the correlation the real
        # forecast achieves at that lead.  Teaches discounting from the full
        # record instead of the few hundred forecast-forced sequences.
        d_va_nz = degrade(d_va, rho_t)
        models_nz = [fit_one(d_tr, d_va_nz, s, rho=rho_t) for s in range(N_SEEDS)]

        # ---- fine-tune on FORECAST meteorology.
        # The pre-trained model learned the rainfall-runoff response from
        # observed forcing and therefore trusts its forcing completely.  At 0.25
        # deg the downscaled S2S rainfall has R2 ~ 0.08, so that trust is
        # misplaced past about lead 7 -- v2's forecast branch collapses to
        # -0.229 at 13-17 d for exactly this reason.  These few hundred forecast
        # sequences cannot teach hydrology, but they can teach discounting.
        ft_tr = (fyear != Y) & (fyear != va_year)
        ft_va = fyear == va_year
        d_ftr = pack(f_rows[ft_tr], met_override=f_met[ft_tr])
        d_fva = pack(f_rows[ft_va], met_override=f_met[ft_va])
        # deepcopy is load-bearing: fit_one mutates the model it is handed and
        # returns the same object, so fine-tuning in place would silently
        # overwrite the pre-trained weights and make all three branches report
        # the fine-tuned model.
        tuned = [fit_one(d_ftr, d_fva, s, model=copy.deepcopy(m), lr=FT_LR,
                         epochs=FT_EPOCHS, patience=FT_PATIENCE, batch=FT_BATCH)
                 for s, m in enumerate(models)]

        # Noise-augmented weights, then the same forecast fine-tune -- separates
        # "augmentation instead of fine-tuning" from "augmentation as well".
        tuned_nz = [fit_one(d_ftr, d_fva, s, model=copy.deepcopy(m), lr=FT_LR,
                            epochs=FT_EPOCHS, patience=FT_PATIENCE, batch=FT_BATCH)
                    for s, m in enumerate(models_nz)]

        # Same fine-tune, but on quantile-mapped forcing, so the effect of
        # Module 3 is separated from the effect of fine-tuning.
        d_qtr = pack(f_rows[ft_tr], met_override=f_met[ft_tr], use_qm=True)
        d_qva = pack(f_rows[ft_va], met_override=f_met[ft_va], use_qm=True)
        tuned_qm = [fit_one(d_qtr, d_qva, s, model=copy.deepcopy(m), lr=FT_LR,
                            epochs=FT_EPOCHS, patience=FT_PATIENCE, batch=FT_BATCH)
                    for s, m in enumerate(models)]

        sel = f_rows[te]
        d_obs = pack(sel)
        d_fc = pack(sel, met_override=f_met[te])
        d_qm = pack(sel, met_override=f_met[te], use_qm=True)
        with torch.no_grad():
            def ens(ms, d):
                return np.stack([m(d["flow"], d["met"]).cpu().numpy() for m in ms]).mean(0)
            p_obs = ens(models, d_obs)
            p_fc = ens(models, d_fc)
            p_ft = ens(tuned, d_fc)
            p_qm = ens(models, d_qm)
            p_qft = ens(tuned_qm, d_qm)
            p_nz = ens(models_nz, d_fc)
            p_nzft = ens(tuned_nz, d_fc)
        p_obs, p_fc, p_ft, p_qm, p_qft, p_nz, p_nzft = (
            np.clip(v * ys + ym, 0, None)
            for v in (p_obs, p_fc, p_ft, p_qm, p_qft, p_nz, p_nzft))

        preds = {"p_obs": p_obs, "p_fc": p_fc, "p_ft": p_ft, "p_qm": p_qm,
                 "p_qft": p_qft, "p_nz": p_nz, "p_nzft": p_nzft}
        for i, ii in enumerate(sel):
            for k in range(HORIZON):
                r = {"init_date": inits[ii], "lead_day": k + 1,
                     "valid_date": vdates[ii][k]}
                for nm, arr in preds.items():
                    # sort across quantiles: pinball does not hard-constrain
                    # monotonicity, and a crossed pair would make an interval
                    # width negative.  Crossings are rare and small; sorting is
                    # the standard remedy and changes nothing else.
                    v = np.sort(arr[i, k])
                    for qi, q in enumerate(QUANTILES):
                        r[f"{nm}_q{int(q * 100):02d}"] = v[qi]
                rows.append(r)
        print(f"  hold out {Y}: train {int(tr.sum()):,} seq, test {int(te.sum())} inits "
              f"({time.time() - t0:.0f}s)")

    pred = pd.DataFrame(rows)
    pred["valid_date"] = pd.to_datetime(pred["valid_date"])
    m = ref.merge(pred.drop(columns=["valid_date"]), on=["init_date", "lead_day"])
    print(f"\nmatched rows: {len(m):,}")

    obs = m["target_inflow"].to_numpy()
    yr = m["valid_date"].dt.year
    if "--smoke" in sys.argv:
        globals()["YEARS"] = sorted(set(int(v) for v in yr.unique()))
    out = {"config": {"past_len": PAST_LEN, "horizon": HORIZON, "d_model": D_MODEL,
                      "enc_layers": N_ENC, "dec_layers": N_DEC, "seeds": N_SEEDS,
                      "met_vars": MET, "n_daily_sequences": int(len(inits))},
           "models": {}}
    print(f"\n=== ensemble-aware probabilistic FutureTST, LOYO over {len(YEARS)} monsoons ===")
    BRANCHES = [("observed met", "p_obs"), ("forecast met", "p_fc"),
                ("forecast + fine-tune", "p_ft"),
                ("forecast + QM", "p_qm"),
                ("forecast + QM + fine-tune", "p_qft"),
                ("forecast + noise-aug", "p_nz"),
                ("forecast + noise-aug + FT", "p_nzft")]

    def q(c, prob):
        return m[f"{c}_q{int(prob * 100):02d}"].to_numpy()

    def crps(c, sub=None):
        """Pinball loss averaged over QUANTILES, x2 -- a discrete CRPS estimator.

        Exact CRPS integrates the pinball loss over all quantile levels; with a
        finite grid this is its Riemann estimate, consistent and comparable
        across models scored on the SAME grid (which is all we need here).
        """
        d = m if sub is None else sub
        o = d["target_inflow"].to_numpy()
        tot = np.zeros(len(d))
        for prob in QUANTILES:
            e = o - d[f"{c}_q{int(prob * 100):02d}"].to_numpy()
            tot += np.maximum(prob * e, (prob - 1.0) * e)
        return float(2 * tot.mean() / len(QUANTILES))

    def cover(c, lo, hi, sub=None):
        d = m if sub is None else sub
        o = d["target_inflow"].to_numpy()
        a = d[f"{c}_q{int(lo * 100):02d}"].to_numpy()
        b = d[f"{c}_q{int(hi * 100):02d}"].to_numpy()
        return float(((o >= a) & (o <= b)).mean()), float((b - a).mean())

    def peak_stats(c, sub):
        """Peak magnitude ratio and |timing error| on the median forecast."""
        g = sub.sort_values(["init_date", "lead_day"])
        pm, dt = [], []
        for _, w in g.groupby("init_date"):
            o = w["target_inflow"].to_numpy()
            s = w[f"{c}_q50"].to_numpy()
            if len(o) < 3 or not np.isfinite(o).all():
                continue
            pm.append(s.max() / max(o.max(), 1e-6))
            dt.append(abs(int(s.argmax()) - int(o.argmax())))
        return (float(np.median(pm)) if pm else float("nan"),
                float(np.mean(dt)) if dt else float("nan"))

    hdr = (f"  {'branch':26s}{'NSE med':>9}{'RMSE':>8}{'CRPS':>8}"
           f"{'cov80':>7}{'cov90':>7}{'width80':>9}{'peak':>7}{'|dt|':>7}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for nm, c in BRANCHES:
        med = q(c, 0.50)
        pf = np.array([nse(m.loc[yr == Y, f"{c}_q50"].to_numpy(),
                           m.loc[yr == Y, "target_inflow"].to_numpy()) for Y in YEARS])
        c80, w80 = cover(c, 0.10, 0.90)
        c90, w90 = cover(c, 0.05, 0.95)
        c50, w50 = cover(c, 0.25, 0.75)
        pk, dt = peak_stats(c, m)
        rec = {"median_NSE": float(np.median(pf)), "mean_NSE": float(pf.mean()),
               "pooled_NSE": float(nse(med, obs)), "pooled_KGE": float(kge(med, obs)),
               "RMSE": float(np.sqrt(((med - obs) ** 2).mean())),
               "CRPS": crps(c),
               "coverage_50": c50, "coverage_80": c80, "coverage_90": c90,
               "width_50": w50, "width_80": w80, "width_90": w90,
               "peak_ratio": pk, "peak_timing_err_days": dt,
               "per_year": {int(Y): float(v) for Y, v in zip(YEARS, pf)}}
        out["models"][nm] = rec
        print(f"  {nm:26s}{rec['median_NSE']:>9.3f}{rec['RMSE']:>8.0f}"
              f"{rec['CRPS']:>8.0f}{c80:>7.2f}{c90:>7.2f}{w80:>9.0f}"
              f"{pk:>7.2f}{dt:>7.2f}")

    print("\n  per lead band (median NSE / CRPS / 80% coverage):")
    bands = {}
    for lo, hi in [(1, 3), (4, 7), (8, 14), (15, 21), (22, 30)]:
        sub = m[(m.lead_day >= lo) & (m.lead_day <= hi)]
        o = sub["target_inflow"].to_numpy()
        band = {}
        for nm, c in BRANCHES:
            per_year = [nse(sub.loc[sub.valid_date.dt.year == Y, f"{c}_q50"].to_numpy(),
                            sub.loc[sub.valid_date.dt.year == Y, "target_inflow"].to_numpy())
                        for Y in YEARS]
            cv, wd = cover(c, 0.10, 0.90, sub)
            pk, dt = peak_stats(c, sub)
            band[nm] = {"NSE_median": float(np.median(per_year)),
                        "NSE_pooled": float(nse(sub[f"{c}_q50"].to_numpy(), o)),
                        "RMSE": float(np.sqrt(((sub[f"{c}_q50"].to_numpy() - o) ** 2).mean())),
                        "CRPS": crps(c, sub), "coverage_80": cv, "width_80": wd,
                        "peak_ratio": pk, "peak_timing_err_days": dt}
        bands[f"{lo}-{hi}"] = band
        best = band["forecast + QM + fine-tune"]
        print(f"    {lo:2d}-{hi:2d} d   QM+FT: NSE {best['NSE_median']:+.3f}  "
              f"CRPS {best['CRPS']:6.0f}  cov80 {best['coverage_80']:.2f}  "
              f"width {best['width_80']:6.0f}  peak {best['peak_ratio']:.2f}")
    out["lead_bands"] = bands
    out["quantiles"] = QUANTILES

    C.METRICS.mkdir(parents=True, exist_ok=True)
    tag = "futuretst_prob_degraded" if DEGRADE_MET else "futuretst_prob"
    # A fast run produces junk by design.  It must never land on the path the
    # real results live at -- that is exactly how results/metrics/futuretst_prob
    # .json came to hold a 1-seed, 2-year artefact that matched nothing in
    # docs/FINDINGS.md.  Suffix it instead.
    if FAST_RUN:
        tag += "_FAST"
    (C.METRICS / f"{tag}.json").write_text(json.dumps(out, indent=2))
    m.to_parquet(C.PROCESSED / f"{tag}_predictions.parquet", index=False)
    print(f"\nwrote {C.METRICS / (tag + '.json')}")


if __name__ == "__main__":
    main()
