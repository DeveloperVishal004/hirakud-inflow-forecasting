# Sub-seasonal Reservoir Inflow Forecasting — Hirakud Dam, Mahanadi Basin

**What this is.** Two independent 30-day inflow forecasting pipelines, built from
two published papers, evaluated against each other and against trivial baselines
on 19 monsoon seasons.

**The result, in one sentence.** Neither a physically-based chain nor a
transformer beats a one-parameter scaling of the raw ECMWF forecast, and the
reason is measurable: catchment rainfall forecasts lose almost all skill beyond
about one week.

*Last rebuilt 7 September 2026. Every number here was measured from the current
artefacts; nothing is quoted from an earlier draft.*

---

## 1. Problem & Objective

Hirakud Dam sits on the Mahanadi with a catchment of about 83,400 km². It is
operated for flood control and irrigation, and both need to know how much water
is arriving. A monsoon flood peak has to be released before it arrives; water
held back too late is a flood, released too early is a lost season.

**Objective.** Forecast daily inflow to Hirakud **1–30 days ahead**, during the
monsoon (June–September), and establish honestly whether it can be done.

**The question the project answers.** Given a sub-seasonal precipitation
forecast, does a physically-based hydrological chain, or a modern sequence
model, produce a usable 30-day inflow forecast — and if not, where exactly is
the skill lost?

Two published approaches are reimplemented and compared:

| | source | approach |
|---|---|---|
| **Layer 2** | Dong et al. (2025), *HESS* 29:2023–2042 | CNN downscaling → hydrological model → LSTM correction |
| **Layer 3** | Ambika et al. (2025), *GRL* 52:e2025GL116707 | FutureTST transformer, inflow direct from forecast weather |

They are kept **strictly independent**. No output of one is ever an input to the
other. That is what makes the comparison meaningful.

---

## 2. Data

| dataset | period | resolution | role |
|---|---|---|---|
| ECMWF S2S reforecast | 2004–2022 | 1.5°, 11 members, leads 1–30 | the forecast being tested |
| IMD gridded rainfall | 2003–2022 | 0.25° | rainfall truth, CNN target |
| ERA5-Land | 2003–2022 | 0.1° → 0.25°, 6-hourly | VIC's observed forcing |
| Hirakud inflow | 2003–2022 | daily | the prediction target |

**Scale of the forecast archive.** 532 initialisations × 30 leads = **15,960
rows**, each with 11 ensemble members and 26 predictor variables on a 7×7 box.

### The two data facts that shape everything

**1. The inflow record is not uniform.** 2003–2014 has near-complete years
(292–363 days). **2015–2022 is monsoon-only** — 153 days per year, of which just
31 fall outside June–September. Any analysis needing a dry season is limited to
2004–2014.

**2. Observed inflow is not natural runoff.**

| | mm/yr | runoff coefficient |
|---|---|---|
| IMD rainfall | 1,281.9 | — |
| observed Hirakud inflow | 366.2 | **0.286** |
| published *natural* value | 449–513 | 0.35–0.40 |

**80–150 mm/yr — 18–29 % of natural runoff — never reaches the gauge.** The
Mahanadi above Hirakud is heavily irrigated, and the dam sees what is left after
upstream use. VIC simulates *natural* runoff and has no abstraction term. This
single fact explains a large part of the model's apparent bias, and it is
returned to in §4.3 and §8.

---

## 3. Overall Architecture

```
                    ECMWF S2S reforecast (1.5°, 11 members, leads 1-30)
                                    |
              +---------------------+---------------------+
              |                                           |
      LAYER 2 (physics)                            LAYER 3 (learned)
              |                                           |
      CNN downscaling (0.25°)                    catchment-mean weather
              |                                     + inflow history
      VIC water balance (150 cells)                       |
              |                                      FutureTST
      RVIC routing (unit hydrographs)                     |
              |                                           |
      LSTM error correction                       30-day inflow
              |                                    (deterministic
      30-day inflow                                 + 7 quantiles)
              |                                           |
              +---------------------+---------------------+
                                    |
                    common evaluation: 309 windows, 19 years
```

The two layers share **only** the raw forecast and the evaluation. Nothing else
crosses between them.

---

## 4. Layer 2 — Physics-based

### 4.1 ECMWF preprocessing

The reforecast arrives as GRIB and needs three corrections before use:

- **Accumulated variables** (`tp`, `cp`, `ssrd`, `sshf`, `slhf`) accumulate from
  initialisation. They must be differenced along the lead axis. Averaging the raw
  values overstates rainfall several-fold.
- **`tcc` is a percentage** (0–100), not the fraction its name suggests.
- **`t2m` and `tcc` are period-processed**, not instantaneous. An instantaneous
  request returns a single day-1 field broadcast across all 30 leads. They are
  requested as 24-hour period windows instead. *Verified across all 532
  initialisations: both vary properly with lead, and correlate with rainfall in
  the physically correct direction (cloud +0.63, temperature −0.40).*

**Three initialisation dates are impossible** (07-12, 07-19, 07-26 are Fridays
in the 2024 model cycle and rejected by the archive), giving 28 usable
initialisations per monsoon.

### 4.2 CNN downscaling

A ResNet (Dong §3.2.1) maps a 3×3 patch of coarse cells plus terrain to
rainfall at each 0.25° catchment cell — 1.5° (~150 km) to 0.25° (~25 km), for
each of 10 perturbed members. Trained leave-one-year-out, so each year's
downscaled field comes from a model that never saw that year.

**It improves rainfall spatially — and degrades the catchment total.**

| lead | corr(CNN rain, IMD) | corr(raw EC, IMD) |
|---|---|---|
| 1–3 d | 0.574 | **0.637** |
| 4–7 d | 0.454 | **0.509** |
| 8–14 d | 0.365 | **0.400** |
| 15–21 d | 0.350 | **0.362** |
| 22–30 d | **0.385** | 0.376 |

The CNN is trained to place rain in space. VIC integrates the basin *total*, and
on that quantity raw ECMWF is better at every lead but the last. This matters —
see §7.

### 4.3 VIC

VIC 5.0.1, classic driver, water-balance mode, **150 active cells**, 6-hourly
time step, driven by ERA5-Land forcing.

**Calibration design.**

| period | role |
|---|---|
| 2003 | spin-up, discarded |
| **2004–2011** | **calibration** — every observed day, 2,740 days |
| 2012–2014 | validation, never scored during the search |
| 2015–2022 | held out entirely |

Eight parameters by differential evolution, 432 evaluations, 135 minutes.
The objective is

```
KGE(discharge) − 2 × [ hinge(runoff coefficient, 0.35–0.40)
                     + hinge(ET/P,               0.55–0.65) ]
```

Streamflow is scored where it is **observed** (daily); the water-balance
partition where it is **defined** (annual). Fitted values:

| parameter | value | | parameter | value |
|---|---|---|---|---|
| b_infilt | 0.183 | | d3 | 2.802 m |
| Ds | 0.0055 | | wpwp_scale | 0.255 |
| Dsmax | 23.54 | | root_scale | 1.814 |
| Ws | 0.833 | | d2 | 1.944 m |

**Skill on the three-way split** (monsoon days):

| split | NSE | KGE | corr | bias | n |
|---|---|---|---|---|---|
| calibration 2004–2011 | 0.581 | 0.413 | 0.931 | **+50.3 %** | 944 |
| validation 2012–2014 | 0.328 | 0.281 | 0.914 | **+58.4 %** | 344 |
| **held out 2015–2022** | 0.416 | 0.191 | **0.937** | **+72.1 %** | 976 |

**Two things to read here.** Correlation is *highest on the held-out years*
(0.937) — VIC's timing generalises; it is not memorising. But bias grows
monotonically across the three periods. A calibration error would be flat; a bias
that grows with time is what increasing upstream abstraction looks like.

**Annual water balance** (2004–2022): runoff coefficient **0.445** against a
0.35–0.40 target, ET/P **0.558** (in band), closure 1.003. The runoff
coefficient cannot be brought into band with physically plausible soil depths —
attempts to force it drove soil layers past 4 m. That residual is the abstraction
term VIC does not have.

### 4.4 RVIC routing

Unit-hydrograph convolution to a single outlet, 150 source cells, 10 lags,
network from Dominant River Tracing. The conversion `mm/day → m³/s` uses cell
area × basin fraction; it is the only place catchment geometry enters.

The 10 lags reach **backwards** into runoff from before the forecast — history
available at issue time, not a look-ahead.

### 4.5 LSTM error correction

The XAJ-LSTM role from Dong §3.4.3, with VIC in place of XAJ. Input: 60 days of
antecedent observed inflow, the routed VIC forecast, and the forecast rainfall
**belonging to that product**. Output: a corrected 30-day hydrograph. Folds are
leave-one-year-out.

**Sample limitation.** The 60-day history requirement leaves **311 of 532
initialisations usable**. Losses fall hardest on early-monsoon dates — from 2015
the record begins in June, so early-June forecasts cannot look back far enough.

**What it does** (per-member, 19 folds):

| chain | VIC alone | with LSTM | t | wins |
|---|---|---|---|---|
| raw EC | −0.676 | −0.010 | **+3.76** | **18/19** |
| EC-CNN | +0.141 | +0.116 | +2.38 | 13/19 |

It is a **bias-correction layer, not a forecasting layer**. Its gain scales with
how much bias is left upstream. Removing the volume error *before* it (a
leave-one-year-out scalar) makes it actively harmful — 7/19 wins on both chains.
There is no useful residual shape for it to learn at this sample size.

---

## 5. Layer 3 — FutureTST

A transformer taking historical inflow and meteorology **plus future forecast
meteorology**, predicting 30 days directly. No hydrological model.

Architecture: gated multivariate encoder, variable-type embeddings, asymmetric
patching (past patched into 8-day tokens, future left unpatched), and 30 learned
horizon queries attending to the encoded past.

### 5.1 Deterministic

Trained leave-one-year-out with a rotating validation year. Two variants are
reported: fine-tuned on forecast weather (`FutureTST + FT`), and the same with
quantile-mapped inputs (`+ QM + FT`).

### 5.2 Probabilistic

Seven quantiles (0.05–0.95) trained with pinball loss, scored by CRPS and
interval coverage. This is the part of the project that remains useful at long
lead: when the deterministic forecast has no skill, a calibrated interval still
does.

> **⚠ Status.** The FutureTST rows in §7 **predate a data repair** (a row-order
> bug that scrambled forecast dates, since fixed) and have **not been re-run**.
> Treat every FutureTST number as provisional. Layer 2 numbers are current.

---

## 6. Experimental Design

### 6.1 Leave-one-year-out

Every learned component — CNN, LSTM, FutureTST, quantile mapping, and the scalar
baseline — is trained leave-one-**year**-out. Never a random split: one valid
date is reached by several initialisations at different leads, so a random split
puts the same observed field on both sides of the partition.

### 6.2 Baselines

Two, both deliberately hard to beat:

- **persistence** — today's inflow held flat for 30 days. The floor.
- **`0.86 × EC`** — a single scalar fitted leave-one-year-out on the VIC chain's
  own output. This is not a naive baseline: it is the physics chain with its
  volume error removed by one number, and it is the number to beat.

### 6.3 Evaluation windows

**309 windows** common to every model — those where 60 days of antecedent inflow
*and* all 30 forecast days are observed, so every model is scored on exactly the
same sample. Scoring is on the **ensemble mean**; per-member scoring inflates
apparent skill and the two are never mixed.

Reported together: **median, mean, and win-count**. Median alone hides a
correction layer that rescues catastrophic years; mean alone is dominated by
them.

---

## 7. Final Results

309 windows, 19 monsoons, 30-day horizon, ensemble mean.

| model | NSE med | NSE mean | RMSE | bias % | peak | \|Δt\| d | vs baseline |
|---|---|---|---|---|---|---|---|
| persistence | −1.023 | −1.241 | 4221 | +21.1 | 0.385 | — | p<0.001, 0/19 |
| **`0.86 × EC`** | **0.168** | 0.069 | **2654** | −2.9 | 0.794 | **6.96** | — |
| `EC → VIC → RVIC` | −0.010 | −0.433 | 2877 | +31.8 | **1.081** | 6.96 | p=0.005, 2/19 |
| `EC-CNN → VIC → RVIC` | 0.161 | −0.075 | 2714 | **+0.7** | 0.833 | 8.43 | p=0.081, 5/19 |
| `EC → VIC → RVIC → LSTM` | 0.142 | **0.088** | 2663 | −10.3 | 0.729 | 6.99 | p=0.520, 9/19 |
| `EC-CNN → VIC → RVIC → LSTM` | 0.140 | 0.034 | 2748 | −15.8 | 0.652 | 7.90 | p=0.531, 6/19 |
| FutureTST + FT ⚠ | **0.186** | −0.095 | 2680 | −17.9 | 0.559 | 7.16 | p=0.277, 11/19 |
| FutureTST + QM + FT ⚠ | 0.137 | 0.036 | 2693 | −17.8 | 0.559 | 7.29 | p=0.582, 10/19 |

**No model is significantly better than `0.86 × EC`.** The best physics
configuration ties it (p = 0.52, 9/19). This survives three independent VIC
calibrations.

### Skill by lead time

Median NSE, same windows:

| band | persistence | `0.86 × EC` | EC-CNN→VIC | EC→VIC→LSTM | FutureTST ⚠ |
|---|---|---|---|---|---|
| 1–3 d | 0.507 | **0.699** | 0.331 | 0.671 | 0.501 |
| 4–7 d | −0.616 | 0.049 | **0.260** | −0.017 | 0.264 |
| 8–14 d | −1.239 | 0.107 | 0.111 | 0.056 | 0.128 |
| 15–21 d | −1.434 | 0.102 | 0.032 | 0.088 | 0.123 |
| **22–30 d** | −1.586 | −0.009 | −0.005 | 0.009 | 0.073 |

**Beyond one week, skill is essentially zero.** Days 1–3 look strong, but
persistence alone scores 0.507 there — that is catchment memory, not forecast
skill.

### The diagnostic experiment

Driving the identical chain with **observed** weather gives the ceiling the
forecast can never exceed:

| | median NSE |
|---|---|
| perfect rainfall, as simulated | 0.394 |
| **perfect rainfall + one volume scalar** | **0.850** |
| perfect rainfall + optimal time shift | 0.471 |
| *actual* `EC-CNN → VIC → RVIC` | 0.161 |

Mean optimal time shift: **0.28 days**.

---

## 8. What We Learned

**1. The rainfall forecast is the binding constraint.** Catchment rainfall
correlates 0.35–0.40 with reality beyond a week. Everything downstream inherits
that ceiling. No post-processor recovers information that is not in the input.

**2. VIC + RVIC is sound.** Given correct water volume, it reproduces 30-day
inflow at **NSE 0.850**, and its timing generalises to unseen years
(correlation 0.937). The hydrology is not the weak link.

**3. The dominant correctable error is volume, not timing.** With perfect
rainfall, the mean optimal time shift is 0.28 days, and one scalar moves the
chain from 0.394 to 0.850. *An earlier version of this project concluded that
phase dominated; that conclusion came from decomposing the forecast chains,
where the timing error belongs to ECMWF rather than to VIC.*

**4. Part of the "bias" is not error at all.** Observed inflow is 18–29 % below
natural runoff because of upstream abstraction. VIC has no term for it, and the
bias grows across calibration → validation → held-out periods, consistent with
increasing irrigation.

**5. The CNN's apparent advantage is bias cancellation.** Compared at equal
volume — a leave-one-year-out scalar applied to both — the downscaled chain
scores **0.157** against raw ECMWF's **0.168**. The CNN is slightly *worse*. Its
lead in the main table comes from a dry catchment mean (+0.7 %) offsetting VIC's
wet bias (+31.8 %): two errors of opposite sign, not better rainfall.

The CNN is trained to place rain in space; VIC integrates the basin *total*, and
on that quantity raw ECMWF correlates better at every lead but the last
(§4.2). *A fix for this was tested and works, but it is a new method rather than
part of Dong et al., so it is held in `archive/beyond_paper_20260907/` for a
further-work section rather than reported here.*

**6. The learned correction is a bias layer.** The LSTM's benefit is
proportional to the bias left upstream (18/19 years where bias is large, 7/19
where it has been removed first). It is not learning hydrological structure.

**7. Nothing beats a single number.** Because `0.86 × EC` *is* the physics chain
with its volume error removed — and volume is the error.

---

## 9. Limitations

**Scope.** One basin, one reservoir, 19 monsoons, 309 evaluation windows. No
claim generalises beyond the Mahanadi above Hirakud.

**FutureTST results are provisional.** They predate a data repair and have not
been re-run. Every FutureTST row in §7 carries ⚠.

**VIC's parameters are effective, not physical.** They absorb upstream
abstraction the model cannot represent. `wpwp_scale` sits at its lower bound and
soil depths near their ceilings — the optimiser is compensating for a missing
process. Do not read them as measured soil properties.

**The 42 % overlap.** VIC is fitted on 8 of the 19 evaluated years. The physics
chains are therefore partly in-sample where FutureTST is not; `fair_comparison.py`
reports the VIC-unseen split separately.

**The LSTM uses 58 % of the data.** 311 of 532 initialisations, losing
early-monsoon dates hardest — the onset period operators most need.

**A 5.5-hour forcing offset.** Radiation accumulates on UTC days, temperature is
averaged on IST days. Measured: this changes daily shortwave by 0.002 % and
redistributes 24 % of longwave between adjacent days. The 6-hourly archive
cannot resolve the boundary, so it is documented rather than corrected.

**Three day-definitions coexist** — IST for meteorology, UTC for radiation,
08:30 IST for IMD rainfall.

**What would change the conclusion.** Only better rainfall forecasts, or
explicit representation of upstream abstraction. Further work on the
hydrological model or the correction layer will not move these numbers.

---

## Where everything else lives

| document | purpose |
|---|---|
| **this file** | what the project is, how it works, what it found |
| `FINDINGS.md` | research log — what was discovered during development, in order |
| `STRUCTURE.md` | which file belongs to which pipeline, and the traps |
| `CURRENT_STATE.md` | what is finished, what is pending |
| `archive/historical_docs/` | superseded drafts, kept for provenance |
| `archive/superseded_20260907/` | code from approaches not carried forward |
