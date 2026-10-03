# Research log — what was discovered during development

**This is the log, not the results.** Current numbers live in
`docs/PROJECT_DOCUMENTATION.md`. This file records what was learned, in the
order it was learned, including the things that turned out to be wrong.

Entries marked **[CORRECTED]** were believed for a time and later disproved by
measurement. They are kept deliberately — a claim that was wrong for months is
worth remembering, and the reason it survived is usually more instructive than
the correction.

---

## The inherited pipeline was broken in six ways

The project began from a notebook reporting **R² 0.873** for rainfall
downscaling. Auditing found six defects, every one inflating the score:

| defect | why it inflated |
|---|---|
| accumulated variables never differenced | running totals fed as daily values |
| ocean cells filled with zeros | 6 of 25 target cells were a constant the model "predicted" perfectly |
| random train/test split | the same observed rainfall map on both sides |
| scaler fitted before splitting | test statistics leaked into training |
| target subsampled to 25 coarse points | **no downscaling was happening at all** |
| rainfall lags indexed to the target date | at lead 17, "yesterday's rain" was 16 days in the future |

**Honest score after fixing all six: R² 0.073–0.078.**

The fifth is the one to remember. The model was predicting coarse rainfall from
coarse rainfall and reporting it as downscaling. It was not a subtle bug; it was
invisible because nobody checked what the target actually contained.

---

## Bugs found and fixed during the work

**The year-label trap.** Code computed a forecast's year as `first_year + index`,
assuming one continuous run. The archive holds **two separate hindcast ranges**,
so 2015–2022 would have been labelled 2004–2011 — *"every downstream fold would
be wrong, silently."*
→ *Read metadata from the data, never reconstruct it from configuration.*

**48 % of training patches were fabricated.** On the original 5×5 download,
nearly half the coarse patches were padding.

**The scrambled forecast dates — the worst defect, and the last found.**
`cnn/quantile_mapping.py` sorted its data into a canonical order but wrote the
**date labels from the un-sorted arrays**. Every row carried another row's date.

It hid because two of the three labels were accidentally right: lead is the
repeating sequence 1…30 under either ordering, and the year came from the
correctly-ordered source. Only the date the forecast was *for* was wrong, and
nothing downstream printed it.

| check | before | after |
|---|---|---|
| rows whose date minus lead equals the true issue date | 300 / 15,960 | 15,960 / 15,960 |
| initialisations paired with their own rainfall | 10 / 532 | 532 / 532 |
| day-1 correlation, forecast rain vs observed | **−0.02** | **+0.68** |

A day-1 rainfall forecast correlating −0.02 with what fell is not a forecast.
→ *When you reorder an array, reorder its labels in the same statement — then
verify with a physical check the labels cannot pass by accident.*

**Calibration multipliers compounded across runs.** `calibrate_vic.py` read the
*active* soil and vegetation files as its starting point. Activating a
calibration overwrites those, so each run began from the previous winner. Because
`wpwp_scale` and `root_scale` are multipliers, they stacked: a run reporting
`wpwp_scale = 0.8755` had an effective value of 0.1532 × 0.8755 = **0.1341**
against the pedotransfer file — *lower* than the value that run was launched to
escape. The bounds no longer meant what they said.
→ Fixed with immutable base files and a guard that refuses to start from a
calibration output. → *A parameter that is a multiplier must be applied to a
fixed reference, never to the last result.*

**Fabricated zeros in the calibration target.** `load_series()` stores missing
gauge days as `0.0` and flags them separately. The calibration and the routing
evaluation both ignored the flag. 213 of 3,287 calibration days were flagged
invalid and **all 213 were zeros, concentrated January–June**. Switching to
all-day scoring made this worse, not better, so it had to be fixed in the same
change.
→ *One project, one definition of "observed". Honour the validity flag everywhere.*

**The spin-up year was being scored.** The docstring said 2003 was discarded. It
was in the calibration years and the objective scored it.

**`t2m` and `tcc` carried no lead information — then did.** Both are
period-processed in the S2S reforecast; an instantaneous request returned a
single day-1 field broadcast across all 30 leads. The download was corrected to
24-hour period windows and the archive rebuilt. The config note warning about it
survived long after it stopped being true, which is its own lesson.

---

## [CORRECTED] Error is dominated by phase

**Believed for months.** An MSE decomposition attributed 66–95 % of squared
error to phase, and the conclusion drawn was that timing error dominates and no
downstream model can fix it.

**Disproved 7 September 2026.** Driving the identical chain with *observed*
weather and decomposing the residual:

- mean optimal time shift: **0.28 days** — timing is essentially perfect
- optimal per-window shifting gains only 0.394 → 0.471
- **a single volume scalar gains 0.394 → 0.850**

The model's error is **volume**, not phase.

**Why the wrong version survived.** The decomposition was run on the *forecast*
chains, where the timing error belongs to ECMWF, not to VIC. Two different
things — the rainfall forecast's phase error and the hydrological model's
error — were measured together and attributed to the model.
→ *To measure a component's error, hold its inputs perfect.*

---

## [CORRECTED] The CNN improves the inflow forecast

**Believed** because the downscaled chain scored far above the raw chain in the
main table.

**Disproved.** Compared at equal volume — a leave-one-year-out scalar applied to
both — the downscaled chain scores **0.157** against raw ECMWF's **0.168**. The
CNN is slightly *worse*.

Its apparent advantage was **bias cancellation**: a dry catchment mean (+0.7 %)
offsetting VIC's wet bias (+31.8 %). Two errors of opposite sign.

The CNN is trained to place rain *in space*; VIC integrates the basin *total*,
and on that quantity raw ECMWF correlates better at every lead but the last.
Keeping the CNN's spatial pattern while taking the catchment total from raw
ECMWF scores **0.200** — better than either on median, mean and worst year. That
fix is a new method rather than part of Dong et al., so it lives in
`archive/beyond_paper_20260907/` as proposed further work; the diagnosis above
stays here because it is an analysis of the paper's own product.
→ *Before crediting a component, remove the bias. Two errors can cancel.*

---

## [CORRECTED] Raw ECMWF is worse than useless at cell scale

**Believed, and quoted everywhere.** Raw ECMWF per-cell R² = **−0.136**, downscaled
**+0.057** — "the sign flip is the claim".

**Disproved 14 September 2026.** The −0.136 figure is produced by no script and
appears in no `results/metrics/` file; it traces only to
`archive/historical_docs/`. Re-measured on the same 224 cells, same mask, same 19
folds and the same **ensemble-mean** regime the CNN is scored in
(`tools/raw_ec_percell_baseline.py`):

| per-cell R² | mean | folds positive |
|---|---|---|
| raw EC, nearest coarse cell | **+0.044** | 17/19 |
| raw EC, 3×3 smoothed | **+0.073** | 19/19 |
| CNN | +0.057 | 19/19 |
| raw EC, **per-member** | **−0.349** | 0/19 |

Raw ECMWF is already positive. The CNN's honest gain is **+0.013**, not a rescue
from negative territory.

**Why the wrong version survived.** It was a **per-member** baseline set against an
**ensemble-mean** model. Averaging ten members is worth ~0.39 of R² here — far more
than the network contributes. This is the **third** time the two regimes have been
mixed; `evaluate_operational.py` was written to stop exactly this, but it governs
the *inflow* metrics and was never applied to the *rainfall* ones.

**The uncomfortable part.** Plain 3×3 smoothing — no learning, no terrain — beats
the trained network. A squared-error score on a noisy right-skewed field rewards
the conditional mean, which is the same behaviour that makes the CNN's heavy-rain
and CRPS scores worse than doing nothing. The surviving case for downscaling is
that VIC needs a value for 150 cells, not that per-cell R² is better.
→ *One regime per metric, enforced in code — not just for the metric that once
burned you.*

---

## The correction layer is a bias layer

Tested three independent ways, all agreeing:

1. Its gain tracks how much bias is left upstream — 18/19 years where the bias
   is large, 13/19 where it is smaller.
2. Remove the volume error *first* with a leave-one-year-out scalar and the LSTM
   becomes actively harmful (7/19 wins on both chains). There is no useful
   residual shape for it to learn at 311 usable windows.
3. It is beaten by a **single scalar** on the median metric.

But it is better than that sounds: on the *mean* it wins, because it rescues the
catastrophic years (+0.38, +0.30, +0.20 in the worst ones) and costs at most
−0.06 in good ones. Median alone systematically undervalues a corrector.
→ *Report median, mean and win-count together. Any one of them lies.*

---

## Part of the bias is not error

Observed inflow implies a runoff coefficient of **0.286**. Published natural
values for the basin are 0.35–0.40. **80–150 mm/yr — 18–29 % of natural runoff —
never reaches the gauge**, because the Mahanadi above Hirakud is heavily
irrigated.

VIC simulates natural runoff and has no abstraction term, so calibrating it to
match the gauge asks it to lose that water through the only sink it has:
evapotranspiration. The optimiser obliged, driving soil depth to every ceiling
it was given — 1.404 of 1.50, then 2.9987 of 3.00, then 3.9991 of 4.00.

Supporting evidence: bias grows monotonically across calibration (+50 %),
validation (+58 %) and held-out (+72 %) periods. A calibration error would be
flat; a bias growing with time is what increasing irrigation looks like.
→ *A bound that binds three times is not a constraint, it is a missing process.*

---

## Evaluation lessons

**Two scoring regimes were mixed twice.** Per-member and ensemble-mean scoring
give different answers; conflating them produced wrong conclusions twice, which
is why `inflow/evaluate_operational.py` exists to define one sample.

**Significance was never run for most models** until `fair_comparison.py`. When
it was, the honest conclusion sharpened: no configuration significantly beats a
one-parameter scaling of the raw forecast.

**The baseline is not naive.** `0.86 × EC` *is* the physics chain with its volume
error removed by one number. That nothing beats it is not a failure of the
models — it is the statement that volume was the error.

**Peak timing for persistence was an artefact.** A flat forecast has no peak;
`argmax` on a constant array returns 0. Now guarded.

---

## Working practice

**Static checks are less reliable than running the thing.** Three times a grep
or an import graph said a file was safe to remove and the smoke test disagreed —
most sharply `cnn/train_gbm.py`, whose removal broke ten modules including the
entire LSTM chain.

**Provisioning scripts look dead to a dependency graph.** Nothing imports
`merge_inflow_2015_2022.py`, but it builds the inflow record. Import-reachability
is the wrong test for a data pipeline.

**`io.open(path, 'w').write(expr)` truncates before evaluating `expr`.** An
exception inside the argument leaves a zero-byte file. Write via a temporary file
and `os.replace`.
