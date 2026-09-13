# Pipeline 1 Explained — the physics route, stage by stage

**For someone who knows neither hydrology nor machine learning.** Every term is
explained the first time it appears. Every number comes from the project's own
results files as they stand today.

```
ECMWF forecast  →  CNN downscaling  →  VIC  →  RVIC  →  LSTM  →  inflow forecast
```

> **Numbers here are current as of 13 September 2026.** If you have an older
> draft of the documentation, several of its headline figures are now wrong —
> see "What changed recently" at the end. Results live in
> `results/metrics/head_to_head.json`; this document explains them.

---

## 0. The problem, in plain terms

You run Hirakud Dam during the monsoon. Rain is forecast over the hills
upstream. You must decide **today** whether to open the gates.

Release too little and heavy rain arrives: the reservoir fills, you dump a huge
volume at once, and the towns below flood. Release too much and the rain never
comes: you have thrown away water the region needed for irrigation and power.

The catch is that by the time you can *measure* the water arriving, the decision
that mattered was made days ago. So you need to know what is coming **before**
it arrives.

**What we predict:** how much water enters the reservoir each day, in cubic
metres per second (m³/s), for the next 30 days.

**Two pieces of vocabulary you need throughout:**

- **Initialisation date** — the day the forecast is issued. Everything the model
  is allowed to know must be known on or before this date.
- **Lead day** — how far ahead. Lead day 1 is tomorrow; lead day 30 is a month
  away.

The catchment — the land that drains into the reservoir — is about
**83,400 km²**, roughly the size of Austria.

---

## 1. Why this is hard: rainfall is not river flow

Suppose we could forecast rainfall *perfectly*. Would we then know the inflow?

**No.** Four independent reasons, and they are why the pipeline needs four
stages instead of a multiplication.

### 1.1 The catchment has a memory

Pour a glass of water on a dry sponge: it soaks in. Pour the same glass on a
sponge that has sat in a bucket all week: it runs straight off.

A catchment behaves exactly like the sponge. What happens to rain depends on
**how wet the ground already was** — hydrologists call this *antecedent
moisture*.

100 mm falling on dry soil in early June may almost entirely soak in. The same
100 mm in August, after weeks of monsoon, becomes almost pure runoff.

This is why the pipeline cannot be "rainfall × constant = inflow", and why the
hydrological model in stage 3 must be started from a **saved soil state** rather
than from scratch.

**How long is the memory?** Measured on this basin: inflow today correlates
**+0.48** with inflow a week ago, **+0.22** with a month ago, and **−0.01** with
60 days ago. So the catchment remembers about a week, forgets almost everything
by a month, and 60 days is pure noise. That number becomes important in stage 5.

### 1.2 The forecast grid is far too coarse

One ECMWF grid cell at this latitude is about **167 × 156 km — roughly
26,000 km²** — and the forecast gives a **single rainfall number** for all of it.

The entire catchment is therefore about **three grid cells' worth of area**.

And because runoff is not proportional to rainfall, averaging is not merely
imprecise — it is *wrong*:

> **runoff(average rainfall) ≠ average of runoff(rainfall)**

Concretely: 100 mm concentrated in one quarter of a cell saturates that quarter
and produces heavy runoff. The same total spread evenly as 25 mm everywhere may
soak in entirely and produce almost nothing. **Same water, completely different
river response.** Averaging before running the physics systematically loses
flood peaks.

That is the argument for stage 2.

### 1.3 Water takes time to travel

Rain and runoff are a **depth** — millimetres over an area. Inflow is a
**discharge** — a volume past one point each second, m³/s.

Converting needs the **area** of each cell, and the **travel time**, because
water does not teleport. Rain falling 300 km upstream does not arrive that
afternoon. So today's inflow is a *mixture*: nearby storms from yesterday plus
distant storms from days ago.

That is the argument for stage 4.

### 1.4 Uncertainty grows with lead time

The forecast centre does not issue one forecast. It issues **eleven slightly
different versions**, called *ensemble members*, because tiny differences in
today's atmosphere grow into large differences a week later. How much they
disagree — the *spread* — is a real measure of confidence.

---

## 2. Stage 1 — the ECMWF forecast

The starting point is a forecast from the **European Centre for Medium-Range
Weather Forecasts**, generally regarded as the best in the world.

| property | value |
|---|---|
| grid | 1.5° (~167 × 156 km), a 7 × 7 box |
| lead days | 1 to 30 |
| ensemble members | 11 (1 control + 10 perturbed) |
| weather variables | 26 |
| forecast dates | **532** across 19 monsoons |

532 dates × 30 lead days = **15,960 rows**. When you see "about 16,000 samples",
this is it.

**Why only 28 dates per monsoon?** The archive runs Mondays and Thursdays only —
31 dates in June–September. Three of them (12, 19, 26 July) are Fridays in this
model version and the server rejects them. They do not exist; it is not a
download to retry.

### The three traps

Raw data from a weather centre is not ready to use.

**Trap 1 — rainfall accumulates.** Total precipitation is reported as a *running
total since the forecast was issued*, like a water meter: 2.4 mm at day 1,
238 mm by day 30. Feed that in raw and the model believes 238 mm fell on day 30
alone. It must be differenced day by day, per member. The project's own comment
calls missing this *"the classic way to get a downscaler that looks brilliant
and is useless."*

**Trap 2 — cloud cover is a percentage,** 0–100, not the 0–1 fraction its name
suggests.

**Trap 3 — two variables once carried no forecast information.** `t2m`
(temperature) and `tcc` (cloud) came back as a *single* day-1 field copied
across all 30 leads, so day 30 equalled day 1. **This has been fixed** — they
are now requested as 24-hour period windows — and verified across all 532
initialisations. Both now vary with lead and correlate with rainfall in the
physically right direction (cloud **+0.63**, temperature **−0.40**).

---

## 3. Stage 2 — the CNN downscaler

### What it does

Instead of predicting the whole rainfall map at once, the network predicts **one
fine cell at a time**, from a small patch of coarse weather *around* that cell
plus the cell's own geography. Run it for all 224 catchment cells and you have
rebuilt the map at 0.25° (~25 km) instead of 1.5° (~150 km).

**Input:** a `(26, 3, 3)` weather patch — all 26 variables on a 3 × 3 block of
coarse cells centred on the target — plus a `(3,)` vector of latitude,
longitude and elevation.

**Output:** one number, rainfall for that cell in mm/day.

### Why a 3 × 3 patch

Not arbitrary — the source paper tested 1×1, 3×3, 5×5 and 7×7 and found 3×3
best. A human forecaster does the same thing: they look at the weather system
*around* a valley, not just the humidity directly overhead.

### Why interpolation would not do

The obvious alternative is to smooth the coarse map onto a finer grid.
**Interpolation adds pixels, not information.** It cannot know that one cell sits
on a windward ridge where air is forced up and rain falls, while its neighbour
sits in a rain shadow. The network can, because it sees terrain and the
surrounding pattern.

### The architecture

A **ResNet** — a network whose key trick is the *skip connection*: alongside the
usual processing, the original input is added back to the output, so a deep
network does not lose track of what it started with.

| stage | operation | output shape |
|---|---|---|
| input | weather patch | (B, 26, 3, 3) |
| block 1 | ResBlock 26 → 64 | (B, 64, 3, 3) |
| block 2 | ResBlock 64 → 32 | (B, 32, 3, 3) |
| block 3 | ResBlock 32 → 16 | (B, 16, 3, 3) |
| flatten | | (B, 144) |
| coords | Linear(3 → 16) + ELU | (B, 16) |
| join | concatenate | (B, 160) |
| head | 160 → 64 → 32 → 1, **softplus** | (B, 1) |

Four choices worth understanding:

- **Padding 1 on every convolution** — the patch is only 3 × 3; without padding
  it would vanish after one layer.
- **Spatial dropout**, which zeroes *whole weather channels* rather than
  individual pixels, so the network cannot lean on any single variable.
- **Coordinates enter as an embedding**, not as extra image channels, so the
  network can say "this place behaves differently" without the convolutions
  having to encode absolute position.
- **Softplus output** — rainfall cannot be negative, and unlike ReLU it does not
  produce dead gradients on dry days, which dominate the data.

### The loss function, and a finding about it

Training minimises:

```
Loss = MSE + b × (1 − TS)        with b = 1.0
```

**MSE** is ordinary squared error. **TS** is the *threat score*, a standard
measure of catching heavy rain: `hits / (hits + false alarms + misses)`, where
"heavy" means above that cell's 90th percentile.

The threat-score term exists because plain squared error would push the network
to predict "not much rain" always — dry days vastly outnumber wet ones, so
hedging is safe. The TS term is supposed to make hedging expensive.

> **Measured finding (September 2026): on this basin the TS term does nothing.**
>
> | term | value |
> |---|---|
> | MSE | **265.3 mm²** |
> | b × (1 − TS) | **1.00** |
> | **TS share of the loss** | **0.375 %** |
>
> MSE is in mm² and reaches hundreds; `(1 − TS)` is capped at 1. The
> extreme-rain term contributes under **0.4 %** of the gradient, and the threat
> score itself is **0.000** — 92 hits against 357,412 misses.
>
> This was tested properly. Raising `b` from 1 to **66** changed nothing.
> Widening the decision boundary so that 100 % of cells had live gradient
> (instead of 0.19 %) also changed nothing. The reason is that **the trained
> network's maximum prediction is ~26 mm**, against a heavy-rain threshold of
> **24.46 mm** and observed maxima of **575 mm**. It cannot reach the threshold,
> so no reweighting of a threshold term can help.
>
> **Under squared error with a weakly-informative input, collapsing toward the
> average is the mathematically correct behaviour.** It is not a bug. It is also
> why the source paper's "specialised loss" did not transfer here.

### What the downscaler achieved

**Per-cell R² across 19 folds:** mean **+0.0571**, positive in **19 of 19
years**. Raw ECMWF at the same cells scores **−0.136** — *worse than ignoring
the forecast*. So the sign flipped, every year.

**But at catchment scale, it is worse than raw ECMWF:**

| lead | corr(CNN rain, observed) | corr(raw EC, observed) |
|---|---|---|
| 1–3 d | 0.574 | **0.637** |
| 4–7 d | 0.454 | **0.509** |
| 8–14 d | 0.365 | **0.400** |
| 15–21 d | 0.350 | **0.362** |
| 22–30 d | **0.385** | 0.376 |

**This is the single most important thing to understand about stage 2.** The
network is trained to place rain *in space*. VIC integrates the basin *total* —
and on the total, raw ECMWF is better at every lead but the last.

And it damps extremes. Rainfall variability, as a fraction of observed:

| lead | raw EC | **CNN** |
|---|---|---|
| 1–3 d | 0.696 | **0.431** |
| 4–7 d | 0.536 | **0.385** |

The CNN keeps only 43 % of the real variability where raw ECMWF keeps 70 %.
That is the damping described above, and it is what drives the low flood-peak
scores later.

---

## 4. Stage 3 — VIC, the hydrological model

**VIC** (Variable Infiltration Capacity) is a physical model of what happens to
water when it lands. For each of **150 cells**, every 6 hours, it solves a
*water balance*: rain arrives, some evaporates, some soaks into layered soil,
some runs off the surface, some drains slowly out of the bottom as baseflow.

It is started from a **saved soil state** for each forecast date — that is how
section 1.1's "memory" enters. A forecast cannot cold-start.

### Calibration: teaching it this particular basin

VIC has free parameters — soil depths, how fast water drains, how deep roots
reach — that cannot be measured directly. They are fitted.

| period | role |
|---|---|
| 2003 | spin-up, discarded |
| **2004–2011** | **calibration**, every observed day — 2,740 days |
| 2012–2014 | validation, never scored during the search |
| 2015–2022 | held out entirely |

Eight parameters, fitted by differential evolution (a search that tries many
combinations and keeps what works), 432 evaluations, 135 minutes.

The objective rewards matching observed flow **and** getting the water
*partition* physically right:

```
KGE(discharge) − 2 × [ hinge(runoff coefficient, 0.35–0.40)
                     + hinge(ET/P,               0.55–0.65) ]
```

Flow is scored where it is **observed** (daily); the partition where it is
**defined** (annually).

### How well it works

| split | NSE | correlation | bias | days |
|---|---|---|---|---|
| calibration 2004–2011 | 0.581 | 0.931 | **+50.3 %** | 944 |
| validation 2012–2014 | 0.328 | 0.914 | **+58.4 %** | 344 |
| **held out 2015–2022** | 0.416 | **0.937** | **+72.1 %** | 976 |

**Two things to read here.**

**The timing generalises.** Correlation is *highest on the years VIC never saw*
(0.937). It is not memorising — it genuinely knows when water arrives.

**The volume is badly wrong, and gets worse over time.** Bias grows +50 % →
+58 % → +72 %. A calibration error would be roughly flat. A bias that **grows
with time** is a clue, and it points at something VIC cannot represent.

### The missing process: people take the water

| | mm/yr | runoff coefficient |
|---|---|---|
| rainfall | 1,281.9 | — |
| **observed inflow** | 366.2 | **0.286** |
| published *natural* value | 449–513 | 0.35–0.40 |

**80–150 mm/yr — 18–29 % of natural runoff — never reaches the gauge.** The
Mahanadi above Hirakud is heavily irrigated. VIC simulates *natural* runoff and
has **no abstraction term**, so it cannot represent water that people divert.

This explains the growing bias (irrigation has expanded), and it explains
something that looked like a modelling failure: when the calibration was pushed
to match observed volume, it responded by growing an **8-metre soil column** —
trying to reproduce human water use by evaporating the difference.

**Do not "fix" a positive VIC bias by widening a soil bound. It is abstraction.**
That warning is now written into `calibrate_vic.py`.

Annual water balance today: runoff coefficient **0.445** (target 0.35–0.40),
ET/P **0.558** (in band), closure **1.003** (water is conserved).

---

## 5. Stage 4 — RVIC routing

VIC gives **runoff per cell, in millimetres**. We need **discharge at one point,
in m³/s**. That is stage 4's whole job, and it is two conversions.

**Depth → volume.** Multiply by the cell's area and the fraction of it inside
the basin:

```
to_cms = 1e-3 × area × basin_fraction / 86400
```

This is the *only* place catchment geometry enters the numbers.

**Now → later.** Water from a distant cell arrives over several days, not at
once. RVIC represents this with a **unit hydrograph**: a small curve saying what
fraction of a cell's runoff arrives after 1 day, 2 days, and so on, out to
**10 days**. The river network comes from terrain (Dominant River Tracing), so
travel times are physical, not fitted.

**One safeguard worth knowing.** The 10-day memory means inflow on forecast day
1 partly depends on runoff from *before* the forecast. That history is taken
from the observed-weather run — it is what was already in the rivers at issue
time, not a peek at the future. Verified: the splice uses exactly the 10 days
**ending on** the initialisation date.

---

## 6. Stage 5 — the LSTM correction

An **LSTM** is a neural network for sequences, able to carry information across
many time steps. Here it does *not* forecast. It **corrects** what VIC produced.

**Inputs:** recent observed inflow (the hydrograph up to the issue date), the
routed VIC forecast, and the forecast rainfall belonging to that product.
**Output:** a corrected 30-day hydrograph.

### The n_seq finding — a real improvement

How many days of history should it read? The source paper calls this an
*"optimised hyperparameter"* and notes it should be *large in snow-affected
basins*, where melt spans hundreds of days. This project had it at **60 days** —
a value borrowed from a setting that does not apply. **The Mahanadi is
monsoon-driven, not snow-fed.**

Two things were wrong with 60:

- **It carried no information.** Inflow autocorrelation at 60 days is −0.013.
  Days 30–60 were pure noise inputs.
- **It threw away data.** A forecast needs that much *continuous* history to be
  usable, so 60 days left only **311 of 532** initialisations. Worse, from 2015
  the record starts in June, so the discarded ones were disproportionately
  **early-monsoon** — the onset, which operators most need.

Sweeping it, scored on the common 309 windows:

| n_seq | median NSE | initialisations usable |
|---|---|---|
| 7 | 0.192 | 495 |
| **15** | **0.200** | **476** |
| 30 | 0.146 | 432 |
| 45 | 0.177 | 376 |
| **60 (old default)** | **0.140** | **311** |

**Changing 60 → 15 lifted the chain from 0.140 to 0.200** — 53 % more training
data and less noise. It is now the default.

*Honest caveat:* 7 and 15 are statistically indistinguishable (p = 0.59). The
solid claim is **"short history, ~7–15 days"**, not an optimum at 15.

### What the LSTM actually is

**A bias-correction layer, not a forecasting layer.** Three independent tests
agree:

1. Its gain tracks how much bias is left upstream — **18 of 19 years** where the
   bias is large, fewer where it is small.
2. Remove the volume error *first* with a simple scalar, and the LSTM becomes
   **actively harmful** (7/19 wins). There is no useful residual shape left for
   it to learn.
3. On the median metric it is beaten by a **single multiplication**.

But it earns its place on the *mean*: it rescues catastrophic years (+0.38,
+0.30, +0.20 in the worst) while costing at most −0.06 in good ones. **Median
alone systematically undervalues a corrector.**

---

## 7. What the whole chain achieves

Scored on 309 windows common to every model, 19 years, ensemble mean:

| model | median NSE | mean | bias % | flood peak |
|---|---|---|---|---|
| persistence (today's flow, held flat) | −1.023 | −1.241 | +21.1 | 0.385 |
| **`0.86 × EC` — the baseline to beat** | **0.168** | 0.069 | −2.9 | 0.794 |
| `EC → VIC → RVIC` | −0.010 | −0.433 | +31.8 | **1.081** |
| `EC-CNN → VIC → RVIC` | 0.161 | −0.075 | +0.7 | 0.833 |
| `EC → VIC → RVIC → LSTM` | 0.118 | 0.070 | −11.1 | 0.744 |
| **`EC-CNN → VIC → RVIC → LSTM`** | **0.200** | 0.040 | −16.0 | 0.666 |

**The full chain — the paper's proposed system — is the best physics
configuration**, and it now exceeds the 0.168 baseline.

**But it is still not *significantly* better:** t = −0.40, p = 0.695, winning in
9 of 19 years. The honest sentence is *"no longer significantly worse, and the
highest-scoring physics chain"* — not *"beats the baseline"*.

### The diagnostic that explains everything

Drive the identical chain with **observed** weather instead of forecast weather.
That is the ceiling no forecast can exceed:

| | median NSE |
|---|---|
| perfect rainfall, as simulated | 0.394 |
| **perfect rainfall + one volume correction** | **0.850** |
| perfect rainfall + optimal timing shift | 0.471 |
| *actual chain with forecast rainfall* | 0.200 |

Read those four numbers carefully, because they contain the whole story.

**VIC + RVIC is not the weak link.** Given the right water volume, it reproduces
30-day inflow at **0.850**. The hydrology works.

**The error is volume, not timing.** The mean optimal time shift is **0.28
days** — timing is essentially perfect. Shifting gains 0.394 → 0.471. One scalar
gains 0.394 → **0.850**.

> ⚠️ **An earlier version of this documentation said the opposite** — that error
> was "dominated by phase, which no downstream model can fix". That was measured
> on the *forecast* chains, where the timing error belongs to ECMWF, not VIC.
> Two different things were measured together and blamed on the model. **If you
> have an older draft, this is the correction that matters most.**

**The gap from 0.850 to 0.200 is the rainfall forecast**, and it is not
recoverable by anything downstream.

### Skill by lead time

| band | `0.86 × EC` | full chain | what is actually happening |
|---|---|---|---|
| 1–3 d | 0.699 | 0.587 | catchment memory — persistence alone scores 0.507 |
| 4–7 d | 0.049 | **0.275** | **where the pipeline earns its keep** |
| 8–14 d | 0.107 | 0.100 | marginal |
| 15–21 d | 0.102 | 0.068 | marginal |
| 22–30 d | −0.009 | −0.010 | zero — climatology does better (0.042) |

**The pipeline's value is concentrated in days 4–14.** Inside day 3 it is
catchment memory, which needs no weather forecast. Beyond day 20, a day-of-year
average beats it.

---

## 8. The honest limits

**Rainfall is the binding constraint.** Catchment rainfall correlates 0.35–0.40
with reality beyond a week. Everything downstream inherits that ceiling. No
post-processor creates information that is not in its input.

**VIC's parameters are *effective*, not physical.** They absorb upstream
abstraction the model cannot represent. `wpwp_scale` sits at its lower bound and
soil depths near their ceilings — the optimiser compensating for a missing
process. Do not read them as measured soil properties.

**42 % of the evaluation is partly in-sample.** VIC is fitted on 8 of the 19
years scored. `inflow/fair_comparison.py` reports the VIC-unseen split
separately.

**The LSTM uses 89 % of forecasts** (476 of 532 at n_seq = 15) — much better
than the 58 % at n_seq = 60, but still not all.

**One basin, 19 monsoons, 309 windows.** Nothing here generalises beyond the
Mahanadi above Hirakud without testing.

---

## 9. What changed recently

If you are holding an older draft, these are the claims that have been
**overturned by measurement**. All are documented in `docs/FINDINGS.md`.

| older claim | now |
|---|---|
| "error is dominated by timing/phase" | **Wrong.** Optimal shift is 0.28 days; the error is volume |
| "the CNN improves the inflow forecast" | **Wrong.** At equal volume it scores 0.157 vs raw ECMWF's 0.168 — its apparent lead was two biases cancelling |
| "the LSTM adds nothing" | **Wrong.** It is a bias layer; its gain scales with upstream bias |
| LSTM history = 60 days | **15 days** — lifted the chain 0.140 → 0.200 |
| the specialised TS loss balances extremes | **It does not function here** — 0.375 % of the loss, and unreachable |
| headline chain ≈ 0.14 | **0.200** after the n_seq fix |

---

## Where to go next

| document | purpose |
|---|---|
| `docs/PROJECT_DOCUMENTATION.md` | the whole project, both pipelines, concise |
| `docs/FINDINGS.md` | research log — what was discovered, in order, including what was wrong |
| `STRUCTURE.md` | which file belongs to which stage |
| `results/metrics/head_to_head.json` | the numbers themselves |

To watch one forecast travel through all five stages with units at each step:

```bash
python tools/trace_one_forecast.py 2018-07-04
```
