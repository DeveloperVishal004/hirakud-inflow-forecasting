# Pipeline 1 Explained — the physics route, stage by stage

**Written for someone who knows neither hydrology nor machine learning.**
Every term is explained the first time it appears. Every number comes from this
project's own results files.

```
ECMWF forecast → CNN downscaling → VIC → RVIC → LSTM → inflow forecast
```

> **Current as of 14 September 2026.** If you are holding an older draft of the
> documentation, six of its headline claims have since been overturned by
> measurement — §14 lists them. The most important: that draft says the error is
> dominated by *timing*. It is dominated by *volume*.

---

## Contents

| § | | |
|---|---|---|
| 1 | [The problem](#1-the-problem) | why a dam needs a forecast at all |
| 2 | [Why rainfall is not river flow](#2-why-rainfall-is-not-river-flow) | the four reasons this needs five stages |
| 3 | [How we measure success](#3-how-we-measure-success) | NSE, KGE, bias, peak ratio |
| 4 | [Stage 1 — ECMWF](#4-stage-1--the-ecmwf-forecast) | the raw ingredient, and its three traps |
| 5 | [Stage 2 — CNN downscaling](#5-stage-2--the-cnn-downscaler) | sharpening the rainfall map |
| 6 | [Stage 3 — VIC](#6-stage-3--vic-the-hydrological-model) | rainfall becomes runoff |
| 7 | [Stage 4 — RVIC](#7-stage-4--rvic-routing) | runoff becomes inflow |
| 8 | [Stage 5 — LSTM](#8-stage-5--the-lstm-correction) | correcting what is left |
| 9 | [One forecast, end to end](#9-one-forecast-end-to-end) | a fully worked example |
| 10 | [How it is tested](#10-how-it-is-tested) | leave-one-year-out, and why |
| 11 | [What the chain achieves](#11-what-the-chain-achieves) | the results |
| 12 | [Why it is not better](#12-why-it-is-not-better) | the diagnostic that explains everything |
| 13 | [Bugs, and what they taught](#13-bugs-and-what-they-taught) | six defects worth knowing |
| 14 | [What changed recently](#14-what-changed-recently) | claims that were overturned |
| 15 | [Questions you may be asked](#15-questions-you-may-be-asked) | and honest answers |
| 16 | [Glossary](#16-glossary) | every term in one place |
| 17 | [How to run it](#17-how-to-run-it) | commands, in order |

---

## 1. The problem

### The operator's dilemma

You run Hirakud Dam during the monsoon. Rain is forecast over the hills
upstream. You must decide **today** whether to open the gates.

**Release too little,** and if heavy rain arrives the reservoir fills. Then you
have no choice: dump a huge volume at once, and the towns below flood.

**Release too much,** and if the rain never comes you have thrown away water the
region needed for irrigation and hydropower.

Here is what makes it genuinely hard. By the time you can *measure* water
arriving at the reservoir, the decision that mattered was made days ago. The
water is already there. To operate safely you must know what is coming **before**
it arrives.

### What exactly is predicted

**Reservoir inflow**: the volume of water entering the reservoir each day,
measured in **cubic metres per second** (m³/s, also called *cumecs*). One m³/s is
a thousand litres every second.

For scale: this basin's inflow ranges from a couple of hundred m³/s in quiet
spells to over 20,000 m³/s in a flood.

Two pieces of vocabulary used throughout:

- **Initialisation date** — the day the forecast is issued. Everything the model
  is allowed to know must be known on or before this date. Nothing after it.
- **Lead day** — how far ahead. Lead day 1 is tomorrow; lead day 30 is a month
  out.

Forecasts are made only during the monsoon (June–September), because that is
when essentially all the water arrives.

### Where

Hirakud Dam sits on the Mahanadi river at 21.52° N, 83.87° E in Odisha, eastern
India.

The **catchment** — the land that drains into the reservoir — is about
**83,400 km²**, roughly the size of Austria. It reaches west into Chhattisgarh
and contains uplands, forest and valley floor. Different parts respond very
differently to the same rain, which matters enormously in §2.

---

## 2. Why rainfall is not river flow

Suppose we could forecast rainfall **perfectly** — every millimetre, in the right
place, on the right day, a month ahead.

Would we then know the inflow?

**No.** Four independent reasons. Each one is why a particular stage of the
pipeline exists.

### 2.1 The catchment has a memory

Pour a glass of water onto a dry kitchen sponge. Almost all of it soaks in.
Barely anything drips out.

Now take a sponge that has been sitting in a bucket for a week and pour the same
glass on it. Almost all of it runs straight off.

A catchment behaves **exactly** like the sponge. What happens to rain depends
entirely on **how wet the ground already was** — hydrologists call this
*antecedent moisture*.

100 mm falling on dry soil in early June may almost entirely soak in, producing
very little river flow. The same 100 mm in August, after weeks of monsoon have
saturated the ground, becomes almost pure runoff and surges into the river.

**Measured on a real forecast** (4 July 2018), the fraction of falling rain that
became runoff:

| lead day | rain (mm/day) | runoff (mm) | **runoff ratio** |
|---|---|---|---|
| 1 | 9.23 | 1.92 | **0.21** |
| 3 | 14.17 | 1.90 | **0.13** |
| 7 | 20.18 | 2.46 | **0.12** |
| 14 | 16.78 | 3.36 | **0.20** |
| 21 | 18.36 | 3.93 | **0.21** |
| 30 | 16.81 | 5.37 | **0.32** |

Look at days 3 and 7: **more rain, less runoff ratio.** The ratio is not a
function of how much rain falls. It is a function of the *accumulated state of
the catchment*.

**Consequence for the pipeline:** the hydrological model cannot start from
scratch. It must be started from a **saved soil state** for each forecast date —
a snapshot of how wet every layer of soil already was. This is stage 3's most
important design feature.

### How long is the memory?

Measured on this basin — the correlation between inflow today and inflow N days
ago:

| gap | correlation |
|---|---|
| 1 day | **+0.911** |
| 3 days | +0.697 |
| 7 days | **+0.484** |
| 14 days | +0.357 |
| 30 days | +0.217 |
| **60 days** | **−0.013** |

The catchment remembers about a week strongly, a month weakly, and **60 days not
at all**. That last number becomes decisive in stage 5 — the original design fed
the correction network 60 days of history, of which the last 30 carried nothing.

### 2.2 The forecast grid is far too coarse

At this latitude, one ECMWF grid cell measures roughly **167 × 156 km — about
26,000 km²** — and the forecast reports a **single rainfall number** for all of
it.

The entire 83,400 km² catchment is therefore about **three grid cells' worth of
area.** Three numbers describing an area the size of Austria.

**Why averaging is not merely imprecise — it is wrong.** Because runoff is not
proportional to rainfall:

> **runoff(average rainfall) ≠ average of runoff(rainfall)**

Concretely: suppose 100 mm falls, concentrated in one quarter of a cell. That
quarter saturates and produces heavy runoff. Now suppose the same total is spread
evenly — a gentle 25 mm everywhere. At 25 mm the soil may absorb all of it,
producing almost nothing.

**Same total water. Completely different river response.**

Averaging rainfall before running the physics **systematically loses flood
peaks** — exactly the events a dam operator cares about most.

**Consequence:** the catchment must be broken into finer pieces. Here it is 224
cells at 0.25° (~25 km), and the water balance runs on **150 active cells**. That
is the argument for stage 2.

### 2.3 Water takes time to travel

Rain and runoff are measured as a **depth** — millimetres spread over an area.

Inflow is measured as a **discharge** — a volume passing one point every second,
in m³/s.

Converting between them needs two things:

1. the **area** of each cell, to turn a depth into a volume
2. the **travel time**, because water does not teleport

Rain falling 300 km upstream does not arrive at the dam that afternoon. It works
down through streams into tributaries into the main channel. So today's inflow is
a **mixture**: water from nearby storms yesterday, plus distant storms several
days ago.

**Consequence:** that is the argument for stage 4.

### 2.4 Uncertainty grows with lead time

The weather centre does not issue *one* forecast. It issues **eleven slightly
different versions**, called **ensemble members**, produced by nudging today's
atmospheric state within its measurement uncertainty and re-running.

Small differences today grow into large differences a week out. How much the
eleven disagree — the **ensemble spread** — is a genuine measure of confidence.

From the same 4 July 2018 forecast:

| lead day | forecast rain (mm/day) | **spread** |
|---|---|---|
| 1 | 9.23 | **2.51** |
| 3 | 14.17 | 6.57 |
| 7 | 20.18 | 8.96 |
| 14 | 16.78 | 9.91 |
| 21 | 18.36 | **12.74** |

Spread grows from 2.5 to 12.7 — the forecast openly losing confidence. And it is
*informative*: across the whole dataset, the correlation between spread and the
size of the eventual inflow error is **+0.475**. When the ensemble disagrees, the
forecast really is less reliable.

---

## 3. How we measure success

Before any results, you need to read the scoreboard.

### NSE — Nash-Sutcliffe Efficiency

The main metric throughout hydrology.

```
NSE = 1 − (sum of squared errors) / (sum of squared deviations from the mean)
```

You do not need the formula. You need the interpretation:

| NSE | meaning |
|---|---|
| **1.0** | perfect |
| **0.0** | exactly as good as always predicting the long-run average |
| **negative** | **worse than predicting the average** |

**An NSE of 0 is not "half right". It means the model added nothing at all.**
This matters because several numbers in this project sit near zero, and one
baseline is around −1.

### KGE — Kling-Gupta Efficiency

NSE has a known weakness: it is dominated by correlation and barely punishes
getting the *volume* wrong. A model with perfect timing and 50 % too much water
can still score well.

KGE splits performance into three parts — correlation, variability, and bias —
and penalises all three. This project uses KGE for **calibration** precisely
because volume error is its main failure mode.

### Bias

`(mean forecast − mean observed) / mean observed`, as a percentage. Positive
means over-predicting. This turns out to be the single most important number in
the whole project.

### Peak ratio

`forecast peak / observed peak`, averaged over forecast windows. **1.0** is
perfect. **Below 1.0** means under-predicting floods — which for a dam operator
is the dangerous direction.

### Median, mean, and win-count — report all three

A model that is usually decent but occasionally catastrophic has a good median
and a terrible mean. A model that rescues disasters has a good mean and an
unremarkable median.

**Any one of the three can mislead.** This project reports all three, because a
correction layer that rescues bad years is invisible in the median — a real
finding we nearly missed.

---

## 4. Stage 1 — the ECMWF forecast

### What it is

The starting point is a forecast from the **European Centre for Medium-Range
Weather Forecasts (ECMWF)**, generally regarded as the best in the world.

Specifically, the **S2S reforecast**. Two terms:

**S2S** = *sub-seasonal to seasonal*, roughly two weeks to two months ahead. This
is the hardest range in weather forecasting. Short-range forecasts work because
today's atmosphere still dominates. Seasonal forecasts work because slow things
like ocean temperature dominate. **Sub-seasonal sits in the gap where neither
helps much.**

**Reforecast** (or *hindcast*) means the centre took its *current* model and
re-ran it on *past* dates. That gives a consistent archive across many years from
one model version — essential if you want to train on history and apply it today.

### What was downloaded

| property | value |
|---|---|
| grid | **1.5°** (~167 × 156 km), a 7 × 7 box |
| region | 16.5–25.5° N, 79.5–88.5° E |
| lead days | 1 to 30 |
| ensemble members | **11** (1 control + 10 perturbed) |
| weather variables | **26** |
| dates per monsoon | **28** |
| **total forecast dates** | **532** |

532 dates × 30 lead days = **15,960 rows**. When you see "about 16,000 samples"
anywhere, this is what it means.

**The 26 variables** are 11 at the surface (total and convective precipitation,
temperature, two wind components, pressure, cloud, three energy fluxes, terrain
elevation) plus 5 upper-air variables (wind ×2, humidity, temperature, geopotential
height) at three altitudes — 200, 500 and 850 hPa. 11 + 5×3 = 26.

**Why upper-air matters.** The project's config explains it in one line: *"a
surface-only forecast cannot place a storm in a particular 25 km cell."* Ground
pressure tells you little about where a storm will organise. Moisture and wind a
few kilometres up tell you a great deal.

**Why only 28 dates per monsoon.** The reforecast runs Mondays and Thursdays
only — 31 dates in June–September. Three of them (12, 19, 26 July) fall on
Fridays in this model version and the server rejects them outright. They do not
exist; it is not a download to retry.

### The three traps

Raw data from a weather centre is **not ready to use**. Three quirks will
silently destroy a model.

#### Trap 1 — rainfall totals accumulate

When ECMWF reports precipitation, it does **not** give that day's rain. It gives
the **running total since the forecast was issued**.

Think of a water meter. Day 1 reads 2.4 mm. By day 30 it reads 238 mm. Feed those
raw numbers to a model and it believes 238 mm fell on day 30 alone.

The fix is to subtract consecutive days, separately for each ensemble member:

```python
a = f[:, :, :, acc_idx]
a[:, 1:] = np.diff(a, axis=1)      # undo accumulation, per member
```

The project's own comment calls missing this *"the classic way to get a
downscaler that looks brilliant and is useless."* Not hypothetical — it was one
of six real defects found in an earlier version (§13).

The same applies to the three energy fluxes, which are additionally divided by
86,400 (seconds in a day) to become average rates.

#### Trap 2 — cloud cover is a percentage

Total cloud cover arrives as **0–100**, not the 0–1 fraction its name implies.

#### Trap 3 — two variables once carried no forecast information

**This was the most serious defect in the archive, and it has been fixed.** The
story is kept because it is the clearest example in this project of a bug that
produces no error message.

**What went wrong.** Temperature (`t2m`) and cloud cover (`tcc`) came back as a
**single day-1 field** instead of 30 daily values, then got copied across all 30
lead days. Temperature at day 30 equalled temperature at day 1. Two of 26
predictor channels were telling the model **nothing** about how conditions evolve.

**Why.** These two variables are *period-processed* in the S2S archive, not
instantaneous. The original request asked for instantaneous lead hours. Nine
surface fields honoured that; these two matched only the one window that happened
to align and returned a single field.

**The fix.** Request them as 24-hour **period windows** instead. All 30 come back.

**The verification**, on the file the network actually trains on — standard
deviation across the 30 lead days, for all 532 initialisations:

| channel | mean | minimum | |
|---|---|---|---|
| rainfall | 0.368 | 0.220 | varies, as it must |
| **temperature** | **0.295** | **0.130** | varies |
| **cloud** | **0.547** | **0.254** | varies |
| terrain elevation | **0.000** | 0.000 | constant — correct, it is static |

Terrain is the control. A broadcast field looks like *that* row. Neither
temperature nor cloud does, in any year.

**And they are physically coherent**, which a copied field could never be. Within
each forecast, across its 30 leads, averaged over all 532:

- **corr(cloud, rainfall) = +0.63** — cloud brings rain. Positive in **99.2 %**
- **corr(temperature, rainfall) = −0.40** — rain cools. Negative in **84.4 %**

A copied field gives correlation of exactly zero, every time.

---

## 5. Stage 2 — the CNN downscaler

### The idea

Ask a human forecaster to predict rainfall in one valley and they do not just
look at the humidity directly overhead. They look at the **surrounding system** —
a front moving in, wind pushing moist air up a ridge.

The network works the same way. Instead of predicting the whole map at once, it
predicts **one fine cell at a time**, using a small patch of coarse weather
*around* that cell plus the cell's own geography. Run it for all 224 catchment
cells and the map is rebuilt at 0.25° (~25 km) instead of 1.5° (~150 km).

### Why not just interpolate?

The obvious alternative is to smooth the coarse map onto a finer grid — bilinear
interpolation or similar.

**This cannot work, and the reason is fundamental: interpolation adds pixels, not
information.** It produces a smoother surface with the same information content.
It cannot know that one 25 km cell sits on a windward ridge where air is forced
upward and rain falls heavily, while its neighbour sits in the rain shadow behind
it. Nothing in the coarse number distinguishes them.

A trained network has two things interpolation does not:

1. **the spatial pattern** of the coarse field — the gradient and structure of
   the weather system, not just its magnitude at one point
2. **fixed terrain** — elevation and position, which never change and strongly
   control where monsoon rain falls

### Inputs and outputs

**Input A:** a weather patch, shape `(26, 3, 3)` — all 26 variables on a 3 × 3
block of coarse cells centred on the target cell.

**Input B:** a location vector, shape `(3,)` — latitude, longitude, elevation.

**Output:** a single number — rainfall for that cell, in mm/day.

**Why 3 × 3?** Not arbitrary. The source paper tested 1×1, 3×3, 5×5 and 7×7 and
found 3×3 best. It is a finding, not a default.

**A near-miss worth recording.** An earlier version downloaded only a 5 × 5 box of
coarse cells. For cells near the edge, a 3 × 3 patch runs off the grid — and the
code filled the gap by repeating the edge row. **48 % of training patches were
affected**: nearly half the data contained duplicated rows presented as real
measurements. Re-downloading on a 7 × 7 box fixed it completely (**0 % clamped**),
and the code now *asserts* this rather than assuming it.

### The architecture

A **ResNet** — short for *residual network*. Its key trick is the **skip
connection**: alongside the usual processing, the original input is added back to
the output. This lets a network go deep without losing track of what it started
with.

| stage | operation | output shape |
|---|---|---|
| input A | weather patch | (B, 26, 3, 3) |
| block 1 | ResBlock 26 → 64 | (B, 64, 3, 3) |
| block 2 | ResBlock 64 → 32 | (B, 32, 3, 3) |
| block 3 | ResBlock 32 → 16 | (B, 16, 3, 3) |
| flatten | 16 × 3 × 3 | (B, 144) |
| input B | coordinates | (B, 3) |
| embed | Linear(3 → 16) + ELU | (B, 16) |
| join | concatenate | (B, 160) |
| head | 160 → 64 → 32 → 1, **softplus** | (B, 1) |

Each ResBlock is: 3×3 convolution → ELU → spatial dropout → 3×3 convolution →
add the skip → ELU.

**Four design choices worth understanding:**

**Padding of 1 on every convolution.** The patch is only 3 × 3. Without padding it
would shrink to nothing after a single layer.

**Spatial dropout rather than ordinary dropout.** Ordinary dropout zeroes
individual pixels; on a 3 × 3 patch that removes almost nothing. Spatial dropout
zeroes **whole weather channels**, forcing the network not to depend on any single
variable.

**Coordinates enter as an embedding, not as extra image channels.** This lets the
network express *"this place behaves differently"* without making the convolutions
encode absolute position.

**Softplus at the output.** Rainfall cannot be negative. Softplus guarantees a
positive output while avoiding the dead gradients ReLU would produce on dry days
— and dry days dominate the data.

### The loss function, and an important finding

Training minimises:

```
Loss = MSE + b × (1 − TS)          with b = 1.0
```

**MSE** is mean squared error — ordinary prediction error.

**TS** is the **threat score**, a standard measure of catching heavy rain:

```
TS = hits / (hits + false alarms + misses)
```

where an event counts as "heavy" if it exceeds that cell's **90th percentile** of
observed rainfall, computed on training years only.

**Why include it?** Plain squared error would push the network to predict "not
much rain" always. Dry days massively outnumber wet ones, so hedging toward the
average is safe and scores well. The threat-score term is supposed to make hedging
expensive.

**Making a hard threshold differentiable.** The threat score is *categorical* — a
prediction either exceeds the threshold or does not — and categorical decisions
have no gradient, so a network cannot learn from them. The solution: replace the
*forecast* side of each decision with a smooth sigmoid centred on the threshold,
while the *observation* side stays a hard yes/no. That is fine, because no
gradient needs to flow back through observations.

**Every term is masked.** Cells where there is no observation contribute nothing
to the loss and nothing to the score. This is not a detail — in the earlier
version, missing cells were filled with **zeros**, making 24 % of the target grid
a free constant the model could "predict" perfectly (§13).

> ### Finding: on this basin, the threat-score term does nothing
>
> Measured on the trained network:
>
> | term | value |
> |---|---|
> | MSE | **265.3 mm²** |
> | b × (1 − TS) | **1.00** |
> | **TS share of the total loss** | **0.375 %** |
> | threat score | **0.000** (92 hits, 357,412 misses) |
>
> MSE is in mm² and reaches hundreds; `(1 − TS)` is capped at 1. The extreme-rain
> term contributes **under 0.4 %** of the gradient.
>
> **This was tested properly, twice.**
>
> *Hypothesis 1 — it is under-weighted.* Raising `b` from 1 to **66** (enough to
> make TS ~20 % of the loss): **no change.** TS stayed 0.0007.
>
> *Hypothesis 2 — the sigmoid is saturated.* The threshold is 24.46 mm while the
> median prediction sits 15.95 mm below it, so 99.74 % of cells were in the flat
> tail with no gradient. Widening the sigmoid until **100 %** of cells had live
> gradient: **still no change.** TS went to 0.0002 — slightly worse.
>
> **Why neither worked.** The trained network's **maximum prediction is ~26 mm**,
> against a threshold of 24.46 mm and observed maxima of **575 mm**. It cannot
> reach the threshold, so no reweighting of a threshold term can matter.
>
> **The deeper reason.** Under squared error with a weakly-informative input,
> collapsing toward the conditional average **is the mathematically correct
> behaviour**. It is not a bug. Predicting 200 mm when you are right 40 % of the
> time is punished enormously; predicting 8 mm always is safe.
>
> This is a substantive negative result: the source paper credits its specialised
> loss for balancing extremes, and **on this basin it does not function**.

### What the downscaler achieved

**Per-cell R², across 19 leave-one-year-out folds:**

| | |
|---|---|
| mean | **+0.0571** |
| median | +0.0589 |
| worst fold | +0.0097 |
| folds positive | **19 of 19** |

Raw ECMWF at the same cells scores **−0.136** — *worse than ignoring the forecast
entirely and predicting the long-run average*. **The sign flipped, in every single
year.** That is a real achievement at the cell level.

### But at catchment scale it is worse than raw ECMWF

**This is the single most important thing to understand about stage 2.**

| lead | corr(CNN rain, observed) | corr(raw EC, observed) |
|---|---|---|
| 1–3 d | 0.574 | **0.637** |
| 4–7 d | 0.454 | **0.509** |
| 8–14 d | 0.365 | **0.400** |
| 15–21 d | 0.350 | **0.362** |
| 22–30 d | **0.385** | 0.376 |

The network is trained to place rain **in space**. VIC integrates the **basin
total** — and on the total, raw ECMWF is better at every lead but the last.

**And it damps extremes.** Rainfall variability as a fraction of observed:

| lead | raw EC | **CNN** |
|---|---|---|
| 1–3 d | 0.696 | **0.431** |
| 4–7 d | 0.536 | **0.385** |
| 8–14 d | 0.475 | 0.396 |

The CNN retains only **43 %** of real variability where raw ECMWF keeps 70 %.
Its 95th percentile is about **55 %** of the observed 95th percentile — it
systematically fails to produce heavy rain.

That damping is the direct cause of poor flood-peak scores later, and it is the
mathematically correct response to a low-skill input under squared error, as
explained above.

---

## 6. Stage 3 — VIC, the hydrological model

### What VIC is

**VIC** stands for *Variable Infiltration Capacity*. It is a physical model of
what happens to water after it lands — not a statistical one. Nothing in it is
learned from inflow data except a handful of tuning parameters.

For each of **150 grid cells**, every **6 hours**, VIC solves a **water balance**.
Water arrives as rain. Then:

- some **evaporates** directly, or is transpired by plants
- some **soaks in**, moving down through three soil layers
- some **runs off the surface** immediately — this happens fast
- some **drains slowly** out of the bottom layer as **baseflow** — this is what
  keeps rivers flowing weeks after the rain stopped

The "variable infiltration capacity" in the name is the model's central idea:
within a single cell, different patches of ground saturate at different rates, so
as a cell wets up, an increasing *fraction* of it produces runoff. That is what
makes the response non-linear — §2.1's sponge, expressed as equations.

**Crucially, VIC is started from a saved soil state** for each forecast date — a
snapshot of moisture in every layer of every cell at the moment the forecast is
issued. This is how the catchment's memory enters the pipeline. A forecast cannot
cold-start.

### What VIC needs to run

Seven weather inputs per cell, every 6 hours: **rainfall, air temperature,
incoming shortwave (sun) and longwave radiation, air pressure, vapour pressure,
and wind speed.**

Rainfall in the forecast comes from stage 2. The other six come from ERA5-Land,
a high-quality reconstruction of past weather — except two (longwave radiation
and vapour pressure) for which no forecast equivalent exists in the archive, so
day-of-year climatology is used. That is an honest limitation, noted here and in
the deviations table of the main documentation.

### Calibration — teaching it this basin

VIC has free parameters that cannot be measured directly: how deep the soil
layers are, how fast water drains, how deep roots reach. These are **fitted**.

| period | role |
|---|---|
| 2003 | spin-up, discarded — the model starts from a guess and needs time to forget it |
| **2004–2011** | **calibration** — every observed day, **2,740 days** |
| 2012–2014 | validation — never scored during the search |
| 2015–2022 | held out entirely |

Eight parameters, fitted by **differential evolution** — a search method that
keeps a population of candidate parameter sets, combines them, and keeps whatever
scores better. 432 evaluations, 135 minutes.

**The objective** rewards matching observed flow *and* getting the water partition
physically right:

```
KGE(discharge) − 2 × [ hinge(runoff coefficient, 0.35–0.40)
                     + hinge(ET/P,               0.55–0.65) ]
```

A *hinge* is zero inside the acceptable band and grows outside it — so the search
is not pushed toward a single point inside a range that is itself uncertain.

Note the split of evidence: **flow is scored where it is observed** (daily), the
**partition where it is defined** (annually). That distinction was a genuine fix —
an earlier version scored only monsoon days, leaving two-thirds of each year
invisible to the search.

**The fitted values:**

| parameter | value | what it controls |
|---|---|---|
| b_infilt | 0.183 | shape of the infiltration curve — the main runoff control |
| Ds | 0.0055 | when non-linear baseflow starts |
| Dsmax | 23.54 | maximum baseflow rate |
| Ws | 0.833 | soil moisture at which baseflow turns non-linear |
| d2 | 1.944 m | thickness of soil layer 2 |
| d3 | 2.802 m | thickness of soil layer 3 |
| wpwp_scale | 0.255 | how much soil water plants can actually reach |
| root_scale | 1.814 | how deep roots go |

### How well it works

Scored on monsoon days, against observed inflow:

| split | NSE | correlation | **bias** | days |
|---|---|---|---|---|
| calibration 2004–2011 | 0.581 | 0.931 | **+50.3 %** | 944 |
| validation 2012–2014 | 0.328 | 0.914 | **+58.4 %** | 344 |
| **held out 2015–2022** | 0.416 | **0.937** | **+72.1 %** | 976 |

**Two things to read here, and they point in opposite directions.**

**The timing generalises beautifully.** Correlation is *highest on the years VIC
never saw* — 0.937. The model is not memorising; it genuinely knows *when* water
arrives, even in years absent from its calibration.

**The volume is badly wrong, and gets worse over time.** Bias grows +50 % → +58 %
→ +72 %. A calibration error would be roughly flat across splits. **A bias that
grows with time is a clue**, and it points at a process the model does not have.

### The missing process — people take the water

| | mm/yr | runoff coefficient |
|---|---|---|
| rainfall | 1,281.9 | — |
| **observed inflow** | 366.2 | **0.286** |
| published *natural* value | 449–513 | 0.35–0.40 |

**80–150 mm/yr — 18–29 % of natural runoff — never reaches the gauge.**

The Mahanadi above Hirakud is heavily irrigated. VIC simulates **natural** runoff
and has **no abstraction term** — no way to represent water that people divert for
farming before it reaches the dam.

This explains everything awkward about stage 3:

- **the growing bias** — irrigation has expanded over two decades
- **why forcing the model to match observed volume fails** — when the calibration
  was pushed to reproduce the depleted gauge, it responded by growing an
  **8-metre soil column**, trying to reproduce human water use by evaporating the
  difference. Soil depth hit every ceiling it was given: 1.404 of 1.5, then 2.999
  of 3.0, then 3.999 of 4.0.

> **Do not "fix" a positive VIC bias by widening a soil bound. It is abstraction.**
>
> That warning is now written into `calibrate_vic.py`, because it is a mistake
> that was made three times.

**Current annual water balance:** runoff coefficient **0.445** (target 0.35–0.40),
ET/P **0.558** (inside band), closure **1.003** — meaning water is conserved to
0.3 %, which confirms the model is internally consistent even where it disagrees
with the gauge.

---

## 7. Stage 4 — RVIC routing

VIC produces **runoff per cell, in millimetres**. We need **discharge at one
point, in m³/s**. Stage 4 does two conversions.

### Depth → volume

```
to_cms = 1e-3 × cell_area × basin_fraction / 86400
```

Millimetres to metres (÷1000), times the area of the cell, times the fraction of
that cell actually inside the basin, divided by the seconds in a day.

**This is the only place in the entire pipeline where catchment geometry enters
the numbers.**

### Now → later: the unit hydrograph

Water from a distant cell does not arrive all at once. It spreads out over days.

RVIC represents this with a **unit hydrograph** — a small curve for each cell
saying what fraction of its runoff arrives after 1 day, 2 days, and so on, out to
**10 days**.

**The actual measured shape for this basin**, summed across all 150 source cells:

| days after runoff | share of water arriving |
|---|---|
| 1 | 7.1 % |
| 2 | 15.6 % |
| 3 | 20.2 % |
| **4** | **29.2 %** ← peak |
| 5 | 21.4 % |
| 6 | 6.1 % |
| 7 | 0.4 % |
| 8–10 | 0.0 % |

**Read that table carefully — it is one of the most informative in this
document.** The typical parcel of water takes **four days** to reach the dam, and
essentially everything has arrived within **six**.

This has a direct operational consequence: rain falling today mostly affects
inflow three to five days from now. The dam's own response time is built into
that curve.

The river network comes from terrain via **Dominant River Tracing** — travel
times are derived from the shape of the land, not fitted to data.

### A safeguard worth understanding

The 10-day memory means inflow on forecast day 1 partly depends on runoff from
**before** the forecast was issued.

That history is taken from the observed-weather run. **It is what was already in
the rivers at issue time — not a peek at the future.** Verified: the splice uses
exactly the 10 days *ending on* the initialisation date. Forecast day 1 is
init + 1, so nothing after the issue date is used.

Without this, forecast day 1 would start from an empty river network and
manufacture a false rising limb.

---

## 8. Stage 5 — the LSTM correction

### What an LSTM is

An **LSTM** (Long Short-Term Memory network) is a neural network built for
sequences. Unlike a plain network, it carries an internal "memory" forward through
time, deciding at each step what to keep and what to forget. That makes it
suitable for hydrographs, where what happened last week matters.

**Here it does not forecast. It corrects.** The source paper describes its LSTM as
*"a post-processing model of the hydrological model"* — it takes what the physics
produced and adjusts it.

**Inputs:** recent observed inflow (the hydrograph up to the issue date), the
routed VIC forecast, and the forecast rainfall belonging to that product.

**Output:** a corrected 30-day hydrograph.

### The n_seq finding — a genuine improvement

How many days of history should it read? The source paper calls this an
*"optimised hyperparameter"* and notes it is *"typically set to a larger value"*
in **snow-affected basins**, where accumulation and melt span hundreds of days.

This project had it fixed at **60 days** — a value inherited from a setting that
does not apply. **The Mahanadi is monsoon-driven, not snow-fed.**

Two things were wrong with 60:

**It carried no information.** Inflow autocorrelation at 60 days is **−0.013**
(§2.1). Days 30–60 were pure noise fed to the network as if they were signal.

**It threw away data.** A forecast needs that much *continuous* history to be
usable at all, so 60 days left only **311 of 532** initialisations. Worse, from
2015 the inflow record starts in June — so the discarded windows were
disproportionately **early-monsoon**: the onset, which is exactly when operators
most need a forecast.

**Sweeping it**, scored on the common 309 evaluation windows:

| n_seq | median NSE | initialisations usable |
|---|---|---|
| 7 | 0.192 | 495 (93 %) |
| **15** | **0.200** | **476 (89 %)** |
| 30 | 0.146 | 432 (81 %) |
| 45 | 0.177 | 376 (71 %) |
| **60 (old default)** | **0.140** | **311 (58 %)** |

**Changing 60 → 15 lifted the whole chain from 0.140 to 0.200** — 53 % more
training data *and* less noise, from one number.

**Honest caveat:** 7 and 15 are statistically indistinguishable (mean difference
+0.004, p = 0.59). The solid claim is **"short history, around 7–15 days"**, not
an optimum at exactly 15. The per-year spread is driven by a couple of bad years
(2009, 2017), not by a smooth trend.

### What the LSTM actually is

**A bias-correction layer, not a forecasting layer.** Three independent tests
agree:

**1. Its gain tracks upstream bias.** On the raw chain, where bias is large, it
wins in **18 of 19 years** (t = +3.76). On chains where an earlier stage already
removed bias, it wins in far fewer.

**2. Remove the bias first and it becomes harmful.** Applying a simple volume
scalar *before* the LSTM — so it would only have to learn the residual shape —
made results **worse** on both chains (7/19 wins). There is no useful residual
structure for it to learn at this sample size.

**3. A single multiplication beats it on the median.** One fitted number scores
0.157–0.168 where the LSTM scores 0.140–0.142.

**But it earns its place on the mean.** Per year, it wins 10 of 19 with a median
difference of +0.002 — yet a **mean** difference of **+0.051**, because it rescues
catastrophic years (+0.38 in 2017, +0.30 in 2009, +0.20 in 2015–16) while costing
at most −0.06 in good ones.

**That is exactly what a correction layer should do**, and the median metric
systematically hides it. It also improves the *worst year* on both chains — which
for flood operation matters more than the average.

---

## 9. One forecast, end to end

Here is the forecast issued **Wednesday 4 July 2018**, traced through all five
stages with real numbers. Run it yourself:

```bash
python tools/trace_one_forecast.py 2018-07-04
```

### Stage 1 — what ECMWF said

| lead | rain (mm/day) | ensemble spread |
|---|---|---|
| 1 | 9.23 | 2.51 |
| 3 | 14.17 | 6.57 |
| 7 | 20.18 | 8.96 |
| 14 | 16.78 | 9.91 |
| 21 | 18.36 | 12.74 |
| 30 | 16.81 | 7.04 |

### Stage 2 — after downscaling, against what actually fell

| lead | CNN (mm/day) | **observed (mm/day)** |
|---|---|---|
| 1 | 11.40 | **2.15** |
| 3 | 10.89 | **5.84** |
| 7 | 12.94 | **4.63** |
| 14 | 14.23 | **10.19** |
| 21 | 13.42 | **3.53** |
| 30 | 14.64 | **3.09** |

**The forecast was wet, and reality was much drier.** Note also how *flat* the
CNN column is — 10.9 to 14.6 across the whole month, while observations swing
from 2.2 to 10.2. That is the damping from §5, visible in a single case.

### Stage 3 — rainfall becomes runoff

Shown in §2.1. Runoff ratio moves from 0.21 down to 0.12 and back up to 0.32 —
not a constant, because the soil state is changing.

### Stage 4 — runoff becomes inflow, and stage 5 the verdict

| lead | date | **VIC forecast** | **observed** |
|---|---|---|---|
| 1 | 05 Jul | 1,664 | **280** |
| 3 | 07 Jul | 1,503 | **226** |
| 7 | 11 Jul | 2,087 | **292** |
| 14 | 18 Jul | 3,201 | **1,578** |
| 21 | 25 Jul | 3,583 | **3,046** |
| 30 | 03 Aug | 4,646 | **1,224** |

Over the whole 30 days:

| | mean (m³/s) | peak (m³/s) | peak day | NSE |
|---|---|---|---|---|
| **observed** | 1,638 | 4,746 | day 19 | — |
| **VIC chain** | 3,048 | 4,646 | **day 30** | **−0.812** |

### What this one case teaches

**The volume error is enormous and it is at the start.** On day 1 VIC predicts
**1,664** against an observed **280** — nearly six times too much. That is the
+50–72 % bias of §6, and it is why stages 5 and the scalar baseline exist at all.

**The peak magnitude is nearly right, the peak timing is not.** VIC's peak of
4,646 is within 2 % of the observed 4,746 — but it lands on **day 30** instead of
**day 19**. In this particular window, timing is the visible failure.

**But do not generalise from one case** — and this is important, because an
earlier version of this documentation did exactly that. Across all 309 windows,
with *perfect* rainfall, the mean optimal time shift is only **0.28 days** (§12).
This window is not representative of the systematic error; it is representative
of what a *forecast* error looks like when the rainfall forecast is wrong.

**That is the distinction §12 exists to make.**

---

## 10. How it is tested

### Leave-one-year-out

To test how a model performs on a year it has never seen, train it on the other
years and score it on the held-out year. Repeat for every year.

Here: **19 separate runs**, each holding out one monsoon. Within each run, 17
years train, 1 year checks progress during training (the *validation* year), and
1 year is held completely aside for the final score (the *test* year).

**Everything is refitted inside each fold** — standardisation statistics, rainfall
thresholds, quantile mappings, the scalar baseline. Nothing from the test year
touches training.

### Why never a random split

This is the single most important methodological point.

One valid date is reached by **several different forecasts at different lead
times**. A forecast issued on 1 July at lead 10, and one issued on 5 July at
lead 6, both predict 11 July.

Split at random and **the same observed rainfall map lands on both sides of the
split**. The model sees the answer during training and scores brilliantly on
something it has effectively memorised.

This is not hypothetical. An earlier version of this project did exactly that and
reported **R² 0.873**. The honest figure, after fixing it and five other defects,
is about **0.078** (§13).

### The common evaluation window

Different models need different amounts of history, so they can score different
numbers of forecasts. Comparing them on different samples would be meaningless.

So every model is judged on the **309 windows** where *every* model can produce a
forecast **and** all 30 observed days exist. Same sample, same target, same
metric.

### Ensemble mean, not per-member

Scoring each of the 10 members separately and averaging the scores gives a
different — and flattering — answer compared to scoring the average of the
members.

This project scores the **ensemble mean**, because that is what an operational
system would use. Conflating the two regimes produced wrong conclusions twice in
this project's history, which is why a dedicated module now defines the sample
once for everyone.

### The baselines

Two, both deliberately hard:

**Persistence** — today's inflow, held flat for 30 days. The floor. It scores
−1.023, which tells you that doing nothing clever is genuinely bad.

**`0.86 × EC`** — a single number, fitted leave-one-year-out, multiplying the
physics chain's own output. **This is not a naive baseline.** It is the physics
chain with its volume error removed by one parameter. It scores **0.168**, and
beating it is the real test.

---

## 11. What the chain achieves

309 windows, 19 monsoons, 30-day horizon, ensemble mean:

| model | median NSE | mean | bias % | peak ratio |
|---|---|---|---|---|
| persistence | −1.023 | −1.241 | +21.1 | 0.385 |
| **`0.86 × EC` — the baseline** | **0.168** | 0.069 | −2.9 | 0.794 |
| `EC → VIC → RVIC` | −0.010 | −0.433 | +31.8 | **1.081** |
| `EC-CNN → VIC → RVIC` | 0.161 | −0.075 | +0.7 | 0.833 |
| `EC → VIC → RVIC → LSTM` | 0.118 | 0.070 | −11.1 | 0.744 |
| **`EC-CNN → VIC → RVIC → LSTM`** | **0.200** | 0.040 | −16.0 | 0.666 |

**The full chain — the paper's proposed system — is the best physics
configuration**, and it exceeds the 0.168 baseline on the median.

### But it is still not significantly better

| model | t | p | wins |
|---|---|---|---|
| `EC → VIC → RVIC` | −3.21 | **0.005** | 2/19 |
| `EC-CNN → VIC → RVIC` | −1.85 | 0.081 | 5/19 |
| `EC → VIC → RVIC → LSTM` | +0.01 | 0.991 | 7/19 |
| **`EC-CNN → VIC → RVIC → LSTM`** | −0.40 | 0.695 | **9/19** |

**The honest sentence is:** *"no longer significantly worse than the baseline, and
the highest-scoring physics configuration"* — **not** *"beats the baseline"*.
Winning 9 of 19 years is a coin flip.

### Skill by lead time — where the pipeline earns its keep

| band | `0.86 × EC` | full chain | what is actually happening |
|---|---|---|---|
| **1–3 d** | 0.699 | 0.587 | catchment memory — persistence alone scores **0.507** |
| **4–7 d** | 0.049 | **0.275** | **the pipeline's real contribution** |
| 8–14 d | 0.107 | 0.100 | marginal |
| 15–21 d | 0.102 | 0.068 | marginal |
| **22–30 d** | −0.009 | −0.010 | zero — day-of-year climatology scores **0.042** |

**Read this table before quoting any headline number.**

Days 1–3 look strong, but a model using **no weather forecast at all** scores
0.507 there. That is the catchment's own memory, not forecasting skill.

Days 4–7 are where the physics chain genuinely beats everything simple — 0.275
against the baseline's 0.049.

Beyond day 20, everything is at zero and a simple seasonal average does better.

**The pipeline's value is concentrated in days 4–14.** That is a real, defensible,
useful result — and much more honest than "a 30-day forecast".

---

## 12. Why it is not better

### The diagnostic that explains everything

Drive the **identical chain** with **observed** weather instead of forecast
weather. That is the ceiling no forecast can ever exceed.

| | median NSE |
|---|---|
| perfect rainfall, as simulated | **0.394** |
| **perfect rainfall + one volume correction** | **0.850** |
| perfect rainfall + optimal timing shift | 0.471 |
| *actual chain, with forecast rainfall* | **0.200** |

**Four numbers containing the whole story.** Read them carefully.

**VIC + RVIC is not the weak link.** Given the right water volume, the hydrology
reproduces 30-day inflow at **NSE 0.850**. The physics works.

**The error is volume, not timing.** The mean optimal time shift across all 309
windows is **0.28 days** — timing is essentially perfect. Shifting every window to
its best possible alignment gains only 0.394 → 0.471. **One scalar multiplication
gains 0.394 → 0.850.**

> ⚠️ **An earlier version of this documentation said the opposite** — that error
> was *"dominated by phase, which no downstream model can fix"*, citing 66–95 %
> phase contribution.
>
> That was measured on the **forecast** chains, where the timing error belongs to
> **ECMWF**, not to VIC. Two different things — the rainfall forecast's timing
> error and the hydrological model's own error — were measured together and both
> attributed to the model.
>
> **To measure a component's error, hold its inputs perfect.** That is what the
> table above does, and it reverses the conclusion.

**The gap from 0.850 to 0.200 is the rainfall forecast**, and nothing downstream
can recover it.

### Why the rainfall forecast caps everything

| lead | corr(catchment rainfall, observed) |
|---|---|
| 1–3 d | 0.637 |
| 4–7 d | 0.509 |
| 8–14 d | 0.400 |
| 15–21 d | 0.362 |
| 22–30 d | 0.376 |

Beyond about a week, catchment rainfall correlates **0.35–0.40** with what
actually falls. That is a property of atmospheric predictability — the practical
limit is roughly 1–2 weeks and it belongs to the atmosphere, not to ECMWF.

**No post-processor creates information that is not in its input.** A better
hydrological model, a bigger neural network, a more sophisticated correction
layer — none of them change that number.

### The CNN's contribution, measured honestly

Comparing the chains at **equal volume** (a leave-one-year-out scalar applied to
both, so bias cannot flatter either):

| | median NSE |
|---|---|
| raw EC | **0.168** |
| EC-CNN | **0.157** |

**The CNN is slightly worse.** Its apparent advantage in the headline table —
0.161 against the raw chain's −0.010 — is **bias cancellation**: the CNN's dry
catchment mean (+0.7 %) offsetting VIC's wet bias (+31.8 %). Two errors of
opposite sign, not better rainfall.

**Before crediting any component, remove the bias. Two errors can cancel.**

---

## 13. Bugs, and what they taught

The dangerous bugs in this kind of work are not the ones that crash. **They are
the ones that make results look better.** A crash gets fixed in an hour; a leak
gets published.

### The six inherited defects

This project began from a notebook reporting **R² 0.873** for rainfall
downscaling. An audit found **six separate defects, every one inflating the
score**:

| defect | why it inflated the result |
|---|---|
| accumulated variables never differenced | running totals fed as if daily |
| ocean cells filled with zeros | 6 of 25 target cells were a constant the model "predicted" perfectly — 24 % of the grid |
| random train/test split | the same observed map on both sides |
| scaler fitted before splitting | test statistics leaked into training |
| target subsampled to 25 coarse points | **no downscaling was happening at all** |
| rainfall lags indexed to the target date | at lead 17, "yesterday's rain" was 16 days in the **future** |

**Honest score after fixing all six: R² 0.073–0.078.**

The fifth is the one to remember. The model was predicting coarse rainfall from
coarse rainfall and reporting it as downscaling. It was not subtle — it was
invisible because **nobody checked what the target actually contained**.

### The scrambled forecast dates

The worst defect found during this project, and the last.

One script sorted its data into a canonical order but wrote the **date labels
from the un-sorted arrays**. Every row carried another row's date.

**It hid because two of the three labels were accidentally right.** The lead
number is the repeating sequence 1…30 under either ordering. The year came from
the correctly-ordered source. Only the date the forecast was *for* was wrong —
and nothing downstream ever printed it.

| check | before | after |
|---|---|---|
| rows whose date minus lead equals the true issue date | 300 / 15,960 | **15,960 / 15,960** |
| initialisations paired with their own rainfall | 10 / 532 | **532 / 532** |
| day-1 correlation, forecast rain vs observed | **−0.02** | **+0.68** |

**A day-1 rainfall forecast that correlates −0.02 with what fell is not a
forecast at all.** That single number is what the labels were costing.

**Lesson:** when you reorder an array, reorder its labels in the same statement —
then verify with a *physical* check the labels cannot pass by accident.

### Calibration multipliers compounding

The calibration script read the **active** soil files as its starting point. But
activating a calibration overwrites those files — so each run began from the
previous winner. Because two parameters are **multipliers**, they stacked: a run
reporting `wpwp_scale = 0.8755` had an effective value of 0.1532 × 0.8755 =
**0.1341** — *lower* than the value that run was launched to escape.

**Lesson:** a parameter that is a multiplier must be applied to a fixed
reference, never to the last result. Fixed with immutable base files and a guard
that refuses to start from a calibration output.

### Fabricated zeros in the calibration target

Missing gauge days are stored as `0.0` with a separate validity flag. The
calibration and the routing evaluation both ignored the flag. **213 of 3,287
calibration days were flagged invalid, and all 213 were zeros**, concentrated in
January–June.

The model was being taught to produce no water on days the gauge simply did not
report — which would have looked like an improved runoff coefficient obtained by
inventing a drought.

**Lesson:** one project, one definition of "observed". Honour the validity flag
everywhere.

### Environment traps

**`~/.vic_mahanadi` is a symlink** into the project directory. Move the project
and VIC silently stops working until it is repointed.

**The Kaggle runner's `--skip-data` flag skips your code too**, because the
project code ships inside the uploaded dataset. The cloud machine then runs the
*previous* version and looks fine.

**Kaggle assigns the wrong GPU by default** — a P100, for which its own PyTorch
build ships no kernels, so every convolution fails. The command-line flag does not
override it, and a CLI push resets whatever the web interface had set.

---

## 14. What changed recently

If you are holding an older draft, these claims have been **overturned by
measurement**. All are documented in `docs/FINDINGS.md`.

| older claim | current position |
|---|---|
| *"error is dominated by timing/phase"* | **Wrong.** Optimal shift is 0.28 days; the error is **volume** |
| *"the CNN improves the inflow forecast"* | **Wrong.** At equal volume: 0.157 vs raw ECMWF's 0.168. Its apparent lead was two biases cancelling |
| *"the LSTM adds nothing"* | **Wrong.** It is a bias layer; its gain scales with upstream bias |
| LSTM history = 60 days | **15 days** — lifted the chain 0.140 → 0.200 |
| *"the specialised loss balances extremes"* | **It does not function here** — 0.375 % of the loss, and unreachable |
| headline chain ≈ 0.14 | **0.200** |

---

## 15. Questions you may be asked

**"Your NSE is 0.2. Isn't that terrible?"**

For 30-day inflow forecasting, no — but the honest framing is that the number is
not the point. With perfect rainfall the same chain reaches 0.850, so the
hydrology is sound; the gap is the rainfall forecast, and catchment rainfall
correlates 0.35–0.40 with reality beyond a week. The useful result is knowing
*where* the skill is lost, which took the diagnostic in §12 to establish.

**"Why doesn't it beat a single multiplication?"**

Because the dominant correctable error *is* volume, and a single multiplication
corrects volume. That baseline is not naive — it is the physics chain with its
bias removed. The finding is that on this basin, bias was most of what there was
to fix.

**"Why use VIC instead of the paper's model?"**

The source paper used XAJ, a lumped conceptual model. VIC is semi-distributed —
it resolves 150 cells with separate soil and vegetation. That matters here
because §2.2 shows averaging rainfall across a 26,000 km² cell destroys flood
peaks. The cost is a much larger setup burden.

**"Is the CNN worth having?"**

At the cell level, clearly yes — it flips R² from −0.136 to +0.057 in all 19
years. At catchment scale, on current evidence, no: at equal volume it is 0.011
worse than raw ECMWF. Both statements are true because they measure different
things, and the chain uses the catchment total.

**"Isn't a +72 % bias disqualifying?"**

Part of it is not model error. 18–29 % of natural runoff is diverted upstream for
irrigation before reaching the dam, and VIC has no term for that. The bias grows
monotonically across calibration → validation → held-out periods, which is what
expanding irrigation looks like and not what a calibration error looks like.

**"Could you just use more data?"**

Not usefully for VIC. Dry-season observations stop in 2014 — 2015–2022 is
monsoon-only — and eight years of daily water balance already over-determines
eight parameters. The binding constraint was never years; it was which *days*
were scored, which is now fixed.

**"What would actually improve this?"**

Better rainfall forecasts, or an explicit abstraction term. Further work on the
hydrological model or the correction layer will not move these numbers, and §12
is the evidence.

---

## 16. Glossary

| term | meaning |
|---|---|
| **antecedent moisture** | how wet the ground already was before rain fell |
| **baseflow** | slow drainage from deep soil that keeps rivers flowing between storms |
| **catchment** | the land area that drains to a point — here, 83,400 km² above the dam |
| **cumec** | cubic metre per second, m³/s — the unit of river discharge |
| **differential evolution** | a search method that evolves a population of candidate parameter sets |
| **discharge** | volume of water passing a point per second |
| **ensemble member** | one of 11 slightly different forecast runs |
| **ensemble spread** | how much the members disagree — a confidence measure |
| **hindcast / reforecast** | today's model re-run on past dates |
| **initialisation date** | the day a forecast is issued |
| **KGE** | Kling-Gupta Efficiency — splits error into correlation, variability, bias |
| **lead day** | how far ahead a forecast looks |
| **LOYO** | leave-one-year-out — train on all years but one, test on that one |
| **LSTM** | a neural network that carries memory across time steps |
| **NSE** | Nash-Sutcliffe Efficiency — 1 is perfect, 0 equals predicting the average |
| **peak ratio** | forecast peak ÷ observed peak; below 1 means under-predicting floods |
| **quantile mapping** | statistical bias correction that reshapes a distribution |
| **ResNet** | a network with skip connections, letting it go deep safely |
| **routing** | moving water through the river network with realistic travel times |
| **runoff coefficient** | fraction of rainfall that becomes streamflow |
| **S2S** | sub-seasonal to seasonal — roughly 2 weeks to 2 months ahead |
| **threat score** | hits ÷ (hits + false alarms + misses), for heavy-rain events |
| **unit hydrograph** | the curve describing how a cell's runoff spreads over arrival days |
| **VIC** | Variable Infiltration Capacity — the physical hydrological model |

---

## 17. How to run it

Interpreter: `/usr/local/bin/python3`. Check inputs first with
`python hydrology/vic/check_inputs.py`.

```bash
# ---- stage 2: rainfall downscaling  (GPU; see tools/kaggle_run.py)
python cnn/loyo_percell_v2.py            # 19-fold LOYO
python cnn/quantile_mapping.py           # the EC and EC-QM benchmarks

# ---- stage 3: VIC  (calibrate once, ~2 h)
python hydrology/vic/calibrate_vic.py --workers 4 --maxiter 8 --popsize 6
hydrology/vic/VIC/vic/drivers/classic/vic_classic.exe \
    -g data/processed/vic/global_param.txt        # full record 2003-2022
python hydrology/vic/water_balance.py             # partition check
python hydrology/vic/route_and_evaluate.py        # cal / val / held-out splits

# ---- stages 3+4: forecasting
python hydrology/vic/vic_states.py                # 532 soil states, ~3 min
for p in ec ec_qm ec_cnn; do
    python hydrology/vic/vic_forecast.py --product $p --workers 4   # ~20 min each
    python hydrology/vic/route_forecast.py --product $p
done

# ---- stage 5: correction
python inflow/lstm_postproc.py --product ec
python inflow/lstm_postproc.py --product ec_cnn

# ---- results
python inflow/head_to_head.py            # the comparison table
python inflow/fair_comparison.py         # significance, VIC-exposure split

# ---- understand one forecast
python tools/trace_one_forecast.py 2018-07-04
```

---

## Where to go next

| document | purpose |
|---|---|
| `docs/PROJECT_DOCUMENTATION.md` | the whole project, both pipelines, concise |
| `docs/FINDINGS.md` | research log — what was discovered, in order, including what was wrong |
| `STRUCTURE.md` | which file belongs to which stage |
| `results/metrics/head_to_head.json` | the numbers themselves |
