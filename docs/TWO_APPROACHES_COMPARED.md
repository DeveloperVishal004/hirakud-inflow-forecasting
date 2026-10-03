# Forecasting Water Into Hirakud Dam
### A comparison of two ways to predict reservoir inflow 1–30 days ahead

**Mahanadi basin, India · 19 monsoon seasons (2004–2022) · 532 forecasts**

> **Revision note, 14 September 2026.** This document has been checked against the
> code and results files as they stand today. Six substantive corrections were
> applied, and they are listed here so that anyone holding an older copy knows what
> changed:
>
> | # | What changed | Where |
> |---|---|---|
> | 1 | **"The main error is timing, not amount" is disproved.** The error is **volume**. | Ch. 1 finding 4, Ch. 3 reason 4 |
> | 2 | Headline significance numbers refreshed from `head_to_head.json` / `fair_comparison.json` | Ch. 1 findings 1, 3, 6 |
> | 3 | The `log_space` contradiction in the loss function **has been fixed** in the code | Ch. 5 |
> | 4 | The threat-score threshold fallback **has been fixed** in the code | Ch. 5 |
> | 5 | The out-of-date-files warning rewritten — those files are now archived | Ch. 1 |
> | 6 | **The "raw ECMWF scores R² −0.136, downscaling flips the sign" claim is wrong.** Raw ECMWF is **+0.044** in the same scoring regime; there is no sign flip. | Ch. 4, Ch. 5 |
>
> Additions: why only 10 of the 11 ensemble members are used (Ch. 2, Ch. 5), and a
> stale sentence in the Chapter 2 summary that contradicted its own Trap 3 section.

## How to read this document

This is written for someone who knows neither hydrology nor machine learning. Every
technical term is explained the first time it appears. Every number comes from the
project's own code and results files, and where the project's files disagree with each
other, that is said openly instead of quietly picking one.

The document tells a story in order. Each chapter answers a question raised by the one
before it. If you read it straight through you will see how each experiment was caused
by a problem discovered in the previous experiment.

**Three things to know before you start.**

**1. This project compares two separate systems. It does not build one.** They are kept
strictly apart on purpose. Nothing produced by the first system is ever fed into the
second. Chapter 8 explains why that separation matters.

**2. The main result is a negative one.** No method tested here is statistically better
than a very simple baseline. That sounds like failure. It is actually the most useful
thing the project found, and Chapters 13 and 15 explain why.

**3. Some of the project's own summary files contained errors.** An audit found fifteen
places where files disagreed with each other. Those are reported as they come up. Several
have since been repaired in the code, and where that is so it is said explicitly rather
than left as a standing warning — a warning that outlives the bug it describes is its own
kind of error, and this project made exactly that mistake twice.

---

# PART I — UNDERSTANDING THE PROBLEM

---

# Chapter 1 — What We Are Forecasting, and Why It Is Hard

## The operator's problem

Imagine you run Hirakud Dam during the monsoon. Rain is forecast over the hills
upstream. You have to decide today whether to open the gates.

If you release too little water and heavy rain arrives, the reservoir fills. Then you
have no choice but to dump a huge volume all at once, and that sudden release floods
the towns below the dam.

If you release too much and the rain never comes, you have thrown away water that the
region needed for irrigation and for generating electricity.

Here is the difficulty. By the time you can actually measure the water arriving at the
reservoir, the decision that mattered was made days ago. The water is already there.
To operate the dam safely you need to know what is coming **before** it arrives.

So the question this whole project tries to answer is:

> Given the weather forecast issued today, how much water will arrive at Hirakud Dam on
> each of the next 30 days — and how sure can we be?

## What exactly is being predicted

The thing being predicted is **reservoir inflow**: the volume of water entering the
reservoir each day, measured in **cubic metres per second** (written m³/s, and also
called "cumecs").

The forecast runs from 1 to 30 days ahead. Each of those days is called a **lead day**.
Lead day 1 is tomorrow, lead day 30 is a month away.

The day the forecast is issued is called the **initialisation date**. Everything the
model is allowed to know must be known on or before that date.

Forecasts are only made during the monsoon, June to September, because that is when
almost all the water arrives.

## Where this is happening

Hirakud Dam sits on the Mahanadi river at 21.52° N, 83.87° E in Odisha, eastern India.

The area of land that drains into the reservoir — the **catchment** — is about
**83,400 km²**. That is roughly the size of Austria. Rain falling anywhere in that area
eventually flows downhill into the Mahanadi and reaches the dam.

The catchment reaches west into the neighbouring state of Chhattisgarh, and includes
uplands, forest and valley floor. Different parts of it behave very differently when
rain falls on them. That variety is going to matter a great deal in Chapter 3.

## Two separate approaches, compared

This project does not invent a new forecasting method. It takes **two methods that were
already published by other researchers**, rebuilds both of them for Hirakud, and puts
them through exactly the same test to see which one works better.

**Approach 1 — the physics route.** Published by Dong and colleagues in 2025 in the
journal *Hydrology and Earth System Sciences*. The idea is to model what physically
happens to the water:

```
weather forecast
  -> sharpen the rainfall map with a neural network
  -> simulate water moving through the soil
  -> route the water down the river network
  -> (optionally) correct the result with a second neural network
  -> inflow at the dam
```

**Approach 2 — the direct route.** Published by Ambika and colleagues in 2025 in
*Geophysical Research Letters*. The idea is to skip the physics entirely:

```
recent river behaviour + recent weather + forecast weather
  -> one large neural network
  -> inflow at the dam
```

**These two are kept completely separate.** Nothing produced by Approach 1 is ever used
by Approach 2. This is enforced in the project's code, not just intended, and Chapter 8
explains exactly why it had to be.

The only two things they share are the raw weather forecast they both start from, and
the final test they are both judged on.

## How the models are tested

The historical record covers **19 monsoon seasons, 2004 to 2022**. Within those seasons
the weather archive supplies **532 forecast dates**.

Testing uses a method called **leave-one-year-out**, usually shortened to **LOYO**.
The idea is simple: to test how well a model would do on a year it has never seen, you
train it on the other years and then score it on the held-out year. Repeat for every
year. Here that means **19 separate runs**, each holding out a different monsoon.

Within each run the years are split three ways: **17 years for training**, **1 year for
checking progress during training** (called the validation year), and **1 year held
completely aside for the final score** (the test year).

One more detail. The weather centre does not issue a single forecast. It issues eleven
slightly different versions, because small differences in today's weather grow into
large differences a week later. These versions are called **ensemble members**. This
project uses the average of **ten** of them as the forecast, because that is what an
operational system would use — and because the eleventh is a different kind of object.
Chapter 2 explains which one is left out and why.

## The main scoring metric

Most results in this document are reported using **Nash-Sutcliffe Efficiency**, or
**NSE**. It is defined as:

```
NSE = 1  -  (sum of squared errors)  /  (sum of squared differences from the average)
```

You do not need the formula. You need the interpretation:

| NSE value | What it means |
|---|---|
| **1.0** | perfect prediction |
| **0.0** | exactly as good as always predicting the long-run average flow |
| **negative** | **worse** than just predicting the average |

An NSE of 0 is not "half right". It means the model added nothing at all. This matters
because several results in this document are near zero, and one baseline is around −1.

## The headline findings, stated up front

There is no suspense in this document. Here is what the project found.

**1. No method beats a very simple baseline, statistically.** The challengers were all
tested against a baseline called `0.86 × EC`, which is the physics chain multiplied by
a single fitted number. The best challenger was the full physics chain,
`EC-CNN → VIC → RVIC → LSTM`, at median NSE **0.200** against the baseline's **0.168**.
The paired test across the 19 years gives **t = −0.40, p = 0.695**, and the challenger
wins in **9 of 19 years**. That is a coin flip, not a difference.

Here is the whole table, on the 309 common evaluation windows:

| model | median NSE | vs baseline | t | p | wins |
|---|---|---|---|---|---|
| persistence | −1.023 | −1.120 | −9.36 | <0.001 | 0/19 |
| **`0.86 × EC` (baseline)** | **0.168** | — | — | — | — |
| `EC → VIC → RVIC` | −0.010 | −0.208 | −3.21 | 0.005 | 2/19 |
| `EC-CNN → VIC → RVIC` | 0.161 | −0.039 | −1.85 | 0.081 | 5/19 |
| `EC → VIC → RVIC → LSTM` | 0.118 | −0.017 | 0.01 | 0.991 | 7/19 |
| **`EC-CNN → VIC → RVIC → LSTM`** | **0.200** | −0.011 | −0.40 | 0.695 | 9/19 |
| FutureTST + FT ⚠ | 0.186 | +0.016 | −1.12 | 0.277 | 11/19 |
| FutureTST + QM + FT ⚠ | 0.137 | 0.000 | −0.56 | 0.582 | 10/19 |

⚠ The two FutureTST rows **predate a data repair and have not been re-run.** They are
provisional everywhere in this project. Do not quote them as final.

Note the oddity in row 6: the best median in the table has a *negative* median margin
over the baseline. That is not a typo. The median of the per-year differences and the
difference of the per-year medians are different quantities, and on a high-variance
model they disagree. It is a small lesson in why one number is never enough.

**2. Sharpening the rainfall made the rainfall better and the inflow worse.** The
neural network improved five-day rainfall accuracy by 5.9 %. But compared at equal
volume — the only fair comparison — the inflow forecast built on that improved rainfall
scores **0.157** against raw ECMWF's **0.168**. Chapter 6 shows why, and it is the most
instructive result in the project.

**3. The correction network is a bias-removal layer, not a forecasting layer.** On the
raw chain it lifts median NSE from **−0.010 to 0.118**, which is a large gain — but all
of it comes from removing a +31.8 % volume bias. Remove that bias first with a single
fitted number and the network becomes actively *harmful* on both chains (7 of 19 years
on each). It has no useful residual shape left to learn. Chapter 11 gives the three
independent tests that establish this.

**4. The main error is volume, not timing.** *(This reverses what earlier versions of
this document said, and the reversal is the single most important correction in it.)*
The forecasts are not mostly wrong about *when* the water arrives. They are wrong about
*how much*. Chapter 15 gives the experiment that settles it; the short version is that
driving the identical chain with observed weather and then applying **one volume
scalar** lifts NSE from 0.394 to **0.850**, whereas allowing the forecast to slide
freely in time gains almost nothing (0.394 to 0.471) and the average best time shift is
**0.28 days** — essentially zero.

The earlier "it's a timing problem" conclusion came from measuring phase error on the
*forecast* chains, where the timing error belongs to ECMWF's rainfall forecast, not to
the water model. Two different errors were measured together and blamed on one of them.

**5. Skill does not fade — it falls off a cliff.** Between forecast day 3 and day 4 the
agreement between forecast and reality drops sharply, and then stays roughly flat for
the rest of the month:

| lead band | baseline | `EC-CNN → VIC → RVIC → LSTM` | FutureTST ⚠ |
|---|---|---|---|
| 1–3 days | **0.699** | 0.587 | 0.501 |
| 4–7 days | 0.049 | **0.275** | 0.264 |
| 8–14 days | 0.107 | 0.100 | **0.128** |
| 15–21 days | 0.102 | 0.068 | **0.123** |
| 22–30 days | −0.009 | −0.010 | **0.073** |

**6. Beyond the first week, nothing is reliable as a single number.** The physics chain
is best at 4–7 days. FutureTST is nominally best at 8–30 days and the only model still
positive at 22–30 — but those numbers are provisional, and an NSE of 0.07 is barely
distinguishable from predicting the seasonal average. The honest claim for long leads
is about **ranges rather than point forecasts**: the probabilistic outputs stay
usefully calibrated at horizons where the single-number forecast is worthless.

## About the project's older files

Three files this document used to warn about — `RESULTS.md`, `INTERVIEW_GUIDE.md` and an
earlier `README.md` — described a previous era of the work: 11 monsoon seasons instead
of 19, a 17-day horizon instead of 30, and a different set of models.

**That warning is now partly out of date, which is worth saying plainly.** `RESULTS.md`
and `INTERVIEW_GUIDE.md` have been moved to `archive/historical_docs/`, and `README.md`
has been rewritten and is current. The rule is now simpler:

> **Anything under `archive/` may contain obsolete numbers and must not be quoted for
> results.** Everything outside `archive/` is current.

The authoritative sources, in order:

| file | what it is for |
|---|---|
| `docs/PROJECT_DOCUMENTATION.md` | the single source of truth: architecture, design, results, limitations |
| `docs/PIPELINE_1_EXPLAINED.md` | the physics chain, stage by stage, in plain language |
| `docs/FINDINGS.md` | the research log — what was learned, in order, including what turned out wrong |
| `results/metrics/head_to_head.json` | the raw numbers behind every results table |
| `CURRENT_STATE.md` | what is running and what is pending, today |

## Chapter summary

A dam operator must decide releases before the water arrives, which requires a forecast
of reservoir inflow 1 to 30 days ahead. The catchment above Hirakud covers 83,400 km²
and the record covers 19 monsoon seasons and 532 forecast dates. Two published methods
are rebuilt and compared: a physics-based chain and a direct machine-learning model.
They are kept strictly separate and judged on the same test using leave-one-year-out
validation. The headline result is that neither beats a one-parameter baseline by a
statistically significant margin — and understanding why turns out to be more useful
than a high score would have been.

Before we can understand any model, we need to know what information the models are
allowed to see. That is the next chapter.

---
# Chapter 2 — The Data We Actually Have

A forecasting model can only ever be as good as what you feed it. This chapter lists
every dataset the project uses, at what resolution, and — importantly — the defects
buried in the weather archive that would silently ruin a model if you did not know
about them.

## The weather forecast: ECMWF S2S

The starting point for both approaches is a forecast from the **European Centre for
Medium-Range Weather Forecasts**, usually written **ECMWF**. It is generally regarded
as the best global weather forecasting centre in the world.

The specific product used is the **S2S reforecast**. Two terms to unpack.

**S2S** stands for **sub-seasonal to seasonal**, meaning roughly two weeks to two
months ahead. This is the hardest range in weather forecasting. Short-range forecasts
work because today's atmospheric conditions still dominate. Long-range seasonal
forecasts work because slow things like ocean temperatures dominate. The sub-seasonal
range sits in the gap where neither helps much.

A **reforecast** (also called a hindcast) means the weather centre took its *current*
model and re-ran it on *past* dates. This gives you a consistent set of forecasts
across many years, all produced by the same model version. That consistency is why the
project uses it.

*(The general reasoning about why a reforecast is preferable to an operational archive
is standard practice in the field; the project's own files record the schedule and
constraints rather than this justification.)*

### What was downloaded

| Property | Value |
|---|---|
| Grid resolution | **1.5°** |
| Region | a 7 × 7 box of grid cells |
| Latitudes | 25.5 down to 16.5 (stored descending, as the file format does) |
| Longitudes | 79.5 to 88.5 |
| Lead days | 1 to 30 |
| Ensemble members | **11** downloaded — one "control" plus 10 perturbed. **10 used.** |
| Weather variables | **26** |
| Forecast dates per season | **28** |
| Total forecast dates | **532** |

The 26 variables are 11 measured at the surface (`tp`, `cp`, `t2m`, `u10`, `v10`,
`msl`, `tcc`, `ssrd`, `sshf`, `slhf`, `orog`) and 5 measured high in the atmosphere
(`u`, `v`, `q`, `t`, `gh`) at three altitudes — the 200, 500 and 850 hPa pressure
levels. Eleven plus five times three equals twenty-six.

The upper-air variables were added deliberately. The project's configuration file
explains why in one sentence: *"a surface-only forecast cannot place a storm in a
particular 25 km cell, which is what heavy-rain R² = −1.25 across all eleven folds had
been saying."* In plain terms: knowing the pressure at ground level tells you almost
nothing about where a storm will organise. Knowing the moisture and wind flow a few
kilometres up tells you a great deal.

### Why eleven members are downloaded but only ten are used

This is a question worth answering properly, because "we threw away 9 % of our data"
deserves a reason.

The eleven members are **not eleven of the same thing**. Member 0 is the **control
run**: the forecast started from the meteorologists' single best estimate of today's
atmosphere, with nothing added. Members 1 to 10 each start from that same estimate
with a small deliberate nudge applied, and those nudges are drawn from the
distribution of plausible measurement error.

So the ten perturbed members are **samples from a distribution** — each one says "here
is one way the weather could plausibly go". The control is the **centre** of that
distribution. Averaging one centre together with ten samples is not the same operation
as averaging eleven samples; it silently double-weights the middle.

Three consequences, in order of importance:

1. **The spread would be wrong.** The disagreement among the members is this project's
   only measure of forecast confidence, and it is a real one — the ensemble spread
   correlates **+0.475** with the eventual size of the error. Mixing in a member that
   was generated by a different procedure contaminates that measure.
2. **The source paper specifies ten.** Dong et al. downscale "10 ensemble members".
   Including the control would be a silent deviation from the method being reproduced.
3. **Consistency.** The control is dropped once, at archive-build time, so every
   downstream stage — the rainfall network, the water model's forcing, and the scoring
   — sees exactly the same ten. There is no stage where a different ensemble sneaks in.

The configuration records both facts (`V2_N_MEMBERS = 11`, `V2_CONTROL_MEMBER = 0`) and
the code in `cnn/loyo_percell_v2.py` does the dropping in one line, with the reason
beside it:

```python
# member 0 is the control; the paper's "10 ensemble members" are the
# perturbed ones, so take 1..10 unless asked for all 11.
f = f[:, :, 1:1 + n_members] if n_members <= 10 else f
```

Passing `--members 11` includes the control, if you ever want to test the difference.
The stored training archive `percell_v2_full.npz` has shape
`(15960, 10, 26, 6, 5)` — ten members, control already removed.

### Why only 28 dates per season

The reforecast is not run every day. It runs on **Mondays and Thursdays only**.

During the June-to-September monsoon that gives 31 scheduled dates. But three of them —
12 July, 19 July and 26 July — fall on Fridays in the model version being used, and the
data server simply rejects requests for them with an error. The project's config file
notes explicitly that these *"are not a download failure to retry."* They do not exist.

So 31 minus 3 gives **28 dates per season**, and 28 × 19 seasons gives **532 forecasts**.

### The shape of the data

Every forecast date covers 30 lead days. Flattening those out gives the basic unit of
the dataset:

```
532 forecast dates  ×  30 lead days  =  15,960 rows
```

Each row is one **(initialisation date, lead day)** pair. Each row carries a full map
of predicted rainfall as its target. When you see "about 16,000 samples" mentioned
anywhere in this project, this is what it means.

## Traps in the weather archive

Raw data from a global weather centre is not ready to use. Several specific quirks in
this archive will silently destroy a model if you miss them.

### Trap 1 — rainfall totals accumulate

When ECMWF reports total precipitation (`tp`) or convective precipitation (`cp`), it
does **not** give you that day's rain. It gives you the **running total since the
forecast was issued**.

Think of a water meter. On day 1 it reads 2.4 mm. By day 30 it reads 238 mm. If you
feed those raw numbers to a model, the model believes 238 mm of rain fell on day 30
alone.

The fix is to subtract each day from the next, separately for every ensemble member:

```python
a = f[:, :, :, acc_idx]
a[:, 1:] = np.diff(a, axis=1)   # undo accumulation, per member
f[:, :, :, acc_idx] = a
```

The project's own comment calls missing this step *"the classic way to get a downscaler
that looks brilliant and is useless."* That is not hypothetical — it was one of six
real defects found in an earlier version of this work (Chapter 16).

The same treatment is needed for `ssrd`, `sshf` and `slhf`, which are energy fluxes
accumulated over each day. Those are also divided by 86,400 (the number of seconds in a
day) to convert them into an average rate.

### Trap 2 — cloud cover is a percentage

Total cloud cover (`tcc`) arrives as a number from **0 to 100**, not as a fraction from
0 to 1. The name suggests a fraction. It is not.

### Trap 3 — two variables that once carried no forecast information (FIXED)

This was the most serious defect in the archive. It has since been repaired, and the
repair is verified. The story is kept because it is the best example in this project
of a bug that produces no error message.

**What went wrong.** Temperature at 2 metres (`t2m`) and total cloud cover (`tcc`) came
back as a **single 0-to-24-hour field** instead of 30 separate daily values. The same
one value was then copied across all 30 lead days — so the temperature at day 30 was
identical to the temperature at day 1. Two of the 26 predictor channels were telling
the model nothing whatsoever about how conditions change over the forecast.

**Why it happened.** These two variables are *period-processed* in the ECMWF S2S
reforecast, not instantaneous. Most surface fields are snapshots — "the value at hour
48". These two are stored as averages over a window, labelled `"0-24"`, `"24-48"` and so
on. The original request asked for instantaneous lead hours (24, 48, … 720). Nine
surface fields honoured that. `t2m` and `tcc` matched only the one window whose label
happened to coincide, `"0-24"`, and returned a single field. No error, no warning — the
code then broadcast that one field across 30 leads because it expected thirty.

**The fix.** `preprocessing/probe_2t_tcc.py` tested the hypothesis on one
initialisation: ask for them as 24-hour **period windows** (`0_24`, `24_48`, …
`696_720`) instead. All 30 came back. `preprocessing/download_s2s_dong.py` now sends
those two variables as a separate period pass, and the archive was rebuilt.

There is a small silver lining. A 24-hour **mean** temperature is arguably a *better*
predictor for a daily rainfall model than an instantaneous midnight snapshot would have
been. The corrected request is not merely less broken than the original intent — it is
better than it.

**The verification.** Measured directly on `data/processed/percell_v2_full.npz`, the
file the rainfall network actually trains on. For each of the 532 initialisations, the
field is averaged over the ten members and the coarse grid, and the standard deviation
is taken **across its own 30 lead days**:

| channel | mean σ | minimum σ | |
|---|---|---|---|
| `tp` (rainfall) | 0.243 | 0.052 | varies, as it must |
| **`t2m`** | **0.225** | **0.047** | varies |
| **`tcc`** | **0.337** | **0.079** | varies |
| `orog` (terrain) | **0.000** | **0.000** | constant, correctly — it is static |

`orog` is the control case. A broadcast field looks like that row: exactly zero, in
every initialisation, with no exceptions. Neither `t2m` nor `tcc` does, in any year, in
any initialisation. (These are standardised units, so the numbers are not degrees or
percentage points; what matters is that they are not zero.)

**They are also physically coherent, which a copied field could never be.** Within each
forecast, across its 30 lead days, over all 532 initialisations:

- `corr(tcc, rainfall)` = **+0.65** on average — cloud brings rain. Positive in
  **97.7 %** of forecasts.
- `corr(t2m, rainfall)` = **−0.34** on average — rain cools. Negative in **75.4 %** of
  forecasts.

A field copied across leads would give a correlation of exactly zero, every time,
without exception. These do not.

**What was left stale, and the lesson in it.** The configuration constant
`V2_LEAD_INVARIANT_VARS` still listed both variables, and still carried a warning that
no model may be credited with skill from them — long after that stopped being true.
Anyone reading the config would have concluded the bug was live. It is now empty, and
the note beside it has been rewritten as a **tripwire** rather than a warning: if that
list is ever non-empty again, the download has reverted to instantaneous hours, and the
probe must be re-run before any model that reads those channels is trusted.

The general lesson is the one this document's revision note also illustrates: **a
warning that outlives the bug it describes is itself a defect.** It costs real time and
it teaches readers the wrong thing about the state of the system.

**One limit on this claim.** The verification covers the v2 archive (2004–2022). The
older v1 archive behind the `+0.1047` baseline has not been checked and may predate the
fix.

### A fourth, smaller trap

Downward solar radiation is called `ssrd` in the newer data source and `ssr` in the
older one. Same physical quantity, different name. The config warns: *"do not map one
onto the other silently."* Separately, the elevation field `orog` goes slightly
negative over the Bay of Bengal corner — that is the shape of the Earth's gravity
field, not missing data.

## The rainfall we are trying to predict: IMD

To train and score any rainfall model you need observations of what actually fell.
These come from the **India Meteorological Department (IMD)** gridded rainfall product.

| Property   | Value                                                                             |
| ---------- | --------------------------------------------------------------------------------- |
| Resolution | **0.25°**                                                                         |
| Grid       | 25 × 25 cells covering 18–24° N, 81–87° E                                         |
| Frequency  | daily                                                                             |
| Period     | 2003–2022                                                                         |
| Units      | millimetres per day — *inferred from how the code uses it, not stated explicitly* |

Not every cell in that 25 × 25 box has an observation. Some are over the sea, some
outside the land mask. Those cells are **masked out** — they are excluded from both
training and scoring. In an earlier version of this project they were filled with
zeros instead, which made 24 % of the target grid a free, trivially predictable
constant and badly inflated the score.

## The weather we use for the physics model: ERA5-Land

The physics model in Chapter 7 needs continuous historical weather to run. That comes
from **ERA5-Land**, a high-quality reconstruction of past weather produced by ECMWF.

It arrives at 0.1° resolution and is converted to the 0.25° grid using area-weighted
averaging. It supplies **six of the seven** weather inputs the physics model needs.

It does **not** supply rainfall. Rainfall stays IMD, deliberately — because IMD is what
the rainfall network in Chapter 5 is trained against, and mixing sources would break
the internal consistency of the whole chain.

Four details in building this forcing data are worth knowing, because each one changes
the answer if you get it wrong:

1. **Wind speed is averaged, not the wind components.** Averaging the east-west and
   north-south components separately and then combining them gives you the speed of the
   *average wind direction*, which is smaller whenever the wind changes direction — and
   monsoon winds change direction constantly.
2. **Radiation accumulates and resets at midnight UTC**, so a day's total is the value
   recorded at midnight of the *following* day. Averaging the raw samples instead would
   overstate radiation several times over.
3. **Daily averages use local days, except radiation, which cannot.** India is 5.5
   hours ahead of UTC, so a local day runs from 18:30 UTC to 18:30 UTC. Radiation's
   accumulation window is fixed to midnight UTC by the archive. The result is a real
   5.5-hour mismatch between the radiation day and the temperature day. **The project
   states this plainly and does not correct it.** Do not describe the radiation and
   temperature days as aligned — they are not.
4. **Regridding is area-weighted.** 0.25 divided by 0.1 is 2.5, not a whole number, so
   each target cell overlaps 3 or 4 source cells unevenly. Simply taking the nearest
   source cell would stamp a checkerboard pattern onto the data.

## The thing we are actually predicting: observed inflow

| Property | Value |
|---|---|
| Total rows | 5,607 daily records |
| Date range | 1 January 2003 to 31 October 2022 |
| Valid records | **5,255** |
| Invalid / flagged | 352 |
| Suspicious zeros flagged | 380 |

**There is a serious gap in this record.** For 2003–2014 it is nearly continuous —
roughly 290 to 363 valid days per year. From **2015 to 2022 there are only 153 valid
days per year**, covering June to October only.

This matters more than it sounds. It means the inflow series is **not continuous** after
2014, which will later reduce the number of scorable forecasts from 532 down to 309
(Chapter 12).

### The validity flag must be honoured everywhere

There is a trap inside this record that cost the project real work. The loader stores a
missing gauge day as the number **`0.0`**, and records its absence in a **separate
validity flag**. Read the numbers without reading the flag and a missing day looks like
a day on which no water at all entered an 83,400 km² reservoir.

Both the calibration and the routing evaluation ignored that flag for a period. **213 of
3,287 calibration days were flagged invalid, and all 213 were zeros**, concentrated in
January to June — exactly where they would most distort the dry-season water balance.
Honouring the flag moved the all-days NSE from 0.369 to 0.403.

The rule this produced: **one project, one definition of "observed"**.

### A near-miss worth recording

The 2015–2022 data arrived in a different unit from the 2003–2014 data. Read as cumecs
(m³/s), it implied a monsoon average flow of 63,254 m³/s and a peak of 868,135 m³/s.

The project checked this against physics rather than trusting the column heading. That
peak would be **twenty times the dam's spillway capacity**, and the implied runoff
would be **10,026 mm of water over a catchment that receives about 1,200 mm of monsoon
rain**. Water cannot appear from nowhere.

The file was actually in **cusecs** (cubic feet per second). Converted properly it gives
1,791 and 24,583 m³/s, which sits sensibly beside the 2004–2014 figures of 2,267 and
29,204. A permanent assertion now enforces this if the file is ever replaced.

## The fixed background data

Four datasets describe things that do not change:

| Dataset | What it provides |
|---|---|
| ETOPO 2022 | ground elevation |
| SoilGrids | soil properties, converted to model parameters |
| MODIS MCD12Q1 | land cover (forest, cropland, and so on) |
| HydroSHEDS | which direction water flows across the terrain |

## Chapter summary

Both approaches start from the same ECMWF S2S reforecast: 1.5° resolution, 26 weather
variables, 11 ensemble members of which 10 are used, 30 lead days, 532 forecast dates,
giving 15,960 (date, lead) rows. The control member is excluded because it is the
centre of the distribution rather than a sample from it, and mixing it in would corrupt
the ensemble spread that serves as the project's confidence signal. Rainfall
observations come from IMD at 0.25°, historical weather for the physics model from
ERA5-Land, and the prediction target from a 5,607-row inflow record that becomes
monsoon-only after 2014 and carries a validity flag that must be honoured everywhere.

Four traps in the weather archive have to be handled explicitly: accumulating rainfall
totals, cloud cover as a percentage, a solar-radiation variable with two different
names — and `t2m` and `tcc`, which **once** carried no forecast information at all.
**That fourth one is fixed and verified**; both variables now vary across the 30 lead
days in every initialisation, and both correlate with rainfall in the direction physics
requires. If you are reading an older copy of this document that says those two
variables still carry no information, that copy is out of date.

We now know what the models can see. The next question is why seeing it is not enough.

---
# Chapter 3 — Why Rainfall Is Not River Flow

Suppose we could forecast rainfall perfectly. Every millimetre, in exactly the right
place, on exactly the right day, a month in advance.

Would we then know the inflow at Hirakud?

No. And the four reasons why are the reasons this project needs complicated models
instead of simple arithmetic. **There are no models in this chapter.** Only physics.

## Reason 1 — the catchment has a memory

Pour a glass of water onto a dry kitchen sponge. Almost all of it soaks in. Barely
anything drips out.

Now take a sponge that has been sitting in a bucket of water for a week and pour the
same glass onto it. Almost all of it runs straight off the sides.

A river catchment behaves exactly like the sponge. What happens to rain depends
entirely on **how wet the ground already was** — a property hydrologists call
**antecedent moisture**.

100 mm of rain falling on dry soil in early June may almost entirely soak in, producing
very little river flow. The same 100 mm falling in August, after weeks of monsoon rain
have saturated the ground, becomes almost pure runoff and surges into the river.

### The measurement

The project has a diagnostic tool that traces a single forecast through every stage.
Run on the forecast issued 4 July 2018, it reports the **runoff ratio** — the fraction
of falling rain that becomes runoff — at six different lead days:

| Lead day | Rain (mm/day) | Runoff (mm) | **Runoff ratio** |
|---|---|---|---|
| 1 | 20.18 | 1.05 | **0.05** |
| 3 | 5.95 | 1.03 | **0.17** |
| 7 | 11.49 | 1.27 | **0.11** |
| 14 | 6.02 | 1.82 | **0.30** |
| 21 | 4.00 | 2.52 | **0.63** |
| 30 | 3.30 | 4.95 | **1.50** |

At lead day 1, only 5 % of the falling rain becomes runoff. By lead day 30, runoff
*exceeds* the rain falling that day — because water that fell earlier is still working
its way through the system.

Notice that the ratio does not rise smoothly. It **falls** from 0.17 to 0.11 between
days 3 and 7. That wobble is itself informative: the ratio is not a simple function of
how far ahead you are looking. It depends on the accumulated state of the catchment.

The project's own explanation is short: *"Rain on already-wet soil runs straight off;
the same rain on dry soil soaks in. This is why VIC needs the state, and why a forecast
cannot cold-start."*

**Important caveat.** This is one forecast window, not a statistic. The diagnostic
tool's own closing note says a good year and a bad year look completely different.

### How long is the memory?

Not very long, as it turns out. The correlation between today's inflow and inflow a
week ago is **0.48**. Between today and 60 days ago it is **−0.01** — essentially
nothing.

The project measured something more useful still: **7 days of history predicts about as
well as 64 days**. That fact will come back in Chapter 9, where a model is given 64 days
of history anyway — and again in Chapter 11, where cutting the correction network's
history window from 60 days to **15** turned out to be the single largest in-paper
improvement anywhere in the physics chain.

## Reason 2 — the forecast is far too coarse, and averaging destroys information

At the Mahanadi's latitude, one 1.5° ECMWF grid cell measures roughly 167 km by 156 km.
That is about **26,000 km²** — and ECMWF reports a **single rainfall number** for all of
it.

The whole 83,400 km² catchment is therefore about **three grid cells' worth of area.**

That single cell contains Chhattisgarh uplands, forest and the Mahanadi valley floor.
All averaged into one number.

### Why averaging is not just imprecise — it is wrong

Because runoff is not proportional to rainfall, the following is true:

> **runoff(average rainfall) ≠ average of runoff(rainfall)**

Here is what that means concretely.

Suppose 100 mm of rain falls, but concentrated in one quarter of a 26,000 km² cell.
That quarter saturates quickly and produces heavy runoff into the river.

Now suppose the weather model reports the same total spread evenly: a gentle 25 mm
everywhere. At 25 mm the soil across the whole cell may absorb all of it, producing
almost nothing.

**Same total volume of rain. Completely different river response.** Averaging rainfall
before running the physics systematically loses flood peaks.

This is why the catchment has to be broken into finer pieces. In this project it is
represented by **224 cells at 0.25° resolution**, and the physics model computes a water
balance on **150 active cells**.

## Reason 3 — water takes time to travel

Rain and runoff are measured as a **depth**: millimetres spread over an area.

Reservoir inflow is measured as a **discharge**: a volume passing a single point every
second, in m³/s.

Converting between them needs two things. First, the **area** of each cell, to turn a
depth into a volume. Second, the **travel time**, because water does not teleport.

Rain falling on a cell a few hundred kilometres upstream does not arrive at the dam that
afternoon. It has to work down through streams into tributaries into the main channel.
So today's inflow is a **mixture**: water from nearby storms yesterday, plus water from
distant storms several days ago.

Any honest forecast has to account for that spread of arrival times. Chapter 7 shows how.

## Reason 4 — uncertainty grows, and it is mostly uncertainty about *how much*

The further ahead you look, the more the ten ensemble members disagree with each other.
That disagreement is called the **ensemble spread**, and it is a real measure of how
confident the forecast is — it correlates **+0.475** with the size of the error that
actually follows.

Now the part that earlier versions of this document got backwards, and that matters
enough to state carefully.

**When the finished forecasts go wrong, they go wrong mainly about the amount of water,
not about the day it arrives.** Chapter 15 gives the experiment; the summary is:

| what was allowed to change | median NSE |
|---|---|
| the chain driven by *observed* weather, as simulated | 0.394 |
| **the same, plus one single volume multiplier** | **0.850** |
| the same, plus the best possible per-window time shift | 0.471 |
| the actual forecast chain | 0.200 |

Allowing the forecast to slide freely in time buys almost nothing. Allowing it a single
scaling factor more than doubles the score. And the average best time shift turns out to
be **0.28 days** — which is to say, essentially none.

### Why the old answer was wrong, and why it survived so long

An earlier analysis decomposed the squared error into bias, amplitude and phase
components and found phase accounting for **66–95 %** of it. That analysis was not
arithmetically wrong. It was run on the wrong thing.

It was run on the **forecast** chains — `EC → VIC → RVIC` and friends. In those chains
the rainfall arriving at the water model is already in the wrong week, because ECMWF put
it there. Decomposing the *output* error therefore measured two different things at once:

- the rainfall forecast's timing error, which belongs to **ECMWF**, and
- the water model's own error, which belongs to **VIC**

...and attributed the sum to the water model. The conclusion drawn from it — "timing
dominates, so no downstream model can help" — was a statement about the weather forecast
wearing the water model's name.

The rule that came out of this is worth more than the finding: **to measure one
component's error, hold its inputs perfect.** Feed the water model observed rainfall and
its phase error essentially vanishes.

### Both things are true at once

This is not a claim that timing is fine everywhere. It is a claim about *which stage
owns the problem*. The finished forecasts really do place the seasonal peak about
**seven days** away from where it happened — around 7.0 days for the baseline and the raw
chain, 7.9 for the full physics chain. That error is real and it is large.

But it enters at the **rainfall** stage, from ECMWF, and it is already present in the
raw forecast before any part of this project touches it. What this project's own
modelling contributes is a **volume** error, and that is the part that could have been
fixed downstream.

## Putting it together: one forecast, end to end

Here is the same 4 July 2018 forecast traced through the whole physical chain, with the
units at each step:

```
ECMWF S2S           10 members × 30 leads × 26 variables, 1.5° grid    mm/day
   |  sharpen the rainfall map (Chapter 5)
0.25° rainfall      224 catchment cells × 30 days                      mm/day
   |  water balance, from a saved soil state (Chapter 7)
runoff              150 cells × 30 days                                mm/day
   |  unit-hydrograph routing, 10 lags, × cell area (Chapter 7)
inflow at the dam   30 days                                            m³/s
   |  compare
observed inflow     30 days                                            m³/s
```

**The units change three times, and each change is a physical step, not bookkeeping.**
If you understand why mm/day becomes m³/s, you understand the hydrology.

## Chapter summary

Even a perfect rainfall forecast would not give a perfect inflow forecast, for four
independent reasons. The catchment remembers — the same rain produces 5 % or 150 %
runoff depending on how wet the soil already was. The forecast grid is so coarse that
the entire catchment is about three cells, and because runoff is non-linear, averaging
rainfall before running the physics loses flood peaks. Water takes days to travel, so
inflow is a mixture of storms from different days. And forecast uncertainty grows with
lead time.

On that last point, note the correction: the uncertainty expresses itself mainly as
error in **how much** water arrives, not when. The seven-day peak-timing error in the
finished forecasts is real, but it is inherited from ECMWF's rainfall forecast rather
than produced by the hydrology — and it was the failure to separate those two that made
this project believe the opposite for months.

Two of these four problems — the coarse grid and the loss of spatial detail — look as
though they could be fixed upstream, by making the rainfall forecast sharper. That is
exactly the bet Approach 1 is built on. Part II tests it.

---

# PART II — APPROACH 1: THE PHYSICS ROUTE

> This part rebuilds the method of **Dong et al. (2025)**. It is **not an exact
> reproduction**, and the differences matter enough to list up front.
>
> **Followed faithfully:** the ResNet architecture of their section 3.2.1 (64 → 32 → 16
> feature maps, 3 × 3 kernels, ELU activations, coordinate embeddings); the **3 × 3 patch
> size**, which is a finding of theirs rather than a default; the form of the loss
> function, squared error plus a differentiable threat-score term; downscaling each
> ensemble member separately; the role of the correction network as a post-processor of
> the hydrological model; and the 30-day horizon.
>
> **Seven deviations:**
>
> | # | Component | Dong et al. | This project |
> |---|---|---|---|
> | **1** | **Hydrological model** | **XAJ** (lumped, conceptual) | **VIC 5** (semi-distributed, 150 cells) |
> | 2 | Predictors | 19 | **26** (a superset) |
> | 3 | Squared-error term | *(their specification)* | computed on **raw millimetres**; log space tried and rejected |
> | 4 | Loss masking | *not verified* | every term masked on observation availability |
> | 5 | Forecast temperature | **delta-corrected** | ECMWF `t2m` used as-is |
> | 6 | Longwave radiation, vapour pressure | paper is silent | **day-of-year climatology** |
> | 7 | Routing | *not verified* | RVIC 1.1.0 with Dominant River Tracing |
>
> Deviation 1 is the largest: it replaces the physical core of their method. Chapter 7
> explains it. Deviations 5 and 6 exist because the weather archive has no forecast
> equivalent for those variables. Deviation 4 was necessary to fix a defect described in
> Chapter 16.
>
> **The basin differs too, in a way that drives the whole result.** Their raw forecast
> carries about 27 % relative error; the Mahanadi's carries **+4.114 %**. Chapter 6 shows
> why that single difference explains most of the gap between their reported outcome
> (−34 % rainfall error) and this project's (−5.888 %).

---
# Chapter 4 — Why Raw ECMWF Rainfall Is Not Enough

Before building anything, it is worth being precise about the problem, and answering
the obvious objection: why not just stretch the coarse map onto a finer grid?

## The arithmetic of the resolution gap

At about 21° N:

| | Cell size | Cell area |
|---|---|---|
| ECMWF forecast, 1.5° | ~167 × 156 km | **~26,000 km²** |
| IMD observations, 0.25° | ~28 × 26 km | **~720 km²** |

The ratio is **36 fine cells inside every coarse cell** — a 6 × 6 grid.

And the catchment, at 83,400 km², is about **three coarse cells' worth of area**,
represented by **224 fine cells**.

Three numbers describing an area the size of Austria. That is the problem in one line.

## Why you cannot simply interpolate

The obvious response is to smooth the coarse map onto the fine grid — bilinear
interpolation, or something like it.

**This cannot work, and the reason is fundamental: interpolation does not add
information.** It produces a smoother surface with more pixels and exactly the same
information content. It cannot know that one 25 km cell sits on a windward ridge where
rain is forced upwards, while its neighbour sits in a rain shadow. Nothing in the coarse
number distinguishes them.

## What a learned model can use that interpolation cannot

A trained model has access to two things interpolation does not:

**The spatial pattern of the coarse field.** Instead of looking only at the single
coarse value sitting above the target cell, it can look at a **neighbourhood** of coarse
cells — so it sees the gradient and structure of the weather system, not just its
magnitude at one point.

**Fixed terrain information.** Elevation and position never change, and they exert a
strong control on where monsoon rain falls. Windward slopes get far more than valley
floors.

## Why the physics model needs fine input anyway

There is a second, independent reason. The physics model in Chapter 7 runs a separate
water balance on **150 grid cells**, each with its own soil properties and land cover.
It needs a rainfall value **per cell, per timestep**.

Feed those 150 cells from about 3 coarse values and every cell inside a coarse box gets
identical rain. The spatial variation the model exists to represent is erased at the
input, before the model has done anything.

## The evidence that it is worth trying — and a correction

Earlier versions of this document made a strong claim here, and it does not survive
checking. The claim was:

> ~~**Raw ECMWF scores R² = −0.136.** Negative — worse than ignoring the forecast.
> After downscaling, the same score is **+0.057**. The **sign flip** is the point.~~

**There is no sign flip.** The −0.136 figure appears in no results file and is produced
by no script in this repository; it traces only to `archive/historical_docs/`, which the
project's own rules forbid quoting for results. Measured properly, on the same 224
cells, the same observation mask, the same 19 leave-one-year-out folds and the same
ensemble-mean regime the downscaler is scored in (`tools/raw_ec_percell_baseline.py`,
`results/metrics/raw_ec_percell_baseline.json`):

| product, per-cell R² | mean | median | folds positive |
|---|---|---|---|
| **raw ECMWF, nearest coarse cell** | **+0.044** | +0.045 | **17 / 19** |
| raw ECMWF, 3 × 3 smoothed | **+0.073** | +0.073 | 19 / 19 |
| **downscaled (the CNN)** | **+0.057** | +0.059 | 19 / 19 |
| *a constant, as a sanity check* | *−0.003* | *−0.002* | *0 / 19* |

Raw ECMWF is **already positive** at the fine cell level. The downscaler's honest gain
over it is about **+0.013**, not a rescue from −0.136.

### Where −0.136 came from

Almost certainly from **per-member scoring**. Score raw ECMWF one ensemble member at a
time and average the results, and it collapses to **−0.349** — comfortably negative, and
the only regime in which a figure like −0.136 is even plausible. But the downscaler's
+0.057 is an **ensemble-mean** score: `loyo_percell_v2.py` averages the ten members
before scoring (`out[b] = acc / n_mem`).

So the old comparison put a per-member baseline next to an ensemble-mean model. That is
not a small slip — averaging ten members removes a large amount of noise, and it is
worth roughly 0.39 of R² here, which is far more than anything the network does.

**This project has now made that exact mistake three times.** `docs/FINDINGS.md` records
the first two under "Two scoring regimes were mixed twice", which is why
`inflow/evaluate_operational.py` was written to define one sample and one regime. That
discipline was applied to the *inflow* metrics and never applied to the *rainfall* ones.

### The harder problem this exposes

Look again at row 2 of the table. Simply **averaging the 3 × 3 block of coarse cells** —
crude smoothing, no learning, no terrain, no training — scores **+0.073**, which is
*better* than the trained network's +0.057, and positive in all 19 folds rather than 17.

That sits very awkwardly beside this chapter's argument that interpolation cannot help
because it adds no information. The argument is sound in principle: smoothing genuinely
adds no information. What the number says is that **this particular target rewards
smoothness**, because a squared-error score on a noisy, heavily right-skewed field
favours a conditional mean — which is exactly the behaviour Chapter 5 identifies in the
network itself, and exactly why its heavy-rain and CRPS scores come out worse than doing
nothing.

In other words: the network and plain smoothing are being pushed toward the same
solution by the same metric, and at this scale smoothing gets there more cleanly.

**What honestly remains of the case for downscaling** is the second reason above, not
this one: the hydrological model needs a value per cell for 150 cells, and it cannot be
given one from three coarse numbers. The per-cell R² does not carry the argument.

**This correction is not yet reflected in `docs/PIPELINE_1_EXPLAINED.md`**, which quotes
−0.136 in two places (around lines 952 and 1875).

## Chapter summary

One ECMWF cell covers about 26,000 km² and contains 36 IMD cells; the whole catchment is
about three coarse cells. Interpolation cannot fix this because it adds no information —
it only adds pixels. A learned model can use the spatial pattern of the coarse field
plus fixed terrain, neither of which interpolation has access to. And the physics model
needs a per-cell rainfall value for 150 cells regardless — and that last reason is the
one that survives measurement. The long-quoted claim that raw ECMWF scores R² = −0.136
at the fine cell level, and that downscaling flips the sign, is **wrong**: it compared a
per-member baseline against an ensemble-mean model. Scored in the same regime, raw ECMWF
is **+0.044**, the downscaler is +0.057, and crude 3 × 3 smoothing is +0.073. Chapter 5
describes the network that produces that +0.057, and what it does and does not buy.

---

# Chapter 5 — The Rainfall Downscaler

This chapter describes the neural network that sharpens the rainfall forecast, exactly
as implemented, and reports what it achieved — including the two places where it made
things **worse**.

## The idea, in one paragraph

Ask a human forecaster to predict rainfall in one valley and they do not just look at
the humidity directly overhead. They look at the surrounding system — a front moving in,
wind pushing moist air up a ridge.

The network works the same way. Rather than predicting the whole 25 × 25 rainfall map at
once, it predicts **one fine cell at a time**, using a small patch of coarse weather
*around* that cell plus the cell's own geography. Run it for all 224 catchment cells and
you have rebuilt the map.

## What goes in

For each prediction — one fine cell, on one forecast date, at one lead day — the network
receives two things:

**A weather patch**, shaped `(26, 3, 3)`. That is all 26 weather variables, on a **3 × 3
block of coarse cells centred on the target cell**.

**A location vector**, shaped `(3,)`: latitude, longitude, and elevation of the target
cell.

### Why 3 × 3, and why it nearly went wrong

The patch size is not arbitrary. Dong and colleagues tested 1×1, 3×3, 5×5 and 7×7, and
found 3×3 best. It is a finding, not a default.

But there was a problem. An earlier version of this project downloaded only a 5 × 5 box
of coarse cells. For cells near the edge of that box, a 3 × 3 patch runs off the grid —
and the code filled the gap by repeating the edge row. **48 % of patches were affected**,
meaning nearly half the training data contained duplicated rows presented as if they
were real measurements.

Re-downloading on a 7 × 7 box fixed it completely: **0 % of patches are now clamped**,
and the code asserts this rather than assuming it.

## What comes out

A single number: rainfall for that cell, in mm/day.

## The architecture, layer by layer

The network is a **ResNet** — short for residual network. The key idea of a ResNet is
the **skip connection**: alongside the usual processing, the original input is added
back to the output. This lets a network go deep without losing track of what it started
with.

For a batch of B samples:

| Stage | Operation | In | Out |
|---|---|---|---|
| Input A | weather patch | — | (B, 26, 3, 3) |
| Block 1 | ResBlock 26 → 64 | (B, 26, 3, 3) | (B, 64, 3, 3) |
| Block 2 | ResBlock 64 → 32 | (B, 64, 3, 3) | (B, 32, 3, 3) |
| Block 3 | ResBlock 32 → 16 | (B, 32, 3, 3) | (B, 16, 3, 3) |
| Flatten | 16 × 3 × 3 | (B, 16, 3, 3) | (B, 144) |
| Input B | coordinates | — | (B, 3) |
| Embed | Linear(3 → 16) + ELU | (B, 3) | (B, 16) |
| Join | concatenate | (B,144)+(B,16) | (B, 160) |
| Head 1 | Linear(160 → 64), ELU, Dropout | (B, 160) | (B, 64) |
| Head 2 | Linear(64 → 32), ELU, Dropout | (B, 64) | (B, 32) |
| Output | Linear(32 → 1), **softplus** | (B, 32) | (B, 1) |

Each ResBlock is: a 3×3 convolution with padding 1, then ELU activation, then spatial
dropout, then a second 3×3 convolution, then the skip connection added, then ELU again.

Four design choices worth explaining:

**Padding of 1 on every convolution.** The patch is only 3 × 3. Without padding it would
shrink to nothing after one layer.

**Spatial dropout (`Dropout2d`) rather than ordinary dropout.** Ordinary dropout zeroes
individual pixels. On a 3 × 3 patch that barely removes any information. Spatial dropout
zeroes **whole feature channels**, forcing the network not to depend on any single
weather variable.

**Coordinates enter through an embedding, not as extra image channels.** This is a
deliberate choice (the code cites Rasp and Lerch, 2018). It lets the network express
"this place behaves differently" without making the convolutions encode absolute
position.

**A softplus at the output.** Rainfall cannot be negative. Softplus guarantees a positive
output while avoiding the dead gradients that ReLU would produce on dry days — and dry
days dominate the data.

## The loss function

The network is trained to minimise:

```
Loss  =  MSE  +  b × (1 − TS)          with b = 1.0
```

**MSE** is mean squared error — ordinary prediction error.

**TS** is the **threat score**, a standard forecasting measure of how well you catch
heavy rain events:

```
TS = Hits / (Hits + False alarms + Misses)
```

An event counts as heavy rain if it exceeds a threshold — here, the **90th percentile of
observed rainfall for that specific cell**, computed on the training years only.

Why include it? Because plain MSE would push the network to smooth everything toward
zero. Dry days massively outnumber wet ones, so predicting "not much rain" is a safe
average strategy. The threat-score term makes that strategy expensive.

### Making a hard threshold differentiable

There is a technical obstacle. The threat score is **categorical** — a prediction either
exceeds the threshold or it does not — and categorical decisions have no gradient, so a
network cannot learn from them.

The solution: replace the *forecast* side of each decision with a smooth sigmoid curve
centred on the threshold, while the *observation* side stays a hard yes/no. That is
fine, because no gradient needs to flow back through the observations.

### Every term is masked

Cells where IMD has no observation contribute **nothing** to the loss and nothing to the
score. This is not a detail. In the earlier version of this project, missing cells were
filled with zeros, which made 24 % of the target grid a free constant that the model
could "predict" perfectly. That was one of six defects that together inflated a reported
score to R² 0.87 when the honest figure was about 0.078 (Chapter 16).

### Two contradictions in the repository — both now RESOLVED

Earlier versions of this document flagged two places where the code and the
configuration disagreed with each other. **Both have since been fixed.** They are kept
here because the *shape* of each disagreement is instructive, and because anyone holding
an older copy of this document needs to know the warnings no longer apply.

**Contradiction A — was the error computed in log space? RESOLVED.**

`cnn/model.py` used to declare `log_space=True` as its default, with a docstring arguing
strongly for computing the error on `log(1 + rain)` so that light rain influences
training. Meanwhile `configs/config.py` set `LOSS_LOG_SPACE = False`, recording that log
space *"made every metric worse in real-mm terms (test R² 0.116 → 0.019, bias −1.5 mm →
−4.6 mm, Spearman 0.51 → 0.48)"* — because it optimises something closer to the median,
and this target's average sits well above its median, so the model systematically
under-predicted.

The production script always passed the configuration value explicitly, so the published
results were never affected. But **any new code calling the loss function without
naming the argument would silently have got the rejected behaviour** — a live trap.

The default is now `log_space: bool = False`, matching the configuration, and the
docstring has been rewritten so the argument for log space is clearly labelled as *why
it was tried*, immediately followed by the measurement that rejected it and an
instruction to re-check light-rain R² and bias — not just overall R² — before anyone
re-enables it. Code and config now agree, and the losing option can no longer be
selected by accident.

**Contradiction B — the threat-score threshold fallback. RESOLVED.**

A cell with very few observations cannot supply a trustworthy 90th percentile. The
configuration said such a cell should fall back to the **domain-wide** training
threshold. The code set it to **0.0** instead — and a threshold of zero makes *every*
prediction a "hit" for that cell, silently removing it from the threat-score term
altogether. A cell with too little data to threshold was being quietly excused from the
part of the loss that exists to handle heavy rain.

The code now computes the domain-wide 90th percentile over the training rows and uses
that, exactly as the configuration always specified:

```python
_dom = y_cells[train_k][m_cells[train_k]]
dom_p90 = float(np.percentile(_dom, C.HEAVY_RAIN_PERCENTILE)) if _dom.size else 0.0
...
    if len(v) >= C.HEAVY_RAIN_MIN_OBS:
        thr[j] = np.percentile(v, C.HEAVY_RAIN_PERCENTILE)
    else:
        thr[j] = dom_p90
        n_fallback += 1
if n_fallback:
    print(f"    {n_fallback}/{len(cells)} cells below "
          f"{C.HEAVY_RAIN_MIN_OBS} obs -> domain p90 {dom_p90:.2f} mm")
```

Two things changed, and the second matters as much as the first. The fallback is now
correct (`dom_p90`, with `HEAVY_RAIN_MIN_OBS = 30` and `HEAVY_RAIN_PERCENTILE = 90`).
And the run now **prints how many cells took the fallback**, so the question the earlier
version of this document had to record as "not verified" is answered by every training
run from now on. A silent fallback became a reported one.

## How it was trained

| Setting | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Gradient clipping | norm 1.0 |
| Maximum epochs | 60 |
| Early-stopping patience | 8 |
| Batch | **16 forecast fields** = 16 × 224 = **3,584 cell-samples per step** |
| Ensemble members used | **10 perturbed** — the control member is excluded (Chapter 2) |
| Dropout | 0.2 |
| Random seed | 42 |
| Folds | 19 (leave-one-year-out) |

**One detail that is easy to miss.** During **training**, each step picks **one ensemble
member at random**. During **scoring**, the network predicts all 10 members and the
**average** is used. That average is what the reference paper scores, and what an
operational system would use.

Mixing those two regimes up is a live hazard in this project: per-member and
ensemble-mean scoring give genuinely different answers, and conflating them produced
wrong conclusions **twice** before `inflow/evaluate_operational.py` was written to
define one sample and one regime for everybody.

### How the folds are built

```python
test_k  = init_year == Y                        # the held-out year
inner   = [y for y in YEARS if y != Y][-1]      # the LAST year that isn't Y
val_k   = init_year == inner                    # used for early stopping
train_k = ~test_k & ~val_k                      # the other 17 years
```

Within each fold, **the standardisation statistics and the rainfall threshold are
recomputed on the training years only.** Nothing from the test year leaks into training.

Splits are on **initialisation year**, never at random. The reason is in the config:
one valid date is reached by about three different forecasts at different lead times, so
a random split would put the same observed rainfall map on both sides.

**One weakness worth flagging.** Because the validation year is "the last year that is
not the test year," **2022 is the validation year in 18 of the 19 folds.** The test year
stays clean, so results are not invalidated — but the decision of when to stop training
is informed by 2022 almost every time. *This is an observation from the audit, not
something the repository records.*

## Results: what improved

**Per-cell R², across 19 folds:**

| | |
|---|---|
| Mean | **+0.0571** |
| Median | **+0.0589** |
| Worst fold | **+0.0097** |
| Folds with a positive score | **19 out of 19** |

Compare with raw ECMWF **scored in the same ensemble-mean regime: +0.044**, positive in
17 of 19 folds. The gain is real but small — about **+0.013** — and it is *not* the sign
flip from −0.136 that this document used to claim. See Chapter 4 for why that figure was
wrong and where it came from.

**Five-day rainfall totals ("pentads"), 715,008 of them across 19 monsoons:**

| Product | RMSE, all | RMSE, heavy | Relative error | CRPS | vs raw |
|---|---|---|---|---|---|
| Raw ECMWF | 46.748 | **111.886** | **+4.114 %** | **23.928** | — |
| Quantile-mapped | 49.604 | 111.531 | −0.055 % | 24.192 | **−6.109 %** |
| **Downscaled** | **43.995** | 112.579 | −1.306 % | 25.676 | **+5.888 %** |

Heavy pentads are those above 99.31 mm.

## Results: what did NOT improve

Look at the bold entries in the raw ECMWF column. **On two metrics, doing nothing beats
the network:**

- **Heavy rainfall:** 112.579 versus 111.886. The network is slightly *worse* on exactly
  the events that matter most for flood management.
- **CRPS**, a score that rewards well-calibrated spread: 25.676 versus 23.928. Worse.

## Why: the network predicts the middle, not the extremes

The explanation is in the project's own diagnostics. The network achieves:

- **Spearman correlation 0.51** — a measure of how well it *ranks* days from dry to wet
- **Pearson correlation 0.36** — a measure of how well it gets the *magnitude* right

**Ranking good, magnitude poor.** That is the signature of a model predicting a
conditional average. Faced with uncertainty, minimising squared error pushes it toward
the middle of the plausible range. So it **damps extremes**.

Hold on to this. It explains everything in Chapter 6 — and it is the same shape of error
as the project's central finding in Chapter 15, where the whole chain turns out to be
wrong about *magnitude* rather than *placement*.

## Why quantile mapping made things worse

Quantile mapping is pure statistical bias correction: it stretches the forecast's
distribution until it matches the observed distribution. It adds no spatial reasoning.
It was included as a **control** — if the network's only real achievement were removing
bias, quantile mapping should match it.

It scored **−6.109 %** — worse than doing nothing.

The reason is in the same table. **Raw ECMWF here has a relative error of only
+4.114 %.** There is barely any systematic bias to remove. Stretching the distribution
of a low-skill forecast adds amplitude without adding information.

This result matters enormously in the next chapter.

## A claim in the repository that needs correcting

`docs/FINDINGS.md` once stated that the −5.888 % figure *"replicates exactly"* across
*"two independent datasets"*, citing **5.888477 %** on the older 11-monsoon dataset and
**5.888199 %** on the 19-monsoon one — a relative difference of 0.000047.

**The two datasets are not independent.** They share the years 2004–2014 — **11 of the
19 years, 58 % of the sample.** Agreement between overlapping datasets is not
corroboration.

Furthermore, agreement to five significant figures is far tighter than sampling noise
should allow, while the relative-error figure on the very same pair moved from +1.662 %
to +4.114 %. Whether some shared cached input explains this is **not verified**.

**The defensible statement is:** the pentad RMSE gain is stable at about 5.9 % when
eight more years are added.

## Chapter summary

The downscaler is a ResNet that predicts one 0.25° cell at a time from a 3 × 3 patch of
26 coarse weather variables plus the cell's latitude, longitude and elevation. It is
trained with squared error plus a differentiable threat-score term so that heavy rain is
not smoothed away, with every term masked to exclude unobserved cells, over 19
leave-one-year-out folds on 10 perturbed ensemble members.

**It works, on rainfall — but by less than this document used to claim.** Per-cell R²
is +0.057, positive in all 19 folds, against raw ECMWF's **+0.044** in the same
ensemble-mean regime: a gain of about +0.013, not a sign flip from −0.136 (Chapter 4).
Crude 3 × 3 smoothing scores +0.073 on the same basis, which is better than the network.
The firmer evidence is at pentad scale: five-day rainfall totals improve by 5.9 % against
the raw forecast, while pure statistical bias correction, offered the same opportunity,
makes them 6.1 % worse. That contrast is what shows the network is doing something
spatial rather than merely rescaling.

**It does not work on the two things that matter most.** Heavy-rain RMSE and CRPS both
come out slightly worse than doing nothing, because a network minimising squared error
predicts the middle of the plausible range and damps the extremes — ranking days well
(Spearman 0.51) while getting their magnitude poorly (Pearson 0.36).

Both of the repository contradictions this chapter used to warn about — the `log_space`
default and the threshold fallback — **have been fixed in the code**, and the fallback
now reports itself on every run rather than happening silently.

So the rainfall is better. The obvious next question is whether better rainfall produces
a better inflow forecast. Chapter 6 answers it, and the answer is no.

---
# Chapters 6 onward — not yet revised

**This revision covers the front matter and Chapters 1 to 5 only.** The draft supplied
for revision was cut off partway through Chapter 5's summary, so Chapters 6 to 16 and
Appendix A were not available to edit and are **not included here**. Nothing in them has
been silently altered or silently dropped — they simply were not in front of me.

If you want them revised too, supply them and they will be updated the same way. In the
meantime, here is what is known to need changing in them, checked against the code and
results files as they stand on 14 September 2026, so the older copy can be read with the
right corrections in mind.

| chapter | what needs changing | correct as of today |
|---|---|---|
| **6** | The comparison of the downscaled chain against raw ECMWF must be made **at equal volume**, or it measures bias cancellation rather than skill. | Equal-volume: `EC-CNN` **0.157** vs raw `EC` **0.168**. The CNN is slightly *worse*. Its apparent advantage came from a dry catchment mean (+0.7 %) offsetting VIC's wet bias (+31.8 %) — two errors of opposite sign. |
| **7** | VIC calibration numbers; the three-way split. | The calibration/validation/held-out split (2004–2011 / 2012–2014 / 2015–2022) is now actually computed, not merely described. Honouring the inflow validity flag moved the all-days NSE from 0.369 to **0.403**. |
| **7** | The soil-depth bounds story. | The optimiser drove soil depth to **every ceiling it was given** — 1.404 of 1.50, then 2.9987 of 3.00, then 3.9991 of 4.00. This is a missing process, not a calibration success: the basin is heavily irrigated and VIC has no abstraction term. Observed runoff coefficient **0.286** against a published natural 0.35–0.40 — **18–29 % of natural runoff never reaches the gauge**. Bias grows monotonically across calibration (+50 %), validation (+58 %) and held-out (+72 %) periods, which is what increasing irrigation looks like and what a calibration error does not. |
| **9 / 11** | The correction network's history window. | Cut from 60 days to **15**. Sweep on the 309 common windows: 7 → 0.192, **15 → 0.200**, 30 → 0.146, 45 → 0.177, 60 → 0.140. This was the largest in-paper improvement in the physics chain, and it follows directly from Chapter 3's finding that 7 days of history predicts about as well as 64. |
| **11** | The rainfall fed to the correction network. | It is now **product-matched** on `(init, lead)`. Previously raw ECMWF rainfall was fed to every product including the downscaled one. Matching is by **date**, never by row position. |
| **11** | What the correction network actually does. | It is a **bias layer**. Three independent tests agree. But report **median, mean and win-count together**: on the median it loses to a single scalar, while on the mean it wins, because it rescues the catastrophic years (+0.38, +0.30, +0.20 in the worst) and costs at most −0.06 in the good ones. Any one of those three numbers alone tells a false story. |
| **12** | Evaluation windows and the frozen lookback. | 309 common windows, 19 years. `EVAL_PAST = 60` is frozen inside `evaluate_operational.py` — it must **not** import the correction network's history length, or tuning that length would silently change which windows every other model is scored on. |
| **13** | The head-to-head table. | Use the table in Chapter 1 of this revision. It now includes the eighth row, `EC-CNN → VIC → RVIC → LSTM` — Dong et al.'s actual proposed system, which had been computed but never tabulated. |
| **15** | **The central finding is reversed.** | The error is **volume**, not phase. Perfect rainfall as simulated 0.394; plus one volume scalar **0.850**; plus optimal per-window time shift only 0.471; mean optimal shift **0.28 days**. The old 66–95 % phase attribution was measured on forecast chains, where the timing error belongs to ECMWF. |
| **15** | What the baseline means. | `0.86 × EC` is not a naive baseline — it **is** the physics chain with its volume error removed by a single number. That nothing beats it is not a failure of the models; it is the statement that volume was the error. |
| **16** | The bug list. | Add: calibration multipliers compounding across runs (fixed with immutable base files and a guard that refuses to start from a calibration output); fabricated zeros in the calibration target (213 of 3,287 days, all zeros, all January–June); the spin-up year being scored despite the docstring saying it was discarded; and the scrambled forecast dates in `quantile_mapping.py`, which is the worst defect found and the last — day-1 rainfall correlation went from **−0.02 to +0.68** when it was fixed. |
| **anywhere** | FutureTST. | **All FutureTST numbers predate the date-label repair and have not been re-run.** They are provisional. Mark them wherever they appear. |
| **Appendix A** | The contradiction list. | Two entries can be struck: the `log_space` default and the threat-score threshold fallback are both fixed in the code (Chapter 5 above). |

Two working-practice notes that belong in Chapter 16 and are easy to lose:

- **Static analysis is less reliable than running the thing.** Three times a grep or an
  import graph said a file was safe to remove and a smoke test disagreed — most sharply
  `cnn/train_gbm.py`, whose removal broke ten modules including the entire correction
  chain. Provisioning scripts look dead to a dependency graph: nothing imports
  `merge_inflow_2015_2022.py`, but it builds the inflow record.
- **`io.open(path, 'w').write(expr)` truncates before evaluating `expr`.** An exception
  inside the argument leaves a zero-byte file. Write via a temporary file and
  `os.replace`.

---

*Revised 14 September 2026 against `head_to_head.json`, `fair_comparison.json`,
`mse_decomposition.json`, `loyo_percell_v2.json`, `raw_ec_percell_baseline.json`,
`configs/config.py`, `cnn/model.py`, `cnn/loyo_percell_v2.py` and
`data/processed/percell_v2_full.npz`. The Chapter 2 Trap 3
verification figures and the Chapter 3 correlation figures were re-measured directly
from the training archive for this revision and supersede the values in earlier copies.*
