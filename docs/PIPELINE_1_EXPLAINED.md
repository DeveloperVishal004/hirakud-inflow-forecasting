# Pipeline 1, Explained From Scratch

### Predicting how much water will arrive at Hirakud Dam, 1 to 30 days ahead

**This document assumes you know nothing about rivers, weather forecasting,
statistics, or artificial intelligence.** Everything is built up from the
beginning. If a word is technical, it is explained the first time it appears and
again in the glossary at the end.

```
ECMWF forecast → CNN downscaling → VIC → RVIC → LSTM → inflow forecast
```

Those five names mean nothing yet. By the end of §6 they will.

> **Current as of 14 September 2026.** If you are holding an older printout,
> six of its main conclusions have since been proven wrong by measurement.
> §12 lists them. The most important reversal: the old version says our errors
> are mostly about *timing*. They are mostly about *quantity*.

---

## Contents

**Part I — The problem**
| § | |
|---|---|
| [1](#1-the-five-minute-version) | The five-minute version |
| [2](#2-the-dam-and-the-decision) | The dam, and the decision |
| [3](#3-what-exactly-are-we-predicting) | What exactly are we predicting |
| [4](#4-why-this-is-hard) | Why this is hard — four reasons |

**Part II — The tools**
| § | |
|---|---|
| [5](#5-what-a-model-is-and-what-training-means) | What a "model" is, and what "training" means |
| [6](#6-the-five-stages-in-plain-words) | The five stages, in plain words |

**Part III — The stages in detail**
| § | |
|---|---|
| [7](#7-stage-1--the-weather-forecast) | Stage 1 — the weather forecast |
| [8](#8-stage-2--sharpening-the-rainfall-map) | Stage 2 — sharpening the rainfall map |
| [9](#9-stage-3--turning-rain-into-runoff) | Stage 3 — turning rain into runoff |
| [10](#10-stage-4--moving-water-down-the-rivers) | Stage 4 — moving water down the rivers |
| [11](#11-stage-5--correcting-what-is-left) | Stage 5 — correcting what is left |

**Part IV — Judging it**
| § | |
|---|---|
| [12](#12-how-we-measure-success) | How we measure success |
| [13](#13-how-the-testing-works) | How the testing works |
| [14](#14-one-forecast-from-start-to-finish) | One forecast, from start to finish |
| [15](#15-the-results) | The results |
| [16](#16-why-it-is-not-better-than-it-is) | Why it is not better than it is |

**Part V — Everything else**
| § | |
|---|---|
| [17](#17-the-story-of-this-project) | The story of this project |
| [18](#18-mistakes-and-what-they-taught) | Mistakes, and what they taught |
| [19](#19-what-changed-recently) | What changed recently |
| [20](#20-questions-you-may-be-asked) | Questions you may be asked |
| [21](#21-common-misunderstandings) | Common misunderstandings |
| [22](#22-glossary) | Glossary |
| [23](#23-how-to-run-it) | How to run it |

---
---

# PART I — THE PROBLEM

---

## 1. The five-minute version

If you read nothing else, read this.

**The situation.** Hirakud Dam in eastern India holds back a huge reservoir. During
the monsoon, enormous amounts of water pour in from the surrounding hills. The
people running the dam must decide, every day, how much water to let out.

**The difficulty.** That decision has to be made *before* the water arrives. Once
you can see the water coming, it is too late to act calmly. So they need a
forecast: how much water will arrive tomorrow, next week, next month?

**What we built.** A chain of five processing steps that turns a weather forecast
into a prediction of water arriving at the dam. This chain is not our invention —
it was published by researchers (Dong and colleagues, 2025) and we rebuilt it for
this Indian river to see whether it works here.

**What we found — the honest answer.** It works, but not well, and we now know
exactly why.

- We can predict reasonably well **4 to 14 days ahead**.
- Closer than 3 days, a much simpler method does just as well — because the river
  is slow, so tomorrow's flow mostly depends on today's, not on the weather.
- Beyond about 20 days, **nothing works** — not our chain, not anything. You may as
  well use the historical average for that date.

**Why it fails beyond two weeks — and this is the key insight.** We tested the
chain twice: once with the real weather forecast, and once by *cheating* — feeding
it the weather that actually happened.

| what we fed it | how well it did (0 = useless, 1 = perfect) |
|---|---|
| the actual weather (cheating) | **0.85** — very good |
| the real forecast | **0.20** — weak |

**The river model is fine. The weather forecast is the problem.** Beyond a week,
the rainfall forecast is only about 35–40 % related to what actually falls. No
amount of clever processing afterwards can recover information that was never
there.

**Why that is a useful result, not a failure.** A great many projects would report
"we achieved 0.20" and stop. We can say *where the skill is lost and how much is
recoverable* — and that tells a dam operator to trust the system for about a week
and use a range of possibilities beyond that, rather than trusting a single number
that looks authoritative and is not.

---

## 2. The dam, and the decision

### The place

Hirakud Dam sits on the Mahanadi river in Odisha, eastern India. It is one of the
longest earthen dams in the world, built in the 1950s, and it does three jobs at
once: prevent floods, store water for irrigation, and generate electricity.

The land that drains into its reservoir — everything uphill whose rain eventually
flows there — is called the **catchment**.

> **Catchment:** the whole area of land that funnels rain into one place. Imagine
> the reservoir as the drain of an enormous, gently sloping bathtub. The catchment
> is the whole tub.

**This catchment is about 83,400 square kilometres.** That is roughly the size of
Austria, or about two and a half times the size of Kerala. It stretches west into
the neighbouring state of Chhattisgarh and contains hills, forest and flat farmland.

### The dilemma, concretely

Imagine you are the dam operator on a July morning. The weather forecast suggests
heavy rain upstream over the coming week. You must decide today: open the gates
and release water, or hold it?

**If you hold too much water back** and the rain does arrive, the reservoir fills
to capacity. Now you have no choice: you must release an enormous volume all at
once. That sudden surge floods the towns downstream. People lose homes. This has
happened.

**If you release too much water** and the rain never comes, you have thrown away
water that farmers needed for their crops and that the region needed for
electricity. A dry reservoir in September means a difficult year.

### Why you cannot simply wait and see

Here is the crux. **By the time you can measure water arriving at the reservoir,
the decision that mattered was made days ago.**

Water takes days to travel down from the hills. Emptying a reservoir also takes
days. So if you wait until you see the flood, you are already too late to make the
gentle, gradual release that would have been safe.

**You have to know what is coming before it arrives.** That is what this project
tries to provide.

---

## 3. What exactly are we predicting

### The quantity

We predict **reservoir inflow**: how much water enters the reservoir each day.

It is measured in **cubic metres per second**, written **m³/s** and often called
**cumecs**.

> **One cubic metre per second** means a cube of water one metre on each side —
> 1,000 litres — flowing past every second.

To give that human scale:

| flow | what it is like |
|---|---|
| **1 m³/s** | a large domestic bathtub emptying every second |
| **250 m³/s** | a quiet day at Hirakud between storms |
| **2,500 m³/s** | a typical busy monsoon day |
| **20,000 m³/s** | a major flood — about eight Olympic swimming pools **every second** |

Over the 30 days we forecast, the flow might range from a couple of hundred to
over twenty thousand. **That is a hundredfold swing**, and getting it wrong in
either direction has consequences.

### The timing words

Two terms appear constantly. They are simple but must be kept straight.

> **Initialisation date** — the day the forecast is made. Think of it as "today".
> The model is allowed to know everything up to this moment and **nothing** after
> it.

> **Lead day** — how far into the future we are looking. Lead day 1 is tomorrow.
> Lead day 30 is a month from now.

So a single forecast is: *"standing on 4 July, here is what I think each of the
next 30 days will bring."* That is **one forecast containing 30 predictions.**

### When we forecast

Only during the monsoon — June to September — because that is when almost all the
water arrives. Outside those months there is little to predict and little at stake.

### How many forecasts we have to learn from

The weather archive gives us **532 forecast dates** spread across **19 monsoon
seasons (2004 to 2022)**.

Each forecast covers 30 days, so:

```
532 forecasts × 30 days each = 15,960 individual day-predictions
```

When you see "about 16,000 samples" mentioned anywhere in this project, that is
where the number comes from.

**Is 532 a lot?** For this kind of work, it is modest. It is one of the quiet
constraints running through everything that follows — several times we will find
that a method cannot be improved simply because there is not enough data to learn
from.

---

## 4. Why this is hard

Here is a natural thought: *if we knew exactly how much rain would fall, we would
know how much water arrives. So the whole problem is just weather forecasting.*

**That is wrong, and understanding why is the foundation of everything else.**

Suppose we had a magic weather forecast — every millimetre of rain, in exactly the
right place, on exactly the right day, a month ahead. We would **still** not know
the inflow. Four independent reasons, and each one is why a particular stage of
our chain exists.

---

### 4.1 Reason one: the ground remembers

**The kitchen sponge.**

Take a dry sponge and pour a glass of water on it. Almost all of it soaks in.
Barely a drop escapes.

Now take a sponge that has been sitting in a bucket of water all week. Pour the
same glass on it. Almost all of it runs straight off the sides.

**Same sponge. Same water. Completely different result** — because of what
happened *before*.

A river catchment behaves exactly like that sponge. Whether rain soaks into the
ground or runs off into the river depends almost entirely on **how wet the ground
already was**.

> **Antecedent moisture** — the technical term for "how wet the ground already
> was". *Antecedent* simply means "what came before".

**In practice:** 100 mm of rain in early June, on ground dried out by eight months
without rain, may almost entirely soak in and barely raise the river. The identical
100 mm in August, after weeks of monsoon have saturated everything, runs straight
off and causes a flood.

**The measurement.** Here is a real forecast (4 July 2018). The "runoff ratio" is
the fraction of falling rain that reached the river rather than soaking in:

| lead day | rain that fell (mm) | water reaching the river (mm) | **runoff ratio** |
|---|---|---|---|
| 1 | 9.23 | 1.92 | **0.21** — a fifth ran off |
| 3 | 14.17 | 1.90 | **0.13** |
| 7 | 20.18 | 2.46 | **0.12** |
| 14 | 16.78 | 3.36 | **0.20** |
| 21 | 18.36 | 3.93 | **0.21** |
| 30 | 16.81 | 5.37 | **0.32** — a third ran off |

**Look at days 3 and 7.** Day 7 had *more* rain (20.18 mm vs 14.17 mm) but a
*lower* runoff ratio (0.12 vs 0.13). More rain produced proportionally less river
water.

**This is the crucial point: the relationship is not a simple multiplication.**
You cannot say "rainfall × 0.2 = river water". The ratio depends on the
accumulated state of the entire landscape, which is constantly changing.

**What this forces us to do.** Our river model cannot start from scratch each
time. It must begin from a **saved snapshot** of how wet every part of the
catchment was on the day the forecast was issued. That snapshot is the ground's
memory, and stage 3 is built around it.

### How long is that memory?

We measured it — comparing river flow today with river flow N days earlier across
the whole record:

| how long ago | how strongly related (1.0 = identical, 0 = unrelated) |
|---|---|
| yesterday | **0.911** — almost identical |
| 3 days ago | 0.697 |
| a week ago | **0.484** — still clearly related |
| 2 weeks ago | 0.357 |
| a month ago | 0.217 — faint |
| **2 months ago** | **−0.013** — nothing at all |

**The catchment remembers roughly a week clearly, a month faintly, and two months
not at all.**

Hold on to that last row. It becomes surprisingly important in §11, where we
discover that our system had been feeding itself 60 days of history — of which the
last 30 contained no information whatsoever.

---

### 4.2 Reason two: the weather forecast is far too blurry

Weather forecasts are calculated on a grid — the world divided into boxes, with
one number per box.

**The boxes in our forecast are enormous.** At this latitude, each one measures
roughly **167 by 156 kilometres**. That is about **26,000 square kilometres** per
box — larger than several Indian states.

And the forecast gives **one single rainfall number** for that entire box.

**So how many boxes cover our catchment?** The catchment is 83,400 km². Each box
is 26,000 km². So:

```
83,400 ÷ 26,000 ≈ 3 boxes
```

**Three numbers to describe an area the size of Austria.**

Within one of those boxes you might have a thunderstorm drenching the hills while
the valley stays completely dry. The forecast cannot tell you that. It reports one
average and moves on.

### Why averaging is not just imprecise — it is actively wrong

This is subtle and it matters enormously.

Because the rain-to-river relationship is not a simple multiplication (§4.1),
averaging the rain *before* calculating the river response gives you a **different
and wrong answer**.

**A worked illustration.**

*Situation A — rain concentrated.* 100 mm of rain falls, but all of it on one
quarter of the box. That quarter saturates quickly, cannot absorb more, and the
rest pours into the river. **Result: a flood.**

*Situation B — rain spread evenly.* The identical total volume falls as a gentle
25 mm everywhere. At 25 mm, the soil across the whole box quietly absorbs almost
all of it. **Result: almost nothing reaches the river.**

**Same total water. Completely different outcome.**

The forecast reports the same number for both. If we simply take its average and
run our calculation, **we systematically miss floods** — which are precisely the
events the dam operator cares about most.

**What this forces us to do.** We must break the catchment into much finer pieces.
We use **224 pieces** at about 25 km each, and calculate water movement separately
on **150 of them** (the ones actually inside the catchment).

**That is the entire reason stage 2 exists** — to turn 3 blurry numbers into 224
sharper ones.

---

### 4.3 Reason three: water takes days to travel

Rain is measured as a **depth** — millimetres spread over an area, like measuring
rainfall in a bucket.

River flow is measured as a **rate** — volume passing one point each second.

Converting between them needs two things:

**First, area.** A depth spread over a large area is a lot of water; over a small
area, very little. So we multiply by the size of each piece of land.

**Second, and less obviously — time.** Water does not teleport. Rain falling 300
kilometres upstream does not reach the dam that afternoon. It has to trickle into
streams, which join into tributaries, which join into the main river.

**So the water arriving at the dam today is a mixture:** rain that fell nearby
yesterday, plus rain that fell far away several days ago, all arriving together.

**What this forces us to do.** We need a model of how long water takes to travel
from each part of the catchment. That is stage 4, and §10 shows the actual measured
travel times — which turn out to be one of the most useful tables in this document.

---

### 4.4 Reason four: forecasts are uncertain, and honest about it

Weather centres do not issue *one* forecast. They issue **eleven slightly different
ones**.

**Why?** Because we cannot measure today's atmosphere perfectly. So forecasters
take their best estimate, then create ten more versions by nudging the starting
conditions slightly — all within the range of what the measurements could not
distinguish. Then they run all eleven forward.

> **Ensemble members** — the eleven different versions of the same forecast.
> **Ensemble spread** — how much they disagree with each other.

If all eleven agree, the forecast is confident. If they scatter wildly, it is not.

**From our real 4 July 2018 forecast:**

| lead day | average forecast rain | **how much the eleven disagree** |
|---|---|---|
| 1 | 9.23 mm | **2.51** — fairly tight |
| 3 | 14.17 mm | 6.57 |
| 7 | 20.18 mm | 8.96 |
| 14 | 16.78 mm | 9.91 |
| 21 | 18.36 mm | **12.74** — very wide |

Disagreement grows from 2.5 to 12.7 as we look further ahead. **The forecast is
openly telling us it is losing confidence.**

**And that honesty is useful.** Across our whole dataset, when the eleven members
disagree more, our final prediction really is more wrong — the relationship measures
**+0.475**, which is a solid link. The spread is a genuine warning signal, not noise.

---

### Summary of Part I

Even a *perfect* rainfall forecast would not give a perfect inflow forecast,
because:

1. **the ground remembers** — the same rain gives wildly different results
   depending on prior wetness → *we need stage 3 and its saved snapshots*
2. **the forecast is too blurry** — 3 numbers for Austria, and averaging loses
   floods → *we need stage 2*
3. **water takes days to travel** → *we need stage 4*
4. **uncertainty grows with time** → *we need to report ranges, not just numbers*

---
---

# PART II — THE TOOLS

---

## 5. What a "model" is, and what "training" means

Two of our five stages use machine learning. If that phrase means nothing to you,
this section is all you need.

### Two completely different kinds of model

This project uses **both**, and keeping them straight is essential.

#### Kind one: a physics model

A **physics model** encodes rules that scientists worked out from understanding
how the world behaves. Nobody taught it by showing examples. It calculates.

Our stage 3 is a physics model. It knows things like *water flows downhill*,
*saturated soil cannot absorb more*, *warm dry air evaporates water faster*. These
rules came from a century of hydrology research, written down as equations.

**Analogy:** a physics model is like a recipe. Follow the steps with your
ingredients and you get the result. The recipe was written by someone who
understood cooking.

#### Kind two: a machine-learning model

A **machine-learning model** is given thousands of examples of "here is the input,
here is the correct answer" and finds patterns connecting them by itself. Nobody
tells it the rules. It discovers them — or rather, it discovers *something* that
reproduces the answers.

**Analogy:** a machine-learning model is like someone who has never been taught to
cook but has watched ten thousand meals being made. They have no theory, but they
have developed strong instincts about what goes with what.

### What "training" actually is

A machine-learning model contains a large number of adjustable knobs called
**parameters** or **weights**.

Training works like this:

1. Show the model one example. It makes a prediction.
2. Compare the prediction to the right answer. Calculate how wrong it was.
3. Nudge every knob slightly in whichever direction reduces that wrongness.
4. Repeat, thousands upon thousands of times.

That's it. No understanding, no insight — just an enormous number of tiny
corrections, gradually settling into a configuration that produces good answers.

> **Loss function** — the formula that scores how wrong a prediction is. This is
> genuinely important: **you get what you measure.** If the loss says "being
> wrong by a lot is catastrophic, being wrong by a little is fine", the model
> becomes cautious and predicts middling values. We will see exactly this happen
> in §8, and it explains one of our biggest problems.

### The danger: memorising instead of learning

Imagine a student who memorises last year's exam paper. They score brilliantly on
that paper and fail completely on a new one.

Machine-learning models do this readily. Given enough knobs, a model can memorise
its training examples without learning anything general.

> **Overfitting** — when a model memorises its training data instead of learning
> patterns that generalise to new data.

**This is not a minor concern. It is the central risk of the entire field**, and
§13 explains the elaborate precautions this project takes against it — precautions
that an earlier version of this work skipped, producing a result that looked
spectacular and meant nothing.

---

## 6. The five stages, in plain words

Here is the whole chain in one page. Details follow in Part III.

```
   ┌──────────────────────────────────────────────────────────┐
   │  STAGE 1   The weather forecast arrives                   │
   │            From Europe's weather centre.                  │
   │            Blurry: 3 numbers covering our whole area.     │
   └──────────────────────────────────────────────────────────┘
                              ↓
   ┌──────────────────────────────────────────────────────────┐
   │  STAGE 2   Sharpen the rainfall map        (AI)          │
   │            Turn 3 blurry numbers into 224 sharper ones,   │
   │            using knowledge of hills and valleys.          │
   └──────────────────────────────────────────────────────────┘
                              ↓
   ┌──────────────────────────────────────────────────────────┐
   │  STAGE 3   Work out how much water reaches the streams    │
   │            (physics)                                      │
   │            For each of 150 pieces of land, every 6 hours: │
   │            how much soaks in, evaporates, runs off.       │
   │            Starts from a snapshot of how wet it was.      │
   └──────────────────────────────────────────────────────────┘
                              ↓
   ┌──────────────────────────────────────────────────────────┐
   │  STAGE 4   Move the water down the rivers   (physics)    │
   │            Add travel time. Water from far away takes     │
   │            days to arrive. Convert depth → flow rate.     │
   └──────────────────────────────────────────────────────────┘
                              ↓
   ┌──────────────────────────────────────────────────────────┐
   │  STAGE 5   Correct the remaining error      (AI)         │
   │            Learn the mistakes the chain usually makes,    │
   │            and adjust for them.                           │
   └──────────────────────────────────────────────────────────┘
                              ↓
                    Forecast: 30 days of inflow
```

**In one sentence each:**

1. **ECMWF** gives us blurry rainfall predictions for the next 30 days.
2. **The CNN** sharpens those into a detailed map using knowledge of the terrain.
3. **VIC** calculates, piece by piece, how much of that rain reaches the streams.
4. **RVIC** carries that water down the river network with realistic travel times.
5. **The LSTM** corrects the systematic mistakes the chain tends to make.

**Stages 3 and 4 are physics. Stages 2 and 5 are machine learning.** That
combination is the whole idea of the approach we are testing — use physics where
we understand the process, and learning where we do not.

---
---

# PART III — THE STAGES IN DETAIL

---

## 7. Stage 1 — the weather forecast

### Where it comes from

Our starting point is a forecast from the **European Centre for Medium-Range
Weather Forecasts (ECMWF)** — widely regarded as the best weather forecasting
organisation in the world.

We use a specific product called the **S2S reforecast**. Two terms to unpack.

**S2S** stands for **sub-seasonal to seasonal** — roughly two weeks to two months
ahead.

> **Why this range is the hardest in all of weather forecasting.** Short-range
> forecasts (1–3 days) work because today's atmosphere still largely determines
> tomorrow's. Seasonal forecasts (3–6 months) work because slow, steady things
> like ocean temperatures dominate. **Sub-seasonal falls in the gap where neither
> helps.** Today's atmosphere has been forgotten; the slow signals are too weak.

**Reforecast** means the weather centre took its *current* model and re-ran it on
*past* dates — essentially asking "what would today's model have predicted for
July 2007?"

Why does that matter? Because weather models are improved constantly. If we
learned from forecasts made by many different model versions, we would be learning
a moving target. A reforecast gives us **one consistent model across twenty
years.**

### What we have

| | |
|---|---|
| grid box size | **1.5°** — about 167 × 156 km |
| area covered | a 7 × 7 grid of boxes around the catchment |
| how far ahead | 1 to 30 days |
| versions of each forecast | **11** (see §4.4) |
| weather variables | **26** |
| forecasts per monsoon | **28** |
| **total forecasts** | **532** |

### What the 26 variables are, and why so many

Eleven describe conditions **at ground level**: rainfall (two kinds), temperature,
wind (two directions), air pressure, cloud cover, three kinds of energy flow, and
the height of the land.

Fifteen describe conditions **high in the atmosphere** — wind, moisture,
temperature and pressure-height, each at three altitudes (roughly 1.5 km, 5.5 km
and 12 km up).

**Why bother with conditions kilometres above the ground?** Because that is where
weather is *organised*. The project's own notes put it bluntly: *"a surface-only
forecast cannot place a storm in a particular 25 km cell."*

Ground-level pressure tells you a storm is somewhere nearby. The flow of moisture
five kilometres up tells you **where it will actually rain**.

### Why exactly 28 forecasts per monsoon

The reforecast is not produced daily. It runs **Mondays and Thursdays only**,
which gives 31 dates across June–September.

But three of them — 12, 19 and 26 July — fall on Fridays in this particular model
version, and the data server rejects requests for them outright. They simply do
not exist. The project's notes flag this explicitly so nobody wastes time retrying:
*"they are not a download failure to retry."*

31 − 3 = **28 dates**, and 28 × 19 seasons = **532 forecasts**.

---

### Three traps hidden in the data

Raw data from a weather centre is **not ready to use**. Three quirks will quietly
destroy your results if you do not know about them. None of them produce an error
message.

#### Trap 1 — the rainfall numbers keep adding up

When ECMWF reports rainfall, it does **not** give you that day's rain. It gives
the **running total since the forecast began**.

> **Think of a water meter in your house.** It does not show today's usage — it
> shows everything used since installation. To find today's usage, you read it
> today, read it yesterday, and subtract.

The forecast's "rainfall" for day 1 might read 2.4 mm; by day 30 it reads 238 mm.
**If you feed those raw numbers into a model, it concludes that 238 mm of rain
fell on day 30 alone** — a catastrophic downpour that never happened.

The fix is to subtract each day from the next. Simple — *once you know*.

The project's own comment on this is memorable: missing this step is *"the classic
way to get a downscaler that looks brilliant and is useless."* And it is not
hypothetical — it was one of six genuine defects found in an earlier version of
this work (§18).

#### Trap 2 — cloud cover is a percentage, not a fraction

Cloud cover arrives as a number from **0 to 100**. The name and the surrounding
variables suggest it should be 0 to 1. It is not. Treat it as a fraction and every
cloud calculation is a hundred times too large.

#### Trap 3 — two variables that briefly contained no information at all

**This was the most serious defect in our data, and it has been fixed.** The story
is worth keeping because it is the clearest example of a bug that produces no error
message, no warning, and no visible symptom.

**What went wrong.** Temperature and cloud cover came back as a **single value for
day 1**, which was then copied across all 30 days. So the temperature our model saw
for day 30 was identical to day 1. Two of our 26 information channels were
constant — telling the model nothing about how conditions change.

**Why it happened.** These two variables are stored differently from the others in
the ECMWF archive. Our request format matched nine variables correctly and these
two only partially — returning one value instead of thirty. No error was raised.

**How we found and fixed it.** A test script requested them in a different format.
All 30 values came back. The download was corrected and the archive rebuilt.

**How we verified the fix.** We measured how much each variable changes across the
30 days, for every one of the 532 forecasts:

| variable | how much it varies | |
|---|---|---|
| rainfall | 0.368 | varies, as it must |
| **temperature** | **0.295** | varies ✓ |
| **cloud cover** | **0.547** | varies ✓ |
| land elevation | **0.000** | constant — **correct**, hills do not move |

**Land elevation is the control.** It *should* be identical across all 30 days —
the terrain does not change. A broken variable would look exactly like that row.
Neither temperature nor cloud does, in any forecast, in any year.

**And we checked they make physical sense**, which a copied value never could:

- when cloud cover is high, rainfall is high — relationship **+0.63**, holds in
  **99.2 %** of forecasts
- when it rains, temperature drops — relationship **−0.40**, holds in **84.4 %**

A copied value would give exactly zero, every time.

---

## 8. Stage 2 — sharpening the rainfall map

### The job

We have about 3 blurry numbers. We need 224 sharp ones. Stage 2 is a neural
network that does this.

### The intuition

Ask an experienced local forecaster to predict rain in one particular valley. They
do not just look at the humidity directly overhead. They look at the **whole
surrounding situation** — a weather system approaching from the south-west, wind
pushing moist air up against a ridge — and combine it with knowledge of that
specific valley.

**Our network does exactly this.** Rather than predicting the whole map at once,
it predicts **one small area at a time**, using:

1. a patch of the blurry forecast *surrounding* that area, and
2. facts about the area itself — where it is, and how high

Run it 224 times and the detailed map is complete.

### Why we cannot just "zoom in"

The obvious alternative is to stretch the blurry map smoothly onto a finer grid —
what photo editors do when you enlarge an image.

**This cannot work, and the reason is fundamental.**

> **Zooming adds pixels. It does not add information.**
>
> Enlarge a blurry photograph and you get a bigger blurry photograph. You never
> recover detail that was not captured.

Smoothing cannot know that one 25 km area sits on a **windward slope**, where air
is pushed upward, cools, and dumps heavy rain — while the area just behind it sits
in that slope's **rain shadow** and stays dry. Nothing in the blurry number
distinguishes them.

**A trained network can know this**, because it has seen thousands of examples and
has access to two things zooming does not:

- **the shape of the surrounding weather** — not just how much rain, but how it is
  arranged across the region
- **the fixed terrain** — the hills never move, and they powerfully control where
  monsoon rain falls

### What goes in and what comes out

For each prediction — one small area, one forecast date, one lead day:

**Input A:** all 26 weather variables, on a **3 × 3 patch** of blurry boxes centred
on the target area.

**Input B:** three numbers — latitude, longitude, and elevation of that area.

**Output:** one number — rainfall there, in mm per day.

**Why a 3 × 3 patch and not bigger or smaller?** This is not a guess. The original
researchers tested 1×1, 3×3, 5×5 and 7×7, and found 3×3 worked best. Too small and
you miss the weather system's shape; too large and you drown the local signal in
distant irrelevance.

> **A near-miss worth recording.** An earlier version of this project downloaded
> only a 5 × 5 area of blurry boxes. For target areas near the edge, a 3 × 3 patch
> ran off the edge of the available data — and the code silently filled the gap by
> **duplicating the edge row**.
>
> **48 % of training examples were affected.** Nearly half the data contained
> invented values presented as real measurements.
>
> Re-downloading a 7 × 7 area fixed it completely — **0 % affected** — and the code
> now actively checks this rather than assuming it.

### How the network is built

It is a **ResNet** — short for *residual network*.

> **The idea behind a ResNet.** Deep networks process information through many
> layers, and information can get distorted along the way — like a message passed
> down a long line of people. A ResNet adds **shortcuts**: alongside the usual
> processing, the original input is passed forward and added back. So each layer
> only needs to learn *what to change*, not reproduce everything from scratch.

The structure:

| step | what happens |
|---|---|
| input | the 26-variable, 3 × 3 weather patch |
| block 1 | expand to 64 internal features |
| block 2 | compress to 32 |
| block 3 | compress to 16 |
| flatten | lay them out as a list of 144 numbers |
| coordinates | convert lat/long/elevation into 16 numbers |
| join | combine: 144 + 16 = 160 numbers |
| head | narrow 160 → 64 → 32 → **1 final answer** |

**Four design decisions worth understanding:**

**Padding.** The patch is only 3 × 3. Each processing step normally shrinks the
edges — so without adding a border, it would vanish after one step.

**Channel dropout.** During training we randomly switch off entire weather
variables. This forces the network not to become dependent on any single one — if
it has learned to rely entirely on humidity, it fails on the days humidity is
hidden, so it learns to use everything.

**Coordinates handled separately.** Position enters through its own small pathway
rather than being mixed in as another weather map. This lets the network say *"this
particular place behaves unusually"* without confusing its understanding of weather
patterns.

**Guaranteed-positive output.** The final step uses a function that can never
produce a negative number. **Rainfall below zero is meaningless**, and without this
the network would happily predict it.

---

### The loss function, and an important discovery

Recall from §5 that the **loss function** decides what the model optimises for.
Here it is:

```
Loss = (ordinary error) + b × (1 − threat score)        with b = 1.0
```

The **ordinary error** part is straightforward: how far off was the prediction,
squared.

The **threat score** part is designed to solve a specific problem.

> **The problem it addresses.** Most days have little or no rain. If a model
> simply predicts "not much rain" every single day, it is right most of the time
> and scores well on ordinary error. But it would **never predict a flood** — the
> one thing we actually need.
>
> The **threat score** measures specifically how well the model catches *heavy
> rain events*:
>
> ```
> threat score = hits ÷ (hits + false alarms + misses)
> ```
>
> Adding it to the loss is meant to make the safe, boring strategy expensive.

### But we discovered it does not work here

This is one of the project's substantive findings, and it took three experiments.

**Measured on the trained network:**

| part of the loss | value |
|---|---|
| ordinary error | **265.3** |
| threat score part | **1.00** |
| **threat score's share of the total** | **0.375 %** |
| actual threat score achieved | **0.000** |

**The two parts are on wildly different scales.** Ordinary error is measured in
millimetres-squared and reaches into the hundreds. The threat score part can never
exceed 1. So the extreme-rain term contributes **less than half a percent** of what
the model is actually optimising.

**In practice, the network is training on ordinary error alone**, and the heavy-rain
protection is decorative.

**We tried to fix it, twice:**

**Attempt 1 — turn up the volume.** We increased its weight 66-fold, enough to make
it a fifth of the total. **No change whatsoever.**

**Attempt 2 — the term might be "stuck".** Investigating why, we found the
heavy-rain threshold is 24.46 mm, while the network's typical prediction sits about
16 mm below it. The mathematics meant **99.7 % of cases produced no learning signal
at all.** We widened the sensitivity so that 100 % of cases produced a signal.
**Still no change** — slightly worse, if anything.

**Why neither worked.** We finally measured the network's **largest prediction ever
made: about 26 mm.** The heavy-rain threshold is 24.46 mm. Real rainfall reaches
**575 mm**.

**The network essentially cannot produce heavy rain at all.** It has collapsed into
a narrow band of cautious, middling values. No amount of adjusting a threshold-based
term can help a model that never reaches the threshold.

> **And here is the deep reason, which is worth understanding properly.**
>
> When a model's input carries only weak information, and the loss punishes large
> errors severely, **predicting the middle is mathematically the correct
> strategy.**
>
> Consider: predicting 200 mm when you are right only 40 % of the time produces
> enormous errors on the 60 % of occasions you are wrong. Predicting a safe 8 mm
> every time produces small errors always.
>
> **The network is not malfunctioning. It is behaving optimally given what it was
> asked to optimise.** The caution is a rational response to uncertainty.
>
> This is a genuine negative finding: the original paper credits this specialised
> loss function for handling extremes well. **On our river, it does not function.**

---

### What stage 2 achieved — the good news

We measure using **R²** (pronounced "R squared"), where 1.0 is perfect, 0 means no
better than always guessing the average, and **negative means worse than guessing
the average**.

**At the level of individual 25 km areas:**

| | |
|---|---|
| average across 19 years | **+0.057** |
| **years with a positive score** | **19 out of 19** |

Compare that with the raw blurry forecast at those same small areas: **−0.136**.

**Negative. The raw forecast is worse than useless at that scale** — you would do
better ignoring it entirely and guessing the long-run average.

**The sign flipped, in every single year.** That is a real achievement.

### What stage 2 achieved — the bad news

**This is the single most important thing to understand about stage 2.**

We also measured how well the sharpened rainfall predicts the **total rain across
the whole catchment** — which is what stage 3 actually uses:

| how far ahead | sharpened (CNN) | **raw forecast** |
|---|---|---|
| 1–3 days | 0.574 | **0.637** |
| 4–7 days | 0.454 | **0.509** |
| 8–14 days | 0.365 | **0.400** |
| 15–21 days | 0.350 | **0.362** |
| 22–30 days | **0.385** | 0.376 |

**The raw forecast is better at every range but the last.**

**How can it be better at small areas and worse overall?** Because the network is
trained to get *each small area* right — to place rain correctly in space. It is
never asked to get the *total* right. And the total is what flows into the river.

Improving the distribution while slightly degrading the sum is entirely possible,
and that is what happened.

**And it flattens everything.** Here is how much variety the rainfall shows,
compared to reality:

| how far ahead | raw forecast | **sharpened** |
|---|---|---|
| 1–3 days | 0.696 | **0.431** |
| 4–7 days | 0.536 | **0.385** |

Real rainfall swings between dry days and downpours. The raw forecast captures
about 70 % of that swing. **The sharpened version captures only 43 %** — it has
smoothed the peaks and troughs toward the middle.

That is the caution described above, and it is the direct cause of our poor
flood-peak performance later.

---

## 9. Stage 3 — turning rain into runoff

### What VIC is

**VIC** stands for **Variable Infiltration Capacity**. It is a **physics model**
(see §5) — it calculates what happens to water using equations derived from
scientific understanding, not from examples.

For each of **150 pieces of land**, every **6 hours**, VIC works out a complete
water budget. Water arrives as rain, and then:

- some **evaporates** back into the air, or is drawn up by plants and breathed out
  through their leaves
- some **soaks into the soil**, moving down through three layers
- some **runs off the surface immediately** — this is fast, and causes floods
- some **drains slowly from deep soil** into the river — this is called
  **baseflow**, and it is why rivers keep flowing for weeks after the rain stops

### What the name means — and it is the key idea

"Variable Infiltration Capacity" sounds impenetrable, but it describes something
intuitive.

> Within a single piece of land, not every square metre is the same. A dip
> collects water and saturates quickly. A slope sheds it. A sandy patch drinks it
> up.
>
> So as a piece of land gets wetter, **an increasing fraction of it becomes
> saturated** — and saturated ground cannot absorb any more. That growing fraction
> produces runoff.
>
> **This is why the response is not a straight line.** It is the sponge from §4.1,
> expressed as equations.

### The snapshot — how memory enters

**VIC starts each forecast from a saved snapshot** of how much moisture was in
every soil layer of every piece of land at the moment the forecast was issued.

This is how §4.1's memory enters the pipeline. Without it, the model would have to
guess how wet the ground was — and as we saw, that guess determines whether rain
produces a trickle or a flood.

**We generate 532 of these snapshots**, one per forecast date.

### What VIC needs to run

Seven weather inputs per piece of land, every six hours: **rainfall, temperature,
sunlight, heat radiation, air pressure, humidity, and wind speed.**

Rainfall comes from stage 2. Five others come from a high-quality reconstruction of
past weather.

**Two of them — heat radiation and humidity — have no forecast equivalent in our
archive.** For those we use the historical average for that date. That is an honest
limitation, recorded here and in the main documentation.

---

### Calibration — teaching it this particular river

VIC contains numbers that cannot be measured directly: how deep the soil layers
are, how fast water drains, how deep plant roots reach. These are **fitted** —
adjusted until the model reproduces observed river flow.

**How the years are divided** — and this division is the backbone of trustworthy
testing:

| years | role |
|---|---|
| 2003 | **warm-up**, then discarded — the model starts from a guess and needs time to forget it |
| **2004–2011** | **calibration** — the model is tuned on these, using every observed day (2,740 days) |
| 2012–2014 | **validation** — never used during tuning |
| 2015–2022 | **completely held back** |

**The search method** is called **differential evolution**. It keeps a population
of candidate settings, combines the promising ones, discards the poor ones, and
repeats — loosely inspired by natural selection. It ran 432 attempts over 135
minutes.

**What it optimises for** — and this is deliberately two things at once:

1. **matching observed river flow** day by day
2. **getting the water budget physically sensible** — a realistic fraction of
   rainfall should become river water, and a realistic fraction should evaporate

Without the second condition, the search can match the river by doing something
physically absurd elsewhere. §17 describes exactly that happening.

### How well VIC works

| period | accuracy (NSE) | timing (correlation) | **quantity error** |
|---|---|---|---|
| calibration 2004–2011 | 0.581 | 0.931 | **+50.3 %** |
| validation 2012–2014 | 0.328 | 0.914 | **+58.4 %** |
| **held back 2015–2022** | 0.416 | **0.937** | **+72.1 %** |

**Two findings here, pointing in opposite directions.**

**The timing is excellent, and it generalises.** Correlation is *highest on the
years the model never saw* — 0.937. **The model genuinely understands when water
arrives**, even in years absent from its tuning. It is not memorising.

**The quantity is badly wrong, and gets worse over time.** It predicts 50 %, then
58 %, then 72 % too much water.

**That pattern is a clue.** A tuning error would be roughly constant across the
three periods. **An error that grows steadily with time points to something
changing in the real world** that the model does not represent.

---

### The missing piece — people take the water

We investigated, and found this:

| | mm per year | fraction of rain reaching the river |
|---|---|---|
| rainfall | 1,281.9 | — |
| **water actually arriving at the dam** | 366.2 | **0.286** |
| what scientific literature expects naturally | 449–513 | 0.35–0.40 |

**Between 80 and 150 mm per year — that is 18 to 29 % of the river's natural flow —
never arrives at the dam.**

**Where does it go?** The Mahanadi above Hirakud is heavily irrigated. Farmers and
irrigation schemes divert water upstream, before it reaches the reservoir.

**VIC calculates natural river flow. It has no concept of people taking water out.**

This single fact explains everything awkward about stage 3:

- **the growing error** — irrigation has expanded over two decades, so more water
  is diverted each year
- **and something that looked like a modelling disaster.** When we forced the
  calibration to match the reduced observed flow, it responded by making the soil
  **eight metres deep** — trying to make water disappear by evaporating it, because
  evaporation was the only removal mechanism available. The soil depth hit every
  limit we set: 1.4 m of a 1.5 m limit, then 3.0 of 3.0, then 4.0 of 4.0.

> **Do not "fix" VIC predicting too much water by allowing deeper soil. It is not
> a soil problem. It is irrigation.**
>
> That warning is now written directly into the calibration code, because the
> mistake was made three times before we understood it.

**Where the water budget stands now:** 44.5 % of rainfall becomes river flow
(target 35–40 %), 55.8 % evaporates (target 55–65 %, so this is correct), and the
budget balances to within **0.3 %** — meaning no water is being created or
destroyed. The model is internally consistent even where it disagrees with the
gauge.

---

## 10. Stage 4 — moving water down the rivers

VIC tells us how much water reached the streams **in each piece of land, measured
as a depth in millimetres**. We need **flow at the dam, in cubic metres per
second**. Stage 4 does two conversions.

### Conversion one: depth into volume

Multiply the depth by the area of that piece of land, and by the fraction of it
actually inside our catchment. Then divide by the number of seconds in a day.

**This is the only place in the entire pipeline where the physical size of the
catchment enters the numbers.**

### Conversion two: now into later

Water from a distant part of the catchment arrives **spread over several days**,
not all at once.

Stage 4 handles this with something called a **unit hydrograph**.

> **A unit hydrograph** is a simple curve that answers: *"if 100 units of water
> enter the streams from this piece of land today, how much arrives at the dam
> tomorrow? The day after? Three days later?"*

**Here is the actual measured answer for this catchment**, added up across all 150
pieces of land:

| days after the water enters the streams | share arriving |
|---|---|
| 1 day | 7.1 % |
| 2 days | 15.6 % |
| 3 days | 20.2 % |
| **4 days** | **29.2 %** ← the peak |
| 5 days | 21.4 % |
| 6 days | 6.1 % |
| 7 days | 0.4 % |
| 8–10 days | 0.0 % |

**This is one of the most useful tables in this document. Read it carefully.**

**The typical parcel of water takes four days to reach the dam**, and virtually
everything has arrived within six.

**What that means practically.** Rain falling today mostly affects the reservoir
three to five days from now. The catchment has a natural response time built into
its geography — and that is exactly why our forecast performs best in the 4-to-14
day range (§15). **The physical structure of the river and the useful range of the
forecast are the same thing seen from two directions.**

The travel times come from the actual shape of the land — analysing which way water
flows downhill across the terrain — not from fitting numbers to data.

### One safeguard worth understanding

Because water takes up to 10 days to arrive, the flow on **forecast day 1** partly
comes from rain that fell **before the forecast was issued**.

We take that earlier water from the historical record. **This is not cheating** —
it is water that was already flowing in the rivers on the day the forecast was
made, which any real operator would know about.

**We verified this carefully**: the historical portion uses exactly the 10 days
*ending on* the forecast date. Forecast day 1 is the day *after*. **Nothing from
the future is used.**

Without this, forecast day 1 would start from completely empty rivers and invent a
false surge as they filled.

---

## 11. Stage 5 — correcting what is left

### What an LSTM is

An **LSTM** (Long Short-Term Memory network) is a neural network designed for
sequences — data where order matters, like a river's flow over time.

> **The idea.** An ordinary network looks at each input independently. An LSTM
> carries a running "memory" forward, deciding at each step what to remember and
> what to forget. That makes it suited to rivers, where what happened last week
> genuinely affects today.

### What it does here — and what it does not

**It does not make a forecast. It corrects one.**

The original researchers describe their version as *"a post-processing model"* —
it takes what the physics produced and adjusts it based on the mistakes the physics
usually makes.

**Inputs:** the recent observed river flow up to the forecast date, the physics
chain's prediction, and the forecast rainfall.

**Output:** a corrected 30-day prediction.

---

### The history-length discovery

How many days of past river flow should it read?

The original paper calls this an *"optimised hyperparameter"* — meaning **it should
be tuned for your particular river** — and notes it should be **large for
snow-fed rivers**, where snow accumulating over months affects flow.

**Our project had it fixed at 60 days.** That value came from a snow-fed setting.
**The Mahanadi is monsoon-fed. It has no snow.**

**Two things were wrong with 60 days:**

**It contained no information.** Recall §4.1: river flow 60 days ago has a
relationship of **−0.013** with today. That is nothing. **Days 30 to 60 were pure
noise being fed to the network as though it were signal.**

**It threw away most of our data.** To use a forecast, we need that many days of
*continuous* prior record. Requiring 60 days left only **311 of our 532 forecasts**
usable.

**And the discarded ones were not random.** From 2015 our records begin in June — so
the forecasts we lost were disproportionately from **early monsoon**. That is the
onset of the rainy season: exactly when a dam operator most needs to know what is
coming.

**We tested every option:**

| days of history | accuracy | forecasts usable |
|---|---|---|
| 7 | 0.192 | 495 (93 %) |
| **15** | **0.200** | **476 (89 %)** |
| 30 | 0.146 | 432 (81 %) |
| 45 | 0.177 | 376 (71 %) |
| **60 (the old setting)** | **0.140** | **311 (58 %)** |

**Changing 60 to 15 improved the entire chain from 0.140 to 0.200** — a gain of
more than 40 %, from changing one number.

**Two effects, both helping:** 53 % more training data, and the removal of 30 days
of noise.

> **An honest caution.** 7 days and 15 days are statistically indistinguishable
> (the difference could easily be chance). **The solid conclusion is "short
> history, around 7 to 15 days" — not "15 is optimal".** Claiming more precision
> than the data supports is how findings fail to replicate.

### What the LSTM actually is

**It is a quantity-correction layer, not a forecasting layer.** Three separate
tests agree:

**Test 1 — its benefit tracks how wrong the physics was.** On the chain with large
quantity errors, it improves results in **18 of 19 years**. Where an earlier stage
already fixed the quantity, it helps far less.

**Test 2 — remove its job and it becomes harmful.** We corrected the quantity error
*before* the LSTM, so it would only need to learn the subtler remaining patterns.
**Results got worse** (helping in only 7 of 19 years). There is no subtle pattern
left for it to learn at this amount of data.

**Test 3 — a single multiplication beats it** on the typical year.

**But it earns its place in a way the typical year hides.** Looking year by year,
it wins 10 of 19 with a tiny typical improvement — yet a **large average
improvement**, because it **rescues catastrophic years**: +0.38 in 2017, +0.30 in
2009, +0.20 in 2015–16, while costing at most −0.06 in good years.

**That is exactly what a safety mechanism should do** — and measuring only typical
performance makes it invisible. It also improves the **worst year** on every chain,
which for flood safety matters more than the average.

---
---

# PART IV — JUDGING IT

---

## 12. How we measure success

Before any results, you need to be able to read the scoreboard.

### NSE — the main score

**NSE** stands for Nash-Sutcliffe Efficiency. It is the standard measure in
hydrology.

| NSE | meaning |
|---|---|
| **1.0** | perfect — every prediction exactly right |
| **0.5** | decent |
| **0.2** | weak but real |
| **0.0** | **exactly as good as always predicting the long-run average** |
| **negative** | **worse than always predicting the average** |

> **The most important thing to understand about NSE: zero does not mean "half
> right". Zero means the model contributed nothing.**
>
> A model scoring 0.0 could be replaced by someone who ignores all data and says
> "about 2,000 m³/s" every single day, with no loss.

This matters because several numbers in this project sit near zero, and one of our
comparison methods scores about **−1.0** — considerably worse than useless.

### Bias — are we too high or too low?

The average prediction compared to the average reality, as a percentage. **+50 %
means predicting half again as much water as actually arrives.**

This turns out to be **the single most important number in the entire project.**

### Peak ratio — do we catch floods?

The predicted flood peak divided by the actual flood peak.

**1.0** is perfect. **Below 1.0 means under-predicting floods** — which for a dam
operator is the dangerous direction, because it means being caught unprepared.

### Typical, average, and how often — report all three

This sounds like statistical pedantry. It is not, and it changed one of our
conclusions.

Consider two models:

- **Model A** is decent most years but occasionally catastrophic.
- **Model B** is unremarkable most years but never disastrous.

Measure only the **typical** year and A looks better. Measure only the **average**
and B looks better. **Both measurements are true and both are incomplete.**

**We report typical, average, and how often each model wins** — because our
correction layer (§11) is exactly Model B, and looking only at typical performance
made its value invisible.

---

## 13. How the testing works

This section is about honesty. It is easy to produce impressive-looking forecast
results by accident.

### Leave-one-year-out

**The principle:** to know how a model performs on a year it has never seen, train
it on all the *other* years and test it on that one. Then repeat, holding out each
year in turn.

We do **19 separate runs**, each holding back a different monsoon. Within each run:
17 years for learning, 1 year for checking progress, 1 year held back for the final
score.

**Everything is recalculated inside each run** — every average, every threshold,
every adjustment. **Nothing from the test year is allowed to influence anything.**

### Why we never split randomly — and the cautionary tale

This is the most important methodological point in the project.

**The problem.** Several different forecasts predict the same day. A forecast made
on 1 July looking 10 days ahead, and one made on 5 July looking 6 days ahead, both
describe 11 July.

**If we split our data randomly**, one of those lands in training and the other in
testing. **The model sees the answer for 11 July during training, then is tested on
11 July.** It scores brilliantly on something it was shown.

**This is not hypothetical.** An earlier version of this project did exactly that
and reported **R² = 0.873** — an excellent-looking result.

**The honest figure, after fixing this and five other defects, is about 0.078.**

The difference between 0.873 and 0.078 is the difference between a model that
appears to work and one that barely does. **Nothing about the code looked wrong.**

### Judging everyone on the same forecasts

Different methods need different amounts of history, so they can produce different
numbers of forecasts. Comparing them on different sets of days would be meaningless
— one might have been given easier days.

**So every method is judged on the 309 forecasts where every method can produce an
answer and we have complete observations.** Same days, same target, same score.

### Using the ensemble average

Recall the eleven forecast versions (§4.4). You can either score all eleven
separately and average the scores, or average the eleven forecasts and score that.

**These give different answers, and the first is flattering.** We use the average
forecast, because that is what a real operator would use.

**Confusing these two produced wrong conclusions twice in this project's history**,
which is why one piece of code now defines the rule once for everybody.

### What we compare against

**Two deliberately tough comparisons.**

**Persistence** — "tomorrow will be like today", held flat for 30 days. This is the
floor. It scores **−1.023**, confirming that doing nothing clever really is bad.

**The scalar baseline** — take our physics chain's own answer and multiply it by a
single fitted number. That is it. One number.

> **This is not a naive comparison. It is our own physics chain with its quantity
> error removed by one multiplication.** It scores **0.168**, and beating it is
> the real test of whether all our sophistication earns its place.

---

## 14. One forecast, from start to finish

Everything above, applied to one real forecast: **Wednesday 4 July 2018**. You can
reproduce this exactly:

```bash
python tools/trace_one_forecast.py 2018-07-04
```

### Stage 1 — what the weather centre predicted

| lead day | rain (mm/day) | disagreement among the 11 versions |
|---|---|---|
| 1 | 9.23 | 2.51 |
| 7 | 20.18 | 8.96 |
| 14 | 16.78 | 9.91 |
| 21 | 18.36 | 12.74 |
| 30 | 16.81 | 7.04 |

### Stage 2 — after sharpening, against what actually fell

| lead day | our sharpened forecast | **what actually fell** |
|---|---|---|
| 1 | 11.40 | **2.15** |
| 3 | 10.89 | **5.84** |
| 7 | 12.94 | **4.63** |
| 14 | 14.23 | **10.19** |
| 21 | 13.42 | **3.53** |
| 30 | 14.64 | **3.09** |

**Two things to notice.**

**The forecast was far too wet** — often three or four times the actual rainfall.

**And look how flat our column is.** It stays between 10.9 and 14.6 for the entire
month, while reality swings between 2.2 and 10.2. **That is the over-cautious
behaviour from §8, visible in a single case.** Reality varies; our forecast barely
does.

### Stages 3 and 4 — the resulting river flow

| lead day | date | **our prediction (m³/s)** | **what actually arrived** |
|---|---|---|---|
| 1 | 5 Jul | 1,664 | **280** |
| 3 | 7 Jul | 1,503 | **226** |
| 7 | 11 Jul | 2,087 | **292** |
| 14 | 18 Jul | 3,201 | **1,578** |
| 21 | 25 Jul | 3,583 | **3,046** |
| 30 | 3 Aug | 4,646 | **1,224** |

Over the whole 30 days:

| | average flow | peak flow | peak on day | score |
|---|---|---|---|---|
| **what actually happened** | 1,638 | 4,746 | **day 19** | — |
| **our forecast** | 3,048 | 4,646 | **day 30** | **−0.812** |

### What this single case teaches

**The quantity error is enormous, and worst at the start.** On day 1 we predicted
**1,664** against an actual **280** — nearly **six times too much**. That is the
systematic over-prediction from §9, and it is why stage 5 and the scalar baseline
exist at all.

**We got the flood's size nearly right, and its date badly wrong.** Our peak of
4,646 is within 2 % of the actual 4,746. But we placed it on **day 30** instead of
**day 19** — eleven days late.

> **Now a warning that matters more than the example itself.**
>
> Looking at this case, you would naturally conclude *"our problem is timing"*.
>
> **An earlier version of this documentation concluded exactly that, and it was
> wrong.** §16 shows the measurement that overturns it: across all 309 forecasts,
> when we remove the rainfall forecast's own errors, our average timing error is
> **0.28 days** — essentially perfect.
>
> **What this case actually shows is what a bad *rainfall forecast* looks like
> after it passes through a good river model.** The weather centre put the rain in
> the wrong week, and our chain faithfully converted wrongly-timed rain into
> wrongly-timed water.
>
> **One example can show you a mechanism. It cannot tell you how common that
> mechanism is.** That distinction is the difference between a diagnosis and a
> guess.

---

## 15. The results

Judged on 309 forecasts, 19 monsoons, using the ensemble average:

| method | typical (median) | average (mean) | quantity error | flood peaks |
|---|---|---|---|---|
| persistence ("tomorrow = today") | −1.023 | −1.241 | +21.1 % | 0.385 |
| **the scalar baseline** | **0.168** | 0.069 | −2.9 % | 0.794 |
| physics chain, raw rainfall | −0.010 | −0.433 | +31.8 % | **1.081** |
| physics chain, sharpened rainfall | 0.161 | −0.075 | +0.7 % | 0.833 |
| physics chain + correction | 0.118 | 0.070 | −11.1 % | 0.744 |
| **full chain (sharpened + correction)** | **0.200** | 0.040 | −16.0 % | 0.666 |

**The full chain — the complete system from the original paper — is the best of the
physics configurations**, and it beats the scalar baseline's 0.168.

### But "beats" needs qualifying

| method | how often it beat the baseline |
|---|---|
| full chain | **9 years out of 19** |

**Nine out of nineteen is a coin flip.** Statistically, we cannot claim the full
chain is genuinely better — only that it is **no longer genuinely worse**, and
scores highest.

> **The honest sentence is:** *"the full chain is the highest-scoring physics
> configuration and is no longer significantly worse than a single
> multiplication."*
>
> **Not:** *"the full chain beats the baseline."*
>
> That distinction is the difference between a defensible claim and one that falls
> apart under questioning.

### Performance by how far ahead — the most useful table here

| how far ahead | baseline | full chain | what is really going on |
|---|---|---|---|
| **1–3 days** | 0.699 | 0.587 | **the river's own momentum** — "tomorrow = today" alone scores 0.507 |
| **4–7 days** | 0.049 | **0.275** | **this is where the pipeline genuinely earns its place** |
| 8–14 days | 0.107 | 0.100 | marginal |
| 15–21 days | 0.102 | 0.068 | marginal |
| **22–30 days** | −0.009 | −0.010 | **nothing works** — the historical average scores 0.042 |

**Read this before quoting any single headline number.**

**Days 1–3 look impressive but are not forecasting.** A method using *no weather
information at all* scores 0.507 there. That is the river's own slowness, not our
skill.

**Days 4–7 are the real achievement** — 0.275 against the baseline's 0.049. Here
the weather forecast genuinely adds knowledge, and our chain extracts it.

**Beyond 20 days everything is at zero**, and a simple "what usually happens on
this date" average does better.

> **So the honest description of this system is:**
>
> *"A forecast useful from about day 4 to day 14. Inside three days, the river's
> own momentum tells you more. Beyond three weeks, use the historical average."*
>
> **That is far more useful to a dam operator than "a 30-day forecast" — because
> it tells them when to trust it.**

Notice too that **days 4–14 is precisely the travel-time window from §10**, where
water takes 4 days to arrive and everything is through within 6. **The forecast is
useful exactly where the catchment's physical response lives.**

---

## 16. Why it is not better than it is

### The experiment that explains everything

We ran the identical chain twice. Once normally. Once **cheating** — feeding it the
weather that actually happened instead of the forecast.

That second run reveals the **ceiling**: the best any forecast could possibly do.

| what we fed it | score |
|---|---|
| actual weather, as-is | **0.394** |
| **actual weather, with the quantity corrected** | **0.850** |
| actual weather, with the timing perfectly corrected | 0.471 |
| *the real forecast (what we actually achieve)* | **0.200** |

**Four numbers containing the entire story. Take them one at a time.**

**Finding 1: the river model is not the problem.** Given the right weather and the
right quantity, our physics reaches **0.850** — genuinely good. The hydrology
works.

**Finding 2: the error is quantity, not timing.** Correcting the timing perfectly
takes us from 0.394 to 0.471 — a small gain. **Correcting the quantity with a single
multiplication takes us from 0.394 to 0.850.** Across all 309 forecasts, the average
timing error is **0.28 days**.

> ### ⚠️ This reverses the old conclusion
>
> An earlier version of this documentation stated that our error was *"dominated
> by timing, which no downstream model can fix."*
>
> **That was measured incorrectly.** It was measured on the chain *using the
> weather forecast* — where the timing error belongs to **the weather centre**, not
> to our river model. Two separate errors were measured together and both blamed on
> our model.
>
> **The lesson, which applies far beyond this project: to measure how good a
> component is, give it perfect inputs.** Otherwise you measure the whole chain and
> attribute the result to one part of it.

**Finding 3: the gap from 0.850 down to 0.200 is entirely the weather forecast.**
And nothing downstream can recover it.

### Why the weather forecast limits everything

How closely does forecast rainfall match what actually falls?

| how far ahead | relationship (1.0 = perfect) |
|---|---|
| 1–3 days | 0.637 |
| 4–7 days | 0.509 |
| 8–14 days | 0.400 |
| 15–21 days | 0.362 |
| 22–30 days | 0.376 |

**Beyond a week, forecast rainfall is only about 35–40 % related to reality.**

This is not a failing of ECMWF. **It is a property of the atmosphere.** The
practical limit of weather predictability is roughly one to two weeks, and it
exists because tiny unmeasurable differences today grow into completely different
weather a fortnight later.

> **No processing afterwards can create information that was never in the input.**
>
> A better river model, a larger neural network, a cleverer correction layer — none
> of them change that 0.35. This is the ceiling, and we are near it.

### And the sharpening did not help

Comparing the raw and sharpened chains **after removing the quantity error from
both** — so neither is flattered by a lucky cancellation:

| | score |
|---|---|
| raw forecast | **0.168** |
| sharpened forecast | **0.157** |

**The sharpened version is slightly worse.**

**Then why does it look better in the results table** (0.161 versus −0.010)?

**Because two errors cancelled out.** The sharpening makes rainfall slightly too
*low*. VIC makes river flow substantially too *high*. One error partly offsets the
other, and the combination looks good.

**That is luck, not skill.** Remove the quantity error from both and the advantage
vanishes.

> **A general lesson worth carrying:** before crediting any part of a system,
> remove the systematic errors. **Two errors of opposite sign can look like
> success.**

---
---

# PART V — EVERYTHING ELSE

---

## 17. The story of this project

Understanding *how* the conclusions were reached makes them easier to trust — and
easier to defend.

**It began with a result that was too good.** An earlier version reported R² 0.873
for rainfall sharpening. Investigation found **six separate defects, every one
inflating the score** (§18). The honest figure was 0.078.

**Then the system was rebuilt properly** — 19 years instead of 11, 30 days instead
of 17, with rigorous testing.

**The first surprise: better rainfall gave worse river forecasts.** Sharpening
improved rainfall accuracy but the resulting inflow was worse. This made no sense
and drove much of what followed.

**A second look at the correction layer.** It appeared to add nothing. Then we
found the date-scrambling bug (§18), fixed it, re-ran — and the conclusion
reversed. It does help, where there is quantity error to remove.

**The river model was recalibrated three times.** Each time the soil depth hit
whatever limit we set. We kept raising the limit. Only on the third occasion did we
ask *why* — and found irrigation (§9). **The model was trying to represent human
water use by growing implausibly deep soil.**

**Then the decisive experiment.** Instead of trying to improve the chain, we fed it
perfect weather to find the ceiling. That single test (§16) reversed the project's
central diagnosis — from "timing" to "quantity" — and explained every earlier
confusion at once.

**Two more findings followed quickly.** The history length was wrong for this
river, worth +0.06 (§11). And the specialised loss function does not function here
at all (§8).

**The lesson running through all of it:** the most valuable measurements were the
ones that tested our *assumptions* rather than our models. The perfect-weather
experiment took an afternoon and was worth more than months of tuning.

---

## 18. Mistakes, and what they taught

> **The dangerous bugs are not the ones that crash. They are the ones that make
> results look better.** A crash gets fixed in an hour. A silent error gets
> published.

### The six inherited defects

| what was wrong | why it inflated the score |
|---|---|
| rainfall totals never un-accumulated | running totals treated as daily rain |
| sea cells filled with zeros | 24 % of the target was a constant the model "predicted" perfectly |
| random data splitting | the same day's answer on both sides of the test |
| statistics calculated before splitting | test-set information leaked into training |
| target secretly coarsened | **no sharpening was happening at all** |
| rainfall history indexed wrongly | "yesterday's rain" was sometimes 16 days in the **future** |

**Honest score after fixing all six: 0.078, down from 0.873.**

**The fifth is the one to remember.** The model was predicting blurry rainfall from
blurry rainfall and reporting it as sharpening. Not subtle — **invisible, because
nobody checked what the target actually contained.**

### The scrambled dates

The worst bug found during this project, and the last.

One script sorted its data into order but wrote the **date labels from the
unsorted version**. **Every row carried a different row's date.**

**It hid because two of the three labels were accidentally correct** — the "days
ahead" number repeats 1–30 regardless of order, and the year came from a correct
source. Only the actual date was wrong, and nothing ever printed it.

| check | before | after |
|---|---|---|
| rows with correct dates | 300 / 15,960 | **15,960 / 15,960** |
| forecasts paired with their own rainfall | 10 / 532 | **532 / 532** |
| day-1 forecast vs actual rainfall | **−0.02** | **+0.68** |

**A day-1 rainfall forecast matching reality at −0.02 is not a forecast at all.**
That single number is what the scrambled labels were costing.

**Lesson:** when you reorder data, reorder its labels in the same operation — then
verify with a **physical check** that labels could not pass by accident.

### The compounding calibration

Our calibration read the *currently active* settings as its starting point. But
activating a calibration overwrites those settings — so each run started from the
previous winner. Two settings were **multipliers**, so they stacked silently: a run
reporting 0.88 actually had an effective value of **0.13**.

**Lesson:** a multiplier must always be applied to a fixed reference, never to the
previous result. Now enforced by a guard that refuses to start from a previous
output.

### Invented zeros in the target

Days when the gauge did not report are stored as **0.0**, with a separate flag
marking them invalid. Our calibration ignored that flag. **213 days were being
treated as "the river carried no water", concentrated in the dry season.**

The model was being taught to produce no water on days nobody measured.

**Lesson:** one project, one definition of "observed" — and honour the validity flag
everywhere.

---

## 19. What changed recently

If you are holding an older printout, these claims have been **overturned by
measurement**:

| older claim | current position |
|---|---|
| *"the error is mostly about timing"* | **Wrong.** Timing error is 0.28 days. **The error is quantity** |
| *"sharpening improves the forecast"* | **Wrong.** At equal quantity it is slightly worse; its apparent gain was two errors cancelling |
| *"the correction layer adds nothing"* | **Wrong.** It corrects quantity, and helps in proportion to how much is wrong |
| history length = 60 days | **15 days** — improved the chain from 0.140 to 0.200 |
| *"the specialised loss handles extremes"* | **It does not function here** — 0.375 % of the loss, and unreachable |
| headline score ≈ 0.14 | **0.200** |

---

## 20. Questions you may be asked

**"A score of 0.2 sounds poor. Is this a failure?"**

For 30-day river forecasting, 0.2 is modest but real. More importantly, the number
is not the contribution. We can show that with perfect weather the same chain
reaches 0.85 — so the river model is sound — and that forecast rainfall is only
35–40 % accurate beyond a week. **Knowing where the skill is lost is more useful
than the score**, because it tells an operator exactly how far to trust the system.

**"Why doesn't it beat a single multiplication?"**

Because the dominant correctable error *is* quantity, and one multiplication
corrects quantity. That baseline is not naive — it is our own chain with its bias
removed. **The finding is that on this river, quantity was most of what there was
to fix.**

**"Why use VIC rather than the model in the original paper?"**

The original used a simpler model treating the catchment as one unit. VIC divides
it into 150 pieces. §4.2 shows why that matters: averaging rainfall across 26,000
km² destroys flood peaks. The cost is a far heavier setup.

**"Is the sharpening stage worth keeping?"**

At the level of individual 25 km areas, clearly yes — it turns a worse-than-useless
−0.136 into +0.057, in all 19 years. At catchment scale, on current evidence, no.
**Both are true because they measure different things**, and the chain uses the
catchment total.

**"Isn't predicting 72 % too much water disqualifying?"**

Part of that is not model error. **18–29 % of the river's natural flow is diverted
for irrigation** before reaching the dam, and VIC has no way to represent that. The
error grows steadily across calibration, validation and held-back periods — which
is what expanding irrigation looks like, and not what a tuning error looks like.

**"Could you just use more data?"**

Not usefully. Dry-season measurements stop in 2014, and eight years of daily data
already over-determines eight settings. **The constraint was never the number of
years — it was which days were being scored**, which is now fixed.

**"What would actually improve this?"**

Better rainfall forecasts, or explicitly modelling irrigation. **Further work on the
river model or the correction layer will not move these numbers**, and §16 is the
evidence.

---

## 21. Common misunderstandings

**"The model is only 20 % accurate."**

NSE is not a percentage. 0.20 means the model explains 20 % more of the variation
than always guessing the average would. It is a comparison against a baseline, not
a success rate.

**"If the rainfall forecast improved, everything would scale up proportionally."**

Not proportionally — but yes, directionally. The ceiling with perfect rainfall is
0.85, so there is a great deal of headroom. That headroom belongs to meteorology,
not hydrology.

**"The AI parts are the sophisticated bits."**

The opposite, here. The physics stages do the heavy lifting and reach 0.85 given
good inputs. **The two machine-learning stages contributed less than expected** —
one is slightly harmful at catchment scale, the other is essentially a
multiplication.

**"A negative result means the project failed."**

The project answered its question: *can these published methods forecast inflow
here, and if not, why?* The answer is "partially, and here is precisely why." **A
well-established negative result with a diagnosis is a genuine contribution** — and
considerably more useful than an unexplained positive one.

**"The 4 July 2018 example shows the model has a timing problem."**

That example shows what a *rainfall forecast* timing error looks like after passing
through the chain. The model's own timing error is 0.28 days. **One case shows a
mechanism; it cannot tell you how common it is.**

---

## 22. Glossary

| term | meaning |
|---|---|
| **antecedent moisture** | how wet the ground already was before rain fell |
| **baseflow** | slow drainage from deep soil keeping rivers flowing between storms |
| **bias** | systematic over- or under-prediction |
| **catchment** | the land area draining to one point — here 83,400 km² |
| **cumec** | cubic metre per second (m³/s), the unit of river flow |
| **differential evolution** | a search method that evolves a population of candidate settings |
| **discharge** | volume of water passing a point per second |
| **downscaling** | turning a coarse map into a detailed one |
| **ensemble member** | one of 11 slightly different forecast runs |
| **ensemble spread** | how much those versions disagree — a confidence measure |
| **inflow** | water entering the reservoir |
| **initialisation date** | the day a forecast is made |
| **lead day** | how far ahead a forecast looks |
| **LOYO** | leave-one-year-out testing |
| **loss function** | the formula scoring how wrong a prediction is during training |
| **LSTM** | a neural network that carries memory across time steps |
| **NSE** | the main accuracy score: 1 perfect, 0 equals guessing the average |
| **overfitting** | memorising training data instead of learning general patterns |
| **parameter / weight** | an adjustable number inside a model |
| **peak ratio** | predicted flood peak ÷ actual; below 1 means under-predicting floods |
| **persistence** | the naive forecast "tomorrow will be like today" |
| **reforecast / hindcast** | today's weather model re-run on past dates |
| **ResNet** | a network with shortcut connections, allowing greater depth |
| **routing** | moving water through the river network with realistic travel times |
| **runoff** | rain that reaches streams rather than soaking in or evaporating |
| **runoff ratio** | the fraction of rainfall that becomes river water |
| **S2S** | sub-seasonal to seasonal — roughly 2 weeks to 2 months ahead |
| **threat score** | a measure of catching heavy-rain events specifically |
| **training** | adjusting a model's internal numbers to fit examples |
| **unit hydrograph** | the curve describing how runoff spreads over arrival days |
| **VIC** | Variable Infiltration Capacity — our physics-based river model |

---

## 23. How to run it

Use `/usr/local/bin/python3`. Check what data is present first:

```bash
python hydrology/vic/check_inputs.py
```

```bash
# ---- Stage 2: rainfall sharpening  (needs a GPU; see tools/kaggle_run.py)
python cnn/loyo_percell_v2.py            # 19-fold leave-one-year-out
python cnn/quantile_mapping.py           # the comparison benchmarks

# ---- Stage 3: the river model  (calibrate once, about 2 hours)
python hydrology/vic/calibrate_vic.py --workers 4 --maxiter 8 --popsize 6
hydrology/vic/VIC/vic/drivers/classic/vic_classic.exe \
    -g data/processed/vic/global_param.txt        # run 2003-2022
python hydrology/vic/water_balance.py             # check the water budget
python hydrology/vic/route_and_evaluate.py        # score the three periods

# ---- Stages 3 and 4: forecasting
python hydrology/vic/vic_states.py                # 532 snapshots, ~3 min
for p in ec ec_qm ec_cnn; do
    python hydrology/vic/vic_forecast.py --product $p --workers 4   # ~20 min each
    python hydrology/vic/route_forecast.py --product $p
done

# ---- Stage 5: correction
python inflow/lstm_postproc.py --product ec
python inflow/lstm_postproc.py --product ec_cnn

# ---- Results
python inflow/head_to_head.py            # the comparison table
python inflow/fair_comparison.py         # significance testing

# ---- Watch one forecast travel through every stage
python tools/trace_one_forecast.py 2018-07-04
```

---

## Where to go next

| document | purpose |
|---|---|
| `docs/PROJECT_DOCUMENTATION.md` | the whole project, both pipelines, concise |
| `docs/FINDINGS.md` | research log — discoveries in order, including what was wrong |
| `STRUCTURE.md` | which file belongs to which stage |
| `results/metrics/head_to_head.json` | the numbers themselves |
