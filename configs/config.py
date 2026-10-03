"""Central configuration for the Hirakud sub-seasonal inflow forecasting pipeline.

Every hard number that the papers or the data forced on us lives here, with the
reason next to it. Nothing downstream should hard-code these.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

# ---------------------------------------------------------------- source files
SOURCE = RAW / "source"
ECMWF_CSV = SOURCE / "ecmwf_s2s_reforecast_final.csv"
INFLOW_XLSX = SOURCE / "Vishal_data.xlsx"
MASTER_PARQUET = SOURCE / "master_df.parquet"  # only used to recover raw obs_rain
IMD_NC_DIR = RAW / "imd"  # *.nc at 0.25 deg -- required for true downscaling

# ---------------------------------------------------------------- time bounds
# Hirakud inflow (Vishal_data.xlsx) runs 2003-01-01..2014-12-31 and the ECMWF
# reforecast starts 2004-06-03.  The intersection is the only period where the
# full chain (forcing -> rainfall -> inflow) can be trained or scored.
YEAR_MIN = 2004
YEAR_MAX = 2014

# ECMWF S2S reforecast we hold only reaches lead day 17, not the 30 d used in
# Dong et al. (2025).  The project targets a 17 day horizon.
LEAD_MIN = 1
LEAD_MAX = 17

# Initialisations exist only for the monsoon (Jun-Sep), 31 per year.  This is
# the project's season: every forecast is issued during the monsoon.
INIT_MONTHS = (6, 7, 8, 9)

# A late-September initialisation reaches into October at long leads.  Those
# valid times are kept: dropping them would truncate the September forecasts to
# a few days and bias the lead-time distribution, and October inflow still
# matters at Hirakud (monthly mean 1,072 cumecs, the fourth wettest month).
# Set False to score strictly inside JJAS.
ALLOW_POST_MONSOON_VALID = True
TARGET_MONTHS = (6, 7, 8, 9, 10) if ALLOW_POST_MONSOON_VALID else (6, 7, 8, 9)

# ------------------------------------------------------------ split by INIT year
# Split on initialisation date, never at random: one valid_time is reached by
# ~3 different initialisations at different leads, so a random split puts the
# same observed rainfall field in train and test (notebook cell 90 did this).
TRAIN_YEARS = tuple(range(2004, 2012))  # 2004-2011  (8 monsoons)
VAL_YEARS = (2012, 2013)
TEST_YEARS = (2014,)

# Days of separation enforced between an init date and the next split's data,
# so a 17-day forecast issued at the end of training cannot overlap validation.
SPLIT_EMBARGO_DAYS = LEAD_MAX

# ---------------------------------------------------------------- coarse grid
COARSE_RES = 1.5
COARSE_LATS = [18.0, 19.5, 21.0, 22.5, 24.0]
COARSE_LONS = [81.0, 82.5, 84.0, 85.5, 87.0]

# Cells where IMD reports no land rainfall at all (Bay of Bengal / outside the
# IMD land mask).  They were previously NaN->0 filled, making 24% of the target
# grid a free constant and inflating R^2.  They are masked out instead.
OCEAN_CELLS = [
    (18.0, 84.0), (18.0, 85.5), (18.0, 87.0),
    (19.5, 85.5), (19.5, 87.0),
    (21.0, 87.0),
]

# ------------------------------------------------------------------ fine grid
# IMD native resolution and the target of the downscaling step.
FINE_RES = 0.25
FINE_LAT_MIN, FINE_LAT_MAX = 18.0, 24.0
FINE_LON_MIN, FINE_LON_MAX = 81.0, 87.0

# Dong et al. (2025) sect. 3.2.1: predictors are read from a 3x3 patch of coarse
# cells centred on the target fine cell (3x3 beat 1x1, 5x5 and 7x7).
PATCH = 3

# Approximate bounding box of the Mahanadi catchment upstream of Hirakud dam
# (dam at 21.52 N, 83.87 E; catchment ~83,400 km^2 reaching into Chhattisgarh).
# Used to focus training and scoring on cells that can actually produce inflow:
# 190 of the 534 land cells in the domain lie east of the catchment and drain to
# the coast, not to the reservoir.
CATCHMENT_LAT_MIN, CATCHMENT_LAT_MAX = 19.75, 23.60
CATCHMENT_LON_MIN, CATCHMENT_LON_MAX = 80.50, 84.25

# ------------------------------------------------------------------- variables
# ECMWF S2S accumulates these from initialisation; they must be differenced
# along lead_day before use.  Verified: tp rises 10.8 -> 157.9 mm over leads 1-17.
ACCUMULATED_VARS = ["tp", "ssr", "sshf", "slhf"]

# Reported at the valid time already; no differencing.
INSTANT_VARS = ["t2m", "tcc", "u10", "v10", "msl"]

STATIC_VARS = ["orog"]

# Accumulated J/m^2 over one day -> mean W/m^2.
SECONDS_PER_DAY = 86400

# ---------------------------------------------------------------------- target
TARGET = "obs_rain"

# Dong et al. (2025) eq. 3-7: threat-score threshold is the 90th percentile of
# observed precipitation, computed per grid cell over the training period.
HEAVY_RAIN_PERCENTILE = 90

# A cell with too few observed rain days makes its own p90 unstable; fall back
# to the domain-wide train p90 for any cell below this count.
HEAVY_RAIN_MIN_OBS = 30

# ---------------------------------------------------------------------- output
CHECKPOINTS = ROOT / "checkpoints"
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
METRICS = RESULTS / "metrics"

# --------------------------------------------------------------------- training
SEED = 42

# Tested empirically, not assumed: land-region training (534 cells) was
# expected to generalise better via more samples and rainfall variability, but
# a direct A/B (see archive/run_land_region/ vs the catchment-only run) showed
# catchment-only training matched the eval region on every metric -- val R^2
# +0.084 vs -0.005 at epoch 1, test R^2 0.116 vs 0.092.  The 310 coastal cells
# outside the catchment appear to be a distribution the model has to spend
# capacity on without it transferring to the reservoir's actual catchment.
TRAIN_REGION = "catchment"
EVAL_REGION = "catchment"

BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
MAX_EPOCHS = 60
EARLY_STOP_PATIENCE = 8
EARLY_STOP_MIN_DELTA = 1e-4
LR_PLATEAU_FACTOR = 0.5
LR_PLATEAU_PATIENCE = 4
MIN_LR = 1e-6
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 0

# Differentiable threat-score loss (model.masked_hybrid_loss): `a` is the
# sigmoid sharpness around the rain threshold, `b` weights (1 - TS) against MSE.
LOSS_SIGMOID_SHARPNESS = 1.0
LOSS_TS_WEIGHT = 1.0

# MSE term computed on log1p(rain) rather than raw mm.  Tried as a fix for R^2
# being negative on the bottom 90% of rain days: it made every metric worse in
# real-mm terms (test R^2 0.116 -> 0.019, bias -1.5mm -> -4.6mm, Spearman 0.51
# -> 0.48) -- log-space MSE optimises something closer to the conditional
# median, and this target's mean sits well above its median (right-skewed),
# so the model learned to systematically underpredict.  Kept as an option
# (--log_space_loss) but off by default; do not re-enable without re-checking
# Light_R2 and Bias, not just overall R^2.
LOSS_LOG_SPACE = False


# =============================================================================
# S2S v2 -- the Dong et al. (2025) specification archive
# =============================================================================
# Everything above describes the ORIGINAL archive (ecmwf_s2s_reforecast_final.csv:
# 5x5 grid, leads 1-17, control member only, 16 columns) and is left untouched so
# the published baselines stay reproducible:
#
#     results/metrics/loyo_field.json     field downscaler      +0.1047 median
#
# The block below describes the re-download (preprocessing/download_s2s_dong.py
# -> preprocessing/build_s2s_archive.py).  New code reads these names; old code
# keeps reading the ones above.  Once the rebuild is scored against the baseline
# and wins, the two can be collapsed into one.

S2S_V2_DIR = PROCESSED / "s2s"          # one .npz per init date

# Wider request box, so every 0.25 deg fine cell has a genuine PATCH x PATCH
# neighbourhood.  On the old 5x5 domain 48 % of patches were clamped against an
# edge, which repeated a row or column of coarse cells as if it were data.
V2_COARSE_LATS = [25.5, 24.0, 22.5, 21.0, 19.5, 18.0, 16.5]   # note: descending,
V2_COARSE_LONS = [79.5, 81.0, 82.5, 84.0, 85.5, 87.0, 88.5]   # as GRIB stores it

# 30 days, matching Dong.  The 17 above was a limit of the old download, not of
# the S2S product.
V2_LEAD_MIN = 1
V2_LEAD_MAX = 30

# Control forecast (member 0) + 10 perturbed members.  The old archive had the
# control only, so no spread information existed at all.
V2_N_MEMBERS = 11
V2_CONTROL_MEMBER = 0

# 26 predictors = 11 surface + 5 upper-air x 3 levels.  The upper-air fields are
# the reason for the re-download: a surface-only forecast cannot place a storm in
# a particular 25 km cell, which is what heavy-rain R2 = -1.25 across all eleven
# folds (results/metrics/tier1_sweep.json) had been saying.
V2_SFC_VARS = ["tp", "cp", "t2m", "u10", "v10", "msl", "tcc",
               "ssrd", "sshf", "slhf", "orog"]
V2_PL_VARS = ["u", "v", "q", "t", "gh"]
V2_PL_LEVELS = [200, 500, 850]
V2_VARIABLES = V2_SFC_VARS + [f"{v}{lev}" for v in V2_PL_VARS for lev in V2_PL_LEVELS]

# ECDS names downward solar radiation `ssrd`; the old archive column was `ssr`.
# Same quantity, different label -- do not map one onto the other silently.
V2_ACCUMULATED_VARS = ["tp", "cp", "ssrd", "sshf", "slhf"]
V2_INSTANT_VARS = ["t2m", "tcc", "u10", "v10", "msl"] + \
                  [f"{v}{lev}" for v in V2_PL_VARS for lev in V2_PL_LEVELS]
V2_STATIC_VARS = ["orog"]

# RESOLVED 2026-09-06 -- this list is now EMPTY, and the note is kept because the
# problem it describes was real and the fix is easy to undo by accident.
#
# 2t and tcc are PERIOD-PROCESSED in the S2S reforecast, not instantaneous.  An
# instantaneous leadtime_hour request returned ONE message labelled "0-24" -- a
# single day-1 field broadcast along the lead axis, carrying no lead-time
# information at all.  download_s2s_dong.py now requests them as 24 h period
# windows (PERIOD_LEADTIME), which returns all 30 daily windows;
# preprocessing/probe_2t_tcc.py is the probe that established this.
#
# Verified on the archive the models actually train on (percell_v2_full.npz):
# across the 30 leads of one initialisation, t2m varies with std 0.671 and tcc
# with 0.866 in standardised units -- comparable to every other predictor.  Only
# `orog` is constant along lead, which is correct: it is static terrain.
#
# If this list is ever non-empty again, the download regressed to instantaneous
# hours.  Re-run the probe before trusting any model that reads those channels.
V2_LEAD_INVARIANT_VARS: list[str] = []

# The S2S reforecast fires Monday and Thursday only.  Three dates inherited from
# the old schedule (07-12, 07-19, 07-26) are Fridays in the 2024 model cycle and
# are rejected by ECDS with HTTP 400 -- they are not a download failure to retry.
V2_UNAVAILABLE_INITS = ["07-12", "07-19", "07-26"]
V2_N_INITS = 28                          # 31 scheduled - 3 impossible

# UNITS, read off the delivered GRIBs (preprocessing/build_s2s_archive.py output),
# not assumed from the ECMWF defaults:
#   tp, cp    mm accumulated from initialisation -- verified monotonic in lead
#             (2.4 mm at lead 1 -> 238 mm at lead 30, domain mean).  Difference
#             along the lead axis before use, as with the old archive.
#   tcc       PERCENT 0-100, not the 0-1 fraction the name suggests.
#   t2m, t*   kelvin.  msl pascal.  q kg/kg.  gh geopotential metres.
#   orog      metres; slightly negative over the Bay of Bengal corner, which is
#             the geoid, not a fill value.
V2_TP_UNITS = "mm"
V2_TCC_SCALE = 100.0          # divide by this to get a 0-1 fraction


# =============================================================================
# Dong et al. (2025) reproduction -- CNN downscaling settings
# =============================================================================
# Predictors listed in sect. 3.2.1: surface elevation, total and convective
# precipitation, and u, v, q, t, gh at 200/500/850 hPa.  That is 18; the paper
# says 19 but names no other.  The archive's 8 extra surface fields (t2m, u10,
# v10, msl, tcc, ssrd, sshf, slhf) are left out; --all-predictors brings them back.
DONG_PREDICTORS = ["orog", "tp", "cp"] + \
                  [f"{v}{lev}" for v in V2_PL_VARS for lev in V2_PL_LEVELS]

# Supplement Table S1: loss = b(1 - TS) + MSE, sigmoid(x) = 1 / (1 + e^(-a x)),
# a = 2 throughout and b rising as raw skill decays with lead.  One model covers
# all leads; the loss computes TS separately in each window and weights it by
# that window's b (see cnn/model.py masked_hybrid_loss).
DONG_LEAD_WINDOWS = [(1, 7), (8, 15), (16, 23), (24, 30)]
DONG_TS_WEIGHTS = [0.4, 0.8, 1.5, 2.0]
DONG_SIGMOID_SHARPNESS = 2.0

# RAINFALL IS SCALED BEFORE THE LOSS.  The paper does not say whether it was, and
# here it decides whether the TS term does anything.  In mm, Mahanadi monsoon rain
# has per-cell variance ~283 mm^2 and a p90 threshold of ~24 mm, so b <= 2 is
# under 1 % of the loss and a = 2 confines the sigmoid's gradient to about
# +/-1 mm around the threshold, where almost no prediction sits.  Dividing by the
# standard deviation of training-year rainfall (~16.8 mm) puts MSE near 1 and
# widens the sigmoid to roughly +/-8 mm, so Table S1's a and b mean something.
# The divisor is computed from the training years only; predictions are
# multiplied back to mm before anything is scored.

# Fixed split, mirroring the paper's 2002-2015 train / 2016-2019 test.  2017 is
# held out of training for early stopping.  QM calibrates on 2004-2017, as the
# paper's QM used its whole training period.
DONG_VAL_YEAR = 2017
DONG_TEST_YEARS = [2018, 2019, 2020, 2021, 2022]

# Dry-day cut-off for quantile mapping (Gudmundsson et al., 2012, as in sect. 3.2.2).
QM_WET_DAY_MM = 0.1
