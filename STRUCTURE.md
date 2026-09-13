# Project structure

Hirakud inflow forecasting, Mahanadi basin. Two papers are implemented here, so
most confusion comes from not knowing which files belong to which. This maps it.

    P1  Dong et al. (2025), HESS 29:2023-2042 -- CNN downscaling + hybrid hydrology
        docs/hess-29-2023-2025.pdf
    P2  FutureTST transformer for S2S forecasting
        docs/S2S Forecasting using Transformers (1).pdf
    v1  the project's original 11-monsoon pipeline.  REMOVED 2026-09-06 -- see
        the "v1 -- REMOVED" section below for what went, what could NOT go and
        why.  Everything that remains runs on 2004-2022.  +0.1047 is not a
        baseline and should not be quoted as one.

---

## The paper-1 chain, in run order

    preprocessing/download_s2s_dong.py     ECMWF S2S reforecast -> data/raw/s2s/*.grib
    preprocessing/build_s2s_archive.py     GRIB -> data/processed/s2s/*.npz
    preprocessing/build_downscaling_v2.py  -> downscaling_v2.npz  (field inputs)
    preprocessing/pack_percell_kaggle.py   -> percell_v2.npz      (per-cell inputs, Kaggle-sized)

    cnn/model.py                 Dong sect 3.2.1 ResNet: 64->32->16, coord embeddings
    cnn/loyo_percell_v2.py       the paper's per-cell downscaler, 19-fold LOYO
    cnn/loyo_field_v2.py         the field-level alternative, same folds
    cnn/quantile_mapping.py      EC and EC-QM benchmarks
    cnn/paper_metrics.py         RMSE/RE/CRPS on 5-day precip; RMSE/RE/NSE/REF on flow

    hydrology/vic/vic_states.py     one VIC state per initialisation (532)
    hydrology/vic/vic_forecast.py   VIC driven by each forecast product
    hydrology/vic/route_forecast.py RVIC unit-hydrograph routing -> inflow
    hydrology/vic/calibrate_vic.py  differential evolution on 8 parameters
    hydrology/vic/water_balance.py  annual runoff coefficient and ET/P -- the
                                    partition check, ANNUAL basis only

    inflow/lstm_postproc.py      LSTM correction of VIC -- the XAJ-LSTM role

## Paper 2 (FutureTST) -- do not delete its inputs

    inflow/futuretst_prob.py (current), futuretst_v3.py, futuretst_v2.py
    data/processed/daily_catchment_met.parquet
    data/processed/forecast_catchment_met_v2.parquet   19 monsoons, leads 1-30
    data/processed/ec_ensemble_catchment.parquet       ensemble met, read at runtime
    data/processed/inflow_dataset.parquet
    data/processed/loyo_oof_rainfall_cnn.parquet   rainfall handoff from the CNN stage
    data/processed/futuretst_pred_tst_vic.parquet
    results/metrics/futuretst_v2.json              baseline: forecast+fine-tune median NSE +0.313

## v1 -- REMOVED 2026-09-06

Everything here now runs on 2004-2022.  The v1 files that nothing needed were
deleted so they cannot cause confusion:

    cnn/loyo_field.py
    preprocessing/build_forcing.py                   v1 forcing, hard-coded 2004-2014
    preprocessing/build_forecast_met.py              superseded by build_forecast_met_v2.py
    data/raw/source/ecmwf_s2s_reforecast_final.csv   38 MB
    data/processed/forecast_catchment_met.parquet    2004-2014, leads 1-17 only
    results/metrics/loyo_field.json                  +0.1047, the old "baseline"

### WHAT COULD NOT BE REMOVED, AND WHY -- read before deleting anything else

"v1" is not a clean folder.  Three things everyone calls v1 are LIVE
dependencies of the 2004-2022 pipeline.  Deleting them breaks it:

    cnn/train_field.py         supplies device / set_seed / r2 / standardise to
                               cnn/loyo_percell_v2.py (the production downscaler),
                               cnn/loyo_field_v2.py AND inflow/lstm_postproc.py
    cnn/field_model.py         FieldDownscaler + masked_field_loss, imported by
                               cnn/loyo_field_v2.py
    data/processed/coarse_grid.npz
                               grid metadata read by build_climatology.py,
                               build_catchment_rainfall.py, inflow/loyo_cv.py,
                               cnn/thresholds.py, cnn/scalers.py
    data/processed/fine_target.npz
                               fine_lats / fine_lons / catchment mask, read by
                               preprocessing/build_daily_catchment_obs.py to build
                               daily_catchment_obs.parquet -- the observed rainfall
                               everything is verified against, and the file that
                               caught the 2026-09 date-label bug

Verified after removal: all of cnn.loyo_percell_v2, cnn.loyo_field_v2,
cnn.quantile_mapping, cnn.paper_metrics, inflow.lstm_postproc,
inflow.head_to_head, inflow.fair_comparison, inflow.futuretst_prob,
hydrology.vic.{vic_forecast,route_forecast,water_balance},
preprocessing.{build_daily_catchment_obs,build_forecast_met_v2,bias_correction}
still import cleanly.

One casualty, accepted: `bias_correction._self_test()` read the deleted v1
parquet.  The runtime path (`BC.fit` / `BC.apply`, used by FutureTST) does not,
so nothing that produces a result is affected.

Why v1 stopped being a baseline.  v1 is 5x5 coarse cells, 10 predictors, leads
1-17, the control member alone, over 11 monsoons.  v2 is 7x7, 26 predictors,
leads 1-30, 10 members, per cell, over 19.  A delta between them is not a
like-for-like measurement of anything, and it could only be computed on the 11
years they share (`V1_FOLDS` stopped at 2014, so 8 of 19 folds had delta = nan).

THE BASELINES THAT ACTUALLY APPLY, all on the full 19 monsoons:

    rainfall   raw ECMWF and EC-QM              cnn/paper_metrics.py
    inflow     0.86 x EC (LOYO scalar)          inflow/head_to_head.py
               persistence                      inflow/head_to_head.py

---

## Directories

    configs/       config.py -- every hard number, with the reason beside it.
                   The `V2_*` block is the paper-1 archive; the block above it
                   is v1 and is deliberately left untouched so v1 stays reproducible.
                   NOTE: C.YEAR_MAX is frozen at 2014 for v1.  Live scripts must
                   read their years from the DATA, never from that constant.
    data/raw/      s2s (GRIBs), imd (observations), vic (ERA5 forcing), source (v1 csv)
    data/processed/ derived arrays. vic/ holds VIC's forcing, states and live
                   calibration scratch -- vic/output is NOT disposable: forecast
                   routing splices 10 days of it for the convolution lags.
    results/metrics/ one JSON per experiment. Written via C.METRICS by many
                   scripts, so do not reorganise -- paths are hard-coded.
    logs/          current paper-1 runs.  logs/archive/ holds the July v1 runs.
    archive/       superseded experiment outputs, kept for provenance (23 MB)
    checkpoints/   trained model weights (17 MB)
    hydrology/vic/VIC/  the VIC 5.0.1 source tree and its compiled arm64 binary

## Two things that bite

1. `~/.vic_mahanadi` is a SYMLINK into data/processed/vic. Move the project and
   VIC silently stops working until it is repointed.
2. `tools/kaggle_run.py --skip-data` skips uploading the DATASET, and the project
   code ships inside that dataset -- so it also skips your code changes and
   Kaggle silently runs the previous version.
