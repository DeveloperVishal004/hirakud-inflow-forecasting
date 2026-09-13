# Sub-seasonal Reservoir Inflow Forecasting — Hirakud Dam, Mahanadi Basin

Forecasting daily inflow to Hirakud Dam **1–30 days ahead** during the monsoon,
and establishing honestly whether it can be done.

Two published approaches are reimplemented and compared on 19 monsoon seasons
(2004–2022). They are kept **strictly independent** — no output of one is ever an
input to the other.

```
LAYER 2 — physics-based        ECMWF → CNN → VIC → RVIC → LSTM → inflow
LAYER 3 — learned              history + forecast weather → FutureTST → inflow
```

> **`docs/PROJECT_DOCUMENTATION.md` is the single source of truth** for the
> architecture, experimental design, results and limitations.
>
> Anything under `archive/` may contain obsolete numbers and **must not be used
> for reporting results.**

---

## Repository layout

    configs/          config.py — every hard number, with the reason beside it
    preprocessing/    ECMWF download/archive, IMD, ERA5, catchment inputs
    cnn/              CNN downscaling (Layer 2), quantile-mapping benchmarks
    hydrology/vic/    VIC setup, calibration, forecasting, RVIC routing
    inflow/           LSTM correction, FutureTST, head-to-head evaluation
    tools/            diagnostics, repair utilities, Kaggle runner
    docs/             PROJECT_DOCUMENTATION.md, FINDINGS.md
    results/metrics/  one JSON per experiment
    archive/          superseded code and documentation (do not quote)

`STRUCTURE.md` maps which file belongs to which layer, and lists the traps.

---

## Running the main pipeline

Assumes inputs are already built (see `hydrology/vic/check_inputs.py`, which
reports what is present at any point). Interpreter: `/usr/local/bin/python3`.

**Layer 2 — physics chain**

```bash
# rainfall
python cnn/loyo_percell_v2.py            # CNN downscaling, 19-fold LOYO (GPU)
python cnn/quantile_mapping.py           # EC and EC-QM benchmarks

# hydrology  (VIC must be calibrated first; ~2 h)
python hydrology/vic/calibrate_vic.py --workers 4 --maxiter 8 --popsize 6
hydrology/vic/VIC/vic/drivers/classic/vic_classic.exe \
    -g data/processed/vic/global_param.txt          # full record 2003-2022
python hydrology/vic/water_balance.py               # annual partition check
python hydrology/vic/route_and_evaluate.py          # cal / val / held-out splits

# forecasting
python hydrology/vic/vic_states.py                  # 532 states, ~3 min
for p in ec ec_qm ec_cnn; do
    python hydrology/vic/vic_forecast.py --product $p --workers 4   # ~20 min each
    python hydrology/vic/route_forecast.py --product $p
done
python inflow/lstm_postproc.py --product ec
python inflow/lstm_postproc.py --product ec_cnn
```

**Layer 3 — FutureTST**

```bash
python inflow/futuretst_prob.py          # needs CUDA; see tools/kaggle_run.py
```

**Comparison**

```bash
python inflow/head_to_head.py            # the results table
python inflow/fair_comparison.py         # significance, VIC-exposure split
```

---

## Two things that bite

1. `~/.vic_mahanadi` is a **symlink** into `data/processed/vic`. Move the project
   and VIC silently stops working until it is repointed.
2. `tools/kaggle_run.py --skip-data` skips uploading the **dataset**, and the
   project code ships inside that dataset — so it also skips your code changes,
   and Kaggle silently runs the previous version.

---

## Data and dependencies — not in this repository

Neither the input data (~6.4 GB) nor the VIC source tree is redistributed here.

| what | where it comes from |
|---|---|
| ECMWF S2S reforecast | ECMWF / Copernicus ECDS — `preprocessing/download_s2s_dong.py` |
| IMD gridded rainfall | India Meteorological Department, 0.25° daily |
| ERA5-Land | Copernicus CDS — `preprocessing/download_era5_land.py` |
| Hirakud inflow | Government of Odisha / Hirakud Dam authority |
| VIC 5.0.1 | https://github.com/UW-Hydro/VIC (GPL-2.0) — build, then point `hydrology/vic/VIC/` at it |
| HydroSHEDS DIR/ACC | https://www.hydrosheds.org |

`python hydrology/vic/check_inputs.py` reports which inputs are present at any
point. `hydrology/vic/DATA_REQUIREMENTS.md` is the full manifest.

## Papers implemented

The PDFs are not redistributed. Both are cited here:

- **Dong, N. et al. (2025).** Deep-learning-based sub-seasonal precipitation and
  streamflow ensemble forecasting. *Hydrology and Earth System Sciences* 29,
  2023–2042. https://doi.org/10.5194/hess-29-2023-2025 (open access, CC BY 4.0)
- **Ambika, A. K. et al. (2025).** FutureTST: sub-seasonal streamflow
  forecasting with transformers. *Geophysical Research Letters* 52,
  e2025GL116707.

## Licence

Code in this repository is MIT (see `LICENSE`). VIC is GPL-2.0 and is **not**
included. The datasets above remain under their providers' terms.

## Status

FutureTST results predate a data repair and have not been re-run; they are
marked provisional in the documentation. Layer 2 results are current as of
7 September 2026.
