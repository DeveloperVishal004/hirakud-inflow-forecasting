# TS-weight (`b`) sweep — how to run it on Kaggle

## Why

Dong et al. eq. 3 is `loss = b(1 - TS) + MSE`. Measured on our trained CNN
(2026-09-08):

    MSE term          265.32 mm^2
    b*(1 - TS)          1.00        (b = 1)
    TS share of loss    0.375 %
    threat score TS     0.000       (92 hits, 77 false alarms, 357,412 misses)

The extreme-rain term is numerically inert: the CNN is training on plain MSE.
That is the direct cause of the damping — catchment rainfall variance is 0.431
of observed at leads 1-3 (raw ECMWF keeps 0.696), and inflow peak ratio is 0.64
against the raw chain's 0.90.

The paper does not specify `b`, so setting it is ours to do — the same situation
as `n_seq`. For the TS term to carry ~20 % of the loss, `b` must be around 66.

## One-time setup (not done on this machine)

    pip install kaggle            # already installed
    # Kaggle -> Settings -> API -> "Create New Token"
    mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
    export KAGGLE_USERNAME=<your-kaggle-username>

## The sweep

Six folds per value, not 19 — enough to see the trend at ~45 min a run instead
of 2.5 h. Kaggle allows ~30 GPU-hours a week and caps a session near 12 h.

    for B in 1 10 30 66 150; do
      python tools/kaggle_run.py "cnn/loyo_percell_v2.py --ts-weight $B --folds 6 --tag _b$B"
    done

`--tag` keeps the outputs apart: each run writes
`results/metrics/loyo_percell_v2_b<B>.json` and
`data/processed/percell_v2_oof_b<B>.npz`, so nothing overwrites the production
`percell_v2_oof.npz`.

## TWO TRAPS, both will cost you a run

1. `tools/kaggle_run.py --skip-data` skips uploading the DATASET, and the code
   ships inside that dataset — so it also skips your code changes and Kaggle
   silently runs the previous version. Do not use it here.
2. Kaggle assigns a **P100** by default and its PyTorch build has no kernels for
   that chip, so every convolution fails. Set **GPU T4 x2** in the web UI, on
   EVERY push. A command-line push resets whatever the UI had set.

## What to judge it on

Raising `b` should RAISE the threat score and peak ratio and LOWER overall R^2.
That is the intended trade, so R^2 is the wrong criterion. Read, in order:

    1. threat score TS               (currently 0.000 -- anything above ~0.1 is progress)
    2. rainfall variance ratio       (currently 0.431 of observed at leads 1-3)
    3. inflow peak ratio after VIC   (currently 0.64; raw ECMWF chain gets 0.90)
    4. overall R^2                   (currently +0.0571 -- expect this to FALL)

A `b` that lifts peak ratio toward 0.9 while holding R^2 near zero is a WIN for
this problem, because the project's failure mode is damped extremes, not mean
accuracy.

## After the sweep

Pick `b`, then run the full 19 folds untagged so it becomes the production file:

    python tools/kaggle_run.py "cnn/loyo_percell_v2.py --ts-weight <B>"

then locally re-run the chain that depends on it:

    python hydrology/vic/vic_forecast.py --product ec_cnn --workers 4
    python hydrology/vic/route_forecast.py --product ec_cnn
    python inflow/lstm_postproc.py --product ec_cnn
    python inflow/head_to_head.py && python inflow/fair_comparison.py
