"""Calibrate XAJ against observed Hirakud inflow, then score it honestly.

Calibration uses differential evolution on NSE, which is the same class of
global search (population-based, derivative-free) that Dong et al. (2025) drive
with particle swarm, and is appropriate here because the XAJ response surface is
non-convex with strongly interacting parameters.

Two things this deliberately does NOT do:

  * It does not calibrate on the years it reports.  Parameters are fitted on the
    training monsoons and scored on the held-out ones, because a conceptual
    model with 17 free parameters can fit almost any single hydrograph.
  * It does not stop at the calibration score.  The question that matters is
    whether XAJ beats the statistical rainfall-to-inflow mapping already in
    place (Ridge, LOYO median NSE +0.338), so that comparison is printed.

Run:  python hydrology/calibrate_xaj.py [--maxiter 60]
Out:  results/metrics/xaj.json, checkpoints/xaj_params.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C
from hydrology.xaj import XAJParams, run_xaj

# Mahanadi catchment upstream of Hirakud.
AREA_KM2 = 83_400.0
WARMUP_DAYS = 365


def load_series() -> pd.DataFrame:
    """Continuous daily rainfall + inflow. XAJ needs an unbroken time axis."""
    rain = pd.read_parquet(C.PROCESSED / "daily_catchment_obs.parquet")
    rain = rain.set_index("date")["rain_obs_daily"].sort_index()

    # Prefer the 2003-2022 record when it exists.  inflow_daily.parquet stops at
    # 2014, so scoring against it silently drops eight monsoons -- the routed
    # comparison reported "overlapping days 4,383" while the VIC run covered
    # 7,304.  Note the extended file is monsoon-only (Jun-Oct) after 2014.
    qf = next((f for f in (C.PROCESSED / "inflow_daily_extended.parquet",
                           C.PROCESSED / "inflow_daily.parquet") if f.exists()),
              C.PROCESSED / "inflow_daily.parquet")
    q = pd.read_parquet(qf)
    q = q.set_index("date").sort_index()

    idx = pd.date_range(max(rain.index.min(), q.index.min()),
                        min(rain.index.max(), q.index.max()), freq="D")
    df = pd.DataFrame({
        "rain": rain.reindex(idx).astype(float),
        "inflow": q["inflow"].reindex(idx).astype(float),
        "valid": q["inflow_valid"].reindex(idx).fillna(False).astype(bool),
    }, index=idx)
    df["rain"] = df["rain"].fillna(0.0)
    df["doy"] = df.index.dayofyear
    df["year"] = df.index.year
    df["is_monsoon"] = df.index.month.isin(C.INIT_MONTHS)
    return df


def nse(sim, obs):
    return 1 - ((sim - obs) ** 2).sum() / ((obs - obs.mean()) ** 2).sum()


def kge(sim, obs):
    r = np.corrcoef(sim, obs)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (sim.std() / obs.std() - 1) ** 2
                       + (sim.mean() / obs.mean() - 1) ** 2)


class Objective:
    """Negative NSE, as a picklable callable.

    A closure would be simpler but cannot cross a process boundary, and
    differential evolution with workers=-1 pickles the objective to send it to
    each worker.  Holding the arrays as attributes keeps it picklable.
    """

    def __init__(self, df, score_mask):
        self.rain = df["rain"].to_numpy()
        self.doy = df["doy"].to_numpy()
        self.obs = df["inflow"].to_numpy()
        self.mask = score_mask

    def __call__(self, v):
        try:
            out = run_xaj(self.rain, self.doy, XAJParams.from_vector(v), AREA_KM2)
        except (FloatingPointError, ValueError, ZeroDivisionError):
            return 1e6
        sim = out["q_cumecs"]
        if not np.all(np.isfinite(sim)):
            return 1e6
        m = self.mask
        return -nse(sim[m], self.obs[m])      # minimise negative NSE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--popsize", type=int, default=18)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    df = load_series()
    print(f"series {df.index.min().date()} -> {df.index.max().date()}  ({len(df)} days)")

    warm = np.zeros(len(df), bool)
    warm[:WARMUP_DAYS] = True
    usable = (~warm) & df["valid"].to_numpy() & df["is_monsoon"].to_numpy()

    is_train = df["year"].isin(C.TRAIN_YEARS).to_numpy()
    is_val = df["year"].isin(C.VAL_YEARS).to_numpy()
    is_test = df["year"].isin(C.TEST_YEARS).to_numpy()
    cal_mask = usable & is_train
    print(f"calibrate on {cal_mask.sum()} monsoon days ({C.TRAIN_YEARS[0]}-{C.TRAIN_YEARS[-1]}), "
          f"validate {int((usable & is_val).sum())}, test {int((usable & is_test).sum())}")

    lo, hi = XAJParams.bounds_arrays()
    print(f"calibrating {len(lo)} parameters by differential evolution "
          f"(maxiter={args.maxiter}, popsize={args.popsize}) ...")

    result = differential_evolution(
        Objective(df, cal_mask), bounds=list(zip(lo, hi)),
        maxiter=args.maxiter, popsize=args.popsize, seed=args.seed,
        tol=1e-4, mutation=(0.5, 1.0), recombination=0.7,
        polish=True, disp=True, workers=-1, updating="deferred",
    )

    p = XAJParams.from_vector(result.x)
    out = run_xaj(df["rain"].to_numpy(), df["doy"].to_numpy(), p, AREA_KM2)
    sim, obs = out["q_cumecs"], df["inflow"].to_numpy()

    metrics = {}
    print("\n=== XAJ performance (monsoon days only) ===")
    for label, m in [("calibration", cal_mask), ("validation", usable & is_val),
                     ("test", usable & is_test)]:
        if m.sum() < 10:
            continue
        metrics[label] = {
            "NSE": float(nse(sim[m], obs[m])), "KGE": float(kge(sim[m], obs[m])),
            "RMSE": float(np.sqrt(((sim[m] - obs[m]) ** 2).mean())),
            "bias": float(sim[m].mean() - obs[m].mean()), "n": int(m.sum()),
        }
        d = metrics[label]
        print(f"  {label:12s} NSE {d['NSE']:+.3f}  KGE {d['KGE']:+.3f}  "
              f"RMSE {d['RMSE']:7.0f}  bias {d['bias']:+7.0f}  (n={d['n']})")

    # Per-year, to expose whether a good average hides bad years.
    print("\n  per-year NSE (monsoon days):")
    per_year = {}
    for yr in sorted(df.loc[~warm, "year"].unique()):
        m = usable & (df["year"] == yr).to_numpy()
        if m.sum() < 30:
            continue
        per_year[int(yr)] = float(nse(sim[m], obs[m]))
        tag = ("cal" if yr in C.TRAIN_YEARS else
               "val" if yr in C.VAL_YEARS else "test")
        print(f"    {yr} [{tag:4s}] {per_year[int(yr)]:+.3f}")

    print("\n=== the comparison that matters ===")
    print("  XAJ is a SIMULATION driven by OBSERVED rainfall -- it is not a")
    print("  forecast.  The like-for-like reference is the statistical mapping")
    print("  fed observed rainfall (perfect-rain), not the operational forecast.")
    ref = C.METRICS / "inflow.json"
    if ref.exists():
        d = json.load(open(ref))
        pr = d.get("perfect-rain|Ridge [PRODUCTION]|test", {}).get("NSE")
        fc = d.get("downscaled|Ridge [PRODUCTION]|test", {}).get("NSE")
        if pr is not None:
            print(f"  Ridge, perfect rainfall (2014 test) : NSE {pr:+.3f}")
            print(f"  Ridge, forecast rainfall (2014 test): NSE {fc:+.3f}")
            if "test" in metrics:
                print(f"  XAJ,   observed rainfall (2014 test): NSE {metrics['test']['NSE']:+.3f}")
                verdict = ("XAJ improves the rainfall->inflow mapping"
                           if metrics["test"]["NSE"] > pr else
                           "XAJ does NOT beat the statistical mapping")
                print(f"  -> {verdict}")
                metrics["comparison"] = {"ridge_perfect_rain": pr,
                                         "ridge_forecast_rain": fc,
                                         "xaj_observed_rain": metrics["test"]["NSE"],
                                         "verdict": verdict}

    params = {n: float(getattr(p, n)) for n in XAJParams.NAMES}
    C.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    (C.CHECKPOINTS / "xaj_params.json").write_text(json.dumps(params, indent=2))
    C.METRICS.mkdir(parents=True, exist_ok=True)
    (C.METRICS / "xaj.json").write_text(json.dumps(
        {"metrics": metrics, "per_year_NSE": per_year, "params": params,
         "area_km2": AREA_KM2, "warmup_days": WARMUP_DAYS,
         "calibration_objective": "NSE on training-monsoon days"}, indent=2))

    # Simulated inflow for the hybrid stage: XAJ state as a feature for Ridge,
    # which is Dong et al.'s XAJ-LSTM design with a linear head.
    pd.DataFrame({"date": df.index, "q_xaj": sim, "runoff_mm": out["runoff_mm"]}) \
        .to_parquet(C.PROCESSED / "xaj_simulation.parquet", index=False)
    print(f"\nwrote {C.METRICS / 'xaj.json'}, {C.CHECKPOINTS / 'xaj_params.json'}, "
          f"{C.PROCESSED / 'xaj_simulation.parquet'}")


if __name__ == "__main__":
    main()
