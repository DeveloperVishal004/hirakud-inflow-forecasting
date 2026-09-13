"""Repair the scrambled `valid` labels in ec_qm_v2*.npz, and rebuild what they fed.

THE BUG (fixed at source in cnn/quantile_mapping.py).  That script reorders the
forecast data into the dataset's row order, then saved `lead`/`valid` taken from
load_members() -- which were never reordered.  Result: data in dataset order,
date labels in filename-outer order.

  * `lead` survived by luck: 1..30 repeats identically under both orderings.
  * `valid` did not: `valid - lead` gave the true initialisation on only
    300 of 15,960 rows (1.9 %).

Everything keyed on `valid` silently read another forecast's rainfall --
inflow/lstm_postproc.py, inflow/futuretst_prob.py, and the derived
ec_ensemble_catchment.parquet.  Measured: 522 of 532 initialisations got the
wrong 30-day series; correlation with the correct series +0.03; lead-1
forecast-vs-observed correlation -0.02 instead of +0.68.

WHY REPAIR RATHER THAN RE-RUN.  ec / ec_qm / obs / obs_mask / init_year / cells
are all in dataset order and provably correct -- `init_year` matches the
dataset's initialisation year on every row, and the in-file observations
reproduce IMD exactly when looked up by the dataset's dates (r = 1.0000).  Only
the label array is wrong, so re-fitting the quantile maps would change nothing
and cost a full rebuild.  This swaps the labels and re-derives the one artefact
built from them.

Every step is verified before anything is written, and the originals are kept
as *.bak_scrambled_labels.

Run:  python tools/repair_ec_qm_labels.py [--apply]
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

APPLY = "--apply" in sys.argv


def obs_series():
    o = pd.read_parquet(C.PROCESSED / "daily_catchment_obs.parquet")
    return pd.Series(o["rain_obs_daily"].to_numpy(), index=pd.to_datetime(o["date"]))


def lead1_corr(valid, lead, prec, dc):
    """Correlation of the lead-1 forecast against observed rain on its valid date.

    This is the honest test of a date labelling: a one-day-ahead forecast must
    track what actually fell.  Scrambled labels drive it to zero.
    """
    s = lead == 1
    look = dc.reindex(pd.DatetimeIndex(valid[s])).to_numpy()
    m = np.isfinite(look) & np.isfinite(prec[s])
    return float(np.corrcoef(look[m], prec[s][m])[0, 1]) if m.sum() > 2 else float("nan")


def main():
    dc = obs_series()
    src = C.PROCESSED / "downscaling_v2_full.npz"
    if not src.exists():
        sys.exit(f"missing {src}")
    d = np.load(src, allow_pickle=True)
    true_init, true_valid, true_lead = d["init"], d["valid"], d["lead"]

    for name in ("ec_qm_v2_full.npz", "ec_qm_v2.npz"):
        f = C.PROCESSED / name
        if not f.exists():
            print(f"{name}: absent, skipping")
            continue
        z = np.load(f, allow_pickle=True)
        if len(z["lead"]) != len(true_lead):
            print(f"{name}: {len(z['lead'])} rows vs dataset {len(true_lead)} -- "
                  f"different build, skipping (repair manually if needed)")
            continue

        cat = z["ec"].mean(2).mean(1)                    # catchment mean, per row
        before = lead1_corr(pd.to_datetime(z["valid"]), z["lead"].astype(int), cat, dc)
        after = lead1_corr(pd.to_datetime(true_valid), true_lead.astype(int), cat, dc)
        iy_ok = np.array_equal(z["init_year"].astype(int),
                               pd.to_datetime(true_init).year.to_numpy())

        print(f"\n{name}")
        print(f"  init_year already agrees with dataset init year : {iy_ok}")
        print(f"  lead-1 corr(forecast, observed)  BEFORE : {before:+.3f}")
        print(f"  lead-1 corr(forecast, observed)  AFTER  : {after:+.3f}")
        if not (after > 0.3 and after > before + 0.2):
            print("  REFUSING to write: the replacement labels do not clearly "
                  "improve the physical check.")
            continue
        if not APPLY:
            print("  (dry run -- pass --apply to write)")
            continue

        shutil.copy2(f, f.with_suffix(".npz.bak_scrambled_labels"))
        out = {k: z[k] for k in z.files}
        out["lead"], out["valid"], out["init"] = true_lead, true_valid, true_init
        np.savez_compressed(f, **out)
        print(f"  written; original kept as {f.name}.bak_scrambled_labels")

    # ---- rebuild the derived ensemble parquet ------------------------------
    f = C.PROCESSED / "ec_qm_v2_full.npz"
    dst = C.PROCESSED / "ec_ensemble_catchment.parquet"
    if not f.exists():
        return
    z = np.load(f, allow_pickle=True)
    if not np.array_equal(pd.to_datetime(z["valid"]).to_numpy(),
                          pd.to_datetime(true_valid).to_numpy()):
        print("\nec_ensemble_catchment.parquet: source labels still scrambled, "
              "not rebuilding")
        return

    # No committed script builds this file -- it is read by futuretst_prob.py
    # and trace_one_forecast.py but written by nothing in the repository.  It is
    # reconstructed here from the repaired archive so it has a source.
    raw = z["ec"].mean(2)          # (rows, members) catchment mean per member
    qm = z["ec_qm"].mean(2)
    df = pd.DataFrame({
        "init_date": pd.to_datetime(z["init"]),
        "lead_day": z["lead"].astype(int),
        "prec_raw_mean": raw.mean(1), "prec_raw_spread": raw.std(1),
        "prec_raw_p10": np.percentile(raw, 10, axis=1),
        "prec_raw_p90": np.percentile(raw, 90, axis=1),
        "prec_qm_mean": qm.mean(1), "prec_qm_spread": qm.std(1),
        "prec_qm_p10": np.percentile(qm, 10, axis=1),
        "prec_qm_p90": np.percentile(qm, 90, axis=1),
    }).sort_values(["init_date", "lead_day"]).reset_index(drop=True)

    chk = lead1_corr(df["init_date"] + pd.to_timedelta(df["lead_day"], unit="D"),
                     df["lead_day"].to_numpy(), df["prec_raw_mean"].to_numpy(), dc)
    print(f"\nec_ensemble_catchment.parquet  rebuilt rows={len(df):,}  "
          f"lead-1 corr {chk:+.3f}")
    if chk < 0.3:
        print("  REFUSING to write: physical check still fails.")
        return
    if not APPLY:
        print("  (dry run -- pass --apply to write)")
        return
    if dst.exists():
        shutil.copy2(dst, dst.with_suffix(".parquet.bak_scrambled_labels"))
    df.to_parquet(dst, index=False)
    print(f"  written; original kept as {dst.name}.bak_scrambled_labels")

    stale = C.PROCESSED / "raw_ec_catchment_prec.parquet"
    if stale.exists():
        if APPLY:
            stale.rename(stale.with_suffix(".parquet.bak_scrambled_labels"))
        print(f"  {stale.name}: derived from the broken labels -- "
              f"{'moved aside' if APPLY else 'would be moved aside'}")


if __name__ == "__main__":
    main()
