"""Follow ONE forecast through every stage of both pipelines, with units.

The fastest way to understand this project is not to read it top to bottom, but
to take a single initialisation date and watch what happens to it -- what goes
in, what comes out, what the units are, and where each number changes meaning.
That is what this prints.

    python tools/trace_one_forecast.py                 # a good default date
    python tools/trace_one_forecast.py 2018-07-05      # any initialisation
    python tools/trace_one_forecast.py --list          # what dates exist

Nothing is recomputed: every stage is read from the artefact that stage wrote,
so the numbers here are exactly the numbers the results tables were built from.

The chain, and the unit change at each step -- the units are where the
understanding actually lives:

    ECMWF S2S      11 members x 30 leads x 26 vars on a 1.5 deg grid   mm/day
      | CNN downscaling            1.5 deg -> 0.25 deg, per catchment cell
    rainfall       224 cells x 30 days                                 mm/day
      | VIC             water balance per cell, 6-hourly, from a saved state
    runoff         150 cells x 30 days                                 mm/day
      | RVIC            unit-hydrograph convolution, 10 lags, x cell area
    inflow         30 days                                             m3/s
      | compare
    observed       30 days                                             m3/s

FutureTST skips the middle two entirely: catchment-mean weather straight to
inflow, which is the comparison the project exists to make.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

W = 78


def rule(title=""):
    if title:
        print(f"\n\033[1m{title}\033[0m\n" + "-" * W)
    else:
        print("-" * W)


def bar(v, lo, hi, width=22, ch="#"):
    """A crude inline bar so magnitudes are visible, not just readable."""
    if not np.isfinite(v):
        return " " * width
    f = 0.0 if hi <= lo else max(0.0, min(1.0, (v - lo) / (hi - lo)))
    n = int(round(f * width))
    return ch * n + "." * (width - n)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    q = np.load(C.PROCESSED / "vic_inflow_ec_cnn.npz", allow_pickle=True)
    inits = pd.to_datetime([str(x) for x in q["inits"]])

    if "--list" in sys.argv:
        for y, g in pd.Series(inits).groupby(inits.year):
            print(f"  {y}: {', '.join(d.strftime('%m-%d') for d in g)}")
        return

    want = pd.Timestamp(args[0]) if args else pd.Timestamp("2018-07-05")
    ki = int(np.argmin(np.abs(inits - want)))
    init = inits[ki]
    if init != want and args:
        print(f"note: {want.date()} is not an initialisation; using {init.date()}\n")

    days = pd.date_range(init + pd.Timedelta(days=1), periods=30)
    print("=" * W)
    print(f"  ONE FORECAST, END TO END".center(W))
    print(f"  issued {init:%A %d %B %Y}".center(W))
    print("=" * W)
    print(f"\n  valid for {days[0]:%d %b} to {days[-1]:%d %b %Y}  (leads 1-30)")

    # ---------------------------------------------------------------- stage 1
    rule("STAGE 1  ECMWF S2S reforecast -- the raw ingredient")
    ens = pd.read_parquet(C.PROCESSED / "ec_ensemble_catchment.parquet")
    ens["init_date"] = pd.to_datetime(ens["init_date"])
    e = ens[ens["init_date"] == init].sort_values("lead_day")
    if e.empty:
        sys.exit(f"no ensemble record for {init.date()}")
    print("  10 perturbed members, 26 predictors, 1.5 deg grid -> catchment mean")
    print(f"  {'lead':>5} {'raw mm/d':>9} {'spread':>8}   {'member disagreement':<24}")
    for L in (1, 3, 7, 14, 21, 30):
        r = e[e.lead_day == L].iloc[0]
        print(f"  {L:>5} {r.prec_raw_mean:>9.2f} {r.prec_raw_spread:>8.2f}   "
              f"{bar(r.prec_raw_spread, 0, 20)}")
    print(f"\n  Spread GROWS with lead -- that is the forecast losing confidence,")
    print(f"  and it is a real signal: Spearman(spread, |inflow error|) = +0.475.")

    # ---------------------------------------------------------------- stage 2
    rule("STAGE 2  CNN downscaling -- 150 km to 25 km  (pipeline 1 only)")
    oof = np.load(C.PROCESSED / "percell_v2_oof.npz", allow_pickle=True)
    valid = pd.to_datetime(oof["valid"])
    oi = (valid - pd.to_timedelta(oof["lead"], unit="D")) == init
    if oi.sum():
        cnn = oof["oof"][oi]                       # (30, 224) mm/day
        obs_r = oof["target"][oi]
        lead_o = oof["lead"][oi]
        order = np.argsort(lead_o)
        cnn, obs_r = cnn[order], obs_r[order]
        print(f"  in : 3x3 coarse patch per cell + lat/lon/elevation")
        print(f"  out: {cnn.shape[1]} catchment cells x {cnn.shape[0]} days, mm/day")
        print(f"\n  {'lead':>5} {'CNN mm/d':>9} {'IMD obs':>9}   {'catchment-mean rain':<24}")
        for L in (1, 3, 7, 14, 21, 30):
            print(f"  {L:>5} {cnn[L-1].mean():>9.2f} {obs_r[L-1].mean():>9.2f}   "
                  f"{bar(cnn[L-1].mean(), 0, 40)}")
        print(f"\n  The CNN's job is SPATIAL: one coarse value becomes 224 cell values,")
        print(f"  informed by terrain.  It cannot fix WHEN the rain falls.")
    else:
        print("  (no downscaled field stored for this initialisation)")

    # ---------------------------------------------------------------- stage 3
    rule("STAGE 3  VIC -- rainfall becomes runoff  (the physics)")
    vf = np.load(C.PROCESSED / "vic_forecast_ec_cnn.npz", allow_pickle=True)
    flux = vf["flux"][ki].mean(0)                  # (150 cells, 30 days) mm
    print(f"  in : 7 forcing variables per cell, 6-hourly, from a SAVED STATE")
    print(f"       (state_{init:%Y%m%d} -- how wet the soil already was)")
    print(f"  out: {flux.shape[0]} cells x {flux.shape[1]} days of runoff + baseflow, mm")
    print(f"\n  {'lead':>5} {'rain mm/d':>10} {'runoff mm':>10}   {'runoff ratio':>12}")
    for L in (1, 3, 7, 14, 21, 30):
        r_in = e[e.lead_day == L].iloc[0].prec_raw_mean
        r_out = flux[:, L-1].mean()
        print(f"  {L:>5} {r_in:>10.2f} {r_out:>10.2f}   "
              f"{(r_out / r_in if r_in > 0.05 else float('nan')):>12.2f}")
    print(f"\n  Runoff ratio is NOT constant -- that is the nonlinearity.  Rain on")
    print(f"  already-wet soil runs straight off; the same rain on dry soil soaks in.")
    print(f"  This is why VIC needs the state, and why a forecast cannot cold-start.")

    # ---------------------------------------------------------------- stage 4
    rule("STAGE 4  RVIC routing -- runoff becomes inflow at the dam")
    ens_q = q["q"][ki]                             # (10 members, 30 days) m3/s
    mean_q = ens_q.mean(0)
    print(f"  in : runoff depth per cell (mm)")
    print(f"  out: discharge at ONE point (m3/s), via unit hydrographs, 10 lags")
    print(f"  mm -> m3/s uses cell area and travel time; the 10 lags reach BACKWARDS")
    print(f"  into observed runoff from before the forecast -- history, not a peek.")

    obs_df = pd.read_parquet(C.PROCESSED / "inflow_daily_extended.parquet")
    obs_s = obs_df[obs_df["inflow_valid"]].set_index("date")["inflow"]
    truth = obs_s.reindex(days).to_numpy()

    ft_path = C.PROCESSED / "futuretst_prob_predictions.parquet"
    ft = None
    if ft_path.exists():
        f = pd.read_parquet(ft_path)
        f["init_date"] = pd.to_datetime(f["init_date"])
        f = f[f["init_date"] == init].sort_values("lead_day")
        if not f.empty:
            ft = f

    rule("STAGE 5  the answer, against what actually happened")
    hdr = f"  {'lead':>5} {'date':>7} {'VIC m3/s':>10} {'FTST P50':>9} " \
          f"{'80% interval':>17} {'OBSERVED':>10}"
    print(hdr)
    hi = np.nanmax([np.nanmax(truth), mean_q.max()]) * 1.05
    for L in (1, 3, 7, 14, 21, 30):
        row = f"  {L:>5} {days[L-1].strftime('%m-%d'):>7} {mean_q[L-1]:>10.0f}"
        if ft is not None and (ft.lead_day == L).any():
            r = ft[ft.lead_day == L].iloc[0]
            row += f" {r.p_qft_q50:>9.0f} {r.p_qft_q10:>7.0f}-{r.p_qft_q90:<9.0f}"
        else:
            row += f" {'-':>9} {'-':>17}"
        t = truth[L-1]
        row += f" {t:>10.0f}" if np.isfinite(t) else f" {'n/a':>10}"
        print(row)

    ok = np.isfinite(truth)
    if ok.sum() > 2:
        def nse(p, o):
            return 1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
        print(f"\n  over all 30 days:")
        print(f"    observed mean {np.nanmean(truth):>8.0f} m3/s   "
              f"peak {np.nanmax(truth):>8.0f} m3/s on day "
              f"{int(np.nanargmax(truth))+1}")
        print(f"    VIC chain     {mean_q[ok].mean():>8.0f} m3/s   "
              f"peak {mean_q[ok].max():>8.0f} m3/s on day {int(mean_q.argmax())+1}"
              f"   NSE {nse(mean_q[ok], truth[ok]):+.3f}")
        if ft is not None:
            p50 = ft.set_index("lead_day")["p_qft_q50"].reindex(range(1, 31)).to_numpy()
            lo = ft.set_index("lead_day")["p_qft_q10"].reindex(range(1, 31)).to_numpy()
            up = ft.set_index("lead_day")["p_qft_q90"].reindex(range(1, 31)).to_numpy()
            cov = ((truth[ok] >= lo[ok]) & (truth[ok] <= up[ok])).mean()
            print(f"    FutureTST     {p50[ok].mean():>8.0f} m3/s   "
                  f"peak {np.nanmax(p50[ok]):>8.0f} m3/s on day "
                  f"{int(np.nanargmax(p50))+1}"
                  f"   NSE {nse(p50[ok], truth[ok]):+.3f}")
            print(f"    80% interval contained the truth on "
                  f"{cov*100:.0f}% of days (nominal 80%)")

        print(f"\n  hydrograph, observed (o) against VIC (v):")
        for L in range(1, 31):
            t = truth[L-1]
            if not np.isfinite(t):
                continue
            print(f"    d{L:<3}{bar(t, 0, hi, 30, 'o')}  {t:>7.0f}")
            print(f"        {bar(mean_q[L-1], 0, hi, 30, 'v')}  {mean_q[L-1]:>7.0f}")

    rule("WHAT TO TAKE FROM THIS ONE CASE")
    print("""  1  The units change three times: mm/day -> mm -> m3/s.  Each change is a
     physical step (downscale, water balance, routing), not bookkeeping.
  2  Ensemble spread grows with lead.  That is the honest signal, and it is
     what the probabilistic model learns to widen its interval from.
  3  The runoff ratio is not constant, which is why a hydrological model
     needs the antecedent state and cannot be replaced by a scaling factor.
  4  Look at WHERE the peak lands, not just how big it is.  Across all 309
     windows the average peak timing error is ~7 days -- that phase error is
     70-90% of the total, and no post-processing can remove it.
  5  Run this on a few dates.  A good year and a bad year look completely
     different, which is exactly why 19-fold cross-validation was necessary.""")
    print()


if __name__ == "__main__":
    main()
