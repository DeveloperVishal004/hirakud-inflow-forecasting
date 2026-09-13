"""Calibrate VIC against observed Hirakud inflow by differential evolution.

Uncalibrated, VIC gets the TIMING right (corr 0.89) and the MAGNITUDE wrong
(+113 % bias, runoff coefficient 0.61 against a published 0.35-0.40).  That is
the signature of a model whose structure is sound and whose free parameters
have never been fitted, which is exactly what this fixes.

SIX PARAMETERS, the conventional VIC set (Ambika et al. 2025 calibrate 5-6):

  b_infilt   shape of the variable-infiltration curve.  Higher -> more of the
             cell saturates at a given moisture -> more surface runoff.  The
             single most important control on the runoff coefficient, and the
             right place to look first given a +113 % bias.
  Ds         fraction of Dsmax at which non-linear baseflow begins
  Dsmax      maximum baseflow rate (mm/day)
  Ws         fraction of maximum soil moisture at which non-linear baseflow
             begins
  d2, d3     thicknesses of soil layers 2 and 3 (m).  These set how much water
             the column can hold, and therefore how much is available for
             transpiration -- the lever on ET/P, which is currently 0.39
             against an expected 0.55-0.65.

Layer depths are calibrated rather than taken from SoilGrids because SoilGrids
reports texture by depth, not the *hydrologically active* column, and because
the Saxton & Rawls clamping documented in build_soil_params.py left the deeper
layers' properties partly assumption-driven.  Fitting them is honest about that.

The remaining soil properties -- Ksat, expt, bubble, Wcr, Wpwp, bulk density --
stay at their pedotransfer values.  They are derived from measurements and are
not free.

THE ABSTRACTION PROBLEM -- read this before touching any bound.

VIC simulates NATURAL runoff.  Hirakud inflow is not natural runoff: the
Mahanadi above the dam is heavily irrigated, so the gauge sees what is left
after upstream use.  Measured on this basin:

    IMD rainfall                     1,281.9 mm/yr
    observed Hirakud inflow            366.2 mm/yr   -> runoff coefficient 0.286
    published NATURAL coefficient      0.35-0.40     -> 448.7-512.8 mm/yr

So 80-150 mm/yr -- 18-29 % of natural runoff -- never reaches the gauge, and VIC
has no term for it.

Calibrating VIC to reproduce the OBSERVED volume therefore asks it to lose that
water through the only sink it has: evapotranspiration.  The optimiser obliges by
driving every ET lever to its limit.  That is exactly what was observed: d2
pinned at 1.404/1.50, then 2.9987/3.00, then 3.9991/4.00, while root_scale rose
to 2.18 and wpwp_scale fell to its floor.  The model was being asked to
reproduce human water use by growing an eight-metre soil column.

DECISION, 2026-09-06.  Calibrate to NATURAL runoff:
  - soil depths capped at physically defensible values (d2 <= 2 m, d3 <= 3 m)
  - the water-balance penalty targets the published NATURAL partition
    (runoff coefficient 0.35-0.40, ET/P 0.55-0.65) with weight 2.0
  - VIC is therefore EXPECTED to over-predict observed inflow, by roughly the
    abstraction.  That residual is real and physical, not calibration error.
    The `0.86 x EC` scalar and the LSTM already remove it downstream -- which
    is part of why a one-parameter scaling of the raw chain is so competitive.

Do not "fix" a positive VIC bias by widening a soil bound.  It is abstraction.

SPLIT, and why each period does what it does.  The inflow record is NOT uniform:
2003-2014 has near-complete years (292-363 days), while 2015-2022 is monsoon-only
(153 days, of which just 31 fall outside JJAS).

    2003        spin-up, discarded
    2004-2011   CALIBRATION, every observed day -- 2,740 days, of which ~1,800
                are outside the monsoon.  These are the only years that can
                constrain the dry-season recession, because they are the only
                ones where it was measured.
    2012-2014   VALIDATION / test, never scored during the search
    2015-2022   HELD OUT.  Monsoon-only, so it could contribute just 248
                non-JJAS days to an all-days objective -- nothing that would
                teach VIC about the water balance.  Held back instead as the
                out-of-sample evaluation set: it is 42 % of the head-to-head
                windows, and the VIC-unseen comparison in
                inflow/fair_comparison.py exists only because it was withheld.

Giving VIC more YEARS is not the lever people expect.  Eight years of daily
water balance already over-determines eight parameters; what was missing was
not years but DAYS -- see the note on --score-days in Objective.

Run:  python hydrology/vic/calibrate_vic.py --workers 8 --maxiter 20
Out:  results/metrics/vic_calibration.json
      data/processed/vic/soil_param_calibrated.txt
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs import config as C
from hydrology.calibrate_xaj import load_series, nse, kge
from hydrology.vic.build_global_param import VICDIR as SAFE_VIC

VIC_EXE = (Path(__file__).resolve().parent / "VIC" / "vic" / "drivers"
           / "classic" / "vic_classic.exe")
CALDIR = SAFE_VIC / "cal"
R_EARTH = 6371.0072

# 1-indexed columns of the classic-driver soil parameter file.
COL = {"b_infilt": 5, "Ds": 6, "Dsmax": 7, "Ws": 8, "c": 9,
       "d1": 23, "d2": 24, "d3": 25,
       "init1": 19, "init2": 20, "init3": 21,
       "wcr1": 41, "wcr2": 42, "wcr3": 43,
       "wpwp1": 44, "wpwp2": 45, "wpwp3": 46}

NAMES = ["b_infilt", "Ds", "Dsmax", "Ws", "d2", "d3", "wpwp_scale", "root_scale"]
BOUNDS = [(0.001, 0.6),     # b_infilt
          (0.001, 1.0),     # Ds
          (0.1, 40.0),      # Dsmax, mm/day
          (0.5, 1.0),       # Ws
          (0.10, 2.00),     # d2, m -- NARROWED BACK to a physical range, and
                            # this is the important change of 2026-09-06.
                            # d2 pinned at EVERY ceiling it was given: 1.404 of
                            # 1.50, 2.9987 of 3.00, 3.9991 of 4.00.  Widening a
                            # fourth time would have been the fourth repeat of
                            # the same mistake.  The reason it pins is not that
                            # the soil is deep -- see THE ABSTRACTION PROBLEM
                            # below -- and no bound can fix it.  2.00 m is the
                            # upper end of a defensible VIC layer 2.
          (0.30, 3.00),     # d3, m -- likewise capped at a physical depth
                            # (it reached 4.09 m in the run that was stopped)
          (0.25, 1.00),     # wpwp_scale -- floor RAISED from 0.15.  The second
                            # run pinned it at 0.153, which puts the wilting
                            # point near 0.09 of saturation: below any measured
                            # soil and past the point where the number means
                            # anything.  The optimiser was not finding physics
                            # there, it was exploiting the only ET lever it had.
                            # `root_scale` is the honest version of that lever.
          (0.75, 3.00)]     # root_scale -- NEW, see the note below.  1.0 leaves
                            # the veg library's 2.00 m rooting depth unchanged;
                            # 3.0 reaches 6.00 m, the full calibrated column.
                            # binding in the SAME direction: more water held in
                            # the column and more of it transpirable.  Leaving
                            # them where they were guarantees the bias survives.

# WHY root_scale IS CALIBRATED -- the diagnosis that produced this run.
#
# The second calibration ended with TWO parameters hard against their bounds,
# both pushing the same way: d2 = 2.9987 of a 3.000 ceiling, wpwp_scale = 0.1532
# of a 0.150 floor.  A thicker store and more extractable water in it: the
# optimiser asking, twice, for more evapotranspiration and being refused.
#
# Widening those two bounds again would have been the obvious move and the wrong
# one, because the geometry says they cannot deliver.  Measured on the resulting
# soil file (hydrology/vic/water_balance.py, and the root depths in
# veg_param.txt):
#
#     soil layers      0.00-0.15 m | 0.15-3.15 m | 3.15-6.11 m
#     rooting depth    0.00-2.00 m, Cv-weighted over all 1,198 veg tiles
#
# LAYER 3 CONTAINS NO ROOTS.  Nearly three metres of soil that cannot transpire,
# only store water and release it as baseflow.  Thickening d3 does not raise ET;
# it builds a bigger baseflow reservoir, which makes the runoff coefficient
# WORSE.  That is why the search kept reaching for the wilting point instead --
# it was the only ET lever left, and it ran out of physical room.
#
# Root depth was never a free parameter.  It should have been: rooting depth is
# among the least well constrained inputs in the whole file -- taken from a
# global veg library, not measured here -- and it is the direct control on how
# much of the column transpires.  `root_scale` multiplies the three root-zone
# depths per tile, leaving the fractions alone, so the profile shape is kept and
# only its reach changes.
#
# WHAT THE NUMBERS HAVE TO MOVE.  Over 2004-2022 the calibrated model gives
# runoff coefficient 0.469 and ET/P 0.538, closure 1.008.  Against three years
# of near-complete observed inflow the catchment's own runoff coefficient is
# 0.289 and VIC's discharge is +59.5 % too high.  Closure being sound means the
# two must move together: roughly 85-100 mm/yr has to leave as vapour instead of
# streamflow.  Forcing is not the culprit -- ERA5 precipitation is 1,244.6 mm/yr
# against IMD's 1,281.9, i.e. 2.9 % DRY.
#
# WHY wpwp_scale IS CALIBRATED, and why the first run without it stalled.
#
# The pedotransfer step had to clamp Saxton & Rawls to keep theta_1500 <
# theta_33 < theta_s: measured bulk density reaches 1.77 g/cm3 where texture
# implies 1.33-1.50, so the density adjustment drove porosity below field
# capacity in 82/146/153 of 256 cells by layer.  Clamping restored the ordering
# by COMPRESSING the plant-available range, leaving Wpwp at 0.57-0.65 of
# saturation against a typical 0.2-0.5 -- i.e. most pore water declared
# unextractable.
#
# The consequence showed up as a calibration that could not move the water
# balance: runoff coefficient went 0.611 -> 0.600 while b_infilt fell 0.20 ->
# 0.04 and d2 tripled.  Runoff was never the constraint; transpiration was.
# Scaling Wpwp (and Wcr with it, preserving Wpwp < Wcr) frees the ET term and
# treats the least trustworthy numbers in the soil file as what they are --
# uncertain -- rather than fixing them and fitting the better-constrained ones
# around them.


def cell_area_km2(lat, res=C.FINE_RES):
    return (R_EARTH ** 2 * np.radians(res)
            * (np.sin(np.radians(lat + res / 2)) - np.sin(np.radians(lat - res / 2))))


# Bands the ANNUAL water balance is asked to land in.  Both are wider than the
# published Mahanadi figures, deliberately, because the two ends disagree and
# pretending otherwise would be fitting to a false precision:
#
#   runoff coefficient  published natural 0.35-0.40.  The catchment's OWN ratio,
#     from three years of near-complete observed inflow against IMD rain, is
#     0.289 -- lower than any published figure.  Hirakud inflow is regulated and
#     upstream-abstracted, so it is not natural runoff; true natural runoff sits
#     somewhere between the two.  The band spans both readings.
#   ET / P  published 0.55-0.65, extended up to 0.70 because closure has to hold:
#     if the runoff coefficient belongs near 0.30 then ET/P belongs near 0.70.
# NATURAL runoff, not observed inflow.  See THE ABSTRACTION PROBLEM in the
# module docstring: observed Hirakud inflow is depleted by upstream use, so it is
# NOT the quantity VIC simulates.  The partition is therefore held to the
# published natural values and the volume mismatch against the gauge is expected.
RC_BAND = (0.35, 0.40)
ET_BAND = (0.55, 0.65)


def hinge(x, lo, hi):
    """Distance outside [lo, hi]; zero inside it."""
    if not np.isfinite(x):
        return 1e3
    return max(0.0, lo - x, x - hi)


class Objective:
    """Negative KGE of routed VIC discharge, penalised for a wrong water balance.

    Picklable, for parallel search.
    """

    def __init__(self, workers, cal_years, quiet=True,
                 wb_weight=1.0, rc_band=RC_BAND, et_band=ET_BAND,
                 score_days="all"):
        import xarray as xr

        self.cal_years = set(cal_years)
        self.quiet = quiet
        self.wb_weight = wb_weight
        self.rc_band = rc_band
        self.et_band = et_band
        self.score_days = score_days
        # READ THE PEDOTRANSFER BASE, NEVER THE ACTIVE FILE.
        #
        # This read `soil_param.txt` / `veg_param.txt` -- the files the
        # production run uses.  Activating a calibration overwrites those, so
        # the NEXT calibration used the previous winner as its starting point.
        # `wpwp_scale` and `root_scale` are MULTIPLIERS, so they compounded:
        # the 2026-09-06 root_scale run reported wpwp_scale = 0.8755 while the
        # effective value against the pedotransfer file was 0.1532 x 0.8755 =
        # 0.1341 -- a wilting point LOWER than the 0.153 that run was launched
        # to escape.  It also silently shifted the search space, so bounds no
        # longer meant what they said.
        #
        # The base files are immutable inputs and are checked below, because a
        # silent drift here is invisible in every downstream number.
        self.base_soil = (SAFE_VIC / "soil_param_base.txt").read_text().splitlines()
        self.base_veg = (SAFE_VIC / "veg_param_base.txt").read_text().splitlines()
        self._check_base()
        self.base_gp = (SAFE_VIC / "global_param.txt").read_text()

        prm = sorted((SAFE_VIC / "rvic" / "case" / "params").glob("*.prm.*.nc"))[-1]
        p = xr.open_dataset(prm)
        self.uh = p["unit_hydrograph"].values[:, :, 0]
        self.slat = p["source_lat"].values
        self.slon = p["source_lon"].values

        b = np.load(C.PROCESSED / "basin.npz", allow_pickle=True)
        self.blats, self.blons, frac = b["lats"], b["lons"], b["fraction"]
        # Per-source conversion from mm/day of depth to m3/s of volume.
        self.conv = np.array([
            cell_area_km2(self.blats[int(np.argmin(np.abs(self.blats - la)))])
            * 1e6
            * frac[int(np.argmin(np.abs(self.blats - la))),
                   int(np.argmin(np.abs(self.blons - lo)))]
            * 1e-3 / 86400.0
            for la, lo in zip(self.slat, self.slon)])

        # WHICH DAYS THE OPTIMISER SEES.  Until 2026-09-06 this was JJAS only:
        # VIC simulated the whole year continuously but was graded on 944 monsoon
        # days out of 2,740 available.  Two-thirds of each year -- the recession,
        # the dry-season baseflow, the soil drying down -- was invisible to the
        # search, so nothing pushed back when a parameter set got them wrong.
        # That is visible in the results: the 2026-09-06 recalibration improved
        # monsoon NSE (0.619 -> 0.668) while all-days NSE FELL (0.538 -> 0.369),
        # and the residual runoff-coefficient error is concentrated outside the
        # monsoon (non-monsoon bias +65 % against the monsoon's +20 %).
        #
        # Scoring every observed day roughly TRIPLES the sample on the same
        # years and is the only term that can constrain the recession.
        # --score-days monsoon restores the old behaviour for comparison.
        # HONOUR `inflow_valid`.  load_series() returns the raw `inflow` column
        # with missing days recorded as 0.0 and flagged in a separate `valid`
        # column; every other script in this project filters on that flag and
        # this one did not.  Measured on the calibration years: 213 of 3,287
        # days (6.5 %) are flagged invalid and ALL 213 are zeros, concentrated
        # in January-June -- the dry season.  Fitting to them teaches VIC to
        # produce no baseflow on days the gauge simply did not report, which
        # would look like an improved runoff coefficient obtained by inventing
        # a drought.  Switching to all-days scoring made this WORSE, not better
        # (JJAS-only saw ~39 of them; all-days sees all 213), so it had to be
        # fixed in the same change.
        _df = load_series()
        obs = _df["inflow"].where(_df["valid"])
        sel = obs.index.year.isin(self.cal_years)
        if self.score_days == "monsoon":
            sel &= obs.index.month.isin(C.INIT_MONTHS)
        # The spin-up year is EXCLUDED from scoring.  It is in cal_years so VIC
        # simulates it -- the run has to start somewhere -- but its errors come
        # from the assumed initial soil moisture, not from the parameters.  The
        # docstring always claimed it was discarded; until now it was not.
        sel &= obs.index.year > min(self.cal_years)
        self.obs = obs[sel].dropna()

        CALDIR.mkdir(parents=True, exist_ok=True)

    def _check_base(self):
        """Refuse to start from a file that is already a calibration output.

        Cheap, and it catches the one mistake that cannot be seen afterwards:
        the multipliers compound, so a run started from a previous winner
        produces plausible numbers that mean something other than they say.
        """
        wp = [float(l.split()[COL["wpwp1"] - 1])
              for l in self.base_soil if l.strip()]
        wp_mean = sum(wp) / len(wp)
        if wp_mean < 0.40:
            raise RuntimeError(
                f"soil_param_base.txt has mean wpwp1 {wp_mean:.4f}; the "
                f"pedotransfer value is ~0.549.  This file looks like a "
                f"CALIBRATION OUTPUT, so wpwp_scale would compound.  Restore it "
                f"from soil_param_uncalibrated_backup.txt.")
        d = [(float(f[2]) + float(f[4]) + float(f[6]))
             for f in (l.split() for l in self.base_veg) if len(f) == 8]
        d_mean = sum(d) / len(d)
        if abs(d_mean - 2.0) > 0.05:
            raise RuntimeError(
                f"veg_param_base.txt has mean rooting depth {d_mean:.3f} m; the "
                f"veg library value is 2.000 m.  This file looks like a "
                f"CALIBRATION OUTPUT, so root_scale would compound.")
        print(f"  base files    wpwp1 {wp_mean:.4f}, rooting depth {d_mean:.3f} m "
              f"-- pedotransfer, not a previous winner")

    # -- write a soil file with the trial parameters substituted
    def _soil(self, v, path):
        b_infilt, Ds, Dsmax, Ws, d2, d3, wp_scale = v
        out = []
        for line in self.base_soil:
            f = line.split()
            f[COL["b_infilt"] - 1] = f"{b_infilt:.5f}"
            f[COL["Ds"] - 1] = f"{Ds:.5f}"
            f[COL["Dsmax"] - 1] = f"{Dsmax:.4f}"
            f[COL["Ws"] - 1] = f"{Ws:.4f}"
            old2 = float(f[COL["d2"] - 1])
            old3 = float(f[COL["d3"] - 1])
            f[COL["d2"] - 1] = f"{d2:.4f}"
            f[COL["d3"] - 1] = f"{d3:.4f}"
            # Initial moisture is a DEPTH-INTEGRATED quantity (mm), so it has to
            # scale with the layer it describes; leaving it fixed while the
            # layer thickens starts the run desiccated and biases early years.
            for k, old, new in (("init2", old2, d2), ("init3", old3, d3)):
                f[COL[k] - 1] = f"{float(f[COL[k] - 1]) * new / old:.4f}"
            # Scale the wilting point, and move the critical point with it so
            # the ordering Wpwp < Wcr < 1 is preserved by construction rather
            # than left to chance.
            for k in range(3):
                wp_col = COL["wpwp1"] - 1 + k
                wc_col = COL["wcr1"] - 1 + k
                wp = float(f[wp_col]) * wp_scale
                wc = float(f[wc_col])
                wc = max(wp + 0.05, min(wc, 0.99))
                f[wp_col] = f"{wp:.5f}"
                f[wc_col] = f"{wc:.5f}"
            out.append(" ".join(f))
        path.write_text("\n".join(out) + "\n")

    # -- write a veg file with the root zones stretched by root_scale
    def _veg(self, root_scale, path):
        """Scale rooting DEPTH, leave the root FRACTIONS alone.

        With ROOT_ZONES 3 each vegetation tile is

            veg_class Cv  d1 f1  d2 f2  d3 f3

        where d* are zone thicknesses in metres and f* the fraction of roots in
        each, summing to 1.  Scaling the three depths and not the fractions
        stretches the profile without changing its shape: the same proportion of
        root stays in the top third, it just reaches deeper.  Scaling fractions
        instead would break their sum and VIC would reject the file.

        Header lines (`cell_id n_tiles`) are passed through untouched -- they are
        the only two-field lines, which is what distinguishes them.
        """
        out = []
        for line in self.base_veg:
            f = line.split()
            if len(f) == 8:                       # a tile line, ROOT_ZONES = 3
                for c in (2, 4, 6):               # depth columns only
                    f[c] = f"{float(f[c]) * root_scale:.4f}"
                out.append("  " + " ".join(f))
            else:
                out.append(line)
        path.write_text("\n".join(out) + "\n")

    def _global(self, d, soil_path, veg_path):
        # Replace by LINE KEY, never by exact string match on a path.  The
        # template addresses VIC through the ~/.vic_mahanadi symlink while
        # VICDIR resolves to the real directory, so the old str.replace()
        # matched nothing after the project moved -- silently.  Every worker
        # then read the SHARED baseline soil file and wrote to the SHARED
        # output, so the search re-evaluated the same parameters ~1,000 times
        # and every route() found an empty directory.  Keying on the first
        # token cannot fail that way.
        override = {"SOIL": f"SOIL         {soil_path}",
                    "VEGPARAM": f"VEGPARAM     {veg_path}",
                    "RESULT_DIR": f"RESULT_DIR   {d / 'output'}/",
                    "LOG_DIR": f"LOG_DIR      {d / 'logs'}/"}
        lines, seen = [], set()
        for line in self.base_gp.splitlines():
            k = line.split()[0] if line.split() else ""
            if k in override:
                lines.append(override[k]); seen.add(k)
            else:
                lines.append(line)
        missing = set(override) - seen
        if missing:
            raise RuntimeError(f"global_param.txt has no {sorted(missing)} line to "
                               f"override -- refusing to run a calibration whose "
                               f"workers would share files")
        gp = "\n".join(lines)
        gp = "\n".join(f"ENDYEAR     {max(self.cal_years)}"
                       if l.split()[:1] == ["ENDYEAR"] else l
                       for l in gp.splitlines())
        p = d / "global_param.txt"
        p.write_text(gp)
        return p

    def _workdir(self):
        """A private directory per PROCESS.

        differential_evolution(workers=N) forks processes, not threads, so any
        thread-based worker id collapses to one directory and the processes
        overwrite each other's soil file and output mid-run -- silently, since
        each still reads *a* valid set of fluxes.  Keying on the pid is what
        actually isolates them.
        """
        import os
        d = CALDIR / f"pid{os.getpid()}"
        if not d.exists():
            (d / "output").mkdir(parents=True, exist_ok=True)
            (d / "logs").mkdir(parents=True, exist_ok=True)
        return d

    def __call__(self, v):
        d = self._workdir()
        try:
            soil = d / "soil_param.txt"
            veg = d / "veg_param.txt"
            self._soil(v[:7], soil)
            self._veg(v[7], veg)
            gp = self._global(d, soil, veg)
            # Clear BOTH outputs and logs.  VIC writes a fresh timestamped log
            # per run; over ~1,000 evaluations those alone reached 2.7 GB per
            # worker and filled the disk, killing the run at the final write.
            for sub, pat in (("output", "fluxes_*"), ("logs", "*")):
                for f in (d / sub).glob(pat):
                    f.unlink()
            r = subprocess.run([str(VIC_EXE), "-g", str(gp)],
                               capture_output=True, text=True, timeout=1800)
            # Delete the logs NOW, not at the start of the next evaluation.
            # VIC writes ~228 MB of log per run; holding it until the next
            # evaluation means every worker sits on a copy at once.  On 8
            # workers that was 2.1 GB of the 4.2 GB free on this disk, and a
            # calibration that dies at the final write after several hours is
            # the most expensive possible failure.  The run is already finished
            # here, so nothing needs them.
            for f in (d / "logs").glob("*"):
                try:
                    f.unlink()
                except OSError:
                    pass
            if r.returncode != 0:
                return 1e6
            q, wb = self._route(d / "output")
            if q is None:
                return 1e6
            j = q.reindex(self.obs.index).dropna()
            if len(j) < 100:
                return 1e6
            # KGE, not NSE.  NSE is dominated by correlation and barely
            # penalises a volume error when the shape is right -- which is
            # exactly this model's failure mode (corr 0.89, bias +58 %).  The
            # first calibration reached NSE 0.485 while leaving the bias at
            # +52 %, because NSE was nearly indifferent to it.  KGE decomposes
            # into correlation, variability AND bias ratio, so the search is
            # actually pushed to fix the water balance.
            score = kge(j.values, self.obs.reindex(j.index).values)

            # WATER-BALANCE PENALTY.  KGE is scored on MONSOON discharge only, so
            # a parameter set can match the hydrograph while partitioning the
            # ANNUAL water balance wrongly -- which is what happened: KGE alone
            # left the runoff coefficient at 0.469 against a physical 0.35-0.40.
            # Streamflow is scored where it is observed; the partition is scored
            # over the whole year, where it is defined.  The penalty is a hinge:
            # zero anywhere inside the band, linear outside it, so the search is
            # not pushed toward a single point inside a range that is itself
            # uncertain.
            #
            # Linear, not squared, on purpose.  A squared hinge is negligible
            # near the band and explodes far from it, which lets a badly
            # partitioned but well-fitting set survive early generations.
            pen = (hinge(wb["runoff_coefficient"], *self.rc_band)
                   + hinge(wb["ET_over_P"], *self.et_band))
            # Kept so the winning set can be re-evaluated in the PARENT process
            # after the search and its balance reported.  differential_evolution
            # forks workers, so an attribute set inside a child is invisible
            # here -- this is only ever read after a deliberate re-run below.
            self.last_wb = dict(wb, KGE=float(score), penalty=float(pen))
            return -(score - self.wb_weight * pen)
        except Exception:  # noqa: BLE001 -- a bad parameter set must not stop the search
            return 1e6

    def _route(self, outdir):
        """Routed discharge, and the ANNUAL water balance of the same run.

        Returns (Series of m3/s, dict of ratios).  The balance is accumulated
        here rather than re-read later because these flux files are deleted at
        the start of the next evaluation -- the only chance to measure them is
        while they exist.  Spin-up is excluded: its storage change is an
        artefact of the assumed initial moisture, not of the parameters.
        """
        files = {}
        for f in outdir.glob("fluxes_*.txt"):
            p = f.stem.split("_")
            files[(round(float(p[1]), 4), round(float(p[2]), 4))] = f
        if not files:
            return None, None
        vol, dates = None, None
        wb = np.zeros(4)                      # runoff, baseflow, evap, prec
        n_cell = 0
        for s, (la, lo) in enumerate(zip(self.slat, self.slon)):
            key = (round(float(la), 4), round(float(lo), 4))
            if key not in files:
                return None, None
            a = np.loadtxt(files[key], skiprows=3)
            if vol is None:
                vol = np.zeros((len(a), len(self.slat)))
                dates = pd.to_datetime(dict(year=a[:, 0].astype(int),
                                            month=a[:, 1].astype(int),
                                            day=a[:, 2].astype(int)))
                post = (dates.dt.year > min(self.cal_years)).values
            vol[:, s] = (a[:, 3] + a[:, 4]) * self.conv[s]
            wb += a[post][:, [3, 4, 5, 6]].sum(axis=0)
            n_cell += 1
        q = np.zeros(len(vol))
        for k in range(self.uh.shape[0]):
            q[k:] += (vol[:len(vol) - k] * self.uh[k][None, :]).sum(axis=1)
        P = wb[3]
        bal = {"runoff_coefficient": float((wb[0] + wb[1]) / P) if P > 0 else np.nan,
               "ET_over_P": float(wb[2] / P) if P > 0 else np.nan,
               "closure": float((wb[0] + wb[1] + wb[2]) / P) if P > 0 else np.nan,
               "P_mm_per_year": float(P / n_cell / (post.sum() / 365.25))
                                if P > 0 and n_cell else np.nan}
        return pd.Series(q, index=dates), bal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--maxiter", type=int, default=20)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--wb-weight", type=float, default=2.0,
                    help="weight on the annual water-balance penalty. 0 "
                         "reproduces the previous KGE-only objective; 1.0 makes "
                         "a 0.10 error in either ratio cost 0.10 of KGE, which "
                         "is the same order as the spread between good and bad "
                         "parameter sets, so neither term can be ignored")
    ap.add_argument("--rc-band", default="0.35,0.40",
                    help="acceptable annual runoff coefficient, lo,hi")
    ap.add_argument("--et-band", default="0.55,0.65",
                    help="acceptable annual ET/P, lo,hi")
    ap.add_argument("--score-days", choices=["all", "monsoon"], default="all",
                    help="which observed days the objective scores. 'all' (the "
                         "default since 2026-09-06) grades the recession and dry "
                         "season too; 'monsoon' restores the JJAS-only objective "
                         "used before that date")
    args = ap.parse_args()
    rc_band = tuple(float(x) for x in args.rc_band.split(","))
    et_band = tuple(float(x) for x in args.et_band.split(","))

    from scipy.optimize import differential_evolution

    cal_years = [C.YEAR_MIN - 1] + list(C.TRAIN_YEARS)
    print(f"VIC calibration")
    print(f"  parameters   {', '.join(NAMES)}")
    print(f"  calibrate    {C.TRAIN_YEARS[0]}-{C.TRAIN_YEARS[-1]} monsoons "
          f"(+{C.YEAR_MIN - 1} spin-up)")
    print(f"  validate     {C.VAL_YEARS}   test {C.TEST_YEARS}  (never scored here)")
    print(f"  budget       popsize {args.popsize} x {len(NAMES)} params "
          f"x {args.maxiter} iters ~ "
          f"{args.popsize * len(NAMES) * (args.maxiter + 1):,} runs "
          f"on {args.workers} workers\n")

    print(f"  objective    KGE(monsoon discharge) - {args.wb_weight:g} x "
          f"[hinge(runoff coeff, {rc_band[0]:.2f}-{rc_band[1]:.2f})")
    print(f"                                     + hinge(ET/P, "
          f"{et_band[0]:.2f}-{et_band[1]:.2f})]")
    print(f"               flow scored where it is OBSERVED (monsoon), the "
          f"partition where it is DEFINED (annual)\n")

    obj = Objective(args.workers, cal_years, wb_weight=args.wb_weight,
                    rc_band=rc_band, et_band=et_band, score_days=args.score_days)
    print(f"  scoring      {args.score_days} observed days in the calibration "
          f"years -- {len(obj.obs):,} days")
    ckpt = C.METRICS / "vic_calibration_checkpoint.json"
    C.METRICS.mkdir(parents=True, exist_ok=True)

    def save_best(xk, convergence=None):
        """Persist the incumbent after every generation.

        disp=True prints only the objective, so a run that is stopped early --
        which, on a machine that is needed for other things, is the normal
        case -- leaves a score nobody can reproduce.
        """
        ckpt.write_text(json.dumps(
            {"params": dict(zip(NAMES, map(float, xk))),
             "elapsed_min": (time.time() - t0) / 60}, indent=2))
        return False

    t0 = time.time()
    res = differential_evolution(
        obj, bounds=BOUNDS, maxiter=args.maxiter, popsize=args.popsize,
        seed=args.seed, tol=1e-3, mutation=(0.5, 1.0), recombination=0.7,
        polish=False, disp=True, workers=args.workers, updating="deferred",
        callback=save_best)
    mins = (time.time() - t0) / 60
    best = dict(zip(NAMES, map(float, res.x)))
    print(f"\n  best penalised score {-res.fun:+.4f}   ({mins:.0f} min, "
          f"{res.nfev} evaluations)")
    for k, v in best.items():
        lo, hi = BOUNDS[NAMES.index(k)]
        span = hi - lo
        at = ""
        if v - lo < 0.01 * span:
            at = "  <-- AT LOWER BOUND, the search wanted less than it was allowed"
        elif hi - v < 0.01 * span:
            at = "  <-- AT UPPER BOUND, the search wanted more than it was allowed"
        print(f"    {k:11s} {v:9.4f}   [{lo:g}, {hi:g}]{at}")

    # ---- write the calibrated soil AND veg files, and report what the winning
    # parameter set actually does to the water balance.  The previous run wrote
    # neither the balance nor a veg file, which is how a runoff coefficient of
    # 0.469 went unnoticed and how `vic_inflow_calibrated.json` ended up with a
    # `water_balance` block no script produces.
    final = C.PROCESSED / "vic" / "soil_param_calibrated.txt"
    final_veg = C.PROCESSED / "vic" / "veg_param_calibrated.txt"
    obj._soil(res.x[:7], final)
    obj._veg(res.x[7], final_veg)
    print(f"\nwrote {final}")
    print(f"wrote {final_veg}")

    # Re-run the winner once, here in the parent, purely to recover its water
    # balance: the search ran in forked workers and nothing they measured came
    # back.  One VIC run, a few minutes, and the alternative is shipping another
    # calibration whose partition nobody looked at.
    print("\n  re-evaluating the winning set to recover its water balance ...")
    obj(res.x)
    wb = getattr(obj, "last_wb", None)
    if wb:
        print("\n  water balance of the winning set (annual, spin-up excluded):")
        print(f"    runoff coefficient {wb['runoff_coefficient']:.3f}   "
              f"target {rc_band[0]:.2f}-{rc_band[1]:.2f}")
        print(f"    ET / P             {wb['ET_over_P']:.3f}   "
              f"target {et_band[0]:.2f}-{et_band[1]:.2f}")
        print(f"    closure            {wb['closure']:.3f}")

    out = {"params": best, "penalised_score": float(-res.fun),
           "objective": "KGE(monsoon) - wb_weight * "
                        "[hinge(runoff_coeff, rc_band) + hinge(ET_over_P, et_band)]",
           "wb_weight": args.wb_weight, "score_days": args.score_days,
           "n_days_scored": int(len(obj.obs)),
           "rc_band": list(rc_band), "et_band": list(et_band),
           "water_balance": wb,
           "n_evaluations": int(res.nfev), "minutes": mins,
           "bounds": dict(zip(NAMES, BOUNDS))}
    C.METRICS.mkdir(parents=True, exist_ok=True)
    (C.METRICS / "vic_calibration.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {C.METRICS / 'vic_calibration.json'}")
    print("\nNext, in order:")
    print("  1. rerun VIC over the full record with BOTH calibrated files")
    print("     (global_param.txt must point SOIL and VEGPARAM at them)")
    print("  2. python hydrology/vic/water_balance.py     -- confirm the "
          "partition held over 2004-2022, not just the calibration years")
    print("  3. python hydrology/vic/route_and_evaluate.py -- validation and test")
    print("  4. everything downstream of VIC is now stale: vic_states, "
          "vic_forecast, route_forecast, lstm_postproc, head_to_head")


if __name__ == "__main__":
    main()
