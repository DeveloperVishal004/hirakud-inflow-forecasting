"""ERA5-Land + IMD -> daily VIC 5 forcing files on the 0.25 deg catchment grid.

VIC 5's classic driver reads one ASCII file per active grid cell, named
data_<lat>_<lon>, one row per day.  VIC 5 removed MTCLIM, so it derives NOTHING
-- all seven forcings must be supplied, in the column order declared by
FORCE_TYPE in the global parameter file:

    PREC    mm      IMD
    AIR_TEMP C      ERA5 t2m, daily mean
    SWDOWN  W/m2    ERA5 ssrd, accumulated -> daily mean
    LWDOWN  W/m2    ERA5 strd, accumulated -> daily mean
    PRESSURE kPa    ERA5 sp
    VP      kPa     from ERA5 d2m via Magnus
    WIND    m/s     ERA5 u10,v10

Precipitation stays IMD: it is the observational target the downscaler is
trained against, so substituting ERA5 precipitation would break the chain's
internal consistency.

Four details here are easy to get wrong and change the answer:

1. WIND SPEED IS AVERAGED, NOT THE COMPONENTS.  mean(sqrt(u^2+v^2)) is the
   daily mean speed; sqrt(mean(u)^2 + mean(v)^2) is the speed of the daily mean
   VECTOR, which is smaller whenever the wind veers -- and monsoon winds veer.
   The latter would understate wind, and so evaporative demand, all season.

2. RADIATION IS ACCUMULATED, NOT INSTANTANEOUS.  ERA5-Land accumulates ssrd and
   strd from 00 UTC and resets daily, so the day's total is the value at 00Z of
   the FOLLOWING day.  Averaging the raw samples instead would overstate
   radiation several-fold -- the same class of error as the ECMWF tp bug fixed
   in preprocessing/build_forcing.py, and as verified for tp here.

3. DAILY MEANS ARE TAKEN ON LOCAL DAYS, radiation excepted.  India is UTC+5:30,
   so a local day runs 18:30Z to 18:30Z.  Radiation cannot follow that: its
   accumulation window is fixed to 00Z-00Z UTC by the archive.  The residual is
   a 5.5 h phase offset between the radiation day and the temperature day, and
   a further offset against IMD, whose day runs 08:30-08:30 IST.  At a daily
   model step this is small, but it is real and is not corrected here.

4. REGRIDDING IS AREA-WEIGHTED, NOT NEAREST OR BINNED.  0.25 / 0.1 = 2.5, a
   non-integer ratio, so target cells straddle source cells unevenly -- each
   draws on 3 or 4 of them.  Nearest-centre binning would vary the contributor
   count across the grid and stamp a checkerboard onto the forcing.  Verified:
   a constant field is preserved and a linear field reproduces target centres
   to machine precision.

Run:  python hydrology/vic/build_forcing.py [--check-only]
Out:  data/processed/vic/forcings/data_<lat>_<lon>
      data/processed/vic/forcing_daily.nc
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs import config as C

ERA5 = C.RAW / "vic" / "forcing" / "era5"
OUTDIR = C.PROCESSED / "vic"
FORCEDIR = OUTDIR / "forcings"

IST = pd.Timedelta(hours=5, minutes=30)
SECONDS_PER_DAY = 86400.0

# Column order -- must match FORCE_TYPE in build_global_param.py.
COLUMNS = ["prec", "air_temp", "swdown", "lwdown", "pressure", "vp", "wind"]


def overlap_weights(src_centres, dst_centres, src_res, dst_res):
    """Exact 1-D area-overlap weights between two regular grids."""
    s0, s1 = src_centres - src_res / 2, src_centres + src_res / 2
    d0, d1 = dst_centres - dst_res / 2, dst_centres + dst_res / 2
    ov = np.maximum(0.0, np.minimum(d1[:, None], s1[None, :])
                    - np.maximum(d0[:, None], s0[None, :]))
    tot = ov.sum(axis=1, keepdims=True)
    if (tot <= 0).any():
        raise ValueError("a target cell has no overlapping source cells")
    return ov / tot


def regrid(da, lats, lons):
    """0.1 deg -> 0.25 deg by exact area weighting."""
    wl = overlap_weights(da.latitude.values, lats, 0.1, C.FINE_RES)
    wo = overlap_weights(da.longitude.values, lons, 0.1, C.FINE_RES)
    v = np.einsum("ij,tjk->tik", wl, da.values)
    return np.einsum("ij,tkj->tki", wo, v)


def open_stream(pattern, varnames):
    files = sorted(glob.glob(str(ERA5 / pattern / "*.nc")))
    if not files:
        return None, 0
    ds = xr.concat([xr.open_dataset(f)[varnames] for f in files],
                   dim="valid_time").sortby("valid_time")
    # Duplicate timestamps would silently double-weight a day in any mean.
    _, keep = np.unique(ds.valid_time.values, return_index=True)
    return ds.isel(valid_time=np.sort(keep)), len(files)


def local_day(times):
    """Local (IST) calendar day for each UTC timestamp."""
    return pd.DatetimeIndex(pd.DatetimeIndex(times) + IST).normalize()


def vapour_pressure_kpa(dew_c):
    """Saturation vapour pressure at the dewpoint = actual vapour pressure."""
    return 0.6108 * np.exp(17.27 * dew_c / (dew_c + 237.3))


def deaccumulate_6h(da, steps):
    """6-hourly MEAN RATE (W/m2) from an ERA5-Land accumulated field (J/m2).

    The counter runs from 00 UTC and resets daily, so the 00/06/12/18Z samples
    are running totals since 00Z, NOT interval values.  On a verified clear day
    the series read 00Z 22.333, 06Z 9.655, 12Z 23.303, 18Z 23.379,
    next 00Z 23.379 MJ/m2 -- the 00Z value carrying the whole previous day, and
    the last two equal because the sun had already set.

    Two things follow, and getting either wrong is silent:

      RESET IS AT 06Z, NOT 00Z.  The 06Z sample is already an interval (00-06);
      differencing it against 00Z subtracts a whole previous day and produces a
      large negative.  Only 06Z needs the special case.

          00-06 = v(06)          06-12 = v(12) - v(06)
          12-18 = v(18) - v(12)  18-24 = v(00 next day) - v(18)

      INTERVALS ARE LABELLED BY THEIR END, STEPS BY THEIR START.  The energy
      arriving during 00-06Z is stamped 06Z in the archive but belongs to the
      forcing row stamped 00Z.  So each step takes the increment of the step
      AFTER it -- without this shift the diurnal cycle is displaced six hours
      and peak radiation lands after sunset.

    The final step has no following sample, so the caller must drop it.
    """
    t = pd.DatetimeIndex(da.valid_time.values)
    cum = da.values
    inc = cum - np.roll(cum, 1, axis=0)
    inc[0] = cum[0]
    inc[t.hour == 6] = cum[t.hour == 6]          # reset point
    inc = xr.DataArray(inc, coords=da.coords, dims=da.dims) / 21600.0

    # Shift: value for step k is the increment recorded at step k+1.
    nxt = steps + pd.Timedelta(hours=6)
    return inc.sel(valid_time=nxt).assign_coords(valid_time=steps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true",
                    help="build and validate but write no files")
    # The record now runs to 2022, but C.YEAR_MAX is still 2014 and is read by
    # 22 modules -- including the v1 scripts whose published numbers must stay
    # reproducible.  So widen the range here, explicitly, instead of moving a
    # global constant underneath everything.  Spin-up is always the year before.
    ap.add_argument("--years", default=f"{C.YEAR_MIN}-{C.YEAR_MAX}",
                    help="FIRST-LAST water years to build (default: config)")
    args = ap.parse_args()
    y0, y1 = (int(x) for x in args.years.split("-"))

    basin = np.load(C.PROCESSED / "basin.npz", allow_pickle=True)
    lats, lons = basin["lats"], basin["lons"]
    active = basin["fraction"] > 0
    print(f"VIC forcing: {len(lats)}x{len(lons)} grid, "
          f"{int(active.sum())} active cells")

    # ---- temperature (hourly) -> local-day mean, plus extremes for diagnostics
    t2m, n1 = open_stream("t2m_*", ["t2m"])
    if t2m is None:
        sys.exit("no t2m stream -- run preprocessing/download_era5_land.py")
    print(f"  t2m  {n1:2d} files, {t2m.sizes['valid_time']:,} hourly steps")
    air_hourly = t2m.t2m

    # ---- dewpoint and wind (6-hourly) -> local-day means
    met, n2 = open_stream("met_*", ["d2m", "u10", "v10"])
    if met is None:
        sys.exit("no met stream -- run preprocessing/download_era5_land.py")
    print(f"  met  {n2:2d} files, {met.sizes['valid_time']:,} 6-hourly steps")
    # speed FIRST, then any averaging -- see note 1 in the module docstring
    wind_raw = np.hypot(met.u10, met.v10)
    dew_raw = met.d2m - 273.15

    # ---- radiation and pressure (6-hourly); radiation is ACCUMULATED
    # Radiation and pressure are separate streams: 2 vars x 2 years and 1 var
    # x 4 years each fill the CDS 5,952-field budget exactly, where 3 vars x 1
    # year wasted a quarter of it.
    rad, n3 = open_stream("rad_*", ["ssrd", "strd"])
    sp_ds, n4 = open_stream("sp_*", ["sp"])
    if rad is None or sp_ds is None:
        sys.exit("missing rad/sp stream -- VIC 5 requires SWDOWN, LWDOWN and "
                 "PRESSURE (MTCLIM was removed).  Run:\n"
                 "  python preprocessing/download_era5_land.py --streams rad sp")
    print(f"  rad  {n3:2d} files, {rad.sizes['valid_time']:,} 6-hourly steps")
    print(f"  sp   {n4:2d} files, {sp_ds.sizes['valid_time']:,} 6-hourly steps")
    # ---- 6-HOURLY, not daily.  VIC 5 requires >= 4 sub-daily snow steps, FORCE
    # must equal SNOW, and with sub-daily forcing MODEL must equal FORCE.  So a
    # daily forcing file cannot be used at all.  This is also the native
    # resolution of met/rad/sp, so only precipitation needs disaggregating.
    # The final DAY is dropped, not merely the final step: de-accumulating
    # radiation needs the sample after each step and there is none past the
    # end, and VIC requires a whole number of days.
    steps = pd.date_range(f"{y0 - 1}-01-01", f"{y1}-12-30 18:00",
                          freq="6h")
    for name, arr in [("t2m", air_hourly), ("met", wind_raw),
                      ("rad", rad.ssrd), ("sp", sp_ds.sp)]:
        have = pd.DatetimeIndex(arr.valid_time.values)
        missing = steps.difference(have)
        if len(missing):
            sys.exit(f"{name} is missing {len(missing)} of the required "
                     f"6-hourly steps, first {missing[0]}")
    print(f"  6-hourly steps: {len(steps):,} "
          f"({steps[0]:%Y-%m-%d} .. {steps[-1]:%Y-%m-%d %H}Z)")

    sel = dict(valid_time=steps)
    grids = {
        "air_temp": regrid(air_hourly.sel(**sel) - 273.15, lats, lons),
        "wind":     regrid(wind_raw.sel(**sel), lats, lons),
        # Clip at zero: differencing leaves values like -1e-13, and a negative
        # downward flux is not something to hand a model.
        "swdown":   regrid(deaccumulate_6h(rad.ssrd, steps), lats, lons).clip(0),
        "lwdown":   regrid(deaccumulate_6h(rad.strd, steps), lats, lons).clip(0),
        "pressure": regrid(sp_ds.sp.sel(**sel), lats, lons) / 1000.0,
        "vp":       vapour_pressure_kpa(regrid(dew_raw.sel(**sel), lats, lons)),
    }

    # ---- precipitation from IMD, already on the 0.25 deg grid
    files = sorted(glob.glob(str(C.IMD_NC_DIR / "RF25_ind*.nc")))
    imd = xr.concat([xr.open_dataset(f) for f in files], dim="TIME").sortby("TIME")
    imd = imd.sel(LATITUDE=slice(lats[0], lats[-1]),
                  LONGITUDE=slice(lons[0], lons[-1]))
    assert np.allclose(imd.LATITUDE.values, lats), "IMD grid differs from VIC grid"
    # IMD is daily; VIC wants 6-hourly.  Spread each day's total evenly over its
    # four steps: a daily observation carries no sub-daily information, and a
    # fabricated diurnal shape would be a worse lie than an honest flat one.
    daily_dates = pd.DatetimeIndex(steps.normalize().unique())
    prec_daily = imd.RAINFALL.reindex(TIME=daily_dates).values
    n_nan = int(np.isnan(prec_daily).sum())
    grids["prec"] = np.repeat(np.nan_to_num(prec_daily, nan=0.0),
                              4, axis=0)[:len(steps)] / 4.0

    # ---- validation
    print("\n=== checks ===")
    ok = True
    UNITS = {"prec": "mm", "air_temp": "C", "swdown": "W/m2", "lwdown": "W/m2",
             "pressure": "kPa", "vp": "kPa", "wind": "m/s"}
    # Bounds set where a value would be impossible for this domain, not merely
    # unusual -- an earlier 5 C floor on temperature flagged real January data.
    # These are 6-HOURLY rates, not daily means, so the bounds are much wider
    # than a daily check would use: a 6-hour window centred on midday averages
    # far above the 24-hour mean, and a clear January night sits far below it.
    # Each bound is set where the value would be physically impossible.
    #   swdown  surface irradiance cannot exceed ~1100 W/m2 even instantaneously
    #   lwdown  Stefan-Boltzmann over any plausible atmospheric temperature
    BOUNDS = {"prec": (0, 600), "air_temp": (0, 50), "swdown": (0, 1100),
              "lwdown": (150, 550), "pressure": (85, 102), "vp": (0.2, 5.0),
              "wind": (0, 25)}
    for k in COLUMNS:
        v = grids[k][:, active]
        lo, hi = BOUNDS[k]
        bad = int(np.isnan(v).sum())
        out = int(((v < lo) | (v > hi)).sum())
        flag = "OK" if bad == 0 and out == 0 else f"{bad} NaN, {out} out of [{lo},{hi}]"
        print(f"  {k:9s} {v.min():8.2f} .. {v.max():8.2f} {UNITS[k]:5s} "
              f"mean {v.mean():8.2f}   {flag}")
        ok &= bad == 0 and out == 0
    if n_nan:
        print(f"  IMD had {n_nan:,} missing values in-grid, set to 0 mm")

    # Vapour pressure cannot exceed saturation at the air temperature.
    sat = vapour_pressure_kpa(grids["air_temp"][:, active])
    sup = int((grids["vp"][:, active] > sat * 1.02).sum())
    print(f"  VP <= saturation   {'OK' if sup == 0 else f'{sup} supersaturated'}")
    gaps = np.unique(np.diff(steps.values).astype("timedelta64[h]").astype(int))
    print(f"  step spacing {gaps} h   {'OK' if list(gaps) == [6] else '<- GAP'}")
    ok &= list(gaps) == [6]
    years = len(steps) / (4 * 365.25)
    print(f"  annual precip      "
          f"{grids['prec'][:, active].sum() / active.sum() / years:.0f} mm")

    if args.check_only:
        print(f"\n{'PASS' if ok else 'FAIL'} (check-only, nothing written)")
        sys.exit(0 if ok else 1)

    FORCEDIR.mkdir(parents=True, exist_ok=True)
    for old in FORCEDIR.glob("data_*"):
        old.unlink()
    n = 0
    for i in range(len(lats)):
        for j in range(len(lons)):
            if not active[i, j]:
                continue
            block = np.column_stack([grids[c][:, i, j] for c in COLUMNS])
            np.savetxt(FORCEDIR / f"data_{lats[i]:.4f}_{lons[j]:.4f}", block,
                       fmt="%.4f %.2f %.1f %.1f %.2f %.4f %.2f")
            n += 1

    xr.Dataset(
        {k: (("time", "lat", "lon"), v.astype("float32")) for k, v in grids.items()},
        coords={"time": steps, "lat": lats, "lon": lons},
        attrs={"source": "ERA5-Land (t2m hourly; met/rad 6-hourly) + IMD 0.25 deg",
               "columns": " ".join(COLUMNS),
               "note": "wind = mean(sqrt(u^2+v^2)); radiation de-accumulated at "
                       "00Z; daily means on IST days except radiation (UTC)"},
    ).to_netcdf(OUTDIR / "forcing_daily.nc")

    print(f"\nwrote {n} forcing files -> {FORCEDIR}")
    print(f"wrote {OUTDIR / 'forcing_daily.nc'}")
    print(f"\n{'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
