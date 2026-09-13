"""Saxton & Rawls (2006) pedotransfer -> the VIC 5 soil parameter file.

SoilGrids gives texture, bulk density and organic carbon.  VIC needs hydraulic
properties: saturated conductivity, the Brooks-Corey exponent, bubbling
pressure, field capacity and wilting point.  Saxton & Rawls (2006, SSSAJ
70:1569-1578) is the standard route between them, fitted to ~2000 USDA samples
and used with a density-adjustment step so that MEASURED bulk density overrides
the value their texture-only equations would imply.

That density adjustment matters here.  The catchment is Vertisol -- 38 % clay,
bulk density 1.22-1.77 g/cm3 -- and texture alone would misstate porosity in
exactly the soils whose shrink-swell behaviour dominates the runoff response.

Which parameters are MEASURED and which are CALIBRATED is the important
distinction, and is kept explicit below:

  measured/derived  Ksat, expt, bubble, Wcr_FRACT, Wpwp_FRACT, resid_moist,
                    bulk_density, quartz, depth, elev, annual_prec
  calibrated        b_infilt, Ds, Dsmax, Ws, c   <- the 5 VIC parameters
                    Ambika et al. (2025) calibrate, left at defaults here

Anything calibrated is a starting value, not a result.  The file is runnable as
written; it is not tuned.

Run:  python hydrology/vic/build_soil_params.py
Out:  data/processed/vic/soil_param.txt  (+ soil_params.npz for inspection)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs import config as C

OUTDIR = C.PROCESSED / "vic"
LAYERS = ["layer1_0-15cm", "layer2_15-60cm", "layer3_60-200cm"]
DEPTH_M = [0.15, 0.45, 1.40]

SOIL_DENSITY = 2650.0     # kg/m3, mineral particle density
OM_FROM_SOC = 1.724       # van Bemmelen: organic matter = 1.724 x organic carbon
DAMPING_DEPTH_M = 4.0
OFF_GMT = 5.5             # India Standard Time; VIC uses it to phase the diurnal cycle

# --- calibration starting values (NOT results) -------------------------------
B_INFILT = 0.2            # variable-infiltration curve shape
DS = 0.001                # fraction of Dsmax at which non-linear baseflow starts
WS = 0.9                  # fraction of max moisture at which it starts
C_EXP = 2.0               # baseflow curve exponent


def saxton_rawls(sand_pct, clay_pct, om_pct, bulk_density):
    """Hydraulic properties from texture, organic matter and bulk density.

    sand/clay/om in percent, bulk_density in g/cm3.  Returns SI-ish units:
    volumetric moisture as fractions, Ksat in mm/day, bubbling pressure in cm.
    """
    S = sand_pct / 100.0
    Cl = clay_pct / 100.0
    OM = om_pct

    # --- wilting point (1500 kPa)
    t1500t = (-0.024 * S + 0.487 * Cl + 0.006 * OM + 0.005 * (S * OM)
              - 0.013 * (Cl * OM) + 0.068 * (S * Cl) + 0.031)
    t1500 = t1500t + (0.14 * t1500t - 0.02)

    # --- field capacity (33 kPa)
    t33t = (-0.251 * S + 0.195 * Cl + 0.011 * OM + 0.006 * (S * OM)
            - 0.027 * (Cl * OM) + 0.452 * (S * Cl) + 0.299)
    t33 = t33t + (1.283 * t33t ** 2 - 0.374 * t33t - 0.015)

    # --- air-entry to 33 kPa moisture
    ts33t = (0.278 * S + 0.034 * Cl + 0.022 * OM - 0.018 * (S * OM)
             - 0.027 * (Cl * OM) - 0.584 * (S * Cl) + 0.078)
    ts33 = ts33t + (0.636 * ts33t - 0.107)

    # --- bubbling pressure (kPa, reported negative by S&R; magnitude used)
    pet = (-21.67 * S - 27.93 * Cl - 81.97 * ts33 + 71.12 * (S * ts33)
           + 8.29 * (Cl * ts33) + 14.05 * (S * Cl) + 27.16)
    pe = pet + (0.02 * pet ** 2 - 0.113 * pet - 0.70)

    # --- saturation from texture, then OVERRIDE with measured bulk density.
    # Without this step the Vertisols' porosity comes from texture alone, which
    # is precisely where a shrink-swell soil departs from the fitted sample.
    tS = t33 + ts33 - 0.097 * S + 0.043
    tS_df = 1.0 - (bulk_density / (SOIL_DENSITY / 1000.0))
    t33_df = t33 - 0.2 * (tS - tS_df)

    # --- enforce the physical ordering theta_1500 < theta_33 < theta_S.
    #
    # Saxton & Rawls was fitted to USDA samples whose bulk density is close to
    # the value their texture equations imply.  These Vertisols are not: the
    # measured density reaches 1.77 g/cm3 where texture implies 1.33-1.50, so
    # the density adjustment drives porosity BELOW field capacity in a
    # substantial minority of cells -- field capacity above total pore space,
    # which cannot happen.  This is the equations being extrapolated past their
    # calibration envelope, not a coding error, so it is clamped explicitly and
    # counted rather than silently absorbed by a downstream np.clip.
    #
    # MIN_AIR is the air-filled porosity retained at saturation.  0.10 of pore
    # space keeps Ksat in the 1-10 mm/day range these soils are reported at,
    # instead of the ~0 that an unclamped negative difference produces.
    MIN_AIR = 0.10
    n_clamped = int((t33_df > (1.0 - MIN_AIR) * tS_df).sum())
    t33_df = np.minimum(t33_df, (1.0 - MIN_AIR) * tS_df)

    # --- Brooks-Corey slope from the TEXTURE-based retention pair, not the
    # density-adjusted one.  B describes the pore-SIZE DISTRIBUTION, which is a
    # property of texture; compacting a soil reduces total pore space without
    # reshaping that distribution.  Using clamped values here would be circular
    # -- the clamp compresses log(theta_33) - log(theta_1500) toward zero and
    # sends B, and hence expt, to physically absurd values (>70 against a
    # normal range of 3-30).
    B = (np.log(1500.0) - np.log(33.0)) / (np.log(np.clip(t33, 1e-4, 0.99))
                                           - np.log(np.clip(t1500, 1e-4, 0.99)))
    lam = 1.0 / B

    # Wilting point only has to sit below field capacity for the OUTPUT states.
    t1500 = np.minimum(t1500, 0.95 * t33_df)
    t33_df = np.clip(t33_df, 1e-4, 0.99)
    t1500 = np.clip(t1500, 1e-4, 0.99)
    ksat_mm_h = 1930.0 * np.clip(tS_df - t33_df, 1e-6, None) ** (3.0 - lam)

    return {
        "theta_s": tS_df,
        "theta_33": t33_df,
        "theta_1500": t1500,
        "ksat_mm_day": ksat_mm_h * 24.0,
        "expt": 3.0 + 2.0 * B,           # VIC's Brooks-Corey exponent
        "bubble_cm": np.abs(pe) * 10.1972,   # kPa -> cm of water
        "lambda": lam,
        "n_clamped": n_clamped,
    }


# The climatology window for the soil parameters, stated EXPLICITLY rather than
# inherited from C.YEAR_MAX.  C.YEAR_MAX is still 2014 -- it belongs to the v1
# archive and is deliberately frozen there -- while the record now runs to 2022.
# The soil parameters VIC is currently calibrated against were built on
# 2004-2014, so that is what is pinned here: changing it silently would move the
# soil parameters out from under the existing calibration without re-running it.
# To rebuild on the full record, change these two lines AND recalibrate VIC.
CLIM_YEAR_MIN = 2004
CLIM_YEAR_MAX = 2014


def annual_precip_mm():
    """Mean annual IMD rainfall per cell -- a VIC input, not a diagnostic."""
    import glob
    import xarray as xr

    files = sorted(glob.glob(str(C.IMD_NC_DIR / "RF25_ind*.nc")))
    ds = xr.concat([xr.open_dataset(f) for f in files], dim="TIME").sortby("TIME")
    ds = ds.sel(TIME=slice(f"{CLIM_YEAR_MIN}-01-01", f"{CLIM_YEAR_MAX}-12-31"))
    r = ds.sel(LATITUDE=slice(C.CATCHMENT_LAT_MIN, C.CATCHMENT_LAT_MAX),
               LONGITUDE=slice(C.CATCHMENT_LON_MIN, C.CATCHMENT_LON_MAX)).RAINFALL
    n_years = CLIM_YEAR_MAX - CLIM_YEAR_MIN + 1
    return (r.sum("TIME") / n_years).values


def mean_air_temp():
    """Annual mean 2 m temperature from whatever ERA5 months are on disk."""
    import xarray as xr

    era = C.RAW / "vic" / "forcing" / "era5"
    files = sorted(era.rglob("*.nc"))
    if not files:
        return None, 0
    tot, n = 0.0, 0
    for f in files:
        d = xr.open_dataset(f)
        if "t2m" not in d:
            continue
        tot += float(d.t2m.mean()) - 273.15
        n += 1
    return (tot / n if n else None), n


def main() -> None:
    soil = np.load(C.PROCESSED / "soilgrids.npz", allow_pickle=True)
    dem = np.load(C.PROCESSED / "dem_vic.npz", allow_pickle=True)
    basin = np.load(C.PROCESSED / "basin.npz", allow_pickle=True)
    if "soc_layer1_0-15cm" not in soil:
        sys.exit("soilgrids.npz has no organic carbon -- re-run "
                 "preprocessing/build_soilgrids.py")

    lats, lons = soil["lats"], soil["lons"]
    assert np.allclose(dem["lats"], lats) and np.allclose(basin["lats"], lats)
    frac = basin["fraction"]
    ny, nx = len(lats), len(lons)
    print(f"VIC soil parameters: {ny}x{nx} cells, "
          f"{int((frac > 0).sum())} active (in basin)")

    props = []
    for k, layer in enumerate(LAYERS):
        p = saxton_rawls(soil[f"sand_{layer}"], soil[f"clay_{layer}"],
                         soil[f"soc_{layer}"] * OM_FROM_SOC,
                         soil[f"bdod_{layer}"])
        props.append(p)
        print(f"  {layer:16s} Ksat {p['ksat_mm_day'].mean():8.1f} mm/d   "
              f"expt {p['expt'].mean():5.2f}   bubble {p['bubble_cm'].mean():6.1f} cm   "
              f"theta_s {p['theta_s'].mean():.3f}"
              + (f"   [{p['n_clamped']} cells clamped]" if p["n_clamped"] else ""))

    aprec = annual_precip_mm()
    avg_t, n_months = mean_air_temp()
    if avg_t is None:
        avg_t = 26.0
        print("\n  NOTE: no ERA5 on disk yet; avg_T set to 26.0 C as a "
              "placeholder. Re-run once the download finishes.")
    else:
        print(f"\n  avg_T from {n_months} ERA5 month(s): {avg_t:.1f} C"
              + ("   <- PARTIAL YEAR, seasonally biased; re-run when complete"
                 if n_months < 12 else ""))

    # ---- assemble the 53-column classic-driver rows
    rows = []
    gid = 0
    for i in range(ny):
        for j in range(nx):
            gid += 1
            run = 1 if frac[i, j] > 0 else 0
            ks = [float(props[k]["ksat_mm_day"][i, j]) for k in range(3)]
            expt = [float(props[k]["expt"][i, j]) for k in range(3)]
            bub = [float(props[k]["bubble_cm"][i, j]) for k in range(3)]
            ts = [float(props[k]["theta_s"][i, j]) for k in range(3)]
            t33 = [float(props[k]["theta_33"][i, j]) for k in range(3)]
            t15 = [float(props[k]["theta_1500"][i, j]) for k in range(3)]
            bd = [float(soil[f"bdod_{LAYERS[k]}"][i, j]) * 1000.0 for k in range(3)]
            quartz = [float(soil[f"sand_{LAYERS[k]}"][i, j]) / 100.0 for k in range(3)]

            # VIC expresses these as fractions of MAXIMUM (saturated) moisture.
            # Wcr is conventionally "~70 % of field capacity", but taken
            # literally as 0.7*theta_33 it falls BELOW the wilting point in
            # heavy clay -- theta_1500 is simply too large a share of
            # theta_33 here -- which would let VIC restrict transpiration at a
            # moisture the plant cannot extract at all.  Taking 70 % of the
            # AVAILABLE water above wilting point is equivalent for sandy soils
            # (where theta_1500 is small) and stays physical for Vertisols.
            wcr = [(t15[k] + 0.7 * (t33[k] - t15[k])) / ts[k] for k in range(3)]
            wpwp = [t15[k] / ts[k] for k in range(3)]
            resid = [0.0, 0.0, 0.0]
            init = [t33[k] * DEPTH_M[k] * 1000.0 for k in range(3)]  # mm
            dsmax = ks[2]     # bottom-layer Ksat; a calibration starting point

            rows.append(
                [run, gid, float(lats[i]), float(lons[j]),
                 B_INFILT, DS, dsmax, WS, C_EXP]
                + expt + ks + [-99.0, -99.0, -99.0] + init
                + [float(dem["elev_mean"][i, j])] + DEPTH_M
                + [avg_t, DAMPING_DEPTH_M] + bub + quartz + bd
                + [SOIL_DENSITY] * 3 + [OFF_GMT] + wcr + wpwp
                + [0.001, 0.0005, float(aprec[i, j])] + resid + [0])

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "soil_param.txt"
    with open(out, "w") as f:
        for r in rows:
            f.write(" ".join(f"{v:g}" if not isinstance(v, int) else str(v)
                             for v in r) + "\n")

    # ---- checks
    print("\n=== checks ===")
    ok = True
    ks_all = np.concatenate([p["ksat_mm_day"].ravel() for p in props])
    ex_all = np.concatenate([p["expt"].ravel() for p in props])
    ts_all = np.concatenate([p["theta_s"].ravel() for p in props])
    checks = [
        ("Ksat mm/day", ks_all, 1.0, 5000.0),
        ("expt", ex_all, 3.0, 30.0),
        ("porosity", ts_all, 0.25, 0.65),
    ]
    for name, arr, lo, hi in checks:
        bad = int(((arr < lo) | (arr > hi)).sum())
        print(f"  {name:14s} {arr.min():8.2f} - {arr.max():8.2f}   "
              f"{'OK' if bad == 0 else f'{bad} outside [{lo}, {hi}]'}")
        ok &= bad == 0

    # Must mirror the formula used to write the rows, or the check is vacuous.
    for k in range(3):
        t33k, t15k, tsk = (props[k]["theta_33"], props[k]["theta_1500"],
                           props[k]["theta_s"])
        wc = (t15k + 0.7 * (t33k - t15k)) / tsk
        wp = t15k / tsk
        bad = int((wp >= wc).sum() + (wc >= 1.0).sum())
        print(f"  layer {k + 1} Wpwp<Wcr<1 {'OK' if bad == 0 else f'{bad} cells violate'}")
        ok &= bad == 0

    ncol = len(rows[0])
    print(f"  columns per row  {ncol}  {'OK' if ncol == 53 else '<- expected 53'}")
    ok &= ncol == 53
    print(f"  rows             {len(rows)}  ({sum(r[0] for r in rows)} active)")

    np.savez_compressed(OUTDIR / "soil_params.npz", lats=lats, lons=lons,
                        **{f"{k}_L{n+1}": props[n][k] for n in range(3)
                           for k in ("ksat_mm_day", "expt", "bubble_cm",
                                     "theta_s", "theta_33", "theta_1500")},
                        annual_prec=aprec, run_cell=(frac > 0).astype(int))
    print(f"\n{'PASS' if ok else 'FAIL'} -- wrote {out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
