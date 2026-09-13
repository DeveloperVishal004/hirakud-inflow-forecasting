"""Read the Dong-spec S2S GRIBs into gridded arrays the CNN can train on.

Replaces the old `data/raw/source/ecmwf_s2s_reforecast_final.csv` route, which
carried 16 columns on a 5x5 grid to lead 17 with no ensemble dimension.  What
`download_s2s_dong.py` fetches instead:

    grid     7 x 7 coarse cells (16.5-25.5 N, 79.5-88.5 E at 1.5 deg)
             -> every 0.25 deg fine cell now gets a genuine 3x3 patch.
                The old 5x5 domain clamped 48 % of patches against an edge.
    leads    1..30 days          (was 1..17)
    members  control + 10 perturbed = 11   (was control only)
    vars     26 = 11 surface + 5 upper-air x 3 pressure levels   (was 16)

LAYOUT.  One .npz per initialisation date, because the whole archive is ~520 MB
and holding it in memory while parsing GRIB is not worth it:

    data/processed/s2s/<MM>_<DD>.npz
        forcing  float32 (year, lead, member, var, lat, lon)
                          11     30     11     26   7    7
        years, leads, members, variables, lats, lons

A GRIB QUIRK, because it cost an afternoon.  cfgrib cannot open these files in
one pass: `2t` and `tcc` sit on different level types AND arrive with a single
period step ("0-24") rather than the 30 instantaneous steps everything else
has, so a merged open raises DatasetBuildError and silently drops both.  They
are therefore read in separate passes and broadcast; see SINGLE_STEP_VARS.

Run:  python preprocessing/build_s2s_archive.py            # all dates on disk
      python preprocessing/build_s2s_archive.py --check    # report, build nothing
Out:  data/processed/s2s/*.npz
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

warnings.filterwarnings("ignore")

SRC = C.RAW / "s2s"
OUT = C.PROCESSED / "s2s"

# cfgrib shortName -> the name used downstream.  ECDS returns `ssrd`, while the
# old archive column was `ssr`; keep the ECDS name and let config follow.
SFC_VARS = ["tp", "cp", "t2m", "u10", "v10", "msl", "tcc",
            "ssrd", "sshf", "slhf", "orog"]
PL_VARS = ["u", "v", "q", "t", "gh"]
PL_LEVELS = [200, 500, 850]
VARIABLES = SFC_VARS + [f"{v}{lev}" for v in PL_VARS for lev in PL_LEVELS]

# Delivered on one 24 h period only, not per lead.  Held constant across leads
# so the array stays rectangular.  Keyed by the GRIB shortName, which for 2 m
# temperature is "2t" -- cfgrib only renames it to t2m once the variable is
# built, so filtering on "t2m" silently matches nothing.
SINGLE_STEP_VARS = {"2t": "t2m", "tcc": "tcc"}
# Time-invariant by nature -- same field at every lead and member.
STATIC_VARS = ["orog"]

N_LEAD = 30
MEMBERS = list(range(11))          # 0 = control forecast, 1..10 = perturbed


def open_grib(path: Path, **keys):
    kw = {"indexpath": ""}
    if keys:
        kw["filter_by_keys"] = keys
    return xr.open_dataset(path, engine="cfgrib", backend_kwargs=kw)


def read_surface(path: Path) -> dict[str, np.ndarray]:
    """-> {var: (year, lead, member, lat, lon)}; single-step vars broadcast."""
    out = {}
    main = open_grib(path)                       # everything with 30 real steps
    for v in main.data_vars:
        out[str(v)] = to_array(main[v])
    for short, name in SINGLE_STEP_VARS.items():  # separate pass, see docstring
        try:
            ds = open_grib(path, shortName=short)
        except Exception:
            continue
        if not ds.data_vars:
            continue
        a = to_array(ds[list(ds.data_vars)[0]], has_step=False)
        out[name] = np.repeat(a, N_LEAD, axis=1)
    return out


def read_surface_period(path: Path) -> dict[str, np.ndarray]:
    """The "sfcp" file: 2t and tcc as 30 daily windows, genuinely lead-varying.

    Read one shortName at a time for the same reason as read_surface -- the two
    sit on different level types and a merged open drops both.
    """
    out = {}
    for short, name in SINGLE_STEP_VARS.items():
        try:
            ds = open_grib(path, shortName=short)
        except Exception:
            continue
        if not ds.data_vars:
            continue
        out[name] = to_array(ds[list(ds.data_vars)[0]])
    return out


def read_pressure(path: Path) -> dict[str, np.ndarray]:
    ds = open_grib(path)
    out = {}
    for v in ds.data_vars:
        for lev in PL_LEVELS:
            out[f"{v}{lev}"] = to_array(ds[v].sel(isobaricInhPa=lev))
    return out


def to_array(da: xr.DataArray, has_step: bool = True) -> np.ndarray:
    """-> (year, lead, member, lat, lon), inserting whatever axes are missing."""
    if "number" not in da.dims:
        da = da.expand_dims("number")
    if has_step:
        da = da.transpose("time", "step", "number", "latitude", "longitude")
    else:
        da = da.expand_dims("step").transpose("time", "step", "number",
                                              "latitude", "longitude")
    return np.asarray(da.values, dtype=np.float32)


def build_date(month: str, day: str, check: bool = False, suffix: str = ""):
    # "sfcp" sorts after "sfc", so its genuine per-lead 2t/tcc overwrite the
    # broadcast single-step values whenever the period file has been fetched.
    files = {(t, k): SRC / f"{t}_{k}_{month}_{day}{suffix}.grib"
             for t in ("cf", "pf") for k in ("sfc", "sfcp", "pl")}
    present = {n: p for n, p in files.items() if p.exists() and p.stat().st_size > 0}
    if not present:
        return None, "no files"
    if check:
        return None, ", ".join(f"{t}_{k}" for t, k in sorted(present))

    # Read whatever is on disk; control lands in member 0, perturbed in 1..10.
    data = {}
    for (t, k), path in sorted(present.items()):
        got = {"sfc": read_surface, "sfcp": read_surface_period,
               "pl": read_pressure}[k](path)
        for v, arr in got.items():
            slot = data.setdefault(v, {})
            slot["cf" if t == "cf" else "pf"] = arr

    n_year = next(iter(next(iter(data.values())).values())).shape[0]
    # ECMWF ships the static fields once, in the control file only -- the terrain
    # is the same for every member by definition.  Without this, orog is NaN for
    # members 1-10 (3.5 % of the whole array) and any NaN-intolerant model drops
    # ten elevenths of the ensemble.
    for v in STATIC_VARS:
        if v in data and "cf" in data[v] and "pf" not in data[v]:
            data[v]["pf"] = np.repeat(data[v]["cf"], len(MEMBERS) - 1, axis=2)
    forcing = np.full((n_year, N_LEAD, len(MEMBERS), len(VARIABLES), 7, 7),
                      np.nan, np.float32)
    filled = []
    for vi, v in enumerate(VARIABLES):
        if v not in data:
            continue
        filled.append(v)
        if "cf" in data[v]:
            forcing[:, :, 0:1, vi] = data[v]["cf"]
        if "pf" in data[v]:
            forcing[:, :, 1:, vi] = data[v]["pf"]
    return forcing, filled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report what is on disk and readable, write nothing")
    ap.add_argument("--hyears", default=None,
                    help="hindcast year range of the GRIBs to read, e.g. 2015-2022.  "
                         "Default reads the unsuffixed 2004-2014 files.  The year axis "
                         "is stamped from this range, not from config, so the two "
                         "archives can be concatenated without guessing which is which.")
    args = ap.parse_args()

    suffix, year0 = "", C.YEAR_MIN
    if args.hyears:
        lo, hi = (int(x) for x in args.hyears.split("-"))
        suffix, year0 = f"_{lo}-{hi}", lo

    OUT.mkdir(parents=True, exist_ok=True)
    dates = [(m, d) for m, days in
             {"06": ["03", "10", "17", "24"],
              "07": ["01", "04", "08", "12", "15", "19", "22", "26", "29"],
              "08": ["01", "05", "08", "12", "15", "19", "22", "26", "29"],
              "09": ["02", "05", "09", "12", "16", "19", "23", "26", "30"]}.items()
             for d in days]

    print(f"{len(VARIABLES)} variables, {N_LEAD} leads, {len(MEMBERS)} members, "
          f"7x7 grid, {len(dates)} scheduled inits")
    print(f"src {SRC}  ->  out {OUT}\n")

    ok = skip = 0
    for m, d in dates:
        res, info = build_date(m, d, check=args.check, suffix=suffix)
        if res is None:
            print(f"  {m}-{d}  SKIP  ({info})")
            skip += 1
            continue
        # A variable whose lead axis is constant carries no forecast information;
        # flag it rather than let a downstream model appear to learn from it.
        lead_varying = np.array(
            [bool(np.nanstd(res[:, :, :, i], axis=1).max() > 0) for i in range(len(VARIABLES))])
        np.savez_compressed(OUT / f"{m}_{d}{suffix}.npz", forcing=res,
                            variables=np.array(VARIABLES),
                            leads=np.arange(1, N_LEAD + 1),
                            members=np.array(MEMBERS),
                            years=np.arange(year0, year0 + res.shape[0]),
                            lead_varying=lead_varying)
        miss = [v for v in VARIABLES if v not in info]
        print(f"  {m}-{d}  ok  {res.shape}  filled {len(info)}/{len(VARIABLES)}"
              + (f"  missing {miss}" if miss else ""))
        ok += 1
    print(f"\n{ok} built, {skip} skipped")


if __name__ == "__main__":
    main()
