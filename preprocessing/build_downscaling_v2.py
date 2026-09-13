"""Assemble the v2 downscaling dataset from the Dong-spec S2S archive.

Turns data/processed/s2s/*.npz (28 init dates, each year x lead x member x var
x 7 x 7) into the sample-aligned arrays the field downscaler trains on, with the
IMD 0.25 deg rainfall field at the matching valid time as the target.

WHAT CHANGES FROM v1 (data/processed/coarse_grid.npz):

    coarse grid   5x5, 10 vars      ->  7x7, 26 vars
    leads         1-17              ->  1-30
    members       control only      ->  control + 10 perturbed
    samples       5,797             ->  9,240 (308 inits x 30 leads)

THE MEMBER DIMENSION.  Members are not flattened into extra samples here.  The
ensemble is summarised into two things the model can actually use:

    mean   the ensemble's best estimate of each field
    sd     how much the members disagree -- a forecast-confidence signal that
           simply did not exist in v1, where there was one member

Training on all 11 members as separate samples (11x the rows, same target) is
the obvious alternative and is left available via --members all; it is an
ablation, not the default, because it multiplies training cost elevenfold to
learn from fields whose target is identical.

ACCUMULATED FIELDS ARE DIFFERENCED.  tp, cp, ssrd, sshf and slhf accumulate from
initialisation -- lead 30's tp is the whole month's rain, not that day's.  They
are differenced along the lead axis, per member, before any averaging.  Missing
this is the classic way to get a downscaler that looks brilliant and is useless.

Run:  python preprocessing/build_downscaling_v2.py
Out:  data/processed/downscaling_v2.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

OUT = C.PROCESSED / "downscaling_v2.npz"
SEC_PER_DAY = C.SECONDS_PER_DAY


def fine_grid():
    lats = np.arange(C.FINE_LAT_MIN, C.FINE_LAT_MAX + C.FINE_RES / 2, C.FINE_RES)
    lons = np.arange(C.FINE_LON_MIN, C.FINE_LON_MAX + C.FINE_RES / 2, C.FINE_RES)
    return lats, lons


def load_archive(members: str, suffix: str = ""):
    """-> coarse (N, C, 7, 7), init dates, lead days, variable names.

    N = init dates x hindcast years x leads, ordered so that `init` and `lead`
    line up with the first axis.
    """
    # Glob the requested year range ONLY.  The archive now holds two ranges with
    # different year axes; a bare "*.npz" would concatenate them and silently
    # mislabel every initialisation in one of the two.
    files = [f for f in sorted(C.S2S_V2_DIR.glob("*.npz"))
             if (f.stem.endswith(suffix) if suffix else "-" not in f.stem)]
    if not files:
        sys.exit(f"no archive in {C.S2S_V2_DIR}; run build_s2s_archive.py first")

    acc_idx = None
    inits, leads, blocks = [], [], []
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        f = z["forcing"]                       # (year, lead, member, var, 7, 7)
        V = [str(x) for x in z["variables"]]
        if acc_idx is None:
            acc_idx = [V.index(v) for v in C.V2_ACCUMULATED_VARS]
            names = V

        # Undo the accumulation, per member, before collapsing anything.
        f = f.copy()
        a = f[:, :, :, acc_idx]
        a[:, 1:] = np.diff(a, axis=1)          # lead 1 already IS the day-1 total
        f[:, :, :, acc_idx] = a

        month, day = fp.stem.split("_")[:2]
        n_year, n_lead = f.shape[0], f.shape[1]
        # Prefer the year axis stamped into the file; fall back to config for
        # archives built before that field existed.
        yrs = ([int(y) for y in z["years"]] if "years" in z.files
               else list(range(C.YEAR_MIN, C.YEAR_MIN + n_year)))
        for yi in range(n_year):
            year = yrs[yi]
            init = pd.Timestamp(f"{year}-{month}-{day}")
            for li in range(n_lead):
                inits.append(init)
                leads.append(li + 1)
        blocks.append(f)

    # (n_date, year, lead, member, var, 7, 7) -> (N, member, var, 7, 7)
    stack = np.concatenate([b.reshape(-1, *b.shape[2:]) for b in blocks], axis=0)
    inits = pd.DatetimeIndex(inits)
    leads = np.array(leads, np.int16)

    if members == "mean":
        mean = stack.mean(axis=1)
        sd = stack.std(axis=1)
        coarse = np.concatenate([mean, sd], axis=1).astype(np.float32)
        names = names + [f"{v}_sd" for v in names]
        member_id = np.zeros(len(coarse), np.int8)
    else:                                       # every member as its own sample
        n, m = stack.shape[0], stack.shape[1]
        coarse = stack.reshape(n * m, *stack.shape[2:]).astype(np.float32)
        inits = inits.repeat(m)
        leads = leads.repeat(m)
        member_id = np.tile(np.arange(m, dtype=np.int8), n)

    # J/m^2 accumulated over a day -> mean W/m^2, so the flux channels sit in a
    # range comparable to the others rather than 1e5 times larger.
    for v in ("ssrd", "sshf", "slhf"):
        for j, nm in enumerate(names):
            if nm in (v, f"{v}_sd"):
                coarse[:, j] /= SEC_PER_DAY
    return coarse, inits, leads, names, member_id


def load_target(valid_times):
    """IMD 0.25 deg rainfall at each valid time -> (N, 25, 25) plus masks."""
    import xarray as xr

    flats, flons = fine_grid()
    years = set(pd.DatetimeIndex(valid_times).year)      # from the data, not config
    rename = {"LATITUDE": "lat", "LONGITUDE": "lon", "TIME": "time", "RAINFALL": "rain"}

    parts = []
    for f in sorted(C.IMD_NC_DIR.glob("*.nc")):
        with xr.open_dataset(f) as ds:
            ds = ds.rename({k: v for k, v in rename.items()
                            if k in ds.variables or k in ds.dims})
            if not pd.to_datetime(ds["time"].values).year.isin(years).any():
                continue
            parts.append(ds["rain"].sel(lat=flats, lon=flons, method="nearest").load())

    rain = xr.concat(parts, dim="time").sortby("time")
    rain = rain.where(~(rain < 0.0))
    # reindex, not sel: a valid time IMD does not cover becomes NaN, never zero
    sel = rain.reindex(time=valid_times)
    target = sel.transpose("time", "lat", "lon").values.astype(np.float32)

    land = np.isfinite(target).any(axis=0)
    mask = np.isfinite(target) & land[None]
    catchment = (
        (flats[:, None] >= C.CATCHMENT_LAT_MIN) & (flats[:, None] <= C.CATCHMENT_LAT_MAX)
        & (flons[None, :] >= C.CATCHMENT_LON_MIN) & (flons[None, :] <= C.CATCHMENT_LON_MAX)
    )
    return np.nan_to_num(target), mask, land, catchment & land, flats, flons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", choices=["mean", "all"], default="mean",
                    help="'mean' = ensemble mean + spread channels (default); "
                         "'all' = each member as its own training sample")
    ap.add_argument("--hyears", default=None,
                    help="which archive year range to build, e.g. 2015-2022.  "
                         "Default builds the original 2004-2014 files.")
    args = ap.parse_args()
    suffix = f"_{args.hyears}" if args.hyears else ""

    print(f"assembling v2 downscaling dataset (members={args.members})")
    coarse, inits, leads, names, member_id = load_archive(args.members, suffix)
    valid = inits + pd.to_timedelta(leads, unit="D")
    print(f"  coarse {coarse.shape}  ({len(names)} channels)")
    print(f"  inits {inits.min().date()}..{inits.max().date()},"
          f" {inits.normalize().nunique()} unique")
    print(f"  leads {leads.min()}..{leads.max()}")

    target, mask, land, catchment, flats, flons = load_target(valid)
    print(f"  target {target.shape}, {mask.mean()*100:.1f}% of cells usable")
    print(f"  catchment cells: {int(catchment.sum())} of {catchment.size}")

    out_path = OUT.with_name(OUT.stem + suffix + OUT.suffix)
    np.savez_compressed(
        out_path, coarse=coarse, channels=np.array(names),
        init=inits.values, lead=leads, member=member_id,
        valid=valid.values, target=target, mask=mask, land=land,
        catchment=catchment, fine_lats=flats, fine_lons=flons,
        init_year=inits.year.values.astype(np.int16),
    )
    print(f"\nwrote {out_path}  ({out_path.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
