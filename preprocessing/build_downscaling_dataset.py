"""Build the 1.5 deg -> 0.25 deg downscaling dataset, Dong et al. (2025) sect. 3.2.1.

The previous pipeline predicted IMD sampled *at* the 25 coarse cells, so input
and output shared a grid and no downscaling took place.  Here the target is the
native 0.25 deg IMD field (25 x 25 = 625 cells over the domain) and each fine
cell is predicted from a 3 x 3 patch of coarse cells centred on it -- the patch
size the paper found best among 1x1, 3x3, 5x5 and 7x7.

Patches are not materialised.  341 inits x 17 leads x 625 cells x 19 vars x 3 x 3
would be ~2.5 GB; the coarse grid is stored once (~11 MB) and `PatchDataset`
gathers patches at access time using a precomputed index map.

Run:  python preprocessing/build_downscaling_dataset.py
Out:  data/processed/coarse_grid.npz      (always)
      data/processed/fine_target.npz      (requires data/raw/imd/*.nc)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

FEATURES = C.INSTANT_VARS + C.ACCUMULATED_VARS + C.STATIC_VARS


def fine_grid() -> tuple[np.ndarray, np.ndarray]:
    """The 0.25 deg IMD grid the model predicts onto."""
    lats = np.arange(C.FINE_LAT_MIN, C.FINE_LAT_MAX + C.FINE_RES / 2, C.FINE_RES)
    lons = np.arange(C.FINE_LON_MIN, C.FINE_LON_MAX + C.FINE_RES / 2, C.FINE_RES)
    return lats, lons


def patch_index_map() -> np.ndarray:
    """For every fine cell, the coarse (row, col) indices of its 3 x 3 patch.

    Returns shape (n_fine_lat, n_fine_lon, PATCH, PATCH, 2).

    The coarse domain is only 5 x 5, so a patch centred on an edge coarse cell
    runs off the grid.  Indices are clamped to the boundary, which repeats the
    edge row/column.  That is a real limitation of the current download, not a
    modelling choice: re-requesting ECMWF over a domain one 1.5 deg ring larger
    (16.5-25.5 N, 79.5-88.5 E) would give every fine cell a genuine 3 x 3
    neighbourhood.  `patch_is_padded` records where clamping happened.
    """
    flats, flons = fine_grid()
    clats = np.array(C.COARSE_LATS)
    clons = np.array(C.COARSE_LONS)
    half = C.PATCH // 2

    idx = np.zeros((len(flats), len(flons), C.PATCH, C.PATCH, 2), dtype=np.int16)
    padded = np.zeros((len(flats), len(flons)), dtype=bool)

    for a, flat in enumerate(flats):
        ci = int(np.abs(clats - flat).argmin())
        for b, flon in enumerate(flons):
            cj = int(np.abs(clons - flon).argmin())
            rows = np.clip(np.arange(ci - half, ci + half + 1), 0, len(clats) - 1)
            cols = np.clip(np.arange(cj - half, cj + half + 1), 0, len(clons) - 1)
            idx[a, b, :, :, 0] = rows[:, None]
            idx[a, b, :, :, 1] = cols[None, :]
            padded[a, b] = len(set(rows)) < C.PATCH or len(set(cols)) < C.PATCH

    return idx, padded


def build_coarse_grid() -> None:
    """Reshape the cleaned forcing into (sample, feature, 5, 5) arrays."""
    df = pd.read_parquet(C.PROCESSED / "forcing_daily.parquet")

    lat_i = {v: i for i, v in enumerate(C.COARSE_LATS)}
    lon_i = {v: j for j, v in enumerate(C.COARSE_LONS)}
    df["li"] = df["latitude"].map(lat_i)
    df["lj"] = df["longitude"].map(lon_i)

    keys = df[["time", "lead_day", "split"]].drop_duplicates().sort_values(["time", "lead_day"])
    keys = keys.reset_index(drop=True)
    key_i = {(t, l): i for i, (t, l) in enumerate(zip(keys["time"], keys["lead_day"]))}
    df["si"] = list(map(key_i.get, zip(df["time"], df["lead_day"])))

    n = len(keys)
    grid = np.full((n, len(FEATURES), len(C.COARSE_LATS), len(C.COARSE_LONS)), np.nan, np.float32)
    for f, name in enumerate(FEATURES):
        grid[df["si"].values, f, df["li"].values, df["lj"].values] = df[name].values

    assert not np.isnan(grid).any(), "coarse grid has holes"

    np.savez_compressed(
        C.PROCESSED / "coarse_grid.npz",
        grid=grid,
        features=np.array(FEATURES),
        time=keys["time"].values.astype("datetime64[D]"),
        lead_day=keys["lead_day"].values.astype(np.int16),
        split=keys["split"].values.astype(str),
        coarse_lats=np.array(C.COARSE_LATS),
        coarse_lons=np.array(C.COARSE_LONS),
    )
    print(f"coarse_grid.npz   {grid.shape}  ({grid.nbytes / 1e6:.1f} MB)")
    print(f"  samples per split: {keys['split'].value_counts().to_dict()}")


def build_fine_target() -> None:
    """Regrid IMD onto the 0.25 deg target grid, aligned to each valid_time.

    IMD is already 0.25 deg, so this is a selection, not an interpolation -- the
    point the previous pipeline lost by collapsing it to 25 coarse points.
    """
    try:
        import xarray as xr
    except ImportError:
        print("\nfine_target: xarray not installed -- skipping")
        return

    files = sorted(C.IMD_NC_DIR.glob("*.nc"))
    if not files:
        print(f"\nfine_target: no NetCDFs in {C.IMD_NC_DIR} -- skipping")
        return

    flats, flons = fine_grid()
    meta = np.load(C.PROCESSED / "coarse_grid.npz", allow_pickle=True)
    valid_times = pd.to_datetime(meta["time"]) + pd.to_timedelta(meta["lead_day"], unit="D")

    # The IMD files are full-year, all-India (129 x 135).  Subset each to the
    # basin window before concatenating, or the stack is ~500 MB of mostly
    # irrelevant grid.  Files are annual; only the overlap years are needed.
    years = set(range(C.YEAR_MIN, C.YEAR_MAX + 1))
    rename = {"LATITUDE": "lat", "LONGITUDE": "lon", "TIME": "time", "RAINFALL": "rain"}

    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            ds = ds.rename({k: v for k, v in rename.items() if k in ds.variables or k in ds.dims})
            if not (pd.to_datetime(ds["time"].values).year.isin(years)).any():
                continue
            parts.append(ds["rain"].sel(lat=flats, lon=flons, method="nearest").load())

    if not parts:
        print(f"\nfine_target: no IMD files cover {C.YEAR_MIN}-{C.YEAR_MAX} -- skipping")
        return

    rain = xr.concat(parts, dim="time").sortby("time")
    # IMD already codes sea / out-of-country cells as NaN; this only guards
    # against negative fill values appearing in a future file revision.
    rain = rain.where(~(rain < 0.0))

    # reindex, not sel: a valid_time with no IMD record becomes NaN, never zero.
    sel = rain.reindex(time=valid_times)
    target = sel.transpose("time", "lat", "lon").values.astype(np.float32)

    # A fine cell is usable only where IMD reports land rainfall at some point.
    land = np.isfinite(target).any(axis=0)
    mask = np.isfinite(target) & land[None, :, :]

    # Cells that can actually route water to Hirakud.  Everything east of the
    # catchment drains to the coast and is irrelevant to reservoir inflow.
    catchment = (
        (flats[:, None] >= C.CATCHMENT_LAT_MIN) & (flats[:, None] <= C.CATCHMENT_LAT_MAX)
        & (flons[None, :] >= C.CATCHMENT_LON_MIN) & (flons[None, :] <= C.CATCHMENT_LON_MAX)
    ) & land

    in_season = pd.Series(valid_times).dt.month.isin(C.TARGET_MONTHS).values
    mask &= in_season[:, None, None]

    np.savez_compressed(
        C.PROCESSED / "fine_target.npz",
        target=np.nan_to_num(target, nan=0.0),  # paired with `mask`; never used alone
        mask=mask,
        land=land,
        catchment=catchment,
        fine_lats=flats,
        fine_lons=flons,
    )
    print(f"\nfine_target.npz   {target.shape}")
    print(f"  land cells      {int(land.sum())} / {land.size}")
    print(f"  in catchment    {int(catchment.sum())}  (rest drains to the coast)")
    print(f"  observed        {mask.sum() / max(land.sum() * len(valid_times), 1):.1%} of land cell-days")
    print(f"  valid_time months kept: {sorted(C.TARGET_MONTHS)}")


class PatchDataset:
    """Yields (patch, target, mask) per fine cell, gathering patches on demand.

    Wrap in torch.utils.data.Dataset by subclassing; kept framework-free here so
    the preprocessing step has no torch dependency.
    """

    def __init__(self, split: str, region: str = "catchment"):
        g = np.load(C.PROCESSED / "coarse_grid.npz", allow_pickle=True)
        keep = g["split"].astype(str) == split
        self.grid = g["grid"][keep]
        self.features = list(g["features"])
        self.lead_day = g["lead_day"][keep]

        t = np.load(C.PROCESSED / "fine_target.npz")
        self.target = t["target"][keep]
        self.mask = t["mask"][keep]
        self.fine_lats, self.fine_lons = t["fine_lats"], t["fine_lons"]

        # Fine-scale terrain, if build_dem.py has been run.  These are static
        # per-cell values, so they join the coordinate vector rather than the
        # convolutional patch -- the patch carries 1.5 deg fields only.
        dem_path = C.PROCESSED / "dem_fine.npz"
        self.dem_names = []
        if dem_path.exists():
            d = np.load(dem_path, allow_pickle=True)
            self.dem_names = ["elev_mean", "elev_std", "elev_min", "elev_max"]
            self.dem = np.stack([d[k] for k in self.dem_names], axis=-1).astype(np.float32)
        else:
            self.dem = None

        # Seasonal context.  ECMWF forecast skill collapses past lead 7, and the
        # optimal prediction there is climatology -- but the model had no date
        # input at all, so it could neither represent the monsoon's seasonal
        # cycle nor fall back on it.  `clim` is the train-only per-cell
        # day-of-year climatology; sin/cos(doy) give a continuous, wrap-safe
        # encoding of where in the season a forecast sits.
        valid = pd.to_datetime(g["time"][keep]) + pd.to_timedelta(self.lead_day, unit="D")
        self.doy = valid.dayofyear.values.astype(np.int16)
        clim_path = C.PROCESSED / "climatology.npz"
        self.season_names = []
        if clim_path.exists():
            self.clim = np.load(clim_path)["clim"]
            self.season_names = ["clim", "sin_doy", "cos_doy"]
        else:
            self.clim = None

        self.pidx, self.padded = patch_index_map()
        # "catchment" trains only on cells that can route water to Hirakud;
        # "land" uses the whole IMD land domain, which gives the shared ResNet
        # more rainfall samples to learn from at the cost of relevance.
        self.cells = np.argwhere(t[{"catchment": "catchment", "land": "land"}[region]])
        self.n_time = self.grid.shape[0]

    def __len__(self) -> int:
        return self.n_time * len(self.cells)

    def __getitem__(self, k: int):
        ti, ci = divmod(k, len(self.cells))
        a, b = self.cells[ci]
        rows = self.pidx[a, b, :, :, 0]
        cols = self.pidx[a, b, :, :, 1]

        patch = self.grid[ti][:, rows, cols]  # (n_feat, PATCH, PATCH)
        # Coordinates go in as embedding inputs, per Rasp & Lerch (2018).
        coords = np.array([self.fine_lats[a], self.fine_lons[b], self.lead_day[ti]], np.float32)
        if self.dem is not None:
            coords = np.concatenate([coords, self.dem[a, b]])
        if self.clim is not None:
            d = self.doy[ti]
            angle = 2.0 * np.pi * d / 366.0
            coords = np.concatenate([
                coords,
                np.array([self.clim[d, a, b], np.sin(angle), np.cos(angle)], np.float32),
            ])
        return patch, coords, self.target[ti, a, b], self.mask[ti, a, b]

    @property
    def n_coords(self) -> int:
        """Width of the static vector, for sizing the model's embedding."""
        return 3 + len(self.dem_names) + len(self.season_names)

    def cell_of(self, k: int) -> tuple[int, int]:
        """The (fine_lat_idx, fine_lon_idx) sample k reads from.

        Mirrors the divmod split in __getitem__: cell index is k's fast axis.
        Lets a caller (e.g. a per-cell rain threshold lookup) address the same
        cell a training sample used, without re-deriving the index arithmetic.
        """
        _, ci = divmod(k, len(self.cells))
        a, b = self.cells[ci]
        return int(a), int(b)


def main() -> None:
    idx, padded = patch_index_map()
    print(f"fine grid         {idx.shape[0]} x {idx.shape[1]} at {C.FINE_RES} deg")
    print(f"  patch           {C.PATCH} x {C.PATCH} coarse cells per fine cell")
    print(f"  edge-padded     {padded.sum()} / {padded.size} fine cells "
          f"({padded.mean():.0%}) -- coarse domain is only 5 x 5")
    print()
    build_coarse_grid()
    build_fine_target()


if __name__ == "__main__":
    main()
