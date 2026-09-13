"""Fetch IMD 2003 rainfall and write it in the same form as the other years.

The IMD archive in data/raw/imd/ starts at 2004, but the VIC run needs 2003 as
a spin-up year: soil moisture entering the 2004 monsoon has to be produced by
the model rather than assumed.  Precipitation is the one forcing VIC cannot run
without, so a missing 2003 would either block spin-up or force a different
precipitation product for that year.

imdlib pulls the same 0.25 deg product from IMD Pune, so this keeps the
precipitation source identical across the whole record -- which matters,
because IMD rainfall is also the observational target the downscaler is trained
against.

The conversion matches the existing files exactly, verified against
RF25_ind2004_rfp25.nc:

    dims       TIME, LATITUDE, LONGITUDE     (not time/lat/lon)
    variable   RAINFALL
    dtype      float32
    missing    -999.0 as _FillValue, read back as NaN
    grid       6.5-38.5 N, 66.5-100.0 E at 0.25 deg (129 x 135)

Anything less would open fine and then misalign silently at concatenation.

Run:  python preprocessing/fetch_imd_2003.py
Out:  data/raw/imd/RF25_ind2003_rfp25.nc
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C

YEAR = 2003
REFERENCE = "RF25_ind2004_rfp25 (1).nc"
OUT = C.IMD_NC_DIR / f"RF25_ind{YEAR}_rfp25.nc"


def main() -> None:
    if OUT.exists():
        print(f"{OUT.name} already present -- nothing to do")
        return

    import imdlib as imd

    tmp = Path(tempfile.mkdtemp(prefix="imd2003_"))
    try:
        ds = imd.get_data("rain", YEAR, YEAR, fn_format="yearwise",
                          file_dir=str(tmp)).get_xarray()

        # imdlib marks missing cells with -999 in the values themselves; the
        # archive files carry it as _FillValue and present NaN on read.
        rain = ds["rain"].where(ds["rain"] > -998.0).astype("float32")

        out = xr.Dataset(
            {"RAINFALL": (("TIME", "LATITUDE", "LONGITUDE"), rain.values)},
            coords={"TIME": ds["time"].values,
                    "LATITUDE": ds["lat"].values.astype("float64"),
                    "LONGITUDE": ds["lon"].values.astype("float64")},
            attrs={"Conventions": "CF-1.0",
                   "history": f"imdlib via preprocessing/fetch_imd_2003.py"},
        )

        # Fail loudly here rather than let a mismatched grid reach the model.
        ref = xr.open_dataset(C.IMD_NC_DIR / REFERENCE)
        assert np.allclose(out.LATITUDE, ref.LATITUDE), "latitude grid differs"
        assert np.allclose(out.LONGITUDE, ref.LONGITUDE), "longitude grid differs"
        assert out.RAINFALL.dtype == ref.RAINFALL.dtype, "dtype differs"

        C.IMD_NC_DIR.mkdir(parents=True, exist_ok=True)
        out.to_netcdf(OUT, encoding={"RAINFALL": {"_FillValue": -999.0,
                                                  "dtype": "float32"}})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    back = xr.open_dataset(OUT)
    a = back.RAINFALL.values
    print(f"wrote {OUT}")
    print(f"  dims      {dict(back.sizes)}")
    print(f"  NaN       {100 * np.isnan(a).mean():.1f}%  "
          f"(2004 reference: {100 * np.isnan(ref.RAINFALL.values).mean():.1f}%)")
    print(f"  rainfall  {np.nanmin(a):.1f} .. {np.nanmax(a):.1f} mm/day")

    cat = back.sel(
        LATITUDE=slice(C.CATCHMENT_LAT_MIN, C.CATCHMENT_LAT_MAX),
        LONGITUDE=slice(C.CATCHMENT_LON_MIN, C.CATCHMENT_LON_MAX)).RAINFALL
    jjas = cat.sel(TIME=cat.TIME.dt.month.isin([6, 7, 8, 9]))
    print(f"  catchment {cat.sizes['LATITUDE']}x{cat.sizes['LONGITUDE']} cells, "
          f"JJAS {YEAR} mean total {float(jjas.sum('TIME').mean()):.0f} mm")


if __name__ == "__main__":
    main()
