"""Build the RVIC routing inputs from routing.npz, and run the parameter step.

VIC produces runoff and baseflow per cell; it does not route them to a gauge.
RVIC convolves those fluxes with a unit hydrograph derived from the flow
network to give discharge at a point -- here, Hirakud.

Four inputs, all derived from products already built and verified:

  flow_direction.nc  D8 codes, basin id and source area from routing.npz
  domain.nc          the cell mask, areas and fractions from basin.npz
  pour_points.csv    the outlet, snapped to the channel in build_basin.py
  uh_box.csv         within-cell response; the standard RVIC box

Flow-direction convention.  RVIC decodes ARCMAP D8 (1=E, 2=SE, 4=S, 8=SW,
16=W, 32=NW, 64=N, 128=NE) whenever the maximum code is >= 10 -- read from
make_uh.py rather than assumed.  That is identical to HydroSHEDS, which is what
build_routing.py wrote, so no remapping is needed.  RVIC also flips the grid to
north-at-top internally if latitude ascends; ours ascends, so it will.

Run:  python hydrology/vic/build_rvic_inputs.py [--run]
Out:  data/processed/vic/rvic/
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs import config as C
from hydrology.vic.build_global_param import VICDIR as SAFE_VIC

REAL = C.PROCESSED / "vic" / "rvic"
RVIC = SAFE_VIC / "rvic"

R_EARTH = 6371.0072
# Lohmann within-cell unit hydrograph, RVIC's distributed default shape.
UH_BOX = [0.00, 0.15, 0.40, 0.25, 0.12, 0.05, 0.02, 0.01]
UH_DT = 3600  # seconds


DIAGONAL = {2, 8, 32, 128}          # SE, SW, NW, NE
NS = {4, 64}                        # S, N


def flow_distance_m(lats, lons, flow):
    """Travel distance out of each cell, following its own D8 direction."""
    res = C.FINE_RES
    dy_m = np.full(len(lats), res * 110574.0)                    # metres N-S
    dx_m = res * 111320.0 * np.cos(np.radians(lats))             # metres E-W
    out = np.empty(flow.shape, dtype="float64")
    for i in range(len(lats)):
        for j in range(len(lons)):
            code = int(flow[i, j])
            if code in NS:
                out[i, j] = dy_m[i]
            elif code in DIAGONAL:
                out[i, j] = np.hypot(dy_m[i], dx_m[i])
            else:                       # E/W, or 0 at the outlet
                out[i, j] = dx_m[i]
    return out


def cell_area_km2(lats, res):
    phi1, phi2 = np.radians(lats + res / 2), np.radians(lats - res / 2)
    return R_EARTH ** 2 * np.radians(res) * (np.sin(phi1) - np.sin(phi2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true",
                    help="also invoke `rvic parameters` on the generated config")
    args = ap.parse_args()

    import xarray as xr

    rt = np.load(C.PROCESSED / "routing.npz", allow_pickle=True)
    bs = np.load(C.PROCESSED / "basin.npz", allow_pickle=True)
    lats, lons = rt["lats"], rt["lons"]
    flow, frac = rt["flow_direction"], bs["fraction"]
    olat, olon = float(rt["outlet_lat"]), float(rt["outlet_lon"])

    REAL.mkdir(parents=True, exist_ok=True)
    ny, nx = len(lats), len(lons)
    area = np.repeat(cell_area_km2(lats, C.FINE_RES)[:, None], nx, axis=1)
    mask = (frac > 0).astype("int32")

    # ---- flow direction file
    # Basin_ID must be non-zero on every routed cell; a single basin here.
    xr.Dataset(
        {
            "Flow_Direction": (("lat", "lon"), flow.astype("int32"),
                               {"units": "-",
                                "description": "ARCMAP D8 (1=E,2=SE,...,128=NE)"}),
            "Basin_ID": (("lat", "lon"), mask.astype("int32")),
            "src_area": (("lat", "lon"), (area * frac).astype("float64"),
                         {"units": "km2"}),
            # METRES, not km.  RVIC divides distance by VELOCITY (m/s) to get
            # travel time; in km the whole 85,000 km2 basin routes in minutes
            # and the unit hydrograph collapses into the first timestep.
            # Distance is per-cell and direction-dependent: diagonal steps are
            # longer, and east-west spacing shrinks with cos(latitude).
            "Flow_Distance": (("lat", "lon"), flow_distance_m(lats, lons, flow),
                              {"units": "m"}),
        },
        coords={"lat": ("lat", lats, {"units": "degrees_north"}),
                "lon": ("lon", lons, {"units": "degrees_east"})},
    ).to_netcdf(REAL / "flow_direction.nc")

    # ---- domain file
    xr.Dataset(
        {
            "mask": (("lat", "lon"), mask),
            "frac": (("lat", "lon"), frac.astype("float64")),
            "area": (("lat", "lon"), (area * 1e6).astype("float64"),
                     {"units": "m2"}),
        },
        coords={"lat": ("lat", lats, {"units": "degrees_north"}),
                "lon": ("lon", lons, {"units": "degrees_east"})},
    ).to_netcdf(REAL / "domain.nc")

    (REAL / "pour_points.csv").write_text(
        "lons,lats,names\n" f"{olon:.4f},{olat:.4f},hirakud\n")

    lines = ["time,UHb"] + [f"{i * UH_DT},{v:.4f}" for i, v in enumerate(UH_BOX)]
    (REAL / "uh_box.csv").write_text("\n".join(lines) + "\n")

    # Every OPTIONS key RVIC reads must be present -- it indexes the dict
    # directly and raises KeyError rather than falling back to a default.
    # The full list was taken from the source, not from the documentation.
    cfg = f"""[OPTIONS]
CASEID          = hirakud
GRIDID          = mahanadi
CASE_DIR        = {RVIC / 'case'}
TEMP_DIR        = {RVIC / 'temp'}
REMAP           = False
AGGREGATE       = False
AGG_PAD         = 25
NETCDF_FORMAT   = NETCDF4
NETCDF_ZLIB     = False
NETCDF_COMPLEVEL = 4
NETCDF_SIGFIGS  = None
SUBSET_DAYS     = 10
CONSTRAIN_FRACTIONS = False
SEARCH_FOR_CHANNEL  = False
CLEAN           = False
VERBOSE         = True
LOG_LEVEL       = INFO

[POUR_POINTS]
FILE_NAME       = {RVIC / 'pour_points.csv'}

[UH_BOX]
FILE_NAME       = {RVIC / 'uh_box.csv'}
HEADER_LINES    = 1

[ROUTING]
FILE_NAME       = {RVIC / 'flow_direction.nc'}
LONGITUDE_VAR   = lon
LATITUDE_VAR    = lat
FLOW_DISTANCE_VAR = Flow_Distance
FLOW_DIRECTION_VAR = Flow_Direction
BASIN_ID_VAR    = Basin_ID
SOURCE_AREA_VAR = src_area
OUTPUT_INTERVAL = 86400
BASIN_FLOWDAYS  = 50
CELL_FLOWDAYS   = 2
VELOCITY        = 1.0
DIFFUSION       = 2000.0

[DOMAIN]
FILE_NAME       = {RVIC / 'domain.nc'}
LONGITUDE_VAR   = lon
LATITUDE_VAR    = lat
LAND_MASK_VAR   = mask
FRACTION_VAR    = frac
AREA_VAR        = area
"""
    (REAL / "rvic_params.cfg").write_text(cfg)

    print(f"wrote RVIC inputs -> {REAL}")
    print(f"  grid           {ny}x{nx}, {int(mask.sum())} routed cells")
    print(f"  outlet         {olat:.2f} N {olon:.2f} E")
    print(f"  basin area     {(area * frac).sum():,.0f} km2  "
          f"(delineated {float(bs['area_km2']):,.0f})")
    print(f"  flow codes     {sorted(set(flow[mask == 1].ravel().tolist()))}")
    print(f"  max code {flow.max()} -> RVIC will use "
          f"{'ARCMAP (1-128)' if flow.max() >= 10 else 'VIC (1-8)'} directions")

    if not args.run:
        print(f"\nnext:  rvic parameters {REAL / 'rvic_params.cfg'}")
        return

    for d in (RVIC / "case", RVIC / "temp"):
        Path(d).mkdir(parents=True, exist_ok=True)
    print("\nrunning `rvic parameters` ...")
    r = subprocess.run(["rvic", "parameters", str(RVIC / "rvic_params.cfg")],
                       capture_output=True, text=True, timeout=3600)
    out = (r.stdout or "") + (r.stderr or "")
    print(f"  exit code {r.returncode}")
    print("\n".join(out.splitlines()[-25:]))
    made = list((RVIC / "case").rglob("*.nc"))
    print(f"\n  produced {len(made)} NetCDF file(s)")
    for m in made[:5]:
        print(f"    {m.name}")
    sys.exit(0 if r.returncode == 0 and made else 1)


if __name__ == "__main__":
    main()
