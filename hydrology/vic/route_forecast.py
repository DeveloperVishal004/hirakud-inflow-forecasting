"""Route forecast VIC runoff to Hirakud inflow with the RVIC unit hydrographs.

Same convolution as hydrology/vic/route_and_evaluate.py -- the network, travel
times and mass conservation are RVIC's -- applied to each forecast instead of
the continuous observed run:

    Q(t) = sum over source cells s, lags k of  UH[k, s] * volume[s, t - k]

THE CONVOLUTION REACHES BACKWARDS.  With 10 lags, inflow on forecast day 1
depends on runoff from the 9 days BEFORE initialisation.  Those are supplied
from the observed-forcing VIC run: at forecast time they are history, not a
look-ahead.  Dropping them would start every forecast from zero discharge and
manufacture a rising limb that is an artefact of the splice.

Run:  python hydrology/vic/route_forecast.py --product ec
Out:  data/processed/vic_inflow_<product>.npz   (n_init, n_member, 30) m3/s
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs import config as C

R_EARTH = 6371.0072
VICOUT = C.PROCESSED / "vic" / "output"
PRM = sorted(glob.glob(str(Path.home() / ".vic_mahanadi" / "rvic" / "case"
                           / "params" / "*.prm.*.nc")))


def cell_area_km2(lat, res=C.FINE_RES):
    return (R_EARTH ** 2 * np.radians(res)
            * (np.sin(np.radians(lat + res / 2)) - np.sin(np.radians(lat - res / 2))))


def main():
    import xarray as xr
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", choices=["ec", "ec_qm", "ec_cnn"], required=True)
    args = ap.parse_args()

    prm = xr.open_dataset(PRM[-1])
    uh = prm["unit_hydrograph"].values[:, :, 0]          # (lags, sources)
    slat, slon = prm["source_lat"].values, prm["source_lon"].values
    n_lag, n_src = uh.shape

    basin = np.load(C.PROCESSED / "basin.npz", allow_pickle=True)
    frac, blats, blons = basin["fraction"], basin["lats"], basin["lons"]

    fc = np.load(C.PROCESSED / f"vic_forecast_{args.product}.npz", allow_pickle=True)
    flux, cells = fc["flux"], fc["cells"]               # (n_init, n_mem, n_cell, 30)
    inits = pd.to_datetime([str(x) for x in fc["inits"]])
    n_init, n_mem, n_cell, lead = flux.shape

    # depth (mm/day) -> volume (m3/s), per source cell in the UH's own order
    area = np.zeros(n_src)
    col = np.full(n_src, -1)
    key = {(round(la, 4), round(lo, 4)): i for i, (la, lo) in enumerate(cells)}
    for s in range(n_src):
        k = (round(float(slat[s]), 4), round(float(slon[s]), 4))
        if k not in key:
            sys.exit(f"source cell {k} has no VIC forecast column")
        col[s] = key[k]
        i = int(np.argmin(np.abs(blats - slat[s])))
        j = int(np.argmin(np.abs(blons - slon[s])))
        area[s] = cell_area_km2(blats[i]) * 1e6 * frac[i, j]
    to_cms = 1e-3 * area / 86400.0

    # observed-run history for the lags
    hist = pd.read_parquet(C.PROCESSED / "vic_daily.parquet")
    hdr = None
    obs_vol = np.zeros((len(hist), n_src), np.float32)
    files = {}
    for f in glob.glob(str(VICOUT / "fluxes_*.txt")):
        p = Path(f).stem.split("_")
        files[(round(float(p[1]), 4), round(float(p[2]), 4))] = f
    for s in range(n_src):
        k = (round(float(slat[s]), 4), round(float(slon[s]), 4))
        if hdr is None:
            hdr = open(files[k]).read().splitlines()[2].split()
        d = pd.read_csv(files[k], sep=r"\s+", skiprows=3, header=None, names=hdr)
        obs_vol[:, s] = (d.OUT_RUNOFF + d.OUT_BASEFLOW).values * to_cms[s]
    obs_dates = pd.to_datetime(hist["date"])
    day_of = {d: i for i, d in enumerate(obs_dates)}
    print(f"routing {n_src} sources, {n_lag} lags, {n_init} inits x {n_mem} members")

    q = np.zeros((n_init, n_mem, lead), np.float32)
    for ki, init in enumerate(inits):
        h0 = day_of.get(init)
        if h0 is None:
            sys.exit(f"init {init} outside the observed VIC record")
        past = obs_vol[h0 - (n_lag - 1):h0 + 1]                    # (n_lag, n_src)
        for mi in range(n_mem):
            fut = flux[ki, mi][col].T * to_cms[None, :]            # (30, n_src)
            vol = np.concatenate([past, fut], 0)                   # (n_lag+30, n_src)
            out = np.zeros(len(vol))
            for k in range(n_lag):
                out[k:] += (vol[:len(vol) - k] * uh[k][None, :]).sum(1)
            q[ki, mi] = out[n_lag:]                                # forecast days only
    out = C.PROCESSED / f"vic_inflow_{args.product}.npz"
    np.savez_compressed(out, q=q, inits=np.array([str(i) for i in inits]),
                        lead=np.arange(1, lead + 1))
    print(f"  inflow {q.shape}  mean {q.mean():.0f} m3/s  max {q.max():.0f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
