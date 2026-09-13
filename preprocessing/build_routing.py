"""Upscale HydroSHEDS 15 arc-sec flow direction to the 0.25 deg routing grid.

VIC produces runoff and baseflow per cell but does not route them; reservoir
inflow needs a routing step, and RVIC needs a flow-direction grid at the VIC
resolution.  HydroSHEDS ships at 15 arc-sec, so it has to be upscaled -- and
upscaling flow direction is not resampling.  Taking a majority or nearest
neighbour of the D8 codes produces a plausible-looking grid whose river network
is disconnected, because direction is a property of the drainage topology, not
a value that averages.

The method here follows the logic of Dominant River Tracing (Wu et al. 2011):
for each coarse cell, find the fine pixel carrying the most upstream area --
that is the trunk channel -- then walk downstream along the fine network until
the path leaves the cell.  The coarse neighbour it enters defines the coarse
direction.  This keeps the upscaled network connected and honours where the
water actually goes, which a majority filter does not.

Validation is by construction: following the upscaled network from every cell
must terminate at the outlet cell containing the dam.  Any cell that does not
is reported rather than silently written out.

HydroSHEDS D8: 1=E 2=SE 4=S 8=SW 16=W 32=NW 64=N 128=NE, 0=sink, 255=nodata.

Run:  python preprocessing/build_routing.py
Out:  data/processed/routing.npz, results/figures/routing.png
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config as C
from preprocessing.build_basin import (D8, DIR_TIF, ACC_TIF, DAM_LAT, DAM_LON,
                                       read_window, snap)

OUT = C.PROCESSED / "routing.npz"

# Coarse-cell offset (drow, dcol) -> D8 code, rows increasing NORTHWARD here
# because the coarse grid is stored south-to-north like the other products.
COARSE_CODE = {(0, 1): 1, (-1, 1): 2, (-1, 0): 4, (-1, -1): 8,
               (0, -1): 16, (1, -1): 32, (1, 0): 64, (1, 1): 128}


def main() -> None:
    basin_f = C.PROCESSED / "basin.npz"
    if not basin_f.exists():
        sys.exit("run preprocessing/build_basin.py first")
    b = np.load(basin_f, allow_pickle=True)
    lats, lons, frac = b["lats"], b["lons"], b["fraction"]
    half = C.FINE_RES / 2

    win = dict(lat=(lats[0] - half, lats[-1] + half),
               lon=(lons[0] - half, lons[-1] + half))
    dirs, transform = read_window(DIR_TIF, win)
    acc, _ = read_window(ACC_TIF, win)
    nrow, ncol = dirs.shape
    print(f"fine grid {dirs.shape} at {transform.a * 3600:.0f} arc-sec")
    print(f"coarse    {len(lats)}x{len(lons)} at {C.FINE_RES} deg")

    lat_px = transform.f + transform.e * (np.arange(nrow) + 0.5)
    lon_px = transform.c + transform.a * (np.arange(ncol) + 0.5)

    def coarse_of(r, c):
        """Coarse (i, j) containing fine pixel (r, c); None if outside."""
        i = int(np.floor((lat_px[r] - (lats[0] - half)) / C.FINE_RES))
        j = int(np.floor((lon_px[c] - (lons[0] - half)) / C.FINE_RES))
        if 0 <= i < len(lats) and 0 <= j < len(lons):
            return i, j
        return None

    outlet_fine, _ = snap(acc, transform, DAM_LAT, DAM_LON)
    outlet_coarse = coarse_of(*outlet_fine)
    print(f"  outlet cell (dam): {lats[outlet_coarse[0]]:.2f} N "
          f"{lons[outlet_coarse[1]]:.2f} E")

    accf = acc.astype("float64")
    accf[acc == np.iinfo(np.uint32).max] = -1

    # Restrict the trunk search to basin pixels.  Without this, a coarse cell
    # sitting on the divide picks its highest-accumulation pixel from whichever
    # river is largest inside it -- often a neighbouring basin flowing away
    # from Hirakud -- and the upscaled network strands that cell.
    fm, flat_b, flon_b = b["fine_mask"], b["fine_lat"], b["fine_lon"]
    ri = np.searchsorted(-flat_b, -lat_px)      # basin arrays run north->south
    ci = np.searchsorted(flon_b, lon_px)
    ri = np.clip(ri, 0, len(flat_b) - 1)
    ci = np.clip(ci, 0, len(flon_b) - 1)
    in_basin = fm[np.ix_(ri, ci)]
    assert abs(flat_b[ri] - lat_px).max() < 0.01, "basin/routing grids misaligned"
    assert abs(flon_b[ci] - lon_px).max() < 0.01, "basin/routing grids misaligned"
    accf = np.where(in_basin, accf, -1)

    flow = np.zeros((len(lats), len(lons)), dtype=np.int16)
    upstream = np.zeros((len(lats), len(lons)), dtype=np.float64)
    unresolved = []

    for i in range(len(lats)):
        rsel = np.where((lat_px >= lats[i] - half) & (lat_px < lats[i] + half))[0]
        for j in range(len(lons)):
            if frac[i, j] <= 0:
                continue                       # outside the basin: no routing
            csel = np.where((lon_px >= lons[j] - half) & (lon_px < lons[j] + half))[0]
            if rsel.size == 0 or csel.size == 0:
                continue
            sub = accf[np.ix_(rsel, csel)]
            dr, dc = np.unravel_index(np.argmax(sub), sub.shape)
            r, c = rsel[dr], csel[dc]
            upstream[i, j] = sub.max()

            # Walk downstream until the path leaves this coarse cell.
            steps = 0
            cur = (i, j)
            while steps < 20000:
                code = int(dirs[r, c])
                if code not in D8:              # sink or nodata
                    break
                dr_, dc_ = D8[code]
                r, c = r + dr_, c + dc_
                if not (0 <= r < nrow and 0 <= c < ncol):
                    break
                nxt = coarse_of(r, c)
                if nxt is None or nxt != cur:
                    break
                steps += 1

            nxt = coarse_of(r, c) if (0 <= r < nrow and 0 <= c < ncol) else None
            if nxt is None:
                # Leaves the domain entirely -- correct only at the outlet.
                flow[i, j] = 0
                if (i, j) != outlet_coarse:
                    unresolved.append((i, j, "exits domain"))
                continue
            off = (nxt[0] - i, nxt[1] - j)
            if off == (0, 0):
                flow[i, j] = 0
                if (i, j) != outlet_coarse:
                    unresolved.append((i, j, "internal sink"))
            elif off in COARSE_CODE:
                flow[i, j] = COARSE_CODE[off]
            else:
                # The trunk skipped a cell diagonally; snap to the dominant step.
                step = (int(np.sign(off[0])), int(np.sign(off[1])))
                flow[i, j] = COARSE_CODE.get(step, 0)
                if flow[i, j] == 0:
                    unresolved.append((i, j, f"offset {off}"))

    flow[outlet_coarse] = 0                    # the outlet drains out of the domain

    # ---- validation: every basin cell must reach the outlet
    print("\n=== connectivity ===")
    bad, longest = [], 0
    for i in range(len(lats)):
        for j in range(len(lons)):
            if frac[i, j] <= 0:
                continue
            ci, cj, n = i, j, 0
            while (ci, cj) != outlet_coarse and n < 4 * flow.size:
                code = int(flow[ci, cj])
                if code == 0:
                    break
                off = next(o for o, k in COARSE_CODE.items() if k == code)
                ci, cj = ci + off[0], cj + off[1]
                if not (0 <= ci < len(lats) and 0 <= cj < len(lons)):
                    break
                n += 1
            longest = max(longest, n)
            if (ci, cj) != outlet_coarse:
                bad.append((i, j))

    n_basin = int((frac > 0).sum())
    print(f"  basin cells                {n_basin}")
    print(f"  reaching the outlet        {n_basin - len(bad)}")
    print(f"  NOT reaching the outlet    {len(bad)}")
    print(f"  longest flow path          {longest} cells")
    if unresolved:
        print(f"  cells needing a fallback   {len(unresolved)}")

    ok = len(bad) == 0
    if not ok:
        for i, j in bad[:10]:
            print(f"    stranded: {lats[i]:.2f} N {lons[j]:.2f} E  "
                  f"basin fraction {frac[i, j]:.2f}")

    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT, lats=lats, lons=lons, flow_direction=flow, fraction=frac,
        upstream_cells_15s=upstream,
        outlet_i=outlet_coarse[0], outlet_j=outlet_coarse[1],
        outlet_lat=lats[outlet_coarse[0]], outlet_lon=lons[outlet_coarse[1]],
        encoding="HydroSHEDS D8: 1=E 2=SE 4=S 8=SW 16=W 32=NW 64=N 128=NE, 0=outlet/none",
        source="upscaled from HydroSHEDS v1 15 arc-sec by dominant river tracing",
    )
    print(f"\n{'PASS' if ok else 'FAIL'} -- wrote {OUT}")

    # ---- figure: arrows must form a tree converging on the dam
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(np.where(frac > 0, frac, np.nan), origin="lower", cmap="YlGnBu",
              vmin=0, vmax=1, extent=[lons[0] - half, lons[-1] + half,
                                      lats[0] - half, lats[-1] + half])
    inv = {k: o for o, k in COARSE_CODE.items()}
    for i in range(len(lats)):
        for j in range(len(lons)):
            if flow[i, j] in inv:
                di, dj = inv[int(flow[i, j])]
                ax.arrow(lons[j], lats[i], dj * 0.14, di * 0.14,
                         head_width=0.05, head_length=0.05,
                         fc="k", ec="k", lw=0.8, length_includes_head=True)
    ax.plot(lons[outlet_coarse[1]], lats[outlet_coarse[0]], "r*", ms=20, mec="k")
    ax.set_title("Upscaled 0.25 deg flow direction (star = Hirakud outlet)")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.tight_layout()
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / "routing.png", dpi=110)
    print(f"wrote {C.FIGURES / 'routing.png'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
