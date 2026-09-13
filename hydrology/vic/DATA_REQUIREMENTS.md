# VIC setup for the Mahanadi above Hirakud — data manifest

Everything VIC 5 needs, what we already hold, and exactly where the rest comes
from. Run `python hydrology/vic/check_inputs.py` at any point to see which
items are still outstanding.

**Domain**: 19.75–23.60 °N, 80.50–84.25 °E (Mahanadi catchment upstream of
Hirakud, ~83,400 km²)
**Resolution**: 0.25° to match the IMD rainfall and the existing downscaler
output. VIC can run finer, but forcing at 0.25° is the binding constraint —
a finer grid would interpolate rather than add information.
**Period**: 2004-01-01 → 2014-12-31 (the inflow record; 2004 is spin-up)

---

## 1. Meteorological forcing — daily, gridded

**VIC 5 requires SEVEN forcings — it does not derive any of them.** An earlier
version of this manifest said the classic driver needs four and generates
humidity, radiation and pressure internally via MTCLIM. **That is VIC 4
behaviour.** VIC 5 removed MTCLIM ([release notes, GH#288]): *"VIC forcings are
now required to be provided at the same time frequency as the model will be run
at."* `vic/drivers/classic/src/vic_force.c` aborts on any missing one:

```
Air temperature must be supplied as a forcing
Precipitation must be supplied as a forcing
Downward shortwave radiation must be supplied as a forcing
Downward longwave radiation must be supplied as a forcing
Atmospheric pressure must be supplied as a forcing
Vapor pressure must be supplied as a forcing
Wind speed must be supplied as a forcing
```

Note there is also **no `TMAX`/`TMIN` forcing type** — the valid names are
`AIR_TEMP PREC SWDOWN LWDOWN PRESSURE VP WIND` (plus optional ALBEDO, LAI,
CATM, FCANOPY, FDIR, PAR, CHANNEL_IN). Temperature enters as `AIR_TEMP` at the
model time step, not as daily extremes.

| VIC forcing | Source | Status |
|---|---|---|
| `PREC` | IMD 0.25° | ✅ |
| `AIR_TEMP` | ERA5-Land `t2m`, hourly → daily mean | ✅ |
| `WIND` | ERA5-Land `u10`,`v10`, 6-hourly | ✅ |
| `VP` | derived from ERA5-Land `d2m` (Magnus) | ✅ |
| `SWDOWN` | ERA5-Land `surface_solar_radiation_downwards` | ⏳ `rad` stream |
| `LWDOWN` | ERA5-Land `surface_thermal_radiation_downwards` | ⏳ `rad` stream |
| `PRESSURE` | ERA5-Land `surface_pressure` | ⏳ `rad` stream |

The cost of the mistake was 12 extra CDS requests (`rad`: 3 variables,
6-hourly, one request per year). Both radiation fields are **accumulated** from
00 UTC and reset daily, so the daily total is the value at 00Z of the following
day — the same convention already verified for `tp`, and retained by 6-hourly
sampling because it includes 00Z.

| Variable | Units | Status | Source |
|---|---|---|---|
| Precipitation | mm/day | ✅ **have**, 2003–2023 | IMD 0.25° (`data/raw/imd/`) |
| Air temperature max/min | °C | ⏳ downloading | ERA5-Land `2m_temperature` |
| Wind speed | m/s | ⏳ downloading | ERA5-Land 10 m u/v → speed |

**Source: ERA5-Land** (CDS, `reanalysis-era5-land`), 0.1° hourly. Fetched by
`preprocessing/download_era5_land.py`; every file checked by
`preprocessing/verify_era5.py` before use. IMD stays the precipitation source —
it is the observational target the downscaler is trained against, so
substituting ERA5 precipitation would break the chain's internal consistency.

2003 rainfall was originally absent (the archive began at 2004), which would
have left the spin-up year without the one forcing VIC cannot run without.
`preprocessing/fetch_imd_2003.py` retrieves it via `imdlib` from the same IMD
product and asserts the grid matches the existing files. JJAS 2003 catchment
mean is 1381 mm against a 2004–2014 mean of 1128 ± 112 mm — a genuinely wet
year (z = +2.3, the rebound after the 2002 drought), not an artefact, so VIC
enters 2004 wetter than average and that is correct.

### Facts established on the June 2003 file — do not re-derive

- **Grid**: 0.1°, 43 × 43 over 19.5–23.7 N, 80.3–84.5 E. CDS snaps the
  requested box outward to its own grid, so the north edge came back at 23.70
  rather than the requested 23.75 — still clear of the catchment's 23.60, but
  only by one cell.
- **Zero NaN**: the box is entirely inland, so there is no ocean mask to carry.
- **`tp` is ACCUMULATED from 00 UTC and resets daily.** Confirmed on 100 % of
  days, not assumed. The daily total for day *D* is `tp` at **00Z on day D+1**
  (verified: 22Z 15.31 → 23Z 15.68 → 00Z 16.09 → 01Z 0.37 mm). Naively summing
  all 24 hourly values overstates June 2003 by 12× — 2218 mm against a true
  ~177 mm. Using the 23Z value instead is close but silently drops one hour per
  day. This is the same class as the ECMWF `tp`/`ssr` bug fixed in
  `preprocessing/build_forcing.py`.
- **`d2m` exceeds `t2m`** on 0.03 % of cells by at most 0.003 °C — ERA5's
  permitted marginal supersaturation. Ignore it; do not "fix" it.

### Variables deliberately excluded

Radiation (`surface_solar_radiation_downwards`, `surface_thermal_radiation_
downwards`) is omitted: VIC's MTCLIM preprocessor derives it from the diurnal
temperature range, and including it would reintroduce accumulated variables for
no gain. `skin_temperature` is omitted because it is ERA5-Land's own
land-surface *output* — feeding it to VIC is circular.

`volumetric_soil_water_layer_1` and `_4` and `total_precipitation` **are**
downloaded but are **not forcing**, for the same circularity reason. Their uses
are (a) soil moisture as a candidate Ridge feature — layer 1 (0–7 cm) responds
in days, layer 4 (100–289 cm) carries multi-week memory, matching the 8–17 day
leads — and later as a reference to validate VIC's own soil moisture against;
(b) `tp` as an independent cross-check on IMD. If soil moisture is used as a
feature it **must be sampled at the init date, never the valid date**, the same
discipline as `LagSource` in `cnn/train_gbm.py`; valid-date sampling is future
information and would leak.

## 2. Soil parameters

One row per grid cell, ~53 columns. The physically important ones:

| Parameter | Meaning | Source |
|---|---|---|
| `b_infilt` | infiltration curve shape | **calibrated** |
| `Ds`, `Dsmax`, `Ws` | baseflow behaviour | **calibrated** |
| `depth[1..3]` | soil layer thicknesses (m) | calibrated / soil DB |
| `sand`, `clay`, `silt` % per layer | texture | soil database |
| `bulk_density`, `soil_density` | kg/m³ | soil database |
| `Wcr_FRACT`, `Wpwp_FRACT` | critical / wilting point | derived from texture |
| `elev` | mean cell elevation | ✅ **have** (`dem_fine.npz`) |
| `annual_prec` | mean annual precipitation | ✅ derivable from IMD |

✅ **Done** — `preprocessing/build_soilgrids.py` → `data/processed/soilgrids.npz`
(16 × 16 at 0.25°), maps at `results/figures/soilgrids.png`, checked by
`preprocessing/verify_soilgrids.py`.

**Source**: **SoilGrids 250 m** (ISRIC) via WCS, delivered in EPSG:4326 so no
reprojection from Interrupted Goode Homolosine is needed here. Preferred over
HWSD for the Mahanadi: HWSD's India coverage derives from older 1:5M FAO
mapping, while SoilGrids is machine-learned from ~240k profiles and resolves
the basin's texture contrasts.

Six SoilGrids depths → three VIC layers by thickness weighting: 0–15 cm,
15–60 cm, 60–200 cm.

Verified: texture closes to 100.00 % (max deviation 0.00 %) in all three
layers, confirming the g/kg → % divisor; bulk density 1.22 → 1.77 g/cm³
increasing with depth, which is real compaction.

**What the soil says about the basin.** Catchment mean 0–15 cm is sand 26 %,
silt 36 %, **clay 38 %** — a clay/silty-clay Vertisol, the black cotton soil the
upper Mahanadi is known for. Clay runs 40–44 % in the northern/upper catchment
and falls to 28–33 % toward the outlet; correlation with elevation is only
+0.10, so this is information the DEM does not already carry.

This bears directly on the earlier GR4J-Horton result. A fixed infiltration
ceiling (`INFIL_MAX`) was rejected because it lost on every held-out year, and
the soil data suggests why: Vertisols do not *have* a fixed infiltration
capacity. Dry and cracked they swallow the first storms through macropores;
wetted and swollen they seal almost completely. A constant ceiling is wrong in
both directions and cannot represent either. VIC's variable-infiltration curve
(`b_infilt`) is the right structure for this, which strengthens the case for
VIC over both XAJ and the Hortonian variant.

✅ **Pedotransfer done** — `hydrology/vic/build_soil_params.py` applies Saxton &
Rawls (2006) and writes `data/processed/vic/soil_param.txt` (256 rows × 53
columns, 150 active cells) plus `soil_params.npz` for inspection.

| | Layer 1 | Layer 2 | Layer 3 |
|---|---|---|---|
| Ksat (mm/day) | 48.5 | 22.0 | 19.2 |
| `expt` | 19.4 | 20.6 | 20.6 |
| bubbling pressure (cm) | 57.6 | 68.9 | 71.1 |
| porosity | 0.433 | 0.409 | 0.402 |

Conductivity falling with depth and porosity falling with compaction are both
correct for this profile. Range checks pass: Ksat 2.5–376 mm/day, `expt`
16.5–23.7, porosity 0.33–0.54, and Wpwp < Wcr < 1 in every cell.

#### A real limitation, recorded rather than hidden

Saxton & Rawls was fitted to USDA samples whose bulk density sits near what
their texture equations imply. **These Vertisols do not.** Measured density
reaches 1.77 g/cm³ where texture implies 1.33–1.50, so the density-adjustment
step drives porosity *below* field capacity — impossible — in a large minority
of cells. They are clamped to retain 10 % air-filled porosity at saturation,
and the count is reported: **82 / 146 / 153 of 256 cells** in layers 1/2/3.

That is over half the domain in the deeper layers. It means Ksat and the
moisture states there are constrained by an assumption, not derived from the
data, and it is a reason to treat `b_infilt`, `Ds`, `Dsmax`, `Ws` as genuinely
free during calibration rather than fine-tuning around a trusted prior.

Two further points where the literal convention had to be departed from:

- **`expt` uses the texture-based retention pair, not the density-adjusted
  one.** `B` describes the pore-size *distribution*, a texture property;
  compaction reduces total pore space without reshaping it. Computing `B` from
  clamped values is circular — it compresses log θ33 − log θ1500 toward zero
  and sends `expt` above 70 against a normal range of 3–30.
- **`Wcr` is 70 % of *available* water above wilting point**, not the literal
  0.7 × θ33. Taken literally, Wcr falls below the wilting point in heavy clay,
  which would have VIC restrict transpiration at a moisture the plant cannot
  extract at all. The two forms are equivalent in sandy soil.

#### Measured vs calibrated

| Derived from data | Calibration starting values (**not** results) |
|---|---|
| Ksat, `expt`, bubble, Wcr, Wpwp, bulk density, quartz, depth, elev, annual_prec, avg_T | `b_infilt` 0.2, `Ds` 0.001, `Dsmax` = layer-3 Ksat, `Ws` 0.9, `c` 2.0 |

`avg_T` = 25.2 °C, from the 12 ERA5-Land months of 2003 currently on disk (a
full calendar year, so not seasonally biased). Worth recomputing over the whole
record once the download completes.

### Elevation: use `dem_vic.npz`, not `dem_fine.npz`

Building the soil grid exposed a latent domain mismatch. `FINE_LON_MIN` is
81.0 (set by the ECMWF downscaling domain) but `CATCHMENT_LON_MIN` is 80.50, so
`dem_fine.npz` has no elevation for the catchment's two westernmost columns —
the headwaters. `preprocessing/build_dem.py --grid vic` writes `dem_vic.npz` on
the catchment grid, asserted to match `soilgrids.npz` exactly. `dem_fine.npz` is
deliberately left untouched: the trained CNN depends on its 25 × 25 shape.

The same mismatch means the CNN's "catchment" region has always been
81.0–84.25 E rather than the 80.50 the config declares. The chain is internally
consistent (`build_daily_catchment_obs.py` averages the same mask), so existing
results stand — but the catchment is narrower than intended, and **HydroSHEDS
will settle the true boundary** when routing is set up.

## 3. Vegetation parameters

| File | Contents | Status |
|---|---|---|
| Class fractions | per cell: IGBP fractions | ✅ `landcover.npz` (`build_landcover.py`) |
| Veg parameter | per cell: classes, Cv, roots | ✅ `vic/veg_param.txt` (`build_veg_params.py`) |
| Veg library | per class: LAI, albedo, roughness, resistances | ✅ `vic/veg_lib.txt` (same script) |

1–8 classes per cell (mean 4.7); Σ Cv ranges 0.746–1.000, the minimum being the
reservoir cell where 23 % water becomes bare soil. No basin cell is left without
vegetation, and only 0.24 % of domain area is lost to the 0.5 % class threshold.

#### The stock VIC library is wrong for this basin

The distributed VIC vegetation library was assembled for North American and
European land cover. Its cropland LAI peaks in June–July and has fallen away by
September — the temperate growing season.

This catchment is **69 % cropland, and that cropland is monsoon kharif**, mostly
rice: sown at the June onset, peaking late August–September, harvested
October–November, bare through the hot pre-monsoon. The stock curve puts peak
transpiration six weeks early and senesces the canopy while the monsoon is still
delivering its heaviest rain — a systematic evapotranspiration error in exactly
the season being forecast.

The cropland, grassland and mosaic curves therefore follow the Indian
phenological calendar:

```
      Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec
LAI   0.5  0.5  0.3  0.2  0.2  0.8  2.5  4.0  4.2  2.5  1.0  0.8
                                         peak Sep (stock library: Jul)
```

Deciduous broadleaf is likewise **dry-season** deciduous — teak and sal drop
their leaves February–May — not winter-deciduous.

Roughness and displacement follow the standard aerodynamic ratios (0.123 h,
0.67 h; Brutsaert 1982). Root fractions put trees deeper than the shallow-rooted
rice that dominates the basin. Note `wind_h` is set above canopy while ERA5
supplies 10 m wind; VIC applies a log profile between the two.

**MODIS MCD12Q1.061 `LC_Type1`, 2010** — one representative mid-record year;
land-cover change is not being studied here. Chosen over ESA WorldCover
(2020/21) and CGLS-LC100 (2015–19) because both post-date the 2004–2014 record
by a decade, over which Mahanadi irrigation expanded materially — they describe
a different catchment. MCD12Q1 also uses the IGBP legend VIC's vegetation
library is already keyed to, so no crosswalk is needed.

Access notes: MCD12Q1 ships as HDF-EOS2 (HDF4), which the installed GDAL cannot
read, so `build_landcover.py` uses `pyhdf` directly. Rather than warping the
sinusoidal grid it inverts every pixel centre to lat/lon and bins it — exact,
where a warp would blur class boundaries. Requires a free Earthdata account;
`earthaccess` handles login. Granules h25v06 and h25v07 cover the catchment
(it straddles the 20 °N tile seam); h26v07 is fetched by the bbox search and
correctly skipped as non-overlapping.

**Composition, basin-area-weighted** (what VIC actually sees):

| Class | % |
|---|---|
| cropland | 68.8 |
| grassland | 9.9 |
| deciduous broadleaf | 8.7 |
| savanna | 5.3 |
| mixed forest | 3.9 |

A predominantly agricultural catchment — the Chhattisgarh rice belt — with
forest confined to the hill margins. Note the basin-weighted cropland fraction
(68.8 %) is well above the bounding-box figure (57.9 %): the box corners pick up
forested terrain that drains elsewhere.

Verified: class fractions sum to 1.000000 everywhere; 3,299–3,390 pixels per
cell against ~3,345 expected at MODIS's true 463.31 m pixel; and the
least-vegetated cell (21.50 °N 83.75 °E, 23.2 % water) is exactly where Hirakud
reservoir sits — an independent end-to-end check on the sinusoidal inversion.

## 4. Routing

VIC produces runoff and baseflow **per cell**; it does not route them to a
gauge. Reservoir inflow needs a routing step.

✅ **Done** — `preprocessing/build_basin.py` → `basin.npz`,
`preprocessing/build_routing.py` → `routing.npz`. Figures: `basin.png`,
`routing.png`.

| Input | Status |
|---|---|
| HydroSHEDS DIR/ACC 15 arc-sec (Asia) | ✅ `data/raw/vic/routing/` |
| Basin mask, 0.25° area fractions | ✅ `basin.npz` |
| Upscaled 0.25° flow direction | ✅ `routing.npz` |
| Outlet cell | 21.50 °N, 84.00 °E (dam snapped 6.1 km onto the modelled channel) |
| Unit hydrograph | Lohmann default, via RVIC |

**Delineated area 85,215 km² against a published 83,400 km² — +2.2 %.** That is
the check that matters: a mis-snapped pour point or a wrong D8 encoding gives an
area wrong by a factor, not by 2 %.

**The rectangular box is only 45 % basin.** Of its 256 cells, 150 contain any
basin and 88 are wholly inside; the basin also reaches east to 84.79 °E, past
the box edge, so the box holds 97.8 % of the true catchment while including a
great deal that drains elsewhere. For VIC this is decisive — routing 55 %
non-contributing area would inflate discharge directly. For the *statistical*
model it turned out not to matter: basin-weighted rainfall correlates with the
box mean at 0.968 and gives no consistent gain against inflow (lag 1 d +0.014,
lag 3 d −0.023, 7 d +0.011, 30 d −0.005), because rainfall over 85,000 km² is
synoptically coherent. **Existing Ridge and CNN results therefore stand.**

### Two things that had to be got right

**Upscaling flow direction is not resampling.** A majority or nearest-neighbour
filter on D8 codes yields a plausible grid whose network is disconnected,
because direction is a property of drainage topology, not a value that
averages. `build_routing.py` follows Dominant River Tracing (Wu et al. 2011):
take each coarse cell's highest-accumulation pixel as its trunk, walk
downstream along the 15 arc-sec network until the path leaves the cell, and let
the coarse neighbour it enters define the direction.

**The trunk search must be restricted to basin pixels.** A coarse cell on the
divide contains pixels of neighbouring basins, and its highest-accumulation
pixel is often a river flowing *away* from Hirakud. Ignoring this stranded
30 of 150 cells; restricting the search fixed all 30. `basin.npz` therefore
stores the 15 arc-sec mask, not only the upscaled fractions.

Validation is by construction: following the upscaled network from every basin
cell must terminate at the outlet. **150/150 reach it**, longest path 20 cells.

**Router**: **RVIC** (`pip install rvic`), the maintained successor to the
classic Lohmann router, reads VIC 5 output directly.

## 5. Software

| Component | Status |
|---|---|
| VIC 5.0.1 classic | ✅ compiled and running |
| RVIC 1.1.0 | ✅ installed, patched, routing parameters generated |
| Calibration | ❌ reuse the differential-evolution harness in `hydrology/compare_models.py` |

## RVIC

`hydrology/vic/build_rvic_inputs.py` builds all four inputs from `routing.npz`
and `basin.npz`, writes the config, and runs `rvic parameters`. **It completes**:
150 source points, one outlet.

RVIC decodes ARCMAP D8 (1=E … 128=NE) whenever the maximum code is ≥ 10 — read
from `make_uh.py` rather than assumed. That is identical to HydroSHEDS, so no
remapping was needed.

### RVIC 1.1.0 does not run on modern Python

Its last release is 2017. `hydrology/vic/patch_rvic.py` applies seven fixes and
verifies by import; re-run it after any `pip install rvic`. Six are language or
library changes, one is an RVIC bug:

| File | Problem |
|---|---|
| `core/pycompat.py` | `SafeConfigParser` removed in Python 3.12 |
| `core/convolution_wrapper.py` | `sysconfig.get_config_var('SO')` removed in 3.8 → `EXT_SUFFIX` |
| `core/config.py` | **RVIC bug**: `isint("1.0")` is True but `int("1.0")` raises. Any whole-number float in the config crashes the parser — and `VELOCITY` is a parameter people calibrate, so `1.5` works and `2.0` does not |
| `parameters.py` | `DataFrame.ix` removed in pandas 1.0 |
| `parameters.py` | `np.float` removed in NumPy 1.24 |
| `core/param_file.py` | Python 3 true division gives float array dimensions and slice bounds (two places) |
| `core/write.py` | `stringtochar` calls `.encode()` on what is now `numpy.bytes_` |

### Flow distance must be in METRES

RVIC divides distance by `VELOCITY` (m/s) to get travel time. Written in
kilometres — as it was first — the whole 85,000 km² basin routes in minutes and
the unit hydrograph collapses entirely into day 0. The generated grid is
per-cell and direction-dependent: north–south uses 110,574 m per 0.25°,
east–west `111,320·cos(lat)`, diagonals the hypotenuse.

### Verified response

```
d0 0.071  d1 0.156  d2 0.202  d3 0.292  d4 0.214  d5 0.061  d6 0.004
peak day 3     50 % by day 3     90 % by day 4     99 % by day 5
per-source unit hydrograph sums 0.986–1.010 (mass conserved)
```

Independently consistent with the geometry: the longest path is 20 cells ×
~27 km = 540 km, which at 1 m/s is 6.25 days to the furthest cell. A 3-day peak
is right for this basin — and it is the concentration time the 1–17 day forecast
has to represent.

`VELOCITY` (1.0 m/s) and `DIFFUSION` (2000 m²/s) are RVIC defaults and are
**calibration parameters**, not measurements.

The **classic** driver was chosen over the image driver because it needs no
NetCDF — `nc-config` is absent on this machine, and the image driver would also
require MPI. Build is a plain `make` in `vic/drivers/classic`; warnings only.

## Running VIC — everything that had to be discovered

`hydrology/vic/smoke_test.py` runs the whole chain on **placeholder radiation**
so that format errors surface without waiting for the CDS queue. It found
seven blockers in sequence; each would otherwise have cost a round trip.
**It now passes**, so soil, vegetation, forcing layout and the global parameter
file are all known-good.

| # | Symptom | Cause / fix |
|---|---|---|
| 1 | `PRT_HEADER has been deprecated` | removed |
| 2 | `PRT_SNOW_BAND has been deprecated` | removed |
| 3 | `snow model steps per day (1) < minimum (4)` | VIC 5 requires ≥ 4 sub-daily snow steps **even with no snow** — the snow model always runs, it just produces zero |
| 4 | `must set QUICK_FLUX to TRUE` | required whenever `FULL_ENERGY` and `FROZEN_SOIL` are both FALSE |
| 5 | log written to `.../Desktop/untitledvic.log` | **VIC splits paths on whitespace.** This project lives under `untitled folder`, so every path was silently truncated. Fixed with a space-free symlink `~/.vic_mahanadi`. It does not error — it just uses the wrong path |
| 6 | log written to `.../logsvic.log` | `RESULT_DIR` and `LOG_DIR` need a **trailing slash**; VIC concatenates filenames directly |
| 7 | `"OUTFILE" must be specified before "OUT_FORMAT"` | `OUTFILE` opens a block; `OUT_FORMAT`, `COMPRESS`, `AGGFREQ`, `OUTVAR` all attach to the block above them |
| 8 | `model must be run at the same time step as the forcing` | with sub-daily forcing, `MODEL_STEPS_PER_DAY` must equal `FORCE_STEPS_PER_DAY` |

### The timestep is forced, not chosen

Three constraints compose to leave exactly one option:

```
snow_steps_per_day >= 4            get_global_param.c  (MIN_SUBDAILY_STEPS_PER_DAY)
FORCE_STEPS_PER_DAY == SNOW        get_global_param.c
MODEL == FORCE  (sub-daily)        read_atmos_data.c
```

So **everything runs 6-hourly**. A daily water balance driven by 6-hourly
forcing is not possible, despite `model_steps_per_day == 1` being explicitly
exempted from the SNOW check — the forcing reader rejects it later. Output is
aggregated back to daily with `AGGFREQ NDAYS 1`.

This is convenient rather than costly: 6-hourly is the **native** resolution of
the `met`, `rad` and `sp` ERA5 streams, so nothing is discarded. Only IMD
precipitation needs disaggregating, and with a daily observation there is no
sub-daily information to preserve — it is spread uniformly over the four steps
rather than given a fabricated diurnal shape.

It does mean the hourly `t2m` download was more than needed: VIC never sees
Tmax/Tmin, only `AIR_TEMP` at the model step. 6-hourly `t2m` would have cost
2 requests instead of 24. Harmless — hourly averages down exactly — but it is
the residue of the same MTCLIM misunderstanding recorded above.

### Verified behaviour

Water balance on the 2004 smoke run, catchment mean:

```
precipitation  1096.9 mm      evaporation  569.2      runoff  503.5
baseflow        106.2         storage      -80.9
residual         -1.136 mm    (0.104 % of P)          CLOSES
```

Closure is a property of the solver, not the forcing, so it is meaningful even
with fake radiation. The 0.1 % residual is canopy interception storage, which
VIC tracks separately from the soil layers. Maximum SWE is 0, as it must be.

The **partition** between ET and runoff is *not* meaningful here — constant
200 W/m² understates pre-monsoon evaporative demand, which is why the runoff
coefficient comes out at 0.56 against a published ~0.35–0.40. That number
should be revisited once real radiation is in.

---

## Realistic effort

| Stage | Effort |
|---|---|
| Acquire + regrid forcing | 1–2 days |
| Soil + vegetation parameter files | 2–3 days |
| Compile VIC, get it running | 1–2 days |
| Routing setup + basin delineation | 1–2 days |
| Calibration (many runs) | 2–5 days |

**~1–2 weeks**, against XAJ's few hours. Worth knowing before committing.

## What VIC has to beat

Same forcing, same held-out years, same metric:

| Model | Forcing | Test NSE (2014) |
|---|---|---|
| **Ridge (statistical)** | observed rainfall | **+0.761** |
| XAJ (17 params) | observed rainfall | +0.699 |
| Ridge | forecast rainfall | +0.435 |

Ambika et al. (2025) calibrated VIC on the Delaware and reported **NSE 0.52**,
which they describe as satisfactory per Moriasi et al. (2007). If VIC lands in
that range here it will *underperform* both models already built — worth
anticipating, because it is a plausible outcome rather than a pessimistic one.

The stronger case for VIC is not beating Ridge outright but supplying
**physically consistent state variables** — soil moisture, baseflow — as extra
predictors for the statistical model. That is the hybrid design in both source
papers, and it is where the gain is most likely to be real.
