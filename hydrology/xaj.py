"""Xin'anjiang (XAJ) conceptual rainfall-runoff model.

Why XAJ and not VIC.  The project brief specifies VIC, but VIC requires gridded
observed temperature, HWSD soil parameters, MODIS land cover and a fine DEM for
flow routing -- none of which this dataset contains.  Dong et al. (2025), the
paper the downscaling half of this project follows, use XAJ, and their headline
hybrid is XAJ-LSTM.  XAJ is lumped and conceptual: it needs catchment-mean
rainfall and potential evapotranspiration only, both available here, and it
answers the same question -- does explicit storage accounting beat a purely
statistical rainfall-to-inflow mapping?

Model structure (Zhao, 1992), four modules:

  1. Evapotranspiration from three tension-water layers (upper, lower, deep),
     drawn down in order.
  2. Runoff generation on the *saturation excess* principle: runoff forms only
     where tension water is already full.  Sub-catchment heterogeneity is
     represented by a distribution curve of tension-water capacity with
     exponent B, so parts of the basin saturate before others.  This is the
     mechanism that makes identical rainfall produce different runoff on a dry
     versus a wet catchment -- the Day A / Day B contrast that motivates the
     whole project.
  3. Partition of generated runoff into surface, interflow and groundwater via
     a free-water reservoir with capacity distribution exponent EX.
  4. Routing of each component through linear reservoirs with its own recession.

State is carried day to day, so antecedent wetness is represented explicitly by
storage rather than by the lagged-rainfall proxies the statistical model uses.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class XAJParams:
    """XAJ parameters with physically meaningful bounds for calibration.

    Defaults are mid-range starting values for a humid/sub-humid monsoon basin;
    they are not tuned.  `BOUNDS` is what the calibrator searches.
    """
    # Evapotranspiration
    K: float = 0.95        # ratio of potential ET to pan/reference ET
    C: float = 0.15        # deep-layer ET coefficient
    # Tension water capacities (mm)
    WUM: float = 20.0      # upper layer
    WLM: float = 80.0      # lower layer
    WDM: float = 60.0      # deep layer
    B: float = 0.3         # tension-water capacity distribution exponent
    IM: float = 0.02       # impervious area fraction
    # Free water / runoff partition
    SM: float = 30.0       # free water storage capacity (mm)
    EX: float = 1.2        # free water capacity distribution exponent
    KI: float = 0.35       # free water -> interflow coefficient
    KG: float = 0.35       # free water -> groundwater coefficient
    # Recessions (daily)
    CI: float = 0.75       # interflow storage recession
    CG: float = 0.98       # groundwater storage recession
    CS: float = 0.20       # surface routing recession
    # Potential evapotranspiration, seasonal.  Observed temperature is not
    # available, so PET is represented as an annual harmonic whose mean and
    # amplitude are calibrated.  This is a deliberate simplification: it lets
    # calibration absorb PET uncertainty rather than invent a temperature series
    # the data cannot support.
    PET_MEAN: float = 3.5  # mm/day
    PET_AMP: float = 1.5   # mm/day
    PET_PHASE: float = 120.0  # day-of-year of the PET maximum

    BOUNDS: dict = field(default_factory=lambda: {
        "K": (0.5, 1.2), "C": (0.05, 0.25),
        "WUM": (5.0, 40.0), "WLM": (40.0, 150.0), "WDM": (20.0, 150.0),
        "B": (0.1, 0.6), "IM": (0.0, 0.05),
        "SM": (5.0, 80.0), "EX": (0.8, 2.0),
        "KI": (0.05, 0.6), "KG": (0.05, 0.6),
        "CI": (0.3, 0.95), "CG": (0.9, 0.999), "CS": (0.05, 0.9),
        "PET_MEAN": (1.5, 6.0), "PET_AMP": (0.0, 3.0), "PET_PHASE": (60.0, 180.0),
    })

    NAMES = ("K", "C", "WUM", "WLM", "WDM", "B", "IM", "SM", "EX",
             "KI", "KG", "CI", "CG", "CS", "PET_MEAN", "PET_AMP", "PET_PHASE")

    @classmethod
    def from_vector(cls, v) -> "XAJParams":
        p = cls()
        for name, val in zip(cls.NAMES, v):
            setattr(p, name, float(val))
        # KI + KG must not exceed 1: together they drain the free-water store,
        # and a sum above unity would create water.  Rescale rather than reject,
        # so the calibrator sees a smooth surface instead of a wall.
        total = p.KI + p.KG
        if total > 0.95:
            p.KI *= 0.95 / total
            p.KG *= 0.95 / total
        return p

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in self.NAMES], float)

    @classmethod
    def bounds_arrays(cls):
        b = cls().BOUNDS
        lo = np.array([b[n][0] for n in cls.NAMES])
        hi = np.array([b[n][1] for n in cls.NAMES])
        return lo, hi


def seasonal_pet(doy: np.ndarray, p: XAJParams) -> np.ndarray:
    """Annual-harmonic PET standing in for a temperature-driven estimate."""
    pet = p.PET_MEAN + p.PET_AMP * np.cos(2 * np.pi * (doy - p.PET_PHASE) / 365.25)
    return np.maximum(pet, 0.1)


def run_xaj(rain: np.ndarray, doy: np.ndarray, p: XAJParams,
            area_km2: float, warmup: int = 365) -> dict:
    """Simulate daily discharge (cumecs) from catchment-mean rainfall.

    `warmup` days are simulated but should be discarded when scoring: the
    initial stores are guesses, and a monsoon basin needs roughly a year to
    forget them.
    """
    n = len(rain)
    pet = seasonal_pet(doy, p)

    WM = p.WUM + p.WLM + p.WDM
    # Start half-full: a dry start would fabricate a spurious wetting-up
    # transient, a full start a spurious flood.
    WU, WL, WD = p.WUM * 0.5, p.WLM * 0.5, p.WDM * 0.5
    S = p.SM * 0.3
    QI_store = QG_store = QS_store = 0.0

    WMM = WM * (1 + p.B) / (1 - p.IM)      # maximum point tension capacity
    SMM = p.SM * (1 + p.EX)                # maximum point free-water capacity

    q_out = np.empty(n)
    runoff_total = np.empty(n)

    for t in range(n):
        P, EP = rain[t], pet[t] * p.K

        # ---- evapotranspiration, upper -> lower -> deep
        if WU + P >= EP:
            EU, EL, ED = EP, 0.0, 0.0
        else:
            EU = WU + P
            rem = EP - EU
            if WL >= p.C * p.WLM:
                EL = rem * WL / p.WLM
                ED = 0.0
            elif WL >= p.C * rem:
                EL, ED = p.C * rem, 0.0
            else:
                EL = WL
                ED = p.C * rem - WL
        E = EU + EL + ED

        # ---- runoff generation (saturation excess on the capacity curve)
        PE = P - E
        if PE <= 0:
            R = 0.0
            WU = max(0.0, WU + P - EU)
            WL = max(0.0, WL - EL)
            WD = max(0.0, WD - ED)
        else:
            W = WU + WL + WD
            A = WMM * (1 - (1 - W / WM) ** (1 / (1 + p.B))) if W < WM else WMM
            if PE + A < WMM:
                R = PE - WM * ((1 - (PE + A) / WMM) ** (1 + p.B)
                               - (1 - A / WMM) ** (1 + p.B)) - (WM - W)
                R = max(0.0, R)
            else:
                R = max(0.0, PE - (WM - W))

            # Refill tension stores with the non-runoff part, upper first.
            infil = PE - R
            WU += P - EU
            if WU > p.WUM:
                excess = WU - p.WUM
                WU = p.WUM
                WL += excess
            WL -= EL
            if WL > p.WLM:
                excess = WL - p.WLM
                WL = p.WLM
                WD += excess
            WD -= ED
            WD = min(WD, p.WDM)
            WU, WL, WD = max(WU, 0.0), max(WL, 0.0), max(WD, 0.0)
            _ = infil

        # ---- partition runoff into surface / interflow / groundwater
        if R > 0:
            AU = SMM * (1 - (1 - S / p.SM) ** (1 / (1 + p.EX))) if S < p.SM else SMM
            if R + AU < SMM:
                RS = R - p.SM * ((1 - (R + AU) / SMM) ** (1 + p.EX)
                                 - (1 - AU / SMM) ** (1 + p.EX)) - (p.SM - S)
                RS = max(0.0, RS)
            else:
                RS = max(0.0, R - (p.SM - S))
            S += R - RS
            S = min(S, p.SM)
            RI, RG = S * p.KI, S * p.KG
            S -= RI + RG
            S = max(S, 0.0)
        else:
            RS = RI = RG = 0.0

        # ---- routing: each component through its own linear reservoir
        QS_store = p.CS * QS_store + (1 - p.CS) * RS
        QI_store = p.CI * QI_store + (1 - p.CI) * RI
        QG_store = p.CG * QG_store + (1 - p.CG) * RG

        depth_mm = QS_store + QI_store + QG_store
        runoff_total[t] = depth_mm
        # mm/day over the catchment -> cumecs
        q_out[t] = depth_mm * area_km2 * 1000.0 / 86400.0

    return {"q_cumecs": q_out, "runoff_mm": runoff_total, "warmup": warmup}
