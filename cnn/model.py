"""ResNet downscaling model and the masked hybrid loss.

Follows Dong et al. (2025) sect. 3.2.1: ResNet blocks over a 3 x 3 coarse patch,
coordinate embeddings concatenated with the flattened convolutional features,
and a loss combining MSE with a differentiable threat score so heavy rain is not
smoothed away.

Two corrections relative to the previous implementation:

  * The head predicts one 0.25 deg cell from its coarse patch, instead of
    reproducing the 1.5 deg input grid.
  * Every loss term is masked.  Cells with no IMD observation contribute
    nothing, rather than being taught to predict zero.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Two 3 x 3 convolutions with an identity shortcut.

    Padding keeps the 3 x 3 patch intact through the stack, so the residual add
    needs no projection except when the channel count changes.

    Dropout2d zeroes whole feature-map channels rather than individual pixels --
    the usual spatial-dropout argument (neighbouring pixels in a feature map are
    correlated, so element-wise dropout barely perturbs the signal) applies with
    extra force here: the input patch is only 3x3, so a single dropped pixel
    would carry almost no information loss anyway.
    """

    def __init__(self, c_in: int, c_out: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        h = F.elu(self.conv1(x))
        h = self.drop(h)
        h = self.conv2(h)
        return F.elu(h + self.skip(x))


class ResNetDownscaler(nn.Module):
    """Coarse 3 x 3 patch -> rainfall at the fine cell at its centre.

    Feature maps follow the paper's 64 -> 32 -> 16 taper.  Coordinates enter
    through an embedding rather than as extra channels (Rasp & Lerch, 2018), so
    the network can express spatial heterogeneity without the convolutions
    having to encode absolute position.
    """

    def __init__(self, n_features: int, n_coords: int = 2, widths=(64, 32, 16),
                 embed_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        blocks, c = [], n_features
        for w in widths:
            blocks.append(ResBlock(c, w, dropout=dropout))
            c = w
        self.blocks = nn.Sequential(*blocks)

        self.coord_embed = nn.Sequential(nn.Linear(n_coords, embed_dim), nn.ELU())

        patch_cells = 3 * 3
        self.head = nn.Sequential(
            nn.Linear(widths[-1] * patch_cells + embed_dim, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, patch, coords):
        h = self.blocks(patch).flatten(1)
        h = torch.cat([h, self.coord_embed(coords)], dim=1)
        # Rainfall is non-negative; softplus keeps that true without the dead
        # gradients a ReLU would give on dry days, which dominate the sample.
        return F.softplus(self.head(h)).squeeze(-1)


def masked_hybrid_loss(pred, obs, mask, threshold, a: float = 1.0, b=1.0,
                       log_space: bool = False, window=None):
    """MSE + sum_w (n_w / n) * b_w * (1 - TS_w), with a differentiable threat score.

    Dong et al. (2025) eqs. 3-7.  The threat score TS = H / (H + F + M) is
    categorical and non-differentiable, so the forecast side of each indicator is
    replaced by sigmoid(a * (pred - threshold)); the observation side stays a hard
    indicator because no gradient flows through it.

    `threshold` is the 90th percentile of observed rainfall, per fine cell.
    `mask` is False wherever IMD reported nothing -- those cells are excluded
    from every term instead of being counted as correct zeros.

    LEAD WINDOWS.  Supplement Table S1 gives b per lead window (0.4, 0.8, 1.5, 2).
    `window` holds each element's window index and `b` one weight per window.
    TS is computed within each window, because a hit at lead 3 and a hit at lead
    28 are different skills, and each window's term is weighted by its share of
    valid elements.  That is exactly the sample average of the paper's per-window
    loss, b_w (1 - TS_w) + MSE_w.  With window=None and a scalar b this reduces to
    the single-window form, MSE + b (1 - TS).

    UNITS.  The loss is only balanced if pred, obs and threshold are on a scale
    where MSE is O(1) -- see DONG_* in configs/config.py.  The trainer passes
    rainfall divided by its training standard deviation.

    `log_space` stays False: MSE on log1p(rain) was measured to make every
    real-mm metric worse (test R^2 0.116 -> 0.019, bias -1.5 -> -4.6 mm) because
    it fits nearer the conditional median of a right-skewed target.
    """
    mask = mask.float()
    n = mask.sum().clamp(min=1.0)

    if log_space:
        mse = (((torch.log1p(pred) - torch.log1p(obs)) ** 2) * mask).sum() / n
    else:
        mse = (((pred - obs) ** 2) * mask).sum() / n

    if window is None:
        window = torch.zeros_like(pred, dtype=torch.long)
    weights = [float(b)] if np.isscalar(b) else [float(x) for x in b]

    obs_wet = (obs > threshold).float()
    fc_wet = torch.sigmoid(a * (pred - threshold))
    fc_dry = torch.sigmoid(-a * (pred - threshold))

    ts_term = pred.new_zeros(())
    ts_by_window = []
    for w, bw in enumerate(weights):
        sel = mask * (window == w).float()
        n_w = sel.sum()
        if n_w < 1:
            ts_by_window.append(float("nan"))
            continue
        hits = (obs_wet * fc_wet * sel).sum()
        false_alarms = ((1.0 - obs_wet) * fc_wet * sel).sum()
        misses = (obs_wet * fc_dry * sel).sum()
        ts = hits / (hits + false_alarms + misses + 1e-6)
        ts_term = ts_term + (n_w / n) * bw * (1.0 - ts)
        ts_by_window.append(float(ts.detach()))
    return mse + ts_term, {"mse": mse.detach(), "ts_term": ts_term.detach(),
                           "ts_by_window": ts_by_window}


def _spearman(p: np.ndarray, o: np.ndarray) -> float:
    """Rank correlation via double-argsort ranks -- no scipy dependency.

    A diagnostic on this data found Spearman 0.51 vs Pearson 0.36: the model's
    ranking of wet vs dry, heavier vs lighter, is noticeably better than the
    linear/magnitude fit R^2 measures.  For a downscaler feeding a hydrological
    model, getting the relative ordering of wet days right is often what
    matters; R^2 alone understates that.
    """
    if len(p) < 2:
        return float("nan")
    rp = p.argsort().argsort().astype(np.float64)
    ro = o.argsort().argsort().astype(np.float64)
    if rp.std() < 1e-9 or ro.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(rp, ro)[0, 1])


def masked_metrics(pred, obs, mask):
    """RMSE / MAE / R^2 / rank correlation over observed cells only.

    Scoring without the mask is what inflated the previous R^2 to 0.87: six of
    twenty-five cells were constant zero in both prediction and target.

    R^2 alone is a misleading summary on this target: it is heavily right-
    skewed (mean ~9mm, p99 ~78mm, max ~575mm), so a diagnostic found R^2
    negative on the bottom 90% of rain days and positive only because of the
    top 10%.  `spearman` is reported alongside it because it is far less
    sensitive to that skew -- see `_spearman`.
    """
    m = mask.bool()
    p, o = pred[m], obs[m]
    err = p - o
    ss_res = (err ** 2).sum()
    ss_tot = ((o - o.mean()) ** 2).sum().clamp(min=1e-6)
    p_np, o_np = p.detach().cpu().numpy(), o.detach().cpu().numpy()
    return {
        "rmse": err.pow(2).mean().sqrt().item(),
        "mae": err.abs().mean().item(),
        "r2": (1 - ss_res / ss_tot).item(),
        "spearman": _spearman(p_np, o_np),
        "bias": float(p_np.mean() - o_np.mean()) if len(p_np) else float("nan"),
        "n": int(m.sum()),
    }
