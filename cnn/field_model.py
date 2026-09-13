"""Field-level downscaler: predict the whole 25x25 rainfall map in one pass.

The per-cell model (PatchDataset + ResNetDownscaler / GBM) predicts each 0.25 deg
cell independently from its own 3x3 coarse patch.  Rainfall fields are spatially
correlated -- a monsoon depression covers many cells at once -- so independent
prediction throws away structure that is genuinely informative, and produces
fields that need not be spatially coherent.

This model consumes the whole coarse field plus full-resolution static and
antecedent fields, and emits the whole fine field jointly.  Every output cell
therefore sees its neighbours through the convolution stack.

Shape note: the fine grid is 25x25, which is small.  A deep U-Net with several
downsampling stages would reduce it to a couple of pixels and destroy the
spatial detail that is the point of the exercise, so this uses ONE down/up level
(25 -> 13 -> 25) with a skip connection.  That buys a multi-scale receptive
field -- enough to see a synoptic system across the domain -- without collapsing
the resolution.  Depth here would be cargo-culting a design meant for 256x256
imagery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with GroupNorm and residual connection.

    GroupNorm rather than BatchNorm: batches are small (tens of maps) and each
    sample is a whole weather field, so batch statistics are noisy and
    correlated across the batch -- exactly the regime where BatchNorm degrades.
    """

    def __init__(self, c_in: int, c_out: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, c_out), c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, c_out), c_out)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        h = F.elu(self.norm1(self.conv1(x)))
        h = self.drop(h)
        h = self.norm2(self.conv2(h))
        return F.elu(h + self.skip(x))


class FieldDownscaler(nn.Module):
    """(coarse fields, fine static/antecedent fields) -> full 0.25 deg rainfall map.

    The coarse ECMWF grid is upsampled to the fine grid before fusion, rather
    than the network learning a transposed-convolution upsampler: with only
    ~4.2k training fields, a learned upsampler from 5x5 is mostly free
    parameters, and bilinear interpolation already encodes the right prior that
    a coarse cell's value applies smoothly across its footprint.
    """

    def __init__(self, n_coarse: int, n_fine_static: int, fine_hw: int = 25,
                 width: int = 64, dropout: float = 0.1, residual: bool = True,
                 levels: int = 1):
        super().__init__()
        self.fine_hw = fine_hw
        self.residual = residual
        self.levels = levels

        self.coarse_encode = nn.Sequential(
            nn.Conv2d(n_coarse, width, 3, padding=1), nn.ELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ELU(),
        )
        self.fuse = ConvBlock(width + n_fine_static, width, dropout)

        self.down = ConvBlock(width, width * 2, dropout)
        self.up = ConvBlock(width * 2 + width, width, dropout)
        if levels >= 2:
            # 25 -> 12 -> 6 -> 12 -> 25.  The docstring argues one level is
            # right; this makes that an ablation rather than an assertion.
            self.down2 = ConvBlock(width * 2, width * 4, dropout)
            self.up2 = ConvBlock(width * 4 + width * 2, width * 2, dropout)
        self.head = nn.Conv2d(width, 1, 1)

        if residual:
            # Start the network at "interpolated ECMWF rainfall, unchanged" and
            # let it learn the correction, rather than reconstructing the field
            # from scratch.  Zero-initialising the head makes the untrained
            # model an exact pass-through, so early training cannot be worse
            # than the interpolation baseline -- useful with only ~4.2k fields.
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            self.tp_gain = nn.Parameter(torch.ones(1))

    def forward(self, coarse, fine_static, tp_interp=None):
        h = self.coarse_encode(coarse)
        h = F.interpolate(h, size=(self.fine_hw, self.fine_hw),
                          mode="bilinear", align_corners=False)
        h = self.fuse(torch.cat([h, fine_static], dim=1))

        # One down/up level for a wider receptive field, with a skip so
        # fine-scale detail survives the round trip.
        d = self.down(F.avg_pool2d(h, 2))
        if self.levels >= 2:
            d2 = self.down2(F.avg_pool2d(d, 2))
            d2 = F.interpolate(d2, size=d.shape[-2:], mode="bilinear", align_corners=False)
            d = self.up2(torch.cat([d2, d], dim=1))
        d = F.interpolate(d, size=h.shape[-2:], mode="bilinear", align_corners=False)
        h = self.up(torch.cat([d, h], dim=1))

        out = self.head(h).squeeze(1)
        if self.residual:
            if tp_interp is None:
                raise ValueError("residual=True requires tp_interp (raw mm, fine grid)")
            # Added inside softplus so the output stays non-negative while the
            # correction remains free to reduce rainfall as well as increase it.
            out = out + self.tp_gain * tp_interp
        return F.softplus(out)


def masked_field_loss(pred, obs, mask, threshold, a: float = 1.0, b: float = 1.0,
                      c: float = 0.0, catchment=None):
    """MSE + b*(1 - differentiable threat score) + c*MSE(catchment mean).

    Same objective as the per-cell model so the comparison is like-for-like;
    only the unit of prediction changed.  Reductions are over all valid cells in
    the batch, which weights each cell equally regardless of how many valid
    cells its field happens to contain.

    `c` > 0 adds a term on the CATCHMENT-MEAN rainfall.  The inflow stage
    consumes only that mean over the 224 catchment cells -- never the individual
    cells -- so per-cell MSE optimises a quantity the pipeline does not use.
    Independent per-cell errors partly cancel on averaging (per-cell R^2 ~0.10
    becomes catchment correlation ~0.51), and nothing in the objective rewards
    getting that cancellation right.  This term does.
    """
    m = mask.float()
    n = m.sum().clamp(min=1.0)
    mse = (((pred - obs) ** 2) * m).sum() / n

    obs_wet = (obs > threshold).float()
    fc_wet = torch.sigmoid(a * (pred - threshold))
    fc_dry = torch.sigmoid(-a * (pred - threshold))
    hits = (obs_wet * fc_wet * m).sum()
    false_alarms = ((1 - obs_wet) * fc_wet * m).sum()
    misses = (obs_wet * fc_dry * m).sum()
    ts = hits / (hits + false_alarms + misses + 1e-6)

    total = mse + b * (1 - ts)
    parts = {"mse": mse.detach(), "ts": ts.detach()}

    if c > 0.0 and catchment is not None:
        cm = catchment.float() * m                      # observed catchment cells
        cn = cm.flatten(1).sum(1).clamp(min=1.0)
        p_mean = (pred * cm).flatten(1).sum(1) / cn
        o_mean = (obs * cm).flatten(1).sum(1) / cn
        cat_mse = ((p_mean - o_mean) ** 2).mean()
        total = total + c * cat_mse
        parts["cat_mse"] = cat_mse.detach()

    return total, parts
