"""
Physical film emulation pipeline (unpaired, real-time).

This replaces the old multi-LUT fusion approach with a physically-decomposed pipeline.

Stage 1: Exposure / log conversion        — ExposureModule
Stage 2: Spectral sensitivity (3x3 matrix) — SpectralMatrix
Stage 3: Per-channel TRC (1D LUT)         — ToneResponseCurve
Stage 4: Residual crossover (small 3D LUT) — ResidualLUT3D
Stage 5: Halation (bright glow)           — Halation
Stage 6: Grain (spatial noise)            — Grain

All stages are differentiable and designed for real-time inference.
The first four stages are per-pixel; stages 5–6 are spatial.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lut import (
    Generator3DLUT_rand,
    TV3D,
    trilinear_port,
)


# ---------------------------------------------------------------------------
# Stage 1: Exposure / log conversion
# ---------------------------------------------------------------------------

class ExposureModule(nn.Module):
    """Per-channel exposure compensation in log domain.

    Computes  x' = clamp(x * 2^EV + bias, 0, 1)
    where EV (per-channel) and bias are learnable scalars.
    Acts as a global gain before the spectral stage.
    """

    def __init__(self):
        super().__init__()
        # per-channel EV in stops, init 0
        self.ev = nn.Parameter(torch.zeros(3))
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gain = torch.pow(2.0, self.ev)            # [3]
        x = x * gain.view(1, 3, 1, 1) + self.bias.view(1, 3, 1, 1)
        return x.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Stage 2: Spectral sensitivity matrix
# ---------------------------------------------------------------------------

class SpectralMatrix(nn.Module):
    """Learnable 3x3 color-mixing matrix + per-channel offset.

    Models the linear part of dye-coupling / spectral overlap.
    Near-identity init to keep training stable.
    """

    def __init__(self):
        super().__init__()
        # init to identity
        self.weight = nn.Parameter(torch.eye(3))
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W] -> apply 3x3 to channel dim
        out = torch.einsum('bchw,co->bohw', x, self.weight)
        out = out + self.bias.view(1, 3, 1, 1)
        return out.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Stage 3: Tone Response Curve (per-channel 1D LUT)
# ---------------------------------------------------------------------------

class ToneResponseCurve(nn.Module):
    """Per-channel 1D LUT with configurable control points.

    Initialises to a soft S-curve approximating H&D response.
    Control points are learnable; TV-1D regularity is exposed via tv_loss().
    """

    def __init__(self, num_points: int = 32):
        super().__init__()
        self.num_points = num_points

        # initialise to a sigmoid-ish S-curve in [0,1]
        x_grid = torch.linspace(0, 1, num_points)
        # toe + linear mid + shoulder
        init_curve = 0.5 - 0.5 * torch.cos(x_grid * math.pi)   # smoothstep
        # blend slightly toward linear for starting point
        init_curve = 0.7 * init_curve + 0.3 * x_grid
        self.curve = nn.Parameter(init_curve.unsqueeze(0).repeat(3, 1))  # [3, P]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x in [0,1], shape [B,3,H,W]
        B, C, H, W = x.shape

        # Process per channel to avoid complex indexing
        results = []
        for c in range(C):
            c_x = x[:, c]  # [B, H, W]
            c_curve = self.curve[c]  # [num_points]

            # Map [0,1] to curve indices [0, num_points-1]
            pos = c_x * (self.num_points - 1)  # [B, H, W]

            # Get lower and upper indices
            idx0 = pos.floor().long().clamp(0, self.num_points - 2)  # [B, H, W]
            idx1 = idx0 + 1  # [B, H, W]

            # Flatten for indexing
            c_x_flat = c_x.reshape(B * H * W)  # [B*H*W]
            idx0_flat = idx0.reshape(B * H * W)  # [B*H*W]
            idx1_flat = idx1.reshape(B * H * W)  # [B*H*W]

            # Gather curve values
            val0 = c_curve[idx0_flat]  # [B*H*W]
            val1 = c_curve[idx1_flat]  # [B*H*W]

            # Compute interpolation coefficient = fractional part of position [0, 1]
            alpha = pos - pos.floor()  # [B, H, W]
            alpha = alpha.reshape(B * H * W)  # [B*H*W]

            # Interpolate
            c_out = val0 + alpha * (val1 - val0)  # [B*H*W]
            results.append(c_out.reshape(B, H, W))  # [B, H, W]

        out = torch.stack(results, dim=1)  # [B, C, H, W]
        return out

    def tv_loss(self) -> torch.Tensor:
        """Second-order smoothness on each channel's curve."""
        d2 = self.curve[:, 2:] - 2 * self.curve[:, 1:-1] + self.curve[:, :-2]
        return (d2 ** 2).mean()


# ---------------------------------------------------------------------------
# Stage 4: Residual 3D LUT (crossover that linear matrix cannot capture)
# ---------------------------------------------------------------------------

class ResidualLUT3D(nn.Module):
    """Small 3D LUT (default 17^3) initialised near-identity to capture
    only the residual non-linear crossover.

    The LUT is initialized as an identity mapping (vertex [i,j,k] maps to
    (i/(d-1), j/(d-1), k/(d-1))) plus tiny random noise, so at init the
    residual is near-zero and the output equals the input.
    """

    def __init__(self, dim: int = 17):
        super().__init__()
        self.dim = dim
        self.lut = Generator3DLUT_rand(dim=dim)

        # Replace random init with identity + tiny noise
        with torch.no_grad():
            identity = self._make_identity_lut(dim)
            noise = torch.randn(3, dim, dim, dim, dtype=identity.dtype) * 0.001
            self.lut.LUT.copy_(identity + noise)

        self.tv_module = TV3D(dim=dim)

    @staticmethod
    def _make_identity_lut(dim: int) -> torch.Tensor:
        """Create an identity 3D LUT of shape [3, dim, dim, dim]."""
        steps = torch.linspace(0, 1, steps=dim)
        lut = torch.empty(3, dim, dim, dim)
        lut[0] = steps.unsqueeze(0).unsqueeze(0).expand(dim, dim, dim)   # R axis
        lut[1] = steps.unsqueeze(-1).unsqueeze(0).expand(dim, dim, dim)  # G axis
        lut[2] = steps.unsqueeze(-1).unsqueeze(-1).expand(dim, dim, dim)  # B axis
        return lut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply trilinear interpolation via custom kernel
        # Generator3DLUT.forward already does trilinear_port.apply(LUT, x)
        out = self.lut(x)
        # residual: input + (lut_out - input) => keeps identity at init
        return x + (out - x)

    def reg_loss(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tv_module(self.lut)


# ---------------------------------------------------------------------------
# Stage 5: Halation
# ---------------------------------------------------------------------------

class Halation(nn.Module):
    """Bright-pixel glow with warm tint.

    1. Extract luminance above threshold.
    2. Gaussian blur with learnable sigma.
    3. Tint warm (more R/G, less B) and add back to image.

    All learnable: threshold, sigma, strength, tint.
    """

    def __init__(self):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(0.80))   # 0..1
        self.log_sigma = nn.Parameter(torch.tensor(2.0))    # sigma ~7.4 px
        self.strength = nn.Parameter(torch.tensor(0.15))
        # warm tint [R, G, B]
        self.tint = nn.Parameter(torch.tensor([1.0, 0.85, 0.55]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # luminance
        lum = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        # bright mask (soft threshold via sigmoid for differentiability)
        mask = torch.sigmoid((lum - self.threshold) * 10.0)   # [B,H,W]
        mask = mask.unsqueeze(1)                              # [B,1,H,W]

        # blurred bright channel (per-channel blur)
        sigma = torch.exp(self.log_sigma)
        radius = max(1, int(sigma.item() * 3))
        # build 1D gaussian kernel
        coords = torch.arange(-radius, radius + 1,
                              device=x.device, dtype=x.dtype)
        k1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        k1d = k1d / k1d.sum()
        # separable conv
        k_h = k1d.view(1, 1, 1, -1)
        k_v = k1d.view(1, 1, -1, 1)
        glow = mask * x                                       # [B,3,H,W]
        pad = (radius, radius, 0, 0)
        glow = F.pad(glow, pad, mode='reflect')
        glow = F.conv2d(glow, k_h.expand(3, -1, -1, -1), groups=3)
        pad = (0, 0, radius, radius)
        glow = F.pad(glow, pad, mode='reflect')
        glow = F.conv2d(glow, k_v.expand(3, -1, -1, -1), groups=3)

        # tint and add
        tint = torch.sigmoid(self.tint).view(1, 3, 1, 1)
        glow = glow * tint
        out = x + self.strength * glow
        return out.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Stage 6: Grain
# ---------------------------------------------------------------------------

class Grain(nn.Module):
    """Film grain via parameterised noise, modulated by local luminance.

    Generates a noise tile per forward pass; variance is higher in
    shadows (negative film behaviour).  A small conv refines the noise
    to have spatial correlation similar to silver halide clusters.

    All learnable: base intensity, shadow boost, grain size (via conv).
    """

    def __init__(self):
        super().__init__()
        self.intensity = nn.Parameter(torch.tensor(0.02))
        self.shadow_boost = nn.Parameter(torch.tensor(2.0))
        # single 3x3 conv to give noise spatial structure
        self.shaper = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            self.shaper.weight.fill_(1.0 / 9.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # generate noise at downsampled resolution then upsample (cheaper, softer)
        scale = 2
        hs, ws = H // scale, W // scale
        noise = torch.randn(B, 1, hs, ws, device=x.device, dtype=x.dtype)
        noise = self.shaper(noise)
        noise = F.interpolate(noise, size=(H, W), mode='bilinear',
                              align_corners=False)
        # normalise
        noise = noise / (noise.std() + 1e-6)

        # luminance-dependent modulation
        lum = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        # more grain in dark areas: weight ~ (1 - lum)^shadow_boost
        mod = torch.pow(1.0 - lum.clamp(0, 1),
                        torch.sigmoid(self.shadow_boost) * 3.0)
        grain = self.intensity * noise * mod
        out = x + grain
        return out.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class FilmPipeline(nn.Module):
    """Full physical film emulation pipeline.

    Forward returns a dict with the output image plus per-stage
    regularisation tensors for the trainer.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # per-pixel stages
        self.exposure = ExposureModule()
        self.spectral = SpectralMatrix()
        self.trc = ToneResponseCurve(
            num_points=getattr(config, "trc_points", 32)
        )
        self.residual_lut = ResidualLUT3D(
            dim=getattr(config, "residual_lut_dim", 17)
        )

        # spatial stages
        self.halation = Halation()
        self.grain = Grain()

        # switches
        self.use_halation = getattr(config, "use_halation", True)
        self.use_grain = getattr(config, "use_grain", True)

    def forward(self, x: torch.Tensor) -> dict:
        out = self.exposure(x)
        out = self.spectral(out)
        out = self.trc(out)
        out = self.residual_lut(out)

        tv_res, mn_res = self.residual_lut.reg_loss()
        tv_trc = self.trc.tv_loss()

        if self.use_halation:
            out = self.halation(out)
        if self.use_grain:
            out = self.grain(out)

        # Safety clamp to ensure output is in valid range
        out = out.clamp(0.0, 1.0)

        return {
            "out": out,
            "tv_res": tv_res,
            "mn_res": mn_res,
            "tv_trc": tv_trc,
        }


# ---------------------------------------------------------------------------
# Backward compatibility alias
# ---------------------------------------------------------------------------

# Alias for backward compatibility - the old name from the multi-LUT fusion approach
digitalFilm_G = FilmPipeline
