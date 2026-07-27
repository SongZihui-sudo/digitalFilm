"""
Physical film emulation pipeline (unpaired, real-time).

This replaces the old multi-LUT fusion approach with a physically-decomposed pipeline.

v2 — Phos-inspired enhancements:
  * PyramidBloom (Gaussian-pyramid multi-scale halation)
  * Density-domain crosstalk as an alternative to linear SpectralMatrix
  * Per-channel / wavelength-dependent scattering radii
  * Pipeline reorder: bloom applied in the linear domain, before TRC

Stage 1: Exposure / log conversion            — ExposureModule
Stage 2: Spectral sensitivity (3x3 matrix)     — SpectralMatrix  (linear domain)
                                            or — DensityCrosstalk (density domain, Phos-style)
Stage 3: Pyramid Bloom (multi-scale halation)  — PyramidBloom   (linear domain)
Stage 4: Per-channel TRC (1D LUT)              — ToneResponseCurve
Stage 5: Residual crossover (small 3D LUT)     — ResidualLUT3D
Stage 6: Grain (spatial noise)                 — Grain

All stages are differentiable and designed for real-time inference.
Stages 1-3 are in the linear domain; stages 4-6 are in the display / curve domain.
"""

import math
from typing import cast
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
# Stage 2b: Density-domain crosstalk + nonlinear density LUT  (Phos-inspired)
# ---------------------------------------------------------------------------

class DensityCrosstalk(nn.Module):
    """Crosstalk in the density domain + nonlinear density residual LUT.

    This module replaces BOTH the old SpectralMatrix AND the old ResidualLUT3D
    with a unified physically-motivated colour module.

    Phos insight:
        Film dyes mix in DENSITY space, not linear space.
        D = -log₁₀(E)  →  dye mixing  →  T = 10^(-D)

    This module does:
        1)  Linear exposure → density:          D = -log₁₀(E + ε)
        2)  3×3 linear crosstalk in density      (additive dye densities)
        3)  Nonlinear residual via a small 3D LUT in density domain
            (captures dye coupling non-idealities)
        4)  Density → transmittance:             T = 10^(-D)

    The 3D density LUT is initialised near-identity so at init the entire
    module behaves like the Phos linear crosstalk model; training gradually
    learns the nonlinear dye interactions.

    To keep training stable, density values are clamped to a reasonable range
    before entering the 3D LUT.
    """

    def __init__(self, eps: float = 1e-6, density_lut_dim: int = 9):
        super().__init__()
        self.eps = eps

        # ---- linear crosstalk matrix (density domain) ----
        weight_init = 0.99 * torch.eye(3) + 0.01 * torch.randn(3, 3)
        self.weight = nn.Parameter(weight_init)
        self.density_bias = nn.Parameter(torch.zeros(3))

        # ---- small 3D LUT in density domain (nonlinear residual) ----
        # dim=9 gives 9³=729 vertices — very small, but enough for residual
        self.density_lut_dim = density_lut_dim
        self.density_lut = Generator3DLUT_rand(dim=density_lut_dim)

        # Init as identity in density domain
        with torch.no_grad():
            identity = self._make_identity_lut(density_lut_dim)
            noise = torch.randn(3, density_lut_dim, density_lut_dim, density_lut_dim,
                                dtype=identity.dtype) * 0.001
            self.density_lut.LUT.copy_(identity + noise)

        self.tv_module = TV3D(dim=density_lut_dim)

    @staticmethod
    def _make_identity_lut(dim: int) -> torch.Tensor:
        steps = torch.linspace(0, 1, steps=dim)
        lut = torch.empty(3, dim, dim, dim)
        lut[0] = steps.unsqueeze(0).unsqueeze(0).expand(dim, dim, dim)
        lut[1] = steps.unsqueeze(-1).unsqueeze(0).expand(dim, dim, dim)
        lut[2] = steps.unsqueeze(-1).unsqueeze(-1).expand(dim, dim, dim)
        return lut

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            out:      image after density-domain processing  [B,3,H,W]
            tv_loss:  TV smoothness of the 3D density LUT    scalar
            mn_loss:  monotonicity of the 3D density LUT    scalar
        """
        B, C, H, W = x.shape

        # 1) Linear → density
        x_safe = x.clamp(min=self.eps)
        density = -torch.log10(x_safe)                            # [B,3,H,W]

        # 2) Linear crosstalk in density domain
        density_mixed = torch.einsum('bchw,co->bohw', density, self.weight)
        density_mixed = density_mixed + self.density_bias.view(1, 3, 1, 1)

        # 3) Nonlinear residual via 3D LUT in density domain
        #    The LUT expects [0,1] input → we remap density to [0,1]
        #    Typical density range for neg film is ~0.1 to ~2.5
        #    We clamp to [0, 2.5] and map to [0,1]
        density_clamped = density_mixed.clamp(0.0, 2.5) / 2.5

        # Apply 3D LUT residual in density domain
        density_lut_out = cast(torch.Tensor, trilinear_port.trilinearPort.apply(
            self.density_lut.LUT, density_clamped
        ))

        # Residual connection: keep identity at init
        density_final = density_mixed + (density_lut_out * 2.5 - density_clamped * 2.5)
        # Ensure non-negative
        density_final = density_final.clamp(min=0.0)

        # 4) Density → transmittance
        out = torch.pow(10.0, -density_final)
        out = out.clamp(0.0, 1.0)

        # Compute regularisation losses
        tv_loss, mn_loss = self.tv_module(self.density_lut)

        return out, tv_loss, mn_loss


# ---------------------------------------------------------------------------
# Stage 2a: Spectral sensitivity matrix (linear domain, lightweight alternative)
# ---------------------------------------------------------------------------

class SpectralMatrix(nn.Module):
    """Learnable 3x3 color-mixing matrix + per-channel offset.

    Models the linear part of dye-coupling / spectral overlap.
    Near-identity init to keep training stable.

    Note: This is kept as a lightweight alternative to DensityCrosstalk
    when speed is preferred over physical accuracy.
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
        return out


# ---------------------------------------------------------------------------
# Stage 3: Pyramid Bloom  (Phos-inspired, fully differentiable)
# ---------------------------------------------------------------------------

class PyramidBloom(nn.Module):
    """Multi-scale bloom via Gaussian pyramid with wavelength-dependent radius.

    Replaces the original threshold-based Halation with a physically-motivated
    approach: build a pyramid, weight each level, and accumulate scattered light.

    Concepts adapted from Phos (ZacharyHu0):
        - Downsample → Gaussian pyramid
        - Each level models a different scatter distance
        - Per-channel radius_mult controls wavelength dependency:
            Red   (λ ≈ 650 nm) → radius_mult > 1.0 (scatters most)
            Green (λ ≈ 550 nm) → radius_mult ≈ 1.0
            Blue  (λ ≈ 450 nm) → radius_mult < 1.0 (scatters least)
        - Additive composite: result = direct * img  +  scatter * bloom

    All parameters are learnable so the model can adapt during training.
    """

    def __init__(self, levels: int = 6):
        super().__init__()
        self.levels = levels

        # Per-channel direct/scatter ratios (initialised to typical film values)
        self.direct = nn.Parameter(torch.tensor([0.95, 0.85, 0.90]))    # R, G, B
        self.scatter = nn.Parameter(torch.tensor([0.20, 0.15, 0.15]))   # R, G, B

        # Wavelength-dependent radius multiplier
        #   Red scatters most   → 1.0 ~ 1.8
        #   Blue scatters least → 0.7 ~ 0.9
        self.log_radius_mult = nn.Parameter(torch.tensor([
            math.log(1.5),    # Red
            math.log(1.0),    # Green
            math.log(0.7),    # Blue
        ]))

        # Base layer weights (0 = lowest level / smallest scale)
        # Phos uses: [0.0, 0.1, 0.2, 0.4, 0.8, 1.0, 1.0] for 7 levels
        self.layer_weights = nn.Parameter(torch.tensor(
            [0.0, 0.1, 0.2, 0.4, 0.8, 1.0, 1.0][:levels + 1],
            dtype=torch.float32,
        ))

        # Global bloom strength scale
        self.strength = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Linear RGB image [B, 3, H, W], values in [0, 1]
        Returns:
            Image with bloom added, clamped to [0, 1]
        """
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype
        radius_mult = torch.exp(self.log_radius_mult)        # [3]

        # ---- per-channel bloom ----
        bloom_channels = []
        for c in range(C):
            ch = x[:, c:c + 1]                               # [B, 1, H, W]
            rm = radius_mult[c].item()

            # 1) Build Gaussian pyramid via average pooling
            pyramid = [ch]
            current = ch
            for _ in range(self.levels):
                current = F.avg_pool2d(current, kernel_size=2, stride=2)
                pyramid.append(current)

            # 2) Upsample and accumulate with weighted contribution
            accum = torch.zeros_like(ch)
            for i in range(1, len(pyramid)):
                base_w = self.layer_weights[i]
                # Wavelength-dependent weight modulation:
                #   Larger radius_mult boosts higher (coarser) levels
                w = base_w * (rm ** (i * 0.5))

                layer_upscaled = F.interpolate(
                    pyramid[i], size=(H, W),
                    mode='bilinear', align_corners=False,
                )
                accum = accum + layer_upscaled * w

            bloom_channels.append(accum)

        bloom = torch.cat(bloom_channels, dim=1)             # [B, 3, H, W]

        # 3) Scale bloom and composite
        bloom = bloom * (torch.sigmoid(self.strength) * 0.15)

        direct_w = torch.sigmoid(self.direct).view(1, 3, 1, 1)
        scatter_w = torch.sigmoid(self.scatter).view(1, 3, 1, 1)

        out = x * direct_w + bloom * scatter_w
        return out.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Stage 4: Tone Response Curve (per-channel 1D LUT)
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
# Stage 5: Residual 3D LUT (crossover that linear matrix cannot capture)
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
# Stage 6: Grain (v2 — smaller, more organic, bounded)
# ---------------------------------------------------------------------------

class Grain(nn.Module):
    """Film grain via parameterised noise, modulated by local luminance.

    v2 improvements:
    * Noise generated at reduced resolution then upsampled → organic grain,
      no single-pixel sparkle.
    * Gaussian-shaped shaper kernel instead of box-filter (no blocky artifacts).
    * Per-channel correlated noise (luminance-grain, not RGB-independent)
      to avoid colour speckle.
    * Intensity bounded via log-param → can't grow unbounded during GAN training.
    * Modulation soft-clamped so deep shadows don't get unnaturally chunky grain.

    All parameters are learnable.
    """

    def __init__(self):
        super().__init__()
        # log-space so intensity is always positive; ~0.018 at init
        self.log_intensity = nn.Parameter(torch.tensor(-4.0))

        # grain coarseness factor: noise is generated at H*scale then upsampled
        # scale ≈ 0.5 means 2× downsampled → ~2-4 px grain at 256px
        self.log_coarseness = nn.Parameter(torch.tensor(math.log(0.5)))

        # shadow boost: how much more grain in darks
        self.shadow_boost = nn.Parameter(torch.tensor(1.5))

        # per-channel gain ratios (real film grain affects RGB unequally)
        self.channel_gain = nn.Parameter(torch.tensor([1.0, 0.95, 0.85]))

        # 5×5 Gaussian-shaped conv to shape noise spectrum (organic texture)
        self.shaper = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False)
        with torch.no_grad():
            self.shaper.weight.copy_(
                self._gaussian_kernel(5, 1.2).view(1, 1, 5, 5)
            )

        # optional fine-scale overlay for natural multi-scale grain appearance
        self.fine_intensity = nn.Parameter(torch.tensor(-5.0))  # ~0.007

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-0.5 * (coords / sigma) ** 2)
        g = g / g.sum()
        return g[:, None] * g[None, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        # ---- coarse layer (main grain texture) ----
        coarseness = torch.sigmoid(self.log_coarseness)  # (0.5, 0.73)
        hs = max(16, int(H * coarseness.item()))
        ws = max(16, int(W * coarseness.item()))

        noise_c = torch.randn(B, 1, hs, ws, device=device, dtype=dtype)
        noise_c = self.shaper(noise_c)
        noise_c = F.interpolate(noise_c, size=(H, W), mode='bilinear',
                                align_corners=False)
        noise_c = noise_c / (noise_c.std() + 1e-6)

        # ---- fine overlay (high-frequency detail, very subtle) ----
        noise_f = torch.randn(B, 1, H, W, device=device, dtype=dtype)
        noise_f = noise_f / (noise_f.std() + 1e-6)

        # ---- composite multi-scale noise ----
        intensity_coarse = torch.exp(self.log_intensity)
        intensity_fine = torch.exp(self.fine_intensity)
        noise = (noise_c * intensity_coarse + noise_f * intensity_fine)

        # ---- luminance-dependent modulation ----
        lum = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        # shadow boost with a soft floor so midtones still get some grain
        boost = torch.sigmoid(self.shadow_boost) * 3.0
        mod = torch.pow(1.0 - lum.clamp(0, 1), boost)
        # soft floor at 0.15 so highlights never get completely clean
        mod = mod * 0.85 + 0.15

        # ---- per-channel gain ----
        chan_g = torch.sigmoid(self.channel_gain)            # [3]
        grain = noise * mod * chan_g.view(1, 3, 1, 1)

        out = x + grain
        return out.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class FilmPipeline(nn.Module):
    """Full physical film emulation pipeline.

    Order (v2, Phos-inspired):
        1.  Exposure              — linear domain gain / bias
        2.  Spectral / density    — colour mixing (linear or density domain)
        3.  Pyramid Bloom         — multi-scale halation (linear domain)
        4.  Tone Response Curve   — per-channel 1D LUT (H&D curve)
        5.  Grain                 — spatial noise

    Two spectral modes:
        - "linear" (default):   SpectralMatrix + ResidualLUT3D
        - "density" (Phos):     DensityCrosstalk (linear crosstalk + 3D density LUT,
                                replaces BOTH SpectralMatrix and ResidualLUT3D)

    Config knobs (read from ``config`` dict):
        spectral_mode      — "linear" or "density" (default "linear")
        use_halation       — enable/disable bloom stage  (default True)
        use_grain          — enable/disable grain stage   (default True)
        bloom_levels       — number of pyramid levels     (default 6)
        trc_points         — 1D LUT control points        (default 32)
        residual_lut_dim   — 3D LUT grid size  (for "linear" mode, default 17)
        density_lut_dim    — density 3D LUT dim (for "density" mode, default 9)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Stage 1 — per-channel exposure
        self.exposure = ExposureModule()

        # Stage 2 — spectral / dye-coupling
        spectral_mode = getattr(config, "spectral_mode", "linear")
        self._mode = spectral_mode

        if spectral_mode == "density":
            # DensityCrosstalk does BOTH the 3×3 linear crosstalk AND the
            # nonlinear residual via a small 3D LUT in density domain
            self.spectral = DensityCrosstalk(
                density_lut_dim=getattr(config, "density_lut_dim", 9)
            )
            self._use_density_lut = True
        else:
            self.spectral = SpectralMatrix()
            self._use_density_lut = False

        # Stage 3 — pyramid bloom (Phos-inspired, linear domain)
        self.bloom = PyramidBloom(
            levels=getattr(config, "bloom_levels", 6),
        )

        # Stage 4 — per-channel 1D LUT
        self.trc = ToneResponseCurve(
            num_points=getattr(config, "trc_points", 32)
        )

        # Stage 5 — residual 3D LUT (only in "linear" mode)
        if not self._use_density_lut:
            self.residual_lut = ResidualLUT3D(
                dim=getattr(config, "residual_lut_dim", 17)
            )

        # Stage 6 — spatial grain
        self.grain = Grain()

        # Stage enable switches
        self.use_bloom = getattr(config, "use_halation", True)
        self.use_grain = getattr(config, "use_grain", True)

    def forward(self, x: torch.Tensor) -> dict:
        # Stage 1 — exposure
        out = self.exposure(x)

        # Stage 2 — spectral / density mixing
        if self._use_density_lut:
            out, tv_res, mn_res = self.spectral(out)
        else:
            out = self.spectral(out)

        # Stage 3 — bloom in the *linear* domain (before tone curve)
        if self.use_bloom:
            out = self.bloom(out)

        # Stage 4 — tone response curve (H&D development)
        out = self.trc(out)
        tv_trc = self.trc.tv_loss()

        # Stage 5 — residual crossover 3D LUT (only in "linear" mode)
        if not self._use_density_lut:
            out = self.residual_lut(out)
            tv_res, mn_res = self.residual_lut.reg_loss()

        # Stage 6 — grain
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
