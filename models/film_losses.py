"""
Film-specific losses for unpaired training.

All losses operate on *distributions* rather than per-pixel
correspondence, because we have unpaired digital (domain A) and
film (domain B) images.

Key losses:
  - SwD / SWD        : Sliced Wasserstein Distance on feature & pixel distributions
  - HistLoss          : per-channel histogram matching via soft binning
  - CovLoss           : 3x3 channel covariance matching
  - TVAniso           : anisotropic TV on the 3D LUT (diagonal = tone, off-diagonal = crossover)
  - LPIPS             : perceptual distribution matching (wrapper, optional)

Usage in trainer:
    film_losses = FilmLossCollection(gan_loss_fn, lpips_fn, config).to(rank)
    loss_dict = film_losses(fake_dict, real_batch, fake_logits)
    g_loss = sum(weighted losses)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lpips


# ---------------------------------------------------------------------------
# Sliced Wasserstein Distance
# ---------------------------------------------------------------------------

def sliced_wasserstein_loss(
    fake: torch.Tensor,
    real: torch.Tensor,
    n_projections: int = 64,
) -> torch.Tensor:
    """SWD between two image batches treated as distributions of pixels.

    Args:
        fake: [B, C, H, W]
        real: [B, C, H, W]
        n_projections: number of random 1D projections

    Returns:
        scalar loss
    """
    B, C, H, W = fake.shape
    # flatten to [B*H*W, C]
    f = fake.permute(0, 2, 3, 1).reshape(-1, C)
    r = real.permute(0, 2, 3, 1).reshape(-1, C)

    # random projection directions on the unit sphere
    device = fake.device
    rng = torch.Generator(device=device).manual_seed(42 + n_projections)
    dirs = torch.randn(n_projections, C, device=fake.device, generator=rng,
                       dtype=fake.dtype)
    dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8)

    # project: [N, B*H*W]
    pf = torch.matmul(dirs, f.t())   # [N, M]
    pr = torch.matmul(dirs, r.t())   # [N, M]

    # sort along sample axis and compare
    pf_sorted, _ = torch.sort(pf, dim=1)
    pr_sorted, _ = torch.sort(pr, dim=1)

    # if batch sizes differ, interpolate to common length
    if pf_sorted.size(1) != pr_sorted.size(1):
        common = min(pf_sorted.size(1), pr_sorted.size(1))
        pf_sorted = F.interpolate(pf_sorted.unsqueeze(1), size=common,
                                   mode='linear', align_corners=False).squeeze(1)
        pr_sorted = F.interpolate(pr_sorted.unsqueeze(1), size=common,
                                   mode='linear', align_corners=False).squeeze(1)

    return (pf_sorted - pr_sorted).pow(2).mean()


# ---------------------------------------------------------------------------
# Histogram matching loss (soft binning)
# ---------------------------------------------------------------------------

class HistLoss(nn.Module):
    """Per-channel soft-histogram matching.

    Bins are placed uniformly in [0, 1].  Soft assignment via sigmoid
    makes the histogram differentiable.
    """

    def __init__(self, n_bins: int = 64, sigma: float = 0.02):
        super().__init__()
        self.n_bins = n_bins
        centers = (torch.arange(n_bins, dtype=torch.float32) + 0.5) / n_bins
        self.register_buffer('centers', centers)
        self.sigma = sigma

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        loss = fake.new_tensor(0.0)
        for c in range(fake.size(1)):
            hf = self._soft_hist(fake[:, c])
            hr = self._soft_hist(real[:, c])
            loss = loss + (hf - hr).pow(2).mean()
        return loss / fake.size(1)

    def _soft_hist(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W] -> [N]
        x = x.reshape(-1).clamp(0, 1)
        # soft assignment: [N, n_bins]
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0).to(x.device)
        w = torch.exp(-0.5 * (diff / self.sigma) ** 2)
        hist = w.sum(dim=0)
        hist = hist / (hist.sum() + 1e-8)
        return hist


# ---------------------------------------------------------------------------
# Channel covariance matching
# ---------------------------------------------------------------------------

def covariance_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Match the 3x3 covariance matrix of RGB channels.

    Captures inter-channel coupling (crossover) statistically.
    """
    B, C, H, W = fake.shape
    f = fake.permute(0, 2, 3, 1).reshape(-1, C)
    r = real.permute(0, 2, 3, 1).reshape(-1, C)

    f_cov = self_cov(f)
    r_cov = self_cov(r)
    return (f_cov - r_cov).pow(2).mean()


def self_cov(x: torch.Tensor) -> torch.Tensor:
    """Covariance of [N, C]."""
    n = x.size(0)
    mean = x.mean(dim=0, keepdim=True)
    xc = x - mean
    return xc.t() @ xc / (n - 1 + 1e-8)


# ---------------------------------------------------------------------------
# Anisotropic TV on 3D LUT (regulariser)
# ---------------------------------------------------------------------------

def anisotropic_tv_lut(
    lut: torch.Tensor,
    diag_weight: float = 0.1,
    offdiag_weight: float = 1.0,
) -> torch.Tensor:
    """Anisotropic total-variation regulariser for a 3D LUT.

    The LUT diagonal (R=G=B, pure tone) is allowed to vary sharply
    (preserves shoulder/toe); off-diagonal directions (crossover)
    are heavily smoothed.

    Args:
        lut: [3, D, D, D] parameter
    """
    # differences along R, G, B axes
    dif_r = lut[:, :, :, 1:] - lut[:, :, :, :-1]
    dif_g = lut[:, :, 1:, :] - lut[:, :, :-1, :]
    dif_b = lut[:, 1:, :, :] - lut[:, :-1, :, :]

    tv = (dif_r ** 2).mean() + (dif_g ** 2).mean() + (dif_b ** 2).mean()
    return offdiag_weight * tv


# ---------------------------------------------------------------------------
# Collection wrapper
# ---------------------------------------------------------------------------

class FilmLossCollection(nn.Module):
    """Bundles all film-specific losses with configurable weights."""

    def __init__(self, config, lpips_net: str = 'vgg'):
        super().__init__()
        self.config = config
        self.hist_loss = HistLoss(
            n_bins=getattr(config, "hist_bins", 64),
            sigma=getattr(config, "hist_sigma", 0.02),
        )
        self.lpips_fn: lpips.LPIPS | None = None  # lazily initialised

        # read weights
        self.w_swd = getattr(config, "lambda_swd", 1.0)
        self.w_hist = getattr(config, "lambda_hist", 1.0)
        self.w_cov = getattr(config, "lambda_cov", 0.5)
        self.w_lpips = getattr(config, "lambda_lpips", 1.0)
        self.w_gan = getattr(config, "lambda_gan", 0.4)
        self.w_tv_res = getattr(config, "lambda_tv_res", 0.01)
        self.w_mn_res = getattr(config, "lambda_mn_res", 0.5)
        self.w_tv_trc = getattr(config, "lambda_tv_trc", 0.01)
        self.w_aniso = getattr(config, "lambda_aniso", 0.01)

    def init_lpips(self, rank: int):
        self.lpips_fn = lpips.LPIPS(net='vgg').to(rank)

    def forward(
        self,
        fake_dict: dict,
        real: torch.Tensor,
        fake_logits: torch.Tensor | None = None,
        gan_loss_fn=None,
    ) -> dict:
        fake = fake_dict["out"]

        losses = {}
        losses["swd"] = sliced_wasserstein_loss(fake, real) * self.w_swd
        losses["hist"] = self.hist_loss(fake, real) * self.w_hist
        losses["cov"] = covariance_loss(fake, real) * self.w_cov

        if self.lpips_fn is not None:
            losses["lpips"] = self.lpips_fn(fake, real).mean() * self.w_lpips

        if fake_logits is not None and gan_loss_fn is not None:
            losses["gan"] = gan_loss_fn(fake_logits, 1.0) * self.w_gan

        # LUT regularisers
        if "tv_res" in fake_dict:
            losses["tv_res"] = fake_dict["tv_res"] * self.w_tv_res
        if "mn_res" in fake_dict:
            losses["mn_res"] = fake_dict["mn_res"] * self.w_mn_res
        if "tv_trc" in fake_dict:
            losses["tv_trc"] = fake_dict["tv_trc"] * self.w_tv_trc

        losses["total"] = sum(losses.values())
        return losses
