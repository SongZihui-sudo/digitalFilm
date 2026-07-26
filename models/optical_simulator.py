"""Depth-aware optical processing for the film-generation pipeline.

The simulator deliberately uses the lightweight ``DefocusLens`` path for
large-format depth of field.  It does not invoke ``Camera`` because the
pipeline already owns decoding, colour handling, and the film-look network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import math

import torch

from models.end2end_imaging.deeplens import DefocusLens
from models.end2end_imaging.network.depth_estimator import DepthAnythingV2Estimator


@dataclass(frozen=True)
class DofSettings:
    enabled: bool = False
    focal_length_mm: float = 210.0
    f_number: float = 5.6
    focus_distance_m: float = 2.5
    sensor_width_mm: float = 127.0
    sensor_height_mm: float = 101.6
    depth_min_mm: float = 500.0
    depth_max_mm: float = 12000.0
    psf_kernel_size: int = 65
    num_layers: int = 32
    render_method: str = "psf_patch"


class OpticalSimulator:
    """Apply optional large-format defocus to RGB tensors."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        depth_model_name: str = "depth-anything/Depth-Anything-V2-Base-hf",
        depth_model_cache_dir: str | None = None,
        depth_infer_size: int = 518,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.depth_model_name = depth_model_name
        # ``cache_dir`` is optional: None delegates to Hugging Face's normal cache.
        self.depth_model_cache_dir = (
            str(Path(depth_model_cache_dir).expanduser().resolve())
            if depth_model_cache_dir
            else None
        )
        self.depth_infer_size = depth_infer_size
        self._depth_estimators: dict[tuple[float, float], DepthAnythingV2Estimator] = {}

    @staticmethod
    def _validate_rgb_tensor(x: torch.Tensor) -> None:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected an RGB tensor shaped [B, 3, H, W]; got {tuple(x.shape)}.")
        if not torch.is_floating_point(x):
            raise ValueError("Optical simulation requires a floating-point RGB tensor in [0, 1].")

    @staticmethod
    def _validate_dof_settings(settings: DofSettings) -> None:
        if settings.focal_length_mm <= 0 or settings.f_number <= 0:
            raise ValueError("focal_length_mm and f_number must be positive.")
        if settings.focus_distance_m <= 0:
            raise ValueError("focus_distance_m must be positive.")
        if settings.sensor_width_mm <= 0 or settings.sensor_height_mm <= 0:
            raise ValueError("Sensor dimensions must be positive.")
        if settings.depth_min_mm <= 0 or settings.depth_max_mm <= settings.depth_min_mm:
            raise ValueError("depth_max_mm must be greater than positive depth_min_mm.")
        if settings.psf_kernel_size < 3 or settings.psf_kernel_size % 2 == 0:
            raise ValueError("psf_kernel_size must be an odd integer of at least 3.")
        if settings.num_layers < 2:
            raise ValueError("num_layers must be at least 2.")

    def _get_depth_estimator(self, settings: DofSettings) -> DepthAnythingV2Estimator:
        key = (float(settings.depth_min_mm), float(settings.depth_max_mm))
        if key not in self._depth_estimators:
            self._depth_estimators[key] = DepthAnythingV2Estimator(
                model_name=self.depth_model_name,
                depth_min_mm=key[0],
                depth_max_mm=key[1],
                infer_size=self.depth_infer_size,
                device=self.device,
                cache_dir=self.depth_model_cache_dir,
            )
        return self._depth_estimators[key]

    def _apply_dof(self, x: torch.Tensor, settings: DofSettings) -> torch.Tensor:
        if not settings.enabled:
            return x
        self._validate_dof_settings(settings)

        _, _, height, width = x.shape
        depth_mm = self._get_depth_estimator(settings).estimate(x)

        # DefocusLens requires physical sensor dimensions and raster resolution
        # to have exactly the same aspect ratio. The input image can be any
        # landscape/portrait/cropped ratio, so preserve the configured sensor
        # diagonal while deriving a matching virtual sensor for this image.
        sensor_diagonal_mm = math.hypot(
            settings.sensor_width_mm,
            settings.sensor_height_mm,
        )
        resolution_diagonal_px = math.hypot(width, height)
        sensor_size = (
            sensor_diagonal_mm * width / resolution_diagonal_px,
            sensor_diagonal_mm * height / resolution_diagonal_px,
        )
        lens = DefocusLens(
            foclen=settings.focal_length_mm,
            fnum=settings.f_number,
            sensor_size=sensor_size,
            sensor_res=(width, height),
            device=self.device,
            dtype=x.dtype,
        )
        lens.refocus(-settings.focus_distance_m * 1000.0)
        return lens.render_rgbd(
            x,
            depth_mm,
            psf_ks=settings.psf_kernel_size,
            num_layers=settings.num_layers,
            method=settings.render_method,
        )

    @torch.no_grad()
    def apply(
        self,
        x: torch.Tensor,
        dof_settings: DofSettings | None = None,
    ) -> torch.Tensor:
        """Run depth-conditioned defocus rendering."""
        self._validate_rgb_tensor(x)
        x = x.to(self.device)
        x = self._apply_dof(x, dof_settings or DofSettings())
        return x.clamp(0.0, 1.0)


# Compatibility alias for existing callers that used the original file's name.
optical_simulator = OpticalSimulator
