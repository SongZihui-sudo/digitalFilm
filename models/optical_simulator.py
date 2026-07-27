"""Depth-aware optical processing for the film-generation pipeline.

The simulator deliberately uses the lightweight ``DefocusLens`` path for
large-format depth of field.  It does not invoke ``Camera`` because the
pipeline already owns decoding, colour handling, and the film-look network.

All optical parameters except aperture (f_number), focal length, sensor size, and focus point
are fixed to sensible defaults for a large-format portrait/general-purpose lens.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import math

import torch

from models.end2end_imaging.deeplens import DefocusLens
from models.end2end_imaging.network.depth_estimator import DepthAnythingV2Estimator


# --- Fixed optical defaults (hidden from the UI) ---
_SENSOR_SIZE_MAP: dict[str, tuple[float, float]] = {
    "m43":        (17.3, 13.0),
    "aps_c":      (23.6, 15.7),
    "full_frame": (36.0, 24.0),
    "medium_645": (56.0, 41.5),
    "large_8x10": (203.2, 254.0),
}
_DEPTH_MIN_MM: float = 100.0
_DEPTH_MAX_MM: float = 10000.0
_PSF_KERNEL_SIZE: int = 65
_NUM_LAYERS: int = 16
_RENDER_METHOD: str = "psf_patch"


@dataclass(frozen=True)
class DofSettings:
    enabled: bool = False
    f_number: float = 5.6
    focal_length_mm: float = 85.0
    sensor_profile: str = "full_frame"
    focus_point_x: float | None = None   # normalized 0-1
    focus_point_y: float | None = None   # normalized 0-1


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
        if settings.focal_length_mm <= 0:
            raise ValueError("focal_length_mm must be positive.")
        if settings.f_number <= 0:
            raise ValueError("f_number must be positive.")
        if settings.focus_point_x is not None and not (0 <= settings.focus_point_x <= 1):
            raise ValueError("focus_point_x must be in [0, 1] when provided.")
        if settings.focus_point_y is not None and not (0 <= settings.focus_point_y <= 1):
            raise ValueError("focus_point_y must be in [0, 1] when provided.")
        if settings.sensor_profile not in _SENSOR_SIZE_MAP:
            raise ValueError(f"Unknown sensor_profile: {settings.sensor_profile}. "
                             f"Must be one of {list(_SENSOR_SIZE_MAP.keys())}.")

    def _get_depth_estimator(self) -> DepthAnythingV2Estimator:
        key = (_DEPTH_MIN_MM, _DEPTH_MAX_MM)
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
        depth_mm = self._get_depth_estimator().estimate(x)

        # --- Determine focus distance from the focus point ---
        if settings.focus_point_x is not None and settings.focus_point_y is not None:
            # Sample the depth map at the clicked point (normalized coords).
            px = int(round(settings.focus_point_x * (width - 1)))
            py = int(round(settings.focus_point_y * (height - 1)))
            px = max(0, min(width - 1, px))
            py = max(0, min(height - 1, py))
            focus_distance_mm = float(depth_mm[0, 0, py, px].item())
            # Clamp to a reasonable range (avoid extreme / invalid depth values).
            focus_distance_mm = max(_DEPTH_MIN_MM * 2, min(_DEPTH_MAX_MM * 0.95, focus_distance_mm))
        else:
            # Default: focus at a classic portrait distance (~2.5 m).
            focus_distance_mm = 2500.0

        # --- Build sensor from the selected profile ---
        sensor_w, sensor_h = _SENSOR_SIZE_MAP[settings.sensor_profile]

        # Step 1: init with native sensor aspect ratio.
        # Use (sensor_w, sensor_h) as the dummy resolution so the assert
        #   sensor_w * res_h == sensor_h * res_w
        # reduces to sensor_w * sensor_h == sensor_h * sensor_w,
        # which is trivially exact in floating point (commutative multiply).
        lens = DefocusLens(
            foclen=settings.focal_length_mm,
            fnum=settings.f_number,
            sensor_size=(sensor_w, sensor_h),
            sensor_res=(sensor_w, sensor_h),
            device=self.device,
            dtype=x.dtype,
        )
        # Step 2: adapt to the actual image resolution while preserving the
        # sensor diagonal (set_sensor_res keeps r_sensor fixed).
        lens.set_sensor_res((width, height))

        lens.refocus(-focus_distance_mm)
        return lens.render_rgbd(
            x,
            depth_mm,
            psf_ks=_PSF_KERNEL_SIZE,
            num_layers=_NUM_LAYERS,
            method=_RENDER_METHOD,
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
