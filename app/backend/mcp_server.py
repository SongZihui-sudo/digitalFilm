"""
Digital Film MCP Server
========================
Provides MCP tools for film-style image processing:

1. film_adjust_basic   — exposure, contrast, saturation, temperature, etc.
2. film_apply_style    — film preset conversion (Kodak Gold, Fuji, etc.)
3. film_simulate_dof   — depth-of-field with Depth Anything V2
4. film_enhance_quality — OSEDiff super-resolution
5. film_process        — combined full pipeline (basic → SR → DOF → film)
"""

import os
import sys
import time
import uuid
import traceback
from typing import Dict, Optional, Any

import numpy as np
import torch
from PIL import Image, ImageEnhance
from fastmcp import FastMCP

# Keep a single CUDA stream alive to minimise memory fragmentation,
# and allow memory segments to be released back to the OS when possible.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from options.options import everyThingOptions
from models.digitalFilm_v2 import digitalFilmv2
from models.optical_simulator import (
    DofSettings as OpticalDofSettings,
    OpticalSimulator,
)
from models.odediff_port import OSEDIFF_port
from utils.utils import (
    ensure_dir,
    resize_to_multiple_of_16,
    pil_to_tensor,
    tensor_to_pil,
    load_image_from_url,
    decode_base64_image,
)

# Config paths — relative to this file (app/backend/)
MODEL_OPTIONS_PATH = os.path.join(CURRENT_DIR, "config", "digitalfilm.yaml")
APP_OPTIONS_PATH = os.path.join(CURRENT_DIR, "config", "mcp_backend.yaml")


# ==============================================================================
# Service
# ==============================================================================

class DigitalFilmMCPService:
    """Manages models and exposes processing methods used by MCP tools."""

    def __init__(self):
        # ---- load config ----
        self.model_config = everyThingOptions(MODEL_OPTIONS_PATH)
        self.model_config.load_config()

        self.app_config = everyThingOptions(APP_OPTIONS_PATH)
        self.app_config.load_config()

        self.device = "cuda" if (
            getattr(self.app_config.opt.global_config, "cuda", False)
            and torch.cuda.is_available()
        ) else "cpu"

        self.root_dir = getattr(
            self.app_config.opt.global_config, "server_root", "./server_data"
        )
        # Make root_dir absolute so relative paths survive CWD changes
        if not os.path.isabs(self.root_dir):
            self.root_dir = os.path.normpath(os.path.join(PROJECT_ROOT, self.root_dir))
        self.output_dir = os.path.join(self.root_dir, "outputs")
        ensure_dir(self.output_dir)

        # ---- load film-style models ----
        self.models: Dict[str, Any] = self._load_models()

        # ---- optical simulator (CPU by default, leaves GPU memory for film net) ----
        optics_config = getattr(self.app_config.opt, "optics", None)
        optical_device = getattr(optics_config, "device", "cpu")
        depth_model_name = getattr(
            optics_config, "depth_model_name", "depth-anything/Depth-Anything-V2-Base-hf"
        )
        depth_model_cache_dir = getattr(optics_config, "depth_model_cache_dir", None)
        if depth_model_cache_dir:
            depth_model_cache_dir = os.path.expanduser(str(depth_model_cache_dir))
            if not os.path.isabs(depth_model_cache_dir):
                depth_model_cache_dir = os.path.join(PROJECT_ROOT, depth_model_cache_dir)
        self.optical_simulator = OpticalSimulator(
            device=optical_device,
            depth_model_name=depth_model_name,
            depth_model_cache_dir=depth_model_cache_dir,
        )

        # ---- OSEDiff super-resolution ----
        osediff_config = getattr(self.app_config.opt, "osediff", None)
        osediff_device = getattr(osediff_config, "device", "cpu")
        self.osediff = OSEDIFF_port(
            getattr(osediff_config, "pretrained_model_name_or_path", ""),
            getattr(osediff_config, "osediff_path", ""),
            getattr(osediff_config, "vae_encoder_tiled_size", 1024),
            getattr(osediff_config, "vae_decoder_tiled_size", 244),
            "fp16",
            osediff_device,
            getattr(osediff_config, "merge_and_unload_lora", True),
        )

    # ----- model loading -----

    def _load_single_model(self, ckpt_path: str):
        if not ckpt_path:
            raise FileNotFoundError("Checkpoint path is empty.")
        model = digitalFilmv2(
            0,
            self.model_config.opt.global_config,
            self.model_config.opt.model_config,
        )
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        model = model.requires_grad_(False)
        model.eval()
        # Keep on CPU — moved to GPU on-demand in _get_model_for_inference()
        return model

    def _load_models(self) -> Dict[str, Any]:
        checkpoints = getattr(self.model_config.opt, "checkpoints", None)
        if checkpoints is None:
            raise ValueError("No checkpoints config found.")
        models = {}
        for preset_name, ckpt_path in checkpoints.items():
            models[preset_name] = self._load_single_model(ckpt_path)
        if not models:
            raise ValueError("No models loaded. Check checkpoints config.")
        return models

    @torch.no_grad()
    def _get_model_for_inference(self, preset: str):
        """Move one model to GPU, yield it, then move it back to CPU.

        Only one model is on GPU at a time, avoiding OOM.
        """
        if preset not in self.models:
            raise ValueError(
                f"Unknown preset '{preset}'. Available: {list(self.models.keys())}"
            )
        model = self.models[preset]
        if self.device == "cuda":
            model = model.to(self.device)
        try:
            yield model
        finally:
            if self.device == "cuda":
                model = model.cpu()
                torch.cuda.empty_cache()

    # ----- image I/O helpers -----

    @staticmethod
    def _load_image(
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
    ) -> Image.Image:
        """Load a PIL Image from one of the three supported sources."""
        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            return Image.open(image_path).convert("RGB")
        if image_url:
            return load_image_from_url(image_url)
        if image_base64:
            return decode_base64_image(image_base64)
        raise ValueError(
            "One of image_path, image_url, or image_base64 must be provided."
        )

    def _save_image(self, image: Image.Image) -> Dict[str, Any]:
        file_name = f"{int(time.time())}_{uuid.uuid4().hex}.png"
        save_path = os.path.normpath(os.path.join(self.output_dir, file_name))
        image.save(save_path, format="PNG")
        return {
            "file_name": file_name,
            "saved_path": save_path,
            "width": image.size[0],
            "height": image.size[1],
        }

    # ----- basic adjustments (PIL-based) -----

    @staticmethod
    def apply_basic_adjustments(
        image: Image.Image,
        exposure: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        temperature: float = 0.0,
        tint: float = 0.0,
        highlights: float = 0.0,
        shadows: float = 0.0,
    ) -> Image.Image:
        """Apply basic photographic adjustments to a PIL Image.  All values
        are roughly in [-100, 100] with 0 meaning no change."""
        image = image.convert("RGB")

        # exposure — multiplicative in linear-ish space
        if exposure != 0:
            arr = np.array(image).astype(np.float32)
            factor = 2.0 ** exposure
            arr *= factor
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            image = Image.fromarray(arr)

        # contrast
        if contrast != 0:
            factor = 1.0 + contrast / 100.0
            factor = max(0.0, factor)
            image = ImageEnhance.Contrast(image).enhance(factor)

        # saturation
        if saturation != 0:
            factor = 1.0 + saturation / 100.0
            factor = max(0.0, factor)
            image = ImageEnhance.Color(image).enhance(factor)

        # temperature / tint — warm-cool and green-magenta shifts
        if temperature != 0 or tint != 0:
            arr = np.array(image).astype(np.float32)
            temp_shift = temperature / 100.0 * 30.0
            tint_shift = tint / 100.0 * 20.0
            arr[..., 0] += temp_shift   # R
            arr[..., 2] -= temp_shift   # B
            arr[..., 1] += tint_shift   # G
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            image = Image.fromarray(arr)

        # highlights / shadows — luminance-masked adjustments
        if highlights != 0 or shadows != 0:
            arr = np.array(image).astype(np.float32)
            luminance = (
                0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
            )
            if highlights != 0:
                strength = highlights / 100.0
                mask = np.clip((luminance - 128) / 127.0, 0, 1)
                arr = arr + strength * mask[..., None] * 40.0
            if shadows != 0:
                strength = shadows / 100.0
                mask = np.clip((128 - luminance) / 128.0, 0, 1)
                arr = arr + strength * mask[..., None] * 40.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            image = Image.fromarray(arr)

        return image

    # ----- film inference -----

    @torch.no_grad()
    def infer_tensor(
        self, x: torch.Tensor, preset: str = "kodak_gold_200"
    ) -> torch.Tensor:
        gen = self._get_model_for_inference(preset)
        model = next(gen)
        try:
            amp = self.device == "cuda"
            dtype = torch.float16 if amp else torch.float32
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=amp):
                return model.g(x)["out"]
        finally:
            try:
                next(gen)
            except StopIteration:
                pass

    @torch.no_grad()
    def infer_pil(
        self, image: Image.Image, preset: str = "kodak_gold_200", max_size: int = 1536
    ) -> Image.Image:
        image = resize_to_multiple_of_16(image.convert("RGB"), max_size=max_size)
        x = pil_to_tensor(image).to(self.device)
        return tensor_to_pil(self.infer_tensor(x, preset=preset))

    # ======================================================================
    # Public processing methods — each maps to an MCP tool
    # ======================================================================

    # ---- Tool 1: basic adjustments ----

    def process_basic(
        self,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        exposure: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        temperature: float = 0.0,
        tint: float = 0.0,
        highlights: float = 0.0,
        shadows: float = 0.0,
    ) -> Dict[str, Any]:
        image = self._load_image(image_path, image_url, image_base64)
        image = self.apply_basic_adjustments(
            image, exposure, contrast, saturation,
            temperature, tint, highlights, shadows,
        )
        meta = self._save_image(image)
        return {
            "ok": True,
            **meta,
            "device": self.device,
        }

    # ---- Tool 2: film style conversion ----

    def process_style(
        self,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        preset: str = "kodak_gold_200",
        max_size: int = 1536,
    ) -> Dict[str, Any]:
        image = self._load_image(image_path, image_url, image_base64)
        output = self.infer_pil(image, preset=preset, max_size=max_size)
        meta = self._save_image(output)
        return {
            "ok": True,
            **meta,
            "preset": preset,
            "available_presets": list(self.models.keys()),
            "device": self.device,
        }

    # ---- Tool 3: depth of field ----

    def process_dof(
        self,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        enabled: bool = True,
        f_number: float = 5.6,
        focal_length_mm: float = 85.0,
        sensor_profile: str = "full_frame",
        focus_point_x: Optional[float] = None,
        focus_point_y: Optional[float] = None,
        max_size: int = 1536,
    ) -> Dict[str, Any]:
        image = self._load_image(image_path, image_url, image_base64)
        image = resize_to_multiple_of_16(image.convert("RGB"), max_size=max_size)
        x = pil_to_tensor(image)

        dof = OpticalDofSettings(
            enabled=enabled,
            f_number=f_number,
            focal_length_mm=focal_length_mm,
            sensor_profile=sensor_profile,
            focus_point_x=focus_point_x,
            focus_point_y=focus_point_y,
        )
        x = self.optical_simulator.apply(x, dof_settings=dof)
        output = tensor_to_pil(x)
        meta = self._save_image(output)
        return {
            "ok": True,
            **meta,
            "dof": {
                "enabled": enabled,
                "f_number": f_number,
                "focal_length_mm": focal_length_mm,
                "sensor_profile": sensor_profile,
                "focus_point_x": focus_point_x,
                "focus_point_y": focus_point_y,
            },
            "device": self.device,
        }

    # ---- Tool 4: quality enhancement (OSEDiff SR) ----

    def _move_osediff_to(self, target_device: str):
        """Move OSEDiff to the given device. Handles non-parameter tensors
        (timesteps, scheduler buffers) that nn.Module.to() skips."""
        if self.osediff.unet.device.type == target_device:
            return
        print(f"[mcp] moving OSEDiff to {target_device} ...", flush=True)
        self.osediff = self.osediff.to(target_device)
        # nn.Module.to() skips these raw tensors; move them explicitly
        if target_device == "cuda":
            self.osediff.timesteps = self.osediff.timesteps.cuda()
            self.osediff.noise_scheduler.alphas_cumprod = (
                self.osediff.noise_scheduler.alphas_cumprod.cuda()
            )
        else:
            self.osediff.timesteps = self.osediff.timesteps.cpu()
            self.osediff.noise_scheduler.alphas_cumprod = (
                self.osediff.noise_scheduler.alphas_cumprod.cpu()
            )
        self.osediff.device = target_device
        torch.cuda.empty_cache()

    @property
    def _osediff_device(self) -> str:
        """Resolve the OSEDiff target device from config."""
        return getattr(self.app_config.opt.osediff, "device", "cpu")

    def process_sr(
        self,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        max_size: int = 1536,
    ) -> Dict[str, Any]:
        image = self._load_image(image_path, image_url, image_base64)
        image = resize_to_multiple_of_16(image.convert("RGB"), max_size=max_size)
        x = pil_to_tensor(image)
        target = self._osediff_device
        print(f"[mcp] SR inference started on {target} ...", flush=True)
        try:
            self._move_osediff_to(target)
            x = self.osediff.process(x, enable=True)
        finally:
            self._move_osediff_to("cpu")
        print("[mcp] SR inference finished.", flush=True)
        output = tensor_to_pil(x)
        meta = self._save_image(output)
        return {
            "ok": True,
            **meta,
            "sr_applied": True,
            "sr_device_used": target,
            "device": self.device,
        }

    # ---- Tool 5: combined full pipeline ----

    def process_full(
        self,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        # basic
        exposure: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        temperature: float = 0.0,
        tint: float = 0.0,
        highlights: float = 0.0,
        shadows: float = 0.0,
        # film style
        preset: str = "kodak_gold_200",
        # dof
        dof_enabled: bool = False,
        f_number: float = 5.6,
        focal_length_mm: float = 85.0,
        sensor_profile: str = "full_frame",
        focus_point_x: Optional[float] = None,
        focus_point_y: Optional[float] = None,
        # sr
        sr_enabled: bool = False,
        # misc
        max_size: int = 1536,
    ) -> Dict[str, Any]:
        image = self._load_image(image_path, image_url, image_base64)

        # 1. Basic adjustments (PIL)
        image = self.apply_basic_adjustments(
            image, exposure, contrast, saturation,
            temperature, tint, highlights, shadows,
        )

        # 2. Resize + to tensor
        image = resize_to_multiple_of_16(image.convert("RGB"), max_size=max_size)
        x = pil_to_tensor(image)

        # 3. SR enhancement
        if sr_enabled:
            target = self._osediff_device
            print(f"[mcp] SR inference started on {target} ...", flush=True)
            try:
                self._move_osediff_to(target)
                x = self.osediff.process(x, enable=True)
            finally:
                self._move_osediff_to("cpu")
            print("[mcp] SR inference finished.", flush=True)

        # 4. DOF (CPU) — before film model to save GPU memory
        if dof_enabled:
            print("[mcp] DOF simulation started — depth inference may take a moment ...", flush=True)
        dof = OpticalDofSettings(
            enabled=dof_enabled,
            f_number=f_number,
            focal_length_mm=focal_length_mm,
            sensor_profile=sensor_profile,
            focus_point_x=focus_point_x,
            focus_point_y=focus_point_y,
        )
        x = self.optical_simulator.apply(x, dof_settings=dof)
        if dof_enabled:
            print("[mcp] DOF simulation finished.", flush=True)

        # 5. Film style (GPU)
        print(f"[mcp] film style conversion ({preset}) started ...", flush=True)
        x = x.to(self.device)
        x = self.infer_tensor(x, preset=preset)

        output = tensor_to_pil(x)
        meta = self._save_image(output)
        return {
            "ok": True,
            **meta,
            "preset": preset,
            "dof_enabled": dof_enabled,
            "sr_enabled": sr_enabled,
            "available_presets": list(self.models.keys()),
            "device": self.device,
        }


# ==============================================================================
# Bootstrap — create service & define MCP tools
# ==============================================================================

service = DigitalFilmMCPService()
mcp = FastMCP("digital-film")


# ---- Tool 1: Basic adjustments ----

@mcp.tool()
def film_adjust_basic(
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    exposure: float = 0,
    contrast: float = 0,
    saturation: float = 0,
    temperature: float = 0,
    tint: float = 0,
    highlights: float = 0,
    shadows: float = 0,
) -> Dict[str, Any]:
    """Apply basic photographic adjustments to an image.

    Each parameter roughly maps to the [-100, 100] range you'd see in
    Lightroom / Capture One, with 0 meaning no change.

    Provide ONE of: image_path (local file), image_url, or image_base64.
    """
    try:
        return service.process_basic(
            image_path=image_path,
            image_url=image_url,
            image_base64=image_base64,
            exposure=exposure,
            contrast=contrast,
            saturation=saturation,
            temperature=temperature,
            tint=tint,
            highlights=highlights,
            shadows=shadows,
        )
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


# ---- Tool 2: Film style conversion ----

@mcp.tool()
def film_apply_style(
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    preset: str = "kodak_gold_200",
    max_size: int = 1536,
) -> Dict[str, Any]:
    """Convert an image to a film-stock look using a trained neural pipeline.

    Available presets are loaded from the checkpoints config (e.g.
    kodak_gold_200, kodak_E100, fuji_color_200).

    Provide ONE of: image_path (local file), image_url, or image_base64.
    """
    try:
        return service.process_style(
            image_path=image_path,
            image_url=image_url,
            image_base64=image_base64,
            preset=preset,
            max_size=max_size,
        )
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


# ---- Tool 3: Depth of field ----

@mcp.tool()
def film_simulate_dof(
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    enabled: bool = True,
    f_number: float = 5.6,
    focal_length_mm: float = 85.0,
    sensor_profile: str = "full_frame",
    focus_point_x: float = 0.5,
    focus_point_y: float = 0.5,
    max_size: int = 1536,
) -> Dict[str, Any]:
    """Simulate realistic depth-of-field using Depth Anything V2 for depth
    estimation and a physically-based defocus renderer.

    sensor_profile: one of m43, aps_c, full_frame, medium_645, large_8x10.
    focus_point_x/y: normalized coords [0, 1]; if omitted, focus at ~2.5 m.

    Provide ONE of: image_path (local file), image_url, or image_base64.
    """
    try:
        return service.process_dof(
            image_path=image_path,
            image_url=image_url,
            image_base64=image_base64,
            enabled=enabled,
            f_number=f_number,
            focal_length_mm=focal_length_mm,
            sensor_profile=sensor_profile,
            focus_point_x=focus_point_x,
            focus_point_y=focus_point_y,
            max_size=max_size,
        )
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


# ---- Tool 4: Quality enhancement ----

@mcp.tool()
def film_enhance_quality(
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    max_size: int = 1536,
) -> Dict[str, Any]:
    """Enhance image quality using OSEDiff super-resolution.

    Applies a diffusion-based enhancement model that improves fine detail,
    reduces artifacts, and sharpens the image.

    Provide ONE of: image_path (local file), image_url, or image_base64.
    """
    try:
        return service.process_sr(
            image_path=image_path,
            image_url=image_url,
            image_base64=image_base64,
            max_size=max_size,
        )
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


# ---- Tool 5: Combined full pipeline ----

@mcp.tool()
def film_process(
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    # basic adjustments
    exposure: float = 0,
    contrast: float = 0,
    saturation: float = 0,
    temperature: float = 0,
    tint: float = 0,
    highlights: float = 0,
    shadows: float = 0,
    # film style
    preset: str = "kodak_gold_200",
    # depth of field
    dof_enabled: bool = False,
    f_number: float = 5.6,
    focal_length_mm: float = 85.0,
    sensor_profile: str = "full_frame",
    focus_point_x: float = 0.5,
    focus_point_y: float = 0.5,
    # super-resolution
    sr_enabled: bool = False,
    # misc
    max_size: int = 1536,
) -> Dict[str, Any]:
    """Run the full film-processing pipeline: basic adjustments → quality
    enhancement → depth-of-field → film-style conversion.

    This combines all four individual tools in the correct processing order.
    Use when you want the complete end-to-end film look in one call.

    Provide ONE of: image_path (local file), image_url, or image_base64.
    """
    try:
        return service.process_full(
            image_path=image_path,
            image_url=image_url,
            image_base64=image_base64,
            exposure=exposure,
            contrast=contrast,
            saturation=saturation,
            temperature=temperature,
            tint=tint,
            highlights=highlights,
            shadows=shadows,
            preset=preset,
            dof_enabled=dof_enabled,
            f_number=f_number,
            focal_length_mm=focal_length_mm,
            sensor_profile=sensor_profile,
            focus_point_x=focus_point_x,
            focus_point_y=focus_point_y,
            sr_enabled=sr_enabled,
            max_size=max_size,
        )
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


if __name__ == "__main__":
    mcp.run()
