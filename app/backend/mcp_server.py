import torch
import os
from PIL import Image, ImageFilter
from typing import Dict, Optional, Any
import time
import uuid
from fastmcp import FastMCP
import numpy as np

from options.options import everyThingOptions
from models.digitalFilm_v2 import digitalFilmv2
from utils.utils import ensure_dir, resize_to_multiple_of_16, pil_to_tensor, tensor_to_pil, load_image_from_url, decode_base64_image


class DigitalFilmMCPService:
    def __init__(self, model_options_path: str, app_options_path: str):
        # 加载配置文件
        self.model_config = everyThingOptions(model_options_path)
        self.model_config.load_config()
        self.app_config = everyThingOptions(app_options_path)
        self.app_config.load_config()
        self.device = "cuda" if (
            getattr(self.app_config.opt.global_config, "cuda", False)
            and torch.cuda.is_available()
        ) else "cpu"
        self.root_dir = getattr(self.app_config.opt.global_config, "server_root", "./server_data")
        self.output_dir = os.path.join(self.root_dir, "outputs")
        ensure_dir(self.output_dir)
        self.base_url = getattr(self.app_config.opt.global_config, "mcp_server_base_url", "")
        # 加载多个模型
        self.models = self.load_models()

    def load_single_model(self, ckpt_path: str):
        if not ckpt_path:
            raise FileNotFoundError("Checkpoint path is empty.")
        model = digitalFilmv2(
            0,
            self.model_config.opt.global_config,
            self.model_config.opt.model_config
        )
        state_dict: dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        model = model.requires_grad_(False)
        model.eval()
        return model
    
    def load_models(self) -> Dict[str, Any]:
        """
        从配置中加载多个模型，返回一个 dict:
        {
            "kodak_gold_200": model,
            "fuji_400h": model,
            ...
        }
        你可以在配置里定义：
        global_config:
          checkpoints:
            kodak_gold_200: /path/to/kodak_gold_200.pth
            fuji_400h: /path/to/fuji_400h.pth
        """
        checkpoints = getattr(self.model_config.opt, "checkpoints", None)
        models = {}
        for preset_name, ckpt_path in checkpoints.items():
            models[preset_name] = self.load_single_model(ckpt_path)
        if not models:
            raise ValueError("No models loaded. Please check checkpoints config.")
        return models
    
    @torch.no_grad()
    def infer_pil(
        self,
        image: Image.Image,
        preset: str = "kodak_gold_200",
        max_size: int = 1536
    ) -> Image.Image:
        if preset not in self.models:
            raise ValueError(
                f"Unknown preset: {preset}. Available presets: {list(self.models.keys())}"
            )
        image = image.convert("RGB")
        image = resize_to_multiple_of_16(image, max_size=max_size)
        x = pil_to_tensor(image).to(self.device)
        model = self.models[preset]
        amp_enabled = (self.device == "cuda")
        amp_dtype = torch.float16 if amp_enabled else torch.float32
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            out = model.g(x)
        out = out["out"]
        result = tensor_to_pil(out)
        return result
    
    def apply_grain(self, image: Image.Image, grain: float = 0.0) -> Image.Image:
        """
        添加胶片颗粒。
        grain 建议范围：0.0 ~ 1.0
        """
        grain = max(0.0, min(1.0, grain))
        if grain <= 0:
            return image
        arr = np.array(image).astype(np.float32)
        # 颗粒强度，按 0~1 映射到较合理范围
        noise_std = 8.0 + grain * 32.0
        noise = np.random.normal(0, noise_std, arr.shape).astype(np.float32)
        # 轻微按亮度调整颗粒，暗部颗粒更明显一点
        luminance = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        luminance = luminance / 255.0
        weight = 1.15 - 0.5 * luminance
        weight = np.expand_dims(weight, axis=-1)
        arr = arr + noise * weight
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    
    def apply_halation(self, image: Image.Image, halation: float = 0.0) -> Image.Image:
        """
        添加高光晕染（红橙色偏暖的 glow）。
        halation 建议范围：0.0 ~ 1.0
        """
        halation = max(0.0, min(1.0, halation))
        if halation <= 0:
            return image
        base = image.convert("RGB")
        arr = np.array(base).astype(np.float32)
        # 取亮部作为 mask
        luminance = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        threshold = 180 - halation * 40   # halation 越大，参与晕染的亮部越多
        mask = np.clip((luminance - threshold) / max(1.0, (255 - threshold)), 0, 1)
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        # 模糊半径随 halation 增长
        blur_radius = 6 + halation * 18
        glow_mask = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        glow_arr = np.array(glow_mask).astype(np.float32) / 255.0
        # 暖色晕染，偏红橙
        halo = np.zeros_like(arr)
        halo[..., 0] = glow_arr * (70 + 120 * halation)   # R
        halo[..., 1] = glow_arr * (20 + 60 * halation)    # G
        halo[..., 2] = glow_arr * (10 + 30 * halation)    # B
        out = arr + halo
        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out)
    
    def save_image(self, image: Image.Image) -> Dict[str, Any]:
        file_name = f"{int(time.time())}_{uuid.uuid4().hex}.png"
        save_path = os.path.join(self.output_dir, file_name)
        image.save(save_path, format="PNG")
        result = {
            "saved_path": save_path,
            "file_name": file_name,
        }
        if self.base_url:
            result["image_url"] = self.base_url.rstrip("/") + f"/outputs/{file_name}"
        else:
            result["image_url"] = ""
        return result
    
    def process(
        self,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        max_size: int = 1536,
        grain: float = 0.0,
        halation: float = 0.0,
        preset: str = "kodak_gold_200"
    ) -> Dict[str, Any]:
        if not image_url and not image_base64:
            raise ValueError("Either image_url or image_base64 must be provided.")
        if image_url:
            image = load_image_from_url(image_url)
        else:
            image = decode_base64_image(image_base64)
        # 模型推理
        output = self.infer_pil(image, preset=preset, max_size=max_size)
        # 后处理
        output = self.apply_halation(output, halation=halation)
        output = self.apply_grain(output, grain=grain)
        meta = self.save_image(output)
        return {
            "ok": True,
            "width": output.size[0],
            "height": output.size[1],
            "saved_path": meta["saved_path"],
            "image_url": meta["image_url"],
            "file_name": meta["file_name"],
            "device": self.device,
            "preset": preset,
            "grain": grain,
            "halation": halation,
            "available_presets": list(self.models.keys()),
        }

MODEL_OPTIONS_PATH = "./options/digitalFilm.yaml"
APP_OPTIONS_PATH = "./options/app.yaml"
service = DigitalFilmMCPService(
    model_options_path=MODEL_OPTIONS_PATH,
    app_options_path=APP_OPTIONS_PATH
)
mcp = FastMCP("digital-film-tools")

@mcp.tool()
def film_generate_image(
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    max_size: int = 1536,
    grain: int = 0,
    halation: int = 0,
    preset: str = "kodak_gold_200"
) -> Dict[str, Any]:
    return service.process(
        image_url=image_url,
        image_base64=image_base64,
        max_size=max_size,
        grain=grain/100,
        halation=halation/100,
        preset=preset
    )


if __name__ == "__main__":
    mcp.run()
