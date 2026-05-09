import os
import time
import uuid
from typing import Dict, Any

import numpy as np
import requests
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image, ImageFilter, ImageEnhance
import os
import sys
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request, Header

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from options.options import everyThingOptions
from models.digitalFilm_v2 import digitalFilmv2
from utils.utils import (
    ensure_dir,
    resize_to_multiple_of_16,
    pil_to_tensor,
    tensor_to_pil,
    load_image_from_url,
)


# =========================
# Pydantic Models
# =========================

class BasicAdjustments(BaseModel):
    exposure: float = 0
    contrast: float = 0
    highlights: float = 0
    shadows: float = 0
    temperature: float = 0
    tint: float = 0
    saturation: float = 0


class FilmStyleSettings(BaseModel):
    preset: str = "kodak_gold_200"
    grain: int = 0
    halation: int = 0


class FilmGenerateRequest(BaseModel):
    image_id: str
    basic: BasicAdjustments = BasicAdjustments()
    film: FilmStyleSettings = FilmStyleSettings()
    max_size: int = 1536


class MasterGetImageRequest(BaseModel):
    image_id: str


class MasterGetImageResponse(BaseModel):
    ok: bool
    image_id: str
    origin_url: str


class MasterRegisterResultRequest(BaseModel):
    image_id: str
    file_name: str
    relative_path: str
    width: int
    height: int
    basic: Dict[str, Any]
    film: Dict[str, Any]
    device: str


class MasterRegisterResultResponse(BaseModel):
    ok: bool
    result_url: str


# =========================
# Service
# =========================

class DigitalFilmHTTPService:
    def __init__(self, model_options_path: str, app_options_path: str):
        self.model_config = everyThingOptions(model_options_path)
        self.model_config.load_config()

        self.app_config = everyThingOptions(app_options_path)
        self.app_config.load_config()

        self.device = "cuda" if (
                getattr(self.app_config.opt.global_config, "cuda", False)
                and torch.cuda.is_available()
        ) else "cpu"

        self.root_dir = getattr(
            self.app_config.opt.global_config,
            "server_root",
            "./server_data"
        )
        self.output_dir = os.path.join(self.root_dir, "result")
        ensure_dir(self.output_dir)

        self.models = self.load_models()

        self.master_server_url = getattr(
            self.app_config.opt.global_config,
            "master_server_url",
            "http://127.0.0.1:8080"
        )

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
        checkpoints = getattr(self.model_config.opt, "checkpoints", None)
        if checkpoints is None:
            raise ValueError("No checkpoints config found.")

        models = {}
        for preset_name, ckpt_path in checkpoints.items():
            models[preset_name] = self.load_single_model(ckpt_path)

        if not models:
            raise ValueError("No models loaded. Please check checkpoints config.")
        return models

    def get_origin_image_url(self, image_id: str, auth_token: str) -> str:
        url = self.master_server_url.rstrip("/") + "/internal/images/get"
        headers = {"Authorization": auth_token} if auth_token else {}

        # 这里的 payload 对应你后端的 GetImageRequest 结构体
        resp = requests.post(url, json={"image_id": image_id}, headers=headers, timeout=10)
        if resp.status_code != 200:
            raise ValueError(f"Fetch origin URL failed: {resp.text}")

        data = resp.json()
        return data.get("origin_url")

    def register_result_to_master(
        self,
        image_id: str,
        file_name: str,
        relative_path: str,
        width: int,
        height: int,
        basic: Dict[str, Any],
        film: Dict[str, Any],
        device: str,
        auth_token: str = None  # 接收从前端透传过来的 Token
    ) -> str:
        url = self.master_server_url.rstrip("/") + "/internal/film/register_result"

        # 构建请求体
        payload = {
            "image_id": image_id,
            "file_name": file_name,
            "relative_path": relative_path,
            "width": width,
            "height": height,
            "basic": basic,
            "film": film,
            "device": device,
        }

        # --- 修改部分：构建请求头 ---
        headers = {}
        if auth_token:
            # 确保将 Token 放入 Authorization 字段
            headers["Authorization"] = auth_token
        # --------------------------

        try:
            # 发送请求时带上 headers
            resp = requests.post(url, json=payload, headers=headers, timeout=30)

            if resp.status_code != 200:
                raise ValueError(f"Master register result failed (Status {resp.status_code}): {resp.text}")

            data = resp.json()
            if not data.get("ok"):
                raise ValueError(f"Master register result logic failed: {data.get('error', 'Unknown error')}")

            result_url = data.get("result_url", "")
            if not result_url:
                raise ValueError("result_url is empty in master response")

            return result_url

        except requests.exceptions.RequestException as e:
            # 捕获网络层面的异常（如超时、连接拒绝）
            raise ValueError(f"Connect to master server failed: {str(e)}")

    def apply_basic_adjustments(self, image: Image.Image, basic: BasicAdjustments) -> Image.Image:
        image = image.convert("RGB")

        # exposure
        if basic.exposure != 0:
            arr = np.array(image).astype(np.float32)
            factor = 2 ** basic.exposure
            arr *= factor
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            image = Image.fromarray(arr)

        # contrast
        if basic.contrast != 0:
            factor = 1.0 + basic.contrast / 100.0
            factor = max(0.0, factor)
            image = ImageEnhance.Contrast(image).enhance(factor)

        # saturation
        if basic.saturation != 0:
            factor = 1.0 + basic.saturation / 100.0
            factor = max(0.0, factor)
            image = ImageEnhance.Color(image).enhance(factor)

        # temperature / tint
        if basic.temperature != 0 or basic.tint != 0:
            arr = np.array(image).astype(np.float32)
            temp_shift = basic.temperature / 100.0 * 30.0
            tint_shift = basic.tint / 100.0 * 20.0

            arr[..., 0] += temp_shift
            arr[..., 2] -= temp_shift
            arr[..., 1] += tint_shift

            arr = np.clip(arr, 0, 255).astype(np.uint8)
            image = Image.fromarray(arr)

        # highlights / shadows
        if basic.highlights != 0 or basic.shadows != 0:
            arr = np.array(image).astype(np.float32)
            luminance = (
                    0.299 * arr[..., 0]
                    + 0.587 * arr[..., 1]
                    + 0.114 * arr[..., 2]
            )

            if basic.highlights != 0:
                strength = basic.highlights / 100.0
                mask = np.clip((luminance - 128) / 127.0, 0, 1)
                arr = arr + strength * mask[..., None] * 40.0

            if basic.shadows != 0:
                strength = basic.shadows / 100.0
                mask = np.clip((128 - luminance) / 128.0, 0, 1)
                arr = arr + strength * mask[..., None] * 40.0

            arr = np.clip(arr, 0, 255).astype(np.uint8)
            image = Image.fromarray(arr)

        return image

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

        with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=amp_enabled
        ):
            out = model.g(x)

        out = out["out"]
        result = tensor_to_pil(out)
        return result

    def apply_grain(self, image: Image.Image, grain: float = 0.0) -> Image.Image:
        grain = max(0.0, min(1.0, grain))
        if grain <= 0:
            return image

        arr = np.array(image).astype(np.float32)
        noise_std = 8.0 + grain * 32.0
        noise = np.random.normal(0, noise_std, arr.shape).astype(np.float32)

        luminance = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        luminance = luminance / 255.0
        weight = 1.15 - 0.5 * luminance
        weight = np.expand_dims(weight, axis=-1)

        arr = arr + noise * weight
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def apply_halation(self, image: Image.Image, halation: float = 0.0) -> Image.Image:
        halation = max(0.0, min(1.0, halation))
        if halation <= 0:
            return image

        base = image.convert("RGB")
        arr = np.array(base).astype(np.float32)

        luminance = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        threshold = 180 - halation * 40
        mask = np.clip((luminance - threshold) / max(1.0, (255 - threshold)), 0, 1)
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")

        blur_radius = 6 + halation * 18
        glow_mask = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        glow_arr = np.array(glow_mask).astype(np.float32) / 255.0

        halo = np.zeros_like(arr)
        halo[..., 0] = glow_arr * (70 + 120 * halation)
        halo[..., 1] = glow_arr * (20 + 60 * halation)
        halo[..., 2] = glow_arr * (10 + 30 * halation)

        out = arr + halo
        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    def save_image(self, image: Image.Image) -> Dict[str, Any]:
        file_name = f"{int(time.time())}_{uuid.uuid4().hex}.png"
        save_path = os.path.join(self.output_dir, file_name)
        image.save(save_path, format="PNG")

        # 给 master 用的相对路径
        relative_path = f"results/{file_name}"

        return {
            "saved_path": save_path,
            "file_name": file_name,
            "relative_path": relative_path,
        }

    def process(
        self,
        image_id: str,
        basic: BasicAdjustments,
        film: FilmStyleSettings,
        max_size: int = 1536,
        auth_token: str = None  # 接收 Token
    ) -> Dict[str, Any]:
        # 1. 从 master 获取原图 URL
        origin_url = self.get_origin_image_url(image_id, auth_token)

        # 2. 下载原图
        image = load_image_from_url(origin_url)

        # 3. basic 调整
        image = self.apply_basic_adjustments(image, basic)

        # 4. 模型推理
        output = self.infer_pil(image, preset=film.preset, max_size=max_size)

        # 5. 后处理
        output = self.apply_halation(output, halation=film.halation / 100.0)
        output = self.apply_grain(output, grain=film.grain / 100.0)

        # 6. 保存本地结果
        meta = self.save_image(output)

        # 7. 把结果登记到 master，拿最终 static_server URL
        result_url = self.register_result_to_master(
            image_id=image_id,
            file_name=meta["file_name"],
            relative_path=meta["relative_path"],
            width=output.size[0],
            height=output.size[1],
            basic=basic.dict(),
            film=film.dict(),
            device=self.device,
            auth_token=auth_token)

        # 8. 返回前端
        return {
            "ok": True,
            "image_id": image_id,
            "result_url": result_url,
            "width": output.size[0],
            "height": output.size[1],
        }


# =========================
# Bootstrap
# =========================

MODEL_OPTIONS_PATH = "./config/digitalFilm.yaml"
APP_OPTIONS_PATH = "./config/image_backend.yaml"

service = DigitalFilmHTTPService(
    model_options_path=MODEL_OPTIONS_PATH,
    app_options_path=APP_OPTIONS_PATH
)

app = FastAPI(title="Digital Film HTTP Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "ok": True,
        "device": service.device,
        "available_presets": list(service.models.keys()),
    }


@app.post("/api/film/generate")
def film_generate(
        req: FilmGenerateRequest,
        authorization: str = Header(None)  # 自动获取请求头中的 Authorization
):
    try:
        return service.process(
            image_id=req.image_id,
            basic=req.basic,
            film=req.film,
            max_size=req.max_size,
            auth_token=authorization,  # 传递 Token
        )
    except Exception as e:
        # 建议记录日志方便排查
        print(f"Generation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app.backend.image_server:app", host="127.0.0.1", port=7070, reload=False)
