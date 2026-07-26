"""Depth estimation wrapper for end-to-end training.

Uses Depth Anything V2 with a pure-PyTorch model loader that mirrors the
HuggingFace module hierarchy *exactly* so safetensors weights load directly
with minimal remapping.  No transformers ``from_pretrained`` involved.
"""

import json
import logging
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_DA_V2_PATCH = 14


# ======================================================================
#  Dinov2 backbone — identical module hierarchy to HF Dinov2Model so
#  safetensors keys match 1:1.
# ======================================================================

class _Dinov2Embeddings(nn.Module):
    """Matches ``Dinov2Embeddings`` from transformers."""

    def __init__(self, config: dict):
        super().__init__()
        patch_size = config.get("patch_size", 14)
        image_size = config.get("image_size", 518)
        self.patch_size = patch_size
        self.image_size = image_size
        self.hidden_size = config.get("hidden_size", 384)

        self.patch_embeddings = nn.Conv2d(
            3, self.hidden_size,
            kernel_size=patch_size, stride=patch_size, bias=True,
        )
        num_patches = (image_size // patch_size) ** 2
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.hidden_size),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.dropout = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, self.hidden_size))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B = pixel_values.shape[0]
        x = self.patch_embeddings(pixel_values)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.position_embeddings
        return self.dropout(x)


# -- sub-modules matching HF Dinov2 attention stack --------------------

class _Dinov2SelfAttention(nn.Module):
    """Separate q/k/v Linear layers — exactly ``Dinov2SelfAttention``."""

    def __init__(self, dim: int, num_heads: int, bias: bool = True):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = dim // num_heads
        self.all_head_size = self.attention_head_size * num_heads
        self.query = nn.Linear(dim, self.all_head_size, bias=bias)
        self.key = nn.Linear(dim, self.all_head_size, bias=bias)
        self.value = nn.Linear(dim, self.all_head_size, bias=bias)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        def _shape(t: torch.Tensor) -> torch.Tensor:
            B, N, _ = t.shape
            return t.view(B, N, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        q, k, v = _shape(q), _shape(k), _shape(v)
        scale = self.attention_head_size ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        if head_mask is not None:
            attn = attn * head_mask
        ctx = attn @ v
        ctx = ctx.transpose(1, 2).contiguous().flatten(2)
        return (ctx, attn) if output_attentions else (ctx,)


class _Dinov2SelfOutput(nn.Module):
    """Matches ``Dinov2SelfOutput`` — dense projection."""

    def __init__(self, dim: int, bias: bool = True):
        super().__init__()
        self.dense = nn.Linear(dim, dim, bias=bias)

    def forward(self, hidden_states, input_tensor):
        return self.dense(hidden_states)


class _Dinov2Attention(nn.Module):
    """HF ``Dinov2Attention``: self.attention + self.output."""

    def __init__(self, dim: int, num_heads: int, bias: bool = True):
        super().__init__()
        self.attention = _Dinov2SelfAttention(dim, num_heads, bias=bias)
        self.output = _Dinov2SelfOutput(dim, bias=bias)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)
        attn_out = self.output(self_outputs[0], hidden_states)
        return (attn_out,) + self_outputs[1:]


class _Dinov2MLP(nn.Module):
    """HF ``Dinov2MLP`` with fc1/fc2 naming."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        inner = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, inner)
        self.fc2 = nn.Linear(inner, dim)

    def forward(self, hidden_states):
        return self.fc2(F.gelu(self.fc1(hidden_states)))


class _Dinov2LayerScale(nn.Module):
    """HF Dinov2 LayerScale: ``self.lambda1`` Parameter."""

    def __init__(self, dim: int):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(dim) * 0.1)

    def forward(self, hidden_states):
        return self.lambda1 * hidden_states


class _Dinov2Layer(nn.Module):
    """HF ``Dinov2Layer`` — norm1, attention, layer_norm2, mlp, layer_scale."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attention = _Dinov2Attention(dim, num_heads)
        self.layer_scale1 = _Dinov2LayerScale(dim)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Dinov2MLP(dim, mlp_ratio)
        self.layer_scale2 = _Dinov2LayerScale(dim)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        sa_out = self.attention(
            self.norm1(hidden_states), head_mask, output_attentions,
        )
        hidden_states = hidden_states + self.layer_scale1(sa_out[0])
        hidden_states = hidden_states + self.layer_scale2(self.mlp(self.layer_norm2(hidden_states)))
        return (hidden_states,) + sa_out[1:]


class _Dinov2Encoder(nn.Module):
    """HF ``Dinov2Encoder`` — ``self.layer`` ModuleList."""

    def __init__(self, num_layers: int, dim: int, num_heads: int):
        super().__init__()
        self.layer = nn.ModuleList([
            _Dinov2Layer(dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, hidden_states, head_mask=None, output_attentions=False,
                output_hidden_states=False, return_dict=False):
        all_hidden = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden = all_hidden + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states,
                head_mask[i] if head_mask is not None else None,
                output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attns = all_attns + (layer_outputs[1],)
        return (hidden_states, all_hidden, all_attns)


class Dinov2Backbone(nn.Module):
    """Minimal Dinov2 ViT backbone matching HF Dinov2Model key layout."""

    def __init__(self, backbone_config: dict):
        super().__init__()
        hidden_size = backbone_config.get("hidden_size", 384)
        num_heads = backbone_config.get("num_attention_heads", 6)
        out_indices = backbone_config.get("out_indices", [9, 10, 11, 12])
        # out_indices is 1-based (HF convention); max=12 → 12 layers (index 0-11)
        num_layers = max(out_indices)

        self.embeddings = _Dinov2Embeddings(backbone_config)
        self.encoder = _Dinov2Encoder(num_layers, hidden_size, num_heads)

        self.out_indices = [i - 1 for i in out_indices]  # 1-based → 0-based
        self.apply_layernorm = backbone_config.get("apply_layernorm", True)
        self.layernorm = (
            nn.LayerNorm(hidden_size, eps=1e-6) if self.apply_layernorm else nn.Identity()
        )
        self._patch_size = backbone_config.get("patch_size", 14)

    def forward(self, pixel_values):
        x = self.embeddings(pixel_values)
        B, Np1, D = x.shape
        N = Np1 - 1
        pH = pW = int(N ** 0.5)

        feature_maps = []
        for i, layer_module in enumerate(self.encoder.layer):
            x = layer_module(x)[0]
            if i in self.out_indices:
                feature_maps.append(self.layernorm(x))

        return feature_maps, pH, pW


# ======================================================================
#  DA-V2 neck / head — attribute names match the vendored HF classes
#  so safetensors loads directly (``strict=True``).
# ======================================================================

class DepthAnythingReassembleLayer(nn.Module):
    """Matches vendored ``DepthAnythingReassembleLayer``."""

    def __init__(self, hidden_size: int, channels: int, factor: float):
        super().__init__()
        self.projection = nn.Conv2d(hidden_size, channels, 1)
        if factor > 1:
            self.resize = nn.ConvTranspose2d(channels, channels,
                                             kernel_size=int(factor), stride=int(factor))
        elif factor == 1:
            self.resize = nn.Identity()
        else:  # factor < 1
            self.resize = nn.Conv2d(channels, channels, 3, stride=int(1 / factor), padding=1)

    def forward(self, x):
        return self.resize(self.projection(x))


class DepthAnythingReassembleStage(nn.Module):
    """Matches vendored ``DepthAnythingReassembleStage``."""

    def __init__(self, hidden_size: int, neck_sizes: list[int], factors: list[float]):
        super().__init__()
        self.layers = nn.ModuleList([
            DepthAnythingReassembleLayer(hidden_size, ch, f)
            for ch, f in zip(neck_sizes, factors)
        ])

    def forward(self, hidden_states, patch_height, patch_width):
        out = []
        for layer, hs in zip(self.layers, hidden_states):
            hs = hs[:, 1:]   # drop cls token
            B, _, C = hs.shape
            hs = hs.reshape(B, patch_height, patch_width, C).permute(0, 3, 1, 2).contiguous()
            out.append(layer(hs))
        return out


class DepthAnythingPreActResidualLayer(nn.Module):
    """Matches vendored ``DepthAnythingPreActResidualLayer``."""

    def __init__(self, channels: int):
        super().__init__()
        self.activation1 = nn.ReLU()
        self.convolution1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.activation2 = nn.ReLU()
        self.convolution2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)

    def forward(self, x):
        return x + self.convolution2(self.activation2(self.convolution1(self.activation1(x))))


class DepthAnythingFeatureFusionLayer(nn.Module):
    """Matches vendored ``DepthAnythingFeatureFusionLayer``."""

    def __init__(self, channels: int):
        super().__init__()
        self.projection = nn.Conv2d(channels, channels, 1, bias=True)
        self.residual_layer1 = DepthAnythingPreActResidualLayer(channels)
        self.residual_layer2 = DepthAnythingPreActResidualLayer(channels)

    def forward(self, feat, residual=None, size=None):
        if residual is not None:
            if feat.shape != residual.shape:
                residual = F.interpolate(residual, size=feat.shape[2:],
                                         mode="bilinear", align_corners=False)
            feat = feat + self.residual_layer1(residual)
        feat = self.residual_layer2(feat)
        mod = {"scale_factor": 2} if size is None else {"size": size}
        return self.projection(F.interpolate(feat, **mod, mode="bilinear", align_corners=True))


class DepthAnythingFeatureFusionStage(nn.Module):
    """Matches vendored ``DepthAnythingFeatureFusionStage``."""

    def __init__(self, num_layers: int, channels: int):
        super().__init__()
        self.layers = nn.ModuleList([
            DepthAnythingFeatureFusionLayer(channels) for _ in range(num_layers)
        ])

    def forward(self, features):
        features = features[::-1]
        fused = []
        size = features[1].shape[2:]
        out = self.layers[0](features[0], size=size)
        fused.append(out)
        for idx, (feat, layer) in enumerate(zip(features[1:], self.layers[1:])):
            next_feat = features[1:][idx + 1] if idx < len(features[1:]) - 1 else None
            size = next_feat.shape[2:] if next_feat is not None else None
            out = layer(out, feat, size=size)
            fused.append(out)
        return fused


class DepthAnythingNeck(nn.Module):
    """Matches vendored ``DepthAnythingNeck``."""

    def __init__(self, config: dict):
        super().__init__()
        hidden = config["reassemble_hidden_size"]
        neck_sizes = config["neck_hidden_sizes"]
        factors = config["reassemble_factors"]
        fusion_ch = config["fusion_hidden_size"]

        self.reassemble_stage = DepthAnythingReassembleStage(hidden, neck_sizes, factors)
        self.convs = nn.ModuleList([
            nn.Conv2d(ch, fusion_ch, 3, 1, 1, bias=False) for ch in neck_sizes
        ])
        self.fusion_stage = DepthAnythingFeatureFusionStage(len(neck_sizes), fusion_ch)

    def forward(self, hidden_states, patch_height, patch_width):
        hs = self.reassemble_stage(hidden_states, patch_height, patch_width)
        return self.fusion_stage([conv(f) for conv, f in zip(self.convs, hs)])


class DepthAnythingDepthEstimationHead(nn.Module):
    """Matches vendored ``DepthAnythingDepthEstimationHead``."""

    def __init__(self, config: dict):
        super().__init__()
        self.head_in_index = config.get("head_in_index", -1)
        self.patch_size = config.get("patch_size", 14)
        feat = config["fusion_hidden_size"]
        hh = config["head_hidden_size"]
        self.conv1 = nn.Conv2d(feat, feat // 2, 3, 1, 1)
        self.conv2 = nn.Conv2d(feat // 2, hh, 3, 1, 1)
        self.activation1 = nn.ReLU()
        self.conv3 = nn.Conv2d(hh, 1, 1, 1, 0)
        self.activation2 = nn.ReLU()

    def forward(self, features, patch_height, patch_width):
        x = features[self.head_in_index]
        x = self.conv1(x)
        x = F.interpolate(
            x,
            size=(patch_height * self.patch_size, patch_width * self.patch_size),
            mode="bilinear", align_corners=True,
        )
        x = self.conv2(x)
        x = self.activation1(x)
        x = self.conv3(x)
        x = self.activation2(x)
        return x.squeeze(1)


# ======================================================================
#  Full model
# ======================================================================

class DepthAnythingV2Model(nn.Module):
    """Full Depth Anything V2 assembled from config + safetensors weights."""

    def __init__(self, config: dict):
        super().__init__()
        backbone_cfg = config.get("backbone_config") or {}
        self.backbone = Dinov2Backbone(backbone_cfg)
        self.neck = DepthAnythingNeck(config)
        self.head = DepthAnythingDepthEstimationHead(config)

    def forward(self, pixel_values):
        hidden_states, patch_h, patch_w = self.backbone(pixel_values)
        features = self.neck(hidden_states, patch_h, patch_w)
        depth = self.head(features, patch_h, patch_w)
        return type("_DAOutput", (), {"predicted_depth": depth})()


# ======================================================================
#  Estimator
# ======================================================================

class DepthAnythingV2Estimator:
    """Estimate metric depth (mm) from sRGB batches."""

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Base-hf",
        depth_min_mm: float = 300.0,
        depth_max_mm: float = 5000.0,
        infer_size: int = 518,
        device=None,
        cache_dir=None,
    ):
        if infer_size % _DA_V2_PATCH != 0:
            raise ValueError(f"infer_size must be multiple of {_DA_V2_PATCH}; got {infer_size}")

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_min_mm = float(depth_min_mm)
        self.depth_max_mm = float(depth_max_mm)
        self.infer_size = int(infer_size)

        snapshot_dir = self._resolve_snapshot(
            Path(cache_dir).expanduser().resolve() if cache_dir else None,
            model_name,
        )
        logger.info(f"Loading depth estimator from: {snapshot_dir}")

        with open(snapshot_dir / "config.json", "r") as f:
            config = json.load(f)

        self.model = DepthAnythingV2Model(config).to(self.device)

        # ------------------------------------------------------------------
        #  Load safetensors weights — since we mirrored the HF hierarchy
        #  exactly, most keys should map 1:1.
        # ------------------------------------------------------------------
        weights_path = snapshot_dir / "model.safetensors"
        if not weights_path.is_file():
            raise FileNotFoundError(f"model.safetensors not found in {snapshot_dir}")
        from safetensors.torch import load_file
        ckpt = load_file(str(weights_path))

        ckpt_keys = set(ckpt.keys())
        model_state = self.model.state_dict()
        model_keys = set(model_state.keys())

        new_state: dict[str, torch.Tensor] = {}

        # Direct match
        for k in ckpt_keys:
            if k in model_keys:
                new_state[k] = ckpt[k]

        unmatch_ckpt = ckpt_keys - set(new_state.keys())
        unmatch_model = model_keys - set(new_state.keys())
        logger.warning(f"Direct: {len(new_state)}/{len(ckpt_keys)} matched. Unmatched ckpt={len(unmatch_ckpt)} model={len(unmatch_model)}")

        if unmatch_ckpt:
            uc = sorted(unmatch_ckpt)
            logger.warning(f"Ckpt unmatched ({len(uc)}):\n" + "\n".join(f"  {k}" for k in uc))
        if unmatch_model:
            um = sorted(unmatch_model)
            logger.warning(f"Model unmatched ({len(um)}):\n" + "\n".join(f"  {k}" for k in um))

        # ---- robust remap -------------------------------------------------
        remap_count = 0
        RULES = [
            # (checkpoint pattern, model pattern) — ckpt_key → model_key
            # patch_embeddings: wrapped vs flat conv
            ("patch_embeddings.projection.", "patch_embeddings."),
            ("patch_embeddings.", "patch_embeddings.projection."),
            # layer_norm2 vs norm2 (both directions)
            (".layer_norm2.", ".norm2."),
            (".norm2.", ".layer_norm2."),
            # layer_norm1 vs norm1
            (".layer_norm1.", ".norm1."),
            (".norm1.", ".layer_norm1."),
        ]

        for ck in sorted(unmatch_ckpt):
            for ck_pat, mod_pat in RULES:
                if ck_pat in ck:
                    mapped = ck.replace(ck_pat, mod_pat)
                    if mapped in unmatch_model:
                        new_state[mapped] = ckpt[ck]
                        unmatch_ckpt.discard(ck)
                        unmatch_model.discard(mapped)
                        remap_count += 1
                        break

        logger.warning(f"Remapped {remap_count} more keys. Left: ckpt={len(unmatch_ckpt)} model={len(unmatch_model)}")

        if unmatch_ckpt:
            logger.warning(f"Ckpt still unmatched: {sorted(unmatch_ckpt)}")
        if unmatch_model:
            logger.warning(f"Model still unmatched: {sorted(unmatch_model)}")

        missing, unexpected = self.model.load_state_dict(new_state, strict=False)
        logger.warning(f"load_state_dict: {len(new_state)} loaded, {len(missing)} missing, {len(unexpected)} unexpected")
        if missing:
            logger.warning(f"Missing: {missing}")
        if unexpected:
            logger.warning(f"Unexpected: {unexpected}")

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self._mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    # ------------------------------------------------------------------
    #  helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_snapshot(cache_dir: Path | None, model_name: str) -> Path:
        org_repo = model_name.replace("/", "--")
        candidates: list[Path] = []
        if cache_dir is not None:
            candidates.append(cache_dir / f"models--{org_repo}")
        hf_default = Path.home() / ".cache" / "huggingface" / "hub"
        if hf_default.is_dir():
            candidates.append(hf_default / f"models--{org_repo}")

        for repo_dir in candidates:
            if not repo_dir.is_dir():
                continue
            ref = repo_dir / "refs" / "main"
            if ref.is_file():
                commit = ref.read_text().strip()
            else:
                snaps_dir = repo_dir / "snapshots"
                snaps = sorted(snaps_dir.iterdir()) if snaps_dir.is_dir() else []
                if not snaps:
                    continue
                commit = snaps[0].name
            snapshot = repo_dir / "snapshots" / commit
            if snapshot.is_dir():
                return snapshot

        raise FileNotFoundError(
            f"Snapshot not found for {model_name}.  Tried: {[str(c) for c in candidates]}"
        )

    # ------------------------------------------------------------------
    #  inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def estimate(self, rgb):
        if rgb.dim() != 4 or rgb.shape[1] != 3:
            raise ValueError(f"Expected (B,3,H,W) sRGB; got {tuple(rgb.shape)}")

        B, _, H, W = rgb.shape
        rgb = rgb.to(self.device)
        rgb_norm = (rgb - self._mean) / self._std
        rgb_resized = F.interpolate(
            rgb_norm, size=(self.infer_size, self.infer_size),
            mode="bilinear", align_corners=False,
        )

        outputs = self.model(pixel_values=rgb_resized)
        disp = outputs.predicted_depth.unsqueeze(1)
        disp = F.interpolate(disp, size=(H, W), mode="bilinear", align_corners=False)

        disp_min = disp.amin(dim=(2, 3), keepdim=True)
        disp_max = disp.amax(dim=(2, 3), keepdim=True)
        disp_norm = (disp - disp_min) / (disp_max - disp_min + 1e-8)

        disp_near = 1.0 / self.depth_min_mm
        disp_far = 1.0 / self.depth_max_mm
        disp_phys = disp_far + disp_norm * (disp_near - disp_far)
        depth_mm = 1.0 / disp_phys
        return depth_mm
