"""
DCAE-based DiffLoss Residual SR Model (with optional CNN Mapper)
=================================================================

Pipeline: LQ → DCAE.encoder → [Mapper(deterministic) + DiffLoss(stochastic)] → decoder → SR

- No SD2.1 dependency (no text_encoder, tokenizer, noise_scheduler).
- Autoencoder is DCAE (frozen), from a HuggingFace checkpoint.
- Mapper (CNN) learns the deterministic latent residual: encoder(target) - encoder(lq).
- DiffLoss learns the stochastic prediction error: (true_residual - mapper_output).
  All supervision (DiffLoss + pixel) is aligned to the teacher, not GT.
- At inference: res_hat = mapper(lq_latent) + diffloss.sample(z=lq_latent) → decode.
- If no mapper_cfg given, DiffLoss predicts the full residual (backward compatible).

Usage:
    from student import DCAESR

    cfg = {"dcae_preset": "dc-ae-f32c32-in-1.0", "dcae_pretrained_path": "...",
           "in_channels": 32, "diffloss_cfg": {...}, "mapper_cfg": {...}}
    model = DCAESR(model_cfg=cfg)
    out = model(lq)          # single forward, no text needed
"""

import os
from typing import List, Optional, Tuple
import sys

import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from efficient_vit.dcae_zoo import dc_ae_from_pretrained
from diffloss.diffloss import DiffLoss


# ═══════════════════════════════════════════════════════════════════════════════
#  Tiled encode / decode utilities for high-resolution inference
# ═══════════════════════════════════════════════════════════════════════════════


def _blend_weight_map(tile_h: int, tile_w: int, feather: int) -> torch.Tensor:
    """
    Create a 2D blend weight map with a linear ramp from 1 in the center
    to 0 at the edges over ``feather`` pixels.  Shape: (1, 1, tile_h, tile_w).
    """
    wy = torch.ones(tile_h)
    wx = torch.ones(tile_w)
    if feather > 0 and tile_h > 2 * feather:
        ramp = torch.linspace(0, 1, feather)
        wy[:feather] = ramp
        wy[-feather:] = ramp.flip(0)
    if feather > 0 and tile_w > 2 * feather:
        ramp = torch.linspace(0, 1, feather)
        wx[:feather] = ramp
        wx[-feather:] = ramp.flip(0)
    w = wy[:, None] * wx[None, :]   # (H, W)
    return w[None, None, ...]       # (1, 1, H, W)


def _tile_coords(total: int, tile: int, feather: int):
    """
    Generate (start, end, blend_valid_start, blend_valid_end) for each tile
    along one spatial dimension.
    Returns list of (s, e, vs, ve).
    """
    if tile >= total:
        return [(0, total, 0, total)]
    coords = []
    pos = 0
    while pos < total:
        e = min(pos + tile, total)
        vs = pos if pos == 0 else pos + feather
        ve = e if e == total else e - feather
        coords.append((pos, e, vs, ve))
        if e == total:
            break
        pos = e - 2 * feather
    return coords


@torch.no_grad()
def tiled_encode(
    encoder: torch.nn.Module,
    x: torch.Tensor,
    tile_size: int = 1024,
    feather: int = 64,
    spatial_compression: int = 32,
) -> torch.Tensor:
    """
    Encode a high-resolution image by tiling with overlapping feather regions.

    Args:
        encoder:   DCAE encoder (callable: pixel → latent).
        x:         (B, 3, H, W) input image, value range matches encoder training.
        tile_size: pixel tile size (before overlap).
        feather:   overlap width in pixel space (maps to feather//compression in latent).
        spatial_compression: latent compression ratio (e.g. 32 for f32c32).

    Returns:
        (B, C, H//comp, W//comp) latent tensor.
    """
    B, C, H, W = x.shape
    comp = spatial_compression
    y_rows = _tile_coords(H, tile_size, feather)
    y_cols = _tile_coords(W, tile_size, feather)

    if len(y_rows) == 1 and len(y_cols) == 1:
        return encoder(x)

    oh, ow = H // comp, W // comp

    # Encode first tile to determine output channels
    first_tile = x[:, :, y_rows[0][0]:y_rows[0][1], y_cols[0][0]:y_cols[0][1]]
    first_lat = encoder(first_tile)
    out_c = first_lat.shape[1]

    accum = torch.zeros(B, out_c, oh, ow, device=x.device, dtype=torch.float32)
    weight_acc = torch.zeros(1, 1, oh, ow, device=x.device, dtype=torch.float32)

    for idx_y, (sy, ey, vsy, vey) in enumerate(y_rows):
        for idx_x, (sx, ex, vsx, vex) in enumerate(y_cols):
            if idx_y == 0 and idx_x == 0:
                lat = first_lat
            else:
                tile = x[:, :, sy:ey, sx:ex]
                lat = encoder(tile)

            th, tw = lat.shape[2], lat.shape[3]
            wt = _blend_weight_map(th, tw, feather // comp).to(device=x.device, dtype=torch.float32)

            l_sy = sy // comp
            l_ey = l_sy + th
            l_sx = sx // comp
            l_ex = l_sx + tw

            accum[:, :, l_sy:l_ey, l_sx:l_ex] += lat.float() * wt
            weight_acc[:, :, l_sy:l_ey, l_sx:l_ex] += wt

    return (accum / weight_acc.clamp(min=1e-8)).to(dtype=x.dtype)


@torch.no_grad()
def tiled_decode(
    decoder: torch.nn.Module,
    x: torch.Tensor,
    tile_size: int = 256,
    feather: int = 16,
    spatial_compression: int = 32,
) -> torch.Tensor:
    """
    Decode a large latent by tiling with overlapping feather regions.

    Args:
        decoder:    DCAE decoder (callable: latent → pixel).
        x:          (B, C, h, w) latent tensor.
        tile_size:  latent tile size (before overlap).
        feather:    overlap width in latent space.
        spatial_compression: latent compression ratio.

    Returns:
        (B, 3, H, W) pixel tensor.
    """
    B, C, h, w = x.shape
    comp = spatial_compression

    y_rows = _tile_coords(h, tile_size, feather)
    y_cols = _tile_coords(w, tile_size, feather)

    if len(y_rows) == 1 and len(y_cols) == 1:
        return decoder(x)

    H, W = h * comp, w * comp

    accum = torch.zeros(B, 3, H, W, device=x.device, dtype=torch.float32)
    weight_acc = torch.zeros(1, 1, H, W, device=x.device, dtype=torch.float32)

    for sy, ey, vsy, vey in y_rows:
        for sx, ex, vsx, vex in y_cols:
            tile = x[:, :, sy:ey, sx:ex]
            px = decoder(tile)  # (B, 3, th*comp, tw*comp)

            ph, pw = px.shape[2], px.shape[3]
            wt = _blend_weight_map(ph, pw, feather * comp).to(device=x.device, dtype=torch.float32)

            p_sy = sy * comp
            p_ey = p_sy + ph
            p_sx = sx * comp
            p_ex = p_sx + pw

            accum[:, :, p_sy:p_ey, p_sx:p_ex] += px.float() * wt
            weight_acc[:, :, p_sy:p_ey, p_sx:p_ex] += wt

    return (accum / weight_acc.clamp(min=1e-8)).to(dtype=x.dtype)


# ═══════════════════════════════════════════════════════════════════════════════
#  LightMapper — lightweight CNN that predicts the deterministic latent residual
# ═══════════════════════════════════════════════════════════════════════════════


class ConvBlock(nn.Module):
    """Conv → GroupNorm → SiLU, optionally with a residual connection."""

    def __init__(self, channels: int, kernel_size: int = 3, groups: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.GroupNorm(min(channels // 4, 32), channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size,
                              padding=padding, groups=groups)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.conv(self.norm(x)))


class ResBlock2d(nn.Module):
    """Residual block: two ConvBlocks with a skip connection."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.block1 = ConvBlock(channels, kernel_size)
        self.block2 = ConvBlock(channels, kernel_size)

    def forward(self, x):
        return x + self.block2(self.block1(x))


class LightMapper(nn.Module):
    """
    Lightweight CNN mapper that predicts the deterministic component of the
    latent residual:  mapper(lq_latent) ≈ encoder(target) - encoder(lq).

    DiffLoss then only needs to model the stochastic prediction error, which
    can be run with fewer sampling steps / a smaller network.

    Architecture:
      lq_latent (B, C, H, W)
        → Conv in (C → hidden)
        → N × ResBlock2d
        → Conv out (hidden → C)
        → output (same shape as input)

    Args:
        in_channels:   latent channel count (e.g. 32 for f32c32).
        hidden:        internal channel width (default 64).
        num_blocks:    number of residual blocks (default 4).
        kernel_size:   conv kernel size (default 3).
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int = 64,
        num_blocks: int = 4,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        padding = kernel_size // 2

        self.conv_in = nn.Conv2d(in_channels, hidden, kernel_size, padding=padding)
        self.blocks = nn.Sequential(*[
            ResBlock2d(hidden, kernel_size) for _ in range(num_blocks)
        ])
        self.norm_out = nn.GroupNorm(min(hidden // 4, 32), hidden)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(hidden, in_channels, kernel_size, padding=padding)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Zero-init the last conv so the mapper starts as identity (output ≈ 0)
        nn.init.constant_(self.conv_out.weight, 0)
        nn.init.constant_(self.conv_out.bias, 0)

    def forward(self, lq_latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lq_latent:  (B, C, H, W) — encoded LQ image latent.
        Returns:
            Deterministic residual estimate, same shape as input.
        """
        h = self.conv_in(lq_latent)
        h = self.blocks(h)
        h = self.act_out(self.norm_out(h))
        return self.conv_out(h)



# ═══════════════════════════════════════════════════════════════════════════════
#  DCAESR — DiffLoss-based SR model (with optional CNN mapper)
# ═══════════════════════════════════════════════════════════════════════════════


class DCAESR(nn.Module):
    """
    DCAE-based lightweight SR model — optional CNN mapper + DiffLoss.

    Pipeline:
      LQ → DCAE.encoder → [Mapper(det.) + DiffLoss(stoch.)] → DCAE.decoder → SR

    If mapper is enabled:
      - Mapper learns the deterministic latent residual.
      - DiffLoss learns the stochastic prediction error (true_residual - mapper_output).
      This means DiffLoss only needs to model the "difficult" part that the mapper
      cannot capture, which can be run with fewer steps / a smaller network.

    If mapper is disabled (mapper_cfg omitted):
      - DiffLoss predicts the full residual (backward compatible).

    All components except DiffLoss (and mapper, if enabled) are frozen.

    Args:
        model_cfg: Config dict with keys:
            - dcae_preset, dcae_pretrained_path  — DCAE autoencoder
            - in_channels                        — latent channel count
            - diffloss_cfg  (optional)           — DiffLoss config
            - mapper_cfg    (optional)           — LightMapper config
              - hidden:        int (default 64)
              - num_blocks:    int (default 4)
              - kernel_size:   int (default 3)
    """

    def __init__(self, model_cfg: dict):
        super().__init__()
        assert model_cfg.get("dcae_preset") is not None, \
            "dcae_preset is required. Use a dcae_* preset from student_zoo."

        self._model_cfg = model_cfg

        # ---- DCAE autoencoder (frozen for full, trainable decoder for pruned) ----
        dcae_preset = model_cfg["dcae_preset"]
        dcae_pretrained = model_cfg.get("dcae_pretrained_path", "")

        # If pretrained path is empty, build from scratch (pruned DCAE — no weights)
        if dcae_pretrained:
            self.autoencoder = dc_ae_from_pretrained(dcae_preset, dcae_pretrained)
        else:
            from efficient_vit.dcae_zoo import dc_ae_zoo
            cfg = dc_ae_zoo(dcae_preset)
            from efficient_vit.efficientvit.dc_ae import DCAE
            self.autoencoder = DCAE(cfg)

        in_channels = model_cfg.get("in_channels", 32)

        # ---- LightMapper (optional, deterministic residual) ----
        mapper_cfg = model_cfg.get("mapper_cfg", None)
        if mapper_cfg is not None:
            self.mapper = LightMapper(
                in_channels=in_channels,
                hidden=mapper_cfg.get("hidden", 64),
                num_blocks=mapper_cfg.get("num_blocks", 4),
                kernel_size=mapper_cfg.get("kernel_size", 3),
            )
        else:
            self.mapper = None

        # ---- DiffLoss (optional, stochastic residual / error) ----
        diffloss_cfg = model_cfg.get("diffloss_cfg", None)
        if diffloss_cfg is not None:
            self.diffloss = DiffLoss(
                target_channels=in_channels,
                z_channels=in_channels,
                depth=diffloss_cfg.get("depth", 4),
                width=diffloss_cfg.get("width", 512),
                num_sampling_steps=diffloss_cfg.get("num_sampling_steps", [16]),
                sampler=diffloss_cfg.get("sampler", "iddpm"),
                vae_scale=diffloss_cfg.get("vae_scale", 0.6),
            )
        else:
            self.diffloss = None

    # ── Freeze / train helpers ─────────────────────────────────────────────

    def set_train(self):
        """Set DiffLoss and mapper to train mode.  Freeze encoder; decoder
        optionally trainable for pruned DCAE pretraining."""
        self.autoencoder.eval()
        self.autoencoder.requires_grad_(False)
        if self.diffloss is not None:
            self.diffloss.train()
        if self.mapper is not None:
            self.mapper.train()

    def set_train_decoder(self):
        """Pretraining mode: freeze encoder, train decoder only."""
        self.autoencoder.encoder.eval()
        self.autoencoder.encoder.requires_grad_(False)
        self.autoencoder.decoder.train()
        self.autoencoder.decoder.requires_grad_(True)
        if self.diffloss is not None:
            self.diffloss.eval()
            self.diffloss.requires_grad_(False)
        if self.mapper is not None:
            self.mapper.eval()
            self.mapper.requires_grad_(False)

    def enable_gradient_checkpointing(self):
        """No-op kept for compatibility; DiffLoss MLP doesn't use checkpointing."""
        pass

    # ── Forward ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward_tiled(
        self,
        lq: torch.Tensor,
        encoder_tile_size: int = 1024,
        decoder_tile_size: int = 256,
        encoder_feather: int = 64,
        decoder_feather: int = 16,
    ) -> torch.Tensor:
        """
        Tile-based inference for high-resolution images.

        Splits the DCAE encoder/decoder into overlapping tiles to stay within
        VRAM limits.  Mapper (CNN) and DiffLoss (MLP) operate on the full
        latent directly — they scale naturally to any spatial size.

        Args:
            lq:                (1, 3, H, W) input image in [-1, 1].
            encoder_tile_size:  pixel tile size for encoder (default 1024).
            decoder_tile_size:  latent tile size for decoder (default 256).
            encoder_feather:    overlap width for encoder in pixel space (default 64).
            decoder_feather:    overlap width for decoder in latent space (default 16).

        Returns:
            (1, 3, H, W) refined SR image in [-1, 1].
        """
        self.autoencoder.eval()
        if self.mapper is not None:
            self.mapper.eval()
        if self.diffloss is not None:
            self.diffloss.eval()

        comp = self._model_cfg.get("spatial_compression", 32)

        # 1. Tiled encode
        lq_latent = tiled_encode(
            self.autoencoder.encoder,
            lq,
            tile_size=encoder_tile_size,
            feather=encoder_feather,
            spatial_compression=comp,
        )

        # 2. Mapper + DiffLoss (operate on full latent — naturally scalable)
        res_mapper = self.mapper(lq_latent) if self.mapper is not None \
            else torch.zeros_like(lq_latent)

        res_diffusion = torch.zeros_like(lq_latent)
        if self.diffloss is not None:
            self._ensure_diffloss_on_device()
            B, C, H, W = lq_latent.shape
            lq_flat = lq_latent.permute(0, 2, 3, 1).reshape(-1, C)
            res_diff_flat = self.diffloss.sample(z=lq_flat)
            res_diffusion = res_diff_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
            res_diffusion = res_diffusion.to(dtype=lq_latent.dtype, device=lq_latent.device)

        gen_latent = lq_latent + res_mapper + res_diffusion

        # 3. Tiled decode
        gen_image = tiled_decode(
            self.autoencoder.decoder,
            gen_latent,
            tile_size=decoder_tile_size,
            feather=decoder_feather,
            spatial_compression=comp,
        ).clamp(-1, 1)

        return gen_image

    # ── Forward (full image, no tiling) ────────────────────────────────────

    def forward(self, lq: torch.Tensor, target: Optional[torch.Tensor] = None):
        """
        Full forward pass.

        With mapper enabled:
          - Mapper predicts the deterministic residual: mapper(lq_latent).
          - DiffLoss learns the prediction error:
              error = (target_latent - lq_latent) - mapper(lq_latent).
          - Inference: res_hat = mapper(lq_latent) + diffloss.sample(z=lq_latent).

        Without mapper (backward compatible):
          - DiffLoss predicts the full residual: target_latent - lq_latent.

        All supervision aligns to the *teacher output* (target), not GT:
        DiffLoss and mapper both learn in latent space,
        pixel losses compare decoder output against target in pixel space.

        Training pipeline (target provided — should be teacher output):
          lq_latent           = encoder(lq)
          target_latent       = encoder(target)
          residual            = target_latent - lq_latent
          res_mapper          = mapper(lq_latent)               ← deterministic
          diff_target         = residual - res_mapper           ← what DiffLoss models
          loss_diff           = diffloss(diff_target, z=lq_latent)
          res_hat             = res_mapper + diffloss.sample(z=lq_latent)
          gen_latent          = lq_latent + res_hat
          gen_image           = decoder(gen_latent)

        Inference pipeline (target=None):
          lq_latent           = encoder(lq)
          res_hat             = mapper(lq_latent) + diffloss.sample(z=lq_latent)
          gen_latent          = lq_latent + res_hat
          gen_image           = decoder(gen_latent)

        Args:
            lq: Low-quality image  (B, 3, H, W), value range [-1, 1]
            target: Teacher output image (B, 3, H, W), value range [-1, 1].
                    All supervision (DiffLoss + pixel) aligns to this.
                    If None, runs inference-only (no DiffLoss training loss).

        Returns:
            A dict with keys:
              - "gen":       refined SR image   (B, 3, H, W)  [-1, 1]
              - "latent":    gen_latent         (B, C, h, w)   lq_latent + res_hat
              - "res_hat":   sampled residual   (B, C, h, w)   mapper + DiffLoss output
              - "loss_diff": DiffLoss training loss (scalar) — only when target
                             given and diffloss enabled; else None
        """
        # ---- Encode LQ latent (frozen) ----
        with torch.no_grad():
            lq_latent = self.autoencoder.encode(lq.to(self.dtype))

        # ---- Mapper: deterministic residual prediction ----
        res_mapper = torch.zeros_like(lq_latent)
        if self.mapper is not None:
            res_mapper = self.mapper(lq_latent)

        # ---- DiffLoss: predict full residual or residual error ----
        loss_diff = None
        res_diffusion = torch.zeros_like(lq_latent)

        if self.diffloss is not None:
            self._ensure_diffloss_on_device()
            B, C, H, W = lq_latent.shape
            lq_flat = lq_latent.permute(0, 2, 3, 1).reshape(-1, C)

            if target is not None:
                # target = teacher output pixels — align everything to teacher
                with torch.no_grad():
                    target_latent = self.autoencoder.encode(target.to(self.dtype))
                residual = target_latent - lq_latent

                # DiffLoss target = full residual (no mapper) or prediction error (with mapper)
                diff_target = residual - res_mapper.detach()
                diff_target_flat = diff_target.permute(0, 2, 3, 1).reshape(-1, C)

                loss_diff = self.diffloss(
                    target=diff_target_flat,
                    z=lq_flat.detach(),
                )

            res_diff_flat = self.diffloss.sample(z=lq_flat)
            res_diffusion = res_diff_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
            res_diffusion = res_diffusion.to(dtype=lq_latent.dtype, device=lq_latent.device)

        # ---- Combined residual: mapper (deterministic) + DiffLoss (stochastic) ----
        res_hat = res_mapper + res_diffusion

        # ---- LQ latent + sampled residual → decode directly ----
        gen_latent = lq_latent + res_hat

        # ---- Decode to pixel space ----
        gen_image = self.autoencoder.decode(gen_latent).clamp(-1, 1)

        return {
            "gen": gen_image,
            "latent": gen_latent,
            "res_hat": res_hat,
            "loss_diff": loss_diff,
        }

    def forward_latent(self, lq: torch.Tensor) -> torch.Tensor:
        """Latent-only forward (no decode).  Memory-efficient for loss computation."""
        with torch.no_grad():
            lq_latent = self.autoencoder.encode(lq.to(self.dtype))

        # Mapper: deterministic residual
        res_mapper = self.mapper(lq_latent) if self.mapper is not None \
            else torch.zeros_like(lq_latent)

        # DiffLoss: stochastic component
        if self.diffloss is not None:
            self._ensure_diffloss_on_device()
            B, C, H, W = lq_latent.shape
            lq_flat = lq_latent.permute(0, 2, 3, 1).reshape(-1, C)
            res_diff_flat = self.diffloss.sample(z=lq_flat)
            res_diffusion = res_diff_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
            res_diffusion = res_diffusion.to(dtype=lq_latent.dtype, device=lq_latent.device)
        else:
            res_diffusion = torch.zeros_like(lq_latent)

        return lq_latent + res_mapper + res_diffusion

    @property
    def dtype(self):
        return next(self.autoencoder.parameters()).dtype

    @property
    def device(self):
        return next(self.autoencoder.parameters()).device

    def _ensure_diffloss_on_device(self):
        """Move diffloss to the same device & dtype as the autoencoder."""
        if self.diffloss is not None:
            ae_dev = self.device
            ae_dtype = self.dtype
            dl_param = next(self.diffloss.parameters())
            if dl_param.device != ae_dev or dl_param.dtype != ae_dtype:
                self.diffloss.to(device=ae_dev, dtype=ae_dtype)

    # ── Checkpoint I/O ─────────────────────────────────────────────────────

    def save_checkpoint(self, path: str):
        """Save DiffLoss, mapper, and autoencoder weights and config."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        ckpt = {
            'dcae_preset': self._model_cfg.get("dcae_preset", "dc-ae-f32c32-in-1.0"),
            'dcae_pretrained_path': self._model_cfg.get("dcae_pretrained_path", ""),
            'in_channels': self._model_cfg.get("in_channels", 32),
            'spatial_compression': self._model_cfg.get("spatial_compression", 32),
        }
        if self.diffloss is not None:
            ckpt['diffloss_state_dict'] = self.diffloss.state_dict()
            ckpt['diffloss_cfg'] = self._model_cfg.get("diffloss_cfg", None)
        if self.mapper is not None:
            ckpt['mapper_state_dict'] = self.mapper.state_dict()
            ckpt['mapper_cfg'] = self._model_cfg.get("mapper_cfg", None)
        # Save full autoencoder weights (needed for pruned DCAE which has no pretrained weights)
        ckpt['autoencoder_state_dict'] = self.autoencoder.state_dict()
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location='cpu')
        if self.diffloss is not None and 'diffloss_state_dict' in ckpt:
            self.diffloss.load_state_dict(ckpt['diffloss_state_dict'])
        if self.mapper is not None and 'mapper_state_dict' in ckpt:
            self.mapper.load_state_dict(ckpt['mapper_state_dict'])
        # Load autoencoder weights if saved (needed for pruned DCAE)
        if 'autoencoder_state_dict' in ckpt:
            self.autoencoder.load_state_dict(ckpt['autoencoder_state_dict'])
        if not (('diffloss_state_dict' in ckpt) or ('mapper_state_dict' in ckpt)
                or ('autoencoder_state_dict' in ckpt)):
            raise RuntimeError(
                "No diffloss_state_dict, mapper_state_dict, or autoencoder_state_dict "
                "found in checkpoint — nothing to load."
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  Part 3: Inference wrapper
# ═══════════════════════════════════════════════════════════════════════════════


class DCAESR_test(DCAESR):
    """
    Inference wrapper.  Forward takes an image tensor, returns an image tensor.
    Compatible with ``torch.no_grad()``.
    """

    def __init__(self, args):
        """
        Args:
            args: Namespace with:
                  - distilled_ckpt_path or osediff_path: path to .pth checkpoint
                  - in_channels, diffloss_cfg (auto-read from checkpoint if saved)
                  - vae_encoder_tiled_size (int, optional): encoder tile size for tiled mode
                  - vae_decoder_tiled_size (int, optional): decoder tile size for tiled mode
        """
        ckpt_path = getattr(args, 'distilled_ckpt_path',
                            getattr(args, 'osediff_path', None))
        if ckpt_path is not None and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            model_cfg = {
                "dcae_preset": ckpt.get("dcae_preset", "dc-ae-f32c32-in-1.0"),
                "dcae_pretrained_path": ckpt.get("dcae_pretrained_path", ""),
                "in_channels": ckpt.get("in_channels", 32),
                "spatial_compression": ckpt.get("spatial_compression", 32),
                "diffloss_cfg": ckpt.get("diffloss_cfg", None),
                "mapper_cfg": ckpt.get("mapper_cfg", None),
            }
            self._diffloss_sd = ckpt.get('diffloss_state_dict', None)
            self._mapper_sd = ckpt.get('mapper_state_dict', None)
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")


        # Store tiled-inference config from args
        self._encoder_tile_size = getattr(args, 'vae_encoder_tiled_size', 1024)
        self._decoder_tile_size = getattr(args, 'vae_decoder_tiled_size', 256)
        self._tile_threshold = getattr(args, 'encoder_tile_threshold', 1024)
        self._decoder_threshold = getattr(args, 'decoder_tile_threshold', 256)

        super().__init__(model_cfg)

        if self._diffloss_sd is not None and self.diffloss is not None:
            self.diffloss.load_state_dict(self._diffloss_sd)
            del self._diffloss_sd
        if getattr(self, '_mapper_sd', None) is not None and self.mapper is not None:
            self.mapper.load_state_dict(self._mapper_sd)
            del self._mapper_sd

        # Move to GPU
        weight_dtype = torch.float32
        mp = getattr(args, 'mixed_precision', 'fp16')
        if mp == "fp16":
            weight_dtype = torch.float16

        self.autoencoder.to("cuda", dtype=weight_dtype)
        if self.diffloss is not None:
            ae_device = next(self.autoencoder.parameters()).device
            ae_dtype = next(self.autoencoder.parameters()).dtype
            self.diffloss.to(device=ae_device, dtype=ae_dtype)
        if self.mapper is not None:
            self.mapper.to("cuda", dtype=weight_dtype)

    @torch.no_grad()
    def forward(self, lq: torch.Tensor, prompt: Optional[str] = None) -> torch.Tensor:
        """
        Inference.  ``prompt`` is accepted for signature compatibility with
        OSEDiff_test but is ignored (DCAE does not use text conditioning).

        Automatically falls back to tiled encoding/decoding when the input
        exceeds the tile-size thresholds.
        """
        comp = self._model_cfg.get("spatial_compression", 32)
        H, W = lq.shape[2], lq.shape[3]
        h_lat = H // comp
        w_lat = W // comp

        if H > self._encoder_tile_size or W > self._encoder_tile_size or \
           h_lat > self._decoder_tile_size or w_lat > self._decoder_tile_size:
            return self.forward_tiled(
                lq,
                encoder_tile_size=self._encoder_tile_size,
                decoder_tile_size=self._decoder_tile_size,
            )

        return super().forward(lq)["gen"]


# ── Compatibility aliases (used by test_osediff.py & test_inference_time.py) ──

# DCAESR_test replaces both old names:
DistilledOSEDiff_test = DCAESR_test
DistilledOSEDiff_inference_time = DCAESR_test
OSEDiff_inference_time_ref = DCAESR_test
