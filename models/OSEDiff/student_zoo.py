"""
Student model zoo — pre-configured DiffLoss + optional Mapper architectures.

All dcae-* presets use DiffLoss (optionally with CNN mapper).
When mapper_cfg is present: mapper predicts deterministic residual,
DiffLoss predicts the prediction error; otherwise DiffLoss predicts full residual.
Name pattern: dcae_{spatial_compression}x{latent_channels}.

Usage:
    from student_zoo import get_model_cfg
    cfg = get_model_cfg("dcae_f32c32")
"""

from typing import Dict, List

# ─── Registry ────────────────────────────────────────────────────────────────
# Shared fields:
#   in_channels           : int          — latent channel count
#   spatial_compression   : int          — DCAE compression ratio
#   dcae_preset           : str          — DCAE pretrained model name
#   dcae_pretrained_path  : str | None   — path to DCAE .pth checkpoint
#   description           : str
#   diffloss_cfg          : dict | None  — DiffLoss config (None disables DiffLoss)
#     - depth              : int          — MLP depth (default 4)
#     - width              : int          — MLP hidden width (default 512)
#     - num_sampling_steps : list[int]    — sampling steps (default [16])
#     - sampler            : str          — "iddpm" or "ddim" (default "iddpm")
#     - vae_scale          : float        — latent scale (default 0.6)
#   mapper_cfg            : dict | None  — LightMapper config (None = no mapper)
#     - hidden             : int          — internal channel width (default 64)
#     - num_blocks         : int          — number of ResBlock2d (default 4)
#     - kernel_size        : int          — conv kernel size (default 3)

MODEL_CFG: Dict[str, dict] = {
    # ── DCAE-f32c32 (full) ──────────────────────────────────────────
    "dcae_f32c32": {
        "in_channels": 32,
        "spatial_compression": 32,
        "dcae_preset": "dc-ae-f32c32-in-1.0",
        "dcae_pretrained_path": "preset/models/mit-han-labdc-ae-f32c32-in-1.0.safetensors",
        "description": "DCAE-f32c32  (32× comp, latent 32ch, full DCAE)",
        "diffloss_cfg": {
            "depth": 4,
            "width": 512,
            "num_sampling_steps": [16],
            "sampler": "iddpm",
            "vae_scale": 0.6,
        },
        "mapper_cfg": {
            "hidden": 64,
            "num_blocks": 4,
            "kernel_size": 3,
        },
    },

    # ── DCAE-f32c32-pruned (channel-halved DCAE) ─────────────────────
    # Encoder: 75% channels; Decoder: 50% channels (AdcSR strategy).
    # No pretrained weights — needs decoder pretraining + distillation.
    "dcae_f32c32_pruned": {
        "in_channels": 32,
        "spatial_compression": 32,
        "dcae_preset": "dc-ae-f32c32-pruned",
        "dcae_pretrained_path": "",
        "description": "DCAE-f32c32-pruned  (32× comp, latent 32ch, pruned DCAE)",
        "diffloss_cfg": {
            "depth": 4,
            "width": 512,
            "num_sampling_steps": [16],
            "sampler": "iddpm",
            "vae_scale": 0.6,
        },
        "mapper_cfg": {
            "hidden": 64,
            "num_blocks": 4,
            "kernel_size": 3,
        },
    },

    # ── DCAE-f64c128 (full) ──────────────────────────────────────────
    "dcae_f64c128": {
        "in_channels": 128,
        "spatial_compression": 64,
        "dcae_preset": "dc-ae-f64c128-in-1.0",
        "dcae_pretrained_path": "preset/models/mit-han-labdc-ae-f64c128-in-1.0.safetensors",
        "description": "DCAE-f64c128  (64× comp, latent 128ch, full DCAE)",
        "diffloss_cfg": {
            "depth": 4,
            "width": 512,
            "num_sampling_steps": [16],
            "sampler": "iddpm",
            "vae_scale": 0.6,
        },
        "mapper_cfg": {
            "hidden": 128,
            "num_blocks": 4,
            "kernel_size": 3,
        },
    },

    # ── DCAE-f64c128-pruned (channel-halved DCAE) ────────────────────
    "dcae_f64c128_pruned": {
        "in_channels": 128,
        "spatial_compression": 64,
        "dcae_preset": "dc-ae-f64c128-pruned",
        "dcae_pretrained_path": "",
        "description": "DCAE-f64c128-pruned  (64× comp, latent 128ch, pruned DCAE)",
        "diffloss_cfg": {
            "depth": 4,
            "width": 512,
            "num_sampling_steps": [16],
            "sampler": "iddpm",
            "vae_scale": 0.6,
        },
        "mapper_cfg": {
            "hidden": 128,
            "num_blocks": 4,
            "kernel_size": 3,
        },
    },
}


def get_model_cfg(name: str) -> dict:
    """Return a copy of the config dict for ``name``."""
    name = name.lower()
    if name not in MODEL_CFG:
        available = ", ".join(MODEL_CFG.keys())
        raise KeyError(
            f"Unknown student model '{name}'. Available: {available}"
        )
    return dict(MODEL_CFG[name])


def list_models() -> List[str]:
    """Return sorted list of available model names."""
    return sorted(MODEL_CFG.keys())


def print_models():
    """Pretty-print the zoo."""
    print("Student model zoo (DiffLoss + optional CNN Mapper):")
    print("=" * 60)
    for name in sorted(MODEL_CFG.keys()):
        cfg = MODEL_CFG[name]
        has_mapper = "mapper_cfg" in cfg
        tag = " [mapper]" if has_mapper else " [no mapper]"
        print(f"  {name:30s}  {cfg['description']}{tag}")
    print("=" * 60)
