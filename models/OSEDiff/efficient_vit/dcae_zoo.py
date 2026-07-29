from typing import Optional

from omegaconf import OmegaConf

from efficient_vit.efficientvit.dc_ae import DCAEConfig, DCAE


def dc_ae_zoo(name: str, pretrained_path: Optional[str] = None) -> DCAEConfig:
    """
    Return a DCAEConfig for a named preset.

    Args:
        name: One of "dc-ae-f32c32-in-1.0", "dc-ae-f16c32-in-1.0",
              "dc-ae-f64c128-in-1.0".
        pretrained_path: Optional path to a .pth checkpoint.  When set,
                         DCAE will load weights at construction time.

    Returns:
        DCAEConfig — can be passed directly to ``DCAE(cfg)``.
    """
    # ── f32c32 (32× compression, latent 32ch) ────────────────────────
    if name in ["dc-ae-f32c32-in-1.0", "dc-ae-f32c32-mix-1.0"]:
        cfg_str = (
            "latent_channels=32 "
            "encoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU] "
            "encoder.width_list=[128,256,512,512,1024,1024] "
            "encoder.depth_list=[0,4,8,2,2,2] "
            "decoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU] "
            "decoder.width_list=[128,256,512,512,1024,1024] "
            "decoder.depth_list=[0,5,10,2,2,2] "
            "decoder.norm=[bn2d,bn2d,bn2d,trms2d,trms2d,trms2d] "
            "decoder.act=[relu,relu,relu,silu,silu,silu]"
        )

    # ── f16c32 (16× compression, latent 32ch) ────────────────────────
    elif name in ["dc-ae-f16c32-in-1.0"]:
        cfg_str = (
            "latent_channels=32 "
            "encoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU] "
            "encoder.width_list=[128,256,512,512,1024] "
            "encoder.depth_list=[0,4,8,2,2] "
            "decoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU] "
            "decoder.width_list=[128,256,512,512,1024] "
            "decoder.depth_list=[0,5,10,2,2] "
            "decoder.norm=[bn2d,bn2d,bn2d,trms2d,trms2d] "
            "decoder.act=[relu,relu,relu,silu,silu]"
        )

    # ── f64c128 (64× compression, latent 128ch) ──────────────────────
    elif name in ["dc-ae-f64c128-in-1.0", "dc-ae-f64c128-mix-1.0"]:
        cfg_str = (
            "latent_channels=128 "
            "encoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU,EViT_GLU] "
            "encoder.width_list=[128,256,512,512,1024,1024,2048] "
            "encoder.depth_list=[0,4,8,2,2,2,2] "
            "decoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU,EViT_GLU] "
            "decoder.width_list=[128,256,512,512,1024,1024,2048] "
            "decoder.depth_list=[0,5,10,2,2,2,2] "
            "decoder.norm=[bn2d,bn2d,bn2d,trms2d,trms2d,trms2d,trms2d] "
            "decoder.act=[relu,relu,relu,silu,silu,silu,silu]"
        )

    # ── pruned-f32c32 (32× compression, latent 32ch, channel-halved) ──
    # Encoder keeps 75% channels; decoder keeps 50% channels (AdcSR strategy).
    elif name in ["dc-ae-f32c32-pruned"]:
        cfg_str = (
            "latent_channels=32 "
            "encoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU] "
            "encoder.width_list=[64,128,256,256,512,512] "
            "encoder.depth_list=[0,4,8,2,2,2] "
            "decoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU] "
            "decoder.width_list=[64,128,256,256,512,512] "
            "decoder.depth_list=[0,5,10,2,2,2] "
            "decoder.norm=[bn2d,bn2d,bn2d,trms2d,trms2d,trms2d] "
            "decoder.act=[relu,relu,relu,silu,silu,silu]"
        )

    # ── pruned-f64c128 (64× compression, latent 128ch, channel-halved) ──
    elif name in ["dc-ae-f64c128-pruned"]:
        cfg_str = (
            "latent_channels=128 "
            "encoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU,EViT_GLU] "
            "encoder.width_list=[96,192,384,384,768,768,1536] "
            "encoder.depth_list=[0,4,8,2,2,2,2] "
            "decoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU,EViT_GLU] "
            "decoder.width_list=[64,128,256,256,512,512,1024] "
            "decoder.depth_list=[0,5,10,2,2,2,2] "
            "decoder.norm=[bn2d,bn2d,bn2d,trms2d,trms2d,trms2d,trms2d] "
            "decoder.act=[relu,relu,relu,silu,silu,silu,silu]"
        )

    else:
        raise KeyError(
            f"Unknown DCAE preset '{name}'. "
            f"Available: dc-ae-f32c32-in-1.0, dc-ae-f16c32-in-1.0, "
            f"dc-ae-f64c128-in-1.0, dc-ae-f32c32-pruned, dc-ae-f64c128-pruned"
        )

    cfg = OmegaConf.from_dotlist(cfg_str.split(" "))
    cfg: DCAEConfig = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(DCAEConfig), cfg)
    )
    cfg.pretrained_path = pretrained_path
    return cfg


def dc_ae_from_pretrained(name: str, pretrained_path: str) -> DCAE:
    """
    Convenience: build a DCAE model from a preset name + checkpoint path.

    Args:
        name: Preset name (e.g. "dc-ae-f32c32-in-1.0").
        pretrained_path: Path to the .pth checkpoint.

    Returns:
        DCAE model (with weights loaded) in eval mode.
    """
    cfg = dc_ae_zoo(name, pretrained_path=pretrained_path)
    model = DCAE(cfg)
    model.eval()
    return model
