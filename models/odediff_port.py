import torch

from .OSEDiff.osediff import OSEDiff_test_for_digitalFilm


class OSEDIFF_port(OSEDiff_test_for_digitalFilm):
    def __init__(self, pretrained_model_name_or_path: str, osediff_path: str, vae_encoder_tiled_size: int = 1024, vae_decoder_tiled_size: int = 224,
                 mixed_precision: str = "fp16", device: str = "cpu", merge_and_unload_lora: bool = True):
        super().__init__(pretrained_model_name_or_path,
                         osediff_path,
                         vae_encoder_tiled_size,
                         vae_decoder_tiled_size,
                         mixed_precision,
                         device,
                         merge_and_unload_lora)

    def process(self, x: torch.Tensor, enable: bool):
        if not enable:
            return x

        # SD VAE is trained on [-1, 1] images; our pipeline feeds [0, 1].
        dev = x.device
        dt = x.dtype
        x = x * 2.0 - 1.0
        x = x.to(device=self.unet.device, dtype=self.weight_dtype)
        out = super().forward(x, "增强画质")
        out = out * 0.5 + 0.5
        out = out.to(device=dev, dtype=dt)
        return out.clamp(0, 1)
