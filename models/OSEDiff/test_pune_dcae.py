"""
Pruned DCAE Decoder Test — Reconstruction Quality Evaluation
=============================================================

Tests a Phase-1-pretrained pruned decoder by running:
  GT image → full encoder (frozen) → pruned decoder (trained) → reconstructed image

Then computes PSNR, SSIM, LPIPS vs the original GT.

Also compares with the full DCAE reconstruction as an upper bound,
and saves side-by-side visualizations.

Usage:
    # Test single decoder checkpoint against a dataset txt file
    python test_decoder.py \
        --full_dcae_model preset/models/mit-han-labdc-ae-f32c32-in-1.0.safetensors \
        --pruned_preset dc-ae-f32c32-pruned \
        --pune_dcae_ckpt experience/decoder_pretrain/checkpoints/decoder_final.pth \
        --test_txt YOUR_TEST_DATASET.txt \
        --output_dir experience/decoder_test

    # Test on a directory of images
    python test_decoder.py \
        --full_dcae_model preset/models/mit-han-labdc-ae-f32c32-in-1.0.safetensors \
        --pruned_preset dc-ae-f32c32-pruned \
        --pune_dcae_ckpt experience/decoder_pretrain/checkpoints/decoder_final.pth \
        --test_image_dir YOUR_IMAGE_DIR \
        --output_dir experience/decoder_test

    # Test on a single image
    python test_decoder.py \
        --full_dcae_model preset/models/mit-han-labdc-ae-f32c32-in-1.0.safetensors \
        --pruned_preset dc-ae-f32c32-pruned \
        --pune_dcae_ckpt experience/decoder_pretrain/checkpoints/decoder_final.pth \
        --test_image path/to/image.png \
        --output_dir experience/decoder_test
"""

import os
import sys
import glob
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm

import pyiqa
from basicsr.utils import img2tensor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from efficient_vit.dcae_zoo import dc_ae_zoo, dc_ae_from_pretrained
from efficient_vit.efficientvit.dc_ae import DCAE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pruned DCAE decoder reconstruction quality test"
    )

    # ---- Models ----
    parser.add_argument("--full_dcae_model", type=str, required=True,
                        help="Path to full pretrained DCAE checkpoint (.safetensors, "
                             "e.g. preset/models/mit-han-labdc-ae-f32c32-in-1.0.safetensors).")
    parser.add_argument("--pruned_preset", type=str, default=None,
                        help="Pruned DCAE preset name. Auto-detected from pune_dcae_ckpt "
                             "if not specified. Manually override only if the ckpt lacks "
                             "this field.")
    parser.add_argument("--pune_dcae_ckpt", type=str, required=True,
                        help="Path to Phase-1-pretrained decoder checkpoint (.pth).")

    # ---- Test data (at least one required) ----
    parser.add_argument("--test_image", type=str, default=None,
                        help="Single test image path.")
    parser.add_argument("--test_image_dir", type=str, default=None,
                        help="Directory of test images (*.png, *.jpg, *.jpeg).")
    parser.add_argument("--test_txt", type=str, default=None,
                        help="Dataset txt file listing image paths (one per line).")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Limit number of test images (useful for quick checks).")

    # ---- Output ----
    parser.add_argument("--output_dir", type=str, default="experience/decoder_test")
    parser.add_argument("--save_images", action="store_true", default=True,
                        help="Save reconstruction comparison images (default: True).")
    parser.add_argument("--no_save_images", action="store_false", dest="save_images",
                        help="Skip saving images, only print metrics.")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Resize test images to this resolution before encoding "
                             "(None = use original size).")

    # ---- Metrics ----
    parser.add_argument("--metrics", type=str, nargs="+",
                        default=["psnr", "ssim", "lpips"],
                        choices=["psnr", "ssim", "lpips", "dists"],
                        help="Metrics to compute.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on.")

    args = parser.parse_args()

    if args.test_image is None and args.test_image_dir is None and args.test_txt is None:
        parser.error("At least one of --test_image, --test_image_dir, --test_txt "
                     "must be provided.")

    return args


# ═══════════════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════════════


def load_image(path: str) -> Image.Image:
    """Load an image as RGB PIL Image."""
    img = Image.open(path).convert("RGB")
    return img


def collect_image_paths(args) -> list:
    """Gather all test image paths from the provided sources."""
    paths = []

    if args.test_image:
        if os.path.isfile(args.test_image):
            paths.append(args.test_image)
        else:
            print(f"[WARN] --test_image path not found: {args.test_image}")

    if args.test_image_dir:
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]:
            paths.extend(sorted(glob.glob(os.path.join(args.test_image_dir, ext))))

    if args.test_txt:
        if os.path.exists(args.test_txt):
            with open(args.test_txt, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and os.path.exists(line):
                        paths.append(line)
        else:
            print(f"[WARN] --test_txt file not found: {args.test_txt}")

    if not paths:
        raise RuntimeError("No valid test images found.")

    if args.max_images:
        paths = paths[:args.max_images]

    return paths


def to_tensor(pil_img: Image.Image, resolution: int = None) -> torch.Tensor:
    """PIL → (1, 3, H, W) tensor in [-1, 1]."""
    if resolution is not None:
        pil_img = pil_img.resize((resolution, resolution), Image.LANCZOS)
    # Ensure dimensions are multiples of 8 (DCAE requirement)
    w, h = pil_img.size
    new_w = w - w % 8
    new_h = h - h % 8
    if new_w != w or new_h != h:
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
    t = transforms.ToTensor()(pil_img)  # [0, 1]
    t = t * 2 - 1                       # [-1, 1]
    return t.unsqueeze(0)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(1, 3, H, W) tensor in [-1, 1] → PIL Image."""
    t = t.squeeze(0).clamp(-1, 1).cpu()
    t = (t + 1) / 2  # → [0, 1]
    return transforms.ToPILImage()(t)


# ═══════════════════════════════════════════════════════════════════════════════
#  Model building
# ═══════════════════════════════════════════════════════════════════════════════


def build_models(args):
    """Build full-DCAE (reference) and pruned-DCAE + Phase-1 decoder."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Load checkpoint to get config ----
    ckpt = torch.load(args.pune_dcae_ckpt, map_location=device)
    ckpt_preset = ckpt.get("pruned_preset", None)

    if args.pruned_preset is not None:
        pruned_preset = args.pruned_preset
    elif ckpt_preset is not None:
        pruned_preset = ckpt_preset
        print(f"  Auto-detected pruned_preset from checkpoint: {pruned_preset}")
    else:
        raise RuntimeError(
            "pruned_preset not found in checkpoint and not provided via "
            "--pruned_preset. Please specify it manually."
        )

    # ---- Full DCAE (upper-bound reference) ----
    if "f32c32" in pruned_preset:
        full_name = "dc-ae-f32c32-in-1.0"
    else:
        full_name = "dc-ae-f64c128-in-1.0"
    full_dcae = dc_ae_from_pretrained(full_name, args.full_dcae_model)
    full_dcae = full_dcae.to(device)
    full_dcae.eval()
    full_dcae.requires_grad_(False)

    print(f"  Full DCAE loaded: {sum(p.numel() for p in full_dcae.parameters()) / 1e6:.1f}M params")

    # ---- Pruned DCAE (test subject) ----
    pruned_cfg = dc_ae_zoo(pruned_preset)
    pruned_dcae = DCAE(pruned_cfg)
    pruned_dcae = pruned_dcae.to(device)
    pruned_dcae.eval()
    pruned_dcae.requires_grad_(False)

    # Load Phase-1 decoder weights
    ckpt = torch.load(args.pune_dcae_ckpt, map_location=device)
    loaded = False
    if "decoder_state_dict" in ckpt:
        pruned_dcae.decoder.load_state_dict(ckpt["decoder_state_dict"])
        loaded = True
        step_info = ckpt.get("step", "unknown")
        loss_info = ckpt.get("best_loss", None)
    elif "autoencoder_state_dict" in ckpt:
        decoder_sd = {k.replace("decoder.", ""): v
                      for k, v in ckpt["autoencoder_state_dict"].items()
                      if k.startswith("decoder.")}
        if decoder_sd:
            pruned_dcae.decoder.load_state_dict(decoder_sd, strict=False)
            loaded = True
            step_info = "unknown"
            loss_info = None

    if not loaded:
        raise RuntimeError(f"Could not find decoder weights in {args.pune_dcae_ckpt}. "
                           f"Keys: {list(ckpt.keys())}")

    n_pruned = sum(p.numel() for p in pruned_dcae.parameters())
    n_dec = sum(p.numel() for p in pruned_dcae.decoder.parameters())
    print(f"  Pruned DCAE loaded: {n_pruned / 1e6:.1f}M total, "
          f"{n_dec / 1e6:.1f}M decoder")
    print(f"  Decoder checkpoint step: {step_info}"
          + (f", best_loss: {loss_info:.6f}" if loss_info is not None else ""))

    full_dtype = next(full_dcae.parameters()).dtype
    pruned_dtype = next(pruned_dcae.decoder.parameters()).dtype

    return full_dcae, pruned_dcae, device, full_dtype, pruned_dtype


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    args = parse_args()

    # ---- Init ----
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Pruned preset: {args.pruned_preset}")
    print(f"Decoder checkpoint: {args.pune_dcae_ckpt}")

    # ---- Output dirs ----
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_images:
        full_img_dir = os.path.join(args.output_dir, "full_recon")
        pruned_img_dir = os.path.join(args.output_dir, "pruned_recon")
        gt_img_dir = os.path.join(args.output_dir, "gt")
        compare_dir = os.path.join(args.output_dir, "comparison")
        os.makedirs(full_img_dir, exist_ok=True)
        os.makedirs(pruned_img_dir, exist_ok=True)
        os.makedirs(gt_img_dir, exist_ok=True)
        os.makedirs(compare_dir, exist_ok=True)

    # ---- Build models ----
    print("\nBuilding models...")
    full_dcae, pruned_dcae, device, full_dtype, pruned_dtype = build_models(args)

    # ---- Init metrics ----
    print("\nInitializing metrics...")
    iqa_metrics = {}
    if "psnr" in args.metrics:
        iqa_metrics["PSNR"] = pyiqa.create_metric(
            "psnr", test_y_channel=True, color_space="ycbcr"
        ).to(device)
    if "ssim" in args.metrics:
        iqa_metrics["SSIM"] = pyiqa.create_metric(
            "ssim", test_y_channel=True, color_space="ycbcr"
        ).to(device)
    if "lpips" in args.metrics:
        iqa_metrics["LPIPS"] = pyiqa.create_metric("lpips", device=device)
    if "dists" in args.metrics:
        iqa_metrics["DISTS"] = pyiqa.create_metric("dists", device=device)
    print(f"  Metrics: {list(iqa_metrics.keys())}")

    # ---- Collect images ----
    print("\nCollecting test images...")
    image_paths = collect_image_paths(args)
    print(f"  Found {len(image_paths)} images")

    # ---- Run evaluation ----
    print("\n===== Running reconstruction evaluation =====\n")

    accum_full = {k: 0.0 for k in iqa_metrics}
    accum_pruned = {k: 0.0 for k in iqa_metrics}
    count = 0

    for i, img_path in enumerate(tqdm(image_paths, desc="Testing")):
        try:
            pil_img = load_image(img_path)
        except Exception as e:
            print(f"  [SKIP] {img_path}: {e}")
            continue

        img_name = Path(img_path).stem
        gt_tensor = to_tensor(pil_img, args.resolution).to(device)
        B, C, H, W = gt_tensor.shape

        # ---- Full DCAE: reconstruction (upper bound) ----
        with torch.no_grad():
            full_latent = full_dcae.encode(gt_tensor.to(dtype=full_dtype))
            full_recon = full_dcae.decode(full_latent).clamp(-1, 1)

        # ---- Pruned DCAE: reconstruction -----
        with torch.no_grad():
            # Encoder: use FULL DCAE encoder (same as Phase 1 training)
            pruned_latent = full_dcae.encode(gt_tensor.to(dtype=full_dtype))
            # Decoder: use PRUNED decoder
            pruned_recon = pruned_dcae.decoder(
                pruned_latent.to(dtype=pruned_dtype)
            ).clamp(-1, 1)

        # ---- Compute metrics ----
        # pyiqa expects tensors in [0, 1]
        gt_01 = (gt_tensor + 1) / 2
        full_01 = (full_recon + 1) / 2
        pruned_01 = (pruned_recon + 1) / 2

        for name, metric in iqa_metrics.items():
            if name in ["LPIPS", "DISTS"]:
                accum_full[name] += metric(full_01.float(), gt_01.float()).item()
                accum_pruned[name] += metric(pruned_01.float(), gt_01.float()).item()
            else:
                accum_full[name] += metric(full_01, gt_01).item()
                accum_pruned[name] += metric(pruned_01, gt_01).item()

        count += 1

        # ---- Save images ----
        if args.save_images:
            gt_pil = tensor_to_pil(gt_tensor)
            full_pil = tensor_to_pil(full_recon)
            pruned_pil = tensor_to_pil(pruned_recon)

            gt_pil.save(os.path.join(gt_img_dir, f"{img_name}.png"))
            full_pil.save(os.path.join(full_img_dir, f"{img_name}.png"))
            pruned_pil.save(os.path.join(pruned_img_dir, f"{img_name}.png"))

            # Side-by-side comparison: GT | Full | Pruned
            gt_np = np.array(gt_pil)
            full_np = np.array(full_pil)
            pruned_np = np.array(pruned_pil)
            comparison = np.hstack([gt_np, full_np, pruned_np])
            Image.fromarray(comparison).save(
                os.path.join(compare_dir, f"{img_name}_compare.png")
            )

    # ---- Print results ----
    print(f"\n===== Results ({count} images) =====\n")
    print(f"{'Metric':<8} {'Full DCAE':<16} {'Pruned DCAE':<16} {'Delta':<12} {'Degradation':<12}")
    print("-" * 64)

    for name in iqa_metrics:
        avg_full = accum_full[name] / count
        avg_pruned = accum_pruned[name] / count
        delta = avg_pruned - avg_full
        # For PSNR/SSIM: higher is better, degradation = negative delta
        # For LPIPS/DISTS: lower is better, degradation = positive delta
        if name in ["LPIPS", "DISTS"]:
            degradation_pct = (delta / max(avg_full, 1e-8)) * 100
        else:
            degradation_pct = (-delta / max(avg_full, 1e-8)) * 100
        print(f"{name:<8} {avg_full:<16.4f} {avg_pruned:<16.4f} "
              f"{delta:+.4f}      {degradation_pct:+.1f}%")

    print("-" * 64)
    print(f"\nImages saved to: {args.output_dir}")
    if args.save_images:
        print(f"  Comparison grids: {compare_dir}/")
        print(f"  Full DCAE recon:  {full_img_dir}/")
        print(f"  Pruned DCAE recon: {pruned_img_dir}/")
        print(f"  Ground truth:     {gt_img_dir}/")

    # ---- Save metrics to txt ----
    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Decoder checkpoint: {args.pune_dcae_ckpt}\n")
        f.write(f"Pruned preset: {args.pruned_preset}\n")
        f.write(f"Images tested: {count}\n\n")
        f.write(f"{'Metric':<8} {'Full DCAE':<16} {'Pruned DCAE':<16} {'Delta':<12} {'Degradation':<12}\n")
        f.write("-" * 64 + "\n")
        for name in iqa_metrics:
            avg_full = accum_full[name] / count
            avg_pruned = accum_pruned[name] / count
            delta = avg_pruned - avg_full
            if name in ["LPIPS", "DISTS"]:
                degradation_pct = (delta / max(avg_full, 1e-8)) * 100
            else:
                degradation_pct = (-delta / max(avg_full, 1e-8)) * 100
            f.write(f"{name:<8} {avg_full:<16.4f} {avg_pruned:<16.4f} "
                    f"{delta:+.4f}      {degradation_pct:+.1f}%\n")
    print(f"\nMetrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
