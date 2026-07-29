"""
Pretraining for Pruned DCAE (Phase 1)
=====================================

Phase 1 (Updated):
  - Use full pretrained DCAE encoder (frozen) as reference.
  - Train the pruned DCAE (Encoder + Decoder, or Decoder only if freeze_pruned_encoder is set).
  - Minimise L1 + L2 + optional LPIPS reconstruction loss.
  - Save full pruned DCAE checkpoints for Phase 2 distillation.

Usage:
    python train_decoder.py \
        --pretrained_dcae preset/models/mit-han-labdc-ae-f32c32-in-1.0.safetensors \
        --pruned_preset dc-ae-f32c32-pruned \
        --output_dir experience/pretrain_f32c32 \
        --dataset_txt_paths_list YOUR_DATASET.txt
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from tqdm.auto import tqdm
from torchvision import transforms

from diffusers.optimization import get_scheduler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from efficient_vit.dcae_zoo import dc_ae_zoo, dc_ae_from_pretrained
from efficient_vit.efficientvit.dc_ae import DCAE
from dataloaders.realsr_dataset import PairedSROnlineTxtDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Autoencoder Pretraining for Pruned DCAE (Phase 1)"
    )

    # ---- Model ----
    parser.add_argument("--pretrained_dcae", type=str, required=True,
                        help="Path to full pretrained DCAE checkpoint (.safetensors).")
    parser.add_argument("--pruned_preset", type=str,
                        default="dc-ae-f32c32-pruned",
                        choices=["dc-ae-f32c32-pruned", "dc-ae-f64c128-pruned"],
                        help="Pruned DCAE preset name.")
    parser.add_argument("--pretrained_pune_dcae_path", type=str, default=None,
                        help="Optional: path to a pretrained pruned DCAE checkpoint to resume from.")
    parser.add_argument("--freeze_pruned_encoder", action="store_true",
                        help="If set, freezes the pruned DCAE encoder and only trains its decoder.")
    parser.add_argument("--latent_channels", type=int, default=32,
                        help="Latent channel count (32 for f32c32, 128 for f64c128).")
    parser.add_argument("--spatial_compression", type=int, default=32,
                        help="Spatial compression ratio (32 for f32c32, 64 for f64c128).")

    # ---- Paths ----
    parser.add_argument("--output_dir", type=str, default="experience/decoder_pretrain")
    parser.add_argument("--logging_dir", type=str, default="logs")

    # ---- Dataset ----
    parser.add_argument("--dataset_txt_paths_list", type=str, nargs="+",
                        default=["YOUR_TXT_FILE_PATH"],
                        help="List of dataset txt file paths.")
    parser.add_argument("--dataset_prob_paths_list", type=int, nargs="+",
                        default=[1],
                        help="Sampling probabilities for each dataset")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Training resolution.")

    # ---- Training schedule ----
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_training_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=50000)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    # ---- Optimizer ----
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                        choices=["linear", "cosine", "cosine_with_restarts",
                                 "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)

    # ---- Loss weights ----
    parser.add_argument("--lambda_l1", type=float, default=1.0,
                        help="Weight for L1 reconstruction loss.")
    parser.add_argument("--lambda_l2", type=float, default=1.0,
                        help="Weight for L2 (MSE) reconstruction loss.")
    parser.add_argument("--lambda_lpips", type=float, default=0.5,
                        help="Weight for LPIPS perceptual loss.")

    # ---- Mixed precision ----
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")

    # ---- Logging ----
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--tracker_project_name", type=str,
                        default="decoder_pretrain")

    # ---- Validation ----
    parser.add_argument("--validation_steps", type=int, default=500,
                        help="Run validation every N steps.")
    parser.add_argument("--validation_image_path", type=str, default=None,
                        help="Path to a single image for validation visualisation.")

    args = parser.parse_args()

    if args.lambda_l1 <= 0 and args.lambda_l2 <= 0 and args.lambda_lpips <= 0:
        parser.error("At least one of --lambda_l1, --lambda_l2, --lambda_lpips must be > 0")

    return args


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def build_models(args):
    """
    Build:
      - full_dcae: frozen reference DCAE (full weights)
      - pruned_dcae: trainable DCAE (encoder + decoder)

    Returns (full_dcae, pruned_dcae).
    """
    # 1. 加载并冻结完整预训练 DCAE（Frozen Reference）
    full_dcae = dc_ae_from_pretrained(
        name="dc-ae-f32c32-in-1.0" if "f32c32" in args.pruned_preset else "dc-ae-f64c128-in-1.0",
        pretrained_path=args.pretrained_dcae,
    )
    full_dcae.eval()
    full_dcae.requires_grad_(False)

    # 2. 初始化剪枝后的 DCAE 结构
    pruned_cfg = dc_ae_zoo(args.pruned_preset)
    pruned_dcae = DCAE(pruned_cfg)
    pruned_dcae.train()
    pruned_dcae.requires_grad_(True)

    # 3. 将 full_dcae 能够匹配 Shape 的权重赋予 pruned_dcae
    full_sd = full_dcae.state_dict()
    model_sd = pruned_dcae.state_dict()

    matched_sd = {}
    mismatched_keys = []

    for k, v in full_sd.items():
        if k in model_sd:
            if v.shape == model_sd[k].shape:
                matched_sd[k] = v
            else:
                mismatched_keys.append((k, v.shape, model_sd[k].shape))

    pruned_dcae.load_state_dict(matched_sd, strict=False)
    print(f"[Model] Initialized pruned_dcae with full DCAE matched weights.")
    print(f"        -> Matched layers loaded: {len(matched_sd)}")
    print(f"        -> Skipped (size mismatch due to pruning): {len(mismatched_keys)}")

    # 4. 如果提供了特定 checkpoint，覆写 pruned_dcae 权重
    pruned_path = getattr(args, "pretrained_pune_dcae_path", None)
    if pruned_path and os.path.exists(pruned_path):
        ckpt = torch.load(pruned_path, map_location="cpu")
        sd_to_load = ckpt.get("autoencoder_state_dict", ckpt.get("state_dict", ckpt))
        
        # Shape 过滤保护
        matched_ckpt_sd = {
            k: v for k, v in sd_to_load.items()
            if k in model_sd and v.shape == model_sd[k].shape
        }
        pruned_dcae.load_state_dict(matched_ckpt_sd, strict=False)
        print(f"[Model] Resumed pruned_dcae from {pruned_path} ({len(matched_ckpt_sd)} layers matched)")

    # 5. 可选：冻结 pruned_dcae 的 Encoder
    if args.freeze_pruned_encoder:
        pruned_dcae.encoder.eval()
        pruned_dcae.encoder.requires_grad_(False)
        print("[Model] Pruned DCAE Encoder is FROZEN. Only Decoder will be trained.")
    else:
        print("[Model] Both Pruned DCAE Encoder and Decoder are TRAINABLE.")

    return full_dcae, pruned_dcae


def main(args):
    # ----- Accelerator setup -----
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    # ----- Build full DCAE + pruned DCAE -----
    if accelerator.is_main_process:
        print("[Phase 1] Building Full DCAE (frozen) & Pruned DCAE (trainable)...")

    full_dcae, pruned_dcae = build_models(args)

    trainable_params = [p for p in pruned_dcae.parameters() if p.requires_grad]

    if accelerator.is_main_process:
        print(f"  Full DCAE params:   {count_params(full_dcae) / 1e6:.2f}M (frozen reference)")
        print(f"  Pruned DCAE params: {count_params(pruned_dcae) / 1e6:.2f}M total")
        print(f"  Trainable params:   {sum(p.numel() for p in trainable_params) / 1e6:.2f}M")

    # ----- Optimizer (trainable parameters of pruned DCAE) -----
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # ----- LPIPS (optional) -----
    net_lpips = None
    if args.lambda_lpips > 0:
        import lpips
        net_lpips = lpips.LPIPS(net='vgg').cuda()
        net_lpips.requires_grad_(False)

    # ----- Dataset -----
    from omegaconf import OmegaConf
    train_args_sim = OmegaConf.create({
        "dataset_txt_paths_list": args.dataset_txt_paths_list,
        "dataset_prob_paths_list": args.dataset_prob_paths_list,
        "resolution": args.resolution,
        "deg_file_path": "params_realesrgan.yml",
        "neg_prompt": "",
    })
    dataset_train = PairedSROnlineTxtDataset(split="train", args=train_args_sim)
    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    # ----- Validation image (optional) -----
    val_image = None
    if args.validation_image_path and os.path.exists(args.validation_image_path):
        from PIL import Image
        val_image = Image.open(args.validation_image_path).convert('RGB')
        val_image = val_image.resize((args.resolution, args.resolution), Image.LANCZOS)
        val_tensor = transforms.ToTensor()(val_image).unsqueeze(0) * 2 - 1  # [-1, 1]

    # ----- Accelerator: prepare -----
    if net_lpips is not None:
        pruned_dcae, optimizer, dl_train, lr_scheduler, net_lpips = accelerator.prepare(
            pruned_dcae, optimizer, dl_train, lr_scheduler, net_lpips
        )
        net_lpips.requires_grad_(False)
    else:
        pruned_dcae, optimizer, dl_train, lr_scheduler = accelerator.prepare(
            pruned_dcae, optimizer, dl_train, lr_scheduler
        )

    # Move frozen full_dcae to device
    full_dcae = full_dcae.to(accelerator.device, dtype=torch.float16
                             if args.mixed_precision == "fp16" else torch.float32)
    enc_dtype = next(full_dcae.parameters()).dtype

    # ----- Trackers -----
    if accelerator.is_main_process:
        args.dataset_txt_paths_list = str(args.dataset_txt_paths_list)
        args.dataset_prob_paths_list = str(args.dataset_prob_paths_list)
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # ----- Training loop -----
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            with accelerator.accumulate(pruned_dcae):
                x_gt = batch["output_pixel_values"]  # (B, 3, H, W)
                
                # ---- Forward Full DCAE for reference latents ----
                with torch.no_grad():
                    latent_ref = full_dcae.encoder(x_gt.to(enc_dtype))

                # ---- Forward Pruned DCAE ----
                unwrapped_pruned = accelerator.unwrap_model(pruned_dcae)
                dec_dtype = next(unwrapped_pruned.parameters()).dtype

                if args.freeze_pruned_encoder:
                    # 如果 Encoder 被冻结，使用 Reference Latent 喂给 Pruned Decoder
                    x_recon = unwrapped_pruned.decoder(latent_ref.to(dtype=dec_dtype))
                else:
                    # 如果 Encoder 可训练，直接使用完整的 End-to-End 前向传播
                    x_recon = pruned_dcae(x_gt.to(dtype=dec_dtype))

                # ---- Reconstruction losses ----
                loss = torch.tensor(0.0, device=x_gt.device)

                # L1 loss
                loss_l1 = torch.tensor(0.0, device=loss.device)
                if args.lambda_l1 > 0:
                    loss_l1 = F.l1_loss(x_recon.float(), x_gt.float(), reduction="mean") * args.lambda_l1
                    loss = loss + loss_l1

                # L2 (MSE) loss
                loss_l2 = torch.tensor(0.0, device=loss.device)
                if args.lambda_l2 > 0:
                    loss_l2 = F.mse_loss(x_recon.float(), x_gt.float(), reduction="mean") * args.lambda_l2
                    loss = loss + loss_l2

                # LPIPS perceptual loss
                loss_lpips = torch.tensor(0.0, device=loss.device)
                if net_lpips is not None and args.lambda_lpips > 0:
                    recon_norm = x_recon.float().clamp(-1, 1) * 0.5 + 0.5
                    gt_norm = x_gt.float().clamp(-1, 1) * 0.5 + 0.5
                    loss_lpips = net_lpips(recon_norm, gt_norm).mean() * args.lambda_lpips
                    loss = loss + loss_lpips

                # Backward
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(pruned_dcae.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # ---- Logging & checkpointing ----
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {
                        "loss": loss.detach().item(),
                        "loss_l1": loss_l1.detach().item() if isinstance(loss_l1, torch.Tensor) else loss_l1,
                        "loss_l2": loss_l2.detach().item() if isinstance(loss_l2, torch.Tensor) else loss_l2,
                    }
                    if net_lpips is not None:
                        logs["loss_lpips"] = loss_lpips.detach().item() if isinstance(loss_lpips, torch.Tensor) else loss_lpips

                    progress_bar.set_postfix(**logs)

                    if global_step % args.checkpointing_steps == 1:
                        # 保存完整剪枝 DCAE 权重（包含 Encoder 和 Decoder）
                        unwrapped_model = accelerator.unwrap_model(pruned_dcae)
                        ckpt = {
                            "autoencoder_state_dict": unwrapped_model.state_dict(),
                            "decoder_state_dict": unwrapped_model.decoder.state_dict(),
                            "pruned_preset": args.pruned_preset,
                            "spatial_compression": args.spatial_compression,
                            "latent_channels": args.latent_channels,
                            "step": global_step,
                        }
                        outf = os.path.join(args.output_dir, "checkpoints", f"dcae_step_{global_step}.pth")
                        torch.save(ckpt, outf)

                    if global_step % args.validation_steps == 1 and val_image is not None:
                        with torch.no_grad():
                            val_gt = val_tensor.to(device=accelerator.device, dtype=enc_dtype)
                            unwrapped_model = accelerator.unwrap_model(pruned_dcae)
                            val_dec_dtype = next(unwrapped_model.parameters()).dtype
                            
                            val_recon = unwrapped_model(val_gt.to(dtype=val_dec_dtype))
                            val_loss = F.l1_loss(val_recon.float(), val_gt.float(), reduction="mean").item()
                            logs["val_l1"] = val_loss

                            val_pil = transforms.ToPILImage()(
                                val_recon[0].float().cpu().clamp(-1, 1) * 0.5 + 0.5
                            )
                            val_pil.save(os.path.join(
                                args.output_dir, "eval", f"recon_step_{global_step}.png"
                            ))

                    accelerator.log(logs, step=global_step)

                    current_loss = loss.detach().item()
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_path = os.path.join(args.output_dir, "checkpoints", "dcae_best.pth")
                        unwrapped_model = accelerator.unwrap_model(pruned_dcae)
                        torch.save({
                            "autoencoder_state_dict": unwrapped_model.state_dict(),
                            "decoder_state_dict": unwrapped_model.decoder.state_dict(),
                            "pruned_preset": args.pruned_preset,
                            "spatial_compression": args.spatial_compression,
                            "latent_channels": args.latent_channels,
                            "step": global_step,
                            "best_loss": best_loss,
                        }, best_path)

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    # ---- Final save ----
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "checkpoints", "dcae_final.pth")
        unwrapped_model = accelerator.unwrap_model(pruned_dcae)
        torch.save({
            "autoencoder_state_dict": unwrapped_model.state_dict(),
            "decoder_state_dict": unwrapped_model.decoder.state_dict(),
            "pruned_preset": args.pruned_preset,
            "spatial_compression": args.spatial_compression,
            "latent_channels": args.latent_channels,
            "step": global_step,
        }, final_path)
        print(f"[✓] DCAE pretraining complete!")
        print(f"    Final:  {final_path}")
        print(f"    Best:   {os.path.join(args.output_dir, 'checkpoints', 'dcae_best.pth')}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
