"""
DCAE-based DiffLoss SR training (Accelerator-based)
=====================================================

Pipeline: LQ -> DCAE.encoder -> DiffLoss(residual sample) -> DCAE.decoder -> SR

Losses:
  - DiffLoss residual loss (gt_latent - lq_latent in latent space)
  - L2 + LPIPS vs. teacher OSEDiff output (perceptual distillation, optional)
"""

import os
import sys
import gc
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs
from torchvision import transforms
from tqdm.auto import tqdm
from pathlib import Path

from diffusers.optimization import get_scheduler

from student import DCAESR
from student_zoo import get_model_cfg, print_models
from dataloaders.realsr_dataset import PairedSROnlineTxtDataset


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="DCAE-based lightweight SR training"
    )

    # ---- Paths ----
    parser.add_argument("--teacher_ckpt", type=str,
                        default="preset/models/osediff.pkl",
                        help="OSEDiff teacher checkpoint (.pkl)")
    parser.add_argument("--teacher_sd_path", type=str, default=None,
                        help="SD2.1 base path for teacher (e.g. preset/models/sd2-1). "
                             "Required when teacher distillation is enabled (lambda_l2/lpips > 0).")
    parser.add_argument("--output_dir", type=str, default="experience/dcae_sr")
    parser.add_argument("--logging_dir", type=str, default="logs")

    # ---- Dataset ----
    parser.add_argument("--dataset_txt_paths_list", type=str, nargs="+",
                        default=["YOUR_TXT_FILE_PATH"],
                        help="List of dataset txt file paths")
    parser.add_argument("--dataset_prob_paths_list", type=int, nargs="+",
                        default=[1],
                        help="Sampling probabilities for each dataset")
    parser.add_argument("--deg_file_path", type=str,
                        default="params_realesrgan.yml")
    parser.add_argument("--resolution", type=int, default=512)

    # ---- Training schedule ----
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=100000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing for the LightMapper.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    # ---- Optimizer ----
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        choices=["linear", "cosine", "cosine_with_restarts",
                                 "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)

    # ---- Mixed precision ----
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")

    # ---- Loss weights ----
    parser.add_argument("--lambda_l2", default=1.0, type=float,
                        help="Weight for L2 distillation loss vs teacher output")
    parser.add_argument("--lambda_lpips", default=2.0, type=float,
                        help="Weight for LPIPS distillation loss vs teacher output")

    # ---- Student model (from student_zoo.py) ----
    parser.add_argument("--student_model", type=str, default="dcae_f32c32",
                        help="Name of the student preset in student_zoo.MODEL_CFG. "
                             "Use 'list' to print available models and exit.")
    parser.add_argument("--list_student_models", action="store_true",
                        help="Print available student model presets and exit.")
    parser.add_argument("--pruned_pune_dcae_path", type=str, default=None,
                        help="Path to a Phase-1-pretrained pruned DCAE checkpoint "
                             "(from train_pruned_dcae.py). When provided, overrides the "
                             "decoder weights in the pruned DCAE before starting "
                             "Phase 2 distillation.")

    # ---- Misc ----
    parser.add_argument("--neg_prompt", type=str,
                        default="painting, oil painting, illustration, drawing, art, "
                                "sketch, cartoon, CG Style, 3D render, unreal engine, "
                                "blurring, dirty, messy, worst quality, low quality, "
                                "frames, watermark, signature, jpeg artifacts, "
                                "deformed, lowres, over-smooth")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--tracker_project_name", type=str,
                        default="train_dcae_sr")
    parser.add_argument("--ram_path", type=str, default=None,
                        help="Path to RAM tagger model for prompt generation")

    # ---- Teacher config ----
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.list_student_models or args.student_model.lower() == "list":
        print_models()
        sys.exit(0)

    # Validate student_model
    try:
        get_model_cfg(args.student_model)
    except KeyError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    return args


def main(args):
    # ----- Accelerator setup -----
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    # ----- 1. Build student model (DCAE + DiffLoss) -----
    if accelerator.is_main_process:
        print("[Distill] Creating student DCAESR (DiffLoss only)...")

    model_cfg = get_model_cfg(args.student_model)

    student = DCAESR(model_cfg)
    student.set_train()

    if student.diffloss is not None:
        n_diff = sum(p.numel() for p in student.diffloss.parameters()
                     if p.requires_grad)
    else:
        n_diff = 0
    if student.mapper is not None:
        n_map = sum(p.numel() for p in student.mapper.parameters()
                    if p.requires_grad)
    else:
        n_map = 0
    n_total = sum(p.numel() for p in student.parameters())
    if accelerator.is_main_process:
        print(f"  DiffLoss params: {n_diff / 1e6:.2f}M (trainable)")
        if student.mapper is not None:
            print(f"  Mapper params:   {n_map / 1e6:.2f}M (trainable)")
        print(f"  Total params:    {n_total / 1e6:.2f}M (incl. frozen DCAE)")

    # Load pretrained pruned DCAE from Phase 1 if provided (full model weights)
    if args.pruned_pune_dcae_path and os.path.exists(args.pruned_pune_dcae_path):
        if accelerator.is_main_process:
            print(f"[Distill] Loading pretrained pruned DCAE from {args.pruned_pune_dcae_path} ...")
        dec_ckpt = torch.load(args.pruned_pune_dcae_path, map_location="cpu")
        if "decoder_state_dict" in dec_ckpt:
            # From train_pruned_dcae.py output (full model checkpoint)
            student.autoencoder.decoder.load_state_dict(dec_ckpt["decoder_state_dict"])
        elif "autoencoder_state_dict" in dec_ckpt:
            # Full autoencoder checkpoint — extract decoder keys
            decoder_sd = {k.replace("decoder.", ""): v
                          for k, v in dec_ckpt["autoencoder_state_dict"].items()
                          if k.startswith("decoder.")}
            if decoder_sd:
                student.autoencoder.decoder.load_state_dict(decoder_sd, strict=False)
        else:
            print("[WARN] Could not find decoder_state_dict or autoencoder_state_dict "
                  "in pretrained_pune_dcae checkpoint — skipping.")
        if accelerator.is_main_process:
            print("  Decoder weights loaded.")
        student.set_train()  # Re-freeze encoder after loading weights

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        student.enable_gradient_checkpointing()
        if accelerator.is_main_process:
            print("[Distill] Gradient checkpointing enabled")

    # ----- 2. Load teacher OSEDiff (for pixel distillation targets) -----
    # Teacher runs in fp16 during data preprocessing — generate all targets first.
    # For online distillation (teacher forward per step), keep teacher loaded.
    # For memory efficiency, teachers are loaded on main process and handled below.
    teacher = None
    if args.teacher_ckpt and (args.lambda_l2 > 0 or args.lambda_lpips > 0):
        from osediff import OSEDiff_test

        if accelerator.is_main_process:
            print(f"[Distill] Loading teacher OSEDiff from {args.teacher_ckpt}...")

        class TeacherArgs:
            pretrained_model_name_or_path = args.teacher_sd_path
            osediff_path = args.teacher_ckpt
            mixed_precision = "fp16"
            vae_encoder_tiled_size = args.vae_encoder_tiled_size
            vae_decoder_tiled_size = args.vae_decoder_tiled_size
            latent_tiled_size = args.latent_tiled_size
            latent_tiled_overlap = args.latent_tiled_overlap
            merge_and_unload_lora = False

        teacher = OSEDiff_test(TeacherArgs())
        teacher.eval()
        teacher.requires_grad_(False)

        # Remove tiled VAE hooks for training images
        if hasattr(teacher.vae.encoder, 'original_forward'):
            teacher.vae.encoder.forward = teacher.vae.encoder.original_forward
        if hasattr(teacher.vae.decoder, 'original_forward'):
            teacher.vae.decoder.forward = teacher.vae.decoder.original_forward

        n_teacher = sum(p.numel() for p in teacher.unet.parameters())
        if accelerator.is_main_process:
            print(f"  Teacher UNet params: {n_teacher / 1e6:.2f}M")

    # ----- 3. Optimizer (DiffLoss + Mapper parameters) -----
    trainable_params = []
    if student.diffloss is not None:
        trainable_params += list(student.diffloss.parameters())
    if student.mapper is not None:
        trainable_params += list(student.mapper.parameters())
    if not trainable_params:
        trainable_params = list(student.parameters())

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

    # ----- 4. LPIPS -----
    import lpips
    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    # ----- 5. Dataset -----
    dataset_train = PairedSROnlineTxtDataset(split="train", args=args)
    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    # ----- 6. RAM tagger (for prompt generation) -----
    model_vlm = None
    inference_ram = None
    ram_transforms = None
    if args.ram_path is not None and os.path.exists(args.ram_path):
        from ram.models.ram_lora import ram
        from ram import inference_ram as _inference_ram
        inference_ram = _inference_ram
        ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        model_vlm = ram(
            pretrained=args.ram_path,
            pretrained_condition=None,
            image_size=384,
            vit='swin_l',
        )
        model_vlm.eval()
        model_vlm.to("cuda", dtype=torch.float16)

    # ----- 7. Accelerator: prepare -----
    if teacher is not None:
        student, teacher, optimizer, dl_train, lr_scheduler, net_lpips = \
            accelerator.prepare(
                student, teacher, optimizer, dl_train, lr_scheduler, net_lpips
            )
        net_lpips.requires_grad_(False)
    else:
        student, optimizer, dl_train, lr_scheduler = \
            accelerator.prepare(student, optimizer, dl_train, lr_scheduler)

    # ----- 8. Trackers -----
    if accelerator.is_main_process:
        args.dataset_txt_paths_list = str(args.dataset_txt_paths_list)
        args.dataset_prob_paths_list = str(args.dataset_prob_paths_list)
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name,
                                  config=tracker_config)

    # ----- 9. Training loop -----
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    global_step = 0

    for epoch in range(args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            with accelerator.accumulate(student):
                x_src = batch["conditioning_pixel_values"]   # LQ (B,3,H,W)
                x_tgt = batch["output_pixel_values"]         # GT (B,3,H,W)
                B = x_src.shape[0]

                # ---- Teacher forward (distillation target) ----
                has_teacher = teacher is not None and (
                    args.lambda_l2 > 0 or args.lambda_lpips > 0
                )
                if has_teacher:
                    # Generate prompt via RAM
                    if model_vlm is not None:
                        x_tgt_ram = ram_transforms(x_tgt * 0.5 + 0.5)
                        caption = inference_ram(
                            x_tgt_ram.to(dtype=torch.float16), model_vlm
                        )
                        prompt_batch = [f'{c}' for c in caption]
                    else:
                        prompt_batch = ["a high quality, sharp, detailed photo"] * B

                    with torch.no_grad():
                        if hasattr(teacher, 'get_pixels_and_embeds'):
                            teacher_pixels, _, _ = teacher.get_pixels_and_embeds(
                                x_src, prompt_batch, args.neg_prompt
                            )
                        else:
                            # Fallback: manual teacher forward
                            prompt_embeds = teacher.encode_prompt(prompt_batch)
                            lq_latent = teacher.vae.encode(
                                x_src.to(teacher.weight_dtype)
                            ).latent_dist.sample() * teacher.vae.config.scaling_factor
                            model_pred = teacher.unet(
                                lq_latent, teacher.timesteps,
                                encoder_hidden_states=prompt_embeds.to(teacher.weight_dtype),
                            ).sample
                            x_denoised = teacher.noise_scheduler.step(
                                model_pred, teacher.timesteps, lq_latent, return_dict=True
                            ).prev_sample
                            teacher_pixels = teacher.vae.decode(
                                x_denoised.to(teacher.weight_dtype) / teacher.vae.config.scaling_factor
                            ).sample.clamp(-1, 1)
                    teacher_pixels = teacher_pixels.float()
                else:
                    teacher_pixels = None

                # ---- Student forward (target = teacher output) ----
                # DiffLoss learns encoder(teacher) - encoder(lq) in latent space,
                # pixel losses compare decoder output vs teacher_pixels.
                # When no teacher, run inference-only (no DiffLoss training).
                student_out = student(x_src, target=teacher_pixels)
                student_pixels = student_out["gen"]
                loss_diff = student_out.get("loss_diff", None)

                # ---- Loss computation (all pixel-level) ----
                loss = torch.tensor(0.0, device=x_src.device)

                if has_teacher and args.lambda_l2 > 0:
                    loss_l2 = F.mse_loss(
                        student_pixels.float(), teacher_pixels,
                        reduction="mean"
                    ) * args.lambda_l2
                    loss = loss + loss_l2
                else:
                    loss_l2 = torch.tensor(0.0, device=loss.device)

                if has_teacher and args.lambda_lpips > 0:
                    loss_lpips_val = net_lpips(
                        student_pixels.float(), teacher_pixels
                    ).mean() * args.lambda_lpips
                    loss = loss + loss_lpips_val
                else:
                    loss_lpips_val = torch.tensor(0.0, device=loss.device)

                # DiffLoss residual loss
                if loss_diff is not None:
                    loss = loss + loss_diff
                else:
                    loss_diff = torch.tensor(0.0, device=loss.device)

                # Backward
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    trainable_params = []
                    if student.diffloss is not None:
                        trainable_params += list(student.diffloss.parameters())
                    if student.mapper is not None:
                        trainable_params += list(student.mapper.parameters())
                    if not trainable_params:
                        trainable_params = list(student.parameters())
                    accelerator.clip_grad_norm_(
                        trainable_params, args.max_grad_norm
                    )

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
                        "loss_l2": loss_l2.detach().item()
                        if isinstance(loss_l2, torch.Tensor) else loss_l2,
                        "loss_lpips": loss_lpips_val.detach().item()
                        if isinstance(loss_lpips_val, torch.Tensor) else loss_lpips_val,
                        "loss_diff": loss_diff.detach().item()
                        if isinstance(loss_diff, torch.Tensor) else loss_diff,
                    }
                    progress_bar.set_postfix(**logs)

                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(
                            args.output_dir, "checkpoints",
                            f"step_{global_step}.pth"
                        )
                        accelerator.unwrap_model(student).save_checkpoint(outf)

                    accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    # ---- Final save ----
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "checkpoints", "step_final.pth")
        accelerator.unwrap_model(student).save_checkpoint(final_path)
        print(f"[✓] Training complete! Final model: {final_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
