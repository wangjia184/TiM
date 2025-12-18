"""
train.py — Pixel-space C2I TiM training (64×64 RGB), **no VAE**, **no CFG**.

This script fixes the key issue in the previous my-app example:
it now trains a true TiM-style transition model objective (x_t interpolation + transition target),
instead of masked patch inpainting.
"""

from __future__ import annotations

import argparse
import os
import glob
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from ema_pytorch import EMA

from dataset import CelebAParquetDataset
from model import TiMC2IModel
from transition import OT_FM, SimpleTransitionSchedule


def to_01(x: torch.Tensor) -> torch.Tensor:
    # x in [-1,1] -> [0,1]
    return (x + 1.0) * 0.5


@torch.no_grad()
def sample_for_tb(schedule: SimpleTransitionSchedule, model: torch.nn.Module, attrs: torch.Tensor, nfe: int, img_size: int) -> torch.Tensor:
    model.eval()
    z = torch.randn(attrs.shape[0], 3, img_size, img_size, device=attrs.device, dtype=torch.float32)
    traj = schedule.sample(model, global_attrs=attrs, z=z, num_steps=nfe, sample_type="transition")
    imgs = traj[-1]
    return to_01(imgs).clamp(0, 1)


def param_norm(model: torch.nn.Module) -> torch.Tensor:
    """L2 norm of trainable parameters (for normalized grad metrics)."""
    with torch.no_grad():
        acc = torch.zeros((), device=next(model.parameters()).device)
        for p in model.parameters():
            if p.requires_grad:
                acc = acc + p.detach().float().pow(2).sum()
        return acc.sqrt()


def main():
    parser = argparse.ArgumentParser()
    # NOTE: resolved relative to repo root at runtime (see below)
    parser.add_argument("--data", type=str, default="celeba/data/train-*.parquet")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--sample_every", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--logdir_root", type=str, default="/workspace/.runs")
    parser.add_argument("--sample_batch", type=int, default=8)
    parser.add_argument("--nfe1", type=int, default=1)
    parser.add_argument("--nfe2", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=0, help="If >0, stop training after this many optimizer steps (smoke test).")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # model knobs
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=2, help="Default aligns with B config for 64x64 RGB.")
    parser.add_argument("--hidden_size", type=int, default=768, help="B config: hidden_size=768")
    parser.add_argument("--depth", type=int, default=12, help="B config: depth=12")
    parser.add_argument("--num_heads", type=int, default=12, help="B config: num_heads=12")
    parser.add_argument("--mlp_ratio", type=float, default=4.0)

    # schedule knobs
    parser.add_argument("--consistency_ratio", type=float, default=0.1, help="Reference C2I configs use 0.1")
    parser.add_argument("--diffusion_ratio", type=float, default=0.5, help="Reference C2I configs use 0.5")
    parser.add_argument("--differential_epsilon", type=float, default=0.005)
    parser.add_argument("--weight_time_type", type=str, default="sqrt", choices=["constant", "reciprocal", "sqrt", "square", "Soft-Min-SNR"])
    parser.add_argument("--weight_time_tangent", action="store_true", default=True)
    parser.add_argument("--weight_time_sigmoid", action="store_true")
    parser.add_argument("--use_dir_loss", action="store_true", help="Add cosine directional loss (reference option).")

    # EMA (ema-pytorch)
    parser.add_argument("--ema_beta", type=float, default=0.9999)
    parser.add_argument("--ema_update_after_step", type=int, default=100)
    parser.add_argument("--ema_update_every", type=int, default=1)

    # checkpointing (optional; off by default to avoid large disk writes)
    parser.add_argument("--ckpt_dir", type=str, default="/workspace/checkpoints_my_app")
    parser.add_argument("--save_every", type=int, default=0, help="If >0, save checkpoint every N optimizer steps.")
    args = parser.parse_args()

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    device = accelerator.device

    # Model (kept transformer-ish like reference, but simpler)
    model = TiMC2IModel(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        attr_dim=8,
        new_condition="t-r",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Transport + schedule (OT_FM is the simplest stable option in bounded t∈(0,1))
    transport = OT_FM(P_mean=-0.4, P_std=1.0, sigma_d=1.0, T_max=1.0, T_min=0.0)
    schedule = SimpleTransitionSchedule(
        transport=transport,
        diffusion_ratio=args.diffusion_ratio,
        consistency_ratio=args.consistency_ratio,
        derivative_type="dde",
        differential_epsilon=args.differential_epsilon,
        weight_t_and_r=True,
        weight_time_type=args.weight_time_type,
        weight_time_tangent=args.weight_time_tangent,
        weight_time_sigmoid=args.weight_time_sigmoid,
        adaptive_weighting=True,
        use_dir_loss=args.use_dir_loss,
    )

    start_epoch = 0
    step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0))
        step = int(ckpt.get("step", 0))

    # Resolve dataset pattern relative to repo root, so `docker exec ... /workspace/my-app/train.py` works reliably.
    repo_root = Path(__file__).resolve().parents[1]  # /workspace
    data_pattern = args.data
    if not os.path.isabs(data_pattern):
        data_pattern = str(repo_root / data_pattern)
    # Validate glob early for a clearer error than HF datasets.
    if len(glob.glob(data_pattern)) == 0:
        raise FileNotFoundError(
            f"No parquet files matched --data pattern: {data_pattern}\n"
            f"Tip: expected files like: {repo_root / 'celeba/data/train-00000-of-00018.parquet'}"
        )
    dataset = CelebAParquetDataset(data_pattern, output_size=args.img_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    # EMA tracks the underlying model (not the DDP wrapper)
    ema = EMA(
        accelerator.unwrap_model(model),
        beta=args.ema_beta,
        update_after_step=args.ema_update_after_step,
        update_every=args.ema_update_every,
    ).to(device)

    writer: SummaryWriter | None = None
    if accelerator.is_main_process:
        os.makedirs(args.logdir_root, exist_ok=True)
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(args.logdir_root, run_name)
        writer = SummaryWriter(log_dir=log_dir)
        writer.add_text("hparams", str(vars(args)), global_step=0)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}") if accelerator.is_main_process else loader

        for batch in pbar:
            with accelerator.accumulate(model):
                x = batch["image"].to(device, dtype=torch.float32)  # [-1,1]
                global_attrs = batch["global_attrs"].to(device, dtype=torch.float32)

                z = torch.randn_like(x)

                with accelerator.autocast():
                    out = schedule.loss(model, x=x, z=z, model_kwargs={"global_attrs": global_attrs})
                    loss = out.loss

                optimizer.zero_grad(set_to_none=True)
                accelerator.backward(loss)

                grad_norm = None
                if accelerator.sync_gradients and args.max_grad_norm and args.max_grad_norm > 0:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()

                if accelerator.sync_gradients:
                    step += 1
                    ema.update()

                    if writer is not None and (step % args.log_every == 0):
                        lr = optimizer.param_groups[0]["lr"]
                        writer.add_scalar("train/loss", float(loss.detach().cpu()), step)
                        writer.add_scalar("train/denoising_loss", float(out.denoising_loss.detach().cpu()), step)
                        writer.add_scalar("train/directional_loss", float(out.directional_loss.detach().cpu()), step)
                        writer.add_scalar("train/weight_mean", float(out.weight_mean.detach().cpu()), step)
                        writer.add_scalar("train/lr", float(lr), step)

                        if grad_norm is not None:
                            gn = float(grad_norm.detach().float().cpu())
                            pn = float(param_norm(accelerator.unwrap_model(model)).detach().float().cpu())
                            writer.add_scalar("train/grad_norm", gn, step)
                            writer.add_scalar("train/param_norm", pn, step)
                            writer.add_scalar("train/grad_norm_normalized", gn / (pn + 1e-12), step)

                    if writer is not None and (step % args.sample_every == 0):
                        sample_attrs = global_attrs[: args.sample_batch].detach()
                        imgs1 = sample_for_tb(schedule, ema.ema_model, sample_attrs, nfe=args.nfe1, img_size=args.img_size)
                        imgs2 = sample_for_tb(schedule, ema.ema_model, sample_attrs, nfe=args.nfe2, img_size=args.img_size)
                        writer.add_images(f"samples/nfe{args.nfe1}", imgs1, step)
                        writer.add_images(f"samples/nfe{args.nfe2}", imgs2, step)

                    if writer is not None and args.save_every and args.save_every > 0 and (step % args.save_every == 0):
                        os.makedirs(args.ckpt_dir, exist_ok=True)
                        ckpt_path = os.path.join(args.ckpt_dir, f"step_{step:08d}.pt")
                        if accelerator.is_main_process:
                            torch.save(
                                {
                                    "epoch": epoch,
                                    "step": step,
                                    "model": accelerator.unwrap_model(model).state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "ema": ema.state_dict(),
                                    "args": vars(args),
                                },
                                ckpt_path,
                            )

                    if accelerator.is_main_process:
                        try:
                            pbar.set_postfix(
                                loss=float(loss.detach().cpu()),
                                denoise=float(out.denoising_loss.detach().cpu()),
                            )
                        except Exception:
                            pass

                    if args.max_steps > 0 and step >= args.max_steps:
                        break

            if args.max_steps > 0 and step >= args.max_steps:
                break

        if args.max_steps > 0 and step >= args.max_steps:
            break

    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()