#!/usr/bin/env python3
"""Fine-tuning script for pi0.5 on HSR data. Supports single-GPU and multi-GPU (DDP).

Usage (single GPU):
    python -m openpi.training.train \\
        --config-name pi05_hsr_mydata \\
        --exp-name my_experiment \\
        --data-root /mnt/s3

Usage (multi-GPU, e.g. p4d.24xlarge の 8xA100):
    torchrun --nproc_per_node=8 -m openpi.training.train \\
        --config-name pi05_hsr_mydata \\
        --exp-name my_experiment \\
        --data-root /mnt/s3
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer_lib
from openpi.models import model as _model
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


logger = logging.getLogger(__name__)

HF_BASE_MODEL_ID = "lerobot/pi05_base"


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def _init_distributed() -> tuple[int, int, int]:
    """Initialize distributed process group if launched via torchrun.

    Returns:
        (rank, local_rank, world_size)
        rank=0 / local_rank=0 / world_size=1  when running single-GPU.
    """
    if "LOCAL_RANK" not in os.environ:
        return 0, 0, 1

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def _is_main(rank: int) -> bool:
    return rank == 0


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune pi0.5 on HSR data")
    parser.add_argument("--config-name", required=True, help="TrainConfig name (e.g. pi05_hsr_mydata)")
    parser.add_argument("--exp-name", default=None, help="Experiment name (used for checkpoint directory)")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Local root for datasets (<data-root>/<repo_id>/). S3マウントパスなど。",
    )
    parser.add_argument("--checkpoint-base-dir", default=None, help="Override checkpoint base directory")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing checkpoint directory")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-GPU batch size (config値を上書き)")
    parser.add_argument("--num-steps", type=int, default=None, help="Override num_train_steps")
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--compile", action="store_true", help="torch.compile でモデルを最適化（初回ステップに数分かかるが以降高速）")
    parser.add_argument(
        "--task-names",
        nargs="+",
        default=None,
        metavar="TASK",
        help="タスク名でエピソードを絞り込む（部分一致）。"
             "例: --task-names 'Make coffee' 'Open the towel stand and hang the towel.' 'Washing dishes in the dishwasher'",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="task_success=True のエピソードのみ使用する。",
    )
    parser.add_argument(
        "--max-episodes-per-task",
        type=int,
        default=None,
        metavar="N",
        help="タスクごとのエピソード上限数。例: --max-episodes-per-task 100",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        metavar="R",
        help="LoRA rank for Action Expert fine-tuning (default: 16). "
             "Set to 0 to disable LoRA and do full fine-tuning.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=None,
        metavar="A",
        help="LoRA alpha (default: 2 * lora_rank).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _resolve_base_weights(pytorch_weight_path: str | None, rank: int) -> Path:
    """Base model weights を解決する。rank=0 のみダウンロードし他は待機。"""
    from huggingface_hub import snapshot_download

    if pytorch_weight_path:
        p = Path(pytorch_weight_path)
        if p.exists():
            if _is_main(rank):
                logger.info("Using weights from config: %s", p)
            return p
        if _is_main(rank):
            logger.warning("pytorch_weight_path '%s' not found. Falling back to HuggingFace Hub.", pytorch_weight_path)

    local_dir_str = f"./checkpoints/{HF_BASE_MODEL_ID.replace('/', '_')}"

    # rank=0 だけダウンロード
    if _is_main(rank):
        logger.info("Downloading base weights from HuggingFace Hub: %s", HF_BASE_MODEL_ID)
        snapshot_download(repo_id=HF_BASE_MODEL_ID, local_dir=local_dir_str)

    # 他のランクはダウンロード完了まで待機
    _barrier(dist.get_world_size() if dist.is_initialized() else 1)

    candidates = list(Path(local_dir_str).glob("**/*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"No .safetensors file found in: {local_dir_str}")
    return candidates[0]


def _load_pytorch_weights(model: PI0Pytorch, weight_path: Path, rank: int) -> None:
    import safetensors.torch

    if _is_main(rank):
        logger.info("Loading weights from %s", weight_path)

    if weight_path.suffix == ".safetensors":
        state_dict = safetensors.torch.load_file(str(weight_path), device="cpu")
    else:
        state_dict = torch.load(str(weight_path), map_location="cpu", weights_only=True)
        state_dict = state_dict.get("model", state_dict.get("state_dict", state_dict))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if _is_main(rank):
        if missing:
            logger.warning("Missing keys (%d): %s", len(missing), missing[:10])
        if unexpected:
            logger.warning("Unexpected keys (%d): %s", len(unexpected), unexpected[:10])
        logger.info("Weights loaded successfully")


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_dir: Path,
) -> None:
    import safetensors.torch

    step_dir = checkpoint_dir / str(step)
    step_dir.mkdir(parents=True, exist_ok=True)

    # DDP でラップされていれば .module を取る
    raw_model = model.module if isinstance(model, DDP) else model
    safetensors.torch.save_file(raw_model.state_dict(), str(step_dir / "model.safetensors"))
    torch.save({"step": step, "optimizer": optimizer.state_dict()}, str(step_dir / "optimizer.pt"))
    (checkpoint_dir / "latest").write_text(str(step))
    logger.info("Saved checkpoint at step %d -> %s", step, step_dir)


def _load_latest_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path,
) -> int:
    import safetensors.torch

    latest_file = checkpoint_dir / "latest"
    if not latest_file.exists():
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    step = int(latest_file.read_text().strip())
    step_dir = checkpoint_dir / str(step)
    logger.info("Resuming from step %d", step)

    raw_model = model.module if isinstance(model, DDP) else model
    state_dict = safetensors.torch.load_file(str(step_dir / "model.safetensors"), device="cpu")
    raw_model.load_state_dict(state_dict, strict=True)

    opt_state = torch.load(str(step_dir / "optimizer.pt"), map_location="cpu", weights_only=False)
    optimizer.load_state_dict(opt_state["optimizer"])
    return step


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def _build_data_loader(
    config: _config.TrainConfig,
    per_gpu_batch_size: int,
    local_root: str | None,
    task_names: list[str] | None,
    success_only: bool,
    max_episodes_per_task: int | None,
    rank: int,
    world_size: int,
) -> torch.utils.data.DataLoader:
    data_config = config.data.create(config.assets_dirs, config.model)

    # Use DirectParquetDataset (bypasses LeRobotDataset's 140GB metadata scan)
    if local_root is not None:
        dataset = _data_loader.create_direct_dataset(
            data_config,
            config.model.action_horizon,
            local_root=local_root,
            task_names=task_names,
            success_only=success_only,
            max_episodes_per_task=max_episodes_per_task,
        )
    else:
        dataset = _data_loader.create_torch_dataset(
            data_config,
            config.model.action_horizon,
            config.model,
            local_root=local_root,
            task_names=task_names,
            success_only=success_only,
            max_episodes_per_task=max_episodes_per_task,
        )
    dataset = _data_loader.transform_dataset(dataset, data_config)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) \
        if world_size > 1 else None

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=4 if config.num_workers > 0 else None,
    )
    return loader, sampler


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def _make_lr_lambda(lr_schedule: _optimizer_lib.CosineDecaySchedule):
    warmup = lr_schedule.warmup_steps
    peak = lr_schedule.peak_lr
    decay_steps = lr_schedule.decay_steps
    end_lr = lr_schedule.decay_lr

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / (warmup + 1)
        progress = min(1.0, (step - warmup) / max(1, decay_steps - warmup))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return (end_lr + (peak - end_lr) * cosine) / peak

    return lr_lambda


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_device(batch: dict, device: torch.device) -> dict:
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device=device, non_blocking=True)
        elif isinstance(v, dict):
            result[k] = _to_device(v, device)
        else:
            result[k] = v
    return result


def _infinite_iter(loader, sampler, epoch_ref: list):
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch_ref[0])
            epoch_ref[0] += 1
        yield from loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    args = parse_args()

    # --- Distributed init ---
    rank, local_rank, world_size = _init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if _is_main(rank):
        logger.info("world_size=%d, device=%s", world_size, device)

    # --- Config ---
    config = _config.get_config(args.config_name)
    if args.exp_name:
        config = dataclasses.replace(config, exp_name=args.exp_name)
    if args.batch_size:
        config = dataclasses.replace(config, batch_size=args.batch_size)
    if args.num_steps:
        config = dataclasses.replace(config, num_train_steps=args.num_steps)
    if args.log_interval:
        config = dataclasses.replace(config, log_interval=args.log_interval)
    if args.save_interval:
        config = dataclasses.replace(config, save_interval=args.save_interval)
    if args.checkpoint_base_dir:
        config = dataclasses.replace(config, checkpoint_base_dir=args.checkpoint_base_dir)
    if args.no_wandb:
        config = dataclasses.replace(config, wandb_enabled=False)

    # per-GPU batch size（config.batch_size を GPU 数で割る）
    assert config.batch_size % world_size == 0, \
        f"batch_size ({config.batch_size}) must be divisible by world_size ({world_size})"
    per_gpu_batch = config.batch_size // world_size

    if _is_main(rank):
        logger.info("Config: %s  |  global_batch=%d  per_gpu_batch=%d  steps=%d",
                    args.config_name, config.batch_size, per_gpu_batch, config.num_train_steps)

    use_amp = device.type == "cuda" and config.pytorch_training_precision == "bfloat16"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # --- Model ---
    assert isinstance(config.model, _model.BaseModelConfig)
    model = PI0Pytorch(config.model).to(device)

    # --- Load base weights (rank=0 がダウンロード、他は待機) ---
    if not args.resume:
        weight_path = _resolve_base_weights(config.pytorch_weight_path, rank)
        _load_pytorch_weights(model, weight_path, rank)

    # --- LoRA fine-tuning (default: rank=16, freeze backbone) ---
    lora_rank = args.lora_rank
    if lora_rank > 0:
        lora_alpha = args.lora_alpha if args.lora_alpha is not None else 2.0 * lora_rank
        param_info = model.apply_lora_finetuning(rank=lora_rank, alpha=lora_alpha)
        if _is_main(rank):
            logger.info(
                "LoRA enabled: rank=%d alpha=%.0f trainable=%.1fM frozen=%.1fM",
                lora_rank, lora_alpha,
                param_info["trainable"] / 1e6, param_info["frozen"] / 1e6,
            )
    else:
        if _is_main(rank):
            logger.info("Full fine-tuning (LoRA disabled)")

    # --- torch.compile ---
    if args.compile:
        if _is_main(rank):
            logger.info("torch.compile を適用中（初回ステップに数分かかります）...")
        model = torch.compile(model)

    # --- DDP wrap ---
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # --- Optimizer (only trainable params) ---
    lr_schedule = config.lr_schedule
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr_schedule.peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_make_lr_lambda(lr_schedule))
    scaler = GradScaler(enabled=use_amp)

    # --- Checkpoint dir (rank=0 のみ作成) ---
    checkpoint_dir = config.checkpoint_dir
    start_step = 0
    if _is_main(rank):
        if checkpoint_dir.exists() and not args.overwrite:
            if args.resume:
                start_step = _load_latest_checkpoint(model, optimizer, checkpoint_dir)
                logger.info("Resumed from step %d", start_step)
            else:
                raise FileExistsError(
                    f"Checkpoint dir {checkpoint_dir} exists. Pass --resume or --overwrite."
                )
        else:
            if args.overwrite and checkpoint_dir.exists():
                import shutil
                shutil.rmtree(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # resume 時は start_step を全ランクで同期
    if args.resume and world_size > 1:
        t = torch.tensor([start_step], device=device)
        dist.broadcast(t, src=0)
        start_step = int(t.item())

    # --- WandB (rank=0 のみ) ---
    wandb_run = None
    if _is_main(rank) and config.wandb_enabled:
        try:
            import wandb
            wandb_run = wandb.init(
                project=config.project_name,
                name=config.exp_name,
                config={
                    "config_name": args.config_name,
                    "global_batch_size": config.batch_size,
                    "world_size": world_size,
                    "num_train_steps": config.num_train_steps,
                    "peak_lr": lr_schedule.peak_lr,
                },
                resume="allow" if args.resume else None,
            )
        except ImportError:
            logger.warning("wandb not installed. Skipping.")

    # --- Data ---
    if _is_main(rank):
        logger.info(
            "Building data loader (data_root=%s, task_names=%s, success_only=%s, max_episodes_per_task=%s)...",
            args.data_root, args.task_names, args.success_only, args.max_episodes_per_task,
        )
    loader, sampler = _build_data_loader(
        config, per_gpu_batch, args.data_root, args.task_names,
        args.success_only, args.max_episodes_per_task, rank, world_size,
    )
    if _is_main(rank):
        logger.info("Dataset ready: %d samples", len(loader.dataset))

    # --- Training loop ---
    import time

    model.train()
    step = start_step
    epoch_ref = [0]
    loader_iter = _infinite_iter(loader, sampler, epoch_ref)

    from openpi.models.model import Observation

    if _is_main(rank):
        logger.info("Start training: step %d / %d", step, config.num_train_steps)

    # ---- 計算時間推定: 最初の5ステップで1ステップあたりの時間を計測 ----
    WARMUP_STEPS = 5
    step_times: list[float] = []
    estimated_sec_per_step: float | None = None
    train_start_time = time.time()

    while step < config.num_train_steps:
        t0 = time.time()

        batch = next(loader_iter)
        obs = Observation.from_dict(_to_device(batch, device))
        actions = batch["actions"].to(device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            loss_per_element = model(obs, actions)
            loss = loss_per_element.mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        raw = model.module if isinstance(model, DDP) else model
        torch.nn.utils.clip_grad_norm_(raw.parameters(), config.optimizer.clip_gradient_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        step += 1
        step_times.append(time.time() - t0)

        # ウォームアップ後に1ステップあたりの推定時間を確定して表示
        if _is_main(rank) and len(step_times) == WARMUP_STEPS and estimated_sec_per_step is None:
            estimated_sec_per_step = sum(step_times) / len(step_times)
            remaining_steps = config.num_train_steps - step
            total_remaining_sec = estimated_sec_per_step * remaining_steps
            total_sec = estimated_sec_per_step * config.num_train_steps
            h_total, r_total = divmod(int(total_sec), 3600)
            m_total = r_total // 60
            h_rem, r_rem = divmod(int(total_remaining_sec), 3600)
            m_rem = r_rem // 60
            logger.info(
                "===== 計算時間推定 =====\n"
                "  1ステップあたり: %.2f 秒\n"
                "  総学習時間 (推定): %d 時間 %d 分\n"
                "  残り時間   (推定): %d 時間 %d 分\n"
                "========================",
                estimated_sec_per_step, h_total, m_total, h_rem, m_rem,
            )

        if _is_main(rank) and step % config.log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - train_start_time
            if estimated_sec_per_step is not None:
                remaining_steps = config.num_train_steps - step
                eta_sec = estimated_sec_per_step * remaining_steps
                h_eta, r_eta = divmod(int(eta_sec), 3600)
                m_eta = r_eta // 60
                h_ela, r_ela = divmod(int(elapsed), 3600)
                m_ela = r_ela // 60
                logger.info(
                    "step=%d/%d loss=%.4f lr=%.2e | 経過 %dh%02dm | ETA %dh%02dm",
                    step, config.num_train_steps, loss.item(), lr,
                    h_ela, m_ela, h_eta, m_eta,
                )
            else:
                logger.info("step=%d loss=%.4f lr=%.2e", step, loss.item(), lr)
            if wandb_run:
                wandb_run.log({"train/loss": loss.item(), "train/lr": lr}, step=step)

        if _is_main(rank) and (step % config.save_interval == 0 or step == config.num_train_steps):
            _save_checkpoint(model, optimizer, step, checkpoint_dir)

    if _is_main(rank):
        logger.info("Training complete at step %d.", step)
        if wandb_run:
            wandb_run.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
