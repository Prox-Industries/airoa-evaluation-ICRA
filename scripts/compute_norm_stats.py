#!/usr/bin/env python3
"""Compute normalization statistics for a LeRobot dataset.

Applies repack + data transforms (matching the training pipeline) before
computing per-key mean/std/quantiles, so that normalization is consistent
with what the model actually sees during training.

Usage:
    # Recommended: use config (applies transforms correctly)
    python scripts/compute_norm_stats.py \
        --config-name my_robot_pi05_lora \
        --data-root /path/to/datasets

    # Direct mode (raw parquet values, no transforms applied):
    python scripts/compute_norm_stats.py \
        --dataset-dir /path/to/my_lerobot_dataset \
        --output-dir ./assets/my_robot_pi05_lora/my_org/my_robot_data

The output is a single `norm_stats.json` file placed in the assets directory
that OpenPI's data pipeline reads at training time.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openpi.shared.normalize import RunningStats, save as save_norm_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Keys to compute normalization stats for (post-transform names).
NORM_KEYS = ("state", "actions")


def _remove_non_numeric(data: dict) -> dict:
    """Remove non-numeric (string, etc.) values from a sample dict."""
    result = {}
    for k, v in data.items():
        if isinstance(v, (np.ndarray, np.generic, int, float)):
            result[k] = v
        elif isinstance(v, dict):
            cleaned = _remove_non_numeric(v)
            if cleaned:
                result[k] = cleaned
    return result


def compute_from_config(
    config_name: str,
    data_root: str | None = None,
    max_samples: int | None = None,
):
    """Compute norm stats using config to resolve dataset, transforms, and output paths.

    This is the recommended mode. It:
    1. Creates a dataset via the config's data pipeline
    2. Applies repack_transforms + data_transforms (same as training)
    3. Computes stats on the transformed "state" and "actions" keys
    """
    from openpi.training.config import get_config
    import openpi.training.data_loader as _data_loader
    import openpi.transforms as _transforms

    config = get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    # Resolve output path
    data_config_factory = config.data
    repo_id = data_config_factory.repo_id
    assets_dir = data_config_factory.assets.assets_dir or str(config.assets_dirs)
    asset_id = data_config_factory.assets.asset_id or repo_id
    output_dir = Path(assets_dir) / asset_id

    # Create raw dataset (no transforms yet)
    dataset = _data_loader.create_torch_dataset(
        data_config,
        action_horizon=config.model.action_horizon,
        model_config=config.model,
        local_root=data_root,
    )

    # Apply repack + data transforms only (NOT normalize or model transforms).
    # This matches the original OpenPI compute_norm_stats behavior.
    transform = _transforms.compose([
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
    ])

    num_samples = len(dataset)
    if max_samples is not None:
        num_samples = min(num_samples, max_samples)

    logger.info(
        "Computing norm stats from %d samples (dataset size: %d), keys: %s",
        num_samples, len(dataset), NORM_KEYS,
    )

    # Initialize running stats for each key
    stats: dict[str, RunningStats] = {k: RunningStats() for k in NORM_KEYS}

    for i in range(num_samples):
        try:
            sample = dataset[i]
            transformed = transform(sample)
            transformed = _remove_non_numeric(transformed)
        except Exception as e:
            logger.warning("Skipping sample %d: %s", i, e)
            continue

        for key in NORM_KEYS:
            if key not in transformed:
                continue
            arr = np.asarray(transformed[key], dtype=np.float32)
            if arr.ndim == 0:
                arr = arr.reshape(1, 1)
            elif arr.ndim == 1:
                arr = arr[np.newaxis, :]  # [1, D]
            elif arr.ndim > 2:
                # For action chunks [H, D], flatten to [H, D] which is already 2D
                arr = arr.reshape(-1, arr.shape[-1])
            stats[key].update(arr)

        if (i + 1) % 500 == 0:
            logger.info("Processed %d / %d samples", i + 1, num_samples)

    # Extract final stats
    result = {}
    for key, rs in stats.items():
        try:
            result[key] = rs.get_statistics()
            logger.info(
                "  %s: mean=%s, std=%s",
                key,
                np.array2string(result[key].mean[:4], precision=3),
                np.array2string(result[key].std[:4], precision=3),
            )
        except ValueError:
            logger.warning("  %s: not enough data, skipping", key)

    if not result:
        raise ValueError(
            "No statistics computed. Check that dataset contains 'state' and 'actions' "
            "keys after repack + data transforms."
        )

    save_norm_stats(output_dir, result)
    logger.info("Saved norm stats to %s", output_dir / "norm_stats.json")


def compute_from_parquets(
    dataset_dir: Path,
    max_episodes: int | None = None,
) -> dict:
    """Compute norm stats by reading parquet files directly (no transforms).

    WARNING: This mode reads raw parquet values without applying transforms.
    Use --config-name mode for correct normalization that matches training.
    """
    import pyarrow.parquet as pq

    logger.warning(
        "Direct parquet mode: stats are computed on RAW values without transforms. "
        "Use --config-name for transform-aware stats that match training."
    )

    meta_dir = dataset_dir / "meta"
    episodes_path = meta_dir / "episodes.jsonl"

    if not episodes_path.exists():
        raise FileNotFoundError(f"episodes.jsonl not found at {episodes_path}")

    # Read episode list
    episodes = []
    with open(episodes_path) as f:
        for line in f:
            episodes.append(json.loads(line))
    if max_episodes:
        episodes = episodes[:max_episodes]

    logger.info("Computing norm stats from %d episodes in %s", len(episodes), dataset_dir)

    # Discover numeric columns from first parquet
    ep0 = episodes[0]
    ep0_idx = ep0["episode_index"]
    chunk = ep0_idx // 1000
    sample_path = dataset_dir / f"data/chunk-{chunk:03d}/episode_{ep0_idx:06d}.parquet"
    sample_table = pq.read_table(sample_path)

    # Find columns that contain list-of-float data (state, action, etc.)
    numeric_columns = []
    for col_name in sample_table.column_names:
        col = sample_table.column(col_name)
        import pyarrow as pa
        if hasattr(col.type, "value_type") and col.type.value_type in (
            pa.float32(), pa.float64(),
        ):
            numeric_columns.append(col_name)

    logger.info("Numeric columns found: %s", numeric_columns)

    # Initialize running stats per column
    stats: dict[str, RunningStats] = {col: RunningStats() for col in numeric_columns}

    for i, ep in enumerate(episodes):
        ep_idx = ep["episode_index"]
        chunk = ep_idx // 1000
        pq_path = dataset_dir / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"

        try:
            table = pq.read_table(pq_path, columns=numeric_columns)
        except Exception as e:
            logger.warning("Skipping episode %d: %s", ep_idx, e)
            continue

        for col_name in numeric_columns:
            col = table.column(col_name)
            # Convert list-of-float column to numpy array [T, D]
            arr = np.array([row.as_py() for row in col], dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[:, None]
            stats[col_name].update(arr)

        if (i + 1) % 100 == 0:
            logger.info("Processed %d / %d episodes", i + 1, len(episodes))

    # Extract final stats
    result = {}
    for col_name, rs in stats.items():
        try:
            result[col_name] = rs.get_statistics()
            logger.info(
                "  %s: mean=%s, std=%s",
                col_name,
                np.array2string(result[col_name].mean[:4], precision=3),
                np.array2string(result[col_name].std[:4], precision=3),
            )
        except ValueError:
            logger.warning("  %s: not enough data, skipping", col_name)

    return result


def main():
    parser = argparse.ArgumentParser(description="Compute normalization statistics")
    parser.add_argument("--config-name", type=str, default=None, help="OpenPI config name (recommended)")
    parser.add_argument("--data-root", type=str, default=None, help="Root directory containing datasets")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Direct path to dataset directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="Direct path to save norm_stats.json")
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit number of episodes (direct mode)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples (config mode)")
    args = parser.parse_args()

    if args.config_name:
        compute_from_config(args.config_name, args.data_root, args.max_samples)
    elif args.dataset_dir and args.output_dir:
        norm_stats = compute_from_parquets(args.dataset_dir, args.max_episodes)
        save_norm_stats(args.output_dir, norm_stats)
        logger.info("Saved norm stats to %s", args.output_dir / "norm_stats.json")
    else:
        parser.error("Specify --config-name or both --dataset-dir and --output-dir")


if __name__ == "__main__":
    main()
