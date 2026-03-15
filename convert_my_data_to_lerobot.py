#!/usr/bin/env python3
"""Convert custom robot data to LeRobot v2.1 format for OpenPI fine-tuning.

This script converts your own robot dataset into the LeRobot directory structure
that OpenPI expects. Adapt the `_load_episode()` function to match your raw data format.

Usage:
    python convert_my_data_to_lerobot.py \
        --raw-dir /path/to/raw/episodes \
        --output-dir ./my_lerobot_dataset \
        --repo-id my_org/my_robot_data \
        --fps 10 \
        --task-name "pick up the cup"

Output structure:
    my_lerobot_dataset/
    ├── meta/
    │   ├── info.json
    │   ├── tasks.jsonl
    │   └── episodes.jsonl
    └── data/
        └── chunk-000/
            └── episode_000000.parquet
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Adapt this section to your raw data format
# ---------------------------------------------------------------------------

# Define your robot's observation/action dimensions here.
STATE_DIM = 7       # e.g. 6 joint angles + 1 gripper
ACTION_DIM = 7      # same as state for simple setups
IMAGE_KEYS = [      # camera names; set to [] if no images
    # "observation.image.top",
]
FPS = 10


def _load_episode(episode_dir: Path) -> dict:
    """Load one episode from your raw data format.

    Returns a dict with:
        "states":  np.ndarray [T, STATE_DIM]  float32
        "actions": np.ndarray [T, ACTION_DIM] float32
        "images":  dict[str, list[np.ndarray]]  (optional) key -> list of [H,W,3] uint8
        "task":    str  (language instruction for this episode)

    Replace this function with your own data loading logic.
    """
    # --- Example: load from .npz ---
    # data = np.load(episode_dir / "episode.npz")
    # return {
    #     "states": data["states"].astype(np.float32),
    #     "actions": data["actions"].astype(np.float32),
    #     "task": "pick up the cup",
    # }

    # --- Dummy data for smoke test ---
    T = 50  # number of timesteps
    return {
        "states": np.random.randn(T, STATE_DIM).astype(np.float32),
        "actions": np.random.randn(T, ACTION_DIM).astype(np.float32),
        "task": "do something",
    }


# ---------------------------------------------------------------------------
# Conversion logic (generally no need to modify below)
# ---------------------------------------------------------------------------

def _make_parquet_table(
    episode_data: dict,
    episode_index: int,
    task_index: int,
    global_frame_offset: int,
    fps: float,
) -> pa.Table:
    """Build a PyArrow table for one episode in LeRobot v2.1 format."""
    states = episode_data["states"]
    actions = episode_data["actions"]
    T = len(states)

    columns = {
        "episode_index": pa.array([episode_index] * T, type=pa.int64()),
        "frame_index": pa.array(list(range(T)), type=pa.int64()),
        "timestamp": pa.array([i / fps for i in range(T)], type=pa.float32()),
        "index": pa.array(list(range(global_frame_offset, global_frame_offset + T)), type=pa.int64()),
        "task_index": pa.array([task_index] * T, type=pa.int64()),
        "next.done": pa.array([False] * (T - 1) + [True], type=pa.bool_()),
    }

    # State as list-of-floats per row
    columns["observation.state"] = pa.array(
        [row.tolist() for row in states], type=pa.list_(pa.float32())
    )

    # Action as list-of-floats per row
    columns["action"] = pa.array(
        [row.tolist() for row in actions], type=pa.list_(pa.float32())
    )

    return pa.table(columns)


def convert(
    raw_dir: Path,
    output_dir: Path,
    repo_id: str,
    fps: float,
    task_name: str | None,
    max_episodes: int | None,
):
    data_dir = output_dir / "data"
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Discover episodes
    if raw_dir.is_dir():
        episode_sources = sorted(raw_dir.iterdir())
    else:
        episode_sources = [raw_dir]

    if max_episodes:
        episode_sources = episode_sources[:max_episodes]

    # Task mapping
    tasks: dict[str, int] = {}
    episodes_meta: list[dict] = []
    global_frame_offset = 0
    total_frames = 0
    chunk_size = 1000

    for ep_idx, ep_source in enumerate(episode_sources):
        print(f"Converting episode {ep_idx}: {ep_source.name}")

        episode_data = _load_episode(ep_source)
        task = task_name or episode_data.get("task", "default_task")

        if task not in tasks:
            tasks[task] = len(tasks)
        task_idx = tasks[task]

        T = len(episode_data["states"])

        # Write parquet
        chunk = ep_idx // chunk_size
        chunk_dir = data_dir / f"chunk-{chunk:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        table = _make_parquet_table(
            episode_data, ep_idx, task_idx, global_frame_offset, fps
        )
        pq.write_table(table, chunk_dir / f"episode_{ep_idx:06d}.parquet")

        # Episode metadata
        episodes_meta.append({
            "episode_index": ep_idx,
            "tasks": [task],
            "length": T,
        })

        global_frame_offset += T
        total_frames += T

    # Write meta/info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type": "my_robot",
        "total_episodes": len(episodes_meta),
        "total_frames": total_frames,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes_meta)}"},
        "data_path": f"data/chunk-{{episode_chunk:03d}}/episode_{{episode_index:06d}}.parquet",
        "video_path": None,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [STATE_DIM],
                "names": None,
            },
            "action": {
                "dtype": "float32",
                "shape": [ACTION_DIM],
                "names": None,
            },
        },
        "repo_id": repo_id,
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

    # Write meta/tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task_str, task_idx in tasks.items():
            f.write(json.dumps({"task_index": task_idx, "task": task_str}) + "\n")

    # Write meta/episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    print(f"\nDone! {len(episodes_meta)} episodes, {total_frames} frames")
    print(f"Output: {output_dir}")
    print(f"Repo ID: {repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Convert custom data to LeRobot format")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory containing raw episode data")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output LeRobot dataset directory")
    parser.add_argument("--repo-id", type=str, default="my_org/my_robot_data", help="LeRobot repo ID")
    parser.add_argument("--fps", type=float, default=FPS, help="Dataset FPS")
    parser.add_argument("--task-name", type=str, default=None, help="Override task name for all episodes")
    parser.add_argument("--max-episodes", type=int, default=None, help="Max episodes to convert")
    args = parser.parse_args()

    convert(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        fps=args.fps,
        task_name=args.task_name,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
