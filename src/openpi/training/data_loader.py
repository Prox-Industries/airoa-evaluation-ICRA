from collections.abc import Iterator, Sequence
import json
import logging
import multiprocessing
import os
import shutil
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import lerobot.common.datasets.utils as _lerobot_utils
import numpy as np

# LeRobot が HuggingFace Hub で revision を検証しようとするのをパッチで無効化
# ローカルデータセット使用時に不要なネットワーク接続を防ぐ
_lerobot_utils.get_safe_version = lambda repo_id, revision: revision or "main"
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def _build_filtered_meta_root(
    source_root: str,
    repo_id: str,
    task_names: list[str] | None,
    success_only: bool,
    max_episodes_per_task: int | None,
    cache_dir: str = "/opt/dlami/nvme/filtered_meta",
) -> str:
    """episodes.jsonl を早期終了スキャンで絞り込み、ローカルSSDに小さな meta を作成する。

    data/ と videos/ はシンボリックリンクで R2 マウントを参照するため、
    LeRobot は高速なローカル meta + R2 経由のデータ読み込みを行う。

    Returns:
        フィルタ済み meta を含むローカルディレクトリのパス（repo_id の親）
    """
    dataset_source = os.path.join(source_root, repo_id)
    # キャッシュキーを設定から生成
    key_parts = [
        repo_id.replace("/", "_"),
        f"tasks={'_'.join(sorted(task_names or []))[:40]}",
        f"success={success_only}",
        f"max={max_episodes_per_task}",
    ]
    cache_key = "_".join(key_parts)
    local_dataset = os.path.join(cache_dir, cache_key, repo_id)
    local_meta = os.path.join(local_dataset, "meta")
    done_flag = os.path.join(local_meta, ".done")

    if os.path.exists(done_flag):
        logging.info("フィルタ済み meta キャッシュを再利用: %s", local_meta)
        return os.path.join(cache_dir, cache_key)

    logging.info("フィルタ済み meta を作成中: %s", local_meta)
    os.makedirs(local_meta, exist_ok=True)

    # info.json / tasks.jsonl をコピー（小さいファイルのみ、episodes_stats.jsonlは42GBのため除外）
    for fname in ("info.json", "tasks.jsonl"):
        src = os.path.join(dataset_source, "meta", fname)
        dst = os.path.join(local_meta, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)

    # episodes.jsonl を早期終了スキャンで絞り込む
    episodes_src = os.path.join(dataset_source, "meta", "episodes.jsonl")
    count_per_task: dict[str, int] = {}
    selected_lines: list[str] = []

    with open(episodes_src) as f:
        for line in f:
            ep = json.loads(line)

            # success フィルタ
            if success_only and not ep.get("task_success", True):
                continue

            # task name フィルタ（short_horizon_task フィールドで照合）
            sht = ep.get("short_horizon_task", "")
            ep_tasks: list[str] = ep.get("tasks", [])
            if task_names:
                # short_horizon_task が task_names のいずれかに完全一致するか確認
                task_key = next((tn for tn in task_names if tn.lower() == sht.lower()), None)
                if task_key is None:
                    continue
            else:
                task_key = sht if sht else (ep_tasks[0] if ep_tasks else "__all__")

            # per-task 上限
            if max_episodes_per_task is not None:
                if count_per_task.get(task_key, 0) >= max_episodes_per_task:
                    continue
                count_per_task[task_key] = count_per_task.get(task_key, 0) + 1

            selected_lines.append(line if line.endswith("\n") else line + "\n")

            # 全タスクが上限に達したら早期終了
            if (
                max_episodes_per_task is not None
                and task_names is not None
                and all(count_per_task.get(t, 0) >= max_episodes_per_task for t in task_names)
            ):
                break

    if not selected_lines:
        raise ValueError(
            f"フィルタ条件に一致するエピソードがありません: "
            f"task_names={task_names}, success_only={success_only}"
        )

    with open(os.path.join(local_meta, "episodes.jsonl"), "w") as f:
        f.writelines(selected_lines)

    logging.info(
        "episodes.jsonl フィルタ完了: %d エピソード選択 %s",
        len(selected_lines), dict(count_per_task),
    )

    # data/ と videos/ はシンボリックリンクで R2 マウントを参照
    for dname in ("data", "videos"):
        link = os.path.join(local_dataset, dname)
        target = os.path.join(dataset_source, dname)
        if not os.path.exists(link) and os.path.exists(target):
            os.symlink(target, link)

    # 完了フラグ
    open(done_flag, "w").close()
    return os.path.join(cache_dir, cache_key)


class DirectParquetDataset(torch.utils.data.Dataset):
    """Lightweight dataset that reads parquet + video directly from S3 mount.

    Bypasses LeRobotDataset's heavy initialization which tries to scan all 2.5M
    episodes (140GB metadata). Instead, reads only the filtered episodes' parquet
    files (24KB each) and decodes video frames (243KB each) on demand.
    """

    def __init__(
        self,
        data_root: str,
        repo_id: str,
        episodes: list[dict],
        tasks: dict[int, str],
        action_horizon: int,
        fps: float,
        action_sequence_keys: tuple[str, ...],
    ):
        self.dataset_dir = os.path.join(data_root, repo_id)
        self.action_horizon = action_horizon
        self.fps = fps
        self.action_sequence_keys = action_sequence_keys
        self.tasks = tasks

        # Read all parquet data upfront (150 episodes × ~24KB = ~3.6MB on disk).
        # Decompressed in memory this is still tiny.
        self._episode_tables: dict[int, "pyarrow.Table"] = {}
        self._frames: list[tuple[int, int]] = []  # (episode_index, frame_index)

        import pyarrow.parquet as pq

        for ep in episodes:
            ep_idx = ep["episode_index"]
            pq_path = self._parquet_path(ep_idx)
            try:
                table = pq.read_table(pq_path)
            except Exception as e:
                logging.warning("Skipping episode %d: %s", ep_idx, e)
                continue

            num_frames = table.num_rows
            self._episode_tables[ep_idx] = table

            # Only include frames where full action horizon is available
            valid = max(0, num_frames - action_horizon + 1)
            for f in range(valid):
                self._frames.append((ep_idx, f))

        if not self._frames:
            raise ValueError("No valid frames found in any episode")

        logging.info(
            "DirectParquetDataset: %d episodes, %d frames",
            len(self._episode_tables), len(self._frames),
        )

    def _parquet_path(self, ep_idx: int) -> str:
        chunk = ep_idx // 1000
        return os.path.join(
            self.dataset_dir,
            f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet",
        )

    def _video_path(self, ep_idx: int, camera: str) -> str:
        chunk = ep_idx // 1000
        return os.path.join(
            self.dataset_dir,
            f"videos/chunk-{chunk:03d}/observation.image.{camera}/episode_{ep_idx:06d}.mp4",
        )

    def _decode_frame(self, ep_idx: int, camera: str, frame_idx: int) -> np.ndarray:
        """Decode a single frame from video. Returns uint8 [H, W, 3]."""
        import av

        path = self._video_path(ep_idx, camera)
        with av.open(path) as container:
            stream = container.streams.video[0]
            # Videos are small (~243KB, ~142 frames) so sequential decode is fast
            for i, frame in enumerate(container.decode(stream)):
                if i == frame_idx:
                    return frame.to_ndarray(format="rgb24")

        raise RuntimeError(f"Could not decode frame {frame_idx} from {path}")

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, index) -> dict:
        ep_idx, frame_idx = self._frames[index]
        table = self._episode_tables[ep_idx]

        # --- State ---
        state = np.array(
            table.column("observation.state")[frame_idx].as_py(),
            dtype=np.float32,
        )

        # --- Actions (stack action_horizon frames) ---
        result: dict = {
            "observation.state": state,
        }
        for key in self.action_sequence_keys:
            col = table.column(key)
            chunk = np.stack(
                [
                    np.array(col[frame_idx + t].as_py(), dtype=np.float32)
                    for t in range(self.action_horizon)
                ]
            )
            result[key] = chunk

        # --- Images (decode from video) ---
        result["observation.image.head"] = self._decode_frame(ep_idx, "head", frame_idx)
        result["observation.image.hand"] = self._decode_frame(ep_idx, "hand", frame_idx)

        # --- Prompt from task_index ---
        task_index = int(table.column("task_index")[frame_idx].as_py())
        if self.tasks:
            prompt = self.tasks.get(task_index, f"task_{task_index}")
            result["prompt"] = prompt
        result["task_index"] = task_index

        return result


def _load_tasks_jsonl(meta_dir: str) -> dict[int, str]:
    """Load tasks.jsonl and return {task_index: task_string} mapping."""
    tasks_path = os.path.join(meta_dir, "tasks.jsonl")
    tasks = {}
    if os.path.exists(tasks_path):
        with open(tasks_path) as f:
            for line in f:
                obj = json.loads(line)
                tasks[obj["task_index"]] = obj["task"]
    return tasks


def _filter_episodes(
    source_root: str,
    repo_id: str,
    task_names: list[str] | None,
    success_only: bool,
    max_episodes_per_task: int | None,
) -> list[dict]:
    """Scan episodes.jsonl and return filtered episode dicts.

    Uses early termination when all tasks hit max_episodes_per_task.
    """
    episodes_path = os.path.join(source_root, repo_id, "meta", "episodes.jsonl")
    count_per_task: dict[str, int] = {}
    selected: list[dict] = []

    logging.info("Scanning episodes.jsonl (early-exit filtering)...")
    with open(episodes_path) as f:
        for line in f:
            ep = json.loads(line)

            if success_only and not ep.get("task_success", True):
                continue

            sht = ep.get("short_horizon_task", "")
            if task_names:
                task_key = next(
                    (tn for tn in task_names if tn.lower() == sht.lower()), None
                )
                if task_key is None:
                    continue
            else:
                task_key = sht or "__all__"

            if max_episodes_per_task is not None:
                if count_per_task.get(task_key, 0) >= max_episodes_per_task:
                    continue
                count_per_task[task_key] = count_per_task.get(task_key, 0) + 1

            selected.append(ep)

            # Early exit when all tasks are full
            if (
                max_episodes_per_task is not None
                and task_names is not None
                and all(
                    count_per_task.get(t, 0) >= max_episodes_per_task
                    for t in task_names
                )
            ):
                break

    logging.info("Filtered episodes: %d selected %s", len(selected), dict(count_per_task))
    if not selected:
        raise ValueError(
            f"No episodes match filter: task_names={task_names}, success_only={success_only}"
        )
    return selected


def create_direct_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    *,
    local_root: str,
    task_names: list[str] | None = None,
    success_only: bool = False,
    max_episodes_per_task: int | None = None,
) -> "DirectParquetDataset":
    """Create a DirectParquetDataset that bypasses LeRobotDataset.

    Reads only the needed parquet/video files from S3 mount, avoiding the
    140GB metadata scan that LeRobotDataset performs on initialization.
    """
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set.")

    # Filter episodes
    episodes = _filter_episodes(
        source_root=local_root,
        repo_id=repo_id,
        task_names=task_names,
        success_only=success_only,
        max_episodes_per_task=max_episodes_per_task,
    )

    # Load task name mapping
    meta_dir = os.path.join(local_root, repo_id, "meta")
    tasks = _load_tasks_jsonl(meta_dir)

    # Read FPS from info.json
    info_path = os.path.join(meta_dir, "info.json")
    with open(info_path) as f:
        info = json.loads(f.read())
    fps = info.get("fps", 10)

    return DirectParquetDataset(
        data_root=local_root,
        repo_id=repo_id,
        episodes=episodes,
        tasks=tasks,
        action_horizon=action_horizon,
        fps=fps,
        action_sequence_keys=data_config.action_sequence_keys,
    )


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    local_root: str | None = None,
    task_names: list[str] | None = None,
    success_only: bool = False,
    max_episodes_per_task: int | None = None,
) -> Dataset:
    """Create a dataset for training.

    Args:
        local_root: If set, load the dataset from this local directory instead of HuggingFace Hub.
                    The dataset is expected to be at ``<local_root>/<repo_id>/``.
        task_names: If set, only episodes whose task description contains one of these strings
                    (case-insensitive substring match) will be included.
        success_only: If True, only include episodes where task_success is True.
        max_episodes_per_task: If set, cap the number of episodes per matched task.
    """
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    # フィルタ条件がある場合: 早期終了スキャンでローカルに小さな meta を作成し
    # LeRobot が 2.5M 行の episodes.jsonl を全件読まないようにする
    if local_root and (task_names or success_only or max_episodes_per_task is not None):
        local_root = _build_filtered_meta_root(
            source_root=local_root,
            repo_id=repo_id,
            task_names=task_names,
            success_only=success_only,
            max_episodes_per_task=max_episodes_per_task,
        )

    root = os.path.join(local_root, repo_id) if local_root else None
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
    episodes = None

    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        tolerance_s=1e-3, # default is 1e-4
        root=root,
        episodes=episodes,
        # video_backend="pyav",
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
