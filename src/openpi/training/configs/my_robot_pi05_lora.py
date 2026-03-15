"""Pi0.5 LoRA fine-tuning config for custom robot data in LeRobot format.

This config defines a lightweight LoRA fine-tuning setup:
  - Model:   pi05_base with LoRA on both PaliGemma (2B) and Action Expert (300M)
  - Data:    LeRobot format, single camera, minimal state/action dims
  - LoRA:    gemma_2b_lora (rank=16) + gemma_300m_lora (rank=32)
  - EMA:     disabled (required for LoRA)

To register this config, add to the _CONFIGS list in config.py:
    from openpi.training.configs.my_robot_pi05_lora import MY_ROBOT_PI05_LORA_CONFIGS
    _CONFIGS = [*_CONFIGS, *MY_ROBOT_PI05_LORA_CONFIGS]

Or use get_config() directly:
    from openpi.training.configs.my_robot_pi05_lora import get_my_robot_config
    config = get_my_robot_config()
"""
from __future__ import annotations

import dataclasses
from typing import override

import numpy as np
import pathlib

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.policies.libero_policy as libero_policy
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms


# ---------------------------------------------------------------------------
# Data config: adapt these to your robot
# ---------------------------------------------------------------------------

# Your robot's action dimension (joints + gripper).
# Must match what you wrote in convert_my_data_to_lerobot.py.
MY_ACTION_DIM = 7

# Number of future action steps to predict per inference call.
# 10 is a good default; increase for smoother trajectories.
MY_ACTION_HORIZON = 10


@dataclasses.dataclass(frozen=True)
class MyRobotDataConfig(_config.DataConfigFactory):
    """Minimal data config for a single-camera robot with LeRobot data.

    Assumes the LeRobot dataset has these keys (set in convert script):
      - observation.state:  [STATE_DIM] float32
      - action:             [ACTION_DIM] float32
      - task_index:         int  (mapped to prompt via tasks.jsonl)

    The repack transform maps LeRobot keys -> OpenPI internal keys.
    """

    default_prompt: str | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        # Repack: map dataset column names to the keys that transforms expect.
        # Left side = key the transforms will see, right side = key in parquet.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                        # If you have a camera image column in parquet, map it here:
                        # "observation/image": "observation.image.top",
                    }
                )
            ]
        )

        # Data transforms: convert raw obs -> model input format.
        # Uses the Libero policy transforms as a simple single-arm template.
        # Replace with your own Inputs/Outputs class if needed.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # Model transforms: tokenize prompt, resize images, pad state/actions.
        model_transforms = _config.ModelTransformFactory(
            default_prompt=self.default_prompt,
        )(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

def _make_model_config() -> pi0_config.Pi0Config:
    """Pi0.5 model config with LoRA variants for memory-efficient fine-tuning."""
    return pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",      # rank=16, alpha=16
        action_expert_variant="gemma_300m_lora", # rank=32, alpha=32
        action_dim=32,                           # internal model dim (padded from MY_ACTION_DIM)
        action_horizon=MY_ACTION_HORIZON,
    )


def _make_freeze_filter():
    """Freeze everything except LoRA params and projection heads."""
    return _make_model_config().get_freeze_filter()


# ---------------------------------------------------------------------------
# Train configs
# ---------------------------------------------------------------------------

def get_my_robot_config(
    repo_id: str = "my_org/my_robot_data",
    assets_dir: str = "./assets/my_robot_pi05_lora",
    num_train_steps: int = 5_000,
    batch_size: int = 16,
) -> _config.TrainConfig:
    """Build a TrainConfig programmatically. Useful for notebooks/scripts."""
    return _config.TrainConfig(
        name="my_robot_pi05_lora",
        model=_make_model_config(),
        data=MyRobotDataConfig(
            repo_id=repo_id,
            assets=_config.AssetsConfig(
                assets_dir=assets_dir,
                asset_id=repo_id,
            ),
            base_config=_config.DataConfig(prompt_from_task=True),
            default_prompt="do something",
        ),
        # Load pi05 base weights. Change to local path if needed.
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # Freeze all base weights; only LoRA adapters + projections are trainable.
        freeze_filter=_make_freeze_filter(),
        # EMA must be None for LoRA fine-tuning.
        ema_decay=None,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=5e-5,
            decay_steps=num_train_steps,
            decay_lr=5e-6,
        ),
        batch_size=batch_size,
        num_workers=4,
        num_train_steps=num_train_steps,
        save_interval=500,
        log_interval=50,
    )


# Smoke test config: 100 steps, batch_size=4
_SMOKE_TEST_CONFIG = _config.TrainConfig(
    name="my_robot_pi05_lora_smoke",
    model=_make_model_config(),
    data=MyRobotDataConfig(
        repo_id="my_org/my_robot_data",
        assets=_config.AssetsConfig(
            assets_dir="./assets/my_robot_pi05_lora",
            asset_id="my_org/my_robot_data",
        ),
        base_config=_config.DataConfig(prompt_from_task=True),
        default_prompt="do something",
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    freeze_filter=_make_freeze_filter(),
    ema_decay=None,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=10,
        peak_lr=5e-5,
        decay_steps=100,
        decay_lr=5e-6,
    ),
    batch_size=4,
    num_workers=0,
    num_train_steps=100,
    save_interval=50,
    log_interval=10,
    overwrite=True,
)

# Production config: 5000 steps
_PRODUCTION_CONFIG = get_my_robot_config()

# Export for registration in config.py
MY_ROBOT_PI05_LORA_CONFIGS = [_SMOKE_TEST_CONFIG, _PRODUCTION_CONFIG]
