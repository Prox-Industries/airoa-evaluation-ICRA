# Reproduction Steps — Final Submission

## Submission Metadata

| Field | Value |
|---|---|
| Team name | Team 27 |
| Fork URL | https://github.com/Prox-Industries/airoa-evaluation-ICRA |
| Branch | `sample-openpi` |
| Commit hash | _filled in at submission time_ |
| Checkpoint path | `s3://airoa-icra-team-27/` (Cloudflare R2) |
| Endpoint | `https://eabeb2a5516ef53a191452e5714fc16b.r2.cloudflarestorage.com` |
| Policy config name | `pi05_hsr_micro_ft` |

The submitted checkpoint is the public Pi0.5 baseline (step 100K, Orbax/JAX,
distributed at `s3://airoa-icra-shared/baseline/100000/`) converted to
PyTorch and corrected with **rank-8 LoRA micro-FT (1000 steps)** that
suppresses a learned `base_t` outlier. The LoRA delta is folded into the
weights at submission time, so the served checkpoint is a plain PyTorch
state_dict (no LoRA wrap at inference).

## Checkpoint Layout (R2)

```
s3://airoa-icra-team-27/
├── model.safetensors                                                 (~6.8 GiB, bf16)
└── assets/
    └── lerobot_datasets/task6891011_level12_v2.5_train/
        └── norm_stats.json                                            (baseline-derived)
```

`norm_stats.json` is reused unchanged from the public baseline at
`s3://airoa-icra-shared/baseline/100000/assets/lerobot_datasets/task6891011_level12_v2.5_train/norm_stats.json`.

## Reproduction Commands

### 1. Clone and check out

```bash
git clone https://github.com/Prox-Industries/airoa-evaluation-ICRA.git
cd airoa-evaluation-ICRA
git checkout sample-openpi
git checkout <commit-hash>
```

### 2. Download the checkpoint from R2

```bash
export AWS_ACCESS_KEY_ID=<team-27 access key>
export AWS_SECRET_ACCESS_KEY=<team-27 secret>
export AWS_ENDPOINT_URL=https://eabeb2a5516ef53a191452e5714fc16b.r2.cloudflarestorage.com
export AWS_REGION=auto

aws s3 cp --recursive s3://airoa-icra-team-27/ ./checkpoint/ \
  --endpoint-url "$AWS_ENDPOINT_URL"
```

After download:

```
./checkpoint/
├── model.safetensors
└── assets/lerobot_datasets/task6891011_level12_v2.5_train/norm_stats.json
```

### 3. Set environment variables

```bash
export POLICY_CHECKPOINT_PATH=$(pwd)/checkpoint
export POLICY_CONFIG_NAME=pi05_hsr_micro_ft
export POLICY_PYTORCH_DEVICE=cuda
export TEST_MODE=true
```

### 4. Build and start the evaluation containers

```bash
./RUN-DOCKER-CONTAINER.sh up
```

The first inference call takes a few minutes due to `torch.compile` /
Triton autotuning on Blackwell. Subsequent calls run at roughly 2–5 Hz.

### 5. Run the test client

```bash
./RUN-DOCKER-CONTAINER.sh shell
# Inside the container:
roslaunch hsr_policy_client hsr_policy_client.launch
```

Successful smoke test indicator (per README):

```
Action executed.
```

### 6. Stop containers

```bash
./RUN-DOCKER-CONTAINER.sh down
```

## What Was Modified

### `src/openpi/training/config.py`

Added a new `TrainConfig` named **`pi05_hsr_micro_ft`** for inference. It
points at `asset_id="lerobot_datasets/task6891011_level12_v2.5_train"`
which matches the `norm_stats.json` shipped inside the submitted checkpoint
directory.

### `src/openpi/policies/hsr_policy.py`

`_encode_actions` now applies a **base-velocity guardrail** ("A1' clip")
before returning actions:

| Action dim | Bound |
|---|---|
| `base_x` (idx 8) | ±0.3 |
| `base_y` (idx 9) | ±0.3 |
| `base_t` (idx 10) | ±1.5 |

These bounds match the soft cap that the corrective micro-FT learned for
`base_t` and the envelope observed for `base_x/y` in training. The clip is
a defense-in-depth layer; the model itself almost never exceeds the bounds
after micro-FT, but the clip prevents rare-burst outliers from reaching the
robot.

### Other files

`src/openpi/models/model.py` already filters the tied `embed_tokens.weight`
key on load (this fix was already on `sample-openpi` and is needed for
PyTorch checkpoints converted from JAX). No additional changes there.

`server/`, `runtime_core/`, `RUN-DOCKER-CONTAINER.sh`, and
`docker-compose.yml` are **not modified**.

## Environment Variables (read by the pipeline)

| Variable | Required | Notes |
|---|---|---|
| `POLICY_CHECKPOINT_PATH` | yes | Absolute host path to checkpoint dir; mounted as `/policy_checkpoint`. |
| `POLICY_CONFIG_NAME` | yes | Use `pi05_hsr_micro_ft`. |
| `POLICY_PYTORCH_DEVICE` | optional | `cuda` recommended. |
| `TEST_MODE` | optional | `true` for the synthetic smoke test loop. |

No other extra env vars are required.

## Hardware / Software Notes

- Tested for the official evaluation environment: NVIDIA RTX 5070 Ti
  (Blackwell, `sm_120`), Ubuntu 24.04, Docker 29 + Compose v2,
  NVIDIA Container Toolkit, ≥16 GB VRAM.
- `server/Dockerfile` (unchanged on this branch) installs PyTorch nightly
  CUDA 12.8, which supports `sm_120`.

## Why the submission is not the unmodified baseline

Per Section 13.2 of the competition rules, submitting an unmodified
baseline is prohibited. The submitted checkpoint is the public baseline
weights **plus** a rank-8 LoRA correction trained for 1000 steps with an
axis-weighted Huber loss on the base axes (peak weight on `base_t`) and a
dim-wise keep-close regularizer toward the frozen baseline. The LoRA delta
is folded into the weights at submission time. The companion runtime
guardrail (A1' clip in `_encode_actions`) is the second line of defense
against rare-burst outliers.
