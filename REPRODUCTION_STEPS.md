# Reproduction Steps — Final Submission

## Submission Metadata

| Field | Value |
|---|---|
| Team name | Team 27 |
| Fork URL | https://github.com/Prox-Industries/airoa-evaluation-ICRA |
| Branch | `sample-openpi` |
| Commit hash | branch tip of `sample-openpi` (pinned in the submission email) |
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

## Quick reproduction (copy-paste)

A single-shell-flow reproduction matching the format requested by the
organizers. Replace the two `<team-27 ...>` placeholders with the R2
keys delivered out-of-band, then paste the rest verbatim.

```bash
# clone + checkout
git clone https://github.com/Prox-Industries/airoa-evaluation-ICRA.git
cd airoa-evaluation-ICRA
git checkout sample-openpi

# download checkpoint from Cloudflare R2 (S3-compatible)
export AWS_ACCESS_KEY_ID=<team-27 access key>
export AWS_SECRET_ACCESS_KEY=<team-27 secret>
export AWS_ENDPOINT_URL=https://eabeb2a5516ef53a191452e5714fc16b.r2.cloudflarestorage.com
export AWS_REGION=auto
mkdir -p checkpoint
aws s3 cp --recursive s3://airoa-icra-team-27/ ./checkpoint/ \
  --endpoint-url "$AWS_ENDPOINT_URL"

# required runtime env
export POLICY_CHECKPOINT_PATH=$(pwd)/checkpoint
export POLICY_CONFIG_NAME=pi05_hsr_micro_ft
export POLICY_PYTORCH_DEVICE=cuda
export TEST_MODE=true

# bring up evaluation containers (build + start)
./RUN-DOCKER-CONTAINER.sh up

# open the client shell and launch the test client
./RUN-DOCKER-CONTAINER.sh shell
# inside the client container:
roslaunch hsr_policy_client hsr_policy_client.launch
```

Expected log: `Action executed.` (smoke-test loop). The first call takes
several minutes due to `torch.compile` / Triton autotuning on Blackwell;
subsequent calls run at roughly 2–5 Hz.

The same flow is documented in detail below.

## Reproduction Commands

### 1. Clone and check out

```bash
git clone https://github.com/Prox-Industries/airoa-evaluation-ICRA.git
cd airoa-evaluation-ICRA
git checkout sample-openpi
# Optionally pin to the exact submission commit (see submission email):
# git checkout <commit-hash-from-email>
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

### `server/ood_recovery.py` (new)

A blocking OOD-detection-and-retry wrapper around the policy. On every
inference call:

1. Run inference with the user prompt.
2. Run inference with the **counterfactual** prompt (Pick<->Place swap of
   the same object family — exhaustive 3-pair set verified across the
   3,971 task6911 parquets).
3. If the gripper-dim disagreement between the two predictions exceeds a
   threshold (and the predictions land on opposite sides of the 0.5
   open/close switch), the frame is flagged OOD.
4. On OOD, **discard the prediction and re-run inference** with a fresh
   flow-matching noise draw. Loop until either the result clears or
   `OOD_MAX_RETRIES` is exhausted.
5. If still OOD after the budget, return the last retry result
   (`OOD_PERSISTENT_ACTION=keep_last`, default).

The wrapper is transparent passthrough when `OOD_ENABLED=false`.

### `server/serve_hsr_policy_ws.py`

Constructs `OODRecoveryConfig` from the environment and wraps the loaded
policy with `OODRecoveryPolicy`. Otherwise unchanged from `sample-openpi`.

### `server/Dockerfile`

Adds default `OOD_*` environment variables so the wrapper is on by
default in the evaluation image. Defaults:

| Variable | Default |
|---|---|
| `OOD_ENABLED` | 1 |
| `OOD_AMBIGUITY_THRESHOLD` | 1.15 |
| `OOD_CHUNK_AGGREGATION` | max |
| `OOD_REQUIRE_CROSSING` | 1 |
| `OOD_RETRY_ON_OOD` | 1 |
| `OOD_MAX_RETRIES` | 3 |
| `OOD_PERSISTENT_ACTION` | keep_last |
| `OOD_MIN_CONSECUTIVE` | 1 |
| `OOD_PA_STRICT` | 1 |

All overridable via `docker run -e OOD_<KEY>=<VALUE>` if a different
behaviour is desired during evaluation.

### Other files

`src/openpi/models/model.py` already filters the tied `embed_tokens.weight`
key on load (this fix was already on `sample-openpi` and is needed for
PyTorch checkpoints converted from JAX). No additional changes there.

### Files explicitly NOT modified

The following files / directories are kept exactly as upstream
`airoa-org/airoa-evaluation-ICRA` `sample-openpi`:

- `runtime_core/**` — WebSocket protocol, server harness
- `deploy/hsr_policy_client/**` — pipeline-managed client logic
- `packages/policy-client/**` — protocol package
- `RUN-DOCKER-CONTAINER.sh` — harness entrypoint
- `docker-compose.yml` — service composition
- `client/Dockerfile` — client image (HSR ROS dependencies)

Verifiable with `git diff upstream/sample-openpi..origin/sample-openpi -- <path>`
returning empty for each path above (where `upstream` points at
`airoa-org/airoa-evaluation-ICRA`).

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
baseline is prohibited. The submission is modified in three independent
layers:

1. **Weights**: the public baseline 100K weights + a rank-8 LoRA
   correction trained for 1000 steps with an axis-weighted Huber loss on
   the base axes (peak weight on `base_t`) and a dim-wise keep-close
   regulariser toward the frozen baseline. The LoRA delta is folded into
   the weights at submission time.
2. **Action post-processing**: the A1' clip in `_encode_actions`
   (`base_x/y ±0.3`, `base_t ±1.5`) — second line of defence against
   rare-burst outliers in the base-velocity axes.
3. **Inference-time control flow**: `server/ood_recovery.py` wraps the
   policy with a blocking discard-and-retry loop on counterfactually
   ambiguous frames (Pick<->Place gripper inversion).
