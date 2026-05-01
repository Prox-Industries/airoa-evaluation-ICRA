# REPRODUCTION_STEPS — Team 27 (Prox Industries)

> Submission note for the ICRA 2026 AIRoA / VLA Pipeline Competition.
> Structured to follow `docs/REPRODUCTION_STEPS.template.md` from
> `airoa-org/airoa-evaluation-ICRA` so the evaluator can run it
> literally.

---

## 1. Overview

| Item | Value |
|---|---|
| Model summary | Pi0.5 baseline 100K + rank-8 LoRA corrective micro-FT (1000 steps) folded into the weights, with an A1' base-velocity clip and a frame-by-frame OOD recovery (gripper-safe hold + forced release after 3 consecutive holds) |
| Framework | OpenPI (PI0Pytorch) |
| Repository | `https://github.com/Prox-Industries/airoa-evaluation-ICRA` |
| Branch | `sample-openpi` |
| Commit hash | branch tip of `sample-openpi` (the exact 40-character hash is included in the submission email) |
| Checkpoint S3 path | `s3://airoa-icra-team-27/` (Cloudflare R2, S3-compatible) |
| Expected VRAM | ≈ 11 GB during inference (PI0Pytorch in bf16 + flow-matching activations) |

---

## 2. Prerequisites

- NVIDIA GPU with ≥ 16 GB VRAM (**Blackwell-compatible**: RTX 5070 Ti, compute 12.0)
- Docker Engine + Docker Compose v2
- NVIDIA Container Toolkit
- AWS CLI v2 (used to download the checkpoint from Cloudflare R2)
- External credentials required: **team-27 R2 access key + secret**, issued together with the team-specific bucket `s3://airoa-icra-team-27/`. **No `HF_TOKEN`, no other gated resource.**

The container image fetches PyTorch nightly (CUDA 12.8) at build time, so
the build host needs internet access; the running container does not.

---

## 3. Reproduction Steps

### 3.1 Clone and checkout

```bash
git clone https://github.com/Prox-Industries/airoa-evaluation-ICRA.git
cd airoa-evaluation-ICRA
git checkout sample-openpi
git rev-parse HEAD  # should match the commit hash given in the submission email
```

### 3.2 Download checkpoint

```bash
export AWS_ACCESS_KEY_ID=<team-27 access key>
export AWS_SECRET_ACCESS_KEY=<team-27 secret>
export AWS_ENDPOINT_URL=https://eabeb2a5516ef53a191452e5714fc16b.r2.cloudflarestorage.com
export AWS_REGION=auto

mkdir -p checkpoint
aws s3 cp --recursive s3://airoa-icra-team-27/ ./checkpoint/ \
    --endpoint-url "$AWS_ENDPOINT_URL"
```

### 3.3 Environment variables

The harness reads the three documented variables. Set them as below.

```bash
export POLICY_CHECKPOINT_PATH=$(pwd)/checkpoint        # MUST be a directory
export POLICY_CONFIG_NAME=pi05_hsr_micro_ft            # OpenPI loader config
export POLICY_PYTORCH_DEVICE=cuda                      # optional; default cuda
```

Optional ICRA-specific variables (consumed by `server/ood_recovery.py`).
**Already given safe defaults in `server/Dockerfile` so leaving them
unset is fine.**

```bash
# Defaults baked into server/Dockerfile (override only if you need to):
#   OOD_ENABLED=1
#   OOD_AMBIGUITY_THRESHOLD=1.15
#   OOD_CHUNK_AGGREGATION=max
#   OOD_REQUIRE_CROSSING=1
#   OOD_MAX_HOLDS=3
#   OOD_PA_STRICT=1
```

For the smoke-test loop (synthetic observations on `localhost`):

```bash
export TEST_MODE=true
```

### 3.4 Start containers

```bash
./RUN-DOCKER-CONTAINER.sh up
```

The build pulls a CUDA 12.8 base image and PyTorch nightly; first build
takes ~15–20 min depending on bandwidth. Subsequent runs reuse the image.

### 3.5 Verify

```bash
# Wait for the policy server to be ready (returns "OK" once the model is loaded):
until curl -s http://localhost:8000/healthz 2>/dev/null | grep -q OK; do sleep 5; done
echo "READY"

# GPU usage sanity check:
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# Run the test client (TEST_MODE=true generates synthetic observations):
./RUN-DOCKER-CONTAINER.sh shell
# inside the client container:
roslaunch hsr_policy_client hsr_policy_client.launch
# expect: "Action executed." printed in a loop
```

After `./RUN-DOCKER-CONTAINER.sh up`, the policy server reaches the
"server listening on 0.0.0.0:8000" log line ~90 seconds later (CUDA
init + safetensors load on Blackwell). Once the WebSocket connection
opens, the first `infer()` call returns in **under 1 second** on an
L40S host (verified — see §7); we expect comparable or better on the
evaluation RTX 5070 Ti. Subsequent calls run at ≈ 2–5 Hz.

### 3.6 Stop

```bash
./RUN-DOCKER-CONTAINER.sh down
```

---

## 4. Files modified relative to the base repo

| Path | Reason |
|---|---|
| `server/serve_hsr_policy_ws.py` | Wrap the loaded policy with `OODRecoveryPolicy` (env-var configured, transparent passthrough when `OOD_ENABLED=0`). |
| `server/ood_recovery.py` *(new)* | Blocking discard-and-retry loop driven by Pick↔Place counterfactual prompt sensitivity at the gripper dimension. |
| `server/Dockerfile` | Add safe ICRA defaults for the `OOD_*` environment variables (image still functions identically with all of them unset / disabled). |
| `src/openpi/training/config.py` | Add a new `TrainConfig` named `pi05_hsr_micro_ft` whose `asset_id` matches the `norm_stats.json` shipped in the submitted checkpoint directory. |
| `src/openpi/policies/hsr_policy.py` | Apply A1' clip in `_encode_actions`: `base_x/y` to `±0.3`, `base_t` to `±1.5`. Defence-in-depth on top of the soft cap that micro-FT learned for `base_t`. |
| `src/openpi/models/model.py` | `load_pytorch` now uses `strict=False` so that tied `embed_tokens.weight` (deduplicated by the JAX→PT converter) does not break loading. HF transformers re-ties on construction. |
| `REPRODUCTION_STEPS.md` *(this file)* | Submission note. |

**Files explicitly not modified** (kept exactly as upstream
`airoa-org/airoa-evaluation-ICRA` `sample-openpi`):

- `runtime_core/**` — WebSocket protocol / server harness
- `deploy/hsr_policy_client/**` — pipeline-managed client logic
- `packages/policy-client/**` — protocol package
- `RUN-DOCKER-CONTAINER.sh` — harness entrypoint
- `docker-compose.yml` — service composition
- `client/Dockerfile` — client image (HSR ROS dependencies)
- `server/entrypoint.sh` — env → CLI mapping (default behaviour preserved)

**Pre-existing fork state, not part of this submission's edits.**
The `Prox-Industries/airoa-evaluation-ICRA` fork carries some additional
content from earlier history that pre-dates this submission and is not
touched by the commits behind this run. Listed here for full audit
transparency:

| Path | Inference-time use | Note |
|---|---|---|
| `src/openpi/models_pytorch/pi0_pytorch.py` | yes | Fork-level adjustments to the `PI0Pytorch` module that are part of the inference path. |
| `src/openpi/transforms.py` | yes | Fork-level adjustments to the transform stack used at inference. |
| `src/openpi/training/data_loader.py` | no | Training-time only. |
| `src/openpi/training/train.py` | no | Training-time only. |
| `src/openpi/training/configs/my_robot_pi05_lora.py` | no | LoRA fine-tuning training config (not invoked at inference). |
| `convert_my_data_to_lerobot.py`, `scripts/compute_norm_stats.py`, `run_train_my_robot_pi05_lora.sh`, `README_my_robot_pi05_lora.md` | no | LoRA fine-tuning helpers (data conversion + training shell scripts). |
| `.docker_cache/policy_cache/big_vision/paligemma_tokenizer.model` | yes | Bundled PaliGemma tokenizer (~4 MB), used at inference; see §5 for details. |
| `.gitignore` | no | Repo metadata only. |

These files are not modified by this submission's commits and are part
of the fork's existing default branch. The OpenPI inference loader path
exercised by `POLICY_CONFIG_NAME=pi05_hsr_micro_ft` does not invoke any
of the training-only files above.

---

## 5. Important notes

- **Server warm-up timing.** After containers are up, the policy server
  takes ~90 seconds to print `server listening on 0.0.0.0:8000` (CUDA
  init + ~6.8 GB safetensors load + transformers patch). Once the
  WebSocket connection opens, the first `infer()` call returns in
  **under 1 second** on an L40S host (verified, see §7). This model
  does not use `torch.compile`, so there is no first-inference autotune
  penalty on Blackwell.
- **Three independent layers of modification on top of the baseline:**
  1. **Weights** — baseline 100K + rank-8 LoRA corrective FT folded in
     (no LoRA wrap at inference time). The training loss combined an
     axis-weighted Huber on the base axes (peak weight on `base_t`) with
     a dim-wise keep-close regulariser toward the frozen baseline.
  2. **Action post-processing (A1' clip)** — hard caps in
     `_encode_actions`: `base_x/y ±0.3`, `base_t ±1.5`. The model learnt
     a soft cap during micro-FT; this clip is a runtime safety net.
  3. **Inference-time control flow (OOD recovery, frame-by-frame)** —
     when the gripper prediction is ambiguous under a Pick↔Place
     counterfactual prompt swap, the wrapper **discards the chunk and
     emits a gripper-safe hold chunk** for one client tick (~100 ms).
     Because the next `infer()` call from the client carries a fresh
     observation, retries naturally happen on *new* obs rather than the
     same one. After `OOD_MAX_HOLDS=3` consecutive holds the wrapper
     forces a release (returns the original chunk anyway and resets the
     counter); OOD detection resumes immediately on the next call. The
     hold chunk has `arm`/`head` deltas at zero and `base` velocity at
     zero, but `gripper` is set to `obs["state"][5]` so the gripper
     retains its current position (preventing an unintended close on
     `action[5]=0.0` under the client's discrete/hybrid threshold of
     0.5). The counter is also reset on prompt change (episode boundary).
- **No external services / gated weights are needed at runtime.** The
  PaliGemma tokenizer asset is bundled at
  `.docker_cache/policy_cache/big_vision/paligemma_tokenizer.model`
  inside the repo and mounted into the container at start-up — no
  `HF_TOKEN` required.
- The submitted checkpoint **is not** the unmodified baseline (the
  `model.safetensors` weights differ from
  `s3://airoa-icra-shared/baseline/100000/` by the LoRA delta described
  above).

---

## 6. Checkpoint file layout

```
checkpoint/
├── model.safetensors
└── assets/
    └── lerobot_datasets/task6891011_level12_v2.5_train/
        └── norm_stats.json
```

Sizes:

```
6.7G    checkpoint/model.safetensors                                       # bf16, PI0Pytorch state_dict
4.7K    checkpoint/assets/lerobot_datasets/task6891011_level12_v2.5_train/norm_stats.json
```

Total size: **≈ 6.8 GB** (one weight file + one normalisation-stats
JSON; nothing else is needed at inference time — see §5).

The OpenPI loader resolves the rest of the model contract from the code
side: model architecture is fixed by `POLICY_CONFIG_NAME=pi05_hsr_micro_ft`
in `src/openpi/training/config.py`; the PaliGemma tokenizer is bundled
in the repo as noted in §5; no per-checkpoint `config.json` /
`tokenizer/` / `preprocessing_config.json` is required by this loader.

---

## 7. Smoke test expected output

Logs from a fresh `./RUN-DOCKER-CONTAINER.sh up` followed by
`roslaunch hsr_policy_client hsr_policy_client.launch` on an L40S host
(verified 2026-05-01):

```
INFO:ood_recovery:OOD recovery ENABLED (frame-by-frame hold mode): threshold=1.150 agg=max max_holds=3 crossing_required=True strict_pa=True
INFO:root:Serving policy config=pi05_hsr_micro_ft checkpoint=/policy_checkpoint on 0.0.0.0:8000
INFO:runtime_core.websocket_policy_server:Connection from ('127.0.0.1', 45046) opened
[INFO] [1777608067.238294]: Action executed.
[INFO] [1777608067.495121]: Action executed.
[INFO] [1777608067.752802]: Action executed.
```

`Action executed.` should appear repeatedly (once per chunk). On the
verification run it appeared 22 548 times across ~50 minutes of
continuous synthetic-observation looping with zero fatal errors.

---

## 8. Contact

- Team: **Team 27 (Prox Industries)**
- Representative: Atsushi Karasawa (`a-karasawa@prox-industries.jp`)
- Submission date: 2026-05-01

---

## Appendix A. Quick copy-paste flow

A single shell sequence equivalent to §3, matching the format requested
by the organizers (mkdir → cp → export → `./RUN-DOCKER-CONTAINER.sh up`
→ `shell` → `roslaunch`).

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

Expected log: `Action executed.` (smoke-test loop). After containers
are up, the server takes ~90 s to reach `server listening on 0.0.0.0:8000`;
the first `infer()` call then returns in <1 s. Subsequent calls run at
roughly 2–5 Hz.
