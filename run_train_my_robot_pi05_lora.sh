#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Pi0.5 LoRA Fine-tuning Pipeline
#
# Steps:
#   1. Convert raw data to LeRobot format  (skip if already done)
#   2. Compute normalization statistics
#   3. Run training
#
# Usage:
#   # Smoke test (100 steps, batch=4)
#   bash run_train_my_robot_pi05_lora.sh smoke
#
#   # Production (5000 steps)
#   bash run_train_my_robot_pi05_lora.sh train
#
#   # Only compute norm stats
#   bash run_train_my_robot_pi05_lora.sh norm
#
#   # Only convert data
#   bash run_train_my_robot_pi05_lora.sh convert
# ============================================================================

# --- Configuration (edit these) -------------------------------------------

# Where your raw data lives
RAW_DATA_DIR="${RAW_DATA_DIR:-./raw_episodes}"

# Where to output the LeRobot dataset
DATASET_DIR="${DATASET_DIR:-./my_lerobot_dataset}"

# LeRobot repo ID (just a namespace, not an actual hub repo)
REPO_ID="${REPO_ID:-my_org/my_robot_data}"

# Where datasets are stored (parent of REPO_ID)
# If DATASET_DIR is ./my_lerobot_dataset and REPO_ID is my_org/my_robot_data,
# then DATA_ROOT should be set so DATA_ROOT/REPO_ID = DATASET_DIR.
DATA_ROOT="${DATA_ROOT:-./my_lerobot_dataset/../}"

# Assets directory for norm stats
ASSETS_DIR="${ASSETS_DIR:-./assets/my_robot_pi05_lora}"

# Experiment name
EXP_NAME="${EXP_NAME:-lora_v1}"

# Number of GPUs
NUM_GPUS="${NUM_GPUS:-1}"

# --- Derived ---
SMOKE_CONFIG="my_robot_pi05_lora_smoke"
PROD_CONFIG="my_robot_pi05_lora"

MODE="${1:-smoke}"

echo "================================================"
echo " Pi0.5 LoRA Fine-tuning Pipeline"
echo " Mode:        ${MODE}"
echo " Dataset:     ${DATASET_DIR}"
echo " Repo ID:     ${REPO_ID}"
echo " Assets:      ${ASSETS_DIR}"
echo " GPUs:        ${NUM_GPUS}"
echo "================================================"

# --- Step 1: Convert data -------------------------------------------------

do_convert() {
    echo ""
    echo ">>> Step 1: Converting raw data to LeRobot format..."
    python convert_my_data_to_lerobot.py \
        --raw-dir "${RAW_DATA_DIR}" \
        --output-dir "${DATASET_DIR}" \
        --repo-id "${REPO_ID}" \
        --fps 10
    echo ">>> Conversion complete."
}

# --- Step 2: Compute norm stats -------------------------------------------

do_norm() {
    echo ""
    echo ">>> Step 2: Computing normalization statistics..."
    python scripts/compute_norm_stats.py \
        --dataset-dir "${DATASET_DIR}" \
        --output-dir "${ASSETS_DIR}/${REPO_ID}"
    echo ">>> Norm stats saved to ${ASSETS_DIR}/${REPO_ID}/norm_stats.json"
}

# --- Step 3: Training -----------------------------------------------------

do_train() {
    local config_name="$1"
    local extra_args="${2:-}"

    echo ""
    echo ">>> Step 3: Training with config=${config_name}..."

    if [ "${NUM_GPUS}" -gt 1 ]; then
        torchrun --nproc_per_node="${NUM_GPUS}" -m openpi.training.train \
            --config-name "${config_name}" \
            --exp-name "${EXP_NAME}" \
            --data-root "${DATA_ROOT}" \
            --no-wandb \
            ${extra_args}
    else
        python -m openpi.training.train \
            --config-name "${config_name}" \
            --exp-name "${EXP_NAME}" \
            --data-root "${DATA_ROOT}" \
            --no-wandb \
            ${extra_args}
    fi

    echo ">>> Training complete."
    echo ">>> Checkpoint: ./checkpoints/${config_name}/${EXP_NAME}/"
}

# --- Dispatch --------------------------------------------------------------

case "${MODE}" in
    convert)
        do_convert
        ;;
    norm)
        do_norm
        ;;
    smoke)
        # Quick end-to-end test
        if [ ! -d "${DATASET_DIR}/meta" ]; then
            do_convert
        fi
        do_norm
        do_train "${SMOKE_CONFIG}" "--overwrite"
        echo ""
        echo "=== Smoke test passed! ==="
        ;;
    train)
        if [ ! -f "${ASSETS_DIR}/${REPO_ID}/norm_stats.json" ]; then
            echo "Norm stats not found. Computing first..."
            do_norm
        fi
        do_train "${PROD_CONFIG}"
        ;;
    all)
        do_convert
        do_norm
        do_train "${PROD_CONFIG}"
        ;;
    *)
        echo "Usage: $0 {smoke|train|convert|norm|all}"
        exit 1
        ;;
esac
