FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs build-essential clang ca-certificates curl \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock LICENSE README.md ./
COPY packages/policy-client ./packages/policy-client
COPY src/openpi ./src/openpi
COPY server ./server
COPY runtime_core ./runtime_core

# Create virtual environment and install locked deps including local workspace packages.
RUN uv venv --python 3.11 /workspace/.venv
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev
RUN uv pip install --python /workspace/.venv/bin/python pytest

# Blackwell GPU (sm_120) requires PyTorch nightly with CUDA 12.8 support.
# Override stable PyTorch installed by uv sync with nightly build.
RUN uv pip install --python /workspace/.venv/bin/python \
    --reinstall --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Support PyTorch checkpoints (same behavior as upstream serve_policy image).
RUN /workspace/.venv/bin/python -c "import transformers; print(transformers.__file__)" | xargs dirname | xargs -I{} cp -r /workspace/src/openpi/models_pytorch/transformers_replace/* {}

ENV PATH="/workspace/.venv/bin:${PATH}"
ENV PYTHONPATH="/workspace:/workspace/src:/workspace/packages/policy-client/src"
ENV OPENPI_DATA_HOME="/policy_assets"

# OOD recovery defaults for the ICRA submission. Overridable at runtime.
# See server/ood_recovery.py for semantics and OOD_* knobs.
ENV OOD_ENABLED=1 \
    OOD_AMBIGUITY_THRESHOLD=1.15 \
    OOD_CHUNK_AGGREGATION=max \
    OOD_REQUIRE_CROSSING=1 \
    OOD_RETRY_ON_OOD=1 \
    OOD_MAX_RETRIES=3 \
    OOD_PERSISTENT_ACTION=keep_last \
    OOD_MIN_CONSECUTIVE=1 \
    OOD_PA_STRICT=1

ENTRYPOINT ["/workspace/server/entrypoint.sh"]
