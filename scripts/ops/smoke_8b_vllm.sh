#!/usr/bin/env bash
set -euo pipefail

ROOT=${RL_DATA_ROOT:-/data/sdb/rl-posttrain}
CODE=${ROOT}/code
MODEL=${MODEL:-${ROOT}/models/qwen3-8b}

cd "${CODE}"
source "${ROOT}/venvs/rl/bin/activate"
source configs/hardware/a100_8x40g_pcie.env

VLLM_USE_V1=${VLLM_USE_V1:-1} \
VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn} \
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} \
python scripts/smoke/vllm_generate.py \
  --model "${MODEL}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.60}" \
  --max-model-len "${MAX_MODEL_LEN:-1024}"
