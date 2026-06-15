#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/rl-posttrain}"
VENV="${VENV:-/data/sdb/rl-posttrain/venvs/rl}"
MODEL="${MODEL:-/data/sdb/rl-posttrain/models/qwen3-8b}"
DATA="${DATA:-${ROOT}/data/eval/critpt_example_eval.jsonl}"
RUN_NAME="${RUN_NAME:-qwen3_8b_critpt_public_baseline_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-/data/sdb/rl-posttrain/logs/eval/${RUN_NAME}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

source "${VENV}/bin/activate"
cd "${ROOT}"

THINKING_ARGS=()
if [[ "${ENABLE_THINKING:-false}" == "true" ]]; then
  THINKING_ARGS+=(--enable-thinking)
fi

python scripts/eval/run_vllm_critpt_eval.py \
  --model "${MODEL}" \
  --data "${DATA}" \
  --out-dir "${OUT_DIR}" \
  --tensor-parallel-size "${TP:-1}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.75}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --max-tokens "${MAX_TOKENS:-1024}" \
  --temperature "${TEMPERATURE:-0.0}" \
  --top-p "${TOP_P:-1.0}" \
  "${THINKING_ARGS[@]}"
