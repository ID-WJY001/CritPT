#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${ROOT:-${SCRIPT_ROOT}}"
VENV="${VENV:-/data/sdb/rl-posttrain/venvs/rl}"
MODEL="${MODEL:-/data/sdb/rl-posttrain/models/qwen3-8b}"
CRITPT_REPO="${CRITPT_REPO:-/data/sdb/rl-posttrain/repos/CritPt}"
CHALLENGES_DIR="${CHALLENGES_DIR:-${CRITPT_REPO}/data/public_test_challenges}"
RUN_NAME="${RUN_NAME:-qwen3_8b_critpt_official70_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-/data/sdb/rl-posttrain/logs/eval/${RUN_NAME}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

source "${VENV}/bin/activate"
cd "${ROOT}"

if [[ ! -d "${CRITPT_REPO}/.git" ]]; then
  mkdir -p "$(dirname "${CRITPT_REPO}")"
  git clone https://github.com/CritPt-Benchmark/CritPt.git "${CRITPT_REPO}"
fi

THINKING_ARGS=()
if [[ "${ENABLE_THINKING:-false}" == "true" ]]; then
  THINKING_ARGS+=(--enable-thinking)
fi

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS+=(--limit "${LIMIT}")
fi

python scripts/eval/run_vllm_critpt_official70.py \
  --model "${MODEL}" \
  --challenges-dir "${CHALLENGES_DIR}" \
  --out-dir "${OUT_DIR}" \
  --tensor-parallel-size "${TP:-1}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --max-model-len "${MAX_MODEL_LEN:-16384}" \
  --max-tokens "${MAX_TOKENS:-2048}" \
  --temperature "${TEMPERATURE:-0.0}" \
  --top-p "${TOP_P:-1.0}" \
  --prompt-style "${PROMPT_STYLE:-code-block}" \
  "${LIMIT_ARGS[@]}" \
  "${THINKING_ARGS[@]}"

echo
echo "Official CritPt batch written to:"
echo "${OUT_DIR}/submission_batch.json"
echo
echo "To submit for official scoring after exporting AA_API_KEY:"
echo "python scripts/eval/submit_critpt_batch.py --batch '${OUT_DIR}/submission_batch.json' --out '${OUT_DIR}/official_score.json'"
