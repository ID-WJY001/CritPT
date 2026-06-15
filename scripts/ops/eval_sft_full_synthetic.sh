#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/rl-posttrain}"
VENV="${VENV:-/data/sdb/rl-posttrain/venvs/rl}"
MODEL="${MODEL:-/data/sdb/rl-posttrain/checkpoints/qwen3_8b_sft_full_critpt_synth_v1/final_model}"
SPLIT="${SPLIT:-val}"
DATA="${DATA:-/data/sdb/rl-posttrain/data/synthetic_critpt/v1/${SPLIT}.jsonl}"
RUN_NAME="${RUN_NAME:-qwen3_8b_sft_full_critpt_synth_v1_${SPLIT}_eval}"
OUT_DIR="${OUT_DIR:-/data/sdb/rl-posttrain/logs/eval/${RUN_NAME}}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

source "${VENV}/bin/activate"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

mkdir -p "${OUT_DIR}"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS+=(--limit "${LIMIT}")
fi

python scripts/eval/generate_synthetic_with_model.py \
  --model "${MODEL}" \
  --data "${DATA}" \
  --out "${OUT_DIR}/predictions.jsonl" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  "${LIMIT_ARGS[@]}"

python scripts/eval/eval_synthetic_critpt.py \
  --data "${DATA}" \
  --predictions "${OUT_DIR}/predictions.jsonl" \
  --out "${OUT_DIR}/score.json"

echo
echo "Synthetic full-SFT eval written to:"
echo "${OUT_DIR}"
