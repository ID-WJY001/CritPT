#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH:-}"

OUT_DIR=${OUT_DIR:-artifacts/data/e1_llm_wrapped}
DATA_DIR=${DATA_DIR:-${RL_DATA_ROOT:-/data/sdb/rl-posttrain}/data}
TRAIN_SIZE=${TRAIN_SIZE:-1400}
VAL_SIZE=${VAL_SIZE:-140}
TEST_SIZE=${TEST_SIZE:-140}
WORKERS=${WORKERS:-16}
WRAP_MODE=${WRAP_MODE:-llm}
PROMPT_STYLE=${PROMPT_STYLE:-code}
OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://yunwu.ai}
JUDGE_MODEL=${JUDGE_MODEL:-gpt-5-mini}
JUDGE_MAX_TOKENS=${JUDGE_MAX_TOKENS:-900}
JUDGE_TIMEOUT_S=${JUDGE_TIMEOUT_S:-90}
JUDGE_TEMPERATURE=${JUDGE_TEMPERATURE:-0.2}
LLM_CACHE_PATH=${LLM_CACHE_PATH:-${DATA_DIR}/judge_cache/e1_background_wrapper.sqlite3}
LLM_WORKERS=${LLM_WORKERS:-8}
SKIP_BUILD_VERIFY=${SKIP_BUILD_VERIFY:-0}

export OPENAI_BASE_URL JUDGE_MODEL JUDGE_MAX_TOKENS JUDGE_TIMEOUT_S JUDGE_TEMPERATURE

if [[ "${WRAP_MODE}" == "llm" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required when WRAP_MODE=llm" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}" "${DATA_DIR}" "$(dirname "${LLM_CACHE_PATH}")"

VERIFY_ARGS=()
if [[ "${SKIP_BUILD_VERIFY}" == "1" || "${SKIP_BUILD_VERIFY}" == "true" ]]; then
  VERIFY_ARGS+=(--skip-verify)
fi

LLM_LIMIT_ARGS=()
if [[ -n "${LLM_LIMIT:-}" ]]; then
  LLM_LIMIT_ARGS+=(--llm-limit "${LLM_LIMIT}")
fi

python3 scripts/data/build_e1_llm_wrapped.py \
  --out-dir "${OUT_DIR}" \
  --train-size "${TRAIN_SIZE}" \
  --val-size "${VAL_SIZE}" \
  --test-size "${TEST_SIZE}" \
  --workers "${WORKERS}" \
  --wrap-mode "${WRAP_MODE}" \
  --llm-cache-path "${LLM_CACHE_PATH}" \
  --llm-workers "${LLM_WORKERS}" \
  "${LLM_LIMIT_ARGS[@]}" \
  "${VERIFY_ARGS[@]}"

python3 scripts/data/export_synthetic_semantic_judge_verl_parquet.py \
  --train-jsonl "${OUT_DIR}/train.jsonl" \
  --val-jsonl "${OUT_DIR}/val.jsonl" \
  --train-out "${DATA_DIR}/critpt_e1_semantic_judge_train.parquet" \
  --val-out "${DATA_DIR}/critpt_e1_semantic_judge_val.parquet" \
  --prompt-style "${PROMPT_STYLE}" \
  --sft-train-out "${OUT_DIR}/train_semantic_sft_messages.jsonl" \
  --sft-val-out "${OUT_DIR}/val_semantic_sft_messages.jsonl"

python3 - <<'PY'
import os
from pathlib import Path

import pandas as pd

data_dir = Path(os.environ.get("DATA_DIR", "/data/sdb/rl-posttrain/data"))
train = pd.read_parquet(data_dir / "critpt_e1_semantic_judge_train.parquet")
val = pd.read_parquet(data_dir / "critpt_e1_semantic_judge_val.parquet")
first = train.iloc[0].to_dict()
extra = first["extra_info"]
assert "code_verifier" not in extra
assert "verifier" not in extra
assert extra["reference_answer"]
assert extra["rubric"]
print({
    "train_rows": len(train),
    "val_rows": len(val),
    "data_source": first["data_source"],
    "reward_style": first["reward_model"]["style"],
    "first_problem_id": extra["problem_id"],
})
PY
