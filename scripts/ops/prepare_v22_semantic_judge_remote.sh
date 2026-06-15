#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH:-}"

OUT_DIR=${OUT_DIR:-artifacts/data/v22_semantic_judge_v20_focused_hard}
DATA_DIR=${DATA_DIR:-${RL_DATA_ROOT:-/data/sdb/rl-posttrain}/data}
TRAIN_SIZE=${TRAIN_SIZE:-1600}
VAL_SIZE=${VAL_SIZE:-240}
TEST_SIZE=${TEST_SIZE:-240}
WORKERS=${WORKERS:-16}
PROMPT_STYLE=${PROMPT_STYLE:-code}
SKIP_BUILD_VERIFY=${SKIP_BUILD_VERIFY:-1}

mkdir -p "${OUT_DIR}" "${DATA_DIR}"

VERIFY_ARGS=()
if [[ "${SKIP_BUILD_VERIFY}" == "1" || "${SKIP_BUILD_VERIFY}" == "true" ]]; then
  VERIFY_ARGS+=(--skip-verify)
fi

python3 scripts/data/build_v20_focused_hard.py \
  --out-dir "${OUT_DIR}" \
  --train-size "${TRAIN_SIZE}" \
  --val-size "${VAL_SIZE}" \
  --test-size "${TEST_SIZE}" \
  --workers "${WORKERS}" \
  "${VERIFY_ARGS[@]}"

python3 scripts/data/export_synthetic_semantic_judge_verl_parquet.py \
  --train-jsonl "${OUT_DIR}/train.jsonl" \
  --val-jsonl "${OUT_DIR}/val.jsonl" \
  --train-out "${DATA_DIR}/critpt_v22_semantic_judge_train.parquet" \
  --val-out "${DATA_DIR}/critpt_v22_semantic_judge_val.parquet" \
  --prompt-style "${PROMPT_STYLE}" \
  --sft-train-out "${OUT_DIR}/train_semantic_sft_messages.jsonl" \
  --sft-val-out "${OUT_DIR}/val_semantic_sft_messages.jsonl"

python3 - <<'PY'
import os
from pathlib import Path

import pandas as pd

data_dir = Path(os.environ.get("DATA_DIR", "/data/sdb/rl-posttrain/data"))
train = pd.read_parquet(data_dir / "critpt_v22_semantic_judge_train.parquet")
val = pd.read_parquet(data_dir / "critpt_v22_semantic_judge_val.parquet")
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
    "has_code_verifier": "code_verifier" in extra,
})
PY
