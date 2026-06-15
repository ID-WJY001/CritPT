#!/usr/bin/env bash
set -euo pipefail

ROOT=${RL_DATA_ROOT:-/data/sdb/rl-posttrain}
CODE=${RL_CODE_ROOT:-${ROOT}/code}
DATA_DIR=${DATA_DIR:-${ROOT}/data/synthetic_critpt/v0}

cd "${CODE}"

python scripts/data/build_synthetic_critpt.py \
  --out-dir "${DATA_DIR}" \
  --train-size "${TRAIN_SIZE:-3500}" \
  --val-size "${VAL_SIZE:-300}" \
  --test-size "${TEST_SIZE:-300}" \
  --seed "${SEED:-20260606}"

python scripts/data/export_synthetic_verl_parquet.py \
  --train-jsonl "${DATA_DIR}/train.jsonl" \
  --val-jsonl "${DATA_DIR}/val.jsonl" \
  --train-out "${ROOT}/data/critpt_synth_code_train.parquet" \
  --val-out "${ROOT}/data/critpt_synth_code_val.parquet"
