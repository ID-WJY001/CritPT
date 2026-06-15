#!/usr/bin/env bash
set -euo pipefail

ROOT=${RL_DATA_ROOT:-/data/sdb/rl-posttrain}
CODE=${RL_CODE_ROOT:-${ROOT}/code}
VERSION=${VERSION:-v1}
DATA_DIR=${DATA_DIR:-${ROOT}/data/synthetic_critpt/${VERSION}}
TRAIN_SIZE=${TRAIN_SIZE:-20000}
VAL_SIZE=${VAL_SIZE:-1000}
TEST_SIZE=${TEST_SIZE:-1000}
SEED=${SEED:-20260607}
WORKERS=${WORKERS:-32}
UPDATE_LATEST=${UPDATE_LATEST:-1}

cd "${CODE}"

python scripts/data/build_synthetic_critpt.py \
  --name "synthetic_critpt_${VERSION}" \
  --out-dir "${DATA_DIR}" \
  --train-size "${TRAIN_SIZE}" \
  --val-size "${VAL_SIZE}" \
  --test-size "${TEST_SIZE}" \
  --seed "${SEED}" \
  --workers "${WORKERS}"

python scripts/data/export_synthetic_verl_parquet.py \
  --train-jsonl "${DATA_DIR}/train.jsonl" \
  --val-jsonl "${DATA_DIR}/val.jsonl" \
  --train-out "${ROOT}/data/critpt_synth_${VERSION}_code_train.parquet" \
  --val-out "${ROOT}/data/critpt_synth_${VERSION}_code_val.parquet"

if [[ "${UPDATE_LATEST}" == "1" ]]; then
  cp "${ROOT}/data/critpt_synth_${VERSION}_code_train.parquet" "${ROOT}/data/critpt_synth_code_train.parquet"
  cp "${ROOT}/data/critpt_synth_${VERSION}_code_val.parquet" "${ROOT}/data/critpt_synth_code_val.parquet"
fi

echo "Synthetic CritPT ${VERSION} is ready:"
echo "  raw:    ${DATA_DIR}"
echo "  train:  ${ROOT}/data/critpt_synth_${VERSION}_code_train.parquet"
echo "  val:    ${ROOT}/data/critpt_synth_${VERSION}_code_val.parquet"
echo "  latest: ${ROOT}/data/critpt_synth_code_train.parquet"
