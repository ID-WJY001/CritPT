#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/sdb/rl-posttrain/code}"
RL_DATA_ROOT="${RL_DATA_ROOT:-/data/sdb/rl-posttrain}"
VENV="${VENV:-${RL_DATA_ROOT}/venvs/rl}"
DATA_ROOT="${DATA_ROOT:-${RL_DATA_ROOT}/data}"
OUT_DIR="${OUT_DIR:-${ROOT}/artifacts/data/v21_operator_precision}"
TRAIN_SIZE="${TRAIN_SIZE:-1800}"
VAL_SIZE="${VAL_SIZE:-240}"
TEST_SIZE="${TEST_SIZE:-240}"
SEED="${SEED:-20260621}"

source "${VENV}/bin/activate"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

python -m pytest tests/test_v21_operator_precision.py -q

python scripts/data/build_v21_operator_precision.py \
  --out-dir "${OUT_DIR}" \
  --train-size "${TRAIN_SIZE}" \
  --val-size "${VAL_SIZE}" \
  --test-size "${TEST_SIZE}" \
  --seed "${SEED}" \
  --workers "${WORKERS:-16}"

python scripts/data/export_synthetic_verl_parquet.py \
  --train-jsonl "${OUT_DIR}/train.jsonl" \
  --val-jsonl "${OUT_DIR}/val.jsonl" \
  --train-out "${DATA_ROOT}/critpt_v21_operator_precision_train.parquet" \
  --val-out "${DATA_ROOT}/critpt_v21_operator_precision_val.parquet" \
  --shuffle-seed "${SEED}"

DATA_ROOT="${DATA_ROOT}" python - <<'PY'
import os
from pathlib import Path
import pandas as pd

data_root = Path(os.environ["DATA_ROOT"])
train = data_root / "critpt_v21_operator_precision_train.parquet"
val = data_root / "critpt_v21_operator_precision_val.parquet"
train_df = pd.read_parquet(train)
val_df = pd.read_parquet(val)
print({
    "train": str(train),
    "train_rows": len(train_df),
    "val": str(val),
    "val_rows": len(val_df),
})
PY
