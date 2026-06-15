#!/usr/bin/env bash
set -euo pipefail

ROOT=${RL_DATA_ROOT:-/data/sdb/rl-posttrain}
CODE_DIR=${CODE_DIR:-${ROOT}/code}
VENV=${VENV:-${ROOT}/venvs/rl}
DATA=${DATA:-${ROOT}/data/synthetic_critpt/v7_compact/test.jsonl}
LIMIT=${LIMIT:-128}
TP=${TP:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-3072}
MAX_TOKENS=${MAX_TOKENS:-2048}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.80}

cd "${CODE_DIR}"
source "${VENV}/bin/activate"
export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"

run_one() {
  local label=$1
  local model=$2
  local out_dir="${ROOT}/logs/eval/${label}"
  mkdir -p "${out_dir}"
  echo "[eval] ${label}"
  echo "[eval] model=${model}"
  python scripts/eval/generate_synthetic_vllm.py \
    --model "${model}" \
    --data "${DATA}" \
    --out "${out_dir}/predictions.jsonl" \
    --prompt-style audit_short \
    --limit "${LIMIT}" \
    --tensor-parallel-size "${TP}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-tokens "${MAX_TOKENS}" \
    --temperature 0.0
  python scripts/eval/eval_synthetic_final_answer.py \
    --data "${DATA}" \
    --predictions "${out_dir}/predictions.jsonl" \
    --out-dir "${out_dir}"
}

run_one finalverifier_base_v8_dense_test128 \
  "${ROOT}/models/qwen3-8b"
run_one finalverifier_v6_hardcase_b8_gs10_v8_dense_test128 \
  "${ROOT}/models/qwen3-8b-auditcot-finalverifier-v6-hardcase-b8-gs10"
run_one finalverifier_v8_dense_n4_b8_gs5_v8_dense_test128 \
  "${ROOT}/models/qwen3-8b-auditshort-finalverifier-v8-dense-n4-b8-gs5"

python - <<'PY'
import json
from pathlib import Path

root = Path("/data/sdb/rl-posttrain/logs/eval")
labels = [
    "finalverifier_base_v8_dense_test128",
    "finalverifier_v6_hardcase_b8_gs10_v8_dense_test128",
    "finalverifier_v8_dense_n4_b8_gs5_v8_dense_test128",
]
for label in labels:
    summary = json.loads((root / label / "final_answer_summary.json").read_text())
    print(label, "acc", summary["acc_mean"], "score", summary["score_mean"])
    for family, values in summary["by_family"].items():
        print(" ", family, "n", values["n"], "acc", round(values["acc_mean"], 4), "score", round(values["score_mean"], 4))
PY
