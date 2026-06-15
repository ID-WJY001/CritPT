#!/usr/bin/env bash
set -euo pipefail

ROOT=${RL_DATA_ROOT:-/data/sdb/rl-posttrain}
CODE_DIR=${CODE_DIR:-${ROOT}/code}
VENV=${VENV:-${ROOT}/venvs/rl}
DATA=${DATA:-${ROOT}/data/synthetic_critpt/v11_template_series_trace/test.jsonl}
LIMIT=${LIMIT:-128}
TP=${TP:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-3072}
MAX_TOKENS=${MAX_TOKENS:-2048}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.80}

cd "${CODE_DIR}"
source "${VENV}/bin/activate"
export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"

labels=()

run_one() {
  local label=$1
  local model=$2
  local out_dir="${ROOT}/logs/eval/${label}"
  mkdir -p "${out_dir}"
  labels+=("${label}")
  echo "[eval] ${label}"
  echo "[eval] model=${model}"
  python scripts/eval/generate_synthetic_vllm.py \
    --model "${model}" \
    --data "${DATA}" \
    --out "${out_dir}/predictions.jsonl" \
    --prompt-style audit_trace \
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

run_one finalverifier_base_v11_template_test128 \
  "${ROOT}/models/qwen3-8b"
run_one finalverifier_v6_hardcase_b8_gs10_v11_template_test128 \
  "${ROOT}/models/qwen3-8b-auditcot-finalverifier-v6-hardcase-b8-gs10"
run_one finalverifier_v9_trace_n4_b8_gs5_v11_template_test128 \
  "${ROOT}/models/qwen3-8b-audittrace-finalverifier-v9-trace-n4-b8-gs5"
run_one finalverifier_v10_curriculum_n4_b8_gs5_v11_template_test128 \
  "${ROOT}/models/qwen3-8b-audittrace-finalverifier-v10-curriculum-n4-b8-gs5"

v11_model="${ROOT}/models/qwen3-8b-audittrace-finalverifier-v11-template-n4-b8-gs5"
if [ -d "${v11_model}" ]; then
  run_one finalverifier_v11_template_n4_b8_gs5_v11_template_test128 "${v11_model}"
fi

python - "${labels[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path("/data/sdb/rl-posttrain/logs/eval")
for label in sys.argv[1:]:
    summary = json.loads((root / label / "final_answer_summary.json").read_text())
    print(label, "acc", summary["acc_mean"], "score", summary["score_mean"])
    for family, values in summary["by_family"].items():
        print(" ", family, "n", values["n"], "acc", round(values["acc_mean"], 4), "score", round(values["score_mean"], 4))
PY
