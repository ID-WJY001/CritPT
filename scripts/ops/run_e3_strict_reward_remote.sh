#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/sdb/rl-posttrain/code}"
RL_DATA_ROOT="${RL_DATA_ROOT:-/data/sdb/rl-posttrain}"
VENV="${VENV:-${RL_DATA_ROOT}/venvs/e1train}"
CONFIG="${CONFIG:-configs/experiments/qwen3_8b_grpo_e3_yunwu_strict_code_judge_from_e2_base.env}"
RUN_NAME="${RUN_NAME:-qwen3_8b_grpo_e3_yunwu_final_answer_judge_from_e2_base}"
PLOT_ROOT="${PLOT_ROOT:-${RL_DATA_ROOT}/plots/e3_realtime}"

cd "${ROOT}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"
if [[ -f "${RL_DATA_ROOT}/.env.judge" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${RL_DATA_ROOT}/.env.judge"
  set +a
fi

mkdir -p "${PLOT_ROOT}" "${RL_DATA_ROOT}/logs/e3_pipeline"

if [[ ! -s "${RL_DATA_ROOT}/data/critpt_e3_final_answer_judge_train.parquet" ]]; then
  echo "missing E3 train parquet under ${RL_DATA_ROOT}/data" >&2
  exit 1
fi
if [[ ! -f "${RL_DATA_ROOT}/models/qwen3-8b-e2-yunwu-semantic-judge-base-gs120/config.json" ]]; then
  echo "missing E2-base merged model" >&2
  exit 1
fi

plot_once() {
  local metrics="${RL_DATA_ROOT}/logs/${RUN_NAME}.metrics.jsonl"
  local out_dir="${PLOT_ROOT}/${RUN_NAME}"
  if [[ -s "${metrics}" ]]; then
    python scripts/ops/plot_verl_metrics_matplotlib.py \
      --metrics-jsonl "${metrics}" \
      --out-dir "${out_dir}" \
      --basename realtime \
      --title "${RUN_NAME}"
  fi
}

echo "[e3] starting strict reward training: ${RUN_NAME}"
(
  while true; do
    plot_once || true
    sleep "${PLOT_INTERVAL_S:-120}"
  done
) &
plot_pid=$!
set +e
bash scripts/train/run_verl_grpo.sh "${CONFIG}"
status=$?
set -e
kill "${plot_pid}" >/dev/null 2>&1 || true
wait "${plot_pid}" >/dev/null 2>&1 || true
plot_once || true
if [[ "${status}" -ne 0 ]]; then
  echo "[e3] training failed: ${RUN_NAME}" >&2
  exit "${status}"
fi
echo "[e3] strict reward training finished: ${RUN_NAME}"
