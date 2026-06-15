#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/sdb/rl-posttrain/code}"
RL_DATA_ROOT="${RL_DATA_ROOT:-/data/sdb/rl-posttrain}"
VENV="${VENV:-${RL_DATA_ROOT}/venvs/e1train}"
CONFIG="${CONFIG:-configs/experiments/qwen3_8b_grpo_e5_failure_aware_from_e4.env}"
RUN_NAME="${RUN_NAME:-qwen3_8b_grpo_e5_failure_aware_from_e4}"
PLOT_ROOT="${PLOT_ROOT:-${RL_DATA_ROOT}/plots/e5_realtime}"
E5_ARTIFACT_DIR="${E5_ARTIFACT_DIR:-${RL_DATA_ROOT}/artifacts/e5_failure_aware}"
DATASET_PREFIX="${DATASET_PREFIX:-critpt_e5_failure_aware}"

cd "${ROOT}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"
if [[ -f "${RL_DATA_ROOT}/.env.judge" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${RL_DATA_ROOT}/.env.judge"
  set +a
fi

mkdir -p "${PLOT_ROOT}" "${RL_DATA_ROOT}/logs/e5_pipeline" "${RL_DATA_ROOT}/data"

if [[ ! -f "${RL_DATA_ROOT}/models/qwen3-8b-e4-official-style-final-answer-from-e3-gs160/config.json" ]]; then
  echo "missing E4 merged model" >&2
  exit 1
fi

if [[ "${FORCE_E5_DATA:-0}" == "1" || ! -s "${RL_DATA_ROOT}/data/${DATASET_PREFIX}_train.parquet" ]]; then
  echo "[e5] building failure-aware final-answer data"
  python scripts/data/build_e5_failure_aware.py \
    --out-dir "${E5_ARTIFACT_DIR}" \
    --data-dir "${RL_DATA_ROOT}/data" \
    --dataset-prefix "${DATASET_PREFIX}" \
    --train-size "${E5_TRAIN_SIZE:-1800}" \
    --val-size "${E5_VAL_SIZE:-240}" \
    --test-size "${E5_TEST_SIZE:-240}" \
    --seed "${E5_SEED:-20260630}" \
    --workers "${E5_VERIFY_WORKERS:-12}"
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

echo "[e5] starting training: ${RUN_NAME}"
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
  echo "[e5] training failed: ${RUN_NAME}" >&2
  exit "${status}"
fi
echo "[e5] training finished: ${RUN_NAME}"
