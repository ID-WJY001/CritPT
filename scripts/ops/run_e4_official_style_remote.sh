#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/sdb/rl-posttrain/code}"
RL_DATA_ROOT="${RL_DATA_ROOT:-/data/sdb/rl-posttrain}"
VENV="${VENV:-${RL_DATA_ROOT}/venvs/e1train}"
CONFIG="${CONFIG:-configs/experiments/qwen3_8b_grpo_e4_official_style_final_answer_from_e3.env}"
RUN_NAME="${RUN_NAME:-qwen3_8b_grpo_e4_official_style_final_answer_from_e3}"
PLOT_ROOT="${PLOT_ROOT:-${RL_DATA_ROOT}/plots/e4_realtime}"
E4_ARTIFACT_DIR="${E4_ARTIFACT_DIR:-${RL_DATA_ROOT}/artifacts/e4_official_style}"

cd "${ROOT}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"
if [[ -f "${RL_DATA_ROOT}/.env.judge" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${RL_DATA_ROOT}/.env.judge"
  set +a
fi

mkdir -p "${PLOT_ROOT}" "${RL_DATA_ROOT}/logs/e4_pipeline" "${RL_DATA_ROOT}/data"

if [[ ! -f "${RL_DATA_ROOT}/models/qwen3-8b-e3-yunwu-final-answer-judge-from-e2-base-gs120/config.json" ]]; then
  echo "missing E3 merged model" >&2
  exit 1
fi

if [[ "${FORCE_E4_DATA:-0}" == "1" || ! -s "${RL_DATA_ROOT}/data/critpt_e4_official_style_train.parquet" ]]; then
  echo "[e4] building official-style final-answer data"
  python scripts/data/build_e4_official_style.py \
    --out-dir "${E4_ARTIFACT_DIR}" \
    --data-dir "${RL_DATA_ROOT}/data" \
    --train-size "${E4_TRAIN_SIZE:-1080}" \
    --val-size "${E4_VAL_SIZE:-144}" \
    --test-size "${E4_TEST_SIZE:-144}" \
    --seed "${E4_SEED:-20260628}"
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

echo "[e4] starting training: ${RUN_NAME}"
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
  echo "[e4] training failed: ${RUN_NAME}" >&2
  exit "${status}"
fi
echo "[e4] training finished: ${RUN_NAME}"
