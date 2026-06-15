#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/sdb/rl-posttrain/code}"
RL_DATA_ROOT="${RL_DATA_ROOT:-/data/sdb/rl-posttrain}"
VENV="${VENV:-${RL_DATA_ROOT}/venvs/e1train}"
CONFIG="${CONFIG:-configs/experiments/qwen3_8b_grpo_e6_teacher_specs_strict_judge_from_e4.env}"
RUN_NAME="${RUN_NAME:-qwen3_8b_grpo_e6_teacher_specs_strict_judge_from_e4}"
PLOT_ROOT="${PLOT_ROOT:-${RL_DATA_ROOT}/plots/e6_realtime}"
E6_ARTIFACT_DIR="${E6_ARTIFACT_DIR:-${RL_DATA_ROOT}/artifacts/e6_teacher_specs}"
DATASET_PREFIX="${DATASET_PREFIX:-critpt_e6_teacher_specs}"

cd "${ROOT}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"
if [[ -f "${RL_DATA_ROOT}/.env.judge" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${RL_DATA_ROOT}/.env.judge"
  set +a
fi

mkdir -p "${PLOT_ROOT}" "${RL_DATA_ROOT}/logs/e6_pipeline" "${RL_DATA_ROOT}/data"

if [[ ! -f "${RL_DATA_ROOT}/models/qwen3-8b-e4-official-style-final-answer-from-e3-gs160/config.json" ]]; then
  echo "missing E4 merged model" >&2
  exit 1
fi

probe_judge() {
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[e6] missing OPENAI_API_KEY; refusing to start LLM-judge training" >&2
    return 1
  fi
  if [[ -z "${OPENAI_BASE_URL:-}" || -z "${JUDGE_MODEL:-}" ]]; then
    echo "[e6] missing OPENAI_BASE_URL or JUDGE_MODEL; refusing to start" >&2
    return 1
  fi
  local url="${OPENAI_BASE_URL%/}/v1/chat/completions"
  local body_path="${RL_DATA_ROOT}/logs/e6_pipeline/judge_probe_body.json"
  local status
  status="$(
    curl -sS --max-time "${JUDGE_PROBE_TIMEOUT_S:-18}" \
      -o "${body_path}" \
      -w "%{http_code}" \
      -H "Authorization: Bearer ${OPENAI_API_KEY}" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${JUDGE_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Return JSON {\\\"ok\\\": true} only.\"}],\"temperature\":0,\"max_tokens\":20,\"response_format\":{\"type\":\"json_object\"}}" \
      "${url}" || true
  )"
  if [[ "${status}" != 2* ]]; then
    echo "[e6] judge probe failed with HTTP ${status}; refusing to start training" >&2
    head -c 600 "${body_path}" >&2 || true
    echo >&2
    return 1
  fi
  echo "[e6] judge probe ok: ${OPENAI_BASE_URL%/} ${JUDGE_MODEL}"
}

if [[ "${E6_SKIP_JUDGE_PROBE:-0}" != "1" ]]; then
  probe_judge
fi

if [[ "${FORCE_E6_DATA:-0}" == "1" || ! -s "${RL_DATA_ROOT}/data/${DATASET_PREFIX}_train.parquet" ]]; then
  echo "[e6] building teacher-spec strict-judge data"
  python scripts/data/build_e6_teacher_specs.py \
    --out-dir "${E6_ARTIFACT_DIR}" \
    --data-dir "${RL_DATA_ROOT}/data" \
    --dataset-prefix "${DATASET_PREFIX}" \
    --train-size "${E6_TRAIN_SIZE:-800}" \
    --val-size "${E6_VAL_SIZE:-120}" \
    --test-size "${E6_TEST_SIZE:-120}" \
    --seed "${E6_SEED:-20260701}" \
    --workers "${E6_SPEC_WORKERS:-2}" \
    --max-attempts-per-example "${E6_MAX_ATTEMPTS_PER_EXAMPLE:-12}"
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

echo "[e6] starting training: ${RUN_NAME}"
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
  echo "[e6] training failed: ${RUN_NAME}" >&2
  exit "${status}"
fi
echo "[e6] training finished: ${RUN_NAME}"
