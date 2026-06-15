#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/sdb/rl-posttrain/code}"
RL_DATA_ROOT="${RL_DATA_ROOT:-/data/sdb/rl-posttrain}"
VENV="${VENV:-${RL_DATA_ROOT}/venvs/e1train}"
E2_DIR="${E2_DIR:-${RL_DATA_ROOT}/data/e2_llm_template_full}"
SMOKE_DIR="${SMOKE_DIR:-${RL_DATA_ROOT}/data/e2_llm_template_smoke}"
TRAIN_PARQUET="${TRAIN_PARQUET:-${RL_DATA_ROOT}/data/critpt_e2_semantic_judge_train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${RL_DATA_ROOT}/data/critpt_e2_semantic_judge_val.parquet}"
PLOT_ROOT="${PLOT_ROOT:-${RL_DATA_ROOT}/plots/e2_realtime}"
E2_TEMPLATE_CACHE_DIR="${E2_TEMPLATE_CACHE_DIR:-${RL_DATA_ROOT}/data/e2_template_cache}"

E2_TRAIN_SIZE="${E2_TRAIN_SIZE:-1400}"
E2_VAL_SIZE="${E2_VAL_SIZE:-140}"
E2_TEST_SIZE="${E2_TEST_SIZE:-140}"
E2_WORKERS="${E2_WORKERS:-8}"
E2_MAX_ATTEMPTS="${E2_MAX_ATTEMPTS:-7}"
E2_SEED="${E2_SEED:-20260625}"
E2_TEMPLATES_PER_FAMILY="${E2_TEMPLATES_PER_FAMILY:-2}"
E2_TEMPLATE_WORKERS="${E2_TEMPLATE_WORKERS:-4}"

export E2_SPEC_MAX_TOKENS="${E2_SPEC_MAX_TOKENS:-6000}"
export E2_SPEC_TIMEOUT_S="${E2_SPEC_TIMEOUT_S:-120}"
export E2_SPEC_TEMPERATURE="${E2_SPEC_TEMPERATURE:-0.2}"
export E2_SPEC_MAX_RETRIES="${E2_SPEC_MAX_RETRIES:-1}"

BASE_CONFIG="${BASE_CONFIG:-configs/experiments/qwen3_8b_grpo_e2_yunwu_semantic_judge_base.env}"
FROM_E1_CONFIG="${FROM_E1_CONFIG:-configs/experiments/qwen3_8b_grpo_e2_yunwu_semantic_judge_from_e1.env}"
E1_CKPT_DIR="${E1_CKPT_DIR:-${RL_DATA_ROOT}/checkpoints/qwen3_8b_grpo_e1_yunwu_semantic_judge/global_step_60/actor}"
E1_MERGED_MODEL="${E1_MERGED_MODEL:-${RL_DATA_ROOT}/models/qwen3-8b-e1-yunwu-semantic-judge-gs60}"
BASE_MODEL="${BASE_MODEL:-${RL_DATA_ROOT}/models/qwen3-8b}"

cd "${ROOT}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"
if [[ -f "${RL_DATA_ROOT}/.env.judge" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${RL_DATA_ROOT}/.env.judge"
  set +a
fi

mkdir -p "${E2_DIR}" "${SMOKE_DIR}" "${PLOT_ROOT}" "${RL_DATA_ROOT}/logs/e2_pipeline"
mkdir -p "${E2_TEMPLATE_CACHE_DIR}"

manifest_total() {
  local manifest="$1"
  python - "$manifest" <<'PY'
import json, sys
path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as handle:
        print(json.load(handle).get("summary", {}).get("total", 0))
except FileNotFoundError:
    print(0)
PY
}

build_dataset() {
  local out_dir="$1"
  local train_size="$2"
  local val_size="$3"
  local test_size="$4"
  local templates_per_family="$5"
  local template_workers="$6"
  local seed="$7"
  local profile_limit="${8:-0}"
  python scripts/data/build_e2_llm_template_data.py \
    --train-size "${train_size}" \
    --val-size "${val_size}" \
    --test-size "${test_size}" \
    --templates-per-family "${templates_per_family}" \
    --template-workers "${template_workers}" \
    --profile-limit "${profile_limit}" \
    --llm-cache-path "${E2_TEMPLATE_CACHE_DIR}/$(basename "${out_dir%.tmp}").sqlite3" \
    --seed "${seed}" \
    --out-dir "${out_dir}"
}

expected_total=$((E2_TRAIN_SIZE + E2_VAL_SIZE + E2_TEST_SIZE))
current_total="$(manifest_total "${E2_DIR}/manifest.json")"
if [[ "${current_total}" -lt "${expected_total}" || "${E2_FORCE_REBUILD:-0}" == "1" ]]; then
  echo "[e2] running smoke template generation"
  rm -rf "${SMOKE_DIR}"
  build_dataset "${SMOKE_DIR}" "${E2_SMOKE_TRAIN_SIZE:-8}" "${E2_SMOKE_VAL_SIZE:-2}" "${E2_SMOKE_TEST_SIZE:-2}" \
    "${E2_SMOKE_TEMPLATES_PER_FAMILY:-1}" "${E2_SMOKE_TEMPLATE_WORKERS:-2}" "$((E2_SEED + 100))" "${E2_SMOKE_PROFILE_LIMIT:-2}"

  echo "[e2] running full template generation: train=${E2_TRAIN_SIZE} val=${E2_VAL_SIZE} test=${E2_TEST_SIZE} templates_per_family=${E2_TEMPLATES_PER_FAMILY}"
  rm -rf "${E2_DIR}.tmp"
  build_dataset "${E2_DIR}.tmp" "${E2_TRAIN_SIZE}" "${E2_VAL_SIZE}" "${E2_TEST_SIZE}" \
    "${E2_TEMPLATES_PER_FAMILY}" "${E2_TEMPLATE_WORKERS}" "${E2_SEED}" 0
  rm -rf "${E2_DIR}"
  mv "${E2_DIR}.tmp" "${E2_DIR}"
else
  echo "[e2] dataset already present: total=${current_total}, expected=${expected_total}"
fi

echo "[e2] exporting semantic judge parquet"
python scripts/data/export_synthetic_semantic_judge_verl_parquet.py \
  --train-jsonl "${E2_DIR}/train.jsonl" \
  --val-jsonl "${E2_DIR}/val.jsonl" \
  --train-out "${TRAIN_PARQUET}" \
  --val-out "${VAL_PARQUET}" \
  --prompt-style code \
  --sft-train-out "${E2_DIR}/train_semantic_sft_messages.jsonl" \
  --sft-val-out "${E2_DIR}/val_semantic_sft_messages.jsonl"

ensure_e1_merged_model() {
  if [[ -f "${E1_MERGED_MODEL}/config.json" ]]; then
    echo "[e2] E1 merged model already exists: ${E1_MERGED_MODEL}"
    return
  fi
  if [[ ! -d "${E1_CKPT_DIR}" ]]; then
    echo "missing E1 checkpoint actor dir: ${E1_CKPT_DIR}" >&2
    exit 1
  fi
  local hf_ckpt_dir="${E1_CKPT_DIR}/huggingface"
  if [[ ! -f "${hf_ckpt_dir}/config.json" ]]; then
    mkdir -p "${hf_ckpt_dir}"
    find "${BASE_MODEL}" -maxdepth 1 -type f \
      \( -name "*.json" -o -name "*.txt" -o -name "*.jinja" -o -name "*.model" \) \
      ! -name "model.safetensors.index.json" \
      -exec cp {} "${hf_ckpt_dir}/" \;
  fi
  if [[ ! -f "${E1_CKPT_DIR}/fsdp_config.json" ]]; then
    local first_shard
    first_shard="$(find "${E1_CKPT_DIR}" -maxdepth 1 -type f -name 'model_world_size_*_rank_0.pt' | head -1)"
    if [[ -z "${first_shard}" ]]; then
      echo "missing E1 FSDP shard under ${E1_CKPT_DIR}" >&2
      exit 1
    fi
    local world_size
    world_size="$(basename "${first_shard}" | sed -E 's/model_world_size_([0-9]+)_rank_0\.pt/\1/')"
    printf '{\n    "FSDP_version": 1,\n    "world_size": %s\n}\n' "${world_size}" > "${E1_CKPT_DIR}/fsdp_config.json"
  fi
  echo "[e2] merging E1 checkpoint -> ${E1_MERGED_MODEL}"
  rm -rf "${E1_MERGED_MODEL}.tmp"
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${E1_CKPT_DIR}" \
    --target_dir "${E1_MERGED_MODEL}.tmp"
  mv "${E1_MERGED_MODEL}.tmp" "${E1_MERGED_MODEL}"
}

plot_once() {
  local run_name="$1"
  local metrics="${RL_DATA_ROOT}/logs/${run_name}.metrics.jsonl"
  local out_dir="${PLOT_ROOT}/${run_name}"
  if [[ -s "${metrics}" ]]; then
    python scripts/ops/plot_verl_metrics_matplotlib.py \
      --metrics-jsonl "${metrics}" \
      --out-dir "${out_dir}" \
      --basename realtime \
      --title "${run_name}"
  fi
}

run_training_with_plots() {
  local config="$1"
  local run_name="$2"
  echo "[e2] starting training: ${run_name}"
  (
    while true; do
      plot_once "${run_name}" || true
      sleep "${PLOT_INTERVAL_S:-120}"
    done
  ) &
  local plot_pid=$!
  set +e
  bash scripts/train/run_verl_grpo.sh "${config}"
  local status=$?
  set -e
  kill "${plot_pid}" >/dev/null 2>&1 || true
  wait "${plot_pid}" >/dev/null 2>&1 || true
  plot_once "${run_name}" || true
  if [[ "${status}" -ne 0 ]]; then
    echo "[e2] training failed: ${run_name}" >&2
    exit "${status}"
  fi
  echo "[e2] training finished: ${run_name}"
}

ensure_e1_merged_model

run_training_with_plots "${BASE_CONFIG}" "qwen3_8b_grpo_e2_yunwu_semantic_judge_base"
run_training_with_plots "${FROM_E1_CONFIG}" "qwen3_8b_grpo_e2_yunwu_semantic_judge_from_e1"

echo "[e2] full pipeline finished"
