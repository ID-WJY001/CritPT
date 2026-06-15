#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/sdb/rl-posttrain/code}"
RL_DATA_ROOT="${RL_DATA_ROOT:-/data/sdb/rl-posttrain}"
VENV="${VENV:-${RL_DATA_ROOT}/venvs/rl}"
RUN_NAME="${RUN_NAME:-qwen3_8b_grpo_v21_operator_precision_from_v20_gs40_n8}"
STEP="${STEP:-30}"
CKPT_DIR="${CKPT_DIR:-${RL_DATA_ROOT}/checkpoints/${RUN_NAME}/global_step_${STEP}/actor}"
TARGET_MODEL="${TARGET_MODEL:-${RL_DATA_ROOT}/models/qwen3-8b-v21-operator-precision-n8-gs${STEP}}"
TMP_MODEL="${TARGET_MODEL}.tmp"
HF_MODEL_CONFIG_PATH="${HF_MODEL_CONFIG_PATH:-${RL_DATA_ROOT}/models/qwen3-8b-v20-focused-hard-n8-gs40}"
if [[ ! -f "${HF_MODEL_CONFIG_PATH}/config.json" ]]; then
  HF_MODEL_CONFIG_PATH="${RL_DATA_ROOT}/models/qwen3-8b-v19-failure-mined-n8-gs60"
fi
if [[ ! -f "${HF_MODEL_CONFIG_PATH}/config.json" ]]; then
  HF_MODEL_CONFIG_PATH="${RL_DATA_ROOT}/models/qwen3-8b"
fi
HF_CKPT_DIR="${CKPT_DIR}/huggingface"
FSDP_CONFIG="${CKPT_DIR}/fsdp_config.json"

source "${VENV}/bin/activate"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "missing checkpoint actor dir: ${CKPT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${HF_MODEL_CONFIG_PATH}/config.json" ]]; then
  echo "missing HF config source: ${HF_MODEL_CONFIG_PATH}/config.json" >&2
  exit 1
fi

if [[ ! -f "${HF_CKPT_DIR}/config.json" ]]; then
  mkdir -p "${HF_CKPT_DIR}"
  find "${HF_MODEL_CONFIG_PATH}" -maxdepth 1 -type f \
    \( -name "*.json" -o -name "*.txt" -o -name "*.jinja" -o -name "*.model" \) \
    ! -name "model.safetensors.index.json" \
    -exec cp {} "${HF_CKPT_DIR}/" \;
fi

if [[ ! -f "${FSDP_CONFIG}" ]]; then
  first_shard="$(find "${CKPT_DIR}" -maxdepth 1 -type f -name "model_world_size_*_rank_0.pt" | head -1)"
  if [[ -z "${first_shard}" ]]; then
    echo "missing model shard needed to reconstruct ${FSDP_CONFIG}" >&2
    exit 1
  fi
  world_size="$(basename "${first_shard}" | sed -E 's/model_world_size_([0-9]+)_rank_0\.pt/\1/')"
  printf '{\n    "FSDP_version": 1,\n    "world_size": %s\n}\n' "${world_size}" > "${FSDP_CONFIG}"
fi

if [[ ! -f "${RL_DATA_ROOT}/.env.aa" ]]; then
  echo "missing ${RL_DATA_ROOT}/.env.aa; official submission will be skipped" >&2
  SUBMIT_AA="${SUBMIT_AA:-false}"
else
  # shellcheck disable=SC1090
  source "${RL_DATA_ROOT}/.env.aa"
  SUBMIT_AA="${SUBMIT_AA:-true}"
fi

if [[ ! -d "${TARGET_MODEL}" ]]; then
  rm -rf "${TMP_MODEL}"
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${CKPT_DIR}" \
    --target_dir "${TMP_MODEL}"
  mv "${TMP_MODEL}" "${TARGET_MODEL}"
else
  echo "merged model already exists: ${TARGET_MODEL}"
fi

run_eval() {
  local style="$1"
  local suffix="$2"
  local out_dir="${RL_DATA_ROOT}/logs/eval/qwen3_8b_v21_gs${STEP}_official70_${suffix}_ctx32768_max4096"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_EVAL:-0,1,2,3}" \
  MODEL="${TARGET_MODEL}" \
  ROOT="${ROOT}" \
  RUN_NAME="$(basename "${out_dir}")" \
  OUT_DIR="${out_dir}" \
  TP="${TP:-4}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}" \
  MAX_TOKENS="${MAX_TOKENS:-4096}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.78}" \
  PROMPT_STYLE="${style}" \
  ENABLE_THINKING=false \
  bash scripts/ops/eval_8b_critpt_official70.sh

  python scripts/eval/analyze_official_submission.py \
    --batch "${out_dir}/submission_batch.json" \
    --out-json "${out_dir}/static_output_analysis.json" \
    --out-md "${out_dir}/static_output_analysis.md"

  if [[ "${SUBMIT_AA}" == "true" ]]; then
    python scripts/eval/submit_critpt_batch.py \
      --batch "${out_dir}/submission_batch.json" \
      --out "${out_dir}/official_score.json"
  else
    echo "SUBMIT_AA=false; skipped AA submission for ${out_dir}"
  fi
}

run_eval raw-compact rawcompact
run_eval code-block codeblock

echo "merged model: ${TARGET_MODEL}"
