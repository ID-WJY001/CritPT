#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/sdb/rl-posttrain/code}"
RL_DATA_ROOT="${RL_DATA_ROOT:-/data/sdb/rl-posttrain}"
VENV="${VENV:-${RL_DATA_ROOT}/venvs/e1train}"
RUN_NAME="${RUN_NAME:-qwen3_8b_grpo_e4_official_style_final_answer_from_e3}"
STEP="${STEP:-160}"
CKPT_DIR="${CKPT_DIR:-${RL_DATA_ROOT}/checkpoints/${RUN_NAME}/global_step_${STEP}/actor}"
TARGET_MODEL="${TARGET_MODEL:-${RL_DATA_ROOT}/models/qwen3-8b-e4-official-style-final-answer-from-e3-gs${STEP}}"
CONFIG_MODEL="${CONFIG_MODEL:-${RL_DATA_ROOT}/models/qwen3-8b-e3-yunwu-final-answer-judge-from-e2-base-gs120}"
CRITPT_REPO="${CRITPT_REPO:-${RL_DATA_ROOT}/repos/CritPt}"
EVAL_LABEL="${EVAL_LABEL:-qwen3_8b_e4_official_style_from_e3_gs${STEP}}"
STYLES="${STYLES:-raw-compact:rawcompact code-block:codeblock}"

CUDA_VISIBLE_DEVICES_EVAL="${CUDA_VISIBLE_DEVICES_EVAL:-0,1,2,3}"
TP="${TP:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.78}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"

source "${VENV}/bin/activate"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
mkdir -p "${RL_DATA_ROOT}/logs/eval" "${RL_DATA_ROOT}/models"

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "missing checkpoint actor dir: ${CKPT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_MODEL}/config.json" ]]; then
  echo "missing config model: ${CONFIG_MODEL}/config.json" >&2
  exit 1
fi

HF_CKPT_DIR="${CKPT_DIR}/huggingface"
mkdir -p "${HF_CKPT_DIR}"
if [[ ! -f "${HF_CKPT_DIR}/config.json" ]]; then
  find "${CONFIG_MODEL}" -maxdepth 1 -type f \
    \( -name "*.json" -o -name "*.txt" -o -name "*.jinja" -o -name "*.model" \) \
    ! -name "model.safetensors.index.json" \
    -exec cp {} "${HF_CKPT_DIR}/" \;
fi

if [[ ! -f "${CKPT_DIR}/fsdp_config.json" ]]; then
  first_shard="$(find "${CKPT_DIR}" -maxdepth 1 -type f -name "model_world_size_*_rank_0.pt" | head -1)"
  if [[ -z "${first_shard}" ]]; then
    echo "missing model shard needed to reconstruct fsdp_config.json" >&2
    exit 1
  fi
  world_size="$(basename "${first_shard}" | sed -E 's/model_world_size_([0-9]+)_rank_0\.pt/\1/')"
  printf '{\n  "FSDP_version": 1,\n  "world_size": %s\n}\n' "${world_size}" > "${CKPT_DIR}/fsdp_config.json"
fi

echo "[e4] merge start: ${TARGET_MODEL}"
if [[ ! -d "${TARGET_MODEL}" ]]; then
  tmp_model="${TARGET_MODEL}.tmp"
  rm -rf "${tmp_model}"
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${CKPT_DIR}" \
    --target_dir "${tmp_model}"
  mv "${tmp_model}" "${TARGET_MODEL}"
else
  echo "merged model already exists: ${TARGET_MODEL}"
fi
echo "[e4] merge done: ${TARGET_MODEL}"

submit_aa=false
if [[ -f "${RL_DATA_ROOT}/.env.aa" ]]; then
  # shellcheck disable=SC1090
  source "${RL_DATA_ROOT}/.env.aa"
  submit_aa=true
elif [[ -n "${AA_API_KEY:-}" ]]; then
  submit_aa=true
fi

run_eval() {
  local style="$1"
  local suffix="$2"
  local out_dir="${RL_DATA_ROOT}/logs/eval/${EVAL_LABEL}_official70_${suffix}_ctx32768_max4096"
  echo "[e4 official70] generating ${style} -> ${out_dir}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_EVAL}" \
  VENV="${VENV}" \
  MODEL="${TARGET_MODEL}" \
  ROOT="${ROOT}" \
  CRITPT_REPO="${CRITPT_REPO}" \
  RUN_NAME="$(basename "${out_dir}")" \
  OUT_DIR="${out_dir}" \
  TP="${TP}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  MAX_TOKENS="${MAX_TOKENS}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
  TEMPERATURE="${TEMPERATURE}" \
  TOP_P="${TOP_P}" \
  PROMPT_STYLE="${style}" \
  ENABLE_THINKING=false \
  bash scripts/ops/eval_8b_critpt_official70.sh

  python scripts/eval/analyze_official_submission.py \
    --batch "${out_dir}/submission_batch.json" \
    --out-json "${out_dir}/static_output_analysis.json" \
    --out-md "${out_dir}/static_output_analysis.md"

  if [[ "${submit_aa}" == "true" ]]; then
    python scripts/eval/submit_critpt_batch.py \
      --batch "${out_dir}/submission_batch.json" \
      --out "${out_dir}/official_score.json" || true
  else
    echo "AA_API_KEY missing; skipped official submission for ${out_dir}"
  fi
}

for style_pair in ${STYLES}; do
  run_eval "${style_pair%%:*}" "${style_pair##*:}"
done

echo "[e4] eval done: ${TARGET_MODEL}"
