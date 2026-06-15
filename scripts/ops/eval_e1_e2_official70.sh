#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/sdb/rl-posttrain/code}"
RL_DATA_ROOT="${RL_DATA_ROOT:-/data/sdb/rl-posttrain}"
VENV="${VENV:-${RL_DATA_ROOT}/venvs/e1train}"
CRITPT_REPO="${CRITPT_REPO:-${RL_DATA_ROOT}/repos/CritPt}"

E1_MODEL="${E1_MODEL:-${RL_DATA_ROOT}/models/qwen3-8b-e1-yunwu-semantic-judge-gs60}"
E2_RUN_NAME="${E2_RUN_NAME:-qwen3_8b_grpo_e2_yunwu_semantic_judge_base}"
E2_STEP="${E2_STEP:-120}"
E2_CKPT_DIR="${E2_CKPT_DIR:-${RL_DATA_ROOT}/checkpoints/${E2_RUN_NAME}/global_step_${E2_STEP}/actor}"
E2_MODEL="${E2_MODEL:-${RL_DATA_ROOT}/models/qwen3-8b-e2-yunwu-semantic-judge-base-gs${E2_STEP}}"
E2_FROM_E1_RUN_NAME="${E2_FROM_E1_RUN_NAME:-qwen3_8b_grpo_e2_yunwu_semantic_judge_from_e1}"
E2_FROM_E1_STEP="${E2_FROM_E1_STEP:-120}"
E2_FROM_E1_FALLBACK_STEP="${E2_FROM_E1_FALLBACK_STEP:-80}"
E2_FROM_E1_MODEL="${E2_FROM_E1_MODEL:-${RL_DATA_ROOT}/models/qwen3-8b-e2-yunwu-semantic-judge-from-e1-gs${E2_FROM_E1_STEP}}"
BASE_MODEL="${BASE_MODEL:-${RL_DATA_ROOT}/models/qwen3-8b}"

WAIT_FOR_TRAIN="${WAIT_FOR_TRAIN:-true}"
POLL_S="${POLL_S:-300}"
STYLES="${STYLES:-raw-compact:rawcompact}"
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

if [[ "${WAIT_FOR_TRAIN}" == "true" ]]; then
  while pgrep -f "verl.trainer.main_ppo" >/dev/null; do
    echo "[official70] training still running; waiting ${POLL_S}s"
    sleep "${POLL_S}"
  done
fi

if [[ ! -d "${CRITPT_REPO}/.git" ]]; then
  mkdir -p "$(dirname "${CRITPT_REPO}")"
  git clone https://github.com/CritPt-Benchmark/CritPt.git "${CRITPT_REPO}"
fi

merge_fsdp_model() {
  local ckpt_dir="$1"
  local target_model="$2"
  local config_model="$3"

  if [[ ! -d "${ckpt_dir}" ]]; then
    echo "missing checkpoint actor dir: ${ckpt_dir}" >&2
    return 1
  fi
  if [[ -d "${target_model}" ]]; then
    echo "[official70] merged model already exists: ${target_model}"
    return 0
  fi

  local hf_ckpt_dir="${ckpt_dir}/huggingface"
  mkdir -p "${hf_ckpt_dir}"
  find "${config_model}" -maxdepth 1 -type f \
    \( -name "*.json" -o -name "*.txt" -o -name "*.jinja" -o -name "*.model" \) \
    ! -name "model.safetensors.index.json" \
    -exec cp {} "${hf_ckpt_dir}/" \;

  if [[ ! -f "${ckpt_dir}/fsdp_config.json" ]]; then
    local first_shard world_size
    first_shard="$(find "${ckpt_dir}" -maxdepth 1 -type f -name "model_world_size_*_rank_0.pt" | head -1)"
    if [[ -z "${first_shard}" ]]; then
      echo "missing model shard needed to reconstruct fsdp_config.json" >&2
      return 1
    fi
    world_size="$(basename "${first_shard}" | sed -E 's/model_world_size_([0-9]+)_rank_0\.pt/\1/')"
    printf '{\n  "FSDP_version": 1,\n  "world_size": %s\n}\n' "${world_size}" > "${ckpt_dir}/fsdp_config.json"
  fi

  local tmp_model="${target_model}.tmp"
  rm -rf "${tmp_model}"
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${ckpt_dir}" \
    --target_dir "${tmp_model}"
  mv "${tmp_model}" "${target_model}"
}

if [[ ! -d "${E2_CKPT_DIR}" ]]; then
  echo "missing E2 checkpoint actor dir: ${E2_CKPT_DIR}" >&2
  exit 1
fi

merge_fsdp_model "${E2_CKPT_DIR}" "${E2_MODEL}" "${BASE_MODEL}"

E2_FROM_E1_CKPT_DIR="${RL_DATA_ROOT}/checkpoints/${E2_FROM_E1_RUN_NAME}/global_step_${E2_FROM_E1_STEP}/actor"
if [[ ! -d "${E2_FROM_E1_CKPT_DIR}" && -n "${E2_FROM_E1_FALLBACK_STEP}" ]]; then
  E2_FROM_E1_STEP="${E2_FROM_E1_FALLBACK_STEP}"
  E2_FROM_E1_CKPT_DIR="${RL_DATA_ROOT}/checkpoints/${E2_FROM_E1_RUN_NAME}/global_step_${E2_FROM_E1_STEP}/actor"
  E2_FROM_E1_MODEL="${RL_DATA_ROOT}/models/qwen3-8b-e2-yunwu-semantic-judge-from-e1-gs${E2_FROM_E1_STEP}"
fi

if [[ -d "${E2_FROM_E1_CKPT_DIR}" ]]; then
  merge_fsdp_model "${E2_FROM_E1_CKPT_DIR}" "${E2_FROM_E1_MODEL}" "${E1_MODEL}"
else
  echo "[official70] no from-E1 checkpoint found; skipping from-E1 eval"
  E2_FROM_E1_MODEL=""
fi

submit_aa=false
if [[ -f "${RL_DATA_ROOT}/.env.aa" ]]; then
  # shellcheck disable=SC1090
  source "${RL_DATA_ROOT}/.env.aa"
  submit_aa=true
elif [[ -n "${AA_API_KEY:-}" ]]; then
  submit_aa=true
fi

run_one() {
  local label="$1"
  local model="$2"
  local style_pair style suffix out_dir

  for style_pair in ${STYLES}; do
    style="${style_pair%%:*}"
    suffix="${style_pair##*:}"
    out_dir="${RL_DATA_ROOT}/logs/eval/${label}_official70_${suffix}_ctx32768_max4096"
    echo "[official70] generating ${label} style=${style} -> ${out_dir}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_EVAL}" \
    VENV="${VENV}" \
    MODEL="${model}" \
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
        --out "${out_dir}/official_score.json"
    else
      echo "[official70] AA_API_KEY missing; skipped official submission for ${out_dir}"
    fi
  done
}

run_one "qwen3_8b_e1_yunwu_semantic_judge_gs60" "${E1_MODEL}"
run_one "qwen3_8b_e2_yunwu_semantic_judge_base_gs${E2_STEP}" "${E2_MODEL}"
if [[ -n "${E2_FROM_E1_MODEL}" ]]; then
  run_one "qwen3_8b_e2_yunwu_semantic_judge_from_e1_gs${E2_FROM_E1_STEP}" "${E2_FROM_E1_MODEL}"
fi

echo "[official70] done"
