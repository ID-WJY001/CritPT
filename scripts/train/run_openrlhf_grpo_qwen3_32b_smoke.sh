#!/usr/bin/env bash
set -euo pipefail

source configs/experiments/qwen3_32b_grpo_openrlhf_smoke.env

mkdir -p "${CHECKPOINT_ROOT}/${RUN_NAME}" "${LOG_ROOT}"

if ! python3 - <<'PY' >/dev/null 2>&1
import importlib
importlib.import_module("openrlhf.cli.train_ppo_ray")
PY
then
  echo "OpenRLHF is not installed in this environment." >&2
  echo "Install it from the official repo before running this smoke script." >&2
  exit 1
fi

ray stop --force >/dev/null 2>&1 || true
ray start --head --num-gpus 8

python3 -m openrlhf.cli.train_ppo_ray \
  --pretrain "${MODEL_NAME}" \
  --save_path "${CHECKPOINT_ROOT}/${RUN_NAME}" \
  --train_file "${TRAIN_DATA}" \
  --eval_file "${VAL_DATA}" \
  --prompt_key prompt \
  --max_len "${MAX_PROMPT_LENGTH}" \
  --generate_max_len "${MAX_RESPONSE_LENGTH}" \
  --zero_stage "${ZERO_STAGE}" \
  --bf16 \
  --load_in_4bit \
  --lora_rank "${LORA_RANK}" \
  --advantage_estimator grpo \
  --n_samples_per_prompt "${NUM_GENERATIONS}" \
  --vllm_tensor_parallel_size "${VLLM_TP_SIZE}" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --gradient_checkpointing \
  2>&1 | tee "${LOG_ROOT}/${RUN_NAME}.log"
