#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: bash scripts/train/run_qwen3_8b_sft_lora.sh CONFIG.env" >&2
  exit 2
fi

CONFIG=$1
source "${CONFIG}"

mkdir -p "${OUTPUT_DIR}" "${LOG_ROOT}"

torchrun --nproc_per_node="${N_GPUS_PER_NODE}" scripts/train/sft_lora.py \
  --model "${MODEL_NAME}" \
  --train-data "${TRAIN_SFT_DATA}" \
  --val-data "${VAL_SFT_DATA}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-length "${MAX_LENGTH}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --logging-steps "${LOGGING_STEPS}" \
  --save-steps "${SAVE_STEPS}" \
  --eval-steps "${EVAL_STEPS}" \
  --max-steps "${MAX_STEPS}" \
  2>&1 | tee "${LOG_ROOT}/${RUN_NAME}.log"
