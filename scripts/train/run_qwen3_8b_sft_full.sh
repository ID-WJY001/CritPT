#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: bash scripts/train/run_qwen3_8b_sft_full.sh CONFIG.env" >&2
  exit 2
fi

CONFIG=$1
source "${CONFIG}"

mkdir -p "${OUTPUT_DIR}" "${LOG_ROOT}"

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29601}

torchrun \
  --nnodes=1 \
  --nproc_per_node="${N_GPUS_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  scripts/train/sft_full.py \
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
  --fsdp-transformer-layer "${FSDP_TRANSFORMER_LAYER:-Qwen3DecoderLayer}" \
  --save-strategy "${SAVE_STRATEGY:-no}" \
  ${SKIP_FINAL_SAVE:+--skip-final-save} \
  2>&1 | tee "${LOG_ROOT}/${RUN_NAME}.log"
