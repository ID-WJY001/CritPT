#!/usr/bin/env bash
set -euo pipefail

source configs/hardware/a100_8x40g_pcie.env

TARGET=${1:-all}
MODELS_DIR=${RL_DATA_ROOT}/models
LOG_DIR=${LOG_ROOT}
MODEL_PROVIDER=${MODEL_PROVIDER:-modelscope}
mkdir -p "${MODELS_DIR}" "${HF_HOME}" "${MODELSCOPE_CACHE}" "${LOG_DIR}"

download_hf() {
  local model_id=$1
  local local_dir=$2
  echo "[download] hf ${model_id} -> ${local_dir}"
  hf download "${model_id}" \
    --local-dir "${local_dir}" \
    --max-workers 8
}

download_modelscope() {
  local model_id=$1
  local local_dir=$2
  echo "[download] modelscope ${model_id} -> ${local_dir}"
  python - "$model_id" "$local_dir" <<'PY'
import sys
from modelscope.hub.snapshot_download import snapshot_download

model_id = sys.argv[1]
local_dir = sys.argv[2]
snapshot_download(model_id, local_dir=local_dir)
PY
}

download_one() {
  local slug=$1
  local model_id=$2
  local local_dir="${MODELS_DIR}/${slug}"
  mkdir -p "${local_dir}"

  echo "[download] start ${model_id}"
  echo "[download] log ${LOG_DIR}/download_${slug}.log"

  {
    date
    if [ "${MODEL_PROVIDER}" = "hf" ]; then
      download_hf "${model_id}" "${local_dir}"
    elif command -v modelscope >/dev/null 2>&1 || python - <<'PY' >/dev/null 2>&1
import modelscope
PY
    then
      download_modelscope "${model_id}" "${local_dir}"
    else
      download_hf "${model_id}" "${local_dir}"
    fi
    date
    du -sh "${local_dir}" || true
  } 2>&1 | tee "${LOG_DIR}/download_${slug}.log"
}

case "${TARGET}" in
  qwen3-14b)
    download_one qwen3-14b Qwen/Qwen3-14B
    ;;
  qwen3-32b)
    download_one qwen3-32b Qwen/Qwen3-32B
    ;;
  all)
    download_one qwen3-14b Qwen/Qwen3-14B
    download_one qwen3-32b Qwen/Qwen3-32B
    ;;
  *)
    echo "unknown target: ${TARGET}" >&2
    echo "usage: bash scripts/remote/download_models.sh [qwen3-14b|qwen3-32b|all]" >&2
    exit 2
    ;;
esac
