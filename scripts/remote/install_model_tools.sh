#!/usr/bin/env bash
set -euo pipefail

source configs/hardware/a100_8x40g_pcie.env

VENV=${RL_DATA_ROOT}/venvs/model-tools
mkdir -p "${RL_DATA_ROOT}/venvs" "${LOG_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

uv venv --python 3.11 "${VENV}"
source "${VENV}/bin/activate"

uv pip install --python "${VENV}/bin/python" --upgrade pip setuptools wheel
uv pip install --python "${VENV}/bin/python" "huggingface_hub[cli]>=0.30.0" "modelscope>=1.18.0"

python - <<'PY'
import huggingface_hub
import modelscope
print("huggingface_hub", huggingface_hub.__version__)
print("modelscope", modelscope.__version__)
PY

echo "activate with: source ${VENV}/bin/activate"
