#!/usr/bin/env bash
set -euo pipefail

ROOT=${RL_DATA_ROOT:-/data/sdb/rl-posttrain}
VENV=${ROOT}/venvs/rl

mkdir -p "${ROOT}"/{venvs,logs,tmp}

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

uv venv --python 3.11 "${VENV}"
source "${VENV}/bin/activate"

uv pip install --python "${VENV}/bin/python" --upgrade pip setuptools wheel

# CUDA 12.4 driver can run cu121/cu124 wheels. cu121 is often the safest PyTorch index.
uv pip install --python "${VENV}/bin/python" --index-url https://download.pytorch.org/whl/cu121 \
  torch torchvision torchaudio

uv pip install --python "${VENV}/bin/python" -r requirements/remote.txt

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("gpu count", torch.cuda.device_count())
PY

echo "activate with: source ${VENV}/bin/activate"
