#!/usr/bin/env bash
set -euo pipefail

ROOT=${RL_DATA_ROOT:-/data/sdb/rl-posttrain}
VENV=${ROOT}/venvs/rl
REPOS=${ROOT}/repos

export UV_LINK_MODE=copy
export PIP_CACHE_DIR=${ROOT}/tmp/pip_cache
export TMPDIR=${ROOT}/tmp
export HF_HOME=${ROOT}/models/hf_cache
export HF_HUB_CACHE=${ROOT}/models/hf_cache/hub
export MODELSCOPE_CACHE=${ROOT}/models/modelscope_cache
export TORCH_HOME=${ROOT}/models/torch_cache

mkdir -p "${ROOT}"/{venvs,logs,tmp,models,repos}

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

uv venv --python 3.11 "${VENV}"
source "${VENV}/bin/activate"

uv pip install --python "${VENV}/bin/python" --upgrade pip setuptools wheel

# CUDA 12.4 driver can run cu121 wheels. Install torch only; vision/audio are
# unnecessary for this project and add fragile extra network index lookups.
"${VENV}/bin/python" -m pip install \
  --timeout 180 \
  --retries 10 \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch

"${VENV}/bin/python" -m pip install \
  --timeout 180 \
  --retries 10 \
  -r requirements/remote.txt

if [ ! -d "${REPOS}/verl/.git" ]; then
  git clone https://github.com/verl-project/verl "${REPOS}/verl"
else
  git -C "${REPOS}/verl" pull --ff-only || true
fi

# verl docs recommend installing from source. We install editable without deps
# after the runtime deps above, so vLLM/Torch stay pinned by this environment.
uv pip install --python "${VENV}/bin/python" --no-deps -e "${REPOS}/verl"

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("gpu count", torch.cuda.device_count())
try:
    import vllm
    print("vllm", vllm.__version__)
except Exception as exc:
    print("vllm import failed", repr(exc))
try:
    import verl
    print("verl import ok")
except Exception as exc:
    print("verl import failed", repr(exc))
PY

echo "activate with: source ${VENV}/bin/activate"
