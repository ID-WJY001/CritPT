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
export PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}

mkdir -p "${ROOT}"/{venvs,logs,tmp,models,repos}

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

rm -rf "${VENV}"
uv venv --python 3.11 "${VENV}"
source "${VENV}/bin/activate"

uv pip install --python "${VENV}/bin/python" --upgrade pip setuptools wheel

# Install torch from the regular PyPI wheel family via a China-friendly mirror.
# The Linux torch 2.6.0 wheel pulls CUDA 12 runtime packages and works with the
# node's 550 driver / CUDA 12.4 runtime.
"${VENV}/bin/python" -m pip install \
  --timeout 180 \
  --retries 10 \
  torch==2.6.0

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
uv pip install --python "${VENV}/bin/python" --no-deps -e .

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
