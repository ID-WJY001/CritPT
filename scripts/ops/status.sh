#!/usr/bin/env bash
set -euo pipefail

ROOT=${RL_DATA_ROOT:-/data/sdb/rl-posttrain}
CODE=${ROOT}/code
VENV=${ROOT}/venvs/rl

cd "${CODE}"

echo "== Where =="
pwd
git log --oneline -1 || true
git status --short || true

echo
echo "== GPU =="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo
echo "== Disk =="
df -h /data/sdb

echo
echo "== Background jobs =="
tmux ls || true
ps -ef | grep -E "verl.trainer|torchrun|vllm|modelscope|huggingface" | grep -v grep || true

echo
echo "== Models =="
python3 scripts/remote/model_inventory.py

echo
echo "== Data =="
ls -lh "${ROOT}/data" || true

echo
echo "== Env =="
source "${VENV}/bin/activate"
python - <<'PY'
import importlib
for name in ["torch", "transformers", "vllm", "verl", "ray"]:
    mod = importlib.import_module(name)
    print(name, getattr(mod, "__version__", "ok"))
PY

echo
echo "OK: if GPUs are 0 MiB and models/env are present, start with:"
echo "bash scripts/ops/start_8b_one_step.sh"
