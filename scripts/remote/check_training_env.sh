#!/usr/bin/env bash
set -euo pipefail

source configs/hardware/a100_8x40g_pcie.env
source /data/sdb/rl-posttrain/venvs/rl/bin/activate

python - <<'PY'
import importlib
import torch

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())

for name in ["transformers", "datasets", "accelerate", "vllm", "verl", "ray", "deepspeed"]:
    try:
        mod = importlib.import_module(name)
        print(name, getattr(mod, "__version__", "ok"))
    except Exception as exc:
        print(name, "FAILED", repr(exc))
        raise
PY

