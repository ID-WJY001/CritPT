#!/usr/bin/env bash
set -euo pipefail

ROOT=${RL_DATA_ROOT:-/data/sdb/rl-posttrain}

mkdir -p "${ROOT}"/{code,models,data,checkpoints,logs,tmp,venvs}
mkdir -p "${ROOT}"/models/{hf_cache,modelscope_cache,torch_cache}
mkdir -p /root/.cache

ln -sfn "${ROOT}/code" /root/rl-posttrain
ln -sfn "${ROOT}" /root/rl-posttrain-artifacts

# Keep large default caches on the data disk even if a future shell forgets to
# source configs/hardware/a100_8x40g_pcie.env.
if [ ! -e /root/.cache/huggingface ]; then
  ln -s "${ROOT}/models/hf_cache" /root/.cache/huggingface
fi

if [ ! -e /root/.cache/modelscope ]; then
  ln -s "${ROOT}/models/modelscope_cache" /root/.cache/modelscope
fi

if [ ! -e /root/.cache/torch ]; then
  ln -s "${ROOT}/models/torch_cache" /root/.cache/torch
fi

chown -R root:root "${ROOT}/code"

echo "single-disk layout is enforced under ${ROOT}"
echo
ls -ld /root/rl-posttrain /root/rl-posttrain-artifacts "${ROOT}" "${ROOT}/code"
echo
df -h "${ROOT}"

