#!/usr/bin/env bash
set -euo pipefail

ROOT=${RL_DATA_ROOT:-/data/sdb/rl-posttrain}

mkdir -p "${ROOT}"/{models,hf_cache,data,checkpoints,logs,tmp}
mkdir -p "${ROOT}"/models/{hf_cache,modelscope_cache}

ln -sfn "${ROOT}" /root/rl-posttrain-artifacts

echo "prepared ${ROOT}"
df -h "${ROOT}"

