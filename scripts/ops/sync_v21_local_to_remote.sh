#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-root@<GPU_HOST>}"
PORT="${PORT:-22}"
IDENTITY="${IDENTITY:-$HOME/.ssh/id_ed25519}"
REMOTE_ROOT="${REMOTE_ROOT:-/data/sdb/rl-posttrain/code}"

rsync -az -e "ssh -i ${IDENTITY} -p ${PORT}" \
  configs/experiments/qwen3_8b_grpo_v21_operator_precision_from_v20_gs40_n8.env \
  "${REMOTE}:${REMOTE_ROOT}/configs/experiments/"

rsync -az -e "ssh -i ${IDENTITY} -p ${PORT}" \
  src/rl_posttrain/critpt_synth/v21_operator_precision.py \
  src/rl_posttrain/critpt_synth/verl_reward_v21_operator_precision.py \
  "${REMOTE}:${REMOTE_ROOT}/src/rl_posttrain/critpt_synth/"

rsync -az -e "ssh -i ${IDENTITY} -p ${PORT}" \
  scripts/data/build_v21_operator_precision.py \
  "${REMOTE}:${REMOTE_ROOT}/scripts/data/"

rsync -az -e "ssh -i ${IDENTITY} -p ${PORT}" \
  scripts/ops/merge_eval_v20_step40.sh \
  scripts/ops/merge_eval_v21_step30.sh \
  scripts/ops/prepare_v21_operator_precision_remote.sh \
  scripts/ops/sync_v21_local_to_remote.sh \
  "${REMOTE}:${REMOTE_ROOT}/scripts/ops/"

rsync -az -e "ssh -i ${IDENTITY} -p ${PORT}" \
  tests/test_v21_operator_precision.py \
  "${REMOTE}:${REMOTE_ROOT}/tests/"

rsync -az -e "ssh -i ${IDENTITY} -p ${PORT}" \
  docs/experiments/qwen3_8b_grpo_v20_focused_hard_from_v19_gs60_n8.zh-CN.md \
  docs/experiments/qwen3_8b_grpo_v21_operator_precision_from_v20_gs40_n8.zh-CN.md \
  docs/experiments/rl_rollout_curve_index.zh-CN.md \
  "${REMOTE}:${REMOTE_ROOT}/docs/experiments/"

echo "synced V20/V21 local changes to ${REMOTE}:${REMOTE_ROOT}"
