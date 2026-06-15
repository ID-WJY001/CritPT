#!/usr/bin/env bash
set -euo pipefail

ROOT=${RL_DATA_ROOT:-/data/sdb/rl-posttrain}
CODE=${ROOT}/code
SESSION=${SESSION:-rl-8b-smoke}
CONFIG=${CONFIG:-configs/experiments/qwen3_8b_grpo_verl_smoke.env}
LOG=${ROOT}/logs/qwen3_8b_grpo_critpt_a10040g_smoke.log

cd "${CODE}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  echo "attach: tmux attach -t ${SESSION}"
  echo "tail:   bash scripts/ops/tail_8b_log.sh"
  exit 0
fi

tmux new-session -d -s "${SESSION}" \
  "cd ${CODE} && source ${ROOT}/venvs/rl/bin/activate && ray stop --force >/tmp/${SESSION}_ray_stop.log 2>&1 || true; cd ${CODE} && source ${ROOT}/venvs/rl/bin/activate && source configs/hardware/a100_8x40g_pcie.env && VLLM_USE_V1=${VLLM_USE_V1:-1} TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1} bash scripts/train/run_verl_grpo.sh ${CONFIG}"

echo "started tmux session: ${SESSION}"
echo "attach: tmux attach -t ${SESSION}"
echo "tail:   bash scripts/ops/tail_8b_log.sh"
echo "log:    ${LOG}"
