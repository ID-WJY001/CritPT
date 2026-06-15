#!/usr/bin/env bash
set -euo pipefail

ROOT=${RL_DATA_ROOT:-/data/sdb/rl-posttrain}
LOG=${LOG:-${ROOT}/logs/qwen3_8b_grpo_critpt_a10040g_smoke.log}

if [ ! -f "${LOG}" ]; then
  echo "log not found yet: ${LOG}"
  echo "start first: bash scripts/ops/start_8b_one_step.sh"
  exit 1
fi

tail -f "${LOG}"
