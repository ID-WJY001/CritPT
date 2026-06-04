#!/usr/bin/env bash
set -euo pipefail

SESSION=${1:-infra-bootstrap}
LOG=/data/sdb/rl-posttrain/logs/infra_bootstrap.log

tmux has-session -t "${SESSION}" 2>/dev/null && {
  echo "tmux session already exists: ${SESSION}"
  tmux ls
  exit 0
}

tmux new-session -d -s "${SESSION}" \
  "cd /root/rl-posttrain && bash scripts/remote/bootstrap_a10040g.sh 2>&1 | tee ${LOG}"

echo "started ${SESSION}"
echo "log: ${LOG}"
tmux ls

