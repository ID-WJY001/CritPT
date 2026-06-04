#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: bash scripts/tmux_start.sh SESSION_NAME COMMAND..." >&2
  exit 2
fi

SESSION=$1
shift

tmux new-session -d -s "${SESSION}" "$*"
tmux attach -t "${SESSION}"
