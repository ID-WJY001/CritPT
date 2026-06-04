#!/usr/bin/env bash
set -euo pipefail

nvidia-smi
nvidia-smi topo -m
df -h
free -h
python3 --version
command -v git && git --version
command -v tmux && tmux -V
command -v docker && docker --version
command -v uv && uv --version

