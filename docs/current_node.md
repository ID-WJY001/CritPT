# Current GPU Node

## SSH

- Host: `36.139.133.196`
- Port: `22`
- User: `root`
- Preferred auth: `~/.ssh/id_ed25519` public key in `/root/.ssh/authorized_keys`

Do not commit passwords. Rotate the temporary password after key auth is proven.

## Observed Hardware

- OS: Ubuntu 22.04.3 LTS
- GPUs: `8x NVIDIA A100-PCIE-40GB`
- Driver: `550.54.15`
- CUDA runtime from driver: `12.4`
- Topology: all GPU pairs show `PHB`
- RAM: `629GiB`
- System disk: `/dev/sda1`, `788G`, mounted on `/`
- Data disk: `/dev/sdb1`, `2.0T`, mounted on `/data/sdb`
- Python: `3.11.12`
- Installed tools: `git`, `tmux`, `docker`, `uv`
- Missing at first check: `nvcc`

## Consequence

This is an A100 PCIe node, not A100 SXM/NVLink. Use it as:

- main line: `Qwen3-14B` GRPO/SFT closed loop;
- stretch: `Qwen3-32B` QLoRA/GRPO smoke;
- avoid: full-parameter 32B RL.

