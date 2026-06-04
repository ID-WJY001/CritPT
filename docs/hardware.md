# Hardware Notes

## Current Node Verdict

The rented node exposes 8 `NVIDIA A100-PCIE-40GB` GPUs. `nvidia-smi topo -m`
shows `PHB` between all GPU pairs, which means there is no NVLink/NVSwitch
fabric. This is still useful, but the training plan should avoid heavy
cross-GPU communication.

## Practical Model Targets

| Target | Feasibility | Notes |
| --- | --- | --- |
| Qwen3-8B | Easy | Good for fast reward/data/debug loops. |
| Qwen3-14B | Recommended | Strong stable RL result for this hardware. |
| Qwen3-32B | Stretch | Use QLoRA/LoRA, short contexts, tiny rollouts. |
| Kimi/MoE giants | Not recommended | Too much engineering risk for a 4-5 day sprint. |

## Defaults

- Data root: `/data/sdb/rl-posttrain`
- Model cache: `/data/sdb/rl-posttrain/models`
- Checkpoints: `/data/sdb/rl-posttrain/checkpoints`
- Logs: `/data/sdb/rl-posttrain/logs`
- vLLM memory utilization for 32B smoke: `0.35-0.45`
- GRPO generations for 32B smoke: `2`

