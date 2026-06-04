# RL Posttrain Infra

中文版接手手册见 [README.zh-CN.md](README.zh-CN.md).

This repo is the launchpad for a CritPT-style post-training project on a rented
8-GPU node. The current target node is:

- `8x NVIDIA A100-PCIE-40GB`
- Ubuntu 22.04
- Driver CUDA runtime 12.4
- 629GiB RAM
- System disk: 788G at `/`
- Data disk: 2T at `/data/sdb`

The machine is good enough for a serious 8B/14B RL closed loop and a cautious
32B QLoRA/GRPO smoke run. Because it is PCIe-only (`PHB` topology, no
NVLink/NVSwitch), full-parameter 32B RL is intentionally out of scope.

## Recommended Run Order

1. Prepare the data disk and workspace.

```bash
bash scripts/remote/prepare_data_disk.sh
```

2. Check the host.

```bash
bash scripts/remote/check_host.sh
```

3. Bootstrap Python/training dependencies.

```bash
bash scripts/remote/bootstrap_a10040g.sh
```

4. Generate a tiny seed dataset and verify the evaluator locally.

```bash
python scripts/data/make_seed_dataset.py --out data/seeds/critpt_seed.jsonl
python scripts/eval/eval_jsonl.py --data data/seeds/critpt_seed.jsonl
```

5. Run the all-reduce smoke test on the GPU node.

```bash
torchrun --standalone --nproc_per_node=8 scripts/smoke/torch_all_reduce.py
```

6. Start with `Qwen3-14B` GRPO for the stable result, then try the guarded
`Qwen3-32B` smoke config.

```bash
bash scripts/train/run_verl_grpo.sh configs/experiments/qwen3_14b_grpo_verl.env
bash scripts/train/run_openrlhf_grpo_qwen3_32b_smoke.sh
```

## Hardware Policy

- Stable result: `Qwen/Qwen3-14B` GRPO.
- Stretch result: `Qwen/Qwen3-32B` QLoRA/GRPO smoke.
- Avoid: full-parameter 32B PPO/GRPO on 8x A100 40G PCIe.
- Keep checkpoints, models, and logs on `/data/sdb`, not `/`.

## Remote Editing

Use Git as the source of truth. The GPU machine can run hotfixes directly, but
every code change must be committed and pushed back so nothing is lost when the
instance is released.

See [docs/current_node.md](docs/current_node.md) for the currently connected
node facts.
