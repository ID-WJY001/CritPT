# 当前状态

更新时间：2026-06-05 05:15 CST

## 结论

这台 8 卡 A100 40G 机器已经完成 RL 前期 infra 搭建，可以开始做 CritPT-style 的 `verl` GRPO smoke 和后续数据/奖励实验。

稳妥主线：`Qwen3-14B`。
大参数 stretch：`Qwen3-32B`，已通过低上下文 vLLM TP=8 加载/生成 smoke，但全参 RL 不建议作为保底。

## 远端入口

```bash
ssh -i ~/.ssh/id_ed25519 -p 22 root@36.139.133.196
cd /root/rl-posttrain
source configs/hardware/a100_8x40g_pcie.env
source /data/sdb/rl-posttrain/venvs/rl/bin/activate
```

重要目录：

```bash
/data/sdb/rl-posttrain/code          # 真实代码目录
/root/rl-posttrain                   # 指向 code 的软链接
/data/sdb/rl-posttrain/models        # 模型权重和缓存
/data/sdb/rl-posttrain/data          # parquet/jsonl 数据
/data/sdb/rl-posttrain/checkpoints   # checkpoint
/data/sdb/rl-posttrain/logs          # 日志
/data/sdb/rl-posttrain/venvs/rl      # RL Python 环境
```

## 已验证硬件

- GPU：`8x NVIDIA A100-PCIE-40GB`
- 拓扑：GPU 间为 `PHB`，无 NVLink/NVSwitch
- Driver：`550.54.15`
- CUDA：`12.4`
- Python：`3.11.12`
- RAM：约 `629GiB`
- 数据盘：`/data/sdb`，`2.0T`，当前约 `103G` 已用，约 `1.8T` 可用
- GPU 当前空闲：8 张卡均 `0 MiB` 显存占用

## 已下载模型

```text
/data/sdb/rl-posttrain/models/qwen3-14b
  size: 27.5GB
  safetensors: 8
  config/tokenizer: ok

/data/sdb/rl-posttrain/models/qwen3-32b
  size: 61.0GB
  safetensors: 17
  config/tokenizer: ok
```

复查命令：

```bash
python scripts/remote/model_inventory.py
```

## 已安装训练环境

关键版本：

```text
torch:        2.6.0+cu124
transformers: 4.52.4
datasets:     2.21.0
accelerate:   1.13.0
vLLM:         0.8.4
verl:         0.9.0.dev
ray:          2.55.1
tensordict:   0.10.0
```

环境检查通过：

```bash
bash scripts/remote/check_training_env.sh
```

已知非阻塞告警：

- `verl` metadata 声明 `numpy<2.0.0`，当前 vLLM 栈使用 `numpy 2.2.6`。实际 `verl/vLLM/ray` import 和 smoke 已通过，先不降级。
- vLLM 在 A100 PCIe 多卡上提示 custom allreduce unsupported，这是 PCIe-only 多卡常见 warning，不影响当前 smoke。

## 已生成数据

```text
/data/sdb/rl-posttrain/data/critpt_train.parquet  # 128 rows, smoke repeat
/data/sdb/rl-posttrain/data/critpt_val.parquet    # 8 rows, smoke repeat
```

生成命令：

```bash
python scripts/data/export_verl_parquet.py \
  --input data/seeds/critpt_seed.jsonl \
  --train-out /data/sdb/rl-posttrain/data/critpt_train.parquet \
  --val-out /data/sdb/rl-posttrain/data/critpt_val.parquet \
  --repeat 128
```

注意：这批数据是 infra smoke 用的重复 seed，不是正式训练数据。正式实验要扩大 synthetic pipeline，并记录数据 hash。

## 已通过 smoke

8 卡 NCCL all-reduce：

```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
torchrun --nnodes=1 --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port=29500 \
  scripts/smoke/torch_all_reduce.py
```

结果：

```text
NCCL version 2.21.5+cuda12.4
all_reduce=36.0 expected=36.0
```

Qwen3-14B vLLM smoke：

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/smoke/vllm_generate.py \
  --model /data/sdb/rl-posttrain/models/qwen3-14b \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.45 \
  --max-model-len 1024
```

结果：模型加载、KV cache、warmup、生成均成功。权重加载约 `6.89GiB/GPU`，engine init 约 `150s`。

Qwen3-32B vLLM smoke：

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/smoke/vllm_generate.py \
  --model /data/sdb/rl-posttrain/models/qwen3-32b \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.35 \
  --max-model-len 512
```

结果：模型加载、KV cache、warmup、生成均成功。权重加载约 `7.64GiB/GPU`，engine init 约 `228s`。

## 可以开始的下一步

14B 主线：

```bash
tmux new -s rl-14b
cd /root/rl-posttrain
source configs/hardware/a100_8x40g_pcie.env
source /data/sdb/rl-posttrain/venvs/rl/bin/activate
bash scripts/train/run_verl_grpo.sh configs/experiments/qwen3_14b_grpo_verl.env
```

只跑 one-step smoke：

```bash
TOTAL_TRAINING_STEPS=1 bash scripts/train/run_verl_grpo.sh configs/experiments/qwen3_14b_grpo_verl.env
```

32B stretch：

```bash
tmux new -s rl-32b-smoke
cd /root/rl-posttrain
source configs/hardware/a100_8x40g_pcie.env
source /data/sdb/rl-posttrain/venvs/rl/bin/activate
bash scripts/train/run_verl_grpo.sh configs/experiments/qwen3_32b_grpo_verl_smoke.env
```

只跑 one-step smoke：

```bash
TOTAL_TRAINING_STEPS=1 bash scripts/train/run_verl_grpo.sh configs/experiments/qwen3_32b_grpo_verl_smoke.env
```

建议先跑 14B 小步闭环，把 reward、失败样本保存、数据合成分布控制跑顺，再扩大 32B。
