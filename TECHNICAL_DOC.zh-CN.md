# 8 卡 A100 40G RL 后训练技术文档

更新时间：2026-06-05 05:15 CST

这份文档记录当前远端机器的完整 infra、模型、环境、数据、smoke 结果和后续 RL 启动方式。目标是你从零登录机器后，不需要翻聊天记录也能接上。

## 1. 机器结论

远端机器：

```bash
ssh -i ~/.ssh/id_ed25519 -p 22 root@36.139.133.196
```

已确认硬件：

```text
GPU:       8x NVIDIA A100-PCIE-40GB
Topology:  PHB, no NVLink/NVSwitch
Driver:    550.54.15
CUDA:      12.4
Python:    3.11.12
RAM:       ~629GiB
System:    /dev/sda1, 788G, mounted on /
Data disk: /dev/sdb1, 2.0T, mounted on /data/sdb
```

判断：

- 这台是 A100 PCIe 40G，不是 A100 80G/SXM/NVLink。
- 适合做 `Qwen3-14B` 的完整 posttrain/RL 闭环。
- `Qwen3-32B` 可以做短上下文、低 batch、LoRA/QLoRA/GRPO smoke 和展示型实验，但不适合作为 4-5 天内的全参 RL 保底。

## 2. 单盘布局

硬规则：所有工作产物都放在 `/data/sdb/rl-posttrain`，避免系统盘被模型/checkpoint/log 填满。

```bash
/data/sdb/rl-posttrain
├── code          # 代码，/root/rl-posttrain 是它的软链接
├── models        # qwen3-14b / qwen3-32b / HF cache / ModelScope cache
├── data          # jsonl/parquet 数据
├── checkpoints   # 训练 checkpoint
├── logs          # 下载/训练/安装日志
├── repos         # verl 源码
├── tmp           # pip/tmp/cache
└── venvs         # Python venv
```

登录后固定执行：

```bash
cd /root/rl-posttrain
source configs/hardware/a100_8x40g_pcie.env
source /data/sdb/rl-posttrain/venvs/rl/bin/activate
```

检查目录是否跑偏：

```bash
bash scripts/remote/enforce_single_disk_layout.sh
df -h /data/sdb
```

## 3. 模型状态

已下载完整：

```bash
python scripts/remote/model_inventory.py
```

当前结果：

```text
qwen3-14b
  path: /data/sdb/rl-posttrain/models/qwen3-14b
  size: 27.5GB
  safetensors_files: 8
  config/tokenizer: ok

qwen3-32b
  path: /data/sdb/rl-posttrain/models/qwen3-32b
  size: 61.0GB
  safetensors_files: 17
  config/tokenizer: ok
```

训练配置已经使用本地模型路径：

```bash
configs/experiments/qwen3_14b_grpo_verl.env
configs/experiments/qwen3_32b_grpo_verl_smoke.env
```

## 4. Python / RL 环境

激活：

```bash
source /data/sdb/rl-posttrain/venvs/rl/bin/activate
```

复建环境：

```bash
cd /root/rl-posttrain
bash scripts/remote/bootstrap_a10040g.sh
```

长任务复建：

```bash
bash scripts/remote/run_training_bootstrap_tmux.sh
tail -f /data/sdb/rl-posttrain/logs/infra_bootstrap.log
```

当前关键版本：

```text
torch        2.6.0+cu124
transformers 4.52.4
datasets     2.21.0
accelerate   1.13.0
vllm         0.8.4
verl         0.9.0.dev
ray          2.55.1
tensordict   0.10.0
```

环境验证：

```bash
bash scripts/remote/check_training_env.sh
```

说明：

- `verl` 从 `/data/sdb/rl-posttrain/repos/verl` editable 安装。
- 当前项目也 editable 安装进 venv，这样 `verl` 自定义 reward 可以 import `rl_posttrain`。
- 暂未安装 OpenRLHF/DeepSpeed 主环境；先以 `verl + FSDP + vLLM` 为主线，减少构建风险。

## 5. CritPT 数据与 Reward

seed 数据：

```bash
data/seeds/critpt_seed.jsonl
```

导出 `verl` parquet：

```bash
python scripts/data/export_verl_parquet.py \
  --input data/seeds/critpt_seed.jsonl \
  --train-out /data/sdb/rl-posttrain/data/critpt_train.parquet \
  --val-out /data/sdb/rl-posttrain/data/critpt_val.parquet \
  --repeat 128
```

当前 smoke 数据：

```text
critpt_train.parquet: 128 rows
critpt_val.parquet:   8 rows
```

自定义 reward：

```bash
src/rl_posttrain/critpt/verl_reward.py
```

它从 parquet 的 `extra_info.verifier` 读取 verifier spec，支持：

- exact
- numeric
- symbolic

评分规则：

```text
答案正确: 1.0
格式有 <answer>...</answer> 但答案错: 0.05
无格式或不可解析: 0.0
```

本地/远端 verifier smoke 已通过：

```text
symbolic_equal -> score 1.0
```

## 6. 已通过 Smoke Tests

### 6.1 8 卡 NCCL all-reduce

命令：

```bash
cd /root/rl-posttrain
source configs/hardware/a100_8x40g_pcie.env
source /data/sdb/rl-posttrain/venvs/rl/bin/activate

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

注意：不要用裸 `torchrun --standalone` 作为唯一命令；这台机器 hostname 解析会刷 warning，固定 `master_addr=127.0.0.1` 更稳。

### 6.2 Qwen3-14B vLLM

命令：

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/smoke/vllm_generate.py \
  --model /data/sdb/rl-posttrain/models/qwen3-14b \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.45 \
  --max-model-len 1024
```

结果：

```text
model loading: ~6.89 GiB/GPU
engine init:   ~150s
generate:      ok
```

### 6.3 Qwen3-32B vLLM

命令：

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/smoke/vllm_generate.py \
  --model /data/sdb/rl-posttrain/models/qwen3-32b \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.35 \
  --max-model-len 512
```

结果：

```text
model loading: ~7.64 GiB/GPU
engine init:   ~228s
generate:      ok
```

结论：32B 的低上下文 rollout 侧可用；真正 RL 仍需保守 batch/response/generation。

## 7. 开始 RL

### 7.1 推荐主线：Qwen3-14B GRPO

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

日志：

```bash
tail -f /data/sdb/rl-posttrain/logs/qwen3_14b_grpo_critpt_a10040g.log
```

checkpoint：

```bash
/data/sdb/rl-posttrain/checkpoints/qwen3_14b_grpo_critpt_a10040g
```

### 7.2 Stretch：Qwen3-32B GRPO Smoke

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

32B 初始建议：

```text
max_prompt_length=512
max_response_length=512
rollout_n=2
rollout_tp=8
vllm_gpu_memory_utilization=0.35-0.38
train_batch_size=32 或更小
ppo_mini_batch_size=8 或更小
micro_batch_size_per_gpu=1
```

如果 OOM，优先降：

1. `MAX_RESPONSE_LENGTH`
2. `ROLLOUT_N`
3. `MAX_PROMPT_LENGTH`
4. `TRAIN_BATCH_SIZE`
5. `VLLM_GPU_MEMORY_UTILIZATION`

## 8. 实验记录规范

每个实验必须记录：

```text
run_name
git commit
config path
model path/hash
data path/hash
reward code path
训练日志路径
checkpoint 路径
评测输出路径
```

建议每次启动前：

```bash
git rev-parse HEAD
sha256sum /data/sdb/rl-posttrain/data/critpt_train.parquet
sha256sum /data/sdb/rl-posttrain/data/critpt_val.parquet
```

## 9. 风险清单

- 这台机器没有 NVLink，32B 多卡通信会慢，别把 32B 当唯一保底。
- 当前 `critpt_train.parquet` 是重复 seed，只用于 infra smoke，正式训练必须扩大 synthetic 数据。
- `verl` metadata 与 `numpy 2.2.6` 有版本告警，但实际 import/vLLM smoke 已通过；如果训练中出现 numpy 相关错误，再考虑单独建 `verl-numpy1` 环境，不要直接破坏当前 vLLM 环境。
- 平台临时密码曾经在聊天中出现过；确认 SSH key 可用后，建议在平台重置密码或禁用密码登录。
- `/data/sdb` 是唯一工作盘；不要把模型/checkpoint 写到 `/root` 或 `/`。

## 10. 快速恢复命令

检查机器：

```bash
cd /root/rl-posttrain
source configs/hardware/a100_8x40g_pcie.env
source /data/sdb/rl-posttrain/venvs/rl/bin/activate
nvidia-smi
df -h /data/sdb
python scripts/remote/model_inventory.py
bash scripts/remote/check_training_env.sh
```

查看任务：

```bash
tmux ls || true
ps -ef | grep -E "torchrun|verl|vllm" | grep -v grep || true
```

清理异常训练进程前先确认没有要保留的任务：

```bash
pkill -f verl.trainer.main_ppo
pkill -f vllm
```
