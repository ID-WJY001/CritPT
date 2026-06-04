# 8卡 A100 RL 后训练 Infra 接手手册

这份文档是给你无痛接手当前机器和代码用的。先看它，不用在聊天记录里翻细节。

## 1. 当前机器结论

- SSH：`ssh -i ~/.ssh/id_ed25519 -p 22 root@36.139.133.196`
- 机器：`8x NVIDIA A100-PCIE-40GB`
- 拓扑：`PHB`，也就是 PCIe 互联，没有 NVLink/NVSwitch
- 内存：约 `629GiB`
- 系统盘：`/`，约 `788G`
- 数据盘：`/data/sdb`，约 `2T`
- 代码目录：`/data/sdb/rl-posttrain/code`，软链接为 `/root/rl-posttrain`
- 模型/日志/checkpoint 根目录：`/data/sdb/rl-posttrain`

这台机器适合：

- 稳妥主线：`Qwen/Qwen3-14B` 做 CritPT-style SFT/GRPO/RL 闭环
- 冲刺尝试：`Qwen/Qwen3-32B` 做 QLoRA/GRPO smoke
- 不建议：全参 32B PPO/GRPO

## 2. 目录约定

硬规则：**所有工作文件都放在 `/data/sdb/rl-posttrain` 这块 2T 数据盘下**。`/root/rl-posttrain` 只是软链接，不存真实代码。

```bash
/root/rl-posttrain                         # 代码软链接
/data/sdb/rl-posttrain/code                # 真实代码目录
/data/sdb/rl-posttrain/models              # 模型权重
/data/sdb/rl-posttrain/models/hf_cache     # Hugging Face cache
/data/sdb/rl-posttrain/models/modelscope_cache
/data/sdb/rl-posttrain/data                # 训练/评测数据
/data/sdb/rl-posttrain/checkpoints         # checkpoint
/data/sdb/rl-posttrain/logs                # 日志
/data/sdb/rl-posttrain/venvs               # Python 环境
```

所有大文件都放 `/data/sdb`，不要放根目录 `/`。

如果不确定目录有没有跑偏，执行：

```bash
bash scripts/remote/enforce_single_disk_layout.sh
```

## 3. 每次登录先做什么

```bash
ssh -i ~/.ssh/id_ed25519 -p 22 root@36.139.133.196
cd /root/rl-posttrain
source configs/hardware/a100_8x40g_pcie.env
df -h /data/sdb
nvidia-smi
tmux ls || true
```

如果有训练正在跑，先进入对应 tmux：

```bash
tmux attach -t model-download
tmux attach -t rl
```

## 4. 已准备的脚本

主机检查：

```bash
bash scripts/remote/check_host.sh
```

准备数据盘目录：

```bash
bash scripts/remote/prepare_data_disk.sh
```

安装基础 Python / PyTorch / 训练依赖：

```bash
bash scripts/remote/bootstrap_a10040g.sh
```

只安装模型下载工具：

```bash
bash scripts/remote/install_model_tools.sh
source /data/sdb/rl-posttrain/venvs/model-tools/bin/activate
```

下载模型：

```bash
bash scripts/remote/download_models.sh qwen3-14b
bash scripts/remote/download_models.sh qwen3-32b
```

当前国内机器默认 `MODEL_PROVIDER=modelscope`，不自动跳 HF 全量下载。若某个小文件失败，优先用 ModelScope 单文件补齐；确认 HF 可用时再临时执行：

```bash
MODEL_PROVIDER=hf bash scripts/remote/download_models.sh qwen3-14b
```

查看模型状态：

```bash
python scripts/remote/model_inventory.py
```

CritPT seed 数据和 verifier：

```bash
PYTHONPATH=src python scripts/data/make_seed_dataset.py --out data/seeds/critpt_seed.jsonl
PYTHONPATH=src python scripts/eval/eval_jsonl.py --data data/seeds/critpt_seed.jsonl
```

8 卡 NCCL smoke：

```bash
torchrun --standalone --nproc_per_node=8 scripts/smoke/torch_all_reduce.py
```

## 5. 模型策略

当前模型下载优先级：

1. `Qwen/Qwen3-14B`：主线，最稳，先拿到完整 RL 闭环
2. `Qwen/Qwen3-32B`：stretch，先做加载/短上下文/QLoRA/GRPO smoke

不要一开始就追求 32B 大 batch 或长上下文。A100 40G PCIe 上的初始限制：

```bash
max_prompt_length=512
max_response_length=512
num_generations=2
vLLM tensor_parallel_size=8
vLLM gpu_memory_utilization=0.35-0.45
LoRA rank=16
ZeRO stage=3
```

如果 32B OOM，立刻降级到 14B 主线，不要硬耗卡时。

## 6. 训练入口

14B verl GRPO 模板：

```bash
bash scripts/train/run_verl_grpo.sh configs/experiments/qwen3_14b_grpo_verl.env
```

32B verl smoke 模板：

```bash
bash scripts/train/run_verl_grpo.sh configs/experiments/qwen3_32b_grpo_verl_smoke.env
```

32B OpenRLHF QLoRA/GRPO smoke 模板：

```bash
bash scripts/train/run_openrlhf_grpo_qwen3_32b_smoke.sh
```

这些训练脚本是启动模板，第一次真正跑之前需要先完成：

- PyTorch/vLLM/verl/OpenRLHF 安装
- 模型下载
- CritPT 数据转成框架需要的格式
- 8 卡 all-reduce smoke
- vLLM 单次 generate smoke

## 7. tmux 使用

新开任务：

```bash
tmux new -s rl
```

后台保持运行：按 `Ctrl-b`，再按 `d`。

重新进入：

```bash
tmux attach -t rl
```

下载模型建议使用：

```bash
tmux new -s model-download
```

## 8. 风险和底线

- 这台是 A100 PCIe 40G，不是 80G，也不是 SXM
- 32B RL 是 stretch，不是保底
- 保底成果应该放在 Qwen3-14B 或 Qwen3-8B 的完整闭环上
- 所有训练日志、数据 hash、config、git commit 都要保留
- 临时密码已经出现在聊天里，建议确认公钥可用后去平台重置/禁用密码登录

## 9. 当前下一步

1. 远端安装模型下载依赖
2. 下载 `Qwen/Qwen3-14B`
3. 下载或尝试下载 `Qwen/Qwen3-32B`
4. 跑模型 inventory 检查
5. 装训练依赖并跑 torch all-reduce
