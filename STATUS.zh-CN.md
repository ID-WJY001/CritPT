# 当前状态

更新时间：2026-06-05 03:30 CST

## 远端机器

- SSH：`ssh -i ~/.ssh/id_ed25519 -p 22 root@36.139.133.196`
- 工作根目录：`/data/sdb/rl-posttrain`
- 代码目录：`/data/sdb/rl-posttrain/code`
- 便捷入口：`/root/rl-posttrain -> /data/sdb/rl-posttrain/code`
- 模型目录：`/data/sdb/rl-posttrain/models`

## 已完成

- 单盘布局已完成：代码、模型、环境、日志、checkpoint 都在 `/data/sdb/rl-posttrain`
- `Qwen/Qwen3-14B` 已下载完整
  - 路径：`/data/sdb/rl-posttrain/models/qwen3-14b`
  - 大小：约 `27.5GB`
  - `safetensors` 分片数：`8`
  - `config/tokenizer/vocab` 已补齐
- 模型下载工具环境已安装
  - 路径：`/data/sdb/rl-posttrain/venvs/model-tools`

## 正在进行

- `Qwen/Qwen3-32B` 正在后台下载
  - tmux：`model-download-32b`
  - 日志：`/data/sdb/rl-posttrain/logs/download_qwen3-32b.log`
  - 路径：`/data/sdb/rl-posttrain/models/qwen3-32b`
  - 当前大小：约 `26.7GB`
  - 当前 `safetensors` 分片数：`3/17`
  - `config.json` 已有，tokenizer 相关文件尚未完整落盘

## 资源状态

- 数据盘：`/data/sdb` 已用约 `55GB`，剩余约 `1.8T`
- GPU：8 张 A100 当前均空闲，显存使用 `0 MiB`

查看进度：

```bash
tmux attach -t model-download-32b
```

退出 tmux 但不中断下载：按 `Ctrl-b`，再按 `d`。

轻量检查：

```bash
cd /root/rl-posttrain
source configs/hardware/a100_8x40g_pcie.env
python3 scripts/remote/model_inventory.py
tail -80 /data/sdb/rl-posttrain/logs/download_qwen3-32b.log
```

## 下一步

1. 安装训练环境：`bash scripts/remote/run_training_bootstrap_tmux.sh`
2. 等 `Qwen3-32B` 完整落盘，若有小文件失败，按 14B 的方式单文件补齐。
3. 跑 8 卡通信测试：`torchrun --standalone --nproc_per_node=8 scripts/smoke/torch_all_reduce.py`
4. 用 14B 先做 vLLM generate smoke。
5. 再决定是否尝试 32B QLoRA/GRPO smoke。
