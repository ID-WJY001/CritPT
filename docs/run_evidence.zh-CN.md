# 实验记录与复现边界

这个仓库不提交完整训练产物。完整 rollout、日志、checkpoint 和官方题逐题输出体积较大，也可能包含远端路径和临时运行信息，因此只保留精选曲线、summary、训练配置和复现脚本。

## 公开保留的证据

```text
artifacts/curated/
configs/experiments/
scripts/ops/
scripts/eval/
docs/experiments.zh-CN.md
```

其中：

- `artifacts/curated/` 保存代表性训练曲线和少量 summary。
- `configs/experiments/` 保存代表性训练配置，覆盖 V13、V14、V19、V20、E4、E5b、E6。
- `scripts/ops/` 保存远端训练、merge、plot、rollout inspection 的操作脚本。
- `scripts/eval/` 保存本地和 official-style evaluation 脚本。

## 未提交的完整产物

以下内容不进入 Git：

```text
完整 rollouts
完整 training logs
verl checkpoints
merged model weights
完整 official70 generated outputs
大规模 parquet/arrow 训练数据
私有 API key 和远端机器信息
```

这些内容可以支撑更完整的审计，但不适合直接放进代码仓库。

## 运行环境

主要远端训练在 8 卡 A100 40GB PCIe 机器上完成。仓库中保留的是可复用的训练配置和脚本，不绑定某一台具体机器。

典型训练链路：

```text
build data
-> launch verl GRPO
-> inspect rollouts
-> plot metrics
-> merge checkpoint
-> run official-style eval
```

## 配置差异

代表配置位于 `configs/experiments/`：

| 配置 | 作用 |
| --- | --- |
| `qwen3_8b_grpo_v13_official_code_format_signal_n4.env` | 官方代码格式训练 |
| `qwen3_8b_grpo_v14_compact_exec_n4.env` | 紧凑可执行答案训练 |
| `qwen3_8b_grpo_v19_failure_mined_from_v18_gs80_n8.env` | failure-mined 数据 |
| `qwen3_8b_grpo_v20_focused_hard_from_v19_gs60_n8.env` | focused hardcases |
| `qwen3_8b_grpo_e4_official_style_final_answer_from_e3.env` | official-style final-answer 训练 |
| `qwen3_8b_grpo_e5b_failure_aware_curriculum_from_e4.env` | failure-aware curriculum |
| `qwen3_8b_grpo_e6_teacher_specs_strict_judge_from_e4.env` | strict teacher-spec 方案 |

## 本地验证

本地单测不依赖模型权重：

```bash
uv run pytest
```

远端训练和 LLM judge 需要私有模型路径、GPU 环境和 API key。

