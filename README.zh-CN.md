# RL 后训练实验仓库

这是一个面向 CritPT 风格科学题的 RL post-training 实验仓库。

目标任务是让模型阅读科学/数学背景题，生成可执行的 Python `answer()` 函数，并返回正确对象。仓库包含数据构造、reward 设计、GRPO 训练配置、rollout 检查、checkpoint 合并和官方风格评测脚本。

最重要的结论也放在前面：

```text
训练和评测闭环可运行。
模型在格式、长度控制和可执行 answer() 结构上有改善。
后期 official-style V/E runs 的 official70 accuracy 仍为 0。
主要瓶颈是数据/reward 与目标 benchmark 语义没有充分对齐。
```

## 文档

1. [docs/overview.zh-CN.md](docs/overview.zh-CN.md)：项目概览和结论。
2. [docs/pipeline.zh-CN.md](docs/pipeline.zh-CN.md)：数据、reward、训练和评测流程。
3. [docs/experiments.zh-CN.md](docs/experiments.zh-CN.md)：V/E 实验摘要。
4. [docs/research_journey.zh-CN.md](docs/research_journey.zh-CN.md)：实验过程复盘。
5. [docs/run_evidence.zh-CN.md](docs/run_evidence.zh-CN.md)：实验记录和复现边界。
6. [docs/rl_diagnostics.zh-CN.md](docs/rl_diagnostics.zh-CN.md)：RL 指标和失败诊断。
7. [artifacts/README.md](artifacts/README.md)：精选曲线和结果证据。

## 仓库里有什么

```text
src/rl_posttrain/
  critpt/              早期 schema、verifier、reward、eval 基础模块
  critpt_synth/        合成题生成器、本地 verifier reward、V/E 数据逻辑
  model_judge/         OpenAI-compatible LLM judge 客户端和 reward wrapper

scripts/
  data/                构造 V/E 训练数据
  eval/                本地评测、官方风格评测、提交包分析
  ops/                 远端运行、画图、merge、rollout 检查
  remote/              GPU 机器初始化脚本
  train/               SFT/GRPO 启动脚本

configs/experiments/  每轮实验的 env 配置
docs/                 项目说明
artifacts/curated/    训练曲线和 summary
tests/                生成器、verifier、judge wrapper 的单元测试
```

## 技术关键词

- 模型：Qwen3-8B 为主
- 训练：GRPO / verl / vLLM rollout
- 数据：程序合成题、官方风格包装题、失败挖掘 hardcase、LLM teacher specs
- reward：本地 final-answer verifier、语义代码 judge、LLM-as-a-judge
- 监控：reward、advantage、KL、entropy、length、clip、rollout 样本
- 评测：heldout synthetic test、官方 70 题风格评测

## 本地复现

本地单测不需要模型权重，也不需要 API key：

```bash
uv run pytest
```

真正的远端训练需要 GPU 机器、模型权重和私有 API key。  
复制 `.env.example` 到私有 `.env` 后再填密钥，不要提交 `.env`。

## 注意事项

- 不提交 API key、密码、私有 `.env`。
- 不提交模型权重、checkpoint、完整 rollout dump、parquet 大数据。
- 只保留精选曲线、summary 和文档。
- 真实远端地址用 `<GPU_HOST>` 这类占位符表达。
