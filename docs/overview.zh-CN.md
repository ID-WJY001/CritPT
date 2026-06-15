# 项目概览

这个仓库是一个面向 CritPT 风格科学题的 RL post-training 实验项目。

目标任务不是普通问答，而是让模型阅读科学/数学背景题，生成可执行的 Python `answer()` 函数，并返回正确对象。项目围绕 Qwen3-8B 搭建了从数据构造、reward 设计、GRPO 训练、rollout 检查到官方风格评测的闭环。

## 做了什么

- 构造 CritPT-style synthetic data，包括程序化题目、官方风格长题面、failure-mined hardcases 和 LLM-assisted teacher specs。
- 实现本地 verifier reward、final-answer reward、length-aware shaping 和 LLM-as-a-judge reward。
- 用 verl GRPO 跑多轮 post-training，并保存曲线、rollout 样本、checkpoint merge/eval 脚本。
- 用 official70 风格评测检查训练是否真的迁移到公开题。

## 结果

结论比较直接：

```text
工程闭环是完整的。
模型学到了格式、长度控制和可执行 answer() 外壳。
但后期 official-style V/E runs 的 official70 accuracy 仍然是 0。
```

这说明当前瓶颈不在“会不会输出代码”，而在数据和 reward 是否真正对齐官方题的解题语义。训练 reward 好看并不等于 benchmark 提升，模型可能只是学会了迎合 synthetic judge。

## 技术结论

```text
完整 RL post-training 闭环已经跑通。
synthetic reward 到 official-style benchmark 的迁移失败是主要问题。
```

后续改进重点应该放在数据和 reward 的语义对齐上，而不是优先增加训练步数或硬件规模。
