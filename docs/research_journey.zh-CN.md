# 实验过程复盘

这份文档记录这个 RL post-training 项目的主要判断过程。它不是完整流水账，而是把实验中几次关键假设、转向和结论整理出来。

## 1. 初始目标

项目一开始的目标是训练 Qwen3-8B 更好地完成 CritPT 风格任务。

这类任务不是普通问答。模型需要阅读较长的科学/数学题面，最后输出一个可执行的 Python `answer()` 函数。评测重点不是解释是否流畅，而是返回对象是否正确。

因此最初的问题被拆成三层：

```text
1. 模型能不能稳定输出正确格式？
2. 模型能不能写出可执行 answer()？
3. 模型能不能真正解官方风格长题？
```

早期实验主要围绕前两层展开。

## 2. 第一阶段：先把格式和执行链路跑通

最早的程序化 synthetic data 用来验证完整训练闭环：

```text
synthetic data -> local verifier -> verl GRPO -> rollout -> checkpoint -> eval
```

这一步的假设是：如果模型先学会 CritPT 的交卷格式，也许 official-style 任务会自然受益。

V13/V14 之后可以看到，模型确实更会输出 `def answer()`，输出也更短、更可执行。但 official70 结果没有提升。

这个阶段得到的结论是：

```text
格式训练有效，但格式不是核心瓶颈。
```

## 3. 第二阶段：从短题转向官方风格长题

官方 70 题的难点不只是 Python 语法，而是从长背景中抽取真正要返回的数学/物理对象。

因此后续实验加入了：

- official-style 长题面包装
- hardcase 数据
- failure-mined 数据
- operator/filter/symbolic focused tasks

V18/V19/V20/V21 这条线的目标，是让训练题更接近官方题的结构。

结果显示，局部 hardcase 能产生 reward 信号，也能修一些特定错误，例如输出过长、候选集合过宽、operator label 不规范等。但 official70 仍然没有被打穿。

这个阶段的结论是：

```text
局部技能可以训练，但 synthetic hardcase 不会自动变成官方题能力。
```

## 4. 第三阶段：从本地 verifier 转向 LLM-as-a-judge

本地 verifier 的优点是稳定、便宜、可重复。缺点是它只会检查我们程序化定义过的答案形式。

为了让 reward 更接近真实语义，E 系列开始引入 LLM-as-a-judge：

- E1 用 LLM 包装题目背景。
- E2 让 LLM 参与生成题目和解法规格。
- E3/E4 收紧 final-answer/code judge。
- E5/E5b 混入 failure-aware curriculum。

这条线的训练 reward 一度更好看，输出格式也更稳定。但合并 checkpoint 后测 official70，准确率仍然没有起来。

这暴露了一个更关键的问题：

```text
LLM judge 如果 rubric 不够严格，也会奖励“看起来像答案”的错误输出。
```

模型可能学会：

- 写出漂亮的 `answer()` 外壳
- 返回看似合理的常数
- 返回占位符列表
- 猜测一个常见表达式

但这不等于真正解题。

## 5. 最重要的负结果

E5b 是最有代表性的负结果。

训练中 reward、长度、格式、KL 都看起来正常，说明 RL 系统本身在工作。但 official-style evaluation 仍然失败。

这个结果推翻了一个早期隐含假设：

```text
训练 reward 变好，不一定意味着目标 benchmark 变好。
```

对于这类任务，reward 必须非常贴近最终评测对象。只让模型“像是在解题”，会诱导出 reward hacking。

## 6. 当前结论

这个项目已经验证了完整 RL post-training 工程链路：

- 数据生成
- reward 设计
- verl GRPO 训练
- rollout 检查
- 曲线监控
- checkpoint 合并
- official-style evaluation

但它没有得到 official70 accuracy 提升。

当前最明确的技术结论是：

```text
瓶颈不是输出格式，也不是单纯训练步数或硬件规模。
瓶颈是训练数据和 reward 与官方题真实语义之间的对齐不足。
```

后续如果继续，应优先改进：

- 更接近官方题分布的数据构造
- 更严格的 LLM-as-a-judge rubric
- 对猜测、占位符、过宽集合的强惩罚
- 更早、更频繁的 official-style evaluation

## 7. 保留价值

虽然 benchmark 没有提升，这轮实验仍然有价值：

```text
它把问题从“模型不会按格式答题”
推进到了
“模型会写答案壳子，但 reward/data 还没有逼它真正解题”。
```

这为下一轮实验限定了更清晰的方向。

