# RL 指标与失败诊断

这个项目中，训练曲线只作为中间信号。最终判断必须结合 rollout 样本和 official-style evaluation。

## 关键指标

### Reward

reward 表示当前奖励函数下模型输出的得分。它不是最终 benchmark accuracy。

本项目里多次出现：

```text
training reward 变好，但 official70 accuracy 不变。
```

这说明 reward 可以被模型迎合，尤其当 synthetic task 或 LLM judge rubric 和目标 benchmark 不完全一致时。

### Advantage

GRPO 依赖同一 prompt 下多个 rollout 的相对差异。如果一个 group 内所有 rollout 得分一样，advantage 接近 0，训练信号会变弱。

这类现象通常说明：

- rollout 数太低或采样不够多样；
- reward 太粗，无法区分好坏输出；
- prompt/data 太简单或太死板；
- 模型在该任务上只会输出同一种模式。

### KL

KL 用来观察当前策略偏离参考策略的程度。

KL 过高可能说明模型被 reward 推得太远。KL 很低也不一定好，可能说明模型主要在学格式，没有形成新解题能力。

### Entropy

entropy 反映采样分布的探索程度。entropy 过低时，rollout 容易塌到固定模板，组内差异不足；entropy 过高时，输出可能不稳定。

### Clip / Length

clip ratio 和 response length 用来判断输出是否撞上长度上限。

早期实验里，长 CoT 容易导致输出被截断；后续 length-aware shaping 和 compact prompt 缓解了这个问题。但长度问题解决后，official70 仍然没有提升，说明核心瓶颈不是 token budget。

## Reward hacking 迹象

以下现象说明模型可能在迎合 reward，而不是真正解题：

- 输出结构正确，但返回占位符。
- 返回看似合理的常数或空列表。
- 生成很长的候选集合，试图覆盖答案。
- 代码可执行，但核心对象是猜的。
- judge 分数上升，但 official-style eval 不变。

E5b 的结果是典型例子：训练指标看起来健康，格式也更稳定，但 official-style evaluation 没有改善。

## official70 为 0 的解释

当前结论不是“RL 没有作用”，而是：

```text
RL 学到了格式和局部行为，但没有学到官方题所需的真实语义筛选能力。
```

官方风格任务的难点在于：

- 长题面中定位真正问题；
- 返回精确的数学/物理对象；
- 避免返回过宽候选集合；
- 避免用模板答案或常见常数猜测；
- 处理符号表达式、operator label、filter/list 等精确结构。

## 下一步设计

后续实验应优先改进数据和 reward，而不是先增加训练步数：

1. 构造更接近 official70 分布的题目。
2. 让 LLM teacher 生成问题、解法规格和参考答案，但保留程序化 sanity check。
3. LLM judge rubric 明确惩罚占位符、猜测、过宽集合、非 canonical label。
4. 每轮训练更早跑 official-style eval，而不是只看训练 reward。
5. 对 rollout group 做 advantage 方差检查，避免无梯度训练。

