# Pipeline

这份文档说明仓库里的 RL 流程，不展开逐轮流水账。

## 1. 数据构造

数据构造代码主要在：

```text
src/rl_posttrain/critpt_synth/
scripts/data/
```

做过几类数据：

- 程序化 synthetic tasks：题目、参考答案和 verifier 都可控。
- official-style prompts：把短题包装成更像官方 70 题的长背景。
- failure-mined hardcases：从失败 rollout 里挖模型常错类型。
- LLM-assisted specs：让 LLM 参与生成题面和解法规格，再做程序校验。

## 2. Reward

reward 代码主要在：

```text
src/rl_posttrain/critpt_synth/verl_reward*.py
src/rl_posttrain/model_judge/
```

主要尝试过：

- 本地 verifier：稳定便宜，适合快速迭代，但容易和官方语义错位。
- final-answer reward：重点检查最终返回对象。
- length-aware shaping：压制长篇绕圈、没有最终答案、输出截断。
- LLM-as-a-judge：更接近语义判断，但需要严格 rubric，避免奖励猜答案。

## 3. 训练

训练主线：

```text
Qwen3-8B
-> vLLM rollout
-> reward function
-> verl GRPO
-> checkpoint
-> merge
-> eval
```

常看的训练指标：

- `reward`：当前 reward 下是否变好。
- `advantage`：同一题多个 rollout 有没有组内差异。
- `kl`：模型有没有偏离初始策略太远。
- `entropy`：输出分布还有没有探索。
- `response length`：有没有变啰嗦或撞上限。
- `clip ratio`：是否大量输出被截断。

## 4. 评测

评测脚本主要在：

```text
scripts/eval/
scripts/ops/merge_eval_*.sh
```

训练中 reward 变好只是中间信号，最终要看：

- rollout 样本是否真的在解题。
- merged checkpoint 是否稳定输出 `answer()`。
- official70 风格评测是否有准确率变化。

这个项目最重要的经验就是：reward 曲线好看不够，必须看真实输出和官方风格评测。

