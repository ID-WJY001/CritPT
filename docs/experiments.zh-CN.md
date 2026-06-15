# 实验摘要

这里保留压缩版实验记录。

## V 系列

| 阶段 | 目的 | 结果 |
| --- | --- | --- |
| V13 | 训练官方风格 `def answer()` 输出 | 格式变好，official70 仍为 0 |
| V14 | 压短输出，强调可执行代码 | 输出更干净，但没有官方提分 |
| V17/V18 | 修 hardcase，包装成长题面 | reward 信号更真实，迁移仍失败 |
| V19 | failure mining | 找到模型常错类型，但效率不够 |
| V20/V21 | operator/filter/symbolic focused hardcases | 局部修补有效，不是通用解法 |

V 系列结论：

```text
模型可以学会格式、可执行性和局部模板。
但 synthetic/local verifier reward 没有自然迁移到 official70。
```

## E 系列

| 阶段 | 目的 | 结果 |
| --- | --- | --- |
| E1 | 用 LLM 包装题目背景 | 数据更自然，但仍偏简单 |
| E2 | 让 LLM 参与生成题目和解法规格 | 难度提升，但 official acc 没起来 |
| E3 | 严格比较候选代码和参考答案 | reward 更关注最终对象 |
| E4 | 官方风格 final-answer 训练 | 格式更稳，official70 仍为 0 |
| E5/E5b | failure-aware curriculum | 训练 reward 一度好看，official70 仍为 0 |
| E6 | strict teacher specs 方案 | 代码准备，未形成正式训练结果 |

E 系列结论：

```text
LLM-as-a-judge 是必要方向，但 rubric 必须严格。
如果 judge 过宽，模型会学会写漂亮外壳、猜常数或返回占位符。
```

## 总结

这轮实验是一次合格的 RL 工程练习，但不是一次成功的 benchmark 提分。  
下一步如果继续，不应该先加卡或加 step，而应该先提高数据和 reward 对官方题语义的对齐程度。
