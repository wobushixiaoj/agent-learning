# 学习路线

## 总目标

不是从头训练大模型，而是建立高级 Agent 工程师需要的 LLM 机制直觉，能够解释
模型行为、诊断系统问题，并在成本、延迟、准确率和可靠性之间做工程取舍。

## Transformer 主线

```text
文本
-> Token 与 Token ID
-> Token Embedding + 位置信息
-> Self-Attention 读取上下文
-> Multi-Head Attention 读取不同关系
-> Residual + LayerNorm + FFN 组成一层
-> 多层堆叠形成上下文化表示
-> 输出 logits 与候选 Token 概率
-> next-token prediction 与训练 loss
-> 推理时逐 Token 生成与 KV Cache
```

## 课程边界

| 阶段 | 只解决的问题 | 暂不讨论 |
| --- | --- | --- |
| Day 1 | 文本如何变成 Transformer 输入向量 | Attention、训练目标 |
| Day 2 | 一个 Token 如何读取其他 Token | 多头、参数训练 |
| Day 3 | 一层 Transformer 为什么需要多头、残差、归一化和 FFN | 多层训练 |
| Day 4 | 多层结果如何变成词表 logits | loss 与优化器 |
| Day 5 | shifted target、交叉熵和 next-token prediction | 推理加速 |
| Day 6 | 自回归生成、KV Cache、延迟与显存 | Agent 系统设计 |

## 与 Agent 岗位的连接

```text
Token 数量 -> 上下文成本和延迟
Attention 与 KV Cache -> 长上下文吞吐和显存
位置编码 -> 长文本外推与上下文边界
logits 与采样 -> Agent 决策稳定性
训练目标与能力边界 -> 为什么 next-token prediction 不等于可靠推理
```
