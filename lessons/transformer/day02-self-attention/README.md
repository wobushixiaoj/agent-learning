# Day 2：Self-Attention 如何读取上下文

本文件只提供章节导航。完整教材来自脚本的真实运行结果：

[直接阅读 Day 2 教材](OUTPUT.md)

## 本页目标

理解 GPT 的一个 Transformer 层在前向计算中，如何让每个 Token 只读取自己和前文，
并生成包含上下文的新向量。

这是训练和推理共用的 Causal Self-Attention 机制，不是训练或推理的完整流程。

## 章节结构

1. 学习目的与训练/推理边界
2. 全页知识流程图
3. 最小示例、Shape 与符号表
4. 从输入 X 得到 Q、K、V
5. `QK^T` 的四个格子
6. 跟踪一行：缩放、Causal Mask、Softmax
7. 对 V 加权求和
8. 进阶：为什么缩放
9. 进阶：Softmax 的具体计算
10. 一页总结

## 示例约定

只使用 2 个 Token，每个向量有 3 个分量：

```text
Agent = [1,2,3]
tools = [3,2,1]
```

为了隔离 Attention 数据流，本页暂时令 `Q=K=V=X`。真实模型使用
`Q=XW_Q`、`K=XW_K`、`V=XW_V`，三者数值通常不同。

## 核心记号

| 符号 | Shape | 含义 |
| --- | --- | --- |
| `q_i/k_i/v_i` | `(3,)` | 单个 Token 在当前 head 的向量 |
| `Q/K/V` | `(2,3)` | 全部 Token 的向量按行组成的矩阵 |
| `QK^T` | `(2,2)` | Token 两两匹配分数 |
| `A` | `(2,2)` | Mask 与 Softmax 后的读取权重 |
| `O=A@V` | `(2,3)` | 融合上下文后的输出 |

## 参考

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
