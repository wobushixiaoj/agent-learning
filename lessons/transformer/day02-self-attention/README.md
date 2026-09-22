# Day 2：“打”如何通过 Self-Attention 读取“我喜欢打”

本文件只提供章节导航。完整教材来自脚本的真实运行结果：

[直接阅读 Day 2 教材](OUTPUT.md)

## 本页目标

沿用 `我喜欢打 -> 网球` 的推理案例，只跟踪最后一个 Token `打`，理解它如何读取
`我`、`喜欢` 和自己，并生成包含上下文的新向量。

本课限定为推理 Prefill 阶段，不讨论训练、Loss 或参数更新。

## 章节结构

1. 当前步骤的输入与输出
2. 用人话理解 Q、K、V
3. `打` 分别匹配三个 Key
4. 缩放、Mask、Softmax
5. 按比例读取三个 Value
6. Attention 输出接下来去哪里
7. 技术附录：矩阵并行、Mask 与缩放

## 示例约定

使用 3 个 Token，每个 Q/K/V 向量有 3 个分量：

```text
[我] [喜欢] [打]
```

具体数字是为手算挑选的教学值。真实模型使用 `Q=XW_Q`、`K=XW_K`、`V=XW_V`
生成向量，且维度远大于 3。

## 核心记号

| 符号 | Shape | 含义 |
| --- | --- | --- |
| `q_i/k_i/v_i` | `(3,)` | 单个 Token 在当前 head 的向量 |
| `Q/K/V` | `(3,3)` | 三个 Token 的向量按行组成的矩阵 |
| `QK^T` | `(3,3)` | Token 两两匹配分数 |
| `A` | `(3,3)` | Mask 与 Softmax 后的读取权重 |
| `O=A@V` | `(3,3)` | 融合上下文后的输出 |

## 参考

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
