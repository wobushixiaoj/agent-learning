# Agent Learning

这是一个持续迭代的 Agent 与 LLM 机制教材仓库，目标岗位是
高级 Agent 应用或平台工程师。

## 学习原则

```text
先看完整位置
-> 明确本节唯一目标
-> 阅读脚本生成的中文教材
-> 从真实输出建立直觉
-> 用练习检验理解
-> 补齐缺失的知识链
-> 再进入下一层
```

课程只保留可复用的知识结论、推导和练习，不包含个人提问过程、错误回答或学习画像。

## 当前课程

完整顺序见 [学习路线](docs/learning-roadmap.md)。

### Day 1：Transformer 输入表示

直接阅读：[Day 1 教材输出](lessons/transformer/day01-input-representation/OUTPUT.md)

章节说明：[Day 1 README](lessons/transformer/day01-input-representation/README.md)

### Day 2：Self-Attention

直接阅读：[Day 2 教材输出](lessons/transformer/day02-self-attention/OUTPUT.md)

章节说明：[Day 2 README](lessons/transformer/day02-self-attention/README.md)

## 环境

当前教程脚本固定使用：

```text
/opt/anaconda3/bin/python3
```

需要安装 `torch` 和 `tiktoken`。这些环境只用于生成教材；学习者默认直接阅读每课的
`OUTPUT.md`，不需要亲自执行脚本。
