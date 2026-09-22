# Agent Learning

这是一个由真实学习问题持续改进的 Agent 与 LLM 机制教材仓库，目标岗位是
高级 Agent 应用或平台工程师。

## 学习原则

```text
先看完整位置
-> 明确本节唯一目标
-> 运行中文教程脚本
-> 从真实输出建立直觉
-> 记录理解断点
-> 把断点补回教材
-> 再进入下一层
```

学习者提出的问题不是对话附注，而是教材需要补齐的知识链。所有问题统一登记在
[学习断点索引](docs/learning-breakpoints.md)，并同步更新对应教材或脚本。

## 当前课程

完整顺序见 [学习路线](docs/learning-roadmap.md)。

### Day 1：Transformer 输入表示

```bash
./lessons/transformer/day01-input-representation/run.sh
```

教材：[Day 1 README](lessons/transformer/day01-input-representation/README.md)

### Day 2：Self-Attention

```bash
./lessons/transformer/day02-self-attention/run.sh
```

教材：[Day 2 README](lessons/transformer/day02-self-attention/README.md)

## 环境

当前教程脚本固定使用：

```text
/opt/anaconda3/bin/python3
```

需要安装 `torch` 和 `tiktoken`。脚本会直接输出中文教程，不要求先阅读源码。
