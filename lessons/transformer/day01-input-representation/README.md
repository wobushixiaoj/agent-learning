# Day 1：文本如何变成 Transformer 输入

## 当前位于哪里

```text
原始文本
-> Token
-> Token ID
-> Token 向量 + 位置信息
-> Transformer 输入
-> [下一节：Self-Attention]
```

## 本节只解决什么

1. Token ID 是词表索引，不是语义向量；
2. Token ID 用来查询 Embedding 表中的一行；
3. 相同 Token ID 在同一张表中得到相同初始向量；
4. 位置信息让模型区分相同 Token 出现在不同位置。

本节不讨论 Attention、预测目标、loss 或训练过程。

## 运行

从仓库根目录执行：

```bash
./lessons/transformer/day01-input-representation/run.sh
```

脚本会按中文教程顺序输出，并在每一步给出结论。

## 理解链

```text
Text
-> Tokenizer 切分 Token
-> 词表给每个 Token 一个 ID
-> ID 查询 Token Embedding
-> 加入位置信息
-> 每个 Token 获得 Transformer 输入向量
```
