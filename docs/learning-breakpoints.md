# 学习断点索引

本文件记录学习者真实提出的问题，以及问题如何反向修改教材。它不是错题流水账，
而是知识链缺口与课程改进的索引。

## Transformer 输入表示

| ID | 原问题或原理解 | 断点类型 | 缺失的逻辑关系 | 教材处理 |
| --- | --- | --- | --- | --- |
| TIN-001 | Token 是否先变成向量，再由向量计算成一个数字？ | concept | Token ID 与 Embedding 的方向相反 | Day 1 增加 `Token -> ID -> 查表得到向量` |
| TIN-002 | Token Embedding 会不会结合 Token 自己的位置？ | concept | Token Embedding 与位置表示属于两个阶段 | Day 1 增加同 ID 查表结果对比 |
| TIN-003 | Position Embedding 是否直接表达主谓宾关系？ | precision | 位置表示提供顺序线索，语法关系由模型学习 | Day 1 增加“位置线索不等于语法标签”说明 |
| TIN-004 | 为什么 Day 1 突然从 Transformer 原理跳到训练过程？ | context-induced | 课程没有区分架构输入和训练目标 | 将 next-token prediction 移到 Day 5，并新增课程地图 |

## Self-Attention

| ID | 原问题或原理解 | 断点类型 | 缺失的逻辑关系 | 教材处理 |
| --- | --- | --- | --- | --- |
| ATT-001 | `(3, 4)` 和三行四列的数字是什么关系？ | prerequisite | Shape 是矩阵尺寸；行对应 Token，列对应向量维度 | Day 2 步骤 1 增加行列、总元素数和 Token 映射 |
| ATT-002 | 步骤 3 太模糊，看不明白 `QK^T` 的计算过程 | prerequisite | 教材从矩阵直接跳到分数，缺少转置、点积、缩放和形状变化 | Day 2 步骤 3 展开完整手算，并加入 `(3,4) @ (4,3) -> (3,3)` |

## 状态说明

- `concept`：概念方向或机制混淆，需要保留修正链路与迁移自测；
- `precision`：主线正确但措辞会导致更高阶误解；
- `context-induced`：教材缺少前置上下文，不归因成学习者能力问题；
- `prerequisite`：课程假设了尚未建立的数学或编程直觉，需要补前置解释。
