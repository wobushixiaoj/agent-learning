from __future__ import annotations

import math

import torch


TOKENS = ["Agent", "tools"]


def section(title: str) -> None:
    print(f"\n## {title}\n")


def subsection(title: str) -> None:
    print(f"\n### {title}\n")


def code_block(value: object) -> None:
    print("```text")
    print(value)
    print("```")


def print_matrix(name: str, matrix: torch.Tensor) -> None:
    print(f"\n**{name}**，形状：`{tuple(matrix.shape)}`")
    code_block(matrix.detach().numpy().round(3))


def print_token_rows(tokens: list[str], matrix: torch.Tensor) -> None:
    print("\n把矩阵的每一行和 Token 对上：")
    for token, row in zip(tokens, matrix.tolist()):
        formatted = ", ".join(f"{value:.1f}" for value in row)
        print(f"- `{token}` -> `[{formatted}]`")


def print_pair_table(name: str, matrix: torch.Tensor) -> None:
    print(f"\n**{name}**（行=读取者，列=信息来源）：\n")
    print("| 读取者 \\ 来源 | " + " | ".join(TOKENS) + " |")
    print("| --- | " + " | ".join("---:" for _ in TOKENS) + " |")
    for token, row in zip(TOKENS, matrix.tolist()):
        values = " | ".join(f"{value:.3f}" for value in row)
        print(f"| {token} | {values} |")


def print_lesson_scope() -> None:
    section("1. 学习目的与边界")
    print("> **本页目标：** 理解 GPT 的一个 Transformer 层在一次前向计算中，")
    print("> 每个 Token 如何只读取自己和前文，并生成包含上下文的新向量。")
    print("\n对 Agent 工程师来说，学习它不是为了手算矩阵，而是为了理解：")
    print("- LLM 为什么能根据上下文改变 Token 表示；")
    print("- GPT 为什么从左到右生成，不能读取未来答案；")
    print("- KV Cache、长上下文成本和推理延迟从哪里来。")
    print("\nCausal Self-Attention 是训练和推理共用的**前向计算机制**：")
    print("\n| 场景 | 因果约束如何出现 |")
    print("| --- | --- |")
    print("| 训练 | 整段 Token 并行计算，用 Mask 遮住未来位置，防止答案泄漏 |")
    print("| 推理 Prefill | 整段 Prompt 并行计算并建立 KV Cache，同样遵守 Mask |")
    print("| 推理 Decode | 一次只处理新 Token，K/V 只有过去和当前，通常无需构造完整 Mask |")
    print("\n本页示例最接近训练或 Prefill 的并行前向计算，但不讨论：")
    print("- 训练专属的 label、loss、反向传播和参数更新；")
    print("- 推理专属的采样、生成循环和 KV Cache 实现。")


def print_knowledge_map() -> None:
    section("2. 全页知识流程图")
    print("```mermaid")
    print("flowchart TD")
    print('    X["输入矩阵 X<br/>2 个 Token × 3 个分量<br/>Shape: (2,3)"]')
    print('    P["生成 Q / K / V<br/>真实模型: XW_Q、XW_K、XW_V<br/>本节简化: Q=K=V=X"]')
    print('    M["完整矩阵 Q、K、V<br/>每个 Shape: (2,3)"]')
    print('    R["取第 i 行<br/>单 Token 向量 q_i、k_i、v_i<br/>每个 Shape: (3,)"]')
    print('    S["批量两两匹配 QK^T<br/>每个 q_i 与所有 k_j 点积<br/>Shape: (2,2)"]')
    print('    D["缩放<br/>除以 sqrt(d_k)<br/>控制分数幅度"]')
    print('    C["Causal Mask<br/>禁止读取未来 Token"]')
    print('    A["Softmax 权重 A<br/>正数且每行和为 1<br/>Shape: (2,2)"]')
    print('    W["对 V 加权求和<br/>A @ V"]')
    print('    O["上下文化输出 O<br/>Shape: (2,3)"]')
    print("    X --> P --> M")
    print('    M -->|"第 i 行"| R')
    print('    M -->|"Q、K"| S')
    print("    S --> D --> C --> A --> W --> O")
    print('    M -->|"V"| W')
    print("```")


def main() -> None:
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0],  # Agent
            [3.0, 2.0, 1.0],  # tools
        ]
    )

    # Identity projections keep the first example hand-calculable.
    query = x
    key = x
    value = x

    raw_scores = query @ key.T
    scale = math.sqrt(query.shape[-1])
    scaled_scores = raw_scores / scale
    unscaled_tools_weights = torch.softmax(raw_scores[1], dim=-1)
    small_scale_weights = torch.softmax(torch.tensor([1.0, 2.0]), dim=-1)
    large_scale_weights = torch.softmax(torch.tensor([10.0, 20.0]), dim=-1)
    causal_mask = torch.triu(
        torch.ones(len(TOKENS), len(TOKENS), dtype=torch.bool), diagonal=1
    )
    masked_scores = scaled_scores.masked_fill(causal_mask, float("-inf"))
    attention_weights = torch.softmax(masked_scores, dim=-1)
    contextual_vectors = attention_weights @ value

    print_lesson_scope()
    print_knowledge_map()

    section("3. 最小示例与统一记号")
    print("本页只使用 2 个 Token，每个 Token 用 3 个数字表示：")
    print_matrix("输入矩阵 X", x)
    print_token_rows(TOKENS, x)
    print("\n数字 `1、2、3` 只是方便手算的占位值，没有预设语义。")

    subsection("3.1 Shape 先读数学定义，再读业务语义")
    print("`X.shape = (2,3)` 的严格定义是 **2 行 3 列**。")
    print("在本例布局中，再把行列解释为：")
    print("- 2 行 = 2 个 Token；")
    print("- 3 列 = 每个 Token 向量有 3 个分量。")
    print("\n> Shape 本身只描述行列；Token 语义来自当前数据布局。")

    subsection("3.2 本页符号表")
    print("| 符号 | Shape | 含义 |")
    print("| --- | --- | --- |")
    print("| `x_i` | `(3,)` | 单个 Token 的输入向量 |")
    print("| `X` | `(2,3)` | 两个 Token 的输入矩阵 |")
    print("| `q_i/k_i/v_i` | `(3,)` | 单个 Token 在当前 head 的 Q/K/V 向量 |")
    print("| `Q/K/V` | `(2,3)` | 全部 Token 的 q/k/v 按行组成的矩阵 |")
    print("| `S=QK^T` | `(2,2)` | 两个 Token 两两之间的匹配分数 |")
    print("| `A` | `(2,2)` | Mask 和 Softmax 后的读取权重 |")
    print("| `O=A@V` | `(2,3)` | 融合上下文后的输出矩阵 |")

    section("4. 从输入 X 得到 Q、K、V")
    subsection("4.1 真实模型与本页简化")
    print("真实模型使用三组不同参数：")
    code_block("Q = X @ W_Q\nK = X @ W_K\nV = X @ W_V")
    print("因此真实模型里的 Q、K、V 数值通常不同。为了只学习 Attention 数据流，")
    print("本页暂时省略投影矩阵，令 `Q=K=V=X`：")
    code_block(
        "Q = K = V = [[1,2,3],\n"
        "             [3,2,1]]"
    )

    subsection("4.2 单 Token 向量与完整矩阵")
    code_block(
        "q_Agent = [1,2,3]    # Q 的第 1 行，Shape: (3,)\n"
        "q_tools = [3,2,1]    # Q 的第 2 行，Shape: (3,)\n\n"
        "Q = [[1,2,3],\n"
        "     [3,2,1]]        # 完整 Q，Shape: (2,3)"
    )
    print("K、V 的组织方式相同。`(2,3)` 的第一维已经同时包含两个 Token，")
    print("所以单个 Token 只有一个 3 维小写 q，不会各自拥有一个 `(2,3)` 的 Q。")
    print("\n在本页限定的“一条序列、一个 layer、一个 head”内有一个大写 Q。")
    print("真实模型的每个 layer/head 各有对应的 Q，工程上通常收进更高维张量。")

    subsection("4.3 逐 Token 与矩阵并行是同一计算")
    code_block(
        "逐 Token：q_Agent = x_Agent @ W_Q\n"
        "           q_tools = x_tools @ W_Q\n\n"
        "矩阵并行：Q = X @ W_Q"
    )
    print("矩阵乘法逐行处理 X，因此 Q 的第 i 行就是 `q_i`。")
    print("真实代码通常直接计算 Q，不一定先逐个创建 q 再调用 `stack`。")

    section("5. 用 Q 和 K 计算两两匹配分数")
    print("概念上，每个 Token 的 `q_i` 都要与所有 Token 的 `k_j` 做点积。")
    print("2 个 Query × 2 个 Key，因此共有 4 次计算：")
    code_block(
        "① q_Agent · k_Agent\n"
        "   [1,2,3] · [1,2,3] = 1x1 + 2x2 + 3x3 = 14\n\n"
        "② q_Agent · k_tools\n"
        "   [1,2,3] · [3,2,1] = 1x3 + 2x2 + 3x1 = 10\n\n"
        "③ q_tools · k_Agent\n"
        "   [3,2,1] · [1,2,3] = 3x1 + 2x2 + 1x3 = 10\n\n"
        "④ q_tools · k_tools\n"
        "   [3,2,1] · [3,2,1] = 3x3 + 2x2 + 1x1 = 14"
    )
    print_pair_table("原始分数 S = QK^T", raw_scores)

    subsection("5.1 为什么矩阵写法需要 K.T")
    print_matrix("K.T", key.T)
    code_block(
        "Q (2,3) @ K.T (3,2) -> S (2,2)\n\n"
        "Q @ K.T = [[14,10],\n"
        "           [10,14]]"
    )
    print("- 中间的 `3`：每次点积使用两个 3 维向量；")
    print("- 外侧的 `2×2`：2 个读取者分别匹配 2 个信息来源。")
    print("\n矩阵写法一次得到全部 4 个分数，和逐 Token 计算完全相同。")

    subsection("5.2 缩放不是归一化：它在 Softmax 前控制尖锐程度")
    print("你对 Softmax 的理解没问题：它确实把结果变成总和为 1 的比例。")
    print("但 **总和归一化不等于消除输入尺度**。Softmax 内部有指数，分数整体放大后，")
    print("最大项会占据更大的比例。")
    small_low, small_high = small_scale_weights.tolist()
    large_low, large_high = large_scale_weights.tolist()
    print("\n两组分数的大小关系都是“第二项是第一项的 2 倍”，但 Softmax 结果不同：")
    print("\n| Softmax 输入 | Softmax 输出 |")
    print("| --- | --- |")
    print(f"| `[1,2]` | `[{small_low:.3f},{small_high:.3f}]` |")
    print(f"| `[10,20]` | `[{large_low:.6f},{large_high:.6f}]` |")
    print("\n`[10,20]` 仍然会被归一化成和为 1，但几乎变成 `[0,1]`。")
    print("因此缩放不是替 Softmax 做归一化，而是在 Softmax 前调节它有多“果断”。")
    print("这也常被称为调节 Softmax 的**温度**。")

    print("\n为什么维度变大时需要调节？点积是 `d_k` 个乘积的求和：")
    code_block("q · k = q_1k_1 + q_2k_2 + ... + q_dk k_dk")
    print("可以把每一项想成一次可能为正、也可能为负的小步。它们会部分抵消，")
    print("但维度越多，最后偏离 0 的典型距离仍会变大，像随机游走：")
    print("- 4 步后的典型偏移量约为 `sqrt(4)=2`；")
    print("- 100 步后的典型偏移量约为 `sqrt(100)=10`。")
    print("\n所以即使 Query 和 Key 没有变得“更相关”，只因维度从 4 增加到 100，")
    print("点积分数的典型幅度也可能放大约 5 倍，Softmax 就会无故变得更尖锐。")

    print("\n对应的标准统计表达是：如果各维分量大致独立、均值为 0、方差为 1，那么：")
    code_block(
        "q · k 的方差 ≈ d_k\n"
        "q · k 的标准差 ≈ sqrt(d_k)"
    )
    print("除以 `sqrt(d_k)`，正好把随维度增长的典型幅度拉回同一量级。")
    print("如果除以 `d_k`，分数通常会被压得过小；`sqrt(d_k)` 对应的是标准差增长速度。")
    print(f"\n本例 `d_k=3`，所以缩放因子 `sqrt(3)={scale:.3f}`。")
    print("注意这里的 3 是向量维度，不是 Token 数量 2。")
    code_block(
        "14 / 1.732 = 8.083\n"
        "10 / 1.732 = 5.774"
    )
    print_pair_table("缩放后的分数 S / sqrt(3)", scaled_scores)
    unscaled_agent_weight, unscaled_self_weight = unscaled_tools_weights.tolist()
    print("\n缩放的直接效果，可以用 `tools` 这一行做对比：")
    print("\n| 输入 Softmax 的分数 | 得到的权重 | 现象 |")
    print("| --- | --- | --- |")
    print(
        f"| 不缩放 `[10,14]` | `[{unscaled_agent_weight:.3f},"
        f"{unscaled_self_weight:.3f}]` | 几乎只保留最大项 |"
    )
    print("| 缩放 `[5.774,8.083]` | `[0.090,0.910]` | 仍有偏好，但没有那么极端 |")
    print("\n分数过大时，Softmax 容易接近 one-hot。训练中这会让梯度变小、学习不稳定；")
    print("推理中则会让权重对细小分数变化过于敏感。缩放先把输入控制在合适范围。")

    section("6. Causal Mask 与 Softmax")
    subsection("6.1 Causal Mask：先排除禁止读取的位置")
    print("`Agent` 位于 `tools` 左边。GPT 从左到右生成，因此：")
    print("- `Agent` 只能读取自己，不能读取未来的 `tools`；")
    print("- `tools` 可以读取前面的 `Agent` 和自己。")
    print("\nMask 把禁止读取的位置改成 `-inf`：")
    print_pair_table("Mask 后的分数", masked_scores)

    subsection("6.2 Softmax 是什么")
    print("Softmax 把一行任意实数分数转换为一组正数权重，并让这一行的权重之和等于 1：")
    code_block("softmax(s_i) = exp(s_i) / sum_j(exp(s_j))")
    print("实际计算通常先减去本行最大值，数值更稳定，而且不会改变最终比例。")
    print("\n以 `tools` 的 Mask 后分数 `[5.774,8.083]` 为例：")
    code_block(
        "① 减去最大值 8.083：[-2.309, 0]\n"
        "② 取指数：           [exp(-2.309), exp(0)] ≈ [0.099, 1.000]\n"
        "③ 除以总和 1.099：   [0.099/1.099, 1/1.099]\n"
        "④ 得到权重：         [0.090, 0.910]"
    )
    print("`-inf` 的指数是 0，因此被 Mask 的位置经过 Softmax 后权重正好是 0。")

    subsection("6.3 为什么需要 Softmax")
    print("原始点积分数不能直接当作稳定的混合比例，因为它们可能为负、数值范围不固定，")
    print("也不保证总和为 1。Softmax 提供了四个性质：")
    print("1. 所有权重都大于等于 0；")
    print("2. 每一行权重之和为 1，可以解释成读取比例；")
    print("3. 分数越大，权重越大，同时保留相对排序；")
    print("4. 函数可微，训练时可以通过反向传播学习 Q/K/V 投影参数。")
    print("\nSoftmax 不是理论上唯一的归一化选择，但它是标准 Scaled Dot-Product Attention 的选择。")
    print("Attention 权重可以理解为本次读取 V 的比例，但不能直接当作模型答案置信度。")
    print("\n将每一行经过 Softmax：")
    print_pair_table("Attention 权重 A", attention_weights)
    row_sums = [round(value, 3) for value in attention_weights.sum(dim=-1).tolist()]
    print(f"\n每一行权重之和：`{row_sums}`")
    print("- `Agent`：100% 读取自己，0% 读取未来的 `tools`；")
    print("- `tools`：约 9% 读取 `Agent`，约 91% 读取自己。")

    section("7. 用权重对 V 加权求和")
    print("每一行权重决定当前 Token 如何混合所有允许读取的 Value 向量：")
    tools_agent_weight, tools_self_weight = attention_weights[1].tolist()
    code_block(
        "o_Agent = 1.000 x v_Agent + 0.000 x v_tools\n"
        "        = [1.000, 2.000, 3.000]\n\n"
        f"o_tools = {tools_agent_weight:.3f} x [1,2,3] + "
        f"{tools_self_weight:.3f} x [3,2,1]\n"
        "        = [2.819, 2.000, 1.181]"
    )
    print_matrix("上下文化输出 O = A @ V", contextual_vectors)
    print("`tools` 的输出已经混入 `Agent` 的信息，因此不再只是原来的 `[3,2,1]`。")

    section("8. 一页总结")
    code_block(
        "X\n"
        "-> Q、K、V\n"
        "-> QK^T 两两匹配\n"
        "-> 除以 sqrt(d_k)\n"
        "-> Causal Mask\n"
        "-> Softmax 权重 A\n"
        "-> A @ V\n"
        "-> 上下文化输出 O"
    )
    print("| 必须分清 | 正确理解 |")
    print("| --- | --- |")
    print("| `q_i` 与 `Q` | `q_i` 是单 Token 向量；Q 收集全部 Token 的 q |")
    print("| 逐 Token 与矩阵计算 | 数学等价；矩阵写法用于并行 |")
    print("| `(2,3)` | 先读 2 行 3 列，再解释为 2 个 Token × 3 个分量 |")
    print("| 缩放 | 抵消点积分数随维度增长，避免 Softmax 过早饱和 |")
    print("| Softmax | 把任意分数变成非负、每行和为 1 的读取权重 |")
    print("| Causal Mask | 训练和推理共用的因果约束，不是训练或推理专属 |")
    print("\n> **核心结论：** Self-Attention 让每个 Token 按权重读取允许访问的上下文，")
    print("> 从自己的输入向量变成包含上下文的输出向量。")


if __name__ == "__main__":
    main()
