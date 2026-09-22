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
    small_scale_weights = torch.softmax(torch.tensor([0.0, 1.0]), dim=-1)
    large_scale_weights = torch.softmax(torch.tensor([0.0, 10.0]), dim=-1)
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

    section("6. 把分数变成读取比例：先只看一行")
    print("这一节先不看完整矩阵，只跟踪 `tools` 这一行。")
    print("它要回答的问题是：`tools` 应该从 `Agent` 和自己各读取多少信息？")

    subsection("6.1 tools 这一行的完整流水线")
    print("上一节已经算出 `tools` 对两个来源的原始分数是 `[10,14]`：")
    print("- `10`：`tools` 与 `Agent` 的匹配分数；")
    print("- `14`：`tools` 与自己的匹配分数。")
    print("\n接下来三个操作各做一件事：")
    print("\n| 阶段 | 这一行的数值 | 只做什么 |")
    print("| --- | --- | --- |")
    print("| 原始分数 | `[10,14]` | 表示两个来源的匹配强弱 |")
    print("| 除以 `sqrt(3)` | `[5.774,8.083]` | 压低数值幅度，不改变谁大谁小 |")
    print("| Causal Mask | `[5.774,8.083]` | `tools` 可以看两个位置，所以这一行不变 |")
    print("| Softmax | `[0.090,0.910]` | 变成总和为 1 的读取比例 |")
    print("\n因此这一行只表达一句话：")
    print("> `tools` 读取约 9% 的 `Agent` Value，再读取约 91% 的自己 Value。")

    subsection("6.2 再看 Agent 这一行为什么不同")
    print("`Agent` 位于 `tools` 前面。GPT 不能让前面的 Token 偷看未来 Token，")
    print("所以 Mask 会删除 `Agent -> tools` 这条连接：")
    print("\n| 阶段 | Agent 这一行 | 含义 |")
    print("| --- | --- | --- |")
    print("| 原始分数 | `[14,10]` | 原本也算出了两个匹配分数 |")
    print("| 缩放后 | `[8.083,5.774]` | 控制数值幅度 |")
    print("| Mask 后 | `[8.083,-inf]` | 第二个位置是未来，禁止读取 |")
    print("| Softmax 后 | `[1.000,0.000]` | 只能读取自己 |")
    print("\n`-inf` 只是“禁止读取”的计算标记。经过 Softmax 后，它的权重会变成 0。")

    subsection("6.3 两行放回完整矩阵")
    print_pair_table("最终 Attention 权重 A", attention_weights)
    print("完整矩阵仍然只讲两句话：")
    print("- 第一行：`Agent` 只能读取自己；")
    print("- 第二行：`tools` 同时读取 `Agent` 和自己。")

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

    section("8. 进阶说明：为什么要除以 sqrt(d_k)")
    print("> 到第 7 节，Attention 主流程已经结束。本节只解释缩放原因，可以稍后再读。")

    subsection("8.1 缩放解决的是“维度越多，点积分数自然越大”")
    print("一个点积分数是 `d_k` 个乘积相加：")
    code_block("q · k = q_1k_1 + q_2k_2 + ... + q_dk k_dk")
    print("`d_k` 越大，参与求和的项越多，分数的典型波动范围也越大。")
    print("这会让高维向量仅仅因为维度更多，就产生更极端的 Attention 权重。")
    print("除以 `sqrt(d_k)` 是对这种维度增益做补偿，不会改变分数的大小顺序。")
    print("\n标准统计结论是：当各维大致独立、均值为 0、方差相近时：")
    code_block(
        "点积的方差约为 d_k\n"
        "点积的标准差约为 sqrt(d_k)\n"
        "点积 / sqrt(d_k) 后，典型波动范围回到相近量级"
    )

    subsection("8.2 Softmax 会放大分数差距")
    small_low, small_high = small_scale_weights.tolist()
    large_low, large_high = large_scale_weights.tolist()
    print("Softmax 虽然总会把结果变成总和为 1 的比例，但不会消除输入差距：")
    print("\n| Softmax 输入 | 输出权重 |")
    print("| --- | --- |")
    print(f"| `[0,1]` | `[{small_low:.3f},{small_high:.3f}]` |")
    print(f"| `[0,10]` | `[{large_low:.6f},{large_high:.6f}]` |")
    print("\n第二组只是把分数差距从 1 放大到 10，Softmax 就几乎只保留第二项。")
    print("所以顺序是：先缩放控制分数差距，再用 Softmax 转成读取比例。")

    subsection("8.3 回到本例")
    unscaled_agent_weight, unscaled_self_weight = unscaled_tools_weights.tolist()
    print(f"本例 `d_k=3`，缩放因子是 `sqrt(3)={scale:.3f}`：")
    print("\n| 做法 | tools 的分数 | Softmax 权重 |")
    print("| --- | --- | --- |")
    print(
        f"| 不缩放 | `[10,14]` | `[{unscaled_agent_weight:.3f},"
        f"{unscaled_self_weight:.3f}]` |"
    )
    print("| 除以 `sqrt(3)` | `[5.774,8.083]` | `[0.090,0.910]` |")
    print("\n缩放保留了“自己分数更高”这个判断，只降低了无谓的极端程度。")

    subsection("8.4 动态观察")
    print("假设未缩放的两个分数是 `[0,sqrt(d_k)]`。维度增大时：")
    print("\n| `d_k` | 不缩放后的权重 | 除以 `sqrt(d_k)` 后的权重 |")
    print("| ---: | --- | --- |")
    print("| 1 | `[26.9%,73.1%]` | `[26.9%,73.1%]` |")
    print("| 16 | `[1.8%,98.2%]` | `[26.9%,73.1%]` |")
    print("| 64 | `[0.03%,99.97%]` | `[26.9%,73.1%]` |")
    print("\n缩放的目标不是让 Attention 平均，而是不让维度数量决定它有多极端。")

    section("9. 进阶说明：Softmax 如何算出 0.090 和 0.910")
    print("对 `tools` 的缩放后分数 `[5.774,8.083]`：")
    code_block(
        "① 每项减去最大值 8.083：[-2.309, 0]\n"
        "② 取指数：                  [0.099, 1.000]\n"
        "③ 除以总和 1.099：          [0.090, 0.910]"
    )
    print("减去最大值只是为了计算稳定，不改变最终比例。")
    print("Softmax 的结果非负、总和为 1，因此可以直接用于混合 Value。")
    print("Attention 权重是本次读取 V 的比例，不等于模型答案的置信度。")

    section("10. 一页总结")
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
