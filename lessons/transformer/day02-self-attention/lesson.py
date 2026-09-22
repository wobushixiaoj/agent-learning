from __future__ import annotations

import math

import torch


TOKENS = ["Agent", "uses", "tools"]


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
        print(f"- {token!r:8} -> [{formatted}]")


def explain_attention_rows(attention_weights: torch.Tensor) -> None:
    print("\n把每一行翻译成人话：")
    for reader_index, reader in enumerate(TOKENS):
        parts = []
        for source, weight in zip(TOKENS, attention_weights[reader_index].tolist()):
            parts.append(f"{source} {weight * 100:.1f}%")
        print(f"- {reader!r} 读取信息的比例：" + "，".join(parts))


def print_pair_table(name: str, matrix: torch.Tensor) -> None:
    print(f"\n**{name}**（行=读取者，列=信息来源）：\n")
    print("| 读取者 \\ 来源 | " + " | ".join(TOKENS) + " |")
    print("| --- | " + " | ".join("---:" for _ in TOKENS) + " |")
    for token, row in zip(TOKENS, matrix.tolist()):
        values = " | ".join(f"{value:.3f}" for value in row)
        print(f"| {token} | {values} |")


def main() -> None:
    section("课程位置与目标")
    print("你现在位于整条链路的这个位置：")
    code_block(
        "文本 -> Token -> Token ID -> 输入向量 X "
        "-> [今天学习 Self-Attention] -> 上下文化向量"
    )
    print(f"\n本节 Token：`{TOKENS}`")
    print(
        "\n> **本节只回答一个问题：** 每个 Token 应该从其他 Token 读取多少信息？"
    )
    print(
        "为了先看清数据流，Q、K、V 暂时直接使用同一份输入向量。"
        "真实模型会通过可训练矩阵分别生成 Q、K、V，后续再学。"
    )

    # Each row is one token's input vector from the previous stage.
    x = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],  # Agent
            [0.0, 1.0, 1.0, 0.0],  # uses
            [1.0, 1.0, 0.0, 1.0],  # tools
        ]
    )

    section("步骤 1：准备 Transformer 输入 X")
    print("Shape 的严格数学含义始终是 `(行数, 列数)`。")
    print("所以 X 的形状 `(3, 4)` 首先只表示：这个矩阵有 3 行、4 列。")
    print("\n然后再解释本例为行列赋予的语义：")
    print("- 3 行：每行放一个 Token 的向量，所以本例对应 3 个 Token")
    print("- 4 列：每行有 4 个分量，所以本例的每个 Token 使用 4 维向量")
    print("- 因此矩阵中共有 3 x 4 = 12 个数")
    print("\n> `(3, 4) = 3 行 4 列` 是严格定义；`3 个 Token、每个向量 4 维` 是本例的语义解释。")
    print("\n本教程当前采用的二维布局：")
    code_block("X.shape = (Token 数量, 每个 Token 的向量维度)")
    print("这个语义映射依赖当前数据布局，不是所有张量都永远这样解释。")
    print_matrix("输入向量 X", x)
    print_token_rows(TOKENS, x)
    print("\n数字写成 1.0、0.0，是因为它们是浮点数；这里先不用解释每一维的语义。")
    print("\n此时每一行主要表示自己的信息，还没有读取其他 Token。")

    # In a real Transformer these are learned linear projections.
    query = x
    key = x
    value = x

    section("步骤 2：同一个输入分别扮演 Q、K、V 三种角色")
    print("Q（Query）：当前 Token 想找什么信息。")
    print("K（Key）：每个 Token 可以用什么特征被匹配。")
    print("V（Value）：匹配成功后，真正被读取的内容。")
    print("\n本节为降低难度，令 Q = K = V = X，所以三个矩阵暂时相同。")
    print_matrix("Q", query)
    print_matrix("K", key)
    print_matrix("V", value)

    raw_scores = query @ key.T
    scale = math.sqrt(x.shape[-1])
    scores = raw_scores / scale

    section("步骤 3：QK^T 计算 Token 两两之间的匹配分数")
    print("本步骤要回答：每个 Token 与每个 Token 的匹配程度是多少？")

    subsection("3.0 先别背规则：它只是批量计算匹配分数")
    print("先只问一个非常具体的问题：**`tools` 和 `Agent` 匹不匹配？**")
    print("\n先看完整 Q，它仍然有 3 行 4 列：\n")
    print("| Q 中的行 | Query 向量 | 本次是否选中 |")
    print("| --- | --- | --- |")
    print("| `Agent` | `[1, 0, 1, 0]` |  |")
    print("| `uses` | `[0, 1, 1, 0]` |  |")
    print("| `tools` | `[1, 1, 0, 1]` | **是，抽出第 3 行** |")
    print("\n> 截图里的 `tools Query` 不是完整 Q，而是从完整 Q 中抽出的一行。")
    print("\n同样，完整 K 也有 3 行 4 列；为了和 `tools` 比较，这次只抽出")
    print("`Agent Key = [1, 0, 1, 0]`，也就是 K 的第 1 行。")
    print("\n当前教程先观察一个 Attention head。在一个 head 中，一个 Token 对应一个")
    print("Query 向量，所以 `tools` 对应一行；三个 Token 合在一起才组成完整 Q 的三行。")
    print("\n现在把选出的两个向量放进手算表。它们各有 4 个特征数字：\n")
    print("| 特征位置 | 1 | 2 | 3 | 4 |")
    print("| --- | ---: | ---: | ---: | ---: |")
    print("| `tools` Query | 1 | 1 | 0 | 1 |")
    print("| `Agent` Key | 1 | 0 | 1 | 0 |")
    print("| 对应位置相乘 | 1×1=1 | 1×0=0 | 0×1=0 | 1×0=0 |")
    print("\n然后把最后一行加起来：\n")
    code_block("1 + 0 + 0 + 0 = 1")
    print("这个 `1` 就是 `tools` 对 `Agent` 的一个原始匹配分数。")
    print("\n但 Attention 不只需要这一个分数：")
    print("- `tools` 还要分别和 `uses`、`tools` 比较；")
    print("- `Agent` 和 `uses` 也要分别和三个 Token 比较；")
    print("- 3 个读取者 × 3 个信息来源，总共要计算 **9 个分数**。")
    print("\n> `Q @ K.T` 只是把这 9 次同样的计算一次性完成。")
    print("\n所以在本节里，可以先把矩阵乘法理解为：**批量计算两两匹配分数**。")
    print("它不是 Attention 新增的一种判断逻辑，只是省去逐个计算的写法。")

    subsection("3.1 严格形状、标准理论与本例语义")
    print("先把三种说法分开：")
    print("- **严格 Shape：** `Q.shape = (3, 4)` 就是 Q 有 3 行、4 列；")
    print("- **本例语义：** 每行放一个 Token 的 Query 向量，所以是 3 个 Query 向量，每个有 4 个分量；")
    print("- **标准理论记号：** `Q: (n_q, d_k)`，本例 `n_q=3`、`d_k=4`。")
    print("\n因此，更准确的原句应该是：")
    print("> Q 是一个 3 行 4 列的矩阵；在本例中，3 行分别对应 3 个 Token 的 Query 向量，每行有 4 个分量。")
    print("\n标准的二维 Attention 形状关系是：")
    code_block(
        "Q:       (n_q, d_k)\n"
        "K:       (n_k, d_k)\n"
        "K.T:     (d_k, n_k)\n"
        "Q @ K.T: (n_q, n_k)"
    )
    print("Self-Attention 中 Q 和 K 来自同一序列，通常 `n_q = n_k = n`。")
    print("本例取 `n=3`、`d_k=4`，所以 Q 和 K 都是 3 行 4 列。")
    print("\n计算时，需要让一个 Query 横着与一个 Key 竖着对齐。")
    print("K.T 就是把 K 的每个 Token 从一行转成一列，形状从 (3, 4) 变成 (4, 3)。")
    print_matrix("转置后的 K.T", key.T)
    print("这样才能进行矩阵乘法：\n")
    code_block("Q (3, 4) @ K.T (4, 3) -> 原始分数 (3, 3)")
    print("这三个形状数字现在可以翻译成人话：")
    print("- 中间的 `4` 和 `4`：每次比较的 Query 和 Key 都有 4 个特征；")
    print("- 左侧的 `3`：有 3 个读取者；")
    print("- 右侧的 `3`：有 3 个候选信息来源；")
    print("- 最终 `3 × 3`：得到 9 个两两匹配分数。")
    print("\n**真实工程实现：** 多头 Attention 通常还会加上 batch 和 head 维度，例如")
    print("`(batch, heads, sequence_length, head_dim)`。不同框架可能调整维度顺序，")
    print("所以不存在唯一的官方存储顺序；稳定的是 `QK.T` 的计算关系。")
    print("\n参考：")
    print("- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)")
    print("- [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)")

    subsection("3.2 一个格子的数字怎么计算？")
    print("每个格子都是一行 Query 与一行 Key 的点积：")
    code_block("点积 = 对应位置相乘，再把乘积全部相加")
    print("\n完整手算 'tools' 这一行：")
    code_block(
        "tools 的 Query = [1, 1, 0, 1]\n"
        "Agent 的 Key   = [1, 0, 1, 0]\n"
        "uses 的 Key    = [0, 1, 1, 0]\n"
        "tools 的 Key   = [1, 1, 0, 1]\n\n"
        "tools 与 Agent：1x1 + 1x0 + 0x1 + 1x0 = 1\n"
        "tools 与 uses： 1x0 + 1x1 + 0x1 + 1x0 = 1\n"
        "tools 与 tools：1x1 + 1x1 + 0x0 + 1x1 = 3"
    )
    print_pair_table("尚未缩放的 QK^T 原始分数", raw_scores)

    subsection("3.3 为什么原始分数还要缩放？")
    print(f"每个向量有 4 维，所以 sqrt(4) = {scale:.1f}。")
    print("把每个原始分数都除以 2：\n")
    code_block(
        "tools -> Agent：1 / 2 = 0.5\n"
        "tools -> uses： 1 / 2 = 0.5\n"
        "tools -> tools：3 / 2 = 1.5"
    )
    print("缩放用于避免向量维度较大时点积过大，让后面的 Softmax 过于极端。")
    print_pair_table("缩放后的 Attention 分数", scores)
    print("\n到这里得到的仍是匹配分数，还不是最终读取比例。")

    # GPT-style causal attention cannot read future tokens.
    causal_mask = torch.triu(
        torch.ones(len(TOKENS), len(TOKENS), dtype=torch.bool), diagonal=1
    )
    masked_scores = scores.masked_fill(causal_mask, float("-inf"))
    attention_weights = torch.softmax(masked_scores, dim=-1)

    section("步骤 4：Causal Mask 禁止偷看未来，再用 Softmax 变成比例")
    print("GPT 按从左到右生成文本，因此当前位置只能读取自己和左边的 Token。")
    print("例如 'Agent' 后面的 'uses' 和 'tools' 对它来说都属于未来。")
    print("\nSoftmax 把允许读取的分数转换为 0 到 1 之间的权重。")
    print("列顺序：", TOKENS)
    print_matrix("Attention 权重", attention_weights)
    print("每一行权重之和：", attention_weights.sum(dim=-1).tolist())
    explain_attention_rows(attention_weights)

    contextual_vectors = attention_weights @ value

    section("步骤 5：按照 Attention 权重，对 V 向量加权求和")
    print("现在不是选择某一个 Token，而是按比例混合多个 Token 的 V。")
    print_matrix("融合上下文后的向量", contextual_vectors)
    print("\n以 'tools' 为例，它的新向量来自：")
    for token, weight in zip(TOKENS, attention_weights[2].tolist()):
        print(f"- `{weight:.3f} x {token!r}` 的 V 向量")
    print("\n因此，'tools' 的新向量不再只包含自己，也混入了前文信息。")

    section("本节总结")
    code_block(
        "输入向量 X -> Q、K、V -> QK^T 匹配分数 "
        "-> Causal Mask -> Softmax 权重 -> 对 V 加权求和 -> 上下文化向量"
    )
    print("\n> **本节只需要记住：** Self-Attention 让每个 Token 按不同权重读取其他 Token 的信息。")
    print("\n### 观察题\n")
    print("为什么 `Agent` 的 Attention 权重是 `[1, 0, 0]`？")


if __name__ == "__main__":
    main()
