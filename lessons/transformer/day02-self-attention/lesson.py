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
    print("X 的形状是 (3, 4)。形状只是在描述下面矩阵有多大：")
    print("- 第一个数字 3：有 3 行，对应 3 个 Token")
    print("- 第二个数字 4：每行有 4 个数，即每个 Token 用 4 维向量表示")
    print("- 因此矩阵中共有 3 x 4 = 12 个数")
    print("\n通用写法：X.shape = (Token 数量, 每个 Token 的向量维度)")
    print("本例就是：(3 个 Token, 每个 Token 4 维)")
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

    subsection("3.0 矩阵乘法速通：本节只需要三条规则")
    print("- **规则 1：** 左矩阵的一行，与右矩阵的一列，计算出结果中的一个格子。")
    print("- **规则 2：** 这一行和这一列做点积，即对应位置相乘，再把乘积相加。")
    print("- **规则 3：** 左矩阵列数必须等于右矩阵行数；结果保留外侧两个数字。")
    print("\n套到本例：")
    print("- `Q (3, 4)`：左边有 3 行，每行有 4 个数")
    print("- `K.T (4, 3)`：右边有 3 列，每列有 4 个数")
    print("- `结果 (3, 3)`：3 行分别乘 3 列，一共得到 9 个格子")
    print("- 中间的 `4` 和 `4` 必须相同，因为一次点积需要两组同样多的数。")
    print("\n可以先把矩阵乘法理解为：批量执行‘一行与一列的点积’。")
    print("它不是把两个矩阵相同位置的数字直接相乘。")
    print_matrix("转置后的 K.T", key.T)
    print("例如结果左上角 = Q 第 1 行 · K.T 第 1 列：\n")
    code_block(
        "[1, 0, 1, 0] · [1, 0, 1, 0]\n"
        "= 1x1 + 0x0 + 1x1 + 0x0\n"
        "= 2"
    )

    subsection("3.1 为什么要写 K.T？")
    print("Q 的形状是 (3, 4)：3 个 Token，每个 Query 有 4 个数。")
    print("K 的形状也是 (3, 4)。")
    print("K.T 表示把 K 转置，所以形状从 (3, 4) 变成 (4, 3)。")
    print("这样才能进行矩阵乘法：\n")
    code_block("Q (3, 4) @ K.T (4, 3) -> 原始分数 (3, 3)")
    print("中间的两个 4 对齐；结果保留外侧的 3 和 3。")
    print("最终 3 x 3 表示：3 个读取者分别与 3 个信息来源进行匹配。")

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
