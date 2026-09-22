from __future__ import annotations

import math

import torch


TOKENS = ["Agent", "uses", "tools"]


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def print_matrix(name: str, matrix: torch.Tensor) -> None:
    print(f"{name}形状：{tuple(matrix.shape)}")
    print(matrix.detach().numpy().round(3))


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


def main() -> None:
    section("Day 2：Self-Attention 如何让 Token 读取上下文")
    print("你现在位于整条链路的这个位置：")
    print(
        "文本 -> Token -> Token ID -> 输入向量 X "
        "-> [今天学习 Self-Attention] -> 上下文化向量"
    )
    print("\n本节 Token：", TOKENS)
    print(
        "本节只回答一个问题：每个 Token 应该从其他 Token 读取多少信息？"
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

    scores = query @ key.T / math.sqrt(x.shape[-1])

    section("步骤 3：QK^T 计算 Token 两两之间的匹配分数")
    print("读矩阵的方法：行表示谁在读取，列表示它准备从谁那里读取。")
    print("行、列顺序都是：", TOKENS)
    print_matrix("Attention 分数", scores)
    print("\n例如最后一行属于 'tools'：")
    print("- 对 'Agent' 的匹配分数是 0.5")
    print("- 对 'uses' 的匹配分数是 0.5")
    print("- 对自己的匹配分数是 1.5")
    print("这里只是匹配分数，还不是最终读取比例。")

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
        print(f"  {weight:.3f} x {token!r} 的 V 向量")
    print("\n因此，'tools' 的新向量不再只包含自己，也混入了前文信息。")

    section("Day 2 完整链路")
    print(
        "输入向量 X -> Q、K、V -> QK^T 匹配分数 "
        "-> Causal Mask -> Softmax 权重 -> 对 V 加权求和 -> 上下文化向量"
    )
    print("\n本节只需要记住：")
    print("Self-Attention 让每个 Token 按不同权重读取其他 Token 的信息。")
    print("\n观察题：为什么 'Agent' 的 Attention 权重是 [1, 0, 0]？")


if __name__ == "__main__":
    main()
