from __future__ import annotations

import math

import torch


TOKENS = ["我", "喜欢", "打"]


def section(title: str) -> None:
    print(f"\n## {title}\n")


def subsection(title: str) -> None:
    print(f"\n### {title}\n")


def code_block(value: object) -> None:
    print("```text")
    print(value)
    print("```")


def main() -> None:
    # Small learned-looking vectors chosen only to make one attention row hand-calculable.
    query = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # 我
            [0.0, 1.0, 0.0],  # 喜欢
            [1.0, 1.0, 0.0],  # 打
        ]
    )
    key = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # 我
            [1.0, 1.0, 0.0],  # 喜欢
            [0.0, 1.0, 1.0],  # 打
        ]
    )
    value = torch.eye(3)

    raw_scores = query @ key.T
    scale = math.sqrt(query.shape[-1])
    scaled_scores = raw_scores / scale
    causal_mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
    masked_scores = scaled_scores.masked_fill(causal_mask, float("-inf"))
    attention_weights = torch.softmax(masked_scores, dim=-1)
    contextual_vectors = attention_weights @ value

    last_raw = raw_scores[-1]
    last_scaled = scaled_scores[-1]
    last_weights = attention_weights[-1]
    last_output = contextual_vectors[-1]

    section("1. 当前位于完整推理的哪一步")
    code_block(
        "我喜欢打\n"
        "-> Token ID\n"
        "-> 初始向量\n"
        "-> [本课] Self-Attention：让“打”读取上下文\n"
        "-> 后续 Transformer 层\n"
        "-> 预测下一个 Token：网"
    )
    print("本课只研究推理过程中的一次 Attention 前向计算，不讨论训练和参数更新。")

    section("2. 本课的输入与输出")
    print("**输入：** `我`、`喜欢`、`打` 三个 Token 的初始向量。")
    print("\n**本课只跟踪：** 最后一个 Token `打`。")
    print("\n**输出：** 融合了 `我喜欢打` 上下文的 `打` 的新向量。")
    print("\n> 本课输出仍然是向量，不是 `网`，也不是下一个 Token 的概率。")
    print("\nDay 1 的输出会真实进入 `XW_Q`、`XW_K`、`XW_V`。但 Day 1 使用的是随机演示参数，")
    print("直接延续会得到难以手算的小数。本课保持同一数据流和同一文本，另外挑选简单的")
    print("Q/K/V 数字来展示机制；这些数字不是 Day 1 随机向量的实际计算结果。")

    section("3. 用一句人话理解 Q、K、V")
    print("当 `打` 读取上下文时，可以把三个对象理解成：")
    print("\n| 对象 | 本例中的问题 |")
    print("| --- | --- |")
    print("| `q_打`（Query） | `打` 当前想从上下文寻找什么？ |")
    print("| 每个 `k`（Key） | `我`、`喜欢`、`打` 各自适不适合被读取？ |")
    print("| 每个 `v`（Value） | 如果读取这个 Token，真正取回什么信息？ |")
    print("\n真实模型通过 `XW_Q`、`XW_K`、`XW_V` 得到这些向量。")
    print("下面使用人为挑选的 3 维数字，只为了完整看清一次计算。")

    section("4. 第一步：打分别和三个 Key 计算匹配分数")
    q_last = query[-1]
    print("`打` 的 Query 是：")
    code_block("q_打 = [1,1,0]")
    print("三个 Token 的 Key 是：")
    code_block(
        "k_我   = [1,0,0]\n"
        "k_喜欢 = [1,1,0]\n"
        "k_打   = [0,1,1]"
    )
    print("`q_打` 分别与三个 Key 点积：")
    code_block(
        "打 -> 我：   [1,1,0] · [1,0,0] = 1\n"
        "打 -> 喜欢： [1,1,0] · [1,1,0] = 2\n"
        "打 -> 打：   [1,1,0] · [0,1,1] = 1"
    )
    print(f"所以 `打` 这一行的原始匹配分数是 `{last_raw.tolist()}`。")
    print("第二个分数最高，表示在这个教学例子里，`打` 最倾向读取 `喜欢`。")

    section("5. 第二步：把三个分数变成读取比例")
    print("从原始分数到读取比例，依次经过三个操作：")
    print("\n| 阶段 | 数值 | 作用 |")
    print("| --- | --- | --- |")
    print(f"| 原始分数 | `{last_raw.tolist()}` | 比较三个来源的匹配强弱 |")
    print(
        f"| 除以 `sqrt(3)` | `[{last_scaled[0]:.3f}, {last_scaled[1]:.3f}, "
        f"{last_scaled[2]:.3f}]` | 控制分数幅度 |"
    )
    print("| Causal Mask | 数值不变 | `打` 是提示词最后一个位置，可以读取三个位置 |")
    print(
        f"| Softmax | `[{last_weights[0]:.3f}, {last_weights[1]:.3f}, "
        f"{last_weights[2]:.3f}]` | 变成总和为 1 的读取比例 |"
    )
    print("\n这组结果只表达一句话：")
    print(
        f"> `打` 从 `我` 读取 {last_weights[0] * 100:.1f}%，从 `喜欢` 读取 "
        f"{last_weights[1] * 100:.1f}%，从自己读取 {last_weights[2] * 100:.1f}%。"
    )

    section("6. 第三步：按比例读取三个 Value")
    print("为了让混合结果一眼可见，本例使用：")
    code_block(
        "v_我   = [1,0,0]\n"
        "v_喜欢 = [0,1,0]\n"
        "v_打   = [0,0,1]"
    )
    print("按照上一步的权重加权求和：")
    code_block(
        f"o_打 = {last_weights[0]:.3f} x v_我\n"
        f"     + {last_weights[1]:.3f} x v_喜欢\n"
        f"     + {last_weights[2]:.3f} x v_打\n\n"
        f"     = [{last_output[0]:.3f}, {last_output[1]:.3f}, {last_output[2]:.3f}]"
    )
    print("这个新向量已经混合了三个位置的信息，因此不再只表示孤立的 `打`。")

    section("7. 本课输出接下来去哪里")
    code_block(
        "包含上下文的“打”向量\n"
        "-> 残差连接、LayerNorm、FFN\n"
        "-> 更多 Transformer 层\n"
        "-> 最终位置向量\n"
        "-> LM Head\n"
        "-> 下一个 Token 的概率"
    )
    print("因此不能说 Attention 预测了 `网`。它只完成了“让当前位置读取上下文”这一步。")

    section("8. 技术附录：逐 Token 写法和矩阵写法是什么关系")
    print("主线只计算了 `打` 这一行：")
    code_block("q_打 @ K.T -> [1,2,1]")
    print("真实实现会把三个 Query 放进大写 `Q`，并行计算所有行：")
    code_block(
        "Q @ K.T =\n"
        "[[1,1,0],   # 我读取各位置的分数\n"
        " [0,1,1],   # 喜欢读取各位置的分数\n"
        " [1,2,1]]   # 打读取各位置的分数"
    )
    print("小写 `q_打` 是大写 `Q` 的第三行。矩阵写法只是同时完成每个 Token 的同类计算。")

    subsection("8.1 Causal Mask 对前两个位置做了什么")
    print("在推理的 Prefill 阶段，三个提示词 Token 会并行计算，但每个位置只能看自己和左侧：")
    print("- `我` 只能读取 `我`；")
    print("- `喜欢` 可以读取 `我、喜欢`；")
    print("- `打` 可以读取 `我、喜欢、打`。")
    print("本课跟踪的 `打` 已经位于最后，所以它这一行没有被 Mask 删除任何位置。")

    subsection("8.2 为什么要除以 sqrt(d_k)")
    print("点积会把 `d_k` 个乘积相加。维度越多，分数的典型幅度通常越大，")
    print("Softmax 也会因此变得过度极端。除以 `sqrt(d_k)` 用来抵消这种维度增益。")
    print("它不会改变哪个分数最大，只控制分数差距进入 Softmax 时不要无谓放大。")

    section("9. 本课结论")
    code_block(
        "打的 Query\n"
        "-> 分别匹配 我 / 喜欢 / 打 的 Key\n"
        "-> 得到匹配分数\n"
        "-> 缩放 + Mask + Softmax\n"
        "-> 得到读取比例\n"
        "-> 按比例混合三个 Value\n"
        "-> 得到包含上下文的“打”向量"
    )


if __name__ == "__main__":
    main()
