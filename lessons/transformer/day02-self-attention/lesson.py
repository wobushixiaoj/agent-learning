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
    last_shifted = last_scaled - last_scaled.max()
    last_exp = torch.exp(last_shifted)
    last_exp_sum = last_exp.sum()

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

    section("5. 从匹配分数到读取比例")
    subsection("5.1 先看这一层解决的三个问题")
    print("输入是带 Token 标签的匹配分数，目标是得到可以混合 Value 的读取比例。")
    print("中间的三个机制分别负责不同问题：")
    code_block(
        "匹配分数 [我:1, 喜欢:2, 打:1]\n"
        "├─ Scaling：控制数值尺度\n"
        "├─ Causal Mask：控制哪些位置允许读取\n"
        "└─ Softmax：把允许读取的分数变成比例\n"
        "最终权重 [我:0.264, 喜欢:0.471, 打:0.264]"
    )
    print("| 机制 | 它回答的问题 | 本例结果 |")
    print("| --- | --- | --- |")
    print(
        f"| Scaling | 分数幅度是否合适？ | `我:{last_scaled[0]:.3f} / "
        f"喜欢:{last_scaled[1]:.3f} / 打:{last_scaled[2]:.3f}` |"
    )
    print("| Causal Mask | 哪些位置允许读取？ | 三个位置都允许 |")
    print(
        f"| Softmax | 每个位置读取多少？ | `我:{last_weights[0]:.3f} / "
        f"喜欢:{last_weights[1]:.3f} / 打:{last_weights[2]:.3f}` |"
    )

    subsection("5.2 Scaling：控制数值尺度")
    print("**核心结论：** 缩放保留分数排序，只压低由向量维度带来的额外幅度。")
    print("\n#### 把本例代入缩放公式\n")
    print("缩放公式要求把每个匹配分数除以 `sqrt(d_k)`。先确定公式中的 `d_k`：")
    print("- `d_k` 表示 Query 和 Key 向量的维度；")
    print("- 本例 `q_打=[1,1,0]`，向量包含 3 个分量；")
    print("- 因此本例 `d_k=3`。")
    print("\n再把 `d_k=3` 代入缩放因子：")
    code_block("sqrt(d_k) = sqrt(3) = 1.732")
    print("三个 Token 使用同一个除数 `1.732`：")
    print("\n| 信息来源 Token | 原始匹配分数 | 缩放计算 | 缩放后分数 |")
    print("| --- | ---: | --- | ---: |")
    print("| `我` | 1 | `1 / 1.732` | 0.577 |")
    print("| `喜欢` | 2 | `2 / 1.732` | 1.155 |")
    print("| `打` | 1 | `1 / 1.732` | 0.577 |")
    print("\n缩放没有改变大小顺序：`喜欢` 的分数仍然最高。它只控制分数幅度，")
    print("避免向量维度增大时，点积分数自然变大并让 Softmax 过度极端。")

    print("\n#### 验证一：独立二 Token 实验，只改变“喜欢”的分数\n")
    print("| 实验 | 固定输入：`我`的分数 | 唯一自变量：`喜欢`的分数 | 观测结果：Softmax 权重（我 / 喜欢） |")
    print("| --- | ---: | ---: | --- |")
    print("| A1 | 0 | 1 | `26.9% / 73.1%` |")
    print("| A2 | 0 | 2 | `11.9% / 88.1%` |")
    print("| A3 | 0 | 4 | `1.8% / 98.2%` |")
    print("\n**观察结果：** 只要输入差距变大，Softmax 就会让输出比例变得更极端。")

    print("\n#### 验证二：固定同一组分数，只改变是否缩放\n")
    print("| 固定条件 | 固定值 |")
    print("| --- | --- |")
    print("| 向量维度 | `d_k=16`，因此 `sqrt(d_k)=4` |")
    print("| 原始分数 | `我:0 / 喜欢:4` |")
    print("| 归一化函数 | 同一个 Softmax |")
    print("\n| 唯一自变量：缩放方式 | 进入 Softmax 的分数（我 / 喜欢） | 观测结果：权重（我 / 喜欢） |")
    print("| --- | --- | --- |")
    print("| 不缩放 | `0 / 4` | `1.8% / 98.2%` |")
    print("| 除以 4 | `0 / 1` | `26.9% / 73.1%` |")
    print("\n**观察结果：** 同一组原始分数经过缩放后，大小顺序不变，但权重不再过度极端。")

    print("\n#### 统计规律如何连接两个验证\n")
    code_block(
        "d_k 增大\n"
        "-> 点积分数的典型幅度约按 sqrt(d_k) 增长\n"
        "-> 验证一：分数差距增大会让 Softmax 权重更极端\n"
        "-> 验证二：除以 sqrt(d_k) 可以抵消这部分额外放大"
    )
    print("真实模型每次点积的具体分数不会固定等于 `我:0 / 喜欢:4`，这里仅用于控制变量观察。")

    subsection("5.3 Causal Mask：控制访问权限")
    print("**核心结论：** Mask 不负责判断相关性，只负责把未来位置彻底排除。")
    print("\n#### 当前案例\n")
    print("Causal 的意思是：当前位置只能读取自己和左侧，不能读取右侧的未来 Token。")
    print("当前 Query Token 是最后一个位置的 `打`：")
    print("\n| 信息来源 Token | 相对 `打` 的位置 | 是否允许读取 | Mask 值 | Mask 前分数 | Mask 后分数 |")
    print("| --- | --- | --- | ---: | ---: | ---: |")
    print("| `我` | 左侧 | 是 | 0 | 0.577 | 0.577 |")
    print("| `喜欢` | 左侧 | 是 | 0 | 1.155 | 1.155 |")
    print("| `打` | 当前 | 是 | 0 | 0.577 | 0.577 |")
    print("\n作为对照，如果 Query Token 是中间位置的 `喜欢`：")
    print("\n| 信息来源 Token | 相对 `喜欢` 的位置 | 是否允许读取 | Mask 值 |")
    print("| --- | --- | --- | ---: |")
    print("| `我` | 左侧 | 是 | 0 |")
    print("| `喜欢` | 当前 | 是 | 0 |")
    print("| `打` | 右侧未来 | 否 | `-inf` |")
    print("\n`-inf` 经过 Softmax 后会得到 0 权重，表示完全禁止读取。")

    subsection("5.4 Softmax：把分数变成分配比例")
    print("**核心结论：** Softmax 把允许读取的任意分数转换成非负、总和为 1 的权重。")
    print("\n#### 把本例代入 Softmax\n")
    print("实际计算先减去三个分数中的最大值 `1.155`，避免指数产生过大的数：")
    print("\n| 信息来源 Token | Mask 后分数 | 减去最大值 1.155 | 取 `exp` | 除以总和 2.123 后的权重 |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for token, score, shifted, exp_value, weight in zip(
        TOKENS, last_scaled, last_shifted, last_exp, last_weights
    ):
        print(
            f"| `{token}` | {score:.3f} | {shifted:.3f} | "
            f"{exp_value:.3f} | {weight:.3f} |"
        )
    print(f"| **合计** |  |  | **{last_exp_sum:.3f}** | **1.000** |")
    print("\n现在三个数都非负，并且总和为 1，所以可以直接解释成读取比例。")

    subsection("5.5 把三个机制重新合起来")
    print("| 信息来源 Token | 原始匹配分数 | Scaling 后 | Mask 后 | Softmax 权重 |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for token, raw, scaled, masked, weight in zip(
        TOKENS, last_raw, last_scaled, masked_scores[-1], last_weights
    ):
        print(
            f"| `{token}` | {raw:.3f} | {scaled:.3f} | "
            f"{masked:.3f} | {weight:.3f} |"
        )
    print("\n这组结果最终只表达一句话：")
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
