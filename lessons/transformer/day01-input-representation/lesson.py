from __future__ import annotations

import tiktoken
import torch
from torch import nn


PROMPT = "我喜欢打"
EMBEDDING_DIM = 4


def section(title: str) -> None:
    print(f"\n## {title}\n")


def code_block(value: object) -> None:
    print("```text")
    print(value)
    print("```")


def main() -> None:
    tokenizer = tiktoken.get_encoding("o200k_base")
    token_ids = tokenizer.encode(PROMPT)
    token_pieces = [tokenizer.decode([token_id]) for token_id in token_ids]

    torch.manual_seed(42)
    input_ids = torch.tensor(token_ids, dtype=torch.long)
    token_embedding = nn.Embedding(tokenizer.n_vocab, EMBEDDING_DIM)
    position_embedding = nn.Embedding(len(token_ids), EMBEDDING_DIM)
    token_vectors = token_embedding(input_ids)
    positions = torch.arange(len(token_ids))
    position_vectors = position_embedding(positions)
    transformer_input = token_vectors + position_vectors

    section("1. 当前位于完整推理的哪一步")
    code_block(
        "我喜欢打\n"
        "-> [本课] Token ID -> 初始向量\n"
        "-> [Day 2] Self-Attention 读取上下文\n"
        "-> 多层 Transformer\n"
        "-> 预测下一个 Token：网"
    )
    print("本课只回答：文本 `我喜欢打` 怎样变成 Transformer 可以接收的向量序列？")
    print("这是推理的输入准备阶段，不涉及训练、Loss 或参数更新。")

    section("2. 本课的输入与输出")
    print("**输入：** 一段文本 `我喜欢打`")
    print(f"\n**输出：** 形状为 `{tuple(transformer_input.shape)}` 的向量矩阵")
    print("\n这 3 行分别对应 `我`、`喜欢`、`打`，每行 4 个数。")
    print("这些向量将直接交给下一步 Self-Attention。")

    section("3. 文本先被切成 Token")
    print("Tokenizer 的结果来自真实运行：")
    print("\n| 位置 | Token | Token ID |")
    print("| ---: | --- | ---: |")
    for index, (piece, token_id) in enumerate(zip(token_pieces, token_ids)):
        print(f"| {index} | `{piece}` | `{token_id}` |")
    print("\n因此：")
    code_block(
        f"文本：{PROMPT}\n"
        f"Token：{token_pieces}\n"
        f"Token ID：{token_ids}"
    )
    print("Token ID 是词表中的行号，不是语义分数，也不是 Embedding 向量。")

    section("4. Token ID 查询 Token Embedding")
    print("Embedding 层可以看成一张可训练的大表：")
    code_block(
        "Token ID 7522  -> 查第 7522 行 -> 得到“我”的初始向量\n"
        "Token ID 69681 -> 查第 69681 行 -> 得到“喜欢”的初始向量\n"
        "Token ID 15552 -> 查第 15552 行 -> 得到“打”的初始向量"
    )
    print("本实验把每个向量缩短为 4 个数，便于阅读。真实模型通常有更多维度。")
    print("\n| Token | Token ID | 查表得到的 4 维向量 |")
    print("| --- | ---: | --- |")
    for piece, token_id, vector in zip(token_pieces, token_ids, token_vectors.tolist()):
        formatted = ", ".join(f"{value:.3f}" for value in vector)
        print(f"| `{piece}` | `{token_id}` | `[{formatted}]` |")
    print("\n当前数值由固定随机种子生成，只用于演示查表和形状，不代表真实语义。")

    section("5. 再加入位置信息")
    print("Token Embedding 只表达 Token 身份。位置向量补充它位于第 0、1、2 个位置。")
    last_index = len(token_ids) - 1
    print("\n以最后一个 Token `打` 为例：")
    code_block(
        f"Token 向量：{token_vectors[last_index].detach().numpy().round(3)}\n"
        f"位置向量： {position_vectors[last_index].detach().numpy().round(3)}\n"
        f"相加结果： {transformer_input[last_index].detach().numpy().round(3)}"
    )
    print("相加后的结果才是 `打` 进入第一个 Transformer 层的输入向量。")

    section("6. 最终交给 Transformer 的内容")
    print("三行向量上下排列成一个矩阵：")
    code_block(transformer_input.detach().numpy().round(3))
    print(f"Shape 是 `{tuple(transformer_input.shape)}`：3 行对应 3 个 Token，4 列对应每个向量的 4 个分量。")
    print("\n> 本课的输出不是预测结果，只是下一步 Self-Attention 的输入。")

    section("7. 本课结论")
    code_block(
        "我喜欢打\n"
        "-> [我] [喜欢] [打]\n"
        "-> [7522, 69681, 15552]\n"
        "-> 查询 Token Embedding\n"
        "-> 加入位置信息\n"
        "-> 3 个初始向量"
    )
    print("下一课将只跟踪最后一个 Token `打`，观察它如何读取整个提示词的上下文。")


if __name__ == "__main__":
    main()
