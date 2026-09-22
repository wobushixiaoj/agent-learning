from __future__ import annotations

import tiktoken
import torch
from torch import nn


TEXT = "Agent uses tools to complete a task."
EMBEDDING_DIM = 8


def section(title: str) -> None:
    print(f"\n## {title}\n")


def code_block(value: object) -> None:
    print("```text")
    print(value)
    print("```")


def show_tokenization(tokenizer: tiktoken.Encoding, text: str) -> list[int]:
    token_ids = tokenizer.encode(text)
    token_pieces = [tokenizer.decode([token_id]) for token_id in token_ids]

    print("**原始文本**")
    code_block(text)
    print("\n**Tokenizer 切分出的 Token**")
    code_block(token_pieces)
    print("\n**每个 Token 在词表中的编号（Token ID）**")
    code_block(token_ids)
    print(f"\nToken 数量：`{len(token_ids)}`")
    print("\n> **本步结论：** Token ID 只是词表索引，还不是语义向量。")

    return token_ids


def show_embeddings(tokenizer: tiktoken.Encoding, token_ids: list[int]) -> None:
    torch.manual_seed(42)

    input_ids = torch.tensor(token_ids, dtype=torch.long)
    token_embedding = nn.Embedding(tokenizer.n_vocab, EMBEDDING_DIM)
    position_embedding = nn.Embedding(len(token_ids), EMBEDDING_DIM)

    token_vectors = token_embedding(input_ids)
    positions = torch.arange(len(token_ids))
    position_vectors = position_embedding(positions)
    transformer_input = token_vectors + position_vectors

    print("Embedding 层是一张可训练的向量表。")
    print("Token ID 用来选择表中的某一行。")
    print(f"\n- Token ID 张量形状：`{tuple(input_ids.shape)}`")
    print(f"- Token Embedding 形状：`{tuple(token_vectors.shape)}`")
    print(f"- Position Embedding 形状：`{tuple(position_vectors.shape)}`")
    print(f"- 最终 Transformer 输入形状：`{tuple(transformer_input.shape)}`")

    print("\n### 以第一个 Token 为例\n")
    print(f"- Token：`{tokenizer.decode([token_ids[0]])!r}`")
    print(f"- Token ID：`{token_ids[0]}`")
    print(f"- Token 向量：`{token_vectors[0].detach().numpy().round(3)}`")
    print(f"- 位置向量：`{position_vectors[0].detach().numpy().round(3)}`")
    print(f"- 两者相加：`{transformer_input[0].detach().numpy().round(3)}`")

    repeated_ids = torch.tensor([token_ids[0], token_ids[0]])
    repeated_vectors = token_embedding(repeated_ids)
    same_vector = torch.equal(repeated_vectors[0], repeated_vectors[1])

    print("\n### 验证同一个 Token ID 是否查到相同向量\n")
    print(f"- 两个 Token ID：`{repeated_ids.tolist()}`")
    print(f"- 查到的 Token 向量是否完全相同：**{'是' if same_vector else '否'}**")
    print(
        "\n> **本步结论：** Token Embedding 表达 Token 是什么；"
        "Position Embedding 提供它在序列中的位置信息。"
    )


def main() -> None:
    tokenizer = tiktoken.get_encoding("gpt2")

    section("学习目标")
    print("本节只学习输入表示，不涉及 Attention、预测目标或训练过程。")

    section("步骤 1：Text -> Token -> Token ID")
    token_ids = show_tokenization(tokenizer, TEXT)

    section("步骤 2：Token ID -> Embedding -> Transformer 输入")
    show_embeddings(tokenizer, token_ids)

    section("本节总结")
    code_block(
        "文本 -> Token -> Token ID -> Token 向量 "
        "+ 位置向量 -> Transformer 输入"
    )
    print("\n下一步 Day 2：让每个 Token 通过 Self-Attention 读取上下文信息。")


if __name__ == "__main__":
    main()
