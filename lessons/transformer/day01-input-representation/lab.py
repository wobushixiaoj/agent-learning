# %% [markdown]
# # Day 1：Tokenizer 与 Embedding
#
# 从上到下依次运行每个 Cell。
# 每次运行前，先预测输出的形状或行为。

# %%
import tiktoken
import torch
from torch import nn

print("Python 环境已就绪。")
print("PyTorch 版本：", torch.__version__)
print("MPS 是否可用：", "是" if torch.backends.mps.is_available() else "否")


# %% [markdown]
# ## 1. 文本如何变成 Token ID
#
# 运行前先想：
#
# - Token 一定是完整英文单词吗？
# - Token ID 是语义分数，还是词表索引？

# %%
TEXT = "Agent uses tools to complete a task."

tokenizer = tiktoken.get_encoding("gpt2")
token_ids = tokenizer.encode(TEXT)
token_pieces = [tokenizer.decode([token_id]) for token_id in token_ids]

print("原始文本：", TEXT)
print("切分出的 Token：", token_pieces)
print("Token ID：", token_ids)
print("Token 数量：", len(token_ids))


# %% [markdown]
# ## 2. 比较英文与中文的 Token 数量
#
# 这个 Tokenizer 来自 GPT-2。观察它如何切分不同语言的文本。

# %%
texts = [
    "Agent uses tools.",
    "智能体调用工具。",
]

for text in texts:
    ids = tokenizer.encode(text)
    pieces = [tokenizer.decode([token_id]) for token_id in ids]
    print(f"\n{text!r}")
    print("Token：", pieces)
    print("Token ID：", ids)
    print("数量：", len(ids))


# %% [markdown]
# ## 3. Token ID 如何变成向量
#
# `nn.Embedding` 是一张可训练的向量表。
# Token ID 用来选择这张表中的某一行。

# %%
EMBEDDING_DIM = 8

torch.manual_seed(42)
token_embedding = nn.Embedding(tokenizer.n_vocab, EMBEDDING_DIM)
input_tensor = torch.tensor(token_ids, dtype=torch.long)
token_vectors = token_embedding(input_tensor)

print("输入 ID 的形状：", tuple(input_tensor.shape))
print("Embedding 表的形状：", tuple(token_embedding.weight.shape))
print("Token 向量的形状：", tuple(token_vectors.shape))
print("第一个 Token ID：", token_ids[0])
print("第一个 Token 向量：", token_vectors[0].detach().numpy().round(3))


# %% [markdown]
# ## 4. 相同 Token ID 是否得到相同 Token 向量
#
# 运行前先预测结果。

# %%
repeated_ids = torch.tensor([token_ids[0], token_ids[0]])
repeated_vectors = token_embedding(repeated_ids)

print("重复的 Token ID：", repeated_ids.tolist())
print(
    "两个向量是否完全相同：",
    "是" if torch.equal(repeated_vectors[0], repeated_vectors[1]) else "否",
)


# %% [markdown]
# ## 5. Position Embedding 如何加入顺序信息
#
# 只有 Token Embedding 时，模型不能区分同一 Token 出现在位置 0 还是位置 5。
# Position Embedding 补充了这部分顺序信息。

# %%
positions = torch.arange(len(token_ids))
position_embedding = nn.Embedding(len(token_ids), EMBEDDING_DIM)
position_vectors = position_embedding(positions)

transformer_input = token_vectors + position_vectors

print("Token 向量形状：", tuple(token_vectors.shape))
print("位置向量形状：", tuple(position_vectors.shape))
print("Transformer 输入形状：", tuple(transformer_input.shape))

print("\n第一个 Token 向量：")
print(token_vectors[0].detach().numpy().round(3))
print("第一个位置向量：")
print(position_vectors[0].detach().numpy().round(3))
print("两者相加后的输入向量：")
print(transformer_input[0].detach().numpy().round(3))


# %% [markdown]
# ## 6. 小实验：交换两个位置向量
#
# Token 向量保持不变，但最终 Transformer 输入会改变。

# %%
swapped_position_vectors = position_vectors.clone()
swapped_position_vectors[[0, 1]] = swapped_position_vectors[[1, 0]]
swapped_input = token_vectors + swapped_position_vectors

print(
    "Token 向量是否保持不变：",
    torch.equal(token_vectors, token_embedding(input_tensor)),
)
print(
    "交换位置后最终输入是否改变：",
    not torch.equal(transformer_input, swapped_input),
)


# %% [markdown]
# ## 7. 回顾
#
# 用自己的话解释下面这条链：
#
# Text
# -> Token
# -> Token ID
# -> Token Embedding
# + Position Embedding
# -> Transformer Input
