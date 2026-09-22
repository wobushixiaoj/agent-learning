# %% [markdown]
# # Day 1：把“我喜欢打”变成 Transformer 输入
#
# 教材默认阅读 `OUTPUT.md`。本文件只用于需要亲手运行时逐段观察。

# %%
import tiktoken
import torch
from torch import nn

PROMPT = "我喜欢打"
tokenizer = tiktoken.get_encoding("o200k_base")
token_ids = tokenizer.encode(PROMPT)
token_pieces = [tokenizer.decode([token_id]) for token_id in token_ids]

print("文本：", PROMPT)
print("Token：", token_pieces)
print("Token ID：", token_ids)

# %%
torch.manual_seed(42)
embedding_dim = 4
input_ids = torch.tensor(token_ids)
token_embedding = nn.Embedding(tokenizer.n_vocab, embedding_dim)
position_embedding = nn.Embedding(len(token_ids), embedding_dim)

token_vectors = token_embedding(input_ids)
position_vectors = position_embedding(torch.arange(len(token_ids)))
transformer_input = token_vectors + position_vectors

print("Token 向量 Shape：", tuple(token_vectors.shape))
print("位置向量 Shape：", tuple(position_vectors.shape))
print("Transformer 输入 Shape：", tuple(transformer_input.shape))
print(transformer_input.detach().numpy().round(3))
