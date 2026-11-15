
import torch
import torch.nn as nn

torch.manual_seed(42)

# 昨天 Qwen2.5 Tokenizer 输出的真实 Token ID
token_ids = torch.tensor(
    [109366, 102217, 100134],
    dtype=torch.long
)

# 昨天查看到的词表大小
vocab_size = 151665

# 教学实验：每个 Token 对应 8 维向量
embedding_dim = 8

embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim
)

vectors = embedding(token_ids)

print("Token IDs:", token_ids.tolist())
print("Embedding 权重形状:", embedding.weight.shape)
print("输入形状:", token_ids.shape)
print("输出形状:", vectors.shape)
print("Embedding 向量:")
print(vectors.detach())
