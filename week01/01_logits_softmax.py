
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(42)

# 一个 Token 的 8 维特征向量
embedding = torch.tensor([
    [-0.9718, 1.3139, -0.2478, 0.5187,
     -0.2360, 0.1484, -0.3488, -2.5427]
])

# 教学实验：输出 3 个候选 Token 的分数
linear = nn.Linear(8, 3)

# 线性层计算 Logits
logits = linear(embedding)

# Softmax 转换为概率
probs = F.softmax(logits, dim=-1)

print("输入形状:", embedding.shape)
print("Logits 形状:", logits.shape)
print("Logits:", logits.detach())
print("预测概率:", probs.detach())
print("概率之和:", probs.sum().item())

# 选出预测概率最高的候选 Token
candidates = ["学习", "模型", "机器人"]
pred_id = probs.argmax(dim=-1).item()

print("预测 Token:", candidates[pred_id])

# ============================
# Cross Entropy 实验
# ============================

# 三个候选：
# 0 = 学习
# 1 = 模型
# 2 = 机器人

# 假设正确答案是「学习」
target = torch.tensor([2])

# 计算交叉熵
loss = F.cross_entropy(logits, target)

# 计算困惑度
ppl = torch.exp(loss)

print("\n===== Loss 实验 =====")
print("正确 Token:", candidates[target.item()])
print("Cross Entropy:", loss.item())
print("Perplexity:", ppl.item())

# 检验手动计算是否一致
manual_loss = -torch.log(probs[0, target.item()])
print("手动计算 Loss:", manual_loss.item())
