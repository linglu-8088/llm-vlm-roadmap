
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# 1. 固定一个 8 维输入向量
x = torch.tensor([
    [-0.9718, 1.3139, -0.2478, 0.5187,
     -0.2360, 0.1484, -0.3488, -2.5427]
])

# 2. 三个候选 Token
candidates = ["学习", "模型", "机器人"]

# 正确答案指定为机器人（索引 2）
target = torch.tensor([2])

# 3. 定义 Linear 层
model = nn.Linear(8, 3)

# 4. 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

print("===== 训练前 =====")
with torch.no_grad():
    logits = model(x)
    probs = F.softmax(logits, dim=-1)
    print("预测概率:", probs.tolist())
    print("预测结果:", candidates[probs.argmax().item()])

# 5. 训练 31 步
for step in range(31):

    optimizer.zero_grad()

    logits = model(x)
    loss = F.cross_entropy(logits, target)

    # 第一次反向传播时观察 Logits 的梯度
    if step == 0:
        logits.retain_grad()

    loss.backward()

    if step == 0:
        print("\n第 0 步的 Logits 梯度:")
        print(logits.grad)

    optimizer.step()

    if step % 5 == 0:
        print(f"Step {step:2d} | Loss: {loss.item():.4f}")

# 6. 训练后的预测
print("\n===== 训练后 =====")
with torch.no_grad():
    logits = model(x)
    probs = F.softmax(logits, dim=-1)

    print("预测概率:", probs.tolist())
    print("预测结果:", candidates[probs.argmax().item()])
