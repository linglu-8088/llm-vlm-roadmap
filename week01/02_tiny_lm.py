
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 构造一个玩具词表
vocab = ["<BOS>", "我喜欢", "深度", "学习", "<EOS>"]
vocab_size = len(vocab)

# 完整句子
sequence = torch.tensor([0, 1, 2, 3, 4], device=device)

# 2. 输入与目标错开一个 Token
inputs = sequence[:-1]
targets = sequence[1:]

print("输入:", [vocab[i] for i in inputs.tolist()])
print("目标:", [vocab[i] for i in targets.tolist()])


# 3. 定义极简下一个 Token 预测网络
class TinyLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids):
        hidden = self.embedding(input_ids)
        logits = self.lm_head(hidden)
        return logits


model = TinyLM(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)

# 4. 查看训练前的预测
with torch.no_grad():
    logits = model(inputs)
    probs = F.softmax(logits, dim=-1)
    print("Logits shape:", logits.shape)
    print("第一个位置的预测概率:", probs[0].tolist())

# 5. 训练：根据当前 Token 预测下一个 Token
for step in range(101):
    optimizer.zero_grad()

    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)

    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        ppl = torch.exp(loss)
        print(
            f"Step {step:3d} | "
            f"Loss: {loss.item():.4f} | "
            f"PPL: {ppl.item():.4f}"
        )

# 6. 从 <BOS> 开始逐个预测
model.eval()
generated = [0]

with torch.no_grad():
    for _ in range(4):
        current = torch.tensor(
            [generated[-1]],
            device=device
        )
        logits = model(current)
        next_id = logits.argmax(dim=-1).item()
        generated.append(next_id)

        if next_id == 4:
            break

print("生成结果:", [vocab[i] for i in generated])

print("\n===== 逐位置预测 =====")

model.eval()

with torch.no_grad():
    logits = model(inputs)
    predictions = logits.argmax(dim=-1)

    for x, pred, target in zip(
        inputs.tolist(),
        predictions.tolist(),
        targets.tolist()
    ):
        print(
            f"输入: {vocab[x]:8s} "
            f"预测: {vocab[pred]:8s} "
            f"目标: {vocab[target]:8s}"
        )
