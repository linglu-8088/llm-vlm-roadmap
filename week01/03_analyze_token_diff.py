
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

path = ROOT / "results/week01/day03_token_stats.csv"

df = pd.read_csv(path)

# 只看中文
zh = df[df["language"] == "zh"]

# 将两种 Tokenizer 的结果按句子对齐
result = zh.pivot(
    index=["pair_id", "text"],
    columns="tokenizer",
    values="token_count"
).reset_index()

# 计算 Token 数量差异
result["diff"] = (
    result["DistilGPT2"] - result["Qwen2.5"]
)

result = result.sort_values(
    "diff", ascending=False
)

print("===== 中文 Token 数量差异最大的 10 条 =====")

print(
    result[
        ["text", "Qwen2.5", "DistilGPT2", "diff"]
    ].head(10).to_string(index=False)
)

print("\n平均差异:", result["diff"].mean())
