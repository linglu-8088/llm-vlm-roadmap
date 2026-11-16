
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer


# 1. 项目路径
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
RESULT_DIR = ROOT / "results" / "week01"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# 2. 生成 50 组中英文对应文本
subjects = [
    ("机器人", "The robot"),
    ("摄像头", "The camera"),
    ("传感器", "The sensor"),
    ("视觉模型", "The vision model"),
    ("机械臂", "The robotic arm"),
    ("训练程序", "The training program"),
    ("数据处理模块", "The data processing module"),
    ("图像分类器", "The image classifier"),
    ("推理服务", "The inference service"),
    ("多模态模型", "The multimodal model"),
]

actions = [
    ("正在处理输入数据。", "is processing input data."),
    ("正在分析测试结果。", "is analyzing the test results."),
    ("正在优化运行速度。", "is optimizing runtime speed."),
    ("正在记录实验日志。", "is recording experiment logs."),
    ("正在生成预测结果。", "is generating predictions."),
]

samples = []

for subject_zh, subject_en in subjects:
    for action_zh, action_en in actions:
        pair_id = len(samples) // 2 + 1

        samples.append({
            "pair_id": pair_id,
            "language": "zh",
            "text": subject_zh + action_zh
        })

        samples.append({
            "pair_id": pair_id,
            "language": "en",
            "text": subject_en + " " + action_en
        })

assert len(samples) == 100

data = pd.DataFrame(samples)
data.to_csv(
    DATA_DIR / "day03_texts_100.csv",
    index=False,
    encoding="utf-8-sig"
)

print("测试文本数量:", len(data))


# 3. 加载两种 Tokenizer
models = {
    "Qwen2.5": "Qwen/Qwen2.5-0.5B",
    "DistilGPT2": "distilbert/distilgpt2"
}

records = []

for model_name, model_id in models.items():
    print(f"\n正在加载: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    for sample in samples:
        text = sample["text"]

        ids = tokenizer.encode(
            text,
            add_special_tokens=False
        )

        decoded = tokenizer.decode(
            ids,
            clean_up_tokenization_spaces=False
        )

        assert decoded == text, (
            f"解码不一致: {model_name}, {text}"
        )

        records.append({
            "pair_id": sample["pair_id"],
            "language": sample["language"],
            "tokenizer": model_name,
            "text": text,
            "char_count": len(text),
            "utf8_bytes": len(text.encode("utf-8")),
            "token_count": len(ids),
            "tokens_per_char": len(ids) / len(text),
        })


# 4. 保存明细结果
df = pd.DataFrame(records)

assert len(df) == 200

df.to_csv(
    RESULT_DIR / "day03_token_stats.csv",
    index=False,
    encoding="utf-8-sig"
)


# 5. Pandas 分组统计
summary = (
    df.groupby(["tokenizer", "language"])
    .agg(
        samples=("token_count", "size"),
        avg_tokens=("token_count", "mean"),
        min_tokens=("token_count", "min"),
        max_tokens=("token_count", "max"),
        avg_tokens_per_char=("tokens_per_char", "mean")
    )
    .reset_index()
)

print("\n===== Tokenizer 对比 =====")
print(summary.to_string(index=False))

summary.to_csv(
    RESULT_DIR / "day03_summary.csv",
    index=False,
    encoding="utf-8-sig"
)


# 6. 绘制平均 Token 数对比图
pivot = summary.pivot(
    index="language",
    columns="tokenizer",
    values="avg_tokens"
)

ax = pivot.plot(
    kind="bar",
    figsize=(8, 5),
    rot=0,
    title="Average Token Count by Language"
)

ax.set_xlabel("Language")
ax.set_ylabel("Average Tokens")
plt.tight_layout()
plt.savefig(
    RESULT_DIR / "day03_average_tokens.png",
    dpi=150
)
plt.close()


# 7. 绘制 Token 数量分布直方图
plt.figure(figsize=(9, 5))

for (model, lang), group in df.groupby(
    ["tokenizer", "language"]
):
    plt.hist(
        group["token_count"],
        bins=15,
        alpha=0.45,
        label=f"{model} - {lang}"
    )

plt.title("Token Count Distribution")
plt.xlabel("Token Count")
plt.ylabel("Number of Sentences")
plt.legend()
plt.tight_layout()
plt.savefig(
    RESULT_DIR / "day03_token_distribution.png",
    dpi=150
)
plt.close()

print("\n实验完成！结果保存在:", RESULT_DIR)
