
from pathlib import Path
import json
import re
import unicodedata

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader


# ========== 1. 路径配置 ==========

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data/raw/day03_texts_100.csv"
PROCESSED = ROOT / "data/processed"
RESULTS = ROOT / "results/week01"

PROCESSED.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)


# ========== 2. 读取和清洗文本 ==========

def clean_text(text):
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


df = pd.read_csv(INPUT, encoding="utf-8-sig")

print("原始样本数量:", len(df))

df = df.dropna(subset=["pair_id", "language", "text"])
df["text"] = df["text"].map(clean_text)

df = df[df["text"] != ""]
df = df.drop_duplicates(
    subset=["language", "text"]
).reset_index(drop=True)

assert len(df) == 100, "请检查原始数据或清洗结果"

df.to_csv(
    PROCESSED / "day04_cleaned.csv",
    index=False,
    encoding="utf-8-sig"
)

print("清洗后样本数量:", len(df))


# ========== 3. 划分 Train / Test ==========

# 按中英文配对 ID 划分，避免配对样本泄漏
pair_ids = sorted(df["pair_id"].unique().tolist())

pair_dataset = Dataset.from_dict({
    "pair_id": pair_ids
})

pair_splits = pair_dataset.train_test_split(
    test_size=0.2,
    seed=42
)

train_pairs = set(pair_splits["train"]["pair_id"])
test_pairs = set(pair_splits["test"]["pair_id"])

assert train_pairs.isdisjoint(test_pairs)

train_df = df[df["pair_id"].isin(train_pairs)]
test_df = df[df["pair_id"].isin(test_pairs)]

dataset = DatasetDict({
    "train": Dataset.from_pandas(
        train_df.reset_index(drop=True),
        preserve_index=False
    ),
    "test": Dataset.from_pandas(
        test_df.reset_index(drop=True),
        preserve_index=False
    )
})

print("\n===== Dataset Split =====")
print(dataset)

assert len(dataset["train"]) == 80
assert len(dataset["test"]) == 20


# ========== 4. 加载 Tokenizer ==========

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    local_files_only=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\nPAD Token:", tokenizer.pad_token)
print("PAD Token ID:", tokenizer.pad_token_id)


# ========== 5. 批量 Tokenization ==========

def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        add_special_tokens=False,
        truncation=True,
        max_length=32,
        padding=False
    )


tokenized = dataset.map(
    tokenize_function,
    batched=True
)

print("\n===== Tokenization 示例 =====")

sample = tokenized["train"][0]

print("文本:", sample["text"])
print("Input IDs:", sample["input_ids"])
print("Attention Mask:", sample["attention_mask"])

tokenized.save_to_disk(
    str(PROCESSED / "day04_tokenized")
)


# ========== 6. 动态 Padding ==========

collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="longest",
    return_tensors="pt"
)

# DataLoader 只需要模型输入字段
keep_columns = {"input_ids", "attention_mask"}

train_data = tokenized["train"].remove_columns([
    col for col in tokenized["train"].column_names
    if col not in keep_columns
])


# ========== 7. PyTorch DataLoader ==========

train_loader = DataLoader(
    train_data,
    batch_size=4,
    shuffle=True,
    collate_fn=collator
)

batch = next(iter(train_loader))

print("\n===== DataLoader Batch =====")
print("input_ids shape:", batch["input_ids"].shape)
print("attention_mask shape:", batch["attention_mask"].shape)

assert batch["input_ids"].shape == batch["attention_mask"].shape
assert batch["input_ids"].shape[0] == 4

print("input_ids:\n", batch["input_ids"])
print("attention_mask:\n", batch["attention_mask"])


# ========== 8. 强制演示 Padding ==========

# 选择最长和最短的两个样本，方便观察补齐
rows = list(tokenized["train"])
shortest = min(rows, key=lambda x: len(x["input_ids"]))
longest = max(rows, key=lambda x: len(x["input_ids"]))

demo_features = [
    {
        "input_ids": row["input_ids"],
        "attention_mask": row["attention_mask"]
    }
    for row in [shortest, longest]
]

demo = collator(demo_features)

print("\n===== Padding 演示 =====")
print("短文本:", shortest["text"])
print("长文本:", longest["text"])
print("input_ids:\n", demo["input_ids"])
print("attention_mask:\n", demo["attention_mask"])

assert demo["input_ids"].shape == demo["attention_mask"].shape
assert torch.all(
    demo["attention_mask"].sum(dim=1) > 0
)


# ========== 9. 保存实验结果 ==========

summary = {
    "raw_samples": 100,
    "cleaned_samples": len(df),
    "train_samples": len(dataset["train"]),
    "test_samples": len(dataset["test"]),
    "batch_size": 4,
    "batch_shape": list(batch["input_ids"].shape),
    "padding_demo_shape": list(demo["input_ids"].shape),
    "padding_demo_mask": demo["attention_mask"].tolist()
}

(RESULTS / "day04_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("\nDay 04 数据处理完成！")


# 检查有效 Token 数量
valid_lengths = batch["attention_mask"].sum(dim=1)

print("\n===== 有效 Token 长度 =====")
print(valid_lengths.tolist())

# 检查输入与 Mask 形状
assert batch["input_ids"].shape == batch["attention_mask"].shape

# 确认 Padding 位置使用 PAD Token ID
pad_positions = batch["attention_mask"] == 0

assert torch.all(
    batch["input_ids"][pad_positions] == tokenizer.pad_token_id
)

print("Padding 检查通过！")

train_ids = set(dataset["train"]["pair_id"])
test_ids = set(dataset["test"]["pair_id"])

overlap = train_ids & test_ids

print("\n===== 数据泄漏检查 =====")
print("训练集 Pair 数:", len(train_ids))
print("测试集 Pair 数:", len(test_ids))
print("交集:", overlap)

assert len(overlap) == 0

print("数据泄漏检查通过！")