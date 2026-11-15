
from transformers import AutoTokenizer

# 只下载 Tokenizer，不加载完整语言模型
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

texts = [
    "我喜欢深度学习",
    "I love deep learning",
    "我正在学习Transformer模型",
    "我喜欢机器学习",
    "我喜欢自然语言处理",
]

print("词表大小:", len(tokenizer))

for text in texts:
    print("\n" + "=" * 50)
    print("原始文本:", text)

    # 文本切分
    tokens = tokenizer.tokenize(text)
    print("Tokens:", tokens)

    # 文本编码成整数 ID
    ids = tokenizer.encode(text, add_special_tokens=False)
    print("Token IDs:", ids)

    readable_tokens = [
        tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False
        )
        for token_id in ids
    ]

    print("可读 Tokens:", readable_tokens)

    for token_id, token_text in zip(ids, readable_tokens):
        print(f"ID={token_id:6d}  Text={token_text!r}")

    # 反向还原文本
    decoded = tokenizer.decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    print("解码结果:", decoded)
    print("Token 数量:", len(ids))

    assert decoded == text
