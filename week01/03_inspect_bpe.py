
from transformers import AutoTokenizer

text = "数据处理模块正在分析测试结果。"

models = {
    "Qwen2.5": "Qwen/Qwen2.5-0.5B",
    "DistilGPT2": "distilbert/distilgpt2"
}

for name, model_id in models.items():
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=True
    )

    ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    print(f"\n===== {name} =====")
    print("Token 数量:", len(ids))
    print("Token IDs:", ids)

    for i, token_id in enumerate(ids):
        token = tokenizer.convert_ids_to_tokens(token_id)
        piece = tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False
        )

        print(
            f"{i:2d} | ID={token_id:6d} "
            f"| Raw={token!r} "
            f"| Decode={piece!r}"
        )

print("\n完整解码:")
restored = tokenizer.decode(
    ids,
    clean_up_tokenization_spaces=False
)

print(restored)
print("是否完全一致:", restored == text)