from cot_framework.data_loader import load_cot_dataset
from cot_framework.sft_formatter import build_sft_dataset
import jsonlines
import os
import random

if __name__ == "__main__":
    print(">>> CWD =", os.getcwd())

    # 1. Load raw dataset
    df = load_cot_dataset("data/CoT_collection.json")

    # 2. Filter tasks
    # Keep only math and reading comprehension tasks (math_dataset, drop, quoref)
    tasks = ["math_dataset", "drop", "quoref"]
    df = df[df["task"].isin(tasks)]
    print(f"[Pipeline] Filtered dataset size: {len(df)}")

    # 3. Format for SFT (Initial formatting: creates 'input' and 'target' fields)
    formatted_raw = build_sft_dataset(df)

    # ================= [Step A: Fix Format (Standardize Schema)] =================
    # Convert {input, target} format to {instruction, input, output} format
    # This ensures compatibility with standard trainers (like SFTTrainer)
    fixed_data = []
    for item in formatted_raw:
        new_item = {
            "instruction": "",  # Leave empty, or add a system prompt like "Please reason step by step."
            "input": item.get("input", ""),
            "output": item.get("target", "")  # Rename 'target' to 'output'
        }
        fixed_data.append(new_item)

    print(f"[Pipeline] Format fixed. Ready to split.")

    # ================= [Step B: Data Split] =================
    random.seed(42)
    random.shuffle(fixed_data)  # Shuffle the fixed data to ensure random distribution

    total_len = len(fixed_data)

    # Set split ratios
    test_ratio = 0.10  # 10% for Testing (Do not use for training)
    sft_ratio = 0.45  # 45% for SFT Stage
    # The remaining 45% will be used for RL (PPO) Stage

    # Calculate split indices
    n_test = int(total_len * test_ratio)
    n_sft = int(total_len * sft_ratio)

    # Slice the data
    test_data = fixed_data[:n_test]  # First 10%
    sft_data = fixed_data[n_test: n_test + n_sft]  # Middle 45%
    rl_data = fixed_data[n_test + n_sft:]  # Last 45%

    print(f"\n--- Data Split Report (Format Fixed) ---")
    print(f"Total samples: {total_len}")
    print(f"Test Set     : {len(test_data)} \t-> saves to data/cot_test.jsonl")
    print(f"SFT Set      : {len(sft_data)} \t-> saves to data/cot_sft_train.jsonl")
    print(f"RL Set       : {len(rl_data)} \t-> saves to data/cot_rl_train.jsonl")

    # ================= [Step C: Save Files] =================

    # Save Test Set
    with jsonlines.open("data/cot_test.jsonl", "w") as writer:
        writer.write_all(test_data)

    # Save SFT Training Set
    with jsonlines.open("data/cot_sft_train.jsonl", "w") as writer:
        writer.write_all(sft_data)

    # Save RL (PPO) Training Set
    with jsonlines.open("data/cot_rl_train.jsonl", "w") as writer:
        writer.write_all(rl_data)

    print("\n[Pipeline] Done! All datasets are fixed and split.")