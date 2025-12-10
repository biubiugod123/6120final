"""
Full Benchmark Evaluation Script: Base vs. DPO vs. ORPO
-------------------------------------------------------
Description:
    This script compares the performance of the original Flan-T5 model against
    the DPO-tuned and ORPO-tuned versions.
    It evaluates them on two tasks:
    1. GSM8K (Arithmetic Reasoning)
    2. StrategyQA (Commonsense Reasoning)

Usage:
    Ensure you have  'results/rl_dpo'
    and 'results/rl_orpo'.
    Run: python evaluate_rl.py
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from tqdm import tqdm

NUM_SAMPLES = 300
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS_TO_EVAL = {
    "Base Model": "google/flan-t5-base",
    "DPO Model": "results/rl_dpo/",
    "ORPO Model": "results/rl_orpo/"
}


def load_sampled_dataset(dataset_name, split, num_samples):
    """load randowm sample"""
    print(f"Loading {dataset_name} ({split})...")
    try:
        if dataset_name == "gsm8k":
            ds = load_dataset("gsm8k", "main", split=split)
        else:
            ds = load_dataset("tasksource/strategy-qa", split="train", download_mode="force_redownload")

        # set random seed
        if len(ds) > num_samples:
            ds = ds.shuffle(seed=42).select(range(num_samples))
        return ds
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return []


def extract_answer_gsm8k(text):
    """
    Parses the model output to extract the numerical answer for GSM8K.
    Strategies:
    1. Look for 'Final Answer: <number>'
    2. Fallback: Find the last number in the text
    """
    text = text.replace(',', '')
    match = re.search(r"Final Answer:.*?(\-?\d+\.?\d*)", text, re.IGNORECASE)
    if match: return match.group(1)
    numbers = re.findall(r"\-?\d+\.?\d*", text)
    if numbers: return numbers[-1]
    return text.strip()


def extract_answer_sqa(text):
    """
    Parses model output for StrategyQA (Boolean Question Answering).
    Returns True for 'yes' and False for 'no'.
    """
    text = text.lower()
    if "yes" in text and "no" not in text: return True
    if "no" in text and "yes" not in text: return False
    return None


def evaluate_dataset(model, tokenizer, dataset, task_name):
    """
    Main evaluation loop for a single model on a single dataset.

    Args:
        model: Loaded Seq2Seq model
        tokenizer: Loaded tokenizer
        dataset: The HF dataset object
        task_name: 'GSM8K' or 'StrategyQA'

    Returns:
        float: Accuracy score (0.0 to 1.0)
    """
    correct = 0
    total = 0
    instruction_template = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n"
        "You must output the final answer at the end in this format: Final Answer: <answer>.\n\n"
        "### Instruction:\n{input}\n\n"
        "### Response:Let's think step by step. "
    )

    inputs_buffer = []
    targets_buffer = []

    print(f"Evaluating {task_name} ({len(dataset)} samples)...")
    progress_bar = tqdm(total=len(dataset))

    for i, item in enumerate(dataset):
        if task_name == "GSM8K":
            question = item['question']
            target = item['answer'].split('####')[-1].strip().replace(',', '')
        else:
            question = item['question']
            target = item['answer']

        prompt = instruction_template.format(input=question)
        inputs_buffer.append(prompt)
        targets_buffer.append(target)

        if len(inputs_buffer) == BATCH_SIZE or i == len(dataset) - 1:
            inputs_tok = tokenizer(inputs_buffer, return_tensors="pt", padding=True, truncation=True,
                                   max_length=512).to(DEVICE)
            with torch.no_grad():
                outputs = model.generate(**inputs_tok, max_new_tokens=128, do_sample=False)
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for pred_text, gold_ans in zip(decoded_preds, targets_buffer):
                if task_name == "GSM8K":
                    pred_val = extract_answer_gsm8k(pred_text)
                    try:
                        if float(pred_val) == float(gold_ans): correct += 1
                    except:
                        pass
                else:
                    pred_bool = extract_answer_sqa(pred_text)
                    if pred_bool == gold_ans: correct += 1

            total += len(inputs_buffer)
            progress_bar.update(len(inputs_buffer))
            inputs_buffer = []
            targets_buffer = []

    progress_bar.close()
    return correct / total if total > 0 else 0.0


def main():
    # 1. prepare data
    gsm8k_data = load_sampled_dataset("gsm8k", "test", NUM_SAMPLES)
    sqa_data = load_sampled_dataset("strategy_qa", "validation", NUM_SAMPLES)

    if not gsm8k_data and not sqa_data:
        print("Failed to load datasets.")
        return

    # 2. evaluate
    results = {}

    for model_label, model_path in MODELS_TO_EVAL.items():
        print(f"\n" + "=" * 40)
        print(f"Evaluating: {model_label}")
        print("=" * 40)

        if model_label != "Base Model" and not os.path.exists(model_path):
            print(f"Path {model_path} not found. Skipping...")
            results[model_label] = {"GSM8K": 0.0, "StrategyQA": 0.0}
            continue

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
            model.eval()

            scores = {}
            if gsm8k_data: scores['GSM8K'] = evaluate_dataset(model, tokenizer, gsm8k_data, "GSM8K")
            if sqa_data: scores['StrategyQA'] = evaluate_dataset(model, tokenizer, sqa_data, "StrategyQA")

            results[model_label] = scores

            del model
            del tokenizer
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error evaluating {model_label}: {e}")
            results[model_label] = {"GSM8K": 0.0, "StrategyQA": 0.0}

    # 3. output
    print("\n" + "=" * 55)
    print(f"{'Task':<12} | {'Base':<8} | {'DPO':<8} | {'ORPO':<8}")
    print("-" * 55)

    tasks = ['GSM8K', 'StrategyQA']
    for t in tasks:
        base = results.get("Base Model", {}).get(t, 0)
        dpo = results.get("DPO Model", {}).get(t, 0)
        orpo = results.get("ORPO Model", {}).get(t, 0)
        print(f"{t:<12} | {base:.3f}    | {dpo:.3f}    | {orpo:.3f}")

        methods = list(MODELS_TO_EVAL.keys())  # ['Base Model', 'DPO Model', 'ORPO Model']
        colors = ['#a1c9f4', '#ff9f9b', '#8de5a1']

        for task_name in tasks:
            scores = [results.get(m, {}).get(task_name, 0.0) for m in methods]

            plt.figure(figsize=(8, 6), dpi=150)
            plt.style.use('seaborn-v0_8-whitegrid')

            x_pos = np.arange(len(methods))
            bars = plt.bar(x_pos, scores, color=colors, width=0.6, alpha=0.9, edgecolor='grey')

            plt.title(f'Performance on {task_name} (N={NUM_SAMPLES})', fontsize=14, fontweight='bold', pad=15)
            plt.ylabel('Accuracy', fontsize=12)
            plt.xticks(x_pos, methods, fontsize=11)

            limit = max(max(scores) * 1.2, 0.05)
            plt.ylim(0, limit)

            plt.bar_label(bars, fmt='%.3f', padding=3, fontsize=11, fontweight='bold')

            filename = f"comparison_{task_name}.png"
            plt.savefig(filename, bbox_inches='tight')
            print(f"\n✅ Plot saved: {filename}")
            plt.close()


if __name__ == "__main__":
    main()
