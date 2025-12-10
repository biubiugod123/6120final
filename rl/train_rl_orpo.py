"""
ORPO Training Script
Model: google/flan-t5-base
Dataset: cot_rl_train.jsonl
command: python rl/train_rl_orpo.py \
    --output_dir results/rl_orpo \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6

"""

import json
import torch
from dataclasses import dataclass, field
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    HfArgumentParser
)
from trl import ORPOTrainer, ORPOConfig
from tqdm import tqdm

@dataclass
class ScriptArguments:
    model_name_or_path: str = field(default="google/flan-t5-base", metadata={"help": "Model name"})
    train_data_path: str = field(default="data/cot_rl_train.jsonl", metadata={"help": "Your training data"})

def load_jsonl(file_path):
    print(f"Loading data from {file_path}...")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def build_preference_dataset(data, tokenizer, model, device):
    print(f"Building ORPO dataset (Generating negatives)...")

    formatted_data = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }

    model.eval()
    model.to(device)

    batch_size = 16
    data_slice = data

    for i in tqdm(range(0, len(data_slice), batch_size)):
        batch_items = data_slice[i : i + batch_size]

        prompts = [item["input"] for item in batch_items]
        targets = [item["output"] for item in batch_items]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_k=50
            )

        generated_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for prompt, target, generated in zip(prompts, targets, generated_responses):
            if generated.strip() == target.strip():
                continue

            formatted_data["prompt"].append(prompt)
            formatted_data["chosen"].append(target)
            formatted_data["rejected"].append(generated)

    print(f"Dataset ready. Effective ORPO Samples: {len(formatted_data['prompt'])}")
    model.cpu()
    torch.cuda.empty_cache()

    return Dataset.from_dict(formatted_data)

def main():
    parser = HfArgumentParser((ScriptArguments, ORPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.is_encoder_decoder = True

    if training_args.output_dir is None:
        training_args.output_dir = "results/rl_orpo"

    print(f"Loading model: {script_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(script_args.model_name_or_path)

    raw_data = load_jsonl(script_args.train_data_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = build_preference_dataset(raw_data, tokenizer, model, device)

    print("Initializing ORPO Trainer...")
    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    print("Starting ORPO Training...")
    trainer.train()

    print(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()