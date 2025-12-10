"""
PPO Training Script for Chain-of-Thought Reasoning

This script trains a language model using PPO (Proximal Policy Optimization)
to improve chain-of-thought reasoning abilities.

Dependencies:
accelerate               1.12.0
datasets                 4.4.1
peft                     0.18.0
transformers             4.57.3
trl                      0.9.6

    pip install accelerate==1.12.0 datasets==4.4.1 peft==0.18.0 transformers==4.57.3 trl==0.9.6

run:
    test: python -m rl.train_rl_ppo --train_data data/cot_sft_formatted.jsonl --max_samples 10 --num_epochs 1 --batch_size 8
"""

import argparse
import json
import os
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
)
from trl import PPOConfig, PPOTrainer, AutoModelForSeq2SeqLMWithValueHead

from cot_framework.reward import RewardFunction


@dataclass
class RLTrainingConfig:
    """Configuration for RL training."""

    # Model configs
    model_name_or_path: str = "google/flan-t5-base"
    sft_checkpoint: str = None
    model_type: str = "seq2seq"

    # Data configs
    train_data_path: str = "data/cot_rl_train.jsonl"
    eval_data_path: str = None
    max_train_samples: int = 10000

    # RL configs
    learning_rate: float = 1e-5
    batch_size: int = 4
    mini_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    ppo_epochs: int = 4
    num_train_epochs: int = 1
    max_steps: int = -1

    # Generation configs
    max_input_length: int = 512
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95

    # Reward configs
    reward_correctness_weight: float = 1.0
    reward_format_weight: float = 0.2
    reward_quality_weight: float = 0.1

    # Other configs
    output_dir: str = "results/rl_ppo"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def load_jsonl_dataset(file_path: str, max_samples: int = None) -> List[Dict]:
    """Load JSONL dataset."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data.append(json.loads(line))
    return data


def prepare_dataset(config: RLTrainingConfig) -> Dataset:
    """
    Prepare dataset for PPO training.

    Returns:
        HuggingFace Dataset with 'query' and 'reference' fields
    """
    # Load data
    data = load_jsonl_dataset(config.train_data_path, config.max_train_samples)

    # Extract queries (inputs) and references (targets)
    queries = [item["input"] for item in data]
    references = [item.get("target", item.get("output", "")) for item in data]

    # Create dataset
    dataset = Dataset.from_dict({
        "query": queries,
        "reference": references
    })

    return dataset

def build_model_and_tokenizer(config: RLTrainingConfig):
    """
    Build model and tokenizer for RL training.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    # Load base model or SFT checkpoint
    model_path = config.sft_checkpoint if config.sft_checkpoint else config.model_name_or_path

    if config.model_type == "seq2seq":
        # 1. Policy model with value head
        model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_path)
        model.generation_config = model.pretrained_model.generation_config

        # 2. Value model
        value_model = model

        # 3. Reference model
        ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_path)
        ref_model.eval()

        for m in (model, ref_model):
            if not hasattr(m, "base_model_prefix"):
                m.base_model_prefix = "pretrained_model"
            if not hasattr(m, "is_gradient_checkpointing"):
                m.is_gradient_checkpointing = False

    else:
        raise NotImplementedError("Causal LM PPO training not implemented yet")

    return model, tokenizer, ref_model, value_model


def prepare_tokenized_dataset(config, tokenizer):
    """
    load data and tokenized dataset
    """
    print("\n[1/4] Loading and processing dataset...")
    dataset = prepare_dataset(config)
    print(f"  Loaded {len(dataset)} training samples")

    query_ref_map = {row["query"]: row["reference"] for row in dataset}
    def tokenize_batch(batch):
        tokenized = tokenizer(
            batch["query"],
            max_length=config.max_input_length,
            truncation=True,
            padding="max_length",
        )
        ref_tokens = tokenizer(
            batch["reference"],
            max_length=config.max_new_tokens,
            truncation=True,
            padding="max_length",
        )
        tokenized["labels"] = ref_tokens["input_ids"]
        return tokenized

    print("  Tokenizing dataset...")
    dataset = dataset.map(tokenize_batch, batched=True)
    dataset = dataset.remove_columns(["query", "reference"])

    return dataset, query_ref_map


def initialize_ppo_components(config):
    """
    initialize the ppo
    """
    print("\n[2/4] Initializing model and components...")
    model, tokenizer, ref_model, value_model = build_model_and_tokenizer(config)

    reward_fn = RewardFunction(weights={
        "correctness": config.reward_correctness_weight,
        "format": config.reward_format_weight,
        "quality": config.reward_quality_weight
    })

    # PPO Config
    ppo_config = PPOConfig(
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        mini_batch_size=config.mini_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        ppo_epochs=config.ppo_epochs,
        seed=config.seed,
    )

    return model, tokenizer, ref_model, reward_fn, ppo_config


def process_batch_tensors(batch, device):
    """
    Handles DataLoader output and ensures shape is always [Batch_Size, Seq_Len].
    """
    query_tensors = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    if isinstance(query_tensors, list):
        query_tensors = torch.stack(query_tensors).to(device)
    else:
        query_tensors = query_tensors.to(device)

    if isinstance(attention_mask, list):
        attention_mask = torch.stack(attention_mask).to(device)
    else:
        attention_mask = attention_mask.to(device)

    if query_tensors.shape[0] > query_tensors.shape[1]:
        query_tensors = query_tensors.T
        attention_mask = attention_mask.T

    return query_tensors, attention_mask


def run_training_loop(config, ppo_trainer, tokenizer, model, reward_fn, query_ref_map):
    """
    core loop
    """
    print(f"\n[4/4] Starting PPO training loop ({config.num_train_epochs} Epochs)...")
    print("=" * 50)

    generation_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_k": config.top_k,
        "top_p": config.top_p,
        "do_sample": True,
    }

    os.makedirs(config.output_dir, exist_ok=True)

    for epoch in range(config.num_train_epochs):
        print(f"\n>>> Epoch {epoch + 1}/{config.num_train_epochs}")
        progress_bar = tqdm(enumerate(ppo_trainer.dataloader), total=len(ppo_trainer.dataloader))

        for step, batch in progress_bar:
            # 1. Process Data (Ensure process_batch_tensors is fixed!)
            query_tensors, attention_mask = process_batch_tensors(batch, config.device)

            # 2. Generate Responses
            with torch.no_grad():
                response_tensors = model.generate(
                    input_ids=query_tensors,
                    attention_mask=attention_mask,
                    **generation_kwargs
                )


            # 3. Decode
            batch_responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            current_queries = tokenizer.batch_decode(query_tensors, skip_special_tokens=True)
            references = [query_ref_map.get(q, "") for q in current_queries]

            # 4. Calculate Rewards
            rewards = reward_fn(batch_responses, references)
            rewards = [torch.tensor(r, dtype=torch.float32, device=config.device) for r in rewards]

            # 5. Prepare List for PPO
            # Because we fixed Step 1, these should now be naturally correct.
            # We simply unbind dim=0 to get a list of [Batch_Size] items.
            query_list = torch.unbind(query_tensors, dim=0)
            response_list = torch.unbind(response_tensors, dim=0)

            # 6. PPO Step
            try:
                stats = ppo_trainer.step(list(query_list), list(response_list), rewards)
            except ValueError as e:
                print(f"  [Error] Step {step} failed: {e}")
                print(f"  Debug Info -> Q_List: {len(query_list)}, R_List: {len(response_list)}, Rewards: {len(rewards)}")
                break # Stop loop to read the error

            # 7. Logging & Checkpointing
            if step % config.logging_steps == 0:
                mean_reward = torch.stack(rewards).mean()
                loss_val = stats.get('ppo/loss/total', 0) if stats else 0
                progress_bar.set_description(f"R: {mean_reward:.3f} | L: {loss_val:.3f}")

            if config.save_steps > 0 and step % config.save_steps == 0 and step > 0:
                save_path = os.path.join(config.output_dir, f"checkpoint-{epoch}-{step}")
                ppo_trainer.save_pretrained(save_path)

    final_save_path = os.path.join(config.output_dir, "final_model")
    ppo_trainer.save_pretrained(final_save_path)
    print(f"\nTraining complete! Final model saved to {final_save_path}")


def train_ppo(config: RLTrainingConfig):
    """
    Main Entry Point: Orchestrates the PPO training process.
    """
    print("=" * 50)
    print("Starting PPO Training for Chain-of-Thought")
    print("=" * 50)

    # 0. Global Setup
    torch.manual_seed(config.seed)

    # 1. Initialize Components (Model, Tokenizer, Config)
    model, tokenizer, ref_model, reward_fn, ppo_config = initialize_ppo_components(config)

    # 2. Prepare Data (Requires tokenizer)
    dataset, query_ref_map = prepare_tokenized_dataset(config, tokenizer)

    # 3. Initialize Trainer
    print("\n[3/4] Initializing PPOTrainer...")
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        tokenizer=tokenizer,
        model=model,
        ref_model=ref_model,
        dataset=dataset,
    )

    # 4. Run Loop
    run_training_loop(config, ppo_trainer, tokenizer, model, reward_fn, query_ref_map)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train CoT model with PPO")

    parser.add_argument("--model_name", type=str, default="google/flan-t5-base")
    parser.add_argument("--sft_checkpoint", type=str, default=None,
                       help="Path to SFT fine-tuned checkpoint")

    parser.add_argument("--train_data", type=str, default="data/cot_sft_formatted.jsonl")
    parser.add_argument("--max_samples", type=int, default=10000)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="results/rl_ppo")

    parser.add_argument("--reward_correctness", type=float, default=1.0)
    parser.add_argument("--reward_format", type=float, default=0.2)
    parser.add_argument("--reward_quality", type=float, default=0.1)

    args = parser.parse_args()

    config = RLTrainingConfig(
        model_name_or_path=args.model_name,
        sft_checkpoint=args.sft_checkpoint,
        train_data_path=args.train_data,
        max_train_samples=args.max_samples,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        output_dir=args.output_dir,
        reward_correctness_weight=args.reward_correctness,
        reward_format_weight=args.reward_format,
        reward_quality_weight=args.reward_quality,
    )

    # Train
    train_ppo(config)




if __name__ == "__main__":
    main()
