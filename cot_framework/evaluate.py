import re
import torch
from typing import List, Tuple, Dict
from tqdm.auto import tqdm


def normalize_text(s: str) -> str:
    """
    Normalize text for comparison:
    - Convert to lowercase
    - Strip whitespace
    - Remove trailing periods
    - Normalize multiple spaces to single space
    """
    s = str(s).strip().lower()
    s = re.sub(r'\.$', '', s)      # Remove trailing period
    s = re.sub(r'\s+', ' ', s)     # Normalize whitespace
    return s


def exact_match(prediction: str, reference: str) -> bool:
    """
    Check if prediction exactly matches reference after normalization.

    Args:
        prediction: Model's predicted answer
        reference: Ground truth answer

    Returns:
        True if exact match, False otherwise
    """
    return normalize_text(prediction) == normalize_text(reference)


def extract_final_answer(text: str) -> str:
    """
    Extract the final answer from model output.
    Looks for "Final Answer:" pattern and extracts content after it.

    Args:
        text: Full model output (may include rationale + final answer)

    Returns:
        Extracted final answer, or the whole text if pattern not found
    """
    # Look for "Final Answer:" pattern
    pattern = r'Final Answer:\s*(.+?)(?:\n|$)'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

    if match:
        return match.group(1).strip()

    # If no "Final Answer:" found, return the last line
    lines = text.strip().split('\n')
    return lines[-1].strip() if lines else text.strip()


def compute_exact_match_accuracy(predictions: List[str],
                                  references: List[str]) -> float:
    """
    Compute exact match accuracy over a list of predictions and references.

    Args:
        predictions: List of model predictions
        references: List of ground truth answers

    Returns:
        Accuracy (float between 0 and 1)
    """
    assert len(predictions) == len(references), \
        f"Prediction count ({len(predictions)}) != reference count ({len(references)})"

    correct = sum(exact_match(p, r) for p, r in zip(predictions, references))
    return correct / len(references) if references else 0.0


def evaluate_model(model,
                   tokenizer,
                   eval_data: List[Dict],
                   device: str = "cuda",
                   batch_size: int = 8,
                   max_input_len: int = 512,
                   max_new_tokens: int = 128) -> Dict[str, float]:
    """
    Evaluate a seq2seq model on CoT dataset.

    Args:
        model: HuggingFace seq2seq model
        tokenizer: Corresponding tokenizer
        eval_data: List of dicts with keys "input" and "target"
        device: Device to run on ("cuda" or "cpu")
        batch_size: Batch size for inference
        max_input_len: Max input length for tokenization
        max_new_tokens: Max tokens to generate

    Returns:
        Dict with metrics: {"accuracy": float, "num_samples": int}
    """
    model.eval()
    model.to(device)

    predictions = []
    references = []

    # Process in batches
    for i in tqdm(range(0, len(eval_data), batch_size), desc="Evaluating"):
        batch = eval_data[i:i+batch_size]

        # Extract inputs and targets
        inputs = [item["input"] for item in batch]
        targets = [item["target"] if "target" in item else item.get("output", "")
                  for item in batch]

        # Tokenize inputs
        enc = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_len
        ).to(device)

        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding for evaluation
                num_beams=1
            )

        # Decode predictions
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Extract final answers from both predictions and targets
        pred_answers = [extract_final_answer(p) for p in decoded]
        ref_answers = [extract_final_answer(t) for t in targets]

        predictions.extend(pred_answers)
        references.extend(ref_answers)

    # Compute accuracy
    accuracy = compute_exact_match_accuracy(predictions, references)

    return {
        "accuracy": accuracy,
        "exact_match": accuracy,
        "num_samples": len(eval_data),
        "predictions": predictions,  # Include for debugging
        "references": references
    }


def evaluate_by_task(model,
                     tokenizer,
                     df,
                     tasks: List[str],
                     device: str = "cuda",
                     max_samples_per_task: int = 500,
                     **eval_kwargs) -> Dict[str, Dict]:
    """
    Evaluate model on multiple tasks separately.

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        df: DataFrame with columns ["task", "source", "target"]
        tasks: List of task names to evaluate
        device: Device to run on
        max_samples_per_task: Max samples to evaluate per task
        **eval_kwargs: Additional arguments for evaluate_model()

    Returns:
        Dict mapping task names to their evaluation results
    """
    results = {}

    for task_name in tasks:
        print(f"\n=== Evaluating on task: {task_name} ===")

        # Filter data for this task
        df_task = df[df["task"] == task_name].copy()

        if len(df_task) == 0:
            print(f"No data for task {task_name}, skipping.")
            continue

        # Sample if needed
        if len(df_task) > max_samples_per_task:
            df_task = df_task.sample(max_samples_per_task, random_state=42)

        # Convert to list of dicts
        eval_data = []
        for _, row in df_task.iterrows():
            eval_data.append({
                "input": row["source"],
                "target": row["target"]
            })

        # Evaluate
        task_results = evaluate_model(
            model, tokenizer, eval_data,
            device=device, **eval_kwargs
        )

        results[task_name] = task_results

        print(f"Task: {task_name}")
        print(f"  Accuracy: {task_results['accuracy']:.4f}")
        print(f"  Samples: {task_results['num_samples']}")

    return results
