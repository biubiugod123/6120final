import re
import numpy as np
from typing import List, Dict, Tuple
from cot_framework.evaluate import normalize_text, extract_final_answer


def compute_correctness_reward(prediction: str,
                               reference: str,
                               correct_reward: float = 1.0,
                               incorrect_penalty: float = -1.0) -> float:
    """
    Compute reward based on answer correctness.

    Args:
        prediction: Model's generated answer
        reference: Ground truth answer
        correct_reward: Reward for correct answer
        incorrect_penalty: Penalty for incorrect answer

    Returns:
        Reward value
    """
    # Extract final answers
    pred_answer = extract_final_answer(prediction)
    ref_answer = extract_final_answer(reference)

    # Compare normalized answers
    if normalize_text(pred_answer) == normalize_text(ref_answer):
        return correct_reward
    else:
        return incorrect_penalty


def compute_format_reward(prediction: str,
                         has_final_answer_bonus: float = 0.1,
                         has_reasoning_bonus: float = 0.1) -> float:
    """
    Compute reward based on output format quality.

    Args:
        prediction: Model's generated text
        has_final_answer_bonus: Bonus if "Final Answer:" is present
        has_reasoning_bonus: Bonus if there's reasoning before final answer

    Returns:
        Format reward (0 to max bonus)
    """
    reward = 0.0

    # Check if "Final Answer:" pattern is present
    if re.search(r'Final Answer:', prediction, re.IGNORECASE):
        reward += has_final_answer_bonus

        # Check if there's reasoning before the final answer
        parts = re.split(r'Final Answer:', prediction, flags=re.IGNORECASE)
        if len(parts) > 1 and len(parts[0].strip()) > 10:  # At least 10 chars of reasoning
            reward += has_reasoning_bonus

    return reward


def compute_reasoning_quality_reward(prediction: str,
                                     reference: str = None,
                                     min_length: int = 20,
                                     max_length: int = 500,
                                     length_bonus: float = 0.2) -> float:
    """
    Compute reward based on reasoning quality (heuristic-based).

    Args:
        prediction: Model's generated text
        reference: Optional reference rationale (not used currently)
        min_length: Minimum reasonable length for rationale
        max_length: Maximum reasonable length (penalize if too long)
        length_bonus: Bonus for appropriate length

    Returns:
        Quality reward
    """
    # Extract reasoning part (everything before "Final Answer:")
    parts = re.split(r'Final Answer:', prediction, flags=re.IGNORECASE)
    reasoning = parts[0].strip() if len(parts) > 1 else prediction.strip()

    reasoning_len = len(reasoning)

    # Reward for appropriate length
    if min_length <= reasoning_len <= max_length:
        return length_bonus
    elif reasoning_len < min_length:
        # Penalize too short reasoning
        return -0.1
    else:
        # Penalize excessively long reasoning
        return -0.05

    return 0.0


def compute_reward(prediction: str,
                  reference: str,
                  weights: Dict[str, float] = None) -> Tuple[float, Dict[str, float]]:
    """
    Compute comprehensive reward for a single prediction.

    Args:
        prediction: Model's generated text
        reference: Ground truth (may include rationale + answer)
        weights: Dict specifying weights for different reward components
                 Default: {"correctness": 1.0, "format": 0.2, "quality": 0.1}

    Returns:
        Tuple of (total_reward, reward_breakdown)
    """
    if weights is None:
        weights = {
            "correctness": 1.0,   # Most important
            "format": 0.2,        # Format compliance
            "quality": 0.1        # Reasoning quality
        }

    # Compute individual reward components
    correctness = compute_correctness_reward(prediction, reference)
    format_reward = compute_format_reward(prediction)
    quality_reward = compute_reasoning_quality_reward(prediction, reference)

    # Weighted sum
    total_reward = (
        weights.get("correctness", 1.0) * correctness +
        weights.get("format", 0.2) * format_reward +
        weights.get("quality", 0.1) * quality_reward
    )

    breakdown = {
        "total": total_reward,
        "correctness": correctness,
        "format": format_reward,
        "quality": quality_reward
    }

    return total_reward, breakdown


def batch_compute_rewards(predictions: List[str],
                         references: List[str],
                         weights: Dict[str, float] = None) -> Tuple[np.ndarray, List[Dict]]:
    """
    Compute rewards for a batch of predictions.

    Args:
        predictions: List of model predictions
        references: List of ground truth answers
        weights: Reward component weights

    Returns:
        Tuple of (rewards_array, list_of_breakdowns)
    """
    assert len(predictions) == len(references), \
        f"Prediction count ({len(predictions)}) != reference count ({len(references)})"

    rewards = []
    breakdowns = []

    for pred, ref in zip(predictions, references):
        reward, breakdown = compute_reward(pred, ref, weights)
        rewards.append(reward)
        breakdowns.append(breakdown)

    return np.array(rewards), breakdowns


def compute_reward_statistics(rewards: np.ndarray,
                              breakdowns: List[Dict] = None) -> Dict:
    """
    Compute statistics over a batch of rewards.

    Args:
        rewards: Array of reward values
        breakdowns: Optional list of reward breakdowns

    Returns:
        Dict with reward statistics
    """
    stats = {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
        "median": float(np.median(rewards))
    }

    # If breakdowns are provided, compute component-wise statistics
    if breakdowns:
        components = ["correctness", "format", "quality"]
        for comp in components:
            comp_values = [b[comp] for b in breakdowns]
            stats[f"{comp}_mean"] = float(np.mean(comp_values))

    return stats


# Example usage for RL training
class RewardFunction:
    """
    Wrapper class for reward computation in RL training.
    """
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize reward function.

        Args:
            weights: Component weights for reward computation
        """
        self.weights = weights or {
            "correctness": 1.0,
            "format": 0.2,
            "quality": 0.1
        }

    def __call__(self, predictions: List[str], references: List[str]) -> np.ndarray:
        """
        Compute rewards for predictions.

        Args:
            predictions: Model outputs
            references: Ground truth

        Returns:
            Array of reward values
        """
        rewards, _ = batch_compute_rewards(predictions, references, self.weights)
        return rewards

    def compute_with_breakdown(self,
                               predictions: List[str],
                               references: List[str]) -> Tuple[np.ndarray, List[Dict]]:
        """
        Compute rewards with detailed breakdown.

        Returns:
            Tuple of (rewards, breakdowns)
        """
        return batch_compute_rewards(predictions, references, self.weights)
