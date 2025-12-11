"""
Fitness evaluation functions for GA objectives.

Implements three objectives:
1. Difficulty: mean entropy of committee predictions
2. Diversity: cosine distance in embedding space
3. Balance: class distribution deviation from uniform
"""

import numpy as np
from typing import Tuple, Dict
import sys
from pathlib import Path
import json
import time

try:
    import torch
except Exception:
    torch = None

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from data.load_data import load_selection_pool
from embeddings.extract_embeddings import load_embeddings

# region agent log
BASE_DIR = Path(__file__).resolve().parents[1]
_AGENT_LOG_PATH = BASE_DIR / "logs" / "debug.log"
_AGENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

def _agent_log(hypothesis_id: str, location: str, message: str, data: Dict) -> None:
    try:
        payload = {
            "sessionId": "debug-session",
            "runId": "baseline",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(_AGENT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
# endregion


# Cache for loaded data
_difficulty_scores = None
_embeddings = None
_normalized_embeddings = None
_torch_normalized_embeddings = None
_labels = None
_pool_size = None


def _load_cached_data():
    """Load and cache difficulty scores, embeddings, and labels."""
    global _difficulty_scores, _embeddings, _labels, _pool_size
    global _normalized_embeddings, _torch_normalized_embeddings
    
    if _difficulty_scores is None:
        # Load difficulty scores
        difficulty_path = config.DIFFICULTY_SCORES_FILE
        if not difficulty_path.exists():
            raise FileNotFoundError(
                f"Difficulty scores not found: {difficulty_path}\n"
                f"Run 'python pretrained_committee_models/run_inference.py' first."
            )
        _difficulty_scores = np.load(difficulty_path, mmap_mode="r")
    
    if _embeddings is None:
        # Load embeddings
        _embeddings = load_embeddings()
        # Precompute normalized embeddings for cosine similarity reuse
        norms = np.linalg.norm(_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        _normalized_embeddings = _embeddings / norms
        # Optionally move normalized embeddings to torch for GPU diversity calc
        if config.DIVERSITY_USE_GPU and torch is not None:
            device = config.DIVERSITY_DEVICE
            dtype = getattr(torch, config.DIVERSITY_TORCH_DTYPE, torch.float32)
            try:
                _torch_normalized_embeddings = torch.as_tensor(_normalized_embeddings, device=device, dtype=dtype)
            except Exception:
                _torch_normalized_embeddings = None
    
    if _labels is None:
        # Load labels
        _, _labels = load_selection_pool()
    
    if _pool_size is None:
        _pool_size = len(_difficulty_scores)
    
    # Validate consistency
    assert len(_difficulty_scores) == len(_embeddings) == len(_labels), \
        "Difficulty scores, embeddings, and labels must have the same length"


def difficulty_objective(indices: np.ndarray) -> float:
    """
    Compute difficulty objective (mean entropy of committee predictions).
    
    Higher difficulty = more informative samples (committee is uncertain).
    
    Args:
        indices: Array of subset indices
        
    Returns:
        Mean difficulty score (to maximize)
    """
    _load_cached_data()
    
    if len(indices) == 0:
        return 0.0
    
    # Get difficulty scores for the subset
    subset_difficulties = _difficulty_scores[indices]
    
    # Return mean difficulty
    return float(np.mean(subset_difficulties))


def diversity_objective(indices: np.ndarray) -> float:
    """
    Compute diversity objective (1 - mean cosine similarity).
    
    Higher diversity = samples are more spread out in embedding space.
    
    Args:
        indices: Array of subset indices
        
    Returns:
        Diversity score (to maximize, range [0, 1])
    """
    _load_cached_data()
    
    if len(indices) < 2:
        return 0.0
    
    start = time.time()
    use_torch = _torch_normalized_embeddings is not None
    if use_torch:
        subset_embeddings = _torch_normalized_embeddings[indices]
        cosine_sim_matrix = subset_embeddings @ subset_embeddings.T
        n = len(indices)
        tri = torch.triu_indices(n, n, offset=1, device=cosine_sim_matrix.device)
        if n > 2048:
            # Batch upper-tri extraction to reduce temporary memory (exact)
            batch = 262144  # ~256k elements per chunk
            total = tri.shape[1]
            sum_vals = 0.0
            count = 0
            for start_idx in range(0, total, batch):
                end_idx = min(start_idx + batch, total)
                vals = cosine_sim_matrix[tri[0, start_idx:end_idx], tri[1, start_idx:end_idx]]
                sum_vals += vals.sum().item()
                count += vals.numel()
            mean_cosine_sim = sum_vals / count
        else:
            upper_triangle = cosine_sim_matrix[tri[0], tri[1]]
            mean_cosine_sim = float(upper_triangle.mean().item())
    else:
        subset_embeddings = _normalized_embeddings[indices]
        cosine_sim_matrix = np.dot(subset_embeddings, subset_embeddings.T)
        n = len(indices)
        upper_triangle = cosine_sim_matrix[np.triu_indices(n, k=1)]
        mean_cosine_sim = np.mean(upper_triangle)
    
    # Diversity = 1 - mean cosine similarity
    diversity = 1.0 - mean_cosine_sim
    duration_ms = (time.time() - start) * 1000
    _agent_log(
        hypothesis_id="H1",
        location="ga/evaluation.py:diversity_objective",
        message="diversity timing",
        data={"k": int(len(indices)), "duration_ms": duration_ms, "torch": use_torch},
    )
    
    return float(np.clip(diversity, 0.0, 1.0))


def balance_objective(indices: np.ndarray) -> float:
    """
    Compute balance objective (1 - deviation from uniform class distribution).
    
    Higher balance = more proportional representation of all classes.
    
    Args:
        indices: Array of subset indices
        
    Returns:
        Balance score (to maximize, range [0, 1])
    """
    _load_cached_data()
    
    if len(indices) == 0:
        return 0.0
    
    # Get labels for the subset
    subset_labels = _labels[indices]
    
    # Compute class distribution
    num_classes = config.NUM_CLASSES
    class_counts = np.bincount(subset_labels, minlength=num_classes)
    class_proportions = class_counts / len(indices)
    
    # Ideal uniform distribution
    ideal_proportion = 1.0 / num_classes
    
    # Compute deviation from uniform (L1 distance)
    deviation = np.sum(np.abs(class_proportions - ideal_proportion))
    
    # Balance = 1 - normalized deviation
    # Maximum deviation is 2 * (1 - 1/num_classes) when all samples are in one class
    max_deviation = 2.0 * (1.0 - ideal_proportion)
    normalized_deviation = deviation / max_deviation if max_deviation > 0 else 0.0
    
    balance = 1.0 - normalized_deviation
    
    return float(np.clip(balance, 0.0, 1.0))


def evaluate_fitness(indices: np.ndarray) -> Tuple[float, float, float]:
    """
    Evaluate fitness of a chromosome (subset).
    
    Returns all three objectives as a tuple.
    
    Args:
        indices: Array of subset indices
        
    Returns:
        Tuple of (difficulty, diversity, balance) - all to maximize
    """
    difficulty = difficulty_objective(indices)
    diversity = diversity_objective(indices)
    balance = balance_objective(indices)
    
    return (difficulty, diversity, balance)


def evaluate_population(population: list) -> list:
    """
    Evaluate fitness for an entire population.
    
    Args:
        population: List of chromosomes (each is an array of indices)
        
    Returns:
        List of fitness tuples, one per individual
    """
    return [evaluate_fitness(chromosome) for chromosome in population]


if __name__ == "__main__":
    # Test evaluation functions
    print("Testing evaluation functions...")
    print("(Note: Requires difficulty scores and embeddings to be computed first)")
    
    try:
        _load_cached_data()
        print(f"Loaded data: {_pool_size} samples")
        
        # Create a test subset
        test_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        difficulty = difficulty_objective(test_indices)
        diversity = diversity_objective(test_indices)
        balance = balance_objective(test_indices)
        
        print("\nTest subset (first 10 indices):")
        print(f"  Difficulty: {difficulty:.4f}")
        print(f"  Diversity: {diversity:.4f}")
        print(f"  Balance: {balance:.4f}")
        
        fitness = evaluate_fitness(test_indices)
        print(f"\nFitness tuple: {fitness}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run the following commands first:")
        print("  1. python pretrained_committee_models/run_inference.py")
        print("  2. python embeddings/extract_embeddings.py")
