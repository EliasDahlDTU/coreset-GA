"""
Mutation operators for genetic algorithm.

Implements three mutation strategies:
1. Index replacement (70%): replace a random number of indices (scaled with k)
2. Block replacement (20%): replace a random 10-20% block with fresh indices
3. Small replacement (10%): replace 2 indices with fresh ones
"""

import numpy as np
import sys
from pathlib import Path
import json
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from ga.population import enforce_uniqueness

# region agent log
BASE_DIR = Path(__file__).resolve().parents[1]
_AGENT_LOG_PATH = BASE_DIR / "logs" / "debug.log"
_AGENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

def _agent_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
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


def mutate_index_replacement(
    chromosome: np.ndarray,
    pool_size: int,
    num_replacements: int = None,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Mutation: replace random indices with new ones.
    
    Args:
        chromosome: Original chromosome
        pool_size: Size of the selection pool
        num_replacements: Number of indices to replace. If None, random between min and max.
        rng: Random number generator. If None, creates a new one.
        
    Returns:
        Mutated chromosome
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if num_replacements is None:
        # Scale mutation strength with subset size.
        # For large k (e.g. 1000), replacing only 1-8 points barely explores the space.
        k = len(chromosome)
        min_rep = max(config.MUTATION_MIN_REPLACEMENTS, int(round(0.005 * k)))  # 0.5% of k
        max_rep = max(config.MUTATION_MAX_REPLACEMENTS, int(round(0.02 * k)))   # 2% of k
        min_rep = min(min_rep, k)
        max_rep = min(max_rep, k)
        if max_rep < min_rep:
            max_rep = min_rep
        num_replacements = int(rng.integers(min_rep, max_rep + 1))
    
    num_replacements = min(num_replacements, len(chromosome))
    
    # Create a copy
    mutated = chromosome.copy()
    
    # Select indices to replace
    indices_to_replace = rng.choice(len(mutated), size=num_replacements, replace=False)
    
    # Build available indices mask (vectorized)
    start = time.time()
    available_mask = np.ones(pool_size, dtype=bool)
    available_mask[mutated] = False
    available_indices = np.flatnonzero(available_mask)
    
    if len(available_indices) < num_replacements:
        # Not enough available indices, just replace what we can
        num_replacements = len(available_indices)
        indices_to_replace = indices_to_replace[:num_replacements]
    
    # Replace with random new indices
    new_indices = rng.choice(available_indices, size=num_replacements, replace=False)
    mutated[indices_to_replace] = new_indices
    
    # Enforce uniqueness and sort
    mutated = enforce_uniqueness(mutated, pool_size, rng=rng)
    
    duration_ms = (time.time() - start) * 1000
    _agent_log(
        hypothesis_id="H3",
        location="ga/mutation.py:mutate_index_replacement",
        message="available_indices timing",
        data={
            "pool_size": int(pool_size),
            "k": int(len(chromosome)),
            "num_replacements": int(num_replacements),
            "available": int(len(available_indices)),
            "duration_ms": duration_ms,
        },
    )
    
    return mutated


def mutate_segment_shuffle(
    chromosome: np.ndarray,
    pool_size: int,
    segment_size: int = None,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Mutation: replace a random contiguous block with fresh indices.

    NOTE: Chromosomes represent *sets* of indices. Shuffling or swapping does not
    change the set, so those operators are no-ops. This operator is intentionally
    a "block replacement" to make it meaningful for set-valued chromosomes.
    
    Args:
        chromosome: Original chromosome
        pool_size: Size of the selection pool
        segment_size: Size of segment to shuffle. If None, random between min and max %.
        rng: Random number generator. If None, creates a new one.
        
    Returns:
        Mutated chromosome
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if len(chromosome) < 2:
        return chromosome.copy()
    
    if segment_size is None:
        # Random segment size between min and max percentage
        min_size = max(1, int(len(chromosome) * config.MUTATION_SEGMENT_MIN_PCT))
        max_size = max(1, int(len(chromosome) * config.MUTATION_SEGMENT_MAX_PCT))
        segment_size = rng.integers(min_size, max_size + 1)
    
    segment_size = min(segment_size, len(chromosome))
    
    mutated = chromosome.copy()

    # Select random block (positions in the chromosome, not dataset index range)
    start_idx = int(rng.integers(0, len(mutated) - segment_size + 1))
    end_idx = start_idx + int(segment_size)

    # Build available indices (exclude current set, but allow reusing the ones we're removing)
    # This increases acceptance vs forbidding all current indices.
    removed = set(mutated[start_idx:end_idx].tolist())
    used = set(mutated.tolist())
    used_minus_removed = used - removed

    available_mask = np.ones(pool_size, dtype=bool)
    if len(used_minus_removed) > 0:
        available_mask[np.fromiter(used_minus_removed, dtype=np.int64)] = False
    available_indices = np.flatnonzero(available_mask)

    fill = rng.choice(available_indices, size=(end_idx - start_idx), replace=False)
    mutated[start_idx:end_idx] = fill

    mutated = enforce_uniqueness(mutated, pool_size, rng=rng)
    return mutated


def mutate_swap(
    chromosome: np.ndarray,
    pool_size: int,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Mutation: replace two random indices with fresh ones (small replacement).

    NOTE: A literal swap is a no-op for set-valued chromosomes after sorting.
    
    Args:
        chromosome: Original chromosome
        pool_size: Size of the selection pool
        rng: Random number generator. If None, creates a new one.
        
    Returns:
        Mutated chromosome
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if len(chromosome) < 2:
        return chromosome.copy()
    
    mutated = chromosome.copy()

    # Choose two positions to replace
    idxs = rng.choice(len(mutated), size=2, replace=False)
    removed = set(mutated[idxs].tolist())
    used = set(mutated.tolist())
    used_minus_removed = used - removed

    available_mask = np.ones(pool_size, dtype=bool)
    if len(used_minus_removed) > 0:
        available_mask[np.fromiter(used_minus_removed, dtype=np.int64)] = False
    available_indices = np.flatnonzero(available_mask)

    new_vals = rng.choice(available_indices, size=2, replace=False)
    mutated[idxs] = new_vals
    mutated = enforce_uniqueness(mutated, pool_size, rng=rng)
    return mutated


def mutate(
    chromosome: np.ndarray,
    pool_size: int,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Apply mutation to a chromosome using probabilistic selection.
    
    Selects mutation operator based on configured probabilities:
    - Index replacement: 70%
    - Segment shuffle: 20%
    - Swap: 10%
    
    Args:
        chromosome: Original chromosome
        pool_size: Size of the selection pool
        rng: Random number generator. If None, creates a new one.
        
    Returns:
        Mutated chromosome
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Select mutation operator based on probabilities
    rand = rng.random()
    
    if rand < config.MUTATION_INDEX_REPLACEMENT_PROB:
        return mutate_index_replacement(chromosome, pool_size, rng=rng)
    elif rand < config.MUTATION_INDEX_REPLACEMENT_PROB + config.MUTATION_SEGMENT_SHUFFLE_PROB:
        return mutate_segment_shuffle(chromosome, pool_size, rng=rng)
    else:
        return mutate_swap(chromosome, pool_size, rng=rng)


if __name__ == "__main__":
    # Test mutation operators
    print("Testing mutation operators...")
    
    k = 10
    pool_size = 100
    rng = np.random.default_rng(42)
    
    # Create test chromosome
    chromosome = np.sort(rng.choice(pool_size, size=k, replace=False))
    print(f"Original chromosome: {chromosome}")
    
    # Test index replacement
    print("\n1. Index replacement mutation:")
    mutated = mutate_index_replacement(chromosome, pool_size, rng=rng)
    print(f"   Mutated: {mutated}")
    print(f"   Changed: {np.sum(chromosome != mutated)} indices")
    
    # Test segment shuffle
    print("\n2. Segment shuffle mutation:")
    mutated = mutate_segment_shuffle(chromosome, pool_size, rng=rng)
    print(f"   Mutated: {mutated}")
    print(f"   Changed: {np.sum(chromosome != mutated)} indices")
    
    # Test swap
    print("\n3. Swap mutation:")
    mutated = mutate_swap(chromosome, pool_size, rng=rng)
    print(f"   Mutated: {mutated}")
    print(f"   Changed: {np.sum(chromosome != mutated)} indices")
    
    # Test probabilistic mutation
    print("\n4. Probabilistic mutation (10 runs):")
    for i in range(10):
        mutated = mutate(chromosome, pool_size, rng=rng)
        mutation_type = (
            "replacement"
            if np.sum(chromosome != mutated) > 2
            else "shuffle"
            if not np.any(np.sort(chromosome) != np.sort(mutated))
            else "swap"
        )
        print(f"   Run {i+1}: {mutation_type} - {mutated[:5]}...")
