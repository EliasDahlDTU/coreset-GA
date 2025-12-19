"""
Crossover operators for genetic algorithm.

Implements set-aware uniform crossover that respects the fixed subset size constraint.
"""

import numpy as np
from typing import Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from ga.population import enforce_uniqueness


def crossover_uniform(
    parent1: np.ndarray,
    parent2: np.ndarray,
    pool_size: int,
    rng: np.random.Generator = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Set-aware uniform crossover.
    
    For each position, randomly choose from either parent.
    If duplicates arise, fill remaining slots with random unseen indices.
    
    Args:
        parent1: First parent chromosome
        parent2: Second parent chromosome
        pool_size: Size of the selection pool
        rng: Random number generator. If None, creates a new one.
        
    Returns:
        Tuple of (child1, child2)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    k = len(parent1)
    if len(parent2) != k:
        raise ValueError(f"Parents must have same length: {len(parent1)} != {len(parent2)}")
    
    # Create children by randomly selecting from parents
    child1 = np.zeros(k, dtype=parent1.dtype)
    child2 = np.zeros(k, dtype=parent2.dtype)
    
    # For each position, randomly choose from either parent
    for i in range(k):
        if rng.random() < 0.5:
            child1[i] = parent1[i]
            child2[i] = parent2[i]
        else:
            child1[i] = parent2[i]
            child2[i] = parent1[i]
    
    # Enforce uniqueness and fill missing slots
    child1 = _fix_child(child1, parent1, parent2, pool_size, rng)
    child2 = _fix_child(child2, parent1, parent2, pool_size, rng)
    
    return child1, child2


def crossover_set_union(
    parent1: np.ndarray,
    parent2: np.ndarray,
    pool_size: int,
    rng: np.random.Generator = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Set-union crossover (recommended for set-valued chromosomes).

    Our chromosome represents a *set* of k unique indices (order doesn't matter).
    The older `crossover_uniform` operates position-wise on sorted arrays, which
    introduces artifacts (early positions are always "small indices", etc.).

    This crossover:
    - Builds the union of parent indices
    - Samples k unique indices from that union when possible
    - If the union is smaller than k (high parent overlap), it fills the remainder
      with random unseen indices from the global pool
    - Produces two different children by re-sampling
    """
    if rng is None:
        rng = np.random.default_rng()

    k = len(parent1)
    if len(parent2) != k:
        raise ValueError(f"Parents must have same length: {len(parent1)} != {len(parent2)}")

    union = np.union1d(parent1, parent2)

    def _make_child() -> np.ndarray:
        if len(union) >= k:
            child = rng.choice(union, size=k, replace=False)
            return np.sort(child)

        # Need to fill: take all union, then fill remaining from pool (excluding union)
        used = set(union.tolist())
        remaining = k - len(union)
        available = np.array([i for i in range(pool_size) if i not in used], dtype=parent1.dtype)
        fill = rng.choice(available, size=remaining, replace=False)
        child = np.concatenate([union.astype(parent1.dtype, copy=False), fill])
        return np.sort(child)

    return _make_child(), _make_child()


def _fix_child(
    child: np.ndarray,
    parent1: np.ndarray,
    parent2: np.ndarray,
    pool_size: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Fix a child chromosome by removing duplicates and filling missing slots.
    
    Args:
        child: Child chromosome (may have duplicates or zeros)
        parent1: First parent (for reference)
        parent2: Second parent (for reference)
        pool_size: Size of the selection pool
        rng: Random number generator
        
    Returns:
        Fixed child chromosome with k unique indices
    """
    k = len(child)

    # NOTE:
    # Dataset indices are in [0, pool_size). Index 0 is a valid sample id.
    # Earlier versions treated 0 as an "uninitialized" sentinel (because children
    # were created with np.zeros), which incorrectly discarded index 0 whenever it
    # appeared in a child. Children are fully populated before we reach this
    # function, so we should never treat any value as a sentinel here.
    unique_values = np.unique(child)
    
    # If we have duplicates or missing values, fix them
    if len(unique_values) < k:
        # Get all indices from both parents as candidates
        all_parent_indices = np.unique(np.concatenate([parent1, parent2]))
        
        # Get used indices
        used_indices = set(unique_values)
        
        # Find available indices (not in child, but could be from parents)
        available_from_parents = [i for i in all_parent_indices if i not in used_indices]
        available_from_pool = [i for i in range(pool_size) if i not in used_indices]
        
        # Prefer indices from parents, then from pool
        num_needed = k - len(unique_values)
        
        if len(available_from_parents) >= num_needed:
            # Use parent indices
            new_indices = rng.choice(available_from_parents, size=num_needed, replace=False)
        else:
            # Use parent indices + pool indices
            new_indices = list(available_from_parents)
            remaining = num_needed - len(new_indices)
            additional = rng.choice(available_from_pool, size=remaining, replace=False)
            new_indices = np.concatenate([new_indices, additional])
        
        # Combine unique values with new indices
        fixed_child = np.concatenate([unique_values, new_indices])
    else:
        fixed_child = unique_values
    
    # Ensure we have exactly k indices
    if len(fixed_child) > k:
        # Randomly select k
        fixed_child = rng.choice(fixed_child, size=k, replace=False)
    elif len(fixed_child) < k:
        # Fill remaining slots
        used_indices = set(fixed_child)
        available_indices = np.array([i for i in range(pool_size) if i not in used_indices])
        num_needed = k - len(fixed_child)
        new_indices = rng.choice(available_indices, size=num_needed, replace=False)
        fixed_child = np.concatenate([fixed_child, new_indices])
    
    return np.sort(fixed_child)


if __name__ == "__main__":
    # Test crossover
    print("Testing crossover operator...")
    
    k = 10
    pool_size = 100
    rng = np.random.default_rng(42)
    
    # Create test parents
    parent1 = np.sort(rng.choice(pool_size, size=k, replace=False))
    parent2 = np.sort(rng.choice(pool_size, size=k, replace=False))
    
    print(f"Parent 1: {parent1}")
    print(f"Parent 2: {parent2}")
    print(f"Overlap: {len(np.intersect1d(parent1, parent2))} indices")
    
    # Test crossover
    child1, child2 = crossover_uniform(parent1, parent2, pool_size, rng=rng)
    
    print(f"\nChild 1: {child1}")
    print(f"  Length: {len(child1)}, Unique: {len(np.unique(child1)) == len(child1)}")
    print(f"  From parent1: {len(np.intersect1d(child1, parent1))} indices")
    print(f"  From parent2: {len(np.intersect1d(child1, parent2))} indices")
    
    print(f"\nChild 2: {child2}")
    print(f"  Length: {len(child2)}, Unique: {len(np.unique(child2)) == len(child2)}")
    print(f"  From parent1: {len(np.intersect1d(child2, parent1))} indices")
    print(f"  From parent2: {len(np.intersect1d(child2, parent2))} indices")
    
    # Test multiple crossovers
    print("\nTesting 10 crossovers:")
    for i in range(10):
        c1, c2 = crossover_uniform(parent1, parent2, pool_size, rng=rng)
        print(f"  Crossover {i+1}: child1 has {len(np.unique(c1))} unique indices")
