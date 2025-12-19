"""
Population management and chromosome representation.

Chromosome: fixed-length list of k unique dataset indices.
"""

import numpy as np
from typing import List, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


_cached_labels = None


def _get_labels():
    global _cached_labels
    if _cached_labels is None:
        from data.load_data import load_selection_pool
        _, _cached_labels = load_selection_pool()
    return _cached_labels


def create_chromosome(k: int, pool_size: int, rng: np.random.Generator = None) -> np.ndarray:
    """
    Create a random chromosome (subset of k unique indices).
    
    Args:
        k: Size of the subset
        pool_size: Size of the selection pool (max valid index + 1)
        rng: Random number generator. If None, creates a new one.
        
    Returns:
        Array of k unique indices in sorted order
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if k > pool_size:
        raise ValueError(f"Cannot create chromosome of size {k} from pool of size {pool_size}")

    # Optional: class-stratified initialization (helps classification representativeness)
    if getattr(config, "GA_STRATIFIED_INIT", False):
        labels = _get_labels()
        num_classes = int(getattr(config, "NUM_CLASSES", int(labels.max()) + 1))

        # Build per-class pools once (fast enough for MNIST scale)
        per_class = [np.flatnonzero(labels == c) for c in range(num_classes)]
        # Ensure each class contributes at least 1 sample when possible
        base = max(1, k // num_classes)
        counts = np.full(num_classes, base, dtype=int)
        # Distribute remainder
        counts[: (k - counts.sum())] += 1

        # Sample per class, allowing fallback if a class has too few points
        chosen = []
        for c in range(num_classes):
            pool = per_class[c]
            take = int(min(counts[c], len(pool)))
            if take > 0:
                chosen.append(rng.choice(pool, size=take, replace=False))

        chosen = np.concatenate(chosen) if len(chosen) else np.array([], dtype=int)

        # If we didn't reach k (rare), fill remaining randomly from global pool excluding chosen
        if len(chosen) < k:
            used = set(chosen.tolist())
            remaining = k - len(chosen)
            available = np.array([i for i in range(pool_size) if i not in used], dtype=int)
            fill = rng.choice(available, size=remaining, replace=False)
            chosen = np.concatenate([chosen, fill])

        return np.sort(chosen.astype(int, copy=False))

    chromosome = rng.choice(pool_size, size=k, replace=False)
    return np.sort(chromosome)


def validate_chromosome(chromosome: np.ndarray, pool_size: int) -> bool:
    """
    Validate that a chromosome is valid.
    
    Args:
        chromosome: Array of indices
        pool_size: Size of the selection pool (max valid index + 1)
        
    Returns:
        True if valid, False otherwise
    """
    if len(chromosome) == 0:
        return False
    
    # Check all indices are unique
    if len(np.unique(chromosome)) != len(chromosome):
        return False
    
    # Check all indices are in valid range
    if np.any(chromosome < 0) or np.any(chromosome >= pool_size):
        return False
    
    return True


def enforce_uniqueness(chromosome: np.ndarray, pool_size: int, rng: np.random.Generator = None) -> np.ndarray:
    """
    Enforce uniqueness in a chromosome by replacing duplicates with random unseen indices.
    
    Args:
        chromosome: Array of indices (may contain duplicates)
        pool_size: Size of the selection pool
        rng: Random number generator. If None, creates a new one.
        
    Returns:
        Array of unique indices
    """
    if rng is None:
        rng = np.random.default_rng()
    
    unique_chromosome = np.unique(chromosome)
    
    # If we lost indices due to duplicates, fill with random unseen ones
    if len(unique_chromosome) < len(chromosome):
        k = len(chromosome)
        used_indices = set(unique_chromosome)
        available_indices = np.array([i for i in range(pool_size) if i not in used_indices])
        
        num_needed = k - len(unique_chromosome)
        if len(available_indices) < num_needed:
            raise ValueError(f"Cannot fill {num_needed} slots from {len(available_indices)} available indices")
        
        new_indices = rng.choice(available_indices, size=num_needed, replace=False)
        unique_chromosome = np.concatenate([unique_chromosome, new_indices])
    
    return np.sort(unique_chromosome)


def initialize_population(
    population_size: int,
    k: int,
    pool_size: int,
    seed: int = None
) -> List[np.ndarray]:
    """
    Initialize a population of random chromosomes.
    
    Args:
        population_size: Number of individuals in the population
        k: Size of each subset
        pool_size: Size of the selection pool
        seed: Random seed for reproducibility
        
    Returns:
        List of chromosomes (each is an array of k unique indices)
    """
    if seed is None:
        seed = config.GA_SEED
    
    rng = np.random.default_rng(seed)
    
    population = []
    for _ in range(population_size):
        chromosome = create_chromosome(k, pool_size, rng=rng)
        population.append(chromosome)
    
    return population


if __name__ == "__main__":
    # Test chromosome creation
    print("Testing chromosome creation...")
    k = 10
    pool_size = 100
    
    chromosome = create_chromosome(k, pool_size)
    print(f"Created chromosome: {chromosome}")
    print(f"Length: {len(chromosome)}, Unique: {len(np.unique(chromosome)) == len(chromosome)}")
    print(f"Valid: {validate_chromosome(chromosome, pool_size)}")
    
    # Test population initialization
    print("\nTesting population initialization...")
    population = initialize_population(5, k, pool_size, seed=42)
    print(f"Population size: {len(population)}")
    for i, chrom in enumerate(population):
        print(f"  Individual {i}: {chrom[:5]}... (length: {len(chrom)})")
    
    # Test uniqueness enforcement
    print("\nTesting uniqueness enforcement...")
    duplicate_chrom = np.array([1, 2, 3, 3, 4, 5, 5, 6, 7, 8])
    fixed = enforce_uniqueness(duplicate_chrom, pool_size, rng=np.random.default_rng(42))
    print(f"Original: {duplicate_chrom}")
    print(f"Fixed: {fixed}")
    print(f"Unique: {len(np.unique(fixed)) == len(fixed)}")
