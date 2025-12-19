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

try:
    import torch
except Exception:
    torch = None

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from data.load_data import load_selection_pool
from embeddings.extract_embeddings import load_embeddings


# Cache for loaded data
_difficulty_scores = None
_embeddings = None
_normalized_embeddings = None
_torch_normalized_embeddings = None
_labels = None
_pool_size = None
_coverage_sample_indices = None  # list[np.ndarray] when using multi-subsample coverage
_coverage_sample_indices_by_class = None  # list[list[np.ndarray]] indexed [class][seed_idx]
_difficulty_target_value = None
_difficulty_target_sigma = None


def _load_cached_data():
    """Load and cache difficulty scores, embeddings, and labels."""
    global _difficulty_scores, _embeddings, _labels, _pool_size
    global _normalized_embeddings, _torch_normalized_embeddings
    global _coverage_sample_indices, _coverage_sample_indices_by_class
    global _difficulty_target_value, _difficulty_target_sigma
    
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

    if _difficulty_target_value is None:
        q = float(getattr(config, "DIFFICULTY_TARGET_QUANTILE", 0.60))
        q = float(np.clip(q, 0.0, 1.0))
        _difficulty_target_value = float(np.quantile(np.asarray(_difficulty_scores), q))
    if _difficulty_target_sigma is None:
        sigma = getattr(config, "DIFFICULTY_TARGET_SIGMA", None)
        if sigma is None:
            sigma = float(np.std(np.asarray(_difficulty_scores)))
            sigma = max(1e-6, 0.25 * sigma)
        _difficulty_target_sigma = float(max(1e-6, sigma))

    if _coverage_sample_indices is None:
        # Fixed subsamples of the pool for stable/cheap coverage estimation.
        m = min(int(getattr(config, "COVERAGE_SAMPLE_SIZE", 4096)), len(_labels))
        seeds = getattr(config, "COVERAGE_SAMPLE_SEEDS", None)
        if seeds is None:
            # Backward compatibility: single seed
            seed = int(getattr(config, "COVERAGE_SAMPLE_SEED", 42))
            seeds = [seed]
        _coverage_sample_indices = []
        for s in seeds:
            rng = np.random.default_rng(int(s))
            _coverage_sample_indices.append(rng.choice(len(_labels), size=m, replace=False))

    if _coverage_sample_indices_by_class is None and getattr(config, "COVERAGE_CLASS_CONDITIONAL", False):
        num_classes = int(getattr(config, "NUM_CLASSES", int(np.max(_labels)) + 1))
        per_class_m = getattr(config, "COVERAGE_PER_CLASS_SAMPLE_SIZE", None)
        if per_class_m is None:
            per_class_m = max(1, int(m // num_classes))
        per_class_m = int(min(per_class_m, len(_labels)))

        # Precompute indices for each class
        class_pools = [np.flatnonzero(_labels == c) for c in range(num_classes)]
        _coverage_sample_indices_by_class = [[None for _ in range(len(_coverage_sample_indices))] for _ in range(num_classes)]

        # For each seed's RNG, sample per class from that class's pool
        for seed_i, s in enumerate(seeds):
            rng = np.random.default_rng(int(s))
            for c in range(num_classes):
                pool = class_pools[c]
                if len(pool) == 0:
                    _coverage_sample_indices_by_class[c][seed_i] = np.array([], dtype=int)
                    continue
                take = min(per_class_m, len(pool))
                _coverage_sample_indices_by_class[c][seed_i] = rng.choice(pool, size=take, replace=False)


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
    
    mean_diff = float(np.mean(subset_difficulties))

    # For small k, prefer "moderate" difficulty rather than maximizing the hardest samples.
    mode = getattr(config, "DIFFICULTY_MODE", "high")
    small_thr = getattr(config, "DIFFICULTY_SMALL_K_THRESHOLD", None)
    small_mode = getattr(config, "DIFFICULTY_MODE_SMALL_K", None)
    if small_thr is not None and small_mode is not None and len(indices) <= int(small_thr):
        mode = small_mode

    if mode == "target_mean":
        # Gaussian bump around a target value (in [0,1] after exp), maximize.
        mu = float(_difficulty_target_value)
        sigma = float(_difficulty_target_sigma)
        z = (mean_diff - mu) / sigma
        return float(np.exp(-0.5 * z * z))

    # Default: maximize mean difficulty
    return mean_diff


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

    # Option: representativeness / coverage objective (often more aligned with training)
    if getattr(config, "DIVERSITY_MODE", "intra") == "coverage":
        _load_cached_data()
        use_torch = _torch_normalized_embeddings is not None
        agg = getattr(config, "COVERAGE_AGGREGATION", "max")
        topk = int(getattr(config, "COVERAGE_TOPK", 5))
        tau = float(getattr(config, "COVERAGE_SOFTMAX_TAU", 0.07))
        weight_by_diff = bool(getattr(config, "COVERAGE_WEIGHT_BY_DIFFICULTY", False))

        def _aggregate_sims_torch(sims: "torch.Tensor") -> "torch.Tensor":
            # sims: (m, k)
            if agg == "max":
                return sims.max(dim=1).values
            if agg == "topk":
                k_eff = min(topk, sims.shape[1])
                return sims.topk(k_eff, dim=1).values.mean(dim=1)
            if agg == "softmax":
                t = max(1e-6, tau)
                # smooth max: t * logsumexp(sims / t)
                return t * torch.logsumexp(sims / t, dim=1)
            # fallback
            return sims.max(dim=1).values

        def _aggregate_sims_np(sims: np.ndarray) -> np.ndarray:
            # sims: (m, k)
            if agg == "max":
                return np.max(sims, axis=1)
            if agg == "topk":
                k_eff = min(topk, sims.shape[1])
                # partial sort for topk
                part = np.partition(sims, -k_eff, axis=1)[:, -k_eff:]
                return np.mean(part, axis=1)
            if agg == "softmax":
                t = max(1e-6, tau)
                x = sims / t
                m = np.max(x, axis=1, keepdims=True)
                lse = np.log(np.sum(np.exp(x - m), axis=1) + 1e-12) + m.squeeze(1)
                return t * lse
            return np.max(sims, axis=1)

        # Class-conditional coverage: compute per class, then average.
        if getattr(config, "COVERAGE_CLASS_CONDITIONAL", False):
            num_classes = int(getattr(config, "NUM_CLASSES", 10))
            subset_labels = _labels[indices]
            per_class_scores = []

            if use_torch:
                # Pre-slice once
                for c in range(num_classes):
                    class_mask = (subset_labels == c)
                    if not np.any(class_mask):
                        per_class_scores.append(0.0)
                        continue

                    class_indices = indices[class_mask]
                    subset_emb_c = _torch_normalized_embeddings[class_indices]

                    seed_means = []
                    for sample_idx in _coverage_sample_indices_by_class[c]:
                        if sample_idx is None or len(sample_idx) == 0:
                            continue
                        sample_emb = _torch_normalized_embeddings[sample_idx]
                        sims = sample_emb @ subset_emb_c.T
                        agg_vals = _aggregate_sims_torch(sims)
                        if weight_by_diff:
                            w = torch.as_tensor(_difficulty_scores[sample_idx], device=agg_vals.device, dtype=agg_vals.dtype)
                            w = (w - w.min()) / (w.max() - w.min() + 1e-12)
                            w = w + 1e-3
                            seed_means.append((agg_vals * w).sum() / w.sum())
                        else:
                            seed_means.append(agg_vals.mean())

                    if len(seed_means) == 0:
                        per_class_scores.append(0.0)
                    else:
                        mean_max_sim_c = float(torch.stack(seed_means).mean().item())
                        per_class_scores.append(float(np.clip((mean_max_sim_c + 1.0) / 2.0, 0.0, 1.0)))
            else:
                for c in range(num_classes):
                    class_mask = (subset_labels == c)
                    if not np.any(class_mask):
                        per_class_scores.append(0.0)
                        continue

                    class_indices = indices[class_mask]
                    subset_emb_c = _normalized_embeddings[class_indices]

                    seed_means = []
                    for sample_idx in _coverage_sample_indices_by_class[c]:
                        if sample_idx is None or len(sample_idx) == 0:
                            continue
                        sample_emb = _normalized_embeddings[sample_idx]
                        sims = np.dot(sample_emb, subset_emb_c.T)
                        agg_vals = _aggregate_sims_np(sims)
                        if weight_by_diff:
                            w = _difficulty_scores[sample_idx].astype(np.float64)
                            w = (w - w.min()) / (w.max() - w.min() + 1e-12)
                            w = w + 1e-3
                            seed_means.append(float(np.sum(agg_vals * w) / np.sum(w)))
                        else:
                            seed_means.append(float(np.mean(agg_vals)))

                    if len(seed_means) == 0:
                        per_class_scores.append(0.0)
                    else:
                        mean_max_sim_c = float(np.mean(seed_means))
                        per_class_scores.append(float(np.clip((mean_max_sim_c + 1.0) / 2.0, 0.0, 1.0)))

            return float(np.mean(per_class_scores)) if len(per_class_scores) else 0.0

        # Global coverage: average across multiple subsamples
        sample_sets = _coverage_sample_indices

        if use_torch:
            subset_embeddings = _torch_normalized_embeddings[indices]
            mean_max_sims = []
            for sample_idx in sample_sets:
                sample_embeddings = _torch_normalized_embeddings[sample_idx]
                # (m,d) @ (d,k) -> (m,k), then take max over subset for each sample point
                sims = sample_embeddings @ subset_embeddings.T
                agg_vals = _aggregate_sims_torch(sims)
                if weight_by_diff:
                    w = torch.as_tensor(_difficulty_scores[sample_idx], device=agg_vals.device, dtype=agg_vals.dtype)
                    w = (w - w.min()) / (w.max() - w.min() + 1e-12)
                    w = w + 1e-3
                    mean_max_sims.append((agg_vals * w).sum() / w.sum())
                else:
                    mean_max_sims.append(agg_vals.mean())
            mean_max_sim = float(torch.stack(mean_max_sims).mean().item())
        else:
            subset_embeddings = _normalized_embeddings[indices]
            mean_max_sims = []
            for sample_idx in sample_sets:
                sample_embeddings = _normalized_embeddings[sample_idx]
                sims = np.dot(sample_embeddings, subset_embeddings.T)
                agg_vals = _aggregate_sims_np(sims)
                if weight_by_diff:
                    w = _difficulty_scores[sample_idx].astype(np.float64)
                    w = (w - w.min()) / (w.max() - w.min() + 1e-12)
                    w = w + 1e-3
                    mean_max_sims.append(float(np.sum(agg_vals * w) / np.sum(w)))
                else:
                    mean_max_sims.append(float(np.mean(agg_vals)))
            mean_max_sim = float(np.mean(mean_max_sims))

        # Cosine similarity is in [-1, 1]. Map to [0, 1] for nicer scaling.
        return float(np.clip((mean_max_sim + 1.0) / 2.0, 0.0, 1.0))
    
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
    
    # Diversity = 1 - mean cosine similarity, reshape with sqrt to emphasize spread
    diversity = 1.0 - mean_cosine_sim
    diversity = float(np.clip(diversity, 0.0, 1.0)) ** 0.5
    return diversity


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
    
    # Hard constraint: must cover every class
    if np.any(class_counts == 0):
        return 0.0
    
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
    
    if balance == 0.0:
        # Infeasible solution (missing class coverage)
        return -1e9, -1e9, -1e9
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
