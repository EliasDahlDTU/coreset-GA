"""
Select best subset from Pareto front.

Implements subset selection strategies:
1. Weighted score (default): weighted sum of normalized objectives
2. Ideal point: closest to utopian point (max difficulty, max diversity, max balance)
3. Maximin: maximize the minimum (weighted) normalized objective (more "representative"/balanced)
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from data.load_data import load_selection_pool, get_class_distribution
from ga.evaluation import evaluate_fitness


def load_pareto_front(k: int, results_dir: Path = None) -> dict:
    """
    Load Pareto front results for a given k.
    
    Args:
        k: Subset size
        results_dir: Results directory. If None, uses config.RESULTS_DIR
        
    Returns:
        Dictionary with Pareto front data
    """
    if results_dir is None:
        results_dir = config.RESULTS_DIR
    
    pareto_path = config.get_pareto_front_path(k)
    
    if not pareto_path.exists():
        raise FileNotFoundError(
            f"Pareto front not found: {pareto_path}\n"
            f"Run 'python experiments/run_k{k}.py' first to generate Pareto front."
        )
    
    with open(pareto_path, 'rb') as f:
        result = pickle.load(f)
    
    return result


def normalize_objectives(fitnesses: list) -> np.ndarray:
    """
    Normalize objectives to [0, 1] range.
    
    Args:
        fitnesses: List of fitness tuples (difficulty, diversity, balance)
        
    Returns:
        Normalized fitness array of shape (n, 3)
    """
    if len(fitnesses) == 0:
        return np.array([])
    
    fitness_array = np.array(fitnesses)

    # Min-max normalize each objective to [0, 1] (per Pareto front).
    # This matches the docstring and makes weighted-score selection easier to reason about.
    obj_min = np.min(fitness_array, axis=0)
    obj_max = np.max(fitness_array, axis=0)
    denom = obj_max - obj_min

    # If an objective is constant across the Pareto front, give it a neutral 0.5 everywhere
    # so it neither helps nor hurts in weighted scoring.
    normalized = np.empty_like(fitness_array, dtype=np.float64)
    for j in range(fitness_array.shape[1]):
        if denom[j] == 0:
            normalized[:, j] = 0.5
        else:
            normalized[:, j] = (fitness_array[:, j] - obj_min[j]) / denom[j]

    return np.clip(normalized, 0.0, 1.0)


def select_by_weighted_score(
    chromosomes: list,
    fitnesses: list,
    weights: dict = None
) -> Tuple[np.ndarray, dict]:
    """
    Select subset using weighted score.
    
    Args:
        chromosomes: List of chromosomes (Pareto-optimal solutions)
        fitnesses: List of fitness tuples
        weights: Dictionary with weights for each objective. If None, uses config.
        
    Returns:
        Tuple of (selected_chromosome, selection_info)
    """
    if weights is None:
        weights = config.SUBSET_SELECTION_WEIGHTS
    
    if len(chromosomes) == 0:
        raise ValueError("No chromosomes provided")
    
    # Normalize objectives
    normalized_fitnesses = normalize_objectives(fitnesses)
    
    # Compute weighted scores
    scores = (
        weights['difficulty'] * normalized_fitnesses[:, 0] +
        weights['diversity'] * normalized_fitnesses[:, 1] +
        weights['balance'] * normalized_fitnesses[:, 2]
    )
    
    # Select best
    best_idx = np.argmax(scores)
    selected_chromosome = chromosomes[best_idx]
    selected_fitness = fitnesses[best_idx]
    
    selection_info = {
        'method': 'weighted_score',
        'weights': weights,
        'score': float(scores[best_idx]),
        'fitness': {
            'difficulty': float(selected_fitness[0]),
            'diversity': float(selected_fitness[1]),
            'balance': float(selected_fitness[2])
        },
        'rank': int(best_idx),
        'total_candidates': len(chromosomes)
    }
    
    return selected_chromosome, selection_info


def select_by_ideal_point(
    chromosomes: list,
    fitnesses: list
) -> Tuple[np.ndarray, dict]:
    """
    Select subset closest to ideal point (utopian point).
    
    Ideal point = (max difficulty, max diversity, max balance)
    
    Args:
        chromosomes: List of chromosomes (Pareto-optimal solutions)
        fitnesses: List of fitness tuples
        
    Returns:
        Tuple of (selected_chromosome, selection_info)
    """
    if len(chromosomes) == 0:
        raise ValueError("No chromosomes provided")
    
    fitness_array = np.array(fitnesses)
    
    # Ideal point (utopian point) = maximum of each objective
    ideal_point = np.max(fitness_array, axis=0)
    
    # Compute distance to ideal point (Euclidean distance)
    distances = np.linalg.norm(fitness_array - ideal_point, axis=1)
    
    # Select closest to ideal
    best_idx = np.argmin(distances)
    selected_chromosome = chromosomes[best_idx]
    selected_fitness = fitnesses[best_idx]
    
    selection_info = {
        'method': 'ideal_point',
        'ideal_point': {
            'difficulty': float(ideal_point[0]),
            'diversity': float(ideal_point[1]),
            'balance': float(ideal_point[2])
        },
        'distance_to_ideal': float(distances[best_idx]),
        'fitness': {
            'difficulty': float(selected_fitness[0]),
            'diversity': float(selected_fitness[1]),
            'balance': float(selected_fitness[2])
        },
        'rank': int(best_idx),
        'total_candidates': len(chromosomes)
    }
    
    return selected_chromosome, selection_info


def select_by_maximin(
    chromosomes: list,
    fitnesses: list,
    weights: dict = None
) -> Tuple[np.ndarray, dict]:
    """
    Select subset by maximizing the minimum (weighted) normalized objective.

    Intuition: weighted sum can pick a solution that is great in 1-2 objectives but
    mediocre in another. Maximin tends to pick a more "representative"/balanced
    point on the Pareto front.
    """
    if weights is None:
        weights = config.SUBSET_SELECTION_WEIGHTS

    if len(chromosomes) == 0:
        raise ValueError("No chromosomes provided")

    normalized_fitnesses = normalize_objectives(fitnesses)  # (n,3) in [0,1]
    w = np.array([weights["difficulty"], weights["diversity"], weights["balance"]], dtype=np.float64)
    w = np.where(w < 0, 0.0, w)
    if float(w.sum()) == 0.0:
        w = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    w = w / w.sum()

    # Apply weights as scaling, then maximize the minimum component
    weighted = normalized_fitnesses * w[None, :]
    scores = np.min(weighted, axis=1)

    best_idx = int(np.argmax(scores))
    selected_chromosome = chromosomes[best_idx]
    selected_fitness = fitnesses[best_idx]

    selection_info = {
        "method": "maximin",
        "weights": {"difficulty": float(w[0]), "diversity": float(w[1]), "balance": float(w[2])},
        "score": float(scores[best_idx]),
        "fitness": {
            "difficulty": float(selected_fitness[0]),
            "diversity": float(selected_fitness[1]),
            "balance": float(selected_fitness[2]),
        },
        "rank": int(best_idx),
        "total_candidates": len(chromosomes),
    }

    return selected_chromosome, selection_info


def select_subset(
    k: int,
    method: str = 'weighted_score',
    weights: dict = None,
    results_dir: Path = None,
    save: bool = True,
    recompute_fitness: bool = False
) -> Tuple[np.ndarray, dict]:
    """
    Select best subset from Pareto front for a given k.
    
    Args:
        k: Subset size
        method: Selection method ('weighted_score' or 'ideal_point')
        weights: Weights for weighted_score method. If None, uses config.
        results_dir: Results directory. If None, uses config.RESULTS_DIR
        save: Whether to save selected subset
        recompute_fitness: If True, recompute objectives for Pareto chromosomes using current config.
        
    Returns:
        Tuple of (selected_chromosome, selection_info)
    """
    # Load Pareto front
    result = load_pareto_front(k, results_dir)
    
    pareto_chromosomes = result['pareto_front']['chromosomes']
    pareto_fitnesses = result['pareto_front']['fitnesses']
    
    if len(pareto_chromosomes) == 0:
        raise ValueError(f"No Pareto-optimal solutions found for k={k}")

    if recompute_fitness:
        # Useful for sweeping objective variants (coverage agg, difficulty weighting, etc.)
        # without re-running NSGA-II. Note this no longer represents the true Pareto
        # front for the new objective, but is a fast approximation.
        pareto_fitnesses = [evaluate_fitness(chrom) for chrom in pareto_chromosomes]
    
    # Default weights: for small k, prefer balance/representativeness; for large k, downweight balance.
    if weights is None:
        small_thr = getattr(config, "SUBSET_SELECTION_SMALL_K_THRESHOLD", None)
        small_w = getattr(config, "SUBSET_SELECTION_WEIGHTS_SMALL_K", None)
        large_thr = getattr(config, "SUBSET_SELECTION_LARGE_K_THRESHOLD", None)
        large_w = getattr(config, "SUBSET_SELECTION_WEIGHTS_LARGE_K", None)

        if small_thr is not None and small_w is not None and k <= int(small_thr):
            weights = small_w
        elif large_thr is not None and large_w is not None and k >= int(large_thr):
            weights = large_w
        else:
            weights = config.SUBSET_SELECTION_WEIGHTS
    
    # Select subset
    if method == 'weighted_score':
        selected_chromosome, selection_info = select_by_weighted_score(
            pareto_chromosomes, pareto_fitnesses, weights
        )
    elif method == 'ideal_point':
        selected_chromosome, selection_info = select_by_ideal_point(
            pareto_chromosomes, pareto_fitnesses
        )
    elif method == 'maximin':
        selected_chromosome, selection_info = select_by_maximin(
            pareto_chromosomes, pareto_fitnesses, weights
        )
    else:
        raise ValueError(
            f"Unknown selection method: {method}. Use 'weighted_score', 'ideal_point', or 'maximin'"
        )
    
    # Add metadata
    selection_info['k'] = k
    selection_info['pareto_front_size'] = len(pareto_chromosomes)
    
    # Save if requested
    if save and config.SAVE_SELECTED_SUBSETS:
        subset_path = config.get_selected_subset_path(k)
        np.save(subset_path, selected_chromosome)
        selection_info['saved_to'] = str(subset_path)
    
    return selected_chromosome, selection_info


def visualize_selected_subset(
    k: int,
    selected_indices: np.ndarray = None,
    save_path: Path = None
):
    """
    Visualize the selected subset (class distribution).
    
    Args:
        k: Subset size
        selected_indices: Selected indices. If None, loads from saved file.
        save_path: Path to save visualization. If None, doesn't save.
    """
    import matplotlib.pyplot as plt
    
    # Load data
    _, labels = load_selection_pool()
    
    if selected_indices is None:
        subset_path = config.get_selected_subset_path(k)
        if not subset_path.exists():
            raise FileNotFoundError(
                f"Selected subset not found: {subset_path}\n"
                f"Run subset selection first."
            )
        selected_indices = np.load(subset_path)
    
    # Get class distribution
    subset_labels = labels[selected_indices]
    class_dist = get_class_distribution(subset_labels, num_classes=config.NUM_CLASSES)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = np.arange(config.NUM_CLASSES)
    bars = ax.bar(classes, class_dist, alpha=0.7, color='steelblue')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, class_dist)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Class Distribution of Selected Subset (k={k})', fontsize=14)
    ax.set_xticks(classes)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Select best subset from Pareto front")
    parser.add_argument(
        "k",
        type=int,
        help="Subset size k"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="weighted_score",
        choices=["weighted_score", "ideal_point", "maximin"],
        help="Selection method (default: weighted_score)"
    )
    parser.add_argument("--w-difficulty", type=float, default=None, help="Override weight for difficulty objective")
    parser.add_argument("--w-diversity", type=float, default=None, help="Override weight for diversity objective")
    parser.add_argument("--w-balance", type=float, default=None, help="Override weight for balance objective")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save selected subset"
    )
    parser.add_argument(
        "--recompute-fitness",
        action="store_true",
        help="Recompute objectives for Pareto chromosomes using current config (for fast sweeps)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualization of selected subset"
    )
    
    args = parser.parse_args()
    
    print(f"Selecting subset for k={args.k} using method '{args.method}'...")

    weights = None
    if args.w_difficulty is not None or args.w_diversity is not None or args.w_balance is not None:
        wd = float(args.w_difficulty) if args.w_difficulty is not None else config.SUBSET_SELECTION_WEIGHTS["difficulty"]
        wv = float(args.w_diversity) if args.w_diversity is not None else config.SUBSET_SELECTION_WEIGHTS["diversity"]
        wb = float(args.w_balance) if args.w_balance is not None else config.SUBSET_SELECTION_WEIGHTS["balance"]
        # Normalize to sum to 1 for easier interpretability
        s = max(1e-12, wd + wv + wb)
        weights = {"difficulty": wd / s, "diversity": wv / s, "balance": wb / s}

    selected_chromosome, selection_info = select_subset(
        k=args.k,
        method=args.method,
        weights=weights,
        save=not args.no_save,
        recompute_fitness=args.recompute_fitness,
    )
    
    print(f"\n✓ Selected subset:")
    print(f"  Method: {selection_info['method']}")
    print(f"  Fitness: difficulty={selection_info['fitness']['difficulty']:.4f}, "
          f"diversity={selection_info['fitness']['diversity']:.4f}, "
          f"balance={selection_info['fitness']['balance']:.4f}")
    print(f"  Rank: {selection_info['rank']+1}/{selection_info['total_candidates']}")
    
    if 'saved_to' in selection_info:
        print(f"  Saved to: {selection_info['saved_to']}")
    
    if args.visualize:
        print("\nCreating visualization...")
        viz_path = config.RESULTS_DIR / f"subset_visualization_k{args.k}.png"
        visualize_selected_subset(args.k, selected_chromosome, save_path=viz_path)

