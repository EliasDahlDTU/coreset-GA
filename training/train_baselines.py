"""
Train baseline models for comparison.

Implements:
1. Random baseline: 5 random subsets per k
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from data.load_data import load_selection_pool
from training.train_cnn import train_cnn


def train_random_baselines(
    k: int,
    num_runs: int = None,
    seed: int = None,
    verbose: bool = True
):
    """
    Train random baseline models (multiple runs per k).
    
    Args:
        k: Subset size
        num_runs: Number of random runs. If None, uses config.NUM_RANDOM_BASELINES
        seed: Base random seed. If None, uses config.BASELINE_SEED
        verbose: Whether to print progress
        
    Returns:
        List of training results (one per run)
    """
    if num_runs is None:
        num_runs = config.NUM_RANDOM_BASELINES
    
    if seed is None:
        seed = config.BASELINE_SEED
    
    # Load selection pool
    _, labels = load_selection_pool()
    pool_size = len(labels)
    
    if verbose:
        print("=" * 60)
        print(f"Training Random Baselines: k={k}, num_runs={num_runs}")
        print("=" * 60)
    
    results = []
    
    for run_num in range(1, num_runs + 1):
        if verbose:
            print(f"\n--- Random Baseline Run {run_num}/{num_runs} ---")
        
        # Generate random subset with unique seed per run
        run_seed = seed + run_num
        np.random.seed(run_seed)
        random_indices = np.random.choice(pool_size, size=k, replace=False)
        
        # Train model
        result = train_cnn(
            train_indices=random_indices,
            k=k,
            subset_type='random',
            run_number=run_num,
            seed=run_seed,
            verbose=verbose
        )
        
        results.append(result)
        
        if verbose:
            print(f"  Test accuracy: {result['test_acc']:.2f}%")
    
    if verbose:
        # Compute statistics
        test_accs = [r['test_acc'] for r in results]
        mean_acc = np.mean(test_accs)
        std_acc = np.std(test_accs)
        
        print(f"\n{'=' * 60}")
        print(f"Random Baseline Summary (k={k}):")
        print(f"  Mean test accuracy: {mean_acc:.2f}%")
        print(f"  Std test accuracy: {std_acc:.2f}%")
        print(f"  Min: {min(test_accs):.2f}%, Max: {max(test_accs):.2f}%")
        print(f"{'=' * 60}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train random baseline models")
    parser.add_argument(
        "k",
        type=int,
        help="Subset size k"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=None,
        help=f"Number of random runs (default: {config.NUM_RANDOM_BASELINES})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Random seed (default: {config.BASELINE_SEED})"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    train_random_baselines(
        k=args.k,
        num_runs=args.num_runs,
        seed=args.seed,
        verbose=not args.quiet
    )

