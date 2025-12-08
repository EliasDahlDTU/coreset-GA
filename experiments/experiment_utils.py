"""
Utility functions for GA experiments.
"""

import pickle
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from data.load_data import load_selection_pool
from ga.nsga2 import run_nsga2


def run_ga_experiment(
    k: int,
    population_size: int = None,
    generations: int = None,
    seed: int = None,
    checkpoint_interval: int = None,
    output_dir: Path = None,
    verbose: bool = True
):
    """
    Run a GA experiment for a given k value.
    
    Args:
        k: Subset size
        population_size: Population size. If None, uses config
        generations: Number of generations. If None, uses config
        seed: Random seed. If None, uses config
        checkpoint_interval: Save checkpoint every N generations. If None, no checkpointing
        output_dir: Output directory. If None, uses config.RESULTS_DIR
        verbose: Whether to print progress
        
    Returns:
        Dictionary with experiment results
    """
    if population_size is None:
        population_size = config.GA_POPULATION_SIZE
    
    if generations is None:
        generations = config.GA_GENERATIONS
    
    if seed is None:
        seed = config.GA_SEED
    
    if output_dir is None:
        output_dir = config.RESULTS_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get pool size
    _, labels = load_selection_pool()
    pool_size = len(labels)
    
    if verbose:
        print("=" * 60)
        print(f"GA Experiment: k={k}, population={population_size}, generations={generations}")
        print(f"Pool size: {pool_size}")
        print(f"Seed: {seed}")
        print("=" * 60)
    
    # Track timing
    start_time = time.time()
    
    # Run NSGA-II
    result = run_nsga2(
        k=k,
        pool_size=pool_size,
        population_size=population_size,
        generations=generations,
        seed=seed,
        verbose=verbose
    )
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Add metadata
    result['metadata'] = {
        'k': k,
        'population_size': population_size,
        'generations': generations,
        'seed': seed,
        'pool_size': pool_size,
        'total_time_seconds': total_time,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    if config.SAVE_PARETO_FRONTS:
        pareto_path = config.get_pareto_front_path(k)
        
        # Save as pickle (preserves numpy arrays)
        with open(pareto_path, 'wb') as f:
            pickle.dump(result, f)
        
        if verbose:
            print(f"\n✓ Saved Pareto front to {pareto_path}")
        
        # Also save a JSON version (for easy inspection, but without numpy arrays)
        json_path = pareto_path.with_suffix('.json')
        json_result = {
            'metadata': result['metadata'],
            'pareto_front_size': len(result['pareto_front']['chromosomes']),
            'history': {
                'generation': result['history']['generation'],
                'pareto_front_size': result['history']['pareto_front_size'],
                'best_difficulty': result['history']['best_difficulty'],
                'best_diversity': result['history']['best_diversity'],
                'best_balance': result['history']['best_balance']
            },
            'pareto_fitnesses': [
                {
                    'difficulty': float(f[0]),
                    'diversity': float(f[1]),
                    'balance': float(f[2])
                }
                for f in result['pareto_front']['fitnesses']
            ]
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_result, f, indent=2)
        
        if verbose:
            print(f"✓ Saved JSON summary to {json_path}")
    
    if verbose:
        print(f"\n✓ Experiment completed in {total_time:.2f} seconds")
        print(f"  Final Pareto front size: {len(result['pareto_front']['chromosomes'])}")
    
    return result

