"""
Run NSGA-II for k=1000 subset selection.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from experiments.experiment_utils import run_ga_experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GA experiment for k=1000")
    parser.add_argument(
        "--population-size",
        type=int,
        default=None,
        help=f"Population size (default: {config.GA_POPULATION_SIZE})"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help=f"Number of generations (default: {config.GA_GENERATIONS})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Random seed (default: {config.GA_SEED})"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    run_ga_experiment(
        k=1000,
        population_size=args.population_size,
        generations=args.generations,
        seed=args.seed,
        verbose=not args.quiet
    )
