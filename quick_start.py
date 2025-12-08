"""
Quick start script to run the complete pipeline for a single k value.

This script automates the full workflow:
1. Dataset preparation
2. Committee model preparation
3. Difficulty score computation
4. Embedding extraction
5. GA experiment
6. Subset selection
7. Model training (GA + random baselines)
8. Evaluation

Usage:
    python quick_start.py --k 100
"""

import subprocess
import sys
from pathlib import Path
import argparse


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n❌ Error in: {description}")
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"✓ Completed: {description}\n")
    return result


def main():
    parser = argparse.ArgumentParser(description="Quick start: run complete pipeline for one k value")
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="Subset size k (default: 100)"
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data preparation (assumes data already exists)"
    )
    parser.add_argument(
        "--skip-committee",
        action="store_true",
        help="Skip committee model preparation"
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip committee inference (assumes difficulty scores exist)"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding extraction (assumes embeddings exist)"
    )
    parser.add_argument(
        "--skip-ga",
        action="store_true",
        help="Skip GA experiment (assumes Pareto front exists)"
    )
    parser.add_argument(
        "--skip-selection",
        action="store_true",
        help="Skip subset selection (assumes selected subset exists)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training"
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip random baseline training"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip model evaluation"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("CORESET-GA QUICK START")
    print(f"Running complete pipeline for k={args.k}")
    print("="*60)
    
    # Step 1: Dataset preparation
    if not args.skip_data:
        run_command(
            ["python", "data/prepare_mnist.py", "--seed", "2025"],
            "Dataset Preparation"
        )
    else:
        print("⏭️  Skipping dataset preparation")
    
    # Step 2: Committee model preparation
    if not args.skip_committee:
        run_command(
            ["python", "pretrained_committee_models/prepare_committee.py"],
            "Committee Model Preparation"
        )
    else:
        print("⏭️  Skipping committee model preparation")
    
    # Step 3: Committee inference
    if not args.skip_inference:
        run_command(
            ["python", "pretrained_committee_models/run_inference.py"],
            "Committee Inference and Difficulty Scores"
        )
    else:
        print("⏭️  Skipping committee inference")
    
    # Step 4: Embedding extraction
    if not args.skip_embeddings:
        run_command(
            ["python", "embeddings/extract_embeddings.py"],
            "Embedding Extraction"
        )
    else:
        print("⏭️  Skipping embedding extraction")
    
    # Step 5: GA experiment
    if not args.skip_ga:
        run_command(
            [f"python", f"experiments/run_k{args.k}.py"],
            f"GA Experiment (k={args.k})"
        )
    else:
        print(f"⏭️  Skipping GA experiment for k={args.k}")
    
    # Step 6: Subset selection
    if not args.skip_selection:
        run_command(
            ["python", "experiments/select_subset.py", str(args.k)],
            "Subset Selection"
        )
    else:
        print(f"⏭️  Skipping subset selection for k={args.k}")
    
    # Step 7: Train GA-selected model
    if not args.skip_training:
        run_command(
            ["python", "training/train_cnn.py", "ga", "--k", str(args.k)],
            "Train CNN on GA-Selected Subset"
        )
    else:
        print("⏭️  Skipping GA-selected model training")
    
    # Step 8: Train random baselines
    if not args.skip_baselines:
        run_command(
            ["python", "training/train_baselines.py", str(args.k)],
            "Train Random Baseline Models"
        )
    else:
        print("⏭️  Skipping random baseline training")
    
    # Step 9: Evaluate models
    if not args.skip_evaluation:
        run_command(
            ["python", "training/evaluate_models.py", "--k-values", str(args.k)],
            "Model Evaluation"
        )
    else:
        print("⏭️  Skipping model evaluation")
    
    print("\n" + "="*60)
    print("✓ QUICK START COMPLETE!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"  1. View results: results/evaluation.json")
    print(f"  2. Generate visualizations: jupyter notebook analysis/final_results.ipynb")
    print(f"  3. View Pareto fronts: jupyter notebook analysis/plot_pareto.ipynb")
    print("="*60)


if __name__ == "__main__":
    main()

