"""
End-to-end pipeline runner for all k values on A100 20GB.

Stages (run once):
1) Data prep
2) Committee prep
3) Committee inference (difficulty scores)
4) Embedding extraction

Stages (per k):
5) GA experiment
6) Subset selection
7) Train GA-selected model
8) Train random baselines

Final:
9) Evaluation across all k

Usage (defaults target A100 20GB settings):
    python run_all.py

Customize k values and skip completed steps:
    python run_all.py --k-values 50 100 200 500 750 1000 \\
        --skip-data --skip-committee --skip-inference --skip-embeddings
"""

import argparse
import subprocess
import sys
from pathlib import Path

import config


def run_command(cmd, description):
    """Run a command; exit on failure."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n❌ Error in: {description}")
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"✓ Completed: {description}\n")


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline across all k values")
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=config.K_VALUES,
        help=f"Subset sizes to run (default: {config.K_VALUES})",
    )
    parser.add_argument("--skip-data", action="store_true", help="Skip data preparation")
    parser.add_argument("--skip-committee", action="store_true", help="Skip committee prep")
    parser.add_argument("--skip-inference", action="store_true", help="Skip committee inference")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding extraction")
    parser.add_argument("--skip-ga", action="store_true", help="Skip GA experiments")
    parser.add_argument("--skip-selection", action="store_true", help="Skip subset selection")
    parser.add_argument("--skip-training", action="store_true", help="Skip GA-model training")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip random baselines")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference/embeddings (default: cuda)",
    )
    parser.add_argument(
        "--inference-batch",
        type=int,
        default=config.COMMITTEE_BATCH_SIZE,
        help=f"Batch size for committee inference (default: {config.COMMITTEE_BATCH_SIZE})",
    )
    parser.add_argument(
        "--embedding-batch",
        type=int,
        default=config.EMBEDDING_BATCH_SIZE,
        help=f"Batch size for embedding extraction (default: {config.EMBEDDING_BATCH_SIZE})",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CORESET-GA: FULL MULTI-K RUNNER")
    print(f"K values: {args.k_values}")
    print("=" * 60)

    # One-time stages
    if not args.skip_data:
        run_command(
            ["python", "data/prepare_mnist.py", "--seed", "42"],
            "Dataset Preparation",
        )
    else:
        print("⏭️  Skipping dataset preparation")

    if not args.skip_committee:
        run_command(
            ["python", "pretrained_committee_models/prepare_committee.py"],
            "Committee Model Preparation",
        )
    else:
        print("⏭️  Skipping committee model preparation")

    if not args.skip_inference:
        run_command(
            [
                "python",
                "pretrained_committee_models/run_inference.py",
                "--device",
                args.device,
                "--batch-size",
                str(args.inference_batch),
            ],
            "Committee Inference and Difficulty Scores",
        )
    else:
        print("⏭️  Skipping committee inference")

    if not args.skip_embeddings:
        run_command(
            [
                "python",
                "embeddings/extract_embeddings.py",
                "--device",
                args.device,
                "--batch-size",
                str(args.embedding_batch),
            ],
            "Embedding Extraction",
        )
    else:
        print("⏭️  Skipping embedding extraction")

    # Per-k stages
    for k in args.k_values:
        print(f"\n{'#' * 60}")
        print(f"Processing k = {k}")
        print(f"{'#' * 60}\n")

        run_k_path = Path(f"experiments/run_k{k}.py")
        if not run_k_path.exists() and not args.skip_ga:
            print(f"❌ Missing GA script for k={k}: {run_k_path}")
            sys.exit(1)

        if not args.skip_ga:
            run_command(
                ["python", str(run_k_path)],
                f"GA Experiment (k={k})",
            )
        else:
            print(f"⏭️  Skipping GA experiment for k={k}")

        if not args.skip_selection:
            run_command(
                ["python", "experiments/select_subset.py", str(k)],
                f"Subset Selection (k={k})",
            )
        else:
            print(f"⏭️  Skipping subset selection for k={k}")

        if not args.skip_training:
            run_command(
                ["python", "training/train_cnn.py", "ga", "--k", str(k)],
                f"Train CNN on GA Subset (k={k})",
            )
        else:
            print(f"⏭️  Skipping GA-selected model training for k={k}")

        if not args.skip_baselines:
            run_command(
                ["python", "training/train_baselines.py", str(k)],
                f"Train Random Baselines (k={k})",
            )
        else:
            print(f"⏭️  Skipping random baselines for k={k}")

    # Final evaluation
    if not args.skip_evaluation:
        eval_k_values = [str(k) for k in args.k_values]
        run_command(
            ["python", "training/evaluate_models.py", "--k-values", *eval_k_values],
            "Evaluate All Models",
        )
    else:
        print("⏭️  Skipping evaluation")

    print("\n" + "=" * 60)
    print("✓ FULL MULTI-K PIPELINE COMPLETE")
    print("=" * 60)
    print("Outputs:")
    print("  - Embeddings & difficulty: embeddings/")
    print("  - GA fronts & selections: results/")
    print("  - Models: final_models/")
    print("  - Eval summary: results/evaluation.json")
    print("=" * 60)


if __name__ == "__main__":
    main()

