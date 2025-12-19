"""
Plot evaluation results from evaluation.json

Usage:
    python analysis/plot_evaluation.py
    python analysis/plot_evaluation.py --output-dir plots/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 11


def load_evaluation_data(json_path):
    """Load evaluation JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def extract_ga_data(data):
    """Extract GA results for all k values."""
    k_values = []
    accuracies = []
    f1_macro = []
    efficiencies = []
    
    for k in data["k_values"]:
        key = f"ga_k{k}"
        if key in data["results"]:
            result = data["results"][key]
            k_values.append(k)
            accuracies.append(result["test_accuracy"])
            f1_macro.append(result["f1_scores"]["macro"] * 100)  # Convert to percentage
            efficiencies.append(result["training_efficiency"])
    
    return {
        "k_values": k_values,
        "accuracies": accuracies,
        "f1_macro": f1_macro,
        "efficiencies": efficiencies,
    }


def extract_random_data(data):
    """Extract random baseline results for all k values."""
    k_values = []
    mean_accuracies = []
    std_accuracies = []
    mean_f1_macro = []
    std_f1_macro = []
    mean_efficiencies = []
    
    for k in data["k_values"]:
        key = f"random_k{k}"
        if key in data["results"]:
            result = data["results"][key]
            k_values.append(k)
            mean_accuracies.append(result["mean_accuracy"])
            std_accuracies.append(result["std_accuracy"])
            
            # Calculate mean/std F1 from individual runs
            f1_scores = [run["f1_scores"]["macro"] * 100 for run in result["individual_runs"]]
            mean_f1_macro.append(np.mean(f1_scores))
            std_f1_macro.append(np.std(f1_scores))
            
            mean_efficiencies.append(result["mean_efficiency"])
    
    return {
        "k_values": k_values,
        "mean_accuracies": mean_accuracies,
        "std_accuracies": std_accuracies,
        "mean_f1_macro": mean_f1_macro,
        "std_f1_macro": std_f1_macro,
        "mean_efficiencies": mean_efficiencies,
    }


def extract_full_dataset_data(data):
    """Extract full dataset model results."""
    if "full_dataset" in data["results"]:
        result = data["results"]["full_dataset"]
        return {
            "k": result["k"],
            "accuracy": result["test_accuracy"],
            "f1_macro": result["f1_scores"]["macro"] * 100,
            "efficiency": result["training_efficiency"],
        }
    return None


def plot_accuracy_comparison(ga_data, random_data, full_data, output_dir):
    """Plot test accuracy vs k for GA vs Random baselines."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot random baselines with error bars
    ax.errorbar(
        random_data["k_values"],
        random_data["mean_accuracies"],
        yerr=random_data["std_accuracies"],
        fmt="o-",
        label="Random Baseline (mean ± std)",
        color="orange",
        capsize=5,
        capthick=2,
        linewidth=2,
        markersize=8,
    )
    
    # Plot GA results
    ax.plot(
        ga_data["k_values"],
        ga_data["accuracies"],
        "s-",
        label="GA-Selected",
        color="blue",
        linewidth=2,
        markersize=8,
    )
    
    # Plot full dataset if available
    if full_data:
        ax.axhline(
            y=full_data["accuracy"],
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Full Dataset (k={full_data['k']})",
        )
    
    ax.set_xlabel("Subset Size (k)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Test Accuracy: GA-Selected vs Random Baselines", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    
    plt.tight_layout()
    output_path = output_dir / "accuracy_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_f1_comparison(ga_data, random_data, full_data, output_dir):
    """Plot macro F1 score vs k for GA vs Random baselines."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot random baselines with error bars
    ax.errorbar(
        random_data["k_values"],
        random_data["mean_f1_macro"],
        yerr=random_data["std_f1_macro"],
        fmt="o-",
        label="Random Baseline (mean ± std)",
        color="orange",
        capsize=5,
        capthick=2,
        linewidth=2,
        markersize=8,
    )
    
    # Plot GA results
    ax.plot(
        ga_data["k_values"],
        ga_data["f1_macro"],
        "s-",
        label="GA-Selected",
        color="blue",
        linewidth=2,
        markersize=8,
    )
    
    # Plot full dataset if available
    if full_data:
        ax.axhline(
            y=full_data["f1_macro"],
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Full Dataset (k={full_data['k']})",
        )
    
    ax.set_xlabel("Subset Size (k)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Macro F1 Score (%)", fontsize=12, fontweight="bold")
    ax.set_title("Macro F1 Score: GA-Selected vs Random Baselines", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    
    plt.tight_layout()
    output_path = output_dir / "f1_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_training_efficiency(ga_data, random_data, full_data, output_dir):
    """Plot training efficiency vs k."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot random baselines
    ax.plot(
        random_data["k_values"],
        random_data["mean_efficiencies"],
        "o-",
        label="Random Baseline (mean)",
        color="orange",
        linewidth=2,
        markersize=8,
    )
    
    # Plot GA results
    ax.plot(
        ga_data["k_values"],
        ga_data["efficiencies"],
        "s-",
        label="GA-Selected",
        color="blue",
        linewidth=2,
        markersize=8,
    )
    
    # Plot full dataset if available
    if full_data:
        ax.axhline(
            y=full_data["efficiency"],
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Full Dataset (k={full_data['k']})",
        )
    
    ax.set_xlabel("Subset Size (k)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Training Efficiency (loss/time)", fontsize=12, fontweight="bold")
    ax.set_title("Training Efficiency: GA-Selected vs Random Baselines", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    plt.tight_layout()
    output_path = output_dir / "training_efficiency.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_per_class_f1_heatmap(data, output_dir):
    """Plot per-class F1 scores as a heatmap."""
    # Extract per-class F1 scores for GA models
    k_values = []
    class_f1_scores = []
    
    for k in data["k_values"]:
        key = f"ga_k{k}"
        if key in data["results"]:
            result = data["results"][key]
            k_values.append(k)
            per_class = result["f1_scores"]["per_class"]
            # Extract F1 scores in order (class_0 to class_9)
            f1_scores = [per_class[f"class_{i}"] * 100 for i in range(10)]
            class_f1_scores.append(f1_scores)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Transpose so k values are on x-axis, classes on y-axis
    heatmap_data = np.array(class_f1_scores).T
    
    im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    
    # Set ticks
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([f"k={k}" for k in k_values])
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"Class {i}" for i in range(10)])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("F1 Score (%)", fontsize=11, fontweight="bold")
    
    # Add text annotations
    for i in range(10):
        for j in range(len(k_values)):
            text = ax.text(
                j, i, f"{heatmap_data[i, j]:.1f}",
                ha="center", va="center",
                color="black" if heatmap_data[i, j] > 50 else "white",
                fontsize=9,
            )
    
    ax.set_xlabel("Subset Size (k)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Class", fontsize=12, fontweight="bold")
    ax.set_title("Per-Class F1 Scores for GA-Selected Models", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    output_path = output_dir / "per_class_f1_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_improvement_over_random(ga_data, random_data, output_dir):
    """Plot improvement of GA over random baseline."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate improvement (GA - Random mean)
    improvements = [
        ga - rand
        for ga, rand in zip(ga_data["accuracies"], random_data["mean_accuracies"])
    ]
    
    # Calculate error bars (using random std)
    errors = random_data["std_accuracies"]
    
    ax.bar(
        range(len(ga_data["k_values"])),
        improvements,
        yerr=errors,
        capsize=5,
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )
    
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax.set_xticks(range(len(ga_data["k_values"])))
    ax.set_xticklabels([f"k={k}" for k in ga_data["k_values"]])
    ax.set_xlabel("Subset Size (k)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy Improvement (%)", fontsize=12, fontweight="bold")
    ax.set_title("GA-Selected Improvement Over Random Baseline", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add value labels on bars
    for i, (improvement, error) in enumerate(zip(improvements, errors)):
        ax.text(
            i, improvement + error + 0.5,
            f"+{improvement:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    
    plt.tight_layout()
    output_path = output_dir / "improvement_over_random.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results")
    parser.add_argument(
        "--json-path",
        type=str,
        default="results/evaluation.json",
        help="Path to evaluation.json file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/plots",
        help="Directory to save plots",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading evaluation data from {args.json_path}...")
    data = load_evaluation_data(args.json_path)
    
    # Extract data
    print("Extracting data...")
    ga_data = extract_ga_data(data)
    random_data = extract_random_data(data)
    full_data = extract_full_dataset_data(data)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_accuracy_comparison(ga_data, random_data, full_data, output_dir)
    plot_f1_comparison(ga_data, random_data, full_data, output_dir)
    plot_training_efficiency(ga_data, random_data, full_data, output_dir)
    plot_per_class_f1_heatmap(data, output_dir)
    plot_improvement_over_random(ga_data, random_data, output_dir)
    
    print(f"\n✓ All plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
