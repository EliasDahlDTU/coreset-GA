"""
Create results table from evaluation.json

Generates a table showing:
- k values
- Mean random baseline accuracy
- Std random baseline accuracy  
- GA-selected accuracy
"""

import json
from pathlib import Path


def create_results_table(json_path="results/evaluation.json"):
    """Create results table from evaluation JSON."""
    
    # Load data
    with open(json_path, "r") as f:
        data = json.load(f)
    
    k_values = data["k_values"]
    results = data["results"]
    
    # Extract data for each k
    table_data = []
    for k in k_values:
        # GA result
        ga_key = f"ga_k{k}"
        ga_acc = results[ga_key]["test_accuracy"]
        
        # Random baseline result
        random_key = f"random_k{k}"
        random_mean = results[random_key]["mean_accuracy"]
        random_std = results[random_key]["std_accuracy"]
        
        table_data.append({
            "k": k,
            "ga_acc": ga_acc,
            "random_mean": random_mean,
            "random_std": random_std
        })
    
    # Create markdown table
    print("=" * 80)
    print("RESULTS TABLE (Markdown Format)")
    print("=" * 80)
    print()
    print("| k | Random Baseline (Mean ± Std) | GA-Selected |")
    print("|---|-------------------------------|-------------|")
    for row in table_data:
        print(f"| {row['k']} | {row['random_mean']:.2f} ± {row['random_std']:.2f} | {row['ga_acc']:.2f} |")
    
    print()
    print("=" * 80)
    print("RESULTS TABLE (LaTeX Format)")
    print("=" * 80)
    print()
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{c|c|c}")
    print("\\hline")
    print("\\textbf{k} & \\textbf{Random Baseline} & \\textbf{GA-Selected} \\\\")
    print("& \\textbf{(Mean ± Std)} & \\textbf{Accuracy} \\\\")
    print("\\hline")
    for row in table_data:
        print(f"{row['k']} & ${row['random_mean']:.2f} \\pm {row['random_std']:.2f}$ & ${row['ga_acc']:.2f}$ \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Test accuracy comparison: GA-selected subsets vs random baselines}")
    print("\\label{tab:results}")
    print("\\end{table}")
    
    print()
    print("=" * 80)
    print("RESULTS TABLE (Plain Text)")
    print("=" * 80)
    print()
    print(f"{'k':<6} {'Random Mean':<15} {'Random Std':<15} {'GA-Selected':<15}")
    print("-" * 60)
    for row in table_data:
        print(f"{row['k']:<6} {row['random_mean']:<15.2f} {row['random_std']:<15.2f} {row['ga_acc']:<15.2f}")
    
    # Also save to markdown file
    output_path = Path("analysis/results_table.md")
    with open(output_path, "w") as f:
        f.write("# Results Table\n\n")
        f.write("| k | Random Baseline (Mean ± Std) | GA-Selected |\n")
        f.write("|---|-------------------------------|-------------|\n")
        for row in table_data:
            f.write(f"| {row['k']} | {row['random_mean']:.2f} ± {row['random_std']:.2f} | {row['ga_acc']:.2f} |\n")
    
    print(f"\n✓ Table saved to {output_path}")


if __name__ == "__main__":
    import sys
    
    json_path = sys.argv[1] if len(sys.argv) > 1 else "results/evaluation.json"
    create_results_table(json_path)
