"""
Evaluate trained models and compute metrics.

Loads all trained models (GA-selected + baselines) and computes:
- Test accuracy
- Per-class F1 score
- Training efficiency (accuracy / k)
- Convergence speed (epochs to 90% of final accuracy)
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import sys
from typing import Dict, List
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from training.cnn_model import create_cnn
from data.load_data import load_test_set, load_selection_pool, create_dataloader


def load_model(model_path: Path, device: str = None):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on. If None, uses config.TRAIN_DEVICE
        
    Returns:
        Loaded model and checkpoint info
    """
    if device is None:
        device = config.TRAIN_DEVICE
    
    device = torch.device(device)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model
    model = create_cnn()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def compute_accuracy(model, dataloader, device):
    """Compute classification accuracy."""
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100.0 * correct / total
    return accuracy, np.array(all_preds), np.array(all_labels)


def compute_f1_score(y_true, y_pred, num_classes=10):
    """
    Compute per-class F1 scores and macro F1.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
        
    Returns:
        Dictionary with per-class F1 and macro F1
    """
    from sklearn.metrics import f1_score
    
    # Per-class F1
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=np.arange(num_classes), zero_division=0)
    
    # Macro F1 (average of per-class F1)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        'per_class': {f'class_{i}': float(f1) for i, f1 in enumerate(f1_per_class)},
        'macro': float(macro_f1)
    }


def compute_convergence_speed(history: dict, target_percent: float = 0.90):
    """
    Compute epochs to reach target_percent of final accuracy.
    
    Args:
        history: Training history with 'val_acc' and 'epoch' keys
        target_percent: Target percentage of final accuracy (default: 0.90)
        
    Returns:
        Number of epochs to reach target, or None if never reached
    """
    if 'val_acc' not in history or len(history['val_acc']) == 0:
        return None
    
    val_accs = history['val_acc']
    final_acc = val_accs[-1]
    target_acc = final_acc * target_percent
    
    # Find first epoch where val_acc >= target_acc
    for i, acc in enumerate(val_accs):
        if acc >= target_acc:
            return i + 1  # Epochs are 1-indexed
    
    return None  # Never reached target


def evaluate_model(
    model_path: Path,
    k: int = None,
    device: str = None
) -> Dict:
    """
    Evaluate a single model.
    
    Args:
        model_path: Path to model checkpoint
        k: Subset size (for efficiency calculation). If None, tries to infer from path.
        device: Device to use. If None, uses config.TRAIN_DEVICE
        
    Returns:
        Dictionary with evaluation results
    """
    if device is None:
        device = config.TRAIN_DEVICE
    
    device = torch.device(device)
    
    # Load model
    model, checkpoint = load_model(model_path, device=device)
    
    # Get k from checkpoint or path if not provided
    if k is None:
        # Try to extract from path (e.g., cnn_ga_k100.pth -> k=100)
        import re
        match = re.search(r'_k(\d+)', model_path.stem)
        if match:
            k = int(match.group(1))
        else:
            # Try full dataset
            if 'full' in model_path.stem:
                _, labels = load_selection_pool()
                k = len(labels)
            else:
                k = None
    
    # Load test set
    test_data, test_labels = load_test_set()
    test_loader = create_dataloader(test_data, test_labels, batch_size=config.get_batch_size(len(test_data)), shuffle=False)
    
    # Evaluate
    test_acc, predictions, true_labels = compute_accuracy(model, test_loader, device)
    
    # Compute F1 scores
    f1_scores = compute_f1_score(true_labels, predictions, num_classes=config.NUM_CLASSES)
    
    # Training efficiency (accuracy / k)
    efficiency = test_acc / k if k is not None else None
    
    # Convergence speed
    history = checkpoint.get('history', {})
    convergence_epochs = compute_convergence_speed(history)
    
    results = {
        'model_path': str(model_path),
        'k': k,
        'test_accuracy': float(test_acc),
        'f1_scores': f1_scores,
        'training_efficiency': float(efficiency) if efficiency is not None else None,
        'convergence_epochs_90pct': convergence_epochs,
        'best_epoch': checkpoint.get('epoch', None),
        'best_val_acc': checkpoint.get('val_acc', None)
    }
    
    return results


def evaluate_all_models(
    k_values: List[int] = None,
    results_dir: Path = None,
    output_file: Path = None,
    device: str = None
) -> Dict:
    """
    Evaluate all trained models for given k values.
    
    Args:
        k_values: List of k values to evaluate. If None, uses config.K_VALUES
        results_dir: Results directory. If None, uses config.RESULTS_DIR
        output_file: Path to save results. If None, uses results/evaluation.json
        device: Device to use. If None, uses config.TRAIN_DEVICE
        
    Returns:
        Dictionary with all evaluation results
    """
    if k_values is None:
        k_values = config.K_VALUES
    
    if results_dir is None:
        results_dir = config.RESULTS_DIR
    
    if output_file is None:
        output_file = results_dir / "evaluation.json"
    
    results_dir = Path(results_dir)
    output_file = Path(output_file)
    final_models_dir = config.FINAL_MODELS_DIR
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'k_values': k_values,
        'results': {}
    }
    
    print("=" * 60)
    print("Evaluating All Models")
    print("=" * 60)
    
    # Import get_model_path from train_cnn
    from training.train_cnn import get_model_path
    
    # Evaluate GA-selected models
    print("\n--- GA-Selected Models ---")
    for k in k_values:
        model_path = get_model_path(k, 'ga', None)
        if model_path.exists():
            print(f"Evaluating GA model for k={k}...")
            result = evaluate_model(model_path, k=k, device=device)
            all_results['results'][f'ga_k{k}'] = result
            print(f"  Test accuracy: {result['test_accuracy']:.2f}%")
        else:
            print(f"  GA model for k={k} not found, skipping...")
    
    # Evaluate random baseline models
    print("\n--- Random Baseline Models ---")
    for k in k_values:
        random_results = []
        for run_num in range(1, config.NUM_RANDOM_BASELINES + 1):
            model_path = get_model_path(k, 'random', run_num)
            if model_path.exists():
                print(f"Evaluating random model for k={k}, run={run_num}...")
                result = evaluate_model(model_path, k=k, device=device)
                random_results.append(result)
                print(f"  Test accuracy: {result['test_accuracy']:.2f}%")
        
        if random_results:
            # Compute statistics
            accs = [r['test_accuracy'] for r in random_results]
            all_results['results'][f'random_k{k}'] = {
                'individual_runs': random_results,
                'mean_accuracy': float(np.mean(accs)),
                'std_accuracy': float(np.std(accs)),
                'min_accuracy': float(np.min(accs)),
                'max_accuracy': float(np.max(accs)),
                'mean_efficiency': float(np.mean([r['training_efficiency'] for r in random_results if r['training_efficiency'] is not None])),
                'num_runs': len(random_results)
            }
    
    # Evaluate full dataset model
    print("\n--- Full Dataset Model ---")
    full_model_path = get_model_path(None, 'full', None)
    if full_model_path.exists():
        print("Evaluating full dataset model...")
        result = evaluate_model(full_model_path, k=None, device=device)
        all_results['results']['full_dataset'] = result
        print(f"  Test accuracy: {result['test_accuracy']:.2f}%")
    else:
        print("  Full dataset model not found, skipping...")
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Evaluation completed! Results saved to {output_file}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=None,
        help="K values to evaluate (default: all from config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: results/evaluation.json)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: from config)"
    )
    
    args = parser.parse_args()
    
    evaluate_all_models(
        k_values=args.k_values,
        output_file=Path(args.output) if args.output else None,
        device=args.device
    )

