"""
Run committee inference on dataset and compute difficulty scores.

Loads all committee models, runs inference on the selection pool,
saves softmax predictions for each model, and computes difficulty scores
(entropy of averaged softmax predictions).
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from data.load_data import load_selection_pool, create_dataloader


def load_committee_model(model_name, model_dir=None, device=None):
    """
    Load a committee model.
    
    Args:
        model_name: Name of the model (e.g., 'resnet18')
        model_dir: Directory containing models. If None, uses config.PRETRAINED_COMMITTEE_MODELS_DIR
        device: Device to load model on. If None, uses config.COMMITTEE_DEVICE
        
    Returns:
        Loaded model in eval mode
    """
    if model_dir is None:
        model_dir = config.PRETRAINED_COMMITTEE_MODELS_DIR
    
    if device is None:
        device = config.COMMITTEE_DEVICE
    
    model_dir = Path(model_dir)
    
    # Try loading full model first (easier)
    full_model_path = model_dir / f"{model_name}_full.pth"
    if full_model_path.exists():
        model = torch.load(full_model_path, map_location=device)
        model.eval()
        return model
    
    # Otherwise load state dict (requires model architecture)
    raise FileNotFoundError(
        f"Model {model_name} not found. Run 'python pretrained_committee_models/prepare_committee.py' first."
    )


def compute_entropy(probs):
    """
    Compute entropy of probability distribution.
    
    Args:
        probs: Probability tensor of shape (batch_size, num_classes)
        
    Returns:
        Entropy values of shape (batch_size,)
    """
    # Add small epsilon to avoid log(0)
    probs = torch.clamp(probs, min=1e-10)
    entropy = -torch.sum(probs * torch.log(probs), dim=1)
    return entropy


def run_committee_inference(
    model_names=None,
    data_dir=None,
    model_dir=None,
    device=None,
    batch_size=None,
    save_predictions=True,
    save_difficulty=True
):
    """
    Run committee inference and compute difficulty scores.
    
    Args:
        model_names: List of model names. If None, uses config.COMMITTEE_MODEL_NAMES
        data_dir: Directory containing data. If None, uses config.DATA_DIR
        model_dir: Directory containing models. If None, uses config.PRETRAINED_COMMITTEE_MODELS_DIR
        device: Device to use. If None, uses config.COMMITTEE_DEVICE
        batch_size: Batch size for inference. If None, uses config.COMMITTEE_BATCH_SIZE
        save_predictions: Whether to save individual model predictions
        save_difficulty: Whether to save difficulty scores
        
    Returns:
        Dictionary with predictions and difficulty scores
    """
    if model_names is None:
        model_names = config.COMMITTEE_MODEL_NAMES
    
    if data_dir is None:
        data_dir = config.DATA_DIR
    
    if model_dir is None:
        model_dir = config.PRETRAINED_COMMITTEE_MODELS_DIR
    
    if device is None:
        device = config.COMMITTEE_DEVICE
    
    if batch_size is None:
        batch_size = config.COMMITTEE_BATCH_SIZE
    
    device = torch.device(device)
    model_dir = Path(model_dir)
    data_dir = Path(data_dir)
    
    # Load data
    print("Loading selection pool data...")
    data, labels = load_selection_pool(data_dir=str(data_dir))
    print(f"  Loaded {len(data)} samples\n")
    
    # Create dataloader
    dataloader = create_dataloader(data, labels, batch_size=batch_size, shuffle=False)
    
    # Load committee models
    print(f"Loading {len(model_names)} committee models...")
    models = {}
    for model_name in model_names:
        print(f"  Loading {model_name}...")
        models[model_name] = load_committee_model(model_name, model_dir=model_dir, device=device)
        print(f"    ✓ Loaded\n")
    
    # Run inference
    print("Running committee inference...")
    all_predictions = {name: [] for name in model_names}
    all_labels = []
    
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(dataloader, desc="Inference"):
            batch_data = batch_data.to(device)
            all_labels.append(batch_labels.numpy())
            
            # Get predictions from each model
            for model_name, model in models.items():
                logits = model(batch_data)
                probs = F.softmax(logits, dim=1)
                all_predictions[model_name].append(probs.cpu().numpy())
    
    # Concatenate all batches
    all_labels = np.concatenate(all_labels, axis=0)
    for model_name in model_names:
        all_predictions[model_name] = np.concatenate(all_predictions[model_name], axis=0)
    
    print(f"  ✓ Completed inference on {len(all_labels)} samples\n")
    
    # Compute averaged predictions
    print("Computing averaged committee predictions...")
    averaged_probs = np.mean([all_predictions[name] for name in model_names], axis=0)
    print(f"  ✓ Averaged predictions shape: {averaged_probs.shape}\n")
    
    # Compute difficulty scores (entropy)
    print("Computing difficulty scores (entropy)...")
    difficulty_scores = compute_entropy(torch.from_numpy(averaged_probs)).numpy()
    print(f"  ✓ Difficulty scores shape: {difficulty_scores.shape}")
    print(f"  ✓ Mean difficulty: {difficulty_scores.mean():.4f}")
    print(f"  ✓ Std difficulty: {difficulty_scores.std():.4f}\n")
    
    # Save results
    embeddings_dir = Path(config.EMBEDDINGS_DIR)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    if save_predictions:
        print("Saving individual model predictions...")
        for model_name in model_names:
            pred_path = embeddings_dir / f"{model_name}_predictions.npy"
            np.save(pred_path, all_predictions[model_name])
            print(f"  ✓ Saved {model_name} predictions to {pred_path}")
        
        # Save averaged predictions
        avg_pred_path = embeddings_dir / "committee_averaged_predictions.npy"
        np.save(avg_pred_path, averaged_probs)
        print(f"  ✓ Saved averaged predictions to {avg_pred_path}\n")
    
    if save_difficulty:
        print("Saving difficulty scores...")
        difficulty_path = config.DIFFICULTY_SCORES_FILE
        np.save(difficulty_path, difficulty_scores)
        print(f"  ✓ Saved difficulty scores to {difficulty_path}\n")
    
    print("✓ Committee inference completed successfully!")
    
    return {
        "predictions": all_predictions,
        "averaged_predictions": averaged_probs,
        "difficulty_scores": difficulty_scores,
        "labels": all_labels
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run committee inference and compute difficulty scores")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to use (default: from config)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory (default: from config)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Model directory (default: from config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: from config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: from config)"
    )
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Don't save individual model predictions"
    )
    parser.add_argument(
        "--no-save-difficulty",
        action="store_true",
        help="Don't save difficulty scores"
    )
    
    args = parser.parse_args()
    
    run_committee_inference(
        model_names=args.models,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        device=args.device,
        batch_size=args.batch_size,
        save_predictions=not args.no_save_predictions,
        save_difficulty=not args.no_save_difficulty
    )

