"""
Data loading utilities for MNIST dataset.

Provides functions to load the prepared data splits (selection pool, validation, test)
and convert them to PyTorch tensors or numpy arrays as needed.
"""

import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from typing import Tuple, Optional, Callable


class MNISTSubsetDataset(Dataset):
    """PyTorch Dataset for a subset of MNIST indices."""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, indices: Optional[np.ndarray] = None):
        """
        Args:
            data: Full dataset array (N, C, H, W)
            labels: Full labels array (N,)
            indices: Optional subset indices. If None, uses all data.
        """
        # Ensure writable tensors to avoid non-writable NumPy warnings
        self.data = torch.as_tensor(np.array(data, copy=True), dtype=torch.float32)
        self.labels = torch.as_tensor(np.array(labels, copy=True), dtype=torch.long)
        
        if indices is not None:
            self.data = self.data[indices]
            self.labels = self.labels[indices]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_dataset_metadata(data_dir: str = "data") -> dict:
    """Load dataset metadata JSON file."""
    metadata_path = Path(data_dir) / "dataset_metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            f"Run 'python data/prepare_mnist.py' first to prepare the dataset."
        )
    
    with open(metadata_path, "r") as f:
        return json.load(f)


def load_selection_pool(data_dir: str = "data", mmap: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the selection pool data and labels.
    
    Args:
        data_dir: Directory containing the data files
        
    Returns:
        Tuple of (data, labels) as numpy arrays
    """
    data_path = Path(data_dir)
    
    mmap_mode = "r" if mmap else None
    data = np.load(data_path / "selection_pool_data.npy", mmap_mode=mmap_mode)
    labels = np.load(data_path / "selection_pool_labels.npy", mmap_mode=mmap_mode)
    
    return data, labels


def load_validation_set(data_dir: str = "data", mmap: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the validation set data and labels.
    
    Args:
        data_dir: Directory containing the data files
        
    Returns:
        Tuple of (data, labels) as numpy arrays
    """
    data_path = Path(data_dir)
    
    mmap_mode = "r" if mmap else None
    data = np.load(data_path / "validation_data.npy", mmap_mode=mmap_mode)
    labels = np.load(data_path / "validation_labels.npy", mmap_mode=mmap_mode)
    
    return data, labels


def load_test_set(data_dir: str = "data", mmap: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the test set data and labels.
    
    Args:
        data_dir: Directory containing the data files
        
    Returns:
        Tuple of (data, labels) as numpy arrays
    """
    data_path = Path(data_dir)
    
    mmap_mode = "r" if mmap else None
    data = np.load(data_path / "test_data.npy", mmap_mode=mmap_mode)
    labels = np.load(data_path / "test_labels.npy", mmap_mode=mmap_mode)
    
    return data, labels


def get_subset_data(
    indices: np.ndarray,
    data_dir: str = "data",
    split: str = "selection_pool",
    mmap: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a subset of data by indices.
    
    Args:
        indices: Array of indices to select
        data_dir: Directory containing the data files
        split: Which split to use ("selection_pool", "validation", "test")
        
    Returns:
        Tuple of (data, labels) for the selected subset
    """
    if split == "selection_pool":
        data, labels = load_selection_pool(data_dir, mmap=mmap)
    elif split == "validation":
        data, labels = load_validation_set(data_dir, mmap=mmap)
    elif split == "test":
        data, labels = load_test_set(data_dir, mmap=mmap)
    else:
        raise ValueError(f"Unknown split: {split}. Use 'selection_pool', 'validation', or 'test'")
    
    # Ensure indices are valid
    indices = np.asarray(indices)
    if np.any(indices >= len(data)) or np.any(indices < 0):
        raise ValueError(f"Invalid indices: some indices out of range [0, {len(data)})")
    
    return data[indices], labels[indices]


def create_dataloader(
    data: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = False,
    indices: Optional[np.ndarray] = None,
    pin_memory: bool = False,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    seed: Optional[int] = None,
) -> DataLoader:
    """
    Create a PyTorch DataLoader from numpy arrays.
    
    Args:
        data: Data array (N, C, H, W)
        labels: Labels array (N,)
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        indices: Optional subset indices. If None, uses all data.
        
    Returns:
        PyTorch DataLoader
    """
    dataset = MNISTSubsetDataset(data, labels, indices)
    # prefetch_factor is only valid when num_workers > 0
    effective_prefetch = prefetch_factor if num_workers > 0 else None
    effective_persistent = persistent_workers if num_workers > 0 else False

    generator = None
    worker_init_fn: Optional[Callable[[int], None]] = None
    if seed is not None:
        # Deterministic shuffling and per-worker RNG seeding.
        generator = torch.Generator()
        generator.manual_seed(int(seed))

        def _seed_worker(worker_id: int) -> None:
            wseed = int(seed) + int(worker_id)
            np.random.seed(wseed)
            random.seed(wseed)
            torch.manual_seed(wseed)

        worker_init_fn = _seed_worker

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        prefetch_factor=effective_prefetch,
        persistent_workers=effective_persistent,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )


def get_class_distribution(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    Get the class distribution (count per class).
    
    Args:
        labels: Array of class labels
        num_classes: Number of classes
        
    Returns:
        Array of shape (num_classes,) with counts per class
    """
    return np.bincount(labels, minlength=num_classes)


def get_class_weights(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    Compute class weights for balanced training (inverse frequency).
    
    Args:
        labels: Array of class labels
        num_classes: Number of classes
        
    Returns:
        Array of shape (num_classes,) with weights per class
    """
    class_counts = get_class_distribution(labels, num_classes)
    total = len(labels)
    
    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)
    
    # Inverse frequency weighting
    weights = total / (num_classes * class_counts)
    
    return weights


if __name__ == "__main__":
    # Example usage
    print("Loading dataset metadata...")
    metadata = load_dataset_metadata()
    print(f"Metadata: {metadata}")
    
    print("\nLoading selection pool...")
    data, labels = load_selection_pool()
    print(f"Selection pool shape: {data.shape}, labels shape: {labels.shape}")
    print(f"Class distribution: {get_class_distribution(labels)}")
    
    print("\nLoading validation set...")
    val_data, val_labels = load_validation_set()
    print(f"Validation shape: {val_data.shape}, labels shape: {val_labels.shape}")
    
    print("\nLoading test set...")
    test_data, test_labels = load_test_set()
    print(f"Test shape: {test_data.shape}, labels shape: {test_labels.shape}")
    
    print("\nExample: Get subset by indices...")
    subset_indices = np.array([0, 1, 2, 3, 4])
    subset_data, subset_labels = get_subset_data(subset_indices)
    print(f"Subset shape: {subset_data.shape}, labels: {subset_labels}")
    
    print("\nExample: Create DataLoader...")
    dataloader = create_dataloader(data, labels, batch_size=32, shuffle=True)
    print(f"DataLoader created with {len(dataloader)} batches")

