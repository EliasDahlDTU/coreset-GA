"""
Default configuration file for coreset-GA project.

Centralizes all hyperparameters, paths, and settings for reproducibility.
"""

from pathlib import Path
import torch

# ============================================================================
# PATHS
# ============================================================================

# Base directories (config/ is one level down from root)
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
PRETRAINED_COMMITTEE_MODELS_DIR = BASE_DIR / "pretrained_committee_models"
RESULTS_DIR = BASE_DIR / "results"
TRAINING_DIR = BASE_DIR / "training"
FINAL_MODELS_DIR = BASE_DIR / "final_models"

# Data files
SELECTION_POOL_DATA = DATA_DIR / "selection_pool_data.npy"
SELECTION_POOL_LABELS = DATA_DIR / "selection_pool_labels.npy"
VALIDATION_DATA = DATA_DIR / "validation_data.npy"
VALIDATION_LABELS = DATA_DIR / "validation_labels.npy"
TEST_DATA = DATA_DIR / "test_data.npy"
TEST_LABELS = DATA_DIR / "test_labels.npy"
DATASET_METADATA = DATA_DIR / "dataset_metadata.json"

# Embedding files
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "embeddings.npy"
DIFFICULTY_SCORES_FILE = EMBEDDINGS_DIR / "difficulty_scores.npy"

# ============================================================================
# DATASET SETTINGS
# ============================================================================

# Subset sizes to test (k values)
K_VALUES = [50, 100, 200, 500, 750, 1000]

# Number of classes (MNIST)
NUM_CLASSES = 10

# Image shape (MNIST: 1 channel, 28x28)
IMAGE_SHAPE = (1, 28, 28)

# ============================================================================
# GENETIC ALGORITHM SETTINGS
# ============================================================================

# Population and evolution
GA_POPULATION_SIZE = 30
GA_GENERATIONS = 20
GA_SEED = 42

# Mutation operator probabilities
MUTATION_INDEX_REPLACEMENT_PROB = 0.70  # Replace 1-5 random indices
MUTATION_SEGMENT_SHUFFLE_PROB = 0.20   # Shuffle 10-20% segment
MUTATION_SWAP_PROB = 0.10              # Swap two indices

# Mutation parameters
MUTATION_MIN_REPLACEMENTS = 1
MUTATION_MAX_REPLACEMENTS = 5
MUTATION_SEGMENT_MIN_PCT = 0.10
MUTATION_SEGMENT_MAX_PCT = 0.20

# Crossover
CROSSOVER_PROB = 0.8  # Probability of crossover

# Selection
TOURNAMENT_SIZE = 2

# NSGA-II settings
NSGA2_ETA_C = 20.0  # Crossover distribution index
NSGA2_ETA_M = 20.0  # Mutation distribution index

# ============================================================================
# COMMITTEE MODELS SETTINGS
# ============================================================================

# Number of committee models
COMMITTEE_SIZE = 3

# Committee model names (to be loaded from pretrained_committee_models/)
COMMITTEE_MODEL_NAMES = [
    "resnet18",
    "vgg11",
    "mobilenet_v2"
]

# Device for committee inference
COMMITTEE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMMITTEE_BATCH_SIZE = 128

# ============================================================================
# EMBEDDING SETTINGS
# ============================================================================

# Embedding dimension
EMBEDDING_DIM = 512

# Feature extractor model (for diversity computation)
FEATURE_EXTRACTOR = "resnet50"  # Backbone without final layer

# Device for embedding extraction
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_BATCH_SIZE = 128

# ============================================================================
# MODEL TRAINING SETTINGS
# ============================================================================

# Training hyperparameters
TRAIN_LEARNING_RATE = 0.001
TRAIN_WEIGHT_DECAY = 1e-6
TRAIN_EPOCHS = 50
TRAIN_BATCH_SIZE_BASE = 32  # Will be adjusted based on subset size

# Early stopping
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MONITOR = "val_accuracy"
EARLY_STOPPING_MODE = "max"

# Model architecture
CNN_CHANNELS = [32, 64, 128]  # Conv block channels
CNN_DENSE_UNITS = 128
CNN_DROPOUT = 0.5

# Training device
TRAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Random seed for training
TRAIN_SEED = 42

# ============================================================================
# BASELINE SETTINGS
# ============================================================================

# Number of random baseline runs
NUM_RANDOM_BASELINES = 5

# Baseline random seed
BASELINE_SEED = 42

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

# Subset selection from Pareto front
SUBSET_SELECTION_WEIGHTS = {
    "difficulty": 1.0 / 3.0,
    "diversity": 1.0 / 3.0,
    "balance": 1.0 / 3.0
}

# Evaluation metrics
COMPUTE_CLASS_WISE_F1 = True
COMPUTE_CALIBRATION_ERROR = True
COMPUTE_CONVERGENCE_SPEED = True

# ============================================================================
# RESULTS AND LOGGING
# ============================================================================

# Save settings
SAVE_PARETO_FRONTS = True
SAVE_SELECTED_SUBSETS = True
SAVE_TRAINING_CURVES = True
SAVE_MODELS = True

# File naming patterns
PARETO_FRONT_PATTERN = "pareto_k{k}.pkl"
SELECTED_SUBSET_PATTERN = "selected_k{k}.npy"
TRAINING_CURVES_PATTERN = "training_curves_k{k}.json"
MODEL_PATTERN = "cnn_k{k}.pth"

# Logging
LOG_LEVEL = "INFO"
LOG_TO_FILE = True
LOG_FILE = RESULTS_DIR / "experiment.log"

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

# Master random seed (for overall experiment reproducibility)
MASTER_SEED = 42

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_batch_size(subset_size: int) -> int:
    """
    Compute batch size based on subset size.
    Ensures multiple batches even for small subsets.
    
    Args:
        subset_size: Size of the training subset
        
    Returns:
        Batch size to use
    """
    return min(TRAIN_BATCH_SIZE_BASE, max(1, subset_size // 2))


def get_pareto_front_path(k: int) -> Path:
    """Get path for Pareto front file for given k."""
    return RESULTS_DIR / PARETO_FRONT_PATTERN.format(k=k)


def get_selected_subset_path(k: int) -> Path:
    """Get path for selected subset file for given k."""
    return RESULTS_DIR / SELECTED_SUBSET_PATTERN.format(k=k)


def get_training_curves_path(k: int) -> Path:
    """Get path for training curves file for given k."""
    return RESULTS_DIR / TRAINING_CURVES_PATTERN.format(k=k)


def get_model_path(k: int) -> Path:
    """Get path for trained model file for given k."""
    return FINAL_MODELS_DIR / MODEL_PATTERN.format(k=k)

