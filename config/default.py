"""
Default configuration file for coreset-GA project.

Centralizes all hyperparameters, paths, and settings for reproducibility.
"""

from pathlib import Path
import torch
import os

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

# Population and evolution (high-budget search)
GA_POPULATION_SIZE = 256
GA_GENERATIONS = 200
GA_SEED = 42

# Mutation operator probabilities
MUTATION_INDEX_REPLACEMENT_PROB = 0.70  # Replace 1-5 random indices
MUTATION_SEGMENT_SHUFFLE_PROB = 0.20   # Shuffle 10-20% segment
MUTATION_SWAP_PROB = 0.10              # Swap two indices

# Mutation parameters
MUTATION_MIN_REPLACEMENTS = 1
MUTATION_MAX_REPLACEMENTS = 8  # Broader exploration for small-k runs
MUTATION_SEGMENT_MIN_PCT = 0.10
MUTATION_SEGMENT_MAX_PCT = 0.20

# Crossover
CROSSOVER_PROB = 0.9  # More crossover to mix good genes under higher budget

# Selection
# Tournament size controls selection pressure. With set-valued chromosomes and noisy objectives,
# slightly higher pressure tends to help the search converge to better fronts.
TOURNAMENT_SIZE = 4

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
COMMITTEE_BATCH_SIZE = 512  # A100-friendly batch size for small MNIST tensors

# ============================================================================
# EMBEDDING SETTINGS
# ============================================================================

# Embedding dimension
EMBEDDING_DIM = 512

# Feature extractor model (for diversity computation)
FEATURE_EXTRACTOR = "resnet50"  # Backbone without final layer

# Device for embedding extraction
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_BATCH_SIZE = 512  # Safe on A100 20GB for MNIST-sized inputs

# ============================================================================
# MODEL TRAINING SETTINGS
# ============================================================================

# Training hyperparameters
TRAIN_LEARNING_RATE = 0.001
TRAIN_WEIGHT_DECAY = 1e-6
TRAIN_EPOCHS = 150  # Longer training; budget allows deeper convergence
TRAIN_BATCH_SIZE_BASE = 512  # Cap; actual batch = min(base, k//2)
TRAIN_NUM_WORKERS = 8       # A100 host can typically sustain this
TRAIN_PIN_MEMORY = True     # Pin host memory for faster H2D
TRAIN_NON_BLOCKING = True   # Non-blocking H2D copies
TRAIN_USE_AMP = True        # Mixed precision (Tensor Cores on A100)
TRAIN_CHANNELS_LAST = True  # Use NHWC for better conv throughput on Ampere
TRAIN_PREFETCH_FACTOR = 8   # Higher prefetch for fast GPU
TRAIN_PERSISTENT_WORKERS = True  # Keep workers alive between epochs
TRAIN_CUDNN_BENCHMARK = True     # Let cuDNN pick fastest algos for fixed shapes
TRAIN_TORCH_COMPILE = True      
TRAIN_TORCH_COMPILE_MODE = "reduce-overhead"  # Lower Python overhead
TRAIN_AUTOCast_DTYPE = "bfloat16"     # Options: "float16", "bfloat16"
ALLOW_TF32 = True                    # Allow TF32 matmuls on Ampere+

# Early stopping (stable defaults)
EARLY_STOPPING_ENABLED = False  # Run full epochs; rely on best-checkpoint selection
EARLY_STOPPING_PATIENCE = 20    # Unused when disabled; higher if re-enabled
EARLY_STOPPING_MONITOR = "val_loss"  # "val_loss" is smoother than accuracy
EARLY_STOPPING_MODE = "min"
EARLY_STOPPING_MIN_DELTA = 0.0   # Track any improvement when saving best

# LR scheduler (ReduceLROnPlateau)
LR_SCHEDULER_USE_PLATEAU = True
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 2
LR_SCHEDULER_COOLDOWN = 1
LR_SCHEDULER_MIN_LR = 1e-5

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

# Number of random baseline runs (stronger comparison)
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

# For small subsets, over-emphasizing difficulty can hurt generalization (you end up picking
# mostly hard/ambiguous points). These defaults bias towards class coverage + representativeness.
SUBSET_SELECTION_SMALL_K_THRESHOLD = 200
SUBSET_SELECTION_WEIGHTS_SMALL_K = {
    "difficulty": 1.0 / 3.0 ,
    "diversity": 1.0 / 3.0,
    "balance": 1.0 / 3.0,
}

# For larger subsets, balance is usually already satisfied (and near-uniform),
# and over-weighting it can hurt by selecting less informative / less representative points.
# These defaults were tuned on MNIST k=1000 and improved GA-selected CNN accuracy.
SUBSET_SELECTION_LARGE_K_THRESHOLD = 500
SUBSET_SELECTION_WEIGHTS_LARGE_K = {
    "difficulty": 1.0 / 3.0 ,
    "diversity": 1.0 / 3.0,
    "balance": 1.0 / 3.0,
}

# Evaluation metrics
COMPUTE_CLASS_WISE_F1 = True
COMPUTE_CALIBRATION_ERROR = True
COMPUTE_CONVERGENCE_SPEED = True

# Diversity evaluation acceleration
DIVERSITY_USE_GPU = True  # Compute diversity on GPU to avoid CPU bottleneck
DIVERSITY_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# NOTE: Diversity drives GA selection. Low precision here can inject noise and make the GA
# behave closer to random search. Use float32 by default for stability; you can switch back
# to bfloat16/float16 if you need speed and are OK with noisier fitness.
DIVERSITY_TORCH_DTYPE = "float32"  # Options: "float16", "bfloat16", "float32"

# Diversity objective mode:
# - "intra": maximize spread within the subset (1 - mean cosine similarity)
# - "coverage": maximize representativeness of the whole pool (mean max cosine similarity
#               from a sampled pool set to the subset). Often better aligned with
#               coreset training than pure intra-diversity.
DIVERSITY_MODE = "coverage"  # Options: "intra", "coverage"
# Coverage objective is an estimate. Defaults use a single fixed subsample for speed/stability.
# You can average multiple subsamples by adding more seeds to COVERAGE_SAMPLE_SEEDS.
COVERAGE_SAMPLE_SIZE = 4096  # Pool points per subsample; higher = more stable, slower
COVERAGE_SAMPLE_SEEDS = [42]

# Class-conditional coverage:
# When enabled, coverage is computed per class by matching sampled points of class c
# only against subset points of class c, then averaging across classes. This is often
# better aligned with classification (MNIST/CIFAR) than global coverage.
COVERAGE_CLASS_CONDITIONAL = False

# If set, overrides per-class sample size; otherwise derived from COVERAGE_SAMPLE_SIZE / NUM_CLASSES.
COVERAGE_PER_CLASS_SAMPLE_SIZE = None

# Coverage aggregation:
# - "max": mean of max similarity to subset (1-NN coverage). Fast, but can be spiky.
# - "topk": mean of top-k similarities (more stable than pure max).
# - "softmax": smooth-max via log-sum-exp with temperature (t -> 0 approximates max).
# Env overrides (for sweeps): CORESETGA_COVERAGE_AGGREGATION / _TOPK / _TAU
COVERAGE_AGGREGATION = os.getenv("CORESETGA_COVERAGE_AGGREGATION", "max")  # Options: "max", "topk", "softmax"
try:
    COVERAGE_TOPK = int(os.getenv("CORESETGA_COVERAGE_TOPK", "5"))
except Exception:
    COVERAGE_TOPK = 5
try:
    COVERAGE_SOFTMAX_TAU = float(os.getenv("CORESETGA_COVERAGE_SOFTMAX_TAU", "0.07"))
except Exception:
    COVERAGE_SOFTMAX_TAU = 0.07

# Optionally weight sampled points by difficulty (hard points matter more for representativeness).
# This can improve alignment with training accuracy on some runs/datasets.
_cov_w = os.getenv("CORESETGA_COVERAGE_WEIGHT_BY_DIFFICULTY", "0").strip().lower()
COVERAGE_WEIGHT_BY_DIFFICULTY = _cov_w in ("1", "true", "yes", "y", "on")

# Difficulty objective shaping:
# Maximizing "hardest" samples can hurt training at very small k (you overfit to ambiguous digits).
# For small k, we can instead target a moderate difficulty level.
DIFFICULTY_MODE = "high"  # Options: "high", "target_mean"
DIFFICULTY_SMALL_K_THRESHOLD = 200
DIFFICULTY_MODE_SMALL_K = "target_mean"
DIFFICULTY_TARGET_QUANTILE = 0.60   # target difficulty value as a quantile of pool difficulty
DIFFICULTY_TARGET_SIGMA = None      # if None, derived from global difficulty std

# GA initialization. Stratified init can be toggled for experimentation, but default stays off
# since it can reduce exploration diversity depending on k/dataset.
GA_STRATIFIED_INIT = False

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

