# Implementation Roadmap

A step-by-step guide to building the coreset-GA system.

---

## === SETUP & DATA PREPARATION ===

- [x] Set up project structure (folders: data/, embeddings/, pretrained_committee_models/, ga/, experiments/, analysis/, results/, final_models/)
- [x] Create requirements.txt with dependencies (numpy, torch, deap, scikit-learn, matplotlib, jupyter, etc.)
- [x] Download/prepare dataset and organize in data/
- [x] Create data loading utilities (train/test splits, class labels, image preprocessing)
- [x] Set up configuration file (config.yaml or config.py) for hyperparameters, paths, k-values

---

## === PRETRAINED COMMITTEE MODELS ===

- [x] Load/download pretrained models for committee (e.g., ResNet18, VGG, MobileNet variants)
- [x] Create committee inference script that:
  - Loads all committee models
  - Runs forward pass on full dataset
  - Saves softmax predictions for each model
- [x] Compute and cache difficulty scores (entropy of averaged softmax) for all samples
- [x] Save difficulty scores to disk (embeddings/difficulty_scores.npy or similar)

---

## === EMBEDDING EXTRACTION ===

- [ ] Load pretrained feature extractor (e.g., ResNet50 backbone, remove final layer)
- [ ] Extract 512-dim embeddings for all dataset images
- [ ] Save embeddings to embeddings/ (one file per image or single .npy array)
- [ ] Create embedding lookup utility for fast access during GA evaluation

---

## === GA CORE COMPONENTS ===

### Population & Chromosome
- [ ] Implement chromosome representation (fixed-length list of k unique indices)
- [ ] Create population initialization (random valid subsets of size k)
- [ ] Add uniqueness validation/enforcement function

### Evaluation Module
- [ ] Implement difficulty objective (load cached scores, compute subset mean)
- [ ] Implement diversity objective (cosine distance in embedding space)
- [ ] Implement balance objective (class distribution deviation from uniform)
- [ ] Create fitness evaluation function that returns (difficulty, diversity, balance) tuple
- [ ] Add caching for objective scores to avoid redundant computation

### Genetic Operators
- [ ] Implement mutation operators:
  - Index replacement (70% probability)
  - Segment shuffle (20% probability)
  - Swap mutation (10% probability)
- [ ] Add post-mutation uniqueness enforcement
- [ ] Implement set-aware uniform crossover
- [ ] Add duplicate handling in crossover (fill with random unseen indices)

### NSGA-II Implementation
- [ ] Implement non-dominated sorting (or use DEAP's NSGA-II)
- [ ] Implement crowding distance calculation
- [ ] Create selection operator (tournament selection on rank + crowding distance)
- [ ] Implement Pareto front extraction
- [ ] Add generation logging (save best individuals, objective values)

---

## === GA EXPERIMENTS ===

- [ ] Create experiment runner script template (experiments/run_k_template.py)
- [ ] Implement run_k50.py, run_k100.py, run_k200.py, run_k500.py, run_k750.py, run_k1000.py
- [ ] Add command-line arguments (k, population size, generations, seed)
- [ ] Implement checkpointing (save population every N generations)
- [ ] Save final Pareto front for each k to results/pareto_k{size}.pkl or .json
- [ ] Log GA metrics (convergence, hypervolume, generation time)

---

## === SUBSET SELECTION ===

- [ ] Implement subset selection from Pareto front:
  - Load Pareto-optimal solutions for given k
  - Compute weighted score (difficulty + diversity + balance)
  - Select best subset (or closest to ideal point)
- [ ] Save selected subset indices to results/selected_k{size}.npy
- [ ] Create visualization of selected subset (class distribution, sample images)

---

## === CNN IMPLEMENTATION ===

- [ ] Define lightweight CNN architecture (3 conv blocks + dense layers)
- [ ] Create model factory function (returns model given number of classes)
- [ ] Implement training loop:
  - Data loading from selected subset
  - Training/validation splits
  - Early stopping
  - Model checkpointing
- [ ] Add training utilities (accuracy calculation, loss tracking, learning rate scheduling)
- [ ] Save trained models to final_models/cnn_k{size}.pth

---

## === BASELINE IMPLEMENTATION ===

- [ ] Implement random baseline (generate 5 random subsets per k, train on each)
- [ ] Implement hardest-only baseline (top-k by difficulty, train model)
- [ ] Implement balanced-only baseline (greedy selection maximizing balance, train model)
- [ ] Implement full-dataset training (upper bound baseline)
- [ ] Save all baseline models to final_models/baseline_{type}_k{size}.pth

---

## === EVALUATION & METRICS ===

- [ ] Create evaluation script that:
  - Loads all trained models (GA-selected + baselines)
  - Runs inference on held-out test set
  - Computes test accuracy, per-class F1, confusion matrix
- [ ] Calculate training efficiency metric (accuracy / k)
- [ ] Log convergence speed (epochs to 90% of final accuracy)
- [ ] Save evaluation results to results/evaluation_k{size}.json

---

## === ANALYSIS & VISUALIZATION ===

- [ ] Create plot_pareto.ipynb:
  - Plot Pareto fronts for each k (3D: difficulty vs diversity vs balance)
  - Show convergence over generations
  - Compare Pareto fronts across different k values
- [ ] Create visualize_subsets.ipynb:
  - Display sample images from selected subsets
  - Show class distribution histograms
  - Compare GA-selected vs random vs baseline subsets
- [ ] Create final results notebook:
  - Plot accuracy vs dataset size Pareto curve
  - Show accuracy comparison (GA vs random vs baselines)
  - Generate efficiency plots (accuracy per sample)
  - Create summary tables

---

## === DOCUMENTATION & CLEANUP ===

- [ ] Add docstrings to all functions and classes
- [ ] Create example usage script (quick start guide)
- [ ] Add README sections for:
  - Installation instructions
  - Quick start example
  - Expected outputs
- [ ] Clean up temporary files and organize results/
- [ ] Create results summary document (key findings, best k values, efficiency gains)

---


