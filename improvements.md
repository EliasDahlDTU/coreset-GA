## High-Budget Improvements (Dec 18, 2025)

This run is configured for accuracy-first, compute-heavy execution (budget ~12h).

### Genetic Algorithm
- Population: 160, Generations: 120 (deeper search).
- Diversity reshaped (sqrt of 1 - cosine sim) to better separate similar subsets.
- Hard class-coverage constraint: any subset missing a class is penalized out.
- Selection weights: tilt toward balance (difficulty/diversity/balance = 0.30/0.30/0.40).
- Objective normalization in subset selection uses z-scores for stability.
- Mutation broadened: max replacements 8, crossover prob 0.9.

### Training
- Epochs: 150, early stopping disabled (best checkpoint still saved on val loss).
- Random baselines: 10 runs for stronger comparison.
- Torch compile enabled, AMP, channels_last, LR on plateau.

### What to rerun
Use existing data/embeddings/committee artifacts:
```
python run_all.py --skip-data --skip-committee --skip-inference --skip-embeddings
```
If you only want evaluation after training: 
```
python training/evaluate_models.py --k-values 50 100 200 500 750 1000
```

### Expected impact
- Better GA guidance via stricter diversity and class coverage.
- More exhaustive GA search.
- Longer training with best-checkpoint selection (no early stop).
- Stronger baseline averaging (10 runs).


