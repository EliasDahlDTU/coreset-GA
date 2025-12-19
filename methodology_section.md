## 2. Methodology

We formulate subset selection as a search problem and solve it using a genetic algorithm (GA). Each individual represents a candidate subset of fixed size k. Because full retraining for fitness evaluation of each candidate is too costly, we must rely on proxy fitness functions derived from pretrained models to reduce overall computation time.

### 2.1 SBSE Formulation

**Search space**: All k-sized subsets of dataset D (combinatorially large)

**Decision variables**: Indices of selected samples

**Objectives**:
- Difficulty (informativeness)
- Diversity (non-redundancy via coverage)
- Class balance

We solve this using NSGA-II, which efficiently explores trade-offs between conflicting objectives and maintains a diverse Pareto front.

### 2.2 Proxy-Based Fitness Function

Training a model for every candidate subset is computationally infeasible. Instead, we use proxy metrics that correlate with downstream model performance.

**Fitness(S) = (Difficulty(S), Diversity(S), Balance(S))**

All metrics are fast to evaluate using precomputed features, enabling large-scale search.

#### Difficulty (Informativeness)

Measured via committee uncertainty. For each sample x, we obtain softmax predictions from three pretrained models: p₁(x), p₂(x), p₃(x). We average these predictions and compute entropy to quantify uncertainty:

[Equation 1: Averaged Committee Prediction]
```
p̄(x) = (1/3) · [p₁(x) + p₂(x) + p₃(x)]
```

[Equation 2: Entropy of Averaged Prediction]
```
H(p̄(x)) = -Σᵢ p̄ᵢ(x) · log(p̄ᵢ(x))
```

[Equation 3: Subset Difficulty]
```
Difficulty(S) = (1/|S|) · Σ_{x∈S} H(p̄(x))
```

Higher entropy indicates higher informativeness, as the committee is uncertain about the sample's class. The difficulty scores H(p̄(x)) are precomputed for all samples in the pool, making subset evaluation O(|S|).

#### Diversity (Coverage)

We measure diversity as how well the subset "covers" the data manifold, rather than intra-subset pairwise similarity. This coverage-based approach better aligns with training objectives. We extract 512-dimensional feature embeddings φ(x) using a pretrained ResNet50 feature extractor (precomputed for all samples).

[Equation 4: Coverage Sample]
```
Sample M points uniformly from pool D: {x₁, x₂, ..., x_M}
```

[Equation 5: Cosine Similarity Matrix]
```
sim(xⱼ, xᵢ) = cos(φ(xⱼ), φ(xᵢ)) = (φ(xⱼ)ᵀ · φ(xᵢ)) / (||φ(xⱼ)|| · ||φ(xᵢ)||)
```

[Equation 6: Maximum Similarity per Sample Point]
```
max_sim(xⱼ, S) = max_{xᵢ∈S} sim(xⱼ, xᵢ)
```

[Equation 7: Mean Maximum Similarity]
```
mean_max_sim(S) = (1/M) · Σ_{j=1}^M max_sim(xⱼ, S)
```

[Equation 8: Diversity Score]
```
Diversity(S) = (mean_max_sim(S) + 1) / 2
```

The score is mapped from [-1, 1] to [0, 1] for normalization. Higher diversity indicates the subset better represents the full data distribution. The computation is O(M · |S|) per evaluation, where M=4096 is the coverage sample size.

#### Balance

Measures deviation from uniform class distribution. Balanced subsets are preferred for robustness and fairness.

[Equation 9: Class Proportions]
```
p_c(S) = |{xᵢ ∈ S : yᵢ = c}| / |S|,  for c ∈ {1, 2, ..., C}
```

[Equation 10: Ideal Uniform Proportion]
```
p* = 1/C
```

[Equation 11: L1 Deviation from Uniform]
```
deviation(S) = Σ_{c=1}^C |p_c(S) - p*|
```

[Equation 12: Maximum Possible Deviation]
```
max_deviation = 2 · (1 - 1/C)
```
(This occurs when all samples belong to a single class.)

[Equation 13: Normalized Deviation]
```
normalized_deviation(S) = deviation(S) / max_deviation
```

[Equation 14: Balance Score]
```
Balance(S) = 1 - normalized_deviation(S)
```

Additionally, we enforce a hard constraint: Balance(S) = 0 if any class is missing from the subset (i.e., if ∃c: p_c(S) = 0). This ensures all classes are represented, which is essential for multi-class classification.
