# Runtime Estimates (A100)

| Stage                              | Original README estimate | Optimized estimate (current setup) |
|------------------------------------|--------------------------|-------------------------------------|
| GA per k                           | 10–30 min                | 5–12 min                            |
| GA all k (50,100,200,500,750,1000) | ~1–3 hours (6 ks × 10–30 min) | ~30–70 min total                   |
| Training per k (GA + baselines)    | 1–3 hours                | ~1–4 min (k≤200), ~4–10 min (k≥500), full dataset ~12–20 min once |
| Training all k (GA + 5 random each) + full | ~12–24 hours total      | ~3–5 hours total GPU time (sequential on one A100; faster with multi-GPU) |
| End-to-end pipeline (sequential)    | ~12–24 hours             | ~4–6 hours likely (up to ~7–8h if IO/CPU slower) |

**Optimizations included:** AMP with bf16, TF32 allowed, channels_last, optional torch.compile, GPU diversity with bf16, pinned memory/non-blocking transfers, dataloader prefetch/persistent workers, mmap loading for large arrays, batched upper-tri diversity on GPU.

