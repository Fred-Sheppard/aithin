**Probe:** hybrid | **input:** canonical | **dataset:** `data/legacy_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 8000 (best at 7000) | **train time:** 18.9 min/run

| Metric (dev, pen-40) | E0b (n=1) | E0a (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 32.5% | 42.5% | -10.0 pt [-30.0, +10.0] |
| Accuracy | 67.5% | 57.5% | +10.0 pt [-10.0, +30.0] |
| Top-3 | 95.0% | 85.0% | +10.0 pt |
| Macro-F1 | 0.636 | 0.492 | +0.144 |
| Aicme acc | 90.0% | 77.5% | +12.5 pt |
| Mean conf | 97.1% | 91.8% | +5.3 pt |
| NLL | 2.519 | 2.701 | -0.183 [-1.720, +1.511] |
| ECE | 0.296 | 0.343 | -0.047 |
| Synthetic val acc | 98.9% | 98.8% | +0.1 pt |
| Synth→real gap (val acc − dev acc) | 31.4% | 41.3% | -9.9 pt |
| Dev-stress acc (supplementary, DR-022) | 66.2% (n=1) | 38.3% (n=1) | +27.9 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E0b | 70% | 68% | 68% | 72% | 70% | 50% |
| E0a | 28% | 32% | 52% | 52% | 10% | 55% |

**Blank-page control:** E (0.96)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E0b | 30.0% | 35.0% |
| E0a | 60.0% | 25.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E0b | 3.0 | 0.0 | 1.0 | 9.0 | 0.0 |
| E0a | 0.0 | 6.0 | 3.0 | 8.0 | 0.0 |

**Figures:** [curves](results/figures/E0b/learning_curves.png) · [confusion](results/figures/E0b/confusion_dev.png) · [taxonomy](results/figures/E0b/taxonomy_dev.png)
