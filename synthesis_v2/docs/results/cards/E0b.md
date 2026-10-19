**Probe:** hybrid | **input:** canonical | **dataset:** `data/legacy_n850_s0` (850/class) | **seeds:** 3 | **steps used:** 7667 (best at 6000) | **train time:** 17.4 min/run

| Metric (dev, pen-40) | E0b (n=3) | E0a (n=3) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 27.5% ± 3.5 | 41.7% ± 7.2 | -14.2 pt [-29.2, +2.5] |
| Accuracy | 72.5% ± 3.5 | 58.3% ± 7.2 | +14.2 pt [-2.5, +29.2] |
| Top-3 | 96.7% ± 1.2 | 83.3% ± 1.2 | +13.3 pt |
| Macro-F1 | 0.675 ± 0.028 | 0.530 ± 0.084 | +0.145 |
| Aicme acc | 92.5% ± 2.0 | 75.8% ± 2.4 | +16.7 pt |
| Mean conf | 97.6% ± 0.4 | 92.2% ± 3.9 | +5.4 pt |
| NLL | 2.190 ± 0.236 | 3.019 ± 0.244 | -0.829 [-2.343, +0.789] |
| ECE | 0.273 ± 0.016 | 0.355 ± 0.042 | -0.082 |
| Synthetic val acc | 99.1% ± 0.1 | 98.6% ± 0.1 | +0.5 pt |
| Synth→real gap (val acc − dev acc) | 26.6% ± 3.4 | 40.3% ± 7.1 | -13.7 pt |
| Dev-stress acc (supplementary, DR-022) | 69.0% (n=3) | 38.2% (n=3) | +30.8 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E0b | 72% | 72% | 71% | 72% | 73% | 52% |
| E0a | 28% | 32% | 52% | 47% | 13% | 56% |

**Blank-page control:** E (0.96), H (0.79), A (0.59)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E0b | 23.3% | 31.7% |
| E0a | 53.3% | 30.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E0b | 2.7 | 0.0 | 0.3 | 8.0 | 0.0 |
| E0a | 0.0 | 5.3 | 4.3 | 7.0 | 0.0 |

**Figures:** [curves](results/figures/E0b/learning_curves.png) · [confusion](results/figures/E0b/confusion_dev.png) · [taxonomy](results/figures/E0b/taxonomy_dev.png)
