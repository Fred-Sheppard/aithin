**Probe:** hybrid | **input:** canonical | **dataset:** `data/legacy_n850_s0` (850/class) | **seeds:** 2 | **steps used:** 7500 (best at 6000) | **train time:** 17.1 min/run

| Metric (dev, pen-40) | E0b (n=2) | E0a (n=3) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 28.7% ± 3.7 | 41.7% ± 7.2 | -12.9 pt [-29.6, +5.8] |
| Accuracy | 71.2% ± 3.7 | 58.3% ± 7.2 | +12.9 pt [-5.8, +29.6] |
| Top-3 | 96.2% ± 1.3 | 83.3% ± 1.2 | +12.9 pt |
| Macro-F1 | 0.670 ± 0.034 | 0.530 ± 0.084 | +0.140 |
| Aicme acc | 92.5% ± 2.5 | 75.8% ± 2.4 | +16.7 pt |
| Mean conf | 97.5% ± 0.4 | 92.2% ± 3.9 | +5.3 pt |
| NLL | 2.299 ± 0.220 | 3.019 ± 0.244 | -0.720 [-2.311, +0.985] |
| ECE | 0.277 ± 0.019 | 0.355 ± 0.042 | -0.078 |
| Synthetic val acc | 99.1% ± 0.1 | 98.6% ± 0.1 | +0.4 pt |
| Synth→real gap (val acc − dev acc) | 27.8% ± 3.6 | 40.3% ± 7.1 | -12.5 pt |
| Dev-stress acc (supplementary, DR-022) | 67.3% (n=2) | 38.2% (n=3) | +29.1 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E0b | 71% | 71% | 70% | 70% | 72% | 49% |
| E0a | 28% | 32% | 52% | 47% | 13% | 56% |

**Blank-page control:** E (0.96), H (0.79)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E0b | 27.5% | 30.0% |
| E0a | 53.3% | 30.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E0b | 2.5 | 0.0 | 0.5 | 8.5 | 0.0 |
| E0a | 0.0 | 5.3 | 4.3 | 7.0 | 0.0 |

**Figures:** [curves](results/figures/E0b/learning_curves.png) · [confusion](results/figures/E0b/confusion_dev.png) · [taxonomy](results/figures/E0b/taxonomy_dev.png)
