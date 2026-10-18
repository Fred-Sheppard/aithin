**Probe:** hybrid | **input:** autocontrast | **dataset:** `data/legacy_n850_s0` (850/class) | **seeds:** 3 | **steps used:** 6833 (best at 4833) | **train time:** 15.6 min/run

| Metric (dev, pen-40) | E0c (n=3) | E0a (n=3) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 18.3% ± 5.9 | 41.7% ± 7.2 | -23.3 pt [-35.0, -11.7] |
| Accuracy | 81.7% ± 5.9 | 58.3% ± 7.2 | +23.3 pt [+11.7, +35.0] |
| Top-3 | 99.2% ± 1.2 | 83.3% ± 1.2 | +15.8 pt |
| Macro-F1 | 0.792 ± 0.072 | 0.530 ± 0.084 | +0.263 |
| Aicme acc | 97.5% ± 2.0 | 75.8% ± 2.4 | +21.7 pt |
| Mean conf | 96.0% ± 0.6 | 92.2% ± 3.9 | +3.9 pt |
| NLL | 0.950 ± 0.386 | 3.019 ± 0.244 | -2.069 [-3.267, -0.986] |
| ECE | 0.179 ± 0.062 | 0.355 ± 0.042 | -0.176 |
| Synthetic val acc | 98.9% ± 0.2 | 98.6% ± 0.1 | +0.2 pt |
| Synth→real gap (val acc − dev acc) | 17.2% ± 6.1 | 40.3% ± 7.1 | -23.1 pt |
| Dev-stress acc (supplementary, DR-022) | 80.1% (n=3) | 38.2% (n=3) | +41.9 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E0c | 82% | 83% | 83% | 73% | 82% | 78% |
| E0a | 28% | 32% | 52% | 47% | 13% | 56% |

**Blank-page control:** R (1.00), I (1.00), R (1.00)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E0c | 20.0% | 16.7% |
| E0a | 53.3% | 30.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E0c | 0.0 | 0.3 | 0.7 | 6.3 | 0.0 |
| E0a | 0.0 | 5.3 | 4.3 | 7.0 | 0.0 |

**Figures:** [curves](results/figures/E0c/learning_curves.png) · [confusion](results/figures/E0c/confusion_dev.png) · [taxonomy](results/figures/E0c/taxonomy_dev.png)
