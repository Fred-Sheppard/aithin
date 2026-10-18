**Probe:** hybrid | **input:** legacy | **dataset:** `data/legacy_n850_s0` (850/class) | **seeds:** 3 | **steps used:** 6833 (best at 4833) | **train time:** 16.1 min/run

| Metric (dev, pen-40) | E0a (n=3) |
|---|---|
| CER | 41.7% ± 7.2 |
| Accuracy | 58.3% ± 7.2 |
| Top-3 | 83.3% ± 1.2 |
| Macro-F1 | 0.530 ± 0.084 |
| Aicme acc | 75.8% ± 2.4 |
| Mean conf | 92.2% ± 3.9 |
| NLL | 3.019 ± 0.244 |
| ECE | 0.355 ± 0.042 |
| Synthetic val acc | 98.6% ± 0.1 |
| Synth→real gap (val acc − dev acc) | 40.3% ± 7.1 |
| Dev-stress acc (supplementary, DR-022) | 38.2% (n=3) |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E0a | 28% | 32% | 52% | 47% | 13% | 56% |

**Blank-page control:** H (0.79), A (0.57), A (0.36)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E0a | 53.3% | 30.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E0a | 0.0 | 5.3 | 4.3 | 7.0 | 0.0 |

**Figures:** [curves](results/figures/E0a/learning_curves.png) · [confusion](results/figures/E0a/confusion_dev.png) · [taxonomy](results/figures/E0a/taxonomy_dev.png)
