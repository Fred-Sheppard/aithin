**Probe:** hybrid | **input:** legacy | **dataset:** `data/legacy_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 7500 (best at 5500) | **train time:** 20.0 min/run

| Metric (dev, pen-40) | E0a (n=1) |
|---|---|
| CER | 42.5% |
| Accuracy | 57.5% |
| Top-3 | 85.0% |
| Macro-F1 | 0.492 |
| Aicme acc | 77.5% |
| Mean conf | 91.8% |
| NLL | 2.701 |
| ECE | 0.343 |
| Synthetic val acc | 98.8% |
| Synth→real gap (val acc − dev acc) | 41.3% |
| Dev-stress acc (supplementary, DR-022) | 38.3% (n=1) |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E0a | 28% | 32% | 52% | 52% | 10% | 55% |

**Blank-page control:** H (0.79)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E0a | 60.0% | 25.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E0a | 0.0 | 6.0 | 3.0 | 8.0 | 0.0 |

**Figures:** [curves](results/figures/E0a/learning_curves.png) · [confusion](results/figures/E0a/confusion_dev.png) · [taxonomy](results/figures/E0a/taxonomy_dev.png)
