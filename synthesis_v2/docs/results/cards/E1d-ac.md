**Probe:** hybrid | **input:** autocontrast | **dataset:** `data/e1d_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 8000 (best at 7500) | **train time:** 1074.3 min/run

| Metric (dev, pen-40) | E1d-ac (n=1) | E1d (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 0.0% | 0.0% | -0.0 pt [-0.0, -0.0] |
| Accuracy | 100.0% | 100.0% | +0.0 pt [+0.0, +0.0] |
| Top-3 | 100.0% | 100.0% | +0.0 pt |
| Macro-F1 | 1.000 | 1.000 | +0.000 |
| Aicme acc | 100.0% | 100.0% | +0.0 pt |
| Mean conf | 100.0% | 100.0% | +0.0 pt |
| NLL | 0.000 | 0.000 | -0.000 [-0.000, +0.000] |
| ECE | 0.000 | 0.000 | -0.000 |
| Synthetic val acc | 98.8% | 99.1% | -0.3 pt |
| Synth→real gap (val acc − dev acc) | -1.2% | -0.9% | -0.3 pt |
| Dev-stress acc (supplementary, DR-022) | 97.5% (n=1) | 98.3% (n=1) | -0.8 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E1d-ac | 100% | 100% | 100% | 100% | 100% | 85% |
| E1d | 100% | 100% | 100% | 100% | 100% | 90% |

**Blank-page control:** A (0.45)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E1d-ac | 0.0% | 0.0% |
| E1d | 0.0% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E1d-ac | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E1d | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Figures:** [curves](results/figures/E1d-ac/learning_curves.png) · [confusion](results/figures/E1d-ac/confusion_dev.png) · [taxonomy](results/figures/E1d-ac/taxonomy_dev.png)
