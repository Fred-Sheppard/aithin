**Probe:** hybrid | **input:** autocontrast | **dataset:** `data/legacy_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 6500 (best at 4500) | **train time:** 14.7 min/run

| Metric (dev, pen-40) | E0c (n=1) | E0a (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 10.0% | 42.5% | -32.5 pt [-47.5, -20.0] |
| Accuracy | 90.0% | 57.5% | +32.5 pt [+20.0, +47.5] |
| Top-3 | 100.0% | 85.0% | +15.0 pt |
| Macro-F1 | 0.893 | 0.492 | +0.401 |
| Aicme acc | 100.0% | 77.5% | +22.5 pt |
| Mean conf | 96.0% | 91.8% | +4.2 pt |
| NLL | 0.436 | 2.701 | -2.265 [-3.427, -1.126] |
| ECE | 0.094 | 0.343 | -0.249 |
| Synthetic val acc | 98.6% | 98.8% | -0.2 pt |
| Synth→real gap (val acc − dev acc) | 8.6% | 41.3% | -32.7 pt |

**Blank-page control:** R (1.00)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E0c | 10.0% | 10.0% |
| E0a | 60.0% | 25.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E0c | 0.0 | 0.0 | 0.0 | 4.0 | 0.0 |
| E0a | 0.0 | 6.0 | 3.0 | 8.0 | 0.0 |

**Figures:** [curves](results/figures/E0c/learning_curves.png) · [confusion](results/figures/E0c/confusion_dev.png) · [taxonomy](results/figures/E0c/taxonomy_dev.png)
