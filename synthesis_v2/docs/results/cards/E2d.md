**Probe:** hybrid | **input:** canonical | **dataset:** `data/e2d_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 8000 (best at 7000) | **train time:** 18.0 min/run

| Metric (dev, pen-40) | E2d (n=1) | E2c (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 0.0% | 0.0% | -0.0 pt [-0.0, -0.0] |
| Accuracy | 100.0% | 100.0% | +0.0 pt [+0.0, +0.0] |
| Top-3 | 100.0% | 100.0% | +0.0 pt |
| Macro-F1 | 1.000 | 1.000 | +0.000 |
| Aicme acc | 100.0% | 100.0% | +0.0 pt |
| Mean conf | 100.0% | 100.0% | +0.0 pt |
| NLL | 0.000 | 0.000 | -0.000 [-0.001, +0.000] |
| ECE | 0.000 | 0.000 | -0.000 |
| Synthetic val acc | 90.9% | 99.0% | -8.1 pt |
| Synth→real gap (val acc − dev acc) | -9.1% | -1.0% | -8.1 pt |
| Dev-stress acc (supplementary, DR-022) | 100.0% (n=1) | 100.0% (n=1) | +0.0 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E2d | 100% | 100% | 100% | 100% | 100% | 100% |
| E2c | 100% | 100% | 100% | 100% | 100% | 100% |

**Blank-page control:** H (0.11)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E2d | 0.0% | 0.0% |
| E2c | 0.0% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E2d | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E2c | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Figures:** [curves](results/figures/E2d/learning_curves.png) · [confusion](results/figures/E2d/confusion_dev.png) · [taxonomy](results/figures/E2d/taxonomy_dev.png) · [contact sheet](results/figures/E2d/contact_sheet.png)
