**Probe:** hybrid | **input:** canonical | **dataset:** `data/e2d_v2_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 8000 (best at 6500) | **train time:** 24.3 min/run

| Metric (dev, pen-40) | E2d-v2 (n=1) | E2c (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 0.0% | 0.0% | -0.0 pt [-0.0, -0.0] |
| Accuracy | 100.0% | 100.0% | +0.0 pt [+0.0, +0.0] |
| Top-3 | 100.0% | 100.0% | +0.0 pt |
| Macro-F1 | 1.000 | 1.000 | +0.000 |
| Aicme acc | 100.0% | 100.0% | +0.0 pt |
| Mean conf | 100.0% | 100.0% | +0.0 pt |
| NLL | 0.000 | 0.000 | -0.000 [-0.001, +0.001] |
| ECE | 0.000 | 0.000 | -0.000 |
| Synthetic val acc | 92.2% | 99.0% | -6.8 pt |
| Synth→real gap (val acc − dev acc) | -7.8% | -1.0% | -6.8 pt |
| Dev-stress acc (supplementary, DR-022) | 99.6% (n=1) | 100.0% (n=1) | -0.4 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E2d-v2 | 100% | 100% | 100% | 100% | 100% | 98% |
| E2c | 100% | 100% | 100% | 100% | 100% | 100% |

**Blank-page control:** A (0.10)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E2d-v2 | 0.0% | 0.0% |
| E2c | 0.0% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E2d-v2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E2c | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Figures:** [curves](results/figures/E2d-v2/learning_curves.png) · [confusion](results/figures/E2d-v2/confusion_dev.png) · [taxonomy](results/figures/E2d-v2/taxonomy_dev.png) · [contact sheet](results/figures/E2d-v2/contact_sheet.png)
