**Probe:** hybrid | **input:** canonical | **dataset:** `data/e2c_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 8000 (best at 7000) | **train time:** 17.2 min/run

| Metric (dev, pen-40) | E2c (n=1) | E2b (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 0.0% | 0.0% | -0.0 pt [-0.0, -0.0] |
| Accuracy | 100.0% | 100.0% | +0.0 pt [+0.0, +0.0] |
| Top-3 | 100.0% | 100.0% | +0.0 pt |
| Macro-F1 | 1.000 | 1.000 | +0.000 |
| Aicme acc | 100.0% | 100.0% | +0.0 pt |
| Mean conf | 100.0% | 100.0% | -0.0 pt |
| NLL | 0.000 | 0.000 | +0.000 [+0.000, +0.001] |
| ECE | 0.000 | 0.000 | +0.000 |
| Synthetic val acc | 99.0% | 98.3% | +0.7 pt |
| Synth→real gap (val acc − dev acc) | -1.0% | -1.7% | +0.7 pt |
| Dev-stress acc (supplementary, DR-022) | 100.0% (n=1) | 100.0% (n=1) | +0.0 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E2c | 100% | 100% | 100% | 100% | 100% | 100% |
| E2b | 100% | 100% | 100% | 100% | 100% | 100% |

**Blank-page control:** F (0.33)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E2c | 0.0% | 0.0% |
| E2b | 0.0% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E2c | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E2b | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Figures:** [curves](results/figures/E2c/learning_curves.png) · [confusion](results/figures/E2c/confusion_dev.png) · [taxonomy](results/figures/E2c/taxonomy_dev.png) · [contact sheet](results/figures/E2c/contact_sheet.png)
