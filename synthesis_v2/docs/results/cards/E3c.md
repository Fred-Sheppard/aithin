**Probe:** hybrid | **input:** render | **dataset:** `data/e3c_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 8000 (best at 7500) | **train time:** 17.1 min/run

| Metric (dev, pen-40) | E3c (n=1) | E3a (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 0.0% | 0.0% | -0.0 pt [-0.0, -0.0] |
| Accuracy | 100.0% | 100.0% | +0.0 pt [+0.0, +0.0] |
| Top-3 | 100.0% | 100.0% | +0.0 pt |
| Macro-F1 | 1.000 | 1.000 | +0.000 |
| Aicme acc | 100.0% | 100.0% | +0.0 pt |
| Mean conf | 99.5% | 100.0% | -0.5 pt |
| NLL | 0.006 | 0.000 | +0.005 [+0.001, +0.014] |
| ECE | 0.005 | 0.000 | +0.005 |
| Synthetic val acc | 98.7% | 100.0% | -1.3 pt |
| Synth→real gap (val acc − dev acc) | -1.3% | 0.0% | -1.3 pt |
| Dev-stress acc (supplementary, DR-022) | 85.0% (n=1) | 84.2% (n=1) | +0.8 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E3c | 100% | 100% | 100% | 100% | 100% | 10% |
| E3a | 100% | 100% | 100% | 100% | 100% | 5% |

**Blank-page control:** B (0.29)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E3c | 0.0% | 0.0% |
| E3a | 0.0% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E3c | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E3a | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Figures:** [curves](results/figures/E3c/learning_curves.png) · [confusion](results/figures/E3c/confusion_dev.png) · [taxonomy](results/figures/E3c/taxonomy_dev.png) · [contact sheet](results/figures/E3c/contact_sheet.png)
