**Probe:** hybrid | **input:** canonical | **dataset:** `data/e1d_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 6500 (best at 4500) | **train time:** 19.7 min/run

| Metric (dev, pen-40) | E1d (n=1) | E1c (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 0.0% | 0.0% | -0.0 pt [-0.0, -0.0] |
| Accuracy | 100.0% | 100.0% | +0.0 pt [+0.0, +0.0] |
| Top-3 | 100.0% | 100.0% | +0.0 pt |
| Macro-F1 | 1.000 | 1.000 | +0.000 |
| Aicme acc | 100.0% | 100.0% | +0.0 pt |
| Mean conf | 100.0% | 100.0% | -0.0 pt |
| NLL | 0.000 | 0.000 | +0.000 [+0.000, +0.000] |
| ECE | 0.000 | 0.000 | +0.000 |
| Synthetic val acc | 99.1% | 100.0% | -0.9 pt |
| Synth→real gap (val acc − dev acc) | -0.9% | 0.0% | -0.9 pt |
| Dev-stress acc (supplementary, DR-022) | 98.3% (n=1) | 97.1% (n=1) | +1.2 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E1d | 100% | 100% | 100% | 100% | 100% | 90% |
| E1c | 100% | 100% | 100% | 100% | 100% | 82% |

**Blank-page control:** H (0.46)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E1d | 0.0% | 0.0% |
| E1c | 0.0% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E1d | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E1c | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Figures:** [curves](results/figures/E1d/learning_curves.png) · [confusion](results/figures/E1d/confusion_dev.png) · [taxonomy](results/figures/E1d/taxonomy_dev.png) · [contact sheet](results/figures/E1d/contact_sheet.png)
