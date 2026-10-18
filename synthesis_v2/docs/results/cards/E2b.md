**Probe:** hybrid | **input:** canonical | **dataset:** `data/e2b_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 7500 (best at 5500) | **train time:** 16.1 min/run

| Metric (dev, pen-40) | E2b (n=1) | E2a (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 0.0% | 0.0% | -0.0 pt [-0.0, -0.0] |
| Accuracy | 100.0% | 100.0% | +0.0 pt [+0.0, +0.0] |
| Top-3 | 100.0% | 100.0% | +0.0 pt |
| Macro-F1 | 1.000 | 1.000 | +0.000 |
| Aicme acc | 100.0% | 100.0% | +0.0 pt |
| Mean conf | 100.0% | 100.0% | -0.0 pt |
| NLL | 0.000 | 0.000 | +0.000 [-0.000, +0.000] |
| ECE | 0.000 | 0.000 | +0.000 |
| Synthetic val acc | 98.3% | 98.1% | +0.2 pt |
| Synth→real gap (val acc − dev acc) | -1.7% | -1.9% | +0.2 pt |
| Dev-stress acc (supplementary, DR-022) | 100.0% (n=1) | 97.9% (n=1) | +2.1 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E2b | 100% | 100% | 100% | 100% | 100% | 100% |
| E2a | 100% | 100% | 100% | 100% | 100% | 88% |

**Blank-page control:** H (0.32)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E2b | 0.0% | 0.0% |
| E2a | 0.0% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E2b | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E2a | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Figures:** [curves](results/figures/E2b/learning_curves.png) · [confusion](results/figures/E2b/confusion_dev.png) · [taxonomy](results/figures/E2b/taxonomy_dev.png) · [contact sheet](results/figures/E2b/contact_sheet.png)
