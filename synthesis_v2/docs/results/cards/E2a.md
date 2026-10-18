**Probe:** hybrid | **input:** canonical | **dataset:** `data/e2a_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 7000 (best at 5000) | **train time:** 15.0 min/run

| Metric (dev, pen-40) | E2a (n=1) | E1d (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 0.0% | 0.0% | -0.0 pt [-0.0, -0.0] |
| Accuracy | 100.0% | 100.0% | +0.0 pt [+0.0, +0.0] |
| Top-3 | 100.0% | 100.0% | +0.0 pt |
| Macro-F1 | 1.000 | 1.000 | +0.000 |
| Aicme acc | 100.0% | 100.0% | +0.0 pt |
| Mean conf | 100.0% | 100.0% | +0.0 pt |
| NLL | 0.000 | 0.000 | -0.000 [-0.000, +0.000] |
| ECE | 0.000 | 0.000 | -0.000 |
| Synthetic val acc | 98.1% | 99.1% | -1.0 pt |
| Synth→real gap (val acc − dev acc) | -1.9% | -0.9% | -1.0 pt |
| Dev-stress acc (supplementary, DR-022) | 97.9% (n=1) | 98.3% (n=1) | -0.4 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E2a | 100% | 100% | 100% | 100% | 100% | 88% |
| E1d | 100% | 100% | 100% | 100% | 100% | 90% |

**Blank-page control:** I (0.30)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E2a | 0.0% | 0.0% |
| E1d | 0.0% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E2a | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E1d | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Figures:** [curves](results/figures/E2a/learning_curves.png) · [confusion](results/figures/E2a/confusion_dev.png) · [taxonomy](results/figures/E2a/taxonomy_dev.png) · [contact sheet](results/figures/E2a/contact_sheet.png)
