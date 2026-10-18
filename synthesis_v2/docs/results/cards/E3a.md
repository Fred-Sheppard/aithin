**Probe:** hybrid | **input:** render | **dataset:** `data/e3a_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 6000 (best at 4000) | **train time:** 12.9 min/run

| Metric (dev, pen-40) | E3a (n=1) | E1d (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 0.0% | 0.0% | -0.0 pt [-0.0, -0.0] |
| Accuracy | 100.0% | 100.0% | +0.0 pt [+0.0, +0.0] |
| Top-3 | 100.0% | 100.0% | +0.0 pt |
| Macro-F1 | 1.000 | 1.000 | +0.000 |
| Aicme acc | 100.0% | 100.0% | +0.0 pt |
| Mean conf | 100.0% | 100.0% | +0.0 pt |
| NLL | 0.000 | 0.000 | -0.000 [-0.000, -0.000] |
| ECE | 0.000 | 0.000 | -0.000 |
| Synthetic val acc | 100.0% | 99.1% | +0.9 pt |
| Synth→real gap (val acc − dev acc) | 0.0% | -0.9% | +0.9 pt |
| Dev-stress acc (supplementary, DR-022) | 84.2% (n=1) | 98.3% (n=1) | -14.2 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E3a | 100% | 100% | 100% | 100% | 100% | 5% |
| E1d | 100% | 100% | 100% | 100% | 100% | 90% |

**Blank-page control:** B (0.53)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E3a | 0.0% | 0.0% |
| E1d | 0.0% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E3a | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E1d | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Figures:** [curves](results/figures/E3a/learning_curves.png) · [confusion](results/figures/E3a/confusion_dev.png) · [taxonomy](results/figures/E3a/taxonomy_dev.png) · [contact sheet](results/figures/E3a/contact_sheet.png)
