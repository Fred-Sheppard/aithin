**Probe:** hybrid | **input:** render | **dataset:** `data/e3c_n850_s0` (850/class) | **seeds:** 3 | **steps used:** 8000 (best at 7500) | **train time:** 18.0 min/run

| Metric (dev, pen-40) | E3c (n=3) | E3a (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 0.0% ± 0.0 | 0.0% | -0.0 pt [-0.0, -0.0] |
| Accuracy | 100.0% ± 0.0 | 100.0% | +0.0 pt [+0.0, +0.0] |
| Top-3 | 100.0% ± 0.0 | 100.0% | +0.0 pt |
| Macro-F1 | 1.000 ± 0.000 | 1.000 | +0.000 |
| Aicme acc | 100.0% ± 0.0 | 100.0% | +0.0 pt |
| Mean conf | 99.6% ± 0.1 | 100.0% | -0.4 pt |
| NLL | 0.004 ± 0.001 | 0.000 | +0.004 [+0.001, +0.010] |
| ECE | 0.004 ± 0.001 | 0.000 | +0.004 |
| Synthetic val acc | 98.8% ± 0.1 | 100.0% | -1.2 pt |
| Synth→real gap (val acc − dev acc) | -1.2% ± 0.1 | 0.0% | -1.2 pt |
| Dev-stress acc (supplementary, DR-022) | 85.4% (n=3) | 84.2% (n=1) | +1.2 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E3c | 100% | 100% | 100% | 100% | 100% | 12% |
| E3a | 100% | 100% | 100% | 100% | 100% | 5% |

**Blank-page control:** B (0.29), B (0.32), H (0.26)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E3c | 0.0% | 0.0% |
| E3a | 0.0% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E3c | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E3a | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Figures:** [curves](results/figures/E3c/learning_curves.png) · [confusion](results/figures/E3c/confusion_dev.png) · [taxonomy](results/figures/E3c/taxonomy_dev.png) · [contact sheet](results/figures/E3c/contact_sheet.png)
