**Probe:** hybrid | **input:** canonical | **dataset:** `data/e1c_n850_s0` (850/class) | **seeds:** 3 | **steps used:** 7167 (best at 6500) | **train time:** 15.9 min/run

| Metric (dev, pen-40) | E1c (n=3) | E1b (n=3) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 0.0% ± 0.0 | 2.5% ± 2.0 | -2.5 pt [-5.8, -0.0] |
| Accuracy | 100.0% ± 0.0 | 97.5% ± 2.0 | +2.5 pt [+0.0, +5.8] |
| Top-3 | 100.0% ± 0.0 | 100.0% ± 0.0 | +0.0 pt |
| Macro-F1 | 1.000 ± 0.000 | 0.973 ± 0.022 | +0.027 |
| Aicme acc | 100.0% ± 0.0 | 100.0% ± 0.0 | +0.0 pt |
| Mean conf | 100.0% ± 0.0 | 99.2% ± 0.6 | +0.8 pt |
| NLL | 0.000 ± 0.000 | 0.052 ± 0.048 | -0.052 [-0.114, -0.004] |
| ECE | 0.000 ± 0.000 | 0.024 ± 0.020 | -0.024 |
| Synthetic val acc | 100.0% ± 0.0 | 99.9% ± 0.0 | +0.1 pt |
| Synth→real gap (val acc − dev acc) | -0.0% ± 0.0 | 2.4% ± 2.1 | -2.4 pt |
| Dev-stress acc (supplementary, DR-022) | 96.5% (n=3) | 87.9% (n=3) | +8.6 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E1c | 100% | 100% | 100% | 100% | 100% | 79% |
| E1b | 87% | 92% | 98% | 99% | 98% | 54% |

**Blank-page control:** H (1.00), H (1.00), B (1.00)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E1c | 0.0% | 0.0% |
| E1b | 5.0% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E1c | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E1b | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |

**Figures:** [curves](results/figures/E1c/learning_curves.png) · [confusion](results/figures/E1c/confusion_dev.png) · [taxonomy](results/figures/E1c/taxonomy_dev.png) · [contact sheet](results/figures/E1c/contact_sheet.png)
