**Probe:** hybrid | **input:** canonical | **dataset:** `data/e1a_prior_n850_s0` (850/class) | **seeds:** 3 | **steps used:** 6000 (best at 4000) | **train time:** 14.3 min/run

| Metric (dev, pen-40) | E1a-prior (n=3) | E0b (n=2) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 1.7% ± 2.4 | 28.7% ± 3.7 | -27.1 pt [-40.4, -14.2] |
| Accuracy | 98.3% ± 2.4 | 71.2% ± 3.7 | +27.1 pt [+14.2, +40.4] |
| Top-3 | 100.0% ± 0.0 | 96.2% ± 1.3 | +3.8 pt |
| Macro-F1 | 0.982 ± 0.025 | 0.670 ± 0.034 | +0.313 |
| Aicme acc | 100.0% ± 0.0 | 92.5% ± 2.5 | +7.5 pt |
| Mean conf | 99.8% ± 0.2 | 97.5% ± 0.4 | +2.3 pt |
| NLL | 0.038 ± 0.053 | 2.299 ± 0.220 | -2.261 [-3.638, -1.058] |
| ECE | 0.015 ± 0.021 | 0.277 ± 0.019 | -0.262 |
| Synthetic val acc | 99.8% ± 0.0 | 99.1% ± 0.1 | +0.8 pt |
| Synth→real gap (val acc − dev acc) | 1.5% ± 2.4 | 27.8% ± 3.6 | -26.3 pt |
| Dev-stress acc (supplementary, DR-022) | 89.6% (n=3) | 67.3% (n=2) | +22.3 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E1a-prior | 95% | 98% | 99% | 100% | 98% | 47% |
| E0b | 71% | 71% | 70% | 70% | 72% | 49% |

**Blank-page control:** A (0.75), B (0.59), B (0.83)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E1a-prior | 3.3% | 0.0% |
| E0b | 27.5% | 30.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E1a-prior | 0.0 | 0.0 | 0.0 | 0.7 | 0.0 |
| E0b | 2.5 | 0.0 | 0.5 | 8.5 | 0.0 |

**Figures:** [curves](results/figures/E1a-prior/learning_curves.png) · [confusion](results/figures/E1a-prior/confusion_dev.png) · [taxonomy](results/figures/E1a-prior/taxonomy_dev.png) · [contact sheet](results/figures/E1a-prior/contact_sheet.png)
