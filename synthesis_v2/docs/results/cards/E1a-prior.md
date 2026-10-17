**Probe:** hybrid | **input:** canonical | **dataset:** `data/e1a_prior_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 5500 (best at 3500) | **train time:** 16.2 min/run

| Metric (dev, pen-40) | E1a-prior (n=1) | E0b (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 5.0% | 32.5% | -27.5 pt [-42.5, -12.5] |
| Accuracy | 95.0% | 67.5% | +27.5 pt [+12.5, +42.5] |
| Top-3 | 100.0% | 95.0% | +5.0 pt |
| Macro-F1 | 0.947 | 0.636 | +0.311 |
| Aicme acc | 100.0% | 90.0% | +10.0 pt |
| Mean conf | 99.5% | 97.1% | +2.4 pt |
| NLL | 0.113 | 2.519 | -2.406 [-3.987, -1.040] |
| ECE | 0.045 | 0.296 | -0.251 |
| Synthetic val acc | 99.9% | 98.9% | +1.0 pt |
| Synth→real gap (val acc − dev acc) | 4.9% | 31.4% | -26.5 pt |
| Dev-stress acc (supplementary, DR-022) | 88.3% (n=1) | 66.2% (n=1) | +22.1 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E1a-prior | 92% | 98% | 98% | 100% | 95% | 48% |
| E0b | 70% | 68% | 68% | 72% | 70% | 50% |

**Blank-page control:** A (0.75)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E1a-prior | 10.0% | 0.0% |
| E0b | 30.0% | 35.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E1a-prior | 0.0 | 0.0 | 0.0 | 2.0 | 0.0 |
| E0b | 3.0 | 0.0 | 1.0 | 9.0 | 0.0 |

**Figures:** [curves](results/figures/E1a-prior/learning_curves.png) · [confusion](results/figures/E1a-prior/confusion_dev.png) · [taxonomy](results/figures/E1a-prior/taxonomy_dev.png) · [contact sheet](results/figures/E1a-prior/contact_sheet.png)
