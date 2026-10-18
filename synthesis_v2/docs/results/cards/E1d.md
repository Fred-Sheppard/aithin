**Probe:** hybrid | **input:** canonical | **dataset:** `data/e1d_n850_s0` (850/class) | **seeds:** 3 | **steps used:** 7167 (best at 5167) | **train time:** 17.3 min/run

| Metric (dev, pen-40) | E1d (n=3) | E1c (n=3) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 0.0% ± 0.0 | 0.0% ± 0.0 | -0.0 pt [-0.0, -0.0] |
| Accuracy | 100.0% ± 0.0 | 100.0% ± 0.0 | +0.0 pt [+0.0, +0.0] |
| Top-3 | 100.0% ± 0.0 | 100.0% ± 0.0 | +0.0 pt |
| Macro-F1 | 1.000 ± 0.000 | 1.000 ± 0.000 | +0.000 |
| Aicme acc | 100.0% ± 0.0 | 100.0% ± 0.0 | +0.0 pt |
| Mean conf | 100.0% ± 0.0 | 100.0% ± 0.0 | -0.0 pt |
| NLL | 0.000 ± 0.000 | 0.000 ± 0.000 | +0.000 [+0.000, +0.000] |
| ECE | 0.000 ± 0.000 | 0.000 ± 0.000 | +0.000 |
| Synthetic val acc | 99.2% ± 0.1 | 100.0% ± 0.0 | -0.7 pt |
| Synth→real gap (val acc − dev acc) | -0.8% ± 0.1 | -0.0% ± 0.0 | -0.7 pt |
| Dev-stress acc (supplementary, DR-022) | 95.7% (n=3) | 96.5% (n=3) | -0.8 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E1d | 100% | 100% | 100% | 100% | 100% | 74% |
| E1c | 100% | 100% | 100% | 100% | 100% | 79% |

**Blank-page control:** H (0.46), H (0.95), H (0.65)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E1d | 0.0% | 0.0% |
| E1c | 0.0% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E1d | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E1c | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Figures:** [curves](results/figures/E1d/learning_curves.png) · [confusion](results/figures/E1d/confusion_dev.png) · [taxonomy](results/figures/E1d/taxonomy_dev.png) · [contact sheet](results/figures/E1d/contact_sheet.png)
