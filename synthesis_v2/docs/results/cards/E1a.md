**Probe:** hybrid | **input:** canonical | **dataset:** `data/e1a_n850_s0` (850/class) | **seeds:** 3 | **steps used:** 6000 (best at 4000) | **train time:** 13.0 min/run

| Metric (dev, pen-40) | E1a (n=3) | E1a-prior (n=3) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 4.2% ± 3.1 | 1.7% ± 2.4 | +2.5 pt [-0.0, +5.8] |
| Accuracy | 95.8% ± 3.1 | 98.3% ± 2.4 | -2.5 pt [-5.8, +0.0] |
| Top-3 | 100.0% ± 0.0 | 100.0% ± 0.0 | +0.0 pt |
| Macro-F1 | 0.957 ± 0.032 | 0.982 ± 0.025 | -0.025 |
| Aicme acc | 100.0% ± 0.0 | 100.0% ± 0.0 | +0.0 pt |
| Mean conf | 99.4% ± 0.4 | 99.8% ± 0.2 | -0.4 pt |
| NLL | 0.264 ± 0.242 | 0.038 ± 0.053 | +0.226 [+0.053, +0.431] |
| ECE | 0.041 ± 0.033 | 0.015 ± 0.021 | +0.026 |
| Synthetic val acc | 99.8% ± 0.1 | 99.8% ± 0.0 | -0.0 pt |
| Synth→real gap (val acc − dev acc) | 4.0% ± 3.1 | 1.5% ± 2.4 | +2.5 pt |
| Dev-stress acc (supplementary, DR-022) | 87.9% (n=3) | 89.6% (n=3) | -1.7 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E1a | 94% | 96% | 95% | 97% | 96% | 50% |
| E1a-prior | 95% | 98% | 99% | 100% | 98% | 47% |

**Blank-page control:** B (0.95), A (0.86), H (1.00)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E1a | 8.3% | 0.0% |
| E1a-prior | 3.3% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E1a | 0.0 | 0.0 | 0.0 | 1.7 | 0.0 |
| E1a-prior | 0.0 | 0.0 | 0.0 | 0.7 | 0.0 |

**Figures:** [curves](results/figures/E1a/learning_curves.png) · [confusion](results/figures/E1a/confusion_dev.png) · [taxonomy](results/figures/E1a/taxonomy_dev.png) · [contact sheet](results/figures/E1a/contact_sheet.png)
