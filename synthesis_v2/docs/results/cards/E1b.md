**Probe:** hybrid | **input:** canonical | **dataset:** `data/e1b_n850_s0` (850/class) | **seeds:** 3 | **steps used:** 4333 (best at 2333) | **train time:** 9.5 min/run

| Metric (dev, pen-40) | E1b (n=3) | E1a (n=3) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 2.5% ± 2.0 | 4.2% ± 3.1 | -1.7 pt [-5.0, +1.7] |
| Accuracy | 97.5% ± 2.0 | 95.8% ± 3.1 | +1.7 pt [-1.7, +5.0] |
| Top-3 | 100.0% ± 0.0 | 100.0% ± 0.0 | +0.0 pt |
| Macro-F1 | 0.973 ± 0.022 | 0.957 ± 0.032 | +0.016 |
| Aicme acc | 100.0% ± 0.0 | 100.0% ± 0.0 | +0.0 pt |
| Mean conf | 99.2% ± 0.6 | 99.4% ± 0.4 | -0.1 pt |
| NLL | 0.052 ± 0.048 | 0.264 ± 0.242 | -0.211 [-0.412, -0.040] |
| ECE | 0.024 ± 0.020 | 0.041 ± 0.033 | -0.017 |
| Synthetic val acc | 99.9% ± 0.0 | 99.8% ± 0.1 | +0.1 pt |
| Synth→real gap (val acc − dev acc) | 2.4% ± 2.1 | 4.0% ± 3.1 | -1.6 pt |
| Dev-stress acc (supplementary, DR-022) | 87.9% (n=3) | 87.9% (n=3) | +0.0 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E1b | 87% | 92% | 98% | 99% | 98% | 54% |
| E1a | 94% | 96% | 95% | 97% | 96% | 50% |

**Blank-page control:** H (0.93), H (0.77), H (0.86)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E1b | 5.0% | 0.0% |
| E1a | 8.3% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E1b | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| E1a | 0.0 | 0.0 | 0.0 | 1.7 | 0.0 |

**Figures:** [curves](results/figures/E1b/learning_curves.png) · [confusion](results/figures/E1b/confusion_dev.png) · [taxonomy](results/figures/E1b/taxonomy_dev.png) · [contact sheet](results/figures/E1b/contact_sheet.png)
