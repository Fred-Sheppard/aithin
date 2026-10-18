**Probe:** hybrid | **input:** canonical | **dataset:** `data/e2c_prior_n850_s0` (850/class) | **seeds:** 3 | **steps used:** 8000 (best at 7000) | **train time:** 18.8 min/run

| Metric (dev, pen-40) | E2c-prior (n=3) | E2c (n=3) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 0.0% ± 0.0 | 0.0% ± 0.0 | -0.0 pt [-0.0, -0.0] |
| Accuracy | 100.0% ± 0.0 | 100.0% ± 0.0 | +0.0 pt [+0.0, +0.0] |
| Top-3 | 100.0% ± 0.0 | 100.0% ± 0.0 | +0.0 pt |
| Macro-F1 | 1.000 ± 0.000 | 1.000 ± 0.000 | +0.000 |
| Aicme acc | 100.0% ± 0.0 | 100.0% ± 0.0 | +0.0 pt |
| Mean conf | 100.0% ± 0.0 | 100.0% ± 0.0 | +0.0 pt |
| NLL | 0.000 ± 0.000 | 0.000 ± 0.000 | -0.000 [-0.001, -0.000] |
| ECE | 0.000 ± 0.000 | 0.000 ± 0.000 | -0.000 |
| Synthetic val acc | 98.3% ± 0.3 | 99.0% ± 0.1 | -0.7 pt |
| Synth→real gap (val acc − dev acc) | -1.7% ± 0.3 | -1.0% ± 0.1 | -0.7 pt |
| Dev-stress acc (supplementary, DR-022) | 99.9% (n=3) | 100.0% (n=3) | -0.1 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E2c-prior | 100% | 100% | 100% | 100% | 100% | 99% |
| E2c | 100% | 100% | 100% | 100% | 100% | 100% |

**Blank-page control:** M (0.62), M (0.70), M (0.60)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E2c-prior | 0.0% | 0.0% |
| E2c | 0.0% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E2c-prior | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E2c | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Figures:** [curves](results/figures/E2c-prior/learning_curves.png) · [confusion](results/figures/E2c-prior/confusion_dev.png) · [taxonomy](results/figures/E2c-prior/taxonomy_dev.png)
