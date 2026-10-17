**Probe:** hybrid | **input:** canonical | **dataset:** `data/e1a_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 5000 (best at 3000) | **train time:** 10.7 min/run

| Metric (dev, pen-40) | E1a (n=1) | E1a-prior (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 7.5% | 5.0% | +2.5 pt [-0.0, +7.5] |
| Accuracy | 92.5% | 95.0% | -2.5 pt [-7.5, +0.0] |
| Top-3 | 100.0% | 100.0% | +0.0 pt |
| Macro-F1 | 0.922 | 0.947 | -0.025 |
| Aicme acc | 100.0% | 100.0% | +0.0 pt |
| Mean conf | 99.0% | 99.5% | -0.5 pt |
| NLL | 0.585 | 0.113 | +0.472 [+0.013, +1.004] |
| ECE | 0.082 | 0.045 | +0.037 |
| Synthetic val acc | 99.8% | 99.9% | -0.1 pt |
| Synth→real gap (val acc − dev acc) | 7.3% | 4.9% | +2.4 pt |
| Dev-stress acc (supplementary, DR-022) | 85.4% (n=1) | 88.3% (n=1) | -2.9 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E1a | 92% | 95% | 90% | 95% | 92% | 48% |
| E1a-prior | 92% | 98% | 98% | 100% | 95% | 48% |

**Blank-page control:** B (0.95)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E1a | 15.0% | 0.0% |
| E1a-prior | 10.0% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E1a | 0.0 | 0.0 | 0.0 | 3.0 | 0.0 |
| E1a-prior | 0.0 | 0.0 | 0.0 | 2.0 | 0.0 |

**Figures:** [curves](results/figures/E1a/learning_curves.png) · [confusion](results/figures/E1a/confusion_dev.png) · [taxonomy](results/figures/E1a/taxonomy_dev.png) · [contact sheet](results/figures/E1a/contact_sheet.png)
