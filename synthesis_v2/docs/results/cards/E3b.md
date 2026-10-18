**Probe:** hybrid | **input:** render | **dataset:** `data/e3b_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 8000 (best at 6500) | **train time:** 17.2 min/run

| Metric (dev, pen-40) | E3b (n=1) | E3a (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 0.0% | 0.0% | -0.0 pt [-0.0, -0.0] |
| Accuracy | 100.0% | 100.0% | +0.0 pt [+0.0, +0.0] |
| Top-3 | 100.0% | 100.0% | +0.0 pt |
| Macro-F1 | 1.000 | 1.000 | +0.000 |
| Aicme acc | 100.0% | 100.0% | +0.0 pt |
| Mean conf | 92.9% | 100.0% | -7.1 pt |
| NLL | 0.078 | 0.000 | +0.078 [+0.051, +0.113] |
| ECE | 0.071 | 0.000 | +0.071 |
| Synthetic val acc | 89.4% | 100.0% | -10.6 pt |
| Synth→real gap (val acc − dev acc) | -10.6% | 0.0% | -10.6 pt |
| Dev-stress acc (supplementary, DR-022) | 84.6% (n=1) | 84.2% (n=1) | +0.4 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E3b | 100% | 100% | 100% | 100% | 100% | 8% |
| E3a | 100% | 100% | 100% | 100% | 100% | 5% |

**Blank-page control:** A (0.28)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E3b | 0.0% | 0.0% |
| E3a | 0.0% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E3b | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E3a | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Figures:** [curves](results/figures/E3b/learning_curves.png) · [confusion](results/figures/E3b/confusion_dev.png) · [taxonomy](results/figures/E3b/taxonomy_dev.png) · [contact sheet](results/figures/E3b/contact_sheet.png)
