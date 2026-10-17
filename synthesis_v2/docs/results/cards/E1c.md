**Probe:** hybrid | **input:** canonical | **dataset:** `data/e1c_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 8000 (best at 8000) | **train time:** 18.4 min/run

| Metric (dev, pen-40) | E1c (n=1) | E1b (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 0.0% | 0.0% | -0.0 pt [-0.0, -0.0] |
| Accuracy | 100.0% | 100.0% | +0.0 pt [+0.0, +0.0] |
| Top-3 | 100.0% | 100.0% | +0.0 pt |
| Macro-F1 | 1.000 | 1.000 | +0.000 |
| Aicme acc | 100.0% | 100.0% | +0.0 pt |
| Mean conf | 100.0% | 100.0% | +0.0 pt |
| NLL | 0.000 | 0.000 | -0.000 [-0.000, -0.000] |
| ECE | 0.000 | 0.000 | -0.000 |
| Synthetic val acc | 100.0% | 99.8% | +0.2 pt |
| Synth→real gap (val acc − dev acc) | 0.0% | -0.2% | +0.2 pt |

**Blank-page control:** H (1.00)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E1c | 0.0% | 0.0% |
| E1b | 0.0% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E1c | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E1b | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Figures:** [curves](results/figures/E1c/learning_curves.png) · [confusion](results/figures/E1c/confusion_dev.png) · [taxonomy](results/figures/E1c/taxonomy_dev.png) · [contact sheet](results/figures/E1c/contact_sheet.png)
