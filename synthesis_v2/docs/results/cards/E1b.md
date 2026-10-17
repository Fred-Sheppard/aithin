**Probe:** hybrid | **input:** canonical | **dataset:** `data/e1b_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 4500 (best at 2500) | **train time:** 9.9 min/run

| Metric (dev, pen-40) | E1b (n=1) | E1a (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 0.0% | 7.5% | -7.5 pt [-15.0, -0.0] |
| Accuracy | 100.0% | 92.5% | +7.5 pt [+0.0, +15.0] |
| Top-3 | 100.0% | 100.0% | +0.0 pt |
| Macro-F1 | 1.000 | 0.922 | +0.078 |
| Aicme acc | 100.0% | 100.0% | +0.0 pt |
| Mean conf | 100.0% | 99.0% | +1.0 pt |
| NLL | 0.000 | 0.585 | -0.585 [-1.263, -0.013] |
| ECE | 0.000 | 0.082 | -0.081 |
| Synthetic val acc | 99.8% | 99.8% | +0.0 pt |
| Synth→real gap (val acc − dev acc) | -0.2% | 7.3% | -7.5 pt |

**Blank-page control:** H (0.93)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E1b | 0.0% | 0.0% |
| E1a | 15.0% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E1b | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E1a | 0.0 | 0.0 | 0.0 | 3.0 | 0.0 |

**Figures:** [curves](results/figures/E1b/learning_curves.png) · [confusion](results/figures/E1b/confusion_dev.png) · [taxonomy](results/figures/E1b/taxonomy_dev.png) · [contact sheet](results/figures/E1b/contact_sheet.png)
