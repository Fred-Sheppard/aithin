# Experiment log: Synthesis v2

There is one card per experiment, in chronological order. The rules:
- The **Motivation, Hypothesis and Change** sections are written *before* the run.
- **Results** are filled in by `train/report.py --exp <id>`.
- **Decision** is written after reviewing the results, using the rule in DR-004.

The protocol is in [DECISIONS.md](DECISIONS.md):
- dev = pen-40;
- test = marker-40, evaluated once;
- early stopping on synthetic validation only;
- 1 seed for screening, 3 seeds to confirm.

## Card template

```
## <ID>: <title>
- Parent: <ID> | Date: <yyyy-mm-dd> | Git: <sha> | Config: configs/<id>.yaml
### Motivation
<report section + observed failure (taxonomy row) this targets>
### Hypothesis
<expected effect; which metric should move and in which direction>
### Change
<config diff vs parent>
### Setup
<n/class, steps used, seeds, probe, eval sets>
### Results
<auto: dev table vs parent with Δ and 95% CI; per-batch; taxonomy; synth-val acc; gap>
### Figures
<auto: contact sheet, confusion matrix, learning curves>
### Decision
ADOPTED | REJECTED | INCONCLUSIVE: <rationale>
### Notes
```

---

## E0-ckpt: re-evaluating the frozen legacy checkpoints (harness check + preprocessing study)
- Parent: none | Date: 2026-10-17 | Config: none (existing checkpoints `best_ogham_hybrid_model_save.h5`, `best_ogham_augmented_model.h5`)

### Motivation
Before any new experiment is trusted, the evaluation harness has to reproduce the team's reported wild-test numbers. We also suspected that the legacy loader (a nearest-neighbour squash of 1000–4000 px images to 128 px) was erasing thin strokes and greying the ink (plan: "preprocessing confound").

### Hypothesis
1. The legacy loader reproduces the reported Hybrid score of 17/40 and the Augmented score of 6/40.
2. Better preprocessing improves real accuracy without any retraining.

### Change
No training. The same checkpoints were scored under three preprocessing modes:
- `legacy`: keras `load_img`, grayscale, nearest squash;
- `autocontrast`: the teammate's branch `image-preprocessing-investigation`, which applies autocontrast at 1% and then a nearest squash;
- `canonical`: DR-005. Autocontrast, ink-bbox crop, square pad and INTER_AREA resize.

### Results
Note: test (marker-40) is reported here only because these are frozen legacy models evaluated for continuity with the team's table. No synthesis decision is made from these test numbers.

| Checkpoint | Preprocessing | dev acc | dev aicme | dev top-3 | dev NLL | test acc | test aicme | test top-3 | Blank-page prediction |
|---|---|---|---|---|---|---|---|---|---|
| Hybrid | legacy | 47.5% | 67.5% | 70.0% | 7.45 | **40.0%** (16/40) | 57.5% | 60.0% | H (0.46) |
| Hybrid | autocontrast | 82.5% | 97.5% | 97.5% | 2.28 | 42.5% | 97.5% | 82.5% | **R (1.00)** |
| Hybrid | canonical | 82.5% | 92.5% | 95.0% | 2.84 | 52.5% | 100% | 90.0% | H (0.45) |
| Augmented | legacy | 10.0% | 35.0% | 37.5% | 10.83 | **15.0%** (6/40) | 45.0% | 40.0% | B (0.92) |
| Augmented | autocontrast | 65.0% | 87.5% | 82.5% | 3.25 | 77.5% | 87.5% | 95.0% | U (0.71) |
| Augmented | canonical | 62.5% | 75.0% | 77.5% | 4.06 | 75.0% | 87.5% | 92.5% | B (0.88) |

**Harness reproduction against the reported table:**
- **Augmented reproduces exactly:** 6/40, CER 85.0%, aicme 45.0%.
- **Hybrid is within one image:** 16/40 against the reported 17/40, with aicme 57.5% against 60.0% and top-3 60.0% against 62.5%.
- Our mean confidence is a few points higher than reported for both models (95.1% against 90.6%, and 91.3% against 88.7%).

The most likely cause of the difference is a slightly different checkpoint file or resize path in the team's script, which is not in the repo. That is within one image, so the harness is accepted (DR-015).

**Error taxonomy for Hybrid with the legacy loader, on test:**
- 16 correct;
- 5 M↔A (diagonal vs across) errors;
- 12 other cross-group errors;
- 7 errors with the right group but an off-by-one stroke count.

With canonical preprocessing, Hybrid on test reaches aicme 100%. All of its remaining errors are stroke-*count* errors on thick marker strokes.

### Decision
**Finding (not an adoption).** Preprocessing alone accounts for +35 to +55 accuracy points on dev for the frozen models. Most of the reported "domain gap" was a preprocessing gap, from greyed ink and thin strokes erased by the nearest squash, rather than a synthesis gap.
- **Autocontrast followed by a nearest squash** creates a failure on blank or near-blank input: it stretches paper noise into confident predictions (blank page → R at 100%).
- **Canonical preprocessing** does not have this problem.
- **The comparison on frozen models is not fair to canonical.** Canonical crops to the ink, which changes the framing the legacy models were trained on.
- So the preprocessing choice is settled by *retraining*, on dev only: E0a (legacy), E0b (canonical) and E0c (autocontrast), all trained on the same legacy data.

### Notes
- Hybrid on marker after canonical preprocessing has 100% group accuracy but only 52.5% exact accuracy. Its remaining errors all come from counting strokes on thick marker ink, which points to the need for E1c (a pen/marker width model).
- The Augmented checkpoint generalises to marker much better than Hybrid once preprocessing is fixed (75% against 52.5%). Its zoom and shift augmentation probably adds scale robustness. This reverses the team's earlier conclusion that augmentation hurt, which came from scoring under the legacy loader.

---

## E0a: legacy generator + legacy loader (retrained in our harness)
- Parent: none | Date: 2026-10-17 | Data: `data/legacy_n850_s0`, the unmodified old generator with external seed 0, 850/class, plus a separately generated synthetic val set of 50/class
- Setup: hybrid probe, batch 32, early stopping on synthetic val (500-step checks, patience 4, cap 8k), 3 seeds

### Motivation
The team's numbers come from checkpoints trained by older scripts. E0a gives the same pipeline a baseline inside *our* protocol, so every later experiment changes exactly one thing.

### Hypothesis
Dev results will be close to the legacy Hybrid checkpoint scored with the legacy loader (47.5% dev accuracy). Seed spread should be large, because the loader erases thin pen strokes.

### Change
None. This is the reference: legacy data with the legacy loader (keras `load_img`, grayscale, nearest-neighbour squash).

### Results
<!-- results:E0a -->
**Probe:** hybrid | **input:** legacy | **dataset:** `data/legacy_n850_s0` (850/class) | **seeds:** 3 | **steps used:** 6833 (best at 4833) | **train time:** 16.1 min/run

| Metric (dev, pen-40) | E0a (n=3) |
|---|---|
| CER | 41.7% ± 7.2 |
| Accuracy | 58.3% ± 7.2 |
| Top-3 | 83.3% ± 1.2 |
| Macro-F1 | 0.530 ± 0.084 |
| Aicme acc | 75.8% ± 2.4 |
| Mean conf | 92.2% ± 3.9 |
| NLL | 3.019 ± 0.244 |
| ECE | 0.355 ± 0.042 |
| Synthetic val acc | 98.6% ± 0.1 |
| Synth→real gap (val acc − dev acc) | 40.3% ± 7.1 |
| Dev-stress acc (supplementary, DR-022) | 38.2% (n=3) |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E0a | 28% | 32% | 52% | 47% | 13% | 56% |

**Blank-page control:** H (0.79), A (0.57), A (0.36)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E0a | 53.3% | 30.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E0a | 0.0 | 5.3 | 4.3 | 7.0 | 0.0 |

**Figures:** [curves](results/figures/E0a/learning_curves.png) · [confusion](results/figures/E0a/confusion_dev.png) · [taxonomy](results/figures/E0a/taxonomy_dev.png)
<!-- /results:E0a -->

### Decision
**REFERENCE.**
- Screening seed 0: dev CER 42.5% (NLL 2.70; aicme 77.5%; pen-1 60% vs pen-2 25% CER).
- The legacy loader hurts most on the faint, loosely cropped pen-1 batch.
- Seeds 1–2 are queued (screen1), and the card will be refreshed.

---

## E0b: legacy generator + canonical preprocessing (DR-005)
- Parent: E0a | Same data, seeds and probe as E0a

### Motivation
E0-ckpt showed that preprocessing alone is worth +35 to +55 dev points on frozen checkpoints. That test is unfair to canonical, because canonical changes the framing the old models were trained on. Retraining on canonical inputs removes that confound.

### Hypothesis
Canonical preprocessing applied to *both* the synthetic and the real images cuts dev CER sharply against E0a, mostly on the pen-1 batch (faint, loose crops). The blank page stays low-confidence.

### Change
`--input canonical`, used for training and evaluation: autocontrast, ink-bbox crop that ignores border-touching components, 10% margin, square pad, INTER_AREA resize to 128.

### Results
<!-- results:E0b -->
**Probe:** hybrid | **input:** canonical | **dataset:** `data/legacy_n850_s0` (850/class) | **seeds:** 2 | **steps used:** 7500 (best at 6000) | **train time:** 17.1 min/run

| Metric (dev, pen-40) | E0b (n=2) | E0a (n=3) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 28.7% ± 3.7 | 41.7% ± 7.2 | -12.9 pt [-29.6, +5.8] |
| Accuracy | 71.2% ± 3.7 | 58.3% ± 7.2 | +12.9 pt [-5.8, +29.6] |
| Top-3 | 96.2% ± 1.3 | 83.3% ± 1.2 | +12.9 pt |
| Macro-F1 | 0.670 ± 0.034 | 0.530 ± 0.084 | +0.140 |
| Aicme acc | 92.5% ± 2.5 | 75.8% ± 2.4 | +16.7 pt |
| Mean conf | 97.5% ± 0.4 | 92.2% ± 3.9 | +5.3 pt |
| NLL | 2.299 ± 0.220 | 3.019 ± 0.244 | -0.720 [-2.311, +0.985] |
| ECE | 0.277 ± 0.019 | 0.355 ± 0.042 | -0.078 |
| Synthetic val acc | 99.1% ± 0.1 | 98.6% ± 0.1 | +0.4 pt |
| Synth→real gap (val acc − dev acc) | 27.8% ± 3.6 | 40.3% ± 7.1 | -12.5 pt |
| Dev-stress acc (supplementary, DR-022) | 67.3% (n=2) | 38.2% (n=3) | +29.1 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E0b | 71% | 71% | 70% | 70% | 72% | 49% |
| E0a | 28% | 32% | 52% | 47% | 13% | 56% |

**Blank-page control:** E (0.96), H (0.79)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E0b | 27.5% | 30.0% |
| E0a | 53.3% | 30.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E0b | 2.5 | 0.0 | 0.5 | 8.5 | 0.0 |
| E0a | 0.0 | 5.3 | 4.3 | 7.0 | 0.0 |

**Figures:** [curves](results/figures/E0b/learning_curves.png) · [confusion](results/figures/E0b/confusion_dev.png) · [taxonomy](results/figures/E0b/taxonomy_dev.png)
<!-- /results:E0b -->

### Decision
**ADOPTED (provisional, 1 seed; seeds 1–2 queued).**

Canonical preprocessing improves on the legacy loader with the same legacy data:
- CER 42.5% → 32.5% (−10 pt);
- aicme 77.5% → 90%;
- pen-1 CER 60% → 30%.

The gain is smaller than the frozen-checkpoint study suggested. The remaining errors are a systematic **under-count by one stroke** on 4–5-stroke letters: N→S, S→F, Q→C, C→T, I→E, E→U (9 of 13 errors).

Hypothesis:
- Canonical normalises scale to the ink box.
- The legacy generator places strokes 0.157 stem lengths apart; real pen strokes are 0.112 apart (median, DR-007 calibration).
- So the model learns "wide" spacing, and five real strokes look like four legacy ones.
- E1 tests this directly.

---

## E0c: legacy generator + autocontrast/nearest (the teammate's branch)
- Parent: E0a | Same data, seeds and probe as E0a

### Motivation
The team is considering the preprocessing on branch `image-preprocessing-investigation`. E0-ckpt found it as effective as canonical on dev (82.5% for the Hybrid checkpoint), but it turns a blank page into a 100%-confident R.

### Hypothesis
E0c ≈ E0b on dev CER. E0c is worse on the blank-page control, and may be worse on pen-1, where the nearest-neighbour squash can still break thin strokes.

### Change
`--input autocontrast`: autocontrast at 1%, then a nearest squash to 128.

### Results
<!-- results:E0c -->
**Probe:** hybrid | **input:** autocontrast | **dataset:** `data/legacy_n850_s0` (850/class) | **seeds:** 3 | **steps used:** 6833 (best at 4833) | **train time:** 15.6 min/run

| Metric (dev, pen-40) | E0c (n=3) | E0a (n=3) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 18.3% ± 5.9 | 41.7% ± 7.2 | -23.3 pt [-35.0, -11.7] |
| Accuracy | 81.7% ± 5.9 | 58.3% ± 7.2 | +23.3 pt [+11.7, +35.0] |
| Top-3 | 99.2% ± 1.2 | 83.3% ± 1.2 | +15.8 pt |
| Macro-F1 | 0.792 ± 0.072 | 0.530 ± 0.084 | +0.263 |
| Aicme acc | 97.5% ± 2.0 | 75.8% ± 2.4 | +21.7 pt |
| Mean conf | 96.0% ± 0.6 | 92.2% ± 3.9 | +3.9 pt |
| NLL | 0.950 ± 0.386 | 3.019 ± 0.244 | -2.069 [-3.267, -0.986] |
| ECE | 0.179 ± 0.062 | 0.355 ± 0.042 | -0.176 |
| Synthetic val acc | 98.9% ± 0.2 | 98.6% ± 0.1 | +0.2 pt |
| Synth→real gap (val acc − dev acc) | 17.2% ± 6.1 | 40.3% ± 7.1 | -23.1 pt |
| Dev-stress acc (supplementary, DR-022) | 80.1% (n=3) | 38.2% (n=3) | +41.9 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E0c | 82% | 83% | 83% | 73% | 82% | 78% |
| E0a | 28% | 32% | 52% | 47% | 13% | 56% |

**Blank-page control:** R (1.00), I (1.00), R (1.00)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E0c | 20.0% | 16.7% |
| E0a | 53.3% | 30.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E0c | 0.0 | 0.3 | 0.7 | 6.3 | 0.0 |
| E0a | 0.0 | 5.3 | 4.3 | 7.0 | 0.0 |

**Figures:** [curves](results/figures/E0c/learning_curves.png) · [confusion](results/figures/E0c/confusion_dev.png) · [taxonomy](results/figures/E0c/taxonomy_dev.png)
<!-- /results:E0c -->

### Decision
**FINDING; the preprocessing choice is deferred to a controlled test on E1-best data.**

- With legacy data, the teammate's autocontrast + nearest squash beats canonical by a wide margin: CER 10.0% vs 32.5%, NLL 0.44 vs 2.52, and better at every check during training.
- This mode keeps the legacy framing, so the legacy spacing mismatch never becomes visible. That is consistent with the E0b hypothesis.
- It still maps a blank page to a 100%-confident "R".
- Canonical stays the protocol input for the E1 chain, which keeps E1 runs comparable with E0b.
- The decisive comparison is queued:
  - **E1-best under canonical vs autocontrast** on the same images. The `_ac` caches were built in DR-020.
  - Seeds 1–2 for E0b and E0c.

---

## E3-ref: line detector + rule decoder, no learning (Phase 2 reference)
- Parent: none | Date: 2026-10-17 | Method: proxy detector v1 (DR-017) → `train/decode_rules.py`. The group comes from stroke side and angle (across strokes more than 17° from perpendicular count as diagonal); the count is the number of strokes.

### Motivation
Plan Phase 2: measure how much of the task the detector representation solves before any learning, and therefore how much the synthesis pipeline has to contribute on top.

### Hypothesis
On clean single-character crops, a geometric detector with counting rules is a strong baseline. Its errors will come from detector failures (split, merged or missed strokes), not from ambiguity.

### Results
| Split | Accuracy | CER | Macro-F1 | Aicme acc | Blank control |
|---|---|---|---|---|---|
| dev (pen-40) | **100%** (40/40) | 0.0% | 1.000 | 100% | no detection |
| test (marker-40, frozen detector, scored once) | **97.5%** (39/40) | 2.5% | 0.973 | 100% | no detection |

- **The single test error** is marker-00031, Z read as R. The detector found 5 strokes: one thick crossing marker stroke was split into a left piece and a right piece that were not paired.
- **Detector accuracy on synthetic QA renders** (200 per config, true letter known):

| Config | E1a-prior | E1a | E1b | E1c | E1d |
|---|---|---|---|---|---|
| Accuracy | 99.5% | 99.5% | 99.0% | 97.5% | 91.0% |

### Decision
**Finding.** On these images, a simple line detector with counting rules beats every CNN probe so far: the best frozen checkpoint reaches 82.5% dev and 75% test. This strongly supports the team's line-detection-first direction.

For the synthesis pipeline this means two things:
- **Probe track:** the value of synthetic data for the image-to-letter probe is still measured in E1 and E2.
- **Detector track:** synthetic data is most valuable for training a learned decoder that is robust to detector failures (E3b and E3c), and for **stress-testing the detector itself**. The hard cases of E1d drop it to 91%.

### Notes
- Mean real-dev stroke geometry was measured with this detector (Phase 1 calibration). Because it decodes all of dev correctly, every dev image contributed measurements.

---

## Generator QA for E1 (applies to the E1a-prior to E1d cards)
- **Property tests** (`tools/test_generator.py`): every config passes. Each sampled glyph satisfies the label validator, and decoding the oracle JSON with the rules recovers the true letter in 1000/1000 cases per config.
- **Generation cost:** 17k train + 1k val images take 0.4–0.6 min on 6 processes, about 3 ms per sample per core.

| Config | Raw rejection rate (top reasons) | Detector decode on QA renders | First attempt (before DR-018) |
|---|---|---|---|
| E1a-prior | 3.0% (spacing, diag angle, convergence) | 99.5% | 99.5% |
| E1a | 7.3% (spacing, convergence) | 99.5% | 99.5% |
| E1b | 11.8% (spacing, convergence) | 99.0% | 89.5% |
| E1c | 24.2% (spacing, convergence) | 97.5% | 80.5% |
| E1d | 26.4% (spacing, convergence) | 91.0% | 76.5% |

- **Calibration** (`tools/measure_real.py`, dev only): figure `results/figures/E1/calibration_e1a_initial.png` (priors) and `calibration_e1a_calibrated.png`. Compared with the priors, the real pen strokes have:
  - longer vowel strokes: 0.64–0.91 of the stem, against a prior of 0.30–0.90;
  - wider across-angle deviations: up to about 12°, against a prior of ±8°;
  - narrower diagonal angles: 26–42°, against a prior of 22–60°;
  - tighter spacing: median 0.112, against a prior of 0.08–0.20;
  - more centred stroke groups.

---

## E1a-prior: parametric stroke model with hand-set prior ranges (uncalibrated)
- Parent: E0b | Config: `configs/e1a_prior.yaml` | Data: `data/e1a_prior_n850_s0`

### Motivation
Of the legacy model's test errors, 16/23 were *group* errors. The legacy geometry also has two defects:
- left strokes are longer than right strokes, because right strokes are clipped;
- diagonals are sheared and not anchored to the stem.

The report (§2) names stroke-level synthesis in trajectory space as the core upgrade.

### Hypothesis
Correct, stem-anchored conventions raise **aicme accuracy** and lower dev CER against E0b, even with uncalibrated ranges.

### Change
- Quick, Draw! polylines are replaced by a parametric stroke model in stem coordinates, with a label-preserving validator.
- The pen stays constant and legacy-like: width 0.015–0.025 of the stem, black ink.
- Prior ranges: spacing 0.08–0.20, side length 0.25–0.70, across length 0.30–0.90, diagonal angle 22–60°, tilt sd 5°.

### Results
<!-- results:E1a-prior -->
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
<!-- /results:E1a-prior -->

### Decision
**ADOPTED (superiority, DR-004).** This is the core upgrade.

Against E0b:
- dev CER 32.5% → **5.0%**, Δ −27.5 pt with 95% CI [−42.5, −12.5];
- NLL 2.52 → 0.11, Δ −2.41 with CI [−3.99, −1.04];
- aicme 90% → 100%;
- the synthetic→real gap shrinks from 31.4 to 4.9 points.

The CIs exclude 0 even on a single seed.

- All letter-group errors disappear. That was the primary hypothesis: correct, stem-anchored side and angle conventions.
- Both remaining errors are off-by-one count errors on pen-1 vowels (E→U, O→A).
- Seeds 1–2 are queued (screen2).

**3-seed confirmation (2026-10-18): CONFIRMED.**
- Dev CER 5.0 / 0.0 / 0.0, mean **1.7%**, against 28.7% for E0b.
- Paired Δ CER −27.1 pt [−40.4, −14.2]; Δ NLL −2.26 [−3.64, −1.06].
- Dev-stress 88.3 / 91.7 / 88.7, mean 89.6%.

---

## E1a: parametric stroke model, ranges calibrated on dev (pen-40)
- Parent: E0b (compared with E1a-prior as well) | Config: `configs/e1a.yaml` | Data: `data/e1a_n850_s0`

### Motivation
It is unclear whether calibrating the generator's ranges to 40 real images helps, or whether broad priors are enough (domain randomisation: cover rather than match).

### Hypothesis
E1a ≤ E1a-prior in dev CER, with the biggest effect on vowels and diagonals, where the priors were furthest from the real data.

### Change
Ranges are set to cover the measured real [min, max] with a margin:
- spacing 0.055–0.19;
- side length 0.18–0.68;
- across length 0.55–0.95;
- across angle sd 5, max 13°;
- diagonal angle 24–48°;
- balance 0.33–0.67;
- group position 0.30–0.80;
- maximum group span 0.60;
- tilt sd 4.

### Results
<!-- results:E1a -->
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
<!-- /results:E1a -->

### Decision
**INCONCLUSIVE on screening (1 seed). The 3-seed confirmation is queued.**

- Calibrating ranges to the real dev geometry did not help seed 0: CER 7.5% vs 5.0%, NLL 0.59 vs 0.11. The difference is one image, inside the seed noise.
- It has the same error pattern as E1a-prior: pen-1 vowel under-counts (I→E, E→U, O→A).
- Under DR-004, "inconclusive" keeps the simpler config, but here both configs are equally simple: they differ only in range values.
- Provisional choice: the **calibrated (measured) ranges**, because they rest on evidence. The E1b–E1d chain was built on them.
- If the 3-seed means show E1a-prior is better on both CER and NLL, the chain is re-based on the prior ranges and the decision is recorded here.

**3-seed confirmation (2026-10-18): NOT ADOPTED (DR-026).**
- Dev CER 7.5 / 0.0 / 5.0, mean 4.2%, against E1a-prior's 1.7%. Paired Δ CER +2.5 pt [−0.0, +5.8].
- **Δ NLL +0.226 [+0.053, +0.431]: significantly worse.**
- Calibrating the generator's ranges to the 40 measured dev images *narrowed* them (shorter vowel range, tighter diagonal angles). This made the probe less well calibrated on real data.
- This supports the domain-randomisation principle of covering the real range rather than matching a small real sample.
- The pre-registered re-base rule fired. It is honoured through the prior-range final candidate E2c-prior (DR-026).

---

## E1b: + hand-drawn imperfection (trajectory space)
- Parent: E1a | Config: `configs/e1b.yaml` | Data: `data/e1b_n850_s0`

### Motivation
Real strokes bow, wobble and hook. Pens lift and leave gaps, and strokes over- or undershoot the stem. The report (§2) says to perturb in trajectory space, where the topology is controlled, rather than with pixel-space elastic warps.

### Hypothesis
Lower dev CER on pen-1/pen-2, which have wobbly ballpoint strokes. Count errors (off-by-one) should fall.

### Change
- curvature ±0.035;
- stem curvature 0.02;
- wave 0.008;
- tremor 0.003;
- side-stroke overshoot −0.02…+0.04;
- hooks p=0.15;
- stem pen-lift gaps p=0.15.

The first attempt (curvature 0.06, wave 0.012) was too strong; see DR-018.

### Results
<!-- results:E1b -->
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
<!-- /results:E1b -->

### Decision
**ADOPTED (provisional, superiority, 1 seed; seeds 1–2 queued).**

Against E1a:
- dev CER 7.5% → **0.0%** (40/40);
- NLL 0.585 → 0.0002.

- The hand-drawn imperfection removed the pen-1 vowel under-counts.
- After the DR-018 fixes, the imperfections stay label-preserving: detector decode on QA renders is 99%.
- **Dev is now saturated (40/40).** From here on, dev differences are measured mainly through NLL and seed means.

**3-seed confirmation (2026-10-18): CONFIRMED (superiority on NLL).**
- Dev CER 0.0 / 2.5 / 5.0, mean 2.5%, against 4.2%. Paired Δ CER −1.7 pt [−5.0, +1.7].
- **Δ NLL −0.211 [−0.412, −0.040].**
- Dev-stress mean 87.9%.
- Seed 0's perfect score was partly luck. The mean improvement over E1a holds.

---

## E1c: + pen and ink model (ballpoint/marker mixture)
- Parent: E1b | Config: `configs/e1c.yaml` | Data: `data/e1c_n850_s0`

### Motivation
- Real ink width ranges from thin ballpoint (0.011–0.029 of stem length, measured on dev) to thick marker.
- The legacy Hybrid checkpoint's remaining test errors under canonical preprocessing were all stroke-*count* errors on thick marker ink.
- Faint ballpoint (pen-1) also needs intensity variation.

### Hypothesis
Lower dev CER, and mainly fewer count errors. This experiment also matters for test (marker), but test is not looked at until the end.

### Change
- A 50/50 mixture of ballpoint (width 0.008–0.030, taper 0–0.5, intensity 0.55–1.0) and marker (width 0.03–0.07, a *prior*; taper 0–0.2; intensity 0.85–1.0).
- Width wobble.
- Stroke spacing ≥ 1.7 × pen width.

### Results
<!-- results:E1c -->
**Probe:** hybrid | **input:** canonical | **dataset:** `data/e1c_n850_s0` (850/class) | **seeds:** 3 | **steps used:** 7167 (best at 6500) | **train time:** 15.9 min/run

| Metric (dev, pen-40) | E1c (n=3) | E1b (n=3) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 0.0% ± 0.0 | 2.5% ± 2.0 | -2.5 pt [-5.8, -0.0] |
| Accuracy | 100.0% ± 0.0 | 97.5% ± 2.0 | +2.5 pt [+0.0, +5.8] |
| Top-3 | 100.0% ± 0.0 | 100.0% ± 0.0 | +0.0 pt |
| Macro-F1 | 1.000 ± 0.000 | 0.973 ± 0.022 | +0.027 |
| Aicme acc | 100.0% ± 0.0 | 100.0% ± 0.0 | +0.0 pt |
| Mean conf | 100.0% ± 0.0 | 99.2% ± 0.6 | +0.8 pt |
| NLL | 0.000 ± 0.000 | 0.052 ± 0.048 | -0.052 [-0.114, -0.004] |
| ECE | 0.000 ± 0.000 | 0.024 ± 0.020 | -0.024 |
| Synthetic val acc | 100.0% ± 0.0 | 99.9% ± 0.0 | +0.1 pt |
| Synth→real gap (val acc − dev acc) | -0.0% ± 0.0 | 2.4% ± 2.1 | -2.4 pt |
| Dev-stress acc (supplementary, DR-022) | 96.5% (n=3) | 87.9% (n=3) | +8.6 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E1c | 100% | 100% | 100% | 100% | 100% | 79% |
| E1b | 87% | 92% | 98% | 99% | 98% | 54% |

**Blank-page control:** H (1.00), H (1.00), B (1.00)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E1c | 0.0% | 0.0% |
| E1b | 5.0% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E1c | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E1b | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |

**Figures:** [curves](results/figures/E1c/learning_curves.png) · [confusion](results/figures/E1c/confusion_dev.png) · [taxonomy](results/figures/E1c/taxonomy_dev.png) · [contact sheet](results/figures/E1c/contact_sheet.png)
<!-- /results:E1c -->

### Decision
**ADOPTED (non-inferiority, DR-021; 1 seed; seeds 1–2 queued).**

- Dev CER is 0.0% → 0.0% and NLL 0.0002 → <0.0001: non-inferior.
- The run is also the most *stable* of the chain. Dev CER is 0 at every check from step 1,000 onwards, a trajectory mean of 0.006 against 0.047 for E1b.
- The pen/marker model exists mainly to cover the marker medium, which only the test set contains. Its payoff is measured once, in the final marker-40 table.

**3-seed confirmation (2026-10-18): CONFIRMED (now superior, not just non-inferior).**
- Dev CER 0 / 0 / 0 (mean 0.0%), against E1b's 2.5%. NLL ≈ 0.
- Dev-stress 97.1 / 97.1 / 95.4, mean **96.5%**, against E1b's 87.9%.
- The pen/marker ink model is what made the chain stable across seeds.

---

## E1d: + hard cases and distractors (detector-crop artefacts)
- Parent: E1c | Config: `configs/e1d.yaml` | Data: `data/e1d_n850_s0`

### Motivation
The report (§1, §2) recommends deliberately generating the hard cases, and SynthTIGER-style "noise text": partial neighbours at the crop edge and stray marks, as a detector crop would produce.

### Hypothesis
Small or no change on dev, whose crops are clean. The value should show on harder real data and on the detector track (E3). A drop in dev CER would indicate over-randomisation.

### Change
- 25% hard cases: near-touching, short strokes, steep tilt, uneven spacing.
- 25% stem extensions past the crop edge, 50% of them with partial neighbour strokes.
- Stray marks (p=0.3, up to 2).

### Results
<!-- results:E1d -->
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
<!-- /results:E1d -->

### Decision
**ADOPTED (non-inferiority, DR-021; 1 seed; seeds 1–2 queued).**

- Dev CER is 0.0% → 0.0% and NLL ≈ 0.0002: non-inferior.
- Synthetic val accuracy falls to 99.1%, which is expected because these are hard cases by design. It is not an over-randomisation alarm, since real dev is unaffected.
- The hard cases and distractors exist for detector crops and the detector track (E3). The proxy detector decodes only 91% of E1d renders, against 97.5–100% on real images, so E1d is a deliberate stress set.
- **E1d is the provisional E1-best** and the base for Phase 2 (E3a–c) and Phase 3 (E2a–d).

**3-seed confirmation (2026-10-18): CONFIRMED (non-inferiority).**
- Dev CER 0 / 0 / 0, NLL ≈ 0.
- Dev-stress 98.3 / 95.0 / 93.8, mean 95.7%, against E1c's 96.5% (−0.8 pt, within the DR-022 margin).
- Synthetic val 99.1–99.4%.

---

## E1d-ac: E1-best data under autocontrast + nearest preprocessing (controlled preprocessing test)
- Parent: E1d | Data: `data/e1d_n850_s0` (the same images, `cache_autocontrast`) | Input: `autocontrast`

### Motivation
E0c beat E0b by a wide margin with *legacy* data (CER 10% vs 32.5%). Our hypothesis is that canonical scale normalisation exposes the legacy generator's spacing mismatch. The question is whether autocontrast + nearest still wins once the data is right.

### Hypothesis
With the calibrated E1 generator, canonical ≤ autocontrast on dev: both should be near 0%. Canonical remains the safer choice because of the blank-page control and because it is scale-invariant. If autocontrast is better by more than 1 image or by NLL, DR-005 is revisited.

### Change
Preprocessing only. Train and dev use autocontrast + nearest squash instead of canonical.

### Results
<!-- results:E1d-ac -->
**Probe:** hybrid | **input:** autocontrast | **dataset:** `data/e1d_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 8000 (best at 7500) | **train time:** 1074.3 min/run

| Metric (dev, pen-40) | E1d-ac (n=1) | E1d (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 0.0% | 0.0% | -0.0 pt [-0.0, -0.0] |
| Accuracy | 100.0% | 100.0% | +0.0 pt [+0.0, +0.0] |
| Top-3 | 100.0% | 100.0% | +0.0 pt |
| Macro-F1 | 1.000 | 1.000 | +0.000 |
| Aicme acc | 100.0% | 100.0% | +0.0 pt |
| Mean conf | 100.0% | 100.0% | +0.0 pt |
| NLL | 0.000 | 0.000 | -0.000 [-0.000, +0.000] |
| ECE | 0.000 | 0.000 | -0.000 |
| Synthetic val acc | 98.8% | 99.1% | -0.3 pt |
| Synth→real gap (val acc − dev acc) | -1.2% | -0.9% | -0.3 pt |
| Dev-stress acc (supplementary, DR-022) | 97.5% (n=1) | 98.3% (n=1) | -0.8 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E1d-ac | 100% | 100% | 100% | 100% | 100% | 85% |
| E1d | 100% | 100% | 100% | 100% | 100% | 90% |

**Blank-page control:** A (0.45)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E1d-ac | 0.0% | 0.0% |
| E1d | 0.0% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E1d-ac | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E1d | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Figures:** [curves](results/figures/E1d-ac/learning_curves.png) · [confusion](results/figures/E1d-ac/confusion_dev.png) · [taxonomy](results/figures/E1d-ac/taxonomy_dev.png)
<!-- /results:E1d-ac -->

### Decision
**FINDING; canonical retained (DR-005 stands).**

| Input on E1-best data | Dev CER | Dev-stress |
|---|---|---|
| Autocontrast | 0.0% | 97.5% |
| Canonical | 0.0% | 98.3% |

- With the new generator, autocontrast + nearest is no better than canonical.
- It still turns a blank page into a confident letter.
- The large E0c advantage was specific to legacy data. This supports the E0b hypothesis: canonical scale normalisation exposed the legacy generator's too-wide stroke spacing, and the calibrated E1 geometry removes that mismatch.
- **Recommendation to the team:** canonical preprocessing pairs with the v2 generator. The autocontrast branch is not needed once the training data is fixed.

---

## E3a: probe trained on canonical renders of exact (oracle) detector JSON
- Parent: E1d | Config: `configs/e3a.yaml` | Data: `data/e3a_n850_s0` (`cache_render`) | Input: `render`. Real images go through the proxy detector v1, then JSON, then canonical render.

### Motivation
The plan's Phase 2 representation fork (DR-008): synthetic and real meet in the detector's representation, which discards pen, ink, paper and camera. Also, E3-ref (rules alone) already scores 100% on dev.

### Hypothesis
Dev CER ≈ 0% (non-inferior to E1d). The representation removes the appearance gap, but it adds a dependency on the detector. Because the training data assumes a perfect detector, E3a may be brittle to real detector errors.

### Change
The probe's input becomes `canonical_render(json)` (DR-008). Training JSON is the exact oracle geometry of E1d samples, with E1d distractors drawn faintly as unattached segments.

### Results
<!-- results:E3a -->
**Probe:** hybrid | **input:** render | **dataset:** `data/e3a_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 6000 (best at 4000) | **train time:** 12.9 min/run

| Metric (dev, pen-40) | E3a (n=1) | E1d (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 0.0% | 0.0% | -0.0 pt [-0.0, -0.0] |
| Accuracy | 100.0% | 100.0% | +0.0 pt [+0.0, +0.0] |
| Top-3 | 100.0% | 100.0% | +0.0 pt |
| Macro-F1 | 1.000 | 1.000 | +0.000 |
| Aicme acc | 100.0% | 100.0% | +0.0 pt |
| Mean conf | 100.0% | 100.0% | +0.0 pt |
| NLL | 0.000 | 0.000 | -0.000 [-0.000, -0.000] |
| ECE | 0.000 | 0.000 | -0.000 |
| Synthetic val acc | 100.0% | 99.1% | +0.9 pt |
| Synth→real gap (val acc − dev acc) | 0.0% | -0.9% | +0.9 pt |
| Dev-stress acc (supplementary, DR-022) | 84.2% (n=1) | 98.3% (n=1) | -14.2 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E3a | 100% | 100% | 100% | 100% | 100% | 5% |
| E1d | 100% | 100% | 100% | 100% | 100% | 90% |

**Blank-page control:** B (0.53)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E3a | 0.0% | 0.0% |
| E1d | 0.0% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E3a | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E1d | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Figures:** [curves](results/figures/E3a/learning_curves.png) · [confusion](results/figures/E3a/confusion_dev.png) · [taxonomy](results/figures/E3a/taxonomy_dev.png) · [contact sheet](results/figures/E3a/contact_sheet.png)
<!-- /results:E3a -->

### Decision
**NOT ADOPTED as the default input.** It fails the DR-022 dev-stress guard.

- On clean dev it ties: CER 0%, NLL ≈ 0, non-inferior.
- On dev-stress it scores **84.2%, against 98.3% for the pixel path (E1d)**, a −14.2 pt drop.
- Every degradation scores 100% except the shadow ramp, which scores **5%**.
- The cause is the proxy detector, not the synthetic data. E3-ref (the detector plus rules, no learning) scores 84.6% on dev-stress with shadow 7%:
  - global Otsu binarisation turns a shadow gradient into "ink";
  - the stem is lost;
  - the canonical render is then blank or wrong.
- **Gate (Phase 2):** raw pixels win with the current detector. E2 (appearance) is therefore the main lever for the probe. The detector path needs **shading-robust binarisation** before it can compete (DR-024).

---

## E3b: E3a + simulated detector errors
- Parent: E3a | Config: `configs/e3b.yaml`

### Motivation
A recogniser fed detector output must tolerate detector failures. The report (§9) has option 2: simulate the detector's typical failures, with rates measured on real data.

### Hypothesis
Non-inferior on dev, where the detector makes no letter-level errors. More robust where the detector fails, which on E1d renders it does 9% of the time. Measured through E3c-style evaluation and the final test.

### Change
- drop 3%/stroke;
- split crossing strokes 3%;
- merge 1%;
- spurious segment 10%/glyph;
- endpoint jitter 0.012;
- stem truncation 3%.

These rates are centred on the proxy detector's measured E1c errors (`tools/measure_detector_errors.py`).

### Results
<!-- results:E3b -->
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
<!-- /results:E3b -->

### Decision
**Non-inferior to E3a, which itself is not the default input.** Kept as the preferred decoder-training recipe for the detector track.

- Dev CER 0%. NLL 0.078 against E3a's 0.000; the Δ is within the DR-021 margin of 0.25.
- Dev-stress 84.6% (+0.4 pt).
- Synthetic val accuracy is 89.4%. Simulated detector errors make about 10% of synthetic samples genuinely ambiguous (a dropped stroke changes the visible count). That is expected for this module, not an alarm.
- The simulated errors cannot help when the detector loses the stem entirely, which is the shadow case.

---

## E3c: render-then-detect (the real detector on synthetic renders)
- Parent: E3a | Config: `configs/e3c.yaml`

### Motivation
Report §9, option 3, the recommended end state: train on the detector's *actual* output for synthetic images, so the recogniser learns the true detector artefacts. Here the renders are E1d renders, with no appearance randomisation. It is re-run on E2-best photos later.

### Hypothesis
Of all the E3 variants, this is the closest to real detector output. Non-inferior on dev.

### Change
Every training sample goes from E1d render to proxy detector v1, then JSON, then canonical render.

### Results
<!-- results:E3c -->
**Probe:** hybrid | **input:** render | **dataset:** `data/e3c_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 8000 (best at 7500) | **train time:** 17.1 min/run

| Metric (dev, pen-40) | E3c (n=1) | E3a (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 0.0% | 0.0% | -0.0 pt [-0.0, -0.0] |
| Accuracy | 100.0% | 100.0% | +0.0 pt [+0.0, +0.0] |
| Top-3 | 100.0% | 100.0% | +0.0 pt |
| Macro-F1 | 1.000 | 1.000 | +0.000 |
| Aicme acc | 100.0% | 100.0% | +0.0 pt |
| Mean conf | 99.5% | 100.0% | -0.5 pt |
| NLL | 0.006 | 0.000 | +0.005 [+0.001, +0.014] |
| ECE | 0.005 | 0.000 | +0.005 |
| Synthetic val acc | 98.7% | 100.0% | -1.3 pt |
| Synth→real gap (val acc − dev acc) | -1.3% | 0.0% | -1.3 pt |
| Dev-stress acc (supplementary, DR-022) | 85.0% (n=1) | 84.2% (n=1) | +0.8 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E3c | 100% | 100% | 100% | 100% | 100% | 10% |
| E3a | 100% | 100% | 100% | 100% | 100% | 5% |

**Blank-page control:** B (0.29)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E3c | 0.0% | 0.0% |
| E3a | 0.0% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E3c | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E3a | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Figures:** [curves](results/figures/E3c/learning_curves.png) · [confusion](results/figures/E3c/confusion_dev.png) · [taxonomy](results/figures/E3c/taxonomy_dev.png) · [contact sheet](results/figures/E3c/contact_sheet.png)
<!-- /results:E3c -->

### Decision
**Non-inferior; the best of the E3 variants. Kept as the decoder-training recipe once the detector is shading-robust.**

- Render-then-detect: dev CER 0%, NLL 0.006, dev-stress 85.0%, synthetic val 98.7%.
- It learns the proxy detector's real artefacts (merged blobs, stray stem-top segments) without needing hand-set error rates.
- Re-run on E2-best photos with a shading-robust detector (proxy-v2) is a follow-up (DR-024).

---

## E2a: + paper and ink appearance
- Parent: E1d | Config: `configs/e2a.yaml`

### Motivation
Report §3 (domain randomisation): the real images have off-white paper, a blue/purple ballpoint cast, show-through and scan grain.

### Hypothesis
Non-inferior on dev (DR-021: a coverage module). Dev is pen on near-white paper, and canonical autocontrast already removes most paper variation, so we expect little effect on dev. The value is robustness on unseen media and phone photos.

### Change
- procedural paper: base 228–255, tint, low-frequency tone, grain, fibres;
- ruled/grid lines at 8%;
- ink colours by pen kind;
- ink bleed at 0.3;
- low-ink streaks at 0.2.

### Results
<!-- results:E2a -->
**Probe:** hybrid | **input:** canonical | **dataset:** `data/e2a_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 7000 (best at 5000) | **train time:** 15.0 min/run

| Metric (dev, pen-40) | E2a (n=1) | E1d (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 0.0% | 0.0% | -0.0 pt [-0.0, -0.0] |
| Accuracy | 100.0% | 100.0% | +0.0 pt [+0.0, +0.0] |
| Top-3 | 100.0% | 100.0% | +0.0 pt |
| Macro-F1 | 1.000 | 1.000 | +0.000 |
| Aicme acc | 100.0% | 100.0% | +0.0 pt |
| Mean conf | 100.0% | 100.0% | +0.0 pt |
| NLL | 0.000 | 0.000 | -0.000 [-0.000, +0.000] |
| ECE | 0.000 | 0.000 | -0.000 |
| Synthetic val acc | 98.1% | 99.1% | -1.0 pt |
| Synth→real gap (val acc − dev acc) | -1.9% | -0.9% | -1.0 pt |
| Dev-stress acc (supplementary, DR-022) | 97.9% (n=1) | 98.3% (n=1) | -0.4 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E2a | 100% | 100% | 100% | 100% | 100% | 88% |
| E1d | 100% | 100% | 100% | 100% | 100% | 90% |

**Blank-page control:** I (0.30)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E2a | 0.0% | 0.0% |
| E1d | 0.0% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E2a | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E1d | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Figures:** [curves](results/figures/E2a/learning_curves.png) · [confusion](results/figures/E2a/confusion_dev.png) · [taxonomy](results/figures/E2a/taxonomy_dev.png) · [contact sheet](results/figures/E2a/contact_sheet.png)
<!-- /results:E2a -->

### Decision
**ADOPTED (non-inferiority, DR-021).**

- Dev CER 0% → 0%, NLL ≈ 0.
- Dev-stress 97.9%, against E1d's 98.3%. The −0.4 pt drop is within the 2.5 pt guard.
- Synthetic val 98.1%.
- Paper and ink variety do not hurt, but do not help measurably on pen-on-white dev either. As expected, canonical autocontrast already normalises paper tone.

---

## E2b: + lighting
- Parent: E2a | Config: `configs/e2b.yaml`

### Motivation
Phone captures have illumination gradients, shadow bands (as in pen 00038), vignetting and white-balance shifts.

### Hypothesis
Non-inferior on dev.

### Change
- gradient up to 0.25;
- shadow band at p=0.3, strength 0.05–0.35;
- vignette up to 0.3;
- white-balance sd 0.025;
- gamma 0.8–1.25.

### Results
<!-- results:E2b -->
**Probe:** hybrid | **input:** canonical | **dataset:** `data/e2b_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 7500 (best at 5500) | **train time:** 16.1 min/run

| Metric (dev, pen-40) | E2b (n=1) | E2a (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 0.0% | 0.0% | -0.0 pt [-0.0, -0.0] |
| Accuracy | 100.0% | 100.0% | +0.0 pt [+0.0, +0.0] |
| Top-3 | 100.0% | 100.0% | +0.0 pt |
| Macro-F1 | 1.000 | 1.000 | +0.000 |
| Aicme acc | 100.0% | 100.0% | +0.0 pt |
| Mean conf | 100.0% | 100.0% | -0.0 pt |
| NLL | 0.000 | 0.000 | +0.000 [-0.000, +0.000] |
| ECE | 0.000 | 0.000 | +0.000 |
| Synthetic val acc | 98.3% | 98.1% | +0.2 pt |
| Synth→real gap (val acc − dev acc) | -1.7% | -1.9% | +0.2 pt |
| Dev-stress acc (supplementary, DR-022) | 100.0% (n=1) | 97.9% (n=1) | +2.1 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E2b | 100% | 100% | 100% | 100% | 100% | 100% |
| E2a | 100% | 100% | 100% | 100% | 100% | 88% |

**Blank-page control:** H (0.32)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E2b | 0.0% | 0.0% |
| E2a | 0.0% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E2b | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E2a | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Figures:** [curves](results/figures/E2b/learning_curves.png) · [confusion](results/figures/E2b/confusion_dev.png) · [taxonomy](results/figures/E2b/taxonomy_dev.png) · [contact sheet](results/figures/E2b/contact_sheet.png)
<!-- /results:E2b -->

### Decision
**ADOPTED (non-inferiority, DR-021).** Robustness gain on dev-stress.

- Dev CER 0%, NLL ≈ 0.
- **Dev-stress 97.9% → 100%**: the shadow degradation goes from 88% to **100%**.
- Lighting randomisation (gradients, shadow bands, vignette) teaches exactly the invariance the shadow ramp tests. As DR-022 notes, this gain is partly by construction.

---

## E2c: + capture degradations
- Parent: E2b | Config: `configs/e2c.yaml`

### Motivation
Defocus (marker batch 1), scan resampling and JPEG.

### Hypothesis
Non-inferior on dev. Watch synthetic-val accuracy for over-randomisation.

### Change
- perspective at p=0.5;
- rotation ±6°;
- Gaussian blur at p=0.5, σ 0.3–1.6;
- motion blur at 0.15;
- rescale at 0.3;
- noise sd up to 6;
- JPEG at p=0.5, quality 30–95.

### Results
<!-- results:E2c -->
**Probe:** hybrid | **input:** canonical | **dataset:** `data/e2c_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 8000 (best at 7000) | **train time:** 17.2 min/run

| Metric (dev, pen-40) | E2c (n=1) | E2b (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 0.0% | 0.0% | -0.0 pt [-0.0, -0.0] |
| Accuracy | 100.0% | 100.0% | +0.0 pt [+0.0, +0.0] |
| Top-3 | 100.0% | 100.0% | +0.0 pt |
| Macro-F1 | 1.000 | 1.000 | +0.000 |
| Aicme acc | 100.0% | 100.0% | +0.0 pt |
| Mean conf | 100.0% | 100.0% | -0.0 pt |
| NLL | 0.000 | 0.000 | +0.000 [+0.000, +0.001] |
| ECE | 0.000 | 0.000 | +0.000 |
| Synthetic val acc | 99.0% | 98.3% | +0.7 pt |
| Synth→real gap (val acc − dev acc) | -1.0% | -1.7% | +0.7 pt |
| Dev-stress acc (supplementary, DR-022) | 100.0% (n=1) | 100.0% (n=1) | +0.0 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E2c | 100% | 100% | 100% | 100% | 100% | 100% |
| E2b | 100% | 100% | 100% | 100% | 100% | 100% |

**Blank-page control:** F (0.33)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E2c | 0.0% | 0.0% |
| E2b | 0.0% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E2c | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E2b | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Figures:** [curves](results/figures/E2c/learning_curves.png) · [confusion](results/figures/E2c/confusion_dev.png) · [taxonomy](results/figures/E2c/taxonomy_dev.png) · [contact sheet](results/figures/E2c/contact_sheet.png)
<!-- /results:E2c -->

### Decision
**ADOPTED (non-inferiority, DR-021).**

- Dev CER 0%, NLL ≈ 0, dev-stress 100%, synthetic val 99.0%.
- Capture degradations (perspective, blur, noise, JPEG, rescale) cost nothing on real dev.
- Their dev-stress categories (blur, lowres, JPEG, rotate) were already at 100% from E1d onwards.

---

## E2d: + artefacts and framing
- Parent: E2c | Config: `configs/e2d.yaml`

### Motivation
Show-through (pen 00024), dark edge bands (pen 00038), scan speckle and very loose crops (pen-1).

### Hypothesis
Non-inferior on dev. These artefacts mainly stress the ink-box crop.

### Change
- show-through at 0.15;
- edge band at 0.15;
- speckle at 0.3;
- loose crop at 0.3, with the character filling 15–60% of the frame.

### Results
<!-- results:E2d -->
**Probe:** hybrid | **input:** canonical | **dataset:** `data/e2d_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 8000 (best at 7000) | **train time:** 18.0 min/run

| Metric (dev, pen-40) | E2d (n=1) | E2c (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 0.0% | 0.0% | -0.0 pt [-0.0, -0.0] |
| Accuracy | 100.0% | 100.0% | +0.0 pt [+0.0, +0.0] |
| Top-3 | 100.0% | 100.0% | +0.0 pt |
| Macro-F1 | 1.000 | 1.000 | +0.000 |
| Aicme acc | 100.0% | 100.0% | +0.0 pt |
| Mean conf | 100.0% | 100.0% | +0.0 pt |
| NLL | 0.000 | 0.000 | -0.000 [-0.001, +0.000] |
| ECE | 0.000 | 0.000 | -0.000 |
| Synthetic val acc | 90.9% | 99.0% | -8.1 pt |
| Synth→real gap (val acc − dev acc) | -9.1% | -1.0% | -8.1 pt |
| Dev-stress acc (supplementary, DR-022) | 100.0% (n=1) | 100.0% (n=1) | +0.0 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E2d | 100% | 100% | 100% | 100% | 100% | 100% |
| E2c | 100% | 100% | 100% | 100% | 100% | 100% |

**Blank-page control:** H (0.11)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E2d | 0.0% | 0.0% |
| E2c | 0.0% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E2d | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E2c | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Figures:** [curves](results/figures/E2d/learning_curves.png) · [confusion](results/figures/E2d/confusion_dev.png) · [taxonomy](results/figures/E2d/taxonomy_dev.png) · [contact sheet](results/figures/E2d/contact_sheet.png)
<!-- /results:E2d -->

### Decision
**Over-randomisation and label-break alarm. Replaced by E2d-v2 after a generator fix (DR-023).**

- Real metrics are unchanged: dev CER 0%, NLL ≈ 0, dev-stress 100%.
- But **synthetic val accuracy fell to 90.9%** (from 99.0%), the DR-010 alarm.
- Inspection of the failing validation images (`data/_e2d_val_errors.png`) found three causes:
  1. **Label-breaking:** perspective/rotation warps pushed loosely framed characters partly or fully out of the frame.
  2. **Crop failure:** heavy shading plus very loose framing defeats the ink-box crop, so the character stays tiny at 128 px. This is a preprocessing robustness limit.
  3. **Show-through ghosts:** at up to 18% strength, the mirrored ghost looks like a second character, and a mirrored B-group letter looks like an H-group letter.
- **Fix (DR-023):**
  - Warps are applied only if at least 97% of the ink stays in frame.
  - Show-through strength is cut to 4–10%.
- **E2d-v2** (the same config, fixed generator) is queued.


---

## E2d-v2: E2d with the label-guarded generator (DR-023)
- Parent: E2c | Config: `configs/e2d.yaml` (unchanged) | Data: `data/e2d_v2_n850_s0` (generator at cb6725c)

### Motivation
E2d's 90.9% synthetic validation accuracy exposed label-breaking warps and over-strong show-through (DR-023).

### Hypothesis
Synthetic val recovers to about 98% or more, and dev and dev-stress stay at the ceiling. If synthetic val stays low, the remaining failures are the crop-under-shading limit (cause 2), which is a preprocessing issue rather than label noise.

### Change
Generator fix only. No config change.

### Results
<!-- results:E2d-v2 -->
**Probe:** hybrid | **input:** canonical | **dataset:** `data/e2d_v2_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 8000 (best at 6500) | **train time:** 24.3 min/run

| Metric (dev, pen-40) | E2d-v2 (n=1) | E2c (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 0.0% | 0.0% | -0.0 pt [-0.0, -0.0] |
| Accuracy | 100.0% | 100.0% | +0.0 pt [+0.0, +0.0] |
| Top-3 | 100.0% | 100.0% | +0.0 pt |
| Macro-F1 | 1.000 | 1.000 | +0.000 |
| Aicme acc | 100.0% | 100.0% | +0.0 pt |
| Mean conf | 100.0% | 100.0% | +0.0 pt |
| NLL | 0.000 | 0.000 | -0.000 [-0.001, +0.001] |
| ECE | 0.000 | 0.000 | -0.000 |
| Synthetic val acc | 92.2% | 99.0% | -6.8 pt |
| Synth→real gap (val acc − dev acc) | -7.8% | -1.0% | -6.8 pt |
| Dev-stress acc (supplementary, DR-022) | 99.6% (n=1) | 100.0% (n=1) | -0.4 pt |

| Dev-stress by degradation | blur | lowres | jpeg | rotate | contrast | shadow |
|---|---|---|---|---|---|---|
| E2d-v2 | 100% | 100% | 100% | 100% | 100% | 98% |
| E2c | 100% | 100% | 100% | 100% | 100% | 100% |

**Blank-page control:** A (0.10)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E2d-v2 | 0.0% | 0.0% |
| E2c | 0.0% | 0.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E2d-v2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E2c | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Figures:** [curves](results/figures/E2d-v2/learning_curves.png) · [confusion](results/figures/E2d-v2/confusion_dev.png) · [taxonomy](results/figures/E2d-v2/taxonomy_dev.png) · [contact sheet](results/figures/E2d-v2/contact_sheet.png)
<!-- /results:E2d-v2 -->

### Decision
_(pending: queued at the head of screen2)_

---

## FINAL: held-out evaluation on marker-40 (frozen configurations, DR-025/DR-027)
- Date: 2026-10-18 | Tool: `train/final_table.py --evaluate` | Full table: [results/final_table.md](results/final_table.md)
- This was computed once, after freezing. No result here feeds back into any decision.

**Headline (hybrid probe, 3 seeds):**

| Pipeline | Marker-40 accuracy | CER | Δ accuracy vs legacy generator E0b [95% CI] |
|---|---|---|---|
| Frozen pipeline **E2c** | **98.3 ± 1.2%** | 1.7% | +35.8 pt [+22.1, +49.6] |
| **E1c** | **100%** (all 3 seeds) | 0% | |
| Team's reported baseline | 42.5% | 57.5% | |

Narrative and conclusions are in [REPORT_NOTES.md](REPORT_NOTES.md#final-results-held-out-marker-40-the-teams-wild-test-set).

*Pending rows:* E2c-prior, the second probe (plain CNN) at E0b/E1d/E2c, E3c seeds 1–2, E0b seed 2. The table is re-run when they finish.
