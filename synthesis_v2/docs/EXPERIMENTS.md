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
**Probe:** hybrid | **input:** legacy | **dataset:** `data/legacy_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 7500 (best at 5500) | **train time:** 20.0 min/run

| Metric (dev, pen-40) | E0a (n=1) |
|---|---|
| CER | 42.5% |
| Accuracy | 57.5% |
| Top-3 | 85.0% |
| Macro-F1 | 0.492 |
| Aicme acc | 77.5% |
| Mean conf | 91.8% |
| NLL | 2.701 |
| ECE | 0.343 |
| Synthetic val acc | 98.8% |
| Synth→real gap (val acc − dev acc) | 41.3% |

**Blank-page control:** H (0.79)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E0a | 60.0% | 25.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E0a | 0.0 | 6.0 | 3.0 | 8.0 | 0.0 |

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
**Probe:** hybrid | **input:** canonical | **dataset:** `data/legacy_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 8000 (best at 7000) | **train time:** 18.9 min/run

| Metric (dev, pen-40) | E0b (n=1) | E0a (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 32.5% | 42.5% | -10.0 pt [-30.0, +10.0] |
| Accuracy | 67.5% | 57.5% | +10.0 pt [-10.0, +30.0] |
| Top-3 | 95.0% | 85.0% | +10.0 pt |
| Macro-F1 | 0.636 | 0.492 | +0.144 |
| Aicme acc | 90.0% | 77.5% | +12.5 pt |
| Mean conf | 97.1% | 91.8% | +5.3 pt |
| NLL | 2.519 | 2.701 | -0.183 [-1.720, +1.511] |
| ECE | 0.296 | 0.343 | -0.047 |
| Synthetic val acc | 98.9% | 98.8% | +0.1 pt |
| Synth→real gap (val acc − dev acc) | 31.4% | 41.3% | -9.9 pt |

**Blank-page control:** E (0.96)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E0b | 30.0% | 35.0% |
| E0a | 60.0% | 25.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E0b | 3.0 | 0.0 | 1.0 | 9.0 | 0.0 |
| E0a | 0.0 | 6.0 | 3.0 | 8.0 | 0.0 |

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
**Probe:** hybrid | **input:** autocontrast | **dataset:** `data/legacy_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 6500 (best at 4500) | **train time:** 14.7 min/run

| Metric (dev, pen-40) | E0c (n=1) | E0a (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 10.0% | 42.5% | -32.5 pt [-47.5, -20.0] |
| Accuracy | 90.0% | 57.5% | +32.5 pt [+20.0, +47.5] |
| Top-3 | 100.0% | 85.0% | +15.0 pt |
| Macro-F1 | 0.893 | 0.492 | +0.401 |
| Aicme acc | 100.0% | 77.5% | +22.5 pt |
| Mean conf | 96.0% | 91.8% | +4.2 pt |
| NLL | 0.436 | 2.701 | -2.265 [-3.427, -1.126] |
| ECE | 0.094 | 0.343 | -0.249 |
| Synthetic val acc | 98.6% | 98.8% | -0.2 pt |
| Synth→real gap (val acc − dev acc) | 8.6% | 41.3% | -32.7 pt |

**Blank-page control:** R (1.00)

| Per batch (dev CER) | pen-1 | pen-2 |
|---|---|---|
| E0c | 10.0% | 10.0% |
| E0a | 60.0% | 25.0% |

| Error taxonomy (mean errors/seed) | group_side_BH | group_diag_across_MA | group_other | count_off_by_one | count_off_more |
|---|---|---|---|---|---|
| E0c | 0.0 | 0.0 | 0.0 | 4.0 | 0.0 |
| E0a | 0.0 | 6.0 | 3.0 | 8.0 | 0.0 |

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
<!-- /results:E1a -->

### Decision
**INCONCLUSIVE on screening (1 seed). The 3-seed confirmation is queued.**

- Calibrating ranges to the real dev geometry did not help seed 0: CER 7.5% vs 5.0%, NLL 0.59 vs 0.11. The difference is one image, inside the seed noise.
- It has the same error pattern as E1a-prior: pen-1 vowel under-counts (I→E, E→U, O→A).
- Under DR-004, "inconclusive" keeps the simpler config, but here both configs are equally simple: they differ only in range values.
- Provisional choice: the **calibrated (measured) ranges**, because they rest on evidence. The E1b–E1d chain was built on them.
- If the 3-seed means show E1a-prior is better on both CER and NLL, the chain is re-based on the prior ranges and the decision is recorded here.

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
<!-- /results:E1b -->

### Decision
**ADOPTED (provisional, superiority, 1 seed; seeds 1–2 queued).**

Against E1a:
- dev CER 7.5% → **0.0%** (40/40);
- NLL 0.585 → 0.0002.

- The hand-drawn imperfection removed the pen-1 vowel under-counts.
- After the DR-018 fixes, the imperfections stay label-preserving: detector decode on QA renders is 99%.
- **Dev is now saturated (40/40).** From here on, dev differences are measured mainly through NLL and seed means.

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
<!-- /results:E1c -->

### Decision
**ADOPTED (non-inferiority, DR-021; 1 seed; seeds 1–2 queued).**

- Dev CER is 0.0% → 0.0% and NLL 0.0002 → <0.0001: non-inferior.
- The run is also the most *stable* of the chain. Dev CER is 0 at every check from step 1,000 onwards, a trajectory mean of 0.006 against 0.047 for E1b.
- The pen/marker model exists mainly to cover the marker medium, which only the test set contains. Its payoff is measured once, in the final marker-40 table.

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
**Probe:** hybrid | **input:** canonical | **dataset:** `data/e1d_n850_s0` (850/class) | **seeds:** 1 | **steps used:** 6500 (best at 4500) | **train time:** 19.7 min/run

| Metric (dev, pen-40) | E1d (n=1) | E1c (n=1) | Δ vs parent [95% CI] |
|---|---|---|---|
| CER | 0.0% | 0.0% | -0.0 pt [-0.0, -0.0] |
| Accuracy | 100.0% | 100.0% | +0.0 pt [+0.0, +0.0] |
| Top-3 | 100.0% | 100.0% | +0.0 pt |
| Macro-F1 | 1.000 | 1.000 | +0.000 |
| Aicme acc | 100.0% | 100.0% | +0.0 pt |
| Mean conf | 100.0% | 100.0% | -0.0 pt |
| NLL | 0.000 | 0.000 | +0.000 [+0.000, +0.000] |
| ECE | 0.000 | 0.000 | +0.000 |
| Synthetic val acc | 99.1% | 100.0% | -0.9 pt |
| Synth→real gap (val acc − dev acc) | -0.9% | 0.0% | -0.9 pt |

**Blank-page control:** H (0.46)

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
<!-- /results:E1d-ac -->

### Decision
_(pending)_

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
<!-- /results:E3a -->

### Decision
_(pending)_

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
<!-- /results:E3b -->

### Decision
_(pending)_

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
<!-- /results:E3c -->

### Decision
_(pending)_

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
<!-- /results:E2a -->

### Decision
_(pending)_

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
<!-- /results:E2b -->

### Decision
_(pending)_

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
<!-- /results:E2c -->

### Decision
_(pending)_

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
<!-- /results:E2d -->

### Decision
_(pending)_
