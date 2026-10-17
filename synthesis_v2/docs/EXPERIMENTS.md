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
<!-- /results:E0a -->

### Decision
_(pending)_

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
<!-- /results:E0b -->

### Decision
_(pending)_

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
<!-- /results:E0c -->

### Decision
_(pending)_

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
<!-- /results:E1a-prior -->

### Decision
_(pending)_

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
<!-- /results:E1a -->

### Decision
_(pending)_

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
<!-- /results:E1b -->

### Decision
_(pending)_

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
<!-- /results:E1c -->

### Decision
_(pending)_

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
<!-- /results:E1d -->

### Decision
_(pending)_
