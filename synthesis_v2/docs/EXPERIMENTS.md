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
