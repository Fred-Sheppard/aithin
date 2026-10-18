# Report notes: Synthesis v2

A narrative summary for the project report, written at the end of each phase. Each section covers:
- what was tried;
- what moved the needle;
- what didn't;
- surprises;
- figures to use.

## Starting point (before Synthesis v2)

The team's wild test set is 40 marker scans: 2 per letter, labelled by hand, with a blank page as a control.

| Wild test set (n=40) | Augmented | Hybrid | Chance |
|---|---|---|---|
| Accuracy | 15.0% (6/40) | 42.5% (17/40) | 5.0% |
| CER (primary) | 85.0% | 57.5% | 95.0% |
| Top-3 accuracy | 42.5% | 62.5% | 15.0% |
| Macro-F1 | 0.103 | 0.353 | — |
| Right letter group (aicme) | 45.0% | 60.0% | 25.0% |
| Mean confidence | 88.7% | 90.6% | — |

What this shows:
- **The domain gap is large.** The model scores about 99% on synthetic validation data but 42.5% on real images.
- **Most errors are letter-group errors.** Of the Hybrid model's 23 errors, 16 assign the wrong group and 7 have the right group but the wrong stroke count.
- **The model is heavily overconfident.** Mean confidence is about 90% at 42.5% accuracy.
- **Generic augmentation of the legacy data made results worse** (15%).

## Phase 0: harness and baseline

**What we built**
- An evaluation harness:
  - the team's metric table, plus NLL and ECE;
  - an Ogham error taxonomy;
  - paired bootstrap CIs over images;
  - a blank-page control;
  - an append-only results ledger.
- A fixed probe: an exact replica of the Hybrid CNN-BiLSTM, plus a second, plain-CNN probe.
- Early stopping on a separately generated synthetic validation set (DR-010, DR-011).
- An overnight job queue.

**Reproducing the reported baseline (E0-ckpt, DR-015).**
- The legacy loader reproduces the Augmented model's wild-test result exactly: 6/40, CER 85.0%, aicme 45.0%.
- The Hybrid result is within one image: 16/40 against the reported 17/40.

**Finding 1: much of the "domain gap" was a preprocessing gap (E0-ckpt).**
- The legacy loader squashes 1000–4000 px photos to 128 px with nearest-neighbour resampling. This erases thin pen strokes and leaves scanned marker ink grey.
- With no retraining, simply changing preprocessing moves the frozen checkpoints:
  - Hybrid: dev 47.5% → 82.5%, test 40% → 52.5%;
  - Augmented: dev 10% → 62.5–65%, test 15% → 75–77.5%.
- The team's branch `image-preprocessing-investigation` (autocontrast then nearest) captures most of this gain, but it turns a blank page into a 100%-confident "R". Canonical preprocessing does not (DR-019).
- *Figures:* `results/figures/phase0/real_all_{raw,legacy,canonical}.png`. The legacy view shows pen-1 strokes broken and faint.

**Finding 2: a line detector with counting rules nearly solves clean single characters (E3-ref).**
- Our proxy line detector, which emits the same JSON schema as the teammate's, plus a rule decoder that uses no learning, scores:
  - **100% on dev** (pen-40);
  - **97.5% on test** (marker-40; one thick crossing stroke was split).
- This supports the team's line-detection-first design.
- It also changes what synthetic data is *for*. Synthetic data:
  - trains decoders that are robust to detector failures;
  - stress-tests the detector. Our hard-case generator (E1d) drops the detector to 91%.

**Finding 3: the label audit found one error.** `ogham-pen-labelled/00022` is labelled Z but shows R (DR-012).

**Compute (DR-014).**
- Metal GPU at batch 32. A BiLSTM probe run takes about 20 minutes, ending at the 8k-step cap or earlier through early stopping.
- The legacy and v2 generators make 17k images in under a minute.

## Phase 1: stroke model (E1)

*Status 2026-10-18: 3-seed confirmations complete. The 3-seed table is directly below. The seed-0 screening table after it is kept as an audit trail.*

**3-seed results (dev, paired bootstrap 95% CI against the parent):**

| Step | Dev CER (3 seeds) | Δ CER [CI] | Δ NLL [CI] | Dev-stress (mean) | Decision |
|---|---|---|---|---|---|
| E0b (legacy generator) | 28.7% | | | 67% | reference |
| **E1a-prior** (stroke model) | **1.7%** (5 / 0 / 0) | **−27.1 [−40.4, −14.2]** | **−2.26 [−3.64, −1.06]** | 90% | adopted |
| E1a (+ calibrated ranges) | 4.2% (7.5 / 0 / 5) | +2.5 [−0.0, +5.8] | **+0.23 [+0.05, +0.43]** | 88% | **rejected** (DR-026) |
| E1b (+ imperfection) | 2.5% (0 / 2.5 / 5) | −1.7 [−5.0, +1.7] | **−0.21 [−0.41, −0.04]** | 88% | adopted |
| E1c (+ pen/marker) | **0.0%** (0 / 0 / 0) | −2.5 | ≈ −0.05 | **96.5%** | adopted |
| E1d (+ hard cases, distractors) | 0.0% (0 / 0 / 0) | 0 | ≈ 0 | 95.7% | adopted (non-inferiority) |

**What we built.** A parametric generator, `ogham_v2/geometry.py` + `pen.py`:
- characters are trajectories in stem coordinates;
- every sample passes a label-preserving validator: stroke count; side; crossing halves; diagonal vs across angle margin; spacing; no converging strokes;
- a supersampled variable-width ballpoint/marker rasteriser;
- exact "oracle" detector JSON for every sample.

Property tests confirm that the oracle round-trips to the true letter in 100% of cases. Generation takes about 3 ms per image per core.

**Calibration** (dev only, `tools/measure_real.py`). The proxy detector measured real pen geometry. The hand-set priors were off in five ways:
- real vowel strokes are long: 0.64–0.91 of the stem;
- across strokes deviate by up to 12°;
- diagonals sit at 26–42°;
- spacing is tighter: median 0.112;
- stroke groups sit centred, with about 0.3 of stem overhang.

*Figures:* `results/figures/E1/calibration_e1a_{initial,calibrated}.png`.

**Results (dev, seed 0):**

| Step | Change | Dev CER | Dev NLL | Decision |
|---|---|---|---|---|
| E0b | legacy generator + canonical preprocessing | 32.5% | 2.52 | reference |
| **E1a-prior** | **parametric stroke model (priors)** | **5.0%** | 0.11 | **adopted**: −27.5 pt, CI [−42.5, −12.5] |
| E1a | + ranges calibrated on dev | 7.5% | 0.59 | inconclusive (1 image); 3 seeds queued |
| E1b | + hand-drawn imperfection | **0.0%** | 0.0002 | adopted (superiority) |
| E1c | + ballpoint/marker pen model | 0.0% | <0.0001 | adopted (non-inferiority, DR-021) |
| E1d | + hard cases and crop distractors | 0.0% | 0.0002 | adopted (non-inferiority, DR-021) |

*Figure:* `results/figures/overview/waterfall_phase1.png`.

**Robustness beyond the dev ceiling.** The dev-stress metric (DR-022) is supplementary: the same 40 real pen images under blur, low resolution, JPEG, rotation, low contrast and shadow. It keeps separating the steps after dev hits 40/40.

| Run | E0a | E0b | E0c | E1a-prior | E1a | E1b | E1c | E1d |
|---|---|---|---|---|---|---|---|---|
| Dev-stress acc | 38.3% | 66.2% | 86.7% | 88.3% | 85.4% | 94.6% | 97.1% | **98.3%** |

The shadow degradation improves most along the chain: 47% at E1a, 72% at E1b, 82% at E1c, 90% at E1d. This independently supports adopting E1c and E1d.

**What moved the needle**
1. **Correct geometry conventions (E1a-prior)** removed every letter-group error and shrank the synthetic→real gap from 31 to 5 points. This matches the report's claim that stroke-level synthesis is the core upgrade, and the baseline diagnosis that most errors were group errors.
2. **Hand-drawn imperfection (E1b)** removed the last errors: vowel under-counts on the faint, loosely cropped pen-1 batch. The first attempt was too strong. Curved, non-parallel strokes converged and made counts ambiguous, which the detector exposed by decoding only 89.5% of renders correctly. The fix (DR-018) was a polyline-distance check in the validator plus softer bows.

**What did not move it, and what hurt**
- **Calibrating the ranges to 40 real images (E1a) hurt.**
  - Over 3 seeds, CER was +2.5 pt (not significant) and NLL was **+0.23, CI [+0.05, +0.43]**, which *is* significant.
  - Fitting the generator's ranges to a small real sample narrowed them, and the probe became less well calibrated on real data.
  - This is a direct, measured instance of the domain-randomisation principle: *cover* the real variation rather than *match* a small sample of it.
  - The final table includes the prior-range version of the frozen pipeline (E2c-prior, DR-026) to test this on marker as well.
- **The pen/marker ink model (E1c)** was adopted on non-inferiority after seed 0, but over 3 seeds it is the step that made results *stable*: CER 0 / 0 / 0, with dev-stress 87.9% → 96.5%.

**Surprises**
1. **Legacy data with autocontrast preprocessing (E0c, 10%) beat legacy data with canonical (E0b, 32.5%).**
   - Canonical normalises scale to the ink box.
   - That exposes the legacy generator's too-wide stroke spacing (0.157 against a real 0.112).
   - E0b's errors are almost all "one stroke too few".
   - Controlled test queued: E1-best under both preprocessings.
2. **Dev saturated at E1b (40/40).** From here, dev can only show regressions, so further modules are judged by non-inferiority on dev (DR-021, pre-registered before seeing E1b–E1d). The held-out marker-40 test, used once, shows whether they generalise.
3. **Two preprocessing bugs** surfaced through the data checks rather than through accuracy:
   - autocontrast before the blank check (DR-019);
   - percentile-based ink detection failing below 1% ink coverage (DR-020).

**Caveats**
- Single seed.
- 40 dev images, all pen.
- The 95% CIs on a 40-image dev set are wide. Only the E0b→E1a-prior step is statistically decisive so far.

## Phase 2: representation fork (E3-ref, E3a, E3b, E3c)

*Screening, seed 0. Figure: `results/figures/overview/waterfall_phase1_3.png`.*

**Question.** Should the recogniser read the line detector's output (a canonical render of its JSON) instead of pixels? Report §9 predicts this "may matter more than any realism trick".

| Variant | Training data | Dev CER | Dev-stress | Shadow subset |
|---|---|---|---|---|
| E3-ref | none: detector + counting rules | 0.0% | 84.6% | 7% |
| E3a | oracle JSON of E1d geometry | 0.0% | 84.2% | 5% |
| E3b | oracle + simulated detector errors | 0.0% | 84.6% | 7% |
| E3c | render-then-detect (the real detector on synthetic renders) | 0.0% | 85.0% | 10% |
| *pixel path, E1d* | E1d images, canonical | 0.0% | **98.3%** | 90% |

**Findings**
1. **On clean crops, the detector representation is already solved.** Rules alone reach 100% dev and 97.5% test. Every learned E3 variant also reaches 100% dev.
2. **The detector path is exactly as robust as the detector.**
   - Every E3 variant inherits the proxy detector's failure under a shadow gradient (5–10%). Global Otsu turns the shadow into ink and the stem is lost.
   - No amount of synthetic decoder training fixes a stem that was never detected.
   - The gate therefore chose the **pixel path** as the probe's default (DR-024).
3. **Among decoder-training recipes**, render-then-detect (E3c) is the best and the most principled. It learns the detector's real artefacts without hand-set error rates. Simulated errors (E3b) make about 10% of synthetic samples genuinely ambiguous.

**Recommendation for the teammate's detector**
- Add shading correction before binarisation: background division or Sauvola thresholding.
- Use our dev-stress shadow set, plus E2b-style generated shaded images (unlimited, exactly labelled), as its regression test.
- This is the clearest example of synthetic data being most useful as a **test generator for the detector**.

## Phase 3: appearance and capture randomisation (E2a–E2d)

*Screening, seed 0.*

| Step | Adds | Dev CER | Dev-stress | Synthetic val | Decision |
|---|---|---|---|---|---|
| E1d | (base) | 0.0% | 98.3% | 99.1% | |
| E2a | paper + ink | 0.0% | 97.9% | 98.1% | adopted (non-inferiority) |
| E2b | + lighting | 0.0% | **100%** | 98.3% | adopted; shadow 88% → 100% |
| E2c | + capture (perspective, blur, noise, JPEG) | 0.0% | 100% | 99.0% | adopted (non-inferiority) |
| E2d | + artefacts and loose crops | 0.0% | 100% | **90.9%** | alarm → generator fix → E2d-v2 queued |

**Findings**
1. **With canonical preprocessing, appearance randomisation costs nothing on real data and buys robustness.**
   - Lighting (E2b) is the module that closes the remaining dev-stress gap, the shadow degradation.
   - Partly by construction, since the stress test and the module target the same effect (DR-022 caveat).
2. **The synthetic-validation alarm worked as designed (E2d).**
   - Real metrics were flat, but synthetic val dropped to 90.9%.
   - Inspection showed label-breaking warps (characters pushed out of frame) and over-strong mirrored show-through.
   - Both are fixed in the generator (DR-023), with E2d-v2 queued.
   - This is the over-randomisation failure mode the report warns about, caught before it reached a decision.
3. **Preprocessing question settled (E1d-ac).**
   - With the v2 generator, autocontrast + nearest (the teammate branch) is no better than canonical: dev 0% for both, dev-stress 97.5% against 98.3%.
   - E0c's large advantage on legacy data came from the legacy generator's spacing mismatch, which canonical exposes. It was not an advantage of autocontrast itself.

**The ceiling problem (important for the remaining plan)**
- From E1b onwards every pixel-path run scores 40/40 on dev, and from E2b onwards 240/240 on dev-stress.
- With only the 80 existing real images, the remaining planned steps have no discriminating signal on dev:
  - leave-one-out ablations;
  - dataset-size scaling;
  - Optuna tuning (E4) on dev NLL, which is now about 0.0000.
- The one unused, unbiased signal is the marker-40 test set, reserved for the final evaluation. The next step needs a decision; see the status note in the conversation.

## Phases 4–5: ablations, scaling, synthesis tuning (E4): deferred (DR-025)

From E1c onwards dev (pen-40) is at 40/40 on every seed, and from E2b onwards dev-stress is at 240/240. With only the 80 existing real images:
- leave-one-out ablations can't separate components;
- dataset-size sweeps can't separate sizes;
- Optuna has no objective, since dev NLL is about 0.

The team decided to freeze the pipeline and run the final test (DR-025). The code is ready:
- `train/tune.py` for Optuna over coarse synthesis knobs;
- `ogham_v2/legacy.py` and `--n-per-class` for size sweeps;
- per-module configs for ablations.

These resume when a harder real dev set exists. The collection protocol is in the plan: about 200 images from at least 4 writers, with phone photos, shadows and lined paper.

## Final results: held-out marker-40 (the team's wild test set)

*Computed once, after freezing (DR-025, DR-027). Full table: `results/final_table.md`. Figure: `results/figures/overview/final_test.png`. Means over 3 training seeds for the hybrid probe; paired bootstrap 95% CI against E0b, the legacy generator under the same preprocessing.*

| Pipeline | Marker-40 accuracy | CER | Δ acc vs E0b [95% CI] | Main remaining error |
|---|---|---|---|---|
| Team's reported baseline (legacy Hybrid checkpoint) | 42.5% (reported); 40.0% (our harness) | 57.5% | | wrong letter group (17/24) |
| E0a: legacy generator, legacy loader (retrained) | 35.8 ± 7.2% | 64.2% | −26.7 [−40.0, −14.2] | stroke count |
| E0b: legacy generator, canonical preprocessing | 62.5% (2 seeds) | 37.5% | reference | stroke count |
| E1a-prior: + parametric stroke model | 87.5 ± 5.4% | 12.5% | **+25.0 [+9.6, +39.2]** | stroke count (5/seed) |
| E1a: + calibrated ranges *(not adopted)* | 75.8 ± 4.2% | 24.2% | +13.3 [−3.3, +28.8] | stroke count |
| E1b: + hand-drawn imperfection | 77.5 ± 3.5% | 22.5% | +15.0 [−0.0, +30.0] | stroke count |
| **E1c: + ballpoint/marker ink model** | **100.0%** (40/40 on all 3 seeds) | **0.0%** | **+37.5 [+23.8, +51.2]** | none |
| E1d: + hard cases, crop distractors | 99.2 ± 1.2% | 0.8% | +36.7 [+22.9, +50.0] | 1 error across 3 seeds |
| E2c: + paper, ink, lighting, capture (calibrated ranges) | 98.3 ± 1.2% | 1.7% | +35.8 [+22.1, +49.6] | 2 errors across 3 seeds |
| **E2c-prior: the same pipeline with the broad prior ranges (recommended, DR-026)** | **100.0%** (40/40 on all 3 seeds) | **0.0%** | **+37.5 [+23.8, +51.2]** | none |
| E3-ref: line detector + rules (no learning) | 97.5% | 2.5% | | split thick crossing stroke |

**Conclusions**
1. **Better synthesis took the probe from 40% to 98–100% on the team's wild test set, without a single real training image.** CER fell from 57.5% to 0–1.7%. The model is unchanged. Only the training data and the preprocessing changed.
2. **Two things did almost all the work:**
   - **(a) correct stroke geometry:** the stem-anchored parametric model, +25 points;
   - **(b) an ink/pen-width model covering thick marker:** +22.5 points on marker, to 100%.

   The second was adopted under a pre-registered non-inferiority rule, because pen-only dev could not show its value. The held-out marker test vindicates that rule (DR-021).
3. **The error taxonomy moves in the expected order.**
   - Legacy models make wrong-group errors.
   - The stroke model removes them, leaving count errors on thick strokes.
   - The marker-width model removes those.
4. **Calibrating the generator to 40 real images hurt** on:
   - dev NLL;
   - marker at the E1a step: 75.8% vs 87.5%;
   - marker in the full pipeline: E2c 98.3% vs E2c-prior 100%.

   Cover the real variation broadly rather than fitting a small sample. The rule that chose the prior ranges was pre-registered on dev (DR-026); marker confirms it.
5. **Appearance and capture randomisation (E2) didn't change clean accuracy** (98–100% either way), but they made the probe robust to shadows and degradations. Dev-stress went from 96% to 100%.
6. **On these clean crops, a line detector with counting rules is as good as the learned probe** (97.5%). With the current detector, the learned detector-input path is limited by the detector's shading robustness (DR-024).
7. **Preprocessing matters as much as the generator for the legacy pipeline.** With good synthetic data, the team's autocontrast branch and our canonical preprocessing are equivalent. Canonical additionally handles blank pages safely.

**Recommended synthesis pipeline (frozen):** `configs/e2c_prior.yaml`, generated with `ogham_v2.generate`:
- parametric stroke model with label validator, using the **broad prior ranges**;
- hand-drawn imperfection;
- 50/50 ballpoint/marker ink model;
- hard cases and crop distractors;
- paper and ink, lighting, and capture randomisation;
- canonical preprocessing;
- 850 images per class (scaling not yet determinable).

**Caveats**
- The test set is 40 images from one session per batch. 95% CIs on single configurations are about ±10 points.
- Dev and test share writers. A writer-disjoint test set is the next step.
- All conclusions are for single-character crops.
- *Still to come in this table:* the second probe (plain CNN) at E0b, E1d and E2c, E3c seeds 1–2, and E0b seed 2.
