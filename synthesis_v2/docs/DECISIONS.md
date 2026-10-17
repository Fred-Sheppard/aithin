# Design decisions: Synthesis v2

Every design choice in the synthesis-v2 investigation is recorded here.
- **Status** is one of *accepted*, *superseded by DR-xxx* or *proposed*.
- **Evidence** links point to experiment cards in [EXPERIMENTS.md](EXPERIMENTS.md).
- DR-001 to DR-014 were decided in planning (2026-10-16/17), before any experiment ran.

---

## DR-001: Build a new pipeline in `synthesis_v2/`; the old pipeline is a reference only
- **Context:** `synthesis/synthesise_data.py` is the only generator the team has, and it produced the models the team has reported on.
- **Options:**
  - (a) refactor it in place;
  - (b) write a new pipeline beside it.
- **Decision:** (b). The old pipeline is never modified. It is run unchanged, through `ogham_v2/legacy.py`, as the E0 baseline.
- **Rationale:** the baseline stays reproducible and comparable. Old and new can be compared side by side, and nobody else's workflow breaks.
- **Status:** accepted.

## DR-002: The existing recogniser is a fixed measuring *probe*, and a second probe confirms the milestones
- **Context:** a better recogniser is being built separately. This investigation is about the data.
- **Decision:**
  - The CNN→BiLSTM in `model/hybrid_model.py:61-89` is reproduced exactly in `train/model.py` and never changed.
  - Only the training data and the input representation vary.
  - A second, structurally different probe (`train/model_alt.py`, a plain CNN) is run at 3 milestones (E0b, the Phase 3 best, and E4).
- **Rationale:**
  - If a synthesis change improves a fixed model's accuracy on real data, it improved generalisation.
  - The second probe checks that a conclusion is not an artefact of one architecture, so it is more likely to transfer to the new model.
- **Status:** accepted.

## DR-003: Real data roles. Dev = pen-40, test = marker-40 (evaluated once)
- **Context:**
  - There are only 80 labelled real images.
  - marker-40 is the team's "wild test set", where the legacy Hybrid model scores 42.5%.
- **Options:**
  - (a) dev = pen, test = marker;
  - (b) mixed batches on both sides;
  - (c) dev = marker, test = pen.
- **Decision:** (a).
  - Every decision, calibration measurement and Optuna trial uses pen-40.
  - marker-40 is evaluated once, on frozen phase winners.
  - A blank page is run alongside as a control.
- **Rationale:**
  - The reported benchmark stays untouched and directly comparable.
  - Tuning on pen and testing on marker is a genuine generalisation test.
- **Guard:** the dev medium (pen) differs from the test medium (marker). To stop tuning favouring pen:
  - the ballpoint/marker width mixture is fixed at 50/50;
  - the marker width comes from priors;
  - neither is ever tuned.
- **Status:** accepted.

## DR-004: CER is the primary metric, NLL breaks ties
- **Context:**
  - The team reports CER, accuracy, top-3, macro-F1, letter-group (aicme) accuracy and mean confidence.
  - With 40 dev images, CER moves in steps of 2.5 points.
- **Decision:**
  - CER (1 − accuracy for single characters) is the headline metric.
  - A change is **adopted** if mean dev CER falls and dev NLL does not worsen.
  - If CER ties, NLL decides.
  - Otherwise the change is **inconclusive**, and the simpler configuration is kept.
  - Paired bootstrap CIs over images are reported alongside.
- **Status:** accepted.

## DR-005: Canonical preprocessing is shared by synthetic and real images
- **Context:**
  - The legacy loader squashes images straight to 128×128 with nearest-neighbour resampling.
  - The real images are 365–4235 px, so thin pen strokes can disappear.
- **Decision:** `ogham_v2/preprocess.py`, in this order:
  - grayscale;
  - autocontrast (cutoff 1%, consistent with branch `image-preprocessing-investigation`);
  - robust ink bounding-box crop (Otsu, ignoring components that touch the border);
  - 10% margin;
  - pad to square, keeping the aspect ratio;
  - INTER_AREA resize to 128.
- **Evidence:**
  - E0a (legacy loader) vs E0b (canonical), with the same data, isolates the gain from preprocessing alone.
  - E0c (the teammate's autocontrast + nearest squash) was added on 2026-10-17 after the E0-ckpt study. On frozen checkpoints, preprocessing alone is worth +35 to +55 dev points. Autocontrast + nearest turns a blank page into a confident prediction (R at 1.00); canonical does not.
- **Status:** accepted, pending E0a/E0b/E0c retraining on dev.

## DR-006: Two virtual environments
- **Decision:** synthesis runs in `.venv-synth` (numpy, OpenCV, Augraphy, Albumentations). Training runs in `.venv-train` (TF 2.18, tensorflow-metal).
- **Interface:** the only contract between them is the dataset folder: `labels.csv`, `cache.npz`, `meta.jsonl` and `oracle.jsonl`.
- **Rationale:** this avoids numpy and dependency pin conflicts. The generator stays independent of any model.
- **Status:** accepted.

## DR-007: Stroke model in stem coordinates with a label-preserving validator
- **Context:**
  - Most of the legacy model's errors are in the *letter group*: 16 of the 23 errors on marker-40.
  - The legacy geometry has two problems:
    - left strokes are longer than right ones, because of clipping;
    - diagonals are sheared and not anchored to the stem.
- **Decision:**
  - Characters are sampled as parametric trajectories in stem-normalised coordinates.
  - Every sample passes a validator:
    - stroke count is correct;
    - side is correct, with overshoot capped;
    - crossing strokes extend at least a minimum length on both sides;
    - strokes are a minimum distance apart;
    - diagonals are ≥22° from horizontal;
    - across strokes are ≤12° from horizontal.
- **Status:** accepted.

## DR-008: The representation fork (E3a/E3b) comes before the appearance work (E2)
- **Decision:** right after E1, test whether training on canonical renders of the line detector's JSON beats raw pixels.
- **Rationale:** if the detector path wins, most pixel-appearance randomisation stops mattering to the recogniser (report §3, §9). It then only matters for render-then-detect (E3c), so the E2 effort is sized accordingly.
- **Status:** accepted.

## DR-009: Dataset size is set by scaling studies, not fixed in advance
- **Context:** the legacy default is 850 images per class.
- **Evidence from the literature:** quantity alone plateaus, while diversity keeps paying off (BeyondWeb, arXiv 2508.10975; aerial-view human detection, arXiv 2405.15939; SynthTIGER).
- **Decision:**
  - Run three sweeps:
    - S0 on legacy data: 850 / 5,000;
    - S1 after E1: 850 / 2,500 / 5,000;
    - S2 after Phase 3: 850 / 2,500 / 5,000, plus 10,000 or on-the-fly generation.
  - N\* is the smallest size within the CI of the best one.
- **Status:** accepted. N\* is to be recorded here when each sweep completes.

## DR-010: No synthetic test set; a separately generated synthetic validation set instead
- **Decision:**
  - Drop the legacy random 10% split.
  - Generate a separate validation set with the same config and a different seed: 50 per class, 1,000 images.
  - It is used only for early stopping and LR scheduling, as a label-breakage alarm, and to measure the synthetic→real gap.
- **Rationale:**
  - The real images are the only meaningful test.
  - A separately generated set costs no training data and avoids near-duplicate leakage.
- **Status:** accepted.

## DR-011: Early stopping on the synthetic validation set only
- **Decision:**
  - Check synthetic validation loss every 500 steps.
  - Patience is 4 checks. ReduceLROnPlateau (×0.2) triggers after 2 checks.
  - Training is capped at 8,000 steps. The best synthetic-val weights are restored.
  - Real dev metrics are logged at every check, for learning curves only, and never used to choose checkpoints.
- **Rationale:**
  - Choosing checkpoints by real score would quietly tune every run on 40 images.
  - Overfitting to the synthetic domain is a property of the data under test, so it should show up in real CER.
- **Status:** accepted. The cap and patience are to be confirmed against the E0b learning curves.

## DR-012: Label override for `ogham-pen-labelled/00022.png` (Z → R)
- **Evidence:** visual inspection shows 5 diagonal strokes. Z has 4, R has 5.
- **Decision:** record the correction in `eval/label_overrides.csv`. The original CSV is never edited.
- **Status:** accepted. Other audit findings are appended to the override file and noted here.

## DR-013: E5 (self-training), E6 (realism models) and E7 (fine-tune ceiling) are deferred
- **Context:** each of them needs real data that does not exist yet: unlabelled photos, or more labelled images.
- **Decision:** out of scope until data is collected. The collection protocol is in the plan.
- **Status:** accepted.

## DR-014: Compute plan (M2 laptop, 8 cores, 16 GB)
- **Benchmark (2026-10-17, `tools/benchmark.py`):** steps/s for each probe and device.

| Probe | Device | b=32 | b=64 | b=128 | Worst case for 8k steps at b=32 |
|---|---|---|---|---|---|
| hybrid (BiLSTM) | Metal GPU | 7.8 (248 img/s) | 4.7 (301) | 2.8 (357) | 17 min |
| hybrid (BiLSTM) | CPU | 5.6 (180 img/s) | 3.0 (193) | 1.5 (198) | 24 min |
| alt (plain CNN) | Metal GPU | 22.3 (713 img/s) | 11.6 (743) | 6.8 (875) | 6 min |
| alt (plain CNN) | CPU | 6.4 (205 img/s) | 3.2 (205) | 1.6 (200) | 21 min |

- **Decision:**
  - Train on the **Metal GPU at batch 32**. Larger batches raise throughput by less than 45%, and batch 32 is the original probe setting.
  - The step cap stays at **8k steps (256k images)**, so the worst case is about 17–22 minutes per hybrid run once validation and dev logging are included. Early stopping usually finishes sooner.
  - That puts the project in the "≤20 min, run as planned" **compute tier**. No Colab is needed for E0–E4.
- **Generation:** legacy data generates at 850/class (17k images) in 23 s, plus 16 s to build caches across 8 processes.
- **Status:** accepted (2026-10-17).

## DR-015: The evaluation harness is accepted after reproducing the reported baseline
- **Evidence:** E0-ckpt.
  - Augmented reproduces exactly (6/40, CER 85.0%, aicme 45.0%).
  - Hybrid is within one image (16/40 against 17/40).
  - Mean confidence is about 3–5 points higher than reported, probably because of a different checkpoint file or resize path in the team's script, which is unavailable.
- **Decision:** accept the harness. This was time-boxed, as the plan specified. Any later run is compared within our harness, never against the external table.
- **Status:** accepted (2026-10-17).

## DR-016: Add E0c (legacy data + autocontrast preprocessing) to Phase 0
- **Context:** the teammate's `image-preprocessing-investigation` branch is a live candidate for the team's preprocessing. E0-ckpt shows it is very effective on dev, but it is unsafe on blank input.
- **Decision:** retrain on legacy data under all three preprocessing modes (E0a legacy, E0b canonical, E0c autocontrast) and choose on dev, using the DR-004 rule. The blank-page control is reported for each.
- **Status:** accepted (2026-10-17).

## DR-017: Proxy line detector as a stand-in for the teammate's detector
- **Context:** the teammate's detector code is not available yet. Calibration (Phase 1) and the representation fork (E3) both need a detector that emits their JSON schema.
- **Decision:** `ogham_v2/detectors/proxy.py`. It runs these steps in order:
  - autocontrast, then ink-bbox crop;
  - Otsu threshold, then skeleton;
  - stem = strongest near-vertical Hough line with a long supporting run;
  - remove the stem band, then fit a PCA line to each remaining skeleton piece;
  - pair collinear opposite-side pieces into "across" strokes.

  Its output schema is identical to the teammate's. `detectors/teammate.py` is the adapter slot for the real detector.
- **Tuning discipline:** two rounds of fixes were driven only by *dev* errors:
  - a lower Hough threshold and a wider band for wavy stems;
  - pieces parallel to the stem are dropped.

  After that the detector was frozen as "proxy v1" and scored once on test (E3-ref). It must not be tuned against test.
- **Validation:**
  - 99–99.5% letter recovery on clean E1a renders, against a target of ≥98%;
  - 100% on dev;
  - 97.5% on test (frozen).
- **Status:** accepted (2026-10-17).

## DR-018: The label validator checks stroke convergence; the across/side rule is shared with the oracle
- **Evidence:** generator QA, recorded on the E1b–E1d cards.
  - The first E1b attempt had curvature 0.06 and wave 0.012, and the validator only checked anchor spacing on the stem.
  - The proxy detector decoded only 89.5% (E1b), 80.5% (E1c) and 76.5% (E1d) of QA renders correctly, against 100% on real dev.
  - Inspection showed curved, slightly non-parallel strokes *converging* at their far ends, which makes the stroke count genuinely ambiguous.
  - Separately, a property test found that side strokes with a permitted overshoot were being labelled "across" in the oracle JSON.
- **Decision:**
  1. The validator rejects any pair of stroke polylines closer than max(0.035, 1.6 × pen width), measured anywhere along their length.
  2. E1b imperfection is softened to curvature 0.035, wave 0.008 and stem curvature 0.02, which look like the real pen strokes.
  3. A stroke counts as "across" only if its shorter half is at least max(0.055, 12% of the stroke). That threshold sits between the maximum side overshoot (0.05) and the minimum crossing half (0.06). The oracle and the proxy detector use the same rule.
- **Result:** detector decode on QA renders rises to E1b 99.0%, E1c 97.5% and E1d 91.0%. Real data scores 97.5–100%, so synthetic data is no longer "harder than reality", apart from the deliberate E1d hard cases.
- **Status:** accepted (2026-10-17).

## DR-019: Blank-page check runs on the raw image, before autocontrast
- **Evidence:**
  - With autocontrast first, a blank page's sensor noise is stretched to full range and treated as ink.
  - The proxy detector then "found" an N on the blank page.
  - Canonical preprocessing turned the blank page into a noise crop.
- **Decision:** `preprocess.is_blank` (1–99 percentile range < 40 on the raw grayscale) runs before autocontrast in canonical preprocessing and in the proxy detector.
- **Verification:** canonical outputs for all 80 real images are byte-identical before and after the change, so no cached result changes. The blank page now maps to pure white, and the detector reports nothing.
- **Note:** training runs that had already started (E0a-s0, E0b-s0) loaded the old code, so their *blank-control* values used the old behaviour. Their image predictions are unaffected.
- **Status:** accepted (2026-10-17).
