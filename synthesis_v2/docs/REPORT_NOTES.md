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
_(to be written)_

## Phase 1: stroke model (E1)
_(to be written)_

## Phase 2: representation fork (E3-ref, E3a, E3b)
_(to be written)_

## Phase 3: appearance randomisation (E2) and render-then-detect (E3c)
_(to be written)_

## Phase 4: ablations
_(to be written)_

## Phase 5: synthesis tuning (E4)
_(to be written)_

## Final results and recommended pipeline
_(to be written)_
