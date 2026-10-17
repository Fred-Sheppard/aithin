# Synthesis v2

This is a new Ogham data-synthesis pipeline, together with the experiments that measure how well its data generalises to real hand-drawn Ogham. The old pipeline in `../synthesis/` is left untouched. It is only used as a reference and as the E0 baseline.

The documents to read are:
- **`docs/EXPERIMENTS.md`**: one card per experiment, giving the hypothesis, the change, the results and the decision.
- **`docs/DECISIONS.md`**: the design-decision records (DR-xxx).
- **`docs/REPORT_NOTES.md`**: a narrative per phase, ready to use in the project report.
- **`docs/results/`**: `ledger.csv` holds every run, `summary.md` holds the tables, and `figures/` holds contact sheets, learning curves, confusion matrices and calibration plots.

## Setup (two virtual environments, DR-006)

```bash
python3.11 -m venv .venv-synth && .venv-synth/bin/pip install -r requirements-synth.txt
python3.11 -m venv .venv-train && .venv-train/bin/pip install -r requirements-train.txt
```

## Pipeline

Generation and training are separate steps.

```
configs/<exp>.yaml ──> ogham_v2.generate (synth venv)
                         geometry.py        stroke model in stem coordinates + label validator   (E1a/E1b/E1d)
                         pen.py             ballpoint/marker rasteriser                           (E1c)
                         appearance.py      paper, ink, lighting, capture, artefacts              (E2a-E2d)
                         detector_schema.py exact "oracle" line-detector JSON per sample
                         canonical_render.py + detector_errors.py / detectors/proxy.py            (E3a-E3c)
                         preprocess.py      canonical preprocessing (shared with real images)
                     ──> data/<dataset>/{train,val}/cache_<mode>.npz, labels.csv, meta.jsonl, oracle.jsonl, images/
                     ──> train/train.py (train venv): fixed probe, early stopping on synthetic val only
                     ──> runs/<exp>/<run>/ + docs/results/ledger.csv
                     ──> train/report.py: experiment card tables and figures
```

## Common commands (run from `synthesis_v2/`)

```bash
# generate a dataset (train 850/class + separately seeded val 50/class)
.venv-synth/bin/python -m ogham_v2.generate --config e1c --out data/e1c_n850_s0 --n-per-class 850 --seed 0

# the legacy generator, unmodified, seeded from outside (E0)
.venv-synth/bin/python -m ogham_v2.legacy --out data/legacy_n850_s0 --n-per-class 850 --seed 0

# visual QA gate: contact sheet
.venv-synth/bin/python tools/contact_sheet.py --dataset data/e1c_n850_s0/train --out sheet.png

# property tests (validator + oracle round-trip)
.venv-synth/bin/python tools/test_generator.py --configs e1a e1b e1c e1d

# train the probe (dev = pen-40 is logged; test = marker-40 is never touched here)
.venv-train/bin/python train/train.py --data data/e1c_n850_s0 --exp E1c --parent E1b --seed 0 --input canonical

# fill in the experiment card and the summary
.venv-train/bin/python train/report.py --exp E1c --parent E1b
.venv-train/bin/python train/report.py --summary

# test-set evaluation: frozen phase winners only (DR-003)
.venv-train/bin/python train/evaluate.py --run runs/E1c/hybrid-canonical-s0 --split test

# overnight queue (resumable; touch runs/queue/<name>/STOP to stop after the current job)
.venv-train/bin/python tools/run_queue.py queues/screen1.yaml

# line detector + rule decoder (E3-ref), calibration and detector error statistics
.venv-synth/bin/python tools/e3ref_eval.py --split dev
.venv-synth/bin/python tools/measure_real.py --synthetic data/e1a_n850_s0/train
.venv-synth/bin/python tools/measure_detector_errors.py --config e1c
```

## Real data (DR-003, DR-012)

| Role | Set | Size |
|---|---|---|
| dev | `../../ogham-pen-labelled` | 40 images |
| test | `../../ogham-marker-labelled` | 40 images (the team's wild test set) |

- Dev drives every decision.
- Test is evaluated only once, on frozen winners.
- Label corrections live in `eval/label_overrides.csv`. The original CSVs are never edited.
