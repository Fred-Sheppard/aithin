# Synthesis v2: Ogham training-data generator

A parametric generator of synthetic handwritten Ogham characters, plus the experiments that measured how well its data generalises to real images. The old pipeline in `../synthesis/` is untouched.

**Headline result.** Trained only on data from the recommended config, the unchanged Hybrid CNN-BiLSTM scores **100% on the team's wild test set** (marker-40), on all 3 training seeds.
- The legacy generator scores 56.7% under the same preprocessing.
- The team's originally reported figure was 42.5%.
- A structurally different plain CNN also goes from 19.2% to 100%.

Details: [`docs/REPORT_NOTES.md`](docs/REPORT_NOTES.md) and [`docs/results/final_table.md`](docs/results/final_table.md).

## Quickstart

### 1. Install the generator environment
Python 3.10–3.12.
```bash
cd synthesis_v2
python3.11 -m venv .venv-synth && .venv-synth/bin/pip install -r requirements-synth.txt
```

### 2. Generate training data with the recommended config
```bash
.venv-synth/bin/python -m ogham_v2.generate --config e2c_prior --out data/ogham_v2 \
    --n-per-class 850 --seed 0 --save-images all
```
This produces 17,000 training images plus 1,000 separately seeded validation images, in about 3–5 minutes on 8 cores. With `--save-images all`, expect roughly 1–2 GB of PNGs.

Output layout, for both `data/ogham_v2/train/` and `data/ogham_v2/val/`:

| File | Contents |
|---|---|
| `labels.csv` | `img_no,img_letter,type`. Same columns and numbering as the legacy generator's `output/labels.csv` |
| `images/00000.png …` | every native-resolution render, named `f"{img_no:05}.png"` (only with `--save-images all`). Drop-in for loaders like `model/hybrid_model.py` |
| `cache_canonical.npz` | `X` (uint8, N×128×128, already preprocessed) and `y` (label index, `model/label_to_letter.json` order) |
| `qa/<LETTER>_<i>.png` | 10 renders per letter for visual checks |
| `meta.jsonl`, `oracle.jsonl` | per-image generator parameters and exact line-detector JSON |

Label indices always follow `../model/label_to_letter.json`: N S F L B Q C T D H R Z NG G M I E U O A.

### 3. Train a model
Two options:
- **Any model, from the PNGs:** point your loader at `train/labels.csv` + `train/images/`. Use `val/` for early stopping. Do not use the real images for that (see DR-010/DR-011).
- **The included probe trainer**, which reads the 128×128 caches directly:
  ```bash
  python3.11 -m venv .venv-train && .venv-train/bin/pip install -r requirements-train.txt
  .venv-train/bin/python train/train.py --data data/ogham_v2 --exp my-run --seed 0 --input canonical
  ```

### 4. At inference, preprocess real images the same way as the training data
This step matters more than any other. The legacy loader (`load_img(target_size=(128,128))`, a nearest-neighbour squash) costs 35–55 accuracy points on real photos (`docs/EXPERIMENTS.md`, card E0-ckpt).

```python
from ogham_v2.preprocess import canonical          # needs numpy, opencv-python(-headless), pillow
x = canonical("photo.png") / 255.0                 # uint8 128x128 -> float [0,1]; ink-box crop, pad, INTER_AREA
probs = model.predict(x[None, ..., None])
```

Pick one preprocessing and use it for both training and inference:

| Preprocessing | Training input | Inference | Blank page |
|---|---|---|---|
| **canonical (recommended)** | `cache_canonical.npz`, or apply `preprocess.canonical` to the PNGs | `preprocess.canonical` | handled safely |
| autocontrast + nearest | generate with `--cache-modes autocontrast`, or apply the same function to the PNGs | `model/image_preprocessing.py::preprocess_image` on branch `image-preprocessing-investigation` | maps to a confident letter |

On the v2 data the two score the same on dev (card E1d-ac). Mixing them, for example training on canonical and predicting with autocontrast, changes the framing and will hurt.

## Configs

`configs/*.yaml` use `base:` inheritance. Each step adds one module on top of the previous one.

| Config | Adds | Status |
|---|---|---|
| `e1a_prior` | parametric stroke model in stem coordinates + label validator | adopted |
| `e1a` | ranges calibrated on real pen images | **not adopted** (hurt; DR-026) |
| `e1b` | hand-drawn imperfection | adopted |
| `e1c` | ballpoint/marker ink model | adopted (key for marker) |
| `e1d` | hard cases, crop distractors | adopted |
| `e2a`–`e2c` | paper and ink, lighting, capture | adopted |
| `e2d` | artefacts, loose crops | not adopted (over-randomised; DR-027) |
| **`e2c_prior`** | `e2c` with the broad prior geometry ranges | **recommended** |
| `e3a`–`e3c` | line-detector representation (canonical render of detector JSON) | detector track (DR-024) |

## Repository map

| Path | What |
|---|---|
| `ogham_v2/` | the generator: `geometry.py` (stroke model + validator), `pen.py` (ink rasteriser), `appearance.py` (paper/lighting/capture), `preprocess.py`, `generate.py`, `legacy.py` (runs the old generator unmodified), `detector_schema.py`, `canonical_render.py`, `detectors/` (proxy line detector) |
| `train/` | the probe models, trainer (early stopping on synthetic validation data only), evaluation, report cards, Optuna tuner, final-table builder |
| `tools/` | contact sheets, generator property tests, calibration, dev-stress robustness test, detector checks, overnight queue runner and watchdog, benchmark |
| `eval/` | real-data roles (dev = pen-40, test = marker-40) and label overrides |
| `docs/` | `EXPERIMENTS.md` (one card per experiment), `DECISIONS.md` (DR-001…027), `REPORT_NOTES.md` (narrative and conclusions), `results/` (ledger, tables, figures) |

## Reproducing the experiments

```bash
.venv-synth/bin/python tools/test_generator.py --configs e2c_prior                         # label-validator property test
.venv-synth/bin/python tools/contact_sheet.py --dataset data/ogham_v2/train --out sheet.png  # visual QA
.venv-synth/bin/python -m ogham_v2.legacy --out data/legacy_n850_s0                          # old generator (E0)
.venv-train/bin/python tools/run_queue.py queues/screen1.yaml                                 # queued training runs
.venv-train/bin/python train/report.py --exp E1c --parent E1b                                 # fill an experiment card
.venv-train/bin/python tools/stress_eval.py --all                                             # dev-stress robustness
.venv-train/bin/python train/final_table.py --evaluate                                        # held-out marker-40 table
```

- **Hardware:** an M2 Mac using `tensorflow-metal` on the GPU, about 15 minutes per training run.
- **Laptop power:** keep the laptop on AC power. macOS throttles background GPU jobs on battery. `tools/run_queue.py` lifts that throttling and runs `caffeinate` for each job.
- **Real data:** the real images (`../../ogham-*-labelled/`) are not part of this repository. Paths are in `ogham_v2/paths.py`.
