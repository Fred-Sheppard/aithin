# Synthesis v2: Ogham training-data generator

This generates synthetic images of handwritten Ogham characters for training a recogniser. Each character is drawn as a set of pen strokes along a stem, with these elements randomised:
- stroke geometry;
- hand-drawn imperfections;
- ballpoint or marker ink;
- paper;
- lighting;
- camera and scanner effects.

Every sample passes a check that it is still exactly the labelled letter. The old generator in `../synthesis/` is untouched.

## 1. Install
This needs Python 3.11 or newer (numpy 2.4 has no wheels for older versions; tested with 3.14). Check that `python3 --version` is new enough, or name the interpreter explicitly.
```bash
cd synthesis_v2
python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt     # numpy, opencv-python-headless, pillow, PyYAML
```

## 2. Generate a dataset
```bash
.venv/bin/python -m ogham_v2.generate --out data/ogham_v2 --save-images all
```
With the defaults this takes about 3–5 minutes on 8 cores. It produces:
- **train:** 850 images per letter × 20 letters = **17,000 images**;
- **val:** 50 per letter = **1,000 images**, generated with a separate seed.

That's 18,000 images in total. Change the counts with `--n-per-class` and `--val-n`.

Useful flags:

| Flag | Default | Effect |
|---|---|---|
| `--config` | `default` | Which config in `configs/` to use |
| `--n-per-class` | 850 | Training images per letter |
| `--val-n` | 50 | Validation images per letter |
| `--seed` | 0 | Random seed |
| `--workers` | 8 | Number of parallel processes |
| `--save-images all` | off | Write every image as a PNG, about 1–2 GB |
| `--cache-modes` | `canonical` | Which preprocessed 128×128 arrays to write: `canonical` and/or `autocontrast` |

Output, written for both `data/ogham_v2/train/` and `data/ogham_v2/val/`:

| File | Contents |
|---|---|
| `labels.csv` | `img_no,img_letter,type`. Same columns and numbering as the legacy generator's `output/labels.csv` |
| `images/00000.png …` | full-resolution renders named `f"{img_no:05}.png"` (with `--save-images all`) |
| `cache_canonical.npz` | `X`: uint8 N×128×128, already preprocessed; `y`: label index in `../model/label_to_letter.json` order (N S F L B Q C T D H R Z NG G M I E U O A) |
| `qa/<LETTER>_<i>.png` | 10 renders per letter for a quick visual check |
| `meta.jsonl` | the sampled generator parameters of each image |

The dataset folder also gets a `dataset.json` recording the resolved config, the seeds and the generator's git commit.

## 3. Train
There are two ways to load the data:
- **From the PNGs:** `train/labels.csv` + `train/images/` work with any loader written for the legacy `output/` folder.
- **From the arrays:** load `cache_canonical.npz` directly.
  ```python
  z = np.load("data/ogham_v2/train/cache_canonical.npz")
  X = z["X"][..., None] / 255.0
  y = z["y"]
  ```

Use `val/` for early stopping and model selection. Keep the real photos for testing only.

## 4. Inference: preprocess real images the same way
Real photos must go through **the same preprocessing as the training data**. The cached arrays use `canonical`:
```python
from ogham_v2.preprocess import canonical
x = canonical("photo.png") / 255.0          # grayscale, autocontrast, crop to the ink, pad square, resize to 128
probs = model.predict(x[None, ..., None])
```

**Training from the PNGs:** apply `canonical` to them when loading.

**Keeping `preprocess_image` from branch `image-preprocessing-investigation`** (autocontrast + nearest-neighbour resize): generate with `--cache-modes autocontrast`, or apply that same function to the PNGs.

Whichever you choose, use it for both training and prediction. Mixing them hurts accuracy, and so does the legacy `load_img(target_size=(128,128))` squash: it can erase thin pen strokes and leaves scanned ink grey.

## 5. Customise
Make a new config that inherits the default and overrides only what you change:
```yaml
# configs/thin_pens.yaml
base: default
pen:
  p_marker: 0.2
```
Then run with `--config thin_pens`.

Every setting is documented in [`configs/default.yaml`](configs/default.yaml).

Check a config before using it:
```bash
.venv/bin/python tools/test_generator.py --config thin_pens                                     # labels stay valid, renders work, deterministic
.venv/bin/python tools/contact_sheet.py --dataset data/ogham_v2/train --out sheet.png              # eyeball 8 samples per letter
```

## Code map

| File | What it does |
|---|---|
| `ogham_v2/geometry.py` | stroke model in stem coordinates, hand-drawn imperfection, hard cases, distractors, and the **label validator** |
| `ogham_v2/pen.py` | ballpoint/marker pen model and the supersampled variable-width rasteriser |
| `ogham_v2/appearance.py` | paper, ink colour, lighting, capture (perspective, blur, noise, JPEG) and artefacts |
| `ogham_v2/preprocess.py` | `canonical` and `autocontrast` preprocessing (shared by training and inference) |
| `ogham_v2/generate.py` | the CLI: parallel generation, seeding, output files |
| `ogham_v2/config.py`, `alphabet.py`, `paths.py` | config loading (with `base:` inheritance), letter specs and label map, paths |
| `configs/default.yaml` | the recommended settings, with comments |
| `tools/test_generator.py`, `tools/contact_sheet.py` | property tests and the visual check |
