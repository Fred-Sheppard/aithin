# Aithin: Another Take on Ogham Transliteration

> Aithin (Gaeilge): To recognise or identify

## Evaluation

`data/eval` holds 119 labelled real drawings that no model is trained on, in four sets: whiteboard photos (55), tattoo photos (24), pen scans (20) and marker scans (20). Photos are reduced to at most 1600 px on the longest side.

Trained models are attached to the [models-v1 release](https://github.com/Fred-Sheppard/aithin/releases/tag/models-v1). Download the one for this version into `models/` and run, from the repository root:

```bash
pip install -r requirements.txt
python model/evaluate.py --model models/<model>.h5
```

This prints letter, direction (stroke group) and stroke-count accuracy for each set and in total. `--out results.json` also writes every prediction.

No released model combines the new preprocessing with `synthesis_v2` data. To train one on the `synthesis_v2` training set (see `synthesis_v2/README.md` to generate it), from `model/`:

```bash
python -c "from hybrid_model import train_hybrid_model; train_hybrid_model('../models/keras_preprocessing_synthesis.h5', csv_path='../synthesis_v2/data/ogham_v2/train/labels.csv', image_dir='../synthesis_v2/data/ogham_v2/train/images')"
```
