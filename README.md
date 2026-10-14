# Aithin: Another Take on Ogham Transliteration

> Aithin (Gaeilge): To recognise or identify

## Evaluation

`data/eval` holds 119 labelled real drawings that no model is trained on, in four sets: whiteboard photos (55), tattoo photos (24), pen scans (20) and marker scans (20). Photos are reduced to at most 1600 px on the longest side.

Trained models are attached to the [models-v1 release](https://github.com/Fred-Sheppard/aithin/releases/tag/models-v1). Download the one for this version into `models/` and run, from the repository root:

```bash
pip install -r requirements.txt
python model/evaluate.py --model models/keras_preprocessing.h5
```

This prints letter, direction (stroke group) and stroke-count accuracy for each set and in total. `--out results.json` also writes every prediction.
