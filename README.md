# Aithin: Another Take on Ogham Transliteration

> Aithin (Gaeilge): To recognise or identify

## Evaluation

`data/eval` holds 119 labelled real drawings that no model is trained on, in four sets: whiteboard photos (55), tattoo photos (24), pen scans (20) and marker scans (20). Photos are reduced to at most 1600 px on the longest side.

Trained models are attached to the [models-v1 release](https://github.com/Fred-Sheppard/aithin/releases/tag/models-v1). Download the one for this version into `models/` and run, from the repository root:

```bash
pip install -r requirements.txt
python model/evaluate.py --model models/convnext_adapted.pt
```

This prints letter, direction (stroke group) and stroke-count accuracy for each set and in total. `--out results.json` also writes every prediction. Keras models (`.h5`) are evaluated the same way, using this version's preprocessing.

## ConvNeXt recogniser

`model/ogham_convnext.py` fine-tunes a pretrained ConvNeXt-Tiny (`convnext_tiny.fb_in22k`, via timm) with two cosine-classifier heads: one for the stroke direction (the letter's group) and one for the stroke count. A letter's score is the sum of the two log-probabilities. Training uses only synthetic images, heavily augmented to look like real drawings; adaptation then fine-tunes briefly on the 69 labelled real drawings in `data/support`, after imprinting the heads with their mean features.

Released models: `convnext_synthetic.pt` (synthetic training only) and `convnext_adapted.pt` (after adaptation), both trained on the `synthesis_v2` training set. To train them yourself, generate that set (see `synthesis_v2/README.md`), then:

```bash
python model/train.py --data synthesis_v2/data/ogham_v2/train    # models/convnext_synthetic.pt
python model/adapt.py --data synthesis_v2/data/ogham_v2/train    # models/convnext_adapted.pt
```

Set `OGHAM_STATUS_FILE=status.txt` to mirror progress bars to a file. To predict a single image, run `python model/client.py --image <image> --model models/convnext_adapted.pt`, or run `python model/draw_interface.py` to draw a letter.
