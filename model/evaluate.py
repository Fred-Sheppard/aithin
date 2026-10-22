"""Score a trained Ogham model (ConvNeXt .pt or Keras .h5) on the held-out sets in data/eval
(each a labels.csv plus images/)."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ogham_data import LETTER_TO_INDEX, LETTERS

MODEL_DIR = Path(__file__).resolve().parent


def load_set(set_dir):
    df = pd.read_csv(set_dir / "labels.csv", dtype=str)
    images = {p.stem: p for p in (set_dir / "images").iterdir()}
    return [images[n] for n in df["img_no"]], list(df["img_letter"])


def convnext_probs(model_path, paths, letters, workers):
    # Each backend's framework is imported only when used, so the Keras model runs without torch
    from torch.utils.data import DataLoader
    from ogham_convnext import OghamDataset, get_device, load_model, predict_batches

    device = get_device()
    model, _ = load_model(model_path, device)
    dataset = OghamDataset(paths, letters, model.mean, model.std)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=workers)
    probs, _, _ = predict_batches(model, loader, device, desc="evaluate")
    return probs.numpy()


def keras_probs(model_path, paths):
    """Softmax outputs, reordered from label_to_letter.json order into LETTERS order."""
    from tensorflow.keras.models import load_model
    from hybrid_model import load_image, preprocess

    model = load_model(model_path)
    with open(MODEL_DIR / "label_to_letter.json") as f:
        label_to_letter = json.load(f)
    columns = [LETTER_TO_INDEX[label_to_letter[str(i)]] for i in range(len(label_to_letter))]
    probs = np.zeros((len(paths), len(LETTERS)), dtype=np.float32)
    for start in range(0, len(paths), 32):
        batch = np.stack([preprocess(load_image(p)) for p in paths[start:start + 32]])
        probs[start:start + len(batch), columns] = model.predict(batch, verbose=0)
    return probs


def metrics(true, pred):
    return {
        "images": len(true),
        "correct": int((pred == true).sum()),
        "letter_acc": float((pred == true).mean()),
        "direction_acc": float((pred // 5 == true // 5).mean()),
        "count_acc": float((pred % 5 == true % 5).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="ConvNeXt checkpoint (.pt) or Keras model (.h5)")
    parser.add_argument("--data", default="data/eval", help="Directory of evaluation sets")
    parser.add_argument("--sets", nargs="+", default=["whiteboard", "tattoos", "pen", "marker"])
    parser.add_argument("--out", help="Write metrics and per-image predictions as JSON")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers (ConvNeXt only)")
    args = parser.parse_args()

    names, paths, letters = [], [], []
    for name in args.sets:
        p, l = load_set(Path(args.data) / name)
        names += [name] * len(p)
        paths += p
        letters += l

    if args.model.endswith(".pt"):
        probs = convnext_probs(args.model, paths, letters, args.workers)
    else:
        probs = keras_probs(args.model, paths)
    names = np.array(names)
    true = np.array([LETTER_TO_INDEX[l] for l in letters])
    pred, confidence = probs.argmax(1), probs.max(1)

    results = {name: metrics(true[names == name], pred[names == name]) for name in args.sets}
    results["total"] = metrics(true, pred)
    print(f"{'Set':12s} {'Correct':>9s} {'Letter':>7s} {'Direction':>10s} {'Count':>7s}")
    for name, m in results.items():
        print(f"{name:12s} {m['correct']:4d}/{m['images']:<4d} {m['letter_acc']:7.3f} "
              f"{m['direction_acc']:10.3f} {m['count_acc']:7.3f}")

    print("\nPer letter (correct/total, most common wrong prediction):")
    for i, letter in enumerate(LETTERS):
        rows = true == i
        if not rows.any():
            continue
        wrong = Counter(LETTERS[j] for j in pred[rows] if j != i).most_common(1)
        wrong_text = f"  -> {wrong[0][0]} ({wrong[0][1]})" if wrong else ""
        print(f"  {letter:3s} {int((pred[rows] == i).sum()):3d}/{int(rows.sum()):<3d}{wrong_text}")

    if args.out:
        output = {
            "model": str(Path(args.model).resolve()),
            "metrics": results,
            "predictions": [
                {"set": n, "image": p.name, "true": LETTERS[t], "predicted": LETTERS[q], "confidence": float(c)}
                for n, p, t, q, c in zip(names.tolist(), paths, true.tolist(), pred.tolist(), confidence.tolist())
            ],
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
