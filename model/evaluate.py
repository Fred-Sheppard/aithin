"""Score a trained Ogham model on the held-out sets in data/eval (each a labels.csv plus images/)."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent
LETTER_TABLE = [
    ["B", "L", "F", "S", "N"],
    ["H", "D", "T", "C", "Q"],
    ["M", "G", "NG", "Z", "R"],
    ["A", "O", "U", "E", "I"],
]
LETTERS = [letter for row in LETTER_TABLE for letter in row]
LETTER_TO_INDEX = {letter: i for i, letter in enumerate(LETTERS)}


def load_set(set_dir):
    df = pd.read_csv(set_dir / "labels.csv", dtype=str)
    images = {p.stem: p for p in (set_dir / "images").iterdir()}
    return [images[n] for n in df["img_no"]], list(df["img_letter"])


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
    parser.add_argument("--model", required=True, help="Trained model (.h5)")
    parser.add_argument("--data", default="data/eval", help="Directory of evaluation sets")
    parser.add_argument("--sets", nargs="+", default=["whiteboard", "tattoos", "pen", "marker"])
    parser.add_argument("--out", help="Write metrics and per-image predictions as JSON")
    args = parser.parse_args()

    names, paths, letters = [], [], []
    for name in args.sets:
        p, l = load_set(Path(args.data) / name)
        names += [name] * len(p)
        paths += p
        letters += l

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
