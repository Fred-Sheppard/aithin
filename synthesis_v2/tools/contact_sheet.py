"""Contact sheet of a generated dataset: one row per letter (label-map order), N preprocessed samples per row.

  python tools/contact_sheet.py --dataset data/ogham_v2/train [--per-class 8] [--cache canonical] --out sheet.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ogham_v2.alphabet import INDEX, LETTERS  # noqa: E402

CELL = 128
LABEL_W = 44


def grid(rows: list, row_labels: list) -> Image.Image:
    ncol = max(len(r) for r in rows)
    sheet = Image.new("L", (LABEL_W + ncol * (CELL + 4), len(rows) * (CELL + 4)), 255)
    d = ImageDraw.Draw(sheet)
    for r, (cells, label) in enumerate(zip(rows, row_labels)):
        y = r * (CELL + 4)
        d.text((4, y + CELL // 2), label, fill=0)
        for c, img in enumerate(cells):
            x = LABEL_W + c * (CELL + 4)
            sheet.paste(Image.fromarray(img).resize((CELL, CELL)), (x, y))
            d.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline=180)
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True, help="a split directory, e.g. data/ogham_v2/train")
    ap.add_argument("--cache", default="canonical")
    ap.add_argument("--per-class", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    z = np.load(a.dataset / f"cache_{a.cache}.npz")
    X, y = z["X"], z["y"]
    rng = np.random.default_rng(0)
    rows = []
    for letter in LETTERS:
        idx = np.flatnonzero(y == INDEX[letter])
        pick = rng.choice(idx, size=min(a.per_class, len(idx)), replace=False) if len(idx) else []
        rows.append([X[i] for i in pick] or [np.full((CELL, CELL), 255, np.uint8)])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    grid(rows, LETTERS).save(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
