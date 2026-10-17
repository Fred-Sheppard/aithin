"""Contact sheet: rows = letters (label-map order), columns = samples. Visual QA gate before any training run.

  python tools/contact_sheet.py --real dev|test|all [--mode canonical|legacy|raw] --out sheet.png
  python tools/contact_sheet.py --dataset data/<exp>/s0/train --per-class 8 --out sheet.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ogham_v2 import preprocess, real_data  # noqa: E402
from ogham_v2.alphabet import LETTERS, INDEX  # noqa: E402

CELL = 128
LABEL_W = 44


def grid(rows: list[list[np.ndarray]], row_labels: list[str], captions=None) -> Image.Image:
    ncol = max(len(r) for r in rows)
    sheet = Image.new("L", (LABEL_W + ncol * (CELL + 4), len(rows) * (CELL + 16)), 255)
    d = ImageDraw.Draw(sheet)
    for r, (cells, label) in enumerate(zip(rows, row_labels)):
        y = r * (CELL + 16)
        d.text((4, y + CELL // 2), label, fill=0)
        for c, img in enumerate(cells):
            x = LABEL_W + c * (CELL + 4)
            sheet.paste(Image.fromarray(img).resize((CELL, CELL)), (x, y))
            d.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline=180)
            if captions and captions[r][c]:
                d.text((x + 2, y + CELL + 1), captions[r][c], fill=0)
    return sheet


def real_sheet(split: str, mode: str) -> Image.Image:
    items = real_data.split_items(split) if split != "all" else real_data.load_items("marker") + real_data.load_items("pen")
    by = {l: [] for l in LETTERS}
    for it in items:
        img = preprocess.apply(mode, it.path) if mode != "raw" else np.asarray(Image.open(it.path).convert("L").resize((CELL, CELL)))
        by[it.letter].append((img, f"{it.id} {it.batch}"))
    rows = [[im for im, _ in by[l]] or [np.full((CELL, CELL), 255, np.uint8)] for l in LETTERS]
    caps = [[cap for _, cap in by[l]] or [""] for l in LETTERS]
    return grid(rows, LETTERS, caps)


def dataset_sheet(ds: Path, per_class: int, cache: str) -> Image.Image:
    z = np.load(ds / f"cache_{cache}.npz")
    X, y = z["X"], z["y"]
    rng = np.random.default_rng(0)
    rows = []
    for l in LETTERS:
        idx = np.flatnonzero(y == INDEX[l])
        pick = rng.choice(idx, size=min(per_class, len(idx)), replace=False) if len(idx) else []
        rows.append([X[i] for i in pick] or [np.full((CELL, CELL), 255, np.uint8)])
    return grid(rows, LETTERS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", choices=["dev", "test", "all"])
    ap.add_argument("--mode", default="canonical", choices=["canonical", "legacy", "raw"])
    ap.add_argument("--dataset", type=Path)
    ap.add_argument("--cache", default="canonical")
    ap.add_argument("--per-class", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sheet = real_sheet(a.real, a.mode) if a.real else dataset_sheet(a.dataset, a.per_class, a.cache)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
