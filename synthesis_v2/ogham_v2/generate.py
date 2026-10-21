"""Generate a synthetic Ogham dataset: a training split plus a separately seeded validation split.

  python -m ogham_v2.generate --out data/ogham_v2 [--config default] [--n-per-class 850] [--seed 0]
         [--val-n 50] [--save-images qa|all] [--cache-modes canonical autocontrast] [--workers 8]

Writes <out>/{train,val}/:
  labels.csv        img_no,img_letter,type (img_no zero-padded; same layout as the legacy generator's output)
  images/           --save-images all: every native-resolution render as <img_no>.png
  cache_<mode>.npz  X = preprocessed uint8 128x128 images, y = label index (model/label_to_letter.json order)
  qa/               the first `qa_per_class` renders per letter as <LETTER>_<i>.png, for visual checks
  meta.jsonl        the sampled generator parameters of every image
and <out>/dataset.json (resolved config, seeds, generator git sha).
"""
import argparse
import csv
import json
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import config as cfgmod
from . import preprocess
from .alphabet import INDEX, LETTERS
from .appearance import apply as appearance
from .geometry import sample_valid
from .paths import ROOT
from .pen import render, sample_pen

CHUNK = 50


def make_sample(letter: str, cfg: dict, rng):
    for attempt in range(20):            # a very wide pen can make 5-stroke letters infeasible: resample the pen
        pen = sample_pen(cfg, rng)
        try:
            glyph = sample_valid(letter, cfg, rng, pen["width"], max_tries=60)
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError(f"no feasible pen/geometry for {letter}")
    glyph.params["pen_resamples"] = attempt
    img, info = render(glyph, pen, cfg, rng)
    if cfg.get("appearance"):
        img = appearance(img, pen, cfg, rng)
    return img, glyph, pen, info


def _write_png(path: Path, img: np.ndarray) -> None:
    cv2.imwrite(str(path), img if img.ndim == 2 else img[..., ::-1])


def _chunk(args):
    cfg, letter, start, count, seed, modes, qa, save_all, dest, n_per_class = args
    rng = np.random.default_rng([seed, INDEX[letter], start])
    out = {m: [] for m in modes}
    metas, qa_imgs = [], []
    for i in range(count):
        img, glyph, pen, info = make_sample(letter, cfg, rng)
        for m in modes:
            out[m].append(preprocess.apply(m, img))
        metas.append({"letter": letter, "pen": {k: v for k, v in pen.items() if k != "intensity"}, **glyph.params,
                      "size": info["size"]})
        if start + i < qa:
            qa_imgs.append((start + i, img))
        if save_all:                     # same numbering as labels.csv: class blocks in label-map order
            _write_png(dest / "images" / f"{INDEX[letter] * n_per_class + start + i:05d}.png", img)
    return letter, start, {m: np.stack(v) for m, v in out.items()}, metas, qa_imgs


def generate_split(cfg: dict, dest: Path, n_per_class: int, seed: int, workers: int = 8, save_images: str = "qa") -> dict:
    modes = cfg.get("cache_modes", ["canonical"])
    qa = cfg.get("qa_per_class", 10)
    save_all = save_images == "all"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "qa").mkdir(exist_ok=True)
    if save_all:
        (dest / "images").mkdir(exist_ok=True)
    tasks = [(cfg, l, s, min(CHUNK, n_per_class - s), seed, modes, qa, save_all, dest, n_per_class)
             for l in LETTERS for s in range(0, n_per_class, CHUNK)]
    results = {}
    with ProcessPoolExecutor(workers) as pool:
        for letter, start, arrs, metas, qa_imgs in pool.map(_chunk, tasks):
            results[(letter, start)] = (arrs, metas)
            for j, img in qa_imgs:
                _write_png(dest / "qa" / f"{letter}_{j:03d}.png", img)
    X = {m: [] for m in modes}
    y, rows = [], []
    with open(dest / "meta.jsonl", "w") as fm:
        n = 0
        for l in LETTERS:
            for s in range(0, n_per_class, CHUNK):
                arrs, metas = results[(l, s)]
                for m in modes:
                    X[m].append(arrs[m])
                for meta in metas:
                    fm.write(json.dumps({"img_no": n, **meta}) + "\n")
                    y.append(INDEX[l])
                    rows.append((f"{n:05d}", l, cfg["name"]))
                    n += 1
    y = np.array(y)
    for m in modes:
        np.savez_compressed(dest / f"cache_{m}.npz", X=np.concatenate(X[m]), y=y)
    with open(dest / "labels.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["img_no", "img_letter", "type"])
        w.writerows(rows)
    return {"n": int(len(y))}


def _git_sha() -> str:
    run = lambda *c: subprocess.run(["git", *c], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    sha = run("rev-parse", "--short", "HEAD")
    return sha + ("+dirty" if run("status", "--porcelain", "ogham_v2", "configs") else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", default="default", help="config name in configs/ or a path to a YAML file")
    ap.add_argument("--n-per-class", type=int, default=850)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-n", type=int, default=50, help="validation images per class (seed + 10000)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache-modes", nargs="+", choices=sorted(preprocess.MODES), help="override the config's cache_modes")
    ap.add_argument("--save-images", choices=["qa", "all"], default="qa",
                    help="'all' also writes every native render as images/<img_no>.png")
    a = ap.parse_args()
    cfg = cfgmod.load(a.config)
    if a.cache_modes:
        cfg["cache_modes"] = a.cache_modes
    out = a.out if a.out.is_absolute() else ROOT / a.out
    shutil.rmtree(out, ignore_errors=True)
    t0 = time.time()
    generate_split(cfg, out / "train", a.n_per_class, a.seed, a.workers, a.save_images)
    generate_split(cfg, out / "val", a.val_n, a.seed + 10_000, a.workers, a.save_images)
    info = {"name": out.name, "config": cfg["name"], "git_sha": _git_sha(), "seed": a.seed,
            "n_per_class": a.n_per_class, "val_n_per_class": a.val_n, "val_seed": a.seed + 10_000,
            "save_images": a.save_images, "created": datetime.now().isoformat(timespec="seconds"),
            "gen_minutes": round((time.time() - t0) / 60, 2), "resolved_config": cfg}
    (out / "dataset.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "resolved_config"}))


if __name__ == "__main__":
    main()
