"""Generate a synthesis-v2 dataset (train + separately seeded val) from an experiment config.

  python -m ogham_v2.generate --config e1a --out data/e1a_n850_s0 [--n-per-class 850] [--seed 0] [--val-n 50]

Writes <out>/{train,val}/: cache_<mode>.npz (uint8 128x128 + labels), labels.csv, meta.jsonl (sampled
parameters), oracle.jsonl (exact detector JSON), images/ (first `qa_per_class` native renders per class),
and <out>/dataset.json.
"""
import argparse
import csv
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import config as cfgmod
from . import preprocess
from .alphabet import INDEX, LETTERS
from .detector_schema import oracle
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
        from .appearance import apply as appearance
        img = appearance(img, pen, cfg, rng)
    return img, glyph, pen, info


def _representation(img, glyph, info, rep: dict, rng):
    """E3: detector-JSON representation of a synthetic sample, rendered canonically."""
    from . import canonical_render, detector_errors
    src = rep.get("source", "oracle")
    if src == "detect":
        from .detectors.proxy import detect
        det = detect(img)
    else:
        det = oracle(glyph, info)
        if src == "oracle_errors":
            det = detector_errors.apply(det, rep["errors"], rng)
    return canonical_render.render(det, rep.get("unattached", "faint"))


def _chunk(args):
    cfg, letter, start, count, seed, modes, qa = args
    rng = np.random.default_rng([seed, INDEX[letter], start])
    out = {m: [] for m in modes}
    metas, oracles, qa_imgs = [], [], []
    rep = cfg.get("representation") or {}
    for i in range(count):
        img, glyph, pen, info = make_sample(letter, cfg, rng)
        for m in modes:
            if m == "render":
                out[m].append(_representation(img, glyph, info, rep, rng))
            else:
                out[m].append(preprocess.apply(m, img))
        metas.append({"letter": letter, "pen": {k: v for k, v in pen.items() if k != "intensity"}, **glyph.params,
                      "size": info["size"]})
        oracles.append(oracle(glyph, info))
        if start + i < qa:
            qa_imgs.append((start + i, img))
    return letter, start, {m: np.stack(v) for m, v in out.items()}, metas, oracles, qa_imgs


def generate_split(cfg: dict, dest: Path, n_per_class: int, seed: int, workers: int = 8) -> dict:
    modes = cfg.get("cache_modes", ["canonical"])
    qa = cfg.get("qa_per_class", 10)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "images").mkdir(exist_ok=True)
    tasks = [(cfg, l, s, min(CHUNK, n_per_class - s), seed, modes, qa) for l in LETTERS for s in range(0, n_per_class, CHUNK)]
    results = {}
    with ProcessPoolExecutor(workers) as pool:
        for letter, start, arrs, metas, oracles, qa_imgs in pool.map(_chunk, tasks):
            results[(letter, start)] = (arrs, metas, oracles)
            for j, img in qa_imgs:
                cv2.imwrite(str(dest / "images" / f"{letter}_{j:03d}.png"), img if img.ndim == 2 else img[..., ::-1])
    X = {m: [] for m in modes}
    y, rows = [], []
    with open(dest / "meta.jsonl", "w") as fm, open(dest / "oracle.jsonl", "w") as fo:
        n = 0
        for l in LETTERS:
            for s in range(0, n_per_class, CHUNK):
                arrs, metas, oracles = results[(l, s)]
                for m in modes:
                    X[m].append(arrs[m])
                for meta, orc in zip(metas, oracles):
                    fm.write(json.dumps({"img_no": n, **meta}) + "\n")
                    fo.write(json.dumps({"img_no": n, "letter": l, **orc}) + "\n")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-per-class", type=int, default=850)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache-modes", nargs="+", help="override the config's cache_modes")
    a = ap.parse_args()
    cfg = cfgmod.load(a.config)
    if a.cache_modes:
        cfg["cache_modes"] = a.cache_modes
    out = a.out if a.out.is_absolute() else ROOT / a.out
    shutil.rmtree(out, ignore_errors=True)
    t0 = time.time()
    generate_split(cfg, out / "train", a.n_per_class, a.seed, a.workers)
    generate_split(cfg, out / "val", a.val_n, a.seed + 10_000, a.workers)
    import subprocess
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "ogham_v2", "configs"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    info = {"name": out.name, "generator": "v2", "config": cfg["name"], "git_sha": sha + ("+dirty" if dirty else ""),
            "seed": a.seed, "n_per_class": a.n_per_class,
            "val_n_per_class": a.val_n, "val_seed": a.seed + 10_000, "created": datetime.now().isoformat(timespec="seconds"),
            "gen_minutes": round((time.time() - t0) / 60, 2), "resolved_config": cfg}
    (out / "dataset.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "resolved_config"}))


if __name__ == "__main__":
    main()
