"""Preprocessed uint8 caches (cache_<mode>.npz) for a folder of PNGs + labels.csv."""
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import preprocess
from .alphabet import INDEX

MODES = ("canonical", "legacy", "autocontrast")


def _one(args):
    mode, path = args
    return preprocess.apply(mode, path)


def read_labels(split_dir: Path):
    with open(split_dir / "labels.csv") as f:
        rows = list(csv.DictReader(f))
    paths = [split_dir / "images" / f"{int(r['img_no']):05d}.png" for r in rows]
    y = np.array([INDEX[r["img_letter"]] for r in rows])
    return paths, y


def build(split_dir: Path, modes=MODES, workers: int = 8) -> None:
    paths, y = read_labels(split_dir)
    with ProcessPoolExecutor(workers) as pool:
        for mode in modes:
            X = np.stack(list(pool.map(_one, [(mode, p) for p in paths], chunksize=64)))
            np.savez_compressed(split_dir / f"cache_{mode}.npz", X=X, y=y)
