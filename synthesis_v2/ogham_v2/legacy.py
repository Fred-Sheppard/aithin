"""Run the OLD generator (aithin/synthesis/synthesise_data.py) unmodified, seeded from outside (DR-001).

  python -m ogham_v2.legacy --out data/legacy_n850_s0 --n-per-class 850 --seed 0 [--val-n 50]

The old code uses paths relative to the working directory ("synthesis/data/lines.ndjson", "output/..."),
so it runs inside a scratch directory that links to the real lines.ndjson.
"""
import argparse
import importlib
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import caches
from .paths import LEGACY_SYNTHESIS, ROOT


def _generate(dest: Path, n_per_class: int, seed: int) -> None:
    work = dest.parent / f"_work_{dest.name}"
    shutil.rmtree(work, ignore_errors=True)
    (work / "synthesis" / "data").mkdir(parents=True)
    os.symlink(LEGACY_SYNTHESIS / "data" / "lines.ndjson", work / "synthesis" / "data" / "lines.ndjson")
    if str(LEGACY_SYNTHESIS) not in sys.path:
        sys.path.insert(0, str(LEGACY_SYNTHESIS))
    cwd = os.getcwd()
    os.chdir(work)
    try:
        random.seed(seed)
        np.random.seed(seed)
        mod = importlib.import_module("synthesise_data")
        mod.generate_dataset(dataset_size=n_per_class)
    finally:
        os.chdir(cwd)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(str(work / "output" / "temp-letters-dir"), str(dest / "images"))
    shutil.move(str(work / "output" / "labels.csv"), str(dest / "labels.csv"))
    shutil.rmtree(work)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-per-class", type=int, default=850)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-n", type=int, default=50)
    a = ap.parse_args()
    out = a.out if a.out.is_absolute() else ROOT / a.out
    shutil.rmtree(out, ignore_errors=True)
    t0 = time.time()
    _generate(out / "train", a.n_per_class, a.seed)
    _generate(out / "val", a.val_n, a.seed + 10_000)
    t1 = time.time()
    for split in ("train", "val"):
        caches.build(out / split)
    info = {"name": out.name, "generator": "legacy", "seed": a.seed, "n_per_class": a.n_per_class,
            "val_n_per_class": a.val_n, "val_seed": a.seed + 10_000, "created": datetime.now().isoformat(timespec="seconds"),
            "gen_minutes": round((t1 - t0) / 60, 2), "cache_minutes": round((time.time() - t1) / 60, 2)}
    (out / "dataset.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
