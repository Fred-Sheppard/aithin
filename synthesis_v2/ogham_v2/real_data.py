"""The 80 labelled real images: dev = pen-40, test = marker-40, plus a blank-page control (DR-003, DR-012)."""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from . import preprocess
from .alphabet import INDEX
from .paths import DATA, EVAL, REAL_SETS


@dataclass
class RealItem:
    set_name: str
    img_no: int
    letter: str
    batch: str
    path: Path

    @property
    def id(self) -> str:
        return f"{self.set_name}-{self.img_no:05d}"


def _batch(set_name: str, source: str) -> str:
    if set_name == "marker":
        return "marker-1" if source == "Ogham Marker.pdf" else "marker-2"
    return "pen-2" if source.startswith("Ogham Pen/") else "pen-1"


def _overrides() -> dict:
    out = {}
    with open(EVAL / "label_overrides.csv") as f:
        for row in csv.DictReader(f):
            out[(row["set"], int(row["img_no"]))] = row["letter"]
    return out


def load_items(set_name: str) -> list[RealItem]:
    root = REAL_SETS[set_name]
    over = _overrides()
    items = []
    with open(root / "labels.csv") as f:
        for row in csv.DictReader(f):
            no = int(row["img_no"])
            letter = over.get((set_name, no), row["img_letter"])
            items.append(RealItem(set_name, no, letter, _batch(set_name, row["source"]), root / "images" / f"{no:05d}.png"))
    return items


def splits() -> dict:
    with open(EVAL / "splits.yaml") as f:
        return yaml.safe_load(f)


def split_items(split: str) -> list[RealItem]:
    return [it for s in splits()[split] for it in load_items(s)]


def blank_page(h: int = 1600, w: int = 1200) -> np.ndarray:
    """Blank white page control: near-white paper with faint sensor noise."""
    rng = np.random.default_rng(0)
    return np.clip(250 + rng.normal(0, 1.5, (h, w)), 0, 255).astype(np.uint8)


def load_split(split: str, mode: str):
    """Returns (X uint8 [N,128,128], y int [N], ids, batches, letters). Cached per preprocessing version."""
    tag = preprocess.VERSION if mode == "canonical" else mode
    cache = DATA / "real_cache" / f"{split}_{mode}_{tag}.npz"
    items = split_items(split)
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["ids"]) == [it.id for it in items] and list(z["letters"]) == [it.letter for it in items]:
            return z["X"], z["y"], list(z["ids"]), list(z["batches"]), list(z["letters"])
    X = np.stack([preprocess.apply(mode, it.path) for it in items])
    y = np.array([INDEX[it.letter] for it in items])
    ids, batches, letters = [it.id for it in items], [it.batch for it in items], [it.letter for it in items]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y, ids=np.array(ids), batches=np.array(batches), letters=np.array(letters))
    return X, y, ids, batches, letters


def load_blank(mode: str) -> np.ndarray:
    return preprocess.apply(mode, blank_page())[None]
