"""Supplementary robustness metric (NOT a decision metric; DR-022): real dev images under capture degradations.

  python tools/stress_eval.py --runs runs/E1d/hybrid-canonical-s0 [runs/... ...]   (train venv)
  python tools/stress_eval.py --all                                                  # every run with weights

Six deterministic degradations of the 40 real pen images (240 images): blur, lowres, jpeg, rotate, contrast,
shadow. Accuracy per degradation and overall -> <run>/metrics_devstress.json + a ledger row (split=dev-stress).
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ogham_v2 import preprocess, real_data  # noqa: E402
from ogham_v2.alphabet import INDEX  # noqa: E402

KINDS = ["blur", "lowres", "jpeg", "rotate", "contrast", "shadow"]


def degrade(img: Image.Image, kind: str, i: int) -> Image.Image:
    g = img.convert("L")
    w, h = g.size
    if kind == "blur":
        return g.filter(ImageFilter.GaussianBlur(0.006 * max(w, h)))
    if kind == "lowres":
        s = 110 / max(w, h)
        return g.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    if kind == "jpeg":
        s = 400 / max(w, h)
        small = g.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
        buf = io.BytesIO()
        small.save(buf, "JPEG", quality=12)
        return Image.open(io.BytesIO(buf.getvalue())).convert("L")
    if kind == "rotate":
        return g.rotate(10 if i % 2 else -10, resample=Image.BILINEAR, expand=True, fillcolor=int(np.median(np.asarray(g))))
    if kind == "contrast":
        a = np.asarray(g, np.float32)
        return Image.fromarray(np.clip(255 - 0.35 * (255 - a), 0, 255).astype(np.uint8))
    if kind == "shadow":
        a = np.asarray(g, np.float32)
        yy, xx = np.mgrid[0:h, 0:w]
        ramp = 1 - 0.5 * np.clip((xx / w + yy / h) / 2 * 1.4 - 0.2, 0, 1)
        return Image.fromarray(np.clip(a * ramp, 0, 255).astype(np.uint8))
    raise ValueError(kind)


def stress_set(mode: str):
    tag = preprocess.VERSION if mode == "canonical" else mode
    if mode == "render":
        from ogham_v2.detectors.proxy import VERSION as tag
    cache = ROOT / "data" / "real_cache" / f"devstress_{mode}_{tag}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["X"], z["y"], list(z["kinds"])
    items = real_data.split_items("dev")
    X, y, kinds = [], [], []
    for i, it in enumerate(items):
        img = Image.open(it.path)
        for k in KINDS:
            X.append(preprocess.apply(mode, degrade(img, k, i)))
            y.append(INDEX[it.letter])
            kinds.append(k)
    X, y = np.stack(X), np.array(y)
    np.savez_compressed(cache, X=X, y=y, kinds=np.array(kinds))
    return X, y, kinds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", type=Path, default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    from train import evaluate as ev
    from train import ledger
    from train.model import build

    runs = list(a.runs)
    if a.all:
        runs = sorted(p.parent for p in (ROOT / "runs").glob("*/*/model.weights.h5") if "tune" not in p.parts[-3])
    for run in runs:
        out = run / "metrics_devstress.json"
        if out.exists() and not a.force:
            continue
        info = json.loads((run / "run.json").read_text())
        X, y, kinds = stress_set(info["input"])
        model = build(info["probe"])
        model.load_weights(run / "model.weights.h5")
        probs = ev.predict(model, X)
        pi = ev.per_image(probs, y)
        k = np.array(kinds)
        m = {"n": int(len(y)), "acc": float(pi["correct"].mean()), "cer": float(1 - pi["correct"].mean()),
             "nll": float(pi["nll"].mean()), "aicme_acc": float(pi["aicme"].mean()),
             "per_kind": {kk: float(pi["correct"][k == kk].mean()) for kk in KINDS}}
        out.write_text(json.dumps(m, indent=2))
        ledger.append({**{kk: info[kk] for kk in ledger.RUN_KEYS if kk in info}, "split": "dev-stress", **m})
        print(f"{run.relative_to(ROOT)}: stress acc {m['acc']:.3f} " + " ".join(f"{kk}={v:.2f}" for kk, v in m["per_kind"].items()), flush=True)


if __name__ == "__main__":
    main()
