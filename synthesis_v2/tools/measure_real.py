"""Calibrate E1 geometry ranges: detector measurements on real DEV images vs a synthetic dataset (same detector).

  python tools/measure_real.py --synthetic data/<ds>/train --out docs/results/figures/E1/calibration
Only dev (pen-40) is measured (DR-003); only images whose decoded letter matches the label are used.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ogham_v2 import detectors, real_data  # noqa: E402
from ogham_v2.alphabet import SPEC  # noqa: E402
from train.decode_rules import decode  # noqa: E402

PARAMS = ["stem_tilt", "side_angle", "across_angle", "diag_angle", "spacing", "length_side", "length_across",
          "length_diag", "balance", "group_pos", "overhang_min", "width_over_stem"]


def measure(det: dict, letter: str, out: dict):
    st = det["stem"]
    out["stem_tilt"].append(st["angle"] - 90)
    strokes = det["strokes"]
    d = SPEC[letter].direction
    out["width_over_stem"].append(det["stroke_width"] / st["length"])
    pos = sorted(s["position"] for s in strokes)
    if len(pos) > 1:
        out["spacing"] += list(np.diff(pos))
    lo, hi = pos[0], 1 - pos[-1]
    out["overhang_min"].append(min(lo, hi))
    out["group_pos"].append(lo / (lo + hi + 1e-9))
    for s in strokes:
        dev = s["angle"] - 90
        if d in ("right", "left"):
            out["side_angle"].append(dev)
            out["length_side"].append(s["length"])
        elif d == "across":
            out["across_angle"].append(dev)
            out["length_across"].append(s["length"])
            out["balance"].append(s["balance"])
        else:
            out["diag_angle"].append(dev)
            out["length_diag"].append(s["length"])
            out["balance"].append(s["balance"])


def collect(items, detect):
    out = {k: [] for k in PARAMS}
    used = 0
    for letter, img in items:
        det = detect(img)
        pred, _ = decode(det)
        if pred == letter:
            measure(det, letter, out)
            used += 1
    return out, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/results/figures/E1/calibration")
    a = ap.parse_args()
    detect = detectors.get("proxy")
    real_items = [(it.letter, np.asarray(Image.open(it.path).convert("L"))) for it in real_data.split_items("dev")]
    real, n_real = collect(real_items, detect)
    syn, n_syn = ({}, 0)
    if a.synthetic:
        syn_items = [(p.stem.split("_")[0], np.asarray(Image.open(p).convert("L"))) for p in sorted(((a.synthetic / "qa") if (a.synthetic / "qa").exists() else (a.synthetic / "images")).glob("*_*.png"))]
        syn, n_syn = collect(syn_items, detect)
    summary = {"n_real_used": n_real, "n_synthetic_used": n_syn, "real": {}, "synthetic": {}}
    for name, src in (("real", real), ("synthetic", syn)):
        for k, v in src.items():
            if v:
                q = np.percentile(v, [0, 5, 50, 95, 100])
                summary[name][k] = {"n": len(v), "min": q[0], "p5": q[1], "median": q[2], "p95": q[3], "max": q[4]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    (a.out.with_suffix(".json")).write_text(json.dumps(summary, indent=2, default=float))
    fig, axes = plt.subplots(3, 4, figsize=(15, 9))
    for ax, k in zip(axes.flat, PARAMS):
        bins = 20
        if real[k]:
            ax.hist(real[k], bins=bins, alpha=0.6, density=True, label=f"real dev (n={len(real[k])})")
        if syn.get(k):
            ax.hist(syn[k], bins=bins, alpha=0.5, density=True, label=f"synthetic (n={len(syn[k])})")
        ax.set_title(k, fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle("Calibration: proxy-detector measurements, real dev (pen-40) vs synthetic", fontsize=12)
    fig.tight_layout()
    fig.savefig(a.out.with_suffix(".png"), dpi=100)
    for k in PARAMS:
        r, s = summary["real"].get(k), summary["synthetic"].get(k)
        fr = f"real p5/med/p95 = {r['p5']:.3f}/{r['median']:.3f}/{r['p95']:.3f} [{r['min']:.3f},{r['max']:.3f}]" if r else "real -"
        fs = f"syn p5/med/p95 = {s['p5']:.3f}/{s['median']:.3f}/{s['p95']:.3f}" if s else ""
        print(f"{k:16s} {fr}   {fs}")


if __name__ == "__main__":
    main()
