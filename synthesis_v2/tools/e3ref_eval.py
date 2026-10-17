"""E3-ref: line detector + rule decoder, no training. Scores dev (and test, for frozen detector versions only).

  python tools/e3ref_eval.py --split dev|test [--detector proxy] [--exp E3-ref]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ogham_v2 import detectors, real_data  # noqa: E402
from ogham_v2.alphabet import INDEX, LETTERS, group_of  # noqa: E402
from train import ledger  # noqa: E402
from train.decode_rules import decode  # noqa: E402
from train.evaluate import taxonomy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--detector", default="proxy")
    ap.add_argument("--exp", default="E3-ref")
    a = ap.parse_args()
    detect = detectors.get(a.detector)
    items = real_data.split_items(a.split)
    preds, errors = [], []
    for it in items:
        det = detect(np.asarray(Image.open(it.path).convert("L")))
        p, info = decode(det)
        preds.append(p)
        if p != it.letter:
            errors.append({"id": it.id, "batch": it.batch, "true": it.letter, "pred": p, **info})
    y = np.array([INDEX[it.letter] for it in items])
    pi = np.array([INDEX[p] if p else -1 for p in preds])
    correct = pi == y
    batches = np.array([it.batch for it in items])
    m = {"n": len(y), "cer": float(1 - correct.mean()), "acc": float(correct.mean()),
         "macro_f1": float(f1_score(y, pi, labels=list(range(20)), average="macro", zero_division=0)),
         "aicme_acc": float(np.mean([p is not None and group_of(p) == it.letter and True or (p is not None and group_of(p) == group_of(it.letter)) for p, it in zip(preds, items)])),
         "taxonomy": taxonomy(np.where(pi < 0, (y + 1) % 20, pi), y),
         "per_batch": {b: {"cer": float(1 - correct[batches == b].mean())} for b in sorted(set(batches))}}
    blank_det = detect(real_data.blank_page())
    bp, _ = decode(blank_det)
    m["blank_pred"], m["blank_maxprob"] = (bp or "none"), ""
    out = ROOT / "runs" / a.exp / f"rules-{a.detector}"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"metrics_{a.split}.json").write_text(json.dumps({**m, "errors": errors}, indent=2, default=str))
    ledger.append({"exp_id": a.exp, "run_id": f"rules-{a.detector}", "probe": "rules", "input": f"{a.detector}-detector",
                   "split": a.split, **ledger.flatten(m)})
    print(json.dumps({k: v for k, v in m.items() if k != "taxonomy"}, indent=1))
    for e in errors:
        print("  error:", e)


if __name__ == "__main__":
    main()
