"""Score a detector with the rule decoder: on synthetic QA renders (vs. the true letter) and/or real images.

  python tools/detector_check.py --dataset data/<ds>/train      # QA renders in images/<LETTER>_<i>.png
  python tools/detector_check.py --real dev [--dump out.jsonl]  # real split
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ogham_v2 import detectors, real_data  # noqa: E402
from ogham_v2.alphabet import SPEC, group_of  # noqa: E402
from train.decode_rules import decode  # noqa: E402


def score(items, detect):
    res = []
    for name, letter, img in items:
        det = detect(img)
        pred, info = decode(det)
        res.append({"id": name, "letter": letter, "pred": pred, "count": info.get("count", 0),
                    "true_count": SPEC[letter].count, "group_ok": pred is not None and group_of(pred) == group_of(letter),
                    "det": det})
    acc = np.mean([r["pred"] == r["letter"] for r in res])
    cnt = np.mean([r["count"] == r["true_count"] for r in res])
    grp = np.mean([r["group_ok"] for r in res])
    return res, {"n": len(res), "decode_acc": float(acc), "count_acc": float(cnt), "group_acc": float(grp)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path)
    ap.add_argument("--real", choices=["dev", "test"])
    ap.add_argument("--detector", default="proxy")
    ap.add_argument("--dump", type=Path)
    ap.add_argument("--show-errors", action="store_true")
    a = ap.parse_args()
    detect = detectors.get(a.detector)
    if a.dataset:
        items = [(p.stem, p.stem.split("_")[0], np.asarray(Image.open(p).convert("L"))) for p in sorted((a.dataset / "images").glob("*.png"))]
    else:
        items = [(it.id, it.letter, np.asarray(Image.open(it.path).convert("L"))) for it in real_data.split_items(a.real)]
    res, summ = score(items, detect)
    print(json.dumps(summ))
    if a.show_errors:
        for r in res:
            if r["pred"] != r["letter"]:
                print(f"  {r['id']}: true {r['letter']} pred {r['pred']} count {r['count']}/{r['true_count']} "
                      f"sides {[s['side'] for s in r['det']['strokes']]} angles {[round(s['angle']) for s in r['det']['strokes']]}")
    if a.dump:
        with open(a.dump, "w") as f:
            for r in res:
                f.write(json.dumps({k: v for k, v in r.items()}, default=float) + "\n")


if __name__ == "__main__":
    main()
