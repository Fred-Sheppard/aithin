"""Segment-level detector error statistics vs. the exact oracle, on synthetic renders (grounds E3b rates).

  python tools/measure_detector_errors.py --config e1c [--n 25]   # n samples per letter
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ogham_v2 import config  # noqa: E402
from ogham_v2.alphabet import LETTERS  # noqa: E402
from ogham_v2.detector_schema import oracle  # noqa: E402
from ogham_v2.detectors.proxy import detect  # noqa: E402
from ogham_v2.generate import make_sample  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="e1c")
    ap.add_argument("--n", type=int, default=25)
    a = ap.parse_args()
    cfg = config.load(a.config)
    rng = np.random.default_rng(123)
    st = {"glyphs": 0, "no_stem": 0, "true_strokes": 0, "missed": 0, "extra": 0, "split_like": 0,
          "side_wrong": 0, "unattached": 0, "jitter": []}
    for l in LETTERS:
        for _ in range(a.n):
            img, glyph, pen, info = make_sample(l, cfg, rng)
            orc, det = oracle(glyph, info), detect(img)
            st["glyphs"] += 1
            st["true_strokes"] += len(orc["strokes"])
            if not det["stem"]:
                st["no_stem"] += 1
                st["missed"] += len(orc["strokes"])
                continue
            st["unattached"] += len(det["unattached"])
            # match detected -> oracle strokes by position (in oracle stem frame both are normalised positions)
            used = set()
            for s in det["strokes"]:
                j = int(np.argmin([abs(s["position"] - o["position"]) for o in orc["strokes"]]))
                o = orc["strokes"][j]
                if abs(s["position"] - o["position"]) > 0.05:
                    st["extra"] += 1
                    continue
                if j in used:
                    st["split_like"] += 1
                    continue
                used.add(j)
                if s["side"] != o["side"]:
                    st["side_wrong"] += 1
                st["jitter"] += [abs(s["position"] - o["position"]), abs(s["length"] - o["length"])]
            st["missed"] += len(orc["strokes"]) - len(used)
    T = st["true_strokes"]
    out = {"config": a.config, "glyphs": st["glyphs"], "no_stem_rate": st["no_stem"] / st["glyphs"],
           "drop_rate": st["missed"] / T, "split_or_dup_rate": st["split_like"] / T, "extra_per_glyph": st["extra"] / st["glyphs"],
           "side_wrong_rate": st["side_wrong"] / T, "unattached_per_glyph": st["unattached"] / st["glyphs"],
           "jitter_median": float(np.median(st["jitter"])), "jitter_p90": float(np.percentile(st["jitter"], 90))}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
