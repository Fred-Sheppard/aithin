"""Property tests for the generator (plan: Verification).

  python tools/test_generator.py --configs e1a e1b e1c e1d [--n 10000]
1. every sampled glyph passes the label-preserving validator (with rejection stats)
2. oracle detector JSON -> rule decoder recovers the true letter 100%
3. rendering never crashes; reports generation speed
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ogham_v2 import config  # noqa: E402
from ogham_v2.alphabet import LETTERS  # noqa: E402
from ogham_v2.detector_schema import oracle  # noqa: E402
from ogham_v2.generate import make_sample  # noqa: E402
from ogham_v2.geometry import sample_glyph, validate  # noqa: E402
from ogham_v2.pen import sample_pen  # noqa: E402
from train.decode_rules import decode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["e1a"])
    ap.add_argument("--n", type=int, default=10000, help="raw geometry samples per letter (validator stats)")
    ap.add_argument("--render-n", type=int, default=100, help="full renders per letter (oracle round-trip)")
    a = ap.parse_args()
    ok_all = True
    for name in a.configs:
        cfg = config.load(name)
        rng = np.random.default_rng(0)
        rej, total = {}, 0
        for l in LETTERS:
            for _ in range(a.n):
                pen = sample_pen(cfg, rng)
                g = sample_glyph(l, cfg, rng, pen["width"])
                ok, why = validate(g, cfg, pen["width"])
                total += 1
                if not ok:
                    rej[why] = rej.get(why, 0) + 1
        t0 = time.time()
        wrong = 0
        for l in LETTERS:
            for _ in range(a.render_n):
                img, glyph, pen, info = make_sample(l, cfg, rng)
                pred, _ = decode(oracle(glyph, info))
                wrong += pred != l
        dt = (time.time() - t0) / (20 * a.render_n)
        rate = sum(rej.values()) / total
        print(f"{name}: raw rejection rate {rate:.1%} {dict(sorted(rej.items(), key=lambda kv: -kv[1]))}; "
              f"oracle round-trip errors {wrong}/{20 * a.render_n}; {1000 * dt:.1f} ms/sample (1 core)")
        ok_all &= wrong == 0
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
