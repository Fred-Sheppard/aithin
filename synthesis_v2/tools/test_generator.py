"""Property tests for the generator. Exit code 0 = all checks pass.

  python tools/test_generator.py [--config default] [--n 2000] [--render-n 20]

1. Label validator: sample raw geometry and report the rejection rate; every sample the generator returns must
   satisfy the validator (stroke count, side, crossing halves, angle margins, spacing, no converging strokes).
2. Rendering: full samples (pen + appearance + preprocessing) for every letter render without error, have ink,
   and preprocess to 128x128.
3. Determinism: the same seed gives byte-identical samples.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ogham_v2 import config, preprocess  # noqa: E402
from ogham_v2.alphabet import LETTERS, SPEC  # noqa: E402
from ogham_v2.generate import make_sample  # noqa: E402
from ogham_v2.geometry import sample_glyph, validate  # noqa: E402
from ogham_v2.pen import sample_pen  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default")
    ap.add_argument("--n", type=int, default=2000, help="raw geometry samples per letter")
    ap.add_argument("--render-n", type=int, default=20, help="full renders per letter")
    a = ap.parse_args()
    cfg = config.load(a.config)
    failures = []

    rng = np.random.default_rng(0)
    rejects, total = {}, 0
    for letter in LETTERS:
        for _ in range(a.n):
            pen = sample_pen(cfg, rng)
            ok, why = validate(sample_glyph(letter, cfg, rng, pen["width"]), cfg, pen["width"])
            total += 1
            if not ok:
                rejects[why] = rejects.get(why, 0) + 1
    print(f"[1] raw rejection rate {sum(rejects.values()) / total:.1%} {dict(sorted(rejects.items(), key=lambda kv: -kv[1]))}")

    t0 = time.time()
    for letter in LETTERS:
        for _ in range(a.render_n):
            img, glyph, pen, _ = make_sample(letter, cfg, rng)
            ok, why = validate(glyph, cfg, pen["width"])
            if not ok:
                failures.append(f"{letter}: generator returned an invalid glyph ({why})")
            if len(glyph.strokes) != SPEC[letter].count:
                failures.append(f"{letter}: {len(glyph.strokes)} strokes, expected {SPEC[letter].count}")
            gray = img if img.ndim == 2 else img.mean(2)
            if gray.min() > 200:
                failures.append(f"{letter}: render has no visible ink")
            for mode in cfg.get("cache_modes", ["canonical"]):
                if preprocess.apply(mode, img).shape != (128, 128):
                    failures.append(f"{letter}: {mode} preprocessing did not give 128x128")
    ms = 1000 * (time.time() - t0) / (len(LETTERS) * a.render_n)
    print(f"[2] {len(LETTERS) * a.render_n} full renders checked, {ms:.1f} ms/sample (1 core)")

    def first(seed):
        return make_sample("N", cfg, np.random.default_rng(seed))[0]
    same = np.array_equal(first(123), first(123))
    print(f"[3] deterministic for a fixed seed: {same}")
    if not same:
        failures.append("same seed gave different samples")

    for f in failures:
        print("FAIL", f)
    print("OK" if not failures else f"{len(failures)} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
