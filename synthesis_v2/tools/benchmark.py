"""Training-throughput benchmark for the compute gate (DR-014).

  python tools/benchmark.py --device gpu|cpu [--steps 40]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--steps", type=int, default=40)
    a = ap.parse_args()
    import numpy as np
    import tensorflow as tf
    if a.device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    from train.model import build

    out = []
    for probe in ("hybrid", "alt"):
        for batch in (32, 64, 128):
            model = build(probe)
            X = np.random.rand(batch * (a.steps + 5), 128, 128, 1).astype("float32")
            y = np.random.randint(0, 20, len(X))
            model.fit(X[:batch * 5], y[:batch * 5], batch_size=batch, epochs=1, verbose=0)  # warm-up / graph build
            t0 = time.time()
            model.fit(X[batch * 5:], y[batch * 5:], batch_size=batch, epochs=1, verbose=0)
            sps = a.steps / (time.time() - t0)
            r = {"device": a.device, "probe": probe, "batch": batch, "steps_per_s": round(sps, 2),
                 "img_per_s": round(sps * batch), "min_per_8k_steps": round(8000 / sps / 60, 1)}
            print(json.dumps(r), flush=True)
            out.append(r)


if __name__ == "__main__":
    main()
