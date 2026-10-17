"""Train a probe on one synthetic dataset; early stopping on the synthetic val set only (DR-010, DR-011).

  python train/train.py --data data/<dataset> --exp <id> --parent <id> --seed 0 [--probe hybrid|alt]
         [--input canonical|legacy|autocontrast] [--batch 64] [--max-steps 8000] [--device gpu|cpu]

Dataset layout: <data>/train/cache_<input>.npz and <data>/val/cache_<input>.npz (X uint8 [N,128,128], y int).
Writes runs/<exp>/<probe>-<input>-s<seed>/ {run.json, history.csv, model.weights.h5, dev_preds.npz, metrics_dev.json}
and appends a dev row to docs/results/ledger.csv. Real test (marker-40) is never touched here.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--parent", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe", default="hybrid", choices=["hybrid", "alt"])
    ap.add_argument("--input", default="canonical", choices=["canonical", "legacy", "autocontrast"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--check-every", type=int, default=500)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lr-patience", type=int, default=2)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--run-name", default="")
    return ap.parse_args()


def batches(X, y, batch, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(len(y))
        for i in range(0, len(perm) - batch + 1, batch):
            idx = np.sort(perm[i:i + batch])
            yield X[idx].astype("float32")[..., None] / 255.0, y[idx]


def main():
    a = parse()
    import numpy as np
    import tensorflow as tf
    if a.device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    import keras

    from ogham_v2 import real_data
    from train import evaluate as ev
    from train import ledger
    from train.model import build

    keras.utils.set_random_seed(a.seed)
    run_name = a.run_name or f"{a.probe}-{a.input}-s{a.seed}"
    out = ROOT / "runs" / a.exp / run_name
    out.mkdir(parents=True, exist_ok=True)

    tr = np.load(a.data / "train" / f"cache_{a.input}.npz")
    va = np.load(a.data / "val" / f"cache_{a.input}.npz")
    Xtr, ytr, Xva, yva = tr["X"], tr["y"], va["X"], va["y"]
    Xdev, ydev, dev_ids, dev_batches, _ = real_data.load_split("dev", a.input)
    ds_info = json.loads((a.data / "dataset.json").read_text()) if (a.data / "dataset.json").exists() else {}

    model = build(a.probe)

    class DevLogger(keras.callbacks.Callback):
        """Logs real-dev metrics at each check for learning curves only (never used for selection)."""

        def __init__(self):
            super().__init__()
            self.best_epoch, self.best = -1, float("inf")

        def on_epoch_end(self, epoch, logs=None):
            m = ev.metrics(ev.predict(self.model, Xdev), ydev)
            logs["dev_cer"], logs["dev_nll"], logs["dev_aicme"] = m["cer"], m["nll"], m["aicme_acc"]
            logs["step"] = (epoch + 1) * a.check_every
            if logs["val_loss"] < self.best:
                self.best, self.best_epoch = logs["val_loss"], epoch

    dev_logger = DevLogger()
    callbacks = [
        dev_logger,
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=a.patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=a.lr_patience, min_lr=1e-6),
        keras.callbacks.CSVLogger(str(out / "history.csv")),
    ]
    t0 = time.time()
    hist = model.fit(
        batches(Xtr, ytr, a.batch, a.seed),
        steps_per_epoch=a.check_every,
        epochs=a.max_steps // a.check_every,
        validation_data=(Xva.astype("float32")[..., None] / 255.0, yva),
        callbacks=callbacks,
        verbose=2,
    )
    minutes = (time.time() - t0) / 60
    model.save_weights(out / "model.weights.h5")

    val_loss, val_acc = model.evaluate(Xva.astype("float32")[..., None] / 255.0, yva, verbose=0)
    probs = ev.predict(model, Xdev)
    m = ev.metrics(probs, ydev, dev_batches)
    blank = ev.predict(model, real_data.load_blank(a.input))[0]
    m["blank_maxprob"], m["blank_pred"] = float(blank.max()), ev.LETTERS[int(blank.argmax())]
    np.savez(out / "dev_preds.npz", probs=probs, y=ydev, ids=np.array(dev_ids), batches=np.array(dev_batches))
    (out / "metrics_dev.json").write_text(json.dumps(m, indent=2))

    info = {
        "exp_id": a.exp, "parent": a.parent, "run_id": run_name, "probe": a.probe, "input": a.input, "seed": a.seed,
        "dataset": str(a.data.relative_to(ROOT)) if a.data.is_relative_to(ROOT) else str(a.data),
        "n_per_class": ds_info.get("n_per_class", int(len(ytr) / 20)), "batch": a.batch, "device": a.device,
        "steps_used": len(hist.history["loss"]) * a.check_every, "best_step": (dev_logger.best_epoch + 1) * a.check_every,
        "train_minutes": round(minutes, 2), "synth_val_acc": float(val_acc), "synth_val_loss": float(val_loss),
        "max_steps": a.max_steps, "check_every": a.check_every, "patience": a.patience,
    }
    (out / "run.json").write_text(json.dumps(info, indent=2))
    ledger.append({**info, "split": "dev", **ledger.flatten(m)})
    print(json.dumps({"run": str(out.relative_to(ROOT)), "dev_cer": m["cer"], "dev_nll": round(m["nll"], 3),
                      "dev_aicme": m["aicme_acc"], "synth_val_acc": round(float(val_acc), 4),
                      "steps": info["steps_used"], "minutes": info["train_minutes"]}))


if __name__ == "__main__":
    main()
