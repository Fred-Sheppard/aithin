"""Metrics matching the team's wild-test table, plus NLL/ECE, the Ogham error taxonomy and paired bootstrap.

CLI (frozen phase winners / legacy checkpoints only, DR-003):
  python train/evaluate.py --run runs/<exp>/<run> --split test
  python train/evaluate.py --checkpoint <file.h5> --name legacy-hybrid --input legacy --split dev|test
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ogham_v2.alphabet import LETTERS, NUM_CLASSES, count_of, group_of  # noqa: E402

GROUP = np.array([group_of(l) for l in LETTERS])
COUNT = np.array([count_of(l) for l in LETTERS])
CHANCE = {"acc": 1 / 20, "cer": 19 / 20, "top3": 3 / 20, "aicme_acc": 1 / 4}


def per_image(probs: np.ndarray, y: np.ndarray) -> dict:
    pred = probs.argmax(1)
    p_true = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    top3 = np.argsort(-probs, 1)[:, :3]
    return {
        "pred": pred,
        "correct": (pred == y).astype(float),
        "top3": (top3 == y[:, None]).any(1).astype(float),
        "aicme": (GROUP[pred] == GROUP[y]).astype(float),
        "nll": -np.log(p_true),
        "conf": probs.max(1),
    }


def taxonomy(pred: np.ndarray, y: np.ndarray) -> dict:
    """Error types: wrong group (side B<->H, diagonal/across M<->A, other) vs right group wrong count."""
    t = {"correct": 0, "group_side_BH": 0, "group_diag_across_MA": 0, "group_other": 0,
         "count_off_by_one": 0, "count_off_more": 0}
    for p, g in zip(pred, y):
        if p == g:
            t["correct"] += 1
        elif GROUP[p] != GROUP[g]:
            pair = {GROUP[p], GROUP[g]}
            key = "group_side_BH" if pair == {"B", "H"} else "group_diag_across_MA" if pair == {"M", "A"} else "group_other"
            t[key] += 1
        else:
            t["count_off_by_one" if abs(COUNT[p] - COUNT[g]) == 1 else "count_off_more"] += 1
    return t


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def metrics(probs: np.ndarray, y: np.ndarray, batches=None) -> dict:
    pi = per_image(probs, y)
    out = {
        "n": int(len(y)),
        "cer": float(1 - pi["correct"].mean()),
        "acc": float(pi["correct"].mean()),
        "top3": float(pi["top3"].mean()),
        "macro_f1": float(f1_score(y, pi["pred"], labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)),
        "aicme_acc": float(pi["aicme"].mean()),
        "mean_conf": float(pi["conf"].mean()),
        "nll": float(pi["nll"].mean()),
        "ece": float(ece(probs, y)),
        "taxonomy": taxonomy(pi["pred"], y),
    }
    if batches is not None:
        b = np.asarray(batches)
        out["per_batch"] = {k: {"cer": float(1 - pi["correct"][b == k].mean()), "nll": float(pi["nll"][b == k].mean()),
                                "aicme_acc": float(pi["aicme"][b == k].mean())} for k in sorted(set(b))}
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, strata, n: int = 1000, seed: int = 0):
    """Mean of (b - a) over images with a 95% CI; resampling images within each stratum (capture batch)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(b, float) - np.asarray(a, float)
    strata = np.asarray(strata)
    groups = [np.flatnonzero(strata == s) for s in sorted(set(strata))]
    boots = np.empty(n)
    for i in range(n):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        boots[i] = d[idx].mean()
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def predict(model, X: np.ndarray, batch: int = 64) -> np.ndarray:
    return model.predict(X.astype("float32")[..., None] / 255.0, batch_size=batch, verbose=0)


def _main():
    import keras  # noqa: F401  (train venv only)
    from ogham_v2 import real_data
    from train import ledger
    from train.model import build

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--name")
    ap.add_argument("--input", choices=["canonical", "legacy", "autocontrast", "render"])
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    a = ap.parse_args()

    if a.run:
        info = json.loads((a.run / "run.json").read_text())
        model = build(info["probe"])
        model.load_weights(a.run / "model.weights.h5")
        mode, out_dir, row = info["input"], a.run, {k: info[k] for k in ledger.RUN_KEYS if k in info}
    else:
        import keras
        model = keras.models.load_model(a.checkpoint, compile=False)
        mode = a.input
        out_dir = ROOT / "runs" / "legacy_checkpoints" / a.name
        out_dir.mkdir(parents=True, exist_ok=True)
        row = {"exp_id": a.name, "run_id": a.name, "probe": "legacy-checkpoint", "input": mode, "seed": ""}

    X, y, ids, batches, _ = real_data.load_split(a.split, mode)
    probs = predict(model, X)
    m = metrics(probs, y, batches)
    blank = predict(model, real_data.load_blank(mode))[0]
    m["blank_maxprob"], m["blank_pred"] = float(blank.max()), LETTERS[int(blank.argmax())]
    np.savez(out_dir / f"{a.split}_preds.npz", probs=probs, y=y, ids=np.array(ids), batches=np.array(batches))
    (out_dir / f"metrics_{a.split}.json").write_text(json.dumps(m, indent=2))
    ledger.append({**row, "split": a.split, **ledger.flatten(m)})
    print(json.dumps({k: v for k, v in m.items() if k != "per_batch"}, indent=2))


if __name__ == "__main__":
    _main()
