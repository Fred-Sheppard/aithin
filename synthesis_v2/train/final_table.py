"""Final held-out table (DR-003, DR-025): evaluate FROZEN configurations once on marker-40 and tabulate.

  python train/final_table.py [--evaluate]      # --evaluate runs evaluate.py --split test for runs lacking it

The pipeline was frozen before this table was produced; nothing here feeds back into any decision.
Rows: legacy checkpoints, the experiment chain (both probes where available), E3-ref (detector + rules).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import evaluate as ev  # noqa: E402

CHAIN = ["E0a", "E0b", "E0c", "E1a-prior", "E1a", "E1b", "E1c", "E1d", "E2a", "E2b", "E2c", "E2c-prior", "E2d-v2", "E3a", "E3b", "E3c"]
REFERENCE = "E0b"
OUT = ROOT / "docs" / "results"


def runs_of(exp, probe):
    return sorted(d for d in (ROOT / "runs" / exp).glob(f"{probe}-*") if (d / "model.weights.h5").exists())


def ensure_test(run: Path):
    if not (run / "test_preds.npz").exists():
        subprocess.run([str(ROOT / ".venv-train/bin/python"), "train/evaluate.py", "--run", str(run), "--split", "test"],
                       cwd=ROOT, check=True, capture_output=True)


def load(run: Path):
    z = np.load(run / "test_preds.npz")
    m = json.loads((run / "metrics_test.json").read_text())
    return z, m


def row(name, probe, items):
    ms = [m for _, m in items]
    agg = lambda k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms])))  # noqa: E731
    per = {b: float(np.mean([m["per_batch"][b]["cer"] for m in ms])) for b in ms[0]["per_batch"]}
    pi = [ev.per_image(z["probs"], z["y"]) for z, _ in items]
    return {"name": name, "probe": probe, "n": len(items), "cer": agg("cer"), "acc": agg("acc"), "top3": agg("top3"),
            "macro_f1": agg("macro_f1"), "aicme": agg("aicme_acc"), "conf": agg("mean_conf"), "nll": agg("nll"),
            "per_batch": per, "blank": ", ".join(f"{m['blank_pred']}({m['blank_maxprob']:.2f})" for m in ms),
            "correct": np.mean([p["correct"] for p in pi], 0), "batches": list(items[0][0]["batches"]),
            "tax": {k: float(np.mean([m["taxonomy"][k] for m in ms])) for k in ms[0]["taxonomy"]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate", action="store_true")
    a = ap.parse_args()
    rows = []
    for name in ["legacy-hybrid", "legacy-augmented", "legacy-hybrid-canonical", "legacy-augmented-canonical"]:
        d = ROOT / "runs" / "legacy_checkpoints" / name
        if (d / "test_preds.npz").exists():
            rows.append(row(name, "checkpoint", [load(d)]))
    for exp in CHAIN:
        for probe in ("hybrid", "alt"):
            runs = runs_of(exp, probe)
            if not runs:
                continue
            if a.evaluate:
                for r in runs:
                    ensure_test(r)
            items = [load(r) for r in runs if (r / "test_preds.npz").exists()]
            if items:
                rows.append(row(exp, probe, items))
    e3 = ROOT / "runs" / "E3-ref" / "rules-proxy" / "metrics_test.json"
    ref_rules = json.loads(e3.read_text()) if e3.exists() else None

    refs = {r["probe"]: r for r in rows if r["name"] == REFERENCE}
    lines = ["# Final held-out table: marker-40 (the team's wild test set)", "",
             "*Computed once, after the pipeline was frozen (DR-025). Means ± SD over seeds. Δ = paired bootstrap vs "
             f"{REFERENCE} (same probe) on per-image accuracy averaged over seeds, 95% CI.*", "",
             "| Config | Probe | Seeds | CER | Acc | Δ acc vs E0b [95% CI] | Top-3 | Macro-F1 | Aicme | Mean conf | NLL | marker-1 CER | marker-2 CER |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    pct = lambda v: f"{100 * v[0]:.1f}" + (f" ± {100 * v[1]:.1f}" if v[1] > 0 else "")  # noqa: E731
    for r in rows:
        ref = refs.get(r["probe"])
        delta = ""
        if ref is not None and r is not ref and len(ref["correct"]) == len(r["correct"]):
            d, lo, hi = ev.paired_bootstrap(ref["correct"], r["correct"], r["batches"])
            delta = f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
        lines.append(f"| {r['name']} | {r['probe']} | {r['n']} | {pct(r['cer'])}% | {pct(r['acc'])}% | {delta} | {pct(r['top3'])}% | "
                     f"{r['macro_f1'][0]:.3f} | {pct(r['aicme'])}% | {pct(r['conf'])}% | {r['nll'][0]:.2f} | "
                     + " | ".join(f"{100 * v:.1f}%" for v in r["per_batch"].values()) + " |")
    if ref_rules:
        lines.append(f"| E3-ref (detector + rules) | rules | — | {100 * ref_rules['cer']:.1f}% | {100 * ref_rules['acc']:.1f}% | | — | "
                     f"{ref_rules['macro_f1']:.3f} | {100 * ref_rules['aicme_acc']:.1f}% | — | — | "
                     + " | ".join(f"{100 * v['cer']:.1f}%" for v in ref_rules["per_batch"].values()) + " |")
    lines += ["", "## Error taxonomy on test (mean errors per seed)", "",
              "| Config | Probe | B↔H side | M↔A diag/across | other group | count ±1 | count ±>1 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        t = r["tax"]
        lines.append(f"| {r['name']} | {r['probe']} | {t['group_side_BH']:.1f} | {t['group_diag_across_MA']:.1f} | {t['group_other']:.1f} | "
                     f"{t['count_off_by_one']:.1f} | {t['count_off_more']:.1f} |")
    (OUT / "final_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    hy = [r for r in rows if r["probe"] in ("hybrid", "checkpoint")]
    fig, ax = plt.subplots(figsize=(14, 4))
    x = np.arange(len(hy))
    ax.bar(x, [100 * r["acc"][0] for r in hy], yerr=[100 * r["acc"][1] for r in hy], capsize=3,
           color=["#8c8c8c" if r["probe"] == "checkpoint" else "#4c72b0" for r in hy])
    if ref_rules:
        ax.axhline(100 * ref_rules["acc"], color="#2a9d55", ls="--", label=f"E3-ref detector + rules ({100 * ref_rules['acc']:.1f}%)")
        ax.legend(fontsize=8)
    for i, r in enumerate(hy):
        ax.text(i, 100 * r["acc"][0] + 2, f"{100 * r['acc'][0]:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, [r["name"] for r in hy], rotation=35, fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("marker-40 accuracy (%)")
    ax.set_title("Held-out test (marker-40): legacy checkpoints (grey) vs synthesis-v2 chain (hybrid probe, mean over seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "overview" / "final_test.png", dpi=110)


if __name__ == "__main__":
    main()
