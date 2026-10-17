"""Experiment cards, figures and summary tables for the report.

  python train/report.py --exp E0b [--parent E0a] [--probe hybrid]   # fills <!-- results:E0b --> in EXPERIMENTS.md
  python train/report.py --summary                                   # docs/results/summary.md
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ogham_v2.alphabet import LETTERS  # noqa: E402
from train import evaluate as ev  # noqa: E402

RESULTS = ROOT / "docs" / "results"
EXPERIMENTS = ROOT / "docs" / "EXPERIMENTS.md"
DECISIONS_CSV = RESULTS / "decisions.csv"
METRICS = [("cer", "CER", True), ("acc", "Accuracy", True), ("top3", "Top-3", True), ("macro_f1", "Macro-F1", False),
           ("aicme_acc", "Aicme acc", True), ("mean_conf", "Mean conf", True), ("nll", "NLL", False),
           ("ece", "ECE", False)]


def load_runs(exp: str, probe: str | None = None):
    runs = []
    for d in sorted((ROOT / "runs" / exp).glob("*/")):
        if not (d / "run.json").exists() or not (d / "dev_preds.npz").exists():
            continue
        info = json.loads((d / "run.json").read_text())
        if probe and info["probe"] != probe:
            continue
        z = np.load(d / "dev_preds.npz")
        runs.append({"dir": d, "info": info, "m": json.loads((d / "metrics_dev.json").read_text()),
                     "probs": z["probs"], "y": z["y"], "ids": list(z["ids"]), "batches": list(z["batches"])})
    return runs


def seed_average(runs):
    """Per-image correctness / NLL averaged over seeds, and mean probabilities."""
    per = [ev.per_image(r["probs"], r["y"]) for r in runs]
    return {"correct": np.mean([p["correct"] for p in per], 0), "nll": np.mean([p["nll"] for p in per], 0),
            "probs": np.mean([r["probs"] for r in runs], 0), "y": runs[0]["y"], "batches": runs[0]["batches"],
            "ids": runs[0]["ids"]}


def agg(runs, key):
    vals = [r["m"][key] if key in r["m"] else r["info"].get(key) for r in runs]
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), 0.0)


def fmt(mean, sd, pct, n):
    if pct:
        return f"{100 * mean:.1f}%" + (f" ± {100 * sd:.1f}" if n > 1 else "")
    return f"{mean:.3f}" + (f" ± {sd:.3f}" if n > 1 else "")


def figures(exp, runs, parent_runs, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    # learning curves
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for r in runs:
        h = list(csv.DictReader(open(r["dir"] / "history.csv")))
        step = [(i + 1) * r["info"]["check_every"] for i in range(len(h))]
        lab = f"seed {r['info']['seed']}"
        axes[0].plot(step, [float(x["val_loss"]) for x in h], label=lab)
        axes[1].plot(step, [float(x["val_accuracy"]) for x in h], label=lab)
        axes[2].plot(step, [float(x["dev_cer"]) for x in h], label=lab)
    for ax, t in zip(axes, ["synthetic val loss", "synthetic val accuracy", "real dev CER (logged only)"]):
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{exp}: learning curves", fontsize=11)
    fig.tight_layout()
    paths["curves"] = out_dir / "learning_curves.png"
    fig.savefig(paths["curves"], dpi=110)
    plt.close(fig)
    # confusion matrix of seed-averaged probabilities
    sa = seed_average(runs)
    cm = np.zeros((20, 20))
    for t, p in zip(sa["y"], sa["probs"].argmax(1)):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(20), LETTERS, fontsize=7)
    ax.set_yticks(range(20), LETTERS, fontsize=7)
    for i, j in zip(*np.nonzero(cm)):
        ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7, color="white" if cm[i, j] > 1 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{exp}: dev confusion (seed-averaged probs)", fontsize=10)
    fig.tight_layout()
    paths["confusion"] = out_dir / "confusion_dev.png"
    fig.savefig(paths["confusion"], dpi=110)
    plt.close(fig)
    # taxonomy vs parent
    keys = ["group_side_BH", "group_diag_across_MA", "group_other", "count_off_by_one", "count_off_more"]
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(len(keys))
    series = [(exp, runs)] + ([("parent", parent_runs)] if parent_runs else [])
    for k, (name, rs) in enumerate(series):
        vals = [np.mean([r["m"]["taxonomy"][key] for r in rs]) for key in keys]
        ax.bar(x + k * 0.38, vals, 0.38, label=name)
    ax.set_xticks(x + 0.19, [k.replace("group_", "grp:").replace("count_", "cnt:") for k in keys], fontsize=8)
    ax.set_ylabel("errors (mean over seeds)")
    ax.legend(fontsize=8)
    ax.set_title(f"{exp}: dev error taxonomy", fontsize=10)
    fig.tight_layout()
    paths["taxonomy"] = out_dir / "taxonomy_dev.png"
    fig.savefig(paths["taxonomy"], dpi=110)
    plt.close(fig)
    return paths


def card(exp, parent, probe):
    runs = load_runs(exp, probe)
    if not runs:
        raise SystemExit(f"no runs for {exp}")
    parent_runs = load_runs(parent, probe) if parent else []
    n, pn = len(runs), len(parent_runs)
    lines = [f"**Probe:** {probe} | **input:** {runs[0]['info']['input']} | **dataset:** `{runs[0]['info']['dataset']}` "
             f"({runs[0]['info']['n_per_class']}/class) | **seeds:** {n} | "
             f"**steps used:** {agg(runs, 'steps_used')[0]:.0f} (best at {agg(runs, 'best_step')[0]:.0f}) | "
             f"**train time:** {agg(runs, 'train_minutes')[0]:.1f} min/run", ""]
    head = f"| Metric (dev, pen-40) | {exp} (n={n}) |" + (f" {parent} (n={pn}) | Δ vs parent [95% CI] |" if pn else "")
    lines += [head, "|---|---|" + ("---|---|" if pn else "")]
    sa = seed_average(runs)
    psa = seed_average(parent_runs) if pn else None
    boot = {}
    if pn:
        d, lo, hi = ev.paired_bootstrap(psa["correct"], sa["correct"], sa["batches"])
        boot["cer"] = (-d, -hi, -lo)
        boot["acc"] = (d, lo, hi)
        boot["nll"] = ev.paired_bootstrap(psa["nll"], sa["nll"], sa["batches"])
    for key, label, pct in METRICS:
        m, sd = agg(runs, key)
        row = f"| {label} | {fmt(m, sd, pct, n)} |"
        if pn:
            pm, psd = agg(parent_runs, key)
            row += f" {fmt(pm, psd, pct, pn)} |"
            if key in boot:
                d, lo, hi = boot[key]
                s = 100 if pct else 1
                u = " pt" if pct else ""
                row += f" {s * d:+.1f}{u} [{s * lo:+.1f}, {s * hi:+.1f}] |" if pct else f" {d:+.3f} [{lo:+.3f}, {hi:+.3f}] |"
            else:
                row += f" {(m - pm) * (100 if pct else 1):+.{1 if pct else 3}f}{' pt' if pct else ''} |"
        lines.append(row)
    for key, label in [("synth_val_acc", "Synthetic val acc"), ("gap", "Synth→real gap (val acc − dev acc)")]:
        if key == "gap":
            vals = [r["info"]["synth_val_acc"] - r["m"]["acc"] for r in runs]
            m, sd = float(np.mean(vals)), float(np.std(vals))
            pv = [r["info"]["synth_val_acc"] - r["m"]["acc"] for r in parent_runs]
        else:
            m, sd = agg(runs, key)
            pv = [r["info"][key] for r in parent_runs]
        row = f"| {label} | {fmt(m, sd, True, n)} |"
        if pn:
            row += f" {fmt(float(np.mean(pv)), float(np.std(pv)), True, pn)} | {100 * (m - np.mean(pv)):+.1f} pt |"
        lines.append(row)
    def stress(rs):
        vals = [json.loads((r["dir"] / "metrics_devstress.json").read_text()) for r in rs if (r["dir"] / "metrics_devstress.json").exists()]
        return vals
    st, pst = stress(runs), stress(parent_runs)
    if st:
        m = float(np.mean([v["acc"] for v in st]))
        row = f"| Dev-stress acc (supplementary, DR-022) | {100 * m:.1f}% (n={len(st)}) |"
        if pn and pst:
            pm = float(np.mean([v["acc"] for v in pst]))
            row += f" {100 * pm:.1f}% (n={len(pst)}) | {100 * (m - pm):+.1f} pt |"
        lines.append(row)
        kinds = list(st[0]["per_kind"])
        lines += ["", "| Dev-stress by degradation | " + " | ".join(kinds) + " |", "|---|" + "---|" * len(kinds)]
        for name, vs in [(exp, st)] + ([(parent, pst)] if pst else []):
            lines.append(f"| {name} | " + " | ".join(f"{100 * np.mean([v['per_kind'][k] for v in vs]):.0f}%" for k in kinds) + " |")
    blanks = ", ".join(f"{r['m']['blank_pred']} ({r['m']['blank_maxprob']:.2f})" for r in runs)
    lines += ["", f"**Blank-page control:** {blanks}", ""]
    batches = sorted(runs[0]["m"]["per_batch"])
    lines += ["| Per batch (dev CER) | " + " | ".join(batches) + " |", "|---|" + "---|" * len(batches)]
    for name, rs in [(exp, runs)] + ([(parent, parent_runs)] if pn else []):
        lines.append(f"| {name} | " + " | ".join(f"{100 * np.mean([r['m']['per_batch'][b]['cer'] for r in rs]):.1f}%"
                                                  for b in batches) + " |")
    tax_keys = ["group_side_BH", "group_diag_across_MA", "group_other", "count_off_by_one", "count_off_more"]
    lines += ["", "| Error taxonomy (mean errors/seed) | " + " | ".join(tax_keys) + " |", "|---|" + "---|" * len(tax_keys)]
    for name, rs in [(exp, runs)] + ([(parent, parent_runs)] if pn else []):
        lines.append(f"| {name} | " + " | ".join(f"{np.mean([r['m']['taxonomy'][k] for r in rs]):.1f}" for k in tax_keys) + " |")
    figs = figures(exp, runs, parent_runs, RESULTS / "figures" / exp)
    lines += ["", "**Figures:** " + " · ".join(f"[{k}]({p.relative_to(ROOT / 'docs')})" for k, p in figs.items())]
    contact = RESULTS / "figures" / exp / "contact_sheet.png"
    if contact.exists():
        lines[-1] += f" · [contact sheet]({contact.relative_to(ROOT / 'docs')})"
    block = "\n".join(lines)
    text = EXPERIMENTS.read_text()
    pat = re.compile(rf"(<!-- results:{re.escape(exp)} -->)(.*?)(<!-- /results:{re.escape(exp)} -->)", re.S)
    if pat.search(text):
        EXPERIMENTS.write_text(pat.sub(lambda mt: f"{mt.group(1)}\n{block}\n{mt.group(3)}", text))
        print(f"updated card {exp} in EXPERIMENTS.md")
    out = RESULTS / "cards" / f"{exp}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(block + "\n")
    print(block)


def summary():
    rows = list(csv.DictReader(open(RESULTS / "ledger.csv")))
    decisions = {}
    if DECISIONS_CSV.exists():
        decisions = {r["exp_id"]: r for r in csv.DictReader(open(DECISIONS_CSV))}
    out = ["# Results summary (auto-generated by `train/report.py --summary`)", ""]
    for split, title in [("dev", "Dev (pen-40): every decision is made here"), ("test", "Test (marker-40): frozen winners only")]:
        groups = defaultdict(list)
        for r in rows:
            if r["split"] == split:
                groups[(r["exp_id"], r["probe"], r["input"])].append(r)
        if not groups:
            continue
        out += [f"## {title}", "", "| Exp | Probe | Input | n | CER | Acc | Top-3 | Macro-F1 | Aicme | Conf | NLL | Synth val | Blank | Decision |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for (exp, probe, inp), rs in groups.items():
            def mean(k, pct=True):
                v = [float(r[k]) for r in rs if r.get(k) not in ("", None)]
                return "" if not v else (f"{100 * np.mean(v):.1f}%" if pct else f"{np.mean(v):.2f}")
            dec = decisions.get(exp, {}).get("decision", "")
            blank = rs[-1].get("blank_pred", "") + (f" ({float(rs[-1]['blank_maxprob']):.2f})" if rs[-1].get("blank_maxprob") else "")
            out.append(f"| {exp} | {probe} | {inp} | {len(rs)} | {mean('cer')} | {mean('acc')} | {mean('top3')} | "
                       f"{mean('macro_f1', False)} | {mean('aicme_acc')} | {mean('mean_conf')} | {mean('nll', False)} | "
                       f"{mean('synth_val_acc')} | {blank} | {dec} |")
        out.append("")
    (RESULTS / "summary.md").write_text("\n".join(out))
    print("\n".join(out))


def waterfall(exps, name):
    """Overview figure: mean dev CER and NLL along an experiment chain, coloured by decision."""
    decisions = {}
    if DECISIONS_CSV.exists():
        decisions = {r["exp_id"]: r["decision"] for r in csv.DictReader(open(DECISIONS_CSV))}
    cer, nll, sd, ns = [], [], [], []
    for e in exps:
        rs = load_runs(e)
        vals = [r["m"]["cer"] for r in rs]
        cer.append(np.mean(vals))
        sd.append(np.std(vals))
        nll.append(np.mean([r["m"]["nll"] for r in rs]))
        ns.append(len(rs))
    colour = lambda d: "#2a9d55" if d.startswith("ADOPTED") else "#c44e52" if d.startswith("REJECTED") else "#8c8c8c"  # noqa: E731
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.8))
    x = np.arange(len(exps))
    a1.bar(x, 100 * np.array(cer), yerr=100 * np.array(sd), color=[colour(decisions.get(e, "")) for e in exps], capsize=3)
    for i, (c, n) in enumerate(zip(cer, ns)):
        a1.text(i, 100 * c + 1.5, f"{100 * c:.1f}%\n(n={n})", ha="center", fontsize=8)
    a1.scatter(x, 100 * np.array(cer), color=[colour(decisions.get(e, "")) for e in exps], zorder=3, s=25)
    a1.set_ylim(0, max(5, 100 * max(cer)) * 1.3)
    a1.set_xticks(x, exps, rotation=30, fontsize=8)
    a1.set_ylabel("dev CER (%)")
    a1.set_title("Dev CER along the chain (green = adopted, grey = reference / finding / inconclusive)", fontsize=9, pad=10)
    a2.plot(x, nll, "o-")
    a2.set_yscale("symlog", linthresh=0.01)
    a2.set_xticks(x, exps, rotation=30, fontsize=8)
    a2.set_ylabel("dev NLL (symlog)")
    a2.set_title("Dev NLL", fontsize=9)
    fig.tight_layout()
    out = RESULTS / "figures" / "overview" / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--waterfall", nargs="+")
    ap.add_argument("--name", default="waterfall")
    ap.add_argument("--exp")
    ap.add_argument("--parent")
    ap.add_argument("--probe", default="hybrid")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.exp:
        card(a.exp, a.parent, a.probe)
    if a.summary:
        summary()
    if a.waterfall:
        waterfall(a.waterfall, a.name)


if __name__ == "__main__":
    main()
