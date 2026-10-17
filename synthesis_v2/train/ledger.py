"""Append-only results ledger: docs/results/ledger.csv (one row per run x split)."""
import csv
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "results" / "ledger.csv"

RUN_KEYS = ["exp_id", "parent", "run_id", "probe", "input", "seed", "dataset", "n_per_class", "batch", "device",
            "steps_used", "best_step", "train_minutes", "synth_val_acc", "synth_val_loss"]
METRIC_KEYS = ["n", "cer", "acc", "top3", "macro_f1", "aicme_acc", "mean_conf", "nll", "ece", "gap",
               "blank_maxprob", "blank_pred", "tax_group_side_BH", "tax_group_diag_across_MA", "tax_group_other",
               "tax_count_off_by_one", "tax_count_off_more"]
COLUMNS = ["timestamp", "git_sha"] + RUN_KEYS + ["split"] + METRIC_KEYS


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return ""


def flatten(m: dict) -> dict:
    out = {k: v for k, v in m.items() if k not in ("taxonomy", "per_batch")}
    for k, v in m.get("taxonomy", {}).items():
        if k != "correct":
            out[f"tax_{k}"] = v
    return out


def append(row: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "git_sha": git_sha(), **row}
    if "synth_val_acc" in row and "acc" in row and row.get("synth_val_acc") not in ("", None):
        row["gap"] = round(float(row["synth_val_acc"]) - float(row["acc"]), 4)
    with open(LEDGER, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
