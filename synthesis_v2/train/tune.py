"""E4: tune coarse synthesis knobs against real dev NLL with Optuna (plan Phase 5). Run in the train venv.

  python train/tune.py --base <best-config> --input canonical|render --trials 30 [--n-per-class 212] [--steps 2000]

Each trial: resolve base config -> apply knob multipliers -> generate (synth venv, same data seed for every trial
= common random numbers) -> train the probe with a reduced budget -> objective = dev NLL averaged with equal weight
over the two pen batches. Excluded from the search (DR-003 guard): pen mixture and stroke widths.
Marker-40 (test) is never used.
"""
import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import optuna
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ogham_v2 import config as cfgmod  # noqa: E402

SYNTH_PY = ROOT / ".venv-synth/bin/python"
TRAIN_PY = ROOT / ".venv-train/bin/python"

# knob: (low, high, [(dotted config path, 'scale'|'set'), ...]); knobs whose paths are absent are skipped
KNOBS = {
    "geom_jitter": (0.5, 2.0, [("geometry.angle_sd", "scale"), ("geometry.spacing_jitter", "scale"),
                               ("geometry.length_jitter", "scale"), ("geometry.balance_jitter", "scale"),
                               ("geometry.tilt_sd", "scale")]),
    "imperfection": (0.0, 2.0, [("imperfection.curvature", "scale"), ("imperfection.wave", "scale"),
                                ("imperfection.tremor", "scale"), ("imperfection.stem_curvature", "scale"),
                                ("imperfection.p_hook", "scale"), ("imperfection.p_stem_gap", "scale")]),
    "p_hard": (0.0, 0.5, [("hard.p", "set")]),
    "p_distractor": (0.0, 0.5, [("distractors.p_stem_extension", "set"), ("distractors.p_stray", "set")]),
    "paper": (0.0, 2.0, [("appearance.paper.grain", "scale"), ("appearance.paper.lowfreq", "scale"),
                         ("appearance.paper.fibre_p", "scale")]),
    "lighting": (0.0, 2.0, [("appearance.lighting.gradient", "scale"), ("appearance.lighting.shadow_p", "scale"),
                            ("appearance.lighting.vignette", "scale")]),
    "blur": (0.0, 2.0, [("appearance.capture.blur_p", "scale"), ("appearance.capture.blur", "scale"),
                        ("appearance.capture.motion_p", "scale")]),
    "noise_jpeg": (0.0, 2.0, [("appearance.capture.noise", "scale"), ("appearance.capture.jpeg_p", "scale"),
                              ("appearance.capture.rescale_p", "scale")]),
    "artefacts": (0.0, 2.0, [("appearance.artefacts.showthrough_p", "scale"), ("appearance.artefacts.edge_band_p", "scale"),
                             ("appearance.artefacts.speckle_p", "scale"), ("appearance.artefacts.loose_p", "scale")]),
    "det_errors": (0.0, 3.0, [("representation.errors.drop", "scale"), ("representation.errors.split", "scale"),
                              ("representation.errors.merge", "scale"), ("representation.errors.spurious", "scale"),
                              ("representation.errors.jitter", "scale")]),
}
PROB_KEYS = ("_p", ".p", "drop", "split", "merge", "spurious")


def _get(cfg, path):
    cur = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or cur.get(k) is None:
            return None
        cur = cur[k]
    return cur


def _set(cfg, path, value):
    keys = path.split(".")
    cur = cfg
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = value


def apply_knobs(base: dict, values: dict) -> dict:
    cfg = copy.deepcopy(base)
    for knob, v in values.items():
        for path, mode in KNOBS[knob][2]:
            cur = _get(cfg, path)
            if cur is None:
                continue
            if mode == "set":
                new = v
            elif isinstance(cur, list):
                lo = cur[0]
                new = [lo, lo + (cur[1] - lo) * v]                 # scale the range's spread above its minimum
            else:
                new = cur * v
            if any(path.endswith(k) for k in PROB_KEYS) and not isinstance(new, list):
                new = min(1.0, max(0.0, new))
            _set(cfg, path, new)
    return cfg


def active_knobs(base: dict) -> list:
    return [k for k, (_, _, paths) in KNOBS.items() if any(_get(base, p) is not None for p, _ in paths)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--input", default="canonical")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--n-per-class", type=int, default=212)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--study", default="E4")
    a = ap.parse_args()
    base = cfgmod.load(a.base)
    knobs = active_knobs(base)
    out = ROOT / "runs" / a.study
    (out / "trials").mkdir(parents=True, exist_ok=True)
    print(f"tuning {a.base} over knobs: {knobs}", flush=True)

    def objective(trial):
        values = {k: trial.suggest_float(k, KNOBS[k][0], KNOBS[k][1]) for k in knobs}
        cfg = apply_knobs(base, values)
        cfg["name"] = f"{a.study}-t{trial.number:03d}"
        cfg.pop("base", None)
        cpath = out / "trials" / f"t{trial.number:03d}.yaml"
        cpath.write_text(yaml.safe_dump(cfg, sort_keys=False))
        data = ROOT / "data" / "tune" / f"{a.study}-t{trial.number:03d}"
        subprocess.run([str(SYNTH_PY), "-m", "ogham_v2.generate", "--config", str(cpath), "--out", str(data),
                        "--n-per-class", str(a.n_per_class), "--val-n", "50", "--seed", "0", "--workers", "6"],
                       cwd=ROOT, check=True, capture_output=True)
        run_name = f"t{trial.number:03d}"
        subprocess.run([str(TRAIN_PY), "train/train.py", "--data", str(data), "--exp", f"{a.study}-tune",
                        "--parent", a.base, "--seed", "0", "--input", a.input, "--max-steps", str(a.steps),
                        "--run-name", run_name], cwd=ROOT, check=True, capture_output=True)
        m = json.loads((ROOT / "runs" / f"{a.study}-tune" / run_name / "metrics_dev.json").read_text())
        nll = sum(b["nll"] for b in m["per_batch"].values()) / len(m["per_batch"])
        trial.set_user_attr("dev_cer", m["cer"])
        trial.set_user_attr("dev_nll", m["nll"])
        shutil.rmtree(data, ignore_errors=True)
        return nll

    study = optuna.create_study(study_name=a.study, direction="minimize", storage=f"sqlite:///{out / 'optuna.db'}",
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=a.trials)
    rows = sorted(study.trials, key=lambda t: t.value if t.value is not None else 1e9)
    summary = [{"trial": t.number, "objective_nll": t.value, **t.params, **t.user_attrs} for t in rows]
    (out / "trials_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary[:5], indent=2))


if __name__ == "__main__":
    main()
