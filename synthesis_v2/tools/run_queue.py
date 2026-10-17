"""Sequential job queue for overnight runs (resumable; skips jobs already done).

  python tools/run_queue.py queues/<name>.yaml [--retry-failed]

Queue YAML: a list of {name, venv: synth|train, cmd: "<command run from synthesis_v2/>"}.
Touch runs/queue/<queue-name>/STOP to stop cleanly after the current job.
State and logs: runs/queue/<queue-name>/{state.json, <job>.log}.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
VENVS = {"synth": ROOT / ".venv-synth/bin/python", "train": ROOT / ".venv-train/bin/python"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queue", type=Path)
    ap.add_argument("--retry-failed", action="store_true")
    a = ap.parse_args()
    jobs = yaml.safe_load(a.queue.read_text())
    qdir = ROOT / "runs" / "queue" / a.queue.stem
    qdir.mkdir(parents=True, exist_ok=True)
    state_file = qdir / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    for job in jobs:
        if (qdir / "STOP").exists():
            print("STOP file found - exiting before the next job", flush=True)
            (qdir / "STOP").unlink()
            break
        name = job["name"]
        prev = state.get(name, {}).get("status")
        if prev == "done" or (prev == "failed" and not a.retry_failed):
            print(f"skip {name} ({prev})", flush=True)
            continue
        cmd = job["cmd"].replace("PY ", f"{VENVS[job.get('venv', 'train')]} ", 1)
        print(f"run  {name}: {cmd}", flush=True)
        t0 = time.time()
        with open(qdir / f"{name}.log", "w") as log:
            rc = subprocess.call(cmd, shell=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        state[name] = {"status": "done" if rc == 0 else "failed", "rc": rc, "minutes": round((time.time() - t0) / 60, 1)}
        state_file.write_text(json.dumps(state, indent=2))
        print(f"     {name}: {state[name]}", flush=True)
    failed = [k for k, v in state.items() if v["status"] == "failed"]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
