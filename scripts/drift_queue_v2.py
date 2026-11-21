"""drift_queue_v2.py: every run of plan_v2.md, one process at a time. Seeds 0 to 2 of every dataset first (Study B2,
then Study C), then the later seeds of Study B2 (plan_v2.md section 10 lists what to cut if the budget is short).
Finished runs (a checkpoint exists) are skipped, so a stopped queue resumes where it was. It stops at the first
failure; the run's own log is in logs/runs_v2/.

--pilot runs one seed of three Study B2 and two Study C datasets, all models, into results/runs_v2_pilot/, for timings only.
--only b2 or --only c leaves the other study out.

Usage: mkdir -p logs && .venv/bin/python -u -m scripts.drift_queue_v2 [--pilot] [--only b2|c] > logs/queue_v2.log 2>&1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from scripts.drift_run import MODELS
from scripts.drift_run_v2 import PILOT, RESULTS, population

ROOT = Path(__file__).resolve().parent.parent
PILOT_JOBS = {"b2": ["drrisk-retinopathy", "NATICUSdroid", "Fitness_Club"], "c": ["adult", "Higgs"]}
C_SEEDS = 3


def jobs(only: str | None = None, pilot: bool = False) -> list[tuple[str, str, str, int]]:
    """(study, model, dataset, seed) in running order."""
    pairs = []
    b2 = population("b2")
    c = population("c")
    if pilot:
        return [(s, m, d, 0) for s in ("b2", "c") if only in (None, s) for d in PILOT_JOBS[s] for m in MODELS]
    if only in (None, "b2"):
        pairs += [(0, "b2", d, int(n)) for d, n in zip(b2.name, b2.seeds)]
    if only in (None, "c"):
        pairs += [(0, "c", d, C_SEEDS) for d in c.name]
    order = [(s, study, d) for s in range(3) for _, study, d, n in pairs if s < n]
    order += [(s, study, d) for _, study, d, n in pairs if study == "b2" for s in range(3, n)]
    return [(study, m, d, s) for s, study, d in order for m in MODELS]


def done(study: str, model: str, dataset: str, seed: int, pilot: bool) -> bool:
    return ((PILOT if pilot else RESULTS) / study / dataset / f"{model}_seed{seed}.joblib").exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--only", choices=["b2", "c"])
    a = ap.parse_args()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short", "scripts", "sdb.py", "pyproject.toml", "uv.lock"],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip()
    queue = jobs(a.only, a.pilot)
    todo = [j for j in queue if not done(*j, a.pilot)]
    print(f"code {head[:7]}, uncommitted code files: {dirty or 'none'}; {len(queue)} runs, {len(todo)} to do", flush=True)
    (ROOT / "logs" / "runs_v2").mkdir(parents=True, exist_ok=True)
    start = time.time()
    for k, (study, model, dataset, seed) in enumerate(todo, start=1):
        t = time.time()
        log = ROOT / "logs" / "runs_v2" / f"{study}_{model}_{dataset}_seed{seed}.log"
        cmd = [sys.executable, "-u", "-m", "scripts.drift_run_v2", study, model, dataset, str(seed)] + (["--pilot"] if a.pilot else [])
        with log.open("w") as f:
            code = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        status = "done" if code == 0 else f"FAILED exit {code}"
        print(f"[{k}/{len(todo)}] {status} {study} {model} {dataset} seed {seed} {time.time() - t:.0f} s, "
              f"elapsed {(time.time() - start) / 3600:.2f} h", flush=True)
        if code:
            sys.exit(f"stopped at {study} {model} {dataset} seed {seed}, see {log}")
    print(f"queue done, {(time.time() - start) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
