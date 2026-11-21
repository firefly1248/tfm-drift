"""drift_queue.py: every run of plan.md, one process at a time, seeds 0..2 of every dataset first and
the extra seeds of the three small datasets last (the reduction order of plan.md section 7).
Finished runs (a checkpoint exists) are skipped, so a stopped queue resumes where it was.
It stops at the first failure; the run's own log is in logs/runs/.

Study A jobs need SDB's prepared files (see study_a_missing in drift_windows.py). Without them the queue
stops before the first run, unless --skip-study-a leaves the Study A jobs out.

Usage: mkdir -p logs && .venv/bin/python -u -m scripts.drift_queue > logs/queue.log 2>&1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from scripts.drift_run import RESULTS
from scripts.drift_windows import (
    CONCEPT_ONLY,
    STUDY_A,
    STUDY_B,
    seeds,
    study_a_available,
    study_a_missing,
)

MODELS = ("logreg", "lgbm", "xgboost", "catboost", "tabpfn35", "tabicl", "tabfm")
CONCEPT = ("ringnorm", "skin-segmentation", "twonorm")  # CONCEPT_ONLY members that passed the gate
ROOT = Path(__file__).resolve().parent.parent
assert set(CONCEPT) <= set(CONCEPT_ONLY)


def jobs(skip_study_a: bool = False) -> list[tuple[str, str, int]]:
    datasets = [*STUDY_B, *CONCEPT, *([] if skip_study_a else STUDY_A)]
    order = [(s, d) for s in range(3) for d in datasets if s in seeds(d)]
    order += [(s, d) for d in STUDY_B for s in seeds(d) if s >= 3]
    return [(m, d, s) for s, d in order for m in MODELS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-study-a", action="store_true", help="leave out the Study A jobs")
    a = ap.parse_args()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short", "scripts", "sdb.py", "pyproject.toml", "uv.lock"],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip()
    queue = jobs(a.skip_study_a)
    todo = [j for j in queue if not (RESULTS / j[1] / f"{j[0]}_seed{j[2]}.joblib").exists()]
    pending_a = sum(j[1] in STUDY_A for j in todo)
    if pending_a and not study_a_available():
        sys.exit(f"{pending_a} Study A jobs are pending. {study_a_missing()} Or pass --skip-study-a.")
    if a.skip_study_a:
        print(f"Study A skipped: {len(jobs()) - len(queue)} jobs left out", flush=True)
    print(f"code {head[:7]}, uncommitted code files: {dirty or 'none'}; {len(queue)} runs, {len(todo)} to do", flush=True)
    (ROOT / "logs" / "runs").mkdir(parents=True, exist_ok=True)
    start = time.time()
    for k, (model, dataset, seed) in enumerate(todo, start=1):
        t = time.time()
        log = ROOT / "logs" / "runs" / f"{model}_{dataset}_seed{seed}.log"
        with log.open("w") as f:
            code = subprocess.run([sys.executable, "-u", "-m", "scripts.drift_run", model, dataset, str(seed)],
                                  cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        status = "done" if code == 0 else f"FAILED exit {code}"
        print(f"[{k}/{len(todo)}] {status} {model} {dataset} seed {seed} {time.time() - t:.0f} s, "
              f"elapsed {(time.time() - start) / 3600:.2f} h", flush=True)
        if code:
            sys.exit(f"stopped at {model} {dataset} seed {seed}, see {log}")
    print(f"queue done, {(time.time() - start) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
