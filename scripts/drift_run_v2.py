"""drift_run_v2.py: one model on one dataset and seed of plan_v2.md; per-block probabilities of label 1 go to
results/runs_v2/<study>/<dataset>/<model>_seed<seed>.joblib. It reuses v1's Run, tuning and models (drift_run.py).

Study B2 (`b2`): arms control, tilt, xup and xdown. Strategies S0 (a fit on B1, on B1x for xup and xdown),
U (a fit on the full-range subsample of P1, scored on the xup and xdown blocks), S1, S2 and, for boosters, S1r
(a full re-tune on every S1 window). Study C (`c`): S0 only, at three window sizes, with the tree limit of
boosters raised to 2,000. Checkpoint keys are (arm, strategy, block); in Study C the strategy carries the window,
S0@1000 and U@1000.

The run prints timings only, never a score. --pilot writes to results/runs_v2_pilot/ instead.

Usage: uv run python -m scripts.drift_run_v2 <study> <model> <dataset> <seed> [--pilot]
"""
from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import sdb
from scripts import drift_run as v1
from scripts.drift_data_v2 import load_openml, load_sdb
from scripts.drift_run import BOOSTERS, MODELS, Run, check_versions
from scripts.drift_windows import minority_is_one, pc_score, pools, windows
from scripts.drift_windows_v2 import (
    LEVELS_C,
    WINDOWS_C,
    block_shares,
    block_size,
    extrap_windows,
    study_c,
    z_extrap,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "runs_v2"
PILOT = ROOT / "results" / "runs_v2_pilot"
C_ROWS = 36_000
MAX_TREES_C = 2_000
ARMS = {"none": None, "cov": None, "xup": "up", "xdown": "down"}


class _Wide:
    """A trial whose n_estimators bound is raised; every other suggestion is the trial's own."""

    def __init__(self, trial, max_trees: int):
        self.trial, self.max_trees = trial, max_trees

    def suggest_float(self, *a, **k):
        return self.trial.suggest_float(*a, **k)

    def suggest_int(self, name, low, high, **k):
        return self.trial.suggest_int(name, low, self.max_trees if name == "n_estimators" else high, **k)


@contextmanager
def trees(max_trees: int):
    """v1's search space with n_estimators up to max_trees (Study C). v1's tune takes `space` from drift_run."""
    original = v1.space
    v1.space = lambda model, trial, narrow: original(model, _Wide(trial, max_trees), narrow)
    try:
        yield
    finally:
        v1.space = original


def strat(lab: np.ndarray, seed: int) -> list:
    return list(StratifiedKFold(4, shuffle=True, random_state=seed).split(np.zeros(len(lab)), lab))


def score(r: Run, stage: str, m, wanted: dict) -> None:
    """Predict the union of the wanted blocks once and store each block: wanted maps a checkpoint key to (rows, labels)."""
    union = np.unique(np.concatenate([rows for rows, _ in wanted.values()]))
    p = r.proba(stage, m, union)
    for key, (rows, lab) in wanted.items():
        r.store(key, rows, p[np.searchsorted(union, rows)], lab)


def tune_fit(r: Run, key: str, idx: np.ndarray, y: np.ndarray, stage: str):
    r.tune(key, idx, y[idx], strat(y[idx], r.seed))
    return r.fit(stage, r.params[key], idx, y[idx])


def run_b2(r: Run, X: pd.DataFrame, y: np.ndarray) -> None:
    P = pools(y, r.seed)
    W = {sc: windows(sc, r.seed, y, P, pc_score(X, r.cat_cols), None) for sc in ("none", "cov")}
    zx = z_extrap(X, r.cat_cols, y)
    mx = block_size(zx, P)
    E = extrap_windows(y, P, zx, r.seed, mx)
    arm = {"none": W["none"], "cov": W["cov"], "xup": E["up"], "xdown": E["down"]}
    blocks = {a: arm[a]["blocks"] for a in arm}
    r.extra = dict(mx=mx, shares=[dict(arm=f"x{d}", block=h, share_z=s, share_feature=f, rows=len(b[0]))
                                  for d in ("up", "down") for h, b in enumerate(E[d]["blocks"], start=1)
                                  for s, f in [block_shares(X, zx, E[d]["b1x"], b[0], d)]])

    def wanted(arms: list[str], strategy: str, hs=(1, 2, 3, 4)) -> dict:
        return {(a, strategy, h): blocks[a][h - 1] for a in arms for h in hs}

    r.tune("B1", P[0], y[P[0]], strat(y[P[0]], r.seed))
    for h in (2, 3, 4):
        idx, lab = arm["none"]["s2"][h]
        r.tune(f"S2_{h}", idx, lab, strat(lab, r.seed))
    for d in ("up", "down"):
        b1x = E[d]["b1x"]
        r.tune(f"B1x_{d}", b1x, y[b1x], strat(y[b1x], r.seed))
        for h in (2, 3, 4):
            idx, lab = E[d]["s2"][h]
            r.tune(f"S2_x{d}_{h}", idx, lab, strat(lab, r.seed))
    r.tune("U", E["u"], y[E["u"]], strat(y[E["u"]], r.seed))

    score(r, "S0", r.fit("S0", r.params["B1"], P[0], y[P[0]]), wanted(["none", "cov"], "S0"))
    for d in ("up", "down"):
        b1x = E[d]["b1x"]
        score(r, f"S0 x{d}", r.fit(f"S0 x{d}", r.params[f"B1x_{d}"], b1x, y[b1x]), wanted([f"x{d}"], "S0"))
    score(r, "U", r.fit("U", r.params["U"], E["u"], y[E["u"]]), {**wanted(["xup"], "U"), **wanted(["xdown"], "U")})

    for a in arm:
        d = ARMS[a]
        for h in (2, 3, 4):
            p1 = r.params[f"B1x_{d}"] if d else r.params["B1"]
            p2 = r.params[f"S2_x{d}_{h}"] if d else r.params[f"S2_{h}"]
            blk = {(a, "S1", h): blocks[a][h - 1]}
            score(r, f"S1 {a} {h}", r.fit(f"S1 {a} {h}", p1, *arm[a]["s1"][h]), blk)
            score(r, f"S2 {a} {h}", r.fit(f"S2 {a} {h}", p2, *arm[a]["s2"][h]), {(a, "S2", h): blocks[a][h - 1]})
            if r.model in BOOSTERS:
                idx, lab = arm[a]["s1"][h]
                r.tune(f"S1r_{a}_{h}", idx, lab, strat(lab, r.seed))
                score(r, f"S1r {a} {h}", r.fit(f"S1r {a} {h}", r.params[f"S1r_{a}_{h}"], idx, lab), {(a, "S1r", h): blocks[a][h - 1]})


def run_c(r: Run, X: pd.DataFrame, y: np.ndarray) -> None:
    z = z_extrap(X, r.cat_cols, y, max_levels=LEVELS_C)
    c = study_c(y, z, r.seed)
    P = c["P"]
    W = {sc: windows(sc, r.seed, y, P, z, None) for sc in ("none", "cov")}
    blocks = {"none": W["none"]["blocks"], "cov": W["cov"]["blocks"], "xup": c["blocks"]["up"], "xdown": c["blocks"]["down"]}
    r.extra = dict(mx=c["mx"])

    def block_set(arms, strategy):
        return {(a, strategy, h): blocks[a][h - 1] for a in arms for h in (1, 2, 3, 4)}

    with trees(MAX_TREES_C):
        for w in WINDOWS_C:
            win = c[w]
            m = tune_fit(r, f"B1_{w}", win["win"], y, f"S0 {w}")
            score(r, f"S0 {w}", m, block_set(["none", "cov"], f"S0@{w}"))
            for d in ("up", "down"):
                m = tune_fit(r, f"B1x_{d}_{w}", win["b1x"][d], y, f"S0 x{d} {w}")
                score(r, f"S0 x{d} {w}", m, block_set([f"x{d}"], f"S0@{w}"))
            m = tune_fit(r, f"U_{w}", win["u"], y, f"U {w}")
            score(r, f"U {w}", m, {**block_set(["xup"], f"U@{w}"), **block_set(["xdown"], f"U@{w}")})


def save(r: Run, study: str, pilot: bool) -> Path:
    path = (PILOT if pilot else RESULTS) / study / r.dataset / f"{r.model}_seed{r.seed}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(model=r.model, dataset=r.dataset, seed=r.seed, study=study, versions=v1.VERSIONS,
                     sdb_commit=sdb.PINNED, params=r.params, preds=r.preds, rows=r.rows, labels=r.labels,
                     times=r.times, extra=r.extra), path)
    return path


def population(study: str) -> pd.DataFrame:
    name = "population_b2_final.csv" if study == "b2" else "population_c_final.csv"
    return pd.read_csv(ROOT / "results" / "v2" / name)


def load(study: str, dataset: str):
    p = population(study).set_index("name").loc[dataset]
    if p.source == "sdb":
        X, y, cats, _ = load_sdb(dataset)
    elif study == "b2":
        X, y, cats, _ = load_openml(int(p.did))
    else:
        X, y, cats, _ = load_openml(int(p.did), cap=C_ROWS, stratified=True)
    return X, minority_is_one(y), cats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=["b2", "c"])
    ap.add_argument("model", choices=MODELS)
    ap.add_argument("dataset")
    ap.add_argument("seed", type=int)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    assert a.dataset in set(population(a.study).name), f"{a.dataset} is not in the {a.study} population"
    check_versions()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    start = time.perf_counter()
    X, y, cats = load(a.study, a.dataset)
    r = Run(a.model, a.dataset, a.seed, X, cats)
    r.extra = {}
    (run_b2 if a.study == "b2" else run_c)(r, X, y)
    path = save(r, a.study, a.pilot)
    t = pd.DataFrame(r.times)
    kind = t.stage.str.split().str[0] + " " + t.stage.str.split().str[1]
    print(t.groupby(kind).seconds.agg(["count", "sum"]).round(1).to_string())
    print(f"{a.study} {a.model} {a.dataset} seed {a.seed}: {time.perf_counter() - start:.0f} s -> {path}")


if __name__ == "__main__":
    main()
