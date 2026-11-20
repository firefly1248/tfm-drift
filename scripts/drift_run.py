"""drift_run.py: one model on one dataset and seed of plan.md; per-block probabilities of label 1 go to
results/runs/<dataset>/<model>_seed<seed>.joblib. Every window comes from scripts/drift_windows.py.

Study B: boosters and logistic regression are tuned on B1 and, per window size, on the control's S2
windows; S0 is one fit on B1 scored on every block in one pass; S1, S2 and the oracle are refits per
scenario. Study A: tuned on B1 and on each S2 window with forward folds. TFMs are never tuned.

The run prints timings only, never a score. --pilot also predicts S0 block by block and reports the
largest difference from the one-pass prediction.

Usage: uv run python -m scripts.drift_run <model> <dataset> <seed> [--pilot]
"""
from __future__ import annotations

import argparse
import importlib.metadata
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

import sdb
from scripts.drift_windows import (
    CONCEPT,
    SCENARIOS,
    STUDY_A,
    STUDY_B,
    folds_a,
    geometry_a,
    load_study_a,
    minority_is_one,
    pc_score,
    pools,
    top_feature,
    windows,
)

TFMS = ("tabfm", "tabpfn35", "tabicl")
BOOSTERS = ("catboost", "lgbm", "xgboost")
MODELS = (*TFMS, *BOOSTERS, "logreg")
VERSIONS = {"tabpfn": "9.0.0", "tabicl": "2.1.1", "tabfm": "1.0.1", "catboost": "1.2.10",
            "lightgbm": "4.6.0", "xgboost": "3.2.0", "optuna": "4.7.0", "scikit-learn": "1.7.2",
            "numpy": "2.4.6", "scipy": "1.16.3", "torch": "2.10.0"}
N_TRIALS = 50
FOLD_JOBS, THREADS = 4, 3  # tuning folds run in parallel, each fit on THREADS threads
TABFM_ROWS = 5000
RESULTS = Path(__file__).resolve().parent.parent / "results" / "runs"


def space(model: str, trial, narrow: bool) -> dict:
    """SDB's search spaces; on fewer than three columns column sampling is fixed at 1."""
    f, i = trial.suggest_float, trial.suggest_int
    common = {"learning_rate": f("learning_rate", 1e-3, 0.3, log=True), "n_estimators": i("n_estimators", 50, 500),
              "subsample": f("subsample", 0.4, 1.0)}
    if model == "lgbm":
        return {**common, "num_leaves": i("num_leaves", 8, 256), "min_child_samples": i("min_child_samples", 5, 100),
                "colsample_bytree": 1.0 if narrow else f("colsample_bytree", 0.4, 1.0),
                "reg_alpha": f("reg_alpha", 1e-8, 10.0, log=True), "reg_lambda": f("reg_lambda", 1e-8, 10.0, log=True)}
    if model == "xgboost":
        return {**common, "max_depth": i("max_depth", 2, 10),
                "colsample_bytree": 1.0 if narrow else f("colsample_bytree", 0.4, 1.0),
                "gamma": f("gamma", 1e-8, 5.0, log=True), "reg_alpha": f("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": f("reg_lambda", 1e-8, 10.0, log=True), "min_child_weight": i("min_child_weight", 1, 10)}
    return {**common, "depth": i("depth", 2, 10), "l2_leaf_reg": f("l2_leaf_reg", 1e-3, 10.0, log=True),
            "colsample_bylevel": 1.0 if narrow else f("colsample_bylevel", 0.4, 1.0),
            "min_data_in_leaf": i("min_data_in_leaf", 1, 20), "random_strength": f("random_strength", 0.0, 10.0)}


def build(model: str, params: dict, cat_cols: list[str], seed: int):
    """An unfitted model. Imports stay local: LightGBM and torch cannot share a process on macOS."""
    if model == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params, objective="binary", subsample_freq=1, class_weight=None,
                              random_state=seed, n_jobs=THREADS, verbose=-1)
    if model == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**params, objective="binary:logistic", enable_categorical=True, tree_method="hist",
                             random_state=seed, n_jobs=THREADS, verbosity=0)
    if model == "logreg":
        from benchmark.models.grid_search import build_grid_search
        return build_grid_search("logreg", None, cat_cols).estimator.set_params(
            lr__random_state=seed, lr__class_weight=None, **params)
    from benchmark.models import wrappers as w
    if model == "catboost":
        return w.CatBoostNativeWrapper(cat_cols=cat_cols, loss_function="Logloss", random_state=seed, verbose=0,
                                       thread_count=THREADS, bootstrap_type="Bernoulli", **params)
    if model == "tabfm":
        return w.TabFMNativeWrapper(cat_cols=cat_cols, random_state=seed)
    if model == "tabpfn35":
        return w.TabPFNNativeWrapper(cat_cols=cat_cols, model_version="v3.5", random_state=seed)
    return w.TabICLNativeWrapper(cat_cols=cat_cols, random_state=seed)


def frame(model: str, X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """LightGBM and XGBoost get pandas categories with the level set of the whole file."""
    if model not in ("lgbm", "xgboost"):
        return X
    return X.assign(**{c: pd.Categorical(X[c], categories=sorted(X[c].unique())) for c in cat_cols})


def tune(model: str, X: pd.DataFrame, y: np.ndarray, folds: list, cat_cols: list[str], seed: int) -> dict:
    """Hyperparameters chosen by mean log loss over the folds; TFMs have none."""
    from benchmark.metrics import LOG_LOSS_SCORER
    if model in TFMS:
        return {}
    if model == "logreg":
        from benchmark.models.grid_search import build_grid_search
        grid = {k: v for k, v in build_grid_search("logreg", None, cat_cols).param_grid.items()
                if k != "lr__class_weight"}
        gs = GridSearchCV(build(model, {}, cat_cols, seed), grid, cv=folds, scoring=LOG_LOSS_SCORER,
                          n_jobs=FOLD_JOBS, refit=False).fit(X, y)
        return gs.best_params_
    narrow = X.shape[1] < 3

    def objective(trial):
        m = build(model, space(model, trial, narrow), cat_cols, seed)
        return cross_val_score(m, X, y, cv=folds, scoring=LOG_LOSS_SCORER, n_jobs=FOLD_JOBS).mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=N_TRIALS)
    return space(model, optuna.trial.FixedTrial(study.best_params), narrow)


class Run:
    """One model, dataset and seed: fits, one-pass predictions, timings, the checkpoint."""

    def __init__(self, model: str, dataset: str, seed: int, X: pd.DataFrame, cat_cols: list[str]):
        self.model, self.dataset, self.seed, self.cat_cols = model, dataset, seed, cat_cols
        self.X = frame(model, X, cat_cols)
        self.params, self.preds, self.rows, self.labels, self.times = {}, {}, {}, {}, []

    def timed(self, stage: str, fn, rows: int):
        start = time.perf_counter()
        out = fn()
        self.times.append(dict(stage=stage, rows=rows, seconds=time.perf_counter() - start))
        return out

    def tune(self, key: str, idx: np.ndarray, lab: np.ndarray, folds: list) -> None:
        self.params[key] = self.timed(f"tune {key}", lambda: tune(self.model, self.X.iloc[idx], lab, folds,
                                                                     self.cat_cols, self.seed), len(idx))

    def fit(self, stage: str, params: dict, idx: np.ndarray, lab: np.ndarray):
        def go():
            m = build(self.model, params, self.cat_cols, self.seed).fit(self.X.iloc[idx], lab)
            assert list(m.classes_) == [0, 1]  # check 4: column 1 is the minority class
            return m
        return self.timed(f"fit {stage}", go, len(idx))

    def proba(self, stage: str, m, rows: np.ndarray) -> np.ndarray:
        return self.timed(f"predict {stage}", lambda: m.predict_proba(self.X.iloc[rows])[:, 1].astype(np.float32),
                          len(rows))

    def store(self, key: tuple, rows: np.ndarray, p: np.ndarray, lab: np.ndarray) -> None:
        self.preds[key], self.rows[key], self.labels[key] = p, rows, lab

    def save(self, study: str) -> Path:
        path = RESULTS / self.dataset / f"{self.model}_seed{self.seed}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(model=self.model, dataset=self.dataset, seed=self.seed, study=study,
                         versions=VERSIONS, sdb_commit=sdb.PINNED, params=self.params, preds=self.preds,
                         rows=self.rows, labels=self.labels, times=self.times), path)
        return path


def run_b(r: Run, y: np.ndarray, z: np.ndarray, x: np.ndarray | None, pilot: bool) -> None:
    scen = [sc for sc in (SCENARIOS if r.dataset in STUDY_B else ["region"]) if sc not in CONCEPT or x is not None]
    P = pools(y, r.seed)
    W = {sc: windows(sc, r.seed, y, P, z, x) for sc in scen}

    def strat(lab):
        return list(StratifiedKFold(4, shuffle=True, random_state=r.seed).split(np.zeros(len(lab)), lab))

    r.tune("B1", P[0], y[P[0]], strat(y[P[0]]))
    with_s2 = "none" in scen
    if with_s2:
        for h in (2, 3, 4):
            idx, lab = W["none"]["s2"][h]
            r.tune(f"S2_{h}", idx, lab, strat(lab))
    s0 = r.fit("S0", r.params["B1"], P[0], y[P[0]])
    outside = np.setdiff1d(np.arange(len(y)), P[0])
    p_out = r.proba("S0 one pass", s0, outside)
    pos = np.empty(len(y), dtype=int)
    pos[outside] = np.arange(len(outside))
    if pilot and r.model in TFMS:
        diff = max(np.abs(r.proba("S0 per block", s0, idx) - p_out[pos[idx]]).max() for idx, _ in W[scen[0]]["blocks"])
        print(f"one-pass check, {scen[0]}: largest difference {diff:.2e}")
    for sc in scen:
        w = W[sc]
        for h, (idx, lab) in enumerate(w["blocks"], start=1):
            r.store((sc, "S0", h), idx, p_out[pos[idx]], lab)
        for h in (2, 3, 4):
            idx, lab = w["blocks"][h - 1]
            m = r.fit(f"S1 {sc} {h}", r.params["B1"], *w["s1"][h])
            r.store((sc, "S1", h), idx, r.proba(f"S1 {sc} {h}", m, idx), lab)
            if with_s2:
                m = r.fit(f"S2 {sc} {h}", r.params[f"S2_{h}"], *w["s2"][h])
                r.store((sc, "S2", h), idx, r.proba(f"S2 {sc} {h}", m, idx), lab)
        if sc in CONCEPT:
            idx, lab = w["blocks"][3]
            m = r.fit(f"oracle {sc}", r.params["B1"], *w["oracle"])
            r.store((sc, "oracle", 4), idx, r.proba(f"oracle {sc}", m, idx), lab)


def run_a(r: Run, y: np.ndarray, t: np.ndarray) -> None:
    g = geometry_a(t)

    def rng(a, e):
        return np.arange(a, e)

    def folds(a, e):
        return [(fit - a, val - a) for fit, val in folds_a(t, a, e)]

    r.tune("B1", rng(*g["b1"]), y[slice(*g["b1"])], folds(*g["b1"]))
    for h in (2, 3, 4):
        r.tune(f"S2_{h}", rng(*g["s2"][h]), y[slice(*g["s2"][h])], folds(*g["s2"][h]))
    s0 = r.fit("S0", r.params["B1"], rng(*g["b1"]), y[slice(*g["b1"])])
    after = rng(g["cuts"][1], len(t))
    p_after = r.proba("S0 one pass", s0, after)
    for h, (a, e) in g["blocks"].items():
        r.store(("none", "S0", h), rng(a, e), p_after[a - after[0]:e - after[0]], y[a:e])
    for h in (2, 3, 4):
        block = rng(*g["blocks"][h])
        for s in ("S1", "S2"):
            a, e = g[s.lower()][h]
            if r.model == "tabfm":
                a = max(a, e - TABFM_ROWS)  # the most recent rows within TabFM's context cap
            params = r.params["B1"] if s == "S1" else r.params[f"S2_{h}"]
            m = r.fit(f"{s} {h}", params, rng(a, e), y[a:e])
            r.store(("none", s, h), block, r.proba(f"{s} {h}", m, block), y[block])


def check_versions() -> None:
    found = {p: importlib.metadata.version(p) for p in VERSIONS}
    assert found == VERSIONS, {p: v for p, v in found.items() if v != VERSIONS[p]}
    sdb.assert_pinned()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=MODELS)
    ap.add_argument("dataset", choices=STUDY_A + STUDY_B + ["ringnorm", "skin-segmentation", "twonorm"])
    ap.add_argument("seed", type=int)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    check_versions()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    start = time.perf_counter()
    if a.dataset in STUDY_A:
        X, y, t, _, cat_cols = load_study_a(a.dataset)
        r = Run(a.model, a.dataset, a.seed, X, cat_cols)
        run_a(r, y, t)
        study = "A"
    else:
        X, y, cat_cols = sdb.load_data_df(a.dataset)
        y = minority_is_one(y)
        top, _ = top_feature(X, y, cat_cols)
        r = Run(a.model, a.dataset, a.seed, X, cat_cols)
        run_b(r, y, pc_score(X, cat_cols), X[top].astype(float).to_numpy() if top else None, a.pilot)
        study = "B"
    path = r.save(study)
    t = pd.DataFrame(r.times)
    kind = t.stage.str.split().str[0] + " " + t.stage.str.split().str[1]
    print(t.groupby(kind).seconds.agg(["count", "sum"]).round(1).to_string())
    print(f"{a.model} {a.dataset} seed {a.seed}: {time.perf_counter() - start:.0f} s -> {path}")


if __name__ == "__main__":
    main()
