import numpy as np
import pandas as pd
import pytest

from scripts import drift_run
from scripts import drift_run_v2 as v2
from scripts.drift_run import Run


def synthetic(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    X["e"] = rng.choice(["u", "v", "w"], n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(X["a"] + 0.5 * X["b"] - 1.5)))).astype(int)
    return X, y


@pytest.fixture
def run_logreg():
    X, y = synthetic(2_400)
    r = Run("logreg", "synthetic", 0, X, ["e"])
    r.extra = {}
    return r, X, y


def test_b2_stores_every_arm_and_strategy_with_labels_that_match(run_logreg):
    r, X, y = run_logreg
    v2.run_b2(r, X, y)
    keys = set(r.preds)
    for a in ("none", "cov", "xup", "xdown"):
        assert {(a, "S0", h) for h in (1, 2, 3, 4)} <= keys
        assert {(a, s, h) for s in ("S1", "S2") for h in (2, 3, 4)} <= keys
    assert {(a, "U", h) for a in ("xup", "xdown") for h in (1, 2, 3, 4)} <= keys
    assert not any(k[1] == "S1r" for k in keys)  # logistic regression is not a booster
    for k, p in r.preds.items():
        assert len(p) == len(r.rows[k]) == len(r.labels[k]) and np.array_equal(r.labels[k], y[r.rows[k]])
        assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
    assert len(r.extra["shares"]) == 8 and r.extra["mx"] >= 1
    # the full-range fit and the B1x fit score the very same rows of an extrapolation arm
    assert np.array_equal(r.rows[("xup", "U", 4)], r.rows[("xup", "S0", 4)])


def test_b2_with_a_booster_retunes_every_s1_window(run_logreg, monkeypatch):
    r, X, y = run_logreg
    monkeypatch.setattr(drift_run, "N_TRIALS", 2)
    r = Run("lgbm", "synthetic", 0, X, ["e"])
    r.extra = {}
    v2.run_b2(r, X, y)
    assert {(a, "S1r", h) for a in ("none", "cov", "xup", "xdown") for h in (2, 3, 4)} <= set(r.preds)
    assert sum(k.startswith("S1r_") for k in r.params) == 12


def test_the_wide_space_reaches_above_500_trees_v1_does_not_and_the_space_is_restored():
    import optuna
    original = drift_run.space

    def draws(max_trees):
        out = []

        def objective(trial):
            sp = drift_run.space("lgbm", trial, False)
            out.append(sp["n_estimators"])
            return 0.0

        with v2.trees(max_trees):
            optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0)).optimize(objective, n_trials=200)
        return max(out)

    assert draws(2_000) > 500 and draws(500) <= 500
    assert drift_run.space is original


def test_study_c_stores_the_window_in_the_strategy():
    X, y = synthetic(36_000, 1)
    r = Run("logreg", "synthetic", 0, X, ["e"])
    r.extra = {}
    v2.run_c(r, X, y)
    keys = set(r.preds)
    for w in (1_000, 5_000, 20_000):
        assert {("none", f"S0@{w}", h) for h in (1, 2, 3, 4)} <= keys
        assert {("xup", f"U@{w}", h) for h in (1, 2, 3, 4)} <= keys
    assert not any(k[1] in ("S1", "S2") for k in keys)
    assert all(np.array_equal(r.labels[k], y[r.rows[k]]) for k in keys)


def test_free_empties_the_mps_cache_for_tfms_only(monkeypatch):
    import sys
    import types

    calls = []
    fake = types.SimpleNamespace(mps=types.SimpleNamespace(empty_cache=lambda: calls.append(1)))
    monkeypatch.setitem(sys.modules, "torch", fake)
    v2.free("catboost")
    assert calls == []
    for model in ("tabfm", "tabpfn35", "tabicl"):
        v2.free(model)
    assert len(calls) == 3
