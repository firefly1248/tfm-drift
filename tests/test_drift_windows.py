"""Tests for the windows of both studies in scripts/drift_windows.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.drift_windows import (
    BLOCK,
    COV_SHIFT,
    FRACS,
    SCENARIOS,
    folds_a,
    geometry_a,
    inclusion,
    oracle_window,
    pc_score,
    pools,
    scenario,
    t0,
    tilt,
    top_feature,
    verdicts,
    windows,
)

rng = np.random.default_rng(0)
n = 3000
X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                  "tied": np.where(rng.random(n) < 0.2, 0.0, rng.normal(size=n)),  # one value, 20 %
                  "c": rng.choice(["u", "v"], n)})
y = (X.a + rng.normal(size=n) > 1).astype(int).to_numpy()
P = pools(y, 0)
z = pc_score(X, ["c"])
x = X.a.to_numpy()


def blocks(sc, seed=1):
    return scenario(sc, y, P, z, x, x[P[0]], np.random.default_rng(seed))


def test_pools_partition_and_stratify():
    assert np.array_equal(np.sort(np.concatenate(P)), np.arange(n))
    sizes = [len(p) for p in P]
    assert max(sizes) - min(sizes) <= 1
    assert np.ptp([y[p].sum() for p in P]) <= 1


def test_z_is_normal_scores():
    assert abs(z.mean()) < 1e-9 and abs(z.std() - 1) < 0.02


@pytest.mark.parametrize("sc", SCENARIOS)
def test_blocks_disjoint_sized_and_never_from_b1(sc):
    out = blocks(sc)
    for h, (idx, lab) in enumerate(out, start=1):
        assert len(idx) == len(lab) == len(P[h]) // 2
        assert set(idx) <= set(P[h])
        assert len(set(idx)) == len(idx)
    assert not set(np.concatenate([i for i, _ in out])) & set(P[0])


def test_inclusion_sums_to_m_and_tilt_hits_target():
    w = np.exp(rng.normal(size=500))
    assert abs(inclusion(w, 100).sum() - 100) < 1e-6
    v, m = z[P[4]] ** 2, len(P[4]) // 2
    d = tilt(v, m, 1 + COV_SHIFT)
    assert abs(inclusion(np.exp(d * (v - v.max())), m) @ v / m - 1 - COV_SHIFT) < 1e-6


def test_covariate_draw_spreads_outward_without_a_direction():
    t4 = [z[blocks("cov", seed)[3][0]] for seed in range(40)]
    assert abs(np.mean([(t ** 2).mean() for t in t4]) - 1 - COV_SHIFT) < 0.03
    assert abs(np.mean([t.mean() for t in t4])) < 0.1  # one pool's own asymmetry, against 0.5 for a mean shift


def test_covariate_and_none_keep_labels():
    for sc in ("none", "cov"):
        assert all(np.array_equal(lab, y[idx]) for idx, lab in blocks(sc))


@pytest.mark.parametrize("r", ["0.5", "2"])
def test_prior_follows_schedule(r):
    for (idx, lab), frac, pool in zip(blocks("prior" + r), FRACS, P[1:]):
        odds = y[pool].sum() / (len(pool) - y[pool].sum()) * float(r) ** frac
        assert abs(lab.sum() - len(idx) * odds / (1 + odds)) <= 0.5
        assert np.array_equal(lab, y[idx])


def test_concept_relabel_counts():
    med = np.median(x[P[0]])
    for (idx, lab), frac in zip(blocks("noise"), FRACS):
        R = x[idx] > med
        assert (lab != y[idx]).sum() == round(0.5 * frac * R.sum())
        assert np.array_equal(lab[~R], y[idx][~R])
    for (idx, lab), frac in zip(blocks("swap"), FRACS):
        R = x[idx] > med
        pos, neg = y[idx][R].sum(), R.sum() - y[idx][R].sum()
        assert (lab != y[idx]).sum() == 2 * round(frac * pos * neg / R.sum())
        assert lab.sum() == y[idx].sum() and np.array_equal(lab[~R], y[idx][~R])
    for (idx, lab), frac in zip(blocks("region"), FRACS):
        R = x[idx] > np.quantile(x[P[0]], 1 - 0.5 * frac)
        assert np.array_equal(lab[R], 1 - y[idx][R]) and np.array_equal(lab[~R], y[idx][~R])


def test_top_feature_skips_tied_and_categorical_columns():
    assert top_feature(X, y, ["c"])[0] == "a"
    assert top_feature(X[["tied", "c"]], y, ["c"])[0] is None


@pytest.mark.parametrize("sc", ["noise", "swap", "region"])
def test_oracle_window_is_unused_rows_at_full_dose(sc):
    r = np.random.default_rng(1)
    out = scenario(sc, y, P, z, x, x[P[0]], r)
    idx, lab = oracle_window(sc, y, P, out, x, x[P[0]], r)
    assert abs(len(idx) - len(P[0])) <= 1 and set(idx) <= set(P[3]) | set(P[4])
    assert not set(idx) & set(np.concatenate([i for i, _ in out]))
    R = x[idx] > np.median(x[P[0]])
    assert np.array_equal(lab[~R], y[idx][~R])
    if sc == "region":
        assert np.array_equal(lab[R], 1 - y[idx][R])
    elif sc == "noise":
        assert (lab != y[idx]).sum() == round(0.5 * R.sum())
    else:
        pos = y[idx][R].sum()
        assert lab.sum() == y[idx].sum()
        assert (lab != y[idx]).sum() == 2 * round(pos * (R.sum() - pos) / R.sum())


@pytest.mark.parametrize("sc", SCENARIOS)
def test_fitting_windows_never_hold_a_row_they_are_scored_on(sc):
    w = windows(sc, 1, y, P, z, x)
    out, b1 = w["blocks"], P[0]
    control = windows("none", 1, y, P, z, x)
    half = t0(P, 1)
    assert set(half) <= set(b1) and len(half) == len(b1) // 2
    assert all(not set(out[a][0]) & set(out[b][0]) for a in range(4) for b in range(a))
    for h in (2, 3, 4):
        test = set(out[h - 1][0])
        (i1, l1), (i2, l2) = w["s1"][h], w["s2"][h]
        assert len(i1) == len(l1) and abs(len(i1) - len(b1)) <= 2
        assert len(i2) == len(l2) == len(b1) + sum(len(b[0]) for b in out[:h - 1])
        assert not test & set(i1) and not test & set(i2)
        assert not test & set(control["s2"][h][0])  # S2 is tuned on the control's windows
        assert np.array_equal(l2[:len(b1)], y[b1])  # B1 is never relabeled
        if sc in ("noise", "swap", "region"):
            assert not set(w["oracle"][0]) & (set(i1) | set(i2) | test)
    assert np.array_equal(w["s1"][2][1], np.concatenate([out[0][1], y[half]]))
    assert np.array_equal(w["s1"][3][1], np.concatenate([out[1][1], out[0][1]]))
    assert np.array_equal(w["s1"][4][1], np.concatenate([out[2][1], out[1][1]]))


def test_gate_names_each_failed_condition():
    row = dict(dataset="d", seed=0, block=4, m=100, b1_minority=40, mean_z2=1.5, positives=30,
               min_class=30, relabeled=0, expected=np.nan, prevalence_change=0)
    g = pd.DataFrame([
        {**row, "scenario": "none"},
        {**row, "scenario": "cov", "mean_z2": 1.2},
        {**row, "scenario": "prior2", "expected": 32.0},
        {**row, "scenario": "noise", "relabeled": 9, "expected": 10},
        {**row, "scenario": "swap", "relabeled": 4, "expected": 4, "prevalence_change": 1},
        {**row, "scenario": "region", "relabeled": 5, "expected": 5, "min_class": 14},
    ])
    v = verdicts(g).set_index("arm").verdict
    assert v["none"] == "pass"
    assert "mean z^2" in v["cov"] and "off schedule" in v["prior"] and "relabel count" in v["noise"]
    assert "moved prevalence" in v["swap"] and "min class 14" in v["region"]


def test_study_a_fitting_and_tuning_rows_precede_their_test_rows():
    t = np.sort(np.random.default_rng(3).integers(0, 1500, 10_000))  # many ties
    g = geometry_a(t)
    cuts = g["cuts"]
    assert cuts[0] == 0 and cuts[-1] == len(t) and all(np.diff(cuts) > 0)
    assert all(t[c] != t[c - 1] for c in cuts[1:-1])  # no timestamp is split by a cut
    assert all(abs(c - BLOCK * k) < 50 for k, c in enumerate(cuts[1:-1], start=1))
    for h, (a, e) in g["blocks"].items():
        fits = [g["b1"]] if h == 1 else [g["s1"][h], g["s2"][h]]
        for fa, fe in fits:
            assert fe == a and fa >= 0 and t[fe - 1] < t[a]
        if h > 1:
            fa, fe = g["s1"][h]
            assert fe - fa >= g["b1"][1] and (fa == 0 or t[fa] != t[fa - 1])
    for a, e in [g["b1"], *g["s2"].values()]:
        for fit, val in folds_a(t, a, e):
            assert fit[0] == a and fit[-1] + 1 == val[0] and val[-1] < e
            assert t[fit[-1]] < t[val[0]]
