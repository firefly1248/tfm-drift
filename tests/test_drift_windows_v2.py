import hashlib

import numpy as np
import pandas as pd
import pytest

from scripts.drift_data_v2 import clean, prepare
from scripts.drift_windows import SCENARIOS, pools, windows
from scripts.drift_windows_v2 import (
    WINDOWS_C,
    Q,
    beyond,
    block_size,
    extrap_blocks,
    extrap_windows,
    gate_b2,
    nested_windows,
    pools_c,
    small_class,
    study_c,
    verdict_b2,
    z_extrap,
)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 2_000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.integers(0, 5, n).astype(float)})
    X["a"] += X["c"] * 0.5
    y = (rng.random(n) < 1 / (1 + np.exp(-(X["a"] - 0.5)))).astype(int)
    y = 1 - y if y.mean() > 0.5 else y
    return X, y


def test_v1_scenarios_and_windows_are_unchanged():
    assert SCENARIOS == ["none", "cov", "prior0.5", "prior2", "noise", "swap", "region"]
    rng = np.random.default_rng(1)
    y = (rng.random(1_000) < 0.3).astype(int)
    z = rng.normal(size=1_000)
    x = rng.normal(size=1_000)
    P = pools(y, 3)
    h = hashlib.sha256()
    for sc in SCENARIOS:
        if sc == "cov":  # a float root-finding step: counts are tested elsewhere, indices may differ by platform
            continue
        for idx, lab in windows(sc, 3, y, P, z, x)["blocks"]:
            h.update(idx.tobytes() + lab.tobytes())
    assert h.hexdigest() == "704250a05f2ec7ca157386af818e738ad93c6b004c9db44c4ee5f27676ddfd2b"


def test_z_is_oriented_toward_label_one_and_breaks_ties(data):
    X, y = data
    z = z_extrap(X, [], y)
    assert y[z > 0].mean() > y[z < 0].mean()
    assert len(np.unique(z)) == len(z)
    Xt = X.assign(a=0.0, b=0.0)  # constant numeric columns: PC1 of c alone has only five values
    assert len(np.unique(z_extrap(Xt, [], y))) == len(y)


def test_extrapolation_blocks_hold_the_stated_share_beyond_the_quantile(data):
    X, y = data
    z = z_extrap(X, [], y)
    P = pools(y, 0)
    mx = block_size(z, P)
    for d in Q:
        for h, (idx, lab) in enumerate(extrap_blocks(y, P, z, 0, d, mx), start=1):
            assert len(idx) == mx and len(set(idx)) == mx
            assert beyond(z, d)[idx].sum() == round(mx * h / 4)
            assert set(idx) <= set(P[h])
            assert np.array_equal(lab, y[idx])


def test_windows_never_hold_the_rows_of_the_block_they_feed(data):
    X, y = data
    z = z_extrap(X, [], y)
    P = pools(y, 1)
    mx = block_size(z, P)
    w = extrap_windows(y, P, z, 1, mx)
    for d in Q:
        r = w[d]
        assert not beyond(z, d)[r["b1x"]].any()
        tested = set(np.concatenate([b[0] for b in r["blocks"]]))
        assert not tested & set(r["b1x"])
        for h in (2, 3, 4):
            for k in ("s1", "s2"):
                assert not set(r[k][h][0]) & set(r["blocks"][h - 1][0])
            assert set(r["s2"][h][0]) >= set(r["b1x"])
    assert set(w["u"]) <= set(P[0]) and len(w["u"]) == min(len(w[d]["b1x"]) for d in Q)


def test_gate_b2_passes_on_balanced_data_and_names_a_failure(data):
    X, y = data
    g = gate_b2(X, y, [], range(2))
    assert verdict_b2(g) == "pass" and (g[g.block == 4].share_z == 1).all()
    assert verdict_b2(g.assign(min_class=3)).startswith("block min class 3")


def test_study_c_pools_and_nested_windows():
    rng = np.random.default_rng(2)
    n = 36_000
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = (rng.random(n) < 0.2).astype(int)
    P = pools_c(y, 0)
    assert [len(p) for p in P] == [20_000, 4_000, 4_000, 4_000, 4_000]
    assert len(np.unique(np.concatenate(P))) == n
    nest = nested_windows(P[0], y, 0)
    assert all(set(nest[a]) <= set(nest[b]) for a, b in zip(WINDOWS_C, WINDOWS_C[1:]))
    assert [len(nest[w]) for w in WINDOWS_C] == list(WINDOWS_C)
    assert abs(y[nest[1_000]].mean() - y[P[0]].mean()) < 0.002
    z = z_extrap(X, [], y)
    c = study_c(y, z, 0)
    assert all(len(c["blocks"][d][3][0]) == c["mx"] for d in Q)
    assert all(not beyond(z, d)[c[w]["b1x"][d]].any() for w in WINDOWS_C for d in Q)


def test_prepare_follows_sdb_conventions_and_clean_deduplicates():
    df = pd.DataFrame({"n": [str(i) for i in range(19)] * 6 + ["x"] * 6, "c": list("abcabcabca") * 12,
                       "empty": [None] * 120})
    df["t"] = ["u"] * 90 + ["v"] * 30
    X, y, cats, counts = prepare(df, "t")
    assert "empty" not in X and cats == ["c"] and X["n"].dtype == float  # more than 90 % parse: "x" becomes missing
    assert counts["rows_dedup"] < counts["rows_raw"] and y.mean() < 0.5
    X2, y2, _, _ = clean(X, y, cats, cap=10, stratified=True)
    assert len(X2) == 10 and abs(y2.mean() - y.mean()) < 0.2
    assert small_class(y2) >= 1


def test_v1_gate_rows_match_the_committed_gate():
    from pathlib import Path

    from scripts.drift_windows import gate_rows
    new = pd.DataFrame(gate_rows("banana")).drop(columns=["top_feature", "top_mi"])
    old = pd.read_csv(Path(__file__).resolve().parent.parent / "results" / "drift_gate.csv")
    old = old[old.dataset == "banana"].drop(columns=["top_feature", "top_mi"], errors="ignore")
    key = ["seed", "scenario", "block"]
    m = new.merge(old, on=key, suffixes=("_new", "_old"))
    assert len(m) == len(new) == len(old)
    for c in ("m", "positives", "min_class", "relabeled", "b1_minority"):
        assert (m[f"{c}_new"] == m[f"{c}_old"]).all()
    assert np.allclose(m.mean_z2_new, m.mean_z2_old, atol=1e-4)
