"""drift_windows_v2.py: the geometry of plan_v2.md, as pure functions, and its gate. Nothing here fits a model.

Study B2 reuses v1's pools, control and tilt (drift_windows.py, unchanged). The extrapolation arms xup and xdown
train on B1x, the rows of P1 on one side of a quantile of z, and score blocks whose share h/4 lies beyond it. Their
reference is a full-range fit U of the same size on the same blocks. Study C uses larger pools and nested windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scripts.drift_windows import (
    FRACS,
    MIN_MINORITY,
    minority_is_one,
    pc_score,
    pools,
    verdicts,
    windows,
)

Q = {"up": 0.6, "down": 0.4}   # B1x holds z at or below the 0.6 quantile (up) or at or above the 0.4 quantile (down)
MX_SHARE = 0.8                 # the extrapolation block holds this share of the scarcest pool region
WINDOWS_C = (1_000, 5_000, 20_000)
LEVELS_C = 20                  # Study C: categorical levels beyond the 20 most frequent share one column


def seeds_v2(n_rows: int) -> range:
    return range(10 if n_rows < 2_000 else 5)


def z_extrap(X: pd.DataFrame, cat_cols: list[str], y: np.ndarray, max_levels: int | None = None) -> np.ndarray:
    """v1's pc_score with ties in PC1 broken by a fixed jitter, and the sign set so that z is higher where
    label 1 is more frequent; max_levels pools rare levels of a categorical column (Study C)."""
    X = X.copy()
    if max_levels:
        for c in cat_cols:
            top = X[c].value_counts().index[:max_levels]
            X[c] = X[c].where(X[c].isin(top), "__other")
    M = pd.get_dummies(X, columns=cat_cols).astype(float)
    M = StandardScaler().fit_transform(M.fillna(M.median()))
    pc = PCA(1, svd_solver="full").fit_transform(M)[:, 0]
    pc = pc + np.random.default_rng(0).normal(0, 1e-9 * pc.std(), len(pc))
    z = norm.ppf((rankdata(pc) - 0.5) / len(pc))
    return -z if y[z > 0].mean() < y[z < 0].mean() else z


def beyond(z: np.ndarray, direction: str) -> np.ndarray:
    q = np.quantile(z, Q[direction])
    return z > q if direction == "up" else z < q


def block_size(z: np.ndarray, P: list[np.ndarray]) -> int:
    """m_x: MX_SHARE of the fewest rows beyond a quantile in any test pool, in either direction."""
    return int(MX_SHARE * min(beyond(z, d)[P[h]].sum() for d in Q for h in range(1, 5)))


def extrap_blocks(y: np.ndarray, P: list[np.ndarray], z: np.ndarray, seed: int, direction: str,
                  mx: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """T1..T4 of xup or xdown: a share h/4 of the m_x rows drawn from the pool's rows beyond the quantile,
    the rest from its rows within the training range, all uniform."""
    out = beyond(z, direction)
    rng = np.random.default_rng([seed, 20 + (direction == "down")])
    blocks = []
    for h, frac in enumerate(FRACS, start=1):
        pool, k = P[h], round(mx * frac)
        idx = np.concatenate([rng.choice(pool[out[pool]], k, replace=False),
                              rng.choice(pool[~out[pool]], mx - k, replace=False)])
        blocks.append((idx, y[idx]))
    return blocks


def extrap_windows(y: np.ndarray, P: list[np.ndarray], z: np.ndarray, seed: int, mx: int) -> dict:
    """Study B2: per direction, B1x, the blocks, the S1 and S2 windows before T2..T4; and the full-range
    window U (a uniform part of P1 as large as the smaller B1x), shared by both directions."""
    res = {}
    for d in Q:
        b1x = P[0][~beyond(z, d)[P[0]]]
        blocks = extrap_blocks(y, P, z, seed, d, mx)
        rng = np.random.default_rng([seed, 30 + (d == "down")])
        t0 = rng.choice(b1x, len(b1x) // 2, replace=False)
        s1 = {}
        for h in (2, 3, 4):
            old = t0 if h == 2 else blocks[h - 3][0]
            part = rng.choice(old, min(len(b1x) - mx, len(old)), replace=False)
            idx = np.concatenate([blocks[h - 2][0], part])
            s1[h] = (idx, y[idx])
        s2 = {h: (np.concatenate([b1x, *[b[0] for b in blocks[:h - 1]]]),
                  y[np.concatenate([b1x, *[b[0] for b in blocks[:h - 1]]])]) for h in (2, 3, 4)}
        res[d] = dict(b1x=b1x, blocks=blocks, s1=s1, s2=s2)
    size = min(len(res[d]["b1x"]) for d in Q)
    res["u"] = np.random.default_rng([seed, 32]).choice(P[0], size, replace=False)
    return res


def block_shares(X: pd.DataFrame, z: np.ndarray, b1x: np.ndarray, idx: np.ndarray, direction: str) -> tuple[float, float]:
    """The share of block rows beyond B1x along z, and the share with a numeric feature outside B1x's range."""
    num = X.select_dtypes(include="number")
    lo, hi = num.iloc[b1x].min(), num.iloc[b1x].max()
    outside_feature = ((num.iloc[idx] < lo) | (num.iloc[idx] > hi)).any(axis=1).mean() if num.shape[1] else np.nan
    return float(beyond(z, direction)[idx].mean()), float(outside_feature)


def gate_e(name: str, X: pd.DataFrame, y: np.ndarray, cat_cols: list[str], seed_range: range,
           pool_fn=pools, z: np.ndarray | None = None) -> str:
    """v1's gate for the control and the tilt on any data: 'pass' or the failed conditions. pool_fn gives the
    pools (Study C passes pools_c)."""
    y = minority_is_one(y)
    z = pc_score(X, cat_cols) if z is None else z
    rows = []
    for seed in seed_range:
        P = pool_fn(y, seed)
        for sc in ("none", "cov"):
            for h, (idx, lab) in enumerate(windows(sc, seed, y, P, z, None)["blocks"], start=1):
                rows.append(dict(dataset=name, seed=seed, scenario=sc, block=h, b1_minority=int(y[P[0]].sum()),
                                 mean_z2=(z[idx] ** 2).mean(), positives=int(lab.sum()), expected=np.nan,
                                 min_class=small_class(lab), relabeled=0, prevalence_change=0))
    v = verdicts(pd.DataFrame(rows))
    bad = v[v.verdict != "pass"]
    return "; ".join(f"{r.arm}: {r.verdict}" for r in bad.itertuples()) or "pass"


def small_class(lab: np.ndarray) -> int:
    return int(min(lab.sum(), len(lab) - lab.sum()))


def gate_b2(X: pd.DataFrame, y: np.ndarray, cat_cols: list[str], seed_range: range) -> pd.DataFrame:
    """One row per seed, direction and block of the extrapolation arms, with what the X-family gate reads."""
    y = minority_is_one(y)
    z = z_extrap(X, cat_cols, y)
    rows = []
    for seed in seed_range:
        P = pools(y, seed)
        mx = block_size(z, P)
        w = extrap_windows(y, P, z, seed, mx)
        u = w["u"]
        for d in Q:
            r = w[d]
            for h in (1, 2, 3, 4):
                idx, lab = r["blocks"][h - 1]
                share_z, share_feat = block_shares(X, z, r["b1x"], idx, d)
                rows.append(dict(seed=seed, direction=d, block=h, mx=mx, m=len(idx), min_class=small_class(lab),
                                 share_z=share_z, share_feature=share_feat,
                                 b1x_rows=len(r["b1x"]), b1x_min_class=small_class(y[r["b1x"]]),
                                 u_min_class=small_class(y[u]),
                                 window_min_class=min([small_class(y[r[k][g][0]]) for k in ("s1", "s2") for g in (2, 3, 4)])))
    return pd.DataFrame(rows)


def verdict_b2(g: pd.DataFrame) -> str:
    """'pass' or the failed conditions: m_x at least 1 and the blocks and windows hold enough of the smaller class."""
    fails = []
    if (g.mx < 1).any():
        fails.append("no rows beyond the quantile")
    if (g.min_class < MIN_MINORITY).any():
        fails.append(f"block min class {g.min_class.min()}")
    if (g.window_min_class < MIN_MINORITY).any():
        fails.append(f"window min class {g.window_min_class.min()}")
    if (g.b1x_min_class < 2 * MIN_MINORITY).any() or (g.u_min_class < 2 * MIN_MINORITY).any():
        fails.append(f"B1x or U min class {min(g.b1x_min_class.min(), g.u_min_class.min())}")
    return "; ".join(fails) or "pass"


# ---------------------------------------------------------------- Study C

def pools_c(y: np.ndarray, seed: int) -> list[np.ndarray]:
    """P1 of 20,000 rows and four disjoint pools of 4,000, each class-stratified, from 36,000 rows."""
    assert len(y) == 36_000
    rng = np.random.default_rng([seed, 40])
    order = {k: rng.permutation(np.flatnonzero(y == k)) for k in (0, 1)}
    take, sizes, P = {0: 0, 1: 0}, [20_000, 4_000, 4_000, 4_000, 4_000], []
    for s in sizes:
        k1 = round(s * y.mean())
        P.append(np.sort(np.concatenate([order[1][take[1]:take[1] + k1], order[0][take[0]:take[0] + s - k1]])))
        take[1] += k1
        take[0] += s - k1
    return P


def nested_windows(P1: np.ndarray, y: np.ndarray, seed: int) -> dict[int, np.ndarray]:
    """Class-stratified windows of WINDOWS_C rows of P1, each inside the next."""
    rng = np.random.default_rng([seed, 41])
    order = {k: rng.permutation(P1[y[P1] == k]) for k in (0, 1)}
    out = {}
    for w in WINDOWS_C:
        k1 = round(w * y[P1].mean())
        out[w] = np.concatenate([order[1][:k1], order[0][:w - k1]])
    return out


def study_c(y: np.ndarray, z: np.ndarray, seed: int) -> dict:
    """Per window size: the control and tilt window, B1x (up and down) and U of the same size; and the blocks
    of every arm (control and tilt through v1's windows, xup and xdown through extrap_blocks)."""
    P = pools_c(y, seed)
    mx = block_size(z, P)
    nest = nested_windows(P[0], y, seed)
    res = dict(P=P, mx=mx, blocks={d: extrap_blocks(y, P, z, seed, d, mx) for d in Q})
    for w, win in nest.items():
        res[w] = dict(win=win, b1x={d: win[~beyond(z, d)[win]] for d in Q})
        size = min(len(v) for v in res[w]["b1x"].values())
        res[w]["u"] = np.random.default_rng([seed, 42, w]).choice(win, size, replace=False)
    return res


def gate_c(X: pd.DataFrame, y: np.ndarray, cat_cols: list[str], seed_range: range) -> pd.DataFrame:
    y = minority_is_one(y)
    z = z_extrap(X, cat_cols, y, max_levels=LEVELS_C)
    rows = []
    for seed in seed_range:
        c = study_c(y, z, seed)
        for w in WINDOWS_C:
            rows.append(dict(seed=seed, window=w, mx=c["mx"], win_min_class=small_class(y[c[w]["win"]]),
                             b1x_min_class=min(small_class(y[v]) for v in c[w]["b1x"].values()),
                             u_min_class=small_class(y[c[w]["u"]]),
                             block_min_class=min(small_class(y[b[0]]) for d in Q for b in c["blocks"][d])))
    return pd.DataFrame(rows)


def verdict_c(g: pd.DataFrame) -> str:
    fails = []
    if (g.mx < 1).any():
        fails.append("no rows beyond the quantile")
    if (g.block_min_class < MIN_MINORITY).any():
        fails.append(f"block min class {g.block_min_class.min()}")
    low = g[["win_min_class", "b1x_min_class", "u_min_class"]].min().min()
    if low < 2 * MIN_MINORITY:
        fails.append(f"window min class {low}")
    return "; ".join(fails) or "pass"

