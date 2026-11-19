"""drift_windows.py: Study B of plan.md, as pure functions of (data, seed), and the
injection gate (check 1), which fits no model.

Per dataset and seed the rows split into five class-stratified pools P1..P5. B1 is P1. Test block
T_h, h = 1..4, is m = floor(|P_(h+1)| / 2) rows of P_(h+1) at dose fraction h/4, drawn by one of
seven scenarios, each from its own random stream. Label 1 is the minority class of the whole dataset
(label 1 on a tie). A concept arm also has an oracle window: the rows of P4 and P5 left out of T3
and T4, relabeled at the full dose. Fitting windows: S1 before T_h is T_(h-1) and T_(h-2), with a
seeded half of B1 standing in for T0; S2 is B1 and T1..T_(h-1).

The CONCEPT_ONLY candidates that pass the gate join the concept population of H5 only, with the
region arm. The gate also records how well a step at the B1 median of the top feature alone
scores the T4 labels of the region arm (step_auc).

Usage: uv run python -m scripts.drift_windows   (writes results/drift_gate.csv)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm, rankdata
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sdb import load_data_df

STUDY_B = ["banana", "bank-marketing-full", "eeg-eye-state", "hill-valley-with-noise", "magic",
           "phoneme", "qsar-biodegradation", "spambase", "statlog-german-credit",
           "thyroid-sick-euthyroid"]
CONCEPT_ONLY = ["autoUniv-au1-1000", "banknote-authentication", "mushroom-expanded", "ringnorm",
                "skin-segmentation", "twonorm"]  # concept candidates outside the main ten, region arm only
SMALL = {"qsar-biodegradation", "hill-valley-with-noise", "statlog-german-credit",
         "autoUniv-au1-1000", "banknote-authentication"}  # under 2,000 rows
SCENARIOS = ["none", "cov", "prior0.5", "prior2", "noise", "swap", "region"]
CONCEPT = ("noise", "swap", "region")
FRACS = (0.25, 0.5, 0.75, 1.0)
COV_SHIFT = 0.5   # target rise of the mean of z^2 at T4, from 1 to 1.5
MIN_MINORITY = 15


def seeds(name: str) -> range:
    return range(10 if name in SMALL else 3)


def minority_is_one(y: np.ndarray) -> np.ndarray:
    return 1 - y if y.mean() > 0.5 else y


def pools(y: np.ndarray, seed: int) -> list[np.ndarray]:
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    return [test for _, test in skf.split(np.zeros(len(y)), y)]


def pc_score(X: pd.DataFrame, cat_cols: list[str]) -> np.ndarray:
    """Normal scores of the ranks of PC1 of the standardized numeric and one-hot columns."""
    M = pd.get_dummies(X, columns=cat_cols).astype(float)
    M = StandardScaler().fit_transform(M.fillna(M.median()))
    pc = PCA(1, svd_solver="full").fit_transform(M)[:, 0]
    return norm.ppf((rankdata(pc) - 0.5) / len(pc))


def inclusion(w: np.ndarray, m: int) -> np.ndarray:
    """Inclusion probabilities of an Efraimidis-Spirakis draw of m rows (Rosen's approximation)."""
    hi = 1.0
    while (1 - np.exp(-w * hi)).sum() < m:
        hi *= 2
    t = brentq(lambda t: (1 - np.exp(-w * t)).sum() - m, 0, hi)
    return 1 - np.exp(-w * t)


def tilt(v: np.ndarray, m: int, target: float) -> float:
    """The d at which the expected mean of v over an exp(d * v)-weighted draw of m rows is target.
    d can be negative when a small pool's mean already exceeds the target."""
    return brentq(lambda d: inclusion(weights(v, d), m) @ v / m - target, -10, 10)


def weights(v: np.ndarray, d: float) -> np.ndarray:
    return np.exp(d * v - (d * v).max())


def top_feature(X: pd.DataFrame, y: np.ndarray, cat_cols: list[str]) -> tuple[str | None, float]:
    """The numeric column with >= 20 distinct values, none above 5 % of the rows, and the highest
    mutual information with the label, and that information. Chosen once per dataset: it is part
    of the drift design."""
    num = X.drop(columns=cat_cols).astype(float)
    ok = [c for c in num if num[c].nunique() >= 20 and num[c].value_counts().max() <= 0.05 * len(num)]
    if not ok:
        return None, np.nan
    mi = mutual_info_classif(num[ok].fillna(num[ok].median()), y, random_state=0)
    return ok[int(np.argmax(mi))], float(mi.max())


def relabel(name: str, lab: np.ndarray, xv: np.ndarray, b1_x: np.ndarray, frac: float,
            rng: np.random.Generator) -> np.ndarray:
    """Labels under a concept arm at dose fraction frac; xv is the top feature of the same rows."""
    lab = lab.copy()
    if name == "region":
        lab[xv > np.nanquantile(b1_x, 1 - 0.5 * frac)] ^= 1
        return lab
    R = np.flatnonzero(xv > np.nanmedian(b1_x))
    if name == "noise":
        lab[rng.choice(R, round(0.5 * frac * len(R)), replace=False)] ^= 1
    else:
        pos, neg = R[lab[R] == 1], R[lab[R] == 0]
        k = round(frac * len(pos) * len(neg) / max(len(R), 1))
        lab[rng.choice(pos, k, replace=False)] = 0
        lab[rng.choice(neg, k, replace=False)] = 1
    return lab


def scenario(name: str, y: np.ndarray, P: list[np.ndarray], z: np.ndarray, x: np.ndarray | None,
             b1_x: np.ndarray | None, rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    """Test blocks T1..T4 as (row indices, labels). x is the top feature, b1_x its B1 values."""
    blocks = []
    for h, frac in enumerate(FRACS, start=1):
        pool = P[h]
        m = len(pool) // 2
        if name == "cov":
            v = z[pool] ** 2
            keys = np.log(rng.random(len(pool))) / weights(v, tilt(v, m, 1 + COV_SHIFT * frac))
            idx = pool[np.argpartition(-keys, m)[:m]]
        elif name.startswith("prior"):
            pos, neg = pool[y[pool] == 1], pool[y[pool] == 0]
            odds = len(pos) / len(neg) * float(name[5:]) ** frac
            k = round(m * odds / (1 + odds))
            idx = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, m - k, replace=False)])
        else:
            idx = rng.choice(pool, m, replace=False)
        lab = relabel(name, y[idx], x[idx], b1_x, frac, rng) if name in CONCEPT else y[idx].copy()
        blocks.append((idx, lab))
    return blocks


def oracle_window(name: str, y: np.ndarray, P: list[np.ndarray], blocks: list, x: np.ndarray,
                  b1_x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """The rows of P4 and P5 left out of T3 and T4, relabeled at the full dose of a concept arm."""
    idx = np.concatenate([np.setdiff1d(P[h], blocks[h - 1][0]) for h in (3, 4)])
    return idx, relabel(name, y[idx], x[idx], b1_x, 1.0, rng)


def t0(P: list[np.ndarray], seed: int) -> np.ndarray:
    """The half of B1 that stands in for T0 in the S1 window before T2."""
    return np.random.default_rng([seed, 100]).choice(P[0], len(P[0]) // 2, replace=False)


def s1_window(h: int, blocks: list, half_b1: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows and labels a sliding refit before T_h, h = 2..4, is fitted on."""
    parts = [blocks[h - 2], blocks[h - 3] if h > 2 else (half_b1, y[half_b1])]
    return np.concatenate([i for i, _ in parts]), np.concatenate([lab for _, lab in parts])


def s2_window(h: int, blocks: list, b1: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows and labels an expanding refit before T_h, h = 2..4, is fitted on."""
    parts = [(b1, y[b1]), *blocks[:h - 1]]
    return np.concatenate([i for i, _ in parts]), np.concatenate([lab for _, lab in parts])


def windows(sc: str, seed: int, y: np.ndarray, P: list[np.ndarray], z: np.ndarray,
            x: np.ndarray | None) -> dict:
    """Everything one scenario fits and scores, from its own random stream: the test blocks, the S1
    and S2 fitting windows before T2..T4 and, for a concept arm, the oracle window. The gate and
    the runner both take their rows and labels from here."""
    rng = np.random.default_rng([seed, SCENARIOS.index(sc)])
    b1_x = x[P[0]] if sc in CONCEPT else None
    blocks = scenario(sc, y, P, z, x, b1_x, rng)
    half = t0(P, seed)
    return dict(blocks=blocks,
                s1={h: s1_window(h, blocks, half, y) for h in (2, 3, 4)},
                s2={h: s2_window(h, blocks, P[0], y) for h in (2, 3, 4)},
                oracle=oracle_window(sc, y, P, blocks, x, b1_x, rng) if sc in CONCEPT else None)


def scheduled(name: str, lab: np.ndarray, xv: np.ndarray, b1_x: np.ndarray, frac: float) -> int:
    """The relabeled count a concept arm's rule prescribes for these rows."""
    if name == "region":
        return int((xv > np.nanquantile(b1_x, 1 - 0.5 * frac)).sum())
    R = xv > np.nanmedian(b1_x)
    if name == "noise":
        return round(0.5 * frac * R.sum())
    pos = lab[R].sum()
    return 2 * round(frac * pos * (R.sum() - pos) / max(R.sum(), 1))


def gate_rows(name: str, scenarios: list[str] = SCENARIOS) -> list[dict]:
    """One row per seed, scenario and block, with the quantities the gate reads. Block 0 is the
    oracle window of a concept arm."""
    X, y, cat_cols = load_data_df(name)
    y = minority_is_one(y)
    z = pc_score(X, cat_cols)
    top, top_mi = top_feature(X, y, cat_cols)
    x = X[top].astype(float).to_numpy() if top else None
    rows = []
    for seed in seeds(name):
        P = pools(y, seed)
        b1_x = x[P[0]] if top else None
        for sc in scenarios:
            if sc in CONCEPT and top is None:
                continue
            w = windows(sc, seed, y, P, z, x)
            scored = [(h, FRACS[h - 1], *b) for h, b in enumerate(w["blocks"], start=1)]
            if sc in CONCEPT:
                scored.append((0, 1.0, *w["oracle"]))
            for h, frac, idx, lab in scored:
                if sc.startswith("prior"):
                    pool = P[h]
                    odds = y[pool].sum() / (len(pool) - y[pool].sum()) * float(sc[5:]) ** frac
                    expected = len(idx) * odds / (1 + odds)
                elif sc in CONCEPT:
                    expected = scheduled(sc, y[idx], x[idx], b1_x, frac)
                else:
                    expected = np.nan
                rows.append(dict(
                    dataset=name, seed=seed, scenario=sc, block=h, m=len(idx), top_feature=top, top_mi=top_mi,
                    b1_minority=int(y[P[0]].sum()), mean_z=z[idx].mean(), mean_z2=(z[idx] ** 2).mean(), positives=int(lab.sum()),
                    min_class=int(min(lab.sum(), len(lab) - lab.sum())),
                    relabeled=int((lab != y[idx]).sum()), expected=expected,
                    prevalence_change=int(lab.sum() - y[idx].sum()),
                    step_auc=roc_auc_score(lab, x[idx] > np.nanmedian(b1_x))
                    if sc == "region" and h == 4 and 0 < lab.sum() < len(lab) else np.nan))
    return rows


def verdicts(g: pd.DataFrame) -> pd.DataFrame:
    """Per dataset and arm: pass or the failed conditions of the gate."""
    arm = g.scenario.map(lambda s: "prior" if s.startswith("prior") else s)
    out = []
    for (ds, a), d in g.assign(arm=arm).groupby(["dataset", "arm"], sort=False):
        fails = []
        if (d.min_class < MIN_MINORITY).any():
            worst = d.loc[d.min_class.idxmin()]
            fails.append(f"min class {worst.min_class} ({worst.scenario}, seed {worst.seed}, T{worst.block})")
        if (d.b1_minority < 2 * MIN_MINORITY).any():
            fails.append("B1 minority under 30")
        if a == "cov" and abs(d[d.block == 4].mean_z2.mean() - 1 - COV_SHIFT) > 0.1:
            fails.append(f"T4 mean z^2 {d[d.block == 4].mean_z2.mean():.3f}")
        if a == "prior" and ((d.positives - d.expected).abs() > 1).any():
            fails.append("prior off schedule")
        if a in CONCEPT and (d.relabeled != d.expected).any():
            fails.append("relabel count off schedule")
        if a == "swap" and (d.prevalence_change != 0).any():
            fails.append("swap moved prevalence")
        out.append(dict(dataset=ds, arm=a, verdict="; ".join(fails) or "pass"))
    return pd.DataFrame(out)


if __name__ == "__main__":
    g = pd.DataFrame([r for name in STUDY_B for r in gate_rows(name)]
                     + [r for name in CONCEPT_ONLY for r in gate_rows(name, ["region"])])
    g.to_csv("results/drift_gate.csv", index=False)
    v = verdicts(g)
    missing = sorted(set(STUDY_B + CONCEPT_ONLY) - set(v[v.arm == "region"].dataset))
    print(v.pivot(index="dataset", columns="arm", values="verdict").to_string())
    print("no top feature (out of the concept arms):", missing or "none")
    t4 = g[g.block == 4].assign(prevalence=g.positives / g.m)
    print(t4[t4.scenario.isin(["none", "cov"])].groupby(["dataset", "scenario"])[["mean_z", "mean_z2", "prevalence"]]
          .mean().unstack().round(3).to_string())
    print(g.groupby("dataset")[["top_feature", "top_mi"]].first().round(3).to_string())
