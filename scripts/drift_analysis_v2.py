"""drift_analysis_v2.py: the analysis of plan_v2.md. Part 0 (cost of a refresh) reads the v1 checkpoints.

Usage: uv run python -m scripts.drift_analysis_v2
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.drift_analysis import holm, signed_rank, signed_rank_null
from scripts.drift_run import RESULTS
from scripts.drift_windows import PUBLIC_A, STUDY_B

OUT = Path(__file__).resolve().parent.parent / "results" / "v2"
DEVICE = {"tabfm": "mps"}  # SDB's TabFM wrapper defaults to mps; every other model ran on cpu


def refresh_rows(model: str, times: list[dict], scenario: str = "none") -> list[dict]:
    """One row per refresh at T2 to T4 of the control scenario (Study A has none: scenario "") and strategy S1 or S2: the fit on the window
    plus the prediction of the block (a TFM fits lazily, so its cost is in the prediction), and for S2 the
    re-tuning of a booster on that window. A model's cost has two parts, reported apart: the fit (and tuning) per
    1,000 window rows, and the prediction per 1,000 block rows."""
    t = {x["stage"]: x for x in times}
    rows = []
    for strategy in ("S1", "S2"):
        for h in (2, 3, 4):
            tag = f"{strategy} {scenario} {h}".replace("  ", " ")
            fit, pred = t[f"fit {tag}"], t[f"predict {tag}"]
            tune = t.get(f"tune S2_{h}", {"seconds": 0.0}) if strategy == "S2" else {"seconds": 0.0}
            rows.append(dict(model=model, device=DEVICE.get(model, "cpu"), strategy=strategy, block=h, window_rows=fit["rows"],
                             block_rows=pred["rows"], fit=fit["seconds"], predict=pred["seconds"], tune=tune["seconds"],
                             refresh=fit["seconds"] + pred["seconds"] + tune["seconds"]))
    return rows


def refresh_cost(runs: Path = RESULTS) -> pd.DataFrame:
    rows = []
    for ds in (*STUDY_B, *PUBLIC_A):  # restricted datasets are left out
        for p in sorted((runs / ds).glob("*.joblib")):
            r = joblib.load(p)
            rows += [dict(row, dataset=ds, seed=r["seed"])
                     for row in refresh_rows(r["model"], r["times"], "none" if ds in STUDY_B else "")]
    d = pd.DataFrame(rows).assign(fit_per_1000_window_rows=lambda x: (x.fit + x.tune) / x.window_rows * 1000,
                                  predict_per_1000_block_rows=lambda x: x.predict / x.block_rows * 1000)
    g = d.groupby(["model", "device", "strategy"])
    return g.agg(refreshes=("refresh", "size"), median_window_rows=("window_rows", "median"), median_block_rows=("block_rows", "median"),
                 median_fit=("fit", "median"), median_tune=("tune", "median"), median_predict=("predict", "median"),
                 median_refresh=("refresh", "median"), fit_per_1000_window_rows=("fit_per_1000_window_rows", "median"),
                 predict_per_1000_block_rows=("predict_per_1000_block_rows", "median")).reset_index()


def one_sided_p(d: np.ndarray) -> float:
    """Exact one-sided signed-rank p-value that the center of symmetry of d is above zero. Zeros are dropped and
    the absolute differences must be distinct."""
    d = d[d != 0]
    n = len(d)
    assert len(np.unique(np.abs(d))) == n, "tied absolute differences: the exact p-value would not be exact"
    w = int(round(np.sum(np.argsort(np.argsort(np.abs(d))) [d > 0] + 1)))
    return float(signed_rank_null(n)[w:].sum() / 2.0 ** n)


def tost(values: pd.Series, margin: float) -> dict:
    """Two one-sided exact signed-rank tests that the pseudo-median of the per-dataset values lies inside
    (-margin, margin): the values shifted by +margin must have a center above zero, and shifted by -margin below.
    The TOST p-value is the larger of the two."""
    v = values.dropna().to_numpy(dtype=float)
    lower, upper = one_sided_p(v + margin), one_sided_p(margin - v)
    return dict(n=len(v), margin=margin, p_lower=lower, p_upper=upper, p=max(lower, upper))


def equivalence_family(values: dict[str, pd.Series], margins: dict[str, float], min_n: int = 25, alpha: float = 0.05) -> pd.DataFrame:
    """E1 to E3: the verdict is the Holm-adjusted TOST p-value; each interval is the Hodges-Lehmann interval at
    the level the adjusted test uses, 1 - 2 * alpha / rank of the test in the Holm order."""
    t = {k: tost(values[k], margins[k]) for k in values}
    testable = {k: r["p"] for k, r in t.items() if r["n"] >= min_n}
    adj = holm(testable) if testable else {}
    order = sorted(testable, key=testable.get)
    rows = []
    for k, r in t.items():
        step = len(testable) - order.index(k) if k in testable else 1
        ci = signed_rank(values[k], level=1 - 2 * alpha / step)
        rows.append(dict(test=k, n=r["n"], margin=r["margin"], est=ci["est"], lo=ci["lo"], hi=ci["hi"], ci_level=1 - 2 * alpha / step,
                         tost_p=r["p"], holm_p=adj.get(k, np.nan),
                         verdict="underpowered" if r["n"] < min_n else ("equivalent" if adj[k] < alpha else "no equivalence shown")))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cost = refresh_cost()
    cost.to_csv(OUT / "refresh_cost.csv", index=False)
    print(cost.round(3).to_string(index=False))
