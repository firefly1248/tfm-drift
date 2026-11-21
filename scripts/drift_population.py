"""drift_population.py: the datasets of Study B2 and Study C, by the rules of plan_v2.md, and their gate. No TFM
and no tuning is run; the only fit is the default LightGBM used for the ceiling and floor rules.

Usage: uv run python -m scripts.drift_population   (writes results/v2/population*.csv)
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from scripts.drift_data_v2 import NotBinary, load_openml, load_sdb
from scripts.drift_windows import PUBLIC_A, STUDY_B
from scripts.drift_windows_v2 import (
    gate_b2,
    gate_c,
    gate_e,
    pools_c,
    seeds_v2,
    verdict_b2,
    verdict_c,
    z_extrap,
)
from sdb import ROOT

OUT = Path(__file__).resolve().parent.parent / "results" / "v2"
SUITES_B2 = {99: "OpenML-CC18", 457: "TabArena-v0.1"}
RESERVE = {271: "AutoML benchmark", 337: "Grinsztajn classification", 334: "Grinsztajn classification, numeric"}
SUITES_C = {**SUITES_B2, **RESERVE}
MIN_ROWS, MAX_FEATURES, MIN_MINORITY_SHARE = 1_000, 500, 0.05
CEILING, FLOOR, LOW_SIGNAL = 0.99, 0.60, 0.65
C_ROWS = 36_000
CAP_B2 = 10_000
# names under which v1 datasets, or versions of them, appear in the sources
V1_ALIASES = {"credit-g", "qsar-biodeg", "spambase", "phoneme", "bank-marketing", "electricity", "banknote-authentication",
              "MagicTelescope", "magic", "ozone-level-8hr", "sick", "hill-valley", "hill-valley-without-noise",
              "bank-marketing-full", "bank-full", "elec", "eeg-eye-state", "banana"}
V1_NAMES = set(STUDY_B) | {"ringnorm", "skin-segmentation", "twonorm"} | set(PUBLIC_A) | V1_ALIASES


def lightgbm_auc(X: pd.DataFrame, y: np.ndarray, cat_cols: list[str]) -> float:
    X = X.copy()
    X[cat_cols] = X[cat_cols].astype("category")
    pred = np.zeros(len(y))
    for fit, val in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = lgb.LGBMClassifier(verbose=-1, n_jobs=4).fit(X.iloc[fit], y[fit])
        pred[val] = m.predict_proba(X.iloc[val])[:, 1]
    return float(roc_auc_score(y, pred))


def suite_candidates(suites: dict[int, str]) -> pd.DataFrame:
    import openml
    rows = []
    for sid, sname in suites.items():
        ids = openml.study.get_suite(sid).data
        meta = openml.datasets.list_datasets(data_id=ids, output_format="dataframe")
        for r in meta.itertuples():
            rows.append(dict(source=f"openml:{sname}", suite=sid, did=int(r.did), name=r.name, version=int(r.version),
                             classes=r.NumberOfClasses))
    return pd.DataFrame(rows)


def sdb_candidates() -> pd.DataFrame:
    meta = pd.read_csv(ROOT / "results" / "datasets.csv").set_index("dataset")
    names = sorted(meta.index[~meta.uci_duplicate]) + sorted(p.stem for p in (ROOT / "datasets" / "external").glob("*.csv"))
    return pd.DataFrame([dict(source="sdb", suite=0, did=-1, name=n, version=0, classes=2) for n in names])


def assess(c: pd.Series, cap: int | None, stratified: bool, rows_needed: int | None = None):
    """Load one candidate and measure it: (frame, label, categorical columns, record)."""
    rec = dict(source=c.source, did=c.did, name=c["name"], version=c.version)
    X, y, cats, n = (load_sdb(c["name"]) if c.source == "sdb" else load_openml(int(c.did), cap=cap, stratified=stratified))
    return X, y, cats, rec | n


def build_b2() -> pd.DataFrame:
    cand = pd.concat([suite_candidates(SUITES_B2), sdb_candidates(), suite_candidates(RESERVE)], ignore_index=True)
    # TabArena (457) first, then CC18, SDB and the reserve suites: the tie rule for copies of a dataset
    cand["rank"] = cand.suite.map({457: 0, 99: 1, 0: 2, 271: 3, 337: 4, 334: 5})
    cand = cand.sort_values(["rank", "name"]).reset_index(drop=True)
    recs, seen = [], {}
    for _, c in cand.iterrows():
        rec = dict(source=c.source, did=c.did, name=c["name"], version=c.version, stage="binary", reserve=c.suite in RESERVE)
        if c.source != "sdb" and c.classes != 2:
            recs.append(rec | dict(stage="not binary", reason=f"{c.classes} classes"))
            continue
        try:
            X, y, cats, rec2 = assess(c, CAP_B2, False)
        except NotBinary as e:
            recs.append(rec | dict(stage="not binary", reason=str(e)))
            continue
        except Exception as e:  # a file that cannot be read is a rejected file, with its reason on record
            recs.append(rec | dict(stage="load", reason=f"{type(e).__name__}: {str(e)[:80]}"))
            continue
        rec |= rec2 | dict(stage="loaded")
        print(f"{c.source} {c['name']}: {rec['rows']} rows, {rec['features']} features, minority {rec['minority']:.3f}", flush=True)
        if c["name"] in V1_NAMES:
            recs.append(rec | dict(stage="v1", reason="a v1 dataset or a version of one"))
            continue
        fp = (rec["rows_dedup"], rec["features"], round(rec["minority"], 3))
        twin = seen.get(fp) or (c["name"] if c["name"] in seen.values() else None)
        if twin:
            recs.append(rec | dict(stage="copy", reason=f"same fingerprint or name as {twin}"))
            continue
        seen[fp] = c["name"]
        if rec["rows"] < MIN_ROWS or rec["features"] > MAX_FEATURES or rec["minority"] < MIN_MINORITY_SHARE:
            recs.append(rec | dict(stage="size", reason=f"{rec['rows']} rows, {rec['features']} features, minority {rec['minority']:.3f}"))
            continue
        auc = lightgbm_auc(X, y, cats)
        rec["lgbm_auc"] = auc
        if not FLOOR <= auc < CEILING:
            recs.append(rec | dict(stage="ceiling or floor", reason=f"LightGBM AUC {auc:.3f}"))
            continue
        sr = seeds_v2(rec["rows"])
        e = gate_e(c["name"], X, y, cats, sr)
        x = verdict_b2(gate_b2(X, y, cats, sr))
        recs.append(rec | dict(stage="gated", reason="", e_gate=e, x_gate=x, e_pass=e == "pass", x_pass=x == "pass",
                               seeds=len(sr), low_signal=auc < LOW_SIGNAL))
    return pd.DataFrame(recs)



def build_c() -> pd.DataFrame:
    cand = pd.concat([suite_candidates(SUITES_C), sdb_candidates()], ignore_index=True)
    cand["rank"] = cand.suite.map({457: 0, 99: 1, 271: 2, 337: 3, 334: 4, 0: 5})
    cand = cand.sort_values(["rank", "name"]).drop_duplicates("name").reset_index(drop=True)
    recs, seen = [], {}
    for _, c in cand.iterrows():
        rec = dict(source=c.source, did=c.did, name=c["name"], version=c.version)
        if c.source != "sdb" and c.classes != 2:
            continue
        try:
            X, y, cats, rec2 = assess(c, None, False)
        except NotBinary:
            continue
        except Exception as e:
            recs.append(rec | dict(stage="load", reason=f"{type(e).__name__}: {str(e)[:80]}"))
            continue
        rec |= rec2
        if rec["rows_dedup"] < C_ROWS:
            continue
        print(f"C candidate {c['name']}: {rec['rows_dedup']} rows after deduplication", flush=True)
        if c["name"] in V1_NAMES:
            recs.append(rec | dict(stage="v1", reason="a v1 dataset or a version of one"))
            continue
        fp = (rec["rows_dedup"], rec["features"], round(rec["minority"], 3))
        if fp in seen:
            recs.append(rec | dict(stage="copy", reason=f"same fingerprint as {seen[fp]}"))
            continue
        seen[fp] = c["name"]
        if rec["features"] > MAX_FEATURES or rec["minority"] < MIN_MINORITY_SHARE:
            recs.append(rec | dict(stage="size", reason=f"{rec['features']} features, minority {rec['minority']:.3f}"))
            continue
        X, y, cats, rec2 = assess(c, C_ROWS, True)
        auc = lightgbm_auc(X, y, cats)
        rec |= rec2 | dict(lgbm_auc=auc)
        if not FLOOR <= auc < CEILING:
            recs.append(rec | dict(stage="ceiling or floor", reason=f"LightGBM AUC {auc:.3f}"))
            continue
        seeds = range(3)
        z = z_extrap(X, cats, y, max_levels=20)
        e = gate_e(c["name"], X, y, cats, seeds, pool_fn=pools_c, z=z)
        x = verdict_c(gate_c(X, y, cats, seeds))
        recs.append(rec | dict(stage="gated", reason="", e_gate=e, x_gate=x, e_pass=e == "pass", x_pass=x == "pass",
                               low_signal=auc < LOW_SIGNAL))
    return pd.DataFrame(recs)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("b2", "both"):
        build_b2().to_csv(OUT / "population.csv", index=False)
    if which in ("c", "both"):
        build_c().to_csv(OUT / "population_c.csv", index=False)
