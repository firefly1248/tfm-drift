"""drift_analysis.py: the analysis of plan.md sections 5 and 6 on the checkpoints in results/runs/.

Scores every stored prediction (results/drift_scores.csv), runs the checks of section 6, the five
primary tests with Holm, their companions and reading rules, and the exploratory tables, all as
results/drift_*.csv. Every table is joined by dataset, scenario, model and seed name (check 7), never
by position. Figures are made by scripts/drift_figures.py from these files.

Usage: uv run python -m scripts.drift_analysis
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import sdb
from scripts.drift_queue import CONCEPT as _CONCEPT
from scripts.drift_run import BOOSTERS, RESULTS, TFMS
from scripts.drift_windows import CONCEPT as ARMS
from scripts.drift_windows import (
    PUBLIC_A,
    STUDY_A,
    STUDY_B,
    minority_is_one,
    pc_score,
    pools,
    seeds,
    top_feature,
)

CONCEPT_DATASETS = list(_CONCEPT)
OUT = Path(__file__).resolve().parent.parent / "results"
CLIP = 1e-6
GUARD = 0.05            # H5: oracle gain needed in both families
MIN_N = 8               # a test needs at least this many non-zero differences
FAMILIES = (("tfm", TFMS), ("booster", BOOSTERS))
PRIORS = ("prior0.5", "prior2")
NO_TOP = "thyroid-sick-euthyroid"  # no admissible top feature: it has no concept arms (plan 4.2)
# H5 population (plan 4.2): the nine main datasets with a top feature, plus the three concept-only ones
H5_POP = [d for d in STUDY_B if d != NO_TOP] + CONCEPT_DATASETS
BAND = (0.5, 0.625)     # quantiles of the top feature in B1: rows in (median, q0.625] sit in the moving band
BENCH = {"tfm": ["tabfm", "tabpfn35", "tabicl"], "booster": ["catboost_logloss", "lgbm_logloss", "xgboost_logloss"]}


# ---------------------------------------------------------------- scoring

def clipped(p: np.ndarray) -> np.ndarray:
    return np.clip(p.astype(np.float64), CLIP, 1 - CLIP)


def metrics(p: np.ndarray, lab: np.ndarray) -> dict:
    """Scores of probabilities of label 1; log loss, Brier and ECE use float64 probabilities clipped to
    [1e-6, 1 - 1e-6] for every model (plan 4.5). Brier is the binary one."""
    from benchmark.calibration import expected_calibration_error  # on sys.path once sdb is imported
    q, n, k = clipped(p), len(lab), int(lab.sum())
    both = 0 < k < n
    prev = k / n
    ap = average_precision_score(lab, p) if both else np.nan
    ll = -(lab * np.log(q) + (1 - lab) * np.log(1 - q))
    return dict(n=n, n_unique=len(np.unique(p)), prevalence=prev, auc=roc_auc_score(lab, p) if both else np.nan, ap=ap,
                skill=(ap - prev) / (1 - prev) if both else np.nan, logloss=ll.mean(),
                brier=np.mean((q - lab) ** 2), ece=expected_calibration_error(lab, np.column_stack([1 - q, q])),
                ll_pos=ll[lab == 1].mean() if k else np.nan, ll_neg=ll[lab == 0].mean() if k < n else np.nan,
                finite=bool(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()))


def s0_star(p_window: np.ndarray, y_window: np.ndarray, p_test: np.ndarray) -> np.ndarray:
    """The static model with its intercept re-estimated by maximum likelihood on the window: the one c
    with sum(expit(logit(p) + c)) = sum(y), so the window's mean prediction equals its prevalence."""
    c = brentq(lambda c: expit(logit(clipped(p_window)) + c).sum() - y_window.sum(), -30, 30)
    return clipped(expit(logit(clipped(p_test)) + c))


def score_dataset(ds: str) -> list[dict]:
    """One row per model, seed, scenario, strategy and block of a dataset, plus S0* at T3 and T4 of
    Study B, the T4 rows outside the moving band (region arm) and the control's calibration slope."""
    study_b = ds not in STUDY_A
    if study_b:
        X, y, cats = sdb.load_data_df(ds)
        y = minority_is_one(y)
        z = pc_score(X, cats)
        top, _ = top_feature(X, y, cats)
        x = X[top].astype(float).to_numpy() if top else None
    out = []
    for path in sorted((RESULTS / ds).glob("*.joblib")):
        d = joblib.load(path)
        key0 = dict(study="B" if study_b else "A", dataset=ds, model=d["model"], seed=d["seed"])
        if study_b and x is not None:
            b1_x = x[pools(y, d["seed"])[0]]
            lo, hi = np.nanmedian(b1_x), np.nanquantile(b1_x, BAND[1])
        for (sc, st, h), p in d["preds"].items():
            lab, rows = d["labels"][sc, st, h], d["rows"][sc, st, h]
            r = dict(**key0, scenario=sc, strategy=st, block=h, **metrics(p, lab))
            if study_b:
                r["z2"], r["relabeled"] = float(np.mean(z[rows] ** 2)), float((lab != y[rows]).mean())
                if sc == "region" and h == 4:
                    keep = ~((x[rows] > lo) & (x[rows] <= hi))
                    r["auc_out"] = roc_auc_score(lab[keep], p[keep]) if 0 < lab[keep].sum() < keep.sum() else np.nan
                if (sc, st, h) == ("none", "S0", 4):
                    r["cal_slope"] = LogisticRegression(penalty=None, max_iter=1000).fit(
                        logit(clipped(p))[:, None], lab).coef_[0, 0]
            out.append(r)
            if study_b and st == "S0" and h in (3, 4):
                win = [d["preds"][sc, "S0", h - 1], d["preds"][sc, "S0", h - 2]]
                labs = [d["labels"][sc, "S0", h - 1], d["labels"][sc, "S0", h - 2]]
                q = s0_star(np.concatenate(win), np.concatenate(labs), p)
                out.append(dict(**key0, scenario=sc, strategy="S0*", block=h, **metrics(q, lab),
                                z2=r["z2"], relabeled=r["relabeled"]))
    return out


def parallel(name: str, items) -> list:
    """Run this module's function `name` over items in worker processes. The function is taken from the
    imported module, not from __main__, which loky workers started with -m cannot unpickle."""
    from scripts import drift_analysis as me
    return Parallel(n_jobs=4)(delayed(getattr(me, name))(i) for i in items)


def score_all() -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parts = parallel("score_dataset", STUDY_B + CONCEPT_DATASETS + PUBLIC_A)
    return pd.DataFrame([r for part in parts for r in part])


# ---------------------------------------------------------------- statistics

def signed_rank_null(n: int) -> np.ndarray:
    """Counts of the subsets of 1..n by sum: the null distribution of the signed-rank statistic."""
    c = np.zeros(n * (n + 1) // 2 + 1)
    c[0] = 1
    for r in range(1, n + 1):
        c = c + np.r_[np.zeros(r), c[:-r]]
    return c


def signed_rank(values: pd.Series, level: float = 0.95) -> dict:
    """Exact two-sided Wilcoxon signed-rank test with the Hodges-Lehmann estimate and its interval from
    the Walsh averages and the exact quantile. Zero differences are dropped. n_nonzero is the n of the
    test; below MIN_N it is untestable."""
    d = values.dropna().to_numpy(dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return dict(n_datasets=int(values.notna().sum()), n_nonzero=0, est=np.nan, lo=np.nan, hi=np.nan, p=np.nan)
    assert len(np.unique(np.abs(d))) == n, "tied absolute differences: the exact p-value would not be exact"
    walsh = np.sort(((d[:, None] + d[None, :]) / 2)[np.triu_indices(n)])
    cdf = np.cumsum(signed_rank_null(n)) / 2.0 ** n
    k = int((cdf <= (1 - level) / 2).sum()) - 1   # largest k with P(W <= k) <= alpha / 2
    lo, hi = (walsh[k], walsh[len(walsh) - 1 - k]) if k >= 0 else (np.nan, np.nan)
    return dict(n_datasets=int(values.notna().sum()), n_nonzero=n, est=float(np.median(walsh)), lo=lo, hi=hi,
                p=float(wilcoxon(d, method="exact").pvalue))


def holm(p: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values of the testable tests."""
    adj, run = {}, 0.0
    order = sorted(p, key=p.get)
    for i, name in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * p[name]))
        adj[name] = run
    return adj


# ---------------------------------------------------------------- per-dataset quantities

def cell(S: pd.DataFrame, metric: str, sc: str, st: str, h: int) -> pd.Series:
    d = S[(S.scenario == sc) & (S.strategy == st) & (S.block == h)]
    return d.set_index(["dataset", "seed", "model"])[metric]


def famean(x: pd.Series, models) -> pd.Series:
    """Per dataset: the mean over the family's models, then over seeds; missing if any cell or any seed
    of the dataset is missing."""
    per = x.unstack("model")[list(models)].mean(axis=1, skipna=False)
    g = per.groupby("dataset")
    n = g.count()
    return g.mean().where(n == [len(seeds(d)) for d in n.index])


def lead(x: pd.Series) -> pd.Series:
    """TFM family mean minus booster family mean, per dataset, averaged over seeds."""
    return famean(x, TFMS) - famean(x, BOOSTERS)


def shares(S: pd.DataFrame, sc: str, metric: str = "auc") -> pd.DataFrame:
    """Per dataset and family at T4: S0, S1, oracle, oracle gain and the recovered share
    (S1 - S0) / (oracle - S0) of family means; guard and the share difference TFM minus booster."""
    parts = {}
    for f, models in FAMILIES:
        t = pd.DataFrame({st: famean(cell(S, metric, sc, st, 4), models) for st in ("S0", "S1", "oracle")})
        t["gain"] = t.oracle - t.S0
        t["share"] = (t.S1 - t.S0) / t.gain
        parts[f] = t
    t = pd.concat(parts, axis=1)
    t.columns = [f"{f}_{c}" for f, c in t.columns]
    t["guard"] = (t.tfm_gain >= GUARD) & (t.booster_gain >= GUARD)
    t["diff"] = (t.tfm_share - t.booster_share).where(t.guard)
    for f, _ in FAMILIES:  # a family's own share is shown only where its own gain reaches the guard
        t[f"{f}_share"] = t[f"{f}_share"].where(t[f"{f}_gain"] >= GUARD)
    return t


def concept_table(S: pd.DataFrame) -> pd.DataFrame:
    """The shares of the three concept arms with the realized relabeled share and prevalence, the
    distance S1 - oracle in ROC AUC and log loss, and, for the region arm, the same share
    recomputed outside the moving band (columns ending _out)."""
    out = []
    for sc in ARMS:
        t = shares(S, sc)
        for f, models in FAMILIES:
            for m in ("auc", "logloss"):
                t[f"{f}_s1_minus_oracle_{m}"] = (famean(cell(S, m, sc, "S1", 4), models)
                                                 - famean(cell(S, m, sc, "oracle", 4), models))
        t4 = S[(S.scenario == sc) & (S.strategy == "S0") & (S.block == 4)].groupby("dataset")
        t["relabeled"], t["prevalence"] = t4.relabeled.mean(), t4.prevalence.mean()
        if sc == "region":
            t = t.join(shares(S, sc, "auc_out").add_suffix("_out"))
        out.append(t.assign(scenario=sc))
    return pd.concat(out).rename_axis("dataset").reset_index()


def model_shares(S: pd.DataFrame) -> pd.DataFrame:
    """Single-model gains and shares in the region arm, the share shown only where the gain reaches GUARD."""
    d = S[(S.scenario == "region") & (S.block == 4) & S.strategy.isin(["S0", "S1", "oracle"])]
    t = d.groupby(["dataset", "model", "strategy"]).auc.mean().unstack("strategy")
    t["gain"] = t.oracle - t.S0
    t["share"] = ((t.S1 - t.S0) / t.gain).where(t.gain >= GUARD)
    return t.reset_index()


def log_error_ratio(S: pd.DataFrame, sc: str) -> pd.Series:
    """log((1 - AUC of TFMs) / (1 - AUC of boosters)) at T4 under S0, errors floored at 0.001."""
    err = {f: (1 - famean(cell(S, "auc", sc, "S0", 4), m)).clip(lower=0.001) for f, m in FAMILIES}
    return np.log(err["tfm"] / err["booster"])


def primary_values(S: pd.DataFrame) -> dict[str, pd.Series]:
    """The per-dataset value of each primary test (plan section 5)."""
    def c(m, sc, st, h):
        return cell(S, m, sc, st, h)

    def refit(m, sc):
        return c(m, sc, "S1", 4) - c(m, sc, "S0", 4)

    def star(sc):
        return c("logloss", sc, "S1", 4) - c("logloss", sc, "S0*", 4)

    v = {"H1": lead(c("auc", "cov", "S0", 4)) - lead(c("auc", "none", "S0", 4)),
         "H2": lead(refit("auc", "cov")) - lead(refit("auc", "none")),
         "H4": lead(c("logloss", "cov", "S0", 4)) - lead(c("logloss", "none", "S0", 4))}
    for r in PRIORS:
        v[f"H3 {r}"] = lead(star(r)) - lead(star("none"))
    v["H3"] = (v["H3 prior0.5"] + v["H3 prior2"]) / 2
    v["H5"] = shares(S, "region")["diff"]
    v = {k: s.reindex(STUDY_B if k != "H5" else H5_POP) for k, s in v.items()}
    return {k: v[k] for k in ("H1", "H2", "H3", "H4", "H5", "H3 prior0.5", "H3 prior2")}


QUANTITY = {"H1": "lead in ROC AUC at T4, S0: covariate minus control",
            "H2": "lead in (S1 - S0) ROC AUC at T4: covariate minus control",
            "H3": "lead in (S1 - S0*) log loss at T4: prior minus control, mean of two levels",
            "H4": "lead in log loss at T4, S0: covariate minus control",
            "H5": "TFM share minus booster share at T4, region, ROC AUC"}


def primary_tests(S: pd.DataFrame, v: dict[str, pd.Series]) -> pd.DataFrame:
    rows = {k: signed_rank(v[k]) for k in QUANTITY}
    testable = {k: r["p"] for k, r in rows.items() if r["n_nonzero"] >= MIN_N}
    adj = holm(testable)
    t = pd.DataFrame.from_dict(rows, orient="index").rename_axis("test").reset_index()
    t["quantity"] = t.test.map(QUANTITY)
    t["holm_p"] = t.test.map(adj)
    t["verdict"] = [("untestable" if k not in adj else "reject" if adj[k] < 0.05 else "not rejected") for k in t.test]
    return t


def leave_one_out(v: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for k in QUANTITY:
        for ds in v[k].dropna().index:
            r = signed_rank(v[k].drop(ds))
            rows.append(dict(test=k, left_out=ds, est=r["est"], p=r["p"], n_nonzero=r["n_nonzero"]))
    return pd.DataFrame(rows)


def companions_h1_h4(S: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per family: the change from control to covariate of error (1 - ROC AUC) and log loss, per dataset;
    and the test of the change of the log error ratio."""
    rows = []
    for f, models in FAMILIES:
        for m in ("auc", "logloss"):
            d = famean(cell(S, m, "cov", "S0", 4), models) - famean(cell(S, m, "none", "S0", 4), models)
            rows.append(pd.DataFrame(dict(dataset=d.index, family=f, change=(-d if m == "auc" else d).values,
                                          metric="error (1 - AUC)" if m == "auc" else "log loss")))
    ratio = (log_error_ratio(S, "cov") - log_error_ratio(S, "none")).reindex(STUDY_B)
    return pd.concat(rows), signed_rank(ratio)


def reading_h1(h1: dict, ratio: dict) -> str:
    """plan 5: a rejection of H1 is a different response only if the log error ratio moves in the same
    direction as the TFMs' advantage. A larger lead is a smaller error ratio, so the two estimates must
    have opposite signs; only the sign of the ratio's estimate is read, as the plan words it."""
    if h1["verdict"] != "reject":
        return "not rejected, no reading"
    return ("a different response to drift" if np.sign(ratio["est"]) == -np.sign(h1["est"])
            else "a change of scale (the log error ratio does not move with the lead)")


def reading_h5(S: pd.DataFrame, h5: dict) -> tuple[str, pd.DataFrame]:
    """plan 5: adaptation only if the lead in S1 - S0 and the share difference outside the band both carry
    H5's sign at exact p < 0.05; otherwise the missing one is named."""
    t = shares(S, "region").join(shares(S, "region", "auc_out").add_suffix("_out"))
    s1s0 = lead(cell(S, "auc", "region", "S1", 4) - cell(S, "auc", "region", "S0", 4)).reindex(H5_POP)
    s1s0 = s1s0.where(t.guard.reindex(H5_POP).fillna(False).astype(bool))
    rule = pd.DataFrame.from_dict({"lead S1-S0": signed_rank(s1s0), "share difference outside band": signed_rank(
        t["diff_out"].where(t.guard).reindex(H5_POP))}, orient="index").rename_axis("quantity").reset_index()
    if h5["verdict"] != "reject":
        return "not rejected, no reading", rule
    ok = [(r.p < 0.05) and np.sign(r.est) == np.sign(h5["est"]) for r in rule.itertuples()]
    if all(ok):
        return "a difference in adaptation", rule
    named = [w for w, good in zip(("a difference in oracle level (the lead in S1 - S0 does not carry the sign)",
                                   "extrapolation across the moving boundary (the share difference outside the band does not)"),
                                  ok) if not good]
    return " and ".join(named), rule


def s1_lead_minus_control(S: pd.DataFrame, sc: str) -> pd.Series:
    """Per dataset: the lead in ROC AUC at T4 under S1, arm minus control (exploratory)."""
    return (lead(cell(S, "auc", sc, "S1", 4)) - lead(cell(S, "auc", "none", "S1", 4))).reindex(STUDY_B)


def extras(S: pd.DataFrame, v: dict[str, pd.Series], ratio: dict) -> pd.DataFrame:
    """Tests reported next to the primaries, with no verdict: each prior level alone, the lead under S1
    (covariate and region arms) minus the control's, and the change of the log error ratio."""
    rows = {"H3, prior 0.5 alone": signed_rank(v["H3 prior0.5"]), "H3, prior 2 alone": signed_rank(v["H3 prior2"]),
            "lead in S1, covariate minus control": signed_rank(s1_lead_minus_control(S, "cov")),
            "lead in S1, region minus control": signed_rank(s1_lead_minus_control(S, "region")),
            "log error ratio, covariate minus control": ratio}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("quantity").reset_index()


def control_diagnostic(S: pd.DataFrame) -> pd.DataFrame:
    """Check 6: the lead's change from T1 to T4 under the control, S0, 99 % Hodges-Lehmann interval."""
    rows = []
    for m in ("auc", "logloss"):
        v = (lead(cell(S, m, "none", "S0", 4)) - lead(cell(S, m, "none", "S0", 1))).reindex(STUDY_B)
        r = signed_rank(v, level=0.99)
        rows.append(dict(metric=m, **r, excludes_zero=bool(r["lo"] > 0 or r["hi"] < 0)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- checks 2 to 4

def expected_cells(ds: str) -> tuple[int, int]:
    """Stored cells of a run (strategies S0, S1, S2 and oracle) and S0* cells (Study B only)."""
    if ds in STUDY_A:
        return 10, 0
    if ds == NO_TOP:
        return 4 * 10, 8
    return (8, 2) if ds in CONCEPT_DATASETS else (7 * 10 + 3, 14)


def checks(S: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Checks 2 to 4 of plan section 6: the table, and the cells behind every line that is not a pass."""
    rows = []
    star = S.strategy == "S0*"
    runs = S[~star].groupby(["dataset", "model", "seed"]).size().rename("found").to_frame()
    runs["found_star"] = S[star].groupby(["dataset", "model", "seed"]).size().reindex(runs.index).fillna(0)
    exp = runs.index.get_level_values("dataset").map(expected_cells)
    runs["expected"], runs["expected_star"] = [e[0] for e in exp], [e[1] for e in exp]
    n_runs = sum(len(seeds(d)) for d in STUDY_B + CONCEPT_DATASETS + PUBLIC_A) * 7
    short = runs[(runs.found != runs.expected) | (runs.found_star != runs.expected_star)]
    score_cols = ["auc", "ap", "logloss", "brier", "ece"]
    missing = S[S[score_cols].isna().any(axis=1)][["dataset", "model", "seed", "scenario", "strategy", "block"]]
    const = S[(S.n_unique == 1) & ~star].groupby(["dataset", "model"]).size().rename("cells").reset_index()
    rows += [("2 runs found", f"{len(runs)} of {n_runs}", "pass" if len(runs) == n_runs else "FAIL"),
             ("2 runs with a missing or extra cell", str(len(short)), "pass" if short.empty else "FAIL"),
             ("2 cells with a missing score", str(len(missing)), "pass" if missing.empty else "FAIL"),
             ("2 non-finite or out-of-range probabilities", str(int((~S.finite).sum())), "pass" if S.finite.all() else "FAIL"),
             ("2 cells with one distinct prediction (a constant model)", f"{int(const.cells.sum())} in {len(const)} (dataset, model) pairs",
              "pass" if const.empty else "REVIEW")]
    cells = S[S.scenario.isin(["none", "cov", *PRIORS]) & ~star]
    by_mean = cells.groupby(["study", "dataset", "model", "scenario", "strategy", "block"])[["ap", "prevalence"]].mean()
    near = by_mean[(by_mean.ap - by_mean.prevalence) < 0.01]
    per_seed = cells[(cells.ap - cells.prevalence) < 0.01]
    rows.append(("3 cells within 0.01 of their AP floor, concept arms exempt: mean over seeds | single seeds",
                 f"{len(near)} of {len(by_mean)} | {len(per_seed)} of {len(cells)}", "pass" if near.empty and per_seed.empty else "REVIEW"))
    prev = S[(S.study == "B") & (S.scenario == "none") & ~star].groupby("dataset").prevalence.mean()
    rows.append(("4 control prevalence per Study B dataset at most 0.51 (label 1 is the minority class)",
                 f"max {prev.max():.3f}", "pass" if prev.max() <= 0.51 else "FAIL"))
    first = S[(S.strategy == "S0") & (S.block == 1) & ((S.scenario == "none") | S.dataset.isin(CONCEPT_DATASETS))]
    auc = first.groupby(["study", "dataset", "model"]).auc.mean()
    rows.append(("4 mean S0 ROC AUC at T1 below 0.5 (control; region arm for the concept-only datasets)",
                 f"{int((auc < 0.5).sum())} of {len(auc)} (dataset, model)", "pass" if (auc >= 0.5).all() else "REVIEW"))
    extra = {"check2_missing": missing, "check2_constant": const, "check3_seed_means": near.reset_index(),
             "check3_single_seeds": per_seed[["study", "dataset", "model", "seed", "scenario", "strategy", "block", "ap", "prevalence"]],
             "check4_auc_below_half": auc[auc < 0.5].reset_index()}
    return pd.DataFrame(rows, columns=["check", "result", "status"]), extra


# ---------------------------------------------------------------- exploratory tables

METRICS = ("auc", "ap", "skill", "logloss", "brier", "ece")


def leads_table(S: pd.DataFrame) -> pd.DataFrame:
    """Family means and the lead, per dataset, scenario, strategy and block, for every metric."""
    parts = []
    for m in METRICS:
        w = S.pivot(index=["dataset", "seed", "scenario", "strategy", "block"], columns="model", values=m)
        t = w[list(TFMS)].mean(axis=1, skipna=False)
        b = w[list(BOOSTERS)].mean(axis=1, skipna=False)
        g = pd.DataFrame(dict(tfm=t, booster=b, lead=t - b)).groupby(["dataset", "scenario", "strategy", "block"])
        mean, n = g.mean(), g["lead"].count()
        mean[n.to_numpy() != [len(seeds(d)) for d in n.index.get_level_values("dataset")]] = np.nan
        parts.append(mean.assign(metric=m))
    return pd.concat(parts).reset_index()


def dose_slopes(S: pd.DataFrame, leads: pd.DataFrame) -> pd.DataFrame:
    """Slope over T1 to T4 of (the S0 lead under an arm minus the control's lead in the same block) against
    (the arm's realized dose minus the control's in the same units), mean over seeds. Dose: mean z^2
    (covariate), log odds of the prevalence (prior), share of relabeled rows (concept arms). dose_t4 is the
    dose at T4 and gives the orientation: the prior 0.5 arm lowers it, the prior 2 arm raises it."""
    b = S[S.strategy == "S0"].groupby(["dataset", "scenario", "block"])[["z2", "prevalence", "relabeled"]].mean()
    units = {"cov": b.z2, **{r: logit(b.prevalence) for r in PRIORS}, **{a: b.relabeled for a in ARMS}}
    rows = []
    for m in ("auc", "logloss"):
        lead_s0 = leads[(leads.strategy == "S0") & (leads.metric == m)].set_index(["dataset", "scenario", "block"]).lead
        for sc, dz in units.items():
            d = pd.DataFrame({"lead": lead_s0.xs(sc, level="scenario") - lead_s0.xs("none", level="scenario"),
                              "dose": dz.xs(sc, level="scenario") - dz.xs("none", level="scenario")}).dropna().reset_index()
            for ds, g in d.groupby("dataset"):
                rows.append(dict(dataset=ds, scenario=sc, metric=m, slope=np.polyfit(g.dose, g.lead, 1)[0],
                                 dose_t4=g.dose[g.block == 4].iloc[0]))
    return pd.DataFrame(rows)


def static_table(S: pd.DataFrame) -> pd.DataFrame:
    """The static model at T4 under the control and every arm, family means; for the prior arms next to the log loss
    reconstructed from the control's class-conditional losses and the block's prevalence."""
    s0 = S[(S.strategy == "S0") & (S.block == 4)]
    ctl = s0[s0.scenario == "none"].set_index(["dataset", "seed", "model"])[["ll_pos", "ll_neg"]].add_suffix("_ctl")
    d = s0.join(ctl, on=["dataset", "seed", "model"])
    d["reconstructed"] = np.where(d.scenario.isin(PRIORS), d.prevalence * d.ll_pos_ctl + (1 - d.prevalence) * d.ll_neg_ctl, np.nan)
    d["family"] = d.model.map({m: f for f, ms in FAMILIES for m in ms})
    return d.dropna(subset=["family"]).groupby(["dataset", "family", "scenario"])[["auc", "logloss", "reconstructed"]].mean().reset_index()


def calibration_table(S: pd.DataFrame) -> pd.DataFrame:
    d = S[S.cal_slope.notna()].copy()
    d["family"] = d.model.map({m: f for f, ms in FAMILIES for m in ms})
    return d.dropna(subset=["family"]).groupby(["dataset", "family"]).cal_slope.mean().unstack().reset_index()


def vs_benchmark(leads: pd.DataFrame) -> pd.DataFrame:
    """The benchmark's random-split lead in PR AUC (its six models) against this study's control lead in AP."""
    ckpt = {m: joblib.load(sdb.ROOT / "results" / "ckpt" / f"{m}.joblib") for f in BENCH.values() for m in f}
    lead_b = {ds: np.mean([np.mean(ckpt[m][ds]["scores"]) for m in BENCH["tfm"]])
              - np.mean([np.mean(ckpt[m][ds]["scores"]) for m in BENCH["booster"]]) for ds in STUDY_B}
    rows = []
    for h in (1, 2, 3, 4):
        d = leads[(leads.scenario == "none") & (leads.strategy == "S0") & (leads.block == h) & (leads.metric == "ap")
                  ].set_index("dataset").lead.reindex(STUDY_B)
        lb = pd.Series(lead_b).reindex(STUDY_B)
        rho, rho2 = spearmanr(lb, d), spearmanr(lb.drop("banana"), d.drop("banana"))  # banana: the benchmark's XGBoost artifact (plan 4.4)
        rows.append(dict(block=h, spearman=rho.statistic, p=rho.pvalue, spearman_without_banana=rho2.statistic,
                         p_without_banana=rho2.pvalue, same_sign=int((np.sign(lb) == np.sign(d)).sum())))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Study A

def resample(rng: np.random.Generator, n: int, L: int | None) -> np.ndarray:
    """Row indices of one bootstrap draw: independent rows (L = None), or a circular moving-block draw of
    ceil(n / L) runs of L consecutive rows, cut to n. L = 1 makes the same draw as L = None."""
    if L is None:
        return rng.integers(n, size=n)
    starts = rng.integers(n, size=-(-n // L))
    return ((starts[:, None] + np.arange(L)) % n).ravel()[:n]


def bootstrap_dataset(ds: str, draws: int = 1000, block: bool = False, L: int | None = None) -> list[dict]:
    """Curves of Study A with percentile intervals from resampling each block's rows; every resample is
    shared by all models, seeds and strategies. Per-model values are seed means; the families are
    means of the three models, and the S2 TFM mean is also given without TabFM (context cap).
    With block = True the rows, which are in time order, are resampled in circular blocks of length L
    (None: round(n ** (1 / 3)) of the test block) and the output has the column L."""
    runs = [joblib.load(p) for p in sorted((RESULTS / ds).glob("*.joblib"))]
    models = sorted({r["model"] for r in runs})
    out = []
    for h in (1, 2, 3, 4):
        cols = [(r["model"], r["seed"], st) for r in runs for st in ("S0", "S1", "S2") if ("none", st, h) in r["preds"]]
        by = {(r["model"], r["seed"]): r for r in runs}
        P = np.column_stack([by[m, s]["preds"]["none", st, h].astype(np.float64) for m, s, st in cols])
        y = next(iter(by.values()))["labels"]["none", "S0", h]
        rows0 = next(iter(by.values()))["rows"]["none", "S0", h]
        assert all(np.array_equal(r["rows"]["none", st, h], rows0) and np.array_equal(r["labels"]["none", st, h], y)
                   for r in runs for st in ("S0", "S1", "S2") if ("none", st, h) in r["rows"])
        rng = np.random.default_rng([STUDY_A.index(ds), h])
        run_L = (L or round(len(y) ** (1 / 3))) if block else None

        def score(idx):
            yb, Q = y[idx][:, None], P[idx]
            k = yb.sum()
            auc = (rankdata(Q, axis=0)[yb[:, 0] == 1].sum(axis=0) - k * (k + 1) / 2) / (k * (len(idx) - k))
            Q = np.clip(Q, CLIP, 1 - CLIP)
            return {"auc": auc, "logloss": -(yb * np.log(Q) + (1 - yb) * np.log(1 - Q)).mean(axis=0)}

        full = score(np.arange(len(y)))
        boot = [score(resample(rng, len(y), run_L)) for _ in range(draws)]
        for metric in ("auc", "logloss"):
            arr = np.stack([full[metric]] + [b[metric] for b in boot])           # (1 + draws, K)
            u = {}
            for st in ("S0", "S1", "S2"):
                have = {m: [i for i, (mm, s, ss) in enumerate(cols) if mm == m and ss == st] for m in models}
                if not have[models[0]]:
                    continue
                per = {m: arr[:, have[m]].mean(axis=1) for m in models}
                fam = {f: np.mean([per[m] for m in ms], axis=0) for f, ms in FAMILIES}
                fam["tfm_no_tabfm"] = np.mean([per["tabpfn35"], per["tabicl"]], axis=0)
                u[st] = {**per, **fam, "lead": fam["tfm"] - fam["booster"], "lead_no_tabfm": fam["tfm_no_tabfm"] - fam["booster"]}
            series = {st: u[st] for st in u} | {f"{st}-S0": {k: u[st][k] - u["S0"][k] for k in u[st]} for st in u if st != "S0"}
            for name, units in series.items():
                for unit, a in units.items():
                    out.append(dict(dataset=ds, block=h, strategy=name, unit=unit, metric=metric, est=a[0],
                                    lo=np.percentile(a[1:], 2.5), hi=np.percentile(a[1:], 97.5),
                                    **({"L": run_L} if block else {})))
    return out


def bootstrap_dataset_block(ds: str, L: int | None, draws: int = 1000) -> list[dict]:
    """Exploratory: bootstrap_dataset with circular moving blocks of L rows (None: round(n ** (1 / 3)))."""
    return bootstrap_dataset(ds, draws, block=True, L=L)


def block_job(item: tuple[str, int | None]) -> list[dict]:
    return bootstrap_dataset_block(*item)


def study_a() -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = pd.DataFrame([r for part in parallel("bootstrap_dataset", PUBLIC_A) for r in part])
    contrast = curves.strategy.str.contains("-")
    c = curves[(contrast & curves.unit.isin(["tfm", "booster", "lead", "lead_no_tabfm"]))
               | (~contrast & curves.unit.isin(["lead", "lead_no_tabfm"]))]
    signs = c.assign(positive=c.est > 0).groupby(["strategy", "block", "unit", "metric"]).positive.agg(["sum", "count"]).reset_index()
    signs["positive_means"] = signs.metric.map({"auc": "higher ROC AUC", "logloss": "higher log loss (worse)"})
    return curves, signs


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks-reviewed", action="store_true",
                    help="continue to the tests although a check is not a pass (plan 6: checks before any result is read)")
    S = score_all()
    S.to_csv(OUT / "drift_scores.csv", index=False)
    SB = S[S.study == "B"]
    chk, extra = checks(S)
    diag = control_diagnostic(SB)
    v = primary_values(SB)
    s1_cov = s1_lead_minus_control(SB, "cov")
    ident = float(np.nanmax(np.abs(s1_cov - v["H1"] - v["H2"])))
    chk.loc[len(chk)] = ("6 control diagnostic: 99 % interval of the T4 minus T1 lead excludes 0 for an outcome",
                         ", ".join(f"{r.metric} [{r.lo:.4f}, {r.hi:.4f}]" for r in diag.itertuples()),
                         "REVIEW" if diag.excludes_zero.any() else "pass")
    chk.loc[len(chk)] = ("identity: H2 = (lead in S1, covariate minus control) - H1, largest gap", f"{ident:.1e}",
                         "pass" if ident < 1e-9 else "FAIL")
    chk.to_csv(OUT / "drift_checks.csv", index=False)
    diag.to_csv(OUT / "drift_control_diagnostic.csv", index=False)
    for name, df in extra.items():
        df.to_csv(OUT / f"drift_{name}.csv", index=False)
    print(chk.to_string(index=False))
    if (chk.status != "pass").any() and not ap.parse_args().checks_reviewed:
        sys.exit("\nnot every check is a pass: read drift_checks.csv and drift_check*.csv, then rerun with --checks-reviewed")

    pd.DataFrame(v).rename_axis("dataset").reset_index().to_csv(OUT / "drift_values.csv", index=False)
    t = primary_tests(SB, v)
    ratio_t, ratio = companions_h1_h4(SB)
    ratio_t.to_csv(OUT / "drift_h1_h4_companions.csv", index=False)
    h1 = t.set_index("test").loc["H1"].to_dict()
    h5 = t.set_index("test").loc["H5"].to_dict()
    read5, rule = reading_h5(SB, h5)
    t["reading"] = t.test.map({"H1": reading_h1(h1, ratio), "H5": read5})
    t.to_csv(OUT / "drift_primary.csv", index=False)
    rule.to_csv(OUT / "drift_h5_reading_rule.csv", index=False)
    leave_one_out(v).to_csv(OUT / "drift_loo.csv", index=False)
    extras(SB, v, ratio).to_csv(OUT / "drift_extras.csv", index=False)
    print("\n", t[["test", "n_datasets", "n_nonzero", "est", "lo", "hi", "p", "holm_p", "verdict", "reading"]].round(4).to_string(index=False))
    print("\nH5 reading rule:\n", rule.round(4).to_string(index=False))

    concept_table(SB).to_csv(OUT / "drift_concept.csv", index=False)
    model_shares(SB).to_csv(OUT / "drift_h5_models.csv", index=False)
    leads = leads_table(S)
    leads.to_csv(OUT / "drift_leads.csv", index=False)
    dose_slopes(SB, leads).to_csv(OUT / "drift_dose_slopes.csv", index=False)
    static_table(SB).to_csv(OUT / "drift_static.csv", index=False)
    calibration_table(SB).to_csv(OUT / "drift_calibration.csv", index=False)
    vs_benchmark(leads).to_csv(OUT / "drift_vs_benchmark.csv", index=False)
    S.groupby(["study", "dataset", "model", "scenario", "strategy", "block"])[list(METRICS)].mean().reset_index().to_csv(
        OUT / "drift_per_model.csv", index=False)
    curves, signs = study_a()
    curves.to_csv(OUT / "drift_study_a_curves.csv", index=False)
    signs.to_csv(OUT / "drift_study_a_signs.csv", index=False)
    blocks = [r for part in parallel("block_job", [(ds, L) for ds in PUBLIC_A for L in (None, 100)]) for r in part]
    pd.DataFrame(blocks).to_csv(OUT / "drift_study_a_curves_block.csv", index=False)


if __name__ == "__main__":
    main()
