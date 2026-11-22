"""drift_analysis_v2.py: the analysis of plan_v2.md. Part 0 (cost of a refresh) reads the v1 checkpoints.

Usage: uv run python -m scripts.drift_analysis_v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from scripts.drift_analysis import holm, metrics, signed_rank, signed_rank_null
from scripts.drift_run import BOOSTERS, MODELS, RESULTS, TFMS
from scripts.drift_run_v2 import RESULTS as RESULTS_V2
from scripts.drift_run_v2 import population
from scripts.drift_windows import PUBLIC_A, STUDY_B

OUT = Path(__file__).resolve().parent.parent / "results" / "v2"
TFMS_C = ("tabpfn35", "tabicl")        # the TFMs that see the whole window at every size in Study C
MARGINS = {"E1": 0.005, "E2": 0.005, "E3": 0.009}
MIN_N = {"X": 20, "C": 8}
WINDOWS = (1_000, 5_000, 20_000)
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


# ---------------------------------------------------------------- scoring

def score_dataset_v2(study: str, ds: str) -> list[dict]:
    """One row per model, seed, arm, strategy and block of a dataset's v2 checkpoints."""
    rows = []
    for path in sorted((RESULTS_V2 / study / ds).glob("*.joblib")):
        d = joblib.load(path)
        for (arm, strategy, h), p in d["preds"].items():
            rows.append(dict(study=study, dataset=ds, model=d["model"], seed=d["seed"], scenario=arm, strategy=strategy,
                             block=h, **metrics(p, d["labels"][arm, strategy, h])))
    return rows


def score_study(study: str) -> pd.DataFrame:
    from scripts import drift_analysis_v2 as me
    parts = Parallel(n_jobs=4)(delayed(me.score_dataset_v2)(study, ds) for ds in population(study).name)
    return pd.DataFrame([r for part in parts for r in part])


def n_seeds(study: str) -> dict[str, int]:
    pop = population(study)
    return dict(zip(pop.name, pop.seeds.astype(int))) if study == "b2" else {d: 3 for d in pop.name}


def cell(S: pd.DataFrame, metric: str, arm: str, strategy: str, h: int) -> pd.Series:
    d = S[(S.scenario == arm) & (S.strategy == strategy) & (S.block == h)]
    return d.set_index(["dataset", "seed", "model"])[metric]


def famean(x: pd.Series, models, nseeds: dict[str, int]) -> pd.Series:
    """Per dataset: the mean over the family's models, then over seeds; missing if any cell or any seed is missing."""
    per = x.unstack("model")[list(models)].mean(axis=1, skipna=False)
    g = per.groupby("dataset")
    return g.mean().where(g.count() == pd.Series({d: nseeds[d] for d in g.count().index}))


def lead_v2(S, metric, arm, strategy, h, nseeds, tfms=TFMS) -> pd.Series:
    x = cell(S, metric, arm, strategy, h)
    return famean(x, tfms, nseeds) - famean(x, BOOSTERS, nseeds)


def b2_values(S: pd.DataFrame, nseeds: dict[str, int]) -> pd.DataFrame:
    """Per-dataset quantities of families E and X and their companions (plan_v2.md 7.1, 7.2)."""
    def L(metric, arm, strategy, h):
        return lead_v2(S, metric, arm, strategy, h, nseeds)

    def fam(metric, arm, models):
        return famean(cell(S, metric, arm, "S0", 4), models, nseeds)

    v = pd.DataFrame(index=sorted(S.dataset.unique()))
    v["E1"] = L("auc", "cov", "S0", 4) - L("auc", "none", "S0", 4)
    v["E2"] = (L("auc", "cov", "S1", 4) - L("auc", "cov", "S0", 4)) - (L("auc", "none", "S1", 4) - L("auc", "none", "S0", 4))
    v["E3"] = L("logloss", "cov", "S0", 4) - L("logloss", "none", "S0", 4)
    for metric, name in (("auc", "X1"), ("logloss", "X2")):
        for d in ("up", "down"):
            v[f"{name}_{d}"] = L(metric, f"x{d}", "S0", 4) - L(metric, f"x{d}", "U", 4)
        v[name] = (v[f"{name}_up"] + v[f"{name}_down"]) / 2
    v["control_lead_auc"], v["control_lead_logloss"] = L("auc", "none", "S0", 4), L("logloss", "none", "S0", 4)
    v["tfm_change_auc"] = fam("auc", "cov", TFMS) - fam("auc", "none", TFMS)
    v["booster_change_auc"] = fam("auc", "cov", BOOSTERS) - fam("auc", "none", BOOSTERS)
    v["refit_gain_tfm"] = famean(cell(S, "auc", "cov", "S1", 4) - cell(S, "auc", "cov", "S0", 4), TFMS, nseeds)
    v["refit_gain_booster"] = famean(cell(S, "auc", "cov", "S1", 4) - cell(S, "auc", "cov", "S0", 4), BOOSTERS, nseeds)

    def log_error_ratio(arm):
        t, b = fam("auc", arm, TFMS), fam("auc", arm, BOOSTERS)
        return np.log((1 - t) / (1 - b))
    v["log_error_ratio_change"] = log_error_ratio("cov") - log_error_ratio("none")
    return v


def c_values(S: pd.DataFrame, nseeds: dict[str, int]) -> pd.DataFrame:
    """Per-dataset quantities of family C (plan_v2.md 7.3) with the primary TFM family, and the three-model version."""
    def L(arm, strategy, tfms, metric="auc"):
        return lead_v2(S, metric, arm, strategy, 4, nseeds, tfms)

    v = pd.DataFrame(index=sorted(S.dataset.unique()))
    for fam_name, tfms in (("", TFMS_C), ("_three", TFMS)):
        for w in WINDOWS:
            v[f"lead{fam_name}_{w}"] = L("none", f"S0@{w}", tfms)
            v[f"tilt{fam_name}_{w}"] = L("cov", f"S0@{w}", tfms) - v[f"lead{fam_name}_{w}"]
            v[f"X1{fam_name}_{w}"] = sum(L(f"x{d}", f"S0@{w}", tfms) - L(f"x{d}", f"U@{w}", tfms) for d in ("up", "down")) / 2
        v[f"C0{fam_name}"] = v[f"lead{fam_name}_20000"]
        v[f"C3{fam_name}"] = v[f"lead{fam_name}_20000"] - v[f"lead{fam_name}_1000"]
        v[f"C1{fam_name}"] = v[f"tilt{fam_name}_20000"] - v[f"tilt{fam_name}_1000"]
        v[f"C2{fam_name}"] = v[f"X1{fam_name}_20000"] - v[f"X1{fam_name}_1000"]
    return v


# ---------------------------------------------------------------- tests

def x_family(v: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """X1 and X2: two-sided exact signed-rank tests, Hodges-Lehmann estimate and 95 % interval, Holm over the testable."""
    r = {k: signed_rank(v[k]) for k in ("X1", "X2")}
    ok = {k: x["p"] for k, x in r.items() if x["n_nonzero"] >= MIN_N["X"]}
    adj = holm(ok) if ok else {}
    return pd.DataFrame([dict(test=k, n=x["n_nonzero"], est=x["est"], lo=x["lo"], hi=x["hi"], p=x["p"], holm_p=adj.get(k, np.nan),
                              verdict="underpowered" if k not in ok else ("reject" if adj[k] < alpha else "not rejected"))
                         for k, x in r.items()])


def c_family(v: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """C0 (one-sided: the control lead at 20,000 is above zero) and C3 (two-sided), Holm over both; C1 and C2 are estimates."""
    rows, ps = [], {}
    for k in ("C0", "C3"):
        x = signed_rank(v[k])
        p = one_sided_p(v[k].dropna().to_numpy(dtype=float)) if k == "C0" else x["p"]
        rows.append(dict(test=k, n=x["n_nonzero"], est=x["est"], lo=x["lo"], hi=x["hi"], p=p))
        if x["n_nonzero"] >= MIN_N["C"]:
            ps[k] = p
    adj = holm(ps) if ps else {}
    out = pd.DataFrame(rows).assign(holm_p=lambda d: d.test.map(adj))
    out["verdict"] = np.where(~out.test.isin(ps), "underpowered", np.where(out.holm_p < alpha, "reject", "not rejected"))
    for k in ("C1", "C2"):
        x = signed_rank(v[k])
        out.loc[len(out)] = dict(test=k, n=x["n_nonzero"], est=x["est"], lo=x["lo"], hi=x["hi"], p=np.nan, holm_p=np.nan, verdict="estimate only")
    return out


def bootstrap_mean(x: pd.Series, reps: int = 10_000) -> tuple[float, float, float]:
    a = x.dropna().to_numpy()
    m = np.random.default_rng(0).choice(a, (reps, len(a))).mean(axis=1)
    return float(a.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def e_companions(v: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    """Reported next to E1 to E3, no verdict: datasets outside the margin, the mean change with a bootstrap interval,
    each family's own change from control to tilt, the control lead and each margin's share of it, the log error ratio,
    and the sensitivity of E1 to the datasets left out."""
    rows = []
    for k, m in MARGINS.items():
        mean, lo, hi = bootstrap_mean(v[k])
        lead = v["control_lead_auc" if k != "E3" else "control_lead_logloss"].abs().median()
        rows.append(dict(item=f"{k} outside margin", value=int((v[k].abs() > m).sum()), n=int(v[k].notna().sum())))
        rows.append(dict(item=f"{k} mean change", value=mean, lo=lo, hi=hi))
        rows.append(dict(item=f"{k} margin as share of the median control lead", value=m / lead))
    for name in ("tfm_change_auc", "booster_change_auc", "refit_gain_tfm", "refit_gain_booster", "log_error_ratio_change"):
        r = signed_rank(v[name])
        rows.append(dict(item=name, value=r["est"], lo=r["lo"], hi=r["hi"], n=r["n_nonzero"]))
    g = pop.set_index("name")
    subsets = {"without TabArena": ~g.source.str.contains("TabArena"), "without reserve datasets": ~g.reserve.astype(bool),
               "without low-signal": ~g.low_signal.astype(bool), "without unresolved terms": g.lic != "unresolved"}
    for label, keep in subsets.items():
        names = [d for d in v.index if keep.get(d, True)]
        rows.append(dict(item=f"E1 {label}", **{k: x for k, x in tost(v.loc[names, "E1"], MARGINS["E1"]).items() if k in ("n", "p")}))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- checks

def expected_keys(model: str, study: str) -> set[tuple]:
    if study == "c":
        out = {(a, f"S0@{w}", h) for a in ("none", "cov") for w in WINDOWS for h in (1, 2, 3, 4)}
        return out | {(a, f"{s}@{w}", h) for a in ("xup", "xdown") for s in ("S0", "U") for w in WINDOWS for h in (1, 2, 3, 4)}
    out = {(a, "S0", h) for a in ("none", "cov", "xup", "xdown") for h in (1, 2, 3, 4)}
    out |= {(a, s, h) for a in ("none", "cov", "xup", "xdown") for s in ("S1", "S2") for h in (2, 3, 4)}
    out |= {(a, "U", h) for a in ("xup", "xdown") for h in (1, 2, 3, 4)}
    if model in BOOSTERS:
        out |= {(a, "S1r", h) for a in ("none", "cov", "xup", "xdown") for h in (2, 3, 4)}
    return out


def checks_v2(study: str) -> pd.DataFrame:
    """Plan 9: every cell present and finite, block rows disjoint within an arm, the extrapolation shares as built,
    and the block rows of S0 and U equal on the extrapolation arms. One row per check."""
    missing = nonfinite = overlap = share = rows_differ = files = 0
    for ds in population(study).name:
        for model in MODELS:
            for seed in range(n_seeds(study)[ds]):
                path = RESULTS_V2 / study / ds / f"{model}_seed{seed}.joblib"
                if not path.exists():
                    files += 1
                    continue
                d = joblib.load(path)
                missing += len(expected_keys(model, study) - set(d["preds"]))
                nonfinite += sum(not np.isfinite(p).all() for p in d["preds"].values())
                s0_names = ["S0"] if study == "b2" else [f"S0@{w}" for w in WINDOWS]
                for arm in ("none", "cov", "xup", "xdown"):
                    for s0 in s0_names:
                        rows = [d["rows"][(arm, s0, h)] for h in (1, 2, 3, 4) if (arm, s0, h) in d["rows"]]
                        overlap += len(np.concatenate(rows)) - len(np.unique(np.concatenate(rows))) if rows else 0
                for arm in ("xup", "xdown"):
                    for h in (1, 2, 3, 4):
                        for s0, u in zip(s0_names, ["U"] if study == "b2" else [f"U@{w}" for w in WINDOWS]):
                            if (arm, u, h) in d["rows"]:
                                rows_differ += not np.array_equal(d["rows"][(arm, u, h)], d["rows"][(arm, s0, h)])
                if study == "b2" and model == MODELS[0]:
                    mx = d["extra"]["mx"]
                    for sh in d["extra"]["shares"]:
                        want = round(mx * sh["block"] / 4) / mx
                        share += abs(sh["share_z"] - want) > 1e-9
    return pd.DataFrame([dict(check="checkpoint files missing", count=files), dict(check="cells missing", count=missing),
                         dict(check="non-finite probabilities", count=nonfinite), dict(check="block rows repeated within an arm", count=overlap),
                         dict(check="extrapolation share differs from m_x h / 4", count=int(share)),
                         dict(check="U and S0 score different rows", count=rows_differ)])


# ---------------------------------------------------------------- secondary, control diagnostic, main

def refit_gain_lead(S: pd.DataFrame, arm: str, nseeds: dict[str, int], refit: str) -> pd.Series:
    """Lead in the refit gain at T4: TFMs' (S1 - S0) minus boosters' (refit - S0), refit S1 or S1r."""
    s0 = cell(S, "auc", arm, "S0", 4)
    tfm = famean(cell(S, "auc", arm, "S1", 4) - s0, TFMS, nseeds)
    booster = famean(cell(S, "auc", arm, refit, 4) - s0, BOOSTERS, nseeds)
    return tfm - booster


def s1r_values(S: pd.DataFrame, nseeds: dict[str, int]) -> pd.DataFrame:
    v = pd.DataFrame(index=sorted(S.dataset.unique()))
    for refit in ("S1", "S1r"):
        for arm in ("none", "cov", "xup", "xdown"):
            v[f"{refit}_{arm}"] = refit_gain_lead(S, arm, nseeds, refit)
        v[f"E2_{refit}"] = v[f"{refit}_cov"] - v[f"{refit}_none"]
    return v


def control_diagnostic(S: pd.DataFrame, nseeds: dict[str, int], strategy: str = "S0") -> pd.DataFrame:
    """Under the control, the 99 % Hodges-Lehmann interval over datasets of the lead's change from T1 to T4, per outcome."""
    rows = []
    for metric in ("auc", "logloss"):
        change = lead_v2(S, metric, "none", strategy, 4, nseeds) - lead_v2(S, metric, "none", strategy, 1, nseeds)
        r = signed_rank(change, level=0.99)
        rows.append(dict(strategy=strategy, metric=metric, n=r["n_nonzero"], est=r["est"], lo=r["lo"], hi=r["hi"],
                         excludes_zero=bool(r["lo"] > 0 or r["hi"] < 0)))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks-reviewed", action="store_true", help="continue after the checks have been read")
    ap.add_argument("--part0", action="store_true", help="only the cost of a refresh, from v1's checkpoints")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.part0:
        cost = refresh_cost()
        cost.to_csv(OUT / "refresh_cost.csv", index=False)
        print(cost.round(3).to_string(index=False))
        return
    checks = {study: checks_v2(study) for study in ("b2", "c")}
    for study, c in checks.items():
        c.to_csv(OUT / f"checks_{study}.csv", index=False)
        print(f"checks, study {study}:\n{c.to_string(index=False)}")
    if any((c["count"] > 0).any() for c in checks.values()) and not a.checks_reviewed:
        sys.exit("a check failed: read the tables in results/v2/checks_*.csv, then rerun with --checks-reviewed")
    Sb, Sc = score_study("b2"), score_study("c")
    nb, nc = n_seeds("b2"), n_seeds("c")
    Sb.to_csv(OUT / "scores_b2.csv", index=False)
    Sc.to_csv(OUT / "scores_c.csv", index=False)
    vb, vc = b2_values(Sb, nb), c_values(Sc, nc)
    vb.to_csv(OUT / "values_b2.csv")
    vc.to_csv(OUT / "values_c.csv")
    pop = population("b2")
    diag = pd.concat([control_diagnostic(Sb, nb).assign(study="b2")] + [
        control_diagnostic(Sc, nc, f"S0@{w}").assign(study="c") for w in WINDOWS])
    diag.to_csv(OUT / "control_diagnostic.csv", index=False)
    if diag.excludes_zero.any():
        print(f"control diagnostic excludes zero:\n{diag[diag.excludes_zero].to_string(index=False)}")
    equivalence_family({k: vb[k] for k in MARGINS}, MARGINS).to_csv(OUT / "e_family.csv", index=False)
    x_family(vb).to_csv(OUT / "x_family.csv", index=False)
    c_family(vc).to_csv(OUT / "c_family.csv", index=False)
    e_companions(vb, pop).to_csv(OUT / "e_companions.csv", index=False)
    s1r_values(Sb, nb).to_csv(OUT / "s1r_values.csv")
    for name in ("e_family", "x_family", "c_family"):
        print(f"{name}:\n{pd.read_csv(OUT / f'{name}.csv').round(4).to_string(index=False)}")


if __name__ == "__main__":
    main()
