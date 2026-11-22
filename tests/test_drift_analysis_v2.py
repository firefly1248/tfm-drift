import pytest

from scripts.drift_analysis_v2 import refresh_rows


def times(scenario: str) -> list[dict]:
    tag = f"{scenario} " if scenario else ""
    out = [dict(stage=f"tune S2_{h}", rows=100 * h, seconds=10.0 * h) for h in (2, 3, 4)]
    for s in ("S1", "S2"):
        for h in (2, 3, 4):
            out += [dict(stage=f"fit {s} {tag}{h}", rows=500, seconds=1.0), dict(stage=f"predict {s} {tag}{h}", rows=250, seconds=0.5)]
    return out


@pytest.mark.parametrize("scenario", ["none", ""])
def test_refresh_is_fit_plus_predict_and_s2_adds_the_retune(scenario):
    r = refresh_rows("catboost", times(scenario), scenario)
    assert len(r) == 6
    s1 = next(x for x in r if x["strategy"] == "S1" and x["block"] == 3)
    s2 = next(x for x in r if x["strategy"] == "S2" and x["block"] == 3)
    assert s1["refresh"] == 1.5 and s2["refresh"] == 1.5 + 30.0 and s1["device"] == "cpu"


def test_tabfm_is_on_mps():
    assert refresh_rows("tabfm", times("none"))[0]["device"] == "mps"


def test_one_sided_p_matches_scipy_and_tost_is_the_larger_shifted_p():
    import numpy as np
    import pandas as pd
    from scipy.stats import wilcoxon

    from scripts.drift_analysis_v2 import equivalence_family, one_sided_p, tost
    rng = np.random.default_rng(0)
    d = rng.normal(0.4, 1, 14)
    assert np.isclose(one_sided_p(d), wilcoxon(d, alternative="greater", method="exact").pvalue)
    v = pd.Series(rng.normal(0, 0.002, 25))
    r = tost(v, 0.005)
    assert np.isclose(r["p_lower"], wilcoxon(v + 0.005, alternative="greater", method="exact").pvalue)
    assert np.isclose(r["p_upper"], wilcoxon(0.005 - v, alternative="greater", method="exact").pvalue)
    assert r["p"] == max(r["p_lower"], r["p_upper"])
    fam = equivalence_family({"E1": v, "E2": v + 0.01, "E3": v.iloc[:10]}, {"E1": 0.005, "E2": 0.005, "E3": 0.009})
    f = fam.set_index("test")
    assert f.loc["E1", "verdict"] == "equivalent" and f.loc["E2", "verdict"] == "no equivalence shown"
    assert f.loc["E3", "verdict"] == "underpowered" and np.isnan(f.loc["E3", "holm_p"])


# ---------------------------------------------------------------- v2 families on synthetic scores

def _scores(study: str, datasets: list[str], seeds: int, arms_strategies, value) :
    import pandas as pd

    from scripts.drift_run import BOOSTERS, TFMS
    rows = []
    for d in datasets:
        for s in range(seeds):
            for m in (*TFMS, *BOOSTERS):
                for arm, st, h in arms_strategies:
                    a, ll = value(d, m, arm, st, h)
                    rows.append(dict(dataset=d, seed=s, model=m, scenario=arm, strategy=st, block=h, auc=a, logloss=ll))
    return pd.DataFrame(rows)


def test_b2_values_compose_e_and_x_from_leads():
    import numpy as np

    from scripts.drift_analysis_v2 import b2_values
    from scripts.drift_run import TFMS
    keys = [(a, s, 4) for a in ("none", "cov", "xup", "xdown") for s in ("S0", "S1")] + [(a, "U", 4) for a in ("xup", "xdown")] \
        + [(a, s, 1) for a in ("none",) for s in ("S0",)]

    def value(d, m, arm, st, h):
        tfm = m in TFMS
        auc = 0.70 + (0.03 if tfm else 0.0) + {"cov": 0.01 if tfm else 0.02, "xup": 0.0, "xdown": 0.0}.get(arm, 0.0)
        auc += {"S1": 0.01, "U": 0.02 if tfm else 0.01}.get(st, 0.0) * (1 if arm != "none" else 1)
        return auc, 0.5 - (auc - 0.7)

    S = _scores("b2", ["a", "b"], 2, keys, value)
    v = b2_values(S, {"a": 2, "b": 2})
    # lead under S0: none 0.03, cov 0.03 + 0.01 - 0.02 = 0.02, so E1 = -0.01
    assert np.allclose(v["E1"], -0.01)
    # S1 adds 0.01 to every model: the refit gain has lead 0 everywhere, so E2 = 0
    assert np.allclose(v["E2"], 0.0)
    # xup: S0 lead is 0.03; U adds 0.02 (TFM) and 0.01 (booster), so the U lead is 0.04 and X1 = 0.03 - 0.04
    assert np.allclose(v["X1"], -0.01) and np.allclose(v["X1_up"], v["X1_down"])
    assert np.allclose(v["tfm_change_auc"], 0.01) and np.allclose(v["booster_change_auc"], 0.02)


def test_x_and_c_families_apply_their_minimums_and_holm():
    import numpy as np
    import pandas as pd

    from scripts.drift_analysis_v2 import c_family, x_family
    rng = np.random.default_rng(0)
    big = pd.DataFrame({"X1": rng.normal(0.02, 0.01, 22), "X2": rng.normal(0, 0.01, 22)})
    r = x_family(big).set_index("test")
    assert r.loc["X1", "verdict"] == "reject" and r.loc["X2", "verdict"] == "not rejected"
    assert x_family(big.iloc[:15]).verdict.eq("underpowered").all()
    v = pd.DataFrame({"C0": rng.normal(0.03, 0.01, 9), "C3": rng.normal(-0.02, 0.01, 9),
                      "C1": rng.normal(0, 0.01, 9), "C2": rng.normal(0, 0.01, 9)})
    c = c_family(v).set_index("test")
    assert c.loc["C0", "verdict"] == "reject" and c.loc["C3", "verdict"] == "reject"
    assert c.loc["C1", "verdict"] == "estimate only" and np.isnan(c.loc["C1", "p"])
    assert c_family(v.iloc[:6]).set_index("test").loc["C0", "verdict"] == "underpowered"


def test_checks_find_a_missing_cell_a_repeated_row_and_a_wrong_share(tmp_path, monkeypatch):
    import joblib
    import numpy as np
    import pandas as pd

    from scripts import drift_analysis_v2 as a2
    from scripts.drift_run import MODELS
    monkeypatch.setattr(a2, "RESULTS_V2", tmp_path)
    monkeypatch.setattr(a2, "population", lambda study: pd.DataFrame({"name": ["ds"], "seeds": [1]}))
    for m in MODELS:
        keys = a2.expected_keys(m, "b2")
        rows = {k: np.arange(10) + 10 * k[2] for k in keys}
        d = dict(preds={k: np.full(10, 0.5, dtype=np.float32) for k in keys}, rows=rows, labels={k: np.zeros(10, int) for k in keys},
                 extra=dict(mx=8, shares=[dict(arm="xup", block=h, share_z=round(8 * h / 4) / 8) for h in (1, 2, 3, 4)]))
        (tmp_path / "b2" / "ds").mkdir(parents=True, exist_ok=True)
        joblib.dump(d, tmp_path / "b2" / "ds" / f"{m}_seed0.joblib")
    ok = a2.checks_v2("b2").set_index("check")["count"]
    assert (ok == 0).all()
    d = joblib.load(tmp_path / "b2" / "ds" / "tabfm_seed0.joblib")
    del d["preds"][("none", "S1", 2)]
    d["rows"][("none", "S0", 2)] = d["rows"][("none", "S0", 1)]
    d["extra"]["shares"][0]["share_z"] = 0.0
    joblib.dump(d, tmp_path / "b2" / "ds" / "tabfm_seed0.joblib")
    bad = a2.checks_v2("b2").set_index("check")["count"]
    assert bad["cells missing"] == 1 and bad["block rows repeated within an arm"] == 10 and bad["extrapolation share differs from m_x h / 4"] == 1
