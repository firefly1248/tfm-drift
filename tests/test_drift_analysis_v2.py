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
