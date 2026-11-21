import numpy as np
import pandas as pd
import pytest
from scipy.stats import wilcoxon

from scripts.drift_analysis import (
    BOOSTERS,
    TFMS,
    expected_cells,
    famean,
    holm,
    lead,
    metrics,
    primary_tests,
    primary_values,
    reading_h1,
    s0_star,
    shares,
    signed_rank,
    signed_rank_null,
)

DATASET = "banana"  # a Study B name, which has three seeds


def test_signed_rank_null_counts_subsets_by_sum():
    assert signed_rank_null(3).tolist() == [1, 1, 1, 2, 1, 1, 1]


def test_signed_rank_matches_scipy_and_the_walsh_quantiles():
    d = np.array([0.3, -0.1, 0.8, 1.2, 0.5, -0.4, 0.9, 1.5, 0.2, 0.7, 0.6, 1.1])
    r = signed_rank(pd.Series(d))
    assert np.isclose(r["p"], wilcoxon(d, method="exact").pvalue)
    n = len(d)
    walsh = np.sort(((d[:, None] + d[None, :]) / 2)[np.triu_indices(n)])
    assert np.isclose(r["est"], np.median(walsh))
    k = 13  # n = 12, two-sided 5 %: reject at W <= 13
    assert (r["lo"], r["hi"]) == (walsh[k], walsh[-1 - k])


def test_signed_rank_99_percent_interval_uses_the_exact_quantile():
    d = np.arange(1.0, 11.0) * np.array([1, -1] * 5)
    walsh = np.sort(((d[:, None] + d[None, :]) / 2)[np.triu_indices(10)])
    r = signed_rank(pd.Series(d), level=0.99)
    k = 3  # n = 10: P(W <= 3) = 5 / 1024 <= 0.005 < P(W <= 4)
    assert (r["lo"], r["hi"]) == (walsh[k], walsh[-1 - k])


def test_signed_rank_interval_covers_zero_at_the_nominal_rate():
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(5000):
        r = signed_rank(pd.Series(rng.standard_normal(10)))
        hits += r["lo"] <= 0 <= r["hi"]
    assert abs(hits / 5000 - 0.951) < 0.015  # exact level at n = 10: 1 - 2 * P(W <= 8)


def test_signed_rank_drops_zeros_and_missing_and_refuses_ties():
    r = signed_rank(pd.Series([0.0, np.nan, 1.0, 2.0, 3.0]))
    assert (r["n_datasets"], r["n_nonzero"]) == (4, 3)
    with pytest.raises(AssertionError):
        signed_rank(pd.Series([1.0, -1.0, 2.0, 3.0]))


def test_holm_is_step_down_and_monotone():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert np.isclose(adj["a"], 0.03) and np.isclose(adj["c"], 0.06) and np.isclose(adj["b"], 0.06)


def test_an_untestable_test_leaves_the_holm_family():
    ten = pd.Series(np.arange(1.0, 11.0) * 0.001)  # all positive: p = 2 / 1024
    v = {k: ten for k in ("H1", "H2", "H3", "H4")} | {"H5": pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])}
    t = primary_tests(None, v).set_index("test")
    assert t.loc["H5", "verdict"] == "untestable" and np.isnan(t.loc["H5", "holm_p"])
    assert np.isclose(t.loc["H1", "holm_p"], min(1.0, 4 * 2 / 1024))  # four tests in the family, not five


def test_s0_star_matches_the_prevalence_of_its_window():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 400)
    y = (rng.random(400) < 0.5 * p + 0.1).astype(int)
    assert np.isclose(s0_star(p, y, p).sum(), y.sum(), atol=1e-3)


def test_metrics_clip_probabilities_and_count_distinct_predictions():
    lab = np.array([1, 1, 0, 0, 0])
    m = metrics(np.array([1.0, 0.9, 0.0, 0.1, 0.2], dtype=np.float32), lab)
    assert np.isfinite(m["logloss"]) and m["n_unique"] == 5 and m["auc"] == 1.0
    assert np.isclose(m["skill"], 1.0) and np.isclose(m["prevalence"], 0.4)
    assert metrics(np.full(5, 0.3, dtype=np.float32), lab)["n_unique"] == 1


def test_expected_cells():
    assert expected_cells("banana") == (73, 14)
    assert expected_cells("thyroid-sick-euthyroid") == (40, 8)
    assert expected_cells("ringnorm") == (8, 2)
    assert expected_cells("hotels") == (10, 0)


def frame(spec: dict[tuple[str, str], dict[str, tuple[float, float]]], seeds: tuple = (0, 1, 2)) -> pd.DataFrame:
    """Block-4 rows of one dataset, all six models. spec[(scenario, strategy)][family] = (auc, logloss)."""
    return pd.DataFrame([dict(dataset=DATASET, seed=s, model=m, scenario=sc, strategy=st, block=4, auc=v[f][0],
                              logloss=v[f][1], auc_out=v[f][0])
                         for s in seeds for (sc, st), v in spec.items()
                         for f, models in (("tfm", TFMS), ("booster", BOOSTERS)) for m in models])


def test_lead_is_tfm_mean_minus_booster_mean_and_joins_by_name_not_position():
    S = frame({("none", "S0"): {"tfm": (0.7, 0), "booster": (0.6, 0)}})
    x = S.set_index(["dataset", "seed", "model"]).auc
    assert np.isclose(lead(x)[DATASET], 0.1)
    assert np.isclose(lead(x.sample(frac=1, random_state=0))[DATASET], 0.1)


def test_a_seed_missing_for_every_model_makes_the_dataset_missing():
    S = frame({("none", "S0"): {"tfm": (0.7, 0), "booster": (0.6, 0)}}, seeds=(0, 1))
    x = S.set_index(["dataset", "seed", "model"]).auc
    assert np.isnan(famean(x, TFMS)[DATASET])


def test_primary_values_compose_difference_in_differences_and_the_prior_levels():
    spec = {("none", "S0"): {"tfm": (0.80, 0.50), "booster": (0.70, 0.60)},
            ("cov", "S0"): {"tfm": (0.82, 0.40), "booster": (0.70, 0.60)},
            ("none", "S1"): {"tfm": (0.81, 0.45), "booster": (0.72, 0.55)},
            ("cov", "S1"): {"tfm": (0.84, 0.0), "booster": (0.72, 0.0)},
            ("none", "S0*"): {"tfm": (0, 0.50), "booster": (0, 0.56)},
            ("prior0.5", "S1"): {"tfm": (0, 0.40), "booster": (0, 0.55)},
            ("prior0.5", "S0*"): {"tfm": (0, 0.50), "booster": (0, 0.56)},
            ("prior2", "S1"): {"tfm": (0, 0.42), "booster": (0, 0.55)},
            ("prior2", "S0*"): {"tfm": (0, 0.50), "booster": (0, 0.56)},
            ("region", "S0"): {"tfm": (0.5, 0), "booster": (0.5, 0)},
            ("region", "S1"): {"tfm": (0.6, 0), "booster": (0.55, 0)},
            ("region", "oracle"): {"tfm": (0.7, 0), "booster": (0.7, 0)}}
    v = primary_values(frame(spec))
    assert np.isclose(v["H1"][DATASET], (0.82 - 0.70) - (0.80 - 0.70))
    assert np.isclose(v["H2"][DATASET], ((0.84 - 0.82) - (0.72 - 0.70)) - ((0.81 - 0.80) - (0.72 - 0.70)))
    assert np.isclose(v["H4"][DATASET], (0.40 - 0.60) - (0.50 - 0.60))
    lead_none = (0.45 - 0.50) - (0.55 - 0.56)
    assert np.isclose(v["H3 prior0.5"][DATASET], ((0.40 - 0.50) - (0.55 - 0.56)) - lead_none)
    assert np.isclose(v["H3 prior2"][DATASET], ((0.42 - 0.50) - (0.55 - 0.56)) - lead_none)
    assert np.isclose(v["H3"][DATASET], (v["H3 prior0.5"][DATASET] + v["H3 prior2"][DATASET]) / 2)
    assert np.isclose(v["H5"][DATASET], 0.5 - 0.25)  # shares 0.1 / 0.2 and 0.05 / 0.2
    assert v["H1"].index.tolist()[0] == DATASET and v["H5"].notna().sum() == 1


def test_share_is_the_recovered_part_of_the_oracle_gain_and_the_guard_removes_small_gains():
    S = frame({("region", "S0"): {"tfm": (0.5, 0), "booster": (0.5, 0)},
               ("region", "S1"): {"tfm": (0.6, 0), "booster": (0.55, 0)},
               ("region", "oracle"): {"tfm": (0.7, 0), "booster": (0.7, 0)}})
    t = shares(S, "region")
    assert np.isclose(t.loc[DATASET, "tfm_share"], 0.5) and np.isclose(t.loc[DATASET, "booster_share"], 0.25)
    assert np.isclose(t.loc[DATASET, "diff"], 0.25)
    S = frame({("region", "S0"): {"tfm": (0.5, 0), "booster": (0.5, 0)},
               ("region", "S1"): {"tfm": (0.6, 0), "booster": (0.51, 0)},
               ("region", "oracle"): {"tfm": (0.7, 0), "booster": (0.52, 0)}})
    t = shares(S, "region")
    assert not t.loc[DATASET, "guard"] and np.isnan(t.loc[DATASET, "diff"]) and np.isnan(t.loc[DATASET, "booster_share"])


def test_h1_reading_needs_the_error_ratio_to_move_against_the_lead():
    rejected = dict(verdict="reject", est=0.01)
    assert reading_h1(rejected, dict(est=-0.1)).startswith("a different response")
    assert reading_h1(rejected, dict(est=0.1)).startswith("a change of scale")
    assert reading_h1(dict(verdict="not rejected", est=0.01), dict(est=-0.1)) == "not rejected, no reading"
