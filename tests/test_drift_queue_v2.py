from scripts.drift_queue_v2 import jobs
from scripts.drift_run import MODELS
from scripts.drift_run_v2 import population


def test_the_queue_runs_seeds_in_the_planned_order_and_counts_every_pair():
    q = jobs()
    b2, c = population("b2"), population("c")
    pairs = int(b2.seeds.sum()) + 3 * len(c)
    assert len(q) == pairs * len(MODELS) and len(set(q)) == len(q)
    first_late = next(i for i, (study, _, _, s) in enumerate(q) if s >= 3)
    assert all(s < 3 for _, _, _, s in q[:first_late]) and all(study == "b2" for study, _, _, s in q[first_late:])
    assert {d for study, _, d, _ in jobs("c")} == set(c.name) and not {s for s, *_ in jobs("c")} - {"c"}


def test_the_pilot_is_one_seed_of_five_datasets():
    q = jobs(pilot=True)
    assert len(q) == 5 * len(MODELS) and {s for *_, s in q} == {0}
    assert {d for _, _, d, _ in q} <= set(population("b2").name) | set(population("c").name)
