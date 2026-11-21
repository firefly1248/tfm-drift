"""drift_power_v2.py: simulated power of the exact signed-rank TOST of plan_v2.md (normal per-dataset values).

Usage: uv run python -m scripts.drift_power_v2   (writes results/v2/power.csv)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.drift_analysis_v2 import OUT, one_sided_p

# v1's per-dataset SD of H1, H2, H4 over the nine datasets without hill-valley-with-noise, and over all ten
SDS = {"nine": {"E1": 0.0041, "E2": 0.0041, "E3": 0.0066}, "ten": {"E1": 0.0072, "E2": 0.0068, "E3": 0.0089}}
MARGIN = {"E1": 0.005, "E2": 0.005, "E3": 0.009}
SHIFTS = {"E1": (0.0, 0.002, 0.003), "E2": (0.0, 0.002, 0.003), "E3": (0.0, 0.003, 0.006)}
REPS = 4_000


def power(test: str, sd: float, shift: float, n: int, alpha: float, rng: np.random.Generator) -> float:
    hits = 0
    for _ in range(REPS):
        v = rng.normal(shift, sd, n)
        hits += max(one_sided_p(v + MARGIN[test]), one_sided_p(MARGIN[test] - v)) < alpha
    return hits / REPS


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rows = [dict(sd_over=k, test=t, sd=sd[t], true_shift=s, n=n, power_alpha_05=power(t, sd[t], s, n, 0.05, rng),
                 power_holm_first_step=power(t, sd[t], s, n, 0.05 / 3, rng))
            for k, sd in SDS.items() for t in sd for s in SHIFTS[t] for n in (25, 30, 40)]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "power.csv", index=False)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
