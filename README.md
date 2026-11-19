# tfm-drift

Do tabular foundation models keep their lead over tuned gradient boosters when the test data drifts? The pre-registered design is in [plan.md](plan.md).

Datasets, the loader and the model wrappers come from [SmallDataBenchmarks](https://github.com/firefly1248/SmallDataBenchmarks), read in place: clone it next to this folder or set `SDB_ROOT`.

```
uv sync
uv run python -m scripts.drift_windows   # injection gate, writes results/drift_gate.csv
uv run pytest
```
