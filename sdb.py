"""SmallDataBenchmarks, read in place: its loader, datasets and checkpoints are not copied here.

The root is SDB_ROOT, or the SmallDataBenchmarks folder next to this project.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("SDB_ROOT", Path(__file__).resolve().parent.parent / "SmallDataBenchmarks"))
sys.path.insert(0, str(ROOT))
PINNED = "b25cf159757157394b8e861ceb51ea04a970f231"  # plan.md: SDB at d8c2e7b (same tree, authors rewritten)


def assert_pinned() -> None:
    """The SDB code this project imports is unchanged since the pinned commit, working tree included."""
    diff = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", PINNED, "--", "benchmark", "config.py"])
    assert diff.returncode == 0, f"SDB benchmark/ or config.py differs from {PINNED[:7]}"


def load_data_df(name: str):
    """The benchmark loader's frame, labels and categorical columns, with its 10,000-row cap."""
    from benchmark.data import load_data_df
    return load_data_df(name, datasets_dir=str(ROOT / "datasets"))
