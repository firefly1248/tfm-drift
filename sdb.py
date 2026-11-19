"""SmallDataBenchmarks, read in place: its loader, datasets and checkpoints are not copied here.

The root is SDB_ROOT, or the SmallDataBenchmarks folder next to this project.
"""
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("SDB_ROOT", Path(__file__).resolve().parent.parent / "SmallDataBenchmarks"))
sys.path.insert(0, str(ROOT))


def load_data_df(name: str):
    """The benchmark loader's frame, labels and categorical columns, with its 10,000-row cap."""
    from benchmark.data import load_data_df
    return load_data_df(name, datasets_dir=str(ROOT / "datasets"))
