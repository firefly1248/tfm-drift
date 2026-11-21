"""pack_runs.py: pack the run checkpoints that may be published into drift_runs.tar.xz and write
results/runs_manifest.json (sha256 of every packed file and of the archive).

The Study B datasets, the concept-only datasets and PUBLIC_A are packed; himalaya and f1-dnf (RESTRICTED) never are.
Extract at the repository root: tar -xJf drift_runs.tar.xz

Usage: uv run python -m scripts.pack_runs
"""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from scripts.drift_queue import CONCEPT, jobs
from scripts.drift_run import RESULTS
from scripts.drift_windows import PUBLIC_A, RESTRICTED, STUDY_B

ROOT = Path(__file__).resolve().parent.parent
ALLOWED = [*STUDY_B, *CONCEPT, *PUBLIC_A]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack(runs: Path, archive: Path, manifest: Path, datasets: list[str], expected: int) -> None:
    files = sorted(p for d in datasets for p in (runs / d).glob("*.joblib"))
    assert len(files) == expected, f"{len(files)} checkpoints, expected {expected}"
    names = {p.relative_to(runs).as_posix(): p for p in files}
    assert not any(r in n for n in names for r in RESTRICTED), "a restricted dataset is in the archive"

    def fixed(info: tarfile.TarInfo) -> tarfile.TarInfo:  # the same files give the same archive
        info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
        return info

    with tarfile.open(archive, "w:xz") as tar:
        for name, p in names.items():
            tar.add(p, arcname=f"results/runs/{name}", filter=fixed)
    manifest.write_text(json.dumps({"archive": {"name": archive.name, "sha256": sha256(archive)},
                                    "files": {n: sha256(p) for n, p in names.items()}}, indent=1) + "\n")


if __name__ == "__main__":
    expected = len([j for j in jobs() if j[1] in ALLOWED])
    pack(RESULTS, ROOT / "drift_runs.tar.xz", ROOT / "results" / "runs_manifest.json", ALLOWED, expected)
    print(f"packed {expected} checkpoints")
