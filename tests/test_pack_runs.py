import json
import tarfile

from scripts.pack_runs import pack


def test_the_archive_and_manifest_leave_out_restricted_datasets(tmp_path):
    runs = tmp_path / "runs"
    for d in ("banana", "himalaya"):
        (runs / d).mkdir(parents=True)
        (runs / d / "catboost_seed0.joblib").write_bytes(d.encode())
    pack(runs, tmp_path / "a.tar.xz", tmp_path / "m.json", ["banana"], expected=1)
    with tarfile.open(tmp_path / "a.tar.xz") as t:
        assert t.getnames() == ["results/runs/banana/catboost_seed0.joblib"]
    assert list(json.loads((tmp_path / "m.json").read_text())["files"]) == ["banana/catboost_seed0.joblib"]
