import pytest

from scripts import drift_queue, drift_windows
from scripts.drift_windows import STUDY_A, load_study_a, study_a_available


@pytest.fixture
def no_prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(drift_windows, "PREPARED", tmp_path)


def test_study_a_is_unavailable_without_the_manifest(no_prepared):
    assert not study_a_available()


def test_load_study_a_names_the_script_that_builds_the_files(no_prepared):
    with pytest.raises(FileNotFoundError, match="es_time_prep.py"):
        load_study_a(STUDY_A[0])


def test_the_queue_without_study_a_holds_no_study_a_dataset():
    assert not {d for _, d, _ in drift_queue.jobs(skip_study_a=True)} & set(STUDY_A)
    assert {d for _, d, _ in drift_queue.jobs()} >= set(STUDY_A)


def test_the_queue_stops_on_pending_study_a_jobs(no_prepared, monkeypatch, tmp_path):
    monkeypatch.setattr(drift_queue, "RESULTS", tmp_path)
    monkeypatch.setattr("sys.argv", ["drift_queue"])
    with pytest.raises(SystemExit, match="es_time_prep.py"):
        drift_queue.main()
