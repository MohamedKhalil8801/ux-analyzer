import hashlib
import json
from pathlib import Path

import pytest

from ux_analyzer.application.checkpoint import (
    CheckpointError,
    ExperimentCheckpointStore,
    finalized_bundle_is_valid,
)


def test_checkpoint_updates_statuses_atomically(tmp_path: Path) -> None:
    store = ExperimentCheckpointStore(tmp_path, ("run-1", "run-2"))

    state = store.initialize()
    store.record_finalized("run-1")
    store.record_failure("run-2", "ProviderFailure")

    persisted = json.loads((tmp_path / "experiment-progress.json").read_text())
    assert state.pending_run_ids == ("run-1", "run-2")
    assert persisted["finalized_run_ids"] == ["run-1"]
    assert persisted["failed_run_ids"] == ["run-2"]
    assert persisted["pending_run_ids"] == []
    assert not (tmp_path / ".experiment-progress.json.tmp").exists()


def test_checkpoint_resume_requires_same_selected_matrix(tmp_path: Path) -> None:
    ExperimentCheckpointStore(tmp_path, ("run-1", "run-2")).initialize()

    with pytest.raises(CheckpointError, match="selected run IDs"):
        ExperimentCheckpointStore(tmp_path, ("run-1", "run-3")).initialize(resume=True)


def test_checkpoint_rejects_malformed_existing_state(tmp_path: Path) -> None:
    (tmp_path / "experiment-progress.json").write_text("not json")

    with pytest.raises(CheckpointError, match="invalid checkpoint"):
        ExperimentCheckpointStore(tmp_path, ("run-1",)).initialize(resume=True)


def test_resume_marks_staging_and_validates_finalized_bundle(tmp_path: Path) -> None:
    run_path = tmp_path / "runs" / "run-1"
    run_path.mkdir(parents=True)
    content = {
        "manifest.json": b"{}\n",
        "timeline.jsonl": b'{"kind":"run-terminated"}\n',
        "result.json": b"{}\n",
    }
    for name, value in content.items():
        (run_path / name).write_bytes(value)
    (run_path / "checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256(value).hexdigest()}  {name}\n"
            for name, value in content.items()
        )
    )
    staging = tmp_path / ".staging" / "run-2"
    staging.mkdir(parents=True)

    store = ExperimentCheckpointStore(tmp_path, ("run-1", "run-2"))
    state = store.initialize(resume=True)

    assert finalized_bundle_is_valid(tmp_path, "run-1")
    assert state.finalized_run_ids == ("run-1",)
    assert state.interrupted_run_ids == ("run-2",)
    assert state.pending_run_ids == ("run-2",)


def test_invalid_checksum_is_not_resumable(tmp_path: Path) -> None:
    run_path = tmp_path / "runs" / "run-1"
    run_path.mkdir(parents=True)
    for name in ("manifest.json", "timeline.jsonl", "result.json"):
        (run_path / name).write_text("{}\n")
    (run_path / "checksums.sha256").write_text(
        f"{'0' * 64}  manifest.json\n"
        f"{hashlib.sha256(b'{}\n').hexdigest()}  timeline.jsonl\n"
        f"{hashlib.sha256(b'{}\n').hexdigest()}  result.json\n"
    )

    assert not finalized_bundle_is_valid(tmp_path, "run-1")
