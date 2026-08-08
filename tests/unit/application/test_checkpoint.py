import hashlib
import json
from pathlib import Path

import pytest

from ux_analyzer.application.checkpoint import (
    CheckpointError,
    ExperimentCheckpointStore,
    finalized_bundle_failures,
    finalized_bundle_is_valid,
)
from ux_analyzer.storage.run_bundle import FilesystemRunBundleWriter


def _write_finalized_bundle(
    root: Path,
    run_id: str,
    *,
    manifest_run_id: str | None = None,
    manifest_prominence_provider_id: str | None = None,
    result_run_id: str | None = None,
    terminal: bool = True,
    outcome: bool = True,
) -> Path:
    run_path = root / "runs" / run_id
    run_path.mkdir(parents=True)
    manifest = {"run_id": manifest_run_id or run_id, "seed": 0}
    if manifest_prominence_provider_id is not None:
        manifest["prominence_provider_id"] = manifest_prominence_provider_id
    content = {
        "manifest.json": json.dumps(manifest).encode(),
        "timeline.jsonl": (
            json.dumps(
                {
                    "sequence": 1,
                    "kind": "run-terminated" if terminal else "run-started",
                    "outcome": {"kind": "verified-success"} if terminal else None,
                }
            ).encode()
            + b"\n"
        ),
        "result.json": json.dumps(
            {
                "run_id": result_run_id or run_id,
                **({"outcome": {"kind": "verified-success"}} if outcome else {}),
            }
        ).encode(),
    }
    for name, value in content.items():
        (run_path / name).write_bytes(value)
    (run_path / "checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256(value).hexdigest()}  {name}\n"
            for name, value in content.items()
        )
    )
    return run_path


def _rewrite_checksums(run_path: Path) -> None:
    files = tuple(
        path
        for path in sorted(run_path.iterdir())
        if path.name != "checksums.sha256" and path.is_file()
    )
    (run_path / "checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in files
        )
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


def test_checkpoint_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    (tmp_path / "experiment-progress.json").write_text(
        '{"schema_version":1,"schema_version":1}', encoding="utf-8"
    )

    with pytest.raises(CheckpointError, match="invalid checkpoint"):
        ExperimentCheckpointStore(tmp_path, ("run-1",)).initialize(resume=True)


def test_resume_marks_staging_and_validates_finalized_bundle(tmp_path: Path) -> None:
    _write_finalized_bundle(tmp_path, "run-1")
    staging = tmp_path / ".staging" / "run-2"
    staging.mkdir(parents=True)

    store = ExperimentCheckpointStore(tmp_path, ("run-1", "run-2"))
    state = store.initialize(resume=True)

    assert finalized_bundle_is_valid(tmp_path, "run-1")
    assert state.finalized_run_ids == ("run-1",)
    assert state.interrupted_run_ids == ("run-2",)
    assert state.pending_run_ids == ("run-2",)


def test_resume_accepts_large_trace_artifact(tmp_path: Path) -> None:
    run = _write_finalized_bundle(tmp_path, "run-trace")
    artifact = run / "artifacts" / "trace.zip"
    artifact.parent.mkdir()
    artifact.write_bytes(b"x" * (64 * 1024 * 1024 + 1))
    checksums = run / "checksums.sha256"
    checksums.write_text(
        checksums.read_text()
        + f"{hashlib.sha256(artifact.read_bytes()).hexdigest()}  artifacts/trace.zip\n"
    )

    assert finalized_bundle_is_valid(tmp_path, "run-trace")


def test_resume_rejects_oversized_trace_artifact_without_loading_payload(
    tmp_path: Path,
) -> None:
    run = _write_finalized_bundle(tmp_path, "run-oversized-trace")
    artifact = run / "artifacts" / "trace.zip"
    artifact.parent.mkdir()
    with artifact.open("wb") as stream:
        stream.truncate(256 * 1024 * 1024 + 1)
    (run / "checksums.sha256").write_text(
        (run / "checksums.sha256").read_text()
        + f"{'0' * 64}  artifacts/trace.zip\n"
    )

    failures = finalized_bundle_failures(run, expected_run_id="run-oversized-trace")

    assert "unreadable checksummed file: artifacts/trace.zip" in failures


@pytest.mark.parametrize(
    ("mutation", "run_id"),
    (
        (lambda run: (run / "unchecksummed.json").write_text("{}"), "run-1"),
        (lambda run: None, "different-run"),
    ),
)
def test_resume_rejects_incomplete_bundle_coverage_or_wrong_selected_run(
    tmp_path: Path, mutation, run_id: str
) -> None:
    run = _write_finalized_bundle(tmp_path, "run-1")
    mutation(run)

    assert not finalized_bundle_is_valid(tmp_path, run_id)


@pytest.mark.parametrize(
    "bundle_options",
    (
        {"manifest_run_id": "other-run"},
        {"result_run_id": "other-run"},
        {"terminal": False},
        {"outcome": False},
    ),
)
def test_resume_rejects_structurally_invalid_or_mismatched_bundle(
    tmp_path: Path, bundle_options: dict[str, object]
) -> None:
    _write_finalized_bundle(tmp_path, "run-1", **bundle_options)

    assert not finalized_bundle_is_valid(tmp_path, "run-1")


def test_resume_rejects_checksum_consistent_duplicate_timeline_sequence(
    tmp_path: Path,
) -> None:
    run = _write_finalized_bundle(tmp_path, "run-1")
    timeline = (
        json.dumps({"sequence": 1, "kind": "run-started"}).encode()
        + b"\n"
        + json.dumps(
            {
                "sequence": 1,
                "kind": "run-terminated",
                "outcome": {"kind": "verified-success"},
            }
        ).encode()
        + b"\n"
    )
    (run / "timeline.jsonl").write_bytes(timeline)
    _rewrite_checksums(run)

    failures = finalized_bundle_failures(run, expected_run_id="run-1")

    assert any("timeline sequence" in failure for failure in failures)


def test_resume_accepts_checksum_valid_unsequenced_legacy_timeline(
    tmp_path: Path,
) -> None:
    run = _write_finalized_bundle(tmp_path, "run-legacy")
    timeline = (
        json.dumps(
            {"kind": "run-terminated", "outcome": {"kind": "verified-success"}}
        ).encode()
        + b"\n"
    )
    (run / "timeline.jsonl").write_bytes(timeline)
    _rewrite_checksums(run)

    assert finalized_bundle_is_valid(tmp_path, "run-legacy")


def test_resume_rejects_mixed_legacy_and_sequenced_timeline(
    tmp_path: Path,
) -> None:
    run = _write_finalized_bundle(tmp_path, "run-mixed")
    timeline = (
        json.dumps({"sequence": 1, "kind": "run-started"}).encode()
        + b"\n"
        + json.dumps(
            {"kind": "run-terminated", "outcome": {"kind": "verified-success"}}
        ).encode()
        + b"\n"
    )
    (run / "timeline.jsonl").write_bytes(timeline)
    _rewrite_checksums(run)

    failures = finalized_bundle_failures(run, expected_run_id="run-mixed")

    assert any("mixes" in failure for failure in failures)


def test_resume_rejects_event_after_terminal_with_consistent_checksums(
    tmp_path: Path,
) -> None:
    run = _write_finalized_bundle(tmp_path, "run-1")
    timeline = (
        json.dumps(
            {
                "sequence": 1,
                "kind": "run-terminated",
                "outcome": {"kind": "verified-success"},
            }
        ).encode()
        + b"\n"
        + json.dumps({"sequence": 2, "kind": "run-started"}).encode()
        + b"\n"
    )
    (run / "timeline.jsonl").write_bytes(timeline)
    _rewrite_checksums(run)

    failures = finalized_bundle_failures(run, expected_run_id="run-1")

    assert any("after terminal" in failure for failure in failures)


def test_resume_binds_bundle_directory_to_embedded_run_id(tmp_path: Path) -> None:
    run = _write_finalized_bundle(
        tmp_path,
        "run-directory",
        manifest_run_id="run-embedded",
        result_run_id="run-embedded",
    )

    failures = finalized_bundle_failures(run)

    assert any("bundle directory" in failure for failure in failures)


def test_resume_rejects_symlinked_manifest_path(tmp_path: Path) -> None:
    run = _write_finalized_bundle(tmp_path, "run-symlink")
    outside = tmp_path / "manifest-outside.json"
    outside.write_bytes((run / "manifest.json").read_bytes())
    (run / "manifest.json").unlink()
    try:
        (run / "manifest.json").symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation unavailable")

    assert not finalized_bundle_is_valid(tmp_path, "run-symlink")


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


def test_resume_rejects_bundle_with_wrong_prominence_provider(tmp_path: Path) -> None:
    _write_finalized_bundle(
        tmp_path,
        "run-foveacast",
        manifest_prominence_provider_id="heuristic",
    )

    assert not finalized_bundle_is_valid(
        tmp_path,
        "run-foveacast",
        expected_prominence_provider_id="foveacast",
    )


def test_resume_quarantines_wrong_provider_bundle_before_retry(tmp_path: Path) -> None:
    run_id = "run-foveacast"
    _write_finalized_bundle(
        tmp_path,
        run_id,
        manifest_prominence_provider_id="heuristic",
    )

    store = ExperimentCheckpointStore(
        tmp_path,
        (run_id,),
        selected_prominence_provider_ids={run_id: "foveacast"},
    )
    state = store.initialize(resume=True)

    assert state.pending_run_ids == (run_id,)
    assert not (tmp_path / "runs" / run_id).exists()
    assert (tmp_path / ".quarantine" / run_id / "manifest.json").is_file()

    writer = FilesystemRunBundleWriter.start(
        tmp_path,
        {
            "run_id": run_id,
            "seed": 0,
            "config_digest": "digest",
            "endpoint_origin": "https://example.test",
            "prominence_provider_id": "foveacast",
        },
    )
    writer.abort("retry regression")
