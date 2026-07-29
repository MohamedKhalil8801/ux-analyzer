from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ux_analyzer.domain.run import ProviderManifest, RunStarted
from ux_analyzer.ports.artifacts import (
    BundleAlreadyFinalizedError,
    BundleManifest,
    RedactionPolicy,
)
from ux_analyzer.storage.run_bundle import FilesystemRunBundleWriter


def bundle_manifest(run_id: str = "run-1") -> BundleManifest:
    return BundleManifest(
        run_id=run_id,
        seed=17,
        config_digest="config-sha256",
        endpoint_origin="https://llm.example.test/v1",
        model_ids={"scent": "scent-model", "cognitive": "cognitive-model"},
        prompt_versions={"scent": "scent-v1", "cognitive": "cognitive-v1"},
        package_version="0.1.0",
        provider_versions={"observation": "fixture-1", "model": "adapter-2"},
        provider_manifests=(
            ProviderManifest(
                provider_id="fixture-provider",
                role="observation",
                model_id=None,
                endpoint_origin="fixture.invalid",
                version="fixture-1",
            ),
        ),
    )


def test_start_uses_atomic_staging_and_finalize_creates_required_files(
    tmp_path: Path,
) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())

    assert writer.staging_path == tmp_path / ".staging" / "run-1"
    assert writer.final_path == tmp_path / "runs" / "run-1"
    assert writer.staging_path.is_dir()
    assert not writer.final_path.exists()

    writer.append_event(RunStarted(run_id="run-1"))
    final_path = writer.finalize({"outcome": "verified-success"})

    assert final_path == tmp_path / "runs" / "run-1"
    assert not writer.staging_path.exists()
    assert (final_path / "manifest.json").is_file()
    assert (final_path / "timeline.jsonl").is_file()
    assert (final_path / "artifacts").is_dir()
    assert (final_path / "checksums.sha256").is_file()
    assert (final_path / "result.json").is_file()


def test_events_are_flushed_redacted_and_assigned_monotonic_sequences(
    tmp_path: Path,
) -> None:
    writer = FilesystemRunBundleWriter.start(
        tmp_path,
        bundle_manifest(),
        redaction=RedactionPolicy(
            exact_values=("fixture-secret", "api-secret"),
            keys=frozenset({"password", "api_key"}),
        ),
    )

    writer.append_event(
        {
            "kind": "run-started",
            "password": "fixture-secret",
            "nested": {"token": "api-secret"},
        }
    )
    assert writer.timeline_path.read_text(encoding="utf-8").count("\n") == 1
    writer.append_event({"kind": "checkpoint", "sequence": 999})

    events = [
        json.loads(line)
        for line in writer.timeline_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["sequence"] for event in events] == [1, 2]
    assert "fixture-secret" not in writer.timeline_path.read_text(encoding="utf-8")
    assert "api-secret" not in writer.timeline_path.read_text(encoding="utf-8")
    assert events[0]["password"] == "[REDACTED]"


def test_finalize_writes_manifest_result_and_checksums_without_secrets(
    tmp_path: Path,
) -> None:
    writer = FilesystemRunBundleWriter.start(
        tmp_path,
        bundle_manifest(),
        redaction=RedactionPolicy(
            exact_values=("api-secret",), keys=frozenset({"api_key"})
        ),
    )
    writer.append_event({"kind": "run-started"})
    artifact = writer.write_artifact("trace.zip", b"trace-bytes")
    final_path = writer.finalize(
        {"status": "complete", "api_key": "api-secret", "artifact": artifact.path}
    )

    manifest_text = (final_path / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["seed"] == 17
    assert manifest["config_digest"] == "config-sha256"
    assert manifest["endpoint_origin"] == "https://llm.example.test"
    assert manifest["model_ids"] == {
        "scent": "scent-model",
        "cognitive": "cognitive-model",
    }
    assert manifest["prompt_versions"] == {
        "scent": "scent-v1",
        "cognitive": "cognitive-v1",
    }
    assert manifest["package_version"] == "0.1.0"
    assert manifest["provider_versions"] == {
        "observation": "fixture-1",
        "model": "adapter-2",
    }
    assert "api-secret" not in manifest_text
    assert "api-secret" not in (final_path / "result.json").read_text(encoding="utf-8")

    checksum_lines = (
        (final_path / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    )
    checksums = {
        path: digest
        for digest, path in (line.split("  ", maxsplit=1) for line in checksum_lines)
    }
    assert "checksums.sha256" not in checksums
    for relative_path, digest in checksums.items():
        actual = hashlib.sha256((final_path / relative_path).read_bytes()).hexdigest()
        assert actual == digest


def test_identical_artifacts_share_content_hash_path(tmp_path: Path) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())

    first = writer.write_artifact("first.png", b"same-content")
    second = writer.write_artifact("second.trace", b"same-content")

    assert first.sha256 == second.sha256
    assert first.path == second.path
    assert len(list((writer.staging_path / "artifacts").iterdir())) == 1


def test_abort_leaves_crash_recovery_marker_in_staging_directory(
    tmp_path: Path,
) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())

    writer.abort("provider stopped unexpectedly")

    marker_path = writer.staging_path / "crash.marker"
    assert marker_path.is_file()
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["reason"] == "provider stopped unexpectedly"
    assert not writer.final_path.exists()


def test_finalized_bundle_refuses_all_mutation(tmp_path: Path) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())
    writer.finalize({"status": "complete"})

    with pytest.raises(BundleAlreadyFinalizedError):
        writer.append_event({"kind": "late-event"})
    with pytest.raises(BundleAlreadyFinalizedError):
        writer.write_artifact("late.txt", b"late")
    with pytest.raises(BundleAlreadyFinalizedError):
        writer.finalize({"status": "again"})
    with pytest.raises(BundleAlreadyFinalizedError):
        writer.abort("too late")


def test_atomic_publish_failure_leaves_internal_error_recovery_marker(
    monkeypatch, tmp_path: Path
) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())
    writer.append_event({"kind": "run-terminated", "outcome": "verified-success"})

    def fail_publish(source: object, destination: object) -> None:
        del source, destination
        raise OSError("atomic publish failed")

    monkeypatch.setattr("ux_analyzer.storage.run_bundle.os.replace", fail_publish)

    with pytest.raises(OSError, match="atomic publish failed"):
        writer.finalize({"outcome": "verified-success"})

    marker = json.loads(
        (writer.staging_path / "crash.marker").read_text(encoding="utf-8")
    )
    assert "finalization failed" in marker["reason"]
    assert "atomic publish failed" in marker["reason"]
    assert marker["outcome"] == "internal-error"
    assert not writer.final_path.exists()
