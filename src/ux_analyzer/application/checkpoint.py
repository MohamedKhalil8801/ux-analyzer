"""Atomic experiment progress checkpoints and finalized-bundle resume checks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from ux_analyzer.application.evaluation import persisted_comparison_sample_is_valid
from ux_analyzer.ports.artifacts import validate_timeline_event_order
from ux_analyzer.storage.run_bundle import (
    secure_assert_ancestors,
    secure_is_link_or_reparse,
    secure_read_bytes,
)

_CHECKPOINT_NAME = "experiment-progress.json"
_SCHEMA_VERSION = 1
_REQUIRED_BUNDLE_FILES = frozenset({"manifest.json", "timeline.jsonl", "result.json"})
_QUARANTINE_DIR = ".quarantine"
_MAX_BUNDLE_JSON_BYTES = 8 * 1024 * 1024
_MAX_BUNDLE_TIMELINE_BYTES = 16 * 1024 * 1024
_MAX_BUNDLE_CHECKSUM_BYTES = 8 * 1024 * 1024
_MAX_BUNDLE_EVENTS = 100_000


def _json_object_without_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON contains duplicate fields")
        result[key] = value
    return result


class CheckpointError(ValueError):
    """Experiment checkpoint is missing required or compatible state."""


@dataclass(frozen=True, slots=True)
class ExperimentCheckpoint:
    """Public immutable projection of one persisted experiment checkpoint."""

    selected_run_ids: tuple[str, ...]
    finalized_run_ids: tuple[str, ...]
    failed_run_ids: tuple[str, ...]
    interrupted_run_ids: tuple[str, ...]
    pending_run_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FinalizedBundleSnapshot:
    """Securely parsed finalized evidence reused by resume and evaluation."""

    bundle: Path
    manifest: dict[str, object]
    result: dict[str, object]
    events: tuple[dict[str, object], ...]


class ExperimentCheckpointStore:
    """Persist per-run experiment progress using atomic same-directory replace."""

    def __init__(
        self,
        output: Path,
        selected_run_ids: tuple[str, ...],
        *,
        selected_prominence_provider_ids: Mapping[str, str] | None = None,
    ) -> None:
        selected = tuple(selected_run_ids)
        if not selected or len(selected) != len(set(selected)):
            raise CheckpointError("selected run IDs must be non-empty and unique")
        if any(Path(run_id).name != run_id or not run_id for run_id in selected):
            raise CheckpointError("selected run IDs must be safe path components")
        self.output = Path(output)
        self.path = self.output / _CHECKPOINT_NAME
        self.temporary_path = self.output / f".{_CHECKPOINT_NAME}.tmp"
        self.selected_run_ids = selected
        selected_providers = dict(selected_prominence_provider_ids or {})
        if set(selected_providers) - set(selected):
            raise CheckpointError(
                "selected prominence providers contain an unknown run ID"
            )
        if any(not provider_id for provider_id in selected_providers.values()):
            raise CheckpointError("selected prominence provider IDs must be non-empty")
        self.selected_prominence_provider_ids = selected_providers
        self._statuses = {run_id: "pending" for run_id in selected}
        self._failure_types: dict[str, str] = {}
        self._interrupted: set[str] = set()

    def initialize(self, *, resume: bool = False) -> ExperimentCheckpoint:
        """Create a checkpoint or load and reconcile one for explicit resume."""

        if self.path.exists():
            if not resume:
                raise CheckpointError(
                    "experiment checkpoint already exists; use --resume or a new output"
                )
            self._load()
        elif not resume:
            self._write()

        if resume:
            self._reconcile_filesystem()
            self._write()
        return self.state

    @property
    def state(self) -> ExperimentCheckpoint:
        """Return current status lists in selected-matrix order."""

        def selected_with(status: str) -> tuple[str, ...]:
            return tuple(
                run_id
                for run_id in self.selected_run_ids
                if self._statuses[run_id] == status
            )

        return ExperimentCheckpoint(
            selected_run_ids=self.selected_run_ids,
            finalized_run_ids=selected_with("finalized"),
            failed_run_ids=selected_with("failed"),
            interrupted_run_ids=tuple(
                run_id
                for run_id in self.selected_run_ids
                if run_id in self._interrupted
            ),
            pending_run_ids=selected_with("pending"),
        )

    def record_finalized(self, run_id: str) -> ExperimentCheckpoint:
        """Record one published bundle after the run agent returns."""

        self._require_selected(run_id)
        self._statuses[run_id] = "finalized"
        self._failure_types.pop(run_id, None)
        self._interrupted.discard(run_id)
        self._write()
        return self.state

    def record_failure(self, run_id: str, error_type: str) -> ExperimentCheckpoint:
        """Record one sanitized execution failure."""

        self._require_selected(run_id)
        self._statuses[run_id] = "failed"
        self._failure_types[run_id] = error_type or "ExecutionFailure"
        self._write()
        return self.state

    def archive_interrupted_staging(self) -> tuple[Path, ...]:
        """Move selected interrupted staging bundles aside before retrying."""

        archived: list[Path] = []
        archive_root = self.output / ".interrupted"
        for run_id in self.state.interrupted_run_ids:
            source = self.output / ".staging" / run_id
            if not source.is_dir():
                continue
            archive_root.mkdir(parents=True, exist_ok=True)
            destination = archive_root / run_id
            suffix = 1
            while destination.exists():
                destination = archive_root / f"{run_id}-{suffix}"
                suffix += 1
            shutil.move(str(source), str(destination))
            archived.append(destination)
        return tuple(archived)

    def _quarantine_invalid_finalized_bundle(self, run_id: str) -> None:
        bundle = self.output / "runs" / run_id
        if not (bundle.exists() or bundle.is_symlink()):
            return
        quarantine_root = self.output / _QUARANTINE_DIR
        quarantine_root.mkdir(parents=True, exist_ok=True)
        destination = quarantine_root / run_id
        suffix = 1
        while destination.exists():
            destination = quarantine_root / f"{run_id}-{suffix}"
            suffix += 1
        shutil.move(str(bundle), str(destination))

    def _require_selected(self, run_id: str) -> None:
        if run_id not in self._statuses:
            raise CheckpointError(f"run ID is not in selected matrix: {run_id}")

    def _load(self) -> None:
        try:
            secure_assert_ancestors(self.path, "experiment checkpoint")
            if secure_is_link_or_reparse(self.path):
                raise RuntimeError(
                    "experiment checkpoint is a symlink or reparse point"
                )
            raw_value = json.loads(
                secure_read_bytes(
                    self.path,
                    "experiment checkpoint",
                    max_bytes=_MAX_BUNDLE_JSON_BYTES,
                ).decode("utf-8"),
                object_pairs_hook=_json_object_without_duplicates,
            )
        except (OSError, RuntimeError, UnicodeError, ValueError) as error:
            raise CheckpointError(f"invalid checkpoint: {error}") from error
        if not isinstance(raw_value, dict):
            raise CheckpointError("invalid checkpoint: expected object")
        value = cast(dict[str, object], raw_value)
        if value.get("schema_version") != _SCHEMA_VERSION:
            raise CheckpointError("invalid checkpoint: unsupported schema")
        selected = value.get("selected_run_ids")
        if selected != list(self.selected_run_ids):
            raise CheckpointError("checkpoint selected run IDs do not match matrix")
        raw_statuses = value.get("statuses")
        if not isinstance(raw_statuses, dict):
            raise CheckpointError("invalid checkpoint: incomplete run statuses")
        statuses = cast(dict[object, object], raw_statuses)
        if set(statuses) != set(self.selected_run_ids):
            raise CheckpointError("invalid checkpoint: incomplete run statuses")
        allowed = {"pending", "finalized", "failed"}
        if any(
            not isinstance(status, str) or status not in allowed
            for status in statuses.values()
        ):
            raise CheckpointError("invalid checkpoint: unknown run status")
        self._statuses = {
            run_id: cast(str, statuses[run_id]) for run_id in self.selected_run_ids
        }
        raw_interrupted = value.get("interrupted_run_ids", [])
        if not isinstance(raw_interrupted, list):
            raise CheckpointError("invalid checkpoint: interrupted runs must be a list")
        interrupted = cast(list[object], raw_interrupted)
        if any(
            not isinstance(run_id, str) or run_id not in self._statuses
            for run_id in interrupted
        ):
            raise CheckpointError("invalid checkpoint: unknown interrupted run ID")
        self._interrupted = set(cast(list[str], interrupted))
        raw_failures = value.get("failure_types", {})
        if not isinstance(raw_failures, dict):
            raise CheckpointError("invalid checkpoint: failure types must be an object")
        failures = cast(dict[object, object], raw_failures)
        if any(
            not isinstance(key, str)
            or key not in self._statuses
            or not isinstance(item, str)
            for key, item in failures.items()
        ):
            raise CheckpointError("invalid checkpoint: malformed failure type")
        self._failure_types = cast(dict[str, str], dict(failures))

    def _reconcile_filesystem(self) -> None:
        for run_id in self.selected_run_ids:
            expected_provider = self.selected_prominence_provider_ids.get(run_id)
            if finalized_bundle_is_valid(
                self.output,
                run_id,
                expected_prominence_provider_id=expected_provider,
            ):
                self._statuses[run_id] = "finalized"
                self._failure_types.pop(run_id, None)
                self._interrupted.discard(run_id)
                continue
            self._quarantine_invalid_finalized_bundle(run_id)
            self._statuses[run_id] = "pending"
            if (self.output / ".staging" / run_id).is_dir():
                self._interrupted.add(run_id)

    def _write(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        state = self.state
        value: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "selected_run_ids": list(state.selected_run_ids),
            "statuses": dict(self._statuses),
            "finalized_run_ids": list(state.finalized_run_ids),
            "failed_run_ids": list(state.failed_run_ids),
            "interrupted_run_ids": list(state.interrupted_run_ids),
            "pending_run_ids": list(state.pending_run_ids),
            "failure_types": dict(self._failure_types),
        }
        content = (
            json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        self.temporary_path.write_text(content, encoding="utf-8")
        os.replace(self.temporary_path, self.path)


def finalized_bundle_is_valid(
    output: Path,
    run_id: str,
    *,
    expected_prominence_provider_id: str | None = None,
) -> bool:
    """Return whether a selected finalized bundle is complete and trustworthy."""

    if not run_id or Path(run_id).name != run_id:
        return False
    bundle = Path(output) / "runs" / run_id
    return not finalized_bundle_failures(
        bundle,
        expected_run_id=run_id,
        expected_prominence_provider_id=expected_prominence_provider_id,
        require_persisted_validity=False,
    )


def finalized_bundle_failures(
    bundle: Path,
    *,
    expected_run_id: str | None = None,
    expected_prominence_provider_id: str | None = None,
    require_persisted_validity: bool = False,
) -> list[str]:
    """Return integrity and terminal-structure failures for a finalized bundle."""

    bundle = Path(bundle)
    checksum_path = bundle / "checksums.sha256"
    if (
        not bundle.is_dir()
        or secure_is_link_or_reparse(bundle)
        or secure_is_link_or_reparse(checksum_path)
        or not checksum_path.is_file()
    ):
        return ["missing checksum file: checksums.sha256"]
    try:
        secure_assert_ancestors(bundle, "finalized bundle")
    except (OSError, RuntimeError, ValueError):
        return ["finalized bundle path contains symlink or reparse point"]
    try:
        lines = (
            secure_read_bytes(
                checksum_path,
                "finalized bundle checksums",
                max_bytes=_MAX_BUNDLE_CHECKSUM_BYTES,
            )
            .decode("utf-8")
            .splitlines()
        )
    except (OSError, RuntimeError, UnicodeError):
        return ["unreadable checksum file: checksums.sha256"]
    failures: list[str] = []
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if "  " not in line:
            failures.append(f"invalid checksum entry at line {line_number}")
            continue
        digest, relative = line.split("  ", maxsplit=1)
        path = PurePosixPath(relative)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in relative
            or relative in checksums
        ):
            failures.append(f"invalid checksum entry at line {line_number}")
            continue
        checksums[relative] = digest

    actual_files: set[str] = set()
    for candidate_index, candidate in enumerate(bundle.rglob("*"), start=1):
        if candidate_index > _MAX_BUNDLE_EVENTS:
            failures.append("finalized bundle contains too many files")
            break
        if secure_is_link_or_reparse(candidate):
            failures.append("finalized bundle contains symlink or reparse point")
            continue
        if candidate.is_file() and candidate.name != "checksums.sha256":
            actual_files.add(candidate.relative_to(bundle).as_posix())
    for required in sorted(_REQUIRED_BUNDLE_FILES):
        if required not in actual_files:
            failures.append(f"missing required bundle file: {required}")
        elif required not in checksums:
            failures.append(f"required file missing checksum: {required}")
    for relative in sorted(actual_files - checksums.keys()):
        failures.append(f"checksum entry missing: {relative}")
    for relative in sorted(checksums.keys() - actual_files):
        failures.append(f"checksummed file missing: {relative}")
    for relative in sorted(actual_files & checksums.keys()):
        path = bundle.joinpath(*PurePosixPath(relative).parts)
        try:
            digest = hashlib.sha256()
            digest.update(
                secure_read_bytes(
                    path,
                    "finalized bundle artifact",
                    max_bytes=_MAX_BUNDLE_TIMELINE_BYTES,
                )
            )
            actual = digest.hexdigest()
        except (OSError, RuntimeError):
            failures.append(f"unreadable checksummed file: {relative}")
            continue
        if actual != checksums[relative]:
            failures.append(f"checksum mismatch: {relative}")

    manifest = _read_json_object(bundle / "manifest.json", "manifest.json", failures)
    result = _read_json_object(bundle / "result.json", "result.json", failures)
    events = _read_json_lines(bundle / "timeline.jsonl", failures)
    if expected_prominence_provider_id is not None:
        actual_provider = manifest.get("prominence_provider_id", "heuristic")
        if actual_provider != expected_prominence_provider_id:
            failures.append(
                "manifest prominence provider ID does not match selected run"
            )
    raw_metrics = result.get("metrics")
    if require_persisted_validity and isinstance(raw_metrics, Mapping):
        raw_metrics_mapping = cast(Mapping[str, object], raw_metrics)
        expected_provider = expected_prominence_provider_id or manifest.get(
            "prominence_provider_id", "heuristic"
        )
        if not isinstance(
            expected_provider, str
        ) or not persisted_comparison_sample_is_valid(
            result,
            raw_metrics_mapping,
            manifest=manifest,
            expected_run_id=expected_run_id or str(manifest.get("run_id", "")),
            expected_prominence_provider_id=expected_provider,
            timeline_events=events,
        ):
            failures.append("persisted comparison validity gate failed")
    embedded_ids = [
        value
        for value in (
            manifest.get("run_id"),
            result.get("run_id"),
            cast(dict[str, object], result.get("metrics", {})).get("run_id")
            if isinstance(result.get("metrics"), dict)
            else None,
        )
        if value is not None
    ]
    selected_id = expected_run_id or (
        manifest.get("run_id") if isinstance(manifest.get("run_id"), str) else None
    )
    if selected_id is not None and bundle.name != selected_id:
        failures.append("bundle directory does not match embedded run ID")
    if (
        selected_id is None
        or not embedded_ids
        or any(value != selected_id for value in embedded_ids)
    ):
        failures.append("embedded run IDs do not match bundle")
    outcome = result.get("outcome")
    if not isinstance(outcome, dict) or not isinstance(
        cast(dict[object, object], outcome).get("kind"), str
    ):
        failures.append("result.json missing terminal outcome")
    terminal = next(
        (event for event in reversed(events) if event.get("kind") == "run-terminated"),
        None,
    )
    if terminal is None:
        failures.append("missing terminal run event")
    else:
        terminal_outcome = terminal.get("outcome")
        if not isinstance(terminal_outcome, dict) or not isinstance(
            cast(dict[object, object], terminal_outcome).get("kind"), str
        ):
            failures.append("terminal run event missing outcome")
    return list(dict.fromkeys(failures))


def _read_json_object(path: Path, name: str, failures: list[str]) -> dict[str, object]:
    try:
        if secure_is_link_or_reparse(path):
            raise RuntimeError("bundle file is a symlink or reparse point")
        value = json.loads(
            secure_read_bytes(
                path,
                f"bundle {name}",
                max_bytes=_MAX_BUNDLE_JSON_BYTES,
            ).decode("utf-8"),
            object_pairs_hook=_json_object_without_duplicates,
        )
    except (OSError, RuntimeError, UnicodeError, ValueError):
        failures.append(f"invalid JSON bundle file: {name}")
        return {}
    if not isinstance(value, dict):
        failures.append(f"bundle file must contain object: {name}")
        return {}
    return cast(dict[str, object], value)


def _read_json_lines(path: Path, failures: list[str]) -> list[dict[str, object]]:
    try:
        if secure_is_link_or_reparse(path):
            raise RuntimeError("timeline is a symlink or reparse point")
        lines = (
            secure_read_bytes(
                path,
                "bundle timeline",
                max_bytes=_MAX_BUNDLE_TIMELINE_BYTES,
            )
            .decode("utf-8")
            .splitlines()
        )
    except (OSError, RuntimeError, UnicodeError):
        failures.append("unreadable bundle file: timeline.jsonl")
        return []
    events: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, object_pairs_hook=_json_object_without_duplicates)
        except ValueError:
            failures.append(f"invalid timeline JSON at line {line_number}")
            continue
        if not isinstance(value, dict):
            failures.append(f"timeline entry is not object at line {line_number}")
            continue
        if len(events) >= _MAX_BUNDLE_EVENTS:
            failures.append("timeline.jsonl contains too many events")
            break
        events.append(cast(dict[str, object], value))
    if not events:
        failures.append("timeline.jsonl contains no events")
    failures.extend(validate_timeline_event_order(events))
    return events


def read_finalized_bundle(
    bundle: Path,
    *,
    expected_run_id: str | None = None,
    expected_prominence_provider_id: str | None = None,
) -> FinalizedBundleSnapshot:
    """Validate and securely parse one finalized bundle for trusted reuse."""

    failures = finalized_bundle_failures(
        bundle,
        expected_run_id=expected_run_id,
        expected_prominence_provider_id=expected_prominence_provider_id,
    )
    if failures:
        raise CheckpointError("invalid finalized bundle: " + "; ".join(failures))
    parsed_failures: list[str] = []
    manifest = _read_json_object(
        Path(bundle) / "manifest.json", "manifest.json", parsed_failures
    )
    result = _read_json_object(
        Path(bundle) / "result.json", "result.json", parsed_failures
    )
    events = _read_json_lines(Path(bundle) / "timeline.jsonl", parsed_failures)
    if parsed_failures:
        raise CheckpointError("invalid finalized bundle: " + "; ".join(parsed_failures))
    return FinalizedBundleSnapshot(Path(bundle), manifest, result, tuple(events))
