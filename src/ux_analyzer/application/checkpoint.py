"""Atomic experiment progress checkpoints and finalized-bundle resume checks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

_CHECKPOINT_NAME = "experiment-progress.json"
_SCHEMA_VERSION = 1
_REQUIRED_BUNDLE_FILES = frozenset({"manifest.json", "timeline.jsonl", "result.json"})


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


class ExperimentCheckpointStore:
    """Persist per-run experiment progress using atomic same-directory replace."""

    def __init__(self, output: Path, selected_run_ids: tuple[str, ...]) -> None:
        selected = tuple(selected_run_ids)
        if not selected or len(selected) != len(set(selected)):
            raise CheckpointError("selected run IDs must be non-empty and unique")
        if any(Path(run_id).name != run_id or not run_id for run_id in selected):
            raise CheckpointError("selected run IDs must be safe path components")
        self.output = Path(output)
        self.path = self.output / _CHECKPOINT_NAME
        self.temporary_path = self.output / f".{_CHECKPOINT_NAME}.tmp"
        self.selected_run_ids = selected
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

    def _require_selected(self, run_id: str) -> None:
        if run_id not in self._statuses:
            raise CheckpointError(f"run ID is not in selected matrix: {run_id}")

    def _load(self) -> None:
        try:
            raw_value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
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
            if finalized_bundle_is_valid(self.output, run_id):
                self._statuses[run_id] = "finalized"
                self._failure_types.pop(run_id, None)
                self._interrupted.discard(run_id)
                continue
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


def finalized_bundle_is_valid(output: Path, run_id: str) -> bool:
    """Return whether a selected finalized bundle passes checksum integrity."""

    if not run_id or Path(run_id).name != run_id:
        return False
    bundle = Path(output) / "runs" / run_id
    checksum_path = bundle / "checksums.sha256"
    if not bundle.is_dir() or not checksum_path.is_file():
        return False
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    checksums: dict[str, str] = {}
    for line in lines:
        if "  " not in line:
            return False
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
            return False
        checksums[relative] = digest
    if not _REQUIRED_BUNDLE_FILES.issubset(checksums):
        return False
    for relative, expected in checksums.items():
        path = bundle.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file():
            return False
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return False
        if actual != expected:
            return False
    return True
