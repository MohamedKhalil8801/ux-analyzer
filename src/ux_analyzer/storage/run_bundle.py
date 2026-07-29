"""Filesystem implementation of immutable run bundle storage."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, cast

from ux_analyzer.domain.run import ProviderManifest
from ux_analyzer.ports.artifacts import (
    REDACTED_VALUE,
    ArtifactReference,
    BundleAlreadyFinalizedError,
    BundleManifest,
    BundleStateError,
    RedactionPolicy,
)

_CHECKSUMS_FILE = "checksums.sha256"
_CRASH_MARKER = "crash.marker"
_ACTIVE_MARKER = ".active"


def _json_value(value: object) -> Any:
    """Convert domain values to JSON data at infrastructure boundary."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {
            "encoding": "base64",
            "data": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): _json_value(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        sequence = cast(Sequence[object], value)
        return [_json_value(item) for item in sequence]
    if is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_value(model_dump(mode="json"))
    raise TypeError(f"cannot serialize artifact value of type {type(value)!r}")


def _redact(value: Any, policy: RedactionPolicy) -> Any:
    if isinstance(value, dict):
        mapping = cast(dict[str, Any], value)
        return {
            key: REDACTED_VALUE if key in policy.keys else _redact(item, policy)
            for key, item in mapping.items()
        }
    if isinstance(value, list):
        sequence = cast(list[Any], value)
        return [_redact(item, policy) for item in sequence]
    if isinstance(value, str):
        redacted = value
        for exact_value in sorted(policy.exact_values, key=len, reverse=True):
            redacted = redacted.replace(exact_value, REDACTED_VALUE)
        return redacted
    return value


def _json_bytes(value: object, policy: RedactionPolicy) -> bytes:
    return (
        json.dumps(
            _redact(_json_value(value), policy),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, content: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _coerce_manifest(value: BundleManifest | Mapping[str, object]) -> BundleManifest:
    if isinstance(value, BundleManifest):
        return value
    provider_manifests_value = value.get("provider_manifests", ())
    provider_manifests: list[ProviderManifest] = []
    if isinstance(provider_manifests_value, Sequence) and not isinstance(
        provider_manifests_value, (str, bytes)
    ):
        provider_items = cast(Sequence[object], provider_manifests_value)
        for item in provider_items:
            if isinstance(item, ProviderManifest):
                provider_manifests.append(item)
                continue
            if not isinstance(item, Mapping):
                raise TypeError("provider manifest must be mapping")
            item_mapping = cast(Mapping[str, object], item)
            provider_manifests.append(
                ProviderManifest(
                    provider_id=str(item_mapping["provider_id"]),
                    role=str(item_mapping["role"]),
                    model_id=cast(str | None, item_mapping.get("model_id")),
                    endpoint_origin=str(item_mapping["endpoint_origin"]),
                    version=str(item_mapping["version"]),
                    prompt_version=cast(str | None, item_mapping.get("prompt_version")),
                    schema_version=cast(str | None, item_mapping.get("schema_version")),
                )
            )
    seed_value = value.get("seed")
    if not isinstance(seed_value, int):
        raise TypeError("manifest seed must be an integer")
    return BundleManifest(
        run_id=str(value["run_id"]),
        seed=seed_value,
        config_digest=str(value["config_digest"]),
        endpoint_origin=str(value["endpoint_origin"]),
        model_ids=cast(Mapping[str, str], value.get("model_ids", {})),
        prompt_versions=cast(Mapping[str, str], value.get("prompt_versions", {})),
        package_version=str(value.get("package_version", "unknown")),
        provider_versions=cast(Mapping[str, str], value.get("provider_versions", {})),
        provider_manifests=tuple(provider_manifests),
    )


class FilesystemRunBundleWriter:
    """Append-only writer that publishes one bundle with an atomic rename."""

    def __init__(
        self,
        *,
        output_dir: Path,
        manifest: BundleManifest,
        redaction: RedactionPolicy,
    ) -> None:
        self.output_dir = output_dir
        self.manifest = manifest
        self.redaction = redaction
        self.staging_path = output_dir / ".staging" / manifest.run_id
        self.final_path = output_dir / "runs" / manifest.run_id
        self.timeline_path = self.staging_path / "timeline.jsonl"
        self._timeline = self.timeline_path.open("a", encoding="utf-8")
        self._next_sequence = 1
        self._finalized = False
        self._aborted = False

    @classmethod
    def start(
        cls,
        output_dir: Path,
        manifest: BundleManifest | Mapping[str, object],
        *,
        redaction: RedactionPolicy | None = None,
    ) -> FilesystemRunBundleWriter:
        """Create staging directory and write initial manifest atomically."""

        bundle_manifest = _coerce_manifest(manifest)
        output_dir = Path(output_dir)
        staging_path = output_dir / ".staging" / bundle_manifest.run_id
        final_path = output_dir / "runs" / bundle_manifest.run_id
        if final_path.exists():
            raise BundleStateError("run bundle already finalized")
        if staging_path.exists():
            raise BundleStateError("run bundle staging directory already exists")

        staging_path.mkdir(parents=True)
        (staging_path / "artifacts").mkdir()
        _write_bytes(
            staging_path / "manifest.json",
            _json_bytes(bundle_manifest.to_dict(), redaction or RedactionPolicy()),
        )
        _write_bytes(
            staging_path / _ACTIVE_MARKER,
            _json_bytes(
                {"run_id": bundle_manifest.run_id}, redaction or RedactionPolicy()
            ),
        )
        return cls(
            output_dir=output_dir,
            manifest=bundle_manifest,
            redaction=redaction or RedactionPolicy(),
        )

    @property
    def run_id(self) -> str:
        return self.manifest.run_id

    @property
    def finalized(self) -> bool:
        return self._finalized

    def _ensure_writable(self) -> None:
        if self._finalized:
            raise BundleAlreadyFinalizedError("finalized run bundle is immutable")
        if self._aborted:
            raise BundleStateError("aborted run bundle cannot be mutated")

    def append_event(self, event: object) -> int:
        """Append one event, overriding caller sequence with monotonic sequence."""

        self._ensure_writable()
        event_value = _json_value(event)
        if not isinstance(event_value, dict):
            raise TypeError("run event must serialize to an object")
        event_value["sequence"] = self._next_sequence
        self._timeline.write(
            json.dumps(
                _redact(event_value, self.redaction),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        self._timeline.flush()
        os.fsync(self._timeline.fileno())
        sequence = self._next_sequence
        self._next_sequence += 1
        return sequence

    def write_artifact(self, name: str, content: bytes | str) -> ArtifactReference:
        """Store artifact under content hash and reuse an existing hash path."""

        self._ensure_writable()
        safe_name = Path(name).name
        if not safe_name or safe_name in {".", ".."}:
            raise ValueError("artifact name must contain a filename")
        raw_content = content.encode("utf-8") if isinstance(content, str) else content
        digest = hashlib.sha256(raw_content).hexdigest()
        relative_path = Path("artifacts") / digest
        destination = self.staging_path / relative_path
        if not destination.exists():
            temporary = destination.with_name(f".{digest}.tmp")
            _write_bytes(temporary, raw_content)
            os.replace(temporary, destination)
        return ArtifactReference(
            path=relative_path.as_posix(),
            sha256=digest,
            size=len(raw_content),
            name=safe_name,
        )

    def finalize(self, result: object) -> Path:
        """Write terminal result and checksums, then atomically publish bundle."""

        self._ensure_writable()
        try:
            self._timeline.flush()
            self._timeline.close()
            _write_bytes(
                self.staging_path / "result.json",
                _json_bytes(result, self.redaction),
            )
            (self.staging_path / _ACTIVE_MARKER).unlink(missing_ok=True)
            checksums = self._checksums()
            checksum_content = "".join(
                f"{digest}  {relative_path}\n" for relative_path, digest in checksums
            ).encode("utf-8")
            _write_bytes(self.staging_path / _CHECKSUMS_FILE, checksum_content)
            self.final_path.parent.mkdir(parents=True, exist_ok=True)
            if self.final_path.exists():
                raise BundleStateError("run bundle final path already exists")
            os.replace(self.staging_path, self.final_path)
        except BaseException as error:
            if not self._timeline.closed:
                self._timeline.close()
            reason = (
                f"finalization failed: {str(error).strip() or type(error).__name__}"
            )
            _write_bytes(
                self.staging_path / _CRASH_MARKER,
                _json_bytes(
                    {
                        "run_id": self.run_id,
                        "outcome": "internal-error",
                        "reason": reason,
                    },
                    self.redaction,
                ),
            )
            (self.staging_path / _ACTIVE_MARKER).unlink(missing_ok=True)
            self._aborted = True
            raise
        self._finalized = True
        return self.final_path

    def abort(self, reason: str) -> Path:
        """Record crash marker while retaining incomplete staging evidence."""

        self._ensure_writable()
        if not reason:
            raise ValueError("abort reason must not be empty")
        self._timeline.flush()
        self._timeline.close()
        _write_bytes(
            self.staging_path / _CRASH_MARKER,
            _json_bytes({"run_id": self.run_id, "reason": reason}, self.redaction),
        )
        (self.staging_path / _ACTIVE_MARKER).unlink(missing_ok=True)
        self._aborted = True
        return self.staging_path

    def _checksums(self) -> list[tuple[str, str]]:
        files = sorted(
            path
            for path in self.staging_path.rglob("*")
            if path.is_file() and path.name not in {_CHECKSUMS_FILE, _ACTIVE_MARKER}
        )
        return [
            (
                path.relative_to(self.staging_path).as_posix(),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in files
        ]
