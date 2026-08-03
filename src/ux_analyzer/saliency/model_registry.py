"""Pinned saliency model manifests and explicit local artifact management."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast
from urllib.request import urlopen

import yaml
from platformdirs import user_data_dir

from ux_analyzer.ports.artifacts import BundleStateError
from ux_analyzer.storage.run_bundle import (
    secure_assert_ancestors,
    secure_ensure_directory,
    secure_is_link_or_reparse,
    secure_make_temporary_directory,
    secure_read_bytes,
    secure_remove_tree,
    secure_replace,
    secure_rmdir,
    secure_unlink,
    secure_write_bytes,
    secure_write_chunks,
)

DEFAULT_MODEL_ID = "foveacast-v0.2.0"
DEFAULT_PRECISION = "fp16"
_DOWNLOAD_TIMEOUT_SECONDS = 120
_CHUNK_SIZE = 1024 * 1024


class ModelRegistryError(RuntimeError):
    """Base error for manifest, download, and model-store failures."""


class ChecksumMismatchError(ModelRegistryError):
    """Raised when downloaded or installed bytes do not match a pinned digest."""


class ModelState(StrEnum):
    """Operator-visible state of one model/provider combination."""

    RUNTIME_MISSING = "runtime missing"
    ARTIFACT_MISSING = "artifact missing"
    CHECKSUM_MISMATCH = "checksum mismatch"
    UNSUPPORTED_PROVIDER = "unsupported provider"
    READY = "ready"


class ArtifactState(StrEnum):
    """Verification state for one manifest artifact."""

    MISSING = "missing"
    CHECKSUM_MISMATCH = "checksum mismatch"
    READY = "ready"


class RuntimeState(StrEnum):
    """Result of probing an installed ONNX Runtime package."""

    MISSING = "runtime missing"
    UNSUPPORTED = "unsupported provider"
    READY = "ready"


@dataclass(frozen=True, slots=True)
class ModelLicense:
    """One license link in the model attribution chain."""

    component: str
    license: str
    attribution: str


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """One immutable, checksum-pinned release artifact."""

    filename: str
    sha256: str
    url: str
    duration: str
    kind: str


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Versioned model release metadata loaded from a package manifest."""

    model_id: str
    provider: str
    version: str
    precision: str
    artifacts: tuple[ModelArtifact, ...]
    licenses: tuple[ModelLicense, ...]
    attribution: str

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("manifest model_id must not be empty")
        if not self.provider.strip() or not self.version.strip():
            raise ValueError("manifest provider and version must not be empty")
        if not self.precision.strip():
            raise ValueError("manifest precision must not be empty")
        artifacts = tuple(self.artifacts)
        if not artifacts:
            raise ValueError("manifest must contain artifacts")
        filenames = tuple(artifact.filename for artifact in artifacts)
        if len(filenames) != len(set(filenames)):
            raise ValueError("manifest artifacts must have unique filenames")
        for artifact in artifacts:
            if len(artifact.sha256) != 64 or any(
                character not in "0123456789abcdef" for character in artifact.sha256
            ):
                raise ValueError(f"invalid SHA-256 for {artifact.filename}")
            if (
                not artifact.filename
                or Path(artifact.filename).name != artifact.filename
            ):
                raise ValueError(f"invalid artifact filename: {artifact.filename}")
            if not artifact.url.startswith(("http://", "https://")):
                raise ValueError(f"invalid artifact URL: {artifact.url}")
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "licenses", tuple(self.licenses))

    @property
    def attribution_text(self) -> str:
        """Return complete human-readable license chain."""

        parts = [
            f"{license_item.component}: {license_item.license}"
            f" ({license_item.attribution})"
            for license_item in self.licenses
        ]
        if self.attribution:
            parts.append(self.attribution)
        return "; ".join(parts)


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    """Sanitized result from an ONNX Runtime/provider probe."""

    state: RuntimeState
    provider: str
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactStatus:
    """Checksum state for one artifact path."""

    artifact: ModelArtifact
    path: Path
    state: ArtifactState
    actual_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class RegistryStatus:
    """Complete local status diagnosis without network access."""

    model_id: str
    state: ModelState
    runtime: RuntimeStatus
    artifacts: tuple[ArtifactStatus, ...]
    diagnostics: tuple[str, ...]
    attribution: str

    @property
    def ready(self) -> bool:
        """Return whether runtime and every pinned artifact are ready."""

        return self.state is ModelState.READY


@dataclass(frozen=True, slots=True)
class InstallResult:
    """Files changed or reused during explicit installation."""

    model_id: str
    downloaded: tuple[str, ...]
    skipped: tuple[str, ...]
    attribution: str


@dataclass(frozen=True, slots=True)
class RemoveResult:
    """Files removed by explicit model removal."""

    model_id: str
    removed: tuple[str, ...]


RuntimeProbe = Callable[[str], RuntimeStatus]


def model_home() -> Path:
    """Resolve model store root from UXA_MODEL_HOME or platformdirs."""

    configured = os.environ.get("UXA_MODEL_HOME")
    if configured:
        return Path(configured).expanduser()
    return Path(user_data_dir("ux-analyzer", "ux-analyzer"))


def _manifest_path(model_id: str) -> Path:
    return Path(__file__).parent / "manifests" / f"{model_id}.yaml"


class _RuntimeModule(Protocol):
    def get_available_providers(self) -> list[str]: ...


def _mapping_list(
    mapping: Mapping[str, object],
    key: str,
) -> tuple[Mapping[str, object], ...]:
    value = mapping.get(key)
    if not isinstance(value, list):
        raise ModelRegistryError(f"manifest {key} must be a list of mappings")
    items = cast(list[object], value)
    if any(not isinstance(item, Mapping) for item in items):
        raise ModelRegistryError(f"manifest {key} must be a list of mappings")
    return tuple(cast(Mapping[str, object], item) for item in items)


def _required_text(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ModelRegistryError(f"manifest {key} must be a non-empty string")
    return value


def _manifest_from_mapping(raw: object) -> ModelManifest:
    if not isinstance(raw, Mapping):
        raise ModelRegistryError("model manifest root must be a mapping")
    manifest = cast(Mapping[str, object], raw)
    try:
        artifacts = tuple(
            ModelArtifact(
                filename=_required_text(item, "filename"),
                sha256=_required_text(item, "sha256"),
                url=_required_text(item, "url"),
                duration=_required_text(item, "duration"),
                kind=_required_text(item, "kind"),
            )
            for item in _mapping_list(manifest, "artifacts")
        )
        licenses = tuple(
            ModelLicense(
                component=_required_text(item, "component"),
                license=_required_text(item, "license"),
                attribution=_required_text(item, "attribution"),
            )
            for item in _mapping_list(manifest, "licenses")
        )
        attribution = manifest.get("attribution", "")
        if not isinstance(attribution, str):
            raise ModelRegistryError("manifest attribution must be a string")
        return ModelManifest(
            model_id=_required_text(manifest, "model_id"),
            provider=_required_text(manifest, "provider"),
            version=_required_text(manifest, "version"),
            precision=_required_text(manifest, "precision"),
            artifacts=artifacts,
            licenses=licenses,
            attribution=attribution,
        )
    except (TypeError, ValueError) as error:
        raise ModelRegistryError("invalid model manifest schema") from error


def load_manifest(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    manifest_path: Path | None = None,
) -> ModelManifest:
    """Load one packaged manifest, rejecting an ID mismatch."""

    path = manifest_path or _manifest_path(model_id)
    try:
        raw: object = cast(object, yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError) as error:
        raise ModelRegistryError(f"unable to load model manifest: {path}") from error
    manifest = _manifest_from_mapping(raw)
    if manifest.model_id != model_id:
        raise ModelRegistryError(
            f"manifest ID {manifest.model_id!r} does not match {model_id!r}"
        )
    return manifest


def _probe_runtime(provider: str) -> RuntimeStatus:
    try:
        runtime = cast(_RuntimeModule, import_module("onnxruntime"))
    except ImportError:
        return RuntimeStatus(
            state=RuntimeState.MISSING,
            provider=provider,
            detail="install saliency-cpu or saliency-directml extra",
        )

    try:
        available = tuple(runtime.get_available_providers())
    except Exception:
        return RuntimeStatus(
            state=RuntimeState.MISSING,
            provider=provider,
            detail="ONNX Runtime provider query failed",
        )

    requested = provider.lower()
    if requested == "auto":
        if os.name == "nt" and "DmlExecutionProvider" in available:
            requested_provider = "DmlExecutionProvider"
        else:
            requested_provider = "CPUExecutionProvider"
    elif requested == "cpu":
        requested_provider = "CPUExecutionProvider"
    elif requested == "directml":
        requested_provider = "DmlExecutionProvider"
    else:
        return RuntimeStatus(
            state=RuntimeState.UNSUPPORTED,
            provider=provider,
            detail=f"unknown execution provider: {provider}",
        )
    if requested_provider not in available:
        return RuntimeStatus(
            state=RuntimeState.UNSUPPORTED,
            provider=requested_provider,
            detail=f"available providers: {', '.join(available) or 'none'}",
        )
    return RuntimeStatus(state=RuntimeState.READY, provider=requested_provider)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    content = secure_read_bytes(path, "model artifact")
    for offset in range(0, len(content), _CHUNK_SIZE):
        digest.update(content[offset : offset + _CHUNK_SIZE])
    return digest.hexdigest()


class ModelRegistry:
    """Manage one versioned model manifest in a content-addressed store."""

    def __init__(
        self,
        *,
        model_home: Path | None = None,
        manifest: ModelManifest | None = None,
        manifest_path: Path | None = None,
        runtime_probe: RuntimeProbe = _probe_runtime,
    ) -> None:
        if manifest is not None and manifest_path is not None:
            raise ValueError("provide manifest or manifest_path, not both")
        self.model_home = Path(model_home or model_home_path()).expanduser()
        self.manifest = manifest or load_manifest(manifest_path=manifest_path)
        self._runtime_probe = runtime_probe

    def _require_model(self, model_id: str) -> None:
        if model_id != self.manifest.model_id:
            raise ModelRegistryError(
                f"unsupported model {model_id!r}; expected {self.manifest.model_id!r}"
            )

    def _require_precision(self, precision: str) -> None:
        if precision != self.manifest.precision:
            raise ModelRegistryError(
                f"unsupported precision {precision!r}; expected {self.manifest.precision!r}"
            )

    def artifact_path(self, artifact: ModelArtifact) -> Path:
        """Return content-addressed path for one manifest artifact."""

        return (
            self.model_home
            / self.manifest.provider
            / self.manifest.version
            / self.manifest.precision
            / artifact.sha256
            / artifact.filename
        )

    @property
    def _release_root(self) -> Path:
        return (
            self.model_home
            / self.manifest.provider
            / self.manifest.version
            / self.manifest.precision
        )

    @property
    def _release_manifest_path(self) -> Path:
        return self._release_root / "manifest.json"

    def _release_manifest_payload(self) -> dict[str, object]:
        return {
            "model_id": self.manifest.model_id,
            "provider": self.manifest.provider,
            "version": self.manifest.version,
            "precision": self.manifest.precision,
            "artifacts": [
                {
                    "filename": artifact.filename,
                    "sha256": artifact.sha256,
                    "duration": artifact.duration,
                    "kind": artifact.kind,
                }
                for artifact in self.manifest.artifacts
            ],
        }

    def _release_manifest_is_valid(self) -> bool:
        try:
            payload = json.loads(
                secure_read_bytes(self._release_manifest_path, "release manifest")
            )
        except (BundleStateError, OSError, json.JSONDecodeError):
            return False
        return payload == self._release_manifest_payload()

    @staticmethod
    def _write_release_manifest(path: Path, payload: Mapping[str, object]) -> None:
        try:
            secure_write_bytes(
                path,
                (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"),
            )
        except (BundleStateError, OSError) as error:
            raise ModelRegistryError(
                f"unable to write release manifest: {path}"
            ) from error

    def _artifact_status(self, artifact: ModelArtifact) -> ArtifactStatus:
        path = self.artifact_path(artifact)
        if not path.is_file():
            return ArtifactStatus(
                artifact=artifact, path=path, state=ArtifactState.MISSING
            )
        try:
            actual = _sha256_file(path)
        except OSError:
            return ArtifactStatus(
                artifact=artifact,
                path=path,
                state=ArtifactState.CHECKSUM_MISMATCH,
            )
        if actual != artifact.sha256:
            return ArtifactStatus(
                artifact=artifact,
                path=path,
                state=ArtifactState.CHECKSUM_MISMATCH,
                actual_sha256=actual,
            )
        return ArtifactStatus(
            artifact=artifact,
            path=path,
            state=ArtifactState.READY,
            actual_sha256=actual,
        )

    def verify_artifact(self, artifact: ModelArtifact) -> ArtifactStatus:
        """Recheck one manifest artifact and return its verified descriptor."""

        if artifact not in self.manifest.artifacts:
            raise ModelRegistryError(
                f"artifact is not part of manifest: {artifact.filename}"
            )
        status = self._artifact_status(artifact)
        if status.state is ArtifactState.CHECKSUM_MISMATCH:
            raise ChecksumMismatchError(
                f"checksum mismatch for {artifact.filename}: "
                f"expected {artifact.sha256}, got {status.actual_sha256 or 'missing'}"
            )
        if status.state is ArtifactState.MISSING:
            raise ModelRegistryError(f"artifact missing: {artifact.filename}")
        return status

    def verified_model_artifacts(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        precision: str = DEFAULT_PRECISION,
    ) -> tuple[ArtifactStatus, ...]:
        """Return all pinned model artifacts after checking installed bytes."""

        self._require_model(model_id)
        self._require_precision(precision)
        statuses = tuple(
            self.verify_artifact(artifact)
            for artifact in self.manifest.artifacts
            if artifact.kind == "model"
        )
        if not statuses:
            raise ModelRegistryError("manifest contains no model artifacts")
        return statuses

    def status(self, model_id: str, *, provider: str = "cpu") -> RegistryStatus:
        """Inspect runtime and local artifacts without making network calls."""

        self._require_model(model_id)
        runtime = self._runtime_probe(provider)
        artifacts = tuple(
            self._artifact_status(artifact) for artifact in self.manifest.artifacts
        )
        if runtime.state is RuntimeState.MISSING:
            state = ModelState.RUNTIME_MISSING
        elif runtime.state is RuntimeState.UNSUPPORTED:
            state = ModelState.UNSUPPORTED_PROVIDER
        elif any(item.state is ArtifactState.CHECKSUM_MISMATCH for item in artifacts):
            state = ModelState.CHECKSUM_MISMATCH
        elif (
            any(item.state is ArtifactState.MISSING for item in artifacts)
            or not self._release_manifest_is_valid()
        ):
            state = ModelState.ARTIFACT_MISSING
        else:
            state = ModelState.READY
        diagnostics = [state.value]
        if runtime.detail:
            diagnostics.append(runtime.detail)
        if (
            state is ModelState.ARTIFACT_MISSING
            and not self._release_manifest_is_valid()
        ):
            diagnostics.append("release manifest missing or invalid")
        return RegistryStatus(
            model_id=model_id,
            state=state,
            runtime=runtime,
            artifacts=artifacts,
            diagnostics=tuple(diagnostics),
            attribution=self.manifest.attribution_text,
        )

    def install(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        precision: str = DEFAULT_PRECISION,
    ) -> InstallResult:
        """Download and atomically install every pinned model/parity artifact."""

        self._require_model(model_id)
        self._require_precision(precision)
        downloaded: list[str] = []
        skipped: list[str] = []
        existing = {
            artifact.filename: self._artifact_status(artifact)
            for artifact in self.manifest.artifacts
        }
        if (
            all(status.state is ArtifactState.READY for status in existing.values())
            and self._release_manifest_is_valid()
        ):
            return InstallResult(
                model_id=model_id,
                downloaded=(),
                skipped=tuple(
                    artifact.filename for artifact in self.manifest.artifacts
                ),
                attribution=self.manifest.attribution_text,
            )

        release_root = self._release_root
        secure_ensure_directory(release_root, "model release root")
        staging_root = secure_make_temporary_directory(
            release_root, ".staging-", "model staging root"
        )
        try:
            for artifact in self.manifest.artifacts:
                staged_target = staging_root / artifact.sha256 / artifact.filename
                status = existing[artifact.filename]
                if status.state is ArtifactState.READY:
                    secure_ensure_directory(
                        staged_target.parent, "staged model artifact parent"
                    )
                    try:
                        secure_write_bytes(
                            staged_target,
                            secure_read_bytes(status.path, "installed model artifact"),
                        )
                    except (BundleStateError, OSError) as error:
                        raise ModelRegistryError(
                            f"unable to stage {artifact.filename}: {error}"
                        ) from error
                    if _sha256_file(staged_target) != artifact.sha256:
                        raise ChecksumMismatchError(
                            f"checksum mismatch for {artifact.filename}"
                        )
                    skipped.append(artifact.filename)
                else:
                    self._download(artifact, staged_target)
                    downloaded.append(artifact.filename)

            self._write_release_manifest(
                staging_root / "manifest.json", self._release_manifest_payload()
            )
            self._publish_staged_release(staging_root)
        finally:
            try:
                secure_remove_tree(
                    staging_root, "model staging cleanup", missing_ok=True
                )
            except (BundleStateError, OSError):
                pass
        return InstallResult(
            model_id=model_id,
            downloaded=tuple(downloaded),
            skipped=tuple(skipped),
            attribution=self.manifest.attribution_text,
        )

    def _publish_staged_release(self, staging_root: Path) -> None:
        """Publish staged files with manifest as complete-release commit marker."""

        moved: list[Path] = []
        backups: list[tuple[Path, Path]] = []
        backup_root = staging_root / ".backups"
        previous_manifest: Path | None = None
        manifest_published = False
        try:
            secure_assert_ancestors(self._release_root, "model release root")
            if secure_is_link_or_reparse(self._release_root):
                raise ModelRegistryError(
                    "model release root must not be a link or reparse point"
                )
            for index, artifact in enumerate(self.manifest.artifacts):
                staged_target = staging_root / artifact.sha256 / artifact.filename
                target = self.artifact_path(artifact)
                if target.is_file() and _sha256_file(target) == artifact.sha256:
                    continue
                secure_ensure_directory(target.parent, "model artifact parent")
                if target.exists() or target.is_symlink():
                    backup = backup_root / f"artifact-{index}"
                    secure_ensure_directory(backup.parent, "model backup parent")
                    secure_replace(target, backup, "model artifact backup")
                    backups.append((target, backup))
                secure_replace(staged_target, target, "model artifact publication")
                moved.append(target)

            if self._release_manifest_path.exists():
                previous_manifest = backup_root / "manifest.json"
                secure_ensure_directory(previous_manifest.parent, "model backup parent")
                secure_replace(
                    self._release_manifest_path,
                    previous_manifest,
                    "model manifest backup",
                )
            secure_replace(
                staging_root / "manifest.json",
                self._release_manifest_path,
                "model manifest publication",
            )
            manifest_published = True
        except (BundleStateError, OSError) as error:
            if manifest_published:
                try:
                    secure_unlink(
                        self._release_manifest_path,
                        "published model manifest rollback",
                        missing_ok=True,
                    )
                except (BundleStateError, OSError):
                    pass
            for target in reversed(moved):
                try:
                    secure_unlink(
                        target, "published model artifact rollback", missing_ok=True
                    )
                except (BundleStateError, OSError):
                    pass
            for target, backup in reversed(backups):
                try:
                    secure_replace(backup, target, "model artifact rollback")
                except (BundleStateError, OSError):
                    pass
            if previous_manifest is not None:
                try:
                    secure_replace(
                        previous_manifest,
                        self._release_manifest_path,
                        "model manifest rollback",
                    )
                except (BundleStateError, OSError):
                    pass
            raise ModelRegistryError(f"release publish failed: {error}") from error

    def _download(self, artifact: ModelArtifact, target: Path) -> None:
        secure_ensure_directory(target.parent, "staged download parent")
        digest = hashlib.sha256()
        try:
            with urlopen(artifact.url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:

                def chunks() -> Iterator[bytes]:
                    while chunk := response.read(_CHUNK_SIZE):
                        digest.update(chunk)
                        yield chunk

                secure_write_chunks(target, chunks(), "model download")
            actual = digest.hexdigest()
            if actual != artifact.sha256:
                raise ChecksumMismatchError(
                    f"checksum mismatch for {artifact.filename}: "
                    f"expected {artifact.sha256}, got {actual}"
                )
        except ChecksumMismatchError:
            raise
        except OSError as error:
            raise ModelRegistryError(
                f"download failed for {artifact.filename}: {error}"
            ) from error
        finally:
            if digest.hexdigest() != artifact.sha256:
                try:
                    secure_unlink(
                        target, "partial model download cleanup", missing_ok=True
                    )
                except (BundleStateError, OSError):
                    pass

    def remove(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        precision: str = DEFAULT_PRECISION,
    ) -> RemoveResult:
        """Remove only this manifest's files and preserve unrelated store data."""

        self._require_model(model_id)
        self._require_precision(precision)
        try:
            secure_assert_ancestors(self._release_root, "model release root")
            for artifact in self.manifest.artifacts:
                secure_assert_ancestors(self.artifact_path(artifact), "model artifact")
        except BundleStateError as error:
            raise ModelRegistryError(str(error)) from error
        removed: list[str] = []
        try:
            if (
                self._release_manifest_path.is_file()
                or self._release_manifest_path.is_symlink()
            ):
                secure_unlink(self._release_manifest_path, "model release manifest")
            for artifact in self.manifest.artifacts:
                path = self.artifact_path(artifact)
                if path.is_file() or path.is_symlink():
                    secure_unlink(path, "model artifact")
                    removed.append(artifact.filename)
                self._remove_empty_parents(path.parent)
        except BundleStateError as error:
            raise ModelRegistryError(str(error)) from error
        return RemoveResult(model_id=model_id, removed=tuple(removed))

    @staticmethod
    def _remove_empty_parents(start: Path) -> None:
        for directory in (
            start,
            start.parent,
            start.parent.parent,
            start.parent.parent.parent,
        ):
            try:
                secure_rmdir(directory, "empty model store directory")
            except (BundleStateError, OSError):
                break


def model_home_path() -> Path:
    """Backward-compatible named resolver for callers that prefer explicit paths."""

    return model_home()


def install_model(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    precision: str = DEFAULT_PRECISION,
    model_home: Path | None = None,
) -> InstallResult:
    """Install one packaged model manifest."""

    return ModelRegistry(model_home=model_home).install(model_id, precision=precision)


def status_model(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    provider: str = "cpu",
    model_home: Path | None = None,
) -> RegistryStatus:
    """Inspect one packaged model manifest."""

    return ModelRegistry(model_home=model_home).status(model_id, provider=provider)


def remove_model(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    precision: str = DEFAULT_PRECISION,
    model_home: Path | None = None,
) -> RemoveResult:
    """Remove one packaged model manifest."""

    return ModelRegistry(model_home=model_home).remove(model_id, precision=precision)


__all__ = [
    "ArtifactState",
    "ArtifactStatus",
    "ChecksumMismatchError",
    "DEFAULT_MODEL_ID",
    "DEFAULT_PRECISION",
    "InstallResult",
    "ModelArtifact",
    "ModelLicense",
    "ModelManifest",
    "ModelRegistry",
    "ModelRegistryError",
    "ModelState",
    "RegistryStatus",
    "RemoveResult",
    "RuntimeState",
    "RuntimeStatus",
    "install_model",
    "load_manifest",
    "model_home",
    "model_home_path",
    "remove_model",
    "status_model",
]
