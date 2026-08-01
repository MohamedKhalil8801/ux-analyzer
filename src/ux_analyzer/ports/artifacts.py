"""Port contracts for immutable benchmark run artifacts."""

from __future__ import annotations

import io
import struct
import zipfile
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import urlsplit

from ux_analyzer import __version__
from ux_analyzer.domain.benchmark import FixtureInputs
from ux_analyzer.domain.run import ProviderManifest, RunSpec

REDACTED_VALUE = "[REDACTED]"


def _empty_string_mapping() -> dict[str, str]:
    return {}


class BundleError(RuntimeError):
    """Base error for run bundle lifecycle failures."""


class BundleAlreadyFinalizedError(BundleError):
    """Raised when a finalized bundle is mutated."""


class BundleStateError(BundleError):
    """Raised when a bundle operation is invalid for its current state."""


@dataclass(frozen=True, slots=True)
class RedactionPolicy:
    """Exact values and mapping keys that must be redacted before JSON output."""

    exact_values: tuple[str, ...] = ()
    keys: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        values = tuple(self.exact_values)
        keys = frozenset(self.keys)
        if any(not value for value in values):
            raise ValueError("redaction exact values must not be empty")
        if any(not key for key in keys):
            raise ValueError("redaction keys must not be empty")
        object.__setattr__(self, "exact_values", values)
        object.__setattr__(self, "keys", keys)

    @classmethod
    def from_fixture_inputs(cls, inputs: FixtureInputs) -> RedactionPolicy:
        """Build policy from scenario values marked sensitive."""

        return cls(
            exact_values=tuple(
                inputs.values[key] for key in sorted(inputs.sensitive_keys)
            ),
            keys=inputs.sensitive_keys,
        )


def sanitize_artifact_content(
    name: str,
    content: bytes,
    policy: RedactionPolicy,
) -> bytes:
    """Remove configured values from persisted screenshots and trace archives."""

    if not policy.exact_values:
        return content
    lowered_name = name.lower()
    if lowered_name.endswith(".zip") or zipfile.is_zipfile(io.BytesIO(content)):
        return _sanitize_zip(content, policy)
    if lowered_name.endswith(".png") or content.startswith(b"\x89PNG\r\n\x1a\n"):
        return _blank_png(content)
    return _replace_exact_bytes(content, policy)


def _sanitize_zip(content: bytes, policy: RedactionPolicy) -> bytes:
    source_buffer = io.BytesIO(content)
    destination_buffer = io.BytesIO()
    try:
        with (
            zipfile.ZipFile(source_buffer, "r") as source,
            zipfile.ZipFile(destination_buffer, "w") as destination,
        ):
            for info in source.infolist():
                member = source.read(info.filename)
                destination.writestr(
                    info,
                    sanitize_artifact_content(info.filename, member, policy),
                )
    except zipfile.BadZipFile:
        return _replace_exact_bytes(content, policy)
    return destination_buffer.getvalue()


def _blank_png(content: bytes) -> bytes:
    if len(content) < 24 or not content.startswith(b"\x89PNG\r\n\x1a\n"):
        return b""
    width, height = struct.unpack(">II", content[16:24])
    if width <= 0 or height <= 0 or width > 16_384 or height > 16_384:
        return b""
    row = b"\x00" + (b"\x00\x00\x00\xff" * width)
    pixels = zlib.compress(row * height)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + _png_chunk(b"IDAT", pixels)
        + _png_chunk(b"IEND", b"")
    )


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)
    )


def _replace_exact_bytes(content: bytes, policy: RedactionPolicy) -> bytes:
    redacted = content
    replacement = REDACTED_VALUE.encode("utf-8")
    for exact_value in sorted(policy.exact_values, key=len, reverse=True):
        for encoding in ("utf-8", "utf-16-le", "utf-16-be"):
            redacted = redacted.replace(exact_value.encode(encoding), replacement)
    return redacted


def _origin_without_credentials(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("endpoint origin must not contain credentials")
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return endpoint.split("?", maxsplit=1)[0].split("#", maxsplit=1)[0].rstrip("/")


@dataclass(frozen=True, slots=True)
class BundleManifest:
    """Reproducibility metadata written to ``manifest.json``."""

    run_id: str
    seed: int
    config_digest: str
    endpoint_origin: str
    scenario_id: str | None = None
    application_version_id: str | None = None
    persona_id: str | None = None
    policy: str | None = None
    model_ids: Mapping[str, str] = field(default_factory=_empty_string_mapping)
    prompt_versions: Mapping[str, str] = field(default_factory=_empty_string_mapping)
    package_version: str = __version__
    provider_versions: Mapping[str, str] = field(default_factory=_empty_string_mapping)
    provider_manifests: tuple[ProviderManifest, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("run_id", self.run_id),
            ("config_digest", self.config_digest),
            ("endpoint_origin", self.endpoint_origin),
            ("package_version", self.package_version),
        ):
            if not value:
                raise ValueError(f"{name} must not be empty")
        object.__setattr__(
            self, "endpoint_origin", _origin_without_credentials(self.endpoint_origin)
        )
        object.__setattr__(
            self, "model_ids", MappingProxyType(dict[str, str](self.model_ids))
        )
        object.__setattr__(
            self,
            "prompt_versions",
            MappingProxyType(dict[str, str](self.prompt_versions)),
        )
        object.__setattr__(
            self,
            "provider_versions",
            MappingProxyType(dict[str, str](self.provider_versions)),
        )
        object.__setattr__(self, "provider_manifests", tuple(self.provider_manifests))

    @classmethod
    def from_run_spec(
        cls,
        run_spec: RunSpec,
        *,
        endpoint_origin: str,
        model_ids: Mapping[str, str] | None = None,
        prompt_versions: Mapping[str, str] | None = None,
        package_version: str = __version__,
        provider_versions: Mapping[str, str] | None = None,
        provider_manifests: tuple[ProviderManifest, ...] = (),
    ) -> BundleManifest:
        """Create bundle metadata without leaking infrastructure into ``RunSpec``."""

        return cls(
            run_id=run_spec.run_id,
            seed=run_spec.seed,
            config_digest=run_spec.config_digest,
            endpoint_origin=endpoint_origin,
            scenario_id=run_spec.scenario.id,
            application_version_id=run_spec.application_version.id,
            persona_id=run_spec.persona.id,
            policy=run_spec.policy.value,
            model_ids=model_ids or {},
            prompt_versions=prompt_versions or {},
            package_version=package_version,
            provider_versions=provider_versions or {},
            provider_manifests=provider_manifests,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible manifest data."""

        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "config_digest": self.config_digest,
            "endpoint_origin": self.endpoint_origin,
            "scenario_id": self.scenario_id,
            "application_version_id": self.application_version_id,
            "persona_id": self.persona_id,
            "policy": self.policy,
            "model_ids": dict(self.model_ids),
            "prompt_versions": dict(self.prompt_versions),
            "package_version": self.package_version,
            "provider_versions": dict(self.provider_versions),
            "provider_manifests": [
                {
                    "provider_id": manifest.provider_id,
                    "role": manifest.role,
                    "model_id": manifest.model_id,
                    "endpoint_origin": _origin_without_credentials(
                        manifest.endpoint_origin
                    ),
                    "version": manifest.version,
                    "prompt_version": manifest.prompt_version,
                    "schema_version": manifest.schema_version,
                }
                for manifest in self.provider_manifests
            ],
        }


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Content-addressed artifact stored inside one run bundle."""

    path: str
    sha256: str
    size: int
    name: str


class RunBundleWriter(Protocol):
    """Append-only run bundle writer implemented by storage adapters."""

    @classmethod
    def start(
        cls,
        output_dir: Path,
        manifest: BundleManifest,
        *,
        redaction: RedactionPolicy | None = None,
    ) -> RunBundleWriter:
        """Create a new staging bundle."""

        ...

    @property
    def run_id(self) -> str:
        """Return immutable run identity."""

        ...

    def append_event(self, event: object) -> int:
        """Append one redacted event and return its monotonic sequence."""

        ...

    def write_artifact(self, name: str, content: bytes | str) -> ArtifactReference:
        """Write or reuse one content-addressed artifact."""

        ...

    def finalize(self, result: object) -> Path:
        """Write terminal files and atomically publish bundle."""

        ...

    def abort(self, reason: str) -> Path:
        """Leave staging bundle with crash marker for recovery."""

        ...
