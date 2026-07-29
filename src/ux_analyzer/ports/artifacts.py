"""Port contracts for immutable benchmark run artifacts."""

from __future__ import annotations

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
