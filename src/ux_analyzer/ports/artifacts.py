"""Port contracts for immutable benchmark run artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import struct
import zipfile
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Protocol, cast
from urllib.parse import urlsplit

from ux_analyzer import __version__
from ux_analyzer.domain.benchmark import FixtureInputs
from ux_analyzer.domain.run import ProviderManifest, RunSpec

REDACTED_VALUE = "[REDACTED]"

_SALIENCY_ARTIFACT_FILENAMES = frozenset(
    {
        "1s.npz",
        "3s.npz",
        "7s.npz",
        "1s-heatmap.png",
        "3s-heatmap.png",
        "7s-heatmap.png",
        "profiles.json",
        "metadata.json",
    }
)
_BEARER_TOKEN_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_CREDENTIAL_URL_PATTERN = re.compile(
    r"(?i)\b[a-z][a-z0-9+.-]*://[^\s/@:]+:[^\s/@]+@[^\s]+"
)
_FIXTURE_SECRET_PATTERN = re.compile(
    r"(?i)\bfixture[-_ ]?(?:secret|token|password|credential)"
    r"(?:[-_ ]?(?:key|value))?(?:\s*[:=]\s*)?[^\s,;]*"
)
_WINDOWS_PATH_PATTERN = re.compile(r"(?<![\w])(?:[A-Za-z]:[\\/]|\\\\)[^\s\"']+")
_POSIX_SENSITIVE_PATH_PATTERN = re.compile(
    r"(?<![\w])/(?:Users|home|tmp|var|private|mnt|opt|workspace|workspaces)"
    r"(?:/[^\s\"']*)?"
)
_SELECTOR_PATTERN = re.compile(r"\[[^\]\r\n]{1,200}\]")
_SENSITIVE_TEXT_MARKER_PATTERN = re.compile(
    r"(?i)(?:^|[^a-z0-9])"
    r"(?:access[\s_-]*token|api[\s_-]*key|authorization|credential|password|secret|token)"
    r"(?:$|[^a-z0-9])"
)
_DEFAULT_SENSITIVE_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "credential",
        "fixture_credential",
        "fixture_password",
        "fixture_secret",
        "fixture_token",
        "password",
        "secret",
        "token",
        "x_api_key",
    }
)
_MAX_SALIENCY_WARNING_LENGTH = 512


def required_saliency_artifact_paths(viewport_id: str) -> frozenset[str]:
    """Return exact eight generated artifact paths for one viewport."""

    validate_saliency_artifact_path(f"saliency/{viewport_id}/metadata.json")
    return frozenset(
        f"saliency/{viewport_id}/{filename}"
        for filename in _SALIENCY_ARTIFACT_FILENAMES
    )


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


def sanitize_log_text(value: str, policy: RedactionPolicy) -> str:
    """Redact configured values and credential/path patterns from log text."""

    redacted = value
    assignment_pattern = _sensitive_assignment_pattern(policy)
    for pattern in (
        _CREDENTIAL_URL_PATTERN,
        _BEARER_TOKEN_PATTERN,
        assignment_pattern,
        _FIXTURE_SECRET_PATTERN,
        _WINDOWS_PATH_PATTERN,
        _POSIX_SENSITIVE_PATH_PATTERN,
        _SELECTOR_PATTERN,
    ):
        redacted = pattern.sub(REDACTED_VALUE, redacted)
    for exact_value in sorted(policy.exact_values, key=len, reverse=True):
        redacted = redacted.replace(exact_value, REDACTED_VALUE)
    return redacted


def _canonical_key(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[\s_-]+", "_", value.strip().casefold())


def _sensitive_assignment_pattern(policy: RedactionPolicy) -> re.Pattern[str]:
    """Match complete sensitive assignments, including quoted spaced values."""

    fragments: set[str] = set()
    for raw_key in _DEFAULT_SENSITIVE_KEYS | {
        _canonical_key(key) for key in policy.keys
    }:
        key = _canonical_key(raw_key)
        if key:
            fragments.add(r"[\s_-]*".join(re.escape(part) for part in key.split("_")))
    alternatives = "|".join(sorted(fragments, key=len, reverse=True))
    return re.compile(
        rf"(?i)(?<!\w)[\"']?(?:{alternatives})[\"']?\s*[:=]\s*"
        r"(?:\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|[^\r\n,;]*)"
    )


def is_sensitive_key(value: object, policy: RedactionPolicy) -> bool:
    """Return whether mapping key belongs to built-in or caller policy."""

    canonical = _canonical_key(value)
    configured = {_canonical_key(key) for key in policy.keys}
    return canonical in _DEFAULT_SENSITIVE_KEYS | configured


def _contains_sensitive_text(value: str, policy: RedactionPolicy) -> bool:
    return (
        sanitize_log_text(value, policy) != value
        or _SENSITIVE_TEXT_MARKER_PATTERN.search(value) is not None
    )


def canonicalize_saliency_artifact_content(
    kind: SaliencyArtifactKind,
    content: bytes,
    *,
    redaction: RedactionPolicy | None = None,
    expected_viewport_id: str | None = None,
) -> bytes:
    """Validate generated saliency bytes before checksum publication."""

    if type(content) is not bytes:
        raise TypeError("saliency artifact content must be bytes")
    if kind is SaliencyArtifactKind.PROFILES:
        return _canonicalize_saliency_json(
            content,
            kind,
            policy=redaction or RedactionPolicy(),
            expected_viewport_id=expected_viewport_id,
        )
    if kind is SaliencyArtifactKind.METADATA:
        return _canonicalize_saliency_json(
            content,
            kind,
            policy=redaction or RedactionPolicy(),
            expected_viewport_id=expected_viewport_id,
        )
    return content


_SALiency_PROFILE_KEYS = frozenset(
    {
        "viewport_id",
        "element_id",
        "immediate",
        "early",
        "eventual",
        "general",
        "aggregates",
        "aggregation_version",
        "prediction_provenance",
    }
)
_SALiency_ESTIMATE_KEYS = frozenset({"kind", "score", "source"})
_SALiency_AGGREGATE_KEYS = frozenset(
    {
        "viewport_id",
        "element_id",
        "duration",
        "density",
        "robust_peak",
        "raw_mass",
        "mass_share",
        "clipped_area",
        "visibility_fraction",
        "occlusion_fraction",
        "raw_score",
        "adjusted_score",
    }
)
_SALiency_PROVENANCE_KEYS = frozenset({"duration", "metadata"})
_SALiency_GEOMETRY_KEYS = frozenset(
    {
        "geometry_version",
        "source_dimensions",
        "native_dimensions",
        "content_dimensions",
        "pad_left",
        "pad_top",
        "pad_right",
        "pad_bottom",
        "scale",
        "scale_x",
        "scale_y",
        "device_pixel_ratio",
        "zoom",
    }
)
_SALiency_PREDICTION_METADATA_KEYS = frozenset(
    {
        "provider_id",
        "model_id",
        "provider_version",
        "model_version",
        "model_checksum",
        "input_dimensions",
        "output_dimensions",
        "geometry",
        "preprocessing_version",
        "inference_duration_ms",
        "execution_provider",
        "warnings",
        "cache_state",
    }
)
_SALiency_CACHE_KEY_KEYS = frozenset(
    {
        "viewport_id",
        "screenshot_sha256",
        "screenshot_dimensions",
        "device_pixel_ratio",
        "zoom",
        "model_checksums",
        "preprocessing_version",
        "precision",
        "execution_provider",
        "aggregation_version",
        "geometry_version",
    }
)
_SALiency_METADATA_KEYS = frozenset(
    {
        "cache_version",
        "cache_key",
        "cache_key_digest",
        "viewport_id",
        "aggregation_version",
        "predictions",
        "saliency_metadata",
        "warnings",
        "artifact_paths",
    }
)
_SALiency_PREDICTION_KEYS = frozenset({"duration", "metadata"})
_SALiency_METADATA_RECORD_KEYS = frozenset(
    {"provider_manifests", "aggregation_version", "warnings"}
)
_SALiency_DURATIONS = frozenset({"1s", "3s", "7s"})
_SALiency_ESTIMATE_KINDS = frozenset(
    {"predicted", "derived", "fallback", "unavailable"}
)


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, item in pairs:
        if key in result:
            raise ValueError("saliency JSON contains duplicate fields")
        result[key] = item
    return result


def _json_mapping(
    value: object,
    *,
    allowed: frozenset[str],
    required: frozenset[str],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    mapping = cast(Mapping[str, object], value)
    unknown = set(mapping) - allowed
    missing = required - set(mapping)
    if unknown:
        raise ValueError(
            f"{name} contains unsupported (not allowlisted) fields: "
            + ", ".join(sorted(str(item) for item in unknown))
        )
    if missing:
        raise ValueError(f"{name} is missing fields: " + ", ".join(sorted(missing)))
    return mapping


def _json_text(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _json_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite number")
    return number


def _json_dimensions(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must contain two positive integers")
    dimensions = cast(list[object], value)
    if len(dimensions) != 2:
        raise ValueError(f"{name} must contain two positive integers")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in dimensions
    ):
        raise ValueError(f"{name} must contain two positive integers")
    return cast(tuple[int, int], tuple(dimensions))


def _json_sha256(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")


def _validate_json_warnings(value: object, name: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    warnings = cast(list[object], value)
    if any(
        type(item) is not str
        or not item.strip()
        or len(item) > _MAX_SALIENCY_WARNING_LENGTH
        or any(character in item for character in "\x00\r\n")
        for item in warnings
    ):
        raise ValueError(f"{name} must contain bounded single-line strings")


def _validate_saliency_geometry(value: object, name: str) -> None:
    geometry = _json_mapping(
        value,
        allowed=_SALiency_GEOMETRY_KEYS,
        required=_SALiency_GEOMETRY_KEYS,
        name=name,
    )
    _json_text(geometry["geometry_version"], f"{name}.geometry_version")
    _json_dimensions(geometry["source_dimensions"], f"{name}.source_dimensions")
    native_dimensions = _json_dimensions(
        geometry["native_dimensions"], f"{name}.native_dimensions"
    )
    content_dimensions = _json_dimensions(
        geometry["content_dimensions"], f"{name}.content_dimensions"
    )
    padding: dict[str, int] = {}
    for field_name in (
        "pad_left",
        "pad_top",
        "pad_right",
        "pad_bottom",
    ):
        offset = geometry[field_name]
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError(f"{name}.{field_name} must be non-negative integer")
        padding[field_name] = offset
    if (
        padding["pad_left"] + content_dimensions[0] + padding["pad_right"]
        != native_dimensions[0]
        or padding["pad_top"] + content_dimensions[1] + padding["pad_bottom"]
        != native_dimensions[1]
    ):
        raise ValueError(f"{name} content and padding must fill native dimensions")
    for field_name in (
        "scale",
        "scale_x",
        "scale_y",
        "device_pixel_ratio",
        "zoom",
    ):
        number = _json_number(geometry[field_name], f"{name}.{field_name}")
        if number <= 0:
            raise ValueError(f"{name}.{field_name} must be greater than zero")


def _validate_saliency_prediction_metadata(value: object, name: str) -> None:
    metadata = _json_mapping(
        value,
        allowed=_SALiency_PREDICTION_METADATA_KEYS,
        required=_SALiency_PREDICTION_METADATA_KEYS,
        name=name,
    )
    for field_name in (
        "provider_id",
        "model_id",
        "provider_version",
        "model_version",
        "preprocessing_version",
        "execution_provider",
    ):
        _json_text(metadata[field_name], f"{name}.{field_name}")
    _json_sha256(metadata["model_checksum"], f"{name}.model_checksum")
    input_dimensions = _json_dimensions(
        metadata["input_dimensions"], f"{name}.input_dimensions"
    )
    _json_dimensions(metadata["output_dimensions"], f"{name}.output_dimensions")
    _validate_saliency_geometry(metadata["geometry"], f"{name}.geometry")
    geometry = cast(Mapping[str, object], metadata["geometry"])
    if tuple(cast(list[object], geometry["native_dimensions"])) != input_dimensions:
        raise ValueError(
            f"{name} geometry native dimensions must match input dimensions"
        )
    inference_duration = _json_number(
        metadata["inference_duration_ms"], f"{name}.inference_duration_ms"
    )
    if inference_duration < 0:
        raise ValueError(f"{name}.inference_duration_ms must be non-negative duration")
    _validate_json_warnings(metadata["warnings"], f"{name}.warnings")
    if metadata["cache_state"] != "miss":
        raise ValueError(f"{name}.cache_state must preserve miss provenance")


def _validate_saliency_profiles(
    value: object,
    *,
    expected_viewport_id: str | None = None,
) -> None:
    if not isinstance(value, list):
        raise ValueError("profiles JSON must be an allowlisted list")
    profiles = cast(list[object], value)
    element_ids: set[str] = set()
    for profile_index, raw_profile in enumerate(profiles):
        profile = _json_mapping(
            raw_profile,
            allowed=_SALiency_PROFILE_KEYS,
            required=_SALiency_PROFILE_KEYS,
            name=f"profile {profile_index}",
        )
        _json_text(profile["viewport_id"], "profile viewport_id")
        element_id = _json_text(profile["element_id"], "profile element_id")
        if element_id in element_ids:
            raise ValueError("saliency profiles must not contain duplicate profiles")
        element_ids.add(element_id)
        if (
            expected_viewport_id is not None
            and profile["viewport_id"] != expected_viewport_id
        ):
            raise ValueError("profile viewport does not match artifact path")
        for estimate_name in ("immediate", "early", "eventual", "general"):
            estimate = profile[estimate_name]
            if estimate is None:
                continue
            estimate_mapping = _json_mapping(
                estimate,
                allowed=_SALiency_ESTIMATE_KEYS,
                required=_SALiency_ESTIMATE_KEYS,
                name=f"profile {estimate_name}",
            )
            kind = _json_text(estimate_mapping["kind"], "estimate kind")
            if kind not in _SALiency_ESTIMATE_KINDS:
                raise ValueError("estimate kind is invalid")
            score = estimate_mapping["score"]
            if score is not None:
                score_value = _json_number(score, "estimate score")
                if not 0.0 <= score_value <= 1.0:
                    raise ValueError("estimate score must be normalized")
            source = estimate_mapping["source"]
            if source is not None:
                _json_text(source, "estimate source")
        aggregates = profile["aggregates"]
        if not isinstance(aggregates, list):
            raise ValueError("profile aggregates must be a list")
        aggregate_values = cast(list[object], aggregates)
        for aggregate_index, raw_aggregate in enumerate(aggregate_values):
            aggregate = _json_mapping(
                raw_aggregate,
                allowed=_SALiency_AGGREGATE_KEYS,
                required=_SALiency_AGGREGATE_KEYS,
                name=f"profile aggregate {aggregate_index}",
            )
            _json_text(aggregate["viewport_id"], "aggregate viewport_id")
            _json_text(aggregate["element_id"], "aggregate element_id")
            if aggregate["viewport_id"] != profile["viewport_id"]:
                raise ValueError("aggregate viewport does not match profile")
            if aggregate["element_id"] != profile["element_id"]:
                raise ValueError("aggregate element does not match profile")
            duration = _json_text(aggregate["duration"], "aggregate duration")
            if duration not in _SALiency_DURATIONS:
                raise ValueError("aggregate duration is invalid")
            for field_name in _SALiency_AGGREGATE_KEYS - {
                "viewport_id",
                "element_id",
                "duration",
            }:
                _json_number(aggregate[field_name], f"aggregate {field_name}")
        _json_text(profile["aggregation_version"], "profile aggregation_version")
        provenance = profile["prediction_provenance"]
        if not isinstance(provenance, list):
            raise ValueError("profile prediction_provenance must be a list")
        provenance_values = cast(list[object], provenance)
        for provenance_index, raw_provenance in enumerate(provenance_values):
            item = _json_mapping(
                raw_provenance,
                allowed=_SALiency_PROVENANCE_KEYS,
                required=_SALiency_PROVENANCE_KEYS,
                name=f"profile provenance {provenance_index}",
            )
            duration = _json_text(item["duration"], "provenance duration")
            if duration not in _SALiency_DURATIONS:
                raise ValueError("provenance duration is invalid")
            _validate_saliency_prediction_metadata(
                item["metadata"], f"profile provenance {provenance_index}.metadata"
            )


def _validate_saliency_metadata(
    value: object,
    *,
    expected_viewport_id: str | None = None,
) -> None:
    metadata = _json_mapping(
        value,
        allowed=_SALiency_METADATA_KEYS,
        required=_SALiency_METADATA_KEYS,
        name="saliency metadata",
    )
    _json_text(metadata["cache_version"], "cache_version")
    cache_key = _json_mapping(
        metadata["cache_key"],
        allowed=_SALiency_CACHE_KEY_KEYS,
        required=_SALiency_CACHE_KEY_KEYS,
        name="cache_key",
    )
    viewport_id = _json_text(metadata["viewport_id"], "metadata viewport_id")
    if expected_viewport_id is not None and viewport_id != expected_viewport_id:
        raise ValueError("metadata viewport does not match artifact path")
    validate_saliency_artifact_path(f"saliency/{viewport_id}/metadata.json")
    _json_text(cache_key["viewport_id"], "cache_key.viewport_id")
    if cache_key["viewport_id"] != viewport_id:
        raise ValueError("cache_key viewport_id does not match metadata")
    _json_sha256(cache_key["screenshot_sha256"], "cache_key.screenshot_sha256")
    _json_dimensions(
        cache_key["screenshot_dimensions"], "cache_key.screenshot_dimensions"
    )
    for field_name in ("device_pixel_ratio", "zoom"):
        if _json_number(cache_key[field_name], f"cache_key.{field_name}") <= 0:
            raise ValueError(f"cache_key.{field_name} must be greater than zero")
    model_checksums = cache_key["model_checksums"]
    if not isinstance(model_checksums, list):
        raise ValueError("cache_key.model_checksums must contain three checksums")
    checksum_values = cast(list[object], model_checksums)
    if len(checksum_values) != 3:
        raise ValueError("cache_key.model_checksums must contain three checksums")
    for checksum in checksum_values:
        _json_sha256(checksum, "cache_key.model_checksum")
    for field_name in (
        "preprocessing_version",
        "precision",
        "execution_provider",
        "aggregation_version",
        "geometry_version",
    ):
        _json_text(cache_key[field_name], f"cache_key.{field_name}")
    _json_sha256(metadata["cache_key_digest"], "cache_key_digest")
    expected_cache_key_digest = hashlib.sha256(
        json.dumps(dict(cache_key), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if metadata["cache_key_digest"] != expected_cache_key_digest:
        raise ValueError("cache_key_digest does not match cache_key")
    _json_text(metadata["aggregation_version"], "aggregation_version")
    if metadata["aggregation_version"] != cache_key["aggregation_version"]:
        raise ValueError("metadata aggregation_version does not match cache_key")
    _validate_json_warnings(metadata["warnings"], "metadata warnings")
    artifact_paths = metadata["artifact_paths"]
    if not isinstance(artifact_paths, list):
        raise ValueError("metadata artifact_paths must be a list")
    artifact_path_values = cast(list[object], artifact_paths)
    expected_artifact_paths = required_saliency_artifact_paths(viewport_id)
    normalized_artifact_paths: set[str] = set()
    for path in artifact_path_values:
        if not isinstance(path, str):
            raise ValueError("metadata artifact paths must be text")
        normalized_path = validate_saliency_artifact_path(path)
        if normalized_path.parts[1] != viewport_id:
            raise ValueError("metadata artifact path viewport does not match metadata")
        normalized_artifact_paths.add(normalized_path.as_posix())
    if normalized_artifact_paths != set(expected_artifact_paths):
        raise ValueError("metadata artifact paths must cover one viewport exactly")
    predictions = metadata["predictions"]
    if not isinstance(predictions, list):
        raise ValueError("metadata predictions must be a list")
    prediction_values = cast(list[object], predictions)
    prediction_by_duration: dict[str, Mapping[str, object]] = {}
    for prediction_index, raw_prediction in enumerate(prediction_values):
        prediction = _json_mapping(
            raw_prediction,
            allowed=_SALiency_PREDICTION_KEYS,
            required=_SALiency_PREDICTION_KEYS,
            name=f"prediction {prediction_index}",
        )
        duration = _json_text(prediction["duration"], "prediction duration")
        if duration not in _SALiency_DURATIONS:
            raise ValueError("prediction duration is invalid")
        if duration in prediction_by_duration:
            raise ValueError("prediction durations must be unique")
        prediction_by_duration[duration] = prediction
        _validate_saliency_prediction_metadata(
            prediction["metadata"], f"prediction {prediction_index}.metadata"
        )
    if set(prediction_by_duration) != set(_SALiency_DURATIONS):
        raise ValueError("metadata predictions must cover 1s, 3s, and 7s")

    expected_durations = ("1s", "3s", "7s")
    baseline_metadata = cast(
        Mapping[str, object], prediction_by_duration[expected_durations[0]]["metadata"]
    )
    baseline_geometry = cast(Mapping[str, object], baseline_metadata["geometry"])
    for index, duration in enumerate(expected_durations):
        prediction_metadata = cast(
            Mapping[str, object], prediction_by_duration[duration]["metadata"]
        )
        geometry = cast(Mapping[str, object], prediction_metadata["geometry"])
        if prediction_metadata["model_checksum"] != checksum_values[index]:
            raise ValueError(
                f"prediction {duration} model checksum does not match cache_key"
            )
        for field_name in (
            "preprocessing_version",
            "execution_provider",
        ):
            if prediction_metadata[field_name] != cache_key[field_name]:
                raise ValueError(
                    f"prediction {duration} {field_name} does not match cache_key"
                )
        if geometry["geometry_version"] != cache_key["geometry_version"]:
            raise ValueError(
                f"prediction {duration} geometry version does not match cache_key"
            )
        if geometry["source_dimensions"] != cache_key["screenshot_dimensions"]:
            raise ValueError(
                f"prediction {duration} source dimensions do not match cache_key"
            )
        if geometry["device_pixel_ratio"] != cache_key["device_pixel_ratio"]:
            raise ValueError(
                f"prediction {duration} geometry DPR does not match cache_key"
            )
        if geometry["zoom"] != cache_key["zoom"]:
            raise ValueError(
                f"prediction {duration} geometry zoom does not match cache_key"
            )
        if prediction_metadata["input_dimensions"] != geometry["native_dimensions"]:
            raise ValueError(
                f"prediction {duration} input dimensions do not match geometry"
            )
        if (
            prediction_metadata["input_dimensions"]
            != baseline_metadata["input_dimensions"]
            or prediction_metadata["output_dimensions"]
            != baseline_metadata["output_dimensions"]
        ):
            raise ValueError("prediction dimensions differ across durations")
        if geometry != baseline_geometry:
            raise ValueError("prediction geometry differs across durations")
        for field_name in (
            "provider_id",
            "model_id",
            "provider_version",
            "model_version",
        ):
            if prediction_metadata[field_name] != baseline_metadata[field_name]:
                raise ValueError(f"prediction {field_name} differs across durations")
    saliency_metadata = _json_mapping(
        metadata["saliency_metadata"],
        allowed=_SALiency_METADATA_RECORD_KEYS,
        required=_SALiency_METADATA_RECORD_KEYS,
        name="saliency_metadata",
    )
    manifests = saliency_metadata["provider_manifests"]
    if not isinstance(manifests, list):
        raise ValueError("saliency provider_manifests must be a list")
    manifest_values = cast(list[object], manifests)
    coerced_manifests = tuple(
        coerce_saliency_provider_manifest(manifest) for manifest in manifest_values
    )
    provider_identities = {
        (manifest.provider_id, manifest.model_id, manifest.version)
        for manifest in coerced_manifests
    }
    for prediction_index, raw_prediction in enumerate(prediction_values):
        prediction = cast(Mapping[str, object], raw_prediction)
        prediction_metadata = cast(Mapping[str, object], prediction["metadata"])
        identity = (
            cast(str, prediction_metadata["provider_id"]),
            cast(str, prediction_metadata["model_id"]),
            cast(str, prediction_metadata["model_version"]),
        )
        if identity not in provider_identities:
            raise ValueError(
                f"prediction {prediction_index} has no matching manifest identity"
            )
    _json_text(saliency_metadata["aggregation_version"], "saliency aggregation_version")
    if saliency_metadata["aggregation_version"] != metadata["aggregation_version"]:
        raise ValueError("saliency aggregation_version does not match metadata")
    _validate_json_warnings(saliency_metadata["warnings"], "saliency warnings")


def _redact_typed_saliency_value(value: object, policy: RedactionPolicy) -> object:
    """Redact typed saliency identifiers before schema validation and canonicalization."""

    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {
            str(key): REDACTED_VALUE
            if is_sensitive_key(key, policy)
            else _redact_typed_saliency_value(item, policy)
            for key, item in mapping.items()
        }
    if isinstance(value, list):
        items = cast(list[object], value)
        return [_redact_typed_saliency_value(item, policy) for item in items]
    if isinstance(value, str):
        if _contains_sensitive_text(value, policy):
            return REDACTED_VALUE
        return value
    return value


def _canonicalize_saliency_json(
    content: bytes,
    kind: SaliencyArtifactKind,
    *,
    policy: RedactionPolicy,
    expected_viewport_id: str | None = None,
) -> bytes:
    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_json_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"unsupported JSON constant: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{kind.value} JSON is invalid") from error
    value = _redact_typed_saliency_value(value, policy)
    if kind is SaliencyArtifactKind.PROFILES:
        _validate_saliency_profiles(value, expected_viewport_id=expected_viewport_id)
    else:
        _validate_saliency_metadata(value, expected_viewport_id=expected_viewport_id)
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sanitize_artifact_content(
    name: str,
    content: bytes,
    policy: RedactionPolicy,
) -> bytes:
    """Remove configured values from persisted screenshots and trace archives."""

    lowered_name = name.lower()
    if lowered_name.endswith(".zip") or zipfile.is_zipfile(io.BytesIO(content)):
        return _sanitize_zip(content, policy)
    if lowered_name.endswith(".png") or content.startswith(b"\x89PNG\r\n\x1a\n"):
        return _blank_png(content) if policy.exact_values else content
    return _sanitize_non_archive_content(content, policy)


def _sanitize_non_archive_content(
    content: bytes,
    policy: RedactionPolicy,
    *,
    force_text: bool = False,
) -> bytes:
    """Sanitize textual fallback bytes, including malformed archive payloads."""

    redacted = _replace_exact_bytes(content, policy)
    try:
        text = redacted.decode("utf-8")
    except UnicodeDecodeError:
        if not force_text:
            return redacted
        text = redacted.decode("utf-8", errors="replace")
    try:
        value = json.loads(text, object_pairs_hook=_json_object)
    except (json.JSONDecodeError, ValueError):
        return sanitize_log_text(text, policy).encode("utf-8")
    return (
        json.dumps(
            _redact_untyped_value(value, policy),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _redact_untyped_value(value: object, policy: RedactionPolicy) -> object:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {
            str(key): REDACTED_VALUE
            if is_sensitive_key(key, policy)
            else _redact_untyped_value(item, policy)
            for key, item in mapping.items()
        }
    if isinstance(value, list):
        return [
            _redact_untyped_value(item, policy) for item in cast(list[object], value)
        ]
    if isinstance(value, str):
        return sanitize_log_text(value, policy)
    return value


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
    except (
        EOFError,
        NotImplementedError,
        OSError,
        RuntimeError,
        zipfile.BadZipFile,
        zlib.error,
    ):
        return _sanitize_non_archive_content(content, policy, force_text=True)
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
    if "?" in endpoint or "#" in endpoint:
        raise ValueError("endpoint origin must not contain query or fragment")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("endpoint origin must not contain credentials")
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return endpoint.split("?", maxsplit=1)[0].split("#", maxsplit=1)[0].rstrip("/")


def validate_saliency_artifact_path(name: str) -> PurePosixPath:
    """Validate one generated saliency artifact logical path."""

    if not name.strip():
        raise ValueError("saliency artifact path must not be empty")
    raw_name = name.replace("\\", "/")
    normalized = PurePosixPath(raw_name)
    if (
        normalized.as_posix() != raw_name
        or len(normalized.parts) != 3
        or normalized.parts[0] != "saliency"
        or normalized.is_absolute()
        or any(part in {"", ".", ".."} or ":" in part for part in normalized.parts)
    ):
        raise ValueError("saliency artifact path must be relative and normalized")
    viewport_id, filename = normalized.parts[1:]
    if (
        not viewport_id
        or any(character.isspace() for character in viewport_id)
        or any(marker in viewport_id for marker in ("[", "]", "#"))
        or filename not in _SALIENCY_ARTIFACT_FILENAMES
    ):
        raise ValueError("saliency artifact filename must be allowlisted")
    if _contains_sensitive_text(viewport_id, RedactionPolicy()):
        raise ValueError("saliency artifact path contains sensitive identifier")
    return normalized


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
    prominence_provider_id: str = "heuristic"
    model_ids: Mapping[str, str] = field(default_factory=_empty_string_mapping)
    prompt_versions: Mapping[str, str] = field(default_factory=_empty_string_mapping)
    package_version: str = __version__
    provider_versions: Mapping[str, str] = field(default_factory=_empty_string_mapping)
    provider_manifests: tuple[ProviderManifest, ...] = ()
    model_trial: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("run_id", self.run_id),
            ("config_digest", self.config_digest),
            ("endpoint_origin", self.endpoint_origin),
            ("package_version", self.package_version),
            ("prominence_provider_id", self.prominence_provider_id),
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
            model_trial=run_spec.model_trial,
            config_digest=run_spec.config_digest,
            endpoint_origin=endpoint_origin,
            scenario_id=run_spec.scenario.id,
            application_version_id=run_spec.application_version.id,
            persona_id=run_spec.persona.id,
            policy=run_spec.policy.value,
            prominence_provider_id=run_spec.prominence_provider_id,
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
            "model_trial": self.model_trial,
            "config_digest": self.config_digest,
            "endpoint_origin": self.endpoint_origin,
            "scenario_id": self.scenario_id,
            "application_version_id": self.application_version_id,
            "persona_id": self.persona_id,
            "policy": self.policy,
            "prominence_provider_id": self.prominence_provider_id,
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


class SaliencyArtifactKind(StrEnum):
    """Generated saliency evidence types with distinct sanitization rules."""

    NATIVE_MAP = "native-map"
    HEATMAP = "heatmap"
    PROFILES = "profiles"
    METADATA = "metadata"


_SALIENCY_MANIFEST_KEYS = frozenset(
    {
        "provider_id",
        "role",
        "model_id",
        "endpoint_origin",
        "version",
        "prompt_version",
        "schema_version",
    }
)


def _require_sha256_text(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def coerce_saliency_provider_manifest(value: object) -> ProviderManifest:
    """Accept only provider-manifest fields allowed in saliency evidence."""

    if isinstance(value, ProviderManifest):
        manifest = value
    else:
        if not isinstance(value, Mapping):
            raise TypeError(
                "saliency provider manifest must be ProviderManifest or mapping"
            )
        mapping = cast(Mapping[str, object], value)
        unknown = set(mapping) - _SALIENCY_MANIFEST_KEYS
        if unknown:
            raise ValueError(
                "saliency provider manifest contains unsupported fields: "
                + ", ".join(sorted(str(item) for item in unknown))
            )
        required = {"provider_id", "role", "endpoint_origin", "version"}
        missing = required - set(mapping)
        if missing:
            raise ValueError(
                "saliency provider manifest is missing fields: "
                + ", ".join(sorted(missing))
            )
        for field_name in required:
            field_value = mapping[field_name]
            if type(field_value) is not str or not field_value.strip():
                raise TypeError(f"saliency provider manifest {field_name} must be text")
        model_id = mapping.get("model_id")
        if model_id is not None and not isinstance(model_id, str):
            raise TypeError("saliency provider manifest model_id must be text or null")
        prompt_version = mapping.get("prompt_version")
        if prompt_version is not None and not isinstance(prompt_version, str):
            raise TypeError(
                "saliency provider manifest prompt_version must be text or null"
            )
        schema_version = mapping.get("schema_version")
        if schema_version is not None and not isinstance(schema_version, str):
            raise TypeError(
                "saliency provider manifest schema_version must be text or null"
            )
        manifest = ProviderManifest(
            provider_id=str(mapping["provider_id"]),
            role=str(mapping["role"]),
            model_id=model_id,
            endpoint_origin=str(mapping["endpoint_origin"]),
            version=str(mapping["version"]),
            prompt_version=prompt_version,
            schema_version=schema_version,
        )
    _origin_without_credentials(manifest.endpoint_origin)
    return manifest


def saliency_provider_manifest_to_dict(value: object) -> dict[str, object]:
    """Serialize only safe, typed provider identity fields."""

    manifest = coerce_saliency_provider_manifest(value)
    return {
        "provider_id": manifest.provider_id,
        "role": manifest.role,
        "model_id": manifest.model_id,
        "endpoint_origin": _origin_without_credentials(manifest.endpoint_origin),
        "version": manifest.version,
        "prompt_version": manifest.prompt_version,
        "schema_version": manifest.schema_version,
    }


@dataclass(frozen=True, slots=True)
class SaliencyCacheHitEvent:
    """Allowlisted saliency cache event; never carries maps or selectors."""

    cache_key: str
    viewport_id: str
    execution_provider: str
    model_checksums: tuple[str, str, str]
    preprocessing_version: str
    precision: str
    provider_manifests: tuple[ProviderManifest, ...]
    artifact_checksums: tuple[ArtifactReference, ...]
    warnings: tuple[str, ...] = ()
    cache_state: str = "hit"

    def __post_init__(self) -> None:
        _require_sha256_text("cache_key", self.cache_key)
        if not self.viewport_id or any(
            separator in self.viewport_id for separator in ("/", "\\", "\x00")
        ):
            raise ValueError(
                "saliency event viewport_id must be one safe path component"
            )
        validate_saliency_artifact_path(f"saliency/{self.viewport_id}/metadata.json")
        for name, value in (
            ("execution_provider", self.execution_provider),
            ("preprocessing_version", self.preprocessing_version),
            ("precision", self.precision),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"saliency event {name} must not be empty")
        checksums = tuple(self.model_checksums)
        if len(checksums) != 3:
            raise ValueError("saliency event requires all three model checksums")
        for checksum in checksums:
            _require_sha256_text("model checksum", checksum)
        manifests = tuple(
            coerce_saliency_provider_manifest(item) for item in self.provider_manifests
        )
        if not manifests:
            raise ValueError("saliency cache event requires provider manifests")
        artifact_checksums = tuple(self.artifact_checksums)
        if not artifact_checksums:
            raise ValueError("saliency cache event requires artifact checksums")
        if self.cache_state not in {"hit", "miss"}:
            raise ValueError("saliency event cache state must be hit or miss")
        artifact_paths: set[str] = set()
        for reference in artifact_checksums:
            if type(reference) is not ArtifactReference:
                raise TypeError(
                    "saliency event artifact checksums must be typed references"
                )
            _require_sha256_text("artifact checksum", reference.sha256)
            validate_saliency_artifact_path(reference.path)
            if PurePosixPath(reference.path).parts[1] != self.viewport_id:
                raise ValueError(
                    "saliency artifact reference viewport does not match event"
                )
            if reference.name != reference.path:
                raise ValueError("saliency artifact reference name must be allowlisted")
            if reference.path in artifact_paths:
                raise ValueError("saliency event artifact paths must be unique")
            artifact_paths.add(reference.path)
            if type(reference.size) is not int or reference.size < 0:
                raise ValueError(
                    "saliency artifact reference size must not be negative"
                )
        expected_paths = {
            f"saliency/{self.viewport_id}/{filename}"
            for filename in _SALIENCY_ARTIFACT_FILENAMES
        }
        if artifact_paths != expected_paths:
            raise ValueError(
                "saliency event artifact checksums must cover one complete viewport exactly"
            )
        warnings = tuple(self.warnings)
        if any(
            type(warning) is not str
            or not warning.strip()
            or len(warning) > _MAX_SALIENCY_WARNING_LENGTH
            or any(character in warning for character in "\x00\r\n")
            for warning in warnings
        ):
            raise ValueError(
                "saliency event warnings must be bounded single-line strings"
            )
        object.__setattr__(self, "model_checksums", checksums)
        object.__setattr__(self, "provider_manifests", manifests)
        object.__setattr__(self, "artifact_checksums", artifact_checksums)
        object.__setattr__(self, "warnings", warnings)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-only allowlisted event data."""

        return {
            "kind": (
                "saliency-cache-hit"
                if self.cache_state == "hit"
                else "saliency-inference-recorded"
            ),
            "cache_state": self.cache_state,
            "cache_key": self.cache_key,
            "viewport_id": self.viewport_id,
            "execution_provider": self.execution_provider,
            "model_checksums": list(self.model_checksums),
            "preprocessing_version": self.preprocessing_version,
            "precision": self.precision,
            "provider_manifests": [
                saliency_provider_manifest_to_dict(manifest)
                for manifest in self.provider_manifests
            ],
            "artifact_checksums": [
                {
                    "path": reference.path,
                    "sha256": reference.sha256,
                    "size": reference.size,
                }
                for reference in self.artifact_checksums
            ],
            "warnings": [
                sanitize_log_text(warning, RedactionPolicy())
                for warning in self.warnings
            ],
        }


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

    @property
    def manifest(self) -> BundleManifest:
        """Return immutable bundle manifest used for provider provenance."""

        ...

    def append_event(self, event: object) -> int:
        """Append one redacted event and return its monotonic sequence."""

        ...

    def append_saliency_event(self, event: SaliencyCacheHitEvent) -> int:
        """Append one typed saliency event without arbitrary JSON values."""

        ...

    def write_artifact(self, name: str, content: bytes | str) -> ArtifactReference:
        """Write or reuse one content-addressed artifact."""

        ...

    def write_named_artifact(
        self, name: str, content: bytes | str
    ) -> ArtifactReference:
        """Write generic checksum-covered content at one logical path."""

        ...

    def write_saliency_artifact(
        self,
        name: str,
        content: bytes | str,
        kind: SaliencyArtifactKind,
    ) -> ArtifactReference:
        """Write generated saliency evidence at its required logical path."""

        ...

    def finalize(self, result: object) -> Path:
        """Write terminal files and atomically publish bundle."""

        ...

    def abort(self, reason: str) -> Path:
        """Leave staging bundle with crash marker for recovery."""

        ...
