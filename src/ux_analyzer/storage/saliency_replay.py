"""Load checksum-validated saliency replay from immutable run bundles."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, cast

from PIL import Image

from ux_analyzer.ports.artifacts import (
    SaliencyArtifactKind,
    canonicalize_saliency_artifact_content,
    parse_saliency_json_content,
    required_saliency_artifact_paths,
    validate_saliency_artifact_path,
)
from ux_analyzer.storage.run_bundle import (
    secure_assert_ancestors,
    secure_is_link_or_reparse,
    secure_read_bytes,
    validate_saliency_heatmap_content,
    validate_saliency_native_map_content,
)

_MAX_SALIENCY_ARTIFACT_BYTES = 8 * 1024 * 1024
_MAX_SOURCE_SCREENSHOT_BYTES = 16 * 1024 * 1024
_REDACTED_IDENTIFIER = "[REDACTED]"
_SALIENCY_DURATIONS = ("1s", "3s", "7s")
_SAFE_PUBLIC_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_SENSITIVE_IDENTIFIER_MARKERS = (
    "access_token",
    "api_key",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)
_SALIENCY_PROFILE_KEYS = frozenset(
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
_SALIENCY_AGGREGATE_KEYS = frozenset(
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
_CANONICAL_PROVIDER_IDS = {
    "heuristic": "heuristic",
    "heuristic-prominence": "heuristic",
    "foveacast": "foveacast",
    "foveacast-prominence": "foveacast",
}


class SaliencyReplayUnavailable(ValueError):
    """Raised when checksum-covered saliency evidence fails schema checks."""


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return cast(dict[str, Any], value)


def _list_of_mappings(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    values = cast(Sequence[object], value)
    return [cast(dict[str, Any], item) for item in values if isinstance(item, Mapping)]


def _kind(event: Mapping[str, object]) -> str:
    return _text(event.get("kind"), "unknown")


def _strings(value: object) -> list[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return []
    values = tuple(cast(Iterable[object], value))
    return [_text(item) for item in values if item is not None]


def _number(value: object, default: float) -> float:
    return (
        float(value)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        else default
    )


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _text(value: object, default: str = "") -> str:
    return default if value is None else str(value)


def _optional_text(value: object) -> str | None:
    return None if value is None else _text(value)


def _optional_bounded(value: object) -> float | None:
    return max(0.0, min(1.0, _number(value, 0))) if _is_number(value) else None


def _unique(values: Sequence[object]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = _text(value)
        if text and text not in result:
            result.append(text)
    return result


def _safe_identifier(value: object, default: str = _REDACTED_IDENTIFIER) -> str:
    text = _text(value, "").strip()
    lowered = text.casefold()
    if (
        not text
        or ".." in text
        or any(marker in lowered for marker in _SENSITIVE_IDENTIFIER_MARKERS)
        or _SAFE_PUBLIC_IDENTIFIER.fullmatch(text) is None
    ):
        return default
    return text


def _safe_checksum(value: object) -> str:
    text = _text(value, "")
    if (
        len(text) == 64
        and text == text.lower()
        and all(character in "0123456789abcdef" for character in text)
    ):
        return text
    return "unavailable"


def _safe_mixture(value: object) -> list[list[object]]:
    values: list[list[object]] = []
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        values = [[key, item] for key, item in mapping.items()]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for raw_value in cast(Sequence[object], value):
            if isinstance(raw_value, Sequence) and not isinstance(
                raw_value, (str, bytes)
            ):
                values.append(list(cast(Sequence[object], raw_value)))
    result: list[list[object]] = []
    for item in values:
        if len(item) != 2:
            continue
        duration, weight = item
        if _text(duration) not in _SALIENCY_DURATIONS or not _is_number(weight):
            continue
        result.append([_text(duration), _number(weight, 0)])
    return result


def _canonical_provider_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return _CANONICAL_PROVIDER_IDS.get(value)


def _is_allowed_screenshot_artifact(path: PurePosixPath) -> bool:
    if len(path.parts) == 2 and path.parts[0] != "artifacts":
        return False
    if len(path.parts) > 2:
        return False
    if path.suffix.casefold() in {".gif", ".jpeg", ".jpg", ".png"}:
        return True
    return len(path.name) == 64 and all(
        character in "0123456789abcdefABCDEF" for character in path.name
    )


def _looks_redacted_source_png(content: bytes) -> bool:
    if not content.startswith(b"\x89PNG"):
        return False
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.width <= 0 or image.height <= 0:
                return False
            rgba = image.convert("RGBA")
            colors = rgba.getcolors(maxcolors=2)
            return colors == [(rgba.width * rgba.height, (0, 0, 0, 255))]
    except (OSError, ValueError):
        return False


def load_saliency_replay(
    run_path: Path,
    events: Sequence[Mapping[str, object]],
    snapshots: Sequence[Mapping[str, object]],
    *,
    expected_provider_id: str = "unavailable",
) -> tuple[dict[str, Any], ...]:
    """Load trusted saliency replay data for bounded evidence consumers."""

    return tuple(
        _saliency_replay(
            Path(run_path),
            [dict(event) for event in events],
            [dict(snapshot) for snapshot in snapshots],
            expected_provider_id=expected_provider_id,
        )
    )


def _saliency_artifact_data_uri(run_path: Path, artifact: str) -> str | None:
    """Read one allowlisted heatmap without escaping bundle containment."""

    try:
        normalized = validate_saliency_artifact_path(artifact)
    except ValueError:
        return None
    if normalized.name not in {
        "1s-heatmap.png",
        "3s-heatmap.png",
        "7s-heatmap.png",
    }:
        return None
    candidate = _secure_bundle_file(run_path, normalized)
    if candidate is None:
        return None
    try:
        content = secure_read_bytes(candidate, "saliency heatmap")
    except (OSError, RuntimeError):
        return None
    if len(content) > _MAX_SALIENCY_ARTIFACT_BYTES:
        return None
    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()
        with Image.open(BytesIO(content)) as image:
            if image.format != "PNG" or image.mode != "L":
                return None
    except (OSError, ValueError):
        return None
    return f"data:image/png;base64,{base64.b64encode(content).decode('ascii')}"


def saliency_artifact_data_uri(run_path: Path, artifact: str) -> str | None:
    """Return a data URI for one validated heatmap artifact."""

    return _saliency_artifact_data_uri(Path(run_path), artifact)


def _secure_bundle_file(root: Path, relative: PurePosixPath) -> Path | None:
    """Resolve bundle file only through real, contained directories."""

    try:
        secure_assert_ancestors(root, "report bundle")
        current = root
        if secure_is_link_or_reparse(current) or not current.is_dir():
            return None
        for index, part in enumerate(relative.parts):
            current = current / part
            if secure_is_link_or_reparse(current):
                return None
            if (
                index < len(relative.parts) - 1
                and current.exists()
                and not current.is_dir()
            ):
                return None
        if not current.is_file():
            return None
        return current
    except (OSError, RuntimeError, ValueError):
        return None


def _saliency_replay(
    run_path: Path,
    events: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
    *,
    expected_provider_id: str = "unavailable",
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for event in events:
        kind = _kind(event)
        if not kind.startswith("saliency-") and kind != "prominence-recorded":
            continue
        if kind == "prominence-recorded" and not any(
            event.get(field_name) is not None
            for field_name in (
                "artifact_namespace",
                "active_provider_id",
                "source_event_id",
            )
        ):
            continue
        raw_namespace = _optional_text(
            event.get("artifact_namespace")
        ) or _optional_text(event.get("viewport_id"))
        namespace = _validated_saliency_namespace(raw_namespace)
        if not namespace:
            continue
        group = groups.setdefault(
            namespace,
            {
                "viewport_id": _safe_identifier(
                    event.get("source_viewport_id"), namespace
                ),
                "artifact_namespace": _safe_identifier(namespace, "unavailable"),
                "provider_id": "unavailable",
                "active_provider_id": "unavailable",
                "search_stage": "unavailable",
                "cache_state": "unavailable",
                "selected_mixture": [],
                "warnings": [],
                "model_checksums": [],
                "execution_provider": "unavailable",
                "timings_ms": [],
                "stage_history": [],
                "profile_event_ids": [],
                "operational_event_ids": [],
                "source_event_ids": [],
                "artifact_ids": [],
            },
        )
        _merge_saliency_event(group, event)

    replay: list[dict[str, Any]] = []
    for namespace in sorted(groups):
        group = groups[namespace]
        try:
            profiles = _read_saliency_profiles(
                run_path / "saliency" / namespace / "profiles.json",
                namespace,
            )
            metadata = _read_saliency_metadata(
                run_path / "saliency" / namespace / "metadata.json",
                namespace,
            )
            source_viewport_id = _text(group.get("viewport_id"), namespace)
            source_snapshot = next(
                (item for item in snapshots if item["id"] == source_viewport_id),
                None,
            )
            _validate_saliency_replay_linkage(
                run_path,
                events,
                group,
                profiles,
                metadata,
                namespace,
                source_snapshot=source_snapshot,
                expected_provider_id=expected_provider_id,
            )
            _validate_saliency_profile_lineage(profiles, source_snapshot)
        except SaliencyReplayUnavailable as error:
            replay.append(
                {
                    **group,
                    "profiles": [],
                    "metadata": {},
                    "entries": [],
                    "replay_available": False,
                    "replay_error": f"Saliency replay unavailable: {error}",
                    "overlay_available": False,
                    "overlay_message": None,
                }
            )
            continue
        entries: list[dict[str, Any]] = []
        for duration in _SALIENCY_DURATIONS:
            artifact = f"saliency/{namespace}/{duration}-heatmap.png"
            prediction: dict[str, Any] = next(
                (
                    item
                    for item in _list_of_mappings(metadata.get("predictions"))
                    if item.get("duration") == duration
                ),
                cast(dict[str, Any], {}),
            )
            prediction_metadata = _mapping(prediction.get("metadata"))
            aggregates = _aggregates_for_duration(profiles, duration)
            entries.append(
                {
                    "duration": duration,
                    "heatmap": _saliency_artifact_data_uri(run_path, artifact),
                    "heatmap_path": artifact,
                    "overlay_available": bool(
                        source_snapshot and source_snapshot.get("screenshot")
                    ),
                    "profiles": profiles,
                    "ranked_elements": _ranked_elements(
                        aggregates, source_snapshot, duration
                    ),
                    "aggregation_components": aggregates,
                    "inference_duration_ms": _number(
                        prediction_metadata.get("inference_duration_ms"),
                        0,
                    ),
                    "provider_id": _safe_identifier(
                        prediction_metadata.get("provider_id"),
                        _text(group.get("provider_id"), "unavailable"),
                    ),
                    "model_id": _safe_identifier(
                        prediction_metadata.get("model_id"), "unavailable"
                    ),
                    "model_version": _safe_identifier(
                        prediction_metadata.get("model_version"), "unavailable"
                    ),
                    "model_checksum": _safe_checksum(
                        prediction_metadata.get("model_checksum")
                    ),
                    "execution_provider": _safe_identifier(
                        prediction_metadata.get("execution_provider"),
                        _text(group.get("execution_provider"), "unavailable"),
                    ),
                    "warnings": _unique(
                        [
                            *_strings(group.get("warnings")),
                            *_strings(prediction_metadata.get("warnings")),
                        ]
                    ),
                }
            )
        replay.append(
            {
                **group,
                "profiles": profiles,
                "metadata": metadata,
                "entries": entries,
                "replay_available": True,
                "replay_error": None,
                "overlay_available": bool(
                    source_snapshot and source_snapshot.get("screenshot")
                ),
                "overlay_message": (
                    "Overlay unavailable due redaction. Heatmap-only artifact retained."
                    if source_snapshot and source_snapshot.get("screenshot_redacted")
                    else None
                ),
            }
        )
    return replay


def _validate_saliency_profile_lineage(
    profiles: list[dict[str, Any]], snapshot: dict[str, Any] | None
) -> None:
    """Keep model-estimate profiles joined to recorded interface elements."""

    if snapshot is None:
        raise SaliencyReplayUnavailable("saliency source viewport is unavailable")
    element_ids = {element["id"] for element in snapshot.get("elements", [])}
    if any(profile["element_id"] not in element_ids for profile in profiles):
        raise SaliencyReplayUnavailable(
            "saliency profile element is not present in source snapshot"
        )


def _merge_saliency_event(group: dict[str, Any], event: dict[str, Any]) -> None:
    for field_name in (
        "provider_id",
        "active_provider_id",
        "search_stage",
        "cache_state",
        "execution_provider",
    ):
        event_value = event.get(field_name)
        if field_name == "search_stage" and event_value is None:
            event_value = event.get("stage")
        if event_value is not None:
            group[field_name] = _safe_identifier(event_value, "unavailable")
    if event.get("selected_mixture") is not None:
        group["selected_mixture"] = _safe_mixture(event["selected_mixture"])
    if event.get("model_checksums") is not None:
        group["model_checksums"] = [
            checksum
            for checksum in _strings(event.get("model_checksums"))
            if _safe_checksum(checksum) != "unavailable"
        ]
    group["warnings"] = _unique(
        [*_strings(group.get("warnings")), *_strings(event.get("warnings"))]
    )
    timings = event.get("timings_ms")
    timing_values: list[object] | tuple[object, ...] = ()
    if isinstance(timings, list | tuple):
        timing_values = cast(list[object] | tuple[object, ...], timings)
        group["timings_ms"] = [
            _number(value, 0) for value in timing_values if _is_number(value)
        ]
    event_id = f"event-{int(_number(event.get('sequence'), 0))}"
    stage_history = _list_of_mappings(group.get("stage_history"))
    stage_history.append(
        {
            "event_id": event_id,
            "kind": _kind(event),
            "search_stage": _safe_identifier(
                event.get("search_stage", event.get("stage")), "unavailable"
            ),
            "selected_mixture": _safe_mixture(event.get("selected_mixture")),
            "cache_state": _safe_identifier(event.get("cache_state"), "unavailable"),
            "timings_ms": [
                _number(value, 0) for value in timing_values if _is_number(value)
            ],
        }
    )
    group["stage_history"] = stage_history
    if _kind(event) == "saliency-profiles-recorded":
        group.setdefault("profile_event_ids", []).append(event_id)
    if _kind(event) == "prominence-recorded":
        group.setdefault("operational_event_ids", []).append(event_id)
    source_event_id = event.get("source_event_id")
    if source_event_id is not None:
        group.setdefault("source_event_ids", []).append(str(source_event_id))
    artifact_ids = event.get("artifact_ids")
    if isinstance(artifact_ids, list | tuple):
        group.setdefault("artifact_ids", []).extend(
            _strings(cast(object, artifact_ids))
        )


def merge_saliency_event(group: dict[str, Any], event: dict[str, Any]) -> None:
    """Merge one trusted saliency event into its replay group."""

    _merge_saliency_event(group, event)


def _read_saliency_profiles(path: Path, namespace: str) -> list[dict[str, Any]]:
    value = _read_saliency_json(path, SaliencyArtifactKind.PROFILES, namespace)
    if not isinstance(value, list):
        raise SaliencyReplayUnavailable("profiles JSON must be a list")
    profile_values = cast(list[object], value)
    profiles: list[dict[str, Any]] = []
    for raw in profile_values:
        if len(profiles) >= 10_000:
            raise SaliencyReplayUnavailable("saliency profile count exceeds limit")
        profile = _mapping(raw)
        if not profile or frozenset(profile) != _SALIENCY_PROFILE_KEYS:
            raise SaliencyReplayUnavailable("profile schema is invalid")
        if _text(profile.get("viewport_id"), "") != namespace:
            raise SaliencyReplayUnavailable("profile viewport is invalid")
        element_id = _safe_identifier(profile.get("element_id"), "unavailable")
        if element_id == "unavailable":
            raise SaliencyReplayUnavailable("profile element ID is invalid")
        item: dict[str, Any] = {
            "viewport_id": namespace,
            "element_id": element_id,
            "aggregation_version": _safe_identifier(
                profile.get("aggregation_version"), "unavailable"
            ),
            "aggregates": [],
            "prediction_provenance": [],
        }
        for estimate_name in ("immediate", "early", "eventual", "general"):
            estimate = _mapping(profile.get(estimate_name))
            if not estimate:
                if estimate_name in {"immediate", "early", "eventual"}:
                    timed_values = [
                        profile.get(name) for name in ("immediate", "early", "eventual")
                    ]
                    if not all(value is None for value in timed_values):
                        raise SaliencyReplayUnavailable(
                            "profile duration coverage is incomplete"
                        )
                    item[estimate_name] = {
                        "kind": "unavailable",
                        "score": None,
                        "source": "unavailable",
                    }
                    continue
                item[estimate_name] = None
                continue
            kind = _text(estimate.get("kind"), "unavailable")
            if kind not in {"predicted", "derived", "fallback", "unavailable"}:
                raise SaliencyReplayUnavailable("profile estimate kind is invalid")
            if estimate_name in {"immediate", "early", "eventual"} and (
                kind == "unavailable"
                or _optional_bounded(estimate.get("score")) is None
            ):
                raise SaliencyReplayUnavailable(
                    "profile duration estimate is unavailable"
                )
            item[estimate_name] = {
                "kind": kind,
                "score": _optional_bounded(estimate.get("score")),
                "source": _safe_identifier(estimate.get("source"), "unavailable"),
            }
        for raw_aggregate in _list_of_mappings(profile.get("aggregates")):
            if frozenset(raw_aggregate) != _SALIENCY_AGGREGATE_KEYS:
                raise SaliencyReplayUnavailable("profile aggregate schema is invalid")
            if len(item["aggregates"]) >= 10_000:
                raise SaliencyReplayUnavailable(
                    "saliency aggregate count exceeds limit"
                )
            if _text(raw_aggregate.get("element_id"), "") != element_id:
                raise SaliencyReplayUnavailable("profile aggregate element is invalid")
            duration = _text(raw_aggregate.get("duration"), "")
            if duration not in _SALIENCY_DURATIONS:
                raise SaliencyReplayUnavailable("profile aggregate duration is invalid")
            item["aggregates"].append(
                {
                    "viewport_id": namespace,
                    "element_id": element_id,
                    "duration": duration,
                    **{
                        field_name: _number(raw_aggregate.get(field_name), 0)
                        for field_name in _SALIENCY_AGGREGATE_KEYS
                        - {"viewport_id", "element_id", "duration"}
                    },
                }
            )
        if {aggregate["duration"] for aggregate in item["aggregates"]} != set(
            _SALIENCY_DURATIONS
        ):
            raise SaliencyReplayUnavailable(
                "profile aggregate duration coverage is incomplete"
            )
        provenance = _list_of_mappings(profile.get("prediction_provenance"))
        provenance_durations = {
            _text(record.get("duration"), "") for record in provenance
        }
        if provenance_durations != set(_SALIENCY_DURATIONS):
            raise SaliencyReplayUnavailable(
                "profile prediction provenance coverage is incomplete"
            )
        item["prediction_provenance"] = [
            {
                "duration": _text(record.get("duration"), "unavailable"),
                "metadata": _safe_prediction_metadata(_mapping(record.get("metadata"))),
            }
            for record in provenance
        ]
        profiles.append(item)
    return profiles


def _read_saliency_metadata(path: Path, namespace: str) -> dict[str, Any]:
    value = _read_saliency_json(path, SaliencyArtifactKind.METADATA, namespace)
    if not isinstance(value, dict):
        raise SaliencyReplayUnavailable("metadata JSON must be an object")
    metadata_value = cast(dict[str, Any], value)
    predictions: list[dict[str, Any]] = []
    for raw in _list_of_mappings(metadata_value.get("predictions")):
        duration = _text(raw.get("duration"), "")
        if duration not in _SALIENCY_DURATIONS:
            raise SaliencyReplayUnavailable("metadata duration is invalid")
        raw_metadata = _mapping(raw.get("metadata"))
        metadata: dict[str, Any] = {
            "provider_id": _safe_identifier(
                raw_metadata.get("provider_id"), "unavailable"
            ),
            "model_id": _safe_identifier(raw_metadata.get("model_id"), "unavailable"),
            "provider_version": _safe_identifier(
                raw_metadata.get("provider_version"), "unavailable"
            ),
            "model_version": _safe_identifier(
                raw_metadata.get("model_version"), "unavailable"
            ),
            "model_checksum": _safe_checksum(raw_metadata.get("model_checksum")),
            "input_dimensions": raw_metadata.get("input_dimensions"),
            "output_dimensions": raw_metadata.get("output_dimensions"),
            "geometry": _mapping(raw_metadata.get("geometry")),
            "preprocessing_version": _safe_identifier(
                raw_metadata.get("preprocessing_version"), "unavailable"
            ),
            "execution_provider": _safe_identifier(
                raw_metadata.get("execution_provider"), "unavailable"
            ),
            "inference_duration_ms": _number(
                raw_metadata.get("inference_duration_ms"), 0
            ),
            "warnings": _strings(raw_metadata.get("warnings")),
        }
        predictions.append({"duration": duration, "metadata": metadata})
    return {
        "viewport_id": _safe_identifier(metadata_value.get("viewport_id"), namespace),
        "cache_key": _mapping(metadata_value.get("cache_key")),
        "cache_key_digest": _safe_checksum(metadata_value.get("cache_key_digest")),
        "artifact_paths": [
            path
            for path in _strings(metadata_value.get("artifact_paths"))
            if _validated_saliency_artifact(path) is not None
        ],
        "predictions": predictions,
        "warnings": _strings(metadata_value.get("warnings")),
    }


def _source_screenshot_digest(
    run_path: Path, snapshot: dict[str, Any] | None
) -> str | None:
    if snapshot is None:
        return None
    artifact = _optional_text(
        snapshot.get("artifact") or snapshot.get("screenshot_artifact")
    )
    if not artifact:
        return None
    relative = PurePosixPath(artifact.replace("\\", "/"))
    if not _is_allowed_screenshot_artifact(relative):
        return None
    candidate = _secure_bundle_file(run_path, relative)
    if candidate is None:
        return None
    try:
        content = secure_read_bytes(candidate, "saliency source screenshot")
    except (OSError, RuntimeError):
        return None
    if len(content) > _MAX_SOURCE_SCREENSHOT_BYTES or _looks_redacted_source_png(
        content
    ):
        return None
    if content.startswith(b"\x89PNG"):
        try:
            with Image.open(BytesIO(content)) as image:
                image.verify()
        except (OSError, ValueError):
            return None
    elif not content.startswith((b"\xff\xd8", b"GIF8")):
        return None
    return hashlib.sha256(content).hexdigest()


def _read_saliency_json(
    path: Path, kind: SaliencyArtifactKind, namespace: str
) -> object:
    root = path.parents[2]
    candidate = _secure_bundle_file(
        root, PurePosixPath("saliency", namespace, path.name)
    )
    if candidate is None:
        raise SaliencyReplayUnavailable("saliency artifact is unavailable")
    try:
        raw = secure_read_bytes(candidate, "saliency JSON")
        if len(raw) > _MAX_SALIENCY_ARTIFACT_BYTES:
            raise SaliencyReplayUnavailable("saliency JSON exceeds size limit")
        parsed = parse_saliency_json_content(
            kind,
            raw,
            expected_viewport_id=namespace,
        )
        canonical = canonicalize_saliency_artifact_content(
            kind,
            raw,
            expected_viewport_id=namespace,
        )
        if canonical != raw:
            raise SaliencyReplayUnavailable("saliency artifact is not canonical")
        return parsed
    except SaliencyReplayUnavailable:
        raise
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise SaliencyReplayUnavailable(
            "saliency artifact schema is invalid"
        ) from error


def _validated_saliency_namespace(value: str | None) -> str | None:
    if not value:
        return None
    try:
        normalized = validate_saliency_artifact_path(f"saliency/{value}/metadata.json")
    except ValueError:
        return None
    return normalized.parts[1]


def _validated_saliency_artifact(value: str) -> str | None:
    try:
        return validate_saliency_artifact_path(value).as_posix()
    except ValueError:
        return None


def _safe_prediction_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider_id": _safe_identifier(metadata.get("provider_id"), "unavailable"),
        "model_id": _safe_identifier(metadata.get("model_id"), "unavailable"),
        "model_version": _safe_identifier(metadata.get("model_version"), "unavailable"),
        "provider_version": _safe_identifier(
            metadata.get("provider_version"), "unavailable"
        ),
        "model_checksum": _safe_checksum(metadata.get("model_checksum")),
        "execution_provider": _safe_identifier(
            metadata.get("execution_provider"), "unavailable"
        ),
        "inference_duration_ms": _number(metadata.get("inference_duration_ms"), 0),
    }


def _validate_saliency_replay_linkage(
    run_path: Path,
    events: list[dict[str, Any]],
    group: dict[str, Any],
    profiles: list[dict[str, Any]],
    metadata: dict[str, Any],
    namespace: str,
    *,
    source_snapshot: dict[str, Any] | None,
    expected_provider_id: str = "unavailable",
) -> None:
    expected_paths = set(required_saliency_artifact_paths(namespace))
    metadata_paths = set(_strings(metadata.get("artifact_paths")))
    if metadata_paths != expected_paths:
        raise SaliencyReplayUnavailable("metadata artifact paths are incomplete")
    cache_key = _mapping(metadata.get("cache_key"))
    expected_screenshot_digest = _safe_checksum(cache_key.get("screenshot_sha256"))
    if expected_screenshot_digest == "unavailable":
        raise SaliencyReplayUnavailable("saliency screenshot digest is missing")
    actual_screenshot_digest = _source_screenshot_digest(run_path, source_snapshot)
    if (
        actual_screenshot_digest is not None
        and actual_screenshot_digest != expected_screenshot_digest
    ):
        raise SaliencyReplayUnavailable("saliency screenshot digest does not match")
    if actual_screenshot_digest is None and "overlay-redacted" not in _strings(
        group.get("warnings")
    ):
        raise SaliencyReplayUnavailable("saliency screenshot digest is unavailable")
    try:
        event_paths = {
            validate_saliency_artifact_path(path).as_posix()
            for path in _strings(group.get("artifact_ids"))
        }
    except ValueError as error:
        raise SaliencyReplayUnavailable("event artifact path is invalid") from error
    if event_paths != expected_paths:
        raise SaliencyReplayUnavailable("event artifact linkage is incomplete")
    canonical_expected_provider = _canonical_provider_id(expected_provider_id)
    if canonical_expected_provider is None:
        raise SaliencyReplayUnavailable("saliency provider identity is invalid")
    for field_name in ("provider_id", "active_provider_id"):
        if _canonical_provider_id(_text(group.get(field_name), "")) != (
            canonical_expected_provider
        ):
            raise SaliencyReplayUnavailable("saliency provider identity is invalid")
    timeline_checksums = set(_strings(group.get("model_checksums")))
    for prediction in _list_of_mappings(metadata.get("predictions")):
        prediction_metadata = _mapping(prediction.get("metadata"))
        if _canonical_provider_id(prediction_metadata.get("provider_id")) != (
            canonical_expected_provider
        ):
            raise SaliencyReplayUnavailable("saliency provider identity is invalid")
        if (
            timeline_checksums
            and _safe_checksum(prediction_metadata.get("model_checksum"))
            not in timeline_checksums
        ):
            raise SaliencyReplayUnavailable("saliency model checksum is invalid")
    metadata_by_duration = {
        _text(prediction.get("duration"), ""): _mapping(prediction.get("metadata"))
        for prediction in _list_of_mappings(metadata.get("predictions"))
    }
    for profile in profiles:
        for provenance in _list_of_mappings(profile.get("prediction_provenance")):
            duration = _text(provenance.get("duration"), "")
            profile_metadata = _mapping(provenance.get("metadata"))
            metadata_for_duration = metadata_by_duration.get(duration)
            if metadata_for_duration is None:
                raise SaliencyReplayUnavailable(
                    "saliency profile prediction duration is invalid"
                )
            for field_name in (
                "provider_id",
                "model_id",
                "model_version",
                "model_checksum",
                "execution_provider",
            ):
                if profile_metadata.get(field_name) != metadata_for_duration.get(
                    field_name
                ):
                    raise SaliencyReplayUnavailable(
                        "saliency profile prediction provenance does not match metadata"
                    )
    event_by_id = {
        f"event-{int(_number(event.get('sequence'), 0))}": event for event in events
    }
    profile_event_ids = {
        event_id for event_id in _strings(group.get("profile_event_ids"))
    }
    relevant_events = [
        event
        for event in events
        if _validated_saliency_namespace(
            _optional_text(event.get("artifact_namespace"))
            or _optional_text(event.get("viewport_id"))
        )
        == namespace
    ]
    for event in relevant_events:
        source_event_id = _optional_text(event.get("source_event_id"))
        if source_event_id is None:
            continue
        source = event_by_id.get(source_event_id)
        current_sequence = int(_number(event.get("sequence"), 0))
        source_sequence = int(_number(source.get("sequence"), 0)) if source else 0
        if source is None or source_sequence >= current_sequence:
            raise SaliencyReplayUnavailable("saliency source event ordering is invalid")
        kind = _kind(event)
        if kind in {
            "saliency-inference-recorded",
            "saliency-cache-hit",
            "saliency-fallback-recorded",
        }:
            if _kind(source) != "viewport-captured":
                raise SaliencyReplayUnavailable(
                    "saliency inference source event kind is invalid"
                )
            source_snapshot = _mapping(source.get("snapshot"))
            if _text(source_snapshot.get("id"), "") != _text(
                event.get("source_viewport_id"), ""
            ):
                raise SaliencyReplayUnavailable(
                    "saliency inference source viewport is invalid"
                )
        elif kind == "saliency-profiles-recorded":
            if _kind(source) == "viewport-captured":
                source_snapshot = _mapping(source.get("snapshot"))
                if _text(source_snapshot.get("id"), "") != _text(
                    event.get("source_viewport_id"), ""
                ):
                    raise SaliencyReplayUnavailable(
                        "saliency profile source viewport is invalid"
                    )
            elif _kind(source) in {
                "saliency-inference-recorded",
                "saliency-cache-hit",
            }:
                if _text(source.get("source_viewport_id"), "") != _text(
                    event.get("source_viewport_id"), ""
                ) or _text(source.get("artifact_namespace"), "") != _text(
                    event.get("artifact_namespace"), ""
                ):
                    raise SaliencyReplayUnavailable(
                        "saliency profile source namespace is invalid"
                    )
                if _canonical_provider_id(_text(source.get("provider_id"), "")) != (
                    _canonical_provider_id(_text(event.get("provider_id"), ""))
                ):
                    raise SaliencyReplayUnavailable(
                        "saliency profile source provider is invalid"
                    )
                if _text(source.get("cache_state"), "") != _text(
                    event.get("cache_state"), ""
                ):
                    raise SaliencyReplayUnavailable(
                        "saliency profile source cache state is invalid"
                    )
            else:
                raise SaliencyReplayUnavailable(
                    "saliency profile source event kind is invalid"
                )
        elif kind == "prominence-recorded":
            if source_event_id not in profile_event_ids:
                raise SaliencyReplayUnavailable(
                    "saliency operational source event kind is invalid"
                )
            if _kind(source) != "saliency-profiles-recorded":
                raise SaliencyReplayUnavailable(
                    "saliency operational source event kind is invalid"
                )
            if _text(source.get("source_viewport_id"), "") != _text(
                event.get("source_viewport_id"), ""
            ) or _text(source.get("artifact_namespace"), "") != _text(
                event.get("artifact_namespace"), ""
            ):
                raise SaliencyReplayUnavailable(
                    "saliency operational source namespace is invalid"
                )
            if _text(source.get("cache_state"), "") != _text(
                event.get("cache_state"), ""
            ):
                raise SaliencyReplayUnavailable(
                    "saliency operational cache state is invalid"
                )
            if _canonical_provider_id(_text(source.get("provider_id"), "")) != (
                _canonical_provider_id(_text(event.get("active_provider_id"), ""))
            ):
                raise SaliencyReplayUnavailable(
                    "saliency operational provider join is invalid"
                )
        else:
            raise SaliencyReplayUnavailable("saliency source event kind is invalid")
    if not profile_event_ids:
        raise SaliencyReplayUnavailable("saliency profile event linkage is missing")
    checksum_path = _secure_bundle_file(run_path, PurePosixPath("checksums.sha256"))
    if checksum_path is None:
        raise SaliencyReplayUnavailable("saliency checksums are unavailable")
    checksums: dict[str, str] = {}
    try:
        for line in (
            secure_read_bytes(checksum_path, "saliency checksums")
            .decode("utf-8")
            .splitlines()
        ):
            if "  " in line:
                digest, relative = line.split("  ", maxsplit=1)
                checksums[relative] = digest
        for relative in sorted(expected_paths):
            artifact = _secure_bundle_file(run_path, PurePosixPath(relative))
            if artifact is None or checksums.get(relative) is None:
                raise SaliencyReplayUnavailable("saliency artifact checksum is missing")
            try:
                content = secure_read_bytes(artifact, "saliency artifact")
            except (OSError, RuntimeError) as error:
                raise SaliencyReplayUnavailable(
                    "saliency artifact is unreadable"
                ) from error
            if len(content) > _MAX_SALIENCY_ARTIFACT_BYTES:
                raise SaliencyReplayUnavailable("saliency artifact exceeds size limit")
            if hashlib.sha256(content).hexdigest() != checksums[relative]:
                raise SaliencyReplayUnavailable("saliency artifact checksum is invalid")
            if PurePosixPath(relative).name.endswith(".npz"):
                prediction = _prediction_metadata_for_path(metadata, relative)
                try:
                    validate_saliency_native_map_content(
                        content,
                        expected_output_dimensions=(
                            int(
                                _number(
                                    prediction.get("output_dimensions", [0, 0])[0], 0
                                )
                            ),
                            int(
                                _number(
                                    prediction.get("output_dimensions", [0, 0])[1], 0
                                )
                            ),
                        ),
                        expected_geometry=_mapping(prediction.get("geometry")),
                    )
                except (IndexError, TypeError, ValueError) as error:
                    raise SaliencyReplayUnavailable(
                        "saliency native map is invalid"
                    ) from error
            elif PurePosixPath(relative).name.endswith("-heatmap.png"):
                try:
                    validate_saliency_heatmap_content(content)
                except ValueError as error:
                    raise SaliencyReplayUnavailable(
                        "saliency heatmap is invalid"
                    ) from error
    except (OSError, RuntimeError, UnicodeError):
        raise SaliencyReplayUnavailable("saliency checksums are unreadable")


def _aggregates_for_duration(
    profiles: list[dict[str, Any]], duration: str
) -> list[dict[str, Any]]:
    return [
        aggregate
        for profile in profiles
        for aggregate in _list_of_mappings(profile.get("aggregates"))
        if aggregate.get("duration") == duration
    ]


def _prediction_metadata_for_path(
    metadata: dict[str, Any], relative_path: str
) -> dict[str, Any]:
    duration = PurePosixPath(relative_path).stem
    prediction = next(
        (
            item
            for item in _list_of_mappings(metadata.get("predictions"))
            if item.get("duration") == duration
        ),
        None,
    )
    prediction_metadata = _mapping(_mapping(prediction).get("metadata"))
    if not prediction_metadata:
        raise ValueError("saliency prediction metadata is missing")
    return prediction_metadata


def _ranked_elements(
    aggregates: list[dict[str, Any]],
    snapshot: dict[str, Any] | None,
    duration: str,
) -> list[dict[str, Any]]:
    elements = {
        element["id"]: element for element in (snapshot or {}).get("elements", [])
    }
    ranked = sorted(
        aggregates,
        key=lambda aggregate: _number(aggregate.get("adjusted_score"), 0),
        reverse=True,
    )
    result: list[dict[str, Any]] = []
    for rank, aggregate in enumerate(ranked, start=1):
        element = elements.get(aggregate["element_id"], {})
        result.append(
            {
                "rank": rank,
                "element_id": aggregate["element_id"],
                "label": _text(element.get("label"), "Unlabelled element"),
                "role": _text(element.get("role"), "other"),
                "bounds": _mapping(element.get("bounds")),
                "duration": duration,
                "adjusted_score": aggregate.get("adjusted_score"),
                "visibility_fraction": aggregate.get("visibility_fraction"),
                "occlusion_fraction": aggregate.get("occlusion_fraction"),
            }
        )
    return result
