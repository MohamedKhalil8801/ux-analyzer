"""Build and resolve the redacted experiment evidence boundary."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import cast

from PIL import Image

from ux_analyzer.application.checkpoint import (
    finalized_bundle_failures,
    read_finalized_bundle,
)
from ux_analyzer.application.experiment import ExperimentResult
from ux_analyzer.domain.expectations import ExpectationKey, FrozenExpectation
from ux_analyzer.domain.findings import EvidenceClass
from ux_analyzer.domain.synthesis import EvidenceRef
from ux_analyzer.ports.artifacts import (
    validate_saliency_artifact_path,
    validate_saliency_heatmap_content,
    validate_saliency_native_map_content,
)
from ux_analyzer.storage.run_bundle import (
    secure_assert_ancestors,
    secure_is_link_or_reparse,
    secure_read_bytes,
)
from ux_analyzer.storage.saliency_replay import load_saliency_replay

_MAX_JSON_BYTES = 8 * 1024 * 1024
_MAX_SCREENSHOT_BYTES = 16 * 1024 * 1024
_MAX_SCREENSHOT_PIXELS = 16 * 1024 * 1024
_MAX_SALIENCY_BYTES = 8 * 1024 * 1024
_MAX_NATIVE_MAP_BYTES = 64 * 1024 * 1024
_MAX_TEXT_LENGTH = 4096
_MAX_GEOMETRY_COORDINATE = 1_000_000.0
_MAX_VIEWPORT_DIMENSION = 32_768
_UX_PRINCIPLE_PACK_VERSION = "ux-principles-v1"
_UX_PRINCIPLE_PACK_DIGEST = (
    "b460e6bde12cd108199e4fa6d96676149a6809d03bcdfc894f4025663e0238df"
)
_SALIENCY_DURATIONS = frozenset({"1s", "3s", "7s"})
_IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/gif"})
_SALIENCY_MEDIA_TYPES = frozenset({"application/octet-stream", "application/x-npz"})
_METRIC_IDENTITY_KEYS = frozenset(
    {
        "abandoned",
        "application_version_id",
        "config_digest",
        "persona_id",
        "policy",
        "prominence_provider_id",
        "run_id",
        "scenario_id",
        "seed",
        "model_trial",
    }
)
_APPROVED_METRIC_CLASSES = {
    "outcome": EvidenceClass.DETERMINISTIC_FACT,
    "inspected-elements": EvidenceClass.DETERMINISTIC_FACT,
    "inspected-regions": EvidenceClass.DETERMINISTIC_FACT,
    "scrolls": EvidenceClass.DETERMINISTIC_FACT,
    "wrong-actions": EvidenceClass.DETERMINISTIC_FACT,
    "backtracks": EvidenceClass.DETERMINISTIC_FACT,
    "verified-completion": EvidenceClass.DETERMINISTIC_FACT,
    "false-success": EvidenceClass.DETERMINISTIC_FACT,
    "navigation-depth": EvidenceClass.DETERMINISTIC_FACT,
    "recovery-actions": EvidenceClass.DETERMINISTIC_FACT,
    "claimed-completion": EvidenceClass.MODEL_ESTIMATE,
    "target-discovery-rank": EvidenceClass.MODEL_ESTIMATE,
    "target-prominence": EvidenceClass.MODEL_ESTIMATE,
    "target-scent": EvidenceClass.MODEL_ESTIMATE,
    "strongest-competing-scent": EvidenceClass.MODEL_ESTIMATE,
    "target-below-fold": EvidenceClass.MODEL_ESTIMATE,
    "unexpected-hierarchy": EvidenceClass.MODEL_ESTIMATE,
    "ambiguous-target": EvidenceClass.MODEL_ESTIMATE,
    "feedback-observed": EvidenceClass.MODEL_ESTIMATE,
    "recovery-success": EvidenceClass.MODEL_ESTIMATE,
    "discovery-cost": EvidenceClass.MODEL_ESTIMATE,
    "inspection-cost": EvidenceClass.MODEL_ESTIMATE,
    "region-cost": EvidenceClass.MODEL_ESTIMATE,
    "scroll-cost": EvidenceClass.MODEL_ESTIMATE,
    "wrong-action-cost": EvidenceClass.MODEL_ESTIMATE,
    "backtrack-cost": EvidenceClass.MODEL_ESTIMATE,
    "uncertainty-cost": EvidenceClass.MODEL_ESTIMATE,
    "abandonment-penalty": EvidenceClass.MODEL_ESTIMATE,
}
_DETERMINISTIC_METRICS = frozenset(
    name
    for name, evidence_class in _APPROVED_METRIC_CLASSES.items()
    if evidence_class is EvidenceClass.DETERMINISTIC_FACT
)
_BOOLEAN_METRICS = frozenset(
    {
        "verified-completion",
        "claimed-completion",
        "false-success",
        "target-below-fold",
        "unexpected-hierarchy",
        "ambiguous-target",
        "feedback-observed",
        "recovery-success",
    }
)
_COUNT_METRICS = frozenset(
    {
        "target-discovery-rank",
        "inspected-elements",
        "inspected-regions",
        "scrolls",
        "wrong-actions",
        "backtracks",
        "navigation-depth",
        "recovery-actions",
    }
)
_PROBABILITY_METRICS = frozenset(
    {
        "target-prominence",
        "target-scent",
        "strongest-competing-scent",
    }
)
_NON_NEGATIVE_METRICS = frozenset(
    {
        "discovery-cost",
        "inspection-cost",
        "region-cost",
        "scroll-cost",
        "wrong-action-cost",
        "backtrack-cost",
        "uncertainty-cost",
        "abandonment-penalty",
    }
)
_SAFE_ID_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_PROVENANCE_MARKERS = (
    "access_token",
    "api_key",
    "authorization",
    "credential",
    "password",
    "private_reasoning",
    "cognitive",
    "reasoning",
    "rationale",
    "request",
    "response",
    "secret",
    "token",
)
_SAFE_OUTCOMES = frozenset(
    {
        "success",
        "verified-success",
        "agent-abandoned",
        "budget-exhausted",
        "timed-out",
        "provider-failure",
        "model-failure",
        "safety-blocked",
        "internal-error",
    }
)
_SCORE_EVENT_KINDS = {
    "prominence-recorded": "prominence",
    "prominence-scored": "prominence",
    "coarse-scent": "scent",
    "coarse-scent-recorded": "scent",
    "full-scent": "scent",
    "full-scent-recorded": "scent",
}
_REF_NAMESPACE_KINDS = {
    "scenario",
    "persona",
    "goal",
    "expectation",
    "viewport",
    "element",
    "screenshot",
    "event",
    "replay",
    "verification",
    "metric",
    "model-estimate",
    "saliency-metadata",
    "heatmap",
    "native-map",
    "ranked-element",
    "limitation",
    "counterevidence",
    "failure",
}


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return cast(Mapping[str, object], value)


def _sequence_values(value: object) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(cast(Sequence[object], value))


def _mappings(value: object) -> tuple[Mapping[str, object], ...]:
    return tuple(
        cast(Mapping[str, object], item)
        for item in _sequence_values(value)
        if isinstance(item, Mapping)
    )


def _text(value: object, default: str = "") -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()[:_MAX_TEXT_LENGTH]
    return default


def _optional_text(value: object) -> str | None:
    result = _text(value)
    return result or None


def _is_id_component(value: object) -> bool:
    return isinstance(value, str) and _SAFE_ID_COMPONENT.fullmatch(value) is not None


def _safe_provenance(value: object, default: str = "unavailable") -> str:
    text = _text(value, default)
    lowered = text.casefold()
    if not _is_id_component(text) or any(
        marker in lowered for marker in _SAFE_PROVENANCE_MARKERS
    ):
        return default
    return text


def _safe_scalar(value: object) -> object | None:
    if isinstance(value, Enum):
        return _safe_scalar(value.value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return None


def _integer(value: object) -> int | None:
    scalar = _safe_scalar(value)
    if type(scalar) is int:
        return scalar
    if isinstance(scalar, float) and scalar.is_integer():
        return int(scalar)
    if isinstance(scalar, str):
        try:
            return int(scalar)
        except ValueError:
            return None
    return None


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in mapping.items()}
        )
    if isinstance(value, (list, tuple)):
        sequence = cast(Sequence[object], value)
        return tuple(_freeze(item) for item in sequence)
    if isinstance(value, (set, frozenset)):
        values = cast(set[object] | frozenset[object], value)
        return tuple(_freeze(item) for item in sorted(values, key=repr))
    if isinstance(value, Enum):
        return _freeze(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("evidence payload contains non-finite number")
        return value
    raise TypeError(f"unsupported evidence payload value: {type(value).__name__}")


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): _json_value(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        sequence = cast(Sequence[object], value)
        return [_json_value(item) for item in sequence]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Enum):
        return _json_value(value.value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported evidence JSON value: {type(value).__name__}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalized_relative_path(value: object, label: str) -> PurePosixPath:
    if isinstance(value, Path):
        raw = value.as_posix()
    elif isinstance(value, str):
        raw = value.replace("\\", "/")
    else:
        raise ValueError(f"{label} must be a relative path")
    normalized = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if (
        not raw
        or normalized.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or normalized.as_posix() != raw
        or any(part in {"", ".", ".."} or ":" in part for part in normalized.parts)
    ):
        raise ValueError(f"{label} contains traversal or is outside the output root")
    return normalized


def _sequence(event: Mapping[str, object]) -> int:
    value = event.get("sequence")
    return value if type(value) is int and value > 0 else 0


def _bounds(value: object) -> dict[str, float] | None:
    raw = _mapping(value)
    result: dict[str, float] = {}
    for name in ("x", "y", "width", "height"):
        number = _safe_scalar(raw.get(name))
        if not isinstance(number, (int, float)) or isinstance(number, bool):
            return None
        if (
            not math.isfinite(float(number))
            or float(number) < 0
            or float(number) > _MAX_GEOMETRY_COORDINATE
        ):
            return None
        result[name] = float(number)
    if (
        result["x"] + result["width"] > _MAX_GEOMETRY_COORDINATE
        or result["y"] + result["height"] > _MAX_GEOMETRY_COORDINATE
    ):
        return None
    return result


def _bounded(value: object) -> float | None:
    number = _safe_scalar(value)
    if not isinstance(number, (int, float)) or isinstance(number, bool):
        return None
    return max(0.0, min(1.0, float(number)))


def _probability(value: object) -> float | None:
    number = _safe_scalar(value)
    if not isinstance(number, (int, float)) or isinstance(number, bool):
        return None
    normalized = float(number)
    if not 0.0 <= normalized <= 1.0:
        return None
    return normalized


def _positive_rank(value: object) -> int | None:
    number = _safe_scalar(value)
    if not isinstance(number, (int, float)) or isinstance(number, bool):
        return None
    normalized = float(number)
    if normalized <= 0.0 or not normalized.is_integer():
        return None
    return int(normalized)


def _safe_checksum(value: object) -> str | None:
    if not isinstance(value, str) or len(value) != 64:
        return None
    if any(character not in "0123456789abcdefABCDEF" for character in value):
        return None
    return value.lower()


def _dimensions(value: object) -> dict[str, int] | None:
    raw = _mapping(value)
    width = _safe_scalar(raw.get("width"))
    height = _safe_scalar(raw.get("height"))
    if (
        not isinstance(width, (int, float))
        or isinstance(width, bool)
        or not isinstance(height, (int, float))
        or isinstance(height, bool)
        or not math.isfinite(float(width))
        or not math.isfinite(float(height))
        or width <= 0
        or height <= 0
        or width > _MAX_VIEWPORT_DIMENSION
        or height > _MAX_VIEWPORT_DIMENSION
        or width * height > _MAX_SCREENSHOT_PIXELS
    ):
        return None
    return {"width": int(width), "height": int(height)}


def _safe_action(value: object) -> dict[str, object]:
    action = _mapping(value)
    result: dict[str, object] = {}
    for name in ("kind", "element_id", "direction"):
        text = _optional_text(action.get(name))
        if text is not None:
            result[name] = text
    for name in ("duration_seconds",):
        number = _safe_scalar(action.get(name))
        if isinstance(number, (int, float)) and not isinstance(number, bool):
            result[name] = number
    return result


def _public_snapshot(event: Mapping[str, object]) -> dict[str, object] | None:
    raw = _mapping(event.get("snapshot"))
    viewport_id = _optional_text(raw.get("id"))
    if viewport_id is None or not _is_id_component(viewport_id):
        return None
    elements: list[dict[str, object]] = []
    right_edges: list[float] = []
    bottom_edges: list[float] = []
    for item in _mappings(raw.get("elements")):
        element_id = _optional_text(item.get("id"))
        bounds = _bounds(item.get("bounds"))
        if element_id is None or not _is_id_component(element_id) or bounds is None:
            continue
        element: dict[str, object] = {
            "id": element_id,
            "role": _text(item.get("role"), "other"),
            "label": _text(item.get("label"), "Unlabelled element"),
            "bounds": bounds,
            "visibility_fraction": _bounded(item.get("visibility_fraction")),
            "occlusion_fraction": _bounded(item.get("occlusion_fraction")),
            "local_contrast": _bounded(item.get("local_contrast")),
            "actionable": bool(item.get("actionable", False)),
            "disabled": bool(item.get("disabled", False)),
        }
        for name in ("region_id",):
            text = _optional_text(item.get(name))
            if text is not None:
                element[name] = text
        for name in ("noticed", "inspected"):
            if isinstance(item.get(name), bool):
                element[name] = item[name]
        elements.append(element)
        right_edges.append(bounds["x"] + bounds["width"])
        bottom_edges.append(bounds["y"] + bounds["height"])
    regions = [
        {
            "id": _text(region.get("id")),
            "label": _text(region.get("label")),
        }
        for region in _mappings(raw.get("regions"))
        if _optional_text(region.get("id")) is not None
    ]
    dimensions: dict[str, int] | None = None
    candidates = (
        _mapping(raw.get("viewport")),
        _mapping(raw.get("viewport_size")),
        _mapping(event.get("viewport")),
        {"width": event.get("viewport_width"), "height": event.get("viewport_height")},
    )
    for candidate in candidates:
        dimensions = _dimensions(candidate)
        if dimensions is not None:
            break
    if dimensions is None:
        dimensions = _dimensions(
            {
                "width": max(1, math.ceil(max(right_edges, default=1))),
                "height": max(1, math.ceil(max(bottom_edges, default=1))),
            }
        ) or {"width": 1, "height": 1}
    artifact = _optional_text(raw.get("screenshot_artifact") or raw.get("artifact"))
    result: dict[str, object] = {
        "id": viewport_id,
        "viewport": dimensions,
        "elements": elements,
        "regions": regions,
    }
    if artifact is not None:
        result["screenshot_artifact"] = artifact
    return result


def _root_relative_artifact(
    root: Path, run_path: Path, raw_path: object
) -> tuple[PurePosixPath, Path] | None:
    try:
        relative_to_run = _normalized_relative_path(raw_path, "artifact path")
        candidate = run_path.joinpath(*relative_to_run.parts)
        secure_assert_ancestors(candidate, "evidence artifact")
        current = run_path
        for index, part in enumerate(relative_to_run.parts):
            current = current / part
            if secure_is_link_or_reparse(current):
                return None
            if index < len(relative_to_run.parts) - 1 and current.exists():
                if not current.is_dir():
                    return None
        root_relative = PurePosixPath(candidate.relative_to(root).as_posix())
        return root_relative, candidate
    except (OSError, RuntimeError, ValueError):
        return None


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    if not path.is_file() or secure_is_link_or_reparse(path):
        raise ValueError(f"missing {label}")
    try:
        value = json.loads(
            secure_read_bytes(path, label, max_bytes=_MAX_JSON_BYTES).decode("utf-8"),
            object_pairs_hook=_json_object_without_duplicates,
        )
    except (OSError, RuntimeError, UnicodeError, ValueError) as error:
        raise ValueError(f"invalid {label}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return cast(dict[str, object], value)


def _json_object_without_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON contains duplicate fields")
        result[key] = value
    return result


def _identity(spec: object) -> dict[str, object]:
    scenario = getattr(spec, "scenario", None)
    version = getattr(spec, "application_version", None)
    persona = getattr(spec, "persona", None)
    policy = getattr(spec, "policy", None)
    policy_value = getattr(policy, "value", policy)
    return {
        "run_id": _text(getattr(spec, "run_id", None)),
        "seed": getattr(spec, "seed", 0),
        "model_trial": getattr(spec, "model_trial", 0),
        "policy": _text(policy_value),
        "prominence_provider_id": _text(
            getattr(spec, "prominence_provider_id", None), "heuristic"
        ),
        "config_digest": _text(getattr(spec, "config_digest", None)),
        "scenario_id": _text(getattr(scenario, "id", None)),
        "scenario_name": _text(getattr(scenario, "name", None)),
        "goal": _text(getattr(scenario, "goal", None)),
        "application_version_id": _text(getattr(version, "id", None)),
        "application_version_label": _text(getattr(version, "label", None)),
        "persona_id": _text(getattr(persona, "id", None)),
        "persona_name": _text(getattr(persona, "name", None)),
    }


def _raw_identity(value: object) -> dict[str, object]:
    raw = _mapping(value)
    scenario = _mapping(raw.get("scenario"))
    version = _mapping(raw.get("application_version"))
    persona = _mapping(raw.get("persona"))
    return {
        "run_id": raw.get("run_id"),
        "seed": raw.get("seed"),
        "model_trial": raw.get("model_trial", 0),
        "policy": raw.get("policy"),
        "prominence_provider_id": raw.get("prominence_provider_id"),
        "config_digest": raw.get("config_digest"),
        "scenario_id": scenario.get("id", raw.get("scenario_id")),
        "scenario_name": scenario.get("name"),
        "goal": scenario.get("goal"),
        "application_version_id": version.get("id", raw.get("application_version_id")),
        "application_version_label": version.get("label"),
        "persona_id": persona.get("id", raw.get("persona_id")),
        "persona_name": persona.get("name"),
    }


def _check_identity(
    label: str,
    candidate: object,
    expected: Mapping[str, object],
    *,
    required: frozenset[str] = frozenset(),
) -> None:
    actual = _raw_identity(candidate)
    for name, expected_value in expected.items():
        if expected_value in (None, ""):
            continue
        if actual.get(name) is None:
            if name in required:
                raise ValueError(f"{label} {name} is missing")
            continue
        actual_value = actual[name]
        if name in {"seed", "model_trial"}:
            actual_integer = _integer(actual_value)
            expected_integer = _integer(expected_value)
            if (
                actual_integer is None
                or expected_integer is None
                or actual_integer != expected_integer
            ):
                raise ValueError(f"{label} {name} does not match run spec")
        elif str(actual_value) != str(expected_value):
            raise ValueError(f"{label} {name} does not match run spec")


def _provenance(
    source: Mapping[str, object],
    metrics: Mapping[str, object],
    manifest: Mapping[str, object],
) -> dict[str, str]:
    provider_id = _text(
        source.get("provider_id"),
        _text(
            source.get("active_provider_id"),
            _text(metrics.get("prominence_provider_id"), "unavailable"),
        ),
    )
    provider_versions = _mapping(manifest.get("provider_versions"))
    model_ids = _mapping(manifest.get("model_ids"))
    return {
        "provider_id": _safe_provenance(provider_id),
        "provider_version": _safe_provenance(
            source.get("provider_version"),
            _text(
                metrics.get("prominence_provider_version"),
                _text(provider_versions.get(provider_id), "unavailable"),
            ),
        ),
        "model_id": _safe_provenance(
            source.get("model_id"),
            _text(
                metrics.get("prominence_model_id"),
                _text(model_ids.get(provider_id), "unavailable"),
            ),
        ),
        "model_version": _safe_provenance(
            source.get("model_version"),
            _text(metrics.get("prominence_model_version"), "unavailable"),
        ),
    }


def _metric_class(name: str, source: Mapping[str, object]) -> EvidenceClass:
    del source
    return _APPROVED_METRIC_CLASSES[name]


def _approved_metric_value(
    name: str,
    value: object,
    *,
    max_discovery_rank: int | None = None,
) -> object | None:
    if name not in _APPROVED_METRIC_CLASSES:
        return None
    if name == "outcome":
        return value if isinstance(value, str) and value in _SAFE_OUTCOMES else None
    if name in _BOOLEAN_METRICS:
        if isinstance(value, bool):
            return value
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) in {0.0, 1.0}
        ):
            return value
        return None
    if name in _COUNT_METRICS:
        count = _integer(value)
        if count is None or count < 0:
            return None
        if name == "target-discovery-rank":
            if count <= 0:
                return None
            if max_discovery_rank is not None and count > max_discovery_rank:
                return None
        return count
    if name in _PROBABILITY_METRICS:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        return number if math.isfinite(number) and 0.0 <= number <= 1.0 else None
    if name in _NON_NEGATIVE_METRICS:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        return number if math.isfinite(number) and number >= 0.0 else None
    return None


def _safe_evidence_id(value: object) -> str | None:
    if not isinstance(value, str) or not value or any(char.isspace() for char in value):
        return None
    parts = value.split(":")
    if len(parts) < 2:
        return None
    if any(
        not _is_id_component(part) and re.fullmatch(r"[0-9a-f]{64}", part) is None
        for part in parts
    ):
        return None
    return value


def _metric_values(
    metrics: Mapping[str, object],
    *,
    max_discovery_rank: int | None = None,
) -> tuple[tuple[str, object, EvidenceClass, Mapping[str, object]], ...]:
    values: list[tuple[str, object, EvidenceClass, Mapping[str, object]]] = []
    explicit_names: set[str] = set()
    for raw in _mappings(metrics.get("metrics")):
        name = _text(raw.get("name"))
        rendered = name.replace("_", "-")
        value = _approved_metric_value(
            rendered,
            raw.get("value"),
            max_discovery_rank=max_discovery_rank,
        )
        if not name or value is None:
            continue
        evidence_class = _metric_class(rendered, raw)
        explicit_names.add(rendered)
        values.append((rendered, value, evidence_class, raw))
    for raw_name, raw_value in metrics.items():
        name = str(raw_name)
        if name in _METRIC_IDENTITY_KEYS or name in {"metrics", "evidence"}:
            continue
        rendered = name.replace("_", "-")
        if rendered in explicit_names:
            continue
        if name == "discovery_cost":
            total = _approved_metric_value(
                "discovery-cost",
                _mapping(raw_value).get("total"),
                max_discovery_rank=max_discovery_rank,
            )
            if total is not None:
                values.append(
                    (
                        "discovery-cost",
                        total,
                        EvidenceClass.MODEL_ESTIMATE,
                        _mapping(raw_value),
                    )
                )
            continue
        value = _safe_scalar(raw_value)
        value = _approved_metric_value(
            rendered,
            value,
            max_discovery_rank=max_discovery_rank,
        )
        if value is None:
            continue
        evidence_class = _metric_class(rendered, metrics)
        values.append((rendered, value, evidence_class, metrics))
    return tuple(values)


def _discovery_rank_bound(
    events: Sequence[Mapping[str, object]],
    snapshots: Sequence[Mapping[str, object]],
) -> int:
    observed_elements = sum(
        len(
            _sequence_values(
                _mapping(event.get("observation")).get("newly_revealed_elements")
            )
        )
        for event in events
    )
    if observed_elements > 0:
        return observed_elements
    return sum(len(_mappings(snapshot.get("elements"))) for snapshot in snapshots)


def _score_values(value: object) -> tuple[Mapping[str, object], ...]:
    return _mappings(value)


def _ranked_payload(value: object) -> dict[str, object]:
    raw = _mapping(value)
    rank = _positive_rank(raw.get("rank"))
    if rank is None:
        return {}
    result: dict[str, object] = {"rank": rank}
    for name in (
        "element_id",
        "label",
        "role",
        "duration",
        "adjusted_score",
        "visibility_fraction",
        "occlusion_fraction",
        "bounds",
    ):
        if name in raw:
            if name == "bounds":
                bounds = _bounds(raw[name])
                if bounds is not None:
                    result[name] = bounds
            elif name in {
                "adjusted_score",
                "visibility_fraction",
                "occlusion_fraction",
            }:
                if raw[name] is None:
                    continue
                number = _probability(raw[name])
                if number is None:
                    return {}
                result[name] = number
            elif name == "element_id":
                text = _optional_text(raw[name])
                if text is not None and _is_id_component(text):
                    result[name] = text
            else:
                text = _optional_text(raw[name])
                if text is not None:
                    result[name] = text
    return result


def _attachment_digest(path: Path, label: str, *, max_bytes: int) -> tuple[bytes, str]:
    try:
        content = secure_read_bytes(path, label, max_bytes=max_bytes)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"{label} is unreadable or oversized") from error
    return content, hashlib.sha256(content).hexdigest()


def _screenshot_media_type(content: bytes) -> str:
    try:
        with Image.open(BytesIO(content)) as image:
            if (
                image.width <= 0
                or image.height <= 0
                or image.width > _MAX_VIEWPORT_DIMENSION
                or image.height > _MAX_VIEWPORT_DIMENSION
                or image.width * image.height > _MAX_SCREENSHOT_PIXELS
            ):
                raise ValueError("screenshot dimensions exceed resource limit")
            image.verify()
            image_format = image.format
    except Image.DecompressionBombError as error:
        raise ValueError("screenshot dimensions exceed resource limit") from error
    except ValueError:
        raise
    except OSError as error:
        raise ValueError("screenshot is not a supported image") from error
    media_type = {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "GIF": "image/gif",
    }.get(image_format or "")
    if media_type is None:
        raise ValueError("screenshot media type is unsupported")
    return media_type


@dataclass(frozen=True, slots=True)
class EvidenceEntry:
    """One immutable, allowlisted piece of experiment evidence."""

    ref: EvidenceRef
    evidence_class: EvidenceClass
    summary: str
    payload: Mapping[str, object]
    attachment_path: Path | None = None

    def __post_init__(self) -> None:
        _validate_ref_namespace(self.ref)
        object.__setattr__(self, "evidence_class", EvidenceClass(self.evidence_class))
        summary = self.summary.strip()
        if not summary:
            raise ValueError("evidence entry summary must not be empty")
        object.__setattr__(self, "summary", summary[:_MAX_TEXT_LENGTH])
        object.__setattr__(
            self,
            "payload",
            cast(Mapping[str, object], _freeze(self.payload)),
        )
        if self.attachment_path is not None:
            path = Path(self.attachment_path)
            if self.ref.artifact_path is None:
                raise ValueError("attachment path requires artifact reference")
            object.__setattr__(self, "attachment_path", path)


@dataclass(frozen=True, slots=True)
class EvidenceCorpus:
    """Immutable experiment-level evidence registry."""

    output_root: Path
    entries: tuple[EvidenceEntry, ...]
    principle_pack_version: str = _UX_PRINCIPLE_PACK_VERSION
    principle_pack_digest: str = _UX_PRINCIPLE_PACK_DIGEST
    metadata: Mapping[str, object] = field(
        default_factory=lambda: cast(Mapping[str, object], {})
    )
    _index: Mapping[str, EvidenceEntry] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        root = Path(self.output_root)
        if not root.is_absolute():
            root = root.absolute()
        object.__setattr__(self, "output_root", root)
        entries = tuple(self.entries)
        index: dict[str, EvidenceEntry] = {}
        for entry in entries:
            evidence_id = entry.ref.evidence_id
            if evidence_id in index:
                raise ValueError(f"duplicate evidence ID: {evidence_id}")
            index[evidence_id] = entry
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "_index", MappingProxyType(index))
        object.__setattr__(
            self,
            "metadata",
            cast(Mapping[str, object], _freeze(self.metadata)),
        )

    def require(self, evidence_id: str) -> EvidenceEntry:
        try:
            return self._index[evidence_id]
        except KeyError as error:
            raise ValueError(f"unknown evidence ID: {evidence_id}") from error

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "evidence-corpus-v1",
            "principle_pack_version": self.principle_pack_version,
            "principle_pack_digest": self.principle_pack_digest,
            "metadata": self.metadata,
            "entries": [
                {
                    "evidence_id": entry.ref.evidence_id,
                    "kind": entry.ref.kind,
                    "run_id": entry.ref.run_id,
                    "viewport_id": entry.ref.viewport_id,
                    "element_id": entry.ref.element_id,
                    "event_id": entry.ref.event_id,
                    "metric_id": entry.ref.metric_id,
                    "artifact_path": entry.ref.artifact_path,
                    "replay_sequence": entry.ref.replay_sequence,
                    "sha256": entry.ref.sha256,
                    "evidence_class": entry.evidence_class.value,
                    "summary": entry.summary,
                    "payload": entry.payload,
                    "attachment_path": (
                        entry.attachment_path.as_posix()
                        if entry.attachment_path is not None
                        else None
                    ),
                }
                for entry in self.entries
            ],
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict()) + "\n"


@dataclass(frozen=True, slots=True, init=False)
class ResolvedEvidence:
    """Validated evidence entries returned by the retrieval boundary."""

    entries: tuple[EvidenceEntry, ...]
    attachment_bytes: int = 0

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(entry.ref.evidence_id for entry in self.entries)

    @classmethod
    def from_entries(
        cls,
        corpus: EvidenceCorpus,
        entries: Sequence[object],
        *,
        max_entries: int,
        max_attachment_bytes: int,
    ) -> ResolvedEvidence:
        _validate_limits(max_entries, max_attachment_bytes)
        normalized = tuple(entries)
        if len(normalized) > max_entries:
            raise ValueError("max_entries limit exceeded")
        canonical: list[EvidenceEntry] = []
        for entry in normalized:
            if not isinstance(entry, EvidenceEntry):
                raise TypeError(
                    "resolved evidence entries must be EvidenceEntry values"
                )
            corpus_entry = corpus.require(entry.ref.evidence_id)
            if corpus_entry is not entry:
                raise ValueError("resolved evidence entries must come from corpus")
            canonical.append(corpus_entry)
        canonical_entries = tuple(canonical)
        validate_evidence_refs(corpus, tuple(entry.ref for entry in canonical_entries))
        total = 0
        for entry in canonical_entries:
            if entry.attachment_path is not None:
                total += _validate_attachment(
                    corpus,
                    entry,
                    max_attachment_bytes=max_attachment_bytes,
                )
                if total > max_attachment_bytes:
                    raise ValueError("cumulative attachment byte limit exceeded")
        instance = object.__new__(cls)
        object.__setattr__(instance, "entries", canonical_entries)
        object.__setattr__(instance, "attachment_bytes", total)
        return instance


def _validate_limits(max_entries: int, max_attachment_bytes: int) -> None:
    if type(max_entries) is not int or max_entries <= 0:
        raise ValueError("max_entries must be greater than zero")
    if type(max_attachment_bytes) is not int or max_attachment_bytes <= 0:
        raise ValueError("max_attachment_bytes must be greater than zero")


def _validate_ref_namespace(ref: EvidenceRef) -> None:
    parts = ref.evidence_id.split(":")
    if not parts or parts[0] not in _REF_NAMESPACE_KINDS:
        raise ValueError("evidence ID namespace is invalid")
    namespace = parts[0]
    if namespace != ref.kind:
        raise ValueError("evidence ID namespace does not match evidence kind")
    if len(parts) < 2 or parts[1] != ref.run_id or not _is_id_component(parts[1]):
        raise ValueError("evidence ID run namespace does not match reference")

    def _matches_component(value: str | None, component: str) -> bool:
        return value is None or value == component

    if namespace in {
        "scenario",
        "persona",
        "goal",
        "expectation",
        "verification",
        "failure",
    }:
        if len(parts) != 2:
            raise ValueError("evidence ID namespace has invalid component count")
        return
    if namespace == "viewport" and (
        len(parts) != 3
        or not _is_id_component(parts[2])
        or not _matches_component(ref.viewport_id, parts[2])
    ):
        raise ValueError("viewport evidence namespace does not match reference")
    if namespace == "element" and (
        len(parts) != 4
        or not _is_id_component(parts[2])
        or not _is_id_component(parts[3])
        or not _matches_component(ref.viewport_id, parts[2])
        or not _matches_component(ref.element_id, parts[3])
    ):
        raise ValueError("element evidence namespace does not match reference")
    if namespace in {"heatmap", "native-map"} and (
        len(parts) != 4
        or not _is_id_component(parts[2])
        or parts[3] not in _SALIENCY_DURATIONS
        or not _matches_component(ref.viewport_id, parts[2])
    ):
        raise ValueError("saliency evidence namespace does not match reference")
    if namespace == "screenshot" and (
        len(parts) != 3
        or not re.fullmatch(r"[0-9a-f]{64}", parts[2])
        or not _matches_component(ref.sha256, parts[2])
    ):
        raise ValueError("screenshot evidence namespace does not match reference")
    if namespace in {"event", "replay"} and len(parts) != 3:
        raise ValueError("event evidence namespace does not match reference")
    if namespace in {"event", "replay"} and (
        not re.fullmatch(r"[1-9][0-9]*", parts[2])
        or (ref.replay_sequence is not None and ref.replay_sequence != int(parts[2]))
        or (ref.event_id is not None and ref.event_id != f"event-{parts[2]}")
    ):
        raise ValueError("event evidence sequence is invalid")
    if namespace == "metric" and (
        len(parts) != 3
        or parts[2] not in _APPROVED_METRIC_CLASSES
        or not _matches_component(ref.metric_id, parts[2])
    ):
        raise ValueError("metric evidence namespace does not match reference")
    if namespace == "model-estimate" and (
        len(parts) != 5
        or parts[2] not in {"prominence", "scent"}
        or not re.fullmatch(r"[1-9][0-9]*", parts[3])
        or not _is_id_component(parts[4])
        or not _matches_component(ref.element_id, parts[4])
    ):
        raise ValueError("model estimate evidence namespace does not match reference")
    if namespace in {"saliency-metadata", "limitation", "counterevidence"} and (
        len(parts) != 3 or not _is_id_component(parts[2])
    ):
        raise ValueError("evidence ID namespace has invalid component")
    if namespace == "ranked-element" and (
        len(parts) != 5
        or not _is_id_component(parts[2])
        or parts[3] not in _SALIENCY_DURATIONS
        or not _is_id_component(parts[4])
        or not _matches_component(ref.viewport_id, parts[2])
        or not _matches_component(ref.element_id, parts[4])
    ):
        raise ValueError("ranked element evidence namespace does not match reference")


def _validate_ref_against_entry(ref: EvidenceRef, entry: EvidenceEntry) -> None:
    expected = entry.ref
    if ref.kind != expected.kind or ref.run_id != expected.run_id:
        raise ValueError("evidence reference does not match corpus entry")
    _validate_ref_namespace(ref)
    for name in (
        "viewport_id",
        "element_id",
        "event_id",
        "metric_id",
        "replay_sequence",
        "artifact_path",
        "sha256",
    ):
        value = getattr(ref, name)
        expected_value = getattr(expected, name)
        if value is not None and value != expected_value:
            raise ValueError("evidence reference does not match corpus entry")
    if ref.artifact_path is not None:
        _normalized_relative_path(ref.artifact_path, "evidence artifact path")


def validate_evidence_refs(corpus: EvidenceCorpus, refs: Sequence[EvidenceRef]) -> None:
    """Validate every requested reference against the immutable registry."""

    raw_references = _sequence_values(refs)
    references: list[EvidenceRef] = []
    for raw_ref in raw_references:
        if not isinstance(raw_ref, EvidenceRef):
            raise TypeError("evidence references must contain EvidenceRef values")
        references.append(raw_ref)
    evidence_ids = tuple(ref.evidence_id for ref in references)
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("duplicate evidence reference")
    for ref in references:
        entry = corpus.require(ref.evidence_id)
        _validate_ref_against_entry(ref, entry)


def _resolve_attachment(corpus: EvidenceCorpus, entry: EvidenceEntry) -> Path:
    attachment = entry.attachment_path
    if attachment is None:
        raise ValueError("evidence entry has no attachment")
    relative = _normalized_relative_path(attachment, "attachment path")
    if entry.ref.artifact_path is None:
        raise ValueError("attachment reference is missing artifact path")
    reference_path = _normalized_relative_path(
        entry.ref.artifact_path, "evidence artifact path"
    )
    if relative != reference_path:
        raise ValueError("attachment path does not match evidence reference")
    root = corpus.output_root
    try:
        secure_assert_ancestors(root, "evidence output root")
        if secure_is_link_or_reparse(root) or not root.is_dir():
            raise ValueError("evidence output root is invalid")
        candidate = root.joinpath(*relative.parts)
        current = root
        for index, part in enumerate(relative.parts):
            current = current / part
            if secure_is_link_or_reparse(current):
                raise ValueError("evidence attachment is a symlink or reparse point")
            if (
                index < len(relative.parts) - 1
                and current.exists()
                and not current.is_dir()
            ):
                raise ValueError("evidence attachment path is not contained")
        if not candidate.is_file():
            raise ValueError("evidence attachment is missing")
        return candidate
    except (OSError, RuntimeError) as error:
        raise ValueError("evidence attachment path is not secure") from error


def _validate_attachment(
    corpus: EvidenceCorpus,
    entry: EvidenceEntry,
    *,
    max_attachment_bytes: int,
) -> int:
    path = _resolve_attachment(corpus, entry)
    kind = entry.ref.kind
    if kind == "screenshot":
        maximum = _MAX_SCREENSHOT_BYTES
    elif kind == "heatmap":
        maximum = _MAX_SALIENCY_BYTES
    elif kind == "native-map":
        maximum = _MAX_NATIVE_MAP_BYTES
    else:
        raise ValueError("unsupported evidence attachment media")
    try:
        content, digest = _attachment_digest(
            path,
            "evidence attachment",
            max_bytes=maximum,
        )
    except ValueError as error:
        raise ValueError(str(error)) from error
    if entry.ref.sha256 is None or digest != entry.ref.sha256:
        raise ValueError("evidence attachment checksum mismatch")
    declared = _optional_text(entry.payload.get("media_type"))
    if kind == "screenshot":
        actual_media = _screenshot_media_type(content)
        if declared not in _IMAGE_MEDIA_TYPES or declared != actual_media:
            raise ValueError("unsupported screenshot media")
    elif kind == "heatmap":
        if declared != "image/png":
            raise ValueError("unsupported heatmap media")
        try:
            validate_saliency_heatmap_content(content)
        except ValueError as error:
            raise ValueError("heatmap content is invalid") from error
    elif kind == "native-map":
        if declared not in _SALIENCY_MEDIA_TYPES:
            raise ValueError("unsupported native-map media")
        try:
            validate_saliency_native_map_content(content)
        except ValueError as error:
            raise ValueError("native-map content is invalid") from error
    if len(content) > max_attachment_bytes:
        raise ValueError("attachment byte limit exceeded")
    return len(content)


class EvidenceResolver:
    """Resolve only bounded, checksum- and namespace-validated evidence."""

    def resolve(
        self,
        corpus: EvidenceCorpus,
        evidence_ids: Sequence[str],
        *,
        max_entries: int,
        max_attachment_bytes: int,
    ) -> ResolvedEvidence:
        _validate_limits(max_entries, max_attachment_bytes)
        requested = tuple(evidence_ids)
        if len(requested) > max_entries:
            raise ValueError("max_entries limit exceeded")
        if len(requested) != len(set(requested)):
            raise ValueError("duplicate evidence ID")
        entries = tuple(corpus.require(evidence_id) for evidence_id in requested)
        return ResolvedEvidence.from_entries(
            corpus,
            entries,
            max_entries=max_entries,
            max_attachment_bytes=max_attachment_bytes,
        )


class _EntryCollector:
    def __init__(self) -> None:
        self.entries: list[EvidenceEntry] = []
        self.by_id: dict[str, EvidenceEntry] = {}

    def add(
        self,
        ref: EvidenceRef,
        evidence_class: EvidenceClass,
        summary: str,
        payload: Mapping[str, object],
        attachment_path: Path | None = None,
    ) -> None:
        existing = self.by_id.get(ref.evidence_id)
        if existing is not None:
            if ref.kind == "screenshot" and ref.sha256 == existing.ref.sha256:
                return
            raise ValueError(f"duplicate evidence ID: {ref.evidence_id}")
        entry = EvidenceEntry(
            ref=ref,
            evidence_class=evidence_class,
            summary=summary,
            payload=payload,
            attachment_path=attachment_path,
        )
        self.entries.append(entry)
        self.by_id[ref.evidence_id] = entry


class EvidenceCorpusBuilder:
    """Create an allowlist from finalized bundles and typed expectations."""

    def build(
        self,
        experiment: ExperimentResult,
        output_root: Path,
        expectations: Mapping[ExpectationKey, FrozenExpectation],
    ) -> EvidenceCorpus:
        root = Path(output_root)
        if not root.is_absolute():
            root = root.absolute()
        if not root.is_dir() or secure_is_link_or_reparse(root):
            raise ValueError("evidence output root is invalid")
        summary = _read_json_object(root / "experiment.json", "experiment.json")
        summary_rows = {
            _text(row.get("run_id")): row
            for row in _mappings(summary.get("run_metrics"))
            if _optional_text(row.get("run_id")) is not None
        }
        specs = tuple(experiment.specs)
        specs_by_id: dict[str, object] = {}
        for spec in specs:
            identity = _identity(spec)
            run_id = _text(identity.get("run_id"))
            if not run_id or run_id in specs_by_id:
                raise ValueError("experiment specs contain duplicate or empty run ID")
            specs_by_id[run_id] = spec
        results_by_id: dict[str, object] = {}
        for result in experiment.results:
            run_id = _text(getattr(result, "run_id", None))
            if not run_id or run_id in results_by_id:
                raise ValueError("experiment results contain duplicate or empty run ID")
            if run_id not in specs_by_id:
                raise ValueError("experiment result is not present in selected specs")
            results_by_id[run_id] = result
        failures_by_id = {
            _text(getattr(failure, "run_id", None)): failure
            for failure in experiment.failures
            if _optional_text(getattr(failure, "run_id", None)) is not None
        }
        collector = _EntryCollector()
        for run_id, spec in specs_by_id.items():
            result = results_by_id.get(run_id)
            if result is None:
                self._add_scope_entries(collector, spec, expectations)
                failure = failures_by_id.get(run_id)
                if failure is not None:
                    collector.add(
                        EvidenceRef(f"failure:{run_id}", "failure", run_id),
                        EvidenceClass.DETERMINISTIC_FACT,
                        f"Run {run_id} did not produce a finalized result.",
                        {
                            "run_id": run_id,
                            "error_type": _safe_provenance(
                                getattr(failure, "error_type", None), "unknown"
                            ),
                        },
                    )
                collector.add(
                    EvidenceRef(f"limitation:{run_id}:failed", "limitation", run_id),
                    EvidenceClass.DETERMINISTIC_FACT,
                    f"Run {run_id} has no finalized evidence.",
                    {
                        "run_id": run_id,
                        "reason": "run did not produce a finalized result",
                    },
                )
                continue
            self._build_run(
                collector,
                root,
                spec,
                result,
                summary_rows.get(run_id, {}),
                expectations,
            )
        return EvidenceCorpus(
            output_root=root,
            entries=tuple(collector.entries),
            principle_pack_version=_UX_PRINCIPLE_PACK_VERSION,
            principle_pack_digest=_UX_PRINCIPLE_PACK_DIGEST,
            metadata={
                "schema_version": "evidence-corpus-v1",
                "experiment_run_ids": tuple(specs_by_id),
                "ux_principles_are_metadata_only": True,
            },
        )

    @staticmethod
    def _add_scope_entries(
        collector: _EntryCollector,
        spec: object,
        expectations: Mapping[ExpectationKey, FrozenExpectation],
    ) -> None:
        expected = _identity(spec)
        run_id = _text(expected["run_id"])
        scenario_id = _text(expected["scenario_id"], "unknown")
        version_id = _text(expected["application_version_id"], "unknown")
        persona_id = _text(expected["persona_id"], "unknown")
        collector.add(
            EvidenceRef(f"scenario:{run_id}", "scenario", run_id),
            EvidenceClass.DETERMINISTIC_FACT,
            f"Scenario {scenario_id} executed for run {run_id}.",
            {
                "id": scenario_id,
                "name": _text(expected["scenario_name"], scenario_id),
                "application_version_id": version_id,
            },
        )
        collector.add(
            EvidenceRef(f"persona:{run_id}", "persona", run_id),
            EvidenceClass.DETERMINISTIC_FACT,
            f"Persona {persona_id} was assigned to run {run_id}.",
            {
                "id": persona_id,
                "name": _text(expected["persona_name"], persona_id),
            },
        )
        collector.add(
            EvidenceRef(f"goal:{run_id}", "goal", run_id),
            EvidenceClass.DETERMINISTIC_FACT,
            f"Recorded goal for run {run_id}.",
            {
                "scenario_id": scenario_id,
                "text": _text(expected["goal"], "Goal unavailable"),
            },
        )
        key = ExpectationKey(version_id, scenario_id, persona_id)
        expectation = expectations.get(key) or expectations.get(
            ExpectationKey(version_id, scenario_id, "*")
        )
        if expectation is None:
            expectation_payload: dict[str, object] = {
                "matched": False,
                "expectation_id": None,
                "key": {
                    "application_version_id": version_id,
                    "scenario_id": scenario_id,
                    "persona_id": persona_id,
                },
                "reason": "no matching frozen expectation",
            }
        else:
            expectation_payload = {
                "matched": True,
                "expectation_id": expectation.expectation_id,
                "schema_version": expectation.schema_version,
                "key": {
                    "application_version_id": expectation.key.application_version_id,
                    "scenario_id": expectation.key.scenario_id,
                    "persona_id": expectation.key.persona_id,
                },
                "desired_outcomes": expectation.desired_outcomes,
                "required_invariants": expectation.required_invariants,
                "acceptable_alternatives": expectation.acceptable_alternatives,
                "reference_paths": expectation.reference_paths,
                "effort_bounds": expectation.effort_bounds,
                "warning_signals": expectation.warning_signals,
            }
        collector.add(
            EvidenceRef(f"expectation:{run_id}", "expectation", run_id),
            EvidenceClass.DETERMINISTIC_FACT,
            (
                f"Frozen expectation {expectation.expectation_id} matched run {run_id}."
                if expectation is not None
                else f"No frozen expectation matched run {run_id}."
            ),
            expectation_payload,
        )

    def _build_run(
        self,
        collector: _EntryCollector,
        root: Path,
        spec: object,
        result: object,
        summary_row: Mapping[str, object],
        expectations: Mapping[ExpectationKey, FrozenExpectation],
    ) -> None:
        expected = _identity(spec)
        run_id = _text(expected["run_id"])
        bundle_value = getattr(result, "bundle_path", None)
        if bundle_value is None:
            raise ValueError(f"run {run_id} has no finalized bundle path")
        bundle = Path(bundle_value)
        expected_bundle = root / "runs" / run_id
        candidates = (
            (bundle,)
            if bundle.is_absolute()
            else (
                bundle.absolute(),
                root / bundle,
            )
        )
        expected_identity = os.path.normcase(os.path.abspath(expected_bundle))
        matching_bundle = next(
            (
                candidate
                for candidate in candidates
                if os.path.normcase(os.path.abspath(candidate)) == expected_identity
            ),
            None,
        )
        if matching_bundle is None:
            raise ValueError(f"run {run_id} bundle path does not match output root")
        bundle = matching_bundle
        provider_id = _text(expected["prominence_provider_id"], "heuristic")
        failures = finalized_bundle_failures(
            bundle,
            expected_run_id=run_id,
            expected_prominence_provider_id=provider_id,
        )
        if failures:
            raise ValueError(
                f"invalid finalized bundle for {run_id}: {'; '.join(failures)}"
            )
        try:
            persisted = read_finalized_bundle(
                bundle,
                expected_run_id=run_id,
                expected_prominence_provider_id=provider_id,
            )
        except (OSError, RuntimeError, UnicodeError, ValueError) as error:
            raise ValueError(f"invalid finalized bundle for {run_id}") from error
        manifest = persisted.manifest
        raw_result = persisted.result
        events = tuple(persisted.events)
        if _text(raw_result.get("run_id")) != run_id:
            raise ValueError(f"result run ID does not match spec for {run_id}")
        _check_identity(
            f"manifest for {run_id}",
            manifest,
            expected,
            required=frozenset(
                {
                    "run_id",
                    "seed",
                    "model_trial",
                    "config_digest",
                    "scenario_id",
                    "application_version_id",
                    "persona_id",
                    "policy",
                    "prominence_provider_id",
                }
            ),
        )
        _check_identity(
            f"result state for {run_id}",
            _mapping(_mapping(raw_result.get("state")).get("spec")),
            expected,
            required=frozenset(expected),
        )
        _check_identity(
            f"experiment summary for {run_id}",
            summary_row,
            expected,
            required=frozenset(
                {
                    "run_id",
                    "seed",
                    "model_trial",
                    "config_digest",
                    "scenario_id",
                    "application_version_id",
                    "persona_id",
                    "policy",
                    "prominence_provider_id",
                }
            ),
        )
        raw_metrics = _mapping(raw_result.get("metrics"))
        _check_identity(
            f"result metrics for {run_id}",
            raw_metrics,
            expected,
            required=frozenset(
                {
                    "run_id",
                    "seed",
                    "model_trial",
                    "config_digest",
                    "scenario_id",
                    "application_version_id",
                    "persona_id",
                    "policy",
                    "prominence_provider_id",
                }
            ),
        )

        snapshots = tuple(
            snapshot
            for event in events
            if (snapshot := _public_snapshot(event)) is not None
        )
        scenario_id = _text(expected["scenario_id"], "unknown")
        version_id = _text(expected["application_version_id"], "unknown")
        persona_id = _text(expected["persona_id"], "unknown")
        collector.add(
            EvidenceRef(f"scenario:{run_id}", "scenario", run_id),
            EvidenceClass.DETERMINISTIC_FACT,
            f"Scenario {scenario_id} executed for run {run_id}.",
            {
                "id": scenario_id,
                "name": _text(expected["scenario_name"], scenario_id),
                "application_version_id": version_id,
            },
        )
        collector.add(
            EvidenceRef(f"persona:{run_id}", "persona", run_id),
            EvidenceClass.DETERMINISTIC_FACT,
            f"Persona {persona_id} was assigned to run {run_id}.",
            {
                "id": persona_id,
                "name": _text(expected["persona_name"], persona_id),
            },
        )
        collector.add(
            EvidenceRef(f"goal:{run_id}", "goal", run_id),
            EvidenceClass.DETERMINISTIC_FACT,
            f"Recorded goal for run {run_id}.",
            {
                "scenario_id": scenario_id,
                "text": _text(expected["goal"], "Goal unavailable"),
            },
        )
        key = ExpectationKey(version_id, scenario_id, persona_id)
        expectation = expectations.get(key) or expectations.get(
            ExpectationKey(version_id, scenario_id, "*")
        )
        expectation_payload: dict[str, object]
        if expectation is None:
            expectation_payload = {
                "matched": False,
                "expectation_id": None,
                "key": {
                    "application_version_id": version_id,
                    "scenario_id": scenario_id,
                    "persona_id": persona_id,
                },
                "reason": "no matching frozen expectation",
            }
        else:
            expectation_payload = {
                "matched": True,
                "expectation_id": expectation.expectation_id,
                "schema_version": expectation.schema_version,
                "key": {
                    "application_version_id": expectation.key.application_version_id,
                    "scenario_id": expectation.key.scenario_id,
                    "persona_id": expectation.key.persona_id,
                },
                "desired_outcomes": expectation.desired_outcomes,
                "required_invariants": expectation.required_invariants,
                "acceptable_alternatives": expectation.acceptable_alternatives,
                "reference_paths": expectation.reference_paths,
                "effort_bounds": expectation.effort_bounds,
                "warning_signals": expectation.warning_signals,
            }
        collector.add(
            EvidenceRef(f"expectation:{run_id}", "expectation", run_id),
            EvidenceClass.DETERMINISTIC_FACT,
            (
                f"Frozen expectation {expectation.expectation_id} matched run {run_id}."
                if expectation is not None
                else f"No frozen expectation matched run {run_id}."
            ),
            expectation_payload,
        )
        for snapshot in snapshots:
            viewport_id = _text(snapshot.get("id"))
            if not viewport_id:
                continue
            collector.add(
                EvidenceRef(
                    f"viewport:{run_id}:{viewport_id}",
                    "viewport",
                    run_id,
                    viewport_id=viewport_id,
                ),
                EvidenceClass.DETERMINISTIC_FACT,
                f"Captured viewport {viewport_id} in run {run_id}.",
                snapshot,
            )
            for element in _mappings(snapshot.get("elements")):
                element_id = _text(element.get("id"))
                if not element_id:
                    continue
                collector.add(
                    EvidenceRef(
                        f"element:{run_id}:{viewport_id}:{element_id}",
                        "element",
                        run_id,
                        viewport_id=viewport_id,
                        element_id=element_id,
                    ),
                    EvidenceClass.DETERMINISTIC_FACT,
                    f"Element {element_id} was captured in viewport {viewport_id}.",
                    {"viewport_id": viewport_id, **dict(element)},
                )
            self._add_screenshot(collector, root, bundle, run_id, snapshot)

        for event in events:
            kind = _text(event.get("kind"))
            sequence = _sequence(event)
            if kind != "action-executed" or sequence <= 0:
                continue
            action = _safe_action(event.get("action"))
            viewport_id = _optional_text(event.get("viewport_id"))
            if viewport_id is None:
                viewport_id = self._viewport_before(events, sequence)
            payload: dict[str, object] = {
                "sequence": sequence,
                "kind": kind,
                "action": action,
                "succeeded": bool(event.get("succeeded", False)),
            }
            if viewport_id is not None:
                payload["viewport_id"] = viewport_id
            event_ref = EvidenceRef(
                f"event:{run_id}:{sequence}",
                "event",
                run_id,
                viewport_id=viewport_id,
                event_id=f"event-{sequence}",
                replay_sequence=sequence,
            )
            collector.add(
                event_ref,
                EvidenceClass.DETERMINISTIC_FACT,
                f"Executed action at sequence {sequence} in run {run_id}.",
                payload,
            )
            collector.add(
                EvidenceRef(
                    f"replay:{run_id}:{sequence}",
                    "replay",
                    run_id,
                    viewport_id=viewport_id,
                    event_id=f"event-{sequence}",
                    replay_sequence=sequence,
                ),
                EvidenceClass.DETERMINISTIC_FACT,
                f"Replay position {sequence} in run {run_id}.",
                payload,
            )

        verification = self._verification(events, raw_result)
        collector.add(
            EvidenceRef(f"verification:{run_id}", "verification", run_id),
            EvidenceClass.DETERMINISTIC_FACT,
            f"Independent verifier result recorded for run {run_id}.",
            verification,
        )
        metrics = raw_metrics or summary_row
        metric_values = list(
            _metric_values(
                metrics,
                max_discovery_rank=_discovery_rank_bound(events, snapshots),
            )
        )
        outcome_kind = _approved_metric_value(
            "outcome", _mapping(raw_result.get("outcome")).get("kind")
        )
        if outcome_kind is not None and not any(
            name == "outcome" for name, *_ in metric_values
        ):
            metric_values.append(
                (
                    "outcome",
                    outcome_kind,
                    EvidenceClass.DETERMINISTIC_FACT,
                    _mapping(raw_result.get("outcome")),
                )
            )
        for name, value, evidence_class, source in metric_values:
            payload: dict[str, object] = {
                "name": name,
                "value": value,
                "evidence_class": evidence_class.value,
            }
            source_ids = source.get("evidence_ids")
            if _sequence_values(source_ids):
                safe_source_ids = tuple(
                    evidence_id
                    for item in _sequence_values(source_ids)
                    if (evidence_id := _safe_evidence_id(item)) is not None
                )
                if safe_source_ids:
                    payload["source_evidence_ids"] = safe_source_ids
            if evidence_class is EvidenceClass.MODEL_ESTIMATE:
                payload["provenance"] = _provenance(source, metrics, manifest)
            collector.add(
                EvidenceRef(
                    f"metric:{run_id}:{name}", "metric", run_id, metric_id=name
                ),
                evidence_class,
                f"Metric {name} recorded for run {run_id}.",
                payload,
            )
        for event in events:
            self._add_score_estimates(collector, run_id, event, metrics, manifest)
        self._add_limitations(collector, run_id, raw_result)
        self._add_saliency(
            collector,
            root,
            bundle,
            run_id,
            events,
            snapshots,
            provider_id,
        )

    @staticmethod
    def _viewport_before(
        events: Sequence[Mapping[str, object]], sequence: int
    ) -> str | None:
        viewport_id: str | None = None
        for event in events:
            if _sequence(event) > sequence:
                break
            if _text(event.get("kind")) == "viewport-captured":
                viewport_id = _optional_text(_mapping(event.get("snapshot")).get("id"))
        return viewport_id

    @staticmethod
    def _verification(
        events: Sequence[Mapping[str, object]], raw_result: Mapping[str, object]
    ) -> dict[str, object]:
        records = [
            _mapping(event.get("result"))
            for event in events
            if _text(event.get("kind")) == "verification-recorded"
            and _mapping(event.get("result"))
        ]
        verification = (
            records[-1] if records else _mapping(raw_result.get("verification"))
        )
        evidence_ids = verification.get("evidence_ids")
        return {
            "verified": bool(verification.get("verified", False)),
            "evidence_ids": tuple(
                _text(value)
                for value in _sequence_values(evidence_ids)
                if _optional_text(value) is not None
            ),
            "details": _optional_text(verification.get("details")),
        }

    @staticmethod
    def _add_screenshot(
        collector: _EntryCollector,
        root: Path,
        bundle: Path,
        run_id: str,
        snapshot: Mapping[str, object],
    ) -> None:
        artifact = snapshot.get("screenshot_artifact")
        resolved = _root_relative_artifact(root, bundle, artifact)
        if resolved is None:
            return
        root_relative, candidate = resolved
        try:
            content, digest = _attachment_digest(
                candidate,
                "source screenshot",
                max_bytes=_MAX_SCREENSHOT_BYTES,
            )
            media_type = _screenshot_media_type(content)
        except ValueError:
            return
        viewport_id = _text(snapshot.get("id"), "unknown")
        collector.add(
            EvidenceRef(
                f"screenshot:{run_id}:{digest}",
                "screenshot",
                run_id,
                viewport_id=viewport_id,
                artifact_path=root_relative.as_posix(),
                sha256=digest,
            ),
            EvidenceClass.DETERMINISTIC_FACT,
            f"Screenshot for viewport {viewport_id} in run {run_id}.",
            {
                "viewport_id": viewport_id,
                "media_type": media_type,
                "sha256": digest,
                "dimensions": _mapping(snapshot.get("viewport")),
            },
            Path(root_relative),
        )

    @staticmethod
    def _add_score_estimates(
        collector: _EntryCollector,
        run_id: str,
        event: Mapping[str, object],
        metrics: Mapping[str, object],
        manifest: Mapping[str, object],
    ) -> None:
        estimate_kind = _SCORE_EVENT_KINDS.get(_text(event.get("kind")))
        if estimate_kind is None:
            return
        sequence = _sequence(event)
        if sequence <= 0:
            return
        provenance = _provenance(event, metrics, manifest)
        for score in _score_values(event.get("scores")):
            element_id = _optional_text(score.get("element_id"))
            if element_id is None or not _is_id_component(element_id):
                continue
            viewport_id = _optional_text(event.get("viewport_id"))
            if viewport_id is not None and not _is_id_component(viewport_id):
                viewport_id = None
            values: dict[str, object] = {}
            for name in (
                "score",
                "raw_score",
                "normalized_probability",
                "first_notice_probability",
                "notice_within_budget_probability",
            ):
                if score.get(name) is None:
                    continue
                value = _probability(score[name])
                if value is None:
                    values = {}
                    break
                values[name] = value
            if not values:
                continue
            evidence_id = (
                f"model-estimate:{run_id}:{estimate_kind}:{sequence}:{element_id}"
            )
            collector.add(
                EvidenceRef(
                    evidence_id,
                    "model-estimate",
                    run_id,
                    viewport_id=viewport_id,
                    element_id=element_id,
                    event_id=f"event-{sequence}",
                ),
                EvidenceClass.MODEL_ESTIMATE,
                f"{estimate_kind.title()} estimate for element {element_id}.",
                {
                    "estimate_kind": estimate_kind,
                    "sequence": sequence,
                    "element_id": element_id,
                    "values": values,
                    "provenance": provenance,
                },
            )

    @staticmethod
    def _add_limitations(
        collector: _EntryCollector,
        run_id: str,
        raw_result: Mapping[str, object],
    ) -> None:
        for suffix, field_name in (
            ("ux-sample-invalid", "ux_sample_invalid_reason"),
            ("evaluation-failure", "evaluation_failure_reason"),
        ):
            reason = _optional_text(raw_result.get(field_name))
            if reason is None:
                continue
            collector.add(
                EvidenceRef(f"limitation:{run_id}:{suffix}", "limitation", run_id),
                EvidenceClass.DETERMINISTIC_FACT,
                f"Recorded {suffix} limitation for run {run_id}.",
                {"kind": suffix, "reason": reason},
            )
        for index, value in enumerate(_sequence_values(raw_result.get("limitations"))):
            text = _optional_text(value)
            if text is None:
                continue
            collector.add(
                EvidenceRef(f"limitation:{run_id}:{index}", "limitation", run_id),
                EvidenceClass.DETERMINISTIC_FACT,
                f"Recorded limitation for run {run_id}.",
                {"text": text},
            )
        values = raw_result.get("counterevidence")
        for index, value in enumerate(_sequence_values(values)):
            raw = _mapping(value)
            payload: dict[str, object] = {}
            if raw:
                for name in ("kind", "summary", "description", "evidence_ids"):
                    if name == "evidence_ids":
                        ids = raw.get(name)
                        if _sequence_values(ids):
                            payload[name] = tuple(
                                _text(item)
                                for item in _sequence_values(ids)
                                if _optional_text(item) is not None
                            )
                    else:
                        text = _optional_text(raw.get(name))
                        if text is not None:
                            payload[name] = text
            else:
                text = _optional_text(value)
                if text is not None:
                    payload["summary"] = text
            if not payload:
                continue
            collector.add(
                EvidenceRef(
                    f"counterevidence:{run_id}:{index}",
                    "counterevidence",
                    run_id,
                ),
                EvidenceClass.DETERMINISTIC_FACT,
                f"Recorded counterevidence for run {run_id}.",
                payload,
            )

    @staticmethod
    def _add_saliency(
        collector: _EntryCollector,
        root: Path,
        bundle: Path,
        run_id: str,
        events: Sequence[Mapping[str, object]],
        snapshots: Sequence[Mapping[str, object]],
        provider_id: str,
    ) -> None:
        try:
            replay_groups = load_saliency_replay(
                bundle,
                events,
                snapshots,
                expected_provider_id=provider_id,
            )
        except (OSError, RuntimeError, ValueError):
            return
        for group in replay_groups:
            namespace = _optional_text(group.get("artifact_namespace"))
            if namespace is None or not _is_id_component(namespace):
                continue
            metadata = _mapping(group.get("metadata"))
            cache_key = _mapping(metadata.get("cache_key"))
            source_screenshot_sha256 = _safe_checksum(
                cache_key.get("screenshot_sha256")
            )
            if not bool(group.get("replay_available", False)):
                error = _optional_text(group.get("replay_error"))
                if error is not None:
                    collector.add(
                        EvidenceRef(
                            f"limitation:{run_id}:saliency-{namespace}",
                            "limitation",
                            run_id,
                        ),
                        EvidenceClass.DETERMINISTIC_FACT,
                        f"Saliency replay unavailable for {namespace}.",
                        {"text": error},
                    )
                continue
            predictions: list[dict[str, object]] = []
            for prediction in _mappings(metadata.get("predictions")):
                prediction_metadata = _mapping(prediction.get("metadata"))
                predictions.append(
                    {
                        "duration": _text(prediction.get("duration")),
                        "metadata": {
                            name: prediction_metadata.get(name)
                            for name in (
                                "provider_id",
                                "model_id",
                                "provider_version",
                                "model_version",
                                "model_checksum",
                                "input_dimensions",
                                "output_dimensions",
                                "geometry",
                                "preprocessing_version",
                                "execution_provider",
                                "inference_duration_ms",
                                "warnings",
                            )
                            if name in prediction_metadata
                        },
                    }
                )
            collector.add(
                EvidenceRef(
                    f"saliency-metadata:{run_id}:{namespace}",
                    "saliency-metadata",
                    run_id,
                    viewport_id=namespace,
                ),
                EvidenceClass.MODEL_ESTIMATE,
                f"Validated saliency metadata for {namespace}.",
                {
                    "namespace": namespace,
                    "viewport_id": _optional_text(group.get("viewport_id")),
                    "provider_id": _text(group.get("provider_id"), "unavailable"),
                    "active_provider_id": _text(
                        group.get("active_provider_id"), "unavailable"
                    ),
                    "cache_state": _text(group.get("cache_state"), "unavailable"),
                    "model_checksums": tuple(
                        _text(item)
                        for item in _sequence_values(group.get("model_checksums"))
                        if _optional_text(item) is not None
                    ),
                    "execution_provider": _text(
                        group.get("execution_provider"), "unavailable"
                    ),
                    "source_screenshot_sha256": source_screenshot_sha256,
                    "artifact_paths": tuple(
                        _text(item)
                        for item in _sequence_values(metadata.get("artifact_paths"))
                        if _optional_text(item) is not None
                    ),
                    "cache_key_digest": _safe_checksum(
                        metadata.get("cache_key_digest")
                    ),
                    "predictions": predictions,
                    "replay_linkage": True,
                },
            )
            for entry in _mappings(group.get("entries")):
                duration = _text(entry.get("duration"))
                if duration not in _SALIENCY_DURATIONS:
                    continue
                heatmap = EvidenceCorpusBuilder._saliency_attachment(
                    collector,
                    root,
                    bundle,
                    run_id,
                    namespace,
                    duration,
                    "heatmap",
                    entry,
                    source_screenshot_sha256,
                )
                EvidenceCorpusBuilder._saliency_attachment(
                    collector,
                    root,
                    bundle,
                    run_id,
                    namespace,
                    duration,
                    "native-map",
                    entry,
                    source_screenshot_sha256,
                )
                ranked = tuple(
                    _ranked_payload(item)
                    for item in _mappings(entry.get("ranked_elements"))
                )
                for item in ranked:
                    element_id = _optional_text(item.get("element_id"))
                    if element_id is None or not _is_id_component(element_id):
                        continue
                    collector.add(
                        EvidenceRef(
                            f"ranked-element:{run_id}:{namespace}:{duration}:{element_id}",
                            "ranked-element",
                            run_id,
                            viewport_id=namespace,
                            element_id=element_id,
                        ),
                        EvidenceClass.MODEL_ESTIMATE,
                        f"Ranked saliency element {element_id} for {duration}.",
                        {
                            **item,
                            "namespace": namespace,
                            "duration": duration,
                            "provenance": {
                                "provider_id": _text(
                                    entry.get("provider_id"), "unavailable"
                                ),
                                "model_id": _text(entry.get("model_id"), "unavailable"),
                                "model_version": _text(
                                    entry.get("model_version"), "unavailable"
                                ),
                            },
                        },
                    )
                del heatmap

    @staticmethod
    def _saliency_attachment(
        collector: _EntryCollector,
        root: Path,
        bundle: Path,
        run_id: str,
        namespace: str,
        duration: str,
        kind: str,
        entry: Mapping[str, object],
        source_screenshot_sha256: str | None,
    ) -> EvidenceEntry | None:
        filename = f"{duration}-heatmap.png" if kind == "heatmap" else f"{duration}.npz"
        logical_path = f"saliency/{namespace}/{filename}"
        try:
            validate_saliency_artifact_path(logical_path)
        except ValueError:
            return None
        resolved = _root_relative_artifact(root, bundle, logical_path)
        if resolved is None:
            return None
        root_relative, candidate = resolved
        try:
            content, digest = _attachment_digest(
                candidate,
                "saliency attachment",
                max_bytes=(
                    _MAX_SALIENCY_BYTES if kind == "heatmap" else _MAX_NATIVE_MAP_BYTES
                ),
            )
            if kind == "heatmap":
                validate_saliency_heatmap_content(content)
            else:
                validate_saliency_native_map_content(content)
        except ValueError:
            return None
        payload: dict[str, object] = {
            "namespace": namespace,
            "duration": duration,
            "media_type": "image/png"
            if kind == "heatmap"
            else "application/octet-stream",
            "provider_id": _text(entry.get("provider_id"), "unavailable"),
            "model_id": _text(entry.get("model_id"), "unavailable"),
            "model_version": _text(entry.get("model_version"), "unavailable"),
            "model_checksum": _safe_checksum(entry.get("model_checksum")),
            "execution_provider": _text(entry.get("execution_provider"), "unavailable"),
            "inference_duration_ms": _safe_scalar(entry.get("inference_duration_ms")),
            "source_screenshot_sha256": source_screenshot_sha256,
            "replay_linkage": True,
            "ranked_elements": tuple(
                _ranked_payload(item)
                for item in _mappings(entry.get("ranked_elements"))
            ),
        }
        reference = EvidenceRef(
            f"{kind}:{run_id}:{namespace}:{duration}",
            kind,
            run_id,
            viewport_id=_optional_text(entry.get("viewport_id"))
            or _optional_text(entry.get("source_viewport_id"))
            or namespace,
            artifact_path=root_relative.as_posix(),
            sha256=digest,
        )
        collector.add(
            reference,
            EvidenceClass.MODEL_ESTIMATE,
            f"Validated {kind} artifact for {namespace} at {duration}.",
            payload,
            Path(root_relative),
        )
        return collector.by_id.get(reference.evidence_id)


__all__ = [
    "EvidenceCorpus",
    "EvidenceCorpusBuilder",
    "EvidenceEntry",
    "EvidenceResolver",
    "ResolvedEvidence",
    "validate_evidence_refs",
]
