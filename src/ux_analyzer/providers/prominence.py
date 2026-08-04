"""Inspectable heuristic prominence scoring for rendered interface snapshots."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import cast

import yaml

from ux_analyzer.application.saliency import (
    ProminenceResult,
)
from ux_analyzer.domain.interface import (
    ElementRole,
    ElementSnapshot,
    GraphRelation,
    ViewportSnapshot,
)

FEATURE_NAMES = (
    "area",
    "center_distance",
    "reading_order",
    "typography",
    "contrast",
    "isolation",
    "actionability",
    "motion",
    "competition",
    "occlusion",
)

_LOWER_IS_BETTER = frozenset({"center_distance", "reading_order"})
_DEFAULT_WEIGHTS = {
    "area": 0.16,
    "center_distance": 0.14,
    "reading_order": 0.10,
    "typography": 0.12,
    "contrast": 0.12,
    "isolation": 0.10,
    "actionability": 0.14,
    "motion": 0.05,
    "competition": -0.08,
    "occlusion": -0.09,
}
DEFAULT_PROMINENCE_WEIGHTS = MappingProxyType(_DEFAULT_WEIGHTS)


def _empty_feature_override_map() -> dict[str, Mapping[str, float]]:
    return {}


@dataclass(frozen=True, slots=True)
class HeuristicProminenceConfig:
    """Versioned provider settings suitable for storage in YAML."""

    version: str = "heuristic-prominence-v1"
    weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_PROMINENCE_WEIGHTS)
    )
    temperature: float = 1.0
    viewport_width: float | None = None
    viewport_height: float | None = None
    feature_overrides: Mapping[str, Mapping[str, float]] = field(
        default_factory=_empty_feature_override_map
    )

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("prominence config version must not be empty")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("prominence temperature must be greater than zero")
        unknown = set(self.weights).difference(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"unknown prominence features: {sorted(unknown)}")
        if not self.weights:
            raise ValueError("prominence config needs at least one weight")
        normalized_weights: dict[str, float] = {}
        for name, weight in self.weights.items():
            if not math.isfinite(weight):
                raise ValueError(f"weight for {name!r} must be finite")
            normalized_weights[name] = float(weight)
        object.__setattr__(self, "weights", MappingProxyType(normalized_weights))
        overrides: dict[str, Mapping[str, float]] = {}
        for element_id, values in self.feature_overrides.items():
            if not element_id:
                raise ValueError("feature override element ID must not be empty")
            unknown_features = set(values).difference(FEATURE_NAMES)
            if unknown_features:
                raise ValueError(
                    f"unknown feature overrides: {sorted(unknown_features)}"
                )
            copied: dict[str, float] = {}
            for name, value in values.items():
                if not math.isfinite(value):
                    raise ValueError(f"feature override {name!r} must be finite")
                copied[name] = float(value)
            overrides[element_id] = MappingProxyType(copied)
        object.__setattr__(self, "feature_overrides", MappingProxyType(overrides))
        for name, value in (
            ("viewport_width", self.viewport_width),
            ("viewport_height", self.viewport_height),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be greater than zero")

    @property
    def raw_feature_overrides(self) -> Mapping[str, Mapping[str, float]]:
        """Explicit name for overrides that supply raw feature measurements."""

        return self.feature_overrides

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> HeuristicProminenceConfig:
        """Build config from a decoded YAML mapping."""

        version = payload.get("version", "heuristic-prominence-v1")
        weights = payload.get("weights", dict(DEFAULT_PROMINENCE_WEIGHTS))
        if not isinstance(version, str) or not isinstance(weights, Mapping):
            raise ValueError("prominence config needs text version and object weights")
        weight_mapping = cast(Mapping[object, object], weights)
        parsed_weights: dict[str, float] = {
            str(name): _finite_number(value, f"weight for {name!r}")
            for name, value in weight_mapping.items()
        }
        overrides_payload = payload.get("feature_overrides", {})
        if not isinstance(overrides_payload, Mapping):
            raise ValueError("feature_overrides must be an object")
        override_mapping = cast(Mapping[object, object], overrides_payload)
        parsed_overrides: dict[str, dict[str, float]] = {}
        for element_id, values in override_mapping.items():
            if not isinstance(values, Mapping):
                raise ValueError("each feature override must be an object")
            feature_mapping = cast(Mapping[object, object], values)
            parsed_overrides[str(element_id)] = {
                str(name): _finite_number(value, f"feature override {name!r}")
                for name, value in feature_mapping.items()
            }
        return cls(
            version=version,
            weights=parsed_weights,
            temperature=_finite_number(payload.get("temperature", 1.0), "temperature"),
            viewport_width=_optional_number(
                payload.get("viewport_width"), "viewport_width"
            ),
            viewport_height=_optional_number(
                payload.get("viewport_height"), "viewport_height"
            ),
            feature_overrides=parsed_overrides,
        )

    @classmethod
    def from_yaml(cls, path: Path) -> HeuristicProminenceConfig:
        """Load versioned weights through a structured YAML parser."""

        with path.open(encoding="utf-8") as config_file:
            payload = yaml.safe_load(config_file)
        if not isinstance(payload, Mapping):
            raise ValueError("prominence YAML root must be an object")
        return cls.from_mapping(cast(Mapping[str, object], payload))


class HeuristicProminenceProvider:
    """Calculate deterministic, weighted prominence for visible elements."""

    id = "heuristic-prominence"

    def __init__(self, config: HeuristicProminenceConfig | None = None) -> None:
        self.config = config

    @property
    def version(self) -> str:
        return (self.config or HeuristicProminenceConfig()).version

    def score(
        self,
        snapshot: ViewportSnapshot,
        config: HeuristicProminenceConfig | None = None,
    ) -> tuple[ProminenceResult, ...]:
        """Score snapshot elements while retaining every feature calculation."""

        settings = config or self.config or HeuristicProminenceConfig()
        elements = snapshot.elements
        if not elements:
            return ()
        near_counts = _near_counts(snapshot)
        viewport_width, viewport_height = _viewport_size(snapshot, settings)
        raw_by_feature = {
            feature: tuple(
                _raw_feature(
                    feature,
                    index,
                    element,
                    near_counts.get(element.id, 0),
                    viewport_width,
                    viewport_height,
                    settings,
                )
                for index, element in enumerate(elements)
            )
            for feature in FEATURE_NAMES
        }
        normalized_by_feature = {
            feature: _normalize(values, feature in _LOWER_IS_BETTER)
            for feature, values in raw_by_feature.items()
        }
        raw_scores: list[float] = []
        contributions: list[dict[str, float]] = []
        for index in range(len(elements)):
            item_contributions = {
                feature: settings.weights.get(feature, 0.0)
                * normalized_by_feature[feature][index]
                for feature in FEATURE_NAMES
            }
            contributions.append(item_contributions)
            raw_scores.append(sum(item_contributions.values()))
        probabilities = _softmax(raw_scores, settings.temperature)
        return tuple(
            ProminenceResult(
                element_id=element.id,
                raw_score=raw_scores[index],
                normalized_probability=probabilities[index],
                first_notice_probability=probabilities[index],
                notice_within_budget_probability=probabilities[index],
                feature_contributions=contributions[index],
                raw_values={
                    feature: raw_by_feature[feature][index] for feature in FEATURE_NAMES
                },
                normalized_values={
                    feature: normalized_by_feature[feature][index]
                    for feature in FEATURE_NAMES
                },
            )
            for index, element in enumerate(elements)
        )


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_number(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _finite_number(value, name)


def _viewport_size(
    snapshot: ViewportSnapshot, config: HeuristicProminenceConfig
) -> tuple[float, float]:
    inferred_width = max(
        element.bounds.x + element.bounds.width for element in snapshot.elements
    )
    inferred_height = max(
        element.bounds.y + element.bounds.height for element in snapshot.elements
    )
    return config.viewport_width or max(
        inferred_width, 1.0
    ), config.viewport_height or max(inferred_height, 1.0)


def _near_counts(snapshot: ViewportSnapshot) -> dict[str, int]:
    counts = {element.id: 0 for element in snapshot.elements}
    for edge in snapshot.graph_edges:
        if GraphRelation(edge.relation) is not GraphRelation.NEAR:
            continue
        if edge.source_id in counts:
            counts[edge.source_id] += 1
        if edge.target_id in counts:
            counts[edge.target_id] += 1
    return counts


def _raw_feature(
    feature: str,
    index: int,
    element: ElementSnapshot,
    near_count: int,
    viewport_width: float,
    viewport_height: float,
    config: HeuristicProminenceConfig,
) -> float:
    override = config.feature_overrides.get(element.id, {}).get(feature)
    if override is not None:
        return override
    bounds = element.bounds
    if feature == "area":
        return bounds.width * bounds.height
    if feature == "center_distance":
        center_x = bounds.x + bounds.width / 2
        center_y = bounds.y + bounds.height / 2
        return math.hypot(center_x - viewport_width / 2, center_y - viewport_height / 2)
    if feature == "reading_order":
        return float(index)
    if feature == "typography":
        role_strength = {
            ElementRole.BUTTON: 0.8,
            ElementRole.LINK: 0.7,
            ElementRole.INPUT: 0.65,
            ElementRole.CHECKBOX: 0.6,
            ElementRole.TAB: 0.7,
            ElementRole.MENU: 0.55,
            ElementRole.TEXT: 0.75,
            ElementRole.OTHER: 0.4,
        }[ElementRole(element.role)]
        return role_strength + min(len(element.label) / 40, 1.0) * 0.2
    if feature == "contrast":
        return element.local_contrast if element.local_contrast is not None else 0.5
    if feature == "isolation":
        return 1 / (1 + near_count)
    if feature == "actionability":
        return float(element.actionable and not element.disabled)
    if feature == "motion":
        return 0.0
    if feature == "competition":
        return float(near_count)
    if feature == "occlusion":
        return (
            element.occlusion_fraction
            if element.occlusion_fraction is not None
            else 1 - element.visibility_fraction
        )
    raise ValueError(f"unknown prominence feature: {feature}")


def _normalize(values: Sequence[float], reverse: bool) -> tuple[float, ...]:
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        return tuple(0.5 for _ in values)
    normalized = tuple((value - minimum) / (maximum - minimum) for value in values)
    if reverse:
        return tuple(1 - value for value in normalized)
    return normalized


def _softmax(logits: Sequence[float], temperature: float) -> tuple[float, ...]:
    if not logits:
        return ()
    safe_temperature = max(temperature, 1e-12)
    maximum = max(logits)
    exponentials = [math.exp((value - maximum) / safe_temperature) for value in logits]
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)
