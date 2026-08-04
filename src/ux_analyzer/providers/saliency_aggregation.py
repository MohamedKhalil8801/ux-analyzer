"""Convert native saliency planes into inspectable element profiles."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ux_analyzer.domain.interface import ElementRole, ElementSnapshot, ViewportSnapshot
from ux_analyzer.domain.saliency import (
    SALIENCY_GEOMETRY_VERSION,
    AttentionDuration,
    AttentionEstimate,
    AttentionEstimateKind,
    ElementAttentionProfile,
    ElementSaliencyAggregate,
    SaliencyGeometry,
    SaliencyPrediction,
    SaliencyPredictionProvenance,
    SaliencyPredictionSet,
)

DEFAULT_AGGREGATION_VERSION = "element-saliency-aggregation-v1"
DEFAULT_DENSITY_WEIGHT = 0.60
DEFAULT_ROBUST_PEAK_WEIGHT = 0.25
DEFAULT_MASS_SHARE_WEIGHT = 0.15

_DEFAULT_SEMANTIC_ROLES = (
    ElementRole.BUTTON,
    ElementRole.CHECKBOX,
    ElementRole.INPUT,
    ElementRole.LINK,
    ElementRole.MENU,
    ElementRole.TAB,
    ElementRole.TEXT,
)


def _default_semantic_roles() -> tuple[ElementRole, ...]:
    return _DEFAULT_SEMANTIC_ROLES


def _default_structural_roles() -> tuple[ElementRole, ...]:
    return (ElementRole.OTHER,)


@dataclass(frozen=True, slots=True)
class SaliencyAggregationConfig:
    """Versioned formula and operational-candidate policy."""

    version: str = DEFAULT_AGGREGATION_VERSION
    density_weight: float = DEFAULT_DENSITY_WEIGHT
    robust_peak_weight: float = DEFAULT_ROBUST_PEAK_WEIGHT
    mass_share_weight: float = DEFAULT_MASS_SHARE_WEIGHT
    temperature: float = 1.0
    meaningful_score_threshold: float = 1e-9
    semantic_roles: tuple[ElementRole | str, ...] = field(
        default_factory=_default_semantic_roles
    )
    structural_roles: tuple[ElementRole | str, ...] = field(
        default_factory=_default_structural_roles
    )

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("saliency aggregation config version must not be empty")
        weights = (
            ("density_weight", self.density_weight),
            ("robust_peak_weight", self.robust_peak_weight),
            ("mass_share_weight", self.mass_share_weight),
        )
        for name, weight in weights:
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if sum(weight for _, weight in weights) <= 0:
            raise ValueError("saliency aggregation weights need positive total")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError(
                "saliency aggregation temperature must be greater than zero"
            )
        if (
            not math.isfinite(self.meaningful_score_threshold)
            or self.meaningful_score_threshold < 0
        ):
            raise ValueError(
                "meaningful_score_threshold must be finite and non-negative"
            )

        semantic_roles = _normalize_roles(self.semantic_roles, "semantic_roles")
        structural_roles = _normalize_roles(self.structural_roles, "structural_roles")
        if not semantic_roles:
            raise ValueError("semantic_roles must not be empty")
        if not structural_roles:
            raise ValueError("structural_roles must not be empty")
        object.__setattr__(self, "semantic_roles", semantic_roles)
        object.__setattr__(self, "structural_roles", structural_roles)


@dataclass(frozen=True, slots=True)
class _SourceBounds:
    left: float
    top: float
    right: float
    bottom: float

    @property
    def area(self) -> float:
        return max(0.0, self.right - self.left) * max(0.0, self.bottom - self.top)


@dataclass(frozen=True, slots=True)
class _DurationEvidence:
    prediction: SaliencyPrediction
    values: np.ndarray[Any, Any]
    aggregates: Mapping[str, ElementSaliencyAggregate]
    source_bounds: Mapping[str, _SourceBounds]


def aggregate_saliency(
    snapshot: ViewportSnapshot,
    predictions: SaliencyPredictionSet | Sequence[SaliencyPrediction],
    config: SaliencyAggregationConfig | None = None,
) -> tuple[ElementAttentionProfile, ...]:
    """Aggregate every snapshot element while exposing operational candidates."""

    settings = config or SaliencyAggregationConfig()
    prediction_items = _prediction_items(snapshot, predictions)
    evidence_by_duration = tuple(
        _aggregate_duration(
            snapshot,
            prediction,
            settings,
        )
        for prediction in prediction_items
    )
    estimates_by_element = _estimates_by_element(
        snapshot, evidence_by_duration, settings
    )
    prediction_provenance = tuple(
        SaliencyPredictionProvenance(
            duration=evidence.prediction.duration,
            metadata=evidence.prediction.metadata,
        )
        for evidence in evidence_by_duration
    )
    aggregates_by_element: dict[str, list[ElementSaliencyAggregate]] = {
        element.id: [] for element in snapshot.elements
    }
    for evidence in evidence_by_duration:
        for element in snapshot.elements:
            aggregates_by_element[element.id].append(evidence.aggregates[element.id])

    profiles: list[ElementAttentionProfile] = []
    for element in snapshot.elements:
        duration_estimates = estimates_by_element[element.id]
        profiles.append(
            ElementAttentionProfile(
                viewport_id=snapshot.id,
                element_id=element.id,
                immediate=duration_estimates.get(AttentionDuration.ONE_SECOND),
                early=duration_estimates.get(AttentionDuration.THREE_SECONDS),
                eventual=duration_estimates.get(AttentionDuration.SEVEN_SECONDS),
                general=duration_estimates.get(AttentionDuration.GENERAL),
                aggregates=tuple(aggregates_by_element[element.id]),
                aggregation_version=settings.version,
                prediction_provenance=prediction_provenance,
            )
        )
    return tuple(profiles)


def _normalize_roles(
    roles: Sequence[ElementRole | str], name: str
) -> tuple[ElementRole, ...]:
    try:
        normalized = tuple(ElementRole(role) for role in roles)
    except ValueError as error:
        raise ValueError(f"{name} contains unsupported element role") from error
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must not contain duplicate roles")
    return normalized


def _prediction_items(
    snapshot: ViewportSnapshot,
    predictions: SaliencyPredictionSet | Sequence[SaliencyPrediction],
) -> tuple[SaliencyPrediction, ...]:
    if isinstance(predictions, SaliencyPredictionSet):
        if predictions.viewport_id != snapshot.id:
            raise ValueError("saliency predictions belong to a different viewport")
        items = predictions.predictions
    else:
        items = tuple(predictions)
    if not items:
        raise ValueError("saliency predictions must not be empty")
    durations = tuple(item.duration for item in items)
    if len(durations) != len(set(durations)):
        raise ValueError("duplicate duration in saliency predictions")
    for prediction in items:
        if prediction.viewport_id != snapshot.id:
            raise ValueError("saliency prediction belongs to a different viewport")
    return items


def _aggregate_duration(
    snapshot: ViewportSnapshot,
    prediction: SaliencyPrediction,
    config: SaliencyAggregationConfig,
) -> _DurationEvidence:
    values = _plane_array(prediction)
    geometry = prediction.metadata.geometry
    if geometry.geometry_version != SALIENCY_GEOMETRY_VERSION:
        raise ValueError(
            f"unsupported saliency geometry version: {geometry.geometry_version!r}"
        )
    source_width, source_height = geometry.source_dimensions
    dpr = geometry.device_pixel_ratio
    zoom = geometry.zoom
    total_mass = _visible_map_mass(values, geometry)
    aggregates: dict[str, ElementSaliencyAggregate] = {}
    source_bounds: dict[str, _SourceBounds] = {}
    for element in snapshot.elements:
        bounds = _source_bounds(element, source_width, source_height, dpr, zoom)
        source_bounds[element.id] = bounds
        map_bounds = _source_bounds_to_map(
            geometry,
            bounds,
            map_width=prediction.plane.width,
            map_height=prediction.plane.height,
        )
        region = _sample_region(values, map_bounds)
        density, robust_peak, raw_mass = _region_statistics(region)
        mass_share = _safe_ratio(raw_mass, total_mass)
        raw_score = _raw_score(
            density,
            robust_peak,
            mass_share,
            config,
        )
        visibility = _clamp_probability(element.visibility_fraction)
        occlusion = _clamp_probability(element.occlusion_fraction or 0.0)
        # Snapshot visibility already combines geometric clipping and occlusion.
        adjusted_score = _finite_nonnegative(raw_score * visibility)
        aggregates[element.id] = ElementSaliencyAggregate(
            viewport_id=snapshot.id,
            element_id=element.id,
            duration=prediction.duration,
            density=density,
            robust_peak=robust_peak,
            raw_mass=raw_mass,
            mass_share=mass_share,
            clipped_area=bounds.area,
            visibility_fraction=visibility,
            occlusion_fraction=occlusion,
            raw_score=raw_score,
            adjusted_score=adjusted_score,
        )
    return _DurationEvidence(
        prediction=prediction,
        values=values,
        aggregates=aggregates,
        source_bounds=source_bounds,
    )


def _plane_array(prediction: SaliencyPrediction) -> np.ndarray[Any, Any]:
    values = np.asarray(prediction.plane.float_values(), dtype=np.float64)
    try:
        return values.reshape((prediction.plane.height, prediction.plane.width))
    except ValueError as error:
        raise ValueError("saliency plane shape does not match dimensions") from error


def _source_bounds(
    element: ElementSnapshot,
    source_width: int,
    source_height: int,
    dpr: float,
    zoom: float,
) -> _SourceBounds:
    scale = dpr * zoom
    left = _scaled_edge(element.bounds.x, scale, source_width)
    top = _scaled_edge(element.bounds.y, scale, source_height)
    right = _scaled_edge(element.bounds.x + element.bounds.width, scale, source_width)
    bottom = _scaled_edge(
        element.bounds.y + element.bounds.height, scale, source_height
    )
    return _SourceBounds(
        left=min(left, right),
        top=min(top, bottom),
        right=max(left, right),
        bottom=max(top, bottom),
    )


def _source_bounds_to_map(
    geometry: SaliencyGeometry,
    bounds: _SourceBounds,
    *,
    map_width: int,
    map_height: int,
) -> tuple[float, float, float, float]:
    source_width, source_height = geometry.source_dimensions
    native_width, native_height = geometry.native_dimensions
    content_width, content_height = geometry.content_dimensions
    input_bounds = (
        geometry.pad_left + bounds.left * content_width / source_width,
        geometry.pad_top + bounds.top * content_height / source_height,
        geometry.pad_left + bounds.right * content_width / source_width,
        geometry.pad_top + bounds.bottom * content_height / source_height,
    )
    return (
        input_bounds[0] * map_width / native_width,
        input_bounds[1] * map_height / native_height,
        input_bounds[2] * map_width / native_width,
        input_bounds[3] * map_height / native_height,
    )


def _visible_map_mass(
    values: np.ndarray[Any, Any], geometry: SaliencyGeometry
) -> float:
    map_width = values.shape[1]
    map_height = values.shape[0]
    content_bounds = (
        geometry.pad_left * map_width / geometry.native_dimensions[0],
        geometry.pad_top * map_height / geometry.native_dimensions[1],
        geometry.content_dimensions[0] * map_width / geometry.native_dimensions[0]
        + geometry.pad_left * map_width / geometry.native_dimensions[0],
        geometry.content_dimensions[1] * map_height / geometry.native_dimensions[1]
        + geometry.pad_top * map_height / geometry.native_dimensions[1],
    )
    region = _sample_region(values, content_bounds)
    if region is None:
        return 0.0
    return _finite_nonnegative(float(np.sum(region, dtype=np.float64)))


def _scaled_edge(value: float, scale: float, limit: int) -> float:
    if value <= 0:
        return 0.0
    scaled = value * scale
    if not math.isfinite(scaled):
        return float(limit)
    return min(float(limit), max(0.0, scaled))


def _sample_region(
    values: np.ndarray[Any, Any],
    map_bounds: tuple[float, float, float, float],
) -> np.ndarray[Any, Any] | None:
    left, top, right, bottom = map_bounds
    left = min(float(values.shape[1]), max(0.0, left))
    right = min(float(values.shape[1]), max(0.0, right))
    top = min(float(values.shape[0]), max(0.0, top))
    bottom = min(float(values.shape[0]), max(0.0, bottom))
    if right <= left or bottom <= top:
        return None
    x_start = max(0, min(values.shape[1] - 1, int(math.floor(left))))
    y_start = max(0, min(values.shape[0] - 1, int(math.floor(top))))
    x_stop = min(values.shape[1], max(x_start + 1, int(math.ceil(right))))
    y_stop = min(values.shape[0], max(y_start + 1, int(math.ceil(bottom))))
    if x_stop <= x_start or y_stop <= y_start:
        return None
    return values[y_start:y_stop, x_start:x_stop]


def _region_statistics(
    region: np.ndarray[Any, Any] | None,
) -> tuple[float, float, float]:
    if region is None or region.size == 0:
        return 0.0, 0.0, 0.0
    density = _clamp_probability(float(np.mean(region, dtype=np.float64)))
    robust_peak = _clamp_probability(float(np.percentile(region, 95)))
    raw_mass = _finite_nonnegative(float(np.sum(region, dtype=np.float64)))
    return density, robust_peak, raw_mass


def _raw_score(
    density: float,
    robust_peak: float,
    mass_share: float,
    config: SaliencyAggregationConfig,
) -> float:
    result = (
        config.density_weight * density
        + config.robust_peak_weight * robust_peak
        + config.mass_share_weight * math.sqrt(_clamp_probability(mass_share))
    )
    return _clamp_probability(result)


def _estimates_by_element(
    snapshot: ViewportSnapshot,
    evidence: Sequence[_DurationEvidence],
    config: SaliencyAggregationConfig,
) -> dict[str, dict[AttentionDuration, AttentionEstimate]]:
    estimates: dict[str, dict[AttentionDuration, AttentionEstimate]] = {
        element.id: {} for element in snapshot.elements
    }
    for duration_evidence in evidence:
        duration = duration_evidence.prediction.duration
        candidates = {
            element.id: duration_evidence.aggregates[element.id].adjusted_score
            for element in snapshot.elements
            if _is_visible_candidate(element, duration_evidence.aggregates[element.id])
        }
        for element in snapshot.elements:
            if element.id not in candidates:
                continue
            if _suppressed_structural_container(
                element,
                snapshot.elements,
                duration_evidence.source_bounds,
                duration_evidence.aggregates,
                config,
            ):
                del candidates[element.id]
        probabilities = _softmax(candidates, config.temperature)
        for element_id, probability in probabilities.items():
            estimates[element_id][_as_duration(duration)] = AttentionEstimate(
                kind=AttentionEstimateKind.PREDICTED,
                score=probability,
                source=_evidence_source(duration_evidence.prediction),
            )
    return estimates


def _is_visible_candidate(
    element: ElementSnapshot, aggregate: ElementSaliencyAggregate
) -> bool:
    return (
        element.visibility_fraction > 0
        and aggregate.clipped_area > 0
        and aggregate.adjusted_score >= 0
    )


def _suppressed_structural_container(
    element: ElementSnapshot,
    elements: Sequence[ElementSnapshot],
    source_bounds: Mapping[str, _SourceBounds],
    aggregates: Mapping[str, ElementSaliencyAggregate],
    config: SaliencyAggregationConfig,
) -> bool:
    role = ElementRole(element.role)
    if element.actionable or role not in config.structural_roles:
        return False
    if role in config.semantic_roles:
        return False
    parent = source_bounds[element.id]
    for child in elements:
        if child.id == element.id:
            continue
        child_bounds = source_bounds[child.id]
        if not _strictly_contains(parent, child_bounds):
            continue
        child_aggregate = aggregates[child.id]
        if (
            (child.actionable or ElementRole(child.role) in config.semantic_roles)
            and child.visibility_fraction > 0
            and child_aggregate.clipped_area > 0
            and child_aggregate.adjusted_score > config.meaningful_score_threshold
        ):
            return True
    return False


def _strictly_contains(parent: _SourceBounds, child: _SourceBounds) -> bool:
    if parent.area <= child.area:
        return False
    return (
        parent.left <= child.left
        and parent.top <= child.top
        and parent.right >= child.right
        and parent.bottom >= child.bottom
    )


def _softmax(scores: Mapping[str, float], temperature: float) -> dict[str, float]:
    if not scores:
        return {}
    identifiers = tuple(scores)
    logits = tuple(
        _finite_nonnegative(scores[element_id]) / temperature
        for element_id in identifiers
    )
    maximum = max(logits)
    exponentials = tuple(math.exp(min(logit - maximum, 0.0)) for logit in logits)
    denominator = sum(exponentials)
    if not math.isfinite(denominator) or denominator <= 0:
        probability = 1.0 / len(identifiers)
        return {element_id: probability for element_id in identifiers}
    return {
        element_id: _clamp_probability(exponentials[index] / denominator)
        for index, element_id in enumerate(identifiers)
    }


def _as_duration(duration: AttentionDuration | str) -> AttentionDuration:
    return AttentionDuration(duration)


def _evidence_source(prediction: SaliencyPrediction) -> str:
    """Return provider provenance used by normalized attention estimates."""

    return prediction.metadata.provider_id


def _clamp_probability(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def _finite_nonnegative(value: float) -> float:
    if not math.isfinite(value) or value < 0:
        return 0.0
    return value


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0 or not math.isfinite(denominator):
        return 0.0
    return _clamp_probability(numerator / denominator)


__all__ = [
    "DEFAULT_AGGREGATION_VERSION",
    "DEFAULT_DENSITY_WEIGHT",
    "DEFAULT_MASS_SHARE_WEIGHT",
    "DEFAULT_ROBUST_PEAK_WEIGHT",
    "SaliencyAggregationConfig",
    "aggregate_saliency",
]
