"""Immutable domain contracts for screenshot saliency evidence."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

SALIENCY_GEOMETRY_VERSION = "saliency-geometry-v1"


class AttentionDuration(StrEnum):
    """Attention horizon represented by a saliency prediction."""

    ONE_SECOND = "1s"
    THREE_SECONDS = "3s"
    SEVEN_SECONDS = "7s"
    GENERAL = "general"


class AttentionEstimateKind(StrEnum):
    """Provenance class for an element attention estimate."""

    PREDICTED = "predicted"
    DERIVED = "derived"
    FALLBACK = "fallback"
    UNAVAILABLE = "unavailable"


class SearchStage(StrEnum):
    """Operational search stage used by later prominence selection."""

    INITIAL = "initial"
    EXPLORATION = "exploration"
    PERSISTENT = "persistent"


def _require_text(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _require_sha256(value: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("screenshot_sha256 must be lowercase SHA-256")


def _positive_dimension(name: str, value: int) -> None:
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _dimensions(name: str, value: tuple[int, int]) -> tuple[int, int]:
    try:
        dimensions = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must contain width and height") from error
    if len(dimensions) != 2:
        raise ValueError(f"{name} must contain width and height")
    width, height = dimensions
    _positive_dimension(f"{name} width", width)
    _positive_dimension(f"{name} height", height)
    return width, height


def _finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _probability(name: str, value: float) -> None:
    _finite(name, value)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def _nonnegative(name: str, value: float) -> None:
    _finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must not be negative")


def _unique_durations(
    durations: Iterable[AttentionDuration | str],
) -> tuple[AttentionDuration, ...]:
    normalized = tuple(AttentionDuration(duration) for duration in durations)
    if len(normalized) != len(set(normalized)):
        raise ValueError("duplicate duration in saliency predictions")
    return normalized


@dataclass(frozen=True, slots=True)
class SaliencyPlane:
    """Normalized little-endian float32 plane without a NumPy dependency."""

    width: int
    height: int
    values: bytes

    def __post_init__(self) -> None:
        _positive_dimension("plane width", self.width)
        _positive_dimension("plane height", self.height)
        try:
            values = bytes(self.values)
        except (TypeError, ValueError) as error:
            raise ValueError("plane values must be bytes") from error
        expected_length = self.width * self.height * 4
        if len(values) != expected_length:
            raise ValueError(
                f"plane byte length must be {expected_length}, got {len(values)}"
            )
        for (value,) in struct.iter_unpack("<f", values):
            _probability("plane value", value)
        object.__setattr__(self, "values", values)

    @property
    def byte_length(self) -> int:
        """Return serialized plane size in bytes."""

        return len(self.values)

    def float_values(self) -> tuple[float, ...]:
        """Return values in row-major order using only the standard library."""

        return tuple(value[0] for value in struct.iter_unpack("<f", self.values))


@dataclass(frozen=True, slots=True)
class SaliencyGeometry:
    """Domain-neutral reversible geometry for one prediction result."""

    geometry_version: str
    source_dimensions: tuple[int, int]
    native_dimensions: tuple[int, int]
    content_dimensions: tuple[int, int]
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    scale: float
    scale_x: float
    scale_y: float
    device_pixel_ratio: float
    zoom: float

    def __post_init__(self) -> None:
        _require_text("geometry_version", self.geometry_version)
        source_dimensions = _dimensions("source dimensions", self.source_dimensions)
        native_dimensions = _dimensions("native dimensions", self.native_dimensions)
        content_dimensions = _dimensions("content dimensions", self.content_dimensions)
        if any(
            type(offset) is not int or offset < 0
            for offset in (
                self.pad_left,
                self.pad_top,
                self.pad_right,
                self.pad_bottom,
            )
        ):
            raise ValueError("geometry padding must be non-negative integers")
        content_width, content_height = content_dimensions
        native_width, native_height = native_dimensions
        if (
            self.pad_left + content_width + self.pad_right != native_width
            or self.pad_top + content_height + self.pad_bottom != native_height
        ):
            raise ValueError("geometry content and padding must fill native dimensions")
        for name, value in (
            ("geometry scale", self.scale),
            ("geometry scale_x", self.scale_x),
            ("geometry scale_y", self.scale_y),
            ("device_pixel_ratio", self.device_pixel_ratio),
            ("zoom", self.zoom),
        ):
            _finite(name, value)
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        object.__setattr__(self, "source_dimensions", source_dimensions)
        object.__setattr__(self, "native_dimensions", native_dimensions)
        object.__setattr__(self, "content_dimensions", content_dimensions)


@dataclass(frozen=True, slots=True)
class SaliencyRequestMetadata:
    """Sanitized, reproducible inputs requested from a saliency provider."""

    viewport_id: str
    screenshot_sha256: str
    screenshot_width: int
    screenshot_height: int
    device_pixel_ratio: float
    zoom: float
    requested_durations: tuple[AttentionDuration | str, ...]
    model_set: tuple[str, ...]
    precision: str
    execution_provider_preference: str

    def __post_init__(self) -> None:
        _require_text("viewport_id", self.viewport_id)
        _require_sha256(self.screenshot_sha256)
        _positive_dimension("screenshot width", self.screenshot_width)
        _positive_dimension("screenshot height", self.screenshot_height)
        _finite("device_pixel_ratio", self.device_pixel_ratio)
        if self.device_pixel_ratio <= 0:
            raise ValueError("device_pixel_ratio must be greater than zero")
        _finite("zoom", self.zoom)
        if self.zoom <= 0:
            raise ValueError("zoom must be greater than zero")
        durations = _unique_durations(self.requested_durations)
        if not durations:
            raise ValueError("requested durations must not be empty")
        models = tuple(self.model_set)
        if not models or any(not model.strip() for model in models):
            raise ValueError("model_set must contain non-empty model IDs")
        if len(models) != len(set(models)):
            raise ValueError("model_set must not contain duplicate model IDs")
        _require_text("precision", self.precision)
        _require_text(
            "execution_provider_preference", self.execution_provider_preference
        )
        object.__setattr__(self, "requested_durations", durations)
        object.__setattr__(self, "model_set", models)

    @property
    def screenshot_dimensions(self) -> tuple[int, int]:
        """Return screenshot width and height as one immutable pair."""

        return self.screenshot_width, self.screenshot_height

    @property
    def dpr(self) -> float:
        """Short vocabulary alias for device pixel ratio."""

        return self.device_pixel_ratio


@dataclass(frozen=True, slots=True)
class SaliencyRequest:
    """Screenshot bytes plus metadata supplied to a saliency provider."""

    screenshot: bytes
    metadata: SaliencyRequestMetadata

    def __post_init__(self) -> None:
        try:
            screenshot = bytes(self.screenshot)
        except (TypeError, ValueError) as error:
            raise ValueError("screenshot must be bytes") from error
        if not screenshot:
            raise ValueError("screenshot must not be empty")
        actual_sha256 = hashlib.sha256(screenshot).hexdigest()
        if actual_sha256 != self.metadata.screenshot_sha256:
            raise ValueError("screenshot_sha256 does not match screenshot bytes")
        object.__setattr__(self, "screenshot", screenshot)


@dataclass(frozen=True, slots=True)
class SaliencyPredictionMetadata:
    """Provider and runtime provenance for one predicted saliency plane."""

    provider_id: str
    model_id: str
    provider_version: str
    model_version: str
    model_checksum: str
    input_dimensions: tuple[int, int]
    output_dimensions: tuple[int, int]
    geometry: SaliencyGeometry
    preprocessing_version: str
    inference_duration_ms: float
    execution_provider: str
    warnings: tuple[str, ...] = ()
    cache_state: str = "miss"

    def __post_init__(self) -> None:
        for name, value in (
            ("provider_id", self.provider_id),
            ("model_id", self.model_id),
            ("provider_version", self.provider_version),
            ("model_version", self.model_version),
            ("model_checksum", self.model_checksum),
            ("preprocessing_version", self.preprocessing_version),
            ("execution_provider", self.execution_provider),
            ("cache_state", self.cache_state),
        ):
            _require_text(name, value)
        input_dimensions = _dimensions("input dimensions", self.input_dimensions)
        output_dimensions = _dimensions("output dimensions", self.output_dimensions)
        if self.geometry.native_dimensions != input_dimensions:
            raise ValueError("geometry native dimensions must match input dimensions")
        _nonnegative("inference duration", self.inference_duration_ms)
        warnings = tuple(self.warnings)
        if any(not warning.strip() for warning in warnings):
            raise ValueError("warnings must contain non-empty strings")
        object.__setattr__(self, "input_dimensions", input_dimensions)
        object.__setattr__(self, "output_dimensions", output_dimensions)
        object.__setattr__(self, "warnings", warnings)


@dataclass(frozen=True, slots=True)
class SaliencyPredictionProvenance:
    """Replay metadata for one duration without duplicating its native map."""

    duration: AttentionDuration | str
    metadata: SaliencyPredictionMetadata

    def __post_init__(self) -> None:
        object.__setattr__(self, "duration", AttentionDuration(self.duration))


@dataclass(frozen=True, slots=True)
class AttentionEstimate:
    """One normalized attention estimate with explicit evidence source."""

    kind: AttentionEstimateKind | str
    score: float | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        kind = AttentionEstimateKind(self.kind)
        if kind is AttentionEstimateKind.UNAVAILABLE:
            if self.score is not None:
                raise ValueError("unavailable estimate must not carry a score")
        else:
            if self.score is None:
                raise ValueError("attention estimate score is required")
            _probability("attention estimate score", self.score)
            if self.source is None or not self.source.strip():
                raise ValueError("attention estimate source is required")
        if self.source is not None and not self.source.strip():
            raise ValueError("attention estimate source must not be empty")
        object.__setattr__(self, "kind", kind)


@dataclass(frozen=True, slots=True)
class ElementSaliencyAggregate:
    """Per-element evidence retained from one native saliency plane."""

    viewport_id: str
    element_id: str
    duration: AttentionDuration | str
    density: float
    robust_peak: float
    raw_mass: float
    mass_share: float
    clipped_area: float
    visibility_fraction: float
    occlusion_fraction: float
    raw_score: float
    adjusted_score: float

    def __post_init__(self) -> None:
        _require_text("viewport_id", self.viewport_id)
        _require_text("element_id", self.element_id)
        object.__setattr__(self, "duration", AttentionDuration(self.duration))
        for name, value in (
            ("density", self.density),
            ("robust_peak", self.robust_peak),
            ("mass_share", self.mass_share),
            ("visibility_fraction", self.visibility_fraction),
            ("occlusion_fraction", self.occlusion_fraction),
            ("raw_score", self.raw_score),
            ("adjusted_score", self.adjusted_score),
        ):
            _probability(name, value)
        _nonnegative("raw_mass", self.raw_mass)
        _nonnegative("clipped_area", self.clipped_area)

    @property
    def p95(self) -> float:
        """Report robust peak under the percentile vocabulary."""

        return self.robust_peak


@dataclass(frozen=True, slots=True)
class ElementAttentionProfile:
    """Immutable duration estimates and aggregate provenance for one element."""

    viewport_id: str
    element_id: str
    immediate: AttentionEstimate | None
    early: AttentionEstimate | None
    eventual: AttentionEstimate | None
    general: AttentionEstimate | None
    aggregates: tuple[ElementSaliencyAggregate, ...]
    aggregation_version: str = ""
    prediction_provenance: tuple[SaliencyPredictionProvenance, ...] = ()

    def __post_init__(self) -> None:
        _require_text("viewport_id", self.viewport_id)
        _require_text("element_id", self.element_id)
        _require_text("aggregation_version", self.aggregation_version)
        aggregates = tuple(self.aggregates)
        durations = tuple(aggregate.duration for aggregate in aggregates)
        if len(durations) != len(set(durations)):
            raise ValueError("duplicate duration in element attention profile")
        for aggregate in aggregates:
            if aggregate.viewport_id != self.viewport_id:
                raise ValueError("aggregate belongs to different viewport")
            if aggregate.element_id != self.element_id:
                raise ValueError("aggregate belongs to different element")
        provenance = tuple(self.prediction_provenance)
        if not provenance:
            raise ValueError("prediction provenance must not be empty")
        provenance_durations = tuple(item.duration for item in provenance)
        if len(provenance_durations) != len(set(provenance_durations)):
            raise ValueError("duplicate duration in prediction provenance")
        object.__setattr__(self, "aggregates", aggregates)
        object.__setattr__(self, "prediction_provenance", provenance)

    def estimate_for(
        self, duration: AttentionDuration | str
    ) -> AttentionEstimate | None:
        """Return estimate associated with one attention duration."""

        normalized = AttentionDuration(duration)
        return {
            AttentionDuration.ONE_SECOND: self.immediate,
            AttentionDuration.THREE_SECONDS: self.early,
            AttentionDuration.SEVEN_SECONDS: self.eventual,
            AttentionDuration.GENERAL: self.general,
        }[normalized]


@dataclass(frozen=True, slots=True)
class ElementAttentionProfileSet:
    """Viewport-scoped collection that prevents duplicate element profiles."""

    viewport_id: str
    profiles: tuple[ElementAttentionProfile, ...]

    def __post_init__(self) -> None:
        _require_text("viewport_id", self.viewport_id)
        profiles = tuple(self.profiles)
        element_ids = tuple(profile.element_id for profile in profiles)
        if len(element_ids) != len(set(element_ids)):
            raise ValueError("duplicate element profile")
        for profile in profiles:
            if profile.viewport_id != self.viewport_id:
                raise ValueError("profile belongs to different viewport")
        object.__setattr__(self, "profiles", profiles)


@dataclass(frozen=True, slots=True)
class SaliencyPrediction:
    """One duration-specific prediction and its provenance."""

    viewport_id: str
    duration: AttentionDuration | str
    plane: SaliencyPlane
    metadata: SaliencyPredictionMetadata

    def __post_init__(self) -> None:
        _require_text("viewport_id", self.viewport_id)
        duration = AttentionDuration(self.duration)
        if self.metadata.output_dimensions != (self.plane.width, self.plane.height):
            raise ValueError("prediction output dimensions must match plane dimensions")
        object.__setattr__(self, "duration", duration)


@dataclass(frozen=True, slots=True)
class SaliencyPredictionSet:
    """All duration predictions produced for one captured viewport."""

    viewport_id: str
    predictions: tuple[SaliencyPrediction, ...]
    request_metadata: SaliencyRequestMetadata | None = None

    def __post_init__(self) -> None:
        _require_text("viewport_id", self.viewport_id)
        predictions = tuple(self.predictions)
        if not predictions:
            raise ValueError("saliency prediction set must not be empty")
        durations = _unique_durations(prediction.duration for prediction in predictions)
        if self.request_metadata is not None:
            if self.request_metadata.viewport_id != self.viewport_id:
                raise ValueError("request metadata belongs to different viewport")
            requested = set(self.request_metadata.requested_durations)
            if any(duration not in requested for duration in durations):
                raise ValueError("prediction duration was not requested")
        for prediction in predictions:
            if prediction.viewport_id != self.viewport_id:
                raise ValueError("prediction belongs to different viewport")
            if self.request_metadata is not None:
                geometry = prediction.metadata.geometry
                if (
                    geometry.source_dimensions
                    != self.request_metadata.screenshot_dimensions
                ):
                    raise ValueError(
                        "prediction geometry source dimensions do not match request"
                    )
                if (
                    geometry.device_pixel_ratio
                    != self.request_metadata.device_pixel_ratio
                ):
                    raise ValueError("prediction geometry DPR does not match request")
                if geometry.zoom != self.request_metadata.zoom:
                    raise ValueError("prediction geometry zoom does not match request")
        object.__setattr__(self, "predictions", predictions)

    def prediction_for(self, duration: AttentionDuration | str) -> SaliencyPrediction:
        """Return prediction for one duration or reject missing evidence."""

        normalized = AttentionDuration(duration)
        for prediction in self.predictions:
            if prediction.duration is normalized:
                return prediction
        raise ValueError(f"no saliency prediction for duration {normalized.value!r}")


# Names used by adapters and callers that emphasize prediction boundaries.
SaliencyPredictionRequest = SaliencyRequest
SaliencyProfileSet = ElementAttentionProfileSet


__all__ = [
    "AttentionDuration",
    "AttentionEstimate",
    "AttentionEstimateKind",
    "ElementAttentionProfile",
    "ElementAttentionProfileSet",
    "ElementSaliencyAggregate",
    "SALIENCY_GEOMETRY_VERSION",
    "SaliencyPlane",
    "SaliencyGeometry",
    "SaliencyPrediction",
    "SaliencyPredictionMetadata",
    "SaliencyPredictionProvenance",
    "SaliencyPredictionRequest",
    "SaliencyPredictionSet",
    "SaliencyProfileSet",
    "SaliencyRequest",
    "SaliencyRequestMetadata",
    "SearchStage",
]
