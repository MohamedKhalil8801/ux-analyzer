"""Adapt model-estimate attention profiles into staged prominence scores."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Protocol, cast

from ux_analyzer.application.saliency import ProminenceBatch, ProminenceResult
from ux_analyzer.domain.interface import ViewportSnapshot
from ux_analyzer.domain.run import ProviderManifest
from ux_analyzer.domain.saliency import (
    SALIENCY_GEOMETRY_VERSION,
    AttentionDuration,
    AttentionEstimate,
    AttentionEstimateKind,
    ElementAttentionProfile,
    SaliencyPrediction,
    SaliencyPredictionRequest,
    SaliencyPredictionSet,
    SaliencyRequestMetadata,
    SearchStage,
)
from ux_analyzer.ports.artifacts import RunBundleWriter
from ux_analyzer.ports.observation import ObservationCapture
from ux_analyzer.providers.saliency_aggregation import (
    SaliencyAggregationConfig,
    aggregate_saliency,
)
from ux_analyzer.storage.saliency_cache import (
    SaliencyCacheEntry,
    SaliencyCacheKey,
    SaliencyCacheMetadata,
    SaliencyCachePayload,
    materialize_saliency_payload_into_bundle,
)


class _SaliencyModelProvider(Protocol):
    model_id: str

    def predict(self, request: SaliencyPredictionRequest) -> SaliencyPredictionSet: ...


class _SaliencyCache(Protocol):
    def load(self, key: SaliencyCacheKey) -> SaliencyCacheEntry | None: ...

    def store(
        self, key: SaliencyCacheKey, payload: SaliencyCachePayload
    ) -> SaliencyCacheEntry: ...

    def materialize_into_bundle(
        self,
        entry: SaliencyCacheEntry,
        writer: RunBundleWriter,
        *,
        provider_manifests: Sequence[ProviderManifest] = (),
        source_viewport_id: str | None = None,
    ) -> object: ...


class _HeuristicProvider(Protocol):
    id: str

    def score(self, snapshot: ViewportSnapshot) -> Sequence[ProminenceResult]: ...


STAGE_MIXTURES: Mapping[SearchStage, Mapping[AttentionDuration, float]] = {
    SearchStage.INITIAL: {AttentionDuration.ONE_SECOND: 1.0},
    SearchStage.EXPLORATION: {AttentionDuration.THREE_SECONDS: 1.0},
    SearchStage.PERSISTENT: {
        AttentionDuration.THREE_SECONDS: 0.25,
        AttentionDuration.SEVEN_SECONDS: 0.75,
    },
}

_INFERENCE_CACHE_VIEWPORT_ID = "native-inference"
_MIN_SOFTMAX_TEMPERATURE = 1e-12

_ModelProviderSource = _SaliencyModelProvider | Callable[[], _SaliencyModelProvider]
_CacheSource = _SaliencyCache | Callable[[], _SaliencyCache]


class AttentionStageSelector:
    """Select one normalized learned prominence distribution per search stage."""

    def __init__(
        self,
        *,
        version: str = "saliency-stage-selector-v1",
        temperature: float = 1.0,
        mixtures: Mapping[SearchStage | str, Mapping[AttentionDuration | str, float]]
        | None = None,
    ) -> None:
        if not version.strip():
            raise ValueError("stage selector version must not be empty")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("stage selector temperature must be greater than zero")
        self.version = version
        self.temperature = temperature
        self.mixtures = _normalize_stage_mixtures(mixtures)

    def select(
        self,
        profiles: Sequence[ElementAttentionProfile],
        stage: SearchStage | str,
    ) -> tuple[ProminenceResult, ...]:
        normalized_stage = SearchStage(stage)
        mixture = self.mixtures[normalized_stage]
        profile_items = tuple(profiles)
        element_ids = tuple(profile.element_id for profile in profile_items)
        if len(element_ids) != len(set(element_ids)):
            raise ValueError("stage profiles must not contain duplicate elements")
        if not profile_items:
            return ()

        scored_profiles: list[ElementAttentionProfile] = []
        raw_scores: list[float] = []
        evidence: list[tuple[str, str]] = []
        raw_components: list[dict[str, float]] = []
        weighted_contributions: list[dict[str, float]] = []
        for profile in profile_items:
            score = 0.0
            component_values: dict[str, float] = {}
            fallback_used = False
            unavailable = False
            source = "foveacast"
            fallback_source: str | None = None
            for duration, weight in mixture.items():
                estimate = profile.estimate_for(duration)
                if (
                    estimate is None
                    or estimate.kind is AttentionEstimateKind.UNAVAILABLE
                ):
                    estimate = _fallback_estimate(profile, duration)
                    fallback_used = True
                    if estimate is not None:
                        fallback_source = estimate.source
                if estimate is None:
                    unavailable = True
                    component_values[duration.value] = 0.0
                    continue
                estimate_score = estimate.score or 0.0
                score += weight * estimate_score
                component_values[duration.value] = estimate_score
                if estimate.kind is AttentionEstimateKind.FALLBACK:
                    fallback_used = True
                if estimate.source is not None:
                    source = estimate.source
            if unavailable:
                continue
            elif fallback_used:
                evidence.append(
                    ("fallback", fallback_source or "missing-duration-fallback")
                )
            else:
                evidence.append(("predicted", source))
            raw_scores.append(score)
            raw_components.append(component_values)
            weighted_contributions.append(
                {
                    f"stage_{duration.value}": weight * component_values[duration.value]
                    for duration, weight in mixture.items()
                }
            )
            scored_profiles.append(profile)

        probabilities = _softmax(raw_scores, self.temperature)
        return tuple(
            ProminenceResult(
                element_id=profile.element_id,
                raw_score=raw_scores[index],
                normalized_probability=probabilities[index],
                first_notice_probability=probabilities[index],
                notice_within_budget_probability=probabilities[index],
                raw_values={
                    f"stage_{duration}": value
                    for duration, value in raw_components[index].items()
                },
                normalized_values={
                    f"stage_{duration}": value
                    for duration, value in raw_components[index].items()
                },
                feature_contributions=weighted_contributions[index],
                provider_id="foveacast",
                stage=normalized_stage.value,
                evidence_kind=evidence[index][0],
                evidence_source=evidence[index][1],
            )
            for index, profile in enumerate(scored_profiles)
        )


@dataclass(frozen=True, slots=True)
class HybridProminenceConfig:
    """Versioned simple blend kept disabled pending focused comparison."""

    version: str = "hybrid-prominence-v1"
    learned_weight: float = 0.70
    enabled: bool = False

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("hybrid version must not be empty")
        if not math.isfinite(self.learned_weight) or not 0 <= self.learned_weight <= 1:
            raise ValueError("hybrid learned weight must be between 0 and 1")


class SimpleHybridProminenceProvider:
    """Blend normalized learned and heuristic components when explicitly invoked."""

    id = "hybrid-prominence"

    def __init__(self, config: HybridProminenceConfig | None = None) -> None:
        self.config = config or HybridProminenceConfig()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def version(self) -> str:
        return self.config.version

    def blend(
        self,
        learned: Sequence[ProminenceResult],
        heuristic: Sequence[ProminenceResult],
    ) -> tuple[ProminenceResult, ...]:
        learned_items = tuple(learned)
        heuristic_items = tuple(heuristic)
        learned_ids = tuple(item.element_id for item in learned_items)
        heuristic_ids = tuple(item.element_id for item in heuristic_items)
        if len(learned_ids) != len(set(learned_ids)):
            raise ValueError("hybrid learned components contain duplicate elements")
        if len(heuristic_ids) != len(set(heuristic_ids)):
            raise ValueError("hybrid heuristic components contain duplicate elements")
        heuristic_by_id = {item.element_id: item for item in heuristic_items}
        if set(learned_ids) != set(heuristic_by_id):
            raise ValueError("hybrid components must score same elements")
        weight = self.config.learned_weight
        blended_raw = tuple(
            weight * learned_item.normalized_probability
            + (1 - weight)
            * heuristic_by_id[learned_item.element_id].normalized_probability
            for learned_item in learned_items
        )
        probabilities = _normalize_probabilities(blended_raw)
        return tuple(
            ProminenceResult(
                element_id=learned_item.element_id,
                raw_score=blended_raw[index],
                normalized_probability=probabilities[index],
                first_notice_probability=probabilities[index],
                notice_within_budget_probability=probabilities[index],
                feature_contributions={
                    "learned_probability": weight * learned_item.normalized_probability,
                    "heuristic_probability": (1 - weight)
                    * heuristic_by_id[learned_item.element_id].normalized_probability,
                },
                raw_values={
                    "learned_probability": learned_item.normalized_probability,
                    "heuristic_probability": heuristic_by_id[
                        learned_item.element_id
                    ].normalized_probability,
                },
                normalized_values={
                    "learned_probability": learned_item.normalized_probability,
                    "heuristic_probability": heuristic_by_id[
                        learned_item.element_id
                    ].normalized_probability,
                },
                component_provenance={
                    "learned": learned_item.evidence_source or learned_item.provider_id,
                    "heuristic": heuristic_by_id[
                        learned_item.element_id
                    ].evidence_source
                    or heuristic_by_id[learned_item.element_id].provider_id,
                },
                provider_id=self.id,
                stage=learned_item.stage,
                evidence_kind="derived",
                evidence_source=self.version,
            )
            for index, learned_item in enumerate(learned_items)
        )


class FoveacastProminenceProvider:
    """Compose Foveacast inference, cache, aggregation, and staged selection."""

    id = "foveacast-prominence"
    version = "foveacast-prominence-v1"

    def __init__(
        self,
        model_provider: _ModelProviderSource,
        cache: _CacheSource | None = None,
        *,
        aggregation_config: SaliencyAggregationConfig | None = None,
        aggregator: Callable[
            [ViewportSnapshot, SaliencyPredictionSet, SaliencyAggregationConfig],
            tuple[ElementAttentionProfile, ...],
        ] = aggregate_saliency,
        stage_selector: AttentionStageSelector | None = None,
        heuristic_provider: _HeuristicProvider | None = None,
        device_pixel_ratio: float = 1.0,
        zoom: float = 1.0,
        cache_enabled: bool = True,
        cache_scope: str = "experiment",
        model_set: Sequence[str] | None = None,
        precision: str | None = None,
        execution_provider_preference: str | None = None,
        fallback_enabled: bool = True,
        fallback_provider_id: str = "heuristic",
    ) -> None:
        self._model_provider_source = model_provider
        self._cache_source = cache
        self._model_provider: _SaliencyModelProvider | None = None
        self._cache: _SaliencyCache | None = None
        self.aggregation_config = aggregation_config or SaliencyAggregationConfig()
        self.aggregator = aggregator
        self.stage_selector = stage_selector or AttentionStageSelector(
            temperature=self.aggregation_config.temperature,
        )
        self.heuristic_provider = heuristic_provider
        self.device_pixel_ratio = device_pixel_ratio
        self.zoom = zoom
        if type(cache_enabled) is not bool:
            raise ValueError("cache_enabled must be boolean")
        if cache_scope != "experiment":
            raise ValueError("saliency cache scope must be 'experiment'")
        self.cache_enabled = cache_enabled
        self.cache_scope = cache_scope
        self.model_set = _normalize_model_set(model_set)
        self.precision = _normalize_optional_text(precision, "saliency precision")
        self.execution_provider_preference = _normalize_optional_text(
            execution_provider_preference,
            "execution provider preference",
        )
        if type(fallback_enabled) is not bool:
            raise ValueError("fallback_enabled must be boolean")
        if fallback_provider_id not in {"heuristic", "heuristic-prominence"}:
            raise ValueError("saliency fallback provider must be 'heuristic'")
        self.fallback_enabled = fallback_enabled
        self.fallback_provider_id = fallback_provider_id
        if not math.isfinite(device_pixel_ratio) or device_pixel_ratio <= 0:
            raise ValueError("device_pixel_ratio must be greater than zero")
        if not math.isfinite(zoom) or zoom <= 0:
            raise ValueError("zoom must be greater than zero")

    def score(
        self,
        capture: ObservationCapture,
        snapshot: ViewportSnapshot,
        stage: SearchStage | str,
        artifacts: RunBundleWriter | None,
    ) -> ProminenceBatch:
        normalized_stage = SearchStage(stage)
        if capture.viewport_id != snapshot.id:
            raise ValueError(
                "saliency capture and snapshot belong to different viewports"
            )
        try:
            if not self.cache_enabled and artifacts is None:
                raise RuntimeError(
                    "cache-disabled saliency requires typed artifact writer"
                )
            model_provider = self._get_model_provider()
            request = self._request(capture, model_provider)
            cache_key = self._request_cache_key(request, model_provider)
            cache = self._get_cache() if self.cache_enabled else None
            if self.cache_enabled and cache_key is not None:
                if cache is None:
                    raise RuntimeError("saliency cache is unavailable")
                entry = cache.load(cache_key)
                if entry is not None:
                    native_predictions = _rebind_prediction_set(
                        entry.predictions, snapshot.id
                    )
                    profiles = tuple(
                        self.aggregator(
                            snapshot,
                            native_predictions,
                            self.aggregation_config,
                        )
                    )
                    self._materialize(
                        entry, artifacts, cache, source_viewport_id=snapshot.id
                    )
                    return self._learned_batch(
                        snapshot,
                        normalized_stage,
                        profiles,
                        cache_state="hit",
                    )

            predictions = model_provider.predict(request)
            profiles = tuple(
                self.aggregator(snapshot, predictions, self.aggregation_config)
            )
            cache_state = "miss"
            cache_predictions = _rebind_prediction_set(
                predictions, _INFERENCE_CACHE_VIEWPORT_ID
            )
            cache_profiles = _rebind_profiles(profiles, _INFERENCE_CACHE_VIEWPORT_ID)
            if cache_key is None:
                raise RuntimeError("saliency model checksums are required")
            payload = SaliencyCachePayload(
                predictions=cache_predictions,
                profiles=cache_profiles,
                metadata=SaliencyCacheMetadata(
                    provider_manifests=(self._provider_manifest(),),
                    aggregation_version=self.aggregation_config.version,
                ),
            )
            if self.cache_enabled:
                if cache is None:
                    raise RuntimeError("saliency cache is unavailable")
                entry = cache.store(cache_key, payload)
                self._materialize(
                    entry, artifacts, cache, source_viewport_id=snapshot.id
                )
                cache_state = entry.cache_state
            else:
                if artifacts is None:
                    raise RuntimeError(
                        "cache-disabled saliency requires artifact writer"
                    )
                materialize_saliency_payload_into_bundle(
                    payload,
                    cache_key,
                    artifacts,
                    provider_manifests=(self._provider_manifest(),),
                )
                cache_state = "disabled"
            return self._learned_batch(
                snapshot,
                normalized_stage,
                profiles,
                cache_state=cache_state,
            )
        except Exception as error:
            return self._fallback_batch(snapshot, normalized_stage, error)

    def _request(
        self,
        capture: ObservationCapture,
        model_provider: _SaliencyModelProvider | None = None,
    ) -> SaliencyPredictionRequest:
        provider = model_provider or self._get_model_provider()
        model_id = str(getattr(provider, "model_id", "foveacast-v0.2.0"))
        configured_model_set = self.model_set
        if configured_model_set is not None and model_id not in configured_model_set:
            raise ValueError(
                f"configured saliency model set does not include provider model {model_id!r}"
            )
        precision = self.precision or str(getattr(provider, "precision", "fp16"))
        preference = self.execution_provider_preference or str(
            getattr(provider, "requested_execution_provider", "cpu")
        )
        metadata = SaliencyRequestMetadata(
            viewport_id=capture.viewport_id,
            screenshot_sha256=hashlib.sha256(capture.screenshot).hexdigest(),
            screenshot_width=capture.viewport.width,
            screenshot_height=capture.viewport.height,
            device_pixel_ratio=self.device_pixel_ratio,
            zoom=self.zoom,
            requested_durations=(
                AttentionDuration.ONE_SECOND,
                AttentionDuration.THREE_SECONDS,
                AttentionDuration.SEVEN_SECONDS,
            ),
            model_set=configured_model_set or (model_id,),
            precision=precision,
            execution_provider_preference=preference,
        )
        return SaliencyPredictionRequest(
            screenshot=capture.screenshot, metadata=metadata
        )

    def _request_cache_key(
        self,
        request: SaliencyPredictionRequest,
        model_provider: _SaliencyModelProvider | None = None,
    ) -> SaliencyCacheKey | None:
        provider = model_provider or self._get_model_provider()
        checksums = _model_checksums(provider)
        if checksums is None:
            return None
        return SaliencyCacheKey(
            viewport_id=_INFERENCE_CACHE_VIEWPORT_ID,
            screenshot_sha256=request.metadata.screenshot_sha256,
            screenshot_dimensions=request.metadata.screenshot_dimensions,
            device_pixel_ratio=request.metadata.device_pixel_ratio,
            zoom=request.metadata.zoom,
            model_checksums=checksums,
            preprocessing_version=str(
                getattr(
                    provider,
                    "preprocessing_version",
                    "foveacast-preprocess-v1",
                )
            ),
            precision=request.metadata.precision,
            execution_provider=str(
                getattr(
                    provider,
                    "actual_execution_provider",
                    getattr(
                        provider,
                        "execution_provider",
                        "CPUExecutionProvider",
                    ),
                )
            ),
            aggregation_version=self.aggregation_config.version,
            geometry_version=SALIENCY_GEOMETRY_VERSION,
        )

    def _materialize(
        self,
        entry: SaliencyCacheEntry,
        artifacts: RunBundleWriter | None,
        cache: _SaliencyCache | None = None,
        *,
        source_viewport_id: str | None = None,
    ) -> None:
        if artifacts is not None:
            (cache or self._get_cache()).materialize_into_bundle(
                entry,
                artifacts,
                provider_manifests=(self._provider_manifest(),),
                source_viewport_id=source_viewport_id,
            )

    def _provider_manifest(self) -> ProviderManifest:
        model_provider = self._get_model_provider()
        model_id = getattr(model_provider, "model_id", None)
        return ProviderManifest(
            provider_id=str(getattr(model_provider, "id", "foveacast")),
            role="prominence",
            model_id=str(model_id) if model_id is not None else None,
            endpoint_origin="internal",
            version=str(
                getattr(
                    model_provider,
                    "model_version",
                    getattr(model_provider, "provider_version", "foveacast-adapter-v1"),
                )
            ),
        )

    def _learned_batch(
        self,
        snapshot: ViewportSnapshot,
        stage: SearchStage,
        profiles: tuple[ElementAttentionProfile, ...],
        *,
        cache_state: str,
    ) -> ProminenceBatch:
        scores = self.stage_selector.select(profiles, stage)
        scored_ids = {score.element_id for score in scores}
        unavailable_profiles = tuple(
            profile for profile in profiles if profile.element_id not in scored_ids
        )
        return ProminenceBatch(
            scores=scores,
            provider_id=self.id,
            active_provider_id="foveacast",
            stage=stage,
            selected_mixture=_selected_mixture(self.stage_selector, stage),
            learned_profiles=profiles,
            unavailable_learned_profiles=unavailable_profiles,
            learned_scores=scores,
            learned_available=True,
            cache_state=cache_state,
        )

    def _fallback_batch(
        self,
        snapshot: ViewportSnapshot,
        stage: SearchStage,
        error: Exception,
    ) -> ProminenceBatch:
        reason = str(error).strip() or type(error).__name__
        if not self.fallback_enabled:
            raise RuntimeError(f"saliency fallback disabled: {reason}") from error
        heuristic = self.heuristic_provider
        if heuristic is None:
            from ux_analyzer.providers.prominence import HeuristicProminenceProvider

            heuristic = HeuristicProminenceProvider()
        try:
            scores = tuple(heuristic.score(snapshot))
        except Exception as heuristic_error:
            reason = f"{reason}; heuristic fallback failed: {heuristic_error}"
            scores = ()
        return ProminenceBatch(
            scores=scores,
            provider_id=self.id,
            active_provider_id=str(getattr(heuristic, "id", "heuristic-prominence")),
            stage=stage,
            selected_mixture=_selected_mixture(self.stage_selector, stage),
            heuristic_scores=scores,
            learned_available=False,
            learned_unavailable_reason=reason,
            fallback_reason=reason,
            cache_state="fallback",
        )

    def _get_model_provider(self) -> _SaliencyModelProvider:
        if self._model_provider is None:
            source = self._model_provider_source
            candidate = (
                source()
                if callable(source) and not hasattr(source, "predict")
                else source
            )
            if not hasattr(candidate, "predict"):
                raise TypeError("saliency model provider must expose predict")
            self._model_provider = cast(_SaliencyModelProvider, candidate)
        return self._model_provider

    def _get_cache(self) -> _SaliencyCache:
        if self._cache is None:
            if self._cache_source is None:
                raise TypeError("saliency cache is required when cache is enabled")
            source = self._cache_source
            candidate = source() if callable(source) else source
            if not hasattr(candidate, "load") or not hasattr(candidate, "store"):
                raise TypeError("saliency cache must expose load and store")
            self._cache = candidate
        return self._cache


def _normalize_stage_mixtures(
    mixtures: Mapping[SearchStage | str, Mapping[AttentionDuration | str, float]]
    | None,
) -> Mapping[SearchStage, Mapping[AttentionDuration, float]]:
    configured = dict(STAGE_MIXTURES)
    if mixtures is not None:
        for raw_stage, raw_mixture in mixtures.items():
            try:
                stage = SearchStage(raw_stage)
            except (TypeError, ValueError) as error:
                raise ValueError("stage mixtures contain unsupported stage") from error
            if not raw_mixture:
                raise ValueError(f"stage mixture for {stage.value} must not be empty")
            normalized: dict[AttentionDuration, float] = {}
            for raw_duration, raw_weight in raw_mixture.items():
                try:
                    duration = AttentionDuration(raw_duration)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "stage mixtures contain unsupported attention duration"
                    ) from error
                if duration is AttentionDuration.GENERAL:
                    raise ValueError("stage mixtures must use timed durations")
                if (
                    type(raw_weight) is bool
                    or not math.isfinite(raw_weight)
                    or raw_weight < 0
                ):
                    raise ValueError(
                        "stage mixture weights must be finite and non-negative"
                    )
                normalized[duration] = float(raw_weight)
            if sum(normalized.values()) <= 0:
                raise ValueError(
                    f"stage mixture for {stage.value} needs positive weight"
                )
            configured[stage] = normalized
    return MappingProxyType(
        {
            stage: MappingProxyType(dict(mixture))
            for stage, mixture in configured.items()
        }
    )


def _selected_mixture(
    selector: AttentionStageSelector, stage: SearchStage
) -> tuple[tuple[str, float], ...]:
    return tuple(
        (duration.value, float(weight))
        for duration, weight in selector.mixtures[stage].items()
        if weight > 0
    )


def _normalize_model_set(model_set: Sequence[str] | None) -> tuple[str, ...] | None:
    if model_set is None:
        return None
    normalized = tuple(model_set)
    if not normalized or any(not model_id.strip() for model_id in normalized):
        raise ValueError("saliency model set must contain non-empty IDs")
    if len(normalized) != len(set(normalized)):
        raise ValueError("saliency model set must contain unique IDs")
    return normalized


def _normalize_optional_text(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _fallback_estimate(
    profile: ElementAttentionProfile,
    missing_duration: AttentionDuration,
) -> AttentionEstimate | None:
    for duration in (
        AttentionDuration.ONE_SECOND,
        AttentionDuration.THREE_SECONDS,
        AttentionDuration.SEVEN_SECONDS,
        AttentionDuration.GENERAL,
    ):
        estimate = profile.estimate_for(duration)
        if estimate is None or estimate.kind is AttentionEstimateKind.UNAVAILABLE:
            continue
        return AttentionEstimate(
            kind=AttentionEstimateKind.FALLBACK,
            score=estimate.score,
            source=f"missing-{missing_duration.value}:{estimate.source}",
        )
    return None


def _rebind_predictions(
    predictions: Sequence[SaliencyPrediction], viewport_id: str
) -> tuple[SaliencyPrediction, ...]:
    return tuple(
        replace(prediction, viewport_id=viewport_id) for prediction in predictions
    )


def _rebind_prediction_set(
    predictions: SaliencyPredictionSet, viewport_id: str
) -> SaliencyPredictionSet:
    request_metadata = predictions.request_metadata
    if request_metadata is not None:
        request_metadata = replace(request_metadata, viewport_id=viewport_id)
    return replace(
        predictions,
        viewport_id=viewport_id,
        predictions=_rebind_predictions(predictions.predictions, viewport_id),
        request_metadata=request_metadata,
    )


def _rebind_profiles(
    profiles: Sequence[ElementAttentionProfile], viewport_id: str
) -> tuple[ElementAttentionProfile, ...]:
    return tuple(
        replace(
            profile,
            viewport_id=viewport_id,
            aggregates=tuple(
                replace(aggregate, viewport_id=viewport_id)
                for aggregate in profile.aggregates
            ),
        )
        for profile in profiles
    )


def _model_checksums(provider: object) -> tuple[str, str, str] | None:
    values: Mapping[object, object] | None = None
    configured = getattr(provider, "model_checksums", None)
    if isinstance(configured, Mapping):
        values = cast(Mapping[object, object], configured)
    else:
        artifacts = getattr(provider, "_artifacts", None)
        if isinstance(artifacts, Mapping):
            artifact_mapping = cast(Mapping[object, object], artifacts)
            values = {
                duration: _artifact_checksum(status)
                for duration, status in artifact_mapping.items()
            }
    if values is None:
        return None
    checksums: list[str] = []
    for duration in (
        AttentionDuration.ONE_SECOND,
        AttentionDuration.THREE_SECONDS,
        AttentionDuration.SEVEN_SECONDS,
    ):
        checksum = values.get(duration, values.get(duration.value))
        if not isinstance(checksum, str) or len(checksum) != 64:
            return None
        checksums.append(checksum)
    return checksums[0], checksums[1], checksums[2]


def _artifact_checksum(status: object) -> str | None:
    artifact = getattr(status, "artifact", None)
    checksum = getattr(artifact, "sha256", None)
    return checksum if isinstance(checksum, str) else None


def _normalize_probabilities(values: Sequence[float]) -> tuple[float, ...]:
    total = sum(values)
    if not math.isfinite(total) or total <= 0:
        return tuple(1.0 / len(values) for _ in values)
    return tuple(value / total for value in values)


def _softmax(scores: Sequence[float], temperature: float) -> tuple[float, ...]:
    if not scores:
        return ()
    safe_temperature = max(temperature, _MIN_SOFTMAX_TEMPERATURE)
    maximum = max(scores)
    exponentials = [math.exp((score - maximum) / safe_temperature) for score in scores]
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


HybridProminenceProvider = SimpleHybridProminenceProvider
DormantHybridProminenceProvider = SimpleHybridProminenceProvider


__all__ = [
    "AttentionStageSelector",
    "DormantHybridProminenceProvider",
    "FoveacastProminenceProvider",
    "HybridProminenceConfig",
    "HybridProminenceProvider",
    "STAGE_MIXTURES",
    "SimpleHybridProminenceProvider",
]
