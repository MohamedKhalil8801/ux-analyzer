from __future__ import annotations

import asyncio
import json
import math
import random
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from threading import Barrier, Lock

import pytest

from ux_analyzer.application.saliency import ProminenceBatch
from ux_analyzer.domain.attention import AttentionState
from ux_analyzer.domain.benchmark import Budget
from ux_analyzer.domain.interface import BoundingBox, ElementSnapshot, ViewportSnapshot
from ux_analyzer.domain.saliency import (
    AttentionDuration,
    AttentionEstimate,
    AttentionEstimateKind,
    ElementAttentionProfile,
    SaliencyGeometry,
    SaliencyPlane,
    SaliencyPrediction,
    SaliencyPredictionMetadata,
    SaliencyPredictionProvenance,
    SaliencyPredictionRequest,
    SaliencyPredictionSet,
    SearchStage,
)
from ux_analyzer.ports.models import ModelRole
from ux_analyzer.ports.observation import ObservationCapture, ViewportSize
from ux_analyzer.providers.attention_policy import (
    AttentionPolicyConfig,
    ProgressiveAttentionPolicy,
)
from ux_analyzer.providers.cognitive import StructuredCognitiveAgent
from ux_analyzer.providers.prominence import (
    HeuristicProminenceProvider,
    ProminenceResult,
)
from ux_analyzer.providers.saliency_aggregation import SaliencyAggregationConfig
from ux_analyzer.providers.saliency_prominence import (
    AttentionStageSelector,
    FoveacastProminenceProvider,
    SimpleHybridProminenceProvider,
    _normalize_stage_scores,
)
from ux_analyzer.storage.saliency_cache import SaliencyCache


def _profile(
    element_id: str,
    *,
    immediate: float | None,
    early: float | None,
    eventual: float | None,
    viewport_id: str = "viewport-1",
) -> ElementAttentionProfile:
    geometry = SaliencyGeometry(
        geometry_version="saliency-geometry-v1",
        source_dimensions=(4, 4),
        native_dimensions=(4, 4),
        content_dimensions=(4, 4),
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        scale=1.0,
        scale_x=1.0,
        scale_y=1.0,
        device_pixel_ratio=1.0,
        zoom=1.0,
    )
    provenance = tuple(
        SaliencyPredictionProvenance(
            duration=duration,
            metadata=SaliencyPredictionMetadata(
                provider_id="foveacast",
                model_id="foveacast-v3",
                provider_version="foveacast-v1",
                model_version="0.2.0",
                model_checksum="a" * 64,
                input_dimensions=(4, 4),
                output_dimensions=(4, 4),
                geometry=geometry,
                preprocessing_version="test-preprocess-v1",
                inference_duration_ms=0.0,
                execution_provider="CPUExecutionProvider",
            ),
        )
        for duration in (
            AttentionDuration.ONE_SECOND,
            AttentionDuration.THREE_SECONDS,
            AttentionDuration.SEVEN_SECONDS,
        )
    )
    return ElementAttentionProfile(
        viewport_id=viewport_id,
        element_id=element_id,
        immediate=_estimate(immediate),
        early=_estimate(early),
        eventual=_estimate(eventual),
        general=None,
        aggregates=(),
        aggregation_version="test-v1",
        prediction_provenance=provenance,
    )


def _estimate(score: float | None) -> AttentionEstimate | None:
    if score is None:
        return None
    return AttentionEstimate(
        kind=AttentionEstimateKind.PREDICTED,
        score=score,
        source="foveacast",
    )


def test_initial_stage_uses_immediate_attention() -> None:
    profiles = (
        _profile("first", immediate=0.9, early=0.1, eventual=0.2),
        _profile("second", immediate=0.1, early=0.9, eventual=0.8),
    )

    selected = AttentionStageSelector().select(profiles, SearchStage.INITIAL)

    assert [item.element_id for item in selected] == ["first", "second"]
    assert selected[0].raw_score == 0.9
    assert selected[0].normalized_probability > selected[1].normalized_probability


def test_stage_mixtures_use_expected_duration_components_and_normalize() -> None:
    profiles = (
        _profile("first", immediate=0.9, early=0.2, eventual=0.1),
        _profile("second", immediate=0.1, early=0.8, eventual=0.7),
    )

    initial = AttentionStageSelector().select(profiles, SearchStage.INITIAL)
    exploration = AttentionStageSelector().select(profiles, SearchStage.EXPLORATION)
    persistent = AttentionStageSelector().select(profiles, SearchStage.PERSISTENT)

    assert initial[0].raw_score == 0.9
    assert exploration[1].raw_score == 0.8
    assert persistent[0].raw_score == 0.25 * 0.2 + 0.75 * 0.1
    assert sum(item.normalized_probability for item in persistent) == 1.0
    assert persistent[1].normalized_probability > persistent[0].normalized_probability


def test_stage_contributions_apply_mixture_weights_for_two_elements() -> None:
    profiles = (
        _profile("first", immediate=0.9, early=0.4, eventual=0.2),
        _profile("second", immediate=0.1, early=0.8, eventual=0.6),
    )

    selected = AttentionStageSelector().select(profiles, SearchStage.PERSISTENT)
    first, second = selected

    assert first.raw_values == {"stage_3s": 0.4, "stage_7s": 0.2}
    assert second.raw_values == {"stage_3s": 0.8, "stage_7s": 0.6}
    assert first.feature_contributions == {
        "stage_3s": pytest.approx(0.1),
        "stage_7s": pytest.approx(0.15),
    }
    assert second.feature_contributions == {
        "stage_3s": pytest.approx(0.2),
        "stage_7s": pytest.approx(0.45),
    }
    assert first.raw_score == pytest.approx(sum(first.feature_contributions.values()))
    assert second.raw_score == pytest.approx(sum(second.feature_contributions.values()))


def test_missing_duration_uses_explicit_fallback_provenance() -> None:
    selected = AttentionStageSelector().select(
        (_profile("first", immediate=0.7, early=None, eventual=None),),
        SearchStage.EXPLORATION,
    )

    assert selected[0].evidence_kind == "fallback"
    assert selected[0].evidence_source == "missing-3s:foveacast"


def test_duration_fallback_invalidates_learned_batch_and_uses_heuristic_scores() -> (
    None
):
    model = _ModelProvider()
    provider = FoveacastProminenceProvider(
        model_provider=model,
        cache=_Cache(),
        aggregator=lambda snapshot, predictions, config: (
            _profile("first", immediate=0.8, early=None, eventual=None),
        ),
        heuristic_provider=HeuristicProminenceProvider(),
    )

    batch = provider.score(_capture(), _snapshot(), SearchStage.EXPLORATION, None)

    assert batch.learned_available is False
    assert batch.cache_state == "fallback"
    assert batch.fallback_reason == "learned duration fallback: missing-3s:foveacast"
    assert batch.active_provider_id == HeuristicProminenceProvider.id


def _snapshot(viewport_id: str = "viewport-1") -> ViewportSnapshot:
    return ViewportSnapshot(
        id=viewport_id,
        elements=(
            ElementSnapshot(
                id="first",
                role="button",
                label="First",
                bounds=BoundingBox(x=0, y=0, width=20, height=20),
                visibility_fraction=1.0,
                actionable=True,
            ),
            ElementSnapshot(
                id="second",
                role="button",
                label="Second",
                bounds=BoundingBox(x=30, y=0, width=20, height=20),
                visibility_fraction=1.0,
                actionable=True,
            ),
        ),
    )


def _capture(viewport_id: str = "viewport-1") -> ObservationCapture:
    screenshot = b"screenshot"
    return ObservationCapture(
        session_id="session-1",
        viewport_id=viewport_id,
        url="https://fixture.test",
        title="Fixture",
        viewport=ViewportSize(width=4, height=4),
        screenshot=screenshot,
    )


def _prediction_set(request: SaliencyPredictionRequest) -> SaliencyPredictionSet:
    geometry = SaliencyGeometry(
        geometry_version="saliency-geometry-v1",
        source_dimensions=(4, 4),
        native_dimensions=(4, 4),
        content_dimensions=(4, 4),
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        scale=1.0,
        scale_x=1.0,
        scale_y=1.0,
        device_pixel_ratio=1.0,
        zoom=1.0,
    )
    durations = (
        (AttentionDuration.ONE_SECOND, "a"),
        (AttentionDuration.THREE_SECONDS, "b"),
        (AttentionDuration.SEVEN_SECONDS, "c"),
    )
    predictions = tuple(
        SaliencyPrediction(
            viewport_id=request.metadata.viewport_id,
            duration=duration,
            plane=SaliencyPlane(
                width=4,
                height=4,
                values=struct.pack("<16f", *([0.5] * 16)),
            ),
            metadata=SaliencyPredictionMetadata(
                provider_id="foveacast",
                model_id="foveacast-v0.2.0",
                provider_version="foveacast-adapter-v1",
                model_version="v0.2.0",
                model_checksum=letter * 64,
                input_dimensions=(4, 4),
                output_dimensions=(4, 4),
                geometry=geometry,
                preprocessing_version="foveacast-preprocess-v1",
                inference_duration_ms=1.0,
                execution_provider="CPUExecutionProvider",
            ),
        )
        for duration, letter in durations
    )
    return SaliencyPredictionSet(
        viewport_id="viewport-1",
        predictions=predictions,
        request_metadata=request.metadata,
    )


class _ModelProvider:
    model_id = "foveacast-v0.2.0"
    model_version = "v0.2.0"
    precision = "fp16"
    preprocessing_version = "foveacast-preprocess-v1"
    actual_execution_provider = "CPUExecutionProvider"
    requested_execution_provider = "cpu"
    model_checksums = {
        AttentionDuration.ONE_SECOND: "a" * 64,
        AttentionDuration.THREE_SECONDS: "b" * 64,
        AttentionDuration.SEVEN_SECONDS: "c" * 64,
    }

    def __init__(self, *, error: Exception | None = None) -> None:
        self.calls = 0
        self.error = error
        self.requests: list[SaliencyPredictionRequest] = []

    def predict(self, request: SaliencyPredictionRequest) -> SaliencyPredictionSet:
        self.calls += 1
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return _prediction_set(request)


@dataclass
class _Entry:
    key: object
    predictions: SaliencyPredictionSet
    profiles: tuple[ElementAttentionProfile, ...]
    cache_state: str = "hit"


class _Cache:
    def __init__(self, *, load_error: Exception | None = None) -> None:
        self.entries: dict[str, _Entry] = {}
        self.load_error = load_error

    def load(self, key: object) -> _Entry | None:
        if self.load_error is not None:
            raise self.load_error
        return self.entries.get(key.digest)

    def store(self, key: object, payload: object) -> _Entry:
        entry = _Entry(
            key=key,
            predictions=payload.predictions,
            profiles=payload.profiles,
            cache_state="miss",
        )
        self.entries[key.digest] = entry
        return entry

    def materialize_into_bundle(self, entry: _Entry, writer: object, **kwargs: object):
        del entry, writer, kwargs
        return ()


def test_cache_disabled_without_writer_rejects_inference_before_model_call() -> None:
    model = _ModelProvider()
    cache = _Cache()
    aggregation_config = SaliencyAggregationConfig(
        version="aggregation-project-v2",
        density_weight=0.5,
        robust_peak_weight=0.3,
        mass_share_weight=0.2,
    )
    stage_selector = AttentionStageSelector(
        version="stage-project-v2",
        temperature=0.9,
        mixtures={
            "initial": {"7s": 1.0},
            "exploration": {"3s": 1.0},
            "persistent": {"3s": 0.25, "7s": 0.75},
        },
    )
    provider = FoveacastProminenceProvider(
        model_provider=model,
        cache=cache,
        cache_enabled=False,
        model_set=("foveacast-v0.2.0", "foveacast-v0.2.0-shadow"),
        precision="fp16",
        execution_provider_preference="directml",
        aggregation_config=aggregation_config,
        aggregator=lambda snapshot, predictions, config: (
            _profile("first", immediate=0.8, early=0.4, eventual=0.2),
            _profile("second", immediate=0.2, early=0.6, eventual=0.9),
        ),
        stage_selector=stage_selector,
        heuristic_provider=HeuristicProminenceProvider(),
        fallback_enabled=True,
        fallback_provider_id="heuristic",
    )

    batch = provider.score(_capture(), _snapshot(), SearchStage.INITIAL, None)

    assert model.requests == []
    assert provider.cache_enabled is False
    assert cache.entries == {}
    assert provider.aggregation_config is aggregation_config
    assert provider.stage_selector.version == "stage-project-v2"
    assert batch.learned_available is False
    assert (
        batch.fallback_reason
        == "cache-disabled saliency requires typed artifact writer"
    )


class _ConcurrentCache(_Cache):
    def __init__(self) -> None:
        super().__init__()
        self._store_barrier = Barrier(2)
        self._store_lock = Lock()

    def store(self, key: object, payload: object) -> _Entry:
        self._store_barrier.wait(timeout=5)
        with self._store_lock:
            existing = self.entries.get(key.digest)
            if existing is not None:
                return replace(existing, cache_state="hit")
            return super().store(key, payload)


def _provider(model: object, cache: _Cache) -> FoveacastProminenceProvider:
    return FoveacastProminenceProvider(
        model_provider=model,
        cache=cache,
        aggregator=lambda snapshot, predictions, config: (
            _profile("first", immediate=0.8, early=0.4, eventual=0.2),
            _profile("second", immediate=0.2, early=0.6, eventual=0.9),
        ),
        stage_selector=AttentionStageSelector(),
        heuristic_provider=HeuristicProminenceProvider(),
    )


def test_identical_screenshot_reuses_native_maps_across_viewports_and_reaggregates() -> (
    None
):
    model = _ModelProvider()
    aggregations: list[tuple[str, tuple[str, ...]]] = []

    def aggregate(snapshot, predictions, config):
        del config
        prediction_items = getattr(predictions, "predictions", predictions)
        aggregations.append(
            (
                snapshot.id,
                tuple(prediction.viewport_id for prediction in prediction_items),
            )
        )
        return (
            _profile(
                "first",
                viewport_id=snapshot.id,
                immediate=0.8,
                early=0.4,
                eventual=0.2,
            ),
            _profile(
                "second",
                viewport_id=snapshot.id,
                immediate=0.2,
                early=0.6,
                eventual=0.9,
            ),
        )

    provider = FoveacastProminenceProvider(
        model_provider=model,
        cache=_Cache(),
        aggregator=aggregate,
        stage_selector=AttentionStageSelector(),
        heuristic_provider=HeuristicProminenceProvider(),
    )

    first = provider.score(_capture(), _snapshot(), SearchStage.INITIAL, None)
    second = provider.score(
        _capture("viewport-2"),
        _snapshot("viewport-2"),
        SearchStage.INITIAL,
        None,
    )

    assert model.calls == 1
    assert aggregations == [
        ("viewport-1", ("viewport-1", "viewport-1", "viewport-1")),
        ("viewport-2", ("viewport-2", "viewport-2", "viewport-2")),
    ]
    assert first.cache_state == "miss"
    assert second.cache_state == "hit"
    assert {profile.viewport_id for profile in second.learned_profiles} == {
        "viewport-2"
    }


def test_stable_cross_viewport_cache_key_round_trips_through_real_cache(
    tmp_path,
) -> None:
    model = _ModelProvider()
    provider = FoveacastProminenceProvider(
        model_provider=model,
        cache=SaliencyCache(tmp_path),
        heuristic_provider=HeuristicProminenceProvider(),
    )

    provider.score(_capture(), _snapshot(), SearchStage.INITIAL, None)
    batch = provider.score(
        _capture("viewport-2"),
        _snapshot("viewport-2"),
        SearchStage.EXPLORATION,
        None,
    )

    assert model.calls == 1
    assert batch.cache_state == "hit"
    assert {profile.viewport_id for profile in batch.learned_profiles} == {"viewport-2"}


def test_unavailable_learned_profiles_stay_evidence_but_not_candidates() -> None:
    model = _ModelProvider()
    provider = FoveacastProminenceProvider(
        model_provider=model,
        cache=_Cache(),
        aggregator=lambda snapshot, predictions, config: (
            _profile("available", immediate=0.8, early=0.4, eventual=0.2),
            _profile("unavailable", immediate=None, early=None, eventual=None),
        ),
        stage_selector=AttentionStageSelector(),
        heuristic_provider=HeuristicProminenceProvider(),
    )

    batch = provider.score(_capture(), _snapshot(), SearchStage.INITIAL, None)

    assert [item.element_id for item in batch.scores] == ["available"]
    assert [item.element_id for item in batch.learned_scores] == ["available"]
    assert {profile.element_id for profile in batch.learned_profiles} == {
        "available",
        "unavailable",
    }
    assert [profile.element_id for profile in batch.unavailable_learned_profiles] == [
        "unavailable"
    ]


def test_model_setup_failure_returns_heuristic_fallback_batch() -> None:
    def setup_model() -> object:
        raise RuntimeError("model setup unavailable")

    provider = FoveacastProminenceProvider(
        model_provider=setup_model,
        cache=_Cache(),
        heuristic_provider=HeuristicProminenceProvider(),
    )

    batch = provider.score(_capture(), _snapshot(), SearchStage.INITIAL, None)

    assert batch.fallback_reason == "model setup unavailable"
    assert batch.learned_available is False
    assert batch.active_provider_id == HeuristicProminenceProvider.id
    assert batch.scores


def test_tiny_valid_temperature_keeps_stage_probabilities_finite() -> None:
    selected = AttentionStageSelector(temperature=1e-320).select(
        (
            _profile("first", immediate=0.9, early=0.2, eventual=0.1),
            _profile("second", immediate=0.1, early=0.8, eventual=0.7),
        ),
        SearchStage.INITIAL,
    )

    assert all(math.isfinite(item.normalized_probability) for item in selected)
    assert sum(item.normalized_probability for item in selected) == pytest.approx(1.0)


def test_temperature_one_max_scales_before_normalizing_huge_scores() -> None:
    probabilities = _normalize_stage_scores((1e308, 9e307), temperature=1.0)

    assert probabilities == pytest.approx((10 / 19, 9 / 19))
    assert probabilities[0] / probabilities[1] == pytest.approx(10 / 9)
    assert sum(probabilities) == pytest.approx(1.0)


def test_log_temperature_scaling_handles_subnormal_and_huge_scores() -> None:
    probabilities = _normalize_stage_scores((5e-324, 1e308), temperature=2.0)

    assert all(math.isfinite(probability) for probability in probabilities)
    assert all(probability >= 0 for probability in probabilities)
    assert probabilities[1] == pytest.approx(1.0)
    assert sum(probabilities) == pytest.approx(1.0)


def test_stage_probabilities_normalize_weighted_scores_without_softmax() -> None:
    selected = AttentionStageSelector().select(
        (
            _profile("first", immediate=0.6, early=0.2, eventual=0.1),
            _profile("second", immediate=0.3, early=0.8, eventual=0.7),
        ),
        SearchStage.INITIAL,
    )

    assert [item.raw_score for item in selected] == [0.6, 0.3]
    assert [item.normalized_probability for item in selected] == pytest.approx(
        (2 / 3, 1 / 3)
    )
    assert selected[0].normalized_probability / selected[1].normalized_probability == (
        pytest.approx(2.0)
    )
    assert sum(item.normalized_probability for item in selected) == pytest.approx(1.0)


def test_stage_probabilities_use_power_temperature_and_preserve_zero() -> None:
    selected = AttentionStageSelector(temperature=2.0).select(
        (
            _profile("first", immediate=0.64, early=0.2, eventual=0.1),
            _profile("second", immediate=0.16, early=0.8, eventual=0.7),
            _profile("zero", immediate=0.0, early=0.5, eventual=0.5),
        ),
        SearchStage.INITIAL,
    )

    assert [item.raw_score for item in selected] == [0.64, 0.16, 0.0]
    assert [item.normalized_probability for item in selected] == pytest.approx(
        (2 / 3, 1 / 3, 0.0)
    )
    assert sum(item.normalized_probability for item in selected) == pytest.approx(1.0)


@pytest.mark.parametrize("invalid_score", (-1.0, math.nan, math.inf))
def test_stage_selector_rejects_nonfinite_or_negative_stage_scores(
    invalid_score: float,
) -> None:
    profile = _profile("invalid", immediate=0.5, early=0.2, eventual=0.1)
    estimate = profile.immediate
    assert estimate is not None
    object.__setattr__(estimate, "score", invalid_score)

    with pytest.raises(
        ValueError, match="stage scores must be finite and non-negative"
    ):
        AttentionStageSelector().select((profile,), SearchStage.INITIAL)


def test_hybrid_rejects_duplicate_learned_ids() -> None:
    learned = ProminenceResult("first", raw_score=0.8, normalized_probability=1.0)
    heuristic = ProminenceResult("first", raw_score=0.2, normalized_probability=1.0)

    with pytest.raises(ValueError, match="duplicate"):
        SimpleHybridProminenceProvider().blend((learned, learned), (heuristic,))


def test_hybrid_preserves_component_provenance() -> None:
    learned = ProminenceResult(
        "first",
        raw_score=0.8,
        normalized_probability=1.0,
        provider_id="foveacast",
        evidence_kind="predicted",
        evidence_source="foveacast-v0.2.0",
    )
    heuristic = ProminenceResult(
        "first",
        raw_score=0.2,
        normalized_probability=1.0,
        provider_id="heuristic-prominence",
        evidence_kind="derived",
        evidence_source="heuristic-prominence-v1",
    )

    blended = SimpleHybridProminenceProvider().blend((learned,), (heuristic,))

    assert blended[0].component_provenance == {
        "learned": "foveacast-v0.2.0",
        "heuristic": "heuristic-prominence-v1",
    }


def test_hybrid_contributions_apply_blend_weights_for_two_elements() -> None:
    learned = (
        ProminenceResult("first", raw_score=0.2, normalized_probability=0.2),
        ProminenceResult("second", raw_score=0.8, normalized_probability=0.8),
    )
    heuristic = (
        ProminenceResult("first", raw_score=0.6, normalized_probability=0.6),
        ProminenceResult("second", raw_score=0.4, normalized_probability=0.4),
    )

    blended = SimpleHybridProminenceProvider().blend(learned, heuristic)
    by_id = {item.element_id: item for item in blended}

    assert by_id["first"].raw_values == {
        "learned_probability": 0.2,
        "heuristic_probability": 0.6,
    }
    assert by_id["second"].raw_values == {
        "learned_probability": 0.8,
        "heuristic_probability": 0.4,
    }
    assert by_id["first"].feature_contributions == {
        "learned_probability": pytest.approx(0.14),
        "heuristic_probability": pytest.approx(0.18),
    }
    assert by_id["second"].feature_contributions == {
        "learned_probability": pytest.approx(0.56),
        "heuristic_probability": pytest.approx(0.12),
    }
    assert all(
        item.raw_score == pytest.approx(sum(item.feature_contributions.values()))
        for item in blended
    )


def _failing_aggregator(snapshot, predictions, config):
    del snapshot, predictions, config
    raise RuntimeError("aggregation unavailable")


def test_stage_changes_do_not_reinfer_cached_predictions() -> None:
    model = _ModelProvider()
    provider = _provider(model, _Cache())

    initial = provider.score(_capture(), _snapshot(), SearchStage.INITIAL, None)
    exploration = provider.score(_capture(), _snapshot(), SearchStage.EXPLORATION, None)

    assert model.calls == 1
    assert initial.stage is SearchStage.INITIAL
    assert exploration.stage is SearchStage.EXPLORATION


class _BarrierModelProvider(_ModelProvider):
    def __init__(self, barrier: Barrier) -> None:
        super().__init__()
        self.barrier = barrier

    def predict(self, request: SaliencyPredictionRequest) -> SaliencyPredictionSet:
        self.barrier.wait(timeout=5)
        return super().predict(request)


def test_concurrent_cache_publication_state_reaches_prominence_batch() -> None:
    barrier = Barrier(2)
    cache = _ConcurrentCache()
    providers = (
        _provider(_BarrierModelProvider(barrier), cache),
        _provider(_BarrierModelProvider(barrier), cache),
    )

    def score(provider: FoveacastProminenceProvider) -> ProminenceBatch:
        return provider.score(_capture(), _snapshot(), SearchStage.INITIAL, None)

    with ThreadPoolExecutor(max_workers=2) as workers:
        batches = tuple(workers.map(score, providers))

    assert sorted(batch.cache_state for batch in batches) == ["hit", "miss"]


def test_model_failure_returns_heuristic_fallback_with_unavailable_learned_profile() -> (
    None
):
    reason = RuntimeError("model runtime unavailable")
    provider = _provider(_ModelProvider(error=reason), _Cache())

    batch = provider.score(_capture(), _snapshot(), SearchStage.INITIAL, None)

    assert isinstance(batch, ProminenceBatch)
    assert batch.fallback_reason == "model runtime unavailable"
    assert batch.learned_available is False
    assert batch.learned_profiles == ()
    assert batch.active_provider_id == HeuristicProminenceProvider.id
    assert all(
        item.provider_id == HeuristicProminenceProvider.id for item in batch.scores
    )


def test_cache_failure_returns_heuristic_fallback_with_reason() -> None:
    provider = _provider(
        _ModelProvider(),
        _Cache(load_error=RuntimeError("cache unavailable")),
    )

    batch = provider.score(_capture(), _snapshot(), SearchStage.INITIAL, None)

    assert batch.fallback_reason == "cache unavailable"
    assert batch.learned_available is False
    assert batch.active_provider_id == HeuristicProminenceProvider.id


def test_aggregation_failure_returns_heuristic_fallback_with_reason() -> None:
    provider = _provider(_ModelProvider(), _Cache())
    provider.aggregator = _failing_aggregator

    batch = provider.score(_capture(), _snapshot(), SearchStage.INITIAL, None)

    assert batch.fallback_reason == "aggregation unavailable"
    assert batch.learned_available is False
    assert batch.active_provider_id == HeuristicProminenceProvider.id


def test_hybrid_is_versioned_dormant_and_preserves_components() -> None:
    learned = _profile("first", immediate=0.8, early=0.8, eventual=0.8)
    heuristic = HeuristicProminenceProvider().score(_snapshot())
    learned_scores = AttentionStageSelector().select((learned,), SearchStage.INITIAL)

    hybrid = SimpleHybridProminenceProvider()
    blended = hybrid.blend(learned_scores, heuristic[:1])

    assert hybrid.enabled is False
    assert hybrid.version == "hybrid-prominence-v1"
    assert blended[0].feature_contributions["learned_probability"] == pytest.approx(0.7)
    assert blended[0].feature_contributions["heuristic_probability"] == pytest.approx(
        0.3 * heuristic[0].normalized_probability
    )


class _CognitiveClient:
    endpoint_origin = "internal"

    def __init__(self) -> None:
        self.calls = []

    async def complete(self, schema, messages, model, role):
        self.calls.append(messages)
        assert model == "cognitive-model"
        assert role is ModelRole.COGNITIVE
        return schema.model_validate({"action": "abandon", "reason": "done"})


def test_cognitive_payload_contains_no_numeric_prominence_scores() -> None:
    client = _CognitiveClient()
    agent = StructuredCognitiveAgent(
        client,
        model="cognitive-model",
        fixture_keys=(),
    )
    score_payload = (
        ProminenceResult(
            "first",
            raw_score=0.123456,
            normalized_probability=0.1,
            feature_contributions={"stage_1s": 0.123456},
            raw_values={"stage_1s": 0.123456},
            normalized_values={"stage_1s": 0.123456},
        ),
        ProminenceResult(
            "second",
            raw_score=0.987654,
            normalized_probability=0.9,
            feature_contributions={"stage_1s": 0.987654},
            raw_values={"stage_1s": 0.987654},
            normalized_values={"stage_1s": 0.987654},
        ),
    )
    selection = ProgressiveAttentionPolicy(
        AttentionPolicyConfig(batch_size=1, cross_region_exploration=0)
    ).next_observation(
        AttentionState.initial(
            Budget(
                max_steps=5,
                max_observations=3,
                max_interactions=2,
                timeout_seconds=10,
            ),
            confidence=0.5,
            frustration=0.0,
        ),
        _snapshot(),
        score_payload,
        (),
        random.Random(1),
    )

    assert selection.selected_ids == ("second",)
    asyncio.run(agent.decide("Find first", selection.observation))

    payload = json.loads(client.calls[0][1].content)
    serialized = json.dumps(payload, sort_keys=True)
    assert payload["newly_revealed_elements"]
    assert "raw_score" not in serialized
    assert "normalized_probability" not in serialized
    assert "feature_contributions" not in serialized
    assert "raw_values" not in serialized
    assert "normalized_values" not in serialized
    for score in score_payload:
        for numeric_value in (
            score.raw_score,
            score.normalized_probability,
            *score.feature_contributions.values(),
        ):
            assert str(numeric_value) not in serialized
