from __future__ import annotations

import struct

import pytest

from ux_analyzer.domain.saliency import (
    AttentionDuration,
    AttentionEstimate,
    AttentionEstimateKind,
    ElementAttentionProfile,
    ElementAttentionProfileSet,
    ElementSaliencyAggregate,
    SaliencyPlane,
    SaliencyPrediction,
    SaliencyPredictionMetadata,
    SaliencyPredictionSet,
    SaliencyRequest,
    SaliencyRequestMetadata,
    SearchStage,
)


def _metadata(
    *, output_dimensions: tuple[int, int] = (2, 2)
) -> SaliencyPredictionMetadata:
    return SaliencyPredictionMetadata(
        provider_id="foveacast",
        model_id="foveacast-v3-1s",
        provider_version="1.0.0",
        model_version="0.2.0",
        model_checksum="sha256:model",
        input_dimensions=(240, 320),
        output_dimensions=output_dimensions,
        preprocessing_version="foveacast-preprocess-v1",
        inference_duration_ms=12.5,
        execution_provider="cpu",
        warnings=(),
        cache_state="miss",
    )


def _plane() -> SaliencyPlane:
    return SaliencyPlane(
        width=2,
        height=2,
        values=struct.pack("<4f", 0.0, 0.25, 0.75, 1.0),
    )


def _prediction(
    *,
    viewport_id: str = "viewport-1",
    duration: AttentionDuration = AttentionDuration.ONE_SECOND,
) -> SaliencyPrediction:
    return SaliencyPrediction(
        viewport_id=viewport_id,
        duration=duration,
        plane=_plane(),
        metadata=_metadata(),
    )


def _aggregate(
    *,
    viewport_id: str = "viewport-1",
    element_id: str = "button-1",
    duration: AttentionDuration = AttentionDuration.ONE_SECOND,
) -> ElementSaliencyAggregate:
    return ElementSaliencyAggregate(
        viewport_id=viewport_id,
        element_id=element_id,
        duration=duration,
        density=0.4,
        robust_peak=0.9,
        raw_mass=1.2,
        mass_share=0.3,
        clipped_area=120.0,
        visibility_fraction=0.8,
        occlusion_fraction=0.1,
        raw_score=0.6,
        adjusted_score=0.432,
    )


def test_saliency_plane_requires_float32_shape_size() -> None:
    with pytest.raises(ValueError, match="plane byte length"):
        SaliencyPlane(width=2, height=2, values=b"short")


def test_saliency_plane_rejects_nonfinite_and_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        SaliencyPlane(width=1, height=1, values=struct.pack("<f", float("nan")))
    with pytest.raises(ValueError, match="between 0 and 1"):
        SaliencyPlane(width=1, height=1, values=struct.pack("<f", 1.1))


def test_request_metadata_normalizes_collections_and_checks_ranges() -> None:
    metadata = SaliencyRequestMetadata(
        viewport_id="viewport-1",
        screenshot_sha256="a" * 64,
        screenshot_width=1280,
        screenshot_height=720,
        device_pixel_ratio=2.0,
        zoom=1.25,
        requested_durations=("1s", "3s"),
        model_set=("model-1",),
        precision="fp16",
        execution_provider_preference="auto",
    )
    request = SaliencyRequest(screenshot=b"png", metadata=metadata)

    assert metadata.requested_durations == (
        AttentionDuration.ONE_SECOND,
        AttentionDuration.THREE_SECONDS,
    )
    assert request.screenshot == b"png"

    with pytest.raises(ValueError, match="device_pixel_ratio"):
        SaliencyRequestMetadata(
            viewport_id="viewport-1",
            screenshot_sha256="hash",
            screenshot_width=1,
            screenshot_height=1,
            device_pixel_ratio=0,
            zoom=1,
            requested_durations=(AttentionDuration.ONE_SECOND,),
            model_set=("model-1",),
            precision="fp32",
            execution_provider_preference="cpu",
        )


def test_prediction_set_rejects_duplicate_durations_and_viewport_mismatch() -> None:
    with pytest.raises(ValueError, match="duplicate duration"):
        SaliencyPredictionSet(
            viewport_id="viewport-1",
            predictions=(_prediction(), _prediction()),
        )

    with pytest.raises(ValueError, match="different viewport"):
        SaliencyPredictionSet(
            viewport_id="viewport-1",
            predictions=(_prediction(viewport_id="viewport-2"),),
        )


def test_prediction_metadata_requires_plane_dimensions_and_finite_latency() -> None:
    with pytest.raises(ValueError, match="output dimensions"):
        SaliencyPrediction(
            viewport_id="viewport-1",
            duration=AttentionDuration.ONE_SECOND,
            plane=_plane(),
            metadata=_metadata(output_dimensions=(1, 1)),
        )

    with pytest.raises(ValueError, match="inference duration"):
        SaliencyPredictionMetadata(
            provider_id="foveacast",
            model_id="model",
            provider_version="1",
            model_version="1",
            model_checksum="checksum",
            input_dimensions=(1, 1),
            output_dimensions=(1, 1),
            preprocessing_version="v1",
            inference_duration_ms=float("inf"),
            execution_provider="cpu",
        )


def test_element_attention_profile_set_rejects_duplicate_profiles() -> None:
    profile = ElementAttentionProfile(
        viewport_id="viewport-1",
        element_id="button-1",
        immediate=AttentionEstimate(
            kind=AttentionEstimateKind.PREDICTED,
            score=0.8,
            source="foveacast",
        ),
        early=None,
        eventual=None,
        general=None,
        aggregates=(_aggregate(),),
    )

    with pytest.raises(ValueError, match="duplicate element profile"):
        ElementAttentionProfileSet(
            viewport_id="viewport-1",
            profiles=(profile, profile),
        )


@pytest.mark.parametrize("kind", AttentionEstimateKind)
def test_attention_estimate_requires_scores_and_sources_except_unavailable(
    kind: AttentionEstimateKind,
) -> None:
    if kind is AttentionEstimateKind.UNAVAILABLE:
        estimate = AttentionEstimate(kind=kind)
        assert estimate.score is None
        return

    with pytest.raises(ValueError, match="score"):
        AttentionEstimate(kind=kind, source="provider")
    with pytest.raises(ValueError, match="source"):
        AttentionEstimate(kind=kind, score=0.5)


def test_unavailable_attention_estimate_cannot_carry_score() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        AttentionEstimate(
            kind=AttentionEstimateKind.UNAVAILABLE,
            score=0.0,
        )


def test_profile_rejects_aggregate_from_other_viewport_or_duplicate_duration() -> None:
    with pytest.raises(ValueError, match="different viewport"):
        ElementAttentionProfile(
            viewport_id="viewport-1",
            element_id="button-1",
            immediate=None,
            early=None,
            eventual=None,
            general=None,
            aggregates=(_aggregate(viewport_id="viewport-2"),),
        )

    with pytest.raises(ValueError, match="duplicate duration"):
        ElementAttentionProfile(
            viewport_id="viewport-1",
            element_id="button-1",
            immediate=None,
            early=None,
            eventual=None,
            general=None,
            aggregates=(
                _aggregate(),
                _aggregate(duration=AttentionDuration.ONE_SECOND),
            ),
        )


def test_search_stage_is_closed_enum() -> None:
    assert SearchStage.INITIAL.value == "initial"
    assert SearchStage.EXPLORATION.value == "exploration"
    assert SearchStage.PERSISTENT.value == "persistent"
