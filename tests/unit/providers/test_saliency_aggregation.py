from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest

from ux_analyzer.domain.interface import BoundingBox, ElementSnapshot, ViewportSnapshot
from ux_analyzer.domain.saliency import (
    AttentionDuration,
    ElementAttentionProfile,
    SaliencyGeometry,
    SaliencyPlane,
    SaliencyPrediction,
    SaliencyPredictionMetadata,
    SaliencyPredictionSet,
    SaliencyRequestMetadata,
)
from ux_analyzer.providers.saliency_aggregation import (
    SaliencyAggregationConfig,
    aggregate_saliency,
)


def _element(
    element_id: str,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    role: str = "button",
    actionable: bool = True,
    visibility_fraction: float = 1.0,
    occlusion_fraction: float | None = None,
) -> ElementSnapshot:
    return ElementSnapshot(
        id=element_id,
        role=role,
        label=element_id,
        bounds=BoundingBox(x=x, y=y, width=width, height=height),
        visibility_fraction=visibility_fraction,
        actionable=actionable,
        occlusion_fraction=occlusion_fraction,
    )


def _snapshot(*elements: ElementSnapshot) -> ViewportSnapshot:
    return ViewportSnapshot(id="viewport-1", elements=elements)


def _prediction(
    values: np.ndarray,
    *,
    duration: AttentionDuration = AttentionDuration.ONE_SECOND,
    screenshot_dimensions: tuple[int, int] | None = None,
    dpr: float = 1.0,
    zoom: float = 1.0,
    provider_id: str = "foveacast",
    geometry_version: str = "saliency-geometry-v1",
) -> SaliencyPrediction:
    height, width = values.shape
    screenshot_width, screenshot_height = screenshot_dimensions or (width, height)
    native_width, native_height = width, height
    scale = min(native_width / screenshot_width, native_height / screenshot_height)
    content_width = max(1, int(round(screenshot_width * scale)))
    content_height = max(1, int(round(screenshot_height * scale)))
    pad_left = (native_width - content_width) // 2
    pad_top = (native_height - content_height) // 2
    return SaliencyPrediction(
        viewport_id="viewport-1",
        duration=duration,
        plane=SaliencyPlane(
            width=width,
            height=height,
            values=np.asarray(values, dtype="<f4").tobytes(order="C"),
        ),
        metadata=SaliencyPredictionMetadata(
            provider_id=provider_id,
            model_id=f"model-{duration.value}",
            provider_version="provider-v1",
            model_version="model-v1",
            model_checksum=f"checksum-{duration.value}",
            input_dimensions=(width, height),
            output_dimensions=(width, height),
            geometry=SaliencyGeometry(
                geometry_version=geometry_version,
                source_dimensions=(screenshot_width, screenshot_height),
                native_dimensions=(native_width, native_height),
                content_dimensions=(content_width, content_height),
                pad_left=pad_left,
                pad_top=pad_top,
                pad_right=native_width - pad_left - content_width,
                pad_bottom=native_height - pad_top - content_height,
                scale=scale,
                scale_x=content_width / screenshot_width,
                scale_y=content_height / screenshot_height,
                device_pixel_ratio=dpr,
                zoom=zoom,
            ),
            preprocessing_version="foveacast-preprocess-v1",
            inference_duration_ms=1.0,
            execution_provider="CPUExecutionProvider",
        ),
    )


def _predictions(
    *values: np.ndarray,
    screenshot_dimensions: tuple[int, int] | None = None,
    dpr: float = 1.0,
    zoom: float = 1.0,
) -> SaliencyPredictionSet:
    durations = (
        AttentionDuration.ONE_SECOND,
        AttentionDuration.THREE_SECONDS,
        AttentionDuration.SEVEN_SECONDS,
    )
    predictions = tuple(
        _prediction(
            values[index],
            duration=durations[index],
            screenshot_dimensions=screenshot_dimensions,
            dpr=dpr,
            zoom=zoom,
        )
        for index in range(len(values))
    )
    first = predictions[0]
    height, width = first.plane.height, first.plane.width
    screenshot_width, screenshot_height = screenshot_dimensions or (width, height)
    return SaliencyPredictionSet(
        viewport_id="viewport-1",
        predictions=predictions,
        request_metadata=SaliencyRequestMetadata(
            viewport_id="viewport-1",
            screenshot_sha256="a" * 64,
            screenshot_width=screenshot_width,
            screenshot_height=screenshot_height,
            device_pixel_ratio=dpr,
            zoom=zoom,
            requested_durations=tuple(item.duration for item in predictions),
            model_set=tuple(item.metadata.model_id for item in predictions),
            precision="fp16",
            execution_provider_preference="cpu",
        ),
    )


def _profile(
    profiles: Sequence[ElementAttentionProfile], element_id: str
) -> ElementAttentionProfile:
    return next(profile for profile in profiles if profile.element_id == element_id)


def test_formula_preserves_native_aggregate_evidence_and_provenance() -> None:
    saliency = np.asarray(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6, 0.7],
            [0.8, 0.9, 1.0, 0.2],
            [0.3, 0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )
    profiles = aggregate_saliency(
        _snapshot(_element("button", x=1, y=1, width=2, height=2)),
        _predictions(saliency),
        SaliencyAggregationConfig(),
    )

    aggregate = _profile(profiles, "button").aggregates[0]
    region = saliency[1:3, 1:3]
    total_mass = float(saliency.sum())
    raw_score = 0.60 * float(region.mean())
    raw_score += 0.25 * float(np.percentile(region, 95))
    raw_score += 0.15 * math.sqrt(float(region.sum()) / total_mass)

    assert aggregate.duration is AttentionDuration.ONE_SECOND
    assert aggregate.density == pytest.approx(float(region.mean()))
    assert aggregate.p95 == pytest.approx(float(np.percentile(region, 95)))
    assert aggregate.raw_mass == pytest.approx(float(region.sum()))
    assert aggregate.mass_share == pytest.approx(float(region.sum()) / total_mass)
    assert aggregate.clipped_area == pytest.approx(4.0)
    assert aggregate.visibility_fraction == 1.0
    assert aggregate.occlusion_fraction == 0.0
    assert aggregate.raw_score == pytest.approx(raw_score)
    assert aggregate.adjusted_score == pytest.approx(raw_score)
    immediate = _profile(profiles, "button").immediate
    assert immediate is not None
    assert immediate.source == "foveacast"
    assert _profile(profiles, "button").aggregation_version == (
        "element-saliency-aggregation-v1"
    )
    assert (
        _profile(profiles, "button").prediction_provenance[0].metadata.model_checksum
        == "checksum-1s"
    )


def test_dpr_and_zoom_scale_css_bounds_before_native_sampling() -> None:
    saliency = np.zeros((8, 8), dtype=np.float32)
    saliency[2:4, 2:4] = 1.0
    profiles = aggregate_saliency(
        _snapshot(_element("button", x=2, y=2, width=2, height=2)),
        _predictions(
            saliency,
            screenshot_dimensions=(16, 16),
            dpr=2.0,
            zoom=1.25,
        ),
        SaliencyAggregationConfig(),
    )

    aggregate = _profile(profiles, "button").aggregates[0]

    assert aggregate.density == pytest.approx(4.0 / 9.0)
    assert aggregate.raw_mass == pytest.approx(4.0)
    assert aggregate.clipped_area == pytest.approx(25.0)


def test_aspect_fit_padding_is_removed_by_inverse_geometry() -> None:
    saliency = np.zeros((8, 8), dtype=np.float32)
    saliency[3:5, 2:4] = 1.0
    profiles = aggregate_saliency(
        _snapshot(_element("button", x=2, y=1, width=2, height=2)),
        _predictions(saliency, screenshot_dimensions=(8, 4)),
        SaliencyAggregationConfig(),
    )

    aggregate = _profile(profiles, "button").aggregates[0]

    assert aggregate.density == pytest.approx(1.0)
    assert aggregate.raw_mass == pytest.approx(4.0)


def test_non_square_hotspot_and_stripe_use_result_geometry() -> None:
    saliency = np.zeros((8, 8), dtype=np.float32)
    saliency[3:5, 2:4] = 1.0
    saliency[2:6, 5:6] = 0.8
    profiles = aggregate_saliency(
        _snapshot(
            _element("hotspot", x=3, y=1, width=3, height=3),
            _element("stripe", x=7.5, y=0, width=1.5, height=6),
            _element("padding", x=0, y=0, width=2, height=1),
        ),
        _predictions(saliency, screenshot_dimensions=(12, 6)),
    )

    hotspot = _profile(profiles, "hotspot").aggregates[0]
    stripe = _profile(profiles, "stripe").aggregates[0]
    padding = _profile(profiles, "padding").aggregates[0]

    assert hotspot.raw_mass > 0
    assert stripe.raw_mass > 0
    assert padding.raw_mass == 0
    assert stripe.density == pytest.approx(0.8)


def test_unsupported_prediction_geometry_version_is_rejected() -> None:
    prediction = _prediction(
        np.ones((4, 4), dtype=np.float32), geometry_version="unknown-geometry-v9"
    )

    with pytest.raises(ValueError, match="unsupported saliency geometry version"):
        aggregate_saliency(
            _snapshot(_element("button", x=0, y=0, width=1, height=1)),
            (prediction,),
        )


def test_viewport_clipping_partial_visibility_and_occlusion_adjust_score() -> None:
    saliency = np.ones((4, 4), dtype=np.float32)
    profiles = aggregate_saliency(
        _snapshot(
            _element(
                "partial",
                x=-1,
                y=-1,
                width=2,
                height=2,
                visibility_fraction=0.375,
                occlusion_fraction=0.25,
            )
        ),
        _predictions(saliency),
        SaliencyAggregationConfig(),
    )

    aggregate = _profile(profiles, "partial").aggregates[0]

    assert aggregate.clipped_area == pytest.approx(1.0)
    assert aggregate.raw_mass == pytest.approx(1.0)
    assert aggregate.adjusted_score == pytest.approx(aggregate.raw_score * 0.375)


def test_effective_visibility_applies_occlusion_adjustment_once() -> None:
    profiles = aggregate_saliency(
        _snapshot(
            _element(
                "covered",
                x=0,
                y=0,
                width=2,
                height=2,
                visibility_fraction=0.75,
                occlusion_fraction=0.25,
            )
        ),
        _predictions(np.ones((4, 4), dtype=np.float32)),
        SaliencyAggregationConfig(),
    )

    aggregate = _profile(profiles, "covered").aggregates[0]

    assert aggregate.visibility_fraction == pytest.approx(0.75)
    assert aggregate.occlusion_fraction == pytest.approx(0.25)
    assert aggregate.adjusted_score == pytest.approx(aggregate.raw_score * 0.75)


def test_offscreen_elements_keep_zero_evidence_but_are_not_candidates() -> None:
    profiles = aggregate_saliency(
        _snapshot(
            _element("offscreen", x=8, y=8, width=2, height=2, visibility_fraction=0.0)
        ),
        _predictions(np.ones((4, 4), dtype=np.float32)),
        SaliencyAggregationConfig(),
    )

    profile = _profile(profiles, "offscreen")
    aggregate = profile.aggregates[0]

    assert aggregate.clipped_area == 0.0
    assert aggregate.raw_mass == 0.0
    assert aggregate.mass_share == 0.0
    assert aggregate.raw_score == 0.0
    assert aggregate.adjusted_score == 0.0
    assert profile.immediate is None


def test_nested_structural_container_is_suppressed_but_evidence_remains() -> None:
    saliency = np.zeros((8, 8), dtype=np.float32)
    saliency[3:5, 3:5] = 1.0
    profiles = aggregate_saliency(
        _snapshot(
            _element(
                "background-card",
                x=0,
                y=0,
                width=8,
                height=8,
                role="other",
                actionable=False,
            ),
            _element("button", x=3, y=3, width=2, height=2),
        ),
        _predictions(saliency),
        SaliencyAggregationConfig(),
    )

    card = _profile(profiles, "background-card")
    button = _profile(profiles, "button")

    assert card.aggregates[0].raw_mass > 0
    assert card.immediate is None
    assert button.immediate is not None
    assert button.immediate.score is not None
    assert button.immediate.score > 0


def test_identical_bounds_do_not_create_false_parent_child_suppression() -> None:
    profiles = aggregate_saliency(
        _snapshot(
            _element(
                "container",
                x=1,
                y=1,
                width=2,
                height=2,
                role="other",
                actionable=False,
            ),
            _element("button", x=1, y=1, width=2, height=2),
        ),
        _predictions(np.ones((4, 4), dtype=np.float32)),
        SaliencyAggregationConfig(),
    )

    assert _profile(profiles, "container").immediate is not None
    assert _profile(profiles, "button").immediate is not None


def test_large_container_does_not_win_from_area_alone() -> None:
    saliency = np.full((8, 8), 0.1, dtype=np.float32)
    saliency[3:5, 3:5] = 1.0
    profiles = aggregate_saliency(
        _snapshot(
            _element(
                "background-card",
                x=0,
                y=0,
                width=8,
                height=8,
                role="other",
                actionable=False,
            ),
            _element("button", x=3, y=3, width=2, height=2),
        ),
        _predictions(saliency),
        SaliencyAggregationConfig(),
    )

    def score(element_id: str) -> float:
        estimate = _profile(profiles, element_id).immediate
        if estimate is None or estimate.score is None:
            return 0.0
        return estimate.score

    assert score("button") > score("background-card")


def test_zero_constant_and_tiny_peak_maps_remain_finite() -> None:
    zero_profiles = aggregate_saliency(
        _snapshot(
            _element("first", x=0, y=0, width=2, height=2),
            _element("second", x=2, y=2, width=2, height=2),
        ),
        _predictions(np.zeros((4, 4), dtype=np.float32)),
        SaliencyAggregationConfig(),
    )
    tiny_peak = np.zeros((4, 4), dtype=np.float32)
    tiny_peak[0, 0] = 1.0
    peak_profiles = aggregate_saliency(
        _snapshot(_element("peak", x=0, y=0, width=1, height=1)),
        _predictions(tiny_peak),
        SaliencyAggregationConfig(),
    )

    for profile in (*zero_profiles, *peak_profiles):
        for aggregate in profile.aggregates:
            assert all(
                math.isfinite(value)
                for value in (
                    aggregate.density,
                    aggregate.p95,
                    aggregate.raw_mass,
                    aggregate.mass_share,
                    aggregate.clipped_area,
                    aggregate.raw_score,
                    aggregate.adjusted_score,
                )
            )
        if profile.immediate is not None:
            assert profile.immediate.score is not None
            assert math.isfinite(profile.immediate.score)

    assert _profile(peak_profiles, "peak").aggregates[0].robust_peak == 1.0


def test_all_duration_maps_produce_duration_specific_aggregate_provenance() -> None:
    predictions = _predictions(
        np.full((2, 2), 0.1, dtype=np.float32),
        np.full((2, 2), 0.2, dtype=np.float32),
        np.full((2, 2), 0.3, dtype=np.float32),
    )

    profile = aggregate_saliency(
        _snapshot(_element("button", x=0, y=0, width=2, height=2)),
        predictions,
        SaliencyAggregationConfig(),
    )[0]

    assert tuple(item.duration for item in profile.aggregates) == (
        AttentionDuration.ONE_SECOND,
        AttentionDuration.THREE_SECONDS,
        AttentionDuration.SEVEN_SECONDS,
    )
    assert profile.immediate is not None
    assert profile.early is not None
    assert profile.eventual is not None
    assert profile.immediate.source == "foveacast"
    assert profile.early.source == "foveacast"
    assert profile.eventual.source == "foveacast"


def test_prediction_set_viewport_mismatch_is_rejected() -> None:
    prediction = _prediction(np.ones((2, 2), dtype=np.float32))
    mismatched = SaliencyPrediction(
        viewport_id="other-viewport",
        duration=prediction.duration,
        plane=prediction.plane,
        metadata=prediction.metadata,
    )

    with pytest.raises(ValueError, match="viewport"):
        aggregate_saliency(
            _snapshot(_element("button", x=0, y=0, width=1, height=1)),
            SaliencyPredictionSet(
                viewport_id="other-viewport", predictions=(mismatched,)
            ),
            SaliencyAggregationConfig(),
        )


def test_config_rejects_invalid_temperature_and_keeps_versioned_defaults() -> None:
    config = SaliencyAggregationConfig()

    assert config.version == "element-saliency-aggregation-v1"
    assert (
        config.density_weight,
        config.robust_peak_weight,
        config.mass_share_weight,
    ) == (
        0.60,
        0.25,
        0.15,
    )
    with pytest.raises(ValueError, match="temperature"):
        SaliencyAggregationConfig(temperature=0.0)
