from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from tests.e2e.saliency_support import (
    FOCUSED_CELL_KEYS,
    CellKey,
    FocusedCellEvidence,
    FocusedPromotionGate,
    OperationalBudget,
    OperationalEvidence,
    OperationalMeasurementStatus,
    OsHighWaterOperationalSampler,
    assert_hotspot_target_alignment,
    material_unexplained_regression,
    run_deterministic_fallback_acceptance,
    run_deterministic_focused_acceptance,
)
from ux_analyzer.application.evaluation import evaluate_experiment_results
from ux_analyzer.application.experiment import ExperimentContext, expand_experiment
from ux_analyzer.config.loader import load_project
from ux_analyzer.domain.interface import BoundingBox, ElementSnapshot, ViewportSnapshot
from ux_analyzer.domain.saliency import (
    AttentionDuration,
    SaliencyGeometry,
    SaliencyPlane,
    SaliencyPrediction,
    SaliencyPredictionMetadata,
    SaliencyPredictionRequest,
    SaliencyPredictionSet,
    SaliencyRequestMetadata,
    SearchStage,
)
from ux_analyzer.ports.observation import ObservationCapture, ViewportSize
from ux_analyzer.providers.cognitive import StructuredCognitiveAgent
from ux_analyzer.providers.prominence import HeuristicProminenceProvider
from ux_analyzer.providers.saliency_aggregation import aggregate_saliency
from ux_analyzer.providers.saliency_prominence import FoveacastProminenceProvider
from ux_analyzer.saliency.model_registry import ModelRegistry, ModelState
from ux_analyzer.storage.saliency_cache import (
    SaliencyCache,
    SaliencyCacheKey,
)

DEMO_PROJECT = Path(__file__).parents[2] / "benchmarks" / "demo" / "project.yaml"
SALIENCY_EXPERIMENT = (
    Path(__file__).parents[2]
    / "benchmarks"
    / "demo"
    / "experiments"
    / "saliency-focused-validation.yaml"
)


def test_saliency_focused_validation_expands_exactly_eight_provider_cells() -> None:
    loaded = load_project(DEMO_PROJECT)
    experiment = next(
        item
        for item in loaded.project.experiments
        if item.id == "saliency-focused-validation"
    )

    specs = expand_experiment(
        ExperimentContext(
            definition=experiment,
            project=loaded.project,
            config_digest=loaded.config_digest_for(experiment.id),
        )
    )

    assert len(specs) == 8
    assert {spec.scenario.id for spec in specs} == {
        "invite-teammate",
        "enable-2fa",
    }
    assert {spec.application_version.id for spec in specs} == {
        "fixture-app-defective",
        "fixture-app-improved",
    }
    assert {spec.persona.id for spec in specs} == {"first-time-nontechnical"}
    assert {spec.policy.value for spec in specs} == {"progressive-prominence-scent"}
    assert {spec.prominence_provider_id for spec in specs} == {
        "heuristic",
        "foveacast",
    }
    assert {spec.seed for spec in specs} == {0}
    assert {spec.model_trial for spec in specs} == {0}
    assert len({spec.run_id for spec in specs}) == 8
    assert SALIENCY_EXPERIMENT.exists()
    fragment = yaml.safe_load(SALIENCY_EXPERIMENT.read_text(encoding="utf-8"))
    assert fragment["id"] == "saliency-focused-validation"
    assert fragment["prominence_provider_ids"] == ["heuristic", "foveacast"]
    assert fragment["policies"] == ["progressive-prominence-scent"]
    assert fragment["seeds"] == [0]
    assert fragment["model_trials"] == [0]


def _typed_gate_cells(
    *, sampler_provider: str = "fake-sampler", synthetic: bool = False
) -> list[FocusedCellEvidence]:
    return [
        FocusedCellEvidence(
            key=key,
            comparison_valid=True,
            learned_valid=key.provider_id == "foveacast",
            fallback_substitution=False,
            verified_completion=True,
            alignment_failures=0,
            leakage_failures=0,
            operational=OperationalEvidence(
                status=OperationalMeasurementStatus.MEASURED,
                latency_ms=4.0,
                peak_rss_bytes=64 * 1024 * 1024,
                measurement_scope="focused-cell",
                sampler_provider=sampler_provider,
                platform="test",
                synthetic=synthetic,
                model_checksums=("1" * 64, "2" * 64, "3" * 64)
                if key.provider_id == "foveacast"
                else (),
            ),
            planted_defect_improved=(
                key.provider_id == "foveacast"
                and key.scenario_id == "invite-teammate"
                and key.version_id == "fixture-app-defective"
            ),
            material_unexplained_regression=False,
        )
        for key in FOCUSED_CELL_KEYS
    ]


def test_focused_promotion_gate_rejects_unassessed_budget_without_inventing_threshold() -> (
    None
):
    result = FocusedPromotionGate().evaluate(_typed_gate_cells())

    assert not result.passed
    assert "operational budget is unassessed" in result.reasons


def test_focused_promotion_gate_rejects_fallback_and_missing_measurement() -> None:
    cells = _typed_gate_cells()
    fallback_key = CellKey(
        scenario_id="enable-2fa",
        version_id="fixture-app-improved",
        provider_id="foveacast",
    )
    index = next(index for index, cell in enumerate(cells) if cell.key == fallback_key)
    cells[index] = replace(
        cells[index],
        fallback_substitution=True,
        operational=OperationalEvidence.unavailable(
            measurement_scope="focused-cell",
            sampler_provider="missing-sampler",
            platform="test",
            reason="peak RSS unsupported",
        ),
    )

    result = FocusedPromotionGate().evaluate(cells)

    assert not result.passed
    assert "operational measurement unavailable" in result.reasons
    assert "fallback substituted for learned output" in result.reasons


def test_focused_promotion_gate_rejects_synthetic_measurement() -> None:
    result = FocusedPromotionGate(
        operational_budget=OperationalBudget(
            max_latency_ms=10.0,
            max_peak_rss_bytes=128 * 1024 * 1024,
        )
    ).evaluate(_typed_gate_cells(synthetic=True))

    assert not result.passed
    assert "synthetic operational evidence cannot promote" in result.reasons


def test_focused_promotion_gate_accepts_assessed_os_measurement() -> None:
    result = FocusedPromotionGate(
        operational_budget=OperationalBudget(
            max_latency_ms=10.0,
            max_peak_rss_bytes=128 * 1024 * 1024,
        )
    ).evaluate(_typed_gate_cells(sampler_provider="os-high-water"))

    assert result.passed


def test_focused_promotion_gate_rejects_malformed_and_unpaired_cells() -> None:
    cells = _typed_gate_cells()
    cells[-1] = replace(
        cells[-1],
        key=CellKey(
            scenario_id="unknown-scenario",
            version_id="fixture-app-improved",
            provider_id="foveacast",
        ),
    )

    result = FocusedPromotionGate().evaluate(cells)

    assert not result.passed
    assert "exact focused cell identities are incomplete" in result.reasons
    assert "learned cell is missing" in result.reasons


@pytest.mark.asyncio
async def test_deterministic_fake_matrix_uses_production_orchestration_and_evaluation(
    tmp_path: Path,
) -> None:
    acceptance = await run_deterministic_focused_acceptance(tmp_path)

    assert len(acceptance.specs) == 8
    assert {CellKey.from_spec(spec) for spec in acceptance.specs} == set(
        FOCUSED_CELL_KEYS
    )
    assert acceptance.failures == ()
    assert len(acceptance.evaluation.cell_aggregates) == 8
    assert len(acceptance.evaluation.variant_comparisons) == 4
    assert acceptance.report_path.is_file()
    assert acceptance.experiment_path.is_file()
    persisted = json.loads(acceptance.experiment_path.read_text(encoding="utf-8"))
    assert len(persisted["run_metrics"]) == 8
    assert len(persisted["cell_aggregates"]) == 8
    assert len(persisted["variant_comparisons"]) == 4
    assert {
        row["prominence_provider_id"] for row in persisted["cell_aggregates"]
    } == {"heuristic", "foveacast"}
    assert {
        row["baseline"]["prominence_provider_id"]
        for row in persisted["variant_comparisons"]
    } == {"heuristic", "foveacast"}
    assert persisted["focused_acceptance"]["gate"]["cell_count"] == 8
    assert persisted["focused_acceptance"]["gate"]["paired_cell_count"] == 4
    assert len(acceptance.cell_evidence) == 8
    gate = FocusedPromotionGate().evaluate(acceptance.cell_evidence)
    assert not gate.passed
    assert gate.cell_count == 8
    assert gate.paired_cell_count == 4
    assert "operational budget is unassessed" in gate.reasons
    assert all(
        evidence.operational.status is OperationalMeasurementStatus.MEASURED
        for evidence in acceptance.cell_evidence
    )
    learned_results = [
        result
        for result in acceptance.results
        if result.state.spec.prominence_provider_id == "foveacast"
    ]
    assert len(learned_results) == 4
    assert all(
        acceptance.model_calls_by_run[result.run_id] == 1
        for result in learned_results
    )
    for result in learned_results:
        assert result.metrics is not None
        assert result.metrics.comparison_valid is True
        assert isinstance(result.bundle_path, Path)
        timeline = (result.bundle_path / "timeline.jsonl").read_text(encoding="utf-8")
        assert "saliency-profiles-recorded" in timeline
        assert '"search_stage":"initial"' in timeline
        assert '"search_stage":"exploration"' in timeline
        assert '"search_stage":"persistent"' in timeline
        assert '"cache_state":"hit"' in timeline
        saliency_files: list[Path] = list((result.bundle_path / "saliency").rglob("*"))
        assert len([path for path in saliency_files if path.is_file()]) == 24

    report = acceptance.report_path.read_text(encoding="utf-8")
    assert "Saliency evidence" in report
    assert "foveacast" in report
    assert "Focused saliency promotion gate" in report
    assert "Operational evidence" in report
    assert "Operational evidence" in acceptance.operational_evidence_path.read_text(
        encoding="utf-8"
    )
    loaded = load_project(DEMO_PROJECT)
    sensitive_values = tuple(
        value
        for scenario in loaded.project.scenarios
        for key, value in scenario.fixture_inputs.values.items()
        if key in scenario.fixture_inputs.sensitive_keys
    )
    private_markers = (
        b'"selector"',
        b'"test_id"',
        b'"hidden_label"',
        b'"destination_url"',
        b'"execution_reference"',
        b"private-target",
        b"private-competitor",
    )
    for path in acceptance.report_path.parent.rglob("*"):
        if not path.is_file():
            continue
        payload = path.read_bytes()
        assert all(value.encode() not in payload for value in sensitive_values)
        assert all(marker not in payload for marker in private_markers)


def test_paired_evaluation_metrics_drive_regression_gate_input(tmp_path: Path) -> None:
    acceptance = asyncio.run(run_deterministic_focused_acceptance(tmp_path))
    learned = next(
        result
        for result in acceptance.results
        if result.state.spec.prominence_provider_id == "foveacast"
        and result.state.spec.scenario.id == "invite-teammate"
        and result.state.spec.application_version.id == "fixture-app-defective"
    )
    heuristic = next(
        result
        for result in acceptance.results
        if result.state.spec.prominence_provider_id == "heuristic"
        and result.state.spec.scenario.id == "invite-teammate"
        and result.state.spec.application_version.id == "fixture-app-defective"
    )
    assert learned.metrics is not None
    assert heuristic.metrics is not None
    assert material_unexplained_regression(learned.metrics, heuristic.metrics) is False

    injected_regression = replace(
        learned.metrics,
        target_prominence=(heuristic.metrics.target_prominence or 0.0) - 0.25,
    )
    assert material_unexplained_regression(injected_regression, heuristic.metrics)


def test_fake_hotspot_maps_to_target_center_with_pixel_tolerance(
    tmp_path: Path,
) -> None:
    model = _DeterministicSaliencyModel()
    provider = FoveacastProminenceProvider(
        model,
        SaliencyCache(tmp_path / "fake-hotspot-cache"),
        heuristic_provider=HeuristicProminenceProvider(),
    )
    snapshot = _focused_snapshot()
    provider.score(_focused_capture(), snapshot, SearchStage.INITIAL, None)

    assert model.last_predictions is not None
    assert assert_hotspot_target_alignment(snapshot, model.last_predictions) == 0


def _focused_snapshot() -> ViewportSnapshot:
    return ViewportSnapshot(
        id="focused-viewport",
        elements=(
            ElementSnapshot(
                id="surface",
                role="other",
                label="Workspace surface",
                bounds=BoundingBox(x=0, y=0, width=8, height=8),
                visibility_fraction=1.0,
                actionable=False,
            ),
            ElementSnapshot(
                id="target",
                role="button",
                label="Invite teammate",
                bounds=BoundingBox(x=1, y=1, width=2, height=2),
                visibility_fraction=1.0,
                actionable=True,
            ),
            ElementSnapshot(
                id="competitor",
                role="button",
                label="Share",
                bounds=BoundingBox(x=5, y=5, width=2, height=2),
                visibility_fraction=1.0,
                actionable=True,
            ),
        ),
    )


def _focused_capture() -> ObservationCapture:
    screenshot = b"deterministic fake saliency screenshot"
    return ObservationCapture(
        session_id="focused-session",
        viewport_id="focused-viewport",
        url="http://fixture.test/focused",
        title="Focused fixture",
        viewport=ViewportSize(width=8, height=8),
        screenshot=screenshot,
    )


class _DeterministicSaliencyModel:
    id = "foveacast"
    model_id = "foveacast-v0.2.0"
    model_version = "v0.2.0"
    provider_version = "foveacast-adapter-v1"
    precision = "fp16"
    preprocessing_version = "foveacast-preprocess-v1"
    actual_execution_provider = "CPUExecutionProvider"
    model_checksums = {"1s": "1" * 64, "3s": "2" * 64, "7s": "3" * 64}

    def __init__(self, *, failure: Exception | None = None) -> None:
        self.calls = 0
        self.failure = failure
        self.last_predictions: SaliencyPredictionSet | None = None

    def predict(self, request: SaliencyPredictionRequest) -> SaliencyPredictionSet:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        geometry = SaliencyGeometry(
            geometry_version="saliency-geometry-v1",
            source_dimensions=request.metadata.screenshot_dimensions,
            native_dimensions=(8, 8),
            content_dimensions=(8, 8),
            pad_left=0,
            pad_top=0,
            pad_right=0,
            pad_bottom=0,
            scale=1.0,
            scale_x=1.0,
            scale_y=1.0,
            device_pixel_ratio=request.metadata.device_pixel_ratio,
            zoom=request.metadata.zoom,
        )
        maps: dict[AttentionDuration, np.ndarray[Any, Any]] = {}
        for duration, target_value in (
            (AttentionDuration.ONE_SECOND, 0.90),
            (AttentionDuration.THREE_SECONDS, 0.75),
            (AttentionDuration.SEVEN_SECONDS, 0.60),
        ):
            values = np.full((8, 8), 0.05, dtype=np.float32)
            values[1:3, 1:3] = target_value
            values[5:7, 5:7] = 0.20
            maps[duration] = values
        predictions = tuple(
            SaliencyPrediction(
                viewport_id=request.metadata.viewport_id,
                duration=duration,
                plane=SaliencyPlane(
                    width=8,
                    height=8,
                    values=values.tobytes(order="C"),
                ),
                metadata=SaliencyPredictionMetadata(
                    provider_id=self.id,
                    model_id=self.model_id,
                    provider_version=self.provider_version,
                    model_version=self.model_version,
                    model_checksum=self.model_checksums[duration.value],
                    input_dimensions=(8, 8),
                    output_dimensions=(8, 8),
                    geometry=geometry,
                    preprocessing_version=self.preprocessing_version,
                    inference_duration_ms=1.0,
                    execution_provider=self.actual_execution_provider,
                ),
            )
            for duration, values in maps.items()
        )
        prediction_set = SaliencyPredictionSet(
            viewport_id=request.metadata.viewport_id,
            predictions=predictions,
            request_metadata=request.metadata,
        )
        self.last_predictions = prediction_set
        return prediction_set


def test_deterministic_fake_maps_cover_cache_stages_profiles_and_cognitive_redaction(
    tmp_path: Path,
) -> None:
    model = _DeterministicSaliencyModel()
    provider = FoveacastProminenceProvider(
        model,
        SaliencyCache(tmp_path / "cache"),
        heuristic_provider=HeuristicProminenceProvider(),
    )
    snapshot = _focused_snapshot()
    capture = _focused_capture()

    initial = provider.score(capture, snapshot, SearchStage.INITIAL, None)
    exploration = provider.score(capture, snapshot, SearchStage.EXPLORATION, None)
    persistent = provider.score(capture, snapshot, SearchStage.PERSISTENT, None)

    assert model.calls == 1
    assert initial.cache_state == "miss"
    assert exploration.cache_state == "hit"
    assert persistent.cache_state == "hit"
    assert initial.stage is SearchStage.INITIAL
    assert exploration.stage is SearchStage.EXPLORATION
    assert persistent.stage is SearchStage.PERSISTENT
    assert dict(persistent.selected_mixture) == {"3s": 0.25, "7s": 0.75}
    assert sum(item.normalized_probability for item in initial.scores) == pytest.approx(
        1.0
    )
    profiles = initial.learned_profiles
    assert {profile.element_id for profile in profiles} == {
        "surface",
        "target",
        "competitor",
    }
    target = next(profile for profile in profiles if profile.element_id == "target")
    surface = next(profile for profile in profiles if profile.element_id == "surface")
    assert target.immediate is not None
    assert target.immediate.score is not None
    assert target.immediate.score > 0.5
    assert surface.immediate is None
    assert len(target.prediction_provenance) == 3
    cache_key = SaliencyCacheKey(
        viewport_id="native-inference",
        screenshot_sha256=hashlib.sha256(capture.screenshot).hexdigest(),
        screenshot_dimensions=(8, 8),
        device_pixel_ratio=1.0,
        zoom=1.0,
        model_checksums=("1" * 64, "2" * 64, "3" * 64),
        preprocessing_version="foveacast-preprocess-v1",
        precision="fp16",
        execution_provider="CPUExecutionProvider",
        aggregation_version="element-saliency-aggregation-v1",
    )
    cache_entry = SaliencyCache(tmp_path / "cache").load(cache_key)
    assert cache_entry is not None
    assert len(cache_entry.artifact_paths) == 8
    assert all(path.exists() for path in cache_entry.artifact_paths.values())

    class _RecordingCognitiveClient:
        endpoint_origin = "internal"

        def __init__(self) -> None:
            self.messages: tuple[object, ...] = ()

        async def complete(
            self,
            schema: Any,
            messages: tuple[object, ...],
            *,
            model: str,
            role: object,
        ) -> Any:
            del model, role
            self.messages = messages
            return schema.model_validate({"action": "abandon", "reason": "done"})

    client = _RecordingCognitiveClient()
    cognitive = StructuredCognitiveAgent(cast(Any, client), model="fake-cognitive")
    from ux_analyzer.domain.attention import ProgressiveObservation

    asyncio.run(
        cognitive.decide(
            "Invite teammate",
            ProgressiveObservation.from_snapshot(
                snapshot, newly_revealed_ids=("target",)
            ),
        )
    )
    prompt = str(getattr(client.messages[1], "content"))
    assert "raw_score" not in prompt
    assert "normalized_probability" not in prompt
    assert "feature_contributions" not in prompt
    assert "selector" not in prompt
    assert "execution_reference" not in prompt


def test_deterministic_fake_model_failure_invalidates_learned_comparison(
    tmp_path: Path,
) -> None:
    provider = FoveacastProminenceProvider(
        _DeterministicSaliencyModel(failure=RuntimeError("fake runtime unavailable")),
        SaliencyCache(tmp_path / "fallback-cache"),
        heuristic_provider=HeuristicProminenceProvider(),
    )

    batch = provider.score(
        _focused_capture(), _focused_snapshot(), SearchStage.INITIAL, None
    )

    assert batch.learned_available is False
    assert batch.cache_state == "fallback"
    assert batch.fallback_reason == "fake runtime unavailable"
    assert batch.active_provider_id == HeuristicProminenceProvider.id


@pytest.mark.asyncio
async def test_fake_saliency_run_records_stages_artifacts_and_fallback_invalidation(
    tmp_path: Path,
) -> None:
    fallback_acceptance = await run_deterministic_fallback_acceptance(
        tmp_path / "fallback"
    )
    fallback_result = fallback_acceptance.result

    assert fallback_result.ux_sample_valid is False
    assert fallback_result.metrics is not None
    assert fallback_result.metrics.comparison_valid is False
    fallback_evaluation = evaluate_experiment_results((fallback_result,))
    assert fallback_evaluation.cell_aggregates == ()
    assert fallback_evaluation.variant_comparisons == ()
    assert fallback_result.ux_sample_invalid_reason == (
        "saliency-fallback: fake runtime unavailable; selector=[REDACTED]"
    )
    assert isinstance(fallback_result.bundle_path, Path)
    events = [
        cast(dict[str, object], json.loads(line))
        for line in (fallback_result.bundle_path / "timeline.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(event.get("kind") == "saliency-fallback-recorded" for event in events)
    assert "data-secret" not in json.dumps(events)
    assert fallback_acceptance.report_path.is_file()
    fallback_report = fallback_acceptance.report_path.read_text(encoding="utf-8")
    assert "Aggregate metrics unavailable." in fallback_report
    assert "Directional gates unavailable." in fallback_report


def test_fake_saliency_report_renders_replay_tabs_and_redacted_payload(
    tmp_path: Path,
) -> None:
    acceptance = asyncio.run(run_deterministic_focused_acceptance(tmp_path))
    html = acceptance.report_path.read_text(encoding="utf-8")

    assert "Saliency evidence" in html
    assert "1s" in html and "3s" in html and "7s" in html
    assert "Heatmap-only artifact" in html
    assert "Aggregation components" in html
    assert "CPUExecutionProvider" in html
    assert "secret-token" not in html
    assert "data-testid=secret" not in html
    assert "execution_reference" not in html


def _real_model_screenshot() -> bytes:
    image = Image.new("RGB", (320, 240), color=(30, 40, 50))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 280, 200), fill=(55, 110, 190), outline=(220, 240, 255))
    draw.rectangle((260, 10, 300, 40), fill=(180, 70, 70), outline=(255, 220, 220))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _real_model_snapshot(scenario_id: str) -> ViewportSnapshot:
    label = (
        "Invite teammate"
        if scenario_id == "invite-teammate"
        else "Enable two-factor authentication"
    )
    return ViewportSnapshot(
        id=f"{scenario_id}-focused-viewport",
        elements=(
            ElementSnapshot(
                id="semantic-container",
                role="other",
                label="Settings surface",
                bounds=BoundingBox(x=0, y=0, width=320, height=240),
                visibility_fraction=1.0,
                actionable=False,
            ),
            ElementSnapshot(
                id="target",
                role="button",
                label=label,
                bounds=BoundingBox(x=40, y=40, width=240, height=160),
                visibility_fraction=1.0,
                actionable=True,
            ),
            ElementSnapshot(
                id="competitor",
                role="button",
                label="Share",
                bounds=BoundingBox(x=260, y=10, width=40, height=30),
                visibility_fraction=1.0,
                actionable=True,
            ),
        ),
    )


@pytest.mark.live
def test_real_foveacast_focused_states_have_maps_profiles_alignment_and_replay_artifacts(
    tmp_path: Path,
) -> None:
    if os.environ.get("UXA_RUN_REAL_SALIENCY_TESTS") != "1":
        pytest.skip("set UXA_RUN_REAL_SALIENCY_TESTS=1 for real-model acceptance")

    registry = ModelRegistry()
    status = registry.status("foveacast-v0.2.0", provider="cpu")
    if status.state is not ModelState.READY:
        pytest.skip(
            "pinned Foveacast CPU model/runtime unavailable: "
            + "; ".join(status.diagnostics)
        )

    from ux_analyzer.adapters.saliency.foveacast import FoveacastSaliencyProvider

    provider = FoveacastSaliencyProvider(
        registry,
        execution_provider_preference="cpu",
    )
    screenshot = _real_model_screenshot()
    operational_rows: list[dict[str, object]] = []
    for scenario_id in ("invite-teammate", "enable-2fa"):
        snapshot = _real_model_snapshot(scenario_id)
        capture = ObservationCapture(
            session_id=f"real-saliency-{scenario_id}",
            viewport_id=snapshot.id,
            url="http://fixture.test/focused",
            title="Focused fixture screenshot",
            viewport=ViewportSize(width=320, height=240),
            screenshot=screenshot,
            snapshot=snapshot,
        )
        request = SaliencyPredictionRequest(
            screenshot=screenshot,
            metadata=SaliencyRequestMetadata(
                viewport_id=snapshot.id,
                screenshot_sha256=hashlib.sha256(screenshot).hexdigest(),
                screenshot_width=320,
                screenshot_height=240,
                device_pixel_ratio=1.0,
                zoom=1.0,
                requested_durations=("1s", "3s", "7s"),
                model_set=("foveacast-v0.2.0",),
                precision="fp16",
                execution_provider_preference="cpu",
            ),
        )
        model_checksums = tuple(
            item.artifact.sha256
            for item in status.artifacts
            if item.artifact.filename.endswith(".onnx")
        )
        assert len(model_checksums) == 3
        sampler = OsHighWaterOperationalSampler(
            {
                CellKey(
                    scenario_id=scenario_id,
                    version_id="fixture-app-improved",
                    provider_id="foveacast",
                ): model_checksums
            }
        )

        async def predict() -> object:
            return provider.predict(request)

        predictions = asyncio.run(
            sampler.measure(
                CellKey(
                    scenario_id=scenario_id,
                    version_id="fixture-app-improved",
                    provider_id="foveacast",
                ),
                predict,
            )
        )
        assert isinstance(predictions, SaliencyPredictionSet)
        operational = sampler.evidence_for(
            CellKey(
                scenario_id=scenario_id,
                version_id="fixture-app-improved",
                provider_id="foveacast",
            )
        )
        assert operational.status in {
            OperationalMeasurementStatus.MEASURED,
            OperationalMeasurementStatus.UNAVAILABLE,
        }
        assert operational.measurement_scope == "focused-cell"
        assert operational.sampler_provider == "os-high-water"
        assert operational.model_checksums == model_checksums
        operational_rows.append(
            operational.to_dict(
                CellKey(
                    scenario_id=scenario_id,
                    version_id="fixture-app-improved",
                    provider_id="foveacast",
                )
            )
        )

        assert len(predictions.predictions) == 3
        assert {
            str(getattr(prediction.duration, "value", prediction.duration))
            for prediction in predictions.predictions
        } == {
            "1s",
            "3s",
            "7s",
        }
        assert all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for prediction in predictions.predictions
            for value in prediction.plane.float_values()
        )
        profiles = aggregate_saliency(snapshot, predictions)
        target = next(profile for profile in profiles if profile.element_id == "target")
        container = next(
            profile
            for profile in profiles
            if profile.element_id == "semantic-container"
        )
        assert target.immediate is not None
        assert target.early is not None
        assert target.eventual is not None
        assert len(target.prediction_provenance) == 3
        assert all(
            aggregate.clipped_area == pytest.approx(240 * 160)
            and aggregate.raw_mass >= 0
            for aggregate in target.aggregates
        )
        assert container.immediate is None
        assert container.early is None
        assert container.eventual is None
        assert assert_hotspot_target_alignment(
            snapshot, predictions, tolerance_px=32.0
        ) == 0

        cache = SaliencyCache(tmp_path / scenario_id)
        prominence = FoveacastProminenceProvider(
            provider,
            cache,
            heuristic_provider=HeuristicProminenceProvider(),
        )
        first_batch = prominence.score(capture, snapshot, SearchStage.INITIAL, None)
        second_batch = prominence.score(
            capture, snapshot, SearchStage.EXPLORATION, None
        )
        assert first_batch.learned_available is True
        assert first_batch.cache_state == "miss"
        assert second_batch.cache_state == "hit"
        cache_key = replace(
            cache_key_from_predictions(predictions),
            viewport_id="native-inference",
        )
        replay = cache.load(cache_key)
        assert replay is not None
        assert len(first_batch.learned_profiles) == len(snapshot.elements)
        assert len(replay.artifact_paths) == 8
        assert all(path.exists() for path in replay.artifact_paths.values())

    operational_path = tmp_path / "real-operational-evidence.json"
    operational_path.write_text(
        json.dumps({"title": "Operational evidence", "cells": operational_rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    assert operational_path.is_file()
    persisted_operational = json.loads(operational_path.read_text(encoding="utf-8"))
    assert len(persisted_operational["cells"]) == 2


def cache_key_from_predictions(predictions: SaliencyPredictionSet) -> SaliencyCacheKey:
    return SaliencyCacheKey.from_prediction_set(
        predictions,
        aggregation_version="element-saliency-aggregation-v1",
    )
