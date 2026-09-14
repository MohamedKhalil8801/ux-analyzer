from __future__ import annotations

import asyncio
import hashlib
import json
import struct
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ux_analyzer.adapters.web.verifier import WebVerifier
from ux_analyzer.application.evaluation import (
    EvaluationTarget,
    RunEvaluationInputs,
    evaluate_run,
    evaluation_target_for,
)
from ux_analyzer.application.experiment import ExperimentFailure, ExperimentResult
from ux_analyzer.application.memory import MemoryPolicy, MemoryPolicyConfig
from ux_analyzer.application.run_agent import (
    RunAgent,
    RunFinalizationError,
    _ux_sample_validity,
)
from ux_analyzer.application.saliency import ProminenceBatch
from ux_analyzer.domain.attention import (
    Abandon,
    AttentionState,
    FullScent,
    InteractWithElement,
    ProgressiveObservation,
)
from ux_analyzer.domain.benchmark import (
    ApplicationVersion,
    ApplicationVersionKind,
    Budget,
    ColourChangeVerifierSpec,
    ExperimentPolicy,
    FixtureInputs,
    FixtureStateVerifierSpec,
    Persona,
    Scenario,
    ScenarioEvaluationTarget,
    VisibleResultVerifierSpec,
)
from ux_analyzer.domain.interface import (
    BoundingBox,
    ElementSnapshot,
    PrivateExecutionReference,
    RegionSnapshot,
    ViewportSnapshot,
)
from ux_analyzer.domain.run import (
    AgentAbandoned,
    BudgetExhausted,
    ProviderManifest,
    VerificationResult,
    VerifiedSuccess,
)
from ux_analyzer.domain.saliency import (
    AttentionDuration,
    SaliencyGeometry,
    SaliencyPlane,
    SaliencyPrediction,
    SaliencyPredictionMetadata,
    SaliencyPredictionRequest,
    SaliencyPredictionSet,
    SearchStage,
)
from ux_analyzer.ports.artifacts import (
    ArtifactReference,
    BundleManifest,
    ProminenceRecordedEvent,
    SaliencyArtifactKind,
    SaliencyFallbackRecordedEvent,
    SaliencyProfilesRecordedEvent,
)
from ux_analyzer.ports.models import (
    CognitiveRunContext,
    ModelCallRecord,
    ModelResponseValidationError,
    ModelRole,
    RetryEvent,
    TokenUsage,
)
from ux_analyzer.ports.observation import (
    ObservationCapture,
    ObservationProviderError,
    ObservationSessionConfig,
    OpenMenuAction,
    PlatformAction,
    PlatformActionResult,
    SafetyBlocked,
    SessionHandle,
    ToggleAction,
    ViewportSize,
    WaitAction,
)
from ux_analyzer.ports.observation import TestAccountId as AccountId
from ux_analyzer.providers.attention_policy import (
    AttentionPolicyConfig,
    ObservationSelection,
    ProgressiveAttentionPolicy,
)
from ux_analyzer.providers.cognitive import CognitiveDecision, StructuredCognitiveAgent
from ux_analyzer.providers.finding_rules import FindingRuleSet
from ux_analyzer.providers.prominence import ProminenceResult
from ux_analyzer.providers.saliency_prominence import FoveacastProminenceProvider
from ux_analyzer.storage.run_bundle import FilesystemRunBundleWriter
from ux_analyzer.storage.saliency_cache import SaliencyCache


@dataclass(frozen=True, slots=True)
class ClaimingDecision:
    action: object
    reason: str
    claimed_success: bool = False


class FakeObservationProvider:
    id = "fake-observer"
    platform = "web"
    version = "fake-observer-v1"

    def __init__(
        self,
        snapshots: tuple[ViewportSnapshot, ...],
        *,
        results: tuple[PlatformActionResult, ...] = (),
        failure: BaseException | None = None,
        start_failure: BaseException | None = None,
        screenshot: bytes | None = None,
        screenshots: tuple[bytes, ...] = (),
    ) -> None:
        self.snapshots = snapshots
        self.results = list(results)
        self.failure = failure
        self.start_failure = start_failure
        self.screenshot = screenshot
        self.screenshots = screenshots
        self.started = 0
        self.reset_count = 0
        self.capture_count = 0
        self.captures: list[ObservationCapture] = []
        self.executed: list[PlatformAction] = []
        self.ended = 0

    async def start_session(self, config: ObservationSessionConfig) -> SessionHandle:
        self.started += 1
        if self.start_failure is not None:
            raise self.start_failure
        return SessionHandle(
            session_id=config.session_id,
            test_account_id=AccountId("test-run"),
            viewport=config.viewport,
            trace_path=config.trace_path,
            blocked_events=[],
        )

    async def reset(self, session: SessionHandle) -> None:
        self.reset_count += 1

    async def capture(self, session: SessionHandle) -> ObservationCapture:
        if self.failure is not None:
            raise self.failure
        snapshot = self.snapshots[min(self.capture_count, len(self.snapshots) - 1)]
        self.capture_count += 1
        capture = ObservationCapture(
            session_id=session.session_id,
            viewport_id=snapshot.id,
            url="http://fixture.test/app/run/improved",
            title="Fixture",
            viewport=session.viewport,
            screenshot=(
                self.screenshots[min(self.capture_count - 1, len(self.screenshots) - 1)]
                if self.screenshots
                else self.screenshot or f"screenshot-{snapshot.id}".encode()
            ),
            snapshot=snapshot,
        )
        self.captures.append(capture)
        return capture

    async def execute(
        self, session: SessionHandle, action: PlatformAction
    ) -> PlatformActionResult:
        self.executed.append(action)
        if self.failure is not None:
            raise self.failure
        if self.results:
            return self.results.pop(0)
        return PlatformActionResult(
            succeeded=True,
            url="http://fixture.test/app/run/improved",
            duration_ms=1,
            state_changed=action.kind not in {"wait"},
        )

    async def end_session(self, session: SessionHandle) -> None:
        self.ended += 1


class FakeProminenceProvider:
    def score(self, snapshot: ViewportSnapshot) -> tuple[ProminenceResult, ...]:
        probability = 1 / len(snapshot.elements)
        return tuple(
            ProminenceResult(
                element_id=element.id,
                raw_score=0,
                normalized_probability=probability,
                feature_contributions={"contrast": probability},
            )
            for element in snapshot.elements
        )


class FakeSaliencyProminenceProvider:
    id = "foveacast-prominence"
    version = "foveacast-prominence-v1"
    model_checksums = {"1s": "1" * 64, "3s": "2" * 64, "7s": "3" * 64}
    actual_execution_provider = "CPUExecutionProvider"
    preprocessing_version = "foveacast-preprocess-v1"
    precision = "fp16"
    cache_key = "a" * 64

    def __init__(self, *, fallback_reason: str | None = None) -> None:
        self.fallback_reason = fallback_reason
        self.calls: list[tuple[str, SearchStage, object]] = []

    def score(
        self,
        capture: ObservationCapture,
        snapshot: ViewportSnapshot,
        stage: SearchStage,
        artifacts: object,
    ) -> ProminenceBatch:
        self.calls.append((capture.viewport_id, SearchStage(stage), artifacts))
        scores = FakeProminenceProvider().score(snapshot)
        configured_mixture = getattr(self, "stage_selector", None)
        mixtures = getattr(configured_mixture, "mixtures", {})
        selected_mixture = tuple(
            (str(getattr(duration, "value", duration)), float(weight))
            for duration, weight in mixtures.get(
                stage, mixtures.get(str(stage), {})
            ).items()
        )
        if self.fallback_reason is not None:
            return ProminenceBatch(
                scores=scores,
                provider_id=self.id,
                active_provider_id="heuristic-prominence",
                stage=stage,
                heuristic_scores=scores,
                learned_available=False,
                learned_unavailable_reason=self.fallback_reason,
                fallback_reason=self.fallback_reason,
                cache_state="fallback",
                selected_mixture=selected_mixture,
            )
        return ProminenceBatch(
            scores=scores,
            provider_id=self.id,
            active_provider_id="foveacast",
            stage=stage,
            learned_scores=scores,
            learned_available=True,
            cache_state="miss",
            selected_mixture=selected_mixture,
        )


class MaterializingFakeSaliencyProminenceProvider(FakeSaliencyProminenceProvider):
    def score(
        self,
        capture: ObservationCapture,
        snapshot: ViewportSnapshot,
        stage: SearchStage,
        artifacts: object,
    ) -> ProminenceBatch:
        batch = super().score(capture, snapshot, stage, artifacts)
        write_saliency_artifact = getattr(artifacts, "write_saliency_artifact", None)
        if callable(write_saliency_artifact):
            for filename, kind in (
                ("1s.npz", SaliencyArtifactKind.NATIVE_MAP),
                ("3s.npz", SaliencyArtifactKind.NATIVE_MAP),
                ("7s.npz", SaliencyArtifactKind.NATIVE_MAP),
                ("1s-heatmap.png", SaliencyArtifactKind.HEATMAP),
                ("3s-heatmap.png", SaliencyArtifactKind.HEATMAP),
                ("7s-heatmap.png", SaliencyArtifactKind.HEATMAP),
                ("profiles.json", SaliencyArtifactKind.PROFILES),
                ("metadata.json", SaliencyArtifactKind.METADATA),
            ):
                write_saliency_artifact(
                    f"saliency/native-inference/{filename}",
                    f"artifact-{filename}".encode(),
                    kind,
                )
        return batch


class NoArtifactSaliencyProminenceProvider(FakeSaliencyProminenceProvider):
    """Model path that claims learned output but materializes no bundle evidence."""


class ValidFakeSaliencyModel:
    id = "foveacast"
    model_id = "foveacast-v0.2.0"
    model_version = "v0.2.0"
    provider_version = "foveacast-adapter-v1"
    precision = "fp16"
    preprocessing_version = "foveacast-preprocess-v1"
    actual_execution_provider = "CPUExecutionProvider"
    requested_execution_provider = "cpu"
    model_checksums = {
        AttentionDuration.ONE_SECOND: "a" * 64,
        AttentionDuration.THREE_SECONDS: "b" * 64,
        AttentionDuration.SEVEN_SECONDS: "c" * 64,
    }

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, request: SaliencyPredictionRequest) -> SaliencyPredictionSet:
        self.calls += 1
        metadata = request.metadata
        geometry = SaliencyGeometry(
            geometry_version="saliency-geometry-v1",
            source_dimensions=metadata.screenshot_dimensions,
            native_dimensions=(4, 4),
            content_dimensions=(4, 4),
            pad_left=0,
            pad_top=0,
            pad_right=0,
            pad_bottom=0,
            scale=4 / metadata.screenshot_width,
            scale_x=4 / metadata.screenshot_width,
            scale_y=4 / metadata.screenshot_height,
            device_pixel_ratio=metadata.device_pixel_ratio,
            zoom=metadata.zoom,
        )
        predictions = tuple(
            SaliencyPrediction(
                viewport_id=metadata.viewport_id,
                duration=duration,
                plane=SaliencyPlane(
                    width=4,
                    height=4,
                    values=struct.pack("<16f", *([value] * 16)),
                ),
                metadata=SaliencyPredictionMetadata(
                    provider_id=self.id,
                    model_id=self.model_id,
                    provider_version=self.provider_version,
                    model_version=self.model_version,
                    model_checksum=self.model_checksums[duration],
                    input_dimensions=(4, 4),
                    output_dimensions=(4, 4),
                    geometry=geometry,
                    preprocessing_version=self.preprocessing_version,
                    inference_duration_ms=1.0,
                    execution_provider=self.actual_execution_provider,
                ),
            )
            for duration, value in (
                (AttentionDuration.ONE_SECOND, 0.25),
                (AttentionDuration.THREE_SECONDS, 0.5),
                (AttentionDuration.SEVEN_SECONDS, 0.75),
            )
        )
        return SaliencyPredictionSet(
            viewport_id=metadata.viewport_id,
            predictions=predictions,
            request_metadata=metadata,
        )


class FakeAttentionPolicy:
    def next_observation(
        self,
        state: Any,
        snapshot: ViewportSnapshot,
        scores: tuple[ProminenceResult, ...],
        coarse_scent: object,
        rng: object,
        *,
        recovery_level: int = 0,
    ) -> ObservationSelection:
        del scores, coarse_scent, rng, recovery_level
        element = next(
            (item for item in snapshot.elements if item.id not in state.noticed_ids),
            None,
        )
        if element is None:
            raise RuntimeError(
                f"no unobserved element: {snapshot.id} {sorted(state.noticed_ids)}"
            )
        observation = ProgressiveObservation.from_snapshot(
            snapshot, newly_revealed_ids=(element.id,)
        )
        return ObservationSelection(
            observation=observation,
            region_id=None,
            element_probabilities={element.id: 1.0},
            region_probabilities={None: 1.0},
            selection_mode="fake",
        )


class ExhaustedAttentionPolicy:
    def next_observation(self, *args: object, **kwargs: object) -> ObservationSelection:
        del args, kwargs
        raise ValueError("no unobserved visible elements remain")


class RepeatingAttentionPolicy:
    def __init__(self) -> None:
        self.recovery_levels: list[int] = []

    def next_observation(
        self,
        state: Any,
        snapshot: ViewportSnapshot,
        scores: tuple[ProminenceResult, ...],
        coarse_scent: object,
        rng: object,
        *,
        recovery_level: int = 0,
    ) -> ObservationSelection:
        del state, scores, coarse_scent, rng
        self.recovery_levels.append(recovery_level)
        element = snapshot.elements[0]
        return ObservationSelection(
            observation=ProgressiveObservation.from_snapshot(
                snapshot, newly_revealed_ids=(element.id,)
            ),
            region_id=None,
            element_probabilities={element.id: 1.0},
            region_probabilities={None: 1.0},
            selection_mode="repeating-fake",
        )


class FakeCognitiveAgent:
    def __init__(self, decisions: tuple[object, ...] = ()) -> None:
        self.decisions = list(decisions)
        self.observations: list[ProgressiveObservation] = []

    async def decide(self, goal: str, observation: ProgressiveObservation) -> object:
        del goal
        self.observations.append(observation)
        decision = self.decisions.pop(0)
        if isinstance(decision, BaseException):
            raise decision
        if hasattr(decision, "__await__"):
            return await decision
        return decision


class ConcurrentModelRecordSource:
    def __init__(self) -> None:
        self._records: list[ModelCallRecord] = []

    @property
    def records(self) -> tuple[ModelCallRecord, ...]:
        return tuple(self._records)

    def append(self, role: ModelRole) -> None:
        self._records.append(
            ModelCallRecord(
                role=role,
                model=f"{role.value}-model",
                endpoint_origin="https://llm.example.test",
                prompt_digest=f"{role.value}-prompt",
                schema_version=f"{role.value}-v1",
                attempts=1,
                latency_ms=1,
                token_usage=TokenUsage(),
                request={},
                response={},
            )
        )


class CoordinatedFullScentEvaluator:
    def __init__(
        self,
        records: ConcurrentModelRecordSource,
        started: asyncio.Event,
        release: asyncio.Event,
    ) -> None:
        self.records = records
        self.started = started
        self.release = release
        self.inputs: tuple[str, AttentionState, ViewportSnapshot] | None = None

    async def evaluate(
        self, goal: str, state: AttentionState, snapshot: ViewportSnapshot
    ) -> tuple[FullScent, ...]:
        self.inputs = (goal, state, snapshot)
        self.started.set()
        await self.release.wait()
        self.records.append(ModelRole.FULL_SCENT)
        return (FullScent.for_element(state, "target", 0.8),)


class ValidationFailingFullScentEvaluator(CoordinatedFullScentEvaluator):
    async def evaluate(
        self, goal: str, state: AttentionState, snapshot: ViewportSnapshot
    ) -> tuple[FullScent, ...]:
        self.inputs = (goal, state, snapshot)
        self.started.set()
        self.records.append(ModelRole.FULL_SCENT)
        raise ModelResponseValidationError(ModelRole.FULL_SCENT, "invalid scent")


class BlockingFullScentEvaluator(CoordinatedFullScentEvaluator):
    def __init__(
        self,
        records: ConcurrentModelRecordSource,
        started: asyncio.Event,
        cancelled: asyncio.Event,
    ) -> None:
        super().__init__(records, started, asyncio.Event())
        self.cancelled = cancelled

    async def evaluate(
        self, goal: str, state: AttentionState, snapshot: ViewportSnapshot
    ) -> tuple[FullScent, ...]:
        self.inputs = (goal, state, snapshot)
        self.started.set()
        try:
            await asyncio.Future()
        finally:
            self.records.append(ModelRole.FULL_SCENT)
            self.cancelled.set()
        return (FullScent.for_element(state, "target", 0.8),)


class CoordinatedCognitiveAgent(FakeCognitiveAgent):
    def __init__(
        self,
        records: ConcurrentModelRecordSource,
        started: asyncio.Event,
        decision: object,
    ) -> None:
        super().__init__((decision,))
        self.records = records
        self.started = started
        self.contexts: list[CognitiveRunContext] = []
        self.decision_inputs: list[tuple[str, ProgressiveObservation]] = []

    def update_context(self, context: CognitiveRunContext) -> None:
        self.contexts.append(context)

    async def decide(self, goal: str, observation: ProgressiveObservation) -> object:
        self.decision_inputs.append((goal, observation))
        self.started.set()
        self.records.append(ModelRole.COGNITIVE)
        return await super().decide(goal, observation)


class BlockingCognitiveAgent(CoordinatedCognitiveAgent):
    def __init__(
        self,
        records: ConcurrentModelRecordSource,
        started: asyncio.Event,
        cancelled: asyncio.Event,
    ) -> None:
        super().__init__(
            records,
            started,
            CognitiveDecision(
                action={"kind": "abandon", "reason": "unreachable"},
                reason="unreachable",
            ),
        )
        self.cancelled = cancelled

    async def decide(self, goal: str, observation: ProgressiveObservation) -> object:
        self.decision_inputs.append((goal, observation))
        self.started.set()
        try:
            await asyncio.Future()
        finally:
            self.records.append(ModelRole.COGNITIVE)
            self.cancelled.set()
        return self.decisions[0]


class FakeVerifier:
    def __init__(self, results: tuple[VerificationResult, ...]) -> None:
        self.results = list(results)
        self.calls = 0

    async def verify(self, session: SessionHandle) -> VerificationResult:
        del session
        self.calls += 1
        if self.results:
            return self.results.pop(0)
        return VerificationResult(verified=False, details="not complete")


class FakeFixtureStateClient:
    def __init__(self, state: dict[str, object]) -> None:
        self.state = state
        self.session_ids: list[str] = []

    async def get_state(self, session_id: str) -> dict[str, object]:
        self.session_ids.append(session_id)
        return self.state


class FakeBundle:
    def __init__(self) -> None:
        self.events: list[object] = []
        self.saliency_contents: dict[str, bytes] = {}
        self.finalized = False
        self.aborted = False
        self.final_result: object | None = None
        self.finalize_error: BaseException | None = None

    def append_event(self, event: object) -> int:
        self.events.append(event)
        return len(self.events)

    def append_saliency_event(self, event: object) -> int:
        self.events.append(event)
        return len(self.events)

    def write_artifact(self, name: str, content: bytes | str) -> ArtifactReference:
        raw = content.encode() if isinstance(content, str) else content
        return ArtifactReference(
            path=f"artifacts/{name}",
            sha256=f"sha-{len(raw)}-{name}",
            size=len(raw),
            name=name,
        )

    def write_saliency_artifact(
        self,
        name: str,
        content: bytes | str,
        kind: SaliencyArtifactKind,
    ) -> ArtifactReference:
        del kind
        raw = content.encode() if isinstance(content, str) else content
        self.saliency_contents[name] = raw
        return ArtifactReference(
            path=name,
            sha256=hashlib.sha256(raw).hexdigest(),
            size=len(raw),
            name=name,
        )

    def verify_artifact(self, reference: ArtifactReference) -> None:
        content = self.saliency_contents.get(reference.path)
        if content is None:
            raise RuntimeError(f"materialized artifact is missing: {reference.path}")
        if len(content) != reference.size:
            raise RuntimeError(f"materialized artifact size mismatch: {reference.path}")
        if hashlib.sha256(content).hexdigest() != reference.sha256:
            raise RuntimeError(
                f"materialized artifact checksum mismatch: {reference.path}"
            )

    def finalize(self, result: object) -> str:
        if self.finalize_error is not None:
            raise self.finalize_error
        self.finalized = True
        self.final_result = result
        return "runs/run-1"

    def abort(self, reason: str) -> str:
        self.aborted = True
        self.final_result = reason
        return "staging/run-1"


class FakeBundleFactory:
    def __init__(self) -> None:
        self.bundle = FakeBundle()

    def start(self, spec: object) -> FakeBundle:
        del spec
        return self.bundle


class FilesystemBundleFactory:
    def __init__(
        self,
        output: Path,
        provider_manifests: tuple[ProviderManifest, ...] = (),
    ) -> None:
        self.output = output
        self.provider_manifests = provider_manifests

    def start(self, spec) -> FilesystemRunBundleWriter:
        return FilesystemRunBundleWriter.start(
            self.output,
            BundleManifest.from_run_spec(
                spec,
                endpoint_origin="https://llm.example.test",
                model_ids={"cognitive": "cognitive-model"},
                prompt_versions={"cognitive": "cognitive-v1"},
                package_version="0.1.0",
                provider_manifests=self.provider_manifests,
            ),
        )


def test_bundle_manifest_preserves_selected_run_semantics() -> None:
    manifest = BundleManifest.from_run_spec(
        _spec(),
        endpoint_origin="https://llm.example.test/v1",
    ).to_dict()

    assert manifest["scenario_id"] == "invite"
    assert manifest["application_version_id"] == "improved"
    assert manifest["persona_id"] == "persona"
    assert manifest["policy"] == "progressive-prominence-scent"
    assert manifest["seed"] == 7


class FakeModelRecordSource:
    def __init__(self) -> None:
        self.records = (
            ModelCallRecord(
                role=ModelRole.COGNITIVE,
                model="cognitive-model",
                endpoint_origin="https://llm.example.test",
                prompt_digest="prompt-digest",
                schema_version="cognitive-v1",
                attempts=2,
                latency_ms=17,
                token_usage=TokenUsage(4, 3, 7),
                request={"messages": [{"content": "[REDACTED]"}]},
                response={"action": {"kind": "abandon"}},
                retries=(
                    RetryEvent(
                        role=ModelRole.COGNITIVE,
                        model="cognitive-model",
                        attempt=1,
                        reason="rate-limit",
                        status_code=429,
                        delay_seconds=0.25,
                    ),
                ),
            ),
        )


def _snapshot(
    viewport_id: str = "viewport-1",
    element_id: str = "target",
    second_element_id: str | None = None,
    *,
    lineage_id: str | None = None,
    second_lineage_id: str | None = None,
    label: str = "Invite teammate",
    region_id: str | None = None,
    region_label: str = "Main content",
) -> ViewportSnapshot:
    elements = [
        ElementSnapshot(
            id=element_id,
            role="button",
            label=label,
            rendered_text=label,
            bounds=BoundingBox(x=10, y=10, width=100, height=30),
            visibility_fraction=1,
            actionable=True,
            region_id=region_id,
            provider_id="fake-observer",
            lineage_id=lineage_id,
            execution_reference=PrivateExecutionReference(
                provider_id="fake-observer",
                viewport_id=viewport_id,
                token=f"token-{element_id}",
            ),
        )
    ]
    if second_element_id is not None:
        elements.append(
            ElementSnapshot(
                id=second_element_id,
                role="button",
                label="Fallback",
                rendered_text="Fallback",
                bounds=BoundingBox(x=130, y=10, width=100, height=30),
                visibility_fraction=1,
                actionable=True,
                region_id=region_id,
                provider_id="fake-observer",
                lineage_id=second_lineage_id,
                execution_reference=PrivateExecutionReference(
                    provider_id="fake-observer",
                    viewport_id=viewport_id,
                    token=f"token-{second_element_id}",
                ),
            )
        )
    return ViewportSnapshot(
        id=viewport_id,
        provider_id="fake-observer",
        elements=tuple(elements),
        regions=(
            (
                RegionSnapshot(
                    id=region_id,
                    label=region_label,
                    element_ids=tuple(element.id for element in elements),
                ),
            )
            if region_id is not None
            else ()
        ),
    )


def _spec(
    *,
    max_steps: int = 6,
    max_model_calls: int = 64,
    timeout_seconds: float | None = 1,
    sensitive_fixture: bool = False,
    evaluation_label: str = "Target",
) -> object:
    version = ApplicationVersion(
        id="improved",
        kind=ApplicationVersionKind.IMPROVED,
        label="Improved",
    )
    scenario = Scenario(
        id="invite",
        name="Invite",
        goal="Invite teammate",
        application_version_ids=(version.id,),
        start_state="dashboard",
        fixture_inputs=FixtureInputs(
            values={"invite_email": "person@example.com"},
            sensitive_keys=frozenset({"invite_email"})
            if sensitive_fixture
            else frozenset(),
        ),
        budget=Budget(
            max_steps=max_steps,
            max_observations=max_steps,
            max_interactions=max_steps,
            max_model_calls=max_model_calls,
            timeout_seconds=timeout_seconds,
        ),
        verifier=VisibleResultVerifierSpec(type="visible-result", text="sent"),
        safeguards=(),
        eligible_persona_ids=("persona",),
        expected_evidence=(),
        evaluation_target=ScenarioEvaluationTarget(
            labels_by_version={"improved": evaluation_label},
            role="button",
        ),
    )
    from ux_analyzer.domain.run import RunSpec

    return RunSpec(
        run_id="run-1",
        seed=7,
        scenario=scenario,
        application_version=version,
        persona=Persona(
            id="persona",
            name="Persona",
            working_memory_capacity=3,
            initial_confidence=0.5,
            initial_frustration=0,
            abandonment_threshold=0.9,
            attention_temperature=1,
        ),
        policy=ExperimentPolicy.PROGRESSIVE_PROMINENCE_SCENT,
        config_digest="config-digest",
    )


def _config(spec: object, tmp_path: Path) -> ObservationSessionConfig:
    return ObservationSessionConfig(
        session_id="run-1",
        start_url="http://fixture.test/app/run-1/improved",
        test_account_id=AccountId("test-run"),
        viewport=ViewportSize(width=1024, height=768),
        trace_path=tmp_path / "trace.zip",
    )


def _session_handle() -> SessionHandle:
    return SessionHandle(
        session_id="run-1",
        test_account_id=AccountId("test-run"),
        viewport=ViewportSize(width=1024, height=768),
        trace_path=Path("trace.zip"),
        blocked_events=[],
    )


def _agent(
    tmp_path: Path,
    provider: FakeObservationProvider,
    cognitive: FakeCognitiveAgent,
    verifier: FakeVerifier,
    bundles: FakeBundleFactory,
    *,
    timeout_seconds: float | None = 1,
    model_record_source: FakeModelRecordSource | None = None,
    result_evaluator=None,
    attention_policy: object | None = None,
    full_scent_evaluator: object | None = None,
    memory_policy: MemoryPolicy | None = None,
    max_model_calls: int = 64,
    prominence_provider: object | None = None,
) -> RunAgent:
    spec = _spec(timeout_seconds=timeout_seconds, max_model_calls=max_model_calls)
    return RunAgent(
        observation_provider=provider,
        prominence_provider=prominence_provider or FakeProminenceProvider(),
        attention_policy=attention_policy or FakeAttentionPolicy(),
        cognitive_agent=cognitive,
        verifier=verifier,
        bundle_factory=bundles,
        session_config_factory=lambda _: _config(spec, tmp_path),
        model_record_source=model_record_source,
        result_evaluator=result_evaluator,
        full_scent_evaluator=full_scent_evaluator,
        memory_policy=memory_policy,
    )


@pytest.mark.asyncio
async def test_saliency_run_uses_capture_stage_and_typed_ordered_events(
    tmp_path: Path,
) -> None:
    saliency = MaterializingFakeSaliencyProminenceProvider()
    bundles = FakeBundleFactory()
    agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(),)),
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Stop."},
                    reason="Stop.",
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
        prominence_provider=saliency,
    )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert result.outcome.kind == "agent-abandoned"
    assert [
        (viewport_id, stage, getattr(artifacts, "_writer", artifacts))
        for viewport_id, stage, artifacts in saliency.calls
    ] == [("viewport-1", SearchStage.INITIAL, bundles.bundle)]
    typed_events = [
        event
        for event in bundles.bundle.events
        if isinstance(
            event,
            (SaliencyProfilesRecordedEvent, ProminenceRecordedEvent),
        )
    ]
    assert [event.kind for event in typed_events] == [
        "saliency-profiles-recorded",
        "prominence-recorded",
    ]
    prominence = typed_events[-1]
    assert isinstance(prominence, ProminenceRecordedEvent)
    prominence_evidence = result.evidence.prominence[0]
    assert prominence_evidence.profile_event_id is not None
    assert prominence_evidence.operational_event_id is not None
    assert prominence_evidence.profile_event_id != prominence_evidence.operational_event_id
    payload = prominence.to_dict()
    assert "scores" not in payload
    assert "raw_map" not in payload
    assert "normalized_probability" not in payload


@pytest.mark.asyncio
async def test_saliency_stage_changes_after_observation_without_new_capture(
    tmp_path: Path,
) -> None:
    saliency = FakeSaliencyProminenceProvider()
    agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(second_element_id="second"),)),
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "inspect", "element_id": "target"},
                    reason="Inspect target.",
                ),
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Stop."}, reason="Stop."
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        FakeBundleFactory(),
        prominence_provider=saliency,
    )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert result.outcome.kind == "agent-abandoned"
    assert [stage for _, stage, _ in saliency.calls] == [
        SearchStage.INITIAL,
        SearchStage.EXPLORATION,
    ]


@pytest.mark.asyncio
async def test_exact_same_capture_reuses_saliency_artifact_paths(
    tmp_path: Path,
) -> None:
    saliency = MaterializingFakeSaliencyProminenceProvider()
    bundles = FakeBundleFactory()
    agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(second_element_id="second"),)),
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "inspect", "element_id": "target"},
                    reason="Inspect target.",
                ),
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Stop."},
                    reason="Stop.",
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
        prominence_provider=saliency,
    )

    await agent.execute(_spec(timeout_seconds=None))

    profile_events = [
        event
        for event in bundles.bundle.events
        if isinstance(event, SaliencyProfilesRecordedEvent)
    ]
    assert len(profile_events) == 2
    assert profile_events[0].artifact_ids == profile_events[1].artifact_ids
    assert "native-inference" not in profile_events[0].viewport_id


@pytest.mark.asyncio
async def test_distinct_captures_get_distinct_saliency_artifact_paths(
    tmp_path: Path,
) -> None:
    saliency = MaterializingFakeSaliencyProminenceProvider()
    bundles = FakeBundleFactory()
    agent = _agent(
        tmp_path,
        FakeObservationProvider(
            (
                _snapshot("viewport-1", second_element_id="second"),
                _snapshot("viewport-2", second_element_id="second"),
            ),
            screenshots=(b"first-capture", b"second-capture"),
        ),
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "scroll", "direction": "down"},
                    reason="Scroll.",
                ),
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Stop."},
                    reason="Stop.",
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
        prominence_provider=saliency,
    )

    await agent.execute(_spec(timeout_seconds=None))

    profile_events = [
        event
        for event in bundles.bundle.events
        if isinstance(event, SaliencyProfilesRecordedEvent)
    ]
    assert len(profile_events) == 2
    assert set(profile_events[0].artifact_ids).isdisjoint(
        profile_events[1].artifact_ids
    )
    assert profile_events[0].viewport_id != profile_events[1].viewport_id


@pytest.mark.asyncio
async def test_learned_batch_without_artifacts_is_recorded_invalid_not_fabricated(
    tmp_path: Path,
) -> None:
    bundles = FakeBundleFactory()
    agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(),)),
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Stop."},
                    reason="Stop.",
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
        prominence_provider=NoArtifactSaliencyProminenceProvider(),
    )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert result.ux_sample_valid is False
    assert result.ux_sample_invalid_reason == (
        "saliency-fallback: learned saliency artifact evidence unavailable"
    )
    assert not any(
        isinstance(event, SaliencyProfilesRecordedEvent)
        for event in bundles.bundle.events
    )
    fallback = next(
        event
        for event in bundles.bundle.events
        if isinstance(event, SaliencyFallbackRecordedEvent)
    )
    assert fallback.reason == "learned saliency artifact evidence unavailable"


@pytest.mark.asyncio
async def test_saliency_profile_event_uses_materialized_artifact_references(
    tmp_path: Path,
) -> None:
    bundles = FakeBundleFactory()
    agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(),)),
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Stop."},
                    reason="Stop.",
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
        prominence_provider=MaterializingFakeSaliencyProminenceProvider(),
    )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert result.outcome.kind == "agent-abandoned"
    profile_event = next(
        event
        for event in bundles.bundle.events
        if isinstance(event, SaliencyProfilesRecordedEvent)
    )
    assert profile_event.viewport_id.startswith("inference-")
    assert profile_event.viewport_id != "native-inference"
    assert profile_event.artifact_ids == tuple(
        sorted(
            f"saliency/{profile_event.viewport_id}/{filename}"
            for filename in (
                "1s.npz",
                "3s.npz",
                "7s.npz",
                "1s-heatmap.png",
                "3s-heatmap.png",
                "7s-heatmap.png",
                "profiles.json",
                "metadata.json",
            )
        )
    )


@pytest.mark.asyncio
async def test_saliency_persistent_stage_uses_recovery_threshold_on_same_viewport(
    tmp_path: Path,
) -> None:
    saliency = FakeSaliencyProminenceProvider()
    agent = _agent(
        tmp_path,
        FakeObservationProvider(
            (
                _snapshot("viewport-1", second_element_id="second"),
                _snapshot("viewport-2", second_element_id="second"),
                _snapshot("viewport-3", second_element_id="second"),
            )
        ),
        FakeCognitiveAgent(
            (
                CognitiveDecision(action={"kind": "wait"}, reason="Wait."),
                CognitiveDecision(action={"kind": "wait"}, reason="Wait."),
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Stop."}, reason="Stop."
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        FakeBundleFactory(),
        attention_policy=RepeatingAttentionPolicy(),
        prominence_provider=saliency,
    )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert result.outcome.kind == "agent-abandoned"
    assert [stage for _, stage, _ in saliency.calls] == [
        SearchStage.INITIAL,
        SearchStage.EXPLORATION,
        SearchStage.PERSISTENT,
    ]


@pytest.mark.asyncio
async def test_custom_stage_mixture_weights_are_recorded_for_each_search_stage(
    tmp_path: Path,
) -> None:
    mixtures = {
        "initial": {"7s": 1.0},
        "exploration": {"1s": 0.2, "3s": 0.3, "7s": 0.5},
        "persistent": {"1s": 0.1, "7s": 0.9},
    }
    saliency = FakeSaliencyProminenceProvider()
    saliency.stage_selector = SimpleNamespace(mixtures=mixtures)
    bundles = FakeBundleFactory()
    agent = _agent(
        tmp_path,
        FakeObservationProvider(
            (
                _snapshot("viewport-1", second_element_id="second"),
                _snapshot("viewport-2", second_element_id="second"),
                _snapshot("viewport-3", second_element_id="second"),
            )
        ),
        FakeCognitiveAgent(
            (
                CognitiveDecision(action={"kind": "wait"}, reason="Wait."),
                CognitiveDecision(action={"kind": "wait"}, reason="Wait."),
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Stop."},
                    reason="Stop.",
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
        attention_policy=RepeatingAttentionPolicy(),
        prominence_provider=saliency,
    )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert result.outcome.kind == "agent-abandoned"
    events = [
        event
        for event in bundles.bundle.events
        if isinstance(event, ProminenceRecordedEvent)
    ]
    assert [event.search_stage for event in events] == [
        "initial",
        "exploration",
        "persistent",
    ]
    assert [dict(event.selected_mixture) for event in events] == [
        mixtures["initial"],
        mixtures["exploration"],
        mixtures["persistent"],
    ]


@pytest.mark.asyncio
async def test_scroll_recapture_resets_saliency_stage_to_initial(
    tmp_path: Path,
) -> None:
    saliency = FakeSaliencyProminenceProvider()
    agent = _agent(
        tmp_path,
        FakeObservationProvider(
            (
                _snapshot("viewport-1"),
                _snapshot("viewport-2"),
            )
        ),
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "scroll", "direction": "down"},
                    reason="Scroll.",
                ),
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Stop."}, reason="Stop."
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        FakeBundleFactory(),
        attention_policy=RepeatingAttentionPolicy(),
        prominence_provider=saliency,
    )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert result.outcome.kind == "agent-abandoned"
    assert [stage for _, stage, _ in saliency.calls] == [
        SearchStage.INITIAL,
        SearchStage.INITIAL,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "platform_action",
    (OpenMenuAction(element_id="target"), ToggleAction(element_id="target")),
    ids=("open-menu", "toggle"),
)
async def test_modal_like_recapture_resets_stage_for_unchanged_semantic_snapshot(
    tmp_path: Path,
    platform_action: object,
) -> None:
    saliency = MaterializingFakeSaliencyProminenceProvider()
    agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot("viewport-1"), _snapshot("viewport-2"))),
        FakeCognitiveAgent(
            (
                ClaimingDecision(action=platform_action, reason="Open modal."),
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Stop."}, reason="Stop."
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        FakeBundleFactory(),
        attention_policy=RepeatingAttentionPolicy(),
        prominence_provider=saliency,
    )

    result = await agent.execute(_spec(timeout_seconds=None, max_steps=10))

    assert result.outcome.kind == "agent-abandoned"
    assert [stage for _, stage, _ in saliency.calls] == [
        SearchStage.INITIAL,
        SearchStage.INITIAL,
    ]


@pytest.mark.asyncio
async def test_click_recapture_resets_stage_for_unchanged_semantic_snapshot(
    tmp_path: Path,
) -> None:
    saliency = FakeSaliencyProminenceProvider()
    agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot("viewport-1"), _snapshot("viewport-2"))),
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "interact", "element_id": "target"},
                    reason="Click target.",
                ),
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Stop."}, reason="Stop."
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        FakeBundleFactory(),
        attention_policy=RepeatingAttentionPolicy(),
        prominence_provider=saliency,
    )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert result.outcome.kind == "agent-abandoned"
    assert [stage for _, stage, _ in saliency.calls] == [
        SearchStage.INITIAL,
        SearchStage.INITIAL,
    ]


@pytest.mark.asyncio
async def test_configured_recovery_threshold_reaches_persistent_stage_at_threshold(
    tmp_path: Path,
) -> None:
    from ux_analyzer.cli import _AttentionPolicyAdapter
    from ux_analyzer.providers.attention_policy import AttentionPolicyConfig

    saliency = FakeSaliencyProminenceProvider()
    attention = RepeatingAttentionPolicy()
    policy = _AttentionPolicyAdapter(
        attention,
        config=AttentionPolicyConfig(recovery_after_misses=1),
    )
    agent = _agent(
        tmp_path,
        FakeObservationProvider(
            (_snapshot("viewport-1"), _snapshot("viewport-2")),
            results=(
                PlatformActionResult(
                    True,
                    "http://fixture.test",
                    1,
                    state_changed=False,
                ),
            ),
        ),
        FakeCognitiveAgent(
            (
                CognitiveDecision(action={"kind": "wait"}, reason="Wait."),
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Stop."}, reason="Stop."
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        FakeBundleFactory(),
        attention_policy=policy,
        prominence_provider=saliency,
    )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert result.outcome.kind == "agent-abandoned"
    assert [stage for _, stage, _ in saliency.calls] == [
        SearchStage.INITIAL,
        SearchStage.PERSISTENT,
    ]


@pytest.mark.asyncio
async def test_capture_aware_saliency_timeline_finalizes_in_filesystem_bundle(
    tmp_path: Path,
) -> None:
    agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(),)),
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Stop."},
                    reason="Stop.",
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        FilesystemBundleFactory(tmp_path),
        prominence_provider=FakeSaliencyProminenceProvider(),
    )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert isinstance(result.bundle_path, Path)
    timeline = [
        json.loads(line)
        for line in (result.bundle_path / "timeline.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    typed_kinds = [
        event["kind"]
        for event in timeline
        if event["kind"]
        in {
            "saliency-fallback-recorded",
            "saliency-profiles-recorded",
            "prominence-recorded",
        }
    ]
    assert typed_kinds == ["saliency-fallback-recorded", "prominence-recorded"]
    assert all(
        "scores" not in event
        for event in timeline
        if event["kind"] == "prominence-recorded"
    )


@pytest.mark.asyncio
async def test_filesystem_result_projects_public_evidence_and_private_snapshot_fields(
    tmp_path: Path,
) -> None:
    private_element = replace(
        _snapshot().elements[0],
        selector="[data-private-result]",
        test_id="private-test-id",
        hidden_label="private-hidden-label",
        destination_url="https://private.example/hidden",
    )
    snapshot = replace(_snapshot(), elements=(private_element,))
    saliency = FoveacastProminenceProvider(
        model_provider=ValidFakeSaliencyModel(),
        cache=SaliencyCache(tmp_path / "saliency-cache"),
    )
    agent = _agent(
        tmp_path,
        FakeObservationProvider((snapshot,)),
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Stop."},
                    reason="Stop.",
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        FilesystemBundleFactory(
            tmp_path / "bundles", provider_manifests=(_saliency_manifest(),)
        ),
        prominence_provider=saliency,
    )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert isinstance(result.bundle_path, Path)
    persisted = json.loads(
        (result.bundle_path / "result.json").read_text(encoding="utf-8")
    )
    serialized = json.dumps(persisted, sort_keys=True)
    for private_value in (
        "token-target",
        "[data-private-result]",
        "private-test-id",
        "private-hidden-label",
        "https://private.example/hidden",
    ):
        assert private_value not in serialized
    for private_key in (
        "execution_reference",
        "selector",
        "test_id",
        "hidden_label",
        "destination_url",
        "token",
    ):
        assert f'"{private_key}"' not in serialized
    for saliency_key in (
        "raw_score",
        "normalized_probability",
        "feature_contributions",
        "raw_values",
        "normalized_values",
    ):
        assert f'"{saliency_key}"' not in serialized

    assert persisted["state"]["snapshots"][0]["provider_id"] == "fake-observer"
    prominence = persisted["evidence"]["prominence"][0]
    assert prominence["viewport_id"] == "viewport-1"
    assert prominence["source_event_id"].startswith("event-")
    assert prominence["profile_event_id"].startswith("event-")
    assert prominence["operational_event_id"] == prominence["source_event_id"]
    selection = persisted["evidence"]["selections"][0]
    assert selection["source_event_id"].startswith("event-")
    assert "element_probabilities" not in selection
    assert "region_probabilities" not in selection
    decision = persisted["evidence"]["decisions"][0]
    assert decision["source_event_id"].startswith("event-")
    assert "decision" not in decision


def _saliency_manifest() -> ProviderManifest:
    return ProviderManifest(
        provider_id="foveacast",
        role="prominence",
        model_id="foveacast-v0.2.0",
        endpoint_origin="internal",
        version="v0.2.0",
    )


@pytest.mark.asyncio
async def test_real_cache_miss_and_hit_materialize_learned_filesystem_evidence(
    tmp_path: Path,
) -> None:
    model = ValidFakeSaliencyModel()
    saliency = FoveacastProminenceProvider(
        model_provider=model,
        cache=SaliencyCache(tmp_path / "experiment"),
    )
    agent = _agent(
        tmp_path,
        FakeObservationProvider(
            (
                _snapshot("viewport-1", "target-1", lineage_id="target-lineage"),
                _snapshot("viewport-2", "target-2", lineage_id="target-lineage"),
            ),
            screenshots=(b"same-screenshot", b"same-screenshot"),
        ),
        FakeCognitiveAgent(
            (
                CognitiveDecision(action={"kind": "wait"}, reason="Wait."),
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Stop."},
                    reason="Stop.",
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        FilesystemBundleFactory(
            tmp_path / "bundles", provider_manifests=(_saliency_manifest(),)
        ),
        attention_policy=RepeatingAttentionPolicy(),
        prominence_provider=saliency,
    )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert result.ux_sample_valid is True
    assert model.calls == 1
    assert isinstance(result.bundle_path, Path)
    timeline = [
        json.loads(line)
        for line in (result.bundle_path / "timeline.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    saliency_events = [
        event
        for event in timeline
        if event["kind"]
        in {
            "saliency-inference-recorded",
            "saliency-cache-hit",
            "saliency-profiles-recorded",
            "prominence-recorded",
        }
    ]
    assert [event["kind"] for event in saliency_events] == [
        "saliency-inference-recorded",
        "saliency-profiles-recorded",
        "prominence-recorded",
        "saliency-cache-hit",
        "saliency-profiles-recorded",
        "prominence-recorded",
    ]
    profiles = [
        event
        for event in saliency_events
        if event["kind"] == "saliency-profiles-recorded"
    ]
    prominence = [
        event for event in saliency_events if event["kind"] == "prominence-recorded"
    ]
    assert [event["source_viewport_id"] for event in profiles] == [
        "viewport-1",
        "viewport-2",
    ]
    assert [event["source_viewport_id"] for event in prominence] == [
        "viewport-1",
        "viewport-2",
    ]
    assert [event["cache_state"] for event in prominence] == ["miss", "hit"]
    assert all(event["selected_element_ids"] for event in prominence)
    assert [event["source_event_id"] for event in prominence] == [
        f"event-{profiles[0]['sequence']}",
        f"event-{profiles[1]['sequence']}",
    ]
    assert [event["artifact_namespace"] for event in prominence] == [
        profiles[0]["artifact_namespace"],
        profiles[1]["artifact_namespace"],
    ]
    for profile in profiles:
        for artifact_id in profile["artifact_ids"]:
            artifact = result.bundle_path / artifact_id
            assert artifact.is_file()
            assert artifact.read_bytes()


@pytest.mark.asyncio
async def test_cache_disabled_inference_materializes_explicit_miss_evidence(
    tmp_path: Path,
) -> None:
    model = ValidFakeSaliencyModel()
    saliency = FoveacastProminenceProvider(
        model_provider=model,
        cache_enabled=False,
    )
    agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(),), screenshot=b"direct-inference"),
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Stop."},
                    reason="Stop.",
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        FilesystemBundleFactory(
            tmp_path / "bundles", provider_manifests=(_saliency_manifest(),)
        ),
        prominence_provider=saliency,
    )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert result.ux_sample_valid is True
    assert model.calls == 1
    assert isinstance(result.bundle_path, Path)
    timeline = [
        json.loads(line)
        for line in (result.bundle_path / "timeline.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    inference = next(
        event for event in timeline if event["kind"] == "saliency-inference-recorded"
    )
    profile = next(
        event for event in timeline if event["kind"] == "saliency-profiles-recorded"
    )
    assert inference["cache_state"] == "disabled"
    assert profile["cache_state"] == "disabled"
    assert len(profile["artifact_ids"]) == 8
    assert not any(event["kind"] == "saliency-fallback-recorded" for event in timeline)


@pytest.mark.asyncio
async def test_capture_scoped_ids_preserve_semantic_exploration_and_persistent_stage(
    tmp_path: Path,
) -> None:
    saliency = MaterializingFakeSaliencyProminenceProvider()
    agent = _agent(
        tmp_path,
        FakeObservationProvider(
            (
                _snapshot("viewport-1", "target-1", region_id="region-1"),
                _snapshot("viewport-2", "target-2", region_id="region-2"),
                _snapshot("viewport-3", "target-3", region_id="region-3"),
            )
        ),
        FakeCognitiveAgent(
            (
                CognitiveDecision(action={"kind": "wait"}, reason="Wait."),
                CognitiveDecision(action={"kind": "wait"}, reason="Wait."),
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Stop."},
                    reason="Stop.",
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        FakeBundleFactory(),
        attention_policy=RepeatingAttentionPolicy(),
        prominence_provider=saliency,
    )

    await agent.execute(_spec(timeout_seconds=None))

    assert [stage for _, stage, _ in saliency.calls] == [
        SearchStage.INITIAL,
        SearchStage.EXPLORATION,
        SearchStage.PERSISTENT,
    ]


@pytest.mark.asyncio
async def test_callable_capture_aware_score_provider_is_supported(
    tmp_path: Path,
) -> None:
    delegate = MaterializingFakeSaliencyProminenceProvider()

    class CallableScore:
        def __call__(
            self,
            capture: ObservationCapture,
            snapshot: ViewportSnapshot,
            stage: SearchStage,
            artifacts: object,
        ) -> ProminenceBatch:
            return delegate.score(capture, snapshot, stage, artifacts)

    class CallableProvider:
        id = delegate.id
        version = delegate.version
        model_checksums = delegate.model_checksums
        actual_execution_provider = delegate.actual_execution_provider
        preprocessing_version = delegate.preprocessing_version
        precision = delegate.precision
        cache_key = delegate.cache_key
        score = CallableScore()

    agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(),)),
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Stop."},
                    reason="Stop.",
            ),
        )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        FakeBundleFactory(),
        prominence_provider=CallableProvider(),
    )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert result.outcome.kind == "agent-abandoned"
    assert delegate.calls[0][0] == "viewport-1"


@pytest.mark.asyncio
async def test_in_memory_writer_detects_tampered_saliency_reference(
    tmp_path: Path,
) -> None:
    class TamperingBundle(FakeBundle):
        def write_saliency_artifact(
            self,
            name: str,
            content: bytes | str,
            kind: SaliencyArtifactKind,
        ) -> ArtifactReference:
            reference = super().write_saliency_artifact(name, content, kind)
            if name.endswith("profiles.json"):
                self.saliency_contents[name] += b"tampered"
            return reference

    class TamperingFactory(FakeBundleFactory):
        def __init__(self) -> None:
            self.bundle = TamperingBundle()

    agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(),)),
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Stop."},
                    reason="Stop.",
                ),
            )
            ),
        FakeVerifier((VerificationResult(verified=False),)),
        TamperingFactory(),
        prominence_provider=MaterializingFakeSaliencyProminenceProvider(),
        )

    with pytest.raises(RuntimeError, match="artifact size mismatch"):
        await agent.execute(_spec(timeout_seconds=None))


@pytest.mark.asyncio
async def test_saliency_fallback_records_typed_event_and_invalidates_sample(
    tmp_path: Path,
) -> None:
    saliency = FakeSaliencyProminenceProvider(
        fallback_reason="model runtime unavailable; selector=[data-secret]"
    )
    bundles = FakeBundleFactory()
    agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(),)),
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Stop."},
                    reason="Stop.",
            ),
        )
                ),
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
        prominence_provider=saliency,
            )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert result.ux_sample_valid is False
    assert result.ux_sample_invalid_reason == (
        "saliency-fallback: model runtime unavailable; selector=[REDACTED]"
        )
    typed_events = [
        event
        for event in bundles.bundle.events
        if isinstance(
            event,
            (SaliencyFallbackRecordedEvent, ProminenceRecordedEvent),
    )
    ]
    assert [event.kind for event in typed_events] == [
        "saliency-fallback-recorded",
        "prominence-recorded",
    ]
    fallback = typed_events[0]
    assert isinstance(fallback, SaliencyFallbackRecordedEvent)
    assert "[data-secret]" not in json.dumps(fallback.to_dict())


@pytest.mark.asyncio
async def test_saliency_fallback_redacts_exact_fixture_secret_from_result_and_report(
    tmp_path: Path,
) -> None:
    secret = "person@example.com"
    saliency = FakeSaliencyProminenceProvider(
        fallback_reason=f"model runtime unavailable; {secret}"
    )
    bundles = FakeBundleFactory()
    spec = _spec(timeout_seconds=None, sensitive_fixture=True)
    agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(),)),
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Stop."},
                    reason="Stop.",
        ),
    )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
        prominence_provider=saliency,
    )

    result = await agent.execute(spec)

    assert secret not in (result.ux_sample_invalid_reason or "")
    fallback = next(
        event
        for event in bundles.bundle.events
        if isinstance(event, SaliencyFallbackRecordedEvent)
    )
    assert secret not in json.dumps(fallback.to_dict())

    from ux_analyzer.cli import _complete_experiment

    summary_path, report_path = _complete_experiment(
        ExperimentResult(
            specs=(spec,),
            results=(result,),
            failures=(
                ExperimentFailure(
                    run_id=spec.run_id,
                    error_type="RunFailure",
                    message=f"fallback {secret}",
                    spec=spec,
                ),
            ),
        ),
        output=tmp_path / "report-output",
        runtime=None,
    )
    assert secret not in summary_path.read_text(encoding="utf-8")
    assert secret not in report_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_exhausted_visible_attention_finalizes_as_agent_abandoned(
    tmp_path: Path,
) -> None:
    bundles = FakeBundleFactory()
    agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(),)),
        FakeCognitiveAgent(()),
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
        attention_policy=ExhaustedAttentionPolicy(),
    )

    result = await agent.execute(_spec())

    assert result.outcome.kind == "agent-abandoned"
    assert result.terminal_reason == "all visible elements examined without progress"
    assert result.verification.verified is False
    assert bundles.bundle.finalized
    assert any(
        isinstance(event, dict) and event.get("kind") == "attention-exhausted"
        for event in bundles.bundle.events
    )


@pytest.mark.asyncio
async def test_verified_success_records_claim_separately_and_finalizes_after_terminal_event(
    tmp_path: Path,
) -> None:
    provider = FakeObservationProvider(
        (_snapshot(), _snapshot("viewport-2", second_element_id="result"))
    )
    verifier = FakeVerifier((VerificationResult(verified=True, evidence_ids=("v1",)),))
    bundles = FakeBundleFactory()
    agent = _agent(
        tmp_path,
        provider,
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "interact", "element_id": "target"},
                    reason="Target matches goal.",
                ),
                CognitiveDecision(
                    action={"kind": "complete"},
                    reason="I saw enough visible evidence.",
                ),
            )
        ),
        verifier,
        bundles,
        result_evaluator=lambda candidate: replace(
            candidate,
            metrics=evaluate_run(candidate, EvaluationTarget("target")),
        ),
    )

    result = await agent.execute(_spec())

    assert result.outcome.kind == "verified-success"
    assert result.verification.verified
    assert result.agent_claimed_success is True
    assert result.metrics is not None
    assert result.metrics.claimed_completion is True
    assert result.metrics.false_success is False
    assert result.state.attention.confidence == pytest.approx(0.6)
    assert bundles.bundle.finalized
    assert bundles.bundle.events[-1].kind == "run-terminated"
    viewport_event = next(
        event
        for event in bundles.bundle.events
        if getattr(event, "kind", None) == "viewport-captured"
    )
    assert viewport_event.viewport_width == 1024
    assert viewport_event.viewport_height == 768
    assert provider.ended == 1
    assert any(
        isinstance(event, dict)
        and event.get("kind") == "decision-recorded"
        and event.get("claimed_success") is True
        for event in bundles.bundle.events
    )
    assert any(
        isinstance(event, dict)
        and event.get("kind") == "agent-claim"
        and event.get("claimed_success") is True
        for event in bundles.bundle.events
    )


@pytest.mark.asyncio
async def test_visible_result_requires_explicit_complete_after_successful_interaction(
    tmp_path: Path,
) -> None:
    initial = _snapshot()
    result_snapshot = _snapshot("viewport-2", second_element_id="result")
    result_snapshot = replace(
        result_snapshot,
        elements=tuple(
            replace(element, label="Sent", rendered_text="Sent")
            if element.id == "result"
            else element
            for element in result_snapshot.elements
        ),
    )
    provider = FakeObservationProvider((initial, result_snapshot))
    verifier = FakeVerifier((VerificationResult(verified=True, evidence_ids=("v1",)),))
    bundles = FakeBundleFactory()
    cognitive = FakeCognitiveAgent(
        (
            CognitiveDecision(
                action={"kind": "interact", "element_id": "target"},
                reason="Visible control matches goal.",
            ),
            CognitiveDecision(
                action={"kind": "complete"},
                reason="I saw enough visible evidence.",
            ),
        )
    )
    agent = _agent(
        tmp_path,
        provider,
        cognitive,
        verifier,
        bundles,
    )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert result.outcome.kind == "verified-success"
    assert verifier.calls == 1
    assert [action.kind for action in provider.executed] == ["click"]
    assert cognitive.observations[1].newly_revealed_elements[0].label == "Sent"
    assert any(
        getattr(event, "action", None).__class__.__name__ == "Complete"
        for event in bundles.bundle.events
        if getattr(event, "kind", None) == "action-proposed"
    )
    assert any(
        getattr(event, "action", None).__class__.__name__ == "Complete"
        for event in bundles.bundle.events
        if getattr(event, "kind", None) == "action-executed"
    )


@pytest.mark.asyncio
async def test_visible_result_terminal_failure_does_not_promote_without_complete(
    tmp_path: Path,
) -> None:
    provider = FakeObservationProvider((_snapshot(),))
    verifier = FakeVerifier((VerificationResult(verified=True),))
    bundles = FakeBundleFactory()
    agent = _agent(
        tmp_path,
        provider,
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "No further search."},
                    reason="No further search.",
                ),
            )
        ),
        verifier,
        bundles,
    )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert result.outcome.kind == "agent-abandoned"
    assert result.verification.verified is False
    assert result.verification.details == "independent verification unavailable"
    assert verifier.calls == 0


@pytest.mark.asyncio
async def test_fixture_state_retains_after_action_auto_verification(
    tmp_path: Path,
) -> None:
    base_spec = _spec(timeout_seconds=None)
    spec = replace(
        base_spec,
        scenario=replace(
            base_spec.scenario,
            verifier=FixtureStateVerifierSpec(
                type="fixture-state",
                resource="workspace",
                field="invite_status",
                operator="equals",
                expected_fixture_key="invite_email",
            ),
        ),
    )
    verifier = FakeVerifier((VerificationResult(verified=True),))
    agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(),)),
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "interact", "element_id": "target"},
                    reason="Target matches goal.",
                ),
            )
        ),
        verifier,
        FakeBundleFactory(),
    )

    result = await agent.execute(spec)

    assert result.outcome.kind == "verified-success"
    assert verifier.calls == 1


@pytest.mark.asyncio
async def test_claimed_but_unverified_success_becomes_agent_abandonment(
    tmp_path: Path,
) -> None:
    provider = FakeObservationProvider((_snapshot(),))
    verifier = FakeVerifier((VerificationResult(verified=False),))
    bundles = FakeBundleFactory()
    agent = _agent(
        tmp_path,
        provider,
        FakeCognitiveAgent(
            (
                ClaimingDecision(
                    action=Abandon(reason="Claimed complete."),
                    reason="Claimed complete.",
                    claimed_success=True,
                ),
            )
        ),
        verifier,
        bundles,
    )

    result = await agent.execute(_spec())

    assert result.outcome.kind == "agent-abandoned", result.terminal_reason
    assert result.agent_claimed_success is True
    assert result.verification.verified is False
    assert provider.ended == 1


@pytest.mark.asyncio
async def test_unverified_explicit_complete_remains_claimed_false_success(
    tmp_path: Path,
) -> None:
    provider = FakeObservationProvider((_snapshot(),))
    verifier = FakeVerifier((VerificationResult(verified=False),))
    agent = _agent(
        tmp_path,
        provider,
        FakeCognitiveAgent(
            (CognitiveDecision(action={"kind": "complete"}, reason="Complete."),)
        ),
        verifier,
        FakeBundleFactory(),
        result_evaluator=lambda candidate: replace(
            candidate,
            metrics=evaluate_run(candidate, EvaluationTarget("target")),
        ),
    )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert result.outcome.kind == "agent-abandoned"
    assert result.verification.verified is False
    assert result.agent_claimed_success is True
    assert result.metrics is not None
    assert result.metrics.claimed_completion is True
    assert result.metrics.false_success is True


@pytest.mark.asyncio
async def test_negative_verification_qualifies_agent_claim_in_result_and_timeline(
    tmp_path: Path,
) -> None:
    """A self-claim must never outrank a negative verification.

    The agent completes, the independent verifier cannot confirm any
    persona-visible result, and the run then keeps searching until the budget
    is gone. The recorded result must not read like a success: the claim is
    preserved for audit, but the outcome cannot be ``verified-success`` and the
    contradiction must be surfaced explicitly in the payload and an event.
    """

    provider = FakeObservationProvider(
        (_snapshot(), _snapshot("viewport-2"), _snapshot("viewport-3"))
    )
    verifier = FakeVerifier((VerificationResult(verified=False),))
    bundles = FakeBundleFactory()
    agent = _agent(
        tmp_path,
        provider,
        FakeCognitiveAgent(
            tuple(
                [
                    CognitiveDecision(
                        action={"kind": "complete"}, reason="I believe it worked."
                    ),
                    *(
                        CognitiveDecision(
                            action={"kind": "inspect", "element_id": "target"},
                            reason="Keep looking.",
                        )
                        for _ in range(20)
                    ),
                ]
            )
        ),
        verifier,
        bundles,
        result_evaluator=lambda candidate: replace(
            candidate,
            metrics=evaluate_run(candidate, EvaluationTarget("target")),
        ),
    )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert result.verification.verified is False
    # The claim is preserved as evidence...
    assert result.agent_claimed_success is True
    # ...but it must not stand unqualified, and the outcome is never success.
    assert getattr(result, "agent_claim_contradicted", False) is True
    assert result.agent_claimed_success_qualified is False
    assert result.outcome.kind != "verified-success"
    # The contradiction is visible in the bundle timeline.
    assert any(
        isinstance(event, dict)
        and event.get("kind") == "agent-claim-contradicted"
        and event.get("verified") is False
        for event in bundles.bundle.events
    )
    assert bundles.bundle.final_result is not None
    persisted = bundles.bundle.final_result
    assert persisted.agent_claim_contradicted is True


@pytest.mark.asyncio
async def test_repeated_verification_failure_terminates_with_distinct_outcome(
    tmp_path: Path,
) -> None:
    """Failed verification must bound the subsequent search.

    Once verification has failed on the evaluation target, only a small
    bounded number of further attempts are allowed, and the run stops with a
    terminal state that is distinguishable from ordinary budget exhaustion so
    a reader can tell "we tried and verification kept failing" apart from "we
    ran out of budget while still working".
    """

    attempts = 40
    provider = FakeObservationProvider(
        tuple(_snapshot(f"viewport-{index}") for index in range(attempts + 2))
    )
    verifier = FakeVerifier((VerificationResult(verified=False),))
    bundles = FakeBundleFactory()
    # The agent keeps re-proposing completion; each attempt triggers an
    # independent verification that cannot confirm a persona-visible result.
    # Without a bound the run would consume the whole attention budget.
    drift: list[object] = [
        CognitiveDecision(
            action={"kind": "complete"}, reason="I believe it worked."
        )
        for _ in range(attempts)
    ]
    agent = _agent(
        tmp_path,
        provider,
        FakeCognitiveAgent(tuple(drift)),
        verifier,
        bundles,
        attention_policy=RepeatingAttentionPolicy(),
    )

    result = await agent.execute(_spec(max_steps=attempts + 2, timeout_seconds=None))

    assert result.outcome.kind == "verification-failed"
    assert result.outcome.kind != "budget-exhausted"
    assert result.terminal_reason is not None
    assert "verification" in result.terminal_reason.casefold()
    assert "budget" not in result.terminal_reason.casefold()
    assert any(
        isinstance(event, dict)
        and event.get("kind") == "verification-failure-exhausted"
        for event in bundles.bundle.events
    )
    # We stopped early rather than consuming the whole attention budget.
    assert verifier.calls < attempts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "outcome"),
    [
        (ObservationProviderError("provider"), "provider-failure"),
        (SafetyBlocked("blocked"), "safety-blocked"),
        (RuntimeError("model failure"), "internal-error"),
    ],
)
async def test_terminal_failures_finalize_and_cleanup(
    tmp_path: Path,
    failure: BaseException,
    outcome: str,
) -> None:
    provider = FakeObservationProvider((_snapshot(),), failure=failure)
    bundles = FakeBundleFactory()
    agent = _agent(
        tmp_path,
        provider,
        FakeCognitiveAgent(()),
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
    )

    result = await agent.execute(_spec())

    assert result.outcome.kind == outcome
    assert result.ux_sample_valid is False
    assert result.ux_sample_invalid_reason == f"{outcome}: {result.terminal_reason}"
    assert bundles.bundle.finalized
    assert bundles.bundle.events[-1].kind == "run-terminated"
    assert provider.ended == 0 or provider.started == 1


@pytest.mark.asyncio
async def test_terminal_verification_failure_preserves_primary_execution_failure(
    tmp_path: Path,
) -> None:
    class SessionNotActiveVerifier(FakeVerifier):
        async def verify(self, session: SessionHandle) -> VerificationResult:
            del session
            self.calls += 1
            raise ObservationProviderError("session not active")

    bundles = FakeBundleFactory()
    verifier = SessionNotActiveVerifier(())
    agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(),), failure=SafetyBlocked("blocked")),
        FakeCognitiveAgent(()),
        verifier,
        bundles,
    )

    result = await agent.execute(_spec())

    assert result.outcome.kind == "safety-blocked"
    assert result.terminal_reason == "blocked"
    assert result.verification == VerificationResult(
        verified=False,
        details="independent verification unavailable",
    )
    terminal = bundles.bundle.events[-1]
    assert terminal.kind == "run-terminated"
    assert terminal.outcome.kind == "safety-blocked"
    assert terminal.verification.details == "independent verification unavailable"


@pytest.mark.asyncio
async def test_model_failure_timeout_and_budget_exhaustion_are_closed_outcomes(
    tmp_path: Path,
) -> None:
    class ModelFailureError(RuntimeError):
        pass

    timeout_agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(),)),
        FakeCognitiveAgent((asyncio.sleep(1),)),
        FakeVerifier((VerificationResult(verified=False),)),
        FakeBundleFactory(),
        timeout_seconds=0.01,
    )
    timeout_result = await timeout_agent.execute(_spec(timeout_seconds=0.01))
    assert timeout_result.outcome.kind == "timed-out"
    assert timeout_result.ux_sample_valid is False
    assert timeout_result.ux_sample_invalid_reason == (
        "timed-out: run timeout exceeded"
    )

    model_bundles = FakeBundleFactory()
    model_agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(),)),
        FakeCognitiveAgent((ModelFailureError("exhausted"),)),
        FakeVerifier((VerificationResult(verified=False),)),
        model_bundles,
    )
    model_result = await model_agent.execute(_spec())
    assert model_result.outcome.kind == "model-failure"
    assert model_result.ux_sample_valid is False
    assert model_result.ux_sample_invalid_reason == "model-failure: exhausted"

    budget_bundles = FakeBundleFactory()
    budget_agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(),)),
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "No budget."},
                    reason="No budget.",
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        budget_bundles,
    )
    budget_result = await budget_agent.execute(_spec(max_steps=1))
    assert budget_result.outcome.kind == "budget-exhausted"
    assert budget_result.ux_sample_valid is True


@pytest.mark.asyncio
async def test_repeated_fixture_input_terminates_with_diagnostic_and_finalizes(
    tmp_path: Path,
) -> None:
    provider = FakeObservationProvider(
        (
            _snapshot(
                second_element_id="submit",
                lineage_id="target-lineage",
                second_lineage_id="submit-lineage",
            ),
            _snapshot(
                "viewport-2",
                second_element_id="submit",
                lineage_id="target-lineage",
                second_lineage_id="submit-lineage",
            ),
        )
    )
    bundles = FakeBundleFactory()
    agent = _agent(
        tmp_path,
        provider,
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={
                        "kind": "type-fixture",
                        "element_id": "target",
                        "fixture_key": "invite_email",
                    },
                    reason="Enter the invite value.",
                ),
                CognitiveDecision(
                    action={
                        "kind": "type-fixture",
                        "element_id": "target",
                        "fixture_key": "invite_email",
                    },
                    reason="Repeat the invite value.",
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
    )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert result.outcome.kind == "budget-exhausted"
    assert result.terminal_reason == "repeated successful fixture input detected"
    assert result.ux_sample_valid is False
    assert result.ux_sample_invalid_reason == (
        "budget-exhausted: repeated successful fixture input detected"
    )
    assert bundles.bundle.finalized
    assert any(
        isinstance(event, dict)
        and event.get("kind") in {"repeated-fixture-input", "repeated-action-detected"}
        for event in bundles.bundle.events
    )


@pytest.mark.asyncio
async def test_model_call_budget_counts_calls_without_an_audit_source(
    tmp_path: Path,
) -> None:
    bundles = FakeBundleFactory()
    cognitive = FakeCognitiveAgent(
        (
            CognitiveDecision(
                action={"kind": "inspect", "element_id": "target"},
                reason="Inspect the only visible control.",
            ),
        )
    )
    agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(),)),
        cognitive,
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
    )

    result = await agent.execute(_spec(max_model_calls=1, timeout_seconds=None))

    assert result.outcome.kind == "budget-exhausted"
    assert result.terminal_reason == "model call budget exhausted"
    assert result.ux_sample_valid is False
    assert result.ux_sample_invalid_reason == (
        "budget-exhausted: model call budget exhausted"
    )
    assert len(cognitive.observations) == 1
    assert any(
        isinstance(event, dict)
        and event.get("kind") == "model-call-budget-exhausted"
        and event.get("model_calls") == 1
        for event in bundles.bundle.events
    )


@pytest.mark.asyncio
async def test_abandonment_at_model_call_limit_defers_to_invalid_model_budget(
    tmp_path: Path,
) -> None:
    provider = FakeObservationProvider((_snapshot(),))
    bundles = FakeBundleFactory()
    cognitive = FakeCognitiveAgent(
        (
            CognitiveDecision(
                action={"kind": "abandon", "reason": "No further progress."},
                reason="No further progress.",
            ),
        )
    )
    agent = _agent(
        tmp_path,
        provider,
        cognitive,
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
        timeout_seconds=None,
        max_model_calls=1,
    )

    result = await agent.execute(_spec(max_model_calls=1, timeout_seconds=None))

    assert result.outcome.kind == "budget-exhausted"
    assert result.terminal_reason == "model call budget exhausted"
    assert result.ux_sample_valid is False
    assert result.ux_sample_invalid_reason == (
        "budget-exhausted: model call budget exhausted"
    )
    assert isinstance(provider_action := provider.executed[0], WaitAction)
    assert provider_action.milliseconds == 0
    assert any(
        isinstance(event, dict)
        and event.get("kind") == "abandonment-deferred"
        for event in bundles.bundle.events
    )


@pytest.mark.asyncio
async def test_full_scent_and_cognitive_overlap_but_commit_evidence_in_logical_order(
    tmp_path: Path,
) -> None:
    records = ConcurrentModelRecordSource()
    full_started = asyncio.Event()
    cognitive_started = asyncio.Event()
    release_full = asyncio.Event()
    full = CoordinatedFullScentEvaluator(records, full_started, release_full)
    decision = CognitiveDecision(
        action={"kind": "abandon", "reason": "Stop."},
        reason="Stop.",
    )
    cognitive = CoordinatedCognitiveAgent(records, cognitive_started, decision)
    bundles = FakeBundleFactory()
    agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(),)),
        cognitive,
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
        full_scent_evaluator=full,
        model_record_source=records,  # type: ignore[arg-type]
    )

    task = asyncio.create_task(agent.execute(_spec(timeout_seconds=None)))
    try:
        await asyncio.wait_for(
            asyncio.gather(full_started.wait(), cognitive_started.wait()), timeout=1
        )
    except TimeoutError:
        release_full.set()
        await asyncio.gather(task, return_exceptions=True)
        raise
    release_full.set()
    result = await task

    assert result.outcome.kind == "agent-abandoned"
    assert full.inputs is not None
    assert full.inputs[0] == "Invite teammate"
    assert full.inputs[2] is not None
    assert full.inputs[2].id == "viewport-1"
    assert full.inputs[1].noticed_ids == frozenset({"target"})
    assert full.inputs[1].current_viewport_id == "viewport-1"
    assert cognitive.contexts == [
        CognitiveRunContext(
            viewport_id="viewport-1",
            working_memory_capacity=3,
            confidence=0.5,
            frustration=0.0,
            abandonment_threshold=0.9,
            attention_temperature=1.0,
        )
    ]
    assert cognitive.decision_inputs[0][0] == "Invite teammate"
    assert tuple(
        element.id
        for element in cognitive.decision_inputs[0][1].newly_revealed_elements
    ) == ("target",)

    model_events = [
        event
        for event in bundles.bundle.events
        if isinstance(event, dict) and event.get("kind") == "model-call-recorded"
    ]
    assert [event["record"].role for event in model_events] == [
        ModelRole.FULL_SCENT,
        ModelRole.COGNITIVE,
    ]
    evidence_kinds = [
        event.get("kind")
        for event in bundles.bundle.events
        if isinstance(event, dict)
        and event.get("kind") in {"full-scent-recorded", "decision-recorded"}
    ]
    assert evidence_kinds == ["full-scent-recorded", "decision-recorded"]
    assert [record.role for record in result.evidence.model_calls] == [
        ModelRole.FULL_SCENT,
        ModelRole.COGNITIVE,
    ]


@pytest.mark.asyncio
async def test_full_scent_uses_sequential_budget_fallback_when_one_call_remains(
    tmp_path: Path,
) -> None:
    records = ConcurrentModelRecordSource()
    full_started = asyncio.Event()
    release_full = asyncio.Event()
    full = CoordinatedFullScentEvaluator(records, full_started, release_full)
    cognitive = CoordinatedCognitiveAgent(
        records,
        asyncio.Event(),
        CognitiveDecision(
            action={"kind": "abandon", "reason": "Must not run."},
            reason="Must not run.",
        ),
    )
    bundles = FakeBundleFactory()
    agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(),)),
        cognitive,
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
        full_scent_evaluator=full,
        model_record_source=records,  # type: ignore[arg-type]
        max_model_calls=1,
    )
    release_full.set()

    result = await agent.execute(_spec(max_model_calls=1, timeout_seconds=None))

    assert result.outcome.kind == "budget-exhausted"
    assert cognitive.decision_inputs == []
    assert [record.role for record in result.evidence.model_calls] == [
        ModelRole.FULL_SCENT
    ]
    assert any(
        isinstance(event, dict)
        and event.get("kind") == "model-call-budget-exhausted"
        and event.get("model_calls") == 1
        for event in bundles.bundle.events
    )


@pytest.mark.asyncio
async def test_parallel_validation_failure_keeps_model_records_and_failure_semantics(
    tmp_path: Path,
) -> None:
    records = ConcurrentModelRecordSource()
    full = ValidationFailingFullScentEvaluator(
        records, asyncio.Event(), asyncio.Event()
    )
    cognitive = CoordinatedCognitiveAgent(
        records,
        asyncio.Event(),
        CognitiveDecision(
            action={"kind": "abandon", "reason": "Unused."},
            reason="Unused.",
        ),
    )
    bundles = FakeBundleFactory()
    agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(),)),
        cognitive,
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
        full_scent_evaluator=full,
        model_record_source=records,  # type: ignore[arg-type]
    )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert result.outcome.kind == "model-failure"
    assert result.terminal_reason == "full-scent: invalid scent"
    model_events = [
        event
        for event in bundles.bundle.events
        if isinstance(event, dict) and event.get("kind") == "model-call-recorded"
    ]
    assert [event["record"].role for event in model_events] == [
        ModelRole.FULL_SCENT,
        ModelRole.COGNITIVE,
    ]
    assert not any(
        isinstance(event, dict)
        and event.get("kind") in {"full-scent-recorded", "decision-recorded"}
        for event in bundles.bundle.events
    )


@pytest.mark.asyncio
async def test_parallel_cognitive_validation_failure_keeps_full_scent_evidence(
    tmp_path: Path,
) -> None:
    records = ConcurrentModelRecordSource()
    full = CoordinatedFullScentEvaluator(records, asyncio.Event(), asyncio.Event())
    full.release.set()
    cognitive = CoordinatedCognitiveAgent(
        records,
        asyncio.Event(),
        ModelResponseValidationError(ModelRole.COGNITIVE, "invalid decision"),
    )
    bundles = FakeBundleFactory()
    agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(),)),
        cognitive,
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
        full_scent_evaluator=full,
        model_record_source=records,  # type: ignore[arg-type]
    )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert result.outcome.kind == "model-failure"
    assert result.terminal_reason == "cognitive: invalid decision"
    evidence_kinds = [
        event.get("kind")
        for event in bundles.bundle.events
        if isinstance(event, dict)
        and event.get("kind") in {"full-scent-recorded", "decision-recorded"}
    ]
    assert evidence_kinds == ["full-scent-recorded"]


@pytest.mark.asyncio
async def test_parallel_cancellation_cancels_both_model_tasks_and_records_calls(
    tmp_path: Path,
) -> None:
    records = ConcurrentModelRecordSource()
    full_cancelled = asyncio.Event()
    cognitive_cancelled = asyncio.Event()
    full = BlockingFullScentEvaluator(records, asyncio.Event(), full_cancelled)
    cognitive = BlockingCognitiveAgent(records, asyncio.Event(), cognitive_cancelled)
    bundles = FakeBundleFactory()
    agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(),)),
        cognitive,
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
        full_scent_evaluator=full,
        model_record_source=records,  # type: ignore[arg-type]
    )

    task = asyncio.create_task(agent.execute(_spec(timeout_seconds=None)))
    await asyncio.gather(full.started.wait(), cognitive.started.wait())
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.gather(full_cancelled.wait(), cognitive_cancelled.wait())

    model_events = [
        event
        for event in bundles.bundle.events
        if isinstance(event, dict) and event.get("kind") == "model-call-recorded"
    ]
    assert [event["record"].role for event in model_events] == [
        ModelRole.FULL_SCENT,
        ModelRole.COGNITIVE,
    ]
    assert bundles.bundle.aborted


@pytest.mark.asyncio
async def test_meaningful_fixture_progress_resets_scroll_repetition(
    tmp_path: Path,
) -> None:
    snapshots = tuple(
        _snapshot(
            f"viewport-{index}",
            f"target-{index}",
            lineage_id="target-lineage",
        )
        for index in range(1, 6)
    )
    provider = FakeObservationProvider(
        snapshots,
        results=(
            PlatformActionResult(True, "http://fixture.test", 1, state_changed=True),
            PlatformActionResult(True, "http://fixture.test", 1, state_changed=True),
            PlatformActionResult(True, "http://fixture.test", 1, state_changed=True),
            PlatformActionResult(True, "http://fixture.test", 1, state_changed=True),
        ),
    )
    attention = RepeatingAttentionPolicy()
    bundles = FakeBundleFactory()
    agent = _agent(
        tmp_path,
        provider,
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "scroll", "direction": "down"},
                    reason="Look below.",
                ),
                CognitiveDecision(
                    action={
                        "kind": "type-fixture",
                        "element_id": "target-2",
                        "fixture_key": "invite_email",
                    },
                    reason="Complete the field.",
                ),
                CognitiveDecision(
                    action={"kind": "scroll", "direction": "down"},
                    reason="Look below again.",
                ),
                CognitiveDecision(
                    action={"kind": "scroll", "direction": "down"},
                    reason="Continue looking.",
                ),
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Test complete."},
                    reason="Test complete.",
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
        attention_policy=attention,
    )

    result = await agent.execute(_spec(max_steps=10, timeout_seconds=None))

    assert result.outcome.kind == "budget-exhausted", result.terminal_reason
    assert result.terminal_reason == "attention budget exhausted"
    assert attention.recovery_levels == [0, 1, 0, 1, 2]
    assert not any(
        isinstance(event, dict) and event.get("kind") == "repeated-action-detected"
        for event in bundles.bundle.events
    )


@pytest.mark.asyncio
async def test_three_consecutive_semantic_stalls_recover_then_abandon(
    tmp_path: Path,
) -> None:
    snapshots = tuple(
        _snapshot(
            f"viewport-{index}",
            f"target-{index}",
            lineage_id="target-lineage",
        )
        for index in range(1, 5)
    )
    provider = FakeObservationProvider(
        snapshots,
        results=tuple(
            PlatformActionResult(True, "http://fixture.test", 1, state_changed=True)
            for _ in range(3)
        ),
    )
    attention = RepeatingAttentionPolicy()
    bundles = FakeBundleFactory()
    scroll = CognitiveDecision(
        action={"kind": "scroll", "direction": "down"},
        reason="Look below.",
    )
    agent = _agent(
        tmp_path,
        provider,
        FakeCognitiveAgent((scroll, scroll, scroll)),
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
        attention_policy=attention,
    )

    result = await agent.execute(_spec(max_steps=10, timeout_seconds=None))

    assert result.outcome.kind == "agent-abandoned"
    assert result.terminal_reason == "repeated actions produced no progress"
    assert attention.recovery_levels == [0, 1, 2]
    assert len(provider.executed) == 3
    recovery_events = [
        event
        for event in bundles.bundle.events
        if isinstance(event, dict) and event.get("kind") == "no-progress-recovery"
    ]
    assert [event["count"] for event in recovery_events] == [1, 2]
    assert any(
        isinstance(event, dict)
        and event.get("kind") == "repeated-action-detected"
        and event.get("count") == 3
        for event in bundles.bundle.events
    )


@pytest.mark.asyncio
async def test_repeated_semantic_action_cycle_finalizes_after_second_cycle(
    tmp_path: Path,
) -> None:
    snapshots = (
        _snapshot("main-1", "share-1", label="Share"),
        _snapshot("dialog-1", "close-1", label="Close"),
        _snapshot("main-2", "share-2", label="Share"),
        _snapshot("dialog-2", "close-2", label="Close"),
        _snapshot("main-3", "share-3", label="Share"),
    )
    provider = FakeObservationProvider(
        snapshots,
        results=tuple(
            PlatformActionResult(
                True,
                "https://fixture.test/invite?token=private-secret",
                1,
                state_changed=True,
            )
            for _ in range(4)
        ),
    )
    bundles = FakeBundleFactory()
    agent = _agent(
        tmp_path,
        provider,
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "interact", "element_id": "share-1"},
                    reason="Open sharing.",
                ),
                CognitiveDecision(
                    action={"kind": "interact", "element_id": "close-1"},
                    reason="Close sharing.",
                ),
                CognitiveDecision(
                    action={"kind": "interact", "element_id": "share-2"},
                    reason="Open sharing again.",
                ),
                CognitiveDecision(
                    action={"kind": "interact", "element_id": "close-2"},
                    reason="Close sharing again.",
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
        attention_policy=RepeatingAttentionPolicy(),
    )

    result = await agent.execute(_spec(max_steps=10, timeout_seconds=None))

    assert result.outcome.kind == "agent-abandoned", result.terminal_reason
    assert result.terminal_reason == "repeated semantic action cycle detected"
    assert len(provider.executed) == 4
    assert bundles.bundle.finalized
    cycle_events = [
        event
        for event in bundles.bundle.events
        if isinstance(event, dict) and event.get("kind") == "repeated-action-cycle"
    ]
    assert cycle_events == [
        {
            "kind": "repeated-action-cycle",
            "cycle_length": 2,
            "reason": "repeated semantic action cycle detected",
        }
    ]
    assert "private-secret" not in repr(cycle_events)


@pytest.mark.asyncio
async def test_failed_action_breaks_semantic_cycle_history(tmp_path: Path) -> None:
    snapshots = (
        _snapshot("state-a-1", "action-a-1", label="Action A"),
        _snapshot("state-b-1", "action-b-1", label="Action B"),
        _snapshot("state-c-1", "action-c-1", label="Action C"),
        _snapshot("state-a-2", "action-a-2", label="Action A"),
        _snapshot("state-b-2", "action-b-2", label="Action B"),
        _snapshot("state-c-2", "action-c-2", label="Action C"),
    )
    provider = FakeObservationProvider(
        snapshots,
        results=(
            PlatformActionResult(True, "https://fixture.test/a", 1),
            PlatformActionResult(True, "https://fixture.test/b", 1),
            PlatformActionResult(False, "https://fixture.test/c", 1, error="blocked"),
            PlatformActionResult(True, "https://fixture.test/a", 1),
            PlatformActionResult(True, "https://fixture.test/b", 1),
        ),
    )
    agent = _agent(
        tmp_path,
        provider,
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "interact", "element_id": "action-a-1"},
                    reason="A.",
                ),
                CognitiveDecision(
                    action={"kind": "interact", "element_id": "action-b-1"},
                    reason="B.",
                ),
                CognitiveDecision(
                    action={"kind": "interact", "element_id": "action-c-1"},
                    reason="C fails.",
                ),
                CognitiveDecision(
                    action={"kind": "interact", "element_id": "action-a-2"},
                    reason="A again.",
                ),
                CognitiveDecision(
                    action={"kind": "interact", "element_id": "action-b-2"},
                    reason="B again.",
                ),
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "done checking"},
                    reason="Stop.",
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        FakeBundleFactory(),
        attention_policy=RepeatingAttentionPolicy(),
    )

    result = await agent.execute(_spec(max_steps=10, timeout_seconds=None))

    assert result.terminal_reason != "repeated semantic action cycle detected"
    assert len(provider.executed) == 5
    assert not any(
        event.kind == "repeated-action-cycle" for event in result.state.events
    )


@pytest.mark.asyncio
async def test_unlimited_run_awaits_run_and_terminal_verification(
    tmp_path: Path,
) -> None:
    class DelayedVerifier(FakeVerifier):
        async def verify(self, session: SessionHandle) -> VerificationResult:
            await asyncio.sleep(0.02)
            return await super().verify(session)

    provider = FakeObservationProvider((_snapshot(),))
    bundles = FakeBundleFactory()
    verifier = DelayedVerifier((VerificationResult(verified=False),))
    delayed_decision = asyncio.sleep(
        0.02,
        result=CognitiveDecision(
            action={"kind": "abandon", "reason": "Observed outcome."},
            reason="Observed outcome.",
        ),
    )
    agent = _agent(
        tmp_path,
        provider,
        FakeCognitiveAgent((delayed_decision,)),
        verifier,
        bundles,
        timeout_seconds=None,
    )

    result = await asyncio.wait_for(
        agent.execute(_spec(timeout_seconds=None)), timeout=1
    )

    assert result.outcome.kind == "agent-abandoned"
    assert result.verification.verified is False
    assert verifier.calls == 0
    assert bundles.bundle.finalized
    assert bundles.bundle.events[-1].kind == "run-terminated"
    assert provider.ended == 1


@pytest.mark.asyncio
async def test_finite_timeout_bounds_terminal_verification(tmp_path: Path) -> None:
    class BlockingVerifier(FakeVerifier):
        async def verify(self, session: SessionHandle) -> VerificationResult:
            await asyncio.sleep(1)
            return await super().verify(session)

    provider = FakeObservationProvider((_snapshot(),))
    bundles = FakeBundleFactory()
    agent = _agent(
        tmp_path,
        provider,
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Stop."},
                    reason="Stop.",
                ),
            )
        ),
        BlockingVerifier((VerificationResult(verified=False),)),
        bundles,
        timeout_seconds=0.5,
    )

    result = await agent.execute(_spec(timeout_seconds=0.5))

    assert result.outcome.kind == "agent-abandoned"
    assert result.verification.verified is False
    assert result.verification.details == "independent verification unavailable"
    assert bundles.bundle.finalized
    assert bundles.bundle.events[-1].kind == "run-terminated"
    assert provider.ended == 1


@pytest.mark.asyncio
async def test_scroll_back_wait_wrong_and_stale_actions_recapture_and_recover(
    tmp_path: Path,
) -> None:
    snapshots = tuple(
        _snapshot(
            f"viewport-{index}",
            f"target-{index}",
            "fallback" if index == 5 else None,
        )
        for index in range(1, 6)
    )
    provider = FakeObservationProvider(
        snapshots,
        results=(
            PlatformActionResult(True, "http://fixture.test", 1, state_changed=True),
            PlatformActionResult(True, "http://fixture.test", 1, state_changed=True),
            PlatformActionResult(True, "http://fixture.test", 1, state_changed=False),
            PlatformActionResult(
                False, "http://fixture.test", 1, state_changed=False, error="wrong"
            ),
        ),
    )
    decisions = (
        CognitiveDecision(
            action={"kind": "scroll", "direction": "down"}, reason="Look below."
        ),
        CognitiveDecision(action={"kind": "back"}, reason="Return."),
        CognitiveDecision(action={"kind": "wait"}, reason="Wait for feedback."),
        CognitiveDecision(
            action={"kind": "interact", "element_id": "target-4"},
            reason="Try candidate.",
        ),
        CognitiveDecision(
            action={"kind": "interact", "element_id": "target-1"},
            reason="Reuse stale target.",
        ),
        CognitiveDecision(
            action={"kind": "abandon", "reason": "No path."}, reason="Stop."
        ),
    )
    bundles = FakeBundleFactory()
    agent = _agent(
        tmp_path,
        provider,
        FakeCognitiveAgent(decisions),
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
    )

    result = await agent.execute(_spec(max_steps=12))

    assert result.outcome.kind == "agent-abandoned", result.terminal_reason
    assert [action.kind for action in provider.executed] == [
        "scroll",
        "back",
        "wait",
        "click",
        "wait",
    ]
    assert len(provider.executed) < len(decisions)
    assert provider.capture_count == 6
    assert any(
        event.get("kind") == "action-rejected"
        for event in bundles.bundle.events
        if isinstance(event, dict)
    )


@pytest.mark.asyncio
async def test_recapture_does_not_rediscover_unchanged_lineage(
    tmp_path: Path,
) -> None:
    provider = FakeObservationProvider(
        (
            _snapshot(
                "viewport-old",
                "target-old",
                lineage_id="lineage-target",
            ),
            _snapshot(
                "viewport-new",
                "target-new",
                "fallback-new",
                lineage_id="lineage-target",
                second_lineage_id="lineage-fallback",
            ),
        )
    )
    cognitive = FakeCognitiveAgent(
        (
            CognitiveDecision(action={"kind": "wait"}, reason="Recapture."),
            CognitiveDecision(
                action={"kind": "abandon", "reason": "Done."}, reason="Done."
            ),
        )
    )
    agent = _agent(
        tmp_path,
        provider,
        cognitive,
        FakeVerifier((VerificationResult(verified=False),)),
        FakeBundleFactory(),
    )

    result = await agent.execute(_spec())

    assert result.outcome.kind == "agent-abandoned"
    assert [
        observation.newly_revealed_elements[0].label
        for observation in cognitive.observations
    ] == ["Invite teammate", "Fallback"]
    assert result.state.attention.noticed_ids == frozenset(
        {"target-new", "fallback-new"}
    )


@pytest.mark.asyncio
async def test_web_verifier_checks_fixture_state_without_exposing_expected_value(
    tmp_path: Path,
) -> None:
    spec = _spec()
    provider = FakeObservationProvider((_snapshot(),))
    session = await provider.start_session(_config(spec, tmp_path))
    state_client = FakeFixtureStateClient({"workspace": {"invite_status": "sent"}})
    verifier = WebVerifier(
        FixtureStateVerifierSpec(
            type="fixture-state",
            resource="workspace",
            field="invite_status",
            operator="equals",
            expected_fixture_key="invite_email",
        ),
        fixture_inputs=FixtureInputs(
            values={"invite_email": "sent"},
        ),
        fixture_state_client=state_client,
    )

    result = await verifier.verify(session)

    assert result.verified
    assert result.evidence_ids == ("fixture:run-1:workspace.invite_status",)
    assert "sent" not in (result.details or "")
    assert state_client.session_ids == ["run-1"]
    assert verifier.last_capture is None


@pytest.mark.asyncio
async def test_web_verifier_checks_fresh_persona_visible_capture(
    tmp_path: Path,
) -> None:
    spec = _spec()
    provider = FakeObservationProvider((_snapshot(),))
    session = await provider.start_session(_config(spec, tmp_path))
    verifier = WebVerifier(
        VisibleResultVerifierSpec(type="visible-result", text="Invite", role="button"),
        observation_provider=provider,
        snapshot_extractor=lambda capture: (
            capture.snapshot
            if capture.snapshot is not None
            else (_ for _ in ()).throw(ValueError("missing snapshot"))
        ),
    )

    result = await verifier.verify(session)

    assert result.verified
    assert result.evidence_ids == ("viewport:viewport-1-verification-1:element:target",)
    assert provider.capture_count == 1
    assert verifier.last_capture is not provider.captures[0]
    assert verifier.last_capture is not None
    assert verifier.last_capture.viewport_id == "viewport-1-verification-1"


@pytest.mark.asyncio
async def test_web_verifier_clears_last_capture_before_each_verification(
    tmp_path: Path,
) -> None:
    spec = _spec()
    provider = FakeObservationProvider((_snapshot(),))
    session = await provider.start_session(_config(spec, tmp_path))
    verifier = WebVerifier(
        VisibleResultVerifierSpec(type="visible-result", text="Invite", role="button"),
        observation_provider=provider,
        snapshot_extractor=lambda capture: capture.snapshot
        if capture.snapshot is not None
        else (_ for _ in ()).throw(ValueError("missing snapshot")),
    )

    await verifier.verify(session)
    provider.failure = RuntimeError("capture failed")
    with pytest.raises(RuntimeError, match="capture failed"):
        await verifier.verify(session)

    assert verifier.last_capture is None


@pytest.mark.asyncio
async def test_colour_change_verifier_detects_material_theme_flip_and_rejects_noise(
    tmp_path: Path,
) -> None:
    """A perceivable visual state change must be verifiable without new text.

    A theme toggle produces no new persona-visible text, so a text verifier can
    never confirm it. The opt-in colour-change mode compares verification-time
    document/body computed colours against a baseline capture and requires a
    *material* difference. Sub-threshold noise (a 1-unit wobble, hover flicker)
    must never pass.
    """

    def _capture(
        *, document_background: str, body_background: str, viewport_background: str
    ) -> ObservationCapture:
        snapshot = _snapshot()
        return ObservationCapture(
            session_id="run-1",
            viewport_id=snapshot.id,
            url="http://fixture.test/app/run/improved",
            title="Fixture",
            viewport=ViewportSize(width=1024, height=768),
            screenshot=b"shot",
            snapshot=snapshot,
            document_background=document_background,
            body_background=body_background,
            viewport_background=viewport_background,
        )

    class ColourProvider:
        def __init__(self, captures: tuple[ObservationCapture, ...]) -> None:
            self.captures = list(captures)

        async def start_session(self, config: ObservationSessionConfig) -> SessionHandle:
            del config
            raise NotImplementedError

        async def capture(self, session: SessionHandle) -> ObservationCapture:
            del session
            return self.captures.pop(0)

    # 1) A real light -> dark theme flip is a material change and must verify.
    baseline = _capture(
        document_background="#ffffff",
        body_background="#ffffff",
        viewport_background="#ffffff",
    )
    flipped = _capture(
        document_background="#111418",
        body_background="#111418",
        viewport_background="#111418",
    )
    provider = ColourProvider((flipped,))
    verifier = WebVerifier(
        ColourChangeVerifierSpec(type="colour-change"),
        observation_provider=provider,
        snapshot_extractor=lambda capture: capture.snapshot,
        baseline_capture=baseline,
    )
    result = await verifier.verify(_session_handle())
    assert result.verified
    assert result.details is not None and "colour" in result.details.casefold()
    assert result.before_state is not None and result.after_state is not None
    assert result.before_state["document_background"] == "#ffffff"
    assert result.after_state["document_background"] == "#111418"

    # 2) A sub-threshold wobble must NOT verify.
    noise = _capture(
        document_background="#fefefe",
        body_background="#ffffff",
        viewport_background="#fdfdfd",
    )
    provider = ColourProvider((noise,))
    verifier = WebVerifier(
        ColourChangeVerifierSpec(type="colour-change"),
        observation_provider=provider,
        snapshot_extractor=lambda capture: capture.snapshot,
        baseline_capture=baseline,
    )
    result = await verifier.verify(_session_handle())
    assert not result.verified
    assert result.before_state is not None and result.after_state is not None


@pytest.mark.asyncio
async def test_visible_result_verifier_semantics_are_unchanged_by_colour_mode(
    tmp_path: Path,
) -> None:
    """Existing visible-result scenarios must behave byte-identically."""

    spec = _spec()
    provider = FakeObservationProvider((_snapshot(),))
    session = await provider.start_session(_config(spec, tmp_path))
    verifier = WebVerifier(
        VisibleResultVerifierSpec(type="visible-result", text="Invite", role="button"),
        observation_provider=provider,
        snapshot_extractor=lambda capture: capture.snapshot
        if capture.snapshot is not None
        else (_ for _ in ()).throw(ValueError("missing snapshot")),
    )

    result = await verifier.verify(session)

    assert result.verified
    assert result.evidence_ids == ("viewport:viewport-1-verification-1:element:target",)
    assert result.details == "persona-visible result matched"
    assert result.before_state is None
    assert result.after_state is None


@pytest.mark.asyncio
async def test_visible_verification_capture_is_persisted_once_with_exact_screenshot(    tmp_path: Path,
) -> None:
    initial = _snapshot()
    verified_snapshot = _snapshot("verification-viewport", element_id="result", label="Sent")
    exact_screenshot = b"exact-verification-screenshot"
    provider = FakeObservationProvider(
        (initial, verified_snapshot),
        screenshots=(b"initial-screenshot", exact_screenshot),
    )
    verifier = WebVerifier(
        VisibleResultVerifierSpec(type="visible-result", text="Sent", role="button"),
        observation_provider=provider,
        snapshot_extractor=lambda capture: capture.snapshot
        if capture.snapshot is not None
        else (_ for _ in ()).throw(ValueError("missing snapshot")),
    )
    bundles = FilesystemBundleFactory(tmp_path / "bundles")
    agent = _agent(
        tmp_path,
        provider,
        FakeCognitiveAgent(
            (CognitiveDecision(action={"kind": "complete"}, reason="Complete."),)
        ),
        verifier,
        bundles,
    )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert provider.capture_count == 2
    assert result.verification.evidence_ids[0].startswith(
        "viewport:verification-viewport"
    )
    screenshot_id = next(
        evidence_id
        for evidence_id in result.verification.evidence_ids
        if evidence_id.startswith("screenshot:")
    )
    screenshot_path = screenshot_id.removeprefix("screenshot:")
    assert any(
        snapshot.id.startswith("verification-viewport-verification-")
        and snapshot.screenshot_artifact == screenshot_path
        for snapshot in result.state.snapshots
    )
    checksum = next(
        item for item in result.state.artifact_checksums if item.path == screenshot_path
    )
    assert checksum.sha256 == hashlib.sha256(exact_screenshot).hexdigest()
    assert (result.bundle_path / screenshot_path).read_bytes() == exact_screenshot
    timeline = [
        json.loads(line)
        for line in (result.bundle_path / "timeline.jsonl").read_text().splitlines()
    ]
    viewport_sequence = next(
        index for index, event in enumerate(timeline)
        if event["kind"] == "viewport-captured"
        and event["snapshot"]["id"].startswith("verification-viewport-verification-")
    )
    verification_sequence = next(
        index for index, event in enumerate(timeline)
        if event["kind"] == "verification-recorded"
    )
    assert viewport_sequence < verification_sequence


@pytest.mark.asyncio
async def test_visible_verifier_falls_back_to_page_text_when_viewport_snapshot_misses_target(
    tmp_path: Path,
) -> None:
    """Regression: the discover-muslim-pedia-features scenario reached the
    Muslim Pedia section header, but the verifier text lived in a feature
    card several viewports below. The viewport snapshot did not contain the
    text, yet the rendered page text did. The page-text fallback unblocks
    such scenarios without weakening the viewport-snapshot primary path.
    """

    class PageTextProvider(FakeObservationProvider):
        async def capture(self, session: SessionHandle) -> ObservationCapture:  # type: ignore[override]
            base = await super().capture(session)
            page_text = (
                "ACT 01 Muslim Pedia An all-in-one Muslim app for iOS. "
                "Rendering engine A live vector Quran engine Ported "
                "DigitalKhatt, a TypeScript calligraphy engine, to Flutter."
            )
            return replace(base, page_text=page_text)

    spec = _spec()
    provider = PageTextProvider((_snapshot(),))
    session = await provider.start_session(_config(spec, tmp_path))
    verifier = WebVerifier(
        VisibleResultVerifierSpec(
            type="visible-result", text="A live vector Quran engine"
        ),
        observation_provider=provider,
        snapshot_extractor=lambda capture: capture.snapshot
        if capture.snapshot is not None
        else (_ for _ in ()).throw(ValueError("missing snapshot")),
    )

    result = await verifier.verify(session)

    assert result.verified
    assert any(eid.endswith(":page-text") for eid in result.evidence_ids)
    assert result.details == "page-text result matched"


@pytest.mark.asyncio
async def test_visible_verifier_does_not_use_page_text_when_target_is_absent(
    tmp_path: Path,
) -> None:
    """The page-text fallback must not promote verification when the target
    is genuinely absent from the document.
    """

    class PageTextProvider(FakeObservationProvider):
        async def capture(self, session: SessionHandle) -> ObservationCapture:  # type: ignore[override]
            base = await super().capture(session)
            return replace(base, page_text="Header Subhead footer text")

    spec = _spec()
    provider = PageTextProvider((_snapshot(),))
    session = await provider.start_session(_config(spec, tmp_path))
    verifier = WebVerifier(
        VisibleResultVerifierSpec(
            type="visible-result", text="Nowhere to be found"
        ),
        observation_provider=provider,
        snapshot_extractor=lambda capture: capture.snapshot
        if capture.snapshot is not None
        else (_ for _ in ()).throw(ValueError("missing snapshot")),
    )

    result = await verifier.verify(session)

    assert not result.verified
    assert result.details == "persona-visible result not found"


@pytest.mark.asyncio
async def test_verification_capture_reusing_viewport_id_gets_distinct_identity(
    tmp_path: Path,
) -> None:
    initial = _snapshot("viewport-1")
    verified_snapshot = _snapshot("viewport-1", element_id="result", label="Sent")
    initial_screenshot = b"initial-screenshot"
    exact_screenshot = b"different-verification-screenshot"
    provider = FakeObservationProvider(
        (initial, verified_snapshot),
        screenshots=(initial_screenshot, exact_screenshot),
    )
    verifier = WebVerifier(
        VisibleResultVerifierSpec(type="visible-result", text="Sent", role="button"),
        observation_provider=provider,
        snapshot_extractor=lambda capture: capture.snapshot
        if capture.snapshot is not None
        else (_ for _ in ()).throw(ValueError("missing snapshot")),
    )
    agent = _agent(
        tmp_path,
        provider,
        FakeCognitiveAgent(
            (CognitiveDecision(action={"kind": "complete"}, reason="Complete."),)
        ),
        verifier,
        FilesystemBundleFactory(tmp_path / "bundles"),
    )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert result.state.snapshots[0].id == "viewport-1"
    assert len(result.state.snapshots) == 2
    persisted_verifier = result.state.snapshots[1]
    assert persisted_verifier.id != "viewport-1"
    assert persisted_verifier.element("result").execution_reference.viewport_id == (
        persisted_verifier.id
    )
    assert (
        f"viewport:{persisted_verifier.id}:element:result"
        in result.verification.evidence_ids
    )
    timeline = [
        json.loads(line)
        for line in (result.bundle_path / "timeline.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    viewport_events = [
        event for event in timeline if event["kind"] == "viewport-captured"
    ]
    assert [event["snapshot"]["id"] for event in viewport_events] == [
        "viewport-1",
        persisted_verifier.id,
    ]
    assert viewport_events[0]["sequence"] != viewport_events[1]["sequence"]
    assert (
        result.bundle_path / persisted_verifier.screenshot_artifact
    ).read_bytes() == exact_screenshot


@pytest.mark.asyncio
async def test_session_start_failure_does_not_cleanup_previous_run_session(
    tmp_path: Path,
) -> None:
    provider = FakeObservationProvider((_snapshot(second_element_id="result"),))
    verifier = FakeVerifier(
        (
            VerificationResult(verified=True, evidence_ids=("v1",)),
            VerificationResult(verified=False),
        )
    )
    agent = _agent(
        tmp_path,
        provider,
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "interact", "element_id": "target"},
                    reason="Target matches goal.",
                ),
                CognitiveDecision(
                    action={"kind": "complete"},
                    reason="I saw enough visible evidence.",
                ),
            )
        ),
        verifier,
        FakeBundleFactory(),
    )

    first = await agent.execute(_spec())
    provider.start_failure = ObservationProviderError("cannot start")
    second = await agent.execute(_spec())

    assert first.outcome.kind == "verified-success"
    assert second.outcome.kind == "provider-failure"
    assert provider.ended == 1


@pytest.mark.asyncio
async def test_terminal_event_append_failure_aborts_bundle_and_cleans_up(
    tmp_path: Path,
) -> None:
    class TerminalEventFailureBundle(FakeBundle):
        def append_event(self, event: object) -> int:
            if getattr(event, "kind", None) == "run-terminated":
                raise RuntimeError("timeline unavailable")
            return super().append_event(event)

    provider = FakeObservationProvider((_snapshot(),))
    bundles = FakeBundleFactory()
    bundles.bundle = TerminalEventFailureBundle()
    agent = _agent(
        tmp_path,
        provider,
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Stop."},
                    reason="Stop.",
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
    )

    with pytest.raises(RuntimeError, match="timeline unavailable"):
        await agent.execute(_spec())

    assert bundles.bundle.aborted
    assert provider.ended == 1


@pytest.mark.asyncio
async def test_run_records_replay_and_role_specific_model_evidence(
    tmp_path: Path,
) -> None:
    bundles = FakeBundleFactory()
    agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(),)),
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Stop."},
                    reason="No viable path.",
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
        model_record_source=FakeModelRecordSource(),
    )

    result = await agent.execute(_spec())

    mapping_events = [
        event for event in bundles.bundle.events if isinstance(event, dict)
    ]
    kinds = {event["kind"] for event in mapping_events}
    assert {
        "prominence-recorded",
        "attention-selection-recorded",
        "decision-recorded",
        "model-call-recorded",
    }.issubset(kinds)
    viewport_event = next(
        event
        for event in bundles.bundle.events
        if getattr(event, "kind", None) == "viewport-captured"
    )
    assert viewport_event.snapshot.screenshot_artifact.startswith("artifacts/")
    prominence = next(
        event for event in mapping_events if event["kind"] == "prominence-recorded"
    )
    assert prominence["scores"][0]["feature_contributions"]
    model_call = next(
        event for event in mapping_events if event["kind"] == "model-call-recorded"
    )
    assert model_call["record"].role is ModelRole.COGNITIVE
    assert model_call["record"].retries[0].status_code == 429
    persisted_model_record = json.dumps(asdict(model_call["record"]))
    assert "super-secret-api-key" not in persisted_model_record
    assert "demo-secret@example.test" not in persisted_model_record
    assert "[REDACTED]" in persisted_model_record
    assert result.evidence.model_calls[0].token_usage.total_tokens == 7


@pytest.mark.asyncio
async def test_evidence_and_finding_replay_links_use_persisted_event_sequences(
    tmp_path: Path,
) -> None:
    bundles = FilesystemBundleFactory(tmp_path)

    def evaluate_with_finding(result):
        metrics = evaluate_run(
            result,
            EvaluationTarget(element_id="target"),
            inputs=RunEvaluationInputs(target_prominence=0.1),
        )
        findings = FindingRuleSet.default().evaluate(metrics)
        return replace(result, metrics=metrics, findings=findings)

    agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(),)),
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Stop."},
                    reason="No viable path.",
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
        result_evaluator=evaluate_with_finding,
    )

    result = await agent.execute(_spec())

    assert isinstance(result.bundle_path, Path)
    timeline = [
        json.loads(line)
        for line in (result.bundle_path / "timeline.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    timeline_events = {f"event-{event['sequence']}": event for event in timeline}
    prominence_evidence = next(
        evidence
        for evidence in result.metrics.evidence
        if evidence.evidence_id.endswith(":target-prominence")
    )
    inspected_evidence = next(
        evidence
        for evidence in result.metrics.evidence
        if evidence.evidence_id.endswith(":inspected-elements")
    )
    finding = next(
        item for item in result.findings if item.category == "weak-target-prominence"
    )
    assert prominence_evidence.source_event_ids == ("event-3",)
    assert timeline_events["event-3"]["kind"] == "prominence-recorded"
    assert "event-5" in inspected_evidence.source_event_ids
    assert "event-3" not in inspected_evidence.source_event_ids
    assert timeline_events["event-5"]["kind"] == "observation-recorded"
    assert "event=event-3" in finding.replay_links[0]


@pytest.mark.asyncio
async def test_bundle_publication_failure_never_returns_success_result(
    tmp_path: Path,
) -> None:
    bundles = FakeBundleFactory()
    bundles.bundle.finalize_error = OSError("atomic publish failed")
    agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(),)),
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "interact", "element_id": "target"},
                    reason="Complete task.",
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=True),)),
        bundles,
    )

    with pytest.raises(RunFinalizationError) as failure:
        await agent.execute(_spec())

    assert failure.value.outcome.kind == "internal-error"
    assert bundles.bundle.aborted
    assert bundles.bundle.finalized is False


@pytest.mark.asyncio
async def test_cancellation_propagates_aborts_staging_and_cleans_up_once(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()

    class BlockingCognitiveAgent(FakeCognitiveAgent):
        async def decide(
            self, goal: str, observation: ProgressiveObservation
        ) -> object:
            del goal, observation
            started.set()
            await asyncio.Future()

    provider = FakeObservationProvider((_snapshot(),))
    output = tmp_path / "cancelled-run"
    agent = _agent(
        tmp_path,
        provider,
        BlockingCognitiveAgent(()),
        FakeVerifier((VerificationResult(verified=False),)),
        FilesystemBundleFactory(output),
    )
    task = asyncio.create_task(agent.execute(_spec()))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    staging = output / ".staging" / "run-1"
    assert provider.ended == 1
    assert (staging / "crash.marker").is_file()
    assert not (staging / "result.json").exists()
    assert not (output / "runs" / "run-1").exists()


@pytest.mark.asyncio
async def test_cleanup_security_failure_aborts_without_publishing_result(
    tmp_path: Path,
) -> None:
    class TraceSanitizationError(RuntimeError):
        pass

    class FailingCleanupProvider(FakeObservationProvider):
        async def end_session(self, session: SessionHandle) -> None:
            del session
            self.ended += 1
            raise TraceSanitizationError("trace sanitization failed")

    provider = FailingCleanupProvider((_snapshot(),))
    bundles = FakeBundleFactory()
    agent = _agent(
        tmp_path,
        provider,
        FakeCognitiveAgent((Abandon(reason="Stop."),)),
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
    )

    with pytest.raises(TraceSanitizationError, match="trace sanitization failed"):
        await agent.execute(_spec())

    assert provider.ended == 1
    assert bundles.bundle.aborted
    assert bundles.bundle.finalized is False


@pytest.mark.asyncio
async def test_run_evaluation_is_persisted_before_bundle_publication(
    tmp_path: Path,
) -> None:
    bundles = FakeBundleFactory()

    def evaluate(result):
        metrics = evaluate_run(result, EvaluationTarget("target"))
        return replace(result, metrics=metrics)

    agent = _agent(
        tmp_path,
        FakeObservationProvider((_snapshot(),)),
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "Stop."},
                    reason="No viable path.",
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
        result_evaluator=evaluate,
    )

    result = await agent.execute(_spec())

    assert result.metrics is not None
    assert result.ux_sample_valid is True
    assert result.metrics.run_id == result.run_id
    persisted = bundles.bundle.final_result
    assert persisted is not None
    assert persisted.metrics == result.metrics


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ("before-observation", "before-target"))
async def test_evaluator_failure_still_publishes_closed_immutable_bundle(
    tmp_path: Path,
    failure_point: str,
) -> None:
    output = tmp_path / failure_point
    bundles = FilesystemBundleFactory(output)
    if failure_point == "before-observation":
        provider = FakeObservationProvider(
            (_snapshot(),),
            start_failure=ObservationProviderError("capture unavailable"),
        )
        cognitive = FakeCognitiveAgent(())
    else:
        provider = FakeObservationProvider((_snapshot(),))
        cognitive = FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "No target."},
                    reason="No target.",
                ),
            )
        )

    def fail_evaluation(result):
        return replace(
            result, metrics=evaluate_run(result, evaluation_target_for(result))
        )

    agent = _agent(
        tmp_path,
        provider,
        cognitive,
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
        result_evaluator=fail_evaluation,
    )

    result = await agent.execute(_spec())

    bundle = output / "runs" / result.run_id
    assert result.state.is_finalized
    assert result.outcome.kind == (
        "provider-failure"
        if failure_point == "before-observation"
        else "agent-abandoned"
    )
    assert result.ux_sample_valid is False
    assert result.metrics is None
    assert result.findings is None
    if failure_point == "before-observation":
        assert result.evaluation_failure_reason is None
        assert result.ux_sample_invalid_reason.startswith("provider-failure:")
    else:
        assert result.evaluation_failure_reason is not None
        assert result.evaluation_failure_reason.startswith(
            "evaluation evidence unavailable:"
        )
        assert result.ux_sample_invalid_reason.startswith("evaluation-failure:")
    assert result.state.events[-1].kind == "run-terminated"
    assert bundle.is_dir()
    assert not (output / ".staging" / result.run_id).exists()
    persisted = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    assert persisted["metrics"] is None
    assert persisted["findings"] is None
    if failure_point == "before-observation":
        assert persisted["evaluation_failure_reason"] is None
    else:
        assert persisted["evaluation_failure_reason"].startswith(
            "evaluation evidence unavailable:"
        )
        assert result.run_id not in result.evaluation_failure_reason
    assert persisted["ux_sample_valid"] is False
    timeline = [
        json.loads(line)
        for line in (bundle / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert timeline[-1]["kind"] == "run-terminated"
    checksums = {
        path: digest
        for digest, path in (
            line.split("  ", maxsplit=1)
            for line in (bundle / "checksums.sha256")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    assert {"manifest.json", "result.json", "timeline.jsonl"}.issubset(checksums)
    for relative_path, digest in checksums.items():
        assert (
            hashlib.sha256((bundle / relative_path).read_bytes()).hexdigest() == digest
        )


@pytest.mark.asyncio
async def test_abandonment_defers_for_reobserve_then_contact_target_succeeds(
    tmp_path: Path,
) -> None:
    provider = FakeObservationProvider(
        (
            _snapshot("viewport-1", "partial", label="Partial view"),
            _snapshot("viewport-2", "contact", label="Contact"),
        )
    )
    bundles = FakeBundleFactory()
    attention = RepeatingAttentionPolicy()
    agent = _agent(
        tmp_path,
        provider,
        FakeCognitiveAgent(
            (
                CognitiveDecision(
                    action={"kind": "abandon", "reason": "No target yet."},
                    reason="No target yet.",
                ),
                CognitiveDecision(
                    action={"kind": "interact", "element_id": "contact"},
                    reason="Contact control is visible.",
                ),
                CognitiveDecision(
                    action={"kind": "complete"},
                    reason="I saw enough visible evidence.",
                ),
            )
        ),
        FakeVerifier((VerificationResult(verified=True),)),
        bundles,
        attention_policy=attention,
    )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert result.outcome.kind == "verified-success"
    assert isinstance(provider.executed[0], WaitAction)
    assert provider.executed[0].milliseconds == 0
    assert provider.executed[1].kind == "click"
    assert provider.executed[1].element_id == "contact"
    assert attention.recovery_levels == [0, 1, 2]
    assert any(
        isinstance(event, dict)
        and event.get("kind") == "abandonment-deferred"
        for event in bundles.bundle.events
    )


@pytest.mark.asyncio
async def test_custom_memory_capacity_keeps_offered_elements_action_valid(
    tmp_path: Path,
) -> None:
    base_snapshot = _snapshot(
        second_element_id="second",
        lineage_id="target",
        second_lineage_id="second",
    )
    third = replace(
        base_snapshot.elements[1],
        id="third",
        label="Third",
        rendered_text="Third",
        bounds=BoundingBox(x=250, y=10, width=100, height=30),
        lineage_id="third",
    )
    fourth = replace(
        third,
        id="fourth",
        label="Fourth",
        rendered_text="Fourth",
        bounds=BoundingBox(x=370, y=10, width=100, height=30),
        lineage_id="fourth",
    )
    snapshot = replace(
        base_snapshot,
        elements=(*base_snapshot.elements, third, fourth),
    )

    class RecordingCognitiveAgent(FakeCognitiveAgent):
        def __init__(self) -> None:
            super().__init__(())
            self.contexts: list[CognitiveRunContext] = []

        def update_context(self, context: CognitiveRunContext) -> None:
            self.contexts.append(context)

        async def decide(
            self, goal: str, observation: ProgressiveObservation
        ) -> object:
            del goal
            self.observations.append(observation)
            if len(self.observations) == 1:
                return CognitiveDecision(action={"kind": "wait"}, reason="Observe more.")
            return CognitiveDecision(
                action={
                    "kind": "interact",
                    "element_id": observation.newly_revealed_elements[0].id,
                },
                reason="Use offered control.",
            )

    cognitive = RecordingCognitiveAgent()
    attention = ProgressiveAttentionPolicy(
        AttentionPolicyConfig(batch_size=2, cross_region_exploration=0)
    )
    agent = _agent(
        tmp_path,
        FakeObservationProvider((snapshot,)),
        cognitive,
        FakeVerifier(
            (VerificationResult(verified=False), VerificationResult(verified=True))
        ),
        FakeBundleFactory(),
        attention_policy=attention,
        memory_policy=MemoryPolicy(MemoryPolicyConfig(working_capacity=1)),
    )

    spec = _spec(timeout_seconds=None)
    spec = replace(
        spec,
        scenario=replace(
            spec.scenario,
            verifier=FixtureStateVerifierSpec(
                type="fixture-state",
                resource="workspace",
                field="invite_status",
                operator="equals",
                expected_fixture_key="invite_email",
            ),
        ),
    )
    result = await agent.execute(spec)

    assert result.outcome.kind == "verified-success"
    assert result.state.attention.memory_capacity == 1
    assert [context.working_memory_capacity for context in cognitive.contexts] == [1, 1]
    offered = cognitive.observations[1]
    assert offered.remembered_elements == ()
    for element in (
        *offered.newly_revealed_elements,
        *offered.remembered_elements,
    ):
        result.state.attention.validate_action(
            InteractWithElement(element_id=element.id), snapshot
        )


@pytest.mark.asyncio
async def test_memory_evicted_noticed_elements_do_not_count_as_fresh_coverage(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(
        second_element_id="second",
        lineage_id="target",
        second_lineage_id="second",
    )
    bundles = FakeBundleFactory()
    decision = CognitiveDecision(
        action={"kind": "abandon", "reason": "No further controls."},
        reason="No further controls.",
    )
    cognitive = FakeCognitiveAgent((decision,) * 6)
    agent = _agent(
        tmp_path,
        FakeObservationProvider((snapshot,)),
        cognitive,
        FakeVerifier((VerificationResult(verified=False),)),
        bundles,
        attention_policy=ProgressiveAttentionPolicy(
            AttentionPolicyConfig(batch_size=2, cross_region_exploration=0)
        ),
        memory_policy=MemoryPolicy(MemoryPolicyConfig(working_capacity=1)),
    )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert result.outcome.kind == "agent-abandoned"
    assert result.terminal_reason == "all visible elements examined without progress"
    assert result.state.attention.noticed_ids == frozenset({"target", "second"})
    assert len(cognitive.observations) == 1
    prominence_events = [
        event
        for event in bundles.bundle.events
        if isinstance(event, dict) and event.get("kind") == "prominence-recorded"
    ]
    assert len(prominence_events) == 1


@pytest.mark.asyncio
async def test_cognitive_completes_from_grouped_stat_rows_retained_in_memory(
    tmp_path: Path,
) -> None:
    base_snapshot = _snapshot(second_element_id="countries")
    snapshot = replace(
        base_snapshot,
        elements=(
            replace(
                base_snapshot.elements[0],
                label="48,000+ cities calibrated",
                rendered_text="48,000+ cities calibrated",
            ),
            replace(
                base_snapshot.elements[1],
                label="130+ countries",
                rendered_text="130+ countries",
            ),
        ),
    )

    class GroupedEvidenceClient:
        endpoint_origin = "https://llm.example.test"

        def __init__(self) -> None:
            self.payloads: list[dict[str, object]] = []

        async def complete(self, schema, messages, model, role):
            del model, role
            payload = json.loads(messages[-1].content)
            self.payloads.append(payload)
            labels = {
                item["label"]
                for item in (
                    *payload["newly_revealed_elements"],
                    *payload["remembered_elements"],
                )
            }
            action = (
                "complete"
                if {"48,000+ cities calibrated", "130+ countries"} <= labels
                else "wait"
            )
            return schema.model_validate(
                {
                    "action": action,
                    "reason": "Current and remembered visible evidence covers the goal.",
                }
            )

    client = GroupedEvidenceClient()
    agent = _agent(
        tmp_path,
        FakeObservationProvider((snapshot,)),
        StructuredCognitiveAgent(client, model="cognitive-model"),
        FakeVerifier((VerificationResult(verified=True),)),
        FakeBundleFactory(),
        attention_policy=ProgressiveAttentionPolicy(
            AttentionPolicyConfig(batch_size=1, cross_region_exploration=0)
        ),
        memory_policy=MemoryPolicy(MemoryPolicyConfig(working_capacity=2)),
    )

    result = await agent.execute(_spec(timeout_seconds=None))

    assert result.outcome.kind == "verified-success"
    assert len(client.payloads) == 2
    assert client.payloads[1]["remembered_elements"]
    assert "verifier" not in json.dumps(client.payloads).lower()


@pytest.mark.parametrize(
    ("outcome", "terminal_reason", "expected_valid", "expected_invalid_reason"),
    (
        (VerifiedSuccess(), None, True, None),
        (AgentAbandoned(reason="No safe path."), "No safe path.", True, None),
        (BudgetExhausted(), "attention budget exhausted", True, None),
        (BudgetExhausted(), "interaction budget exhausted", True, None),
        (
            BudgetExhausted(),
            "model call budget exhausted",
            False,
            "budget-exhausted: model call budget exhausted",
        ),
        (
            BudgetExhausted(),
            "repeated successful fixture input detected",
            False,
            "budget-exhausted: repeated successful fixture input detected",
        ),
        (
            BudgetExhausted(),
            "provider runtime limit exhausted",
            False,
            "budget-exhausted: provider runtime limit exhausted",
        ),
    ),
)
def test_ux_sample_validity_separates_human_and_automation_budgets(
    outcome: object,
    terminal_reason: str | None,
    expected_valid: bool,
    expected_invalid_reason: str | None,
) -> None:
    validity = _ux_sample_validity(outcome, terminal_reason)  # type: ignore[arg-type]

    assert validity == (expected_valid, expected_invalid_reason)


@pytest.mark.asyncio
async def test_target_interaction_counts_as_progress_and_allows_completion(
    tmp_path: Path,
) -> None:
    """The first target click is patience; identical repeats get rejected.

    Regression: a successful click on the verifier target (e.g. a Download CV
    link that changes nothing in the DOM) previously counted as no progress,
    so three identical clicks abandoned the run before the model could
    declare completion. Now the first click is protected from no-progress
    abandonment, identical repeats are rejected with explicit feedback, and
    the model still gets its chance to declare completion.
    """

    snapshot = _snapshot(
        "viewport-1",
        "target",
        lineage_id="target-lineage",
        label="Invite teammate",
    )
    snapshots = (snapshot, snapshot, snapshot, snapshot)
    provider = FakeObservationProvider(
        snapshots,
        results=(PlatformActionResult(True, "http://fixture.test", 1, state_changed=False),),
    )
    bundles = FakeBundleFactory()
    interact = CognitiveDecision(
        action={"kind": "interact", "element_id": "target"},
        reason="Click the target.",
    )
    complete = CognitiveDecision(
        action={"kind": "complete"},
        reason="Target is visible.",
    )
    agent = _agent(
        tmp_path,
        provider,
        FakeCognitiveAgent((interact, interact, interact, complete)),
        FakeVerifier((VerificationResult(verified=True),)),
        bundles,
        attention_policy=RepeatingAttentionPolicy(),
    )

    result = await agent.execute(
        _spec(max_steps=10, timeout_seconds=None, evaluation_label="Invite teammate")
    )

    assert result.outcome.kind == "verified-success", result.terminal_reason
    assert len(provider.executed) == 1
    rejections = [
        event
        for event in bundles.bundle.events
        if isinstance(event, dict) and event.get("kind") == "action-rejected"
    ]
    assert len(rejections) == 2
    assert any(
        isinstance(event, dict) and event.get("kind") == "target-interaction-progress"
        for event in bundles.bundle.events
    )


@pytest.mark.asyncio
async def test_no_change_interaction_repeat_is_rejected_with_feedback(
    tmp_path: Path,
) -> None:
    """Repeating an interaction that changed nothing is rejected at the boundary.

    The first click succeeds silently (no interface change). The model repeats
    it; the boundary rejects the repeat with explicit feedback instead of
    silently executing another no-op, and the platform executes it only once.
    """

    snapshot = _snapshot(
        "viewport-1",
        "target",
        lineage_id="target-lineage",
        label="Invite teammate",
    )
    snapshots = (snapshot, snapshot, snapshot, snapshot)
    provider = FakeObservationProvider(
        snapshots,
        results=(PlatformActionResult(True, "http://fixture.test", 1, state_changed=False),),
    )
    bundles = FakeBundleFactory()
    cognitive = ContextRecordingCognitiveAgent(
        (
            CognitiveDecision(
                action={"kind": "interact", "element_id": "target"},
                reason="Click the target.",
            ),
            CognitiveDecision(
                action={"kind": "interact", "element_id": "target"},
                reason="Click it again.",
            ),
            CognitiveDecision(action={"kind": "complete"}, reason="Done."),
        )
    )
    agent = _agent(
        tmp_path,
        provider,
        cognitive,
        FakeVerifier((VerificationResult(verified=True),)),
        bundles,
        attention_policy=RepeatingAttentionPolicy(),
    )

    result = await agent.execute(
        _spec(max_steps=8, timeout_seconds=None, evaluation_label="Invite teammate")
    )

    assert result.outcome.kind == "verified-success", result.terminal_reason
    assert len(provider.executed) == 1
    rejections = [
        event
        for event in bundles.bundle.events
        if isinstance(event, dict)
        and event.get("kind") == "action-rejected"
        and "previous interaction" in event.get("reason", "")
    ]
    assert len(rejections) == 1


@pytest.mark.asyncio
async def test_web_verifier_matches_nonbreaking_space_labels(tmp_path: Path) -> None:
    """Verifier containment normalizes NBSP so real headings are not hidden."""

    provider = FakeObservationProvider(
        (_snapshot(label="Muslim\u00a0Pedia"),)
    )
    session = await provider.start_session(_config(_spec(), tmp_path))
    verifier = WebVerifier(
        VisibleResultVerifierSpec(
            type="visible-result", text="Muslim Pedia", role="button"
        ),
        observation_provider=provider,
        snapshot_extractor=lambda capture: (
            capture.snapshot
            if capture.snapshot is not None
            else (_ for _ in ()).throw(ValueError("missing snapshot"))
        ),
    )

    result = await verifier.verify(session)

    assert result.verified
    assert result.evidence_ids == (
        "viewport:viewport-1-verification-1:element:target",
    )


class ContextRecordingCognitiveAgent(FakeCognitiveAgent):
    def __init__(self, decisions: tuple[object, ...] = ()) -> None:
        super().__init__(decisions)
        self.contexts: list[CognitiveRunContext] = []

    def update_context(self, context: CognitiveRunContext) -> None:
        self.contexts.append(context)


@pytest.mark.asyncio
async def test_failed_complete_verification_is_reported_to_next_decision(
    tmp_path: Path,
) -> None:
    """After an unverified completion, the model is told verified=false."""

    provider = FakeObservationProvider(
        tuple(_snapshot(f"viewport-{index}", "target", lineage_id="target-lineage") for index in range(1, 5)),
        results=(
            PlatformActionResult(True, "http://fixture.test", 1, state_changed=False),
            PlatformActionResult(True, "http://fixture.test", 1, state_changed=False),
            PlatformActionResult(True, "http://fixture.test", 1, state_changed=False),
        ),
    )
    bundles = FakeBundleFactory()
    cognitive = ContextRecordingCognitiveAgent(
        (
            CognitiveDecision(action={"kind": "complete"}, reason="Claim."),
            CognitiveDecision(
                action={"kind": "interact", "element_id": "target"},
                reason="Explore more.",
            ),
            CognitiveDecision(action={"kind": "wait"}, reason="Observe."),
            CognitiveDecision(action={"kind": "wait"}, reason="Observe."),
            CognitiveDecision(action={"kind": "wait"}, reason="Observe."),
            CognitiveDecision(action={"kind": "wait"}, reason="Observe."),
        )
    )
    agent = _agent(
        tmp_path,
        provider,
        cognitive,
        FakeVerifier(()),
        bundles,
        attention_policy=RepeatingAttentionPolicy(),
    )

    result = await agent.execute(
        _spec(max_steps=8, timeout_seconds=None, evaluation_label="Invite teammate")
    )

    assert result.outcome.kind == "budget-exhausted", result.terminal_reason
    complete_feedback = [
        context.previous_action_result
        for context in cognitive.contexts
        if isinstance(context.previous_action_result, dict)
        and context.previous_action_result.get("verified") is False
    ]
    assert complete_feedback, "failed completion must surface verified=false"

@pytest.mark.asyncio
async def test_target_interaction_surfaces_target_engaged_to_next_decision(
    tmp_path: Path,
) -> None:
    """Acting on the goal's own control is reported as target_engaged.

    Regression: a "Download CV" link whose click leaves the page unchanged
    reported only state_changed: false, so the model read a successful
    interaction with the target as a failed attempt and looped on it until
    the run was abandoned, never declaring completion.
    """

    snapshot = _snapshot(
        "viewport-1",
        "target",
        lineage_id="target-lineage",
        label="Invite teammate",
    )
    provider = FakeObservationProvider(
        (snapshot, snapshot, snapshot, snapshot),
        results=(PlatformActionResult(True, "http://fixture.test", 1, state_changed=False),),
    )
    cognitive = ContextRecordingCognitiveAgent(
        (
            CognitiveDecision(
                action={"kind": "interact", "element_id": "target"},
                reason="Click the target.",
            ),
            CognitiveDecision(action={"kind": "wait"}, reason="Observe."),
        )
    )
    agent = _agent(
        tmp_path,
        provider,
        cognitive,
        FakeVerifier(()),
        FakeBundleFactory(),
        attention_policy=RepeatingAttentionPolicy(),
    )

    await agent.execute(
        _spec(max_steps=8, timeout_seconds=None, evaluation_label="Invite teammate")
    )

    engaged = [
        context.previous_action_result
        for context in cognitive.contexts
        if isinstance(context.previous_action_result, dict)
        and context.previous_action_result.get("target_engaged") is True
    ]
    assert engaged, "interacting with the evaluation target must report target_engaged"
    assert engaged[0]["succeeded"] is True
    # The semantic truth is still reported alongside it.
    assert engaged[0]["state_changed"] is False

class CompletesOnTargetEngaged(FakeCognitiveAgent):
    """Stands in for the cognitive-v3 rule: complete once the goal control
    has been engaged, instead of reading state_changed: false as a failure
    and retrying the same click until the run is abandoned.
    """

    def __init__(self) -> None:
        super().__init__(())
        self.contexts: list[CognitiveRunContext] = []

    def update_context(self, context: CognitiveRunContext) -> None:
        self.contexts.append(context)

    async def decide(self, goal: str, observation: ProgressiveObservation) -> object:
        feedback = self.contexts[-1].previous_action_result if self.contexts else None
        if isinstance(feedback, dict) and feedback.get("target_engaged") is True:
            return CognitiveDecision(action={"kind": "complete"}, reason="Goal control acted on.")
        return CognitiveDecision(
            action={"kind": "interact", "element_id": "target"},
            reason="Click the goal control.",
        )


@pytest.mark.asyncio
async def test_goal_target_click_completes_without_a_page_change(
    tmp_path: Path,
) -> None:
    """A run reaches completion after clicking the goal's own control.

    Replay of complete-download-cv: the click succeeds, the page
    legitimately does not change, and target_engaged is the only signal
    telling the next decision that the goal was satisfied. Without it the
    model re-reads state_changed: false as a failed attempt, retries the
    same control, trips the anti-repeat guard, and the run ends abandoned
    without ever proposing complete.
    """

    snapshot = _snapshot(
        "viewport-1",
        "target",
        lineage_id="target-lineage",
        label="Download CV",
    )
    provider = FakeObservationProvider(
        (snapshot, snapshot, snapshot),
        results=(PlatformActionResult(True, "http://fixture.test", 1, state_changed=False),),
    )
    cognitive = CompletesOnTargetEngaged()
    agent = _agent(
        tmp_path,
        provider,
        cognitive,
        FakeVerifier((VerificationResult(verified=True, evidence_ids=("v1",)),)),
        FakeBundleFactory(),
        result_evaluator=lambda candidate: replace(
            candidate,
            metrics=evaluate_run(candidate, EvaluationTarget("target")),
        ),
        attention_policy=RepeatingAttentionPolicy(),
    )

    result = await agent.execute(
        _spec(max_steps=8, timeout_seconds=None, evaluation_label="Download CV")
    )

    assert result.outcome.kind == "verified-success", result.terminal_reason
    assert result.agent_claimed_success is True
    assert any(
        isinstance(context.previous_action_result, dict)
        and context.previous_action_result.get("target_engaged") is True
        for context in cognitive.contexts
    )


@pytest.mark.asyncio
async def test_colour_change_verifier_is_constructible_without_baseline() -> None:
    """The CLI construction path must not demand a baseline up front.

    The run agent seeds the colour-change baseline from the run's *pre-action
    capture* (``ColourChangeVerifierSpec.set_baseline``) after the browser
    session starts, so a constructor that rejects ``baseline_capture=None``
    makes the opt-in verifier unusable in every real run: the CLI builds the
    verifier before any capture exists. Construction must succeed without a
    baseline; the requirement is enforced at ``verify()`` time instead.
    """

    from ux_analyzer.ports.observation import ObservationCapture
    from ux_analyzer.adapters.web.verifier import WebVerifier
    from ux_analyzer.adapters.web.verifier import WebVerificationError
    from ux_analyzer.ports.observation import ViewportSize

    def _colour_capture(*, background: str) -> ObservationCapture:
        snapshot = _snapshot()
        return ObservationCapture(
            session_id="run-1",
            viewport_id=snapshot.id,
            url="http://fixture.test/app/run/improved",
            title="Fixture",
            viewport=ViewportSize(width=1024, height=768),
            screenshot=b"shot",
            snapshot=snapshot,
            document_background=background,
            body_background=background,
            viewport_background=background,
        )

    class ColourProvider:
        def __init__(self, captures: tuple[ObservationCapture, ...]) -> None:
            self.captures = list(captures)

        async def start_session(self, config: ObservationSessionConfig) -> SessionHandle:
            del config
            raise NotImplementedError

        async def capture(self, session: SessionHandle) -> ObservationCapture:
            del session
            if not self.captures:
                raise AssertionError("no captures left")
            return self.captures.pop(0)

    baseline = _colour_capture(background="#ffffff")
    flipped = _colour_capture(background="#111418")
    provider = ColourProvider((flipped,))

    # Construction without a baseline: must NOT raise (the CLI path).
    verifier = WebVerifier(
        ColourChangeVerifierSpec(type="colour-change"),
        observation_provider=provider,
        snapshot_extractor=lambda capture: capture.snapshot,
    )

    # verify() before any baseline is defined must fail loudly, not silently
    # compare against nothing.
    with pytest.raises(WebVerificationError, match="baseline"):
        await verifier.verify(_session_handle())

    # After the run agent seeds the baseline from the pre-action capture, the
    # same instance verifies a real theme flip.
    verifier.set_baseline(baseline)
    result = await verifier.verify(_session_handle())
    assert result.verified
    assert result.before_state["document_background"] == "#ffffff"
    assert result.after_state["document_background"] == "#111418"
