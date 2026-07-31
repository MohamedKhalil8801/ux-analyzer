from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from ux_analyzer.adapters.web.verifier import WebVerifier
from ux_analyzer.application.evaluation import (
    EvaluationTarget,
    evaluate_run,
    evaluation_target_for,
)
from ux_analyzer.application.run_agent import RunAgent, RunFinalizationError
from ux_analyzer.domain.attention import Abandon, ProgressiveObservation
from ux_analyzer.domain.benchmark import (
    ApplicationVersion,
    ApplicationVersionKind,
    Budget,
    ExperimentPolicy,
    FixtureInputs,
    FixtureStateVerifierSpec,
    Persona,
    Scenario,
    VisibleResultVerifierSpec,
)
from ux_analyzer.domain.interface import (
    BoundingBox,
    ElementSnapshot,
    PrivateExecutionReference,
    ViewportSnapshot,
)
from ux_analyzer.domain.run import VerificationResult
from ux_analyzer.ports.artifacts import ArtifactReference, BundleManifest
from ux_analyzer.ports.models import (
    ModelCallRecord,
    ModelRole,
    RetryEvent,
    TokenUsage,
)
from ux_analyzer.ports.observation import (
    ObservationCapture,
    ObservationProviderError,
    ObservationSessionConfig,
    PlatformAction,
    PlatformActionResult,
    SafetyBlocked,
    SessionHandle,
    ViewportSize,
)
from ux_analyzer.ports.observation import TestAccountId as AccountId
from ux_analyzer.providers.attention_policy import ObservationSelection
from ux_analyzer.providers.cognitive import CognitiveDecision
from ux_analyzer.providers.prominence import ProminenceResult
from ux_analyzer.storage.run_bundle import FilesystemRunBundleWriter


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
    ) -> None:
        self.snapshots = snapshots
        self.results = list(results)
        self.failure = failure
        self.start_failure = start_failure
        self.started = 0
        self.reset_count = 0
        self.capture_count = 0
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
        return ObservationCapture(
            session_id=session.session_id,
            viewport_id=snapshot.id,
            url="http://fixture.test/app/run/improved",
            title="Fixture",
            viewport=session.viewport,
            screenshot=f"screenshot-{snapshot.id}".encode(),
            snapshot=snapshot,
        )

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


class FakeAttentionPolicy:
    def next_observation(
        self,
        state: Any,
        snapshot: ViewportSnapshot,
        scores: tuple[ProminenceResult, ...],
        coarse_scent: object,
        rng: object,
    ) -> ObservationSelection:
        del scores, coarse_scent, rng
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
        self.finalized = False
        self.aborted = False
        self.final_result: object | None = None
        self.finalize_error: BaseException | None = None

    def append_event(self, event: object) -> int:
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
    def __init__(self, output: Path) -> None:
        self.output = output

    def start(self, spec) -> FilesystemRunBundleWriter:
        return FilesystemRunBundleWriter.start(
            self.output,
            BundleManifest.from_run_spec(
                spec,
                endpoint_origin="https://llm.example.test",
                model_ids={"cognitive": "cognitive-model"},
                prompt_versions={"cognitive": "cognitive-v1"},
                package_version="0.1.0",
            ),
        )


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
) -> ViewportSnapshot:
    elements = [
        ElementSnapshot(
            id=element_id,
            role="button",
            label="Invite teammate",
            bounds=BoundingBox(x=10, y=10, width=100, height=30),
            visibility_fraction=1,
            actionable=True,
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
                bounds=BoundingBox(x=130, y=10, width=100, height=30),
                visibility_fraction=1,
                actionable=True,
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
    )


def _spec(*, max_steps: int = 6, timeout_seconds: float = 1) -> object:
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
        fixture_inputs=FixtureInputs(values={"invite_email": "person@example.com"}),
        budget=Budget(
            max_steps=max_steps,
            max_observations=max_steps,
            max_interactions=max_steps,
            timeout_seconds=timeout_seconds,
        ),
        verifier=VisibleResultVerifierSpec(type="visible-result", text="sent"),
        safeguards=(),
        eligible_persona_ids=("persona",),
        expected_evidence=(),
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


def _agent(
    tmp_path: Path,
    provider: FakeObservationProvider,
    cognitive: FakeCognitiveAgent,
    verifier: FakeVerifier,
    bundles: FakeBundleFactory,
    *,
    timeout_seconds: float = 1,
    model_record_source: FakeModelRecordSource | None = None,
    result_evaluator=None,
) -> RunAgent:
    spec = _spec(timeout_seconds=timeout_seconds)
    return RunAgent(
        observation_provider=provider,
        prominence_provider=FakeProminenceProvider(),
        attention_policy=FakeAttentionPolicy(),
        cognitive_agent=cognitive,
        verifier=verifier,
        bundle_factory=bundles,
        session_config_factory=lambda _: _config(spec, tmp_path),
        model_record_source=model_record_source,
        result_evaluator=result_evaluator,
    )


@pytest.mark.asyncio
async def test_verified_success_records_claim_separately_and_finalizes_after_terminal_event(
    tmp_path: Path,
) -> None:
    provider = FakeObservationProvider((_snapshot(),))
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
            )
        ),
        verifier,
        bundles,
    )

    result = await agent.execute(_spec())

    assert result.outcome.kind == "verified-success"
    assert result.verification.verified
    assert result.agent_claimed_success is False
    assert result.state.attention.confidence == pytest.approx(0.55)
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
    assert bundles.bundle.finalized
    assert bundles.bundle.events[-1].kind == "run-terminated"
    assert provider.ended == 0 or provider.started == 1


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
    ]
    assert len(provider.executed) < len(decisions)
    assert provider.capture_count == 5
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
    assert result.evidence_ids == ("viewport:viewport-1:element:target",)
    assert provider.capture_count == 1


@pytest.mark.asyncio
async def test_session_start_failure_does_not_cleanup_previous_run_session(
    tmp_path: Path,
) -> None:
    provider = FakeObservationProvider((_snapshot(),))
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
        provider = FakeObservationProvider(
            (
                ViewportSnapshot(
                    id="viewport-1",
                    provider_id="fake-observer",
                    elements=(),
                ),
            )
        )
        cognitive = FakeCognitiveAgent(())

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
        else "internal-error"
    )
    assert result.metrics is None
    assert result.findings is None
    assert result.evaluation_failure_reason == "result evaluation failed: ValueError"
    assert result.state.events[-1].kind == "run-terminated"
    assert bundle.is_dir()
    assert not (output / ".staging" / result.run_id).exists()
    persisted = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    assert persisted["metrics"] is None
    assert persisted["findings"] is None
    assert (
        persisted["evaluation_failure_reason"] == "result evaluation failed: ValueError"
    )
    assert result.run_id not in result.evaluation_failure_reason
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
