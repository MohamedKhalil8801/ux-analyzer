"""Application use case for one complete attention-guided benchmark run."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from ux_analyzer.application.action_validation import (
    ActionValidationError,
    ValidatedAction,
    validate_action,
)
from ux_analyzer.application.state_updates import (
    ApplicationState,
    StateUpdateConfig,
    apply_failure,
    apply_interaction_result,
    apply_observation,
)
from ux_analyzer.domain.attention import Abandon, InteractWithElement
from ux_analyzer.domain.interface import (
    PrivateExecutionReference,
    ViewportSnapshot,
)
from ux_analyzer.domain.run import (
    ActionExecuted,
    ActionProposed,
    AgentAbandoned,
    ArtifactChecksum,
    BudgetExhausted,
    InternalError,
    ModelFailure,
    ObservationRecorded,
    ProviderFailure,
    ProviderManifest,
    RunEvent,
    RunOutcome,
    RunSpec,
    RunStarted,
    RunState,
    RunTerminated,
    TimedOut,
    VerificationRecorded,
    VerificationResult,
    VerifiedSuccess,
    ViewportCaptured,
)
from ux_analyzer.domain.run import (
    SafetyBlocked as SafetyBlockedOutcome,
)
from ux_analyzer.ports.artifacts import ArtifactReference, RunBundleWriter
from ux_analyzer.ports.models import CoarseScentEvaluator, FullScentEvaluator
from ux_analyzer.ports.observation import (
    ObservationCapture,
    ObservationProvider,
    ObservationProviderError,
    ObservationSessionConfig,
    SafetyBlocked,
    SessionHandle,
)
from ux_analyzer.ports.verification import VerificationProvider
from ux_analyzer.providers.attention_policy import ObservationSelection
from ux_analyzer.providers.memory import MemoryPolicy
from ux_analyzer.providers.prominence import ProminenceResult


class ProminenceProvider(Protocol):
    """Application-facing prominence scoring capability."""

    def score(self, snapshot: ViewportSnapshot) -> Sequence[ProminenceResult]: ...


class AttentionPolicy(Protocol):
    """Application-facing progressive observation policy."""

    def next_observation(
        self,
        state: object,
        snapshot: ViewportSnapshot,
        scores: Sequence[ProminenceResult],
        coarse_scent: object,
        rng: random.Random,
    ) -> ObservationSelection: ...


class CognitiveAgent(Protocol):
    """Application-facing decision capability."""

    async def decide(self, goal: str, observation: object) -> object: ...


class RunBundleFactory(Protocol):
    """Creates one append-only bundle for one run."""

    def start(self, spec: RunSpec) -> RunBundleWriter: ...


SessionConfigFactory = Callable[[RunSpec], ObservationSessionConfig]
SnapshotExtractor = Callable[[ObservationCapture], ViewportSnapshot]
ProviderManifestFactory = Callable[[RunSpec], Sequence[ProviderManifest]]


@dataclass(frozen=True, slots=True)
class RunResult:
    """Closed result returned after bundle finalization or abort."""

    run_id: str
    outcome: RunOutcome
    verification: VerificationResult
    agent_claimed_success: bool
    state: RunState
    bundle_path: object | None = None
    terminal_reason: str | None = None

    @property
    def claimed_success(self) -> bool:
        """Compatibility alias for callers using shorter claim vocabulary."""

        return self.agent_claimed_success


@dataclass(frozen=True, slots=True)
class _Execution:
    state: RunState
    outcome: RunOutcome
    verification: VerificationResult | None
    agent_claimed_success: bool
    terminal_reason: str | None


@dataclass(slots=True)
class _RunContext:
    """Mutable per-execution cursor; never shared between run calls."""

    state: RunState
    application_state: ApplicationState
    session: SessionHandle | None = None


class RunAgent:
    """Coordinate one isolated run through platform and model ports."""

    def __init__(
        self,
        *,
        observation_provider: ObservationProvider,
        prominence_provider: ProminenceProvider,
        attention_policy: AttentionPolicy,
        cognitive_agent: CognitiveAgent,
        verifier: VerificationProvider,
        bundle_factory: RunBundleFactory | None = None,
        bundle_writer_factory: RunBundleFactory | None = None,
        session_config_factory: SessionConfigFactory | None = None,
        session_factory: SessionConfigFactory | None = None,
        coarse_scent_evaluator: CoarseScentEvaluator | None = None,
        full_scent_evaluator: FullScentEvaluator | None = None,
        memory_policy: MemoryPolicy | None = None,
        state_update_config: StateUpdateConfig | None = None,
        snapshot_extractor: SnapshotExtractor | None = None,
        provider_manifest_factory: ProviderManifestFactory | None = None,
    ) -> None:
        selected_bundle_factory = bundle_factory or bundle_writer_factory
        selected_session_factory = session_config_factory or session_factory
        if selected_bundle_factory is None:
            raise ValueError("run agent needs bundle factory")
        if selected_session_factory is None:
            raise ValueError("run agent needs session config factory")
        self.observation_provider = observation_provider
        self.prominence_provider = prominence_provider
        self.attention_policy = attention_policy
        self.cognitive_agent = cognitive_agent
        self.verifier = verifier
        self.bundle_factory = selected_bundle_factory
        self.session_config_factory = selected_session_factory
        self.coarse_scent_evaluator = coarse_scent_evaluator
        self.full_scent_evaluator = full_scent_evaluator
        self.memory_policy = memory_policy
        self.state_update_config = state_update_config or StateUpdateConfig()
        self.snapshot_extractor = snapshot_extractor or _snapshot_from_capture
        self.provider_manifest_factory = provider_manifest_factory

    async def execute(self, spec: RunSpec) -> RunResult:
        """Execute, independently verify, finalize, and always clean up one run."""

        writer: RunBundleWriter | None = None
        initial_state = RunState.initial(spec)
        context = _RunContext(
            state=initial_state,
            application_state=ApplicationState.from_attention(initial_state.attention),
        )
        artifact_checksums: list[ArtifactChecksum] = []
        deadline = (
            asyncio.get_running_loop().time() + spec.scenario.budget.timeout_seconds
        )

        try:
            writer = self.bundle_factory.start(spec)
            artifact_checksums.append(
                _artifact_checksum(writer.write_artifact("run-start.txt", spec.run_id))
            )
            context.state = _record(
                context.state,
                RunStarted(run_id=spec.run_id),
                writer,
            )

            try:
                execution = await asyncio.wait_for(
                    self._run(spec, context, writer, artifact_checksums),
                    timeout=spec.scenario.budget.timeout_seconds,
                )
            except TimeoutError:
                execution = _Execution(
                    state=context.state,
                    outcome=TimedOut(),
                    verification=None,
                    agent_claimed_success=False,
                    terminal_reason="run timeout exceeded",
                )
            except BaseException as error:
                execution = _Execution(
                    state=context.state,
                    outcome=_outcome_for_error(error),
                    verification=None,
                    agent_claimed_success=False,
                    terminal_reason=_safe_error_message(error),
                )

            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                execution = _timed_out_execution(execution, writer)
            else:
                try:
                    execution = await asyncio.wait_for(
                        self._verify_terminal(
                            execution,
                            writer,
                            context.session,
                        ),
                        timeout=remaining,
                    )
                except TimeoutError:
                    execution = _timed_out_execution(execution, writer)
            return self._finalize(spec, execution, writer, artifact_checksums)
        except BaseException as error:
            if writer is not None:
                _abort_bundle(
                    writer, f"run execution failed: {_safe_error_message(error)}"
                )
            raise
        finally:
            await self._end_session(context.session)

    async def _run(
        self,
        spec: RunSpec,
        context: _RunContext,
        writer: RunBundleWriter,
        artifact_checksums: list[ArtifactChecksum],
    ) -> _Execution:
        session = await self.observation_provider.start_session(
            self.session_config_factory(spec)
        )
        context.session = session
        await self.observation_provider.reset(session)
        await self._capture(context, writer, artifact_checksums)
        rng = random.Random(spec.seed)
        claimed_success = False

        while True:
            application_state = context.application_state
            if application_state.budgets.steps <= 0 or (
                application_state.budgets.observations <= 0
            ):
                return _Execution(
                    state=context.state,
                    outcome=BudgetExhausted(),
                    verification=None,
                    agent_claimed_success=claimed_success,
                    terminal_reason="attention budget exhausted",
                )
            snapshot = context.state.current_snapshot
            if snapshot is None:
                raise RuntimeError("run has no current viewport")

            scores = tuple(self.prominence_provider.score(snapshot))
            coarse_scent: object = ()
            if self.coarse_scent_evaluator is not None:
                coarse_scent = await self.coarse_scent_evaluator.evaluate(
                    spec.scenario.goal, snapshot
                )
                writer.append_event(
                    {
                        "kind": "coarse-scent-recorded",
                        "scores": tuple(
                            {"element_id": item.element_id, "score": item.score}
                            for item in coarse_scent
                        ),
                    }
                )

            selection = self.attention_policy.next_observation(
                application_state.attention,
                snapshot,
                scores,
                coarse_scent,
                rng,
            )
            observation = selection.observation
            context.state = _record(
                context.state,
                ObservationRecorded(observation=observation),
                writer,
            )
            context.application_state = _require_application_state(
                apply_observation(
                    application_state,
                    observation,
                    snapshot=snapshot,
                    memory_policy=self.memory_policy,
                )
            )

            if context.application_state.budgets.steps <= 0:
                return _Execution(
                    state=context.state,
                    outcome=BudgetExhausted(),
                    verification=None,
                    agent_claimed_success=claimed_success,
                    terminal_reason="attention budget exhausted",
                )

            if self.full_scent_evaluator is not None:
                full_scent = await self.full_scent_evaluator.evaluate(
                    spec.scenario.goal,
                    context.application_state.attention,
                    snapshot,
                )
                writer.append_event(
                    {
                        "kind": "full-scent-recorded",
                        "scores": tuple(
                            {"element_id": item.element_id, "score": item.score}
                            for item in full_scent
                        ),
                    }
                )

            decision = await self.cognitive_agent.decide(
                spec.scenario.goal, observation
            )
            claim = _agent_claim(decision)
            claimed_success = claimed_success or claim
            if claim:
                writer.append_event({"kind": "agent-claim", "claimed_success": True})

            try:
                validated = validate_action(
                    decision,
                    context.application_state.attention,
                    snapshot,
                    fixture_inputs=spec.scenario.fixture_inputs,
                )
            except ActionValidationError as error:
                writer.append_event({"kind": "action-rejected", "reason": str(error)})
                context.application_state = _require_application_state(
                    apply_failure(
                        context.application_state,
                        reason="invalid-action",
                        config=self.state_update_config,
                        memory_policy=self.memory_policy,
                    )
                )
                if context.application_state.abandoned:
                    return _Execution(
                        state=context.state,
                        outcome=AgentAbandoned(
                            reason=context.application_state.abandonment_reason
                            or "invalid action threshold crossed"
                        ),
                        verification=None,
                        agent_claimed_success=claimed_success,
                        terminal_reason=context.application_state.abandonment_reason,
                    )
                continue

            context.state = _record(
                context.state,
                ActionProposed(action=validated.domain_action),
                writer,
            )
            if isinstance(validated.domain_action, Abandon):
                return _Execution(
                    state=context.state,
                    outcome=AgentAbandoned(reason=validated.domain_action.reason),
                    verification=None,
                    agent_claimed_success=claimed_success,
                    terminal_reason=validated.domain_action.reason,
                )

            if validated.platform_action is None:
                context.state = _record(
                    context.state,
                    ActionExecuted(
                        action=validated.domain_action,
                        viewport_id=snapshot.id,
                        succeeded=True,
                    ),
                    writer,
                )
                context.application_state = _require_application_state(
                    apply_interaction_result(
                        context.application_state,
                        validated.domain_action,
                        True,
                        snapshot=snapshot,
                        config=self.state_update_config,
                        memory_policy=self.memory_policy,
                    )
                )
                continue

            result = await self.observation_provider.execute(
                session, validated.platform_action
            )
            context.state = _record(
                context.state,
                ActionExecuted(
                    action=validated.domain_action,
                    viewport_id=snapshot.id,
                    execution_reference=_execution_reference(snapshot, validated),
                    succeeded=result.succeeded,
                    error=result.error,
                ),
                writer,
            )
            context.application_state = _require_application_state(
                apply_interaction_result(
                    context.application_state,
                    validated.domain_action,
                    result,
                    snapshot=snapshot,
                    config=self.state_update_config,
                    memory_policy=self.memory_policy,
                )
            )
            if context.application_state.abandoned:
                return _Execution(
                    state=context.state,
                    outcome=AgentAbandoned(
                        reason=context.application_state.abandonment_reason
                        or "abandonment threshold crossed"
                    ),
                    verification=None,
                    agent_claimed_success=claimed_success,
                    terminal_reason=context.application_state.abandonment_reason,
                )

            if result.state_changed:
                verification = await self._verify(context.state, writer, session)
                context.state = verification[0]
                if verification[1].verified:
                    return _Execution(
                        state=context.state,
                        outcome=VerifiedSuccess(),
                        verification=verification[1],
                        agent_claimed_success=claimed_success,
                        terminal_reason=None,
                    )

            await self._capture(context, writer, artifact_checksums)

    async def _capture(
        self,
        context: _RunContext,
        writer: RunBundleWriter,
        artifact_checksums: list[ArtifactChecksum],
    ) -> None:
        session = context.session
        if session is None:
            raise RuntimeError("capture requires active session")
        capture = await self.observation_provider.capture(session)
        snapshot = self.snapshot_extractor(capture)
        artifact_checksums.append(
            _artifact_checksum(
                writer.write_artifact(f"{snapshot.id}.png", capture.screenshot)
            )
        )
        context.state = _record(
            context.state,
            ViewportCaptured(snapshot=snapshot),
            writer,
        )

    async def _verify(
        self,
        state: RunState,
        writer: RunBundleWriter,
        session: SessionHandle,
    ) -> tuple[RunState, VerificationResult]:
        result = await self.verifier.verify(session)
        return (
            _record(state, VerificationRecorded(result=result), writer),
            result,
        )

    async def _verify_terminal(
        self,
        execution: _Execution,
        writer: RunBundleWriter,
        session: SessionHandle | None,
    ) -> _Execution:
        if execution.verification is not None:
            return execution

        if session is None:
            result = VerificationResult(
                verified=False,
                details="independent verification unavailable",
            )
            state = _record(
                execution.state,
                VerificationRecorded(result=result),
                writer,
            )
            return replace(execution, state=state, verification=result)

        try:
            result = await self.verifier.verify(session)
        except BaseException as error:
            result = VerificationResult(
                verified=False,
                details="independent verification failed",
            )
            state = _record(
                execution.state,
                VerificationRecorded(result=result),
                writer,
            )
            return replace(
                execution,
                state=state,
                outcome=_outcome_for_error(error),
                terminal_reason=_safe_error_message(error),
                verification=result,
            )

        state = _record(
            execution.state,
            VerificationRecorded(result=result),
            writer,
        )
        outcome = VerifiedSuccess() if result.verified else execution.outcome
        return replace(execution, state=state, outcome=outcome, verification=result)

    def _finalize(
        self,
        spec: RunSpec,
        execution: _Execution,
        writer: RunBundleWriter,
        artifact_checksums: Sequence[ArtifactChecksum],
    ) -> RunResult:
        verification = execution.verification or VerificationResult(
            verified=False,
            details="independent verification unavailable",
        )
        state = execution.state
        if state.verification is None:
            state = _record(state, VerificationRecorded(result=verification), writer)
        manifests = self._provider_manifests(spec)
        terminal = RunTerminated(
            outcome=execution.outcome,
            verification=verification,
            provider_manifests=manifests,
            configuration_digest=spec.config_digest,
            artifact_checksums=tuple(artifact_checksums),
        )
        try:
            final_state = _record(state, terminal, writer)
        except BaseException as error:
            _abort_bundle(
                writer, f"terminal event failed: {_safe_error_message(error)}"
            )
            raise

        result = RunResult(
            run_id=spec.run_id,
            outcome=execution.outcome,
            verification=verification,
            agent_claimed_success=execution.agent_claimed_success,
            state=final_state,
            terminal_reason=execution.terminal_reason,
        )
        try:
            bundle_path = writer.finalize(result)
        except BaseException as error:
            bundle_path = _abort_bundle(
                writer,
                f"finalization failed: {_safe_error_message(error)}",
            )
        return replace(result, bundle_path=bundle_path)

    def _provider_manifests(self, spec: RunSpec) -> tuple[ProviderManifest, ...]:
        if self.provider_manifest_factory is not None:
            manifests = tuple(self.provider_manifest_factory(spec))
            if manifests:
                return manifests
        providers: list[tuple[object, str]] = [
            (self.observation_provider, "observation"),
        ]
        for provider, fallback_role in (
            (self.coarse_scent_evaluator, "coarse-scent"),
            (self.full_scent_evaluator, "full-scent"),
            (self.cognitive_agent, "cognitive"),
            (self.verifier, "verification"),
        ):
            if provider is not None:
                providers.append((provider, fallback_role))
        return tuple(_provider_manifest(provider, role) for provider, role in providers)

    async def _end_session(self, session: SessionHandle | None) -> None:
        if session is None:
            return
        try:
            await self.observation_provider.end_session(session)
        except BaseException:
            return


def _require_application_state(state: object) -> ApplicationState:
    if not isinstance(state, ApplicationState):
        raise RuntimeError("application state transition returned invalid state")
    return state


def _record(state: RunState, event: RunEvent, writer: RunBundleWriter) -> RunState:
    next_state = state.apply(event)
    writer.append_event(event)
    return next_state


def _snapshot_from_capture(capture: ObservationCapture) -> ViewportSnapshot:
    if capture.snapshot is None:
        raise RuntimeError("observation capture has no normalized viewport snapshot")
    return capture.snapshot


def _artifact_checksum(reference: ArtifactReference) -> ArtifactChecksum:
    return ArtifactChecksum(path=reference.path, sha256=reference.sha256)


def _execution_reference(
    snapshot: ViewportSnapshot,
    validated: ValidatedAction,
) -> PrivateExecutionReference | None:
    if not isinstance(validated.domain_action, InteractWithElement):
        return None
    return snapshot.element(validated.domain_action.element_id).execution_reference


def _agent_claim(decision: object) -> bool:
    return bool(
        getattr(
            decision,
            "claimed_success",
            getattr(decision, "claim_success", False),
        )
    )


def _provider_manifest(provider: object, role: str) -> ProviderManifest:
    manifest = getattr(provider, "manifest", None)
    if callable(manifest):
        manifest = manifest()
    if manifest is not None:
        model_id = getattr(manifest, "model_id", None)
        manifest_role = getattr(manifest, "role", role)
        role_value = getattr(manifest_role, "value", manifest_role)
        return ProviderManifest(
            provider_id=str(getattr(manifest, "provider_id", type(provider).__name__)),
            role=str(role_value),
            model_id=str(model_id) if model_id is not None else None,
            endpoint_origin=str(getattr(manifest, "endpoint_origin", "internal")),
            version=str(
                getattr(
                    manifest,
                    "provider_version",
                    getattr(provider, "version", "unknown"),
                )
            ),
        )
    return ProviderManifest(
        provider_id=str(getattr(provider, "id", type(provider).__name__)),
        role=role,
        model_id=None,
        endpoint_origin=str(getattr(provider, "endpoint_origin", "internal")),
        version=str(getattr(provider, "version", "unknown")),
    )


def _outcome_for_error(error: BaseException) -> RunOutcome:
    if isinstance(error, TimeoutError):
        return TimedOut()
    if isinstance(error, SafetyBlocked):
        return SafetyBlockedOutcome(reason=_safe_error_message(error))
    if isinstance(error, ObservationProviderError):
        return ProviderFailure(reason=_safe_error_message(error))
    if error.__class__.__name__ in {
        "ModelFailureError",
        "ModelProviderError",
    }:
        return ModelFailure(reason=_safe_error_message(error))
    if error.__class__.__name__ in {
        "ProviderFailure",
        "WebVerificationError",
    }:
        return ProviderFailure(reason=_safe_error_message(error))
    return InternalError(reason=_safe_error_message(error))


def _safe_error_message(error: BaseException) -> str:
    message = str(error).strip()
    return message or error.__class__.__name__


def _timed_out_execution(
    execution: _Execution,
    writer: RunBundleWriter,
) -> _Execution:
    result = VerificationResult(
        verified=False,
        details="independent verification unavailable before timeout",
    )
    state = _record(
        execution.state,
        VerificationRecorded(result=result),
        writer,
    )
    return replace(
        execution,
        state=state,
        outcome=TimedOut(),
        verification=result,
        terminal_reason="run timeout exceeded",
    )


def _abort_bundle(writer: RunBundleWriter, reason: str) -> object | None:
    try:
        return writer.abort(reason or "run aborted")
    except BaseException:
        return None
