"""Application use case for one complete attention-guided benchmark run."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Protocol, cast

from ux_analyzer.application.action_validation import (
    ActionValidationError,
    ValidatedAction,
    validate_action,
)
from ux_analyzer.application.memory import MemoryPolicy
from ux_analyzer.application.progress import (
    TransitionProgressSignature,
    element_progress_identity,
    made_meaningful_progress,
    repeated_cycle_length,
    transition_progress_signature,
)
from ux_analyzer.application.state_updates import (
    ApplicationState,
    StateUpdateConfig,
    apply_failure,
    apply_interaction_result,
    apply_observation,
    reconcile_snapshot_state,
)
from ux_analyzer.domain.attention import (
    Abandon,
    FullScent,
    InteractWithElement,
    PersonaObservation,
)
from ux_analyzer.domain.findings import Finding
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
from ux_analyzer.ports.artifacts import (
    ArtifactReference,
    BundleManifest,
    RunBundleWriter,
    SaliencyArtifactKind,
    SaliencyCacheHitEvent,
)
from ux_analyzer.ports.models import (
    CoarseScentEvaluator,
    CognitiveRunContext,
    FullScentEvaluator,
    ModelCallRecord,
    ModelResponseValidationError,
    ModelRole,
)
from ux_analyzer.ports.observation import (
    ObservationCapture,
    ObservationProvider,
    ObservationProviderError,
    ObservationSessionConfig,
    SafetyBlocked,
    SessionHandle,
)
from ux_analyzer.ports.verification import VerificationProvider

if TYPE_CHECKING:
    from ux_analyzer.application.evaluation import RunMetrics

_ATTENTION_EXHAUSTED_MESSAGE = "no unobserved visible elements remain"
_ATTENTION_EXHAUSTED_REASON = "all visible elements examined without progress"


class ProminenceResult(Protocol):
    """Application-facing prominence evidence shape."""

    element_id: str
    raw_score: float
    normalized_probability: float
    first_notice_probability: float | None
    notice_within_budget_probability: float | None
    feature_contributions: Mapping[str, float]
    raw_values: Mapping[str, float]
    normalized_values: Mapping[str, float]


class ObservationSelection(Protocol):
    """Application-facing observation selection shape."""

    observation: PersonaObservation
    region_id: str | None
    element_probabilities: object
    region_probabilities: object
    selection_mode: str
    recovery_selected_ids: Sequence[str]
    next_recovery_state: object

    @property
    def selected_ids(self) -> Sequence[str]: ...


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
        *,
        recovery_level: int = 0,
    ) -> ObservationSelection: ...


class CognitiveAgent(Protocol):
    """Application-facing decision capability."""

    async def decide(self, goal: str, observation: object) -> object: ...


class RunBundleFactory(Protocol):
    """Creates one append-only bundle for one run."""

    def start(self, spec: RunSpec) -> RunBundleWriter: ...


class ModelRecordSource(Protocol):
    """Role-call audit source scoped to one run."""

    @property
    def records(self) -> Sequence[ModelCallRecord]: ...


@dataclass(slots=True)
class _StageTiming:
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0


class RunProfiler:
    """Optional wall-clock profiler for one run; never changes run evidence."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._started = perf_counter()
        self._stages: dict[str, _StageTiming] = {}
        self._samples: list[dict[str, object]] = []

    @property
    def enabled(self) -> bool:
        return self._path is not None

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        started = perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (perf_counter() - started) * 1000
            timing = self._stages.setdefault(stage, _StageTiming())
            timing.count += 1
            timing.total_ms += elapsed_ms
            timing.max_ms = max(timing.max_ms, elapsed_ms)
            self._samples.append(
                {"stage": stage, "elapsed_ms": round(elapsed_ms, 3)}
            )

    def write(self, *, run_id: str) -> None:
        if self._path is None:
            return
        total_ms = (perf_counter() - self._started) * 1000
        stages = {
            name: {
                "count": timing.count,
                "total_ms": round(timing.total_ms, 3),
                "average_ms": round(timing.total_ms / timing.count, 3),
                "max_ms": round(timing.max_ms, 3),
                "share_of_run": round(timing.total_ms / total_ms, 4)
                if total_ms > 0
                else 0.0,
            }
            for name, timing in self._stages.items()
        }
        payload = {
            "run_id": run_id,
            "elapsed_ms": round(total_ms, 3),
            "stages": stages,
            "samples": self._samples,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(f".{self._path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._path)


class _ProfiledRunBundleWriter:
    """Measure bundle operations while preserving writer contract."""

    def __init__(self, writer: RunBundleWriter, profiler: RunProfiler) -> None:
        self._writer = writer
        self._profiler = profiler

    @property
    def run_id(self) -> str:
        return self._writer.run_id

    @property
    def manifest(self) -> BundleManifest:
        return self._writer.manifest

    def append_event(self, event: object) -> int:
        with self._profiler.measure("bundle.append_event"):
            return self._writer.append_event(event)

    def append_saliency_event(self, event: SaliencyCacheHitEvent) -> int:
        with self._profiler.measure("bundle.append_saliency_event"):
            return self._writer.append_saliency_event(event)

    def write_artifact(self, name: str, content: bytes | str) -> ArtifactReference:
        with self._profiler.measure("bundle.write_artifact"):
            return self._writer.write_artifact(name, content)

    def write_named_artifact(
        self, name: str, content: bytes | str
    ) -> ArtifactReference:
        with self._profiler.measure("bundle.write_named_artifact"):
            return self._writer.write_named_artifact(name, content)

    def write_saliency_artifact(
        self,
        name: str,
        content: bytes | str,
        kind: SaliencyArtifactKind,
    ) -> ArtifactReference:
        with self._profiler.measure("bundle.write_saliency_artifact"):
            return self._writer.write_saliency_artifact(name, content, kind)

    def finalize(self, result: object) -> Path:
        with self._profiler.measure("bundle.finalize"):
            return self._writer.finalize(result)

    def abort(self, reason: str) -> Path:
        with self._profiler.measure("bundle.abort"):
            return self._writer.abort(reason)


SessionConfigFactory = Callable[[RunSpec], ObservationSessionConfig]
SnapshotExtractor = Callable[[ObservationCapture], ViewportSnapshot]
ProviderManifestFactory = Callable[[RunSpec], Sequence[ProviderManifest]]
ResultEvaluator = Callable[["RunResult"], "RunResult"]


class RunFinalizationError(RuntimeError):
    """Bundle publication failed; no successful run result was published."""

    def __init__(self, reason: str) -> None:
        self.outcome = InternalError(reason=reason)
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class ProminenceEvidence:
    viewport_id: str
    scores: tuple[ProminenceResult, ...]
    source_event_id: str | None = None


@dataclass(frozen=True, slots=True)
class ScentEvidence:
    kind: str
    viewport_id: str
    scores: tuple[object, ...]
    source_event_id: str | None = None


@dataclass(frozen=True, slots=True)
class SelectionEvidence:
    viewport_id: str
    selected_ids: tuple[str, ...]
    selection_mode: str
    region_id: str | None
    element_probabilities: object
    region_probabilities: object
    recovery_selected_ids: tuple[str, ...] = ()
    source_event_id: str | None = None


@dataclass(frozen=True, slots=True)
class DecisionEvidence:
    viewport_id: str
    decision: object
    source_event_id: str | None = None


@dataclass(frozen=True, slots=True)
class RunEvidence:
    prominence: tuple[ProminenceEvidence, ...] = ()
    scent: tuple[ScentEvidence, ...] = ()
    selections: tuple[SelectionEvidence, ...] = ()
    decisions: tuple[DecisionEvidence, ...] = ()
    model_calls: tuple[ModelCallRecord, ...] = ()
    screenshot_artifacts: tuple[ArtifactChecksum, ...] = ()
    state_event_ids: tuple[str, ...] = ()


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
    evidence: RunEvidence = field(default_factory=RunEvidence)
    metrics: RunMetrics | None = None
    findings: tuple[Finding, ...] | None = ()
    evaluation_failure_reason: str | None = None
    ux_sample_valid: bool = True
    ux_sample_invalid_reason: str | None = None

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
    profiler: RunProfiler
    session: SessionHandle | None = None
    prominence: list[ProminenceEvidence] = field(
        default_factory=lambda: list[ProminenceEvidence]()
    )
    scent: list[ScentEvidence] = field(default_factory=lambda: list[ScentEvidence]())
    selections: list[SelectionEvidence] = field(
        default_factory=lambda: list[SelectionEvidence]()
    )
    decisions: list[DecisionEvidence] = field(
        default_factory=lambda: list[DecisionEvidence]()
    )
    model_calls: list[ModelCallRecord] = field(
        default_factory=lambda: list[ModelCallRecord]()
    )
    screenshot_artifacts: list[ArtifactChecksum] = field(
        default_factory=lambda: list[ArtifactChecksum]()
    )
    state_event_ids: list[str] = field(default_factory=lambda: list[str]())
    model_record_cursor: int = 0
    model_call_count: int = 0
    completed_fixture_inputs: set[tuple[str, str]] = field(
        default_factory=lambda: set[tuple[str, str]]()
    )
    last_action_fingerprint: tuple[object, ...] | None = None
    consecutive_action_count: int = 0
    no_progress_count: int = 0
    transition_history: list[TransitionProgressSignature] = field(
        default_factory=lambda: list[TransitionProgressSignature]()
    )
    previous_action: dict[str, object] | None = None
    previous_action_result: dict[str, object] | None = None


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
        model_record_source: ModelRecordSource | None = None,
        result_evaluator: ResultEvaluator | None = None,
        profile_path: Path | None = None,
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
        self.model_record_source = model_record_source
        self.result_evaluator = result_evaluator
        self.profile_path = profile_path

    async def execute(self, spec: RunSpec) -> RunResult:
        """Execute, independently verify, finalize, and always clean up one run."""

        writer: RunBundleWriter | None = None
        initial_state = RunState.initial(spec)
        profiler = RunProfiler(self.profile_path)
        set_profiler = getattr(self.observation_provider, "set_profiler", None)
        if callable(set_profiler):
            set_profiler(profiler)
        context = _RunContext(
            state=initial_state,
            application_state=ApplicationState.from_attention(initial_state.attention),
            profiler=profiler,
        )
        artifact_checksums: list[ArtifactChecksum] = []
        timeout_seconds = spec.scenario.budget.timeout_seconds
        deadline = (
            asyncio.get_running_loop().time() + timeout_seconds
            if timeout_seconds is not None
            else None
        )

        try:
            with profiler.measure("bundle.start"):
                writer = self.bundle_factory.start(spec)
            if profiler.enabled:
                writer = _ProfiledRunBundleWriter(writer, profiler)
            artifact_checksums.append(
                _artifact_checksum(writer.write_artifact("run-start.txt", spec.run_id))
            )
            context.state = _record(
                context.state,
                RunStarted(run_id=spec.run_id),
                writer,
                context.state_event_ids,
            )

            try:
                run = self._run(spec, context, writer, artifact_checksums)
                execution = (
                    await run
                    if timeout_seconds is None
                    else await asyncio.wait_for(run, timeout=timeout_seconds)
                )
            except TimeoutError:
                execution = _Execution(
                    state=context.state,
                    outcome=TimedOut(),
                    verification=None,
                    agent_claimed_success=False,
                    terminal_reason="run timeout exceeded",
                )
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                execution = _Execution(
                    state=context.state,
                    outcome=_outcome_for_error(error),
                    verification=None,
                    agent_claimed_success=False,
                    terminal_reason=_safe_error_message(error),
                )

            if deadline is None:
                with profiler.measure("verification.terminal"):
                    execution = await self._verify_terminal(
                        execution,
                        writer,
                        context.session,
                        context.state_event_ids,
                    )
            else:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    execution = _timed_out_execution(
                        execution, writer, context.state_event_ids
                    )
                else:
                    try:
                        with profiler.measure("verification.terminal"):
                            execution = await asyncio.wait_for(
                                self._verify_terminal(
                                    execution,
                                    writer,
                                    context.session,
                                    context.state_event_ids,
                                ),
                                timeout=remaining,
                            )
                    except TimeoutError:
                        execution = _timed_out_execution(
                            execution, writer, context.state_event_ids
                        )
            trace_session = context.session
            context.session = None
            await self._end_session(trace_session)
            if trace_session is not None and trace_session.trace_path.is_file():
                artifact_checksums.append(
                    _artifact_checksum(
                        writer.write_artifact(
                            trace_session.trace_path.name,
                            trace_session.trace_path.read_bytes(),
                        )
                    )
                )
            return self._finalize(spec, execution, writer, artifact_checksums, context)
        except asyncio.CancelledError as cancellation:
            if writer is not None:
                _abort_bundle(writer, "run execution cancelled")
            session = context.session
            context.session = None
            try:
                await self._end_session(session)
            except BaseException as cleanup_error:
                raise cancellation from cleanup_error
            raise
        except BaseException as error:
            if writer is not None:
                _abort_bundle(
                    writer, f"run execution failed: {_safe_error_message(error)}"
                )
            raise
        finally:
            try:
                with profiler.measure("browser.end_session"):
                    await self._end_session(context.session)
            finally:
                profiler.write(run_id=spec.run_id)

    async def _run(
        self,
        spec: RunSpec,
        context: _RunContext,
        writer: RunBundleWriter,
        artifact_checksums: list[ArtifactChecksum],
    ) -> _Execution:
        with context.profiler.measure("browser.start_session"):
            session = await self.observation_provider.start_session(
                self.session_config_factory(spec)
            )
        context.session = session
        with context.profiler.measure("browser.reset"):
            await self.observation_provider.reset(session)
        await self._capture(context, writer, artifact_checksums)
        rng = random.Random(spec.seed)
        claimed_success = False

        while True:
            application_state = context.application_state
            if context.model_call_count >= spec.scenario.budget.max_model_calls:
                writer.append_event(
                    {
                        "kind": "model-call-budget-exhausted",
                        "model_calls": context.model_call_count,
                        "limit": spec.scenario.budget.max_model_calls,
                    }
                )
                return _Execution(
                    state=context.state,
                    outcome=BudgetExhausted(),
                    verification=None,
                    agent_claimed_success=claimed_success,
                    terminal_reason="model call budget exhausted",
                )
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

            with context.profiler.measure("prominence.score"):
                scores = tuple(self.prominence_provider.score(snapshot))
            prominence_sequence = writer.append_event(
                {
                    "kind": "prominence-recorded",
                    "viewport_id": snapshot.id,
                    "scores": tuple(_prominence_payload(score) for score in scores),
                }
            )
            context.prominence.append(
                ProminenceEvidence(
                    snapshot.id, scores, source_event_id=_event_id(prominence_sequence)
                )
            )
            coarse_scent: object = ()
            if self.coarse_scent_evaluator is not None:
                try:
                    context.model_call_count += 1
                    with context.profiler.measure("model.coarse_scent"):
                        coarse_scent = await self.coarse_scent_evaluator.evaluate(
                            spec.scenario.goal, snapshot
                        )
                except ModelResponseValidationError as error:
                    return self._model_failure(
                        context, writer, error, claimed_success=claimed_success
                    )
                finally:
                    self._record_model_calls(context, writer)
                if context.model_call_count >= spec.scenario.budget.max_model_calls:
                    writer.append_event(
                        {
                            "kind": "model-call-budget-exhausted",
                            "model_calls": context.model_call_count,
                            "limit": spec.scenario.budget.max_model_calls,
                        }
                    )
                    return _Execution(
                        state=context.state,
                        outcome=BudgetExhausted(),
                        verification=None,
                        agent_claimed_success=claimed_success,
                        terminal_reason="model call budget exhausted",
                    )
                coarse_sequence = writer.append_event(
                    {
                        "kind": "coarse-scent-recorded",
                        "scores": tuple(
                            {"element_id": item.element_id, "score": item.score}
                            for item in coarse_scent
                        ),
                    }
                )
                context.scent.append(
                    ScentEvidence(
                        "coarse-scent",
                        snapshot.id,
                        tuple(coarse_scent),
                        source_event_id=_event_id(coarse_sequence),
                    )
                )

            try:
                with context.profiler.measure("attention.select"):
                    selection = self.attention_policy.next_observation(
                        application_state.attention,
                        snapshot,
                        scores,
                        coarse_scent,
                        rng,
                        recovery_level=context.no_progress_count,
                    )
            except ValueError as error:
                if str(error) != _ATTENTION_EXHAUSTED_MESSAGE:
                    raise
                writer.append_event(
                    {
                        "kind": "attention-exhausted",
                        "reason": _ATTENTION_EXHAUSTED_REASON,
                    }
                )
                return _Execution(
                    state=context.state,
                    outcome=AgentAbandoned(reason=_ATTENTION_EXHAUSTED_REASON),
                    verification=None,
                    agent_claimed_success=claimed_success,
                    terminal_reason=_ATTENTION_EXHAUSTED_REASON,
                )
            observation = selection.observation
            selection_record = SelectionEvidence(
                viewport_id=snapshot.id,
                selected_ids=tuple(selection.selected_ids),
                selection_mode=str(selection.selection_mode),
                region_id=getattr(selection, "region_id", None),
                element_probabilities=getattr(selection, "element_probabilities", {}),
                region_probabilities=getattr(selection, "region_probabilities", {}),
                recovery_selected_ids=tuple(
                    getattr(selection, "recovery_selected_ids", ())
                ),
            )
            selection_sequence = writer.append_event(
                {
                    "kind": "attention-selection-recorded",
                    "viewport_id": snapshot.id,
                    "selected_ids": selection_record.selected_ids,
                    "selection_mode": selection_record.selection_mode,
                    "region_id": selection_record.region_id,
                    "element_probabilities": selection_record.element_probabilities,
                    "region_probabilities": selection_record.region_probabilities,
                    "recovery_selected_ids": selection_record.recovery_selected_ids,
                }
            )
            context.selections.append(
                replace(
                    selection_record,
                    source_event_id=_event_id(selection_sequence),
                )
            )
            context.state = _record(
                context.state,
                ObservationRecorded(observation=observation),
                writer,
                context.state_event_ids,
            )
            context.application_state = _require_application_state(
                apply_observation(
                    application_state,
                    observation,
                    snapshot=snapshot,
                    memory_policy=self.memory_policy,
                    recovery_state=getattr(selection, "next_recovery_state", None),
                )
            )
            _sync_run_attention(context)

            if context.application_state.budgets.steps <= 0:
                return _Execution(
                    state=context.state,
                    outcome=BudgetExhausted(),
                    verification=None,
                    agent_claimed_success=claimed_success,
                    terminal_reason="attention budget exhausted",
                )

            parallel_model_calls = (
                self.full_scent_evaluator is not None
                and spec.scenario.budget.max_model_calls - context.model_call_count
                >= 2
            )
            full_scent: tuple[FullScent, ...] | None = None
            if self.full_scent_evaluator is not None and not parallel_model_calls:
                try:
                    context.model_call_count += 1
                    with context.profiler.measure("model.full_scent"):
                        full_scent = await self.full_scent_evaluator.evaluate(
                            spec.scenario.goal,
                            context.application_state.attention,
                            snapshot,
                        )
                except ModelResponseValidationError as error:
                    return self._model_failure(
                        context, writer, error, claimed_success=claimed_success
                    )
                finally:
                    self._record_model_calls(context, writer)
                if context.model_call_count >= spec.scenario.budget.max_model_calls:
                    writer.append_event(
                        {
                            "kind": "model-call-budget-exhausted",
                            "model_calls": context.model_call_count,
                            "limit": spec.scenario.budget.max_model_calls,
                        }
                    )
                    return _Execution(
                        state=context.state,
                        outcome=BudgetExhausted(),
                        verification=None,
                        agent_claimed_success=claimed_success,
                        terminal_reason="model call budget exhausted",
                    )

            update_context = getattr(self.cognitive_agent, "update_context", None)
            if callable(update_context):
                update_context(
                    CognitiveRunContext(
                        viewport_id=snapshot.id,
                        previous_action=context.previous_action,
                        previous_action_result=context.previous_action_result,
                        completed_fixture_keys=tuple(
                            sorted(
                                fixture_key
                                for _, fixture_key in context.completed_fixture_inputs
                            )
                        ),
                        fixture_input_complete=bool(context.completed_fixture_inputs),
                        working_memory_capacity=spec.persona.working_memory_capacity,
                        confidence=context.application_state.confidence,
                        frustration=context.application_state.frustration,
                        abandonment_threshold=spec.persona.abandonment_threshold,
                        attention_temperature=spec.persona.attention_temperature,
                    )
                )
            parallel_error: BaseException | None = None
            decision: object | None = None
            try:
                if parallel_model_calls:
                    context.model_call_count += 2
                    full_result, cognitive_result = (
                        await self._evaluate_parallel_calls(
                            context,
                            spec.scenario.goal,
                            context.application_state.attention,
                            snapshot,
                            observation,
                        )
                    )
                    if isinstance(full_result, BaseException):
                        raise full_result
                    full_scent = cast(tuple[FullScent, ...], full_result)
                    if isinstance(cognitive_result, BaseException):
                        parallel_error = cognitive_result
                    else:
                        decision = cognitive_result
                else:
                    context.model_call_count += 1
                    with context.profiler.measure("model.cognitive"):
                        decision = await self.cognitive_agent.decide(
                            spec.scenario.goal, observation
                        )
            except ModelResponseValidationError as error:
                return self._model_failure(
                    context, writer, error, claimed_success=claimed_success
                )
            finally:
                self._record_model_calls(
                    context,
                    writer,
                    ordered_roles=(
                        (ModelRole.FULL_SCENT, ModelRole.COGNITIVE)
                        if parallel_model_calls
                        else ()
                    ),
                )
            if full_scent is not None:
                full_sequence = writer.append_event(
                    {
                        "kind": "full-scent-recorded",
                        "scores": tuple(
                            {"element_id": item.element_id, "score": item.score}
                            for item in full_scent
                        ),
                    }
                )
                context.scent.append(
                    ScentEvidence(
                        "full-scent",
                        snapshot.id,
                        tuple(full_scent),
                        source_event_id=_event_id(full_sequence),
                    )
                )
            if parallel_error is not None:
                if isinstance(parallel_error, ModelResponseValidationError):
                    return self._model_failure(
                        context,
                        writer,
                        parallel_error,
                        claimed_success=claimed_success,
                    )
                raise parallel_error
            decision_sequence = writer.append_event(
                {
                    "kind": "decision-recorded",
                    "viewport_id": snapshot.id,
                    "decision": decision,
                    "reason": getattr(decision, "reason", None),
                    "action": getattr(decision, "action", None),
                    "claimed_success": _agent_claim(decision),
                }
            )
            context.decisions.append(
                DecisionEvidence(
                    snapshot.id,
                    decision,
                    source_event_id=_event_id(decision_sequence),
                )
            )
            claim = _agent_claim(decision)
            claimed_success = claimed_success or claim
            if claim:
                writer.append_event({"kind": "agent-claim", "claimed_success": True})

            try:
                with context.profiler.measure("action.validate"):
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
                _sync_run_attention(context)
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

            action_fingerprint = _action_fingerprint(validated, snapshot)
            fixture_identity = _fixture_identity(validated, snapshot)
            action_element_id = _interaction_element_id(validated)
            if (
                fixture_identity is not None
                and fixture_identity in context.completed_fixture_inputs
            ):
                writer.append_event(
                    {
                        "kind": "repeated-fixture-input",
                        "element_id": action_element_id,
                        "fixture_key": validated.fixture_key,
                        "reason": "fixture input was already entered successfully",
                    }
                )
                return _Execution(
                    state=context.state,
                    outcome=BudgetExhausted(),
                    verification=None,
                    agent_claimed_success=claimed_success,
                    terminal_reason="repeated successful fixture input detected",
                )
            context.state = _record(
                context.state,
                ActionProposed(action=validated.domain_action),
                writer,
                context.state_event_ids,
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
                    context.state_event_ids,
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
                _sync_run_attention(context)
                context.previous_action = _safe_action(validated)
                context.previous_action_result = {
                    "succeeded": True,
                    "state_changed": False,
                    "meaningful_progress": True,
                }
                context.no_progress_count = 0
                context.last_action_fingerprint = None
                context.consecutive_action_count = 0
                continue

            with context.profiler.measure("browser.execute"):
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
                    platform_action_kind=validated.platform_action.kind,
                    navigation_occurred=result.navigation_occurred,
                    state_changed=result.state_changed,
                ),
                writer,
                context.state_event_ids,
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
            _sync_run_attention(context)
            context.previous_action = _safe_action(validated)
            fixture_completed = False
            if result.succeeded and fixture_identity is not None:
                context.completed_fixture_inputs.add(fixture_identity)
                fixture_completed = True
                writer.append_event(
                    {
                        "kind": "fixture-input-completed",
                        "element_id": action_element_id,
                        "fixture_key": validated.fixture_key,
                    }
                )
            await self._capture(context, writer, artifact_checksums)
            current_snapshot = context.state.current_snapshot
            if current_snapshot is None:
                raise RuntimeError("recapture did not produce a current snapshot")
            meaningful_progress = made_meaningful_progress(
                snapshot,
                current_snapshot,
                succeeded=result.succeeded,
                navigation_occurred=result.navigation_occurred,
                fixture_completed=fixture_completed,
            )
            cycle_length = None
            if result.succeeded:
                context.transition_history.append(
                    transition_progress_signature(
                        action_fingerprint,
                        current_snapshot,
                        result.url,
                    )
                )
                context.transition_history = context.transition_history[-8:]
                cycle_length = repeated_cycle_length(context.transition_history)
            else:
                context.transition_history.clear()
            context.previous_action_result = {
                "succeeded": result.succeeded,
                "state_changed": result.state_changed,
                "navigation_occurred": result.navigation_occurred,
                "meaningful_progress": meaningful_progress,
                "error": result.error,
            }
            if meaningful_progress:
                context.no_progress_count = 0
                context.last_action_fingerprint = None
                context.consecutive_action_count = 0
            else:
                context.no_progress_count += 1
                if context.last_action_fingerprint == action_fingerprint:
                    context.consecutive_action_count += 1
                else:
                    context.last_action_fingerprint = action_fingerprint
                    context.consecutive_action_count = 1
                if (
                    context.consecutive_action_count < 3
                    and context.no_progress_count < 5
                ):
                    writer.append_event(
                        {
                            "kind": "no-progress-recovery",
                            "count": context.no_progress_count,
                            "action": validated.domain_action,
                            "reason": "action succeeded but the recaptured interface did not change semantically",
                        }
                )

            if result.succeeded:
                with context.profiler.measure("verification.run"):
                    verification = await self._verify(
                        context.state, writer, session, context.state_event_ids
                    )
                context.state = verification[0]
                if verification[1].verified:
                    return _Execution(
                        state=context.state,
                        outcome=VerifiedSuccess(),
                        verification=verification[1],
                        agent_claimed_success=claimed_success,
                        terminal_reason=None,
                    )

            if cycle_length is not None:
                writer.append_event(
                    {
                        "kind": "repeated-action-cycle",
                        "cycle_length": cycle_length,
                        "reason": "repeated semantic action cycle detected",
                    }
                )
                return _Execution(
                    state=context.state,
                    outcome=AgentAbandoned(
                        reason="repeated semantic action cycle detected"
                    ),
                    verification=None,
                    agent_claimed_success=claimed_success,
                    terminal_reason="repeated semantic action cycle detected",
                )

            stalled_out = (
                context.consecutive_action_count >= 3
                or context.no_progress_count >= 5
            )
            if not meaningful_progress and stalled_out:
                if context.consecutive_action_count >= 3:
                    writer.append_event(
                        {
                            "kind": "repeated-action-detected",
                            "action": validated.domain_action,
                            "count": context.consecutive_action_count,
                            "reason": "identical action repeated without semantic progress",
                        }
                    )
                writer.append_event(
                    {
                        "kind": "no-progress-detected",
                        "count": context.no_progress_count,
                        "reason": "consecutive actions exhausted semantic progress recovery",
                    }
                )
                return _Execution(
                    state=context.state,
                    outcome=AgentAbandoned(
                        reason="repeated actions produced no progress"
                    ),
                    verification=None,
                    agent_claimed_success=claimed_success,
                    terminal_reason="repeated actions produced no progress",
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


    async def _capture(
        self,
        context: _RunContext,
        writer: RunBundleWriter,
        artifact_checksums: list[ArtifactChecksum],
    ) -> None:
        session = context.session
        if session is None:
            raise RuntimeError("capture requires active session")
        with context.profiler.measure("observation.capture.total"):
            capture = await self.observation_provider.capture(session)
        snapshot = self.snapshot_extractor(capture)
        previous_snapshot = context.state.current_snapshot
        screenshot = writer.write_artifact(f"{snapshot.id}.png", capture.screenshot)
        screenshot_checksum = _artifact_checksum(screenshot)
        artifact_checksums.append(screenshot_checksum)
        context.screenshot_artifacts.append(screenshot_checksum)
        snapshot = replace(snapshot, screenshot_artifact=screenshot.path)
        if previous_snapshot is not None:
            context.application_state = _require_application_state(
                reconcile_snapshot_state(
                    context.application_state,
                    previous_snapshot,
                    snapshot,
                )
            )
            _sync_run_attention(context)
        context.state = _record(
            context.state,
            ViewportCaptured(
                snapshot=snapshot,
                viewport_width=capture.viewport.width,
                viewport_height=capture.viewport.height,
            ),
            writer,
            context.state_event_ids,
            )

    async def _evaluate_parallel_calls(
        self,
        context: _RunContext,
        goal: str,
        attention: object,
        snapshot: ViewportSnapshot,
        observation: PersonaObservation,
    ) -> tuple[object, object]:
        full_scent_evaluator = self.full_scent_evaluator
        if full_scent_evaluator is None:
            raise RuntimeError("parallel evaluation requires full scent evaluator")

        async def evaluate_full_scent() -> tuple[FullScent, ...]:
            with context.profiler.measure("model.full_scent"):
                return await full_scent_evaluator.evaluate(
                    goal, attention, snapshot
                )

        async def evaluate_cognitive() -> object:
            with context.profiler.measure("model.cognitive"):
                return await self.cognitive_agent.decide(goal, observation)

        tasks = (
            asyncio.create_task(evaluate_full_scent()),
            asyncio.create_task(evaluate_cognitive()),
        )
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        full_result, cognitive_result = results
        return cast(tuple[object, object], tuple(results))

    async def _verify(
        self,
        state: RunState,
        writer: RunBundleWriter,
        session: SessionHandle,
        state_event_ids: list[str],
    ) -> tuple[RunState, VerificationResult]:
        result = await self.verifier.verify(session)
        return (
            _record(
                state,
                VerificationRecorded(result=result),
                writer,
                state_event_ids,
            ),
            result,
        )

    async def _verify_terminal(
        self,
        execution: _Execution,
        writer: RunBundleWriter,
        session: SessionHandle | None,
        state_event_ids: list[str],
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
                state_event_ids,
            )
            return replace(execution, state=state, verification=result)

        try:
            result = await self.verifier.verify(session)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            result = VerificationResult(
                verified=False,
                details="independent verification failed",
            )
            state = _record(
                execution.state,
                VerificationRecorded(result=result),
                writer,
                state_event_ids,
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
            state_event_ids,
        )
        outcome = VerifiedSuccess() if result.verified else execution.outcome
        return replace(execution, state=state, outcome=outcome, verification=result)

    def _finalize(
        self,
        spec: RunSpec,
        execution: _Execution,
        writer: RunBundleWriter,
        artifact_checksums: Sequence[ArtifactChecksum],
        context: _RunContext,
    ) -> RunResult:
        verification = execution.verification or VerificationResult(
            verified=False,
            details="independent verification unavailable",
        )
        state = execution.state
        if state.verification is None:
            state = _record(
                state,
                VerificationRecorded(result=verification),
                writer,
                context.state_event_ids,
            )
        manifests = self._provider_manifests(spec)
        terminal = RunTerminated(
            outcome=execution.outcome,
            verification=verification,
            provider_manifests=manifests,
            configuration_digest=spec.config_digest,
            artifact_checksums=tuple(artifact_checksums),
        )
        try:
            final_state = _record(state, terminal, writer, context.state_event_ids)
        except BaseException as error:
            _abort_bundle(
                writer, f"terminal event failed: {_safe_error_message(error)}"
            )
            raise

        ux_sample_valid, ux_sample_invalid_reason = _ux_sample_validity(
            execution.outcome,
            execution.terminal_reason,
        )
        result = RunResult(
            run_id=spec.run_id,
            outcome=execution.outcome,
            verification=verification,
            agent_claimed_success=execution.agent_claimed_success,
            state=final_state,
            terminal_reason=execution.terminal_reason,
            ux_sample_valid=ux_sample_valid,
            ux_sample_invalid_reason=ux_sample_invalid_reason,
            findings=() if ux_sample_valid else None,
            evidence=RunEvidence(
                prominence=tuple(context.prominence),
                scent=tuple(context.scent),
                selections=tuple(context.selections),
                decisions=tuple(context.decisions),
                model_calls=tuple(context.model_calls),
                screenshot_artifacts=tuple(context.screenshot_artifacts),
                state_event_ids=tuple(context.state_event_ids),
            ),
        )
        if self.result_evaluator is not None and result.ux_sample_valid:
            try:
                with context.profiler.measure("evaluation.run"):
                    result = self.result_evaluator(result)
            except Exception as error:
                evaluation_reason = _evaluation_failure_reason(error)
                result = replace(
                    result,
                    metrics=None,
                    findings=None,
                    evaluation_failure_reason=evaluation_reason,
                    ux_sample_valid=False,
                    ux_sample_invalid_reason=f"evaluation-failure: {evaluation_reason}",
                )
        try:
            bundle_path = writer.finalize(result)
        except BaseException as error:
            reason = f"finalization failed: {_safe_error_message(error)}"
            _abort_bundle(
                writer,
                reason,
            )
            raise RunFinalizationError(reason) from error
        return replace(result, bundle_path=bundle_path)

    def _record_model_calls(
        self,
        context: _RunContext,
        writer: RunBundleWriter,
        *,
        ordered_roles: Sequence[ModelRole] = (),
    ) -> None:
        source = self.model_record_source
        if source is None:
            return
        records = tuple(source.records)
        new_records = list(records[context.model_record_cursor :])
        if ordered_roles:
            role_order = {role: index for index, role in enumerate(ordered_roles)}
            new_records.sort(
                key=lambda record: role_order.get(record.role, len(role_order))
            )
        for record in new_records:
            context.model_calls.append(record)
            writer.append_event({"kind": "model-call-recorded", "record": record})
        context.model_record_cursor = len(records)

    def _model_failure(
        self,
        context: _RunContext,
        writer: RunBundleWriter,
        error: ModelResponseValidationError,
        *,
        claimed_success: bool,
    ) -> _Execution:
        writer.append_event(
            {
                "kind": "model-failure",
                "role": error.role.value,
                "reason": error.reason,
                "response_summary": error.response_summary,
            }
        )
        return _Execution(
            state=context.state,
            outcome=ModelFailure(reason=error.reason),
            verification=None,
            agent_claimed_success=claimed_success,
            terminal_reason=f"{error.role.value}: {error.reason}",
        )

    def _provider_manifests(self, spec: RunSpec) -> tuple[ProviderManifest, ...]:
        if self.provider_manifest_factory is not None:
            manifests = tuple(self.provider_manifest_factory(spec))
            if manifests:
                return manifests
        providers: list[tuple[object, str]] = [
            (self.observation_provider, "observation"),
            (self.prominence_provider, "prominence"),
            (self.attention_policy, "attention"),
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
        await _shielded_await(self.observation_provider.end_session(session))


def _require_application_state(state: object) -> ApplicationState:
    if not isinstance(state, ApplicationState):
        raise RuntimeError("application state transition returned invalid state")
    return state


def _sync_run_attention(context: _RunContext) -> None:
    context.state = replace(
        context.state, attention=context.application_state.attention
    )


def _record(
    state: RunState,
    event: RunEvent,
    writer: RunBundleWriter,
    state_event_ids: list[str],
) -> RunState:
    next_state = state.apply(event)
    state_event_ids.append(_event_id(writer.append_event(event)))
    return next_state


def _event_id(sequence: int) -> str:
    return f"event-{sequence}"


def _snapshot_from_capture(capture: ObservationCapture) -> ViewportSnapshot:
    if capture.snapshot is None:
        raise RuntimeError("observation capture has no normalized viewport snapshot")
    return capture.snapshot


def _artifact_checksum(reference: ArtifactReference) -> ArtifactChecksum:
    return ArtifactChecksum(path=reference.path, sha256=reference.sha256)


def _prominence_payload(score: ProminenceResult) -> dict[str, object]:
    return {
        "element_id": score.element_id,
        "raw_score": score.raw_score,
        "normalized_probability": score.normalized_probability,
        "first_notice_probability": score.first_notice_probability,
        "notice_within_budget_probability": score.notice_within_budget_probability,
        "feature_contributions": dict(score.feature_contributions),
        "raw_values": dict(score.raw_values),
        "normalized_values": dict(score.normalized_values),
    }


def _execution_reference(
    snapshot: ViewportSnapshot,
    validated: ValidatedAction,
) -> PrivateExecutionReference | None:
    if not isinstance(validated.domain_action, InteractWithElement):
        return None
    return snapshot.element(validated.domain_action.element_id).execution_reference


def _fixture_identity(
    validated: ValidatedAction, snapshot: ViewportSnapshot
) -> tuple[str, str] | None:
    if validated.fixture_key is None or not isinstance(
        validated.domain_action, InteractWithElement
    ):
        return None
    element = snapshot.element(validated.domain_action.element_id)
    return (element.lineage_id or element.id, validated.fixture_key)


def _interaction_element_id(validated: ValidatedAction) -> str | None:
    action = validated.domain_action
    return action.element_id if isinstance(action, InteractWithElement) else None


def _action_fingerprint(
    validated: ValidatedAction, snapshot: ViewportSnapshot
) -> tuple[object, ...]:
    action = validated.domain_action
    element_id = getattr(action, "element_id", None)
    if isinstance(element_id, str):
        element = snapshot.element(element_id)
        target: object = element_progress_identity(snapshot, element)
    else:
        target = None
    return (
        str(getattr(action, "kind", type(action).__name__)),
        target,
        validated.fixture_key,
        getattr(action, "direction", None),
    )


def _safe_action(validated: ValidatedAction) -> dict[str, object]:
    action = validated.domain_action
    result: dict[str, object] = {
        "kind": str(getattr(action, "kind", type(action).__name__))
    }
    element_id = getattr(action, "element_id", None)
    if isinstance(element_id, str):
        result["element_id"] = element_id
    if validated.fixture_key is not None:
        result["fixture_key"] = validated.fixture_key
    direction = getattr(action, "direction", None)
    if isinstance(direction, str):
        result["direction"] = direction
    return result


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
            prompt_version=str(getattr(manifest, "prompt_version", "")) or None,
            schema_version=str(getattr(manifest, "schema_version", "")) or None,
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
        "ModelResponseValidationError",
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


def _evaluation_failure_reason(error: BaseException) -> str:
    safe_reason = getattr(error, "safe_reason", None)
    if isinstance(safe_reason, str) and safe_reason:
        return safe_reason
    return f"result evaluation failed: {error.__class__.__name__}"


def _ux_sample_validity(
    outcome: RunOutcome,
    terminal_reason: str | None,
) -> tuple[bool, str | None]:
    kind = str(getattr(outcome, "kind", "internal-error"))
    if kind in {"verified-success", "agent-abandoned", "budget-exhausted"}:
        return True, None
    reason = terminal_reason or getattr(outcome, "reason", None) or "run unavailable"
    return False, f"{kind}: {reason}"


async def _shielded_await(awaitable: Awaitable[None]) -> None:
    operation = asyncio.ensure_future(awaitable)
    cancellation: asyncio.CancelledError | None = None
    while not operation.done():
        try:
            await asyncio.shield(operation)
        except asyncio.CancelledError as error:
            cancellation = error
    operation_error = operation.exception()
    if cancellation is not None:
        raise cancellation from operation_error
    if operation_error is not None:
        raise operation_error
    operation.result()


def _timed_out_execution(
    execution: _Execution,
    writer: RunBundleWriter,
    state_event_ids: list[str],
) -> _Execution:
    result = VerificationResult(
        verified=False,
        details="independent verification unavailable before timeout",
    )
    state = _record(
        execution.state,
        VerificationRecorded(result=result),
        writer,
        state_event_ids,
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
