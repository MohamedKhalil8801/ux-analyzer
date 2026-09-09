"""Application use case for one complete attention-guided benchmark run."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import random
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
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
    snapshot_progress_signature,
    transition_progress_signature,
)
from ux_analyzer.application.saliency import (
    ProminenceBatch,
    ProminenceProvider,
    ProminenceResult,
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
    Complete,
    FullScent,
    InteractWithElement,
    PersonaObservation,
    Wait,
)
from ux_analyzer.domain.benchmark import VisibleResultVerifierSpec
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
from ux_analyzer.domain.saliency import ElementAttentionProfile, SearchStage
from ux_analyzer.ports.artifacts import (
    ArtifactReference,
    BundleManifest,
    ProminenceRecordedEvent,
    RedactionPolicy,
    RunBundleWriter,
    SaliencyArtifactContext,
    SaliencyArtifactKind,
    SaliencyFallbackRecordedEvent,
    SaliencyProfilesRecordedEvent,
    required_saliency_artifact_paths,
    sanitize_log_text,
    validate_saliency_artifact_path,
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
    WaitAction,
)
from ux_analyzer.ports.verification import VerificationProvider

if TYPE_CHECKING:
    from ux_analyzer.application.evaluation import RunMetrics

_ATTENTION_EXHAUSTED_MESSAGE = "no unobserved visible elements remain"
_ATTENTION_EXHAUSTED_REASON = "all visible elements examined without progress"
_HUMAN_BUDGET_TERMINAL_REASONS = frozenset(
    {
        "attention budget exhausted",
        "step budget exhausted",
        "observation budget exhausted",
        "interaction budget exhausted",
        "action budget exhausted",
        "human attention budget exhausted",
        "human action budget exhausted",
    }
)


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
            self._samples.append({"stage": stage, "elapsed_ms": round(elapsed_ms, 3)})

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

    @property
    def saliency_artifact_context(self) -> SaliencyArtifactContext | None:
        return self._writer.saliency_artifact_context

    def append_event(self, event: object) -> int:
        with self._profiler.measure("bundle.append_event"):
            return self._writer.append_event(event)

    def append_saliency_event(self, event: object) -> int:
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

    def verify_artifact(self, reference: ArtifactReference) -> None:
        with self._profiler.measure("bundle.verify_artifact"):
            self._writer.verify_artifact(reference)

    def finalize(self, result: object) -> Path:
        with self._profiler.measure("bundle.finalize"):
            return self._writer.finalize(result)

    def abort(self, reason: str) -> Path:
        with self._profiler.measure("bundle.abort"):
            return self._writer.abort(reason)


class _SaliencyArtifactTrackingWriter:
    """Capture references returned by capture-aware artifact writes."""

    def __init__(
        self, writer: RunBundleWriter, context: SaliencyArtifactContext
    ) -> None:
        self._writer = writer
        self._context = context
        self.references: list[ArtifactReference] = []

    @property
    def run_id(self) -> str:
        return self._writer.run_id

    @property
    def manifest(self) -> BundleManifest:
        return self._writer.manifest

    @property
    def saliency_artifact_context(self) -> SaliencyArtifactContext:
        return self._context

    def append_event(self, event: object) -> int:
        return self._writer.append_event(event)

    def append_saliency_event(self, event: object) -> int:
        return self._writer.append_saliency_event(event)

    def write_artifact(self, name: str, content: bytes | str) -> ArtifactReference:
        return self._writer.write_artifact(name, content)

    def write_named_artifact(
        self, name: str, content: bytes | str
    ) -> ArtifactReference:
        return self._writer.write_named_artifact(name, content)

    def write_saliency_artifact(
        self,
        name: str,
        content: bytes | str,
        kind: SaliencyArtifactKind,
    ) -> ArtifactReference:
        reference = self._writer.write_saliency_artifact(
            self._remap_path(name), content, kind
        )
        self.references.append(reference)
        return reference

    def verify_artifact(self, reference: ArtifactReference) -> None:
        self._writer.verify_artifact(reference)

    def _remap_path(self, name: str) -> str:
        normalized = validate_saliency_artifact_path(name)
        return f"saliency/{self._context.artifact_namespace}/{normalized.name}"

    def _remap_reference(self, reference: ArtifactReference) -> ArtifactReference:
        path = self._remap_path(reference.path)
        return replace(reference, path=path, name=path)

    def finalize(self, result: object) -> Path:
        return self._writer.finalize(result)

    def abort(self, reason: str) -> Path:
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
    profile_event_id: str | None = None
    operational_event_id: str | None = None
    stage: str | None = None
    profiles: tuple[ElementAttentionProfile, ...] = ()

    def __post_init__(self) -> None:
        operational_event_id = self.operational_event_id or self.source_event_id
        if (
            self.operational_event_id is not None
            and self.source_event_id is not None
            and self.operational_event_id != self.source_event_id
        ):
            raise ValueError("prominence source and operational event IDs must match")
        object.__setattr__(self, "source_event_id", operational_event_id)
        object.__setattr__(self, "operational_event_id", operational_event_id)
        if self.stage is not None:
            try:
                stage = SearchStage(self.stage)
            except ValueError as error:
                raise ValueError("prominence evidence stage is invalid") from error
            object.__setattr__(self, "stage", stage.value)
        object.__setattr__(self, "scores", tuple(self.scores))
        object.__setattr__(self, "profiles", tuple(self.profiles))


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

    def to_persistence_dict(self) -> dict[str, object]:
        """Return public evidence references without raw decision or score data."""

        return {
            "prominence": tuple(
                {
                    "viewport_id": record.viewport_id,
                    "source_event_id": record.source_event_id,
                    "profile_event_id": record.profile_event_id,
                    "operational_event_id": record.operational_event_id,
                    "stage": record.stage,
                }
                for record in self.prominence
            ),
            "scent": tuple(
                {
                    "kind": record.kind,
                    "viewport_id": record.viewport_id,
                    "source_event_id": record.source_event_id,
                }
                for record in self.scent
            ),
            "selections": tuple(
                {
                    "viewport_id": record.viewport_id,
                    "selected_ids": record.selected_ids,
                    "selection_mode": record.selection_mode,
                    "region_id": record.region_id,
                    "recovery_selected_ids": record.recovery_selected_ids,
                    "source_event_id": record.source_event_id,
                }
                for record in self.selections
            ),
            "decisions": tuple(
                {
                    "viewport_id": record.viewport_id,
                    "source_event_id": record.source_event_id,
                }
                for record in self.decisions
            ),
            "model_calls": tuple(
                _model_call_persistence_dict(record) for record in self.model_calls
            ),
            "screenshot_artifacts": self.screenshot_artifacts,
            "state_event_ids": self.state_event_ids,
        }


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

    def to_persistence_dict(self) -> dict[str, object]:
        """Return result fields with explicit public evidence persistence."""

        return {
            "run_id": self.run_id,
            "outcome": self.outcome,
            "verification": self.verification,
            "agent_claimed_success": self.agent_claimed_success,
            "state": self.state,
            "bundle_path": self.bundle_path,
            "terminal_reason": self.terminal_reason,
            "evidence": self.evidence.to_persistence_dict(),
            "metrics": self.metrics,
            "findings": self.findings,
            "evaluation_failure_reason": self.evaluation_failure_reason,
            "ux_sample_valid": self.ux_sample_valid,
            "ux_sample_invalid_reason": self.ux_sample_invalid_reason,
        }


@dataclass(frozen=True, slots=True)
class _Execution:
    state: RunState
    outcome: RunOutcome
    verification: VerificationResult | None
    agent_claimed_success: bool
    terminal_reason: str | None


class _RunStalled(Exception):
    """Raised when a run recorded no progress heartbeat within its stall budget."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _WallDeadlineExceeded(Exception):
    """Raised when the optional wall-clock budget cap elapsed.

    Distinct from :class:`TimeoutError` so that internal timeouts from
    providers or the browser are never mislabeled as run-budget timeouts.
    """


@dataclass(slots=True)
class _ProgressHeartbeat:
    """Tracks wall-clock time since the last recorded run progress.

    A beat records forward motion (completed capture, model call, action, or
    verification). The run watchdog treats a long silent stretch as a stall.
    While an operation is in flight (a model call, browser action, capture,
    or verification), the silent allowance is extended: slow-but-alive work
    must not be misread as a hang.
    """

    loop: asyncio.AbstractEventLoop
    stall_seconds: float | None
    last_beat: float = field(init=False)
    in_flight: int = 0

    def __post_init__(self) -> None:
        self.last_beat = self.loop.time()

    def beat(self) -> None:
        self.last_beat = self.loop.time()

    def begin_operation(self) -> None:
        self.in_flight += 1

    def end_operation(self) -> None:
        self.in_flight = max(0, self.in_flight - 1)

    def idle_seconds(self) -> float:
        return self.loop.time() - self.last_beat

    def effective_stall_seconds(self) -> float | None:
        if self.stall_seconds is None:
            return None
        if self.in_flight:
            return max(self.stall_seconds, _INFLIGHT_STALL_SECONDS)
        return self.stall_seconds

    def stall_remaining(self) -> float | None:
        if self.stall_seconds is None:
            return None
        return self.effective_stall_seconds() - self.idle_seconds()


_INFLIGHT_STALL_SECONDS = 240.0


@contextmanager
def _operation_in_flight(
    progress: _ProgressHeartbeat | None,
) -> Iterator[None]:
    """Mark one awaited operation as in-flight work, not a hang."""

    if progress is not None:
        progress.begin_operation()
    try:
        yield
    finally:
        if progress is not None:
            progress.end_operation()


def _beat(progress: _ProgressHeartbeat | None) -> None:
    if progress is not None:
        progress.beat()


def _normalized_target_label(value: str) -> str:
    """Whitespace-safe casefolded label for evaluation-target matching."""

    return " ".join(value.replace("\u00a0", " ").split()).casefold()


async def _cancel_and_wait(task: asyncio.Future[object]) -> None:
    task.cancel()
    try:
        await task
    except BaseException:  # noqa: BLE001 - cancellation result is irrelevant
        pass


async def _wait_for_run_progress(
    run_task: asyncio.Future[_Execution],
    *,
    heartbeat: _ProgressHeartbeat,
    wall_deadline: float | None,
    stall_seconds: float | None,
    loop: asyncio.AbstractEventLoop,
) -> _Execution:
    """Await the run task, ending it only when progress actually stops.

    The task is never cancelled while it keeps producing progress
    heartbeats. A wall-clock cap (``wall_deadline``) or a silent stretch
    (``stall_seconds``) cancels it; otherwise the run continues.
    """

    if wall_deadline is None and stall_seconds is None:
        return await run_task
    poll_interval = 0.25
    while True:
        now = loop.time()
        if wall_deadline is not None and now >= wall_deadline:
            await _cancel_and_wait(run_task)
            raise _WallDeadlineExceeded
        remaining_stall = heartbeat.stall_remaining()
        if remaining_stall is not None and remaining_stall <= 0:
            await _cancel_and_wait(run_task)
            raise _RunStalled(
                f"stalled: no progress for {heartbeat.idle_seconds():.1f}s "
                f"(stall_timeout_seconds={stall_seconds:g})"
            )
        wait = poll_interval
        if wall_deadline is not None:
            wait = min(wait, max(0.05, wall_deadline - now))
        if remaining_stall is not None:
            wait = min(wait, max(0.05, remaining_stall))
        try:
            done, _pending = await asyncio.wait({run_task}, timeout=wait)
        except asyncio.CancelledError:
            await _cancel_and_wait(run_task)
            raise
        if done:
            return run_task.result()


def _progress_remaining(
    *,
    wall_deadline: float | None,
    heartbeat: _ProgressHeartbeat,
    loop: asyncio.AbstractEventLoop,
) -> float | None:
    """Time budget left for terminal verification, or None when unbounded."""

    caps: list[float] = []
    if wall_deadline is not None:
        caps.append(wall_deadline - loop.time())
    stall_remaining = heartbeat.stall_remaining()
    if stall_remaining is not None:
        caps.append(stall_remaining)
    if not caps:
        return None
    return min(caps)


@dataclass(frozen=True, slots=True)
class _CapturedViewport:
    """Local capture data passed to model providers without entering RunState."""

    capture: ObservationCapture
    snapshot: ViewportSnapshot
    screenshot_sha256: str
    source_event_id: str


@dataclass(frozen=True, slots=True)
class _SaliencyRuntimeMetadata:
    provider_id: str
    model_checksums: tuple[str, ...]
    execution_provider: str
    preprocessing_version: str
    precision: str
    cache_key: str | None
    timings_ms: tuple[float, ...]
    warnings: tuple[str, ...]


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
    deferred_wait_pending: bool = False
    no_change_interaction_element_id: object | None = None
    transition_history: list[TransitionProgressSignature] = field(
        default_factory=lambda: list[TransitionProgressSignature]()
    )
    previous_action: dict[str, object] | None = None
    previous_action_result: dict[str, object] | None = None
    current_capture: _CapturedViewport | None = None
    current_viewport_identity: tuple[object, ...] | None = None
    observed_viewport_identity: tuple[object, ...] | None = None
    stage_reset_for_capture: bool = True
    saliency_fallback_reason: str | None = None
    redaction_policy: RedactionPolicy = field(default_factory=RedactionPolicy)
    saliency_artifact_references: list[ArtifactReference] = field(default_factory=list)


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
        """Execute with automatic fresh-context retries for internal/model flakes.

        Real UX outcomes (verified-success, agent-abandoned, budget-exhausted)
        are returned immediately. Internal flakes (internal-error, model-failure,
        provider-failure) are retried with a fresh browser session/attention
        context up to two additional times, without requiring a user flag.
        Reproducible re-execution of successful samples still requires
        ``run_count`` / ``seeds`` — that is an explicit experiment flag.
        """

        max_attempts = 3
        last_result: RunResult | None = None
        for attempt in range(max_attempts):
            result = await self._execute_once(spec, attempt)
            last_result = result
            if attempt == max_attempts - 1 or not _is_retryable_result(result):
                if attempt > 0:
                    # Annotate that a retry succeeded — evaluation keeps the last
                    # attempt's bundle; intermediate aborted bundles are removed.
                    pass
                return result
            # Retryable internal/model flake — remove the failed bundle so the
            # next attempt can re-use the same deterministic run_id.
            bundle_path = getattr(result, "bundle_path", None)
            if bundle_path is not None:
                try:
                    import shutil
                    from pathlib import Path as _Path

                    path = _Path(bundle_path)  # type: ignore[arg-type]
                    if path.exists():
                        shutil.rmtree(path)
                    # Also remove the zipped trace that ExperimentResult keeps.
                    trace_zip = path.parent.parent / "traces" / f"{path.name}.zip"
                    if trace_zip.exists():
                        trace_zip.unlink()
                except BaseException:
                    pass
            # Backoff before fresh-context retry to soften Zen rate limits.
            await asyncio.sleep(min(2**attempt, 5))
        assert last_result is not None
        return last_result

    async def _execute_once(self, spec: RunSpec, attempt: int = 0) -> RunResult:
        """Single attempt with fresh browser context and attention state."""

        writer: RunBundleWriter | None = None
        initial_state = RunState.initial(spec)
        effective_memory_capacity = _effective_memory_capacity(
            spec, self.memory_policy
        )
        if initial_state.attention.memory_capacity != effective_memory_capacity:
            initial_state = replace(
                initial_state,
                attention=replace(
                    initial_state.attention,
                    memory_capacity=effective_memory_capacity,
                ),
            )
        profiler = RunProfiler(self.profile_path)
        set_profiler = getattr(self.observation_provider, "set_profiler", None)
        if callable(set_profiler):
            set_profiler(profiler)
        context = _RunContext(
            state=initial_state,
            application_state=ApplicationState.from_attention(initial_state.attention),
            profiler=profiler,
            redaction_policy=RedactionPolicy.from_fixture_inputs(
                spec.scenario.fixture_inputs
            ),
        )
        artifact_checksums: list[ArtifactChecksum] = []
        timeout_seconds = spec.scenario.budget.timeout_seconds
        stall_seconds = spec.scenario.budget.stall_timeout_seconds
        loop = asyncio.get_running_loop()
        heartbeat = _ProgressHeartbeat(loop=loop, stall_seconds=stall_seconds)
        wall_deadline = (
            loop.time() + timeout_seconds if timeout_seconds is not None else None
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
                run_task = asyncio.ensure_future(
                    self._run(
                        spec,
                        context,
                        writer,
                        artifact_checksums,
                        attempt=attempt,
                        progress=heartbeat,
                    )
                )
                execution = await _wait_for_run_progress(
                    run_task,
                    heartbeat=heartbeat,
                    wall_deadline=wall_deadline,
                    stall_seconds=stall_seconds,
                    loop=loop,
                )
            except _WallDeadlineExceeded:
                execution = _Execution(
                    state=context.state,
                    outcome=TimedOut(),
                    verification=None,
                    agent_claimed_success=False,
                    terminal_reason="run timeout exceeded",
                )
            except _RunStalled as stalled:
                execution = _Execution(
                    state=context.state,
                    outcome=TimedOut(),
                    verification=None,
                    agent_claimed_success=False,
                    terminal_reason=stalled.reason,
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

            with _operation_in_flight(heartbeat):
                remaining = _progress_remaining(
                    wall_deadline=wall_deadline,
                    heartbeat=heartbeat,
                    loop=loop,
                )
                if remaining is not None and remaining <= 0:
                    execution = _timed_out_execution(
                        execution, writer, context.state_event_ids
                    )
                elif remaining is None:
                    with profiler.measure("verification.terminal"):
                        execution = await self._verify_terminal(
                            execution,
                            writer,
                            context,
                            artifact_checksums,
                        )
                else:
                    try:
                        with profiler.measure("verification.terminal"):
                            execution = await asyncio.wait_for(
                                self._verify_terminal(
                                    execution,
                                    writer,
                                    context,
                                    artifact_checksums,
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
        *,
        attempt: int = 0,
        progress: _ProgressHeartbeat | None = None,
    ) -> _Execution:
        with context.profiler.measure("browser.start_session"):
            session = await self.observation_provider.start_session(
                self.session_config_factory(spec)
            )
        context.session = session
        _beat(progress)
        with context.profiler.measure("browser.reset"):
            await self.observation_provider.reset(session)
        _beat(progress)
        with _operation_in_flight(progress):
            await self._capture(context, writer, artifact_checksums)
        _beat(progress)
        rng = random.Random(spec.seed + attempt * 997)
        claimed_success = False

        while True:
            _beat(progress)
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
            captured = context.current_capture
            snapshot = context.state.current_snapshot
            if captured is None or snapshot is None:
                raise RuntimeError("run has no current viewport capture")
            if context.deferred_wait_pending:
                context.deferred_wait_pending = False
                if not _has_unobserved_visible_element(
                    context.application_state, snapshot
                ):
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

            with context.profiler.measure("prominence.score"):
                scores, prominence_batch, artifact_references = self._score_prominence(
                    captured, snapshot, context, writer
                )
            if artifact_references:
                known_paths = {
                    reference.path for reference in context.saliency_artifact_references
                }
                context.saliency_artifact_references.extend(
                    reference
                    for reference in artifact_references
                    if reference.path not in known_paths
                )
            prominence_sequence, profile_event_id = self._record_prominence(
                context,
                writer,
                snapshot,
                scores=scores,
                batch=prominence_batch,
                artifact_references=artifact_references,
            )
            prominence_stage = _search_stage(
                self.attention_policy, context, snapshot
            ).value
            context.prominence.append(
                ProminenceEvidence(
                    snapshot.id,
                    scores,
                    source_event_id=_event_id(prominence_sequence),
                    profile_event_id=profile_event_id,
                    operational_event_id=_event_id(prominence_sequence),
                    stage=prominence_stage,
                    profiles=(
                        prominence_batch.learned_profiles
                        if prominence_batch is not None
                        else ()
                    ),
                )
            )
            coarse_scent: object = ()
            if self.coarse_scent_evaluator is not None:
                try:
                    context.model_call_count += 1
                    with context.profiler.measure("model.coarse_scent"):
                        with _operation_in_flight(progress):
                            coarse_scent = (
                                await self.coarse_scent_evaluator.evaluate(
                                    spec.scenario.goal, snapshot
                                )
                            )
                    _beat(progress)
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
            except (ValueError, RuntimeError) as error:
                if not _is_attention_exhausted_error(error):
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
            context.observed_viewport_identity = context.current_viewport_identity
            context.stage_reset_for_capture = False
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
                and spec.scenario.budget.max_model_calls - context.model_call_count >= 2
            )
            full_scent: tuple[FullScent, ...] | None = None
            if self.full_scent_evaluator is not None and not parallel_model_calls:
                try:
                    context.model_call_count += 1
                    with context.profiler.measure("model.full_scent"):
                        with _operation_in_flight(progress):
                            full_scent = await self.full_scent_evaluator.evaluate(
                                spec.scenario.goal,
                                context.application_state.attention,
                                snapshot,
                            )
                    _beat(progress)
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
                        working_memory_capacity=context.application_state.attention.memory_capacity,
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
                    with _operation_in_flight(progress):
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
                        with _operation_in_flight(progress):
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
            _beat(progress)
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

            # Anti-repeat guard: the cognitive prompt forbids repeating a
            # semantic action whose previous execution left the interface
            # unchanged; the boundary enforces it with explicit rejection
            # feedback instead of silently executing a no-op again. Element
            # identity uses the snapshot-stable semantic identity, not the
            # per-capture element ID (recaptures re-label the same node).
            proposed_action = getattr(decision, "action", decision)
            proposed_kind = getattr(proposed_action, "kind", None)
            proposed_element_id = getattr(proposed_action, "element_id", None)
            if (
                isinstance(proposed_element_id, str)
                and context.no_change_interaction_element_id is not None
                and proposed_kind
                in {"interact", "interact-with-element", "type-fixture"}
            ):
                try:
                    proposed_identity = element_progress_identity(
                        snapshot, snapshot.element(proposed_element_id)
                    )
                except (KeyError, ValueError):
                    proposed_identity = None
                if proposed_identity == context.no_change_interaction_element_id:
                    writer.append_event(
                        {
                            "kind": "action-rejected",
                            "element_id": proposed_element_id,
                            "reason": "previous interaction on this element succeeded without changing the interface; repeating it cannot progress the task",
                        }
                    )
                    # Surface the rejection in the next cognitive decision's
                    # previous-action feedback so the model adapts instead of
                    # insisting on the same no-op.
                    context.previous_action = {
                        "kind": str(proposed_kind),
                        "element_id": proposed_element_id,
                    }
                    context.previous_action_result = {
                        "succeeded": False,
                        "state_changed": False,
                        "navigation_occurred": False,
                        "meaningful_progress": False,
                        "error": (
                            "repeat rejected: the previous interaction on "
                            "this element produced no interface change"
                        ),
                    }
                    context.application_state = _require_application_state(
                        apply_failure(
                            context.application_state,
                            reason="repeated-no-change-interaction",
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

            deferred_abandonment = False
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
                if _accept_abandonment(context, spec):
                    return _Execution(
                        state=context.state,
                        outcome=AgentAbandoned(reason=validated.domain_action.reason),
                        verification=None,
                        agent_claimed_success=claimed_success,
                        terminal_reason=validated.domain_action.reason,
                    )
                writer.append_event(
                    {
                        "kind": "abandonment-deferred",
                        "reason": validated.domain_action.reason,
                        "recovery_level": context.no_progress_count + 1,
                    }
                )
                validated = ValidatedAction(
                    decision=decision,
                    domain_action=Wait(),
                    platform_action=WaitAction(milliseconds=0),
                )
                action_fingerprint = _action_fingerprint(validated, snapshot)
                deferred_abandonment = True
                context.deferred_wait_pending = True
                context.state = _record(
                    context.state,
                    ActionProposed(action=validated.domain_action),
                    writer,
                    context.state_event_ids,
                )

            if isinstance(validated.domain_action, Complete):
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
                with context.profiler.measure("verification.run"):
                    with _operation_in_flight(progress):
                        verification = await self._verify(
                            context, writer, artifact_checksums
                        )
                _beat(progress)
                context.state = verification[0]
                context.previous_action_result = {
                    "succeeded": True,
                    "state_changed": False,
                    "navigation_occurred": False,
                    "meaningful_progress": False,
                    "verified": verification[1].verified,
                }
                if verification[1].verified:
                    return _Execution(
                        state=context.state,
                        outcome=VerifiedSuccess(),
                        verification=verification[1],
                        agent_claimed_success=claimed_success,
                        terminal_reason=None,
                    )
                if context.application_state.budgets.steps <= 0:
                    return _Execution(
                        state=context.state,
                        outcome=BudgetExhausted(),
                        verification=verification[1],
                        agent_claimed_success=claimed_success,
                        terminal_reason="attention budget exhausted",
                    )
                continue

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
                context.no_change_interaction_element_id = None
                continue

            with context.profiler.measure("browser.execute"):
                with _operation_in_flight(progress):
                    result = await self.observation_provider.execute(
                        session, validated.platform_action
                    )
            _beat(progress)
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
            with _operation_in_flight(progress):
                await self._capture(context, writer, artifact_checksums)
            _beat(progress)
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
            target_labels = {
                _normalized_target_label(label)
                for label in spec.scenario.evaluation_target.labels_by_version.values()
            }
            interacted_with_target = False
            if action_element_id is not None and result.succeeded:
                acted_label = None
                try:
                    acted_label = snapshot.element(action_element_id).label
                except Exception:  # noqa: BLE001 - unknown targets stay unclassified
                    acted_label = None
                if acted_label is not None and (
                    _normalized_target_label(acted_label) in target_labels
                ):
                    interacted_with_target = True
                    writer.append_event(
                        {
                            "kind": "target-interaction-progress",
                            "element_id": action_element_id,
                            "reason": "successful interaction with the evaluation target is progress",
                        }
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
            if (
                result.succeeded
                and isinstance(validated.domain_action, InteractWithElement)
                and not result.navigation_occurred
                and not meaningful_progress
            ):
                # The adapter's state_changed heuristic can report True for
                # focus/scroll side effects; semantic snapshot comparison is
                # the reliable "interface did not change" signal.
                try:
                    context.no_change_interaction_element_id = (
                        element_progress_identity(
                            snapshot, snapshot.element(action_element_id)
                        )
                    )
                except (KeyError, ValueError):
                    context.no_change_interaction_element_id = None
            else:
                context.no_change_interaction_element_id = None
            context.previous_action_result = {
                "succeeded": result.succeeded,
                "state_changed": result.state_changed,
                "navigation_occurred": result.navigation_occurred,
                "meaningful_progress": meaningful_progress,
                "error": result.error,
            }
            if deferred_abandonment:
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
                            "reason": "abandonment was deferred for bounded re-observation",
                        }
                    )
            elif interacted_with_target:
                # A successful interaction with the evaluation target never
                # reads as "stuck": the no-progress counter resets so the run
                # cannot be abandoned for absent app feedback. Repeated
                # identical target clicks still accumulate their own
                # patience counter (see stalled_out below) so a loop stays
                # bounded. The model still sees the semantic truth
                # (state_changed) through previous_action_result above.
                context.no_progress_count = 0
                if context.last_action_fingerprint == action_fingerprint:
                    context.consecutive_action_count += 1
                else:
                    context.last_action_fingerprint = action_fingerprint
                    context.consecutive_action_count = 1
            elif meaningful_progress:
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

            if (
                result.succeeded
                and not deferred_abandonment
                and not _is_visible_result_verifier(spec)
            ):
                with context.profiler.measure("verification.run"):
                    with _operation_in_flight(progress):
                        verification = await self._verify(
                            context, writer, artifact_checksums
                        )
                _beat(progress)
                context.state = verification[0]
                if verification[1].verified:
                    return _Execution(
                        state=context.state,
                        outcome=VerifiedSuccess(),
                        verification=verification[1],
                        agent_claimed_success=claimed_success,
                        terminal_reason=None,
                    )

            if cycle_length is not None and not interacted_with_target:
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

            target_patience = 5 if interacted_with_target else 3
            stalled_out = (
                context.consecutive_action_count >= target_patience
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
        *,
        capture: ObservationCapture | None = None,
        snapshot: ViewportSnapshot | None = None,
    ) -> _CapturedViewport:
        session = context.session
        if session is None:
            raise RuntimeError("capture requires active session")
        if capture is None:
            with context.profiler.measure("observation.capture.total"):
                capture = await self.observation_provider.capture(session)
        if snapshot is None:
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
        current_identity = _viewport_identity(snapshot)
        context.stage_reset_for_capture = (
            context.current_viewport_identity is None
            or current_identity != context.current_viewport_identity
            or _stage_reset_action(context.previous_action)
        )
        context.current_viewport_identity = current_identity
        duplicate_viewport = any(
            existing.id == snapshot.id for existing in context.state.snapshots
        )
        if not duplicate_viewport:
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
            source_event_id = context.state_event_ids[-1]
        else:
            source_event_id = (
                context.current_capture.source_event_id
                if context.current_capture is not None
                else context.state_event_ids[-1]
            )
        captured = _CapturedViewport(
            capture=capture,
            snapshot=snapshot,
            screenshot_sha256=hashlib.sha256(capture.screenshot).hexdigest(),
            source_event_id=source_event_id,
        )
        context.current_capture = captured
        return captured

    def _score_prominence(
        self,
        captured: _CapturedViewport,
        snapshot: ViewportSnapshot,
        context: _RunContext,
        writer: RunBundleWriter,
    ) -> tuple[
        tuple[ProminenceResult, ...],
        ProminenceBatch | None,
        tuple[ArtifactReference, ...],
    ]:
        stage = _search_stage(self.attention_policy, context, snapshot)
        score_method = self.prominence_provider.score
        if _capture_score_method(score_method):
            tracking_writer = _SaliencyArtifactTrackingWriter(
                writer,
                SaliencyArtifactContext(
                    source_viewport_id=snapshot.id,
                    artifact_namespace=_saliency_artifact_namespace(
                        self.prominence_provider, captured, snapshot
                    ),
                    source_event_id=captured.source_event_id,
                ),
            )
            result = score_method(captured.capture, snapshot, stage, tracking_writer)
            artifact_references = tuple(tracking_writer.references)
        else:
            result = score_method(snapshot)
            artifact_references = ()
        if isinstance(result, ProminenceBatch):
            return tuple(result.scores), result, artifact_references
        if not isinstance(result, Sequence) or isinstance(result, (str, bytes)):
            raise TypeError("prominence provider must return scores or ProminenceBatch")
        return (
            tuple(cast(Sequence[ProminenceResult], result)),
            None,
            artifact_references,
        )

    def _record_prominence(
        self,
        context: _RunContext,
        writer: RunBundleWriter,
        snapshot: ViewportSnapshot,
        *,
        scores: tuple[ProminenceResult, ...],
        batch: ProminenceBatch | None,
        artifact_references: Sequence[ArtifactReference] = (),
    ) -> tuple[int, str | None]:
        if batch is None:
            stage = _search_stage(self.attention_policy, context, snapshot)
            return (
                writer.append_event(
                    {
                        "kind": "prominence-recorded",
                        "viewport_id": snapshot.id,
                        "search_stage": stage.value,
                        "scores": tuple(_prominence_payload(score) for score in scores),
                    }
                ),
                None,
            )

        stage = _search_stage(self.attention_policy, context, snapshot)
        stage_value = stage.value
        metadata = _saliency_runtime_metadata(
            self.prominence_provider,
            context.current_capture,
            batch,
        )
        artifact_identity: tuple[str, tuple[str, ...]] | None = None
        artifact_failure_reason: str | None = None
        profile_event_id: int | None = None
        if batch.learned_available:
            if len(metadata.model_checksums) != 3:
                raise RuntimeError("saliency profile provenance lacks model checksums")
            try:
                artifact_identity = _saliency_artifact_identity(
                    snapshot.id, artifact_references
                )
            except RuntimeError:
                artifact_failure_reason = (
                    "learned saliency artifact evidence unavailable"
                )
            if artifact_identity is not None:
                artifact_viewport_id, artifact_ids = artifact_identity
                profile_event_id = _append_typed_saliency_event(
                    writer,
                    SaliencyProfilesRecordedEvent(
                        viewport_id=artifact_viewport_id,
                        provider_id=metadata.provider_id,
                        search_stage=stage_value,
                        model_checksums=cast(
                            tuple[str, str, str], metadata.model_checksums
                        ),
                        execution_provider=metadata.execution_provider,
                        preprocessing_version=metadata.preprocessing_version,
                        precision=metadata.precision,
                        cache_key=metadata.cache_key,
                        timings_ms=metadata.timings_ms,
                        artifact_ids=artifact_ids,
                        warnings=metadata.warnings,
                        source_viewport_id=snapshot.id,
                        artifact_namespace=artifact_viewport_id,
                        source_event_id=(
                            context.current_capture.source_event_id
                            if context.current_capture is not None
                            else None
                        ),
                        cache_state=batch.cache_state,
                    ),
                )
            else:
                artifact_failure_reason = artifact_failure_reason or (
                    "learned saliency artifact evidence unavailable"
                )
        if not batch.learned_available or artifact_failure_reason is not None:
            reason = (
                artifact_failure_reason
                or batch.fallback_reason
                or batch.learned_unavailable_reason
            )
            if not reason:
                reason = "learned prominence unavailable"
            reason = sanitize_log_text(reason, context.redaction_policy)
            context.saliency_fallback_reason = reason
            _append_typed_saliency_event(
                writer,
                SaliencyFallbackRecordedEvent(
                    viewport_id=snapshot.id,
                    provider_id=str(batch.provider_id),
                    fallback_provider_id=(
                        str(batch.active_provider_id)
                        if not batch.learned_available
                        else "unavailable"
                    ),
                    search_stage=stage_value,
                    reason=reason,
                    model_checksums=metadata.model_checksums,
                    execution_provider=metadata.execution_provider,
                    cache_key=metadata.cache_key,
                    warnings=metadata.warnings,
                    source_viewport_id=snapshot.id,
                    cache_state="fallback",
                ),
            )
        prominence_sequence = _append_typed_saliency_event(
            writer,
            ProminenceRecordedEvent(
                viewport_id=snapshot.id,
                provider_id=str(batch.provider_id),
                active_provider_id=str(batch.active_provider_id),
                search_stage=stage_value,
                selected_mixture=_stage_mixture(self.prominence_provider, batch, stage),
                selected_element_ids=tuple(score.element_id for score in scores),
                source_viewport_id=snapshot.id,
                artifact_namespace=(
                    artifact_identity[0] if artifact_identity is not None else None
                ),
                source_event_id=(
                    _event_id(profile_event_id)
                    if profile_event_id is not None
                    else None
                ),
                cache_state=(
                    batch.cache_state if batch.learned_available else "fallback"
                ),
            ),
        )
        return prominence_sequence, (
            _event_id(profile_event_id) if profile_event_id is not None else None
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
                return await full_scent_evaluator.evaluate(goal, attention, snapshot)

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
        context: _RunContext,
        writer: RunBundleWriter,
        artifact_checksums: list[ArtifactChecksum],
    ) -> tuple[RunState, VerificationResult]:
        session = context.session
        if session is None:
            raise RuntimeError("verification requires active session")
        result = await self.verifier.verify(session)
        result = await self._persist_verification_capture(
            context, writer, artifact_checksums, result
        )
        return (
            _record(
                context.state,
                VerificationRecorded(result=result),
                writer,
                context.state_event_ids,
            ),
            result,
        )

    async def _persist_verification_capture(
        self,
        context: _RunContext,
        writer: RunBundleWriter,
        artifact_checksums: list[ArtifactChecksum],
        result: VerificationResult,
    ) -> VerificationResult:
        capture = getattr(self.verifier, "last_capture", None)
        if not isinstance(capture, ObservationCapture):
            return result
        snapshot = getattr(self.verifier, "last_snapshot", None)
        if not isinstance(snapshot, ViewportSnapshot):
            snapshot = self.snapshot_extractor(capture)
        persisted = await self._capture(
            context,
            writer,
            artifact_checksums,
            capture=capture,
            snapshot=snapshot,
        )
        screenshot_path = persisted.snapshot.screenshot_artifact
        if screenshot_path is None:
            raise RuntimeError("verification capture lacks persisted screenshot")
        screenshot_evidence_id = f"screenshot:{screenshot_path}"
        if screenshot_evidence_id in result.evidence_ids:
            return result
        return replace(
            result,
            evidence_ids=(*result.evidence_ids, screenshot_evidence_id),
        )

    async def _verify_terminal(
        self,
        execution: _Execution,
        writer: RunBundleWriter,
        context: _RunContext,
        artifact_checksums: list[ArtifactChecksum],
    ) -> _Execution:
        if execution.verification is not None:
            return execution

        if _is_visible_result_verifier(execution.state.spec):
            result = execution.state.verification or VerificationResult(
                verified=False,
                details="independent verification unavailable",
            )
            state = execution.state
            if state.verification is None:
                state = _record(
                    state,
                    VerificationRecorded(result=result),
                    writer,
                    context.state_event_ids,
                )
            return replace(execution, state=state, verification=result)

        session = context.session
        if session is None:
            result = VerificationResult(
                verified=False,
                details="independent verification unavailable",
            )
            state = _record(
                execution.state,
                VerificationRecorded(result=result),
                writer,
                context.state_event_ids,
            )
            return replace(execution, state=state, verification=result)

        try:
            result = await self.verifier.verify(session)
            result = await self._persist_verification_capture(
                context, writer, artifact_checksums, result
            )
        except asyncio.CancelledError:
            raise
        except BaseException:
            result = VerificationResult(
                verified=False,
                details="independent verification failed",
            )
            state = _record(
                execution.state,
                VerificationRecorded(result=result),
                writer,
                context.state_event_ids,
            )
            return replace(
                execution,
                state=state,
                verification=result,
            )

        state = _record(
            context.state,
            VerificationRecorded(result=result),
            writer,
            context.state_event_ids,
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
        _validate_materialized_saliency_artifacts(
            writer, context.saliency_artifact_references
        )
        terminal_artifact_checksums = list(artifact_checksums)
        known_artifacts = {
            (item.path, item.sha256) for item in terminal_artifact_checksums
        }
        for reference in context.saliency_artifact_references:
            checksum = _artifact_checksum(reference)
            identity = (checksum.path, checksum.sha256)
            if identity not in known_artifacts:
                terminal_artifact_checksums.append(checksum)
                known_artifacts.add(identity)
        manifests = self._provider_manifests(spec)
        terminal = RunTerminated(
            outcome=execution.outcome,
            verification=verification,
            provider_manifests=manifests,
            configuration_digest=spec.config_digest,
            artifact_checksums=tuple(terminal_artifact_checksums),
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
        if context.saliency_fallback_reason is not None:
            ux_sample_valid = False
            ux_sample_invalid_reason = "saliency-fallback: " + sanitize_log_text(
                context.saliency_fallback_reason,
                context.redaction_policy,
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
        result = _enforce_saliency_fallback_validity(
            result,
            context.saliency_fallback_reason,
            context.redaction_policy,
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


def _capture_score_method(method: object) -> bool:
    """Detect capture-aware functions, decorated functions, and callable objects."""

    try:
        inspect.signature(method).bind(object(), object(), object(), object())
    except (TypeError, ValueError):
        return False
    return True


def _search_stage(
    attention_policy: object,
    context: _RunContext,
    snapshot: ViewportSnapshot,
) -> object:
    if (
        context.stage_reset_for_capture
        or context.current_viewport_identity is None
        or context.observed_viewport_identity != context.current_viewport_identity
    ):
        from ux_analyzer.domain.saliency import SearchStage

        return SearchStage.INITIAL
    threshold = getattr(
        getattr(attention_policy, "config", None), "recovery_after_misses", 2
    )
    if context.no_progress_count >= threshold:
        from ux_analyzer.domain.saliency import SearchStage

        return SearchStage.PERSISTENT
    from ux_analyzer.domain.saliency import SearchStage

    return SearchStage.EXPLORATION


def _stage_reset_action(action: Mapping[str, object] | None) -> bool:
    if action is None:
        return False
    return any(
        action_kind
        in {
            "back",
            "close-menu",
            "click",
            "navigate",
            "open-menu",
            "scroll",
            "toggle",
        }
        for action_kind in (
            action.get("kind"),
            action.get("platform_action_kind"),
        )
    )


def _viewport_identity(snapshot: ViewportSnapshot) -> tuple[object, ...]:
    """Use semantic capture identity; viewport and region IDs are namespaces."""

    return snapshot_progress_signature(snapshot)


def _saliency_artifact_namespace(
    provider: object,
    captured: _CapturedViewport,
    snapshot: ViewportSnapshot,
) -> str:
    cache_key = getattr(provider, "cache_key", None)
    if not isinstance(cache_key, str):
        model_provider = getattr(provider, "_model_provider", None)
        get_model_provider = getattr(provider, "_get_model_provider", None)
        if model_provider is None and callable(get_model_provider):
            try:
                model_provider = get_model_provider()
            except Exception:
                model_provider = None
        request_builder = getattr(provider, "_request", None)
        key_builder = getattr(provider, "_request_cache_key", None)
        if (
            model_provider is not None
            and callable(request_builder)
            and callable(key_builder)
        ):
            try:
                request = request_builder(captured.capture, model_provider)
                key = key_builder(request, model_provider)
                candidate = getattr(key, "digest", None)
                if isinstance(candidate, str):
                    cache_key = candidate
            except Exception:
                cache_key = None
    identity = json.dumps(
        {
            "cache_key": cache_key,
            "screenshot_sha256": captured.screenshot_sha256,
            "viewport_id": snapshot.id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"inference-{hashlib.sha256(identity).hexdigest()}"


def _stage_mixture(
    provider: object,
    batch: ProminenceBatch,
    stage: object,
) -> tuple[tuple[str, float], ...]:
    value = str(getattr(stage, "value", stage))
    configured = getattr(batch, "selected_mixture", None)
    if isinstance(configured, tuple) and configured:
        return tuple((str(duration), float(weight)) for duration, weight in configured)
    if configured is None:
        selector = getattr(provider, "stage_selector", None)
        mixtures = getattr(selector, "mixtures", None)
        if isinstance(mixtures, Mapping):
            configured = mixtures.get(stage, mixtures.get(value))
    if isinstance(configured, Mapping):
        durations = tuple(
            (str(getattr(duration, "value", duration)), float(weight))
            for duration, weight in configured.items()
            if isinstance(weight, (int, float))
            and not isinstance(weight, bool)
            and weight > 0
        )
        if durations:
            return durations
    return {
        "initial": (("1s", 1.0),),
        "exploration": (("3s", 1.0),),
        "persistent": (("3s", 0.25), ("7s", 0.75)),
    }[value]


def _saliency_artifact_identity(
    snapshot_id: str,
    references: Sequence[ArtifactReference],
) -> tuple[str, tuple[str, ...]]:
    if not references:
        raise RuntimeError("learned saliency artifact references are missing")
    _validate_saliency_references(references)
    viewport_ids = {PurePosixPath(reference.path).parts[1] for reference in references}
    if len(viewport_ids) != 1:
        raise RuntimeError(
            "saliency artifacts must belong to one materialized viewport"
        )
    viewport_id = next(iter(viewport_ids))
    actual_paths = {reference.path for reference in references}
    if actual_paths != set(required_saliency_artifact_paths(viewport_id)):
        raise RuntimeError("saliency artifacts must cover one viewport exactly")
    return viewport_id, tuple(sorted(actual_paths))


def _validate_saliency_references(
    references: Sequence[ArtifactReference],
) -> None:
    paths: set[str] = set()
    for reference in references:
        validate_saliency_artifact_path(reference.path)
        if reference.name != reference.path:
            raise RuntimeError("saliency artifact reference name must match path")
        if (
            len(reference.sha256) != 64
            or reference.sha256 != reference.sha256.lower()
            or any(
                character not in "0123456789abcdef" for character in reference.sha256
            )
        ):
            raise RuntimeError("saliency artifact reference checksum is invalid")
        if reference.size < 0:
            raise RuntimeError("saliency artifact reference size must not be negative")
        if reference.path in paths:
            raise RuntimeError("saliency artifact references must be unique")
        paths.add(reference.path)


def _validate_materialized_saliency_artifacts(
    writer: RunBundleWriter,
    references: Sequence[ArtifactReference],
) -> None:
    if not references:
        return
    _validate_saliency_references(references)
    for reference in references:
        try:
            writer.verify_artifact(reference)
        except Exception as error:
            raise RuntimeError(str(error)) from error


def _append_typed_saliency_event(
    writer: RunBundleWriter,
    event: object,
) -> int:
    append = getattr(writer, "append_saliency_event", None)
    if not callable(append):
        raise TypeError("capture-aware prominence requires typed saliency event writer")
    return int(append(event))


def _saliency_runtime_metadata(
    provider: object,
    captured: _CapturedViewport | None,
    batch: ProminenceBatch,
) -> _SaliencyRuntimeMetadata:
    model_provider = getattr(provider, "_model_provider", None)
    get_model_provider = getattr(provider, "_get_model_provider", None)
    if model_provider is None and callable(get_model_provider):
        try:
            model_provider = get_model_provider()
        except Exception:
            model_provider = None

    metadata_by_duration: dict[str, object] = {}
    for profile in batch.learned_profiles:
        for provenance in getattr(profile, "prediction_provenance", ()):
            raw_duration = getattr(provenance, "duration", "")
            duration_value = getattr(raw_duration, "value", raw_duration)
            duration = str(duration_value)
            metadata_by_duration[duration] = getattr(provenance, "metadata", None)

    configured_checksums = getattr(model_provider, "model_checksums", None)
    if configured_checksums is None:
        configured_checksums = getattr(provider, "model_checksums", None)
    checksums: list[str] = []
    timings: list[float] = []
    warnings: list[str] = []
    for duration in ("1s", "3s", "7s"):
        metadata = metadata_by_duration.get(duration)
        checksum = getattr(metadata, "model_checksum", None)
        if checksum is None and isinstance(configured_checksums, Mapping):
            checksum = configured_checksums.get(duration)
        if isinstance(checksum, str) and len(checksum) == 64:
            checksums.append(checksum)
        timing = getattr(metadata, "inference_duration_ms", None)
        if isinstance(timing, (int, float)) and not isinstance(timing, bool):
            timings.append(float(timing))
        for warning in getattr(metadata, "warnings", ()):
            if isinstance(warning, str):
                warnings.append(warning)

    provider_id = next(
        (
            str(getattr(metadata_by_duration[duration], "provider_id"))
            for duration in ("1s", "3s", "7s")
            if duration in metadata_by_duration
            and getattr(metadata_by_duration[duration], "provider_id", None)
        ),
        str(batch.active_provider_id),
    )
    execution_provider = next(
        (
            str(getattr(metadata_by_duration[duration], "execution_provider"))
            for duration in ("1s", "3s", "7s")
            if duration in metadata_by_duration
            and getattr(metadata_by_duration[duration], "execution_provider", None)
        ),
        str(
            getattr(
                model_provider,
                "actual_execution_provider",
                getattr(provider, "actual_execution_provider", "CPUExecutionProvider"),
            )
        ),
    )
    preprocessing_version = next(
        (
            str(getattr(metadata_by_duration[duration], "preprocessing_version"))
            for duration in ("1s", "3s", "7s")
            if duration in metadata_by_duration
            and getattr(metadata_by_duration[duration], "preprocessing_version", None)
        ),
        str(
            getattr(
                model_provider,
                "preprocessing_version",
                getattr(provider, "preprocessing_version", "foveacast-preprocess-v1"),
            )
        ),
    )
    precision = str(
        getattr(
            model_provider,
            "precision",
            getattr(provider, "precision", "fp16"),
        )
    )
    cache_key = getattr(provider, "cache_key", None)
    if not isinstance(cache_key, str):
        cache_key = None
    if cache_key is None and captured is not None and model_provider is not None:
        request_builder = getattr(provider, "_request", None)
        key_builder = getattr(provider, "_request_cache_key", None)
        if callable(request_builder) and callable(key_builder):
            try:
                request = request_builder(captured.capture, model_provider)
                key = key_builder(request, model_provider)
                digest = getattr(key, "digest", None)
                if isinstance(digest, str):
                    cache_key = digest
            except Exception:
                cache_key = None
    return _SaliencyRuntimeMetadata(
        provider_id=provider_id,
        model_checksums=tuple(checksums),
        execution_provider=execution_provider,
        preprocessing_version=preprocessing_version,
        precision=precision,
        cache_key=cache_key,
        timings_ms=tuple(timings) if len(timings) == 3 else (),
        warnings=tuple(warnings),
    )


def _require_application_state(state: object) -> ApplicationState:
    if not isinstance(state, ApplicationState):
        raise RuntimeError("application state transition returned invalid state")
    return state


def _is_attention_exhausted_error(error: BaseException) -> bool:
    message = str(error)
    return message == _ATTENTION_EXHAUSTED_MESSAGE or message.startswith(
        "no unobserved element:"
    )


def _effective_memory_capacity(
    spec: RunSpec, memory_policy: MemoryPolicy | None
) -> int:
    persona_capacity = spec.persona.working_memory_capacity
    if memory_policy is None:
        return persona_capacity
    return min(persona_capacity, memory_policy.config.working_capacity)


def _has_unobserved_visible_element(
    state: ApplicationState, snapshot: ViewportSnapshot
) -> bool:
    noticed_ids = state.attention.noticed_ids
    return any(
        element.visibility_fraction > 0 and element.id not in noticed_ids
        for element in snapshot.elements
    )


def _is_visible_result_verifier(spec: RunSpec) -> bool:
    return isinstance(spec.scenario.verifier, VisibleResultVerifierSpec)


def _accept_abandonment(context: _RunContext, spec: RunSpec) -> bool:
    """Accept model abandonment only after bounded search can no longer continue."""

    application_state = context.application_state
    attention = application_state.attention
    return (
        application_state.abandoned
        or attention.frustration >= spec.persona.abandonment_threshold
        or context.consecutive_action_count >= 3
        or context.no_progress_count >= 5
        or attention.budgets.steps <= 0
        or attention.budgets.observations <= 0
    )


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


def _model_call_persistence_dict(record: ModelCallRecord) -> dict[str, object]:
    """Persist model-call audit metadata without request or response payloads."""

    return {
        "role": record.role,
        "model": record.model,
        "endpoint_origin": record.endpoint_origin,
        "prompt_digest": record.prompt_digest,
        "schema_version": record.schema_version,
        "attempts": record.attempts,
        "latency_ms": record.latency_ms,
        "token_usage": record.token_usage,
        "retries": record.retries,
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
    platform_action_kind = getattr(validated.platform_action, "kind", None)
    if isinstance(platform_action_kind, str):
        result["platform_action_kind"] = platform_action_kind
    return result


def _agent_claim(decision: object) -> bool:
    action = getattr(decision, "action", decision)
    if getattr(action, "kind", None) == "complete":
        return True
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


def _enforce_saliency_fallback_validity(
    result: RunResult,
    fallback_reason: str | None,
    redaction_policy: RedactionPolicy,
) -> RunResult:
    """Make fallback invalidity authoritative after optional result evaluation."""

    if fallback_reason is None:
        return result
    safe_reason = sanitize_log_text(fallback_reason, redaction_policy)
    invalid_reason = f"saliency-fallback: {safe_reason}"
    metrics = result.metrics
    if metrics is not None:
        metrics = replace(
            metrics,
            prominence_fallback=True,
            prominence_fallback_reason=safe_reason,
            comparison_valid=False,
        )
    return replace(
        result,
        metrics=metrics,
        findings=None,
        ux_sample_valid=False,
        ux_sample_invalid_reason=invalid_reason,
    )


def _ux_sample_validity(
    outcome: RunOutcome,
    terminal_reason: str | None,
) -> tuple[bool, str | None]:
    kind = str(getattr(outcome, "kind", "internal-error"))
    if kind in {"verified-success", "agent-abandoned"}:
        return True, None
    reason = terminal_reason or getattr(outcome, "reason", None) or "run unavailable"
    if kind == "budget-exhausted":
        normalized_reason = " ".join(reason.casefold().split())
        if normalized_reason in _HUMAN_BUDGET_TERMINAL_REASONS:
            return True, None
    return False, f"{kind}: {reason}"


_RETRYABLE_INTERNAL_SUBSTRINGS: tuple[str, ...] = (
    "cannot inspect",
    "unremembered",
    "unnoticed",
    "not present in viewport",
    "observation references element not present",
)

_RETRYABLE_MODEL_SUBSTRINGS: tuple[str, ...] = (
    "rate-limit",
    "429",
    "502",
    "503",
    "500 server error",
    "server error",
    "timeout",
    "transient",
    "overloaded",
)


def _is_retryable_result(result: RunResult | _Execution) -> bool:
    """Internal/model flakes that deserve a fresh-context retry without a user flag.

    Real UX signals (verified-success, agent-abandoned, budget-exhausted,
    or an evaluation failure with a valid sample) are never retried here.
    Only hallucination-style internal errors and transient model failures are
    retried — deterministic test fixtures stay single-attempt so tests keep
    their single-call expectations.
    """

    outcome = getattr(result, "outcome", None)
    kind = str(getattr(outcome, "kind", "")) if outcome is not None else ""
    if isinstance(result, RunResult) and result.ux_sample_valid:
        return False
    reason_parts: list[str] = []
    for attr in ("terminal_reason", "ux_sample_invalid_reason"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value:
            reason_parts.append(value)
    if outcome is not None:
        for attr in ("reason", "message"):
            value = getattr(outcome, attr, None)
            if isinstance(value, str) and value:
                reason_parts.append(value)
    reason = " ".join(reason_parts).casefold()
    if kind == "internal-error":
        return any(sub in reason for sub in _RETRYABLE_INTERNAL_SUBSTRINGS)
    if kind == "model-failure":
        return any(sub in reason for sub in _RETRYABLE_MODEL_SUBSTRINGS)
    return False


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
        terminal_reason=execution.terminal_reason or "run timeout exceeded",
    )


def _abort_bundle(writer: RunBundleWriter, reason: str) -> object | None:
    try:
        return writer.abort(reason or "run aborted")
    except BaseException:
        return None
