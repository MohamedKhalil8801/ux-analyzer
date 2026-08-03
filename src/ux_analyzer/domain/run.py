"""Immutable benchmark run state and its single event reducer."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Literal

from ux_analyzer.domain.attention import (
    AttentionAction,
    AttentionState,
    InteractWithElement,
    PersonaObservation,
)
from ux_analyzer.domain.benchmark import (
    ApplicationVersion,
    ExperimentPolicy,
    Persona,
    Scenario,
    resolve_prominence_provider_id,
)
from ux_analyzer.domain.interface import (
    PrivateExecutionReference,
    ViewportSnapshot,
)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Independent verifier result, separate from agent claims."""

    verified: bool
    evidence_ids: tuple[str, ...] = ()
    details: str | None = None

    def __post_init__(self) -> None:
        evidence_ids = tuple(self.evidence_ids)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("verification contains duplicate evidence ID")
        object.__setattr__(self, "evidence_ids", evidence_ids)


@dataclass(frozen=True, slots=True)
class ProviderManifest:
    """Sanitized provider identity recorded in a finalized run."""

    provider_id: str
    role: str
    model_id: str | None
    endpoint_origin: str
    version: str
    prompt_version: str | None = None
    schema_version: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("provider_id", self.provider_id),
            ("role", self.role),
            ("endpoint_origin", self.endpoint_origin),
            ("version", self.version),
        ):
            if not value:
                raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True, slots=True)
class ArtifactChecksum:
    """Checksum for one finalized artifact."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.path or not self.sha256:
            raise ValueError("artifact checksum path and digest are required")


@dataclass(frozen=True, slots=True)
class RunSpec:
    """One immutable scenario/version/persona/policy/seed/trial assignment."""

    run_id: str
    seed: int
    scenario: Scenario
    application_version: ApplicationVersion
    persona: Persona
    policy: ExperimentPolicy
    config_digest: str
    model_trial: int = 0
    prominence_provider_id: str = "heuristic"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if not self.config_digest:
            raise ValueError("config_digest must not be empty")
        resolve_prominence_provider_id(self.prominence_provider_id)
        if self.application_version.id not in self.scenario.application_version_ids:
            raise ValueError("application version is not eligible for scenario")
        if self.persona.id not in self.scenario.eligible_persona_ids:
            raise ValueError("persona is not eligible for scenario")


class RunStatus(StrEnum):
    """Lifecycle status values."""

    CREATED = "created"
    RUNNING = "running"
    FINALIZED = "finalized"


class RunOutcomeKind(StrEnum):
    """Closed terminal outcome discriminator values."""

    VERIFIED_SUCCESS = "verified-success"
    AGENT_ABANDONED = "agent-abandoned"
    BUDGET_EXHAUSTED = "budget-exhausted"
    TIMED_OUT = "timed-out"
    PROVIDER_FAILURE = "provider-failure"
    MODEL_FAILURE = "model-failure"
    SAFETY_BLOCKED = "safety-blocked"
    INTERNAL_ERROR = "internal-error"


class RunEventKind(StrEnum):
    """Closed event discriminator values."""

    RUN_STARTED = "run-started"
    VIEWPORT_CAPTURED = "viewport-captured"
    OBSERVATION_RECORDED = "observation-recorded"
    ACTION_PROPOSED = "action-proposed"
    ACTION_EXECUTED = "action-executed"
    VERIFICATION_RECORDED = "verification-recorded"
    RUN_TERMINATED = "run-terminated"


@dataclass(frozen=True, slots=True)
class VerifiedSuccess:
    kind: Literal["verified-success"] = "verified-success"


@dataclass(frozen=True, slots=True)
class AgentAbandoned:
    kind: Literal["agent-abandoned"] = "agent-abandoned"
    reason: str = ""


@dataclass(frozen=True, slots=True)
class BudgetExhausted:
    kind: Literal["budget-exhausted"] = "budget-exhausted"


@dataclass(frozen=True, slots=True)
class TimedOut:
    kind: Literal["timed-out"] = "timed-out"


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    kind: Literal["provider-failure"] = "provider-failure"
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ModelFailure:
    kind: Literal["model-failure"] = "model-failure"
    reason: str = ""


@dataclass(frozen=True, slots=True)
class SafetyBlocked:
    kind: Literal["safety-blocked"] = "safety-blocked"
    reason: str = ""


@dataclass(frozen=True, slots=True)
class InternalError:
    kind: Literal["internal-error"] = "internal-error"
    reason: str = ""


type RunOutcome = (
    VerifiedSuccess
    | AgentAbandoned
    | BudgetExhausted
    | TimedOut
    | ProviderFailure
    | ModelFailure
    | SafetyBlocked
    | InternalError
)


@dataclass(frozen=True, slots=True)
class RunStarted:
    kind: Literal["run-started"] = "run-started"
    run_id: str = ""


@dataclass(frozen=True, slots=True)
class ViewportCaptured:
    kind: Literal["viewport-captured"] = "viewport-captured"
    snapshot: ViewportSnapshot = None  # type: ignore[assignment]
    viewport_width: int | None = None
    viewport_height: int | None = None

    def __post_init__(self) -> None:
        dimensions = (self.viewport_width, self.viewport_height)
        if any(value is not None and value <= 0 for value in dimensions):
            raise ValueError("viewport dimensions must be positive")
        if (self.viewport_width is None) != (self.viewport_height is None):
            raise ValueError("viewport dimensions must be recorded together")


@dataclass(frozen=True, slots=True)
class ObservationRecorded:
    kind: Literal["observation-recorded"] = "observation-recorded"
    observation: PersonaObservation = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class ActionProposed:
    kind: Literal["action-proposed"] = "action-proposed"
    action: AttentionAction = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class ActionExecuted:
    kind: Literal["action-executed"] = "action-executed"
    action: AttentionAction = None  # type: ignore[assignment]
    viewport_id: str = ""
    execution_reference: PrivateExecutionReference | None = None
    succeeded: bool = True
    error: str | None = None
    platform_action_kind: str | None = None
    navigation_occurred: bool = False
    state_changed: bool = False


@dataclass(frozen=True, slots=True)
class VerificationRecorded:
    kind: Literal["verification-recorded"] = "verification-recorded"
    result: VerificationResult = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class RunTerminated:
    kind: Literal["run-terminated"] = "run-terminated"
    outcome: RunOutcome = None  # type: ignore[assignment]
    verification: VerificationResult | None = None
    provider_manifests: tuple[ProviderManifest, ...] = ()
    configuration_digest: str = ""
    artifact_checksums: tuple[ArtifactChecksum, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_manifests", tuple(self.provider_manifests))
        object.__setattr__(self, "artifact_checksums", tuple(self.artifact_checksums))


type RunEvent = (
    RunStarted
    | ViewportCaptured
    | ObservationRecorded
    | ActionProposed
    | ActionExecuted
    | VerificationRecorded
    | RunTerminated
)


@dataclass(frozen=True, slots=True)
class RunState:
    """Immutable state; ``apply`` is sole lifecycle transition API."""

    spec: RunSpec
    status: str
    events: tuple[RunEvent, ...]
    snapshots: tuple[ViewportSnapshot, ...]
    attention: AttentionState
    outcome: RunOutcome | None
    verification: VerificationResult | None
    provider_manifests: tuple[ProviderManifest, ...]
    configuration_digest: str | None
    artifact_checksums: tuple[ArtifactChecksum, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", RunStatus(self.status))
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "snapshots", tuple(self.snapshots))
        object.__setattr__(self, "provider_manifests", tuple(self.provider_manifests))
        object.__setattr__(self, "artifact_checksums", tuple(self.artifact_checksums))

    @classmethod
    def initial(cls, spec: RunSpec) -> RunState:
        return cls(
            spec=spec,
            status=RunStatus.CREATED,
            events=(),
            snapshots=(),
            attention=AttentionState.initial(
                budget=spec.scenario.budget,
                confidence=spec.persona.initial_confidence,
                frustration=spec.persona.initial_frustration,
                memory_capacity=spec.persona.working_memory_capacity,
                current_subgoal=spec.scenario.goal,
            ),
            outcome=None,
            verification=None,
            provider_manifests=(),
            configuration_digest=None,
            artifact_checksums=(),
        )

    @property
    def is_finalized(self) -> bool:
        return self.status == RunStatus.FINALIZED

    @property
    def current_snapshot(self) -> ViewportSnapshot | None:
        return self.snapshots[-1] if self.snapshots else None

    def apply(self, event: RunEvent) -> RunState:
        """Apply one valid event and return new immutable state."""

        if self.is_finalized:
            raise ValueError("cannot apply event after terminal outcome")
        if isinstance(event, RunStarted):
            if self.status != RunStatus.CREATED:
                raise ValueError("run can only start from created state")
            if event.run_id != self.spec.run_id:
                raise ValueError("run-started event has wrong run ID")
            return replace(
                self,
                status=RunStatus.RUNNING,
                events=self.events + (event,),
            )
        if self.status != RunStatus.RUNNING:
            raise ValueError("run event requires running state")
        if isinstance(event, ViewportCaptured):
            if any(snapshot.id == event.snapshot.id for snapshot in self.snapshots):
                raise ValueError("duplicate viewport snapshot ID")
            return replace(
                self,
                snapshots=self.snapshots + (event.snapshot,),
                events=self.events + (event,),
            )
        if isinstance(event, ObservationRecorded):
            current_snapshot = self.current_snapshot
            if current_snapshot is None:
                raise ValueError("observation requires captured viewport")
            if event.observation.viewport_id != current_snapshot.id:
                raise ValueError("observation references stale viewport")
            captured_ids = {element.id for element in current_snapshot.elements}
            observed_ids = {
                element.id
                for element in (
                    *event.observation.newly_revealed_elements,
                    *event.observation.remembered_elements,
                )
            }
            if not observed_ids.issubset(captured_ids):
                raise ValueError(
                    "observation references element not present in viewport"
                )
            return replace(
                self,
                attention=self.attention.after_observation(event.observation),
                events=self.events + (event,),
            )
        if isinstance(event, ActionProposed):
            current_snapshot = self.current_snapshot
            if current_snapshot is None:
                raise ValueError("action requires captured viewport")
            self.attention.validate_action(event.action, current_snapshot)
            return replace(self, events=self.events + (event,))
        if isinstance(event, ActionExecuted):
            current_snapshot = self.current_snapshot
            if current_snapshot is None:
                raise ValueError("action requires captured viewport")
            if event.viewport_id != current_snapshot.id:
                raise ValueError("action executed against stale viewport")
            self.attention.validate_action(event.action, current_snapshot)
            if isinstance(event.action, InteractWithElement):
                reference = event.execution_reference
                if reference is None:
                    raise ValueError("interaction execution requires private reference")
                current_snapshot.validate_execution_reference(
                    event.action.element_id, reference
                )
            return replace(
                self,
                attention=self.attention.after_action(event.action),
                events=self.events + (event,),
            )
        if isinstance(event, VerificationRecorded):
            return replace(
                self,
                verification=event.result,
                events=self.events + (event,),
            )
        verification = event.verification or self.verification
        if verification is None:
            raise ValueError("finalized run requires verification result")
        if not event.provider_manifests:
            raise ValueError("finalized run requires provider manifests")
        if event.configuration_digest != self.spec.config_digest:
            raise ValueError("finalized run configuration digest mismatch")
        if not event.artifact_checksums:
            raise ValueError("finalized run requires artifact checksums")
        return replace(
            self,
            status=RunStatus.FINALIZED,
            events=self.events + (event,),
            outcome=event.outcome,
            verification=verification,
            provider_manifests=tuple(event.provider_manifests),
            configuration_digest=event.configuration_digest,
            artifact_checksums=tuple(event.artifact_checksums),
        )
