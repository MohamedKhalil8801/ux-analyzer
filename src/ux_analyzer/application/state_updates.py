"""Deterministic application-layer attention state transitions."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ux_analyzer.domain.attention import (
    Abandon,
    AttentionAction,
    AttentionState,
    Back,
    InspectElement,
    InteractWithElement,
    NoticeElements,
    PersonaObservation,
    RememberedElement,
    Scroll,
    Wait,
)
from ux_analyzer.domain.interface import ViewportSnapshot
from ux_analyzer.ports.observation import (
    BackAction,
    ClearTextAction,
    ClickAction,
    DoubleClickAction,
    DragAction,
    OpenMenuAction,
    PlatformAction,
    PlatformActionResult,
    ScrollAction,
    SelectOptionAction,
    SubmitAction,
    ToggleAction,
    TypeTextAction,
    WaitAction,
)
from ux_analyzer.providers.memory import (
    MemoryEntry,
    MemoryPolicy,
    MemoryPolicyConfig,
    MemoryState,
)


@dataclass(frozen=True, slots=True)
class StateUpdateConfig:
    """Versioned, model-independent state transition formulas."""

    version: str = "state-updates-v1"
    success_confidence_delta: float = 0.05
    success_frustration_delta: float = -0.1
    failure_confidence_delta: float = -0.1
    failure_frustration_delta: float = 0.2
    abandonment_threshold: float = 0.9

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("state update version must not be empty")
        for name, value in (
            ("success_confidence_delta", self.success_confidence_delta),
            ("success_frustration_delta", self.success_frustration_delta),
            ("failure_confidence_delta", self.failure_confidence_delta),
            ("failure_frustration_delta", self.failure_frustration_delta),
            ("abandonment_threshold", self.abandonment_threshold),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.abandonment_threshold <= 1:
            raise ValueError("abandonment_threshold must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class ApplicationState:
    """Application state joining domain attention with episodic memory."""

    attention: AttentionState
    memory: MemoryState
    abandoned: bool = False
    abandonment_reason: str | None = None

    @classmethod
    def from_attention(
        cls,
        attention: AttentionState,
        memory: MemoryState | None = None,
    ) -> ApplicationState:
        if memory is None:
            memory = MemoryState(
                working=tuple(
                    MemoryEntry(
                        key=item.element_id,
                        value=item.label,
                        viewport_id=item.viewport_id,
                    )
                    for item in attention.memory
                )
            )
        return cls(attention=attention, memory=memory)

    @property
    def budgets(self):
        return self.attention.budgets

    @property
    def confidence(self) -> float:
        return self.attention.confidence

    @property
    def frustration(self) -> float:
        return self.attention.frustration


StateLike = AttentionState | ApplicationState


def apply_observation(
    state: StateLike,
    observation: PersonaObservation,
    *,
    snapshot: ViewportSnapshot | None = None,
    memory_policy: MemoryPolicy | None = None,
) -> StateLike:
    """Consume observation budget and update bounded working memory."""

    attention, memory, abandoned, reason = _parts(state)
    if snapshot is not None and snapshot.id != observation.viewport_id:
        raise ValueError("observation references stale viewport")
    next_attention = attention.after_observation(observation)
    policy = memory_policy or MemoryPolicy(
        config=MemoryPolicyConfig(working_capacity=attention.memory_capacity)
    )
    next_memory = policy.update(memory, observation=observation)
    next_attention = _sync_working_memory(next_attention, next_memory)
    return _build(state, next_attention, next_memory, abandoned, reason)


def apply_interaction_result(
    state: StateLike,
    action: AttentionAction | PlatformAction,
    result: PlatformActionResult | bool,
    *,
    snapshot: ViewportSnapshot | None = None,
    config: StateUpdateConfig | None = None,
    memory_policy: MemoryPolicy | None = None,
) -> StateLike:
    """Consume action budgets and apply deterministic success/failure effects."""

    settings = config or StateUpdateConfig()
    attention, memory, abandoned, reason = _parts(state)
    domain_action = _as_attention_action(action)
    if snapshot is not None:
        attention.validate_action(domain_action, snapshot)
    next_attention = attention.after_action(domain_action)
    succeeded = result if isinstance(result, bool) else result.succeeded
    if succeeded:
        next_attention = _adjust_emotion(
            next_attention,
            confidence_delta=settings.success_confidence_delta,
            frustration_delta=settings.success_frustration_delta,
        )
        next_memory = memory
    else:
        next_attention = _adjust_emotion(
            next_attention,
            confidence_delta=settings.failure_confidence_delta,
            frustration_delta=settings.failure_frustration_delta,
        )
        target_id = _target_id(domain_action)
        if target_id is not None:
            next_attention = next_attention.mark_failed_candidate(target_id)
            failure = MemoryEntry.failure(
                target_id,
                getattr(result, "error", None) or "interaction-failed",
                viewport_id=next_attention.current_viewport_id,
            )
            policy = memory_policy or MemoryPolicy()
            next_memory = policy.update(memory, failures=(failure,), elapsed_steps=0)
        else:
            next_memory = memory
    next_abandoned, next_reason = _abandonment(
        next_attention, settings, abandoned, reason
    )
    return _build(state, next_attention, next_memory, next_abandoned, next_reason)


def apply_failure(
    state: StateLike,
    element_id: str | None = None,
    *,
    reason: str = "failure",
    config: StateUpdateConfig | None = None,
    memory_policy: MemoryPolicy | None = None,
) -> StateLike:
    """Record failure, clamp emotion, retain failure memory, and signal abandonment."""

    settings = config or StateUpdateConfig()
    attention, memory, abandoned, previous_reason = _parts(state)
    next_attention = _adjust_emotion(
        attention,
        confidence_delta=settings.failure_confidence_delta,
        frustration_delta=settings.failure_frustration_delta,
    )
    if element_id:
        next_attention = next_attention.mark_failed_candidate(element_id)
    next_memory = memory
    if element_id:
        policy = memory_policy or MemoryPolicy()
        next_memory = policy.update(
            memory,
            failures=(
                MemoryEntry.failure(
                    element_id,
                    reason,
                    viewport_id=next_attention.current_viewport_id,
                ),
            ),
            elapsed_steps=0,
        )
    next_abandoned, next_reason = _abandonment(
        next_attention, settings, abandoned, previous_reason
    )
    return _build(state, next_attention, next_memory, next_abandoned, next_reason)


def should_abandon(
    state: AttentionState | ApplicationState,
    threshold: float,
) -> bool:
    """Return whether clamped frustration reaches configured threshold."""

    if not 0 <= threshold <= 1:
        raise ValueError("abandonment threshold must be between 0 and 1")
    attention = state.attention if isinstance(state, ApplicationState) else state
    return attention.frustration >= threshold


def _parts(
    state: StateLike,
) -> tuple[AttentionState, MemoryState, bool, str | None]:
    if isinstance(state, ApplicationState):
        return state.attention, state.memory, state.abandoned, state.abandonment_reason
    application_state = ApplicationState.from_attention(state)
    return application_state.attention, application_state.memory, False, None


def _build(
    original: StateLike,
    attention: AttentionState,
    memory: MemoryState,
    abandoned: bool,
    reason: str | None,
) -> StateLike:
    if isinstance(original, ApplicationState):
        return replace(
            original,
            attention=attention,
            memory=memory,
            abandoned=abandoned,
            abandonment_reason=reason,
        )
    return attention  # type: ignore[return-value]


def _sync_working_memory(
    attention: AttentionState,
    memory: MemoryState,
) -> AttentionState:
    remembered = tuple(
        RememberedElement(
            element_id=entry.key,
            viewport_id=entry.viewport_id or attention.current_viewport_id or "unknown",
            label=entry.value,
        )
        for entry in memory.working[-attention.memory_capacity :]
    )
    return replace(attention, memory=remembered)


def _adjust_emotion(
    attention: AttentionState,
    *,
    confidence_delta: float,
    frustration_delta: float,
) -> AttentionState:
    return replace(
        attention,
        confidence=_clamp(attention.confidence + confidence_delta),
        frustration=_clamp(attention.frustration + frustration_delta),
    )


def _abandonment(
    attention: AttentionState,
    config: StateUpdateConfig,
    already_abandoned: bool,
    previous_reason: str | None,
) -> tuple[bool, str | None]:
    if already_abandoned:
        return True, previous_reason
    if should_abandon(attention, config.abandonment_threshold):
        return True, "abandonment-threshold-crossed"
    return False, None


def _as_attention_action(action: AttentionAction | PlatformAction) -> AttentionAction:
    if isinstance(
        action,
        (
            Abandon,
            Back,
            InspectElement,
            InteractWithElement,
            NoticeElements,
            Scroll,
            Wait,
        ),
    ):
        return action
    if isinstance(action, ScrollAction):
        return Scroll(direction=action.direction)
    if isinstance(action, WaitAction):
        return Wait(duration_seconds=action.milliseconds / 1000)
    if isinstance(action, BackAction):
        return Back()
    if isinstance(
        action,
        (
            ClickAction,
            DoubleClickAction,
            TypeTextAction,
            ClearTextAction,
            SelectOptionAction,
            ToggleAction,
            SubmitAction,
            OpenMenuAction,
            DragAction,
        ),
    ):
        target_id = _target_id(action)
        if target_id is not None:
            return InteractWithElement(element_id=target_id)
    raise ValueError("action result needs supported action")


def _target_id(action: object) -> str | None:
    value = getattr(action, "element_id", None)
    return value if isinstance(value, str) else None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
