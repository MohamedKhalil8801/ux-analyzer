from __future__ import annotations

from dataclasses import replace

from ux_analyzer.application.state_updates import (
    ApplicationState,
    StateUpdateConfig,
    apply_failure,
    apply_interaction_result,
    apply_observation,
    reconcile_snapshot_state,
)
from ux_analyzer.domain.attention import (
    AttentionState,
    InspectElement,
    InteractWithElement,
    ProgressiveObservation,
    Scroll,
)
from ux_analyzer.domain.benchmark import Budget
from ux_analyzer.domain.interface import (
    BoundingBox,
    ElementSnapshot,
    PrivateExecutionReference,
    ViewportSnapshot,
)
from ux_analyzer.ports.observation import PlatformActionResult
from ux_analyzer.providers.memory import (
    MemoryEntry,
    MemoryPolicy,
    MemoryPolicyConfig,
    MemoryState,
)


def _snapshot() -> ViewportSnapshot:
    return ViewportSnapshot(
        id="viewport-1",
        provider_id="fixture",
        elements=(
            ElementSnapshot(
                id="target",
                role="button",
                label="Invite teammate",
                bounds=BoundingBox(x=10, y=10, width=100, height=30),
                visibility_fraction=1.0,
                actionable=True,
                provider_id="fixture",
                execution_reference=PrivateExecutionReference(
                    provider_id="fixture", viewport_id="viewport-1", token="target"
                ),
            ),
            ElementSnapshot(
                id="email",
                role="input",
                label="Email",
                bounds=BoundingBox(x=10, y=50, width=180, height=30),
                visibility_fraction=1.0,
                actionable=True,
                provider_id="fixture",
                execution_reference=PrivateExecutionReference(
                    provider_id="fixture", viewport_id="viewport-1", token="email"
                ),
            ),
        ),
    )


def _attention() -> AttentionState:
    return AttentionState.initial(
        Budget(
            max_steps=10, max_observations=5, max_interactions=3, timeout_seconds=10
        ),
        confidence=0.5,
        frustration=0.2,
        memory_capacity=2,
        current_subgoal="Invite teammate",
    )


def _observed_state() -> ApplicationState:
    snapshot = _snapshot()
    observation = ProgressiveObservation.from_snapshot(
        snapshot, newly_revealed_ids=("target", "email")
    )
    return apply_observation(
        ApplicationState.from_attention(_attention()), observation, snapshot=snapshot
    )


def test_memory_policy_enforces_capacities_decay_and_failure_retention() -> None:
    policy = MemoryPolicy(
        MemoryPolicyConfig(
            working_capacity=2,
            episodic_capacity=2,
            decay_rate=0.25,
            failure_retention_bonus=10.0,
        )
    )
    memory = policy.update(
        MemoryState(episodic=(MemoryEntry(key="old", value="Old", strength=1.0),)),
        working=(
            MemoryEntry(key="one", value="One", strength=1.0),
            MemoryEntry(key="two", value="Two", strength=1.0),
        ),
        elapsed_steps=0,
    )
    memory = policy.update(
        memory,
        working=(MemoryEntry(key="three", value="Three", strength=1.0),),
        failures=(MemoryEntry.failure("failed", "Failed action"),),
        elapsed_steps=1,
    )

    assert tuple(item.key for item in memory.working) == ("two", "three")
    assert len(memory.working) == 2
    assert len(memory.episodic) == 2
    assert memory.episodic[0].key == "failed"
    assert memory.working[-1].strength == 1.0
    assert memory.working[0].strength == 0.75


def test_apply_observation_consumes_budget_and_updates_working_memory() -> None:
    state = _observed_state()

    assert state.attention.budgets.steps == 9
    assert state.attention.budgets.observations == 4
    assert state.attention.current_viewport_id == "viewport-1"
    assert state.attention.remembered_ids == frozenset({"target", "email"})
    assert len(state.memory.working) == 2


def test_failed_interaction_consumes_budgets_marks_candidate_and_clamps_state() -> None:
    state = _observed_state()
    updated = apply_interaction_result(
        state,
        InteractWithElement(element_id="target"),
        PlatformActionResult(
            succeeded=False,
            url="https://fixture.invalid/app",
            duration_ms=5,
            error="wrong target",
        ),
        config=StateUpdateConfig(
            failure_confidence_delta=-2.0,
            failure_frustration_delta=2.0,
            abandonment_threshold=1.0,
        ),
    )

    assert updated.attention.budgets.steps == 8
    assert updated.attention.budgets.interactions == 2
    assert "target" in updated.attention.failed_candidates
    assert updated.attention.confidence == 0.0
    assert updated.attention.frustration == 1.0
    assert updated.memory.episodic[0].key == "target"


def test_inspection_and_scroll_consume_steps_without_interaction_budget() -> None:
    state = _observed_state()

    inspected = apply_interaction_result(
        state,
        InspectElement(element_id="target"),
        True,
    )
    scrolled = apply_interaction_result(inspected, Scroll(direction="down"), True)

    assert inspected.attention.budgets.steps == 8
    assert inspected.attention.budgets.interactions == 3
    assert "target" in inspected.attention.inspected_ids
    assert scrolled.attention.budgets.steps == 7
    assert scrolled.attention.budgets.interactions == 3


def test_failure_crossing_threshold_marks_application_state_abandoned() -> None:
    state = _observed_state()

    updated = apply_failure(
        state,
        element_id="target",
        config=StateUpdateConfig(
            failure_frustration_delta=0.9,
            abandonment_threshold=0.8,
        ),
    )

    assert updated.abandoned
    assert updated.abandonment_reason == "abandonment-threshold-crossed"


def test_bare_attention_state_transition_remains_supported() -> None:
    snapshot = _snapshot()
    observation = ProgressiveObservation.from_snapshot(
        snapshot, newly_revealed_ids=("target",)
    )

    updated = apply_observation(_attention(), observation, snapshot=snapshot)

    assert isinstance(updated, AttentionState)
    assert updated.current_viewport_id == snapshot.id


def test_failure_clamps_confidence_and_frustration_at_bounds() -> None:
    state = ApplicationState.from_attention(
        replace(_attention(), confidence=0.0, frustration=1.0)
    )

    updated = apply_failure(
        state,
        element_id="missing",
        config=StateUpdateConfig(
            failure_confidence_delta=-1.0,
            failure_frustration_delta=1.0,
            abandonment_threshold=1.0,
        ),
    )

    assert updated.attention.confidence == 0.0
    assert updated.attention.frustration == 1.0


def test_recapture_reconciles_attention_and_memory_through_safe_lineage() -> None:
    previous = ViewportSnapshot(
        id="viewport-old",
        provider_id="fixture",
        elements=(
            ElementSnapshot(
                id="target-old",
                role="button",
                label="Invite teammate",
                bounds=BoundingBox(x=10, y=10, width=100, height=30),
                visibility_fraction=1,
                actionable=True,
                provider_id="fixture",
                lineage_id="lineage-target",
            ),
        ),
    )
    current = ViewportSnapshot(
        id="viewport-new",
        provider_id="fixture",
        elements=(
            ElementSnapshot(
                id="target-new",
                role="button",
                label="Invite teammate",
                bounds=BoundingBox(x=10, y=10, width=100, height=30),
                visibility_fraction=1,
                actionable=True,
                provider_id="fixture",
                lineage_id="lineage-target",
            ),
        ),
    )
    observed = apply_observation(
        ApplicationState.from_attention(_attention()),
        ProgressiveObservation.from_snapshot(
            previous, newly_revealed_ids=("target-old",)
        ),
        snapshot=previous,
    )
    inspected = apply_interaction_result(
        observed,
        InspectElement(element_id="target-old"),
        True,
        snapshot=previous,
    )
    failed = apply_interaction_result(
        inspected,
        InteractWithElement(element_id="target-old"),
        PlatformActionResult(
            succeeded=False,
            url="https://fixture.invalid/app",
            duration_ms=1,
            error="wrong target",
        ),
        snapshot=previous,
    )

    reconciled = reconcile_snapshot_state(failed, previous, current)

    assert reconciled.attention.current_viewport_id == "viewport-new"
    assert reconciled.attention.noticed_ids == frozenset({"target-new"})
    assert reconciled.attention.inspected_ids == frozenset({"target-new"})
    assert reconciled.attention.failed_candidates == frozenset({"target-new"})
    assert reconciled.attention.remembered_ids == frozenset({"target-new"})
    assert reconciled.attention.current_observation_ids == frozenset({"target-new"})
    assert reconciled.memory.working[0].key == "target-new"
    assert reconciled.memory.working[0].viewport_id == "viewport-new"
    assert reconciled.memory.episodic[0].key == "target-new"
    assert reconciled.memory.episodic[0].viewport_id == "viewport-new"
