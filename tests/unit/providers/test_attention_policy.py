from __future__ import annotations

import random
from dataclasses import replace

import pytest

from ux_analyzer.application.state_updates import apply_observation
from ux_analyzer.domain.attention import (
    AttentionRecoveryMiss,
    AttentionState,
    CoarseScent,
    InteractWithElement,
    ProgressiveObservation,
)
from ux_analyzer.domain.benchmark import Budget
from ux_analyzer.domain.interface import (
    BoundingBox,
    ElementSnapshot,
    RegionSnapshot,
    ViewportSnapshot,
)
from ux_analyzer.providers.attention_policy import (
    AttentionPolicyConfig,
    ProgressiveAttentionPolicy,
)
from ux_analyzer.providers.prominence import ProminenceResult


def _element(
    element_id: str,
    *,
    region_id: str | None = None,
    x: float = 10,
    lineage_id: str | None = None,
    actionable: bool = True,
    disabled: bool = False,
) -> ElementSnapshot:
    return ElementSnapshot(
        id=element_id,
        role="button",
        label=element_id.title(),
        bounds=BoundingBox(x=x, y=10, width=80, height=30),
        visibility_fraction=1.0,
        actionable=actionable,
        disabled=disabled,
        region_id=region_id,
        provider_id="fixture",
        lineage_id=lineage_id,
    )


def _state() -> AttentionState:
    return AttentionState.initial(
        Budget(
            max_steps=10, max_observations=10, max_interactions=5, timeout_seconds=10
        ),
        confidence=0.5,
        frustration=0.0,
    )


def _scores(*element_ids: str) -> tuple[ProminenceResult, ...]:
    probability = 1 / len(element_ids)
    return tuple(
        ProminenceResult(
            element_id=element_id,
            raw_score=0.0,
            normalized_probability=probability,
        )
        for element_id in element_ids
    )


def test_region_first_selection_reveals_bounded_batch_with_region_context() -> None:
    snapshot = ViewportSnapshot(
        id="viewport-1",
        elements=(
            _element("profile", region_id="profile"),
            _element("security", region_id="security", x=200),
            _element("recovery", region_id="security", x=300),
        ),
        regions=(
            RegionSnapshot(id="profile", label="Profile", element_ids=("profile",)),
            RegionSnapshot(
                id="security", label="Security", element_ids=("security", "recovery")
            ),
        ),
    )
    policy = ProgressiveAttentionPolicy(
        AttentionPolicyConfig(
            batch_size=2,
            cross_region_exploration=0,
            region_priors={"security": 100.0},
        )
    )

    selection = policy.next_observation(
        _state(),
        snapshot,
        _scores("profile", "security", "recovery"),
        (),
        random.Random(4),
    )

    assert selection.region_id == "security"
    assert selection.selected_ids == ("security", "recovery")
    assert selection.observation.region_context is not None
    assert selection.observation.region_context.id == "security"
    assert len(selection.observation.newly_revealed_elements) == 2


def test_region_first_batch_reserves_cross_region_exploration_slot() -> None:
    snapshot = ViewportSnapshot(
        id="viewport-1",
        elements=(
            _element("overview", region_id="navigation"),
            _element("security", region_id="navigation", x=100),
            _element("email", region_id="invite-form", x=200),
            _element("send", region_id="invite-form", x=300),
        ),
        regions=(
            RegionSnapshot(
                id="navigation",
                label="Primary navigation",
                element_ids=("overview", "security"),
            ),
            RegionSnapshot(
                id="invite-form",
                label="Invite teammate",
                element_ids=("email", "send"),
            ),
        ),
    )
    scores = (
        ProminenceResult("overview", raw_score=0.0, normalized_probability=0.05),
        ProminenceResult("security", raw_score=0.0, normalized_probability=0.05),
        ProminenceResult("email", raw_score=0.0, normalized_probability=0.45),
        ProminenceResult("send", raw_score=0.0, normalized_probability=0.45),
    )
    policy = ProgressiveAttentionPolicy(
        AttentionPolicyConfig(
            batch_size=2,
            cross_region_exploration=1,
            region_priors={"navigation": 100.0},
        )
    )

    selection = policy.next_observation(
        _state(), snapshot, scores, (), random.Random(4)
    )

    selected_regions = {
        snapshot.element(element_id).region_id for element_id in selection.selected_ids
    }
    assert selected_regions == {"navigation", "invite-form"}
    assert selection.selection_mode == "region-first+cross-region"
    assert selection.region_id is None
    assert selection.observation.region_context is None


def test_default_progressive_batch_reveals_two_elements() -> None:
    assert AttentionPolicyConfig().batch_size == 2
    assert AttentionPolicyConfig().cross_region_exploration == 1

    snapshot = ViewportSnapshot(
        id="viewport-1",
        elements=(
            _element("one"),
            _element("two", x=200),
            _element("three", x=300),
        ),
    )

    selection = ProgressiveAttentionPolicy().next_observation(
        _state(),
        snapshot,
        _scores("one", "two", "three"),
        (),
        random.Random(1),
    )

    assert len(selection.selected_ids) == 2


def test_forgotten_element_can_be_revealed_again() -> None:
    snapshot = ViewportSnapshot(
        id="viewport-1",
        elements=(
            _element("send"),
            _element("email", x=200),
        ),
    )
    state = AttentionState.initial(
        Budget(
            max_steps=10,
            max_observations=10,
            max_interactions=5,
            timeout_seconds=10,
        ),
        confidence=0.5,
        frustration=0.0,
        memory_capacity=1,
    )
    state = state.after_observation(
        ProgressiveObservation.from_snapshot(
            snapshot, newly_revealed_ids=("send",)
        )
    )
    state = state.after_observation(
        ProgressiveObservation.from_snapshot(
            snapshot, newly_revealed_ids=("email",)
        )
    )
    assert state.remembered_ids == frozenset({"email"})
    assert "send" in state.noticed_ids

    selection = ProgressiveAttentionPolicy(
        AttentionPolicyConfig(batch_size=1)
    ).next_observation(
        state,
        snapshot,
        _scores("send", "email"),
        (),
        random.Random(1),
    )

    assert selection.selected_ids == ("send",)


def test_high_scent_actionable_is_forced_after_two_consecutive_misses() -> None:
    snapshot = ViewportSnapshot(
        id="viewport-1",
        provider_id="fixture",
        elements=tuple(
            _element(str(index), x=index * 100, lineage_id=str(index))
            for index in range(9)
        )
        + (_element("submit", x=900, lineage_id="submit"),),
    )
    scent = CoarseScent.from_element(snapshot, snapshot.element("submit"), 1.0)
    policy = ProgressiveAttentionPolicy(
        AttentionPolicyConfig(batch_size=2, coarse_scent_weight=0.0)
    )
    rng = random.Random(4)
    scores = _scores(*(element.id for element in snapshot.elements))

    first = policy.next_observation(_state(), snapshot, scores, (scent,), rng)
    assert first.recovery_selected_ids == ()
    assert first.next_recovery_state == (
        AttentionRecoveryMiss(
            provider_id="fixture", lineage_id="submit", consecutive_misses=1
        ),
    )
    next_state = apply_observation(
        _state(),
        first.observation,
        snapshot=snapshot,
        recovery_state=first.next_recovery_state,
    )
    second = policy.next_observation(next_state, snapshot, scores, (scent,), rng)
    assert second.recovery_selected_ids == ()
    assert second.next_recovery_state == (
        AttentionRecoveryMiss(
            provider_id="fixture", lineage_id="submit", consecutive_misses=2
        ),
    )
    next_state = apply_observation(
        next_state,
        second.observation,
        snapshot=snapshot,
        recovery_state=second.next_recovery_state,
    )
    third = policy.next_observation(next_state, snapshot, scores, (scent,), rng)

    assert third.selection_mode.endswith("+recovery")
    assert third.recovery_selected_ids == ("submit",)
    assert third.selected_ids[0] == "submit"
    assert len(third.selected_ids) == 2
    assert third.next_recovery_state == ()


def test_recovery_consumes_normal_rng_draw_before_forcing_target() -> None:
    snapshot = ViewportSnapshot(
        id="viewport-1",
        provider_id="fixture",
        elements=tuple(
            _element(str(index), x=index * 100, lineage_id=str(index))
            for index in range(10)
        ),
    )
    scent = CoarseScent.from_element(snapshot, snapshot.element("9"), 1.0)
    scores = _scores(*(element.id for element in snapshot.elements))
    policy = ProgressiveAttentionPolicy(
        AttentionPolicyConfig(batch_size=2, coarse_scent_weight=0.0)
    )
    rng = random.Random(4)
    first = policy.next_observation(_state(), snapshot, scores, (scent,), rng)
    state = apply_observation(
        _state(),
        first.observation,
        snapshot=snapshot,
        recovery_state=first.next_recovery_state,
    )
    second = policy.next_observation(state, snapshot, scores, (scent,), rng)
    state = apply_observation(
        state,
        second.observation,
        snapshot=snapshot,
        recovery_state=second.next_recovery_state,
    )
    third = policy.next_observation(state, snapshot, scores, (scent,), rng)
    assert third.recovery_selected_ids == ("9",)

    expected_rng = random.Random(4)
    first_control = policy.next_observation(_state(), snapshot, scores, (), expected_rng)
    control_state = apply_observation(
        _state(), first_control.observation, snapshot=snapshot
    )
    second_control = policy.next_observation(
        control_state, snapshot, scores, (), expected_rng
    )
    control_state = apply_observation(
        control_state, second_control.observation, snapshot=snapshot
    )
    policy.next_observation(control_state, snapshot, scores, (), expected_rng)

    assert rng.getstate() == expected_rng.getstate()


@pytest.mark.parametrize(
    ("actionable", "disabled", "score"),
    [(False, False, 1.0), (True, True, 1.0), (True, False, 0.89)],
)
def test_only_enabled_actionable_strong_scent_accrues_recovery_misses(
    actionable: bool, disabled: bool, score: float
) -> None:
    target = _element(
        "target",
        x=500,
        lineage_id="target",
        actionable=actionable,
        disabled=disabled,
    )
    snapshot = ViewportSnapshot(
        id="viewport-1",
        provider_id="fixture",
        elements=tuple(_element(str(index), x=index * 100) for index in range(5))
        + (target,),
    )
    scent = CoarseScent.from_element(snapshot, target, score)
    selection = ProgressiveAttentionPolicy(
        AttentionPolicyConfig(coarse_scent_weight=0.0)
    ).next_observation(
        _state(),
        snapshot,
        _scores(*(element.id for element in snapshot.elements)),
        (scent,),
        random.Random(4),
    )
    assert selection.next_recovery_state == ()


def test_recovery_does_not_track_non_unique_or_missing_lineage() -> None:
    snapshot = ViewportSnapshot(
        id="viewport-1",
        provider_id="fixture",
        elements=(
            _element("one", lineage_id="duplicate"),
            _element("two", x=200, lineage_id="duplicate"),
            _element("target", x=400),
        ),
    )
    scent = tuple(
        CoarseScent.from_element(snapshot, element, 1.0)
        for element in snapshot.elements
    )
    selection = ProgressiveAttentionPolicy(
        AttentionPolicyConfig(batch_size=1, coarse_scent_weight=0.0)
    ).next_observation(
        _state(), snapshot, _scores("one", "two", "target"), scent, random.Random(1)
    )
    assert selection.next_recovery_state == ()


def test_mixed_region_recovery_clears_region_context() -> None:
    snapshot = ViewportSnapshot(
        id="viewport-1",
        provider_id="fixture",
        elements=(
            _element("normal-one", region_id="normal", lineage_id="normal-one"),
            _element(
                "normal-two", region_id="normal", x=200, lineage_id="normal-two"
            ),
            _element("target", region_id="target", x=400, lineage_id="target"),
        ),
        regions=(
            RegionSnapshot(
                id="normal",
                label="Normal",
                element_ids=("normal-one", "normal-two"),
            ),
            RegionSnapshot(id="target", label="Target", element_ids=("target",)),
        ),
    )
    state = replace(
        _state(),
        recovery_misses=(AttentionRecoveryMiss("fixture", "target", 2),),
    )
    scent = CoarseScent.from_element(snapshot, snapshot.element("target"), 1.0)
    policy = ProgressiveAttentionPolicy(
        AttentionPolicyConfig(
            coarse_scent_weight=0.0,
            cross_region_exploration=0,
            region_priors={"normal": 100.0},
        )
    )

    selection = policy.next_observation(
        state,
        snapshot,
        _scores("normal-one", "normal-two", "target"),
        (scent,),
        random.Random(4),
    )

    assert selection.recovery_selected_ids == ("target",)
    assert selection.region_id is None
    assert selection.observation.region_context is None


def test_coarse_scent_and_failure_penalty_change_candidate_probabilities() -> None:
    snapshot = ViewportSnapshot(
        id="viewport-1",
        elements=(_element("target"), _element("failed", x=200)),
    )
    scent = CoarseScent.from_element(snapshot, snapshot.element("target"), 1.0)
    state = _state().mark_failed_candidate("failed")
    policy = ProgressiveAttentionPolicy(
        AttentionPolicyConfig(coarse_scent_weight=2.0, failure_penalty=0.8)
    )

    selection = policy.next_observation(
        state,
        snapshot,
        _scores("target", "failed"),
        (scent,),
        random.Random(3),
    )

    assert (
        selection.element_probabilities["target"]
        > selection.element_probabilities["failed"]
    )


def test_sparse_snapshot_falls_back_to_element_sampling() -> None:
    snapshot = ViewportSnapshot(
        id="viewport-1",
        elements=(_element("one"), _element("two", x=200)),
    )

    selection = ProgressiveAttentionPolicy().next_observation(
        _state(), snapshot, _scores("one", "two"), (), random.Random(1)
    )

    assert selection.region_id is None
    assert len(selection.selected_ids) == 2
    assert selection.observation.region_context is None


def test_blank_rendered_text_is_not_selected_as_persona_visible() -> None:
    blank = replace(_element("blank"), rendered_text="   ")
    snapshot = ViewportSnapshot(id="viewport-1", elements=(blank,))

    with pytest.raises(ValueError, match="no unobserved visible elements remain"):
        ProgressiveAttentionPolicy(
            AttentionPolicyConfig(batch_size=1)
        ).next_observation(
            _state(), snapshot, _scores("blank"), (), random.Random(1)
        )


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_batch_size_is_limited_to_one_through_three(batch_size: int) -> None:
    elements = tuple(_element(f"item-{index}", x=index * 100) for index in range(4))
    snapshot = ViewportSnapshot(id="viewport-1", elements=elements)

    selection = ProgressiveAttentionPolicy(
        AttentionPolicyConfig(batch_size=batch_size)
    ).next_observation(
        _state(),
        snapshot,
        _scores(*(element.id for element in elements)),
        (),
        random.Random(8),
    )

    assert len(selection.selected_ids) == batch_size


def test_same_seed_repeats_and_different_seed_can_change_path() -> None:
    snapshot = ViewportSnapshot(
        id="viewport-1",
        elements=(_element("one"), _element("two", x=200)),
    )
    policy = ProgressiveAttentionPolicy()
    scores = _scores("one", "two")

    first = policy.next_observation(_state(), snapshot, scores, (), random.Random(1))
    repeat = policy.next_observation(_state(), snapshot, scores, (), random.Random(1))
    different = policy.next_observation(
        _state(), snapshot, scores, (), random.Random(2)
    )

    assert first.selected_ids == repeat.selected_ids
    assert first.selected_ids != different.selected_ids


def test_observation_memory_budget_keeps_all_visible_elements_action_valid() -> None:
    snapshot = ViewportSnapshot(
        id="viewport-1",
        elements=tuple(
            _element(element_id, x=index * 100)
            for index, element_id in enumerate(
                ("old-one", "old-two", "old-three", "new-one", "new-two")
            )
        ),
    )
    state = replace(_state(), memory_capacity=3)
    for element_id in ("old-one", "old-two", "old-three"):
        state = apply_observation(
            state,
            ProgressiveObservation.from_snapshot(
                snapshot, newly_revealed_ids=(element_id,)
            ),
            snapshot=snapshot,
        )

    selection = ProgressiveAttentionPolicy(
        AttentionPolicyConfig(batch_size=2)
    ).next_observation(
        state,
        snapshot,
        _scores(*(element.id for element in snapshot.elements)),
        (),
        random.Random(1),
    )

    assert tuple(
        element.id for element in selection.observation.remembered_elements
    ) == ("old-three",)
    observed = apply_observation(
        state, selection.observation, snapshot=snapshot
    )
    for element_id in ("old-three", *selection.selected_ids):
        observed.validate_action(
            InteractWithElement(element_id=element_id), snapshot
        )


def test_recovery_covers_never_noticed_visible_elements_before_forgotten_ones() -> None:
    snapshot = ViewportSnapshot(
        id="viewport-1",
        elements=(
            _element("forgotten"),
            _element("fresh-one", x=200),
            _element("fresh-two", x=300),
        ),
    )
    state = replace(
        _state(),
        noticed_ids=frozenset({"forgotten"}),
        memory=(),
    )
    scores = (
        ProminenceResult("forgotten", 0.0, 0.99),
        ProminenceResult("fresh-one", 0.0, 0.1),
        ProminenceResult("fresh-two", 0.0, 0.1),
    )

    selection = ProgressiveAttentionPolicy(
        AttentionPolicyConfig(batch_size=2)
    ).next_observation(
        state, snapshot, scores, (), random.Random(1), recovery_level=1
    )

    assert set(selection.selected_ids) == {"fresh-one", "fresh-two"}
    assert "forgotten" not in selection.selected_ids


def test_observation_memory_exposes_no_remembered_elements_when_batch_fills_capacity() -> None:
    snapshot = ViewportSnapshot(
        id="viewport-1",
        elements=tuple(
            _element(element_id, x=index * 100)
            for index, element_id in enumerate(("old", "new-one", "new-two"))
        ),
    )
    state = replace(_state(), memory_capacity=1)
    state = apply_observation(
        state,
        ProgressiveObservation.from_snapshot(snapshot, newly_revealed_ids=("old",)),
        snapshot=snapshot,
    )

    selection = ProgressiveAttentionPolicy(
        AttentionPolicyConfig(batch_size=2)
    ).next_observation(
        state,
        snapshot,
        _scores(*(element.id for element in snapshot.elements)),
        (),
        random.Random(1),
    )

    assert selection.observation.remembered_elements == ()


def test_recovery_samples_fresh_candidates_by_prominence_without_scent() -> None:
    snapshot = ViewportSnapshot(
        id="viewport-1",
        elements=(
            _element("forgotten"),
            _element("fresh-high", x=200),
            _element("fresh-low", x=300),
        ),
    )
    state = replace(
        _state(),
        noticed_ids=frozenset({"forgotten"}),
        memory=(),
    )
    scores = (
        ProminenceResult("forgotten", 0.0, 0.99),
        ProminenceResult("fresh-high", 0.0, 0.9),
        ProminenceResult("fresh-low", 0.0, 0.1),
    )

    selection = ProgressiveAttentionPolicy(
        AttentionPolicyConfig(batch_size=1)
    ).next_observation(
        state, snapshot, scores, (), random.Random(1), recovery_level=1
    )

    assert selection.selected_ids == ("fresh-high",)
