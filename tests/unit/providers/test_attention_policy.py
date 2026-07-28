from __future__ import annotations

import random

import pytest

from ux_analyzer.domain.attention import AttentionState, CoarseScent
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
) -> ElementSnapshot:
    return ElementSnapshot(
        id=element_id,
        role="button",
        label=element_id.title(),
        bounds=BoundingBox(x=x, y=10, width=80, height=30),
        visibility_fraction=1.0,
        actionable=True,
        region_id=region_id,
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
        AttentionPolicyConfig(batch_size=2, region_priors={"security": 100.0})
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
    assert len(selection.selected_ids) == 1
    assert selection.observation.region_context is None


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
