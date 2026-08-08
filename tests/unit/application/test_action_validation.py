from __future__ import annotations

from dataclasses import replace

import pytest

from ux_analyzer.application.action_validation import (
    ActionValidationError,
    validate_action,
)
from ux_analyzer.application.state_updates import ApplicationState, apply_observation
from ux_analyzer.domain.attention import AttentionState, ProgressiveObservation
from ux_analyzer.domain.benchmark import Budget, FixtureInputs
from ux_analyzer.domain.interface import (
    BoundingBox,
    ElementSnapshot,
    ViewportSnapshot,
)
from ux_analyzer.ports.observation import ScrollAction as PlatformScrollAction
from ux_analyzer.ports.observation import TypeTextAction
from ux_analyzer.providers.cognitive import CognitiveDecision


def _snapshot(
    *, viewport_id: str = "viewport-1", actionable: bool = True
) -> ViewportSnapshot:
    return ViewportSnapshot(
        id=viewport_id,
        elements=(
            ElementSnapshot(
                id="target",
                role="button",
                label="Invite teammate",
                bounds=BoundingBox(x=10, y=10, width=100, height=30),
                visibility_fraction=1.0,
                actionable=actionable,
            ),
            ElementSnapshot(
                id="email",
                role="input",
                label="Email",
                bounds=BoundingBox(x=10, y=50, width=180, height=30),
                visibility_fraction=1.0,
                actionable=True,
            ),
        ),
    )


def _state(snapshot: ViewportSnapshot) -> ApplicationState:
    attention = AttentionState.initial(
        Budget(max_steps=4, max_observations=2, max_interactions=1, timeout_seconds=10),
        confidence=0.5,
        frustration=0.0,
        memory_capacity=3,
    )
    observation = ProgressiveObservation.from_snapshot(
        snapshot, newly_revealed_ids=("target", "email")
    )
    return apply_observation(ApplicationState.from_attention(attention), observation)


def test_validate_cognitive_interaction_returns_typed_platform_action() -> None:
    snapshot = _snapshot()
    decision = CognitiveDecision(
        action={"kind": "interact", "element_id": "target"},
        reason="Target matches goal.",
    )

    validated = validate_action(decision, _state(snapshot).attention, snapshot)

    assert validated.domain_action.kind == "interact-with-element"
    assert validated.platform_action.kind == "click"
    assert validated.platform_action.element_id == "target"


def test_validate_type_action_resolves_fixture_key_and_rejects_invented_value() -> None:
    snapshot = _snapshot()
    fixtures = FixtureInputs(
        values={"invite_email": "person@example.com"},
        sensitive_keys=frozenset({"invite_email"}),
    )
    state = _state(snapshot).attention

    validated = validate_action(
        TypeTextAction(element_id="email", text="invite_email"),
        state,
        snapshot,
        fixture_inputs=fixtures,
    )

    assert validated.platform_action.text == "person@example.com"
    assert validated.fixture_key == "invite_email"

    with pytest.raises(ActionValidationError, match="fixture"):
        validate_action(
            TypeTextAction(element_id="email", text="invented@example.com"),
            state,
            snapshot,
            fixture_inputs=fixtures,
        )


def test_validate_cognitive_fixture_type_action_resolves_only_configured_value() -> (
    None
):
    snapshot = _snapshot()
    fixtures = FixtureInputs(
        values={"invite_email": "person@example.com"},
        sensitive_keys=frozenset({"invite_email"}),
    )
    state = _state(snapshot).attention

    validated = validate_action(
        CognitiveDecision.model_validate(
            {
                "action": {
                    "kind": "type-fixture",
                    "element_id": "email",
                    "fixture_key": "invite_email",
                },
                "reason": "Fill visible email input from scenario fixture.",
            }
        ),
        state,
        snapshot,
        fixture_inputs=fixtures,
    )

    assert validated.platform_action.kind == "type-text"
    assert validated.platform_action.text == "person@example.com"
    assert validated.fixture_key == "invite_email"


def test_validate_rejects_unremembered_unactionable_and_stale_targets() -> None:
    snapshot = _snapshot(actionable=False)
    state = _state(snapshot).attention
    decision = CognitiveDecision(
        action={"kind": "interact", "element_id": "target"},
        reason="Try target.",
    )

    with pytest.raises(ActionValidationError, match="actionable"):
        validate_action(decision, state, snapshot)

    with pytest.raises(ActionValidationError, match="remembered"):
        unremembered_snapshot = ViewportSnapshot(
            id=snapshot.id,
            elements=(
                *snapshot.elements,
                ElementSnapshot(
                    id="unremembered",
                    role="button",
                    label="Unremembered",
                    bounds=BoundingBox(x=10, y=90, width=100, height=30),
                    visibility_fraction=1.0,
                    actionable=True,
                ),
            ),
        )
        validate_action(
            CognitiveDecision(
                action={"kind": "interact", "element_id": "unremembered"},
                reason="Try unknown.",
            ),
            replace(
                state,
                noticed_ids=state.noticed_ids | frozenset({"unremembered"}),
            ),
            unremembered_snapshot,
        )

    stale = _snapshot(viewport_id="viewport-2", actionable=True)
    with pytest.raises(ActionValidationError, match="stale"):
        validate_action(decision, state, stale)


def test_validate_consumable_actions_checks_step_and_interaction_budgets() -> None:
    snapshot = _snapshot()
    state = _state(snapshot).attention

    no_steps = replace(state, budgets=replace(state.budgets, steps=0))
    with pytest.raises(ActionValidationError, match="step budget"):
        validate_action(PlatformScrollAction(direction="down"), no_steps, snapshot)

    no_interactions = replace(state, budgets=replace(state.budgets, interactions=0))
    with pytest.raises(ActionValidationError, match="interaction budget"):
        validate_action(
            CognitiveDecision(
                action={"kind": "interact", "element_id": "target"},
                reason="Try target.",
            ),
            no_interactions,
            snapshot,
        )


def test_validate_abandon_decision_without_target() -> None:
    snapshot = _snapshot()
    decision = CognitiveDecision(
        action={"kind": "abandon", "reason": "No useful path."},
        reason="No useful path.",
    )

    validated = validate_action(decision, _state(snapshot).attention, snapshot)

    assert validated.domain_action.kind == "abandon"
    assert validated.platform_action is None


def test_validate_complete_decision_without_target() -> None:
    snapshot = _snapshot()

    validated = validate_action(
        CognitiveDecision(
            action={"kind": "complete"},
            reason="Visible result is sufficient.",
        ),
        _state(snapshot).attention,
        snapshot,
    )

    assert validated.domain_action.kind == "complete"
    assert validated.platform_action is None
