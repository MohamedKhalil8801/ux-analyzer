"""Validate model and platform actions at application boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal, cast

from ux_analyzer.domain.attention import (
    Abandon,
    AttentionAction,
    AttentionState,
    Back,
    InspectElement,
    InteractWithElement,
    Scroll,
    Wait,
)
from ux_analyzer.domain.benchmark import FixtureInputs
from ux_analyzer.domain.interface import BoundingBox, ViewportSnapshot
from ux_analyzer.ports.observation import (
    BackAction,
    ClearTextAction,
    ClickAction,
    DoubleClickAction,
    DragAction,
    OpenMenuAction,
    PlatformAction,
    ScrollAction,
    SelectOptionAction,
    SubmitAction,
    ToggleAction,
    TypeTextAction,
    WaitAction,
)


class ActionValidationError(ValueError):
    """Raised when proposed action violates application safety contracts."""


@dataclass(frozen=True, slots=True)
class ValidatedAction:
    """Validated domain action plus optional executable platform action."""

    decision: object
    domain_action: AttentionAction
    platform_action: PlatformAction | None
    fixture_key: str | None = None

    @property
    def action(self) -> PlatformAction | None:
        return self.platform_action

    @property
    def attention_action(self) -> AttentionAction:
        return self.domain_action


def validate_action(
    decision: object,
    state: AttentionState,
    snapshot: ViewportSnapshot,
    *,
    fixture_inputs: FixtureInputs | Mapping[str, str] | None = None,
) -> ValidatedAction:
    """Validate target identity, budgets, actionability, and fixture data."""

    try:
        domain_action, platform_action, fixture_key = _translate(
            decision, snapshot, fixture_inputs
        )
        state.validate_action(domain_action, snapshot)
        _validate_budgets(state, domain_action)
    except (TypeError, ValueError) as error:
        raise ActionValidationError(str(error)) from error
    return ValidatedAction(
        decision=decision,
        domain_action=domain_action,
        platform_action=platform_action,
        fixture_key=fixture_key,
    )


def _translate(
    decision: object,
    snapshot: ViewportSnapshot,
    fixture_inputs: FixtureInputs | Mapping[str, str] | None,
) -> tuple[AttentionAction, PlatformAction | None, str | None]:
    action = getattr(decision, "action", decision)
    kind = _kind(action)
    element_id = _element_id(action)
    if kind is None:
        raise ValueError("action must provide typed kind")

    if kind in {"inspect", "inspect-element"}:
        if not isinstance(element_id, str):
            raise ValueError("inspect action needs element ID")
        return InspectElement(element_id=element_id), None, None
    if kind in {"interact", "interact-with-element"}:
        if not isinstance(element_id, str):
            raise ValueError("interaction action needs element ID")
        return (
            InteractWithElement(element_id=element_id),
            ClickAction(element_id=element_id, bounds=_bounds(snapshot, element_id)),
            None,
        )
    if kind == "type-fixture":
        if not isinstance(element_id, str):
            raise ValueError("fixture type action needs element ID")
        fixture_key = getattr(action, "fixture_key", None)
        if not isinstance(fixture_key, str):
            raise ValueError("fixture type action needs fixture key")
        values = _fixture_values(fixture_inputs)
        if fixture_key not in values:
            raise ValueError("typed value must resolve through scenario fixture")
        return (
            InteractWithElement(element_id=element_id),
            TypeTextAction(
                element_id=element_id,
                text=values[fixture_key],
                bounds=_bounds(snapshot, element_id),
            ),
            fixture_key,
        )
    if kind == "scroll":
        direction_value = getattr(action, "direction", "down")
        if direction_value not in {"up", "down"}:
            raise ValueError("scroll direction must be up or down")
        direction = cast(Literal["up", "down"], direction_value)
        domain = Scroll(direction=direction)
        return domain, ScrollAction(direction=direction, amount="medium"), None
    if kind == "wait":
        return Wait(), WaitAction(milliseconds=0), None
    if kind == "back":
        return Back(), BackAction(), None
    if kind == "abandon":
        reason = getattr(action, "reason", "")
        return Abandon(reason=reason), None, None

    if isinstance(action, TypeTextAction):
        if not isinstance(element_id, str):
            raise ValueError("type action needs element ID")
        values = _fixture_values(fixture_inputs)
        fixture_key = action.text
        if fixture_key not in values:
            raise ValueError("typed value must resolve through scenario fixture")
        resolved = replace(action, text=values[fixture_key])
        return (
            InteractWithElement(element_id=element_id),
            resolved,
            fixture_key,
        )

    if isinstance(
        action,
        (
            ClickAction,
            DoubleClickAction,
            ClearTextAction,
            SelectOptionAction,
            ToggleAction,
            SubmitAction,
            OpenMenuAction,
            DragAction,
        ),
    ):
        target_id = _element_id(action)
        if target_id is None:
            raise ValueError(f"unsupported action kind: {kind}")
        return InteractWithElement(element_id=target_id), action, None
    raise ValueError(f"unsupported action kind: {kind}")


def _validate_budgets(state: AttentionState, action: AttentionAction) -> None:
    if isinstance(action, Abandon):
        return
    if isinstance(action, Scroll) and state.budgets.steps <= 0:
        raise ValueError("step budget exhausted")
    if state.budgets.steps <= 0:
        raise ValueError("step budget exhausted")
    if isinstance(action, InteractWithElement) and state.budgets.interactions <= 0:
        raise ValueError("interaction budget exhausted")


def _fixture_values(
    fixture_inputs: FixtureInputs | Mapping[str, str] | None,
) -> Mapping[str, str]:
    if fixture_inputs is None:
        raise ValueError("typed action requires scenario fixture inputs")
    if isinstance(fixture_inputs, FixtureInputs):
        return fixture_inputs.values
    return fixture_inputs


def _bounds(snapshot: ViewportSnapshot, element_id: str) -> BoundingBox:
    return snapshot.element(element_id).bounds


def _element_id(action: object) -> str | None:
    value = getattr(action, "element_id", None)
    return value if isinstance(value, str) else None


def _kind(action: object) -> str | None:
    value = getattr(action, "kind", None)
    return value if isinstance(value, str) else None
