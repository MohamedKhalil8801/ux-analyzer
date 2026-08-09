from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from ux_analyzer.domain.expectations import ExpectationKey, FrozenExpectation


def test_frozen_expectation_allows_multiple_paths_but_requires_outcome() -> None:
    expectation = FrozenExpectation(
        expectation_id="invite-first-time-v1",
        schema_version="frozen-expectation-v1",
        key=ExpectationKey("fixture-improved", "invite", "first-time"),
        desired_outcomes=("A teammate receives a valid invitation.",),
        required_invariants=("The user confirms the invite before completion.",),
        acceptable_alternatives=(
            "Invite from the team page.",
            "Invite from onboarding.",
        ),
        reference_paths=(("open-team", "invite", "confirm"),),
        effort_bounds={"max_wrong_actions": 1, "max_backtracks": 1},
        warning_signals=("Repeatedly opens unrelated settings.",),
    )

    assert len(expectation.acceptable_alternatives) == 2
    assert expectation.reference_paths == (("open-team", "invite", "confirm"),)


def test_frozen_expectation_normalizes_collections_and_freezes_mapping() -> None:
    effort_bounds = {"max_wrong_actions": 1}
    expectation = FrozenExpectation(
        expectation_id="expectation-1",
        schema_version="schema-1",
        key=ExpectationKey("app-1", "scenario-1", "persona-1"),
        desired_outcomes=["Outcome"],
        required_invariants=["Invariant"],
        acceptable_alternatives=["Alternative"],
        reference_paths=[["open", "finish"]],
        effort_bounds=effort_bounds,
        warning_signals=["Warning"],
    )

    assert expectation.desired_outcomes == ("Outcome",)
    assert expectation.required_invariants == ("Invariant",)
    assert expectation.acceptable_alternatives == ("Alternative",)
    assert expectation.reference_paths == (("open", "finish"),)
    assert expectation.warning_signals == ("Warning",)
    assert isinstance(expectation.effort_bounds, MappingProxyType)

    effort_bounds["max_wrong_actions"] = 99
    assert expectation.effort_bounds["max_wrong_actions"] == 1.0

    with pytest.raises(TypeError):
        expectation.effort_bounds["max_backtracks"] = 1  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        expectation.expectation_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expectation_id", ""),
        ("schema_version", ""),
    ],
)
def test_frozen_expectation_rejects_empty_identity_content(
    field: str, value: str
) -> None:
    values = {
        "expectation_id": "expectation-1",
        "schema_version": "schema-1",
        "key": ExpectationKey("app-1", "scenario-1", "persona-1"),
        "desired_outcomes": ("Outcome",),
    }
    values[field] = value

    with pytest.raises(ValueError):
        FrozenExpectation(**values)  # type: ignore[arg-type]


def test_frozen_expectation_requires_at_least_one_desired_outcome() -> None:
    with pytest.raises(ValueError, match="desired outcome"):
        FrozenExpectation(
            expectation_id="expectation-1",
            schema_version="schema-1",
            key=ExpectationKey("app-1", "scenario-1", "persona-1"),
            desired_outcomes=(),
        )


@pytest.mark.parametrize("empty_index", [0, 1, 2])
def test_expectation_key_rejects_each_empty_identity_part(empty_index: int) -> None:
    values = ["app-1", "scenario-1", "persona-1"]
    values[empty_index] = " "

    with pytest.raises(ValueError):
        ExpectationKey(*values)


def test_expectation_key_is_immutable() -> None:
    key = ExpectationKey("app-1", "scenario-1", "persona-1")

    with pytest.raises(FrozenInstanceError):
        key.persona_id = "changed"  # type: ignore[misc]
