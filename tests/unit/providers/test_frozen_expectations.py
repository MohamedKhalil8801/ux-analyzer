from __future__ import annotations

import pytest

from ux_analyzer.domain.expectations import ExpectationKey, FrozenExpectation
from ux_analyzer.providers.frozen_expectations import FrozenExpectationProvider


def _expectation(
    expectation_id: str,
    *,
    application_version_id: str = "version-a",
    scenario_id: str = "scenario-a",
    persona_id: str = "persona-a",
) -> FrozenExpectation:
    return FrozenExpectation(
        expectation_id=expectation_id,
        schema_version="frozen-expectation-v1",
        key=ExpectationKey(
            application_version_id=application_version_id,
            scenario_id=scenario_id,
            persona_id=persona_id,
        ),
        desired_outcomes=("The intended outcome is achieved.",),
    )


def test_resolve_prefers_exact_persona_match_over_wildcard() -> None:
    wildcard = _expectation("wildcard", persona_id="*")
    exact = _expectation("exact", persona_id="persona-a")
    provider = FrozenExpectationProvider((wildcard, exact))

    assert provider.resolve(exact.key) is exact
    assert (
        provider.resolve(ExpectationKey("version-a", "scenario-a", "persona-b"))
        is wildcard
    )


def test_resolve_does_not_fall_back_across_version_or_scenario() -> None:
    provider = FrozenExpectationProvider((_expectation("wildcard", persona_id="*"),))

    assert (
        provider.resolve(ExpectationKey("version-b", "scenario-a", "persona-b")) is None
    )
    assert (
        provider.resolve(ExpectationKey("version-a", "scenario-b", "persona-b")) is None
    )


def test_provider_rejects_duplicate_keys() -> None:
    first = _expectation("first")
    second = _expectation("second")

    with pytest.raises(ValueError, match="duplicate frozen expectation key"):
        FrozenExpectationProvider((first, second))
