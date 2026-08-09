"""Immutable, versioned expectations for scenario-level UX analysis."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


def _require_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _tuple_of_strings(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a collection of strings")
    normalized = tuple(values)
    for value in normalized:
        _require_non_empty(value, field_name)
    return normalized


@dataclass(frozen=True, slots=True)
class ExpectationKey:
    """Stable identity for an expectation's application and scenario scope."""

    application_version_id: str
    scenario_id: str
    persona_id: str

    def __post_init__(self) -> None:
        _require_non_empty(self.application_version_id, "application_version_id")
        _require_non_empty(self.scenario_id, "scenario_id")
        _require_non_empty(self.persona_id, "persona_id")


@dataclass(frozen=True, slots=True)
class FrozenExpectation:
    """Immutable outcome contract that permits more than one valid route."""

    expectation_id: str
    schema_version: str
    key: ExpectationKey
    desired_outcomes: tuple[str, ...]
    required_invariants: tuple[str, ...] = ()
    acceptable_alternatives: tuple[str, ...] = ()
    reference_paths: tuple[tuple[str, ...], ...] = ()
    effort_bounds: Mapping[str, float] = field(default_factory=dict)
    warning_signals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.expectation_id, "expectation_id")
        _require_non_empty(self.schema_version, "schema_version")
        if not isinstance(self.key, ExpectationKey):
            raise TypeError("key must be an ExpectationKey")

        desired_outcomes = _tuple_of_strings(self.desired_outcomes, "desired_outcomes")
        if not desired_outcomes:
            raise ValueError("at least one desired outcome is required")

        reference_paths: list[tuple[str, ...]] = []
        for path in self.reference_paths:
            reference_paths.append(_tuple_of_strings(path, "reference_paths"))

        if not isinstance(self.effort_bounds, Mapping):
            raise TypeError("effort_bounds must be a mapping")
        effort_bounds = {
            _require_non_empty(str(name), "effort_bounds key"): float(value)
            for name, value in self.effort_bounds.items()
        }

        object.__setattr__(self, "desired_outcomes", desired_outcomes)
        object.__setattr__(
            self,
            "required_invariants",
            _tuple_of_strings(self.required_invariants, "required_invariants"),
        )
        object.__setattr__(
            self,
            "acceptable_alternatives",
            _tuple_of_strings(self.acceptable_alternatives, "acceptable_alternatives"),
        )
        object.__setattr__(self, "reference_paths", tuple(reference_paths))
        object.__setattr__(self, "effort_bounds", MappingProxyType(effort_bounds))
        object.__setattr__(
            self,
            "warning_signals",
            _tuple_of_strings(self.warning_signals, "warning_signals"),
        )


__all__ = ["ExpectationKey", "FrozenExpectation"]
