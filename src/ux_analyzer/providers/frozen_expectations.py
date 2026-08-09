"""Resolve immutable, version-scoped frozen expectations."""

from __future__ import annotations

from collections.abc import Iterable
from types import MappingProxyType

from ux_analyzer.domain.expectations import ExpectationKey, FrozenExpectation


class FrozenExpectationProvider:
    """Resolve exact expectation keys with a same-scope persona fallback."""

    provider_id = "frozen-expectation-v1"
    provider_version = "1"

    def __init__(self, expectations: Iterable[FrozenExpectation] = ()) -> None:
        documents: dict[ExpectationKey, FrozenExpectation] = {}
        for expectation in expectations:
            if not isinstance(expectation, FrozenExpectation):
                raise TypeError("expectations must contain FrozenExpectation values")
            if expectation.key in documents:
                raise ValueError(f"duplicate frozen expectation key: {expectation.key}")
            documents[expectation.key] = expectation
        self._documents = MappingProxyType(documents)

    def resolve(self, key: ExpectationKey) -> FrozenExpectation | None:
        """Return the exact expectation or its same-version wildcard fallback."""

        exact = self._documents.get(key)
        if exact is not None:
            return exact
        return self._documents.get(
            ExpectationKey(
                application_version_id=key.application_version_id,
                scenario_id=key.scenario_id,
                persona_id="*",
            )
        )
