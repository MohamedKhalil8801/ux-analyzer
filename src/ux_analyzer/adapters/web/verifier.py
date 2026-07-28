"""Independent web verification adapters for controlled fixture runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol, cast
from urllib.parse import quote, urlsplit

import httpx

from ux_analyzer.domain.benchmark import (
    FixtureInputs,
    FixtureStateVerifierSpec,
    VerifierOperator,
    VerifierSpec,
    VisibleResultVerifierSpec,
)
from ux_analyzer.domain.interface import ElementRole, ViewportSnapshot
from ux_analyzer.domain.run import VerificationResult
from ux_analyzer.ports.observation import (
    ObservationCapture,
    ObservationProvider,
    SessionHandle,
)
from ux_analyzer.ports.verification import VerificationProvider


class WebVerificationError(RuntimeError):
    """Raised when web verifier cannot obtain trustworthy evidence."""


class FixtureStateClient(Protocol):
    """Port for reading private state from one controlled fixture origin."""

    async def get_state(self, session_id: str) -> Mapping[str, object]: ...


class HttpFixtureStateClient:
    """HTTP client for the fixture's private state control endpoint."""

    def __init__(
        self,
        origin: str,
        *,
        http_client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.origin = _validate_origin(origin)
        if timeout_seconds <= 0:
            raise ValueError("verification timeout must be greater than zero")
        self._http_client = http_client
        self._timeout_seconds = timeout_seconds

    async def get_state(self, session_id: str) -> Mapping[str, object]:
        if not session_id:
            raise ValueError("session ID must not be empty")
        client = self._http_client or httpx.AsyncClient(timeout=self._timeout_seconds)
        owns_client = self._http_client is None
        try:
            response = await client.get(
                f"{self.origin}/__control/state/{quote(session_id, safe='')}"
            )
            if response.status_code != 200:
                raise WebVerificationError("fixture state request failed")
            try:
                payload = response.json()
            except ValueError as error:
                raise WebVerificationError(
                    "fixture state response was not JSON"
                ) from error
            if not isinstance(payload, Mapping):
                raise WebVerificationError("fixture state response was not an object")
            return cast(Mapping[str, object], payload)
        except httpx.HTTPError as error:
            raise WebVerificationError("fixture state request failed") from error
        finally:
            if owns_client:
                await client.aclose()


SnapshotExtractor = Callable[[ObservationCapture], ViewportSnapshot]


class WebVerifier(VerificationProvider):
    """Dispatch typed scenario verification without trusting agent claims."""

    def __init__(
        self,
        spec: VerifierSpec,
        *,
        fixture_inputs: FixtureInputs | Mapping[str, str] | None = None,
        fixture_state_client: FixtureStateClient | None = None,
        fixture_control_origin: str | None = None,
        observation_provider: ObservationProvider | None = None,
        snapshot_extractor: SnapshotExtractor | None = None,
    ) -> None:
        self.spec = spec
        self._fixture_inputs = _fixture_values(fixture_inputs)
        self._fixture_state_client = fixture_state_client
        if fixture_control_origin is not None:
            if fixture_state_client is not None:
                raise ValueError(
                    "fixture state client and control origin are mutually exclusive"
                )
            self._fixture_state_client = HttpFixtureStateClient(fixture_control_origin)
        self._observation_provider = observation_provider
        self._snapshot_extractor = snapshot_extractor
        if isinstance(spec, FixtureStateVerifierSpec):
            if self._fixture_inputs is None:
                raise ValueError("fixture-state verification needs fixture inputs")
            if self._fixture_state_client is None:
                raise ValueError("fixture-state verification needs state client")
        else:
            if observation_provider is None or snapshot_extractor is None:
                raise ValueError(
                    "visible-result verification needs observation provider and extractor"
                )

    async def verify(self, session: SessionHandle) -> VerificationResult:
        if isinstance(self.spec, FixtureStateVerifierSpec):
            return await self._verify_fixture_state(session, self.spec)
        return await self._verify_visible_result(session, self.spec)

    async def _verify_fixture_state(
        self,
        session: SessionHandle,
        spec: FixtureStateVerifierSpec,
    ) -> VerificationResult:
        if self._fixture_state_client is None or self._fixture_inputs is None:
            raise WebVerificationError("fixture-state verifier is not configured")
        if spec.expected_fixture_key not in self._fixture_inputs:
            raise WebVerificationError("expected fixture key is not configured")
        state = await self._fixture_state_client.get_state(session.session_id)
        actual = _lookup_path(
            state, (*spec.resource.split("."), *spec.field.split("."))
        )
        expected = self._fixture_inputs[spec.expected_fixture_key]
        verified = _compare(actual, expected, VerifierOperator(spec.operator))
        evidence_id = f"fixture:{session.session_id}:{spec.resource}.{spec.field}"
        details = (
            f"fixture state matched {spec.resource}.{spec.field}"
            if verified
            else f"fixture state did not match {spec.resource}.{spec.field}"
        )
        return VerificationResult(
            verified=verified,
            evidence_ids=(evidence_id,),
            details=details,
        )

    async def _verify_visible_result(
        self,
        session: SessionHandle,
        spec: VisibleResultVerifierSpec,
    ) -> VerificationResult:
        if self._observation_provider is None or self._snapshot_extractor is None:
            raise WebVerificationError("visible-result verifier is not configured")
        capture = await self._observation_provider.capture(session)
        snapshot = self._snapshot_extractor(capture)
        evidence_id = f"viewport:{snapshot.id}"
        for element in snapshot.elements:
            if spec.text not in element.label:
                continue
            if spec.role is not None and ElementRole(element.role).value != spec.role:
                continue
            return VerificationResult(
                verified=True,
                evidence_ids=(f"{evidence_id}:element:{element.id}",),
                details="persona-visible result matched",
            )
        return VerificationResult(
            verified=False,
            evidence_ids=(evidence_id,),
            details="persona-visible result not found",
        )


FixtureStateWebVerifier = WebVerifier
VisibleResultWebVerifier = WebVerifier


def _fixture_values(
    fixture_inputs: FixtureInputs | Mapping[str, str] | None,
) -> Mapping[str, str] | None:
    if fixture_inputs is None:
        return None
    if isinstance(fixture_inputs, FixtureInputs):
        return fixture_inputs.values
    return fixture_inputs


def _validate_origin(origin: str) -> str:
    parsed = urlsplit(origin.rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("fixture control origin must be an HTTP(S) origin")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("fixture control origin must not contain credentials")
    if parsed.path or parsed.query or parsed.fragment:
        raise ValueError("fixture control origin must not contain path or query")
    return f"{parsed.scheme}://{parsed.netloc}"


def _lookup_path(state: Mapping[str, object], path: tuple[str, ...]) -> object | None:
    current: object = state
    for part in path:
        mapping = _object_mapping(current)
        if mapping is None:
            return None
        current = mapping.get(part)
    return current


def _object_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    mapping = cast(Mapping[object, object], value)
    return {str(key): item for key, item in mapping.items()}


def _compare(actual: object, expected: str, operator: VerifierOperator) -> bool:
    if operator is VerifierOperator.EQUALS:
        return actual == expected
    if operator is VerifierOperator.NOT_EQUALS:
        return actual != expected
    if operator is VerifierOperator.CONTAINS:
        if isinstance(actual, str):
            return expected in actual
        if isinstance(actual, (list, tuple, set, frozenset)):
            return expected in actual
        return False
    if operator is VerifierOperator.TRUTHY:
        return bool(actual)
    if operator is VerifierOperator.FALSY:
        return not bool(actual)
    raise ValueError(f"unsupported verifier operator: {operator}")
