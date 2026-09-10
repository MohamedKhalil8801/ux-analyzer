"""Independent web verification adapters for controlled fixture runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
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
        self._last_capture: ObservationCapture | None = None
        self._last_snapshot: ViewportSnapshot | None = None
        self._verification_capture_count = 0
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

    @property
    def last_capture(self) -> ObservationCapture | None:
        return self._last_capture

    @property
    def last_snapshot(self) -> ViewportSnapshot | None:
        return self._last_snapshot

    async def verify(self, session: SessionHandle) -> VerificationResult:
        self._last_capture = None
        self._last_snapshot = None
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
        self._verification_capture_count += 1
        verification_viewport_id = (
            f"{capture.viewport_id}-verification-{self._verification_capture_count}"
        )
        capture = replace(capture, viewport_id=verification_viewport_id)
        snapshot = self._snapshot_extractor(capture)
        snapshot = _relabel_snapshot(snapshot, verification_viewport_id)
        capture = replace(capture, snapshot=snapshot)
        self._last_capture = capture
        self._last_snapshot = snapshot
        evidence_id = f"viewport:{snapshot.id}"
        effective_visible_elements = tuple(
            element
            for element in snapshot.elements
            if element.visibility_fraction > 0 and element.rendered_text is not None
        )
        effective_rendered = tuple(
            _normalized_verifier_text(element.rendered_text or "")
            for element in effective_visible_elements
        )
        normalized_all_of = tuple(
            _normalized_verifier_text(item) for item in spec.all_of
        )
        normalized_target = _normalized_verifier_text(spec.text)
        for element in snapshot.elements:
            rendered_text = element.rendered_text
            if rendered_text is None or element.visibility_fraction <= 0:
                continue
            if normalized_target not in _normalized_verifier_text(rendered_text):
                continue
            if spec.role is not None and ElementRole(element.role).value != spec.role:
                continue
            if not all(
                any(required_text in text for text in effective_rendered)
                for required_text in normalized_all_of
            ):
                continue
            return VerificationResult(
                verified=True,
                evidence_ids=(f"{evidence_id}:element:{element.id}",),
                details="persona-visible result matched",
            )
        # Fallback: if the target is rendered somewhere in the document but
        # sits below the current viewport (so no element in the viewport
        # snapshot contains it), accept the full page text. This guards
        # against verifier false-negatives when the agent reaches a section
        # that *contains* the result but the extractor has filtered the text
        # out of the viewport snapshot, or the result lives a few scrolls
        # below the section header. ``capture.page_text`` is the unfiltered
        # rendered text of the document (see ``ObservationCapture``). A
        # distinct detail string keeps the signal auditable in run records.
        if capture.page_text is not None:
            if normalized_target in _normalized_verifier_text(capture.page_text):
                if all(
                    required_text in _normalized_verifier_text(capture.page_text)
                    for required_text in normalized_all_of
                ):
                    return VerificationResult(
                        verified=True,
                        evidence_ids=(f"{evidence_id}:page-text",),
                        details="page-text result matched",
                    )
        return VerificationResult(
            verified=False,
            evidence_ids=(evidence_id,),
            details="persona-visible result not found",
        )


FixtureStateWebVerifier = WebVerifier
VisibleResultWebVerifier = WebVerifier


def _normalized_verifier_text(value: str) -> str:
    """Normalize rendered text for containment checks.

    Non-breaking spaces and other odd whitespace must not hide a genuinely
    visible label: replace them, collapse runs, and strip.
    """

    return " ".join(value.replace("\u00a0", " ").split())


def _relabel_snapshot(
    snapshot: ViewportSnapshot, viewport_id: str
) -> ViewportSnapshot:
    """Give verifier captures unique IDs while preserving internal references."""

    elements = tuple(
        replace(
            element,
            execution_reference=(
                replace(element.execution_reference, viewport_id=viewport_id)
                if element.execution_reference is not None
                else None
            ),
        )
        for element in snapshot.elements
    )
    return replace(
        snapshot,
        id=viewport_id,
        elements=elements,
    )


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
