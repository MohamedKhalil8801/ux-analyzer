"""Fail-closed browser network policy for bundled fixture origins."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit

from playwright.async_api import Request, Route

from ux_analyzer.ports.observation import BlockedRequest, SafetyBlocked

_NON_NETWORK_SCHEMES = frozenset({"about", "blob", "data"})
_FIXTURE_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "fixture.test"})


def _empty_blocked_requests() -> list[BlockedRequest]:
    return []


def _origin(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("fixture origin must be an HTTP or HTTPS origin")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("fixture origin must not contain credentials")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("fixture origin must not contain path or query")
    hostname = parsed.hostname.lower()
    if hostname not in _FIXTURE_HOSTS:
        raise ValueError("only bundled fixture origins may be allowlisted")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("fixture origin has invalid port") from error
    host = f"[{hostname}]" if ":" in hostname else hostname
    return f"{parsed.scheme.lower()}://{host}{f':{port}' if port else ''}"


def _request_origin(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme in _NON_NETWORK_SCHEMES:
        return value
    if parsed.scheme not in {"http", "https"}:
        return value
    if not parsed.scheme or not parsed.netloc:
        return value
    return _origin(f"{parsed.scheme}://{parsed.netloc}")


@dataclass(frozen=True, slots=True)
class BrowserAllowedOrigins:
    """Exact fixture origins accepted by browser requests."""

    origins: frozenset[str]

    def __post_init__(self) -> None:
        normalized = frozenset(_origin(item) for item in self.origins)
        if not normalized:
            raise ValueError("at least one fixture origin is required")
        object.__setattr__(self, "origins", normalized)

    @classmethod
    def fixture_only(cls, origins: Iterable[str]) -> BrowserAllowedOrigins:
        return cls(frozenset(origins))

    def allows(self, url: str) -> bool:
        parsed = urlsplit(url)
        if parsed.scheme in _NON_NETWORK_SCHEMES:
            return True
        try:
            return _request_origin(url) in self.origins
        except ValueError:
            return False

    def require_allowed(
        self, url: str, *, resource_type: str = "other", kind: str = "request"
    ) -> None:
        if self.allows(url):
            return
        origin = _request_origin(url)
        raise SafetyBlocked(
            f"browser {kind} blocked for foreign origin {origin!r} ({resource_type})"
        )


@dataclass(slots=True)
class NetworkPolicy:
    """Playwright route handler and sanitized blocked-request journal."""

    allowed_origins: BrowserAllowedOrigins
    blocked_requests: list[BlockedRequest] = field(
        default_factory=_empty_blocked_requests
    )

    @classmethod
    def fixture_only(cls, origins: Iterable[str]) -> NetworkPolicy:
        return cls(BrowserAllowedOrigins.fixture_only(origins))

    @property
    def origins(self) -> frozenset[str]:
        return self.allowed_origins.origins

    def check(
        self, url: str, *, resource_type: str = "other", kind: str = "request"
    ) -> None:
        try:
            self.allowed_origins.require_allowed(
                url, resource_type=resource_type, kind=kind
            )
        except SafetyBlocked:
            event = BlockedRequest(
                url=url,
                origin=_request_origin(url),
                resource_type=resource_type,
                kind=kind,
            )
            self.blocked_requests.append(event)
            raise

    def record_popup(self, url: str) -> None:
        origin = _request_origin(url)
        if origin in self.origins:
            return
        self.blocked_requests.append(
            BlockedRequest(
                url=url,
                origin=origin,
                resource_type="document",
                kind="popup",
            )
        )

    async def handle_route(self, route: Route, request: Request) -> None:
        try:
            self.check(
                request.url,
                resource_type=request.resource_type,
                kind="request",
            )
        except SafetyBlocked:
            await route.abort(error_code="blockedbyclient")
            return
        response = await route.fetch(max_redirects=0)
        location = response.headers.get("location")
        if location:
            redirect_url = urljoin(request.url, location)
            try:
                self.check(
                    redirect_url,
                    resource_type=request.resource_type,
                    kind="redirect",
                )
            except SafetyBlocked:
                await route.abort(error_code="blockedbyclient")
                return
        await route.fulfill(response=response)
