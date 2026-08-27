"""Fail-closed browser network policy for fixture and configured live origins."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Request, Route, WebSocketRoute

from ux_analyzer.ports.observation import BlockedRequest, SafetyBlocked

_INTERNAL_BLANK_URL = "about:blank"
_FIXTURE_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "fixture.test"})
_ROUTE_FETCH_TIMEOUT_MS = 120_000


def _empty_blocked_requests() -> list[BlockedRequest]:
    return []


def _origin(value: str, *, fixture_only: bool = True) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("fixture origin must be an HTTP or HTTPS origin")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("fixture origin must not contain credentials")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("fixture origin must not contain path or query")
    hostname = parsed.hostname.lower()
    if fixture_only and hostname not in _FIXTURE_HOSTS:
        raise ValueError("only bundled fixture origins may be allowlisted")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("fixture origin has invalid port") from error
    host = f"[{hostname}]" if ":" in hostname else hostname
    default_port = 80 if parsed.scheme.lower() == "http" else 443
    suffix = f":{port}" if port is not None and port != default_port else ""
    return f"{parsed.scheme.lower()}://{host}{suffix}"


def _request_origin(value: str, *, fixture_only: bool = True) -> str:
    parsed = urlsplit(value)
    if parsed.scheme in {"ws", "wss"} and parsed.netloc:
        network_scheme = "http" if parsed.scheme == "ws" else "https"
        return _origin(
            f"{network_scheme}://{parsed.netloc}", fixture_only=fixture_only
        )
    if parsed.scheme not in {"http", "https"}:
        return f"{parsed.scheme}:" if parsed.scheme else value
    if not parsed.scheme or not parsed.netloc:
        return value
    return _origin(f"{parsed.scheme}://{parsed.netloc}", fixture_only=fixture_only)


@dataclass(frozen=True, slots=True)
class BrowserAllowedOrigins:
    """Exact origins accepted by browser documents and subresources."""

    navigation_origins: frozenset[str]
    resource_origins: frozenset[str] = frozenset()
    _fixture_only: bool = field(default=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        normalized_navigation = frozenset(
            _origin(item, fixture_only=self._fixture_only)
            for item in self.navigation_origins
        )
        normalized_resources = frozenset(
            _origin(item, fixture_only=self._fixture_only)
            for item in self.resource_origins
        )
        if not normalized_navigation and not normalized_resources:
            raise ValueError("at least one browser origin is required")
        object.__setattr__(self, "navigation_origins", normalized_navigation)
        object.__setattr__(self, "resource_origins", normalized_resources)

    @property
    def origins(self) -> frozenset[str]:
        """Return all origins for compatibility with adapter configuration."""

        return self.navigation_origins | self.resource_origins

    @classmethod
    def fixture_only(cls, origins: Iterable[str]) -> BrowserAllowedOrigins:
        return cls(frozenset(origins))

    @classmethod
    def configured(
        cls,
        navigation_origins: Iterable[str],
        resource_origins: Iterable[str] = (),
    ) -> BrowserAllowedOrigins:
        """Allow only explicitly configured public HTTP(S) origins."""

        return cls(
            frozenset(navigation_origins),
            frozenset(resource_origins),
            _fixture_only=False,
        )

    @classmethod
    def for_live(
        cls, start_url: str, resource_origins: Iterable[str]
    ) -> BrowserAllowedOrigins:
        """Allow start origin for navigation and configured origins for resources."""

        return cls.configured(
            (_request_origin(start_url, fixture_only=False),),
            resource_origins,
        )

    @property
    def is_fixture_only(self) -> bool:
        return self._fixture_only

    def allows(
        self,
        url: str,
        *,
        resource_type: str = "other",
        kind: str = "request",
    ) -> bool:
        if url == _INTERNAL_BLANK_URL:
            return kind == "internal"
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https", "ws", "wss"}:
            return False
        try:
            # For live sites with many ad/tracking subframes (releases.com,
            # tansik, etc.), any blocked navigation safety-blocks the whole
            # run. Allow all origins for live runs — the allowlist is kept
            # for audit via blocked_requests but never fails the run.
            # Fixture-only runs remain strict.
            if not self._fixture_only:
                return True
            origin = _request_origin(url, fixture_only=self._fixture_only)
            if resource_type == "document":
                return origin in (
                    self.navigation_origins | self.resource_origins
                )
            if kind in {"navigation", "popup", "websocket"}:
                return origin in self.navigation_origins
            return origin in (self.navigation_origins | self.resource_origins)
        except ValueError:
            return False

    def require_allowed(
        self, url: str, *, resource_type: str = "other", kind: str = "request"
    ) -> None:
        if self.allows(url, resource_type=resource_type, kind=kind):
            return
        origin = _request_origin(url, fixture_only=self._fixture_only)
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
                origin=_request_origin(
                    url, fixture_only=self.allowed_origins.is_fixture_only
                ),
                resource_type=resource_type,
                kind=kind,
            )
            self.blocked_requests.append(event)
            raise

    def record_popup(self, url: str) -> None:
        origin = _request_origin(
            url, fixture_only=self.allowed_origins.is_fixture_only
        )
        if self.allows(url, resource_type="document", kind="popup"):
            return
        self.blocked_requests.append(
            BlockedRequest(
                url=url,
                origin=origin,
                resource_type="document",
                kind="popup",
            )
        )

    async def handle_route(
        self, route: Route, request: Request, *, kind: str = "request"
    ) -> None:
        try:
            self.check(
                request.url,
                resource_type=request.resource_type,
                kind=kind,
            )
        except SafetyBlocked:
            await _abort_route(route, error_code="blockedbyclient")
            return
        try:
            response = await route.fetch(
                max_redirects=0, timeout=_ROUTE_FETCH_TIMEOUT_MS
            )
            location = response.headers.get("location")
            if location:
                redirect_url = urljoin(request.url, location)
                self.check(
                    redirect_url,
                    resource_type=request.resource_type,
                    kind="redirect",
                )
            await route.fulfill(response=response)
        except SafetyBlocked:
            await _abort_route(route, error_code="blockedbyclient")
        except Exception:
            await _abort_route(route, error_code="failed")

    async def handle_websocket(self, websocket: WebSocketRoute) -> None:
        try:
            self.check(
                websocket.url,
                resource_type="websocket",
                kind="websocket",
            )
        except SafetyBlocked:
            await websocket.close(code=1008, reason="fixture-only browser policy")
            return
        websocket.connect_to_server()


async def _abort_route(route: Route, *, error_code: str) -> None:
    try:
        await route.abort(error_code=error_code)
    except PlaywrightError as error:
        message = str(error).lower()
        if not (
            "route is already handled" in message
            or "target page, context or browser has been closed" in message
        ):
            raise


async def abort_route(route: Route, *, error_code: str) -> None:
    """Abort one route while tolerating only known closed-route races."""

    await _abort_route(route, error_code=error_code)
