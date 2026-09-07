from __future__ import annotations

from types import SimpleNamespace

import pytest
from playwright.async_api import Error as PlaywrightError

from ux_analyzer.adapters.web.network_policy import BrowserAllowedOrigins, NetworkPolicy
from ux_analyzer.ports.observation import SafetyBlocked


class _FakeRoute:
    def __init__(self, response: object) -> None:
        self.response = response
        self.fetch_arguments: dict[str, object] | None = None
        self.fulfilled = False
        self.aborted = False

    async def fetch(self, **kwargs: object) -> object:
        self.fetch_arguments = kwargs
        return self.response

    async def fulfill(self, *, response: object) -> None:
        del response
        self.fulfilled = True

    async def abort(self, *, error_code: str) -> None:
        del error_code
        self.aborted = True


class _FailingRoute(_FakeRoute):
    async def fetch(self, **kwargs: object) -> object:
        self.fetch_arguments = kwargs
        raise RuntimeError("route fetch failed")


class _AbortFailingRoute(_FakeRoute):
    def __init__(self, response: object, error: BaseException) -> None:
        super().__init__(response)
        self.error = error

    async def abort(self, *, error_code: str) -> None:
        del error_code
        raise self.error


@pytest.mark.asyncio
async def test_allowed_route_fetch_disables_redirects_and_uses_finite_timeout() -> None:
    policy = NetworkPolicy(BrowserAllowedOrigins.fixture_only(("http://fixture.test",)))
    route = _FakeRoute(SimpleNamespace(headers={}))
    request = SimpleNamespace(
        url="http://fixture.test/media.mp4", resource_type="media"
    )

    await policy.handle_route(route, request)

    timeout = route.fetch_arguments["timeout"]
    assert isinstance(timeout, int)
    assert 0 < timeout <= 120_000
    assert route.fetch_arguments == {
        "max_redirects": 0,
        "timeout": timeout,
    }
    assert route.fulfilled
    assert not route.aborted


@pytest.mark.asyncio
async def test_route_fetch_failure_is_aborted_without_escaping_handler() -> None:
    policy = NetworkPolicy(BrowserAllowedOrigins.fixture_only(("http://fixture.test",)))
    route = _FailingRoute(SimpleNamespace(headers={}))
    request = SimpleNamespace(
        url="http://fixture.test/media.mp4", resource_type="media"
    )

    await policy.handle_route(route, request)

    assert route.aborted


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    (
        "Route is already handled!",
        "Target page, context or browser has been closed",
    ),
)
async def test_known_closed_or_handled_route_abort_errors_are_ignored(
    message: str,
) -> None:
    policy = NetworkPolicy(BrowserAllowedOrigins.fixture_only(("http://fixture.test",)))
    route = _AbortFailingRoute(
        SimpleNamespace(headers={"location": "https://foreign.example/media"}),
        PlaywrightError(message),
    )
    request = SimpleNamespace(
        url="http://fixture.test/media.mp4", resource_type="media"
    )

    await policy.handle_route(route, request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", (PlaywrightError("abort failed"), RuntimeError("abort failed"))
)
async def test_unexpected_route_abort_errors_propagate(error: BaseException) -> None:
    policy = NetworkPolicy(BrowserAllowedOrigins.fixture_only(("http://fixture.test",)))
    route = _AbortFailingRoute(
        SimpleNamespace(headers={"location": "https://foreign.example/media"}),
        error,
    )
    request = SimpleNamespace(
        url="http://fixture.test/media.mp4", resource_type="media"
    )

    with pytest.raises(type(error), match="abort failed"):
        await policy.handle_route(route, request)


def test_live_origins_allow_all_connections_and_journal_only() -> None:
    """Live runs deliberately allow every connection to keep runs unblocked.

    Ad-heavy live sites would otherwise safety-block the whole run, so live
    policies never reject a scheme-valid request; the allowlist stays as
    recorded context. Fixture-only policies remain strict.
    """

    origins = BrowserAllowedOrigins.for_live(
        "https://target.example/work",
        ("https://fonts.example",),
    )

    assert origins.navigation_origins == frozenset({"https://target.example"})
    assert origins.resource_origins == frozenset({"https://fonts.example"})
    for url, resource_type, kind in (
        ("https://target.example/work", "document", "navigation"),
        ("https://target.example/app.js", "script", "request"),
        ("https://fonts.example/site.css", "stylesheet", "request"),
        ("https://fonts.example/font.woff2", "font", "request"),
        ("https://fonts.example/page", "document", "navigation"),
        ("https://fonts.example/page", "document", "popup"),
        ("https://fonts.example/page", "document", "redirect"),
        ("wss://fonts.example/socket", "websocket", "websocket"),
        ("https://foreign.example/page", "document", "navigation"),
    ):
        assert origins.allows(url, resource_type=resource_type, kind=kind)
        origins.require_allowed(url, resource_type=resource_type, kind=kind)


def test_fixture_only_origins_stay_strict() -> None:
    origins = BrowserAllowedOrigins.fixture_only(("http://fixture.test",))

    assert origins.allows(
        "http://fixture.test/app", resource_type="document", kind="navigation"
    )
    assert origins.is_fixture_only
    with pytest.raises(SafetyBlocked):
        origins.require_allowed(
            "http://127.0.0.1:9999/page", resource_type="document", kind="navigation"
        )
    # Non-fixture hosts cannot even be classified as origins.
    with pytest.raises(ValueError):
        origins.require_allowed(
            "https://foreign.example/page",
            resource_type="document",
            kind="navigation",
        )
