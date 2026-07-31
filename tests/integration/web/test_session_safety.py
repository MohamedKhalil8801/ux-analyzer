from __future__ import annotations

# ruff: noqa: E402
import asyncio
import socket
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import pytest
import pytest_asyncio
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from playwright.async_api import async_playwright

REPOSITORY_ROOT = Path(__file__).parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from fixture_app.app import app as fixture_app
from ux_analyzer.adapters.web.network_policy import (
    BrowserAllowedOrigins,
    NetworkPolicy,
)
from ux_analyzer.adapters.web.session import (
    PlaywrightSessionAdapter,
    ProviderFailure,
    SafetyBlocked,
)
from ux_analyzer.ports.observation import (
    ClickAction,
    NavigateAction,
    ObservationSessionConfig,
    PlatformAction,
    ScrollAction,
    ViewportSize,
)
from ux_analyzer.ports.observation import (
    TestAccountId as AccountId,
)
from ux_analyzer.ports.verification import VerificationProvider


@fixture_app.get("/__test-redirect")
async def _test_redirect(target: str = Query(...)) -> RedirectResponse:
    return RedirectResponse(target, status_code=307)


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _server_app() -> FastAPI:
    server_app = FastAPI()

    @server_app.get("/redirect")
    async def redirect(target: str) -> RedirectResponse:
        return RedirectResponse(target, status_code=307)

    @server_app.get("/pixel")
    async def pixel() -> Response:
        return Response(content=b"pixel", media_type="image/gif")

    @server_app.get("/model")
    async def model() -> dict[str, str]:
        return {"status": "ok"}

    @server_app.get("/page")
    async def page() -> HTMLResponse:
        return HTMLResponse("<html><body><h1>Fixture</h1></body></html>")

    return server_app


@pytest_asyncio.fixture
async def running_servers() -> tuple[str, str]:
    fixture_port = _free_port()
    foreign_port = _free_port()
    fixture_config = uvicorn.Config(
        fixture_app, host="127.0.0.1", port=fixture_port, log_level="error"
    )
    foreign_config = uvicorn.Config(
        _server_app(), host="127.0.0.1", port=foreign_port, log_level="error"
    )
    fixture_server = uvicorn.Server(fixture_config)
    foreign_server = uvicorn.Server(foreign_config)
    fixture_task = asyncio.create_task(fixture_server.serve())
    foreign_task = asyncio.create_task(foreign_server.serve())

    async with httpx.AsyncClient() as client:
        for url in (
            f"http://127.0.0.1:{fixture_port}/app/ready/improved",
            f"http://127.0.0.1:{foreign_port}/page",
        ):
            for _ in range(100):
                try:
                    if (await client.get(url)).status_code < 500:
                        break
                except httpx.ConnectError:
                    await asyncio.sleep(0.01)
            else:
                raise RuntimeError(f"server did not start: {url}")

    yield (
        f"http://127.0.0.1:{fixture_port}",
        f"http://127.0.0.1:{foreign_port}",
    )

    fixture_server.should_exit = True
    foreign_server.should_exit = True
    await asyncio.gather(fixture_task, foreign_task)


@pytest_asyncio.fixture
async def browser_adapter(running_servers: tuple[str, str], tmp_path: Path) -> Any:
    fixture_origin, _ = running_servers
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        adapter = PlaywrightSessionAdapter(
            browser=browser,
            allowed_origins=NetworkPolicy.fixture_only((fixture_origin,)).origins,
            trace_directory=tmp_path / "traces",
        )
        try:
            yield adapter
        finally:
            await adapter.close()
            await browser.close()


def _session_config(origin: str, trace_path: Path, account_id: str = "test-account-1"):
    return ObservationSessionConfig(
        session_id="session-1",
        start_url=f"{origin}/app/session-1/improved",
        test_account_id=AccountId(account_id),
        viewport=ViewportSize(width=1024, height=768),
        trace_path=trace_path,
    )


@pytest.mark.asyncio
async def test_platform_action_union_is_closed_and_platform_neutral() -> None:
    actions: tuple[PlatformAction, ...] = (
        ClickAction(element_id="invite"),
        NavigateAction(url="http://fixture.test/page"),
        ScrollAction(direction="down", amount=400),
    )

    assert [action.kind for action in actions] == ["click", "navigate", "scroll"]
    assert "selector" not in ClickAction.__annotations__


def test_verification_port_has_no_browser_dependency() -> None:
    assert VerificationProvider.__module__ == "ux_analyzer.ports.verification"


def test_fixture_policy_rejects_foreign_schemes_and_explicit_about_navigation() -> None:
    origins = BrowserAllowedOrigins.fixture_only(("http://fixture.test",))

    assert origins.allows("about:blank", kind="internal")
    for url in (
        "about:config",
        "data:text/html,unsafe",
        "blob:http://fixture.test/unsafe",
        "file:///etc/passwd",
        "ftp://fixture.test/file",
        "javascript:alert(1)",
    ):
        assert not origins.allows(url)
        with pytest.raises(SafetyBlocked):
            origins.require_allowed(url)

    with pytest.raises(SafetyBlocked):
        origins.require_allowed(
            "about:blank", resource_type="document", kind="navigation"
        )


@pytest.mark.asyncio
async def test_foreign_navigation_and_redirect_are_blocked(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
) -> None:
    fixture_origin, foreign_origin = running_servers
    session = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / "redirect.zip")
    )

    with pytest.raises(SafetyBlocked):
        await browser_adapter.execute(
            session, NavigateAction(url=f"{foreign_origin}/page")
        )

    redirect_session_config = ObservationSessionConfig(
        session_id="redirect-session",
        start_url=f"{fixture_origin}/app/redirect-session/improved",
        test_account_id=AccountId("test-redirect"),
        viewport=ViewportSize(width=1024, height=768),
        trace_path=tmp_path / "redirect-follow.zip",
    )
    redirect_session = await browser_adapter.start_session(redirect_session_config)
    with pytest.raises(SafetyBlocked):
        await browser_adapter.execute(
            redirect_session,
            NavigateAction(
                url=(
                    f"{fixture_origin}/__test-redirect?target="
                    f"{quote(f'{foreign_origin}/page', safe='')}"
                )
            ),
        )

    assert all(event.origin == foreign_origin for event in session.blocked_events)
    assert all(
        event.origin == foreign_origin for event in redirect_session.blocked_events
    )


@pytest.mark.asyncio
async def test_foreign_popup_form_fetch_and_image_are_blocked(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
) -> None:
    fixture_origin, foreign_origin = running_servers
    session = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / "resources.zip")
    )
    page = browser_adapter.page_for_testing(session)
    await page.goto("about:blank")
    await page.set_content(
        "<html><body><iframe name='sink' hidden></iframe></body></html>"
    )
    await page.evaluate(
        """
        ([foreign]) => {
          const image = new Image();
          image.src = `${foreign}/pixel`;
          document.body.append(image);
          const form = document.createElement('form');
          form.action = `${foreign}/page`;
          form.method = 'post';
          form.target = 'sink';
          document.body.append(form);
          form.requestSubmit();
          fetch(`${foreign}/page`).catch(() => {});
          window.open(`${foreign}/page`, 'foreign-popup');
        }
        """,
        [foreign_origin],
    )
    await page.wait_for_timeout(500)

    resource_types = {event.resource_type for event in session.blocked_events}
    assert {"document", "fetch", "image"}.issubset(resource_types)
    assert any(event.kind == "popup" for event in session.blocked_events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    (
        "data:text/html,<h1>unsafe</h1>",
        "blob:http://fixture.test/unsafe-page",
        "about:blank",
    ),
)
async def test_non_fixture_page_schemes_are_blocked_as_navigation(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
    url: str,
) -> None:
    fixture_origin, _ = running_servers
    session = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / f"scheme-{quote(url, safe='')}.zip")
    )

    with pytest.raises(SafetyBlocked):
        await browser_adapter.execute(session, NavigateAction(url=url))


@pytest.mark.asyncio
async def test_foreign_websocket_is_closed_and_recorded(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
) -> None:
    fixture_origin, foreign_origin = running_servers
    foreign_websocket = foreign_origin.replace("http://", "ws://") + "/socket"
    session = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / "websocket.zip")
    )
    page = browser_adapter.page_for_testing(session)

    result = await page.evaluate(
        """
        url => new Promise(resolve => {
          const socket = new WebSocket(url);
          socket.addEventListener('open', () => resolve('opened'));
          socket.addEventListener('error', () => resolve('rejected'));
          socket.addEventListener('close', () => resolve('closed'));
        })
        """,
        foreign_websocket,
    )

    assert result in {"closed", "rejected"}
    assert any(
        event.kind == "websocket" and event.origin == foreign_origin
        for event in session.blocked_events
    )


@pytest.mark.asyncio
async def test_fixture_assets_permitted_model_traffic_unaffected(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
) -> None:
    fixture_origin, foreign_origin = running_servers
    session = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / "assets.zip")
    )
    capture = await browser_adapter.capture(session)

    assert capture.viewport == ViewportSize(width=1024, height=768)
    assert not any(event.origin == fixture_origin for event in session.blocked_events)

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{foreign_origin}/model")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_sessions_use_isolated_fixed_contexts_and_denied_capabilities(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
) -> None:
    fixture_origin, _ = running_servers
    first = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / "first.zip", "test-first")
    )
    second_config = _session_config(
        fixture_origin, tmp_path / "second.zip", "test-second"
    )
    second_config = ObservationSessionConfig(
        session_id="session-2",
        start_url=second_config.start_url.replace("session-1", "session-2"),
        test_account_id=second_config.test_account_id,
        viewport=second_config.viewport,
        trace_path=second_config.trace_path,
    )
    second = await browser_adapter.start_session(second_config)

    first_page = browser_adapter.page_for_testing(first)
    second_page = browser_adapter.page_for_testing(second)
    await first_page.evaluate("document.cookie = 'private=first'")

    assert await first_page.evaluate("document.cookie") == "private=first"
    assert await second_page.evaluate("document.cookie") == ""
    assert await first_page.evaluate("innerWidth") == 1024
    assert await first_page.evaluate("innerHeight") == 768
    assert await first_page.evaluate("Notification.permission") == "denied"
    assert (
        await first_page.evaluate(
            """async () => new Promise(resolve => {
            navigator.geolocation.getCurrentPosition(
              () => resolve('granted'),
              () => resolve('denied')
            );
        })"""
        )
        == "denied"
    )
    assert (
        await first_page.evaluate(
            """async () => {
            if (!navigator.clipboard) return 'denied';
            try {
              await navigator.clipboard.writeText('secret');
              return 'granted';
            } catch (error) {
              return 'denied';
            }
        }"""
        )
        == "denied"
    )

    await browser_adapter.end_session(first)
    await browser_adapter.end_session(second)
    assert first.trace_path.is_file()
    assert second.trace_path.is_file()


@pytest.mark.asyncio
async def test_non_test_account_is_rejected_before_context_creation(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
) -> None:
    fixture_origin, _ = running_servers

    with pytest.raises(ValueError, match="test account"):
        _session_config(fixture_origin, tmp_path / "unsafe.zip", "prod-user")

    assert browser_adapter.active_session_count == 0


@pytest.mark.asyncio
async def test_provider_failure_retains_trace_and_cleans_up(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
) -> None:
    fixture_origin, _ = running_servers
    session = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / "failure.zip")
    )

    with pytest.raises(ProviderFailure):
        await browser_adapter.execute(
            session, ClickAction(element_id="missing-private-target")
        )

    assert browser_adapter.active_session_count == 0
    assert session.trace_path.is_file()
