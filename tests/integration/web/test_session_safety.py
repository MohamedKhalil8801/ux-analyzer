from __future__ import annotations

# ruff: noqa: E402
import asyncio
import gc
import json
import os
import socket
import sys
import zipfile
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote

import httpx
import pytest
import pytest_asyncio
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import (
    HTMLResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright

REPOSITORY_ROOT = Path(__file__).parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

_STREAMING_VIDEO_SIZE = 18_800_000
_STREAMING_VIDEO_STARTED: asyncio.Event | None = None
_FOREIGN_NAVIGATION_HIT: asyncio.Event | None = None
_FOREIGN_REDIRECT_MEDIA_HIT: asyncio.Event | None = None
_FONT_CSS_HIT: asyncio.Event | None = None
_FONT_FILE_HIT: asyncio.Event | None = None

from fixture_app.app import app as fixture_app
from ux_analyzer.adapters.web import session as web_session
from ux_analyzer.adapters.web.extractor import capture as capture_snapshot
from ux_analyzer.adapters.web.extractor import capture_with_diagnostics
from ux_analyzer.adapters.web.network_policy import (
    BrowserAllowedOrigins,
    NetworkPolicy,
)
from ux_analyzer.adapters.web.session import (
    PlaywrightSessionAdapter,
    ProviderFailure,
    SafetyBlocked,
    _blocked_document_navigation,
)
from ux_analyzer.adapters.web.verifier import WebVerifier
from ux_analyzer.domain.attention import ProgressiveObservation
from ux_analyzer.domain.benchmark import VisibleResultVerifierSpec
from ux_analyzer.ports.artifacts import BundleStateError, RedactionPolicy
from ux_analyzer.ports.observation import (
    BackAction,
    BlockedRequest,
    ClearTextAction,
    ClickAction,
    DoubleClickAction,
    DragAction,
    NavigateAction,
    ObservationSessionConfig,
    OpenMenuAction,
    PlatformAction,
    PressKeyAction,
    ScrollAction,
    SelectOptionAction,
    SessionHandle,
    SubmitAction,
    ToggleAction,
    TypeTextAction,
    ViewportSize,
    WaitAction,
)
from ux_analyzer.ports.observation import (
    TestAccountId as AccountId,
)
from ux_analyzer.ports.verification import VerificationProvider
from ux_analyzer.providers.cognitive import StructuredCognitiveAgent


@fixture_app.get("/__test-redirect")
async def _test_redirect(target: str = Query(...)) -> RedirectResponse:
    return RedirectResponse(target, status_code=307)


@fixture_app.post("/__test-slow-feedback")
async def _test_slow_feedback() -> HTMLResponse:
    await asyncio.sleep(0.2)
    return HTMLResponse(
        "<html><body><p role='status'>Two-factor authentication enabled.</p></body></html>"
    )


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

    @server_app.get("/blocked-navigation")
    async def blocked_navigation() -> HTMLResponse:
        if _FOREIGN_NAVIGATION_HIT is not None:
            _FOREIGN_NAVIGATION_HIT.set()
        return HTMLResponse("<html><body><h1>Foreign target</h1></body></html>")

    @server_app.get("/redirected-media")
    async def redirected_media() -> Response:
        if _FOREIGN_REDIRECT_MEDIA_HIT is not None:
            _FOREIGN_REDIRECT_MEDIA_HIT.set()
        return Response(content=b"foreign-media", media_type="image/png")

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
        navigation_origins=(origin,),
        fixture_only=True,
    )


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlink trace fixture unavailable: {error}")


@fixture_app.get("/__test-font.css")
async def _test_font_css() -> Response:
    if _FONT_CSS_HIT is not None:
        _FONT_CSS_HIT.set()
    return Response(
        content=(
            b"@font-face { font-family: TestFont; "
            b"src: url('/__test-font.woff2') format('woff2'); }"
            b"body { font-family: TestFont; }"
        ),
        media_type="text/css",
        headers={"access-control-allow-origin": "*"},
    )


@fixture_app.get("/__test-font.woff2")
async def _test_font_file() -> Response:
    if _FONT_FILE_HIT is not None:
        _FONT_FILE_HIT.set()
    return Response(
        content=b"font-bytes",
        media_type="font/woff2",
        headers={"access-control-allow-origin": "*"},
    )


@fixture_app.get("/__test-streaming-video")
async def _test_streaming_video() -> StreamingResponse:
    async def stream() -> AsyncIterator[bytes]:
        if _STREAMING_VIDEO_STARTED is not None:
            _STREAMING_VIDEO_STARTED.set()
        yield b"0" * 65_536
        await asyncio.sleep(5)
        yield b"0" * (_STREAMING_VIDEO_SIZE - 65_536)

    return StreamingResponse(
        stream(),
        media_type="video/mp4",
        headers={"content-length": str(_STREAMING_VIDEO_SIZE)},
    )


def _reset_streaming_video_started() -> asyncio.Event:
    global _STREAMING_VIDEO_STARTED
    _STREAMING_VIDEO_STARTED = asyncio.Event()
    return _STREAMING_VIDEO_STARTED


def _reset_foreign_navigation_hit() -> asyncio.Event:
    global _FOREIGN_NAVIGATION_HIT
    _FOREIGN_NAVIGATION_HIT = asyncio.Event()
    return _FOREIGN_NAVIGATION_HIT


def _reset_foreign_redirect_media_hit() -> asyncio.Event:
    global _FOREIGN_REDIRECT_MEDIA_HIT
    _FOREIGN_REDIRECT_MEDIA_HIT = asyncio.Event()
    return _FOREIGN_REDIRECT_MEDIA_HIT


def _reset_font_hits() -> tuple[asyncio.Event, asyncio.Event]:
    global _FONT_CSS_HIT, _FONT_FILE_HIT
    _FONT_CSS_HIT = asyncio.Event()
    _FONT_FILE_HIT = asyncio.Event()
    return _FONT_CSS_HIT, _FONT_FILE_HIT


@fixture_app.get("/__test-download.pdf")
async def _test_download_pdf() -> Response:
    return Response(
        content=b"%PDF-1.4\n% test download\n",
        media_type="application/pdf",
        headers={"content-disposition": 'attachment; filename="test.pdf"'},
    )


@fixture_app.get("/__test-delayed-navigation")
async def _test_delayed_navigation() -> HTMLResponse:
    return HTMLResponse(
        "<html><body><p role='status'>Delayed navigation settled.</p></body></html>"
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


def test_public_policy_requires_explicit_exact_origins() -> None:
    origins = BrowserAllowedOrigins.for_live(
        "HTTPS://Portfolio.Example:443/work",
        ("HTTPS://Fonts.Example:443/",),
    )

    assert origins.origins == frozenset(
        {"https://portfolio.example", "https://fonts.example"}
    )
    assert origins.navigation_origins == frozenset({"https://portfolio.example"})
    assert origins.resource_origins == frozenset({"https://fonts.example"})
    # Live policies allow every connection (ad-heavy sites must not kill
    # runs); the allowlist still records the explicitly configured origins.
    assert origins.allows(
        "https://portfolio.example/work", resource_type="document", kind="navigation"
    )
    assert origins.allows("https://fonts.example/site.css", resource_type="stylesheet")
    assert origins.allows("https://cdn.example/site.css")
    assert origins.allows("https://portfolio.example.evil/work")

    with pytest.raises(ValueError, match="bundled fixture origins"):
        BrowserAllowedOrigins.fixture_only(("https://portfolio.example",))


@pytest.mark.parametrize("url", ("", ":", "about:blank"))
def test_transient_popup_urls_are_download_artifacts(url: str) -> None:
    from ux_analyzer.adapters.web.session import _is_transient_popup_url

    assert _is_transient_popup_url(url)
    assert not _is_transient_popup_url("https://foreign.example/page")


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
async def test_clicking_foreign_visible_link_returns_blocked_result_and_recaptures(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
) -> None:
    fixture_origin, foreign_origin = running_servers
    foreign_hit = _reset_foreign_navigation_hit()
    session = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / "foreign-link.zip")
    )
    page = browser_adapter.page_for_testing(session)
    await page.set_content(
        f"<a href='{foreign_origin}/blocked-navigation'>Foreign visible link</a>"
    )
    link = page.get_by_role("link", name="Foreign visible link")
    bounds = await link.bounding_box()
    assert bounds is not None

    result = await browser_adapter.execute(
        session,
        ClickAction(
            element_id="foreign-link",
            bounds=type(
                "Bounds",
                (),
                {
                    "x": bounds["x"],
                    "y": bounds["y"],
                    "width": bounds["width"],
                    "height": bounds["height"],
                },
            )(),
        ),
    )

    assert not result.succeeded
    assert not result.state_changed
    assert result.error == "navigation blocked by safety policy"
    assert browser_adapter.active_session_count == 1
    await browser_adapter.capture(session)
    assert not foreign_hit.is_set()
    assert any(
        event.origin == foreign_origin and event.resource_type == "document"
        for event in session.blocked_events
    )


@pytest.mark.asyncio
async def test_clicking_foreign_popup_link_returns_blocked_result_and_recaptures(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
) -> None:
    fixture_origin, foreign_origin = running_servers
    foreign_hit = _reset_foreign_navigation_hit()
    config = replace(
        _session_config(fixture_origin, tmp_path / "foreign-popup-link.zip"),
        action_settle_ms=200,
    )
    session = await browser_adapter.start_session(config)
    page = browser_adapter.page_for_testing(session)
    await page.set_content(
        f"<a target='_blank' href='{foreign_origin}/blocked-navigation'>Foreign popup link</a>"
    )
    link = page.get_by_role("link", name="Foreign popup link")
    bounds = await link.bounding_box()
    assert bounds is not None

    result = await browser_adapter.execute(
        session,
        ClickAction(
            element_id="foreign-popup-link",
            bounds=type(
                "Bounds",
                (),
                {
                    "x": bounds["x"],
                    "y": bounds["y"],
                    "width": bounds["width"],
                    "height": bounds["height"],
                },
            )(),
        ),
    )

    assert not result.succeeded
    assert not result.state_changed
    assert result.error == "navigation blocked by safety policy"
    assert browser_adapter.active_session_count == 1
    await browser_adapter.capture(session)
    assert not foreign_hit.is_set()
    assert any(event.kind == "popup" for event in session.blocked_events)


@pytest.mark.asyncio
async def test_click_error_after_blocked_navigation_returns_recoverable_result(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_origin, foreign_origin = running_servers
    session = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / "click-error.zip")
    )
    managed = browser_adapter._sessions[session.session_id]

    async def fail_click(*_args: object, **_kwargs: object) -> None:
        managed.policy.blocked_requests.append(
            BlockedRequest(
                url=f"{foreign_origin}/blocked-navigation",
                origin=foreign_origin,
                resource_type="document",
                kind="request",
            )
        )
        raise PlaywrightError("click failed")

    monkeypatch.setattr(
        PlaywrightSessionAdapter,
        "_click",
        staticmethod(fail_click),
    )

    result = await browser_adapter.execute(
        session,
        ClickAction(
            element_id="foreign-link",
            bounds=type(
                "Bounds",
                (),
                {"x": 1, "y": 1, "width": 10, "height": 10},
            )(),
        ),
    )

    assert not result.succeeded
    assert not result.state_changed
    assert result.error == "navigation blocked by safety policy"
    assert browser_adapter.active_session_count == 1
    await browser_adapter.capture(session)


@pytest.mark.asyncio
async def test_cleanup_removes_popup_listener_and_serializes_concurrent_calls(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_origin, _ = running_servers
    session = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / "concurrent-cleanup.zip")
    )
    managed = browser_adapter._sessions[session.session_id]
    listener = managed.popup_listener
    assert listener is not None
    removed_listeners: list[object] = []
    original_remove_listener = managed.context.remove_listener

    def record_remove_listener(event: str, callback: object) -> None:
        removed_listeners.append(callback)
        original_remove_listener(event, callback)  # type: ignore[arg-type]

    monkeypatch.setattr(managed.context, "remove_listener", record_remove_listener)
    unroute_started = asyncio.Event()
    release_unroute = asyncio.Event()

    async def hold_unroute(*, behavior: str) -> None:
        assert behavior == "ignoreErrors"
        unroute_started.set()
        await release_unroute.wait()

    monkeypatch.setattr(managed.context, "unroute_all", hold_unroute)

    first = asyncio.create_task(browser_adapter.end_session(session))
    await asyncio.wait_for(unroute_started.wait(), timeout=2)
    assert managed.popup_listener is None

    second = asyncio.create_task(browser_adapter.end_session(session))
    await asyncio.sleep(0)
    assert not second.done()

    release_unroute.set()
    await asyncio.gather(first, second)

    assert removed_listeners == [listener]
    assert browser_adapter.active_session_count == 0


@pytest.mark.asyncio
async def test_route_callbacks_during_cleanup_abort_or_close_without_tasks(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_origin, _ = running_servers
    session = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / "callback-gate.zip")
    )
    managed = browser_adapter._sessions[session.session_id]
    route_handler = managed.route_handler
    websocket_handler = managed.websocket_handler
    assert route_handler is not None
    assert websocket_handler is not None
    unroute_started = asyncio.Event()
    release_unroute = asyncio.Event()

    async def hold_unroute(*, behavior: str) -> None:
        assert behavior == "ignoreErrors"
        unroute_started.set()
        await release_unroute.wait()

    monkeypatch.setattr(managed.context, "unroute_all", hold_unroute)
    first = asyncio.create_task(browser_adapter.end_session(session))
    await asyncio.wait_for(unroute_started.wait(), timeout=2)

    aborted_codes: list[str] = []

    async def abort(*, error_code: str) -> None:
        aborted_codes.append(error_code)

    closed_websockets: list[tuple[int, str]] = []

    async def close(*, code: int, reason: str) -> None:
        closed_websockets.append((code, reason))

    route = SimpleNamespace(abort=abort)
    websocket = SimpleNamespace(close=close)
    request = SimpleNamespace(
        url=f"{fixture_origin}/pixel", resource_type="image"
    )

    await route_handler(route, request)
    await websocket_handler(websocket)

    assert aborted_codes == ["blockedbyclient"]
    assert closed_websockets == [(1008, "session cleanup in progress")]
    assert managed.route_tasks == set()
    assert managed.popup_tasks == set()

    release_unroute.set()
    await first
    assert browser_adapter.active_session_count == 0


@pytest.mark.asyncio
async def test_duplicate_session_id_rejected_while_session_active(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
) -> None:
    fixture_origin, _ = running_servers
    session = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / "duplicate-active.zip")
    )
    duplicate_config = replace(
        _session_config(fixture_origin, tmp_path / "duplicate-active-2.zip"),
        session_id=session.session_id,
    )

    with pytest.raises(ProviderFailure, match="session ID already active"):
        await browser_adapter.start_session(duplicate_config)

    assert browser_adapter.active_session_count == 1


@pytest.mark.asyncio
async def test_duplicate_session_id_rejected_during_cleanup_without_replacing_mapping(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_origin, _ = running_servers
    session = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / "duplicate-cleanup.zip")
    )
    managed = browser_adapter._sessions[session.session_id]
    unroute_started = asyncio.Event()
    release_unroute = asyncio.Event()

    async def hold_unroute(*, behavior: str) -> None:
        assert behavior == "ignoreErrors"
        unroute_started.set()
        await release_unroute.wait()

    monkeypatch.setattr(managed.context, "unroute_all", hold_unroute)
    cleanup = asyncio.create_task(browser_adapter.end_session(session))
    await asyncio.wait_for(unroute_started.wait(), timeout=2)

    replacement: SessionHandle | None = None
    try:
        duplicate_config = replace(
            _session_config(fixture_origin, tmp_path / "replacement.zip"),
            session_id=session.session_id,
        )
        with pytest.raises(ProviderFailure, match="session ID already active"):
            replacement = await browser_adapter.start_session(duplicate_config)
    finally:
        release_unroute.set()
        await cleanup
        if replacement is not None:
            await browser_adapter.end_session(replacement)

    assert browser_adapter.active_session_count == 0
    assert not (tmp_path / "replacement.zip").exists()


def _action_bounds() -> object:
    return type(
        "Bounds",
        (),
        {"x": 1, "y": 1, "width": 10, "height": 10},
    )()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "recoverable"),
    (
        pytest.param(ClickAction(element_id="target", bounds=_action_bounds()), True, id="click"),
        pytest.param(DoubleClickAction(element_id="target", bounds=_action_bounds()), True, id="double-click"),
        pytest.param(TypeTextAction(element_id="target", text="x", bounds=_action_bounds()), True, id="type"),
        pytest.param(SelectOptionAction(element_id="target", option="x", bounds=_action_bounds()), True, id="select"),
        pytest.param(ToggleAction(element_id="target", bounds=_action_bounds()), True, id="toggle"),
        pytest.param(SubmitAction(element_id="target", bounds=_action_bounds()), True, id="submit"),
        pytest.param(OpenMenuAction(element_id="target", bounds=_action_bounds()), True, id="open-menu"),
        pytest.param(ClearTextAction(element_id="target", bounds=_action_bounds()), True, id="clear"),
        pytest.param(BackAction(), False, id="back"),
        pytest.param(WaitAction(), False, id="wait"),
        pytest.param(PressKeyAction(key="Escape"), False, id="press-key"),
        pytest.param(ScrollAction(direction="down", amount=10), False, id="scroll"),
        pytest.param(DragAction(element_id="target", end_x=20, end_y=20, bounds=_action_bounds()), False, id="drag"),
        pytest.param(NavigateAction(url="/page"), False, id="navigate"),
    ),
)
async def test_blocked_navigation_is_recoverable_only_for_user_interactions(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: PlatformAction,
    recoverable: bool,
) -> None:
    fixture_origin, foreign_origin = running_servers
    session = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / f"action-{action.kind}.zip")
    )
    managed = browser_adapter._sessions[session.session_id]

    async def no_op_click(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        PlaywrightSessionAdapter,
        "_click",
        staticmethod(no_op_click),
    )

    async def no_op_back(*, wait_until: str) -> None:
        del wait_until

    monkeypatch.setattr(managed.page, "go_back", no_op_back)

    async def record_blocked_navigation(*_args: object, **_kwargs: object) -> None:
        managed.policy.blocked_requests.append(
            BlockedRequest(
                url=f"{foreign_origin}/blocked-navigation",
                origin=foreign_origin,
                resource_type="document",
                kind="request",
            )
        )

    monkeypatch.setattr(
        "ux_analyzer.adapters.web.session._settle_after_action",
        record_blocked_navigation,
    )

    if recoverable:
        result = await browser_adapter.execute(session, action)
        assert not result.succeeded
        assert not result.state_changed
        assert result.error == "navigation blocked by safety policy"
        assert browser_adapter.active_session_count == 1
    else:
        with pytest.raises(SafetyBlocked):
            await browser_adapter.execute(session, action)
        assert browser_adapter.active_session_count == 0


@pytest.mark.asyncio
async def test_streaming_media_route_does_not_leak_into_next_session(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
) -> None:
    fixture_origin, _ = running_servers
    streaming_started = _reset_streaming_video_started()
    first = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / "streaming-first.zip", "test-first")
    )
    first_page = browser_adapter.page_for_testing(first)
    await first_page.evaluate(
        """
        () => {
          const video = document.createElement('video');
          video.src = '/__test-streaming-video';
          video.preload = 'auto';
          document.body.append(video);
        }
        """
    )
    await asyncio.wait_for(streaming_started.wait(), timeout=2)

    loop = asyncio.get_running_loop()
    loop_errors: list[dict[str, object]] = []
    previous_exception_handler = loop.get_exception_handler()

    def record_loop_error(
        _loop: asyncio.AbstractEventLoop, context: dict[str, object]
    ) -> None:
        loop_errors.append(context)

    loop.set_exception_handler(record_loop_error)
    try:
        await browser_adapter.end_session(first)
        second = await browser_adapter.start_session(
            replace(
                _session_config(
                    fixture_origin,
                    tmp_path / "streaming-second.zip",
                    "test-second",
                ),
                session_id="session-2",
                start_url=f"{fixture_origin}/app/session-2/improved",
            )
        )
        capture = await browser_adapter.capture(second)
        await asyncio.sleep(0.05)
    finally:
        loop.set_exception_handler(previous_exception_handler)

    assert browser_adapter.active_session_count == 1
    assert capture.url == f"{fixture_origin}/app/session-2/improved"
    assert loop_errors == []


@pytest.mark.asyncio
async def test_foreign_popup_then_streaming_teardown_has_no_unhandled_tasks(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
) -> None:
    fixture_origin, foreign_origin = running_servers
    loop = asyncio.get_running_loop()
    loop_errors: list[dict[str, object]] = []
    previous_exception_handler = loop.get_exception_handler()

    def record_loop_error(
        _loop: asyncio.AbstractEventLoop, context: dict[str, object]
    ) -> None:
        loop_errors.append(context)

    loop.set_exception_handler(record_loop_error)
    try:
        foreign_session = await browser_adapter.start_session(
            replace(
                _session_config(
                    fixture_origin,
                    tmp_path / "ordered-foreign-popup.zip",
                ),
                session_id="ordered-foreign-popup",
                start_url=f"{fixture_origin}/app/ordered-foreign-popup/improved",
                action_settle_ms=200,
            )
        )
        foreign_page = browser_adapter.page_for_testing(foreign_session)
        await foreign_page.set_content(
            f"<a target='_blank' href='{foreign_origin}/page'>Foreign popup</a>"
        )
        foreign_link = foreign_page.get_by_role("link", name="Foreign popup")
        foreign_bounds = await foreign_link.bounding_box()
        assert foreign_bounds is not None

        result = await browser_adapter.execute(
            foreign_session,
            ClickAction(
                element_id="ordered-foreign-popup",
                bounds=type(
                    "Bounds",
                    (),
                    {
                        "x": foreign_bounds["x"],
                        "y": foreign_bounds["y"],
                        "width": foreign_bounds["width"],
                        "height": foreign_bounds["height"],
                    },
                )(),
            ),
        )
        assert result.error == "navigation blocked by safety policy"
        assert browser_adapter.active_session_count == 1

        streaming_started = _reset_streaming_video_started()
        streaming_session = await browser_adapter.start_session(
            replace(
                _session_config(
                    fixture_origin,
                    tmp_path / "ordered-streaming.zip",
                ),
                session_id="ordered-streaming",
                start_url=f"{fixture_origin}/app/ordered-streaming/improved",
            )
        )
        streaming_page = browser_adapter.page_for_testing(streaming_session)
        await streaming_page.evaluate(
            """
            () => {
              const video = document.createElement('video');
              video.src = '/__test-streaming-video';
              video.preload = 'auto';
              document.body.append(video);
            }
            """
        )
        await asyncio.wait_for(streaming_started.wait(), timeout=2)
        await browser_adapter.end_session(streaming_session)

        next_session = await browser_adapter.start_session(
            replace(
                _session_config(
                    fixture_origin,
                    tmp_path / "ordered-next.zip",
                ),
                session_id="ordered-next",
                start_url=f"{fixture_origin}/app/ordered-next/improved",
            )
        )
        capture = await browser_adapter.capture(next_session)
        await asyncio.sleep(0.05)
    finally:
        loop.set_exception_handler(previous_exception_handler)

    assert capture.url == f"{fixture_origin}/app/ordered-next/improved"
    assert loop_errors == []


@pytest.mark.asyncio
async def test_teardown_with_inflight_route_has_no_unhandled_tasks(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_origin, _ = running_servers
    original_route = NetworkPolicy.handle_route

    async def slow_handle_route(
        self: NetworkPolicy, route: Any, request: Any, *, kind: str = "request"
    ) -> None:
        await asyncio.sleep(0.3)
        await original_route(self, route, request, kind=kind)

    monkeypatch.setattr(NetworkPolicy, "handle_route", slow_handle_route)

    from playwright._impl import _browser_context as _pw_context

    original_update = _pw_context.BrowserContext._update_interception_patterns

    async def slow_update(self: Any) -> None:
        await asyncio.sleep(0.3)
        await original_update(self)

    monkeypatch.setattr(
        _pw_context.BrowserContext,
        "_update_interception_patterns",
        slow_update,
    )

    session = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / "inflight.zip", "test-inflight")
    )
    page = browser_adapter.page_for_testing(session)
    await page.evaluate(
        """
        () => {
          for (let i = 0; i < 8; i++) {
            const img = document.createElement('img');
            img.src = '/page?i=' + i;
            document.body.append(img);
          }
        }
        """
    )
    await asyncio.sleep(0.05)

    loop = asyncio.get_running_loop()
    loop_errors: list[dict[str, object]] = []
    previous_exception_handler = loop.get_exception_handler()

    def record_loop_error(
        _loop: asyncio.AbstractEventLoop, context: dict[str, object]
    ) -> None:
        loop_errors.append(context)

    loop.set_exception_handler(record_loop_error)
    try:
        await browser_adapter.end_session(session)
        gc.collect()
        await asyncio.sleep(0.3)
        gc.collect()
        await asyncio.sleep(0.3)
    finally:
        loop.set_exception_handler(previous_exception_handler)

    assert loop_errors == []


@pytest.mark.asyncio
async def test_same_origin_pdf_popup_is_download_artifact_and_session_stays_active(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
) -> None:
    fixture_origin, _ = running_servers
    config = replace(
        _session_config(fixture_origin, tmp_path / "pdf-popup.zip"),
        action_settle_ms=100,
    )
    session = await browser_adapter.start_session(config)
    page = browser_adapter.page_for_testing(session)
    popup_pages: list[Any] = []
    popup_urls: list[str] = []

    def record_popup(popup: Any) -> None:
        popup_pages.append(popup)
        popup_urls.append(popup.url)

    page.context.on("page", record_popup)
    try:
        await page.set_content(
            "<a target='_blank' href='/__test-download.pdf'>Download PDF</a>"
        )
        link = page.get_by_role("link", name="Download PDF")
        bounds = await link.bounding_box()
        assert bounds is not None

        await browser_adapter.execute(
            session,
            ClickAction(
                element_id="download-pdf",
                bounds=type(
                    "Bounds",
                    (),
                    {
                        "x": bounds["x"],
                        "y": bounds["y"],
                        "width": bounds["width"],
                        "height": bounds["height"],
                    },
                )(),
            ),
        )
        assert popup_pages
        assert popup_pages[0].is_closed()
        capture = await browser_adapter.capture(session)
    finally:
        page.context.remove_listener("page", record_popup)

    assert popup_urls
    assert popup_urls[0] in {"", ":", "about:blank"}
    assert browser_adapter.active_session_count == 1
    assert not any(event.kind == "popup" for event in session.blocked_events)
    assert capture.url == f"{fixture_origin}/app/session-1/improved"


def test_blocked_subresources_do_not_classify_action_navigation() -> None:
    events = [
        BlockedRequest(
            url="https://foreign.example/pixel",
            origin="https://foreign.example",
            resource_type="image",
            kind="request",
        ),
        BlockedRequest(
            url="https://foreign.example/model",
            origin="https://foreign.example",
            resource_type="fetch",
            kind="request",
        ),
    ]

    assert _blocked_document_navigation(events, 0) is None


@pytest.mark.asyncio
async def test_live_session_navigates_foreign_origins_without_blocks(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
) -> None:
    """Live sessions deliberately allow every connection; nothing is blocked.

    Fixture-only sessions keep strict behavior, so the fixture session still
    serves the app while the live session may navigate anywhere.
    """

    fixture_origin, live_origin = running_servers
    fixture_session = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / "fixture-mixed.zip")
    )
    session = await browser_adapter.start_session(
        ObservationSessionConfig(
            session_id="live-isolated",
            start_url=f"{live_origin}/page",
            test_account_id=AccountId("test-live-isolated"),
            viewport=ViewportSize(width=1024, height=768),
            trace_path=tmp_path / "live-isolated.zip",
            navigation_origins=(live_origin,),
            fixture_only=False,
        )
    )

    assert browser_adapter.page_for_testing(fixture_session).url.endswith(
        "/app/session-1/improved"
    )
    assert browser_adapter.page_for_testing(session).url == f"{live_origin}/page"
    foreign_navigation = await browser_adapter.execute(
        session, NavigateAction(url=f"{fixture_origin}/page")
    )

    assert foreign_navigation.succeeded
    assert not any(
        event.origin == fixture_origin for event in session.blocked_events
    )


@pytest.mark.asyncio
async def test_live_session_requires_explicit_allowed_origins(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
) -> None:
    fixture_origin, _ = running_servers
    config = ObservationSessionConfig(
        session_id="live-without-origins",
        start_url=f"{fixture_origin}/page",
        test_account_id=AccountId("test-live-without-origins"),
        viewport=ViewportSize(width=1024, height=768),
        trace_path=tmp_path / "live-without-origins.zip",
        fixture_only=False,
    )

    with pytest.raises(ProviderFailure, match="explicit allowed origins"):
        await browser_adapter.start_session(config)


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
async def test_foreign_media_redirect_is_blocked_before_foreign_endpoint_receives_request(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
) -> None:
    fixture_origin, foreign_origin = running_servers
    foreign_hit = _reset_foreign_redirect_media_hit()
    session = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / "foreign-media-redirect.zip")
    )
    page = browser_adapter.page_for_testing(session)
    target = quote(f"{foreign_origin}/redirected-media", safe="")
    await page.set_content(
        f"<img src='{fixture_origin}/__test-redirect?target={target}'>"
    )
    await page.wait_for_timeout(500)

    assert not foreign_hit.is_set()
    assert any(
        event.kind == "redirect"
        and event.origin == foreign_origin
        and event.resource_type == "image"
        for event in session.blocked_events
    )


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
async def test_live_resource_origin_loads_fonts_and_navigation_is_allowed(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
) -> None:
    """Live sessions load resource-origin assets and navigate anywhere."""

    fonts_origin, target_origin = running_servers
    css_hit, font_hit = _reset_font_hits()
    config = ObservationSessionConfig(
        session_id="live-origin-separation",
        start_url=f"{target_origin}/page",
        test_account_id=AccountId("test-live-origin-separation"),
        viewport=ViewportSize(width=1024, height=768),
        trace_path=tmp_path / "live-origin-separation.zip",
        navigation_origins=(target_origin,),
        resource_origins=(fonts_origin,),
        fixture_only=False,
    )
    session = await browser_adapter.start_session(config)
    page = browser_adapter.page_for_testing(session)
    requested_urls: list[str] = []
    failed_requests: list[str | None] = []
    page.on("request", lambda request: requested_urls.append(request.url))
    page.on("requestfailed", lambda request: failed_requests.append(request.failure))
    await page.set_content(
        "<span style='font-family: TestFont'>Font result</span>"
    )
    await page.add_style_tag(url=f"{fonts_origin}/__test-font.css")
    await asyncio.wait_for(css_hit.wait(), timeout=2)
    await asyncio.wait_for(font_hit.wait(), timeout=2)
    assert f"{fonts_origin}/__test-font.css" in requested_urls
    assert f"{fonts_origin}/__test-font.woff2" in requested_urls
    assert failed_requests == []

    await page.set_content(
        f"<a target='_blank' href='{fonts_origin}/app/popup/improved'>Font popup</a>"
    )
    link = page.get_by_role("link", name="Font popup")
    bounds = await link.bounding_box()
    assert bounds is not None
    popup_result = await browser_adapter.execute(
        session,
        ClickAction(
            element_id="font-popup",
            bounds=type(
                "Bounds",
                (),
                {
                    "x": bounds["x"],
                    "y": bounds["y"],
                    "width": bounds["width"],
                    "height": bounds["height"],
                },
            )(),
        ),
    )

    assert popup_result.succeeded
    assert browser_adapter.active_session_count == 1
    foreign_navigation = await browser_adapter.execute(
        session,
        NavigateAction(url=f"{fonts_origin}/app/navigation/improved"),
    )
    assert foreign_navigation.succeeded
    assert not any(
        event.origin == fonts_origin for event in session.blocked_events
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
async def test_click_waits_for_post_action_document_before_capture(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
) -> None:
    fixture_origin, _ = running_servers
    session_id = "post-action-capture"
    session = await browser_adapter.start_session(
        ObservationSessionConfig(
            session_id=session_id,
            start_url=f"{fixture_origin}/app/{session_id}/improved/settings",
            test_account_id=AccountId("test-post-action-capture"),
            viewport=ViewportSize(width=1024, height=768),
            trace_path=tmp_path / "post-action.zip",
        )
    )
    page = browser_adapter.page_for_testing(session)
    await page.set_content(
        "<form method='post' action='/__test-slow-feedback'>"
        "<button type='submit'>Enable two-factor authentication</button>"
        "</form>"
    )
    button = page.get_by_role("button", name="Enable two-factor authentication")
    bounds = await button.bounding_box()
    assert bounds is not None

    await browser_adapter.execute(
        session,
        ClickAction(
            element_id="enable-two-factor",
            bounds=type(
                "Bounds",
                (),
                {
                    "x": bounds["x"],
                    "y": bounds["y"],
                    "width": bounds["width"],
                    "height": bounds["height"],
                },
            )(),
        ),
    )
    captured = await browser_adapter.capture(session)
    snapshot = await capture_snapshot(page, captured.viewport_id)

    assert any(
        "Two-factor authentication enabled" in element.label
        for element in snapshot.elements
    )


@pytest.mark.asyncio
async def test_delayed_click_navigation_is_settled_before_capture(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
) -> None:
    fixture_origin, _ = running_servers
    session = await browser_adapter.start_session(
        ObservationSessionConfig(
            session_id="delayed-navigation",
            start_url=f"{fixture_origin}/app/delayed-navigation/improved",
            test_account_id=AccountId("test-delayed-navigation"),
            viewport=ViewportSize(width=1024, height=768),
            trace_path=tmp_path / "delayed-navigation.zip",
            navigation_origins=(fixture_origin,),
            fixture_only=True,
            navigation_settle_ms=100,
            action_settle_ms=350,
        )
    )
    page = browser_adapter.page_for_testing(session)
    await page.set_content("<button>Continue</button>")
    await page.evaluate(
        """
        () => document.querySelector('button').addEventListener('click', () => {
          setTimeout(() => location.assign('/__test-delayed-navigation'), 180);
        })
        """
    )
    button = page.get_by_role("button", name="Continue")
    bounds = await button.bounding_box()
    assert bounds is not None

    started = monotonic()
    await browser_adapter.execute(
        session,
        ClickAction(
            element_id="continue",
            bounds=type(
                "Bounds",
                (),
                {
                    "x": bounds["x"],
                    "y": bounds["y"],
                    "width": bounds["width"],
                    "height": bounds["height"],
                },
            )(),
        ),
    )
    assert monotonic() - started >= 0.55

    captured = await browser_adapter.capture(session)
    snapshot = await capture_snapshot(page, captured.viewport_id)

    assert captured.url == f"{fixture_origin}/__test-delayed-navigation"
    assert any(
        "Delayed navigation settled." in element.label for element in snapshot.elements
    )


@pytest.mark.asyncio
async def test_client_state_change_uses_action_settle_window(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
) -> None:
    fixture_origin, _ = running_servers
    session = await browser_adapter.start_session(
        ObservationSessionConfig(
            session_id="client-state-change",
            start_url=f"{fixture_origin}/app/client-state-change/improved",
            test_account_id=AccountId("test-client-state-change"),
            viewport=ViewportSize(width=1024, height=768),
            trace_path=tmp_path / "client-state-change.zip",
            action_settle_ms=250,
        )
    )
    page = browser_adapter.page_for_testing(session)
    await page.set_content(
        """
        <button id="change">Change state</button>
        <p id="status">Pending</p>
        """
    )
    await page.evaluate(
        """
        () => document.querySelector('#change').addEventListener('click', () => {
          setTimeout(() => { document.querySelector('#status').textContent = 'Ready'; }, 150);
        })
        """
    )
    button = page.get_by_role("button", name="Change state")
    bounds = await button.bounding_box()
    assert bounds is not None

    await browser_adapter.execute(
        session,
        ClickAction(
            element_id="change-state",
            bounds=type(
                "Bounds",
                (),
                {
                    "x": bounds["x"],
                    "y": bounds["y"],
                    "width": bounds["width"],
                    "height": bounds["height"],
                },
            )(),
        ),
    )

    assert await page.locator("#status").text_content() == "Ready"


@pytest.mark.asyncio
async def test_configured_navigation_and_action_waits_complete_before_recapture(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
) -> None:
    fixture_origin, _ = running_servers
    config = ObservationSessionConfig(
        session_id="settled-session",
        start_url=f"{fixture_origin}/app/settled-session/improved",
        test_account_id=AccountId("test-settled"),
        viewport=ViewportSize(width=1024, height=768),
        trace_path=tmp_path / "settled.zip",
        navigation_settle_ms=80,
        action_settle_ms=60,
    )

    started = monotonic()
    session = await browser_adapter.start_session(config)
    assert monotonic() - started >= 0.06

    started = monotonic()
    await browser_adapter.execute(
        session,
        ScrollAction(direction="down", amount=100),
    )
    assert monotonic() - started >= 0.04

    started = monotonic()
    await browser_adapter.execute(
        session,
        NavigateAction(url=f"{fixture_origin}/page"),
    )
    assert monotonic() - started >= 0.12


@pytest.mark.asyncio
async def test_extraction_result_returns_screenshot_from_geometry_capture(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
) -> None:
    fixture_origin, _ = running_servers
    session = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / "aligned.zip")
    )
    page = browser_adapter.page_for_testing(session)

    result = await capture_with_diagnostics(page, "aligned-viewport")

    assert result.screenshot


@pytest.mark.asyncio
async def test_visible_result_uses_only_rendered_text_inside_viewport(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
) -> None:
    fixture_origin, _ = running_servers
    session = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / "visible-text.zip")
    )
    page = browser_adapter.page_for_testing(session)
    await page.set_content(
        """
        <style>
          .sr-only { position:absolute !important; width:1px !important; height:1px !important; padding:0 !important; margin:-1px !important; overflow:hidden !important; clip:rect(0px, 0px, 0px, 0px) !important; white-space:nowrap !important; border:0 !important; }
          .clip-path-hidden { clip-path:inset(100%); }
          .equivalent-clip-path-hidden { clip-path:inset(60% 10% 40% 10%); }
          .partially-clipped { clip-path:inset(10%); }
          .one-pixel-hidden { position:absolute; width:1px; height:1px; overflow:hidden; }
          .transparent-text { color:transparent; }
          .zero-font { font-size:0; }
        </style>
        <nav aria-label="Primary secret"><h2 style="position:absolute;top:2000px">Offscreen heading</h2><a>Navigation item</a></nav>
        <label for="visible-input">Visible input label</label>
        <input id="visible-input">
        <label for="hidden-input" class="sr-only">Hidden input label</label>
        <input id="hidden-input" aria-label="Semantic hidden input">
        <button aria-label="ARIA-only result"></button>
        <button id="hidden-result" aria-label="Hidden result"><span class="sr-only">Hidden result</span></button>
        <button aria-label="Clip path result"><span class="clip-path-hidden" style="clip-path:inset(100%)">Clip path result</span></button>
        <button aria-label="Equivalent clip path result"><span class="equivalent-clip-path-hidden" style="clip-path:inset(60% 10% 40% 10%)">Equivalent clip path result</span></button>
        <button class="partially-clipped" style="clip-path:inset(10%)">Partially clipped visible result</button>
        <span class="one-pixel-hidden" style="position:absolute;width:1px;height:1px;overflow:hidden">One pixel result</span>
        <span class="transparent-text" style="color:transparent">Transparent result</span>
        <span class="zero-font" style="font-size:0">Zero font result</span>
        <div id="visible-container">
          Visible container
          <span id="offscreen-descendant">Offscreen descendant result</span>
        </div>
        <div id="offscreen-container">
          <button>Offscreen result</button>
        </div>
        <p>Visible result</p>
        <p>Nearby visible text remains</p>
        """
    )
    await page.evaluate(
        """
        () => {
          const container = document.querySelector('#visible-container');
          const descendant = document.querySelector('#offscreen-descendant');
          const offscreen = document.querySelector('#offscreen-container');
          document.querySelectorAll('.sr-only').forEach((node) => {
            node.style.cssText = 'position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0px,0px,0px,0px);white-space:nowrap;border:0';
          });
          document.querySelector('.clip-path-hidden').style.cssText = 'clip-path:inset(100%)';
          document.querySelector('.equivalent-clip-path-hidden').style.cssText = 'clip-path:inset(60% 10% 40% 10%)';
          document.querySelector('.partially-clipped').style.cssText = 'clip-path:inset(10%)';
          document.querySelector('.one-pixel-hidden').style.cssText = 'position:absolute;width:1px;height:1px;overflow:hidden';
          document.querySelector('.transparent-text').style.cssText = 'color:transparent';
          document.querySelector('.zero-font').style.cssText = 'font-size:0';
          document.querySelector('nav h2').style.cssText = 'position:absolute;top:2000px;left:0';
          container.style.cssText = 'position:relative;width:300px;height:40px;overflow:hidden';
          descendant.style.cssText = 'position:absolute;top:2000px;left:0';
          offscreen.style.cssText = 'position:absolute;top:2000px;left:0';
        }
        """
    )

    extracted = await capture_with_diagnostics(page, "visible-text-viewport")
    by_label = {element.label: element for element in extracted.snapshot.elements}

    assert by_label["ARIA-only result"].rendered_text == ""
    assert by_label["Hidden result"].rendered_text == ""
    visible_input = next(
        element for element in extracted.snapshot.elements if element.label == "Visible input label"
    )
    hidden_input = next(
        element for element in extracted.snapshot.elements if element.label == "Semantic hidden input"
    )
    assert visible_input.rendered_text == "Visible input label"
    assert hidden_input.rendered_text == ""
    assert by_label["Visible result"].rendered_text == "Visible result"
    serialized_elements = json.dumps(
        [element.rendered_text for element in extracted.snapshot.elements]
    )
    for hidden_text in (
        "Hidden result",
        "Clip path result",
        "Equivalent clip path result",
        "One pixel result",
        "Transparent result",
        "Zero font result",
    ):
        assert hidden_text not in serialized_elements, hidden_text
    assert "Nearby visible text remains" in serialized_elements
    assert "Partially clipped visible result" in serialized_elements
    nav_region = next(
        region for region in extracted.snapshot.regions if region.label == "Primary secret"
    )
    assert nav_region.rendered_label == "Navigation"
    assert "Offscreen descendant result" not in by_label
    assert "Offscreen result" not in by_label

    # These tests exercise the viewport-snapshot path in isolation. The real
    # WebSession.capture populates page_text from document.body.innerText,
    # which would let the page-text fallback match hidden text below. Setting
    # page_text=None disables the fallback so the strict path is what is
    # actually under test here.
    captured = replace(
        await browser_adapter.capture(session),
        snapshot=extracted.snapshot,
        page_text=None,
    )

    class SnapshotProvider:
        async def capture(self, _session: object):
            return captured

    def snapshot_extractor(capture: object):
        snapshot = getattr(capture, "snapshot")
        if snapshot is None:
            raise AssertionError("missing extracted snapshot")
        return snapshot

    provider = SnapshotProvider()
    for text, expected in (
        ("ARIA-only result", False),
        ("Hidden result", False),
        ("Clip path result", False),
        ("Equivalent clip path result", False),
        ("One pixel result", False),
        ("Transparent result", False),
        ("Zero font result", False),
        ("Offscreen result", False),
        ("Visible result", True),
        ("Nearby visible text remains", True),
        ("Partially clipped visible result", True),
    ):
        verifier = WebVerifier(
            VisibleResultVerifierSpec(type="visible-result", text=text),
            observation_provider=provider,  # type: ignore[arg-type]
            snapshot_extractor=snapshot_extractor,  # type: ignore[arg-type]
        )
        result = await verifier.verify(session)
        assert result.verified is expected

    hidden_element = next(
        element for element in extracted.snapshot.elements if element.label == "Hidden result"
    )
    clipped_element = next(
        element
        for element in extracted.snapshot.elements
        if element.label == "Clip path result"
    )
    visible_element = next(
        element
        for element in extracted.snapshot.elements
        if element.label == "Nearby visible text remains"
    )
    partially_clipped_element = next(
        element
        for element in extracted.snapshot.elements
        if element.label == "Partially clipped visible result"
    )
    observation = ProgressiveObservation.from_snapshot(
        extracted.snapshot,
        newly_revealed_ids=(
            hidden_element.id,
            clipped_element.id,
            visible_element.id,
            partially_clipped_element.id,
        ),
        region_id=nav_region.id,
    )

    class CognitiveClient:
        endpoint_origin = "https://llm.example.test"

        def __init__(self) -> None:
            self.payload = ""

        async def complete(self, schema, messages, model, role):
            del model, role
            self.payload = messages[-1].content
            return schema.model_validate({"action": "wait", "reason": "Observe."})

    cognitive_client = CognitiveClient()
    await StructuredCognitiveAgent(
        cognitive_client, model="cognitive-model"
    ).decide("Find visible result", observation)
    assert "Hidden result" not in cognitive_client.payload
    assert "Clip path result" not in cognitive_client.payload
    assert "Equivalent clip path result" not in cognitive_client.payload
    assert "Nearby visible text remains" in cognitive_client.payload
    assert "Partially clipped visible result" in cognitive_client.payload
    assert '"region_label":"Navigation"' in cognitive_client.payload
    assert "Primary secret" not in cognitive_client.payload


@pytest.mark.asyncio
async def test_visible_result_requires_all_of_in_effective_visible_elements(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
) -> None:
    fixture_origin, _ = running_servers
    session = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / "all-of-text.zip")
    )
    page = browser_adapter.page_for_testing(session)
    await page.set_content(
        """
        <button>Frontend Engineer</button>
        <p>PAIR Systems</p>
        <p>React</p>
        <p>TypeScript</p>
        <p id="offscreen">offscreen-only</p>
        """
    )
    await page.evaluate(
        """
        () => {
          document.querySelector('#offscreen').style.cssText =
            'position:absolute;top:2000px;left:0';
        }
        """
    )

    extracted = await capture_with_diagnostics(page, "all-of-viewport")
    # See test_visible_result_uses_only_rendered_text_inside_viewport: isolate
    # the viewport-snapshot path by disabling the page-text fallback.
    captured = replace(
        await browser_adapter.capture(session),
        snapshot=extracted.snapshot,
        page_text=None,
    )

    class SnapshotProvider:
        async def capture(self, _session: object):
            return captured

    provider = SnapshotProvider()
    verifier = WebVerifier(
        VisibleResultVerifierSpec(
            type="visible-result",
            text="Frontend Engineer",
            role="button",
            all_of=("PAIR Systems", "React", "TypeScript"),
        ),
        observation_provider=provider,  # type: ignore[arg-type]
        snapshot_extractor=lambda capture: capture.snapshot,  # type: ignore[union-attr]
    )

    result = await verifier.verify(session)

    assert result.verified

    missing_visible_text = WebVerifier(
        VisibleResultVerifierSpec(
            type="visible-result",
            text="Frontend Engineer",
            role="button",
            all_of=("PAIR Systems", "offscreen-only"),
        ),
        observation_provider=provider,  # type: ignore[arg-type]
        snapshot_extractor=lambda capture: capture.snapshot,  # type: ignore[union-attr]
    )

    assert not (await missing_visible_text.verify(session)).verified


@pytest.mark.asyncio
async def test_visible_result_rejects_fully_occluded_rendered_text(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
) -> None:
    fixture_origin, _ = running_servers
    session = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / "occluded-text.zip")
    )
    page = browser_adapter.page_for_testing(session)
    await page.set_content(
        """
        <button id="covered">Covered result</button>
        <div id="overlay">Blocking overlay</div>
        """
    )
    await page.evaluate(
        """
        () => {
          const button = document.querySelector('#covered');
          const overlay = document.querySelector('#overlay');
          button.style.cssText = 'position:fixed;top:100px;left:100px;z-index:1';
          overlay.style.cssText = 'position:fixed;inset:0;z-index:2;background:white';
        }
        """
    )

    extracted = await capture_with_diagnostics(page, "occluded-text-viewport")
    covered = next(
        element
        for element in extracted.snapshot.elements
        if element.label == "Covered result"
    )
    assert covered.rendered_text == "Covered result"
    assert covered.visibility_fraction == 0

    # Isolate the viewport-snapshot path; see
    # test_visible_result_uses_only_rendered_text_inside_viewport.
    captured = replace(
        await browser_adapter.capture(session),
        snapshot=extracted.snapshot,
        page_text=None,
    )

    class SnapshotProvider:
        async def capture(self, _session: object):
            return captured

    verifier = WebVerifier(
        VisibleResultVerifierSpec(type="visible-result", text="Covered result"),
        observation_provider=SnapshotProvider(),  # type: ignore[arg-type]
        snapshot_extractor=lambda capture: capture.snapshot,  # type: ignore[union-attr]
    )

    result = await verifier.verify(session)

    assert not result.verified


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


@pytest.mark.asyncio
async def test_trace_sanitization_failure_propagates_after_context_cleanup(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_origin, _ = running_servers
    session = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / "sanitize-failure.zip")
    )

    page = browser_adapter.page_for_testing(session)
    raw_path = browser_adapter._sessions[session.session_id].raw_trace_path

    def fail_sanitization(*_args: object) -> bytes:
        assert page.is_closed()
        raise RuntimeError("trace sanitization failed")

    monkeypatch.setattr(
        "ux_analyzer.adapters.web.session.sanitize_artifact_content",
        fail_sanitization,
    )

    with pytest.raises(RuntimeError, match="trace sanitization failed"):
        await browser_adapter.end_session(session)

    assert browser_adapter.active_session_count == 0
    assert not session.trace_path.exists()
    assert not raw_path.exists()
    assert not session.trace_path.with_name(f".{session.trace_path.name}.sanitized").exists()


@pytest.mark.asyncio
async def test_trace_finalization_uses_private_raw_archive_and_publishes_sanitized_trace(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_origin, _ = running_servers
    sensitive = "invitee@example.test"
    config = replace(
        _session_config(fixture_origin, tmp_path / "private-raw.zip"),
        artifact_redaction=RedactionPolicy(exact_values=(sensitive,)),
    )
    session = await browser_adapter.start_session(config)
    managed = browser_adapter._sessions[session.session_id]
    raw_path: Path | None = None

    async def write_raw_trace(*, path: str) -> None:
        nonlocal raw_path
        raw_path = Path(path)
        with zipfile.ZipFile(raw_path, "w") as archive:
            archive.writestr("trace.network", f"email = '{sensitive}'")

    monkeypatch.setattr(managed.context.tracing, "stop", write_raw_trace)

    await browser_adapter.end_session(session)

    assert raw_path is not None
    assert raw_path != session.trace_path
    assert raw_path.parent == session.trace_path.parent
    assert raw_path.name == f".{session.trace_path.name}.raw"
    assert not raw_path.exists()
    with zipfile.ZipFile(session.trace_path) as archive:
        sanitized_member = archive.read("trace.network")
    assert sensitive.encode("utf-8") not in sanitized_member
    assert b"[REDACTED]" in sanitized_member


@pytest.mark.asyncio
async def test_trace_finalization_unlinks_hostile_staging_links_without_touching_targets(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_origin, _ = running_servers
    session = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / "hostile-staging.zip")
    )
    managed = browser_adapter._sessions[session.session_id]
    raw_target = tmp_path / "outside-raw.txt"
    sanitized_target = tmp_path / "outside-sanitized.txt"
    raw_target.write_bytes(b"outside raw sentinel")
    sanitized_target.write_bytes(b"outside sanitized sentinel")
    raw_link = managed.raw_trace_path
    sanitized_link = session.trace_path.with_name(
        f".{session.trace_path.name}.sanitized"
    )
    _symlink_or_skip(raw_link, raw_target)
    _symlink_or_skip(sanitized_link, sanitized_target)

    async def write_raw_trace(*, path: str) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("trace.network", "safe trace")

    monkeypatch.setattr(managed.context.tracing, "stop", write_raw_trace)

    await browser_adapter.end_session(session)

    assert raw_target.read_bytes() == b"outside raw sentinel"
    assert sanitized_target.read_bytes() == b"outside sanitized sentinel"
    assert not os.path.lexists(raw_link)
    assert not os.path.lexists(sanitized_link)
    assert session.trace_path.is_file()
    assert not session.trace_path.is_symlink()


@pytest.mark.asyncio
async def test_trace_replace_retries_transient_sharing_violation_after_context_cleanup(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_origin, _ = running_servers
    session = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / "transient-replace.zip")
    )
    page = browser_adapter.page_for_testing(session)
    real_replace = web_session.secure_replace_exclusive_file
    attempts = 0

    def replace_with_transient_failure(*args: object, **kwargs: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise BundleStateError(
                "atomic trace publication failed with NTSTATUS 0xc0000043"
            )
        assert page.is_closed()
        real_replace(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        web_session,
        "secure_replace_exclusive_file",
        replace_with_transient_failure,
    )

    await browser_adapter.end_session(session)

    assert attempts == 2
    assert session.trace_path.is_file()


@pytest.mark.asyncio
async def test_persistent_trace_replace_failure_discards_trace_and_propagates(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_origin, _ = running_servers
    session = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / "persistent-replace.zip")
    )
    attempts = 0
    raw_path = browser_adapter._sessions[session.session_id].raw_trace_path

    def always_fail_replace(*_args: object, **_kwargs: object) -> None:
        nonlocal attempts
        attempts += 1
        raise BundleStateError(
            "atomic trace publication failed with NTSTATUS 0xc0000043"
        )

    monkeypatch.setattr(
        web_session,
        "secure_replace_exclusive_file",
        always_fail_replace,
    )

    with pytest.raises(BundleStateError, match="0xc0000043"):
        await browser_adapter.end_session(session)

    assert attempts == 3
    assert browser_adapter.active_session_count == 0
    assert not session.trace_path.exists()
    assert not session.trace_path.with_name(f".{session.trace_path.name}.sanitized").exists()
    assert not raw_path.exists()


@pytest.mark.asyncio
async def test_trace_stop_failure_removes_partially_written_sensitive_bytes(
    browser_adapter: Any,
    running_servers: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_origin, _ = running_servers
    sensitive = b"invitee@example.test:246810"
    session = await browser_adapter.start_session(
        _session_config(fixture_origin, tmp_path / "partial-sensitive.zip")
    )
    page = browser_adapter.page_for_testing(session)
    managed = browser_adapter._sessions[session.session_id]
    raw_path = managed.raw_trace_path

    async def fail_after_partial_write(*, path: str) -> None:
        Path(path).write_bytes(sensitive)
        raise PlaywrightError("trace stop failed")

    monkeypatch.setattr(managed.context.tracing, "stop", fail_after_partial_write)

    await browser_adapter.end_session(session)

    assert browser_adapter.active_session_count == 0
    assert page.is_closed()
    assert not raw_path.exists()
    artifact_bytes = b"".join(
        path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    )
    assert sensitive not in artifact_bytes
