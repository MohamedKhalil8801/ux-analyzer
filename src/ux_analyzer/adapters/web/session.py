"""Playwright session adapter with fail-closed browser safety controls."""

from __future__ import annotations

import asyncio
import gc
import os
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic, sleep
from typing import Any, cast
from uuid import uuid4

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Request,
    Route,
    WebSocketRoute,
)
from playwright.async_api import Error as PlaywrightError

from ux_analyzer.adapters.web.network_policy import (
    BrowserAllowedOrigins,
    NetworkPolicy,
    abort_route,
)
from ux_analyzer.analysis.page_capture import TAP_INSTRUMENTATION_JS
from ux_analyzer.ports.artifacts import BundleStateError, sanitize_artifact_content
from ux_analyzer.ports.observation import (
    BackAction,
    BlockedRequest,
    ClearTextAction,
    ClickAction,
    DoubleClickAction,
    NavigateAction,
    ObservationCapture,
    ObservationProviderError,
    ObservationSessionConfig,
    OpenMenuAction,
    PlatformAction,
    PlatformActionResult,
    PressKeyAction,
    SafetyBlocked,
    ScrollAction,
    SelectOptionAction,
    SessionHandle,
    SubmitAction,
    ToggleAction,
    TypeTextAction,
    WaitAction,
)
from ux_analyzer.storage.run_bundle import (
    SecureDirectoryHandle,
    SecureExclusiveFile,
    secure_create_exclusive_file,
    secure_open_directory,
    secure_replace_exclusive_file,
    secure_unlink,
)

__all__ = [
    "BrowserAllowedOrigins",
    "PlaywrightSessionAdapter",
    "ProviderFailure",
    "SafetyBlocked",
]


class ProviderFailure(ObservationProviderError):
    """Raised after a non-safety provider error closes its session."""


_NAVIGATION_START_GRACE_SECONDS = 0.1
_TRANSIENT_POPUP_URLS = frozenset(("", ":", "about:blank"))
_TRACE_REPLACE_ATTEMPTS = 3
_TRACE_REPLACE_DELAY_SECONDS = 0.01
# Playwright fires route-interception updates and route fulfil/continue calls
# as unowned tasks. Closing a context while one is in flight leaves a failed
# task that nothing retrieves, so the event loop reports a spurious
# "Task exception was never retrieved" during teardown.
_TEARDOWN_TASK_SETTLE_SECONDS = 1.0
_TEARDOWN_TASK_POLL_SECONDS = 0.01
# Playwright's connection reader loop runs for the whole browser lifetime.
_PLAYWRIGHT_SESSION_TASKS = frozenset({"Connection.run"})
_RECOVERABLE_USER_ACTIONS = (
    ClickAction,
    DoubleClickAction,
    TypeTextAction,
    SelectOptionAction,
    ToggleAction,
    SubmitAction,
    OpenMenuAction,
    ClearTextAction,
)


def _empty_task_set() -> set[asyncio.Task[None]]:
    return set()


@dataclass(slots=True)
class _ManagedSession:
    config: ObservationSessionConfig
    handle: SessionHandle
    context: BrowserContext
    page: Page
    policy: NetworkPolicy
    raw_trace_path: Path
    trace_started: bool = False
    capture_index: int = 0
    cleanup_started: bool = False
    popup_listener: Callable[[Page], None] | None = None
    route_handler: Callable[[Route, Request], Awaitable[None]] | None = None
    websocket_handler: Callable[[WebSocketRoute], Awaitable[None]] | None = None
    cleanup_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cleanup_complete: asyncio.Event = field(default_factory=asyncio.Event)
    cleanup_error: BaseException | None = None
    route_tasks: set[asyncio.Task[None]] = field(default_factory=_empty_task_set)
    popup_tasks: set[asyncio.Task[None]] = field(default_factory=_empty_task_set)


class PlaywrightSessionAdapter:
    """Safe web observation provider backed by one isolated context per run."""

    id = "playwright-web"
    platform = "web"

    def __init__(
        self,
        *,
        browser: Browser,
        allowed_origins: BrowserAllowedOrigins | Iterable[str] | None,
        trace_directory: Path,
    ) -> None:
        self._browser = browser
        if isinstance(allowed_origins, BrowserAllowedOrigins):
            self._default_allowed_origins = allowed_origins
        elif allowed_origins is None:
            self._default_allowed_origins = None
        else:
            self._default_allowed_origins = BrowserAllowedOrigins.fixture_only(
                allowed_origins
            )
        self._trace_directory = trace_directory
        self._sessions: dict[str, _ManagedSession] = {}

    @property
    def active_session_count(self) -> int:
        return len(self._sessions)

    async def start_session(self, config: ObservationSessionConfig) -> SessionHandle:
        allowed_origins = self._allowed_origins_for(config)
        allowed_origins.require_allowed(
            config.start_url, resource_type="document", kind="navigation"
        )
        if config.session_id in self._sessions:
            raise ProviderFailure("session ID already active")
        with secure_open_directory(
            config.trace_path.parent,
            "trace directory",
            create=True,
        ):
            pass
        raw_trace_path = _raw_trace_path(config.trace_path)
        _discard_trace(
            config.trace_path,
            raw_trace_path,
        )
        policy = NetworkPolicy(allowed_origins)
        context: BrowserContext | None = None
        managed: _ManagedSession | None = None
        try:
            context = await self._browser.new_context(
                viewport={
                    "width": config.viewport.width,
                    "height": config.viewport.height,
                },
                accept_downloads=False,
                permissions=[],
                service_workers="block",
            )
            await context.add_init_script(
                """
                (() => {
                  const denied = () => {
                    const error = new DOMException('Permission denied', 'NotAllowedError');
                    return Promise.reject(error);
                  };
                  Object.defineProperty(navigator, 'geolocation', {
                    configurable: false,
                    value: {
                      getCurrentPosition: (_success, failure) => {
                        if (typeof failure === 'function') {
                          failure({code: 1, message: 'Permission denied'});
                        }
                      },
                      watchPosition: (_success, failure) => {
                        if (typeof failure === 'function') {
                          failure({code: 1, message: 'Permission denied'});
                        }
                        return 0;
                      },
                      clearWatch: () => {}
                    }
                  });
                  if (typeof Notification !== 'undefined') {
                    Object.defineProperty(Notification, 'permission', {
                      configurable: false,
                      value: 'denied'
                    });
                    Notification.requestPermission = () => Promise.resolve('denied');
                  }
                  if (navigator.clipboard) {
                    navigator.clipboard.writeText = denied;
                    navigator.clipboard.readText = denied;
                  }
                })();
                """
            )
            await context.add_init_script(TAP_INSTRUMENTATION_JS)
            await context.tracing.start(
                screenshots=config.trace_screencast,
                snapshots=True,
                sources=False,
            )
            page = await context.new_page()
            account_id = config.test_account_id
            if isinstance(account_id, str):
                raise ProviderFailure("session requires test account ID")
            handle = SessionHandle(
                session_id=config.session_id,
                test_account_id=account_id,
                viewport=config.viewport,
                trace_path=config.trace_path,
                blocked_events=policy.blocked_requests,
            )
            managed = _ManagedSession(
                config=config,
                handle=handle,
                context=context,
                page=page,
                policy=policy,
                raw_trace_path=raw_trace_path,
                trace_started=True,
            )
            assert managed is not None
            if handle.session_id in self._sessions:
                raise ProviderFailure("session ID already active")
            self._sessions[handle.session_id] = managed

            def handle_popup(popup: Page) -> None:
                self._schedule_popup_cleanup(managed, popup)

            managed.popup_listener = handle_popup
            context.on("page", handle_popup)

            async def handle_websocket(websocket: WebSocketRoute) -> None:
                if managed.cleanup_started:
                    await websocket.close(
                        code=1008,
                        reason="session cleanup in progress",
                    )
                    return
                await _track_task(
                    managed.route_tasks,
                    policy.handle_websocket(websocket),
                )

            async def handle_route(route: Route, request: Request) -> None:
                if managed.cleanup_started:
                    await abort_route(route, error_code="blockedbyclient")
                    return
                await _track_task(
                    managed.route_tasks,
                    policy.handle_route(
                        route,
                        request,
                        kind=_request_kind(request, managed.page),
                    ),
                )

            managed.websocket_handler = handle_websocket
            managed.route_handler = handle_route
            await context.route_web_socket(
                "**/*", handle_websocket
            )
            await context.route("**/*", handle_route)
            await self._navigate(managed, config.start_url)
            return handle
        except asyncio.CancelledError as cancellation:
            try:
                if managed is not None:
                    await _shielded_cleanup(self._cleanup(managed))
                elif context is not None:
                    await _shielded_cleanup(context.close())
            except BaseException as cleanup_error:
                raise cancellation from cleanup_error
            raise
        except BaseException:
            if managed is not None:
                await self._cleanup(managed)
            elif context is not None:
                await context.close()
            raise

    async def capture(self, session: SessionHandle) -> ObservationCapture:
        managed = self._active(session)
        try:
            screenshot = await managed.page.screenshot(type="png")
            managed.capture_index += 1
            # ``document.body.innerText`` returns the rendered text as the user
            # would see it if they scrolled, which is exactly the "page text"
            # the verifier needs when the visible-result text lives below the
            # current viewport. ``display: none`` content is excluded, but
            # off-screen content is included. The evaluate is wrapped so a
            # navigation race or page eval failure cannot break capture — the
            # viewport snapshot alone remains authoritative.
            page_text: str | None
            try:
                page_text = await managed.page.evaluate(
                    "() => document.body && document.body.innerText"
                )
            except BaseException:  # noqa: BLE001 - capture must not fail
                page_text = None
            # Perceivable colour signals for opt-in state-change verification.
            # The document/body computed background is what a sighted user sees
            # as the page "theme"; the viewport background is the sampled
            # dominant colour of the current screen. Wrapped so a page eval
            # failure never breaks capture.
            colours: Mapping[str, object] | None
            try:
                colours = await managed.page.evaluate(
                    """() => {
                        const read = (el) => {
                            if (!el) return null;
                            const bg = getComputedStyle(el).backgroundColor;
                            return bg && bg !== 'rgba(0, 0, 0, 0)' ? bg : null;
                        };
                        return {
                            document_background: read(document.documentElement),
                            body_background: read(document.body),
                        };
                    }"""
                )
            except BaseException:  # noqa: BLE001 - capture must not fail
                colours = None
            return ObservationCapture(
                session_id=session.session_id,
                viewport_id=f"{session.session_id}-viewport-{managed.capture_index}",
                url=managed.page.url,
                title=await managed.page.title(),
                viewport=session.viewport,
                screenshot=screenshot,
                page_text=page_text if isinstance(page_text, str) else None,
                document_background=_optional_colour(colours, "document_background"),
                body_background=_optional_colour(colours, "body_background"),
                viewport_background=await _viewport_background(managed.page),
            )
        except asyncio.CancelledError as cancellation:
            try:
                await _shielded_cleanup(self._cleanup(managed))
            except BaseException as cleanup_error:
                raise cancellation from cleanup_error
            raise
        except BaseException as error:
            await self._cleanup(managed)
            if isinstance(error, SafetyBlocked):
                raise
            raise ProviderFailure("browser capture failed") from error

    async def execute(
        self, session: SessionHandle, action: PlatformAction
    ) -> PlatformActionResult:
        managed = self._active(session)
        started = monotonic()
        initial_url = managed.page.url
        blocked_before = len(managed.policy.blocked_requests)
        navigation_started = asyncio.Event()

        def record_navigation(request: Request) -> None:
            if (
                request.is_navigation_request()
                and request.frame == managed.page.main_frame
            ):
                navigation_started.set()

        managed.page.on("request", record_navigation)
        try:
            navigation_occurred = False
            if isinstance(action, NavigateAction):
                await self._navigate(managed, action.url, settle=False)
                navigation_occurred = True
            elif isinstance(action, BackAction):
                await managed.page.go_back(wait_until="domcontentloaded")
                navigation_occurred = True
            elif isinstance(action, ScrollAction):
                amount = _scroll_amount(action)
                if action.direction == "up":
                    amount = -amount
                await managed.page.evaluate(
                    "amount => window.scrollBy(0, amount)", amount
                )
            elif isinstance(action, WaitAction):
                await managed.page.wait_for_timeout(action.milliseconds)
            elif isinstance(action, PressKeyAction):
                await managed.page.keyboard.press(action.key)
            elif isinstance(action, ClickAction):
                await self._click(managed.page, action.element_id, action.bounds)
            elif isinstance(action, DoubleClickAction):
                await self._click(
                    managed.page, action.element_id, action.bounds, double=True
                )
            elif isinstance(action, TypeTextAction):
                await self._click(managed.page, action.element_id, action.bounds)
                await managed.page.keyboard.press("ControlOrMeta+A")
                await managed.page.keyboard.press("Backspace")
                await managed.page.keyboard.type(action.text)
            elif isinstance(action, ClearTextAction):
                await self._click(managed.page, action.element_id, action.bounds)
                await managed.page.keyboard.press("ControlOrMeta+A")
                await managed.page.keyboard.press("Backspace")
            elif isinstance(action, SelectOptionAction):
                await self._click(managed.page, action.element_id, action.bounds)
                await managed.page.keyboard.type(action.option)
                await managed.page.keyboard.press("Enter")
            elif isinstance(action, (ToggleAction, SubmitAction, OpenMenuAction)):
                await self._click(managed.page, action.element_id, action.bounds)
            else:
                if action.bounds is None:
                    raise ProviderFailure("drag action requires target bounds")
                start_x, start_y = _center(action.bounds)
                await managed.page.mouse.move(start_x, start_y)
                await managed.page.mouse.down()
                await managed.page.mouse.move(action.end_x, action.end_y)
                await managed.page.mouse.up()
            await _settle_after_action(
                managed.page,
                navigation_started,
                managed.config.navigation_settle_ms,
                managed.config.action_settle_ms,
            )
            await _await_popup_tasks(managed.popup_tasks)
            blocked_navigation = _blocked_document_navigation(
                managed.policy.blocked_requests, blocked_before
            )
            if blocked_navigation is not None:
                if _is_recoverable_user_interaction(action):
                    return _blocked_action_result(managed, started)
                raise SafetyBlocked(
                    "browser action blocked for foreign origin "
                    f"{blocked_navigation.origin!r}"
                )
            return PlatformActionResult(
                succeeded=True,
                url=managed.page.url,
                duration_ms=_elapsed_ms(started),
                navigation_occurred=(
                    navigation_occurred or managed.page.url != initial_url
                ),
                state_changed=not isinstance(action, (WaitAction, PressKeyAction)),
            )
        except SafetyBlocked:
            await self._cleanup(managed)
            raise
        except asyncio.CancelledError as cancellation:
            try:
                await _shielded_cleanup(self._cleanup(managed))
            except BaseException as cleanup_error:
                raise cancellation from cleanup_error
            raise
        except BaseException as error:
            blocked_navigation = _blocked_document_navigation(
                managed.policy.blocked_requests, blocked_before
            )
            if blocked_navigation is not None and _is_recoverable_user_interaction(
                action
            ):
                return _blocked_action_result(managed, started)
            await self._cleanup(managed)
            if blocked_navigation is not None:
                raise SafetyBlocked(
                    "browser action blocked for foreign origin "
                    f"{blocked_navigation.origin!r}"
                ) from error
            if isinstance(error, ProviderFailure):
                raise
            raise ProviderFailure("browser action failed") from error
        finally:
            managed.page.remove_listener("request", record_navigation)

    async def reset(self, session: SessionHandle) -> None:
        managed = self._active(session)
        try:
            await managed.context.clear_cookies()
            await self._navigate(managed, managed.config.start_url)
        except SafetyBlocked:
            await self._cleanup(managed)
            raise
        except asyncio.CancelledError as cancellation:
            try:
                await _shielded_cleanup(self._cleanup(managed))
            except BaseException as cleanup_error:
                raise cancellation from cleanup_error
            raise
        except BaseException as error:
            await self._cleanup(managed)
            raise ProviderFailure("browser reset failed") from error

    async def end_session(self, session: SessionHandle) -> None:
        managed = self._sessions.get(session.session_id)
        if managed is not None:
            await self._cleanup(managed)

    async def close(self) -> None:
        for managed in tuple(self._sessions.values()):
            await self._cleanup(managed)

    def page_for_testing(self, session: SessionHandle) -> Page:
        """Expose page only to adapter integration tests."""

        return self._active(session).page

    def _active(self, session: SessionHandle) -> _ManagedSession:
        managed = self._sessions.get(session.session_id)
        if managed is None or managed.cleanup_started:
            raise ProviderFailure("session is not active")
        return managed

    def _allowed_origins_for(
        self, config: ObservationSessionConfig
    ) -> BrowserAllowedOrigins:
        if not config.fixture_only and not config.navigation_origins:
            raise ProviderFailure("live session requires explicit allowed origins")
        if config.navigation_origins or config.resource_origins:
            if config.fixture_only:
                return BrowserAllowedOrigins.fixture_only(config.navigation_origins)
            return BrowserAllowedOrigins.configured(
                config.navigation_origins, config.resource_origins
            )
        if self._default_allowed_origins is None:
            raise ProviderFailure("session requires explicit allowed origins")
        return self._default_allowed_origins

    async def _navigate(
        self, managed: _ManagedSession, url: str, *, settle: bool = True
    ) -> None:
        managed.policy.check(url, resource_type="document", kind="navigation")
        blocked_before = len(managed.policy.blocked_requests)
        try:
            await managed.page.goto(url, wait_until="domcontentloaded")
        except PlaywrightError as error:
            new_blocks = managed.policy.blocked_requests[blocked_before:]
            if new_blocks:
                raise SafetyBlocked(
                    f"browser navigation blocked for foreign origin "
                    f"{new_blocks[-1].origin!r}"
                ) from error
            raise ProviderFailure("browser navigation failed") from error
        new_blocks = managed.policy.blocked_requests[blocked_before:]
        if new_blocks:
            raise SafetyBlocked(
                f"browser navigation blocked for foreign origin "
                f"{new_blocks[-1].origin!r}"
            )
        if settle:
            await _wait_for_navigation_settle(
                managed.page, managed.config.navigation_settle_ms
            )

    async def _close_foreign_popup(self, managed: _ManagedSession, popup: Page) -> None:
        try:
            await popup.wait_for_load_state("domcontentloaded", timeout=500)
        except PlaywrightError:
            pass
        try:
            popup_url = popup.url
            if _is_transient_popup_url(popup_url):
                await popup.close()
            elif not managed.policy.allowed_origins.allows(popup_url, kind="popup"):
                managed.policy.record_popup(popup_url)
                await popup.close()
        except PlaywrightError:
            pass

    def _schedule_popup_cleanup(self, managed: _ManagedSession, popup: Page) -> None:
        _track_task(
            managed.popup_tasks,
            self._close_foreign_popup(managed, popup),
        )

    @staticmethod
    async def _click(
        page: Page,
        element_id: str,
        bounds: object,
        *,
        double: bool = False,
    ) -> None:
        if bounds is None:
            raise ProviderFailure(
                f"element action requires private execution bounds: {element_id}"
            )
        x, y = _center(bounds)
        if double:
            await page.mouse.dblclick(x, y)
        else:
            await page.mouse.click(x, y)

    async def _cleanup(self, managed: _ManagedSession) -> None:
        cleanup_owner = False
        wait_for_cleanup = False
        cleanup_error: BaseException | None = None
        async with managed.cleanup_lock:
            if managed.cleanup_complete.is_set():
                cleanup_error = managed.cleanup_error
            elif managed.cleanup_started:
                wait_for_cleanup = True
            else:
                managed.cleanup_started = True
                cleanup_owner = True

        if not cleanup_owner:
            if wait_for_cleanup:
                await managed.cleanup_complete.wait()
                cleanup_error = managed.cleanup_error
            if cleanup_error is not None:
                raise cleanup_error
            return

        trace_resources = ExitStack()
        trace_parent: SecureDirectoryHandle | None = None
        raw_trace: SecureExclusiveFile | None = None
        try:
            try:
                listener = managed.popup_listener
                managed.popup_listener = None
                if listener is not None:
                    managed.context.remove_listener("page", listener)
            except BaseException as error:
                cleanup_error = error

            try:
                await managed.context.unroute_all(behavior="ignoreErrors")
                await _cancel_and_gather_tasks(managed.route_tasks)
                await _cancel_and_gather_tasks(managed.popup_tasks)
                if managed.trace_started:
                    trace_parent = trace_resources.enter_context(
                        secure_open_directory(
                            managed.handle.trace_path.parent,
                            "trace directory",
                            create=False,
                        )
                    )
                    _discard_file(managed.raw_trace_path)
                    raw_trace = trace_resources.enter_context(
                        secure_create_exclusive_file(
                            trace_parent,
                            managed.raw_trace_path.name,
                            "raw trace archive",
                            readable=True,
                        )
                    )
                    await managed.context.tracing.stop(path=str(managed.raw_trace_path))
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error

            try:
                await managed.context.close()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error

            if cleanup_error is not None:
                _discard_trace(managed.handle.trace_path, managed.raw_trace_path)
                if isinstance(cleanup_error, PlaywrightError):
                    return
                raise cleanup_error

            if (
                managed.trace_started
                and trace_parent is not None
                and raw_trace is not None
            ):
                try:
                    _sanitize_trace(managed, trace_parent, raw_trace)
                except PlaywrightError:
                    _discard_trace(managed.handle.trace_path, managed.raw_trace_path)
                except BaseException:
                    _discard_trace(managed.handle.trace_path, managed.raw_trace_path)
                    raise
        except BaseException as error:
            managed.cleanup_error = error
            raise
        finally:
            trace_resources.close()
            await _drain_playwright_teardown_tasks()
            if self._sessions.get(managed.handle.session_id) is managed:
                self._sessions.pop(managed.handle.session_id, None)
            managed.route_handler = None
            managed.websocket_handler = None
            managed.cleanup_complete.set()


def _sanitize_trace(
    managed: _ManagedSession,
    parent: SecureDirectoryHandle,
    raw_trace: SecureExclusiveFile,
) -> None:
    raw_path = managed.raw_trace_path
    sanitized = sanitize_artifact_content(
        managed.handle.trace_path.name,
        _read_descriptor(raw_trace.descriptor),
        managed.config.artifact_redaction,
    )
    legacy_temporary = managed.handle.trace_path.with_name(
        f".{managed.handle.trace_path.name}.sanitized"
    )
    _discard_file(legacy_temporary)
    temporary_name = f".{managed.handle.trace_path.name}.sanitized.{uuid4().hex}.tmp"
    with secure_create_exclusive_file(
        parent,
        temporary_name,
        "sanitized trace temporary file",
    ) as temporary:
        _write_descriptor(temporary.descriptor, sanitized)
        _replace_trace_with_retry(
            parent,
            temporary,
            managed.handle.trace_path.name,
        )
    secure_unlink(
        raw_path,
        "raw trace cleanup",
        missing_ok=True,
        expected_identity=raw_trace.identity,
    )


def _read_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _write_descriptor(descriptor: int, content: bytes) -> None:
    offset = 0
    while offset < len(content):
        written = os.write(descriptor, content[offset:])
        if written <= 0:
            raise OSError("failed to write sanitized trace")
        offset += written
    os.fsync(descriptor)


def _replace_trace_with_retry(
    parent: SecureDirectoryHandle,
    source: SecureExclusiveFile,
    destination_name: str,
) -> None:
    for attempt in range(_TRACE_REPLACE_ATTEMPTS):
        try:
            secure_replace_exclusive_file(
                parent,
                source,
                destination_name,
                "sanitized trace publication",
                replace_existing=True,
            )
            return
        except (PermissionError, BundleStateError) as error:
            if (
                isinstance(error, BundleStateError)
                and "0xc0000043" not in str(error).lower()
            ):
                raise
            if attempt == _TRACE_REPLACE_ATTEMPTS - 1:
                raise
            sleep(_TRACE_REPLACE_DELAY_SECONDS)


def _raw_trace_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.raw")


def _discard_trace(
    path: Path,
    raw_path: Path | None = None,
    *,
    include_published: bool = True,
) -> None:
    candidates = (
        (path,) if include_published else ()
    ) + (path.with_name(f".{path.name}.sanitized"),)
    if raw_path is not None:
        candidates += (raw_path,)
    for candidate in candidates:
        try:
            _discard_file(candidate)
        except FileNotFoundError:
            continue


def _discard_file(path: Path) -> None:
    secure_unlink(path, "trace cleanup", missing_ok=True)


def _retrieve_task_exception(task: asyncio.Task[Any]) -> None:
    if not task.cancelled():
        task.exception()


def _is_playwright_task(task: asyncio.Task[Any]) -> bool:
    """Return True when the task runs Playwright's own coroutine code.

    Playwright fires route-interception updates and route fulfil/continue
    calls as unowned tasks. Their exceptions are never read, so closing a
    context mid-flight makes the event loop report them as unretrieved when
    the finished task is collected. Matching on the coroutine's source file
    keeps the drain to Playwright's tasks and away from ours or the caller's.
    """

    code = getattr(task.get_coro(), "cr_code", None)
    filename = getattr(code, "co_filename", "") or ""
    return "playwright" in filename.replace("\\", "/").split("/")


def _is_playwright_teardown_task(task: asyncio.Task[Any]) -> bool:
    """Return True for a transient Playwright task spawned around teardown.

    The connection's reader loop outlives a session, so waiting for every
    Playwright task to finish would never end. Only the short-lived tasks
    Playwright creates around route handling settle during teardown.
    """

    if not _is_playwright_task(task):
        return False
    code = getattr(task.get_coro(), "cr_code", None)
    return getattr(code, "co_qualname", "") not in _PLAYWRIGHT_SESSION_TASKS


def _is_unretrieved_playwright_task(context: Mapping[str, object]) -> bool:
    """Return True for an unretrieved-task report about a Playwright task."""

    if context.get("message") != "Task exception was never retrieved":
        return False
    if not isinstance(context.get("exception"), PlaywrightError):
        return False
    future = context.get("future")
    return isinstance(future, asyncio.Task) and _is_playwright_task(
        cast("asyncio.Task[Any]", future)
    )


async def _drain_playwright_teardown_tasks() -> None:
    """Let Playwright's unowned teardown tasks finish and read them out.

    A finished task whose exception nothing retrieves is reported by the
    event loop as ``Task exception was never retrieved`` when it is garbage
    collected, which the loop may do before anyone can look at the result.
    Holding a reference to every transient Playwright task, waiting for them
    to settle, and reading their results keeps teardown quiet. A loop filter
    for the same reports covers any task that slipped past the sampling.
    """

    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()

    def filtered(
        active_loop: asyncio.AbstractEventLoop, context: dict[str, Any]
    ) -> None:
        if _is_unretrieved_playwright_task(context):
            return
        if previous is not None:
            previous(active_loop, context)
        else:
            active_loop.default_exception_handler(context)

    loop.set_exception_handler(filtered)
    try:
        held: set[asyncio.Task[Any]] = {
            task for task in asyncio.all_tasks() if _is_playwright_task(task)
        }
        deadline = monotonic() + _TEARDOWN_TASK_SETTLE_SECONDS
        while monotonic() < deadline:
            held.update(
                task for task in asyncio.all_tasks() if _is_playwright_task(task)
            )
            if not any(
                not task.done() and _is_playwright_teardown_task(task)
                for task in held
            ):
                break
            await asyncio.sleep(_TEARDOWN_TASK_POLL_SECONDS)
        for task in held:
            if task.done():
                _retrieve_task_exception(task)
        gc.collect()
    finally:
        loop.set_exception_handler(previous)


def _track_task(
    tasks: set[asyncio.Task[None]], awaitable: Coroutine[Any, Any, None]
) -> asyncio.Task[None]:
    task = asyncio.create_task(awaitable)
    tasks.add(task)

    def finish(completed: asyncio.Task[None]) -> None:
        tasks.discard(completed)
        _retrieve_task_exception(completed)

    task.add_done_callback(finish)
    return task


async def _cancel_and_gather_tasks(tasks: set[asyncio.Task[None]]) -> None:
    while tasks:
        current = tuple(tasks)
        for task in current:
            if not task.done():
                task.cancel()
        await asyncio.gather(*current, return_exceptions=True)


async def _await_popup_tasks(tasks: set[asyncio.Task[None]]) -> None:
    await asyncio.sleep(0)
    while tasks:
        current = tuple(tasks)
        results = await asyncio.gather(*current, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException) and not isinstance(
                result, asyncio.CancelledError
            ):
                raise ProviderFailure("popup cleanup failed") from result
        await asyncio.sleep(0)


def _is_transient_popup_url(url: str) -> bool:
    return url in _TRANSIENT_POPUP_URLS


async def _shielded_cleanup(awaitable: Awaitable[None]) -> None:
    cleanup = asyncio.ensure_future(awaitable)
    cancellation: asyncio.CancelledError | None = None
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError as error:
            cancellation = error
    cleanup_error = cleanup.exception()
    if cancellation is not None:
        raise cancellation from cleanup_error
    if cleanup_error is not None:
        raise cleanup_error


def _optional_colour(colours: Mapping[str, object] | None, key: str) -> str | None:
    """Read one optional CSS colour string from an evaluated colour mapping."""

    if colours is None:
        return None
    value = colours.get(key)
    return value if isinstance(value, str) and value else None


async def _viewport_background(page: Page) -> str | None:
    """Sample the dominant background colour of the current viewport.

    Uses a coarse CSS-pixel grid over the viewport and reports the most common
    opaque ``backgroundColor``. A coarse sample is deliberate: it is what a
    sighted user perceives as "the screen colour" and it is stable against
    anti-aliasing at element edges. Failures return ``None`` so capture never
    breaks.
    """

    try:
        value = await page.evaluate(
            """() => {
                const step = 32;
                const counts = new Map();
                const w = window.innerWidth;
                const h = window.innerHeight;
                for (let y = step / 2; y < h; y += step) {
                    for (let x = step / 2; x < w; x += step) {
                        const el = document.elementFromPoint(x, y);
                        if (!el) continue;
                        const bg = getComputedStyle(el).backgroundColor;
                        if (!bg || bg === 'rgba(0, 0, 0, 0)') continue;
                        counts.set(bg, (counts.get(bg) || 0) + 1);
                    }
                }
                let best = null;
                let bestCount = 0;
                for (const [colour, count] of counts) {
                    if (count > bestCount) {
                        best = colour;
                        bestCount = count;
                    }
                }
                return best;
            }"""
        )
    except BaseException:  # noqa: BLE001 - capture must not fail
        return None
    return value if isinstance(value, str) and value else None


def _center(bounds: object) -> tuple[float, float]:
    if not hasattr(bounds, "x"):
        raise ProviderFailure("action target has no bounds")
    x = float(getattr(bounds, "x"))
    y = float(getattr(bounds, "y"))
    width = float(getattr(bounds, "width"))
    height = float(getattr(bounds, "height"))
    return x + width / 2, y + height / 2


def _scroll_amount(action: ScrollAction) -> int:
    if isinstance(action.amount, int):
        return action.amount
    return {"small": 240, "medium": 640, "large": 1024}[action.amount]


def _elapsed_ms(started: float) -> int:
    return int((monotonic() - started) * 1000)


def _blocked_action_result(
    managed: _ManagedSession, started: float
) -> PlatformActionResult:
    return PlatformActionResult(
        succeeded=False,
        url=managed.page.url,
        duration_ms=_elapsed_ms(started),
        error="navigation blocked by safety policy",
    )


def _is_recoverable_user_interaction(action: PlatformAction) -> bool:
    return isinstance(action, _RECOVERABLE_USER_ACTIONS)


async def _settle_after_action(
    page: Page,
    navigation_started: asyncio.Event,
    navigation_settle_ms: int,
    action_settle_ms: int,
) -> None:
    settle_window_seconds = max(
        _NAVIGATION_START_GRACE_SECONDS, action_settle_ms / 1000
    )
    try:
        await asyncio.wait_for(
            navigation_started.wait(), timeout=settle_window_seconds
        )
    except TimeoutError:
        return
    await page.wait_for_load_state("domcontentloaded")
    await _wait_for_navigation_settle(page, navigation_settle_ms)
    await page.wait_for_timeout(action_settle_ms)


async def _wait_for_navigation_settle(page: Page, navigation_settle_ms: int) -> None:
    await page.wait_for_timeout(navigation_settle_ms)


def _blocked_document_navigation(
    blocked_events: list[BlockedRequest], start: int
) -> BlockedRequest | None:
    return next(
        (
            event
            for event in blocked_events[start:]
            if event.resource_type == "document"
            and event.kind in {"request", "redirect", "popup"}
        ),
        None,
    )


def _request_kind(request: Request, main_page: Page) -> str:
    if request.resource_type != "document" or not request.is_navigation_request():
        return "request"
    try:
        return "popup" if request.frame.page != main_page else "request"
    except PlaywrightError:
        return "popup"
