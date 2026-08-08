"""Playwright session adapter with fail-closed browser safety controls."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic, sleep
from typing import Any

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
from ux_analyzer.ports.artifacts import sanitize_artifact_content
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
        config.trace_path.parent.mkdir(parents=True, exist_ok=True)
        config.trace_path.unlink(missing_ok=True)
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
            await context.tracing.start(
                screenshots=True,
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
            return ObservationCapture(
                session_id=session.session_id,
                viewport_id=f"{session.session_id}-viewport-{managed.capture_index}",
                url=managed.page.url,
                title=await managed.page.title(),
                viewport=session.viewport,
                screenshot=screenshot,
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
                    await managed.context.tracing.stop(
                        path=str(managed.handle.trace_path)
                    )
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error

            try:
                await managed.context.close()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error

            if cleanup_error is not None:
                _discard_trace(managed.handle.trace_path)
                if isinstance(cleanup_error, PlaywrightError):
                    return
                raise cleanup_error

            if managed.trace_started:
                try:
                    _sanitize_trace(managed)
                except PlaywrightError:
                    _discard_trace(managed.handle.trace_path)
                except BaseException:
                    _discard_trace(managed.handle.trace_path)
                    raise
        except BaseException as error:
            managed.cleanup_error = error
            raise
        finally:
            if self._sessions.get(managed.handle.session_id) is managed:
                self._sessions.pop(managed.handle.session_id, None)
            managed.route_handler = None
            managed.websocket_handler = None
            managed.cleanup_complete.set()


def _sanitize_trace(managed: _ManagedSession) -> None:
    path = managed.handle.trace_path
    if not path.is_file():
        return
    sanitized = sanitize_artifact_content(
        path.name,
        path.read_bytes(),
        managed.config.artifact_redaction,
    )
    temporary = path.with_name(f".{path.name}.sanitized")
    temporary.write_bytes(sanitized)
    _replace_trace_with_retry(temporary, path)


def _replace_trace_with_retry(source: Path, destination: Path) -> None:
    for attempt in range(_TRACE_REPLACE_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == _TRACE_REPLACE_ATTEMPTS - 1:
                raise
            sleep(_TRACE_REPLACE_DELAY_SECONDS)


def _discard_trace(path: Path) -> None:
    for candidate in (path, path.with_name(f".{path.name}.sanitized")):
        try:
            with candidate.open("r+b") as trace:
                trace.truncate(0)
                trace.flush()
                os.fsync(trace.fileno())
        except FileNotFoundError:
            continue
        candidate.unlink(missing_ok=True)


def _retrieve_task_exception(task: asyncio.Task[None]) -> None:
    if not task.cancelled():
        task.exception()


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
