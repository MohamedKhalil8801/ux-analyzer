"""Playwright session adapter with fixture-only browser safety controls."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from playwright.async_api import Browser, BrowserContext, Page
from playwright.async_api import Error as PlaywrightError

from ux_analyzer.adapters.web.network_policy import (
    BrowserAllowedOrigins,
    NetworkPolicy,
)
from ux_analyzer.ports.observation import (
    BackAction,
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


class PlaywrightSessionAdapter:
    """Safe web observation provider backed by one isolated context per run."""

    id = "playwright-web"
    platform = "web"

    def __init__(
        self,
        *,
        browser: Browser,
        allowed_origins: BrowserAllowedOrigins | Iterable[str],
        trace_directory: Path,
    ) -> None:
        self._browser = browser
        self._allowed_origins = (
            allowed_origins
            if isinstance(allowed_origins, BrowserAllowedOrigins)
            else BrowserAllowedOrigins.fixture_only(allowed_origins)
        )
        self._trace_directory = trace_directory
        self._sessions: dict[str, _ManagedSession] = {}

    @property
    def active_session_count(self) -> int:
        return len(self._sessions)

    async def start_session(self, config: ObservationSessionConfig) -> SessionHandle:
        self._allowed_origins.require_allowed(
            config.start_url, resource_type="document", kind="navigation"
        )
        config.trace_path.parent.mkdir(parents=True, exist_ok=True)
        config.trace_path.unlink(missing_ok=True)
        policy = NetworkPolicy(self._allowed_origins)
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
            await context.route(
                "**/*", lambda route, request: policy.handle_route(route, request)
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
            self._sessions[handle.session_id] = managed
            context.on(
                "page",
                lambda popup: asyncio.create_task(
                    self._close_foreign_popup(managed, popup)
                ),
            )
            await self._navigate(managed, config.start_url)
            return handle
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
        try:
            navigation_occurred = False
            if isinstance(action, NavigateAction):
                await self._navigate(managed, action.url)
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
            return PlatformActionResult(
                succeeded=True,
                url=managed.page.url,
                duration_ms=_elapsed_ms(started),
                navigation_occurred=navigation_occurred,
                state_changed=not isinstance(action, (WaitAction, PressKeyAction)),
            )
        except SafetyBlocked:
            await self._cleanup(managed)
            raise
        except BaseException as error:
            await self._cleanup(managed)
            if isinstance(error, ProviderFailure):
                raise
            raise ProviderFailure("browser action failed") from error

    async def reset(self, session: SessionHandle) -> None:
        managed = self._active(session)
        try:
            await managed.context.clear_cookies()
            await self._navigate(managed, managed.config.start_url)
        except SafetyBlocked:
            await self._cleanup(managed)
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

    async def _navigate(self, managed: _ManagedSession, url: str) -> None:
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

    async def _close_foreign_popup(self, managed: _ManagedSession, popup: Page) -> None:
        try:
            await popup.wait_for_load_state("domcontentloaded", timeout=500)
        except PlaywrightError:
            pass
        if not managed.policy.allowed_origins.allows(popup.url):
            managed.policy.record_popup(popup.url)
            await popup.close()

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
        if managed.cleanup_started:
            return
        managed.cleanup_started = True
        self._sessions.pop(managed.handle.session_id, None)
        try:
            if managed.trace_started:
                await managed.context.tracing.stop(path=str(managed.handle.trace_path))
        except PlaywrightError:
            managed.handle.trace_path.touch(exist_ok=True)
        finally:
            try:
                await managed.context.close()
            except PlaywrightError:
                pass


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
