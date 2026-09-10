"""Platform-neutral observation, session, and action contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from ux_analyzer.domain.interface import BoundingBox, ViewportSnapshot
from ux_analyzer.ports.artifacts import RedactionPolicy


class ObservationProviderError(RuntimeError):
    """Base error for observation provider failures."""


class SafetyBlocked(ObservationProviderError):
    """Raised when a provider refuses an unsafe browser operation."""


@dataclass(frozen=True, slots=True)
class TestAccountId:
    """Identifier restricted to disposable benchmark accounts."""

    value: str

    def __post_init__(self) -> None:
        if not self.value.startswith("test-"):
            raise ValueError("test account ID must start with 'test-'")


@dataclass(frozen=True, slots=True)
class ViewportSize:
    """Fixed viewport dimensions for one isolated run."""

    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("viewport dimensions must be greater than zero")


@dataclass(frozen=True, slots=True)
class ObservationSessionConfig:
    """Inputs needed to create one isolated platform session."""

    session_id: str
    start_url: str
    test_account_id: TestAccountId | str
    viewport: ViewportSize
    trace_path: Path
    artifact_redaction: RedactionPolicy = RedactionPolicy()
    navigation_settle_ms: int = 0
    action_settle_ms: int = 0
    navigation_origins: tuple[str, ...] = ()
    resource_origins: tuple[str, ...] = ()
    fixture_only: bool = True

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session ID must not be empty")
        if not self.start_url:
            raise ValueError("start URL must not be empty")
        account_id = self.test_account_id
        if isinstance(account_id, str):
            account_id = TestAccountId(account_id)
        object.__setattr__(self, "test_account_id", account_id)
        object.__setattr__(self, "navigation_origins", tuple(self.navigation_origins))
        object.__setattr__(self, "resource_origins", tuple(self.resource_origins))
        if not self.trace_path.name:
            raise ValueError("trace path must contain a filename")
        for name, milliseconds in (
            ("navigation settle", self.navigation_settle_ms),
            ("action settle", self.action_settle_ms),
        ):
            if isinstance(milliseconds, bool) or milliseconds < 0:
                raise ValueError(f"{name} duration must not be negative")


@dataclass(slots=True)
class SessionHandle:
    """Opaque platform session identity with safe diagnostic state."""

    session_id: str
    test_account_id: TestAccountId
    viewport: ViewportSize
    trace_path: Path
    blocked_events: list[BlockedRequest]


@dataclass(frozen=True, slots=True)
class BlockedRequest:
    """Sanitized record of one browser request denied by policy."""

    url: str
    origin: str
    resource_type: str
    kind: str


@dataclass(frozen=True, slots=True)
class ObservationCapture:
    """Platform-neutral capture returned before extraction normalization."""

    session_id: str
    viewport_id: str
    url: str
    title: str
    viewport: ViewportSize
    screenshot: bytes
    snapshot: ViewportSnapshot | None = None
    # Full rendered text of the document (i.e. ``document.body.innerText`` at
    # capture time). Optional so existing tests / fakes can leave it unset;
    # verifiers that need to look beyond the current viewport use it as a
    # "page text" fallback when the viewport snapshot alone does not match.
    page_text: str | None = None


@dataclass(frozen=True, slots=True)
class PlatformActionResult:
    """Result of executing one platform-neutral action."""

    succeeded: bool
    url: str
    duration_ms: int
    navigation_occurred: bool = False
    state_changed: bool = False
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ClickAction:
    """Click one persona-visible element."""

    kind: Literal["click"] = "click"
    element_id: str = ""
    bounds: BoundingBox | None = None

    def __post_init__(self) -> None:
        if not self.element_id:
            raise ValueError("click action needs element ID")


@dataclass(frozen=True, slots=True)
class DoubleClickAction:
    """Double-click one persona-visible element."""

    kind: Literal["double-click"] = "double-click"
    element_id: str = ""
    bounds: BoundingBox | None = None

    def __post_init__(self) -> None:
        if not self.element_id:
            raise ValueError("double-click action needs element ID")


@dataclass(frozen=True, slots=True)
class TypeTextAction:
    """Type fixture-provided text into one element."""

    kind: Literal["type-text"] = "type-text"
    element_id: str = ""
    text: str = ""
    bounds: BoundingBox | None = None

    def __post_init__(self) -> None:
        if not self.element_id:
            raise ValueError("type action needs element ID")


@dataclass(frozen=True, slots=True)
class ClearTextAction:
    """Clear one persona-visible text field."""

    kind: Literal["clear-text"] = "clear-text"
    element_id: str = ""
    bounds: BoundingBox | None = None

    def __post_init__(self) -> None:
        if not self.element_id:
            raise ValueError("clear action needs element ID")


@dataclass(frozen=True, slots=True)
class SelectOptionAction:
    """Select one option in one persona-visible control."""

    kind: Literal["select-option"] = "select-option"
    element_id: str = ""
    option: str = ""
    bounds: BoundingBox | None = None

    def __post_init__(self) -> None:
        if not self.element_id or not self.option:
            raise ValueError("select action needs element ID and option")


@dataclass(frozen=True, slots=True)
class ToggleAction:
    """Toggle one persona-visible control."""

    kind: Literal["toggle"] = "toggle"
    element_id: str = ""
    bounds: BoundingBox | None = None

    def __post_init__(self) -> None:
        if not self.element_id:
            raise ValueError("toggle action needs element ID")


@dataclass(frozen=True, slots=True)
class SubmitAction:
    """Submit one persona-visible form."""

    kind: Literal["submit"] = "submit"
    element_id: str = ""
    bounds: BoundingBox | None = None

    def __post_init__(self) -> None:
        if not self.element_id:
            raise ValueError("submit action needs element ID")


@dataclass(frozen=True, slots=True)
class OpenMenuAction:
    """Open one persona-visible menu."""

    kind: Literal["open-menu"] = "open-menu"
    element_id: str = ""
    bounds: BoundingBox | None = None

    def __post_init__(self) -> None:
        if not self.element_id:
            raise ValueError("menu action needs element ID")


@dataclass(frozen=True, slots=True)
class PressKeyAction:
    """Press one keyboard key in current page context."""

    kind: Literal["press-key"] = "press-key"
    key: str = ""

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("key action needs key")


@dataclass(frozen=True, slots=True)
class DragAction:
    """Drag one element by platform-neutral coordinates."""

    kind: Literal["drag"] = "drag"
    element_id: str = ""
    end_x: float = 0
    end_y: float = 0
    bounds: BoundingBox | None = None

    def __post_init__(self) -> None:
        if not self.element_id:
            raise ValueError("drag action needs element ID")


@dataclass(frozen=True, slots=True)
class NavigateAction:
    """Navigate to an explicitly requested URL through provider policy."""

    kind: Literal["navigate"] = "navigate"
    url: str = ""

    def __post_init__(self) -> None:
        if not self.url:
            raise ValueError("navigate action needs URL")


@dataclass(frozen=True, slots=True)
class ScrollAction:
    """Scroll current view by bounded platform-neutral distance."""

    kind: Literal["scroll"] = "scroll"
    direction: Literal["up", "down"] = "down"
    amount: int | Literal["small", "medium", "large"] = "medium"

    def __post_init__(self) -> None:
        if isinstance(self.amount, int) and self.amount <= 0:
            raise ValueError("scroll amount must be greater than zero")


@dataclass(frozen=True, slots=True)
class BackAction:
    """Navigate one step backward in session history."""

    kind: Literal["back"] = "back"


@dataclass(frozen=True, slots=True)
class WaitAction:
    """Wait for interface state to settle."""

    kind: Literal["wait"] = "wait"
    milliseconds: int = 0

    def __post_init__(self) -> None:
        if self.milliseconds < 0:
            raise ValueError("wait duration must not be negative")


type PlatformAction = (
    ClickAction
    | DoubleClickAction
    | TypeTextAction
    | ClearTextAction
    | SelectOptionAction
    | ToggleAction
    | SubmitAction
    | OpenMenuAction
    | PressKeyAction
    | DragAction
    | NavigateAction
    | ScrollAction
    | BackAction
    | WaitAction
)


class ObservationProvider(Protocol):
    """Platform-neutral lifecycle contract for observation providers."""

    id: str
    platform: Literal["web", "desktop", "mobile"]

    async def start_session(
        self, config: ObservationSessionConfig
    ) -> SessionHandle: ...

    async def capture(self, session: SessionHandle) -> ObservationCapture: ...

    async def execute(
        self, session: SessionHandle, action: PlatformAction
    ) -> PlatformActionResult: ...

    async def reset(self, session: SessionHandle) -> None: ...

    async def end_session(self, session: SessionHandle) -> None: ...
