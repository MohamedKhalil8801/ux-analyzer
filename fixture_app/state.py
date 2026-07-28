"""Isolated in-memory state for controlled benchmark sessions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import Lock

DEFAULT_FIXTURE_INPUTS: dict[str, str] = {
    "invite_email": "person@example.com",
    "invite_role": "member",
    "totp_code": "123456",
}


@dataclass(slots=True)
class WorkspaceState:
    """Private workspace state used by invite verification."""

    invited_email: str | None = None
    invited_role: str | None = None
    invite_status: str = "idle"

    def as_dict(self) -> dict[str, str | None]:
        return {
            "invited_email": self.invited_email,
            "invited_role": self.invited_role,
            "invite_status": self.invite_status,
        }


@dataclass(slots=True)
class SecurityState:
    """Private security state used by two-factor verification."""

    two_factor_enabled: bool = False
    two_factor_status: str = "disabled"

    def as_dict(self) -> dict[str, bool | str]:
        return {
            "two_factor_enabled": self.two_factor_enabled,
            "two_factor_status": self.two_factor_status,
        }


@dataclass(slots=True)
class SessionState:
    """Complete state contract shared by defective and improved UI versions."""

    session_id: str
    fixture_inputs: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_FIXTURE_INPUTS)
    )
    workspace: WorkspaceState = field(default_factory=WorkspaceState)
    security: SecurityState = field(default_factory=SecurityState)

    def invite(self, email: str, role: str) -> None:
        self.workspace.invited_email = email
        self.workspace.invited_role = role
        self.workspace.invite_status = "sent"

    def enable_two_factor(self) -> None:
        self.security.two_factor_enabled = True
        self.security.two_factor_status = "enabled"

    def as_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "fixture_inputs": dict(self.fixture_inputs),
            "workspace": self.workspace.as_dict(),
            "security": self.security.as_dict(),
            "completion": {
                "invite-teammate": self.workspace.invite_status == "sent",
                "enable-2fa": self.security.two_factor_enabled,
            },
        }


class SessionStore:
    """Thread-safe process-local store. No persistence or network side effects."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._deleted: set[str] = set()
        self._lock = Lock()

    def get_or_create(self, session_id: str) -> SessionState:
        with self._lock:
            self._deleted.discard(session_id)
            return self._sessions.setdefault(session_id, SessionState(session_id))

    def reset(
        self, session_id: str, inputs: Mapping[str, str] | None = None
    ) -> SessionState:
        fixture_inputs = dict(DEFAULT_FIXTURE_INPUTS)
        if inputs is not None:
            for key in fixture_inputs:
                supplied_value = inputs.get(key)
                if supplied_value is not None and supplied_value.strip():
                    fixture_inputs[key] = supplied_value
        with self._lock:
            self._deleted.discard(session_id)
            state = SessionState(session_id=session_id, fixture_inputs=fixture_inputs)
            self._sessions[session_id] = state
            return state

    def invite(self, session_id: str, email: str, role: str) -> SessionState:
        with self._lock:
            self._deleted.discard(session_id)
            state = self._sessions.setdefault(session_id, SessionState(session_id))
            state.invite(email, role)
            return state

    def enable_two_factor(self, session_id: str) -> SessionState:
        with self._lock:
            self._deleted.discard(session_id)
            state = self._sessions.setdefault(session_id, SessionState(session_id))
            state.enable_two_factor()
            return state

    def snapshot(self, session_id: str) -> dict[str, object] | None:
        with self._lock:
            if session_id in self._deleted:
                return None
            state = self._sessions.setdefault(session_id, SessionState(session_id))
            return state.as_dict()

    def delete(self, session_id: str) -> bool:
        with self._lock:
            deleted = self._sessions.pop(session_id, None) is not None
            self._deleted.add(session_id)
            return deleted


session_store = SessionStore()
