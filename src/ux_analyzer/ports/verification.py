"""Platform-neutral reset and independent verification ports."""

from __future__ import annotations

from typing import Protocol

from ux_analyzer.domain.run import VerificationResult
from ux_analyzer.ports.observation import SessionHandle


class ResetProvider(Protocol):
    """Reset one isolated benchmark session to its configured start state."""

    async def reset(self, session: SessionHandle) -> None: ...


class VerificationProvider(Protocol):
    """Determine official task outcome independently from agent claims."""

    async def verify(self, session: SessionHandle) -> VerificationResult: ...


Verifier = VerificationProvider
