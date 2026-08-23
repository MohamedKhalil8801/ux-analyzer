"""Shared bounded serialization primitives for structured model transports."""

from __future__ import annotations

import json
import math

def _configured_request_max_bytes() -> int:
    """Return transport ceiling, honoring UXA_LLM_REQUEST_MAX_BYTES when set."""

    import os

    raw = os.environ.get("UXA_LLM_REQUEST_MAX_BYTES", "").strip()
    if not raw:
        return 750_000
    try:
        value = int(raw)
    except ValueError:
        return 750_000
    return max(100_000, value)


MODEL_REQUEST_MAX_BYTES = 750_000
# Backwards-compat alias used by import-time constants in report_synthesis;
# callers that need the live configured value must call
# _configured_request_max_bytes() or model_request_max_bytes().
MODEL_ATTACHMENT_MAX_BYTES = 16 * 1024 * 1024


def model_request_max_bytes() -> int:
    """Return the live configured transport ceiling."""

    return _configured_request_max_bytes()


class TransportBudgetError(ValueError):
    """Raised when one structured model request exceeds its byte ceiling."""


class TransportEvidenceUnavailableError(TransportBudgetError):
    """Raised when a role re-requests evidence excluded by transport fitting."""

    reason = "visual evidence unavailable"

    def __init__(self, unavailable_count: int) -> None:
        if not 1 <= unavailable_count <= 32:
            raise ValueError("unavailable evidence count must be between 1 and 32")
        self.unavailable_count = unavailable_count
        super().__init__(self.reason)


def serialize_transport_json(
    value: object,
    *,
    sort_keys: bool = True,
) -> bytes:
    """Serialize transport JSON deterministically as strict UTF-8 bytes."""

    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=sort_keys,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def transport_size(*parts: bytes) -> int:
    """Return total outbound byte count for one transport request."""

    return sum(len(part) for part in parts)


def enforce_transport_size(*parts: bytes) -> int:
    """Reject an outbound request above the hard transport ceiling."""

    size = transport_size(*parts)
    if size > _configured_request_max_bytes():
        raise TransportBudgetError(
            "report synthesis request exceeds transport-safe byte budget"
        )
    return size


def require_finite_float(value: float, *, context: str) -> None:
    """Reject non-standard JSON float values before serialization."""

    if not math.isfinite(value):
        raise ValueError(f"{context} must contain finite float values")
