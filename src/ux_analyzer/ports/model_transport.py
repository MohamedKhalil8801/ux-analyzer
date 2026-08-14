"""Shared bounded serialization primitives for structured model transports."""

from __future__ import annotations

import json
import math

MODEL_REQUEST_MAX_BYTES = 750_000
# Evidence may be resolved above one request's encoded budget so the provider can
# select a fitting subset and report omitted attachments explicitly.
MODEL_ATTACHMENT_MAX_BYTES = 16 * 1024 * 1024


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
    if size > MODEL_REQUEST_MAX_BYTES:
        raise TransportBudgetError(
            "report synthesis request exceeds transport-safe byte budget"
        )
    return size


def require_finite_float(value: float, *, context: str) -> None:
    """Reject non-standard JSON float values before serialization."""

    if not math.isfinite(value):
        raise ValueError(f"{context} must contain finite float values")
