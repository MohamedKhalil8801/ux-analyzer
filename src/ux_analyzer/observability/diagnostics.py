"""Structured, timestamped diagnostics for post-run forensics.

The synthesis pipeline rejects model output for a documented set of reasons
and returns a *sanitized* reason to the model. That sanitization is correct for
the corpus boundary, but it has a second effect that is not intended: a
persisted attempt records that output was rejected without recording what was
rejected. When a live run fails, that is the difference between a diagnosis
and a guess.

This module closes the gap without reopening the boundary. Every event carries

* a UTC timestamp with millisecond resolution,
* the emitting layer, role, and stage,
* a sanitized reason drawn from our own fixed vocabulary, never from model text,
* the field path that failed,
* a SHA-256 fingerprint and byte length of the offending text, and
* an optional structured extra payload.

The offending text itself is passed in only to be fingerprinted. It is never
stored, logged, returned, or attached to an exception. Two runs that produce
identical rejected text produce identical fingerprints, which is what makes
the value useful for grouping failures across attempts.

Every other value is truncated to a fixed length rather than filtered for
wording. Those values are counts, indices, schema names, and reason codes
drawn from this codebase's own vocabulary; filtering them for phrases would
reproduce the substring-guard failure this module exists to make diagnosable.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO, cast

_RING_LIMIT = 2048
_FINGERPRINT_LENGTH = 16
_MAX_VALUE_LENGTH = 200
_TRUTHY = frozenset({"1", "true", "yes", "on"})

_lock = threading.Lock()
_ring: deque[Mapping[str, Any]] = deque(maxlen=_RING_LIMIT)
_file_handle: TextIO | None = None
_file_path: Path | None = None
_echo_stderr = False


def fingerprint(text: str) -> str:
    """Return a stable short digest of ``text``.

    The digest identifies content without disclosing it. It is not a security
    boundary: it is a correlation key, and it is only safe because the text it
    summarizes is already excluded from the corpus boundary.
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_FINGERPRINT_LENGTH]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().casefold() in _TRUTHY


def configure(*, path: str | os.PathLike[str] | None = None, echo_stderr: bool | None = None) -> None:
    """Configure diagnostic sinks.

    ``path`` appends JSON lines to a log file. ``echo_stderr`` mirrors every
    event to standard error. Both are also settable through the
    ``UXA_DIAGNOSTIC_LOG`` and ``UXA_DIAGNOSTIC_STDERR`` environment
    variables, which are read on first use when this is not called. An
    in-memory ring buffer always records, so a caller can retrieve recent
    events without configuring any sink.
    """

    global _file_handle, _file_path, _echo_stderr
    with _lock:
        if _file_handle is not None:
            try:
                _file_handle.close()
            finally:
                _file_handle = None
                _file_path = None
        _echo_stderr = _truthy("UXA_DIAGNOSTIC_STDERR") if echo_stderr is None else echo_stderr
        target = path if path is not None else os.environ.get("UXA_DIAGNOSTIC_LOG")
        if not target:
            return
        resolved = Path(target)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        _file_path = resolved
        _file_handle = resolved.open("a", encoding="utf-8")


def _safe_text(value: object) -> str:
    """Coerce a context value to a bounded, printable string.

    Context values are counts, indices, schema names, and reason codes drawn
    from this codebase's own vocabulary. ``text=`` is the only parameter that
    ever receives model-authored content, and it is fingerprinted rather than
    passed through here.
    """

    rendered = str(value)
    return rendered if len(rendered) <= _MAX_VALUE_LENGTH else rendered[:_MAX_VALUE_LENGTH]


def _sanitize_extras(extras: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in extras.items():
        safe_key = _safe_text(key)
        if isinstance(value, str):
            sanitized[safe_key] = _safe_text(value)
        elif isinstance(value, (int, float, bool)) or value is None:
            sanitized[safe_key] = value
        elif isinstance(value, (list, tuple)):
            items = cast(Sequence[object], value)
            sanitized[safe_key] = [
                _safe_text(item) if isinstance(item, str) else item for item in items
            ]
        else:
            sanitized[safe_key] = _safe_text(value)
    return sanitized


def record(
    event: str,
    *,
    layer: str = "application",
    role: str | None = None,
    stage: str | None = None,
    reason: str | None = None,
    reason_code: str | None = None,
    field_path: str | None = None,
    text: str | None = None,
    **extras: Any,
) -> Mapping[str, Any]:
    """Record one diagnostic event and return the stored mapping.

    ``reason`` must come from this codebase's own fixed vocabulary; it is
    truncated, not word-filtered. ``text`` is the only parameter that accepts
    model-authored content. It is fingerprinted and discarded, and never appears
    in the returned mapping.
    """

    payload: dict[str, Any] = {
        "timestamp": _utc_now(),
        "event": _safe_text(event),
        "layer": _safe_text(layer),
    }
    if role is not None:
        payload["role"] = _safe_text(role)
    if stage is not None:
        payload["stage"] = _safe_text(stage)
    if reason_code is not None:
        payload["reason_code"] = _safe_text(reason_code)
    if reason is not None:
        payload["reason"] = _safe_text(reason)
    if field_path is not None:
        payload["field"] = _safe_text(field_path)
    if text is not None:
        payload["field_hash"] = fingerprint(text)
        payload["field_length"] = len(text)
    if extras:
        payload["extras"] = _sanitize_extras(extras)

    line = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
    with _lock:
        _ring.append(payload)
        handle = _file_handle
        echo = _echo_stderr
    if echo:
        print(line, file=sys.stderr)
    if handle is not None:
        try:
            handle.write(line + "\n")
            handle.flush()
        except (OSError, ValueError):
            pass
    return payload


def recent(limit: int | None = None) -> tuple[Mapping[str, Any], ...]:
    """Return the most recent recorded events, oldest first."""

    with _lock:
        events = tuple(_ring)
    if limit is None:
        return events
    return events[-limit:] if limit > 0 else ()


def drain() -> tuple[Mapping[str, Any], ...]:
    """Return and clear the in-memory ring buffer."""

    with _lock:
        events = tuple(_ring)
        _ring.clear()
    return events


def reset() -> None:
    """Clear buffered events and close the file sink. Intended for tests."""

    configure(path=None, echo_stderr=False)
    with _lock:
        _ring.clear()


__all__ = [
    "configure",
    "drain",
    "fingerprint",
    "recent",
    "record",
    "reset",
]
