"""Safe semantic progress detection for recaptured interface snapshots."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from urllib.parse import urlsplit

from ux_analyzer.domain.interface import ElementSnapshot, ViewportSnapshot

SemanticSnapshotSignature = tuple[tuple[object, ...], ...]
TransitionProgressSignature = tuple[object, ...]


def snapshot_progress_signature(
    snapshot: ViewportSnapshot,
) -> SemanticSnapshotSignature:
    """Return a stable, private-data-free signature of meaningful UI state."""

    occurrences: Counter[tuple[object, ...]] = Counter()
    signature: list[tuple[object, ...]] = []
    for element in snapshot.elements:
        semantic = _element_semantics(element)
        occurrence = occurrences[semantic]
        occurrences[semantic] += 1
        identity = element.lineage_id or ("semantic", semantic, occurrence)
        signature.append((identity, *semantic))
    return tuple(sorted(signature, key=repr))


def made_meaningful_progress(
    before: ViewportSnapshot,
    after: ViewportSnapshot,
    *,
    succeeded: bool,
    navigation_occurred: bool,
    fixture_completed: bool,
) -> bool:
    """Decide whether an action advanced the task rather than merely ran."""

    if not succeeded:
        return False
    return (
        navigation_occurred
        or fixture_completed
        or snapshot_progress_signature(before) != snapshot_progress_signature(after)
    )


def transition_progress_signature(
    action_fingerprint: tuple[str, str | None, str | None],
    before: ViewportSnapshot,
    after: ViewportSnapshot,
    current_url: str | None,
) -> TransitionProgressSignature:
    """Return a private-data-free signature for one action/UI transition."""

    return (
        action_fingerprint,
        snapshot_progress_signature(before),
        snapshot_progress_signature(after),
        _safe_url_origin_path(current_url),
    )


def repeated_cycle_length(
    history: Sequence[TransitionProgressSignature],
    *,
    min_length: int = 2,
    max_length: int = 4,
) -> int | None:
    """Return the repeated suffix period when two complete periods match."""

    for length in range(min_length, max_length + 1):
        if len(history) >= length * 2 and history[-length:] == history[-2 * length : -length]:
            return length
    return None


def _element_semantics(element: ElementSnapshot) -> tuple[object, ...]:
    return (
        str(element.role),
        element.label,
        element.region_id,
        _visibility_bucket(element.visibility_fraction),
        element.actionable,
        element.disabled,
    )


def _visibility_bucket(fraction: float) -> str:
    if fraction <= 0:
        return "hidden"
    if fraction >= 1:
        return "full"
    return "partial"


def _safe_url_origin_path(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlsplit(url)
    hostname = parsed.hostname
    if not parsed.scheme or not hostname:
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    host = f"[{hostname}]" if ":" in hostname else hostname
    authority = f"{host}:{port}" if port is not None else host
    return f"{parsed.scheme}://{authority}{parsed.path}"
