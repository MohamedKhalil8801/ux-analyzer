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

    region_labels = {region.id: region.label for region in snapshot.regions}
    occurrences: Counter[tuple[object, ...]] = Counter()
    signature: list[tuple[object, ...]] = []
    for element in snapshot.elements:
        region_label = (
            region_labels.get(element.region_id)
            if element.region_id is not None
            else None
        )
        semantic = _element_semantics(element, region_label)
        occurrence = occurrences[semantic]
        occurrences[semantic] += 1
        identity = ("semantic", semantic, occurrence)
        signature.append((identity, *semantic))
    return tuple(sorted(signature, key=repr))


def element_progress_identity(
    snapshot: ViewportSnapshot, element: ElementSnapshot
) -> object:
    """Return stable safe identity for an action target across recaptures."""

    region_labels = {region.id: region.label for region in snapshot.regions}
    region_label = (
        region_labels.get(element.region_id) if element.region_id is not None else None
    )
    semantic = _element_semantics(element, region_label)
    occurrence = next(
        index
        for index, candidate in enumerate(
            candidate
            for candidate in snapshot.elements
            if _element_semantics(
                candidate,
                region_labels.get(candidate.region_id)
                if candidate.region_id is not None
                else None,
            )
            == semantic
        )
        if candidate is element
    )
    return (
        "semantic-target",
        *semantic,
        occurrence,
    )


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
    action_fingerprint: tuple[object, ...],
    after: ViewportSnapshot,
    current_url: str | None,
) -> TransitionProgressSignature:
    """Return a safe signature for an action and the state it produced."""

    return (
        action_fingerprint,
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
        if (
            len(history) >= length * 2
            and history[-length:] == history[-2 * length : -length]
        ):
            return length
    return None


def _element_semantics(
    element: ElementSnapshot, region_label: str | None
) -> tuple[object, ...]:
    return (
        str(element.role),
        element.label,
        _normalized_region_label(region_label),
        _visibility_bucket(element.visibility_fraction),
        element.actionable,
        element.disabled,
    )


def _normalized_region_label(label: str | None) -> str | None:
    return " ".join(label.split()).casefold() if label else None


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
