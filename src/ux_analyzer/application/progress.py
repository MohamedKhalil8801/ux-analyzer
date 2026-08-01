"""Safe semantic progress detection for recaptured interface snapshots."""

from __future__ import annotations

from collections import Counter

from ux_analyzer.domain.interface import ElementSnapshot, ViewportSnapshot

SemanticSnapshotSignature = tuple[tuple[object, ...], ...]


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
