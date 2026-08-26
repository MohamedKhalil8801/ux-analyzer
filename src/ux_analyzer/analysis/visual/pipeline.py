"""Aggregate visual detectors across all UI-fundamental pieces."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ux_analyzer.analysis.visual.snapshot import Snapshot
from ux_analyzer.analysis.visual.types import FUNDAMENTALS, VisualIssue

Analyzer = Callable[[Snapshot], Sequence[VisualIssue]]

_PIECES: list[tuple[str, Analyzer]] = []  # populated by register_piece


def register_piece(name: str, analyzer: Analyzer) -> None:
    """Add a piece analyzer to the pipeline (idempotent by name)."""
    if any(existing == name for existing, _ in _PIECES):
        return
    _PIECES.append((name, analyzer))


def analyze_snapshot(snapshot: Snapshot) -> list[VisualIssue]:
    """Run every registered piece and merge issues (sorted for stability)."""
    issues: list[VisualIssue] = []
    for _, analyze in tuple(_PIECES):
        try:
            issues.extend(analyze(snapshot))
        except Exception:  # noqa: BLE001 - one piece must not kill the rest
            continue
    issues.sort(key=lambda i: (i.fundamental, i.severity, i.check_id))
    return issues


def fundamentals_flagged(issues: Sequence[VisualIssue]) -> set[str]:
    """Map issues to the set of fundamentals a reviewer would flag."""
    flagged: set[str] = set()
    for issue in issues:
        if issue.fundamental in FUNDAMENTALS:
            flagged.add(issue.fundamental)
    return flagged


def _load_default_pieces() -> None:
    from ux_analyzer.analysis.visual import (  # noqa: PLC0415
        color,
        hierarchy,
        spacing,
        typography,
    )

    register_piece("spacing", spacing.analyze_spacing)
    register_piece("typography", typography.analyze_typography)
    register_piece("color", color.analyze_color)
    register_piece("hierarchy", hierarchy.analyze_hierarchy)


_load_default_pieces()
