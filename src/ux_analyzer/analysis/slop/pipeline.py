"""Aggregate slop analysis: run patterns + copy patterns + scoring.

Produces the machine-readable slop report used by the CLI, the live-page
audit, and the bench harness. Shape mirrors the reference CLI JSON so outputs
compare field-for-field.
"""

from __future__ import annotations

from typing import Any

from ux_analyzer.analysis.slop.copy_patterns import run_copy_patterns
from ux_analyzer.analysis.slop.patterns import run_patterns
from ux_analyzer.analysis.slop.scoring import (
    combine_axes,
    score_copy,
    score_patterns,
)
from ux_analyzer.analysis.visual.snapshot import Snapshot


def analyze_slop(
    snap: Snapshot,
    *,
    viewport_w: int = 1440,
    viewport_h: int = 900,
    doc_height: int | None = None,
    scroll_y: int = 0,
    text_context: dict[str, Any] | None = None,
    surface: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run the full slop detection over a captured snapshot.

    Returns a dict with ``url``-independent fields:

    - ``score`` / ``tier`` / ``grade`` / ``verdict`` / ``patternsFlagged``
    - ``patterns``: 27 rows with per-pattern evidence
    - ``copy``: the 9-pattern copy axis summary + rows
    - ``unified*``: combined multi-axis result
    """
    from ux_analyzer.analysis.slop.context import build_context

    ctx = build_context(
        snap,
        viewport_w=viewport_w,
        viewport_h=viewport_h,
        doc_height=doc_height or 0,
        scroll_y=scroll_y,
        text_context=text_context,
        surface=surface,
    )
    rows = run_patterns(ctx)
    summary = score_patterns(rows)

    copy_rows = run_copy_patterns(text_context)
    copy_summary = score_copy(copy_rows, (text_context or {}).get("wordCount") or 0)
    copy_summary["patterns"] = copy_rows

    axes = combine_axes({"design": summary, "copy": copy_summary})

    return {
        **summary,
        "patterns": rows,
        "copy": copy_summary,
        **axes,
    }
