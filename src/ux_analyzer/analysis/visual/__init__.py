"""Visual UI-fundamental analysis package.

Detectors in this package judge rendered pages against UI design
fundamentals (white space, typography, scale, contrast, color, alignment,
visual hierarchy). They operate on a `VisualSnapshot`: a computed-style tree
captured from a rendered page (see `extract.py`), so analysis is
deterministic and testable without a browser.
"""

from ux_analyzer.analysis.visual.snapshot import (
    Box,
    Snapshot,
    SNode,
    load_snapshot_json,
)
from ux_analyzer.analysis.visual.types import FUNDAMENTALS, VisualIssue

__all__ = [
    "FUNDAMENTALS",
    "Box",
    "Snapshot",
    "SNode",
    "VisualIssue",
    "load_snapshot_json",
]
