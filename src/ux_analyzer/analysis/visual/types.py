"""Shared types for visual UI-fundamental detectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

FUNDAMENTALS = (
    "white-space",
    "contrast",
    "color",
    "typography",
    "scale",
    "alignment",
    "visual-hierarchy",
)

SEVERITY_ORDER = {"critical": 0, "medium": 1, "low": 2}


@dataclass(frozen=True, slots=True)
class VisualIssue:
    """An evidence-backed visual design-fundamental finding.

    `fundamental` is one of :data:`FUNDAMENTALS`. `element_refs` are stable
    node paths (see `Snapshot.selector`) so reports and overlays can point at
    the offending elements.
    """

    fundamental: str
    check_id: str
    title: str
    description: str
    severity: str  # critical | medium | low
    evidence: dict[str, Any] = field(default_factory=dict)
    element_refs: tuple[str, ...] = ()