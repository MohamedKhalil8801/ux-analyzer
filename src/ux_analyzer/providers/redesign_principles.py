"""Versioned, static Redesign Principle Pack (plan Task 2, ADR 0007).

The pack is interpretive doctrine for the redesign pipeline: principles may
name or explain a design observation, never prove a problem exists. All
statements are original standalone language; Gestalt, Nielsen, lawsofux.com,
and WCAG 2.2 materials are cited as sources, never quoted at length. The pack
performs no I/O and is immutable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

REDESIGN_PRINCIPLE_PACK_VERSION = "redesign-principles-2026-09"


@dataclass(frozen=True)
class RedesignPrinciple:
    """One interpretive design principle (id, name, statement, source)."""

    id: str
    name: str
    statement: str
    source: str


def _principle(
    principle_id: str,
    name: str,
    statement: str,
    source: str,
) -> RedesignPrinciple:
    return RedesignPrinciple(
        id=principle_id,
        name=name,
        statement=statement,
        source=source,
    )


_REDESIGN_PRINCIPLES: tuple[RedesignPrinciple, ...] = (
    _principle(
        "gestalt-proximity",
        "Proximity",
        "Elements placed close together are perceived as belonging to one group, so physical distance should reflect actual relationships between content and actions.",
        "Gestalt principles of perceptual organization; see also lawsofux.com (proximity)",
    ),
    _principle(
        "gestalt-similarity",
        "Similarity",
        "Elements that share visual attributes such as color, shape, or size are read as related even when they are not adjacent, so shared styling should carry shared meaning.",
        "Gestalt principles of perceptual organization; see also lawsofux.com (similarity)",
    ),
    _principle(
        "gestalt-common-region",
        "Common Region",
        "Elements inside a shared boundary or background are perceived as one group, so enclosing surfaces should group things that genuinely belong together.",
        "Gestalt principles of perceptual organization; see also lawsofux.com (common region)",
    ),
    _principle(
        "nielsen-consistency",
        "Consistency and Standards",
        "Using the same visual treatment, wording, and placement for the same meaning across a product lets people transfer what they have already learned.",
        "Nielsen Norman Group usability heuristics; see also lawsofux.com (consistency)",
    ),
    _principle(
        "nielsen-recognition",
        "Recognition Rather than Recall",
        "Making objects, actions, and options visible reduces the memory burden of remembering them from one screen to the next.",
        "Nielsen Norman Group usability heuristics",
    ),
    _principle(
        "nielsen-aesthetic-minimalist",
        "Aesthetic and Minimalist Design",
        "Screens communicate best when every element present either supports the user's goal or is removed, because irrelevant detail competes with relevant information.",
        "Nielsen Norman Group usability heuristics",
    ),
    _principle(
        "wcag-contrast-1.4.3",
        "Contrast (Minimum)",
        "Text should meet the minimum contrast ratio of 4.5:1 against its background (3:1 for large text) so that it stays readable for people with low vision or on poor displays.",
        "WCAG 2.2 success criterion 1.4.3",
    ),
    _principle(
        "wcag-target-size-2.5.8",
        "Target Size (Minimum)",
        "Interactive targets need enough physical size, with adequate spacing, so that people can select them reliably regardless of input method or motor precision.",
        "WCAG 2.2 success criterion 2.5.8",
    ),
    _principle(
        "wcag-labels-3.3.2",
        "Labels or Instructions",
        "Inputs need visible labels or instructions so people know what information is expected before they commit an entry.",
        "WCAG 2.2 success criterion 3.3.2",
    ),
    _principle(
        "wcag-reflow-1.4.10",
        "Reflow",
        "Content should reflow to a single column at narrow widths without losing information or requiring two-dimensional scrolling.",
        "WCAG 2.2 success criterion 1.4.10",
    ),
    _principle(
        "wcag-heading-order-1.3.1",
        "Info and Relationships",
        "Programmatic and visual structure such as heading order, lists, and label associations should match what the layout communicates visually.",
        "WCAG 2.2 success criterion 1.3.1",
    ),
    _principle(
        "copy-tone-fit",
        "Tone Fits the Moment",
        "Interface copy should match the emotional stakes of the moment and speak in the user's vocabulary, avoiding jargon and unnecessary cheerfulness.",
        "Content design practice; see also lawsofux.com (mental models) for vocabulary alignment",
    ),
    _principle(
        "copy-clarity-over-cleverness",
        "Clarity Over Cleverness",
        "Buttons and links should describe the outcome of the action in specific verbs rather than brand voice or puns, so people can predict what happens next.",
        "Content design practice",
    ),
    _principle(
        "hierarchy-f-pattern",
        "Scannable Hierarchy",
        "People scan pages in patterns anchored at the top and leading edges, so the most important information and actions belong where scanning starts and weight should step down deliberately.",
        "Scanning behavior research (F-pattern, Nielsen Norman Group); see also lawsofux.com (serial position effect)",
    ),
    _principle(
        "progressive-disclosure",
        "Progressive Disclosure",
        "Showing the primary path first and revealing advanced or secondary options on demand keeps first impressions manageable without hiding capability.",
        "Interaction design practice; see also lawsofux.com (progressive disclosure)",
    ),
    _principle(
        "whitespace-breathing-room",
        "Whitespace as Structure",
        "Empty space is not wasted space: consistent spacing separates and joins content, signals importance, and gives dense pages a readable rhythm.",
        "Visual design practice; see also lawsofux.com (proximity)",
    ),
    _principle(
        "visual-hierarchy-scale",
        "Scale Carries Meaning",
        "Size, weight, and color steps should form a deliberate hierarchy so that the eye reads importance before detail.",
        "Visual design practice",
    ),
    _principle(
        "structural-reuse-patterns",
        "Reuse Established Patterns",
        "A radical departure should be justified by a concrete user problem, because familiar structural patterns carry a learning cost when broken.",
        "Interaction design practice; see also lawsofux.com (Jakob's law)",
    ),
)

_REDESIGN_PRINCIPLES_BY_ID: dict[str, RedesignPrinciple] = {
    principle.id: principle for principle in _REDESIGN_PRINCIPLES
}


def redesign_principle_pack() -> tuple[RedesignPrinciple, ...]:
    """Return the immutable pack in stable declaration order (no I/O)."""

    return _REDESIGN_PRINCIPLES


def redesign_principle_ids() -> tuple[str, ...]:
    """Return the pack's principle ids in declaration order."""

    return tuple(_REDESIGN_PRINCIPLES_BY_ID)


def is_known_redesign_principle_id(principle_id: str) -> bool:
    """Return whether ``principle_id`` names a principle in the pack."""

    return principle_id in _REDESIGN_PRINCIPLES_BY_ID


def redesign_principle_digest() -> str:
    """Return the SHA-256 of canonical ASCII JSON for this pack version."""

    payload = {
        "principles": [
            {
                "id": principle.id,
                "name": principle.name,
                "statement": principle.statement,
                "source": principle.source,
            }
            for principle in sorted(
                _REDESIGN_PRINCIPLES, key=lambda item: item.id
            )
        ],
        "version": REDESIGN_PRINCIPLE_PACK_VERSION,
    }
    canonical_json = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(canonical_json).hexdigest()


__all__ = [
    "REDESIGN_PRINCIPLE_PACK_VERSION",
    "RedesignPrinciple",
    "is_known_redesign_principle_id",
    "redesign_principle_digest",
    "redesign_principle_ids",
    "redesign_principle_pack",
]
