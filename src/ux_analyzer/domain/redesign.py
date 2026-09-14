"""Immutable Design Proposal domain contracts (plan Task 1, ADR 0007).

Design Proposals are model estimates produced by the redesign pipeline. They
are deliberately independent of the run evidence corpus: they never cite
Evidence References and never claim run-evidence status (CONTEXT.md language:
Design Proposal, Redesign Attempt Status).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import cast

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_non_empty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _tuple_of_strings(values: object, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a collection of strings")
    normalized = tuple(cast(Iterable[object], values))
    return tuple(_require_non_empty(item, field_name) for item in normalized)


def _box_mapping(value: object, field_name: str) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    raw = {
        str(key): item
        for key, item in cast(Mapping[object, object], value).items()
    }
    if set(raw) != {"x", "y", "width", "height"}:
        raise ValueError(f"{field_name} must have exactly x, y, width, height keys")
    numbers: dict[str, float] = {}
    for key in ("x", "y", "width", "height"):
        item = raw[key]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{field_name}.{key} must be a number")
        number = float(item)
        if number != number or number in (float("inf"), float("-inf")):
            raise ValueError(f"{field_name}.{key} must be finite")
        if number < 0:
            raise ValueError(f"{field_name} must not contain negative values")
        numbers[key] = number
    return MappingProxyType(numbers)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DesignCategory(StrEnum):
    """The nine proposal families (plan Task 1 interfaces)."""

    GROUPING = "grouping"
    WHITESPACE = "whitespace"
    UNIFICATION = "unification"
    RADICAL_REDESIGN = "radical-redesign"
    COPY = "copy"
    RELOCATION = "relocation"
    SIMPLIFICATION = "simplification"
    NEW_SECTION = "new-section"
    ACCESSIBILITY = "accessibility"


class Impact(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Effort(StrEnum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class RedesignAttemptStatus(StrEnum):
    """Mirrors the Synthesis Status pattern for the redesign pipeline."""

    ACCEPTED = "accepted"
    NO_PROPOSALS = "no-proposals"
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"


DELIBERATE_CHOICE_CATEGORIES = frozenset(
    {
        DesignCategory.GROUPING,
        DesignCategory.UNIFICATION,
        DesignCategory.SIMPLIFICATION,
        DesignCategory.RELOCATION,
    }
)


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeliberateChoiceCheck:
    """Names a potentially-intentional pattern and why the proposal still stands."""

    pattern: str
    rationale: str

    def __post_init__(self) -> None:
        _require_non_empty(self.pattern, "pattern")
        _require_non_empty(self.rationale, "rationale")


@dataclass(frozen=True, slots=True)
class SectionReference:
    """Page-scoped reference into a persisted page capture."""

    url: str
    section_label: str
    box: Mapping[str, float]
    summary: str

    def __post_init__(self) -> None:
        _require_non_empty(self.url, "url")
        _require_non_empty(self.section_label, "section_label")
        _require_non_empty(self.summary, "summary")
        object.__setattr__(self, "box", _box_mapping(self.box, "box"))


@dataclass(frozen=True, slots=True)
class DesignProposal:
    """One schema-validated model-estimate design suggestion."""

    proposal_id: str
    page_url: str
    category: DesignCategory
    title: str
    observation: str
    rationale: str
    change: str
    principle_ids: tuple[str, ...]
    impact: Impact
    effort: Effort
    section_refs: tuple[SectionReference, ...]
    also_affects: tuple[str, ...] = ()
    deliberate_choice_check: DeliberateChoiceCheck | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.proposal_id, "proposal_id")
        _require_non_empty(self.page_url, "page_url")
        object.__setattr__(self, "category", DesignCategory(self.category))
        _require_non_empty(self.title, "title")
        _require_non_empty(self.observation, "observation")
        _require_non_empty(self.rationale, "rationale")
        _require_non_empty(self.change, "change")
        principles = _tuple_of_strings(self.principle_ids, "principle_ids")
        if len(principles) != len(set(principles)):
            raise ValueError("principle_ids must be unique")
        object.__setattr__(self, "principle_ids", principles)
        object.__setattr__(self, "impact", Impact(self.impact))
        object.__setattr__(self, "effort", Effort(self.effort))
        refs = tuple(self.section_refs)
        if not refs:
            raise ValueError("section_refs must contain at least one reference")
        for ref in refs:
            if not isinstance(ref, SectionReference):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError("section_refs must contain SectionReference values")
            if ref.url != self.page_url:
                raise ValueError(
                    "section_refs must reference the proposal's own page_url"
                )
        object.__setattr__(self, "section_refs", refs)
        object.__setattr__(
            self,
            "also_affects",
            _tuple_of_strings(self.also_affects, "also_affects"),
        )
        requires_check = self.category in DELIBERATE_CHOICE_CATEGORIES
        if requires_check and self.deliberate_choice_check is None:
            raise ValueError(
                "deliberate_choice_check is required for grouping, unification, "
                "simplification, and relocation proposals"
            )
        if not requires_check and self.deliberate_choice_check is not None:
            raise ValueError(
                "deliberate_choice_check must be None outside deliberate-choice "
                "categories"
            )


@dataclass(frozen=True, slots=True)
class KilledProposal:
    """A proposal removed by the critic/merger, preserved with its reason."""

    proposal_id: str
    reason: str

    def __post_init__(self) -> None:
        _require_non_empty(self.proposal_id, "proposal_id")
        _require_non_empty(self.reason, "reason")


@dataclass(frozen=True, slots=True)
class PageUnderstanding:
    """Per-page holistic reading; audience is always labeled an inference."""

    page_url: str
    intent: str
    audience_inference: str
    section_relationships: str

    def __post_init__(self) -> None:
        _require_non_empty(self.page_url, "page_url")
        _require_non_empty(self.intent, "intent")
        _require_non_empty(self.audience_inference, "audience_inference")
        _require_non_empty(self.section_relationships, "section_relationships")


def _tuple_of_proposals(values: object) -> tuple[DesignProposal, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("proposals must be a collection")
    normalized = tuple(cast(Iterable[object], values))
    for item in normalized:
        if not isinstance(item, DesignProposal):
            raise TypeError("proposals must contain DesignProposal values")
    return cast(tuple[DesignProposal, ...], normalized)


def _tuple_of_killed(values: object) -> tuple[KilledProposal, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("killed must be a collection")
    normalized = tuple(cast(Iterable[object], values))
    for item in normalized:
        if not isinstance(item, KilledProposal):
            raise TypeError("killed must contain KilledProposal values")
    return cast(tuple[KilledProposal, ...], normalized)


def _tuple_of_understanding(values: object) -> tuple[PageUnderstanding, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("page_understanding must be a collection")
    normalized = tuple(cast(Iterable[object], values))
    for item in normalized:
        if not isinstance(item, PageUnderstanding):
            raise TypeError("page_understanding must contain PageUnderstanding values")
    return cast(tuple[PageUnderstanding, ...], normalized)


@dataclass(frozen=True, slots=True)
class RedesignAttempt:
    """Immutable result of one redesign pass (mirrors SynthesisAttempt)."""

    attempt_id: str
    status: RedesignAttemptStatus
    proposals: tuple[DesignProposal, ...] = ()
    killed: tuple[KilledProposal, ...] = ()
    page_understanding: tuple[PageUnderstanding, ...] = ()
    consistency_notes: tuple[str, ...] = ()
    pack_version: str = ""
    audience: str = ""
    created_at: str | None = None
    unavailable_reason: str = ""
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.attempt_id, "attempt_id")
        object.__setattr__(self, "status", RedesignAttemptStatus(self.status))
        object.__setattr__(self, "proposals", _tuple_of_proposals(self.proposals))
        seen_ids: set[str] = set()
        for proposal in self.proposals:
            if proposal.proposal_id in seen_ids:
                raise ValueError("proposals must have unique proposal_id values")
            seen_ids.add(proposal.proposal_id)
        object.__setattr__(self, "killed", _tuple_of_killed(self.killed))
        object.__setattr__(
            self, "page_understanding", _tuple_of_understanding(self.page_understanding)
        )
        object.__setattr__(
            self,
            "consistency_notes",
            _tuple_of_strings(self.consistency_notes, "consistency_notes"),
        )
        if self.pack_version:
            _require_non_empty(self.pack_version, "pack_version")
        if self.audience:
            _require_non_empty(self.audience, "audience")
        if self.created_at is not None:
            _require_non_empty(self.created_at, "created_at")
        if self.status is RedesignAttemptStatus.UNAVAILABLE:
            _require_non_empty(
                self.unavailable_reason,
                "unavailable_reason (required when status is unavailable)",
            )
        if self.status is RedesignAttemptStatus.REJECTED:
            if not self.rejection_reasons:
                raise ValueError(
                    "rejection_reasons is required when status is rejected"
                )
            object.__setattr__(
                self,
                "rejection_reasons",
                _tuple_of_strings(self.rejection_reasons, "rejection_reasons"),
            )
        killed_ids = {killed.proposal_id for killed in self.killed}
        if killed_ids & seen_ids:
            raise ValueError(
                "killed proposals must not appear in published proposals"
            )


__all__ = [
    "DELIBERATE_CHOICE_CATEGORIES",
    "DeliberateChoiceCheck",
    "DesignCategory",
    "DesignProposal",
    "Effort",
    "Impact",
    "KilledProposal",
    "PageUnderstanding",
    "RedesignAttempt",
    "RedesignAttemptStatus",
    "SectionReference",
]
