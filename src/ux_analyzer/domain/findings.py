"""Evidence-typed metrics and findings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EvidenceClass(StrEnum):
    """Trust classes allowed in benchmark output."""

    DETERMINISTIC_FACT = "deterministic-fact"
    MODEL_ESTIMATE = "model-estimate"
    UNSUPPORTED_HUMAN_CLAIM = "unsupported-human-claim"


class UnsupportedHumanClaimError(ValueError):
    """Raised when unsupported human claims are promoted to findings."""


class FindingSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Reproducibility(StrEnum):
    SEEDED = "seeded"
    REPRODUCIBLE = "reproducible"
    MODEL_DEPENDENT = "model-dependent"
    NOT_REPRODUCIBLE = "not-reproducible"


@dataclass(frozen=True, slots=True)
class Evidence:
    """Evidence record that can be referenced by a finding."""

    evidence_id: str
    evidence_class: EvidenceClass | str
    description: str
    source_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.description:
            raise ValueError("evidence ID and description are required")
        object.__setattr__(self, "evidence_class", EvidenceClass(self.evidence_class))
        object.__setattr__(self, "source_event_ids", tuple(self.source_event_ids))


@dataclass(frozen=True, slots=True)
class Metric:
    """Named numeric measurement with explicit evidence class."""

    name: str
    value: float
    evidence_class: EvidenceClass | str
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("metric name must not be empty")
        object.__setattr__(self, "evidence_class", EvidenceClass(self.evidence_class))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))


@dataclass(frozen=True, slots=True)
class Finding:
    """Evidence-backed observation; unsupported human claims cannot enter."""

    finding_id: str
    category: str
    severity: FindingSeverity | str
    reproducibility: Reproducibility | str
    evidence_class: EvidenceClass | str
    evidence_ids: tuple[str, ...]
    limitations: tuple[str, ...]
    generated_explanation: str | None = None

    def __post_init__(self) -> None:
        if not self.finding_id or not self.category:
            raise ValueError("finding ID and category are required")
        evidence_class = EvidenceClass(self.evidence_class)
        if evidence_class is EvidenceClass.UNSUPPORTED_HUMAN_CLAIM:
            raise UnsupportedHumanClaimError(
                "unsupported human claim cannot become finding"
            )
        evidence_ids = tuple(self.evidence_ids)
        if not evidence_ids:
            raise ValueError("finding requires evidence IDs")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("finding contains duplicate evidence ID")
        object.__setattr__(self, "severity", FindingSeverity(self.severity))
        object.__setattr__(
            self, "reproducibility", Reproducibility(self.reproducibility)
        )
        object.__setattr__(self, "evidence_class", evidence_class)
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(self, "limitations", tuple(self.limitations))
