"""Immutable contracts for evidence-grounded report synthesis."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from types import MappingProxyType

from ux_analyzer.domain.findings import (
    EvidenceClass,
    FindingSeverity,
    Reproducibility,
)


def _require_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _tuple_of_strings(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a collection of strings")
    normalized = tuple(values)
    for value in normalized:
        _require_non_empty(value, field_name)
    return normalized


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(_freeze_value(item) for item in value)
    return value


def _mapping_proxy(values: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(values, Mapping):
        raise TypeError("value must be a mapping")
    return MappingProxyType(
        {str(key): _freeze_value(value) for key, value in values.items()}
    )


def _float_mapping(values: Mapping[str, float]) -> Mapping[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError("usage must be a mapping")
    return MappingProxyType({str(key): float(value) for key, value in values.items()})


def _tuple_of_refs(
    values: Iterable[EvidenceRef], field_name: str
) -> tuple[EvidenceRef, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a collection of EvidenceRef values")
    normalized = tuple(values)
    if any(not isinstance(value, EvidenceRef) for value in normalized):
        raise TypeError(f"{field_name} must contain EvidenceRef values")
    return normalized


def _tuple_of_text_or_refs(
    values: Iterable[str | EvidenceRef], field_name: str
) -> tuple[str | EvidenceRef, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a collection")
    normalized = tuple(values)
    for value in normalized:
        if isinstance(value, EvidenceRef):
            continue
        _require_non_empty(value, field_name)
    return normalized


class SynthesisStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"
    NO_ISSUES = "no-issues"


class ObjectionSeverity(StrEnum):
    BLOCKING = "blocking"
    MATERIAL = "material"
    EDITORIAL = "editorial"


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """Stable reference to a persisted piece of run or visual evidence."""

    evidence_id: str
    kind: str
    run_id: str
    viewport_id: str | None = None
    element_id: str | None = None
    event_id: str | None = None
    metric_id: str | None = None
    artifact_path: str | None = None
    replay_sequence: int | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.evidence_id, "evidence ID")
        _require_non_empty(self.kind, "evidence kind")
        _require_non_empty(self.run_id, "run ID")
        for field_name in (
            "viewport_id",
            "element_id",
            "event_id",
            "metric_id",
            "sha256",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_non_empty(value, field_name)
        if self.artifact_path is not None:
            object.__setattr__(self, "artifact_path", str(self.artifact_path))
            _require_non_empty(self.artifact_path, "artifact_path")
        if self.replay_sequence is not None and self.replay_sequence < 0:
            raise ValueError("replay_sequence must not be negative")


@dataclass(frozen=True, slots=True)
class SynthesisFinding:
    """Reviewed, plain-language finding grounded in resolvable evidence."""

    finding_id: str
    title: str
    issue: str
    impact: str
    root_cause: str
    fixes: tuple[str, ...]
    severity: FindingSeverity | str
    confidence: float
    evidence_refs: tuple[EvidenceRef, ...]
    affected_surfaces: tuple[str, ...] = ()
    principles: tuple[str, ...] = ()
    counterevidence: tuple[str | EvidenceRef, ...] = ()
    limitations: tuple[str, ...] = ()
    reviewer_state: str = "pending"
    evidence_class: EvidenceClass | str = EvidenceClass.MODEL_ESTIMATE
    reproducibility: Reproducibility | str = Reproducibility.MODEL_DEPENDENT
    severity_justification: str = ""
    reviewer_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "finding_id",
            "title",
            "issue",
            "impact",
            "root_cause",
            "reviewer_state",
        ):
            _require_non_empty(getattr(self, field_name), field_name)

        fixes = _tuple_of_strings(self.fixes, "fixes")
        if not fixes:
            raise ValueError("finding requires at least one fix")

        evidence_refs = _tuple_of_refs(self.evidence_refs, "evidence_refs")
        if not evidence_refs:
            raise ValueError("finding requires evidence IDs")
        evidence_ids = tuple(reference.evidence_id for reference in evidence_refs)
        if any(not evidence_id.strip() for evidence_id in evidence_ids):
            raise ValueError("evidence ID must not be empty")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("finding contains duplicate evidence ID")

        confidence = float(self.confidence)
        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        object.__setattr__(self, "fixes", fixes)
        object.__setattr__(self, "severity", FindingSeverity(self.severity))
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "evidence_refs", evidence_refs)
        object.__setattr__(
            self,
            "affected_surfaces",
            _tuple_of_strings(self.affected_surfaces, "affected_surfaces"),
        )
        object.__setattr__(
            self, "principles", _tuple_of_strings(self.principles, "principles")
        )
        object.__setattr__(
            self,
            "counterevidence",
            _tuple_of_text_or_refs(self.counterevidence, "counterevidence"),
        )
        object.__setattr__(
            self, "limitations", _tuple_of_strings(self.limitations, "limitations")
        )
        object.__setattr__(self, "evidence_class", EvidenceClass(self.evidence_class))
        object.__setattr__(
            self,
            "reproducibility",
            Reproducibility(self.reproducibility),
        )
        if self.severity_justification:
            _require_non_empty(self.severity_justification, "severity_justification")
        object.__setattr__(
            self,
            "reviewer_notes",
            _tuple_of_strings(self.reviewer_notes, "reviewer_notes"),
        )


@dataclass(frozen=True, slots=True)
class SynthesisObjection:
    """Reviewer challenge that can block or qualify finding publication."""

    objection_id: str
    finding_id: str
    severity: ObjectionSeverity | str
    message: str
    evidence_refs: tuple[EvidenceRef, ...] = ()
    reviewer_role: str = ""
    resolved: bool = False
    resolution: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.objection_id, "objection ID")
        _require_non_empty(self.finding_id, "finding ID")
        _require_non_empty(self.message, "objection message")
        if self.reviewer_role:
            _require_non_empty(self.reviewer_role, "reviewer_role")
        if self.resolution is not None and self.resolution:
            _require_non_empty(self.resolution, "resolution")
        object.__setattr__(self, "severity", ObjectionSeverity(self.severity))
        object.__setattr__(
            self, "evidence_refs", _tuple_of_refs(self.evidence_refs, "evidence_refs")
        )


def _tuple_of_findings(
    values: Iterable[SynthesisFinding], field_name: str
) -> tuple[SynthesisFinding, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a collection of SynthesisFinding values")
    normalized = tuple(values)
    if any(not isinstance(value, SynthesisFinding) for value in normalized):
        raise TypeError(f"{field_name} must contain SynthesisFinding values")
    return normalized


def _tuple_of_objections(
    values: Iterable[SynthesisObjection], field_name: str
) -> tuple[SynthesisObjection, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(
            f"{field_name} must be a collection of SynthesisObjection values"
        )
    normalized = tuple(values)
    if any(not isinstance(value, SynthesisObjection) for value in normalized):
        raise TypeError(f"{field_name} must contain SynthesisObjection values")
    return normalized


@dataclass(frozen=True, slots=True)
class SynthesisAttempt:
    """Immutable, auditable result of one report-synthesis attempt."""

    attempt_id: str
    status: SynthesisStatus | str
    corpus_digest: str = ""
    expectation_digest: str = ""
    principle_pack_digest: str = ""
    model_manifest: Mapping[str, object] = field(default_factory=dict)
    role_manifest: Mapping[str, object] = field(default_factory=dict)
    prompt_version: str = ""
    schema_version: str = "synthesis-v1"
    retrieval_log: tuple[Mapping[str, object], ...] = ()
    usage: Mapping[str, float] = field(default_factory=dict)
    candidate_findings: tuple[SynthesisFinding, ...] = ()
    objections: tuple[SynthesisObjection, ...] = ()
    rejected_findings: tuple[SynthesisFinding, ...] = ()
    findings: tuple[SynthesisFinding, ...] = ()
    limitations: tuple[str, ...] = ()
    fallback_available: bool = True
    created_at: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.attempt_id, "attempt ID")
        _require_non_empty(self.schema_version, "schema_version")
        object.__setattr__(self, "status", SynthesisStatus(self.status))

        for field_name in (
            "corpus_digest",
            "expectation_digest",
            "principle_pack_digest",
            "prompt_version",
        ):
            value = getattr(self, field_name)
            if value:
                _require_non_empty(value, field_name)
        if self.created_at is not None:
            _require_non_empty(self.created_at, "created_at")

        object.__setattr__(self, "model_manifest", _mapping_proxy(self.model_manifest))
        object.__setattr__(self, "role_manifest", _mapping_proxy(self.role_manifest))
        object.__setattr__(self, "usage", _float_mapping(self.usage))
        object.__setattr__(
            self,
            "retrieval_log",
            tuple(_mapping_proxy(entry) for entry in self.retrieval_log),
        )
        object.__setattr__(
            self,
            "candidate_findings",
            _tuple_of_findings(self.candidate_findings, "candidate_findings"),
        )
        object.__setattr__(
            self,
            "objections",
            _tuple_of_objections(self.objections, "objections"),
        )
        object.__setattr__(
            self,
            "rejected_findings",
            _tuple_of_findings(self.rejected_findings, "rejected_findings"),
        )
        object.__setattr__(
            self,
            "findings",
            _tuple_of_findings(self.findings, "findings"),
        )
        object.__setattr__(
            self, "limitations", _tuple_of_strings(self.limitations, "limitations")
        )

    @property
    def final_findings(self) -> tuple[SynthesisFinding, ...]:
        """Alias used by artifact consumers for the published finding list."""

        return self.findings

    @property
    def candidates(self) -> tuple[SynthesisFinding, ...]:
        """Alias used by role orchestration for candidate findings."""

        return self.candidate_findings


__all__ = [
    "EvidenceRef",
    "ObjectionSeverity",
    "SynthesisAttempt",
    "SynthesisFinding",
    "SynthesisObjection",
    "SynthesisStatus",
]
