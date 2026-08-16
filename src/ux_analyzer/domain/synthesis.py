"""Immutable contracts for evidence-grounded report synthesis."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import cast

from ux_analyzer.domain.findings import (
    EvidenceClass,
    FindingSeverity,
    Reproducibility,
    UnsupportedHumanClaimError,
)


def _require_non_empty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _tuple_of_strings(values: object, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a collection of strings")
    normalized = tuple(cast(Iterable[object], values))
    return tuple(_require_non_empty(value, field_name) for value in normalized)


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return MappingProxyType(
            {str(key): _freeze_value(item) for key, item in mapping.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in cast(Iterable[object], value))
    if isinstance(value, (set, frozenset)):
        return tuple(_freeze_value(item) for item in cast(Iterable[object], value))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported metadata value: {type(value).__name__}")


def _mapping_proxy(values: object) -> Mapping[str, object]:
    if not isinstance(values, Mapping):
        raise TypeError("value must be a mapping")
    mapping = cast(Mapping[object, object], values)
    return MappingProxyType(
        {str(key): _freeze_value(value) for key, value in mapping.items()}
    )


def _float_mapping(values: object) -> Mapping[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError("usage must be a mapping")
    mapping = cast(Mapping[object, object], values)
    normalized: dict[str, float] = {}
    for key, value in mapping.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("usage values must be numeric")
        normalized[str(key)] = float(value)
    return MappingProxyType(normalized)


def _tuple_of_refs(values: object, field_name: str) -> tuple[EvidenceRef, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a collection of EvidenceRef values")
    normalized = tuple(cast(Iterable[object], values))
    if any(not isinstance(value, EvidenceRef) for value in normalized):
        raise TypeError(f"{field_name} must contain EvidenceRef values")
    return cast(tuple[EvidenceRef, ...], normalized)


def _tuple_of_text_or_refs(
    values: object, field_name: str
) -> tuple[str | EvidenceRef, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a collection")
    normalized = tuple(cast(Iterable[object], values))
    for value in normalized:
        if isinstance(value, EvidenceRef):
            continue
        _require_non_empty(value, field_name)
    return cast(tuple[str | EvidenceRef, ...], normalized)


class SynthesisStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"
    NO_ISSUES = "no-issues"


REPORT_ADJUDICATOR_ROLE = "report-adjudicator"
CANONICAL_SYNTHESIS_ROLES = (
    "report-analyst",
    "report-evidence-auditor",
    "report-pattern-reviewer",
    REPORT_ADJUDICATOR_ROLE,
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_AUDIT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_REASON_CODE_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class ObjectionSeverity(StrEnum):
    BLOCKING = "blocking"
    MATERIAL = "material"
    EDITORIAL = "editorial"


def _sha256_digest(value: object, field_name: str) -> str:
    digest = _require_non_empty(value, field_name)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return digest


@dataclass(frozen=True, slots=True)
class SynthesisRoleReceipt:
    """Digest-bound proof that one canonical synthesis role completed."""

    role: str
    provider_id: str
    model_id: str
    prompt_digest: str
    schema_digest: str
    output_digest: str

    def __post_init__(self) -> None:
        role = _require_non_empty(self.role, "role")
        if role not in CANONICAL_SYNTHESIS_ROLES:
            raise ValueError("role receipt must use a canonical synthesis role")
        provider_id = _require_non_empty(self.provider_id, "provider_id")
        model_id = _require_non_empty(self.model_id, "model_id")
        if provider_id == "unavailable" or model_id == "unavailable":
            raise ValueError("role receipt requires provider_id and model_id provenance")
        _sha256_digest(self.prompt_digest, "prompt_digest")
        _sha256_digest(self.schema_digest, "schema_digest")
        _sha256_digest(self.output_digest, "output_digest")


@dataclass(frozen=True, slots=True)
class RejectedCandidateAudit:
    """Sanitized identity and disposition for an invalid analyst candidate."""

    finding_id: str
    source_role: str
    reason_code: str
    output_digest: str

    def __post_init__(self) -> None:
        finding_id = _require_non_empty(self.finding_id, "finding_id")
        if _AUDIT_ID_PATTERN.fullmatch(finding_id) is None:
            raise ValueError("finding_id must be a bounded audit identifier")
        if self.source_role != "report-analyst":
            raise ValueError("rejected candidate source_role must be report-analyst")
        reason_code = _require_non_empty(self.reason_code, "reason_code")
        if _REASON_CODE_PATTERN.fullmatch(reason_code) is None:
            raise ValueError("reason_code must be a lowercase hyphenated code")
        _sha256_digest(self.output_digest, "output_digest")


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
            _require_non_empty(self.artifact_path, "artifact_path")
        replay_sequence: object = self.replay_sequence
        if replay_sequence is not None:
            if type(replay_sequence) is not int:
                raise TypeError("replay_sequence must be an exact integer")
            if replay_sequence < 0:
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
        evidence_class = EvidenceClass(self.evidence_class)
        if evidence_class is EvidenceClass.UNSUPPORTED_HUMAN_CLAIM:
            raise UnsupportedHumanClaimError(
                "unsupported human claim cannot become finding"
            )
        object.__setattr__(self, "evidence_class", evidence_class)
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
    objection_type: str = "other"
    evidence_refs: tuple[EvidenceRef, ...] = ()
    reviewer_role: str = ""
    resolved: bool = False
    resolution: str | None = None
    resolved_by_role: str | None = None
    resolution_evidence_refs: tuple[EvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.objection_id, "objection ID")
        _require_non_empty(self.finding_id, "finding ID")
        objection_type = _require_non_empty(self.objection_type, "objection_type")
        if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", objection_type) is None:
            raise ValueError("objection_type must be a lowercase hyphenated code")
        _require_non_empty(self.message, "objection message")
        if self.reviewer_role:
            _require_non_empty(self.reviewer_role, "reviewer_role")
        if self.resolved and self.resolution is None:
            raise ValueError("resolution must not be empty")
        if self.resolution is not None:
            _require_non_empty(self.resolution, "resolution")
        if self.resolved_by_role is not None:
            _require_non_empty(self.resolved_by_role, "resolved_by_role")
        object.__setattr__(self, "severity", ObjectionSeverity(self.severity))
        object.__setattr__(
            self, "evidence_refs", _tuple_of_refs(self.evidence_refs, "evidence_refs")
        )
        resolution_refs = _tuple_of_refs(
            self.resolution_evidence_refs,
            "resolution_evidence_refs",
        )
        resolution_evidence_ids = tuple(ref.evidence_id for ref in resolution_refs)
        if len(resolution_evidence_ids) != len(set(resolution_evidence_ids)):
            raise ValueError("resolution contains duplicate evidence ID")
        object.__setattr__(self, "resolution_evidence_refs", resolution_refs)


def final_finding_preserves_candidate(
    final: SynthesisFinding,
    candidate: SynthesisFinding,
    *,
    objections: Iterable[SynthesisObjection] = (),
) -> bool:
    """Return whether a final preserves reviewed claim and supporting evidence."""

    if final.finding_id != candidate.finding_id:
        return False
    candidate_evidence_ids = {ref.evidence_id for ref in candidate.evidence_refs}
    final_evidence_ids = {ref.evidence_id for ref in final.evidence_refs}
    if not candidate_evidence_ids <= final_evidence_ids:
        return False

    reviewed_fields = (
        "issue",
        "impact",
        "root_cause",
        "fixes",
        "severity",
        "confidence",
        "affected_surfaces",
        "principles",
        "counterevidence",
        "severity_justification",
    )
    changed_fields = {
        field_name
        for field_name in reviewed_fields
        if getattr(final, field_name) != getattr(candidate, field_name)
    }
    if not changed_fields:
        return True

    authorization_types = {
        "title": frozenset(
            {"affected-surface", "citation-accuracy", "visual-interpretation"}
        ),
        "issue": frozenset(
            {
                "affected-surface",
                "citation-accuracy",
                "factual-support",
                "severity",
                "visual-interpretation",
            }
        ),
        "impact": frozenset(
            {
                "affected-surface",
                "citation-accuracy",
                "factual-support",
                "severity",
            }
        ),
        "root_cause": frozenset(
            {
                "affected-surface",
                "citation-accuracy",
                "factual-support",
                "fix-leverage",
                "visual-interpretation",
            }
        ),
        "fixes": frozenset({"fix-leverage"}),
        "severity": frozenset({"severity"}),
        "confidence": frozenset(
            {
                "citation-accuracy",
                "contradiction",
                "counterexample",
                "factual-support",
                "severity",
                "visual-interpretation",
            }
        ),
        "affected_surfaces": frozenset(
            {"affected-surface", "recurrence", "shared-cause"}
        ),
        "principles": frozenset({"recurrence", "shared-cause"}),
        "counterevidence": frozenset({"contradiction", "counterexample"}),
        "severity_justification": frozenset(
            {
                "citation-accuracy",
                "factual-support",
                "severity",
                "visual-interpretation",
            }
        ),
    }
    relevant = tuple(
        objection
        for objection in objections
        if objection.finding_id == candidate.finding_id
        and objection.resolved
        and objection.resolution is not None
        and objection.resolved_by_role == REPORT_ADJUDICATOR_ROLE
        and objection.resolution_evidence_refs
    )
    return all(
        any(
            objection.objection_type in authorization_types[field_name]
            for objection in relevant
        )
        for field_name in changed_fields
    )


def _tuple_of_findings(values: object, field_name: str) -> tuple[SynthesisFinding, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a collection of SynthesisFinding values")
    normalized = tuple(cast(Iterable[object], values))
    if any(not isinstance(value, SynthesisFinding) for value in normalized):
        raise TypeError(f"{field_name} must contain SynthesisFinding values")
    return cast(tuple[SynthesisFinding, ...], normalized)


def _tuple_of_objections(
    values: object, field_name: str
) -> tuple[SynthesisObjection, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(
            f"{field_name} must be a collection of SynthesisObjection values"
        )
    normalized = tuple(cast(Iterable[object], values))
    if any(not isinstance(value, SynthesisObjection) for value in normalized):
        raise TypeError(f"{field_name} must contain SynthesisObjection values")
    return cast(tuple[SynthesisObjection, ...], normalized)


def _tuple_of_role_receipts(values: object) -> tuple[SynthesisRoleReceipt, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("role_receipts must be a collection")
    normalized = tuple(cast(Iterable[object], values))
    if any(not isinstance(value, SynthesisRoleReceipt) for value in normalized):
        raise TypeError("role_receipts must contain SynthesisRoleReceipt values")
    return cast(tuple[SynthesisRoleReceipt, ...], normalized)


def _tuple_of_rejected_candidate_audits(
    values: object,
) -> tuple[RejectedCandidateAudit, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("rejected_candidate_audits must be a collection")
    normalized = tuple(cast(Iterable[object], values))
    if any(not isinstance(value, RejectedCandidateAudit) for value in normalized):
        raise TypeError(
            "rejected_candidate_audits must contain RejectedCandidateAudit values"
        )
    return cast(tuple[RejectedCandidateAudit, ...], normalized)


@dataclass(frozen=True, slots=True)
class SynthesisAttempt:
    """Immutable, auditable result of one report-synthesis attempt."""

    attempt_id: str
    status: SynthesisStatus | str
    corpus_digest: str = ""
    expectation_digest: str = ""
    principle_pack_digest: str = ""
    model_manifest: Mapping[str, object] = field(
        default_factory=lambda: dict[str, object]()
    )
    role_manifest: Mapping[str, object] = field(
        default_factory=lambda: dict[str, object]()
    )
    prompt_version: str = ""
    schema_version: str = "synthesis-v2"
    retrieval_log: tuple[Mapping[str, object], ...] = ()
    usage: Mapping[str, float] = field(default_factory=lambda: dict[str, float]())
    role_receipts: tuple[SynthesisRoleReceipt, ...] = ()
    rejected_candidate_audits: tuple[RejectedCandidateAudit, ...] = ()
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
            "role_receipts",
            _tuple_of_role_receipts(self.role_receipts),
        )
        object.__setattr__(
            self,
            "rejected_candidate_audits",
            _tuple_of_rejected_candidate_audits(self.rejected_candidate_audits),
        )
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
    "CANONICAL_SYNTHESIS_ROLES",
    "EvidenceRef",
    "ObjectionSeverity",
    "REPORT_ADJUDICATOR_ROLE",
    "RejectedCandidateAudit",
    "SynthesisAttempt",
    "SynthesisFinding",
    "SynthesisObjection",
    "SynthesisRoleReceipt",
    "SynthesisStatus",
    "final_finding_preserves_candidate",
]
