"""Neutral contracts shared by report-synthesis application and providers."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ux_analyzer.domain.findings import EvidenceClass, FindingSeverity, Reproducibility
from ux_analyzer.domain.synthesis import (
    EvidenceRef,
    ObjectionSeverity,
    SynthesisFinding,
    SynthesisObjection,
)
from ux_analyzer.ports.models import ModelManifest

REPORT_SYNTHESIS_SCHEMA_VERSION = "report-synthesis-v1"

_OBJECTION_TYPES = {
    "affected-surface",
    "citation-accuracy",
    "contradiction",
    "counterexample",
    "factual-support",
    "fix-leverage",
    "recurrence",
    "severity",
    "shared-cause",
    "visual-interpretation",
    "other",
}

_SENSITIVE_KEY_MARKERS = (
    "private_reasoning",
    "prior_agent",
    "prior_finding",
    "finding_prose",
    "raw_response",
    "raw_prompt",
    "chat_history",
    "conversation_history",
    "chain_of_thought",
    "decision_rationale",
)

FORBIDDEN_NARRATIVE_MARKERS = (
    "chain of thought",
    "private reasoning",
    "prior agent",
    "prior finding",
    "finding prose",
    "raw model response",
    "raw response",
    "raw prompt",
    "system prompt",
    "chat history",
    "conversation history",
    "decision rationale",
    "existing finding prose",
)


def _canonical_narrative_text(value: str) -> str:
    return re.sub(r"[\s_-]+", " ", value.casefold()).strip()


def contains_forbidden_narrative(value: str) -> bool:
    """Return whether text attempts to carry excluded model narrative."""

    normalized = _canonical_narrative_text(value)
    return any(
        _canonical_narrative_text(marker) in normalized
        for marker in FORBIDDEN_NARRATIVE_MARKERS
    )


def redact_forbidden_narrative(value: str) -> str:
    """Replace excluded narrative with a stable redaction marker."""

    return "[redacted]" if contains_forbidden_narrative(value) else value


def is_sensitive_key(value: object) -> bool:
    """Return whether a mapping key belongs to excluded model narrative."""

    normalized = _canonical_narrative_text(str(value))
    return any(
        _canonical_narrative_text(marker) in normalized
        for marker in _SENSITIVE_KEY_MARKERS
    ) or contains_forbidden_narrative(str(value))


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _text_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a collection of strings")
    normalized = tuple(values)
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    for value in normalized:
        _require_text(value, field_name)
    return normalized


@dataclass(frozen=True, slots=True)
class UxPrinciple:
    """Immutable interpretive lens for evidence-backed UX analysis."""

    principle_id: str
    name: str
    explanation: str
    diagnostic_questions: tuple[str, ...]
    misuse_warning: str
    applicability_cues: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.principle_id, "principle_id")
        _require_text(self.name, "name")
        _require_text(self.explanation, "explanation")
        _require_text(self.misuse_warning, "misuse_warning")
        object.__setattr__(
            self,
            "diagnostic_questions",
            _text_tuple(self.diagnostic_questions, "diagnostic_questions"),
        )
        object.__setattr__(
            self,
            "applicability_cues",
            _text_tuple(self.applicability_cues, "applicability_cues"),
        )


class _RoleSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceReference(_RoleSchema):
    """Transport form of the domain ``EvidenceRef`` contract."""

    model_config = ConfigDict(strict=True)

    evidence_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    viewport_id: str | None = None
    element_id: str | None = None
    event_id: str | None = None
    metric_id: str | None = None
    artifact_path: str | None = None
    replay_sequence: int | None = Field(default=None, ge=0)
    sha256: str | None = None

    def to_domain(self) -> EvidenceRef:
        return EvidenceRef(**self.model_dump(mode="python"))


def _new_evidence_references() -> list[EvidenceReference]:
    return []


def _new_counterevidence() -> list[str | EvidenceReference]:
    return []


class CandidateFinding(_RoleSchema):
    """Candidate or final finding fields shared by synthesis role responses."""

    finding_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    issue: str = Field(min_length=1)
    impact: str = Field(min_length=1)
    root_cause: str = Field(min_length=1)
    fixes: list[str] = Field(min_length=1)
    severity: FindingSeverity
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_refs: list[EvidenceReference] = Field(min_length=1)
    affected_surfaces: list[str] = Field(default_factory=list)
    principles: list[str] = Field(default_factory=list)
    counterevidence: list[str | EvidenceReference] = Field(
        default_factory=_new_counterevidence
    )
    limitations: list[str] = Field(default_factory=list)
    reviewer_state: str = Field(default="candidate", min_length=1)
    evidence_class: EvidenceClass = EvidenceClass.MODEL_ESTIMATE
    reproducibility: Reproducibility = Reproducibility.MODEL_DEPENDENT
    severity_justification: str = ""
    reviewer_notes: list[str] = Field(default_factory=list)

    @field_validator(
        "fixes",
        "affected_surfaces",
        "principles",
        "limitations",
        "reviewer_notes",
        mode="before",
    )
    @classmethod
    def _normalize_text_lists(cls, value: object) -> object:
        if value is None:
            return []
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError("finding text fields must be lists of strings")
        return [
            item.strip() if isinstance(item, str) else item
            for item in cast(Sequence[object], value)
        ]

    @field_validator(
        "fixes", "affected_surfaces", "principles", "limitations", "reviewer_notes"
    )
    @classmethod
    def _reject_empty_text(cls, value: list[str]) -> list[str]:
        if any(not item for item in value):
            raise ValueError("finding text fields must not contain empty values")
        return value

    @field_validator("evidence_refs")
    @classmethod
    def _reject_duplicate_refs(
        cls, value: list[EvidenceReference]
    ) -> list[EvidenceReference]:
        evidence_ids = [item.evidence_id for item in value]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("finding contains duplicate evidence ID")
        return value

    def to_domain(self, *, reviewer_state: str | None = None) -> SynthesisFinding:
        values = self.model_dump(mode="python")
        values["fixes"] = tuple(self.fixes)
        values["evidence_refs"] = tuple(ref.to_domain() for ref in self.evidence_refs)
        values["affected_surfaces"] = tuple(self.affected_surfaces)
        values["principles"] = tuple(self.principles)
        values["counterevidence"] = tuple(
            item.to_domain() if isinstance(item, EvidenceReference) else item
            for item in self.counterevidence
        )
        values["limitations"] = tuple(self.limitations)
        values["reviewer_notes"] = tuple(self.reviewer_notes)
        if reviewer_state is not None:
            values["reviewer_state"] = reviewer_state
        return SynthesisFinding(**values)


class TypedObjection(_RoleSchema):
    """Typed challenge emitted by an evidence or pattern reviewer."""

    objection_id: str = Field(min_length=1)
    finding_id: str = Field(min_length=1)
    objection_type: str = Field(default="other", min_length=1)
    severity: ObjectionSeverity
    message: str = Field(min_length=1)
    evidence_refs: list[EvidenceReference] = Field(
        default_factory=_new_evidence_references
    )
    reviewer_role: str = ""
    resolved: bool = False
    resolution: str | None = None

    @field_validator("objection_type")
    @classmethod
    def _normalize_objection_type(cls, value: str) -> str:
        normalized = value.strip().lower().replace("_", "-")
        if normalized not in _OBJECTION_TYPES:
            raise ValueError("objection_type is not a supported typed objection")
        return normalized

    @field_validator("evidence_refs")
    @classmethod
    def _reject_duplicate_objection_refs(
        cls, value: list[EvidenceReference]
    ) -> list[EvidenceReference]:
        evidence_ids = [item.evidence_id for item in value]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("objection contains duplicate evidence ID")
        return value

    def to_domain(self) -> SynthesisObjection:
        return SynthesisObjection(
            objection_id=self.objection_id,
            finding_id=self.finding_id,
            severity=self.severity,
            message=self.message,
            evidence_refs=tuple(ref.to_domain() for ref in self.evidence_refs),
            reviewer_role=self.reviewer_role,
            resolved=self.resolved,
            resolution=self.resolution,
        )


def _new_candidate_findings() -> list[CandidateFinding]:
    return []


def _new_typed_objections() -> list[TypedObjection]:
    return []


class ObjectionResolution(_RoleSchema):
    """Adjudicator's explicit disposition for one reviewer objection."""

    objection_id: str = Field(min_length=1)
    finding_id: str = Field(min_length=1)
    resolved: bool
    resolution: str = Field(min_length=1)
    evidence_refs: list[EvidenceReference] = Field(
        default_factory=_new_evidence_references
    )

    @field_validator("evidence_refs")
    @classmethod
    def _reject_duplicate_resolution_refs(
        cls, value: list[EvidenceReference]
    ) -> list[EvidenceReference]:
        evidence_ids = [item.evidence_id for item in value]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("resolution contains duplicate evidence ID")
        return value


def _new_objection_resolutions() -> list[ObjectionResolution]:
    return []


class _InvestigativeResponse(_RoleSchema):
    """Common retrieval state shared by every report role response."""

    schema_version: ClassVar[str] = REPORT_SYNTHESIS_SCHEMA_VERSION
    complete: bool
    evidence_requests: list[str] = Field(default_factory=list)
    unavailable_evidence_ids: list[str] = Field(default_factory=list, max_length=32)
    limitations: list[str] = Field(default_factory=list, max_length=16)

    @field_validator(
        "evidence_requests",
        "unavailable_evidence_ids",
        "limitations",
        mode="before",
    )
    @classmethod
    def _normalize_common_lists(cls, value: object) -> object:
        if value is None:
            return []
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError("response list fields must be lists of strings")
        return [
            item.strip() if isinstance(item, str) else item
            for item in cast(Sequence[object], value)
        ]

    @field_validator("evidence_requests", "unavailable_evidence_ids")
    @classmethod
    def _validate_evidence_id_values(cls, value: list[str]) -> list[str]:
        if any(not item for item in value):
            raise ValueError("evidence ID lists must not contain empty IDs")
        if len(value) != len(set(value)):
            raise ValueError("evidence ID lists must contain unique IDs")
        return value

    @field_validator("limitations")
    @classmethod
    def _validate_limitations(cls, value: list[str]) -> list[str]:
        if any(not item or len(item) > 512 for item in value):
            raise ValueError("limitations must contain bounded non-empty text")
        return value

    @model_validator(mode="after")
    def _require_request_when_incomplete(self) -> _InvestigativeResponse:
        if not self.complete and not self.evidence_requests:
            raise ValueError(
                "incomplete response requires at least one evidence_requests value"
            )
        if self.unavailable_evidence_ids and not self.limitations:
            raise ValueError("unavailable evidence declarations require limitations")
        return self


class AnalystResponse(_InvestigativeResponse):
    """Structured candidate findings emitted by the report analyst."""

    schema_version: ClassVar[str] = "report-analyst-response-v2"
    candidate_findings: list[CandidateFinding] = Field(
        default_factory=_new_candidate_findings
    )


class EvidenceAuditResponse(_InvestigativeResponse):
    """Structured factual and visual objections emitted by the evidence auditor."""

    schema_version: ClassVar[str] = "report-evidence-auditor-response-v2"
    objections: list[TypedObjection] = Field(default_factory=_new_typed_objections)

    @property
    def typed_objections(self) -> list[TypedObjection]:
        return self.objections


class PatternReviewResponse(_InvestigativeResponse):
    """Structured recurrence and severity objections emitted by the pattern reviewer."""

    schema_version: ClassVar[str] = "report-pattern-reviewer-response-v2"
    objections: list[TypedObjection] = Field(default_factory=_new_typed_objections)

    @property
    def typed_objections(self) -> list[TypedObjection]:
        return self.objections


class AdjudicationResponse(_InvestigativeResponse):
    """Structured final findings and explicit objection resolutions."""

    schema_version: ClassVar[str] = "report-adjudicator-response-v2"
    final_findings: list[CandidateFinding] = Field(
        default_factory=_new_candidate_findings
    )
    objection_resolutions: list[ObjectionResolution] = Field(
        default_factory=_new_objection_resolutions
    )


# Descriptive aliases keep role-specific contracts discoverable to providers.
EvidenceRefSchema = EvidenceReference
FindingSchema = CandidateFinding
FinalFinding = CandidateFinding
ObjectionSchema = TypedObjection
InvestigativeResponse = _InvestigativeResponse
ReportAnalystResponse = AnalystResponse
ReportEvidenceAuditorResponse = EvidenceAuditResponse
ReportPatternReviewerResponse = PatternReviewResponse
ReportAdjudicatorResponse = AdjudicationResponse
AuditorResponse = EvidenceAuditResponse
PatternResponse = PatternReviewResponse


class ReportAnalystPort(Protocol):
    """Application port for the isolated analyst role."""

    @property
    def manifest(self) -> ModelManifest: ...

    async def analyze(
        self,
        corpus_manifest: Any,
        principles: Any = None,
        *,
        resolved_evidence: Any = None,
        previous_output: Any = None,
    ) -> object: ...


class ReportEvidenceAuditorPort(Protocol):
    """Application port for the isolated evidence auditor role."""

    @property
    def manifest(self) -> ModelManifest: ...

    async def audit(
        self,
        corpus_manifest: Any,
        principles: Any = None,
        candidate_findings: Any = (),
        *,
        resolved_evidence: Any = None,
        previous_output: Any = None,
    ) -> object: ...


class ReportPatternReviewerPort(Protocol):
    """Application port for the isolated pattern reviewer role."""

    @property
    def manifest(self) -> ModelManifest: ...

    async def review(
        self,
        corpus_manifest: Any,
        principles: Any = None,
        candidate_findings: Any = (),
        *,
        resolved_evidence: Any = None,
        previous_output: Any = None,
    ) -> object: ...


class ReportAdjudicatorPort(Protocol):
    """Application port for the isolated adjudicator role."""

    @property
    def manifest(self) -> ModelManifest: ...

    async def adjudicate(
        self,
        corpus_manifest: Any,
        principles: Any = None,
        candidate_findings: Any = (),
        objections: Any = (),
        *,
        resolved_evidence: Any = None,
        previous_output: Any = None,
    ) -> object: ...


class SynthesisCorpusPort(Protocol):
    """Canonical corpus metadata required by immutable artifact storage."""

    @property
    def digest(self) -> str: ...

    @property
    def principle_pack_digest(self) -> str: ...

    def to_json(self) -> str: ...


__all__ = [
    "AdjudicationResponse",
    "AnalystResponse",
    "AuditorResponse",
    "CandidateFinding",
    "EvidenceAuditResponse",
    "EvidenceReference",
    "EvidenceRefSchema",
    "FinalFinding",
    "FindingSchema",
    "FORBIDDEN_NARRATIVE_MARKERS",
    "InvestigativeResponse",
    "ObjectionResolution",
    "ObjectionSchema",
    "PatternResponse",
    "PatternReviewResponse",
    "REPORT_SYNTHESIS_SCHEMA_VERSION",
    "ReportAdjudicatorPort",
    "ReportAdjudicatorResponse",
    "ReportAnalystPort",
    "ReportAnalystResponse",
    "ReportEvidenceAuditorPort",
    "ReportEvidenceAuditorResponse",
    "ReportPatternReviewerPort",
    "ReportPatternReviewerResponse",
    "SynthesisCorpusPort",
    "TypedObjection",
    "UxPrinciple",
    "contains_forbidden_narrative",
    "redact_forbidden_narrative",
    "is_sensitive_key",
]
