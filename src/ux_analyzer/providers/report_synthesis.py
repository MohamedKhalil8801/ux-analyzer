"""Fresh-context structured providers for evidence-grounded report synthesis."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import ClassVar, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from ux_analyzer.application.evidence_corpus import (
    EvidenceCorpus,
    EvidenceEntry,
    ResolvedEvidence,
)
from ux_analyzer.domain.findings import EvidenceClass, FindingSeverity, Reproducibility
from ux_analyzer.domain.synthesis import (
    EvidenceRef,
    ObjectionSeverity,
    SynthesisFinding,
    SynthesisObjection,
)
from ux_analyzer.ports.models import (
    ChatMessage,
    ModelAttachment,
    ModelManifest,
    ModelResponseValidationError,
    ModelRole,
    StructuredModelClient,
)
from ux_analyzer.providers.ux_principles import UxPrinciple, ux_principles

REPORT_SYNTHESIS_SCHEMA_VERSION = "report-synthesis-v1"

_SAFE_EVIDENCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(?::[A-Za-z0-9._-]+)+$")
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


class _RoleSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceReference(_RoleSchema):
    """Transport form of the existing domain ``EvidenceRef`` contract."""

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

    @field_validator("evidence_requests", mode="before")
    @classmethod
    def _normalize_evidence_requests(cls, value: object) -> object:
        if value is None:
            return []
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError("evidence_requests must be a list of strings")
        return [
            item.strip() if isinstance(item, str) else item
            for item in cast(Sequence[object], value)
        ]

    @field_validator("evidence_requests")
    @classmethod
    def _validate_evidence_request_values(cls, value: list[str]) -> list[str]:
        if any(not item for item in value):
            raise ValueError("evidence_requests must not contain empty IDs")
        if len(value) != len(set(value)):
            raise ValueError("evidence_requests must contain unique IDs")
        return value

    @model_validator(mode="after")
    def _require_request_when_incomplete(self) -> _InvestigativeResponse:
        if not self.complete and not self.evidence_requests:
            raise ValueError(
                "incomplete response requires at least one evidence_requests value"
            )
        return self


class AnalystResponse(_InvestigativeResponse):
    """Structured candidate findings emitted by the report analyst."""

    schema_version: ClassVar[str] = "report-analyst-response-v1"
    candidate_findings: list[CandidateFinding] = Field(
        default_factory=_new_candidate_findings
    )


class EvidenceAuditResponse(_InvestigativeResponse):
    """Structured factual and visual objections emitted by the evidence auditor."""

    schema_version: ClassVar[str] = "report-evidence-auditor-response-v1"
    objections: list[TypedObjection] = Field(default_factory=_new_typed_objections)

    @property
    def typed_objections(self) -> list[TypedObjection]:
        return self.objections


class PatternReviewResponse(_InvestigativeResponse):
    """Structured recurrence and severity objections emitted by the pattern reviewer."""

    schema_version: ClassVar[str] = "report-pattern-reviewer-response-v1"
    objections: list[TypedObjection] = Field(default_factory=_new_typed_objections)

    @property
    def typed_objections(self) -> list[TypedObjection]:
        return self.objections


class AdjudicationResponse(_InvestigativeResponse):
    """Structured final findings and explicit objection resolutions."""

    schema_version: ClassVar[str] = "report-adjudicator-response-v1"
    final_findings: list[CandidateFinding] = Field(
        default_factory=_new_candidate_findings
    )
    objection_resolutions: list[ObjectionResolution] = Field(
        default_factory=_new_objection_resolutions
    )


# Descriptive aliases keep role-specific names discoverable to orchestration code.
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


ManifestInput = EvidenceCorpus | Mapping[str, object]


def _canonical_narrative_text(value: str) -> str:
    return re.sub(r"[\s_-]+", " ", value.casefold()).strip()


def contains_forbidden_narrative(value: str) -> bool:
    normalized = _canonical_narrative_text(value)
    return any(
        _canonical_narrative_text(marker) in normalized
        for marker in FORBIDDEN_NARRATIVE_MARKERS
    )


def redact_forbidden_narrative(value: str) -> str:
    return "[redacted]" if contains_forbidden_narrative(value) else value


def _is_sensitive_key(key: object) -> bool:
    normalized = _canonical_narrative_text(str(key))
    return any(
        _canonical_narrative_text(marker) in normalized
        for marker in _SENSITIVE_KEY_MARKERS
    ) or contains_forbidden_narrative(str(key))


def _safe_string(value: str) -> str:
    return redact_forbidden_narrative(value)


def _safe_prompt_value(value: object, *, depth: int = 0) -> object:
    if depth > 8:
        return "[truncated]"
    if isinstance(value, EvidenceEntry):
        return _safe_entry(value, depth=depth + 1)
    if isinstance(value, EvidenceRef):
        return _safe_evidence_ref(value)
    if isinstance(value, BaseModel):
        return _safe_prompt_value(value.model_dump(mode="python"), depth=depth + 1)
    if isinstance(value, UxPrinciple):
        return _safe_prompt_value(asdict(value), depth=depth + 1)
    if isinstance(value, SynthesisFinding):
        return _safe_prompt_value(_domain_finding_payload(value), depth=depth + 1)
    if isinstance(value, SynthesisObjection):
        return _safe_prompt_value(_domain_objection_payload(value), depth=depth + 1)
    if is_dataclass(value) and not isinstance(value, type):
        return _safe_prompt_value(asdict(value), depth=depth + 1)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        result: dict[str, object] = {}
        for key, item in mapping.items():
            if _is_sensitive_key(key):
                continue
            safe_item = _safe_prompt_value(item, depth=depth + 1)
            result[str(key)] = safe_item
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            _safe_prompt_value(item, depth=depth + 1)
            for item in cast(Iterable[object], value)
        ]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, str):
        return _safe_string(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def _safe_evidence_ref(reference: EvidenceRef) -> dict[str, object]:
    return {
        name: value
        for name in (
            "evidence_id",
            "kind",
            "run_id",
            "viewport_id",
            "element_id",
            "event_id",
            "metric_id",
            "artifact_path",
            "replay_sequence",
            "sha256",
        )
        if (value := getattr(reference, name)) is not None
    }


def _safe_entry(entry: EvidenceEntry, *, depth: int = 0) -> dict[str, object]:
    return {
        "evidence_id": entry.ref.evidence_id,
        "kind": entry.ref.kind,
        "run_id": entry.ref.run_id,
        "reference": _safe_evidence_ref(entry.ref),
        "evidence_class": entry.evidence_class.value,
        "summary": _safe_prompt_value(entry.summary, depth=depth + 1),
        "payload": _safe_prompt_value(entry.payload, depth=depth + 1),
    }


def _domain_finding_payload(finding: SynthesisFinding) -> dict[str, object]:
    return {
        "finding_id": finding.finding_id,
        "title": finding.title,
        "issue": finding.issue,
        "impact": finding.impact,
        "root_cause": finding.root_cause,
        "fixes": finding.fixes,
        "severity": cast(FindingSeverity, finding.severity).value,
        "confidence": finding.confidence,
        "evidence_refs": [_safe_evidence_ref(ref) for ref in finding.evidence_refs],
        "affected_surfaces": finding.affected_surfaces,
        "principles": finding.principles,
        "counterevidence": [
            _safe_evidence_ref(item) if isinstance(item, EvidenceRef) else item
            for item in finding.counterevidence
        ],
        "limitations": finding.limitations,
        "reviewer_state": finding.reviewer_state,
        "evidence_class": cast(EvidenceClass, finding.evidence_class).value,
        "reproducibility": cast(Reproducibility, finding.reproducibility).value,
        "severity_justification": finding.severity_justification,
        "reviewer_notes": finding.reviewer_notes,
    }


def _domain_objection_payload(objection: SynthesisObjection) -> dict[str, object]:
    return {
        "objection_id": objection.objection_id,
        "finding_id": objection.finding_id,
        "severity": cast(ObjectionSeverity, objection.severity).value,
        "message": objection.message,
        "evidence_refs": [_safe_evidence_ref(ref) for ref in objection.evidence_refs],
        "reviewer_role": objection.reviewer_role,
        "resolved": objection.resolved,
        "resolution": objection.resolution,
    }


def _manifest_payload(manifest: object) -> dict[str, object]:
    if isinstance(manifest, EvidenceCorpus):
        source: object = manifest.to_dict()
    elif isinstance(manifest, Mapping):
        source = cast(Mapping[str, object], manifest)
    else:
        raise TypeError("corpus manifest must be an EvidenceCorpus or mapping")
    payload = cast(dict[str, object], _safe_prompt_value(source))
    entries = payload.get("entries")
    if isinstance(entries, list):
        normalized_entries: list[object] = []
        for entry in cast(list[object], entries):
            if isinstance(entry, EvidenceEntry):
                normalized_entries.append(_safe_entry(entry))
            else:
                normalized_entries.append(entry)
        payload["entries"] = normalized_entries
    return payload


def _known_evidence_ids(manifest: ManifestInput) -> frozenset[str]:
    if isinstance(manifest, EvidenceCorpus):
        return frozenset(entry.ref.evidence_id for entry in manifest.entries)
    manifest_mapping = manifest
    values: set[str] = set()
    entries = manifest_mapping.get("entries", ())
    if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
        for entry in cast(Sequence[object], entries):
            if isinstance(entry, EvidenceEntry):
                values.add(entry.ref.evidence_id)
            elif isinstance(entry, Mapping):
                entry_mapping = cast(Mapping[object, object], entry)
                evidence_id = entry_mapping.get("evidence_id")
                if not isinstance(evidence_id, str):
                    reference = entry_mapping.get("reference")
                    if isinstance(reference, Mapping):
                        evidence_id = cast(Mapping[object, object], reference).get(
                            "evidence_id"
                        )
                if isinstance(evidence_id, str) and evidence_id.strip():
                    values.add(evidence_id.strip())
    evidence_ids = manifest_mapping.get("evidence_ids", ())
    if isinstance(evidence_ids, Sequence) and not isinstance(
        evidence_ids, (str, bytes)
    ):
        values.update(
            item.strip()
            for item in cast(Sequence[object], evidence_ids)
            if isinstance(item, str) and item.strip()
        )
    return frozenset(values)


def _resolved_payload(resolved: ResolvedEvidence | None) -> list[object]:
    if resolved is None:
        return []
    return [_safe_entry(entry) for entry in resolved.entries]


def _principle_payload(
    principles: object | None,
) -> tuple[UxPrinciple, ...]:
    if principles is None:
        return ux_principles()
    if isinstance(principles, (str, bytes)):
        raise TypeError("principles must be a sequence of UxPrinciple values")
    if not isinstance(principles, Sequence):
        raise TypeError("principles must be a sequence of UxPrinciple values")
    normalized_values = tuple(cast(Sequence[object], principles))
    if any(not isinstance(item, UxPrinciple) for item in normalized_values):
        raise TypeError("principles must contain UxPrinciple values")
    return cast(tuple[UxPrinciple, ...], normalized_values)


def _attachment_values(
    manifest: ManifestInput,
    resolved: ResolvedEvidence | None,
) -> tuple[ModelAttachment, ...]:
    if resolved is None or not isinstance(manifest, EvidenceCorpus):
        return ()
    attachments: list[ModelAttachment] = []
    for entry in resolved.entries:
        if entry.attachment_path is None or entry.ref.sha256 is None:
            continue
        media_type = entry.payload.get("media_type")
        if media_type not in {"image/png", "image/jpeg"}:
            continue
        typed_media_type = cast(Literal["image/png", "image/jpeg"], media_type)
        attachments.append(
            ModelAttachment(
                evidence_id=entry.ref.evidence_id,
                path=manifest.output_root / entry.attachment_path,
                media_type=typed_media_type,
                sha256=entry.ref.sha256,
            )
        )
    return tuple(attachments)


def _canonical_json(value: object) -> str:
    return json.dumps(
        _safe_prompt_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


class _ReportRole:
    response_schema: ClassVar[type[_InvestigativeResponse]]
    role: ClassVar[ModelRole]
    prompt_version: ClassVar[str]

    def __init__(self, client: StructuredModelClient, *, model: object) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("report model must not be empty")
        self.client = client
        self.model: str = model

    @property
    def manifest(self) -> ModelManifest:
        return ModelManifest(
            provider_id=str(
                getattr(
                    client := self.client, "provider_id", "openai-compatible-structured"
                )
            ),
            role=self.role,
            model_id=self.model,
            endpoint_origin=str(getattr(client, "endpoint_origin", "unknown")),
            prompt_version=self.prompt_version,
            schema_version=self.response_schema.schema_version,
            provider_version=str(
                getattr(client, "provider_version", "openai-compatible-v1")
            ),
        )

    @property
    def prompt(self) -> str:
        return _COMMON_PROMPT + "\n\n" + self._role_prompt

    @property
    def _role_prompt(self) -> str:
        raise NotImplementedError

    async def _complete(
        self,
        manifest: ManifestInput,
        principles: Sequence[UxPrinciple] | None,
        *,
        role_input: Mapping[str, object] | None = None,
        resolved_evidence: ResolvedEvidence | None = None,
        previous_output: _InvestigativeResponse | None = None,
    ) -> _InvestigativeResponse:
        normalized_principles = _principle_payload(principles)
        if previous_output is not None and not isinstance(
            previous_output, self.response_schema
        ):
            raise ValueError("previous output must belong to this role")
        message_payload: dict[str, object] = {
            "corpus_manifest": _manifest_payload(manifest),
            "resolved_evidence": _resolved_payload(resolved_evidence),
            "ux_principle_pack": [asdict(item) for item in normalized_principles],
            "role_input": dict(role_input or {}),
        }
        if previous_output is not None:
            message_payload["prior_structured_output"] = previous_output.model_dump(
                mode="json"
            )
        messages = (
            ChatMessage(role="system", content=self.prompt),
            ChatMessage(
                role="user",
                content=_canonical_json(message_payload),
                attachments=_attachment_values(manifest, resolved_evidence),
            ),
        )
        try:
            response = await self.client.complete(
                self.response_schema,
                messages,
                model=self.model,
                role=self.role,
            )
        except ModelResponseValidationError:
            raise
        except ValidationError as error:
            raise ModelResponseValidationError(
                self.role,
                "response schema validation failed",
                response_summary={"schema": self.response_schema.__name__},
            ) from error
        except (TypeError, ValueError) as error:
            raise ModelResponseValidationError(
                self.role,
                "response schema validation failed",
                response_summary={"schema": self.response_schema.__name__},
            ) from error

        if not isinstance(response, self.response_schema):
            try:
                response = self.response_schema.model_validate(response)
            except ValidationError as error:
                raise ModelResponseValidationError(
                    self.role,
                    "response schema validation failed",
                    response_summary={"schema": self.response_schema.__name__},
                ) from error
        self._validate_response(response, manifest, normalized_principles)
        return response

    def _validate_response(
        self,
        response: _InvestigativeResponse,
        manifest: ManifestInput,
        principles: Sequence[UxPrinciple],
    ) -> None:
        known_ids = _known_evidence_ids(manifest)
        if not response.complete and not response.evidence_requests:
            self._invalid("incomplete response needs evidence requests")
        for evidence_id in response.evidence_requests:
            self._validate_requested_id(evidence_id, known_ids)

        principle_ids = {principle.principle_id for principle in principles}
        for finding in self._findings(response):
            self._validate_finding(finding, known_ids, principle_ids)
        for objection in self._objections(response):
            self._validate_refs(objection.evidence_refs, known_ids)
        for resolution in self._resolutions(response):
            self._validate_refs(resolution.evidence_refs, known_ids)

    def _validate_requested_id(
        self,
        evidence_id: str,
        known_ids: frozenset[str],
    ) -> None:
        if _SAFE_EVIDENCE_ID.fullmatch(evidence_id) is None:
            self._invalid("invalid evidence request ID")
        if evidence_id.startswith("principle:"):
            self._invalid("principles are not evidence")
        if not known_ids:
            self._invalid("evidence request cannot be validated without corpus IDs")
        if evidence_id not in known_ids:
            self._invalid(f"unknown evidence ID: {evidence_id}")

    def _validate_refs(
        self,
        refs: Sequence[EvidenceReference],
        known_ids: frozenset[str],
    ) -> None:
        evidence_ids = [reference.evidence_id for reference in refs]
        if len(evidence_ids) != len(set(evidence_ids)):
            self._invalid("duplicate evidence ID")
        for reference in refs:
            if reference.evidence_id.startswith("principle:"):
                self._invalid("principles are not evidence")
            if reference.evidence_id not in known_ids:
                self._invalid(f"unknown evidence ID: {reference.evidence_id}")

    def _validate_finding(
        self,
        finding: CandidateFinding,
        known_ids: frozenset[str],
        principle_ids: set[str],
    ) -> None:
        refs = tuple(finding.evidence_refs) + tuple(
            item
            for item in finding.counterevidence
            if isinstance(item, EvidenceReference)
        )
        self._validate_refs(refs, known_ids)
        unknown_principles = set(finding.principles) - principle_ids
        if unknown_principles:
            self._invalid("finding references an unknown UX principle")
        if not finding.evidence_refs:
            self._invalid("principles are not evidence")

    @staticmethod
    def _findings(response: _InvestigativeResponse) -> tuple[CandidateFinding, ...]:
        if isinstance(response, AnalystResponse):
            return tuple(response.candidate_findings)
        if isinstance(response, AdjudicationResponse):
            return tuple(response.final_findings)
        return ()

    @staticmethod
    def _objections(response: _InvestigativeResponse) -> tuple[TypedObjection, ...]:
        if isinstance(response, (EvidenceAuditResponse, PatternReviewResponse)):
            return tuple(response.objections)
        return ()

    @staticmethod
    def _resolutions(
        response: _InvestigativeResponse,
    ) -> tuple[ObjectionResolution, ...]:
        if isinstance(response, AdjudicationResponse):
            return tuple(response.objection_resolutions)
        return ()

    def _invalid(self, reason: str) -> None:
        raise ModelResponseValidationError(
            self.role,
            reason,
            response_summary={"schema": self.response_schema.__name__},
        )


_COMMON_PROMPT = """You are one isolated report-synthesis role.

Use only the allowlisted structured evidence in corpus_manifest and resolved_evidence. Do not use prior prompts, raw model responses, private reasoning, chat history, cognitive prose, or existing finding prose. Frozen Expectations describe outcomes, invariants, acceptable alternatives, and warning signals. Treat reference paths as examples, not the only correct path. Judge path deviations only through observed user impact and supported outcomes.

The UX principle pack is optional interpretive guidance. Principles are not evidence and cannot determine severity. A principle may help name or explain an issue only when observed evidence supports it. Never request or cite a principle as an evidence ID.

Every factual claim and finding must use resolvable evidence IDs from corpus_manifest. If more evidence is needed, set complete to false and request only valid evidence IDs listed in corpus_manifest. Return only the structured response schema; do not include private reasoning or extra fields."""


class ReportAnalyst(_ReportRole):
    """Discover evidence-backed UX issues and plausible root causes."""

    role = ModelRole.REPORT_ANALYST
    prompt_version = "report-analyst-v1"
    response_schema = AnalystResponse

    @property
    def _role_prompt(self) -> str:
        return "Discover material UX issues and their likely root causes. Emit candidate findings with plain language, concrete fixes, evidence references, limitations, and justified severity."

    async def analyze(
        self,
        corpus_manifest: ManifestInput,
        principles: Sequence[UxPrinciple] | None = None,
        *,
        resolved_evidence: ResolvedEvidence | None = None,
        previous_output: AnalystResponse | None = None,
    ) -> AnalystResponse:
        response = await self._complete(
            corpus_manifest,
            principles,
            resolved_evidence=resolved_evidence,
            previous_output=previous_output,
            role_input={"task": "discover issues and root causes"},
        )
        return cast(AnalystResponse, response)


class EvidenceAuditor(_ReportRole):
    """Challenge factual and visual support for analyst candidates."""

    role = ModelRole.REPORT_EVIDENCE_AUDITOR
    prompt_version = "report-evidence-auditor-v1"
    response_schema = EvidenceAuditResponse

    @property
    def _role_prompt(self) -> str:
        return "Challenge factual support, visual interpretation, citation accuracy, and contradictions in candidate findings. Emit typed objections tied to the evidence needed to resolve each challenge."

    async def audit(
        self,
        corpus_manifest: ManifestInput,
        principles: Sequence[UxPrinciple] | None = None,
        candidate_findings: Sequence[
            CandidateFinding | SynthesisFinding | Mapping[str, object]
        ] = (),
        *,
        resolved_evidence: ResolvedEvidence | None = None,
        previous_output: EvidenceAuditResponse | None = None,
    ) -> EvidenceAuditResponse:
        response = await self._complete(
            corpus_manifest,
            principles,
            resolved_evidence=resolved_evidence,
            previous_output=previous_output,
            role_input={
                "task": "audit factual and visual support",
                "candidate_findings": list(candidate_findings),
            },
        )
        return cast(EvidenceAuditResponse, response)

    async def review(
        self,
        corpus_manifest: ManifestInput,
        principles: Sequence[UxPrinciple] | None = None,
        candidate_findings: Sequence[
            CandidateFinding | SynthesisFinding | Mapping[str, object]
        ] = (),
        *,
        resolved_evidence: ResolvedEvidence | None = None,
        previous_output: EvidenceAuditResponse | None = None,
    ) -> EvidenceAuditResponse:
        return await self.audit(
            corpus_manifest,
            principles,
            candidate_findings,
            resolved_evidence=resolved_evidence,
            previous_output=previous_output,
        )


class PatternReviewer(_ReportRole):
    """Review recurrence, cross-surface impact, severity, and fix leverage."""

    role = ModelRole.REPORT_PATTERN_REVIEWER
    prompt_version = "report-pattern-reviewer-v1"
    response_schema = PatternReviewResponse

    @property
    def _role_prompt(self) -> str:
        return "Check recurrence, affected surfaces, counterexamples, shared causes, severity, and fix leverage. Emit typed objections when a pattern or priority claim is not established by evidence."

    async def review(
        self,
        corpus_manifest: ManifestInput,
        principles: Sequence[UxPrinciple] | None = None,
        candidate_findings: Sequence[
            CandidateFinding | SynthesisFinding | Mapping[str, object]
        ] = (),
        *,
        resolved_evidence: ResolvedEvidence | None = None,
        previous_output: PatternReviewResponse | None = None,
    ) -> PatternReviewResponse:
        response = await self._complete(
            corpus_manifest,
            principles,
            resolved_evidence=resolved_evidence,
            previous_output=previous_output,
            role_input={
                "task": "review recurrence, scope, severity, and fix leverage",
                "candidate_findings": list(candidate_findings),
            },
        )
        return cast(PatternReviewResponse, response)


class ReportAdjudicator(_ReportRole):
    """Resolve reviewer objections and write plain-language final findings."""

    role = ModelRole.REPORT_ADJUDICATOR
    prompt_version = "report-adjudicator-v1"
    response_schema = AdjudicationResponse

    @property
    def _role_prompt(self) -> str:
        return "This role resolves objections and writes plain-language final findings. Publish only findings with supported evidence, concrete fixes, justified severity, and explicit resolutions for every objection."

    async def adjudicate(
        self,
        corpus_manifest: ManifestInput,
        principles: Sequence[UxPrinciple] | None = None,
        candidate_findings: Sequence[
            CandidateFinding | SynthesisFinding | Mapping[str, object]
        ] = (),
        objections: Sequence[
            TypedObjection | SynthesisObjection | Mapping[str, object]
        ] = (),
        *,
        resolved_evidence: ResolvedEvidence | None = None,
        previous_output: AdjudicationResponse | None = None,
    ) -> AdjudicationResponse:
        response = await self._complete(
            corpus_manifest,
            principles,
            resolved_evidence=resolved_evidence,
            previous_output=previous_output,
            role_input={
                "task": "resolve objections and write final findings",
                "candidate_findings": list(candidate_findings),
                "objections": list(objections),
            },
        )
        return cast(AdjudicationResponse, response)

    async def resolve(
        self,
        corpus_manifest: ManifestInput,
        principles: Sequence[UxPrinciple] | None = None,
        candidate_findings: Sequence[
            CandidateFinding | SynthesisFinding | Mapping[str, object]
        ] = (),
        objections: Sequence[
            TypedObjection | SynthesisObjection | Mapping[str, object]
        ] = (),
        *,
        resolved_evidence: ResolvedEvidence | None = None,
        previous_output: AdjudicationResponse | None = None,
    ) -> AdjudicationResponse:
        return await self.adjudicate(
            corpus_manifest,
            principles,
            candidate_findings,
            objections,
            resolved_evidence=resolved_evidence,
            previous_output=previous_output,
        )


__all__ = [
    "AdjudicationResponse",
    "AnalystResponse",
    "AuditorResponse",
    "CandidateFinding",
    "EvidenceAuditResponse",
    "EvidenceAuditor",
    "EvidenceReference",
    "EvidenceRefSchema",
    "FinalFinding",
    "FindingSchema",
    "InvestigativeResponse",
    "ManifestInput",
    "ObjectionResolution",
    "ObjectionSchema",
    "PatternResponse",
    "PatternReviewResponse",
    "PatternReviewer",
    "ReportAdjudicator",
    "ReportAdjudicatorResponse",
    "ReportAnalyst",
    "ReportAnalystResponse",
    "ReportEvidenceAuditorResponse",
    "ReportPatternReviewerResponse",
    "REPORT_SYNTHESIS_SCHEMA_VERSION",
    "TypedObjection",
    "FORBIDDEN_NARRATIVE_MARKERS",
    "contains_forbidden_narrative",
    "redact_forbidden_narrative",
]
