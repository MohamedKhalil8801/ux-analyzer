"""Application orchestration for isolated, evidence-grounded report synthesis."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from inspect import isawaitable
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from ux_analyzer.application.evidence_corpus import (
    EvidenceCorpus,
    EvidenceResolver,
    ResolvedEvidence,
    validate_evidence_refs,
)
from ux_analyzer.domain.findings import EvidenceClass
from ux_analyzer.domain.synthesis import (
    EvidenceRef,
    ObjectionSeverity,
    SynthesisAttempt,
    SynthesisFinding,
    SynthesisObjection,
    SynthesisStatus,
)
from ux_analyzer.ports.models import (
    ModelCallRecord,
    ModelManifest,
    ModelResponseValidationError,
    ModelRole,
)
from ux_analyzer.ports.report_synthesis import (
    AdjudicationResponse,
    AnalystResponse,
    CandidateFinding,
    EvidenceAuditResponse,
    ObjectionResolution,
    PatternReviewResponse,
    ReportAdjudicatorPort,
    ReportAnalystPort,
    ReportEvidenceAuditorPort,
    ReportPatternReviewerPort,
    TypedObjection,
    UxPrinciple,
    contains_forbidden_narrative,
    redact_forbidden_narrative,
)

REPORT_SYNTHESIS_APPLICATION_SCHEMA_VERSION = "synthesis-v1"
REPORT_SYNTHESIS_PROMPT_VERSION = "report-synthesis-orchestrator-v1"
MAX_RETRIEVAL_ROUNDS = 3
DEFAULT_MAX_RETRIEVAL_ENTRIES = 32
DEFAULT_MAX_ATTACHMENT_BYTES = 16 * 1024 * 1024

_PRINCIPLE_AUTHORITY_MARKERS = (
    "principle proves",
    "principles prove",
    "principle determines",
    "principles determine",
    "principle makes this",
    "principles make this",
    "law makes this",
    "law proves",
)
_CAUSAL_MARKERS = re.compile(
    r"\b(?:because|due to|causes?|caused by|results? in|leads? to|drives?)\b",
    re.IGNORECASE,
)
_HARM_MARKERS = (
    "abandon",
    "block",
    "cannot",
    "could not",
    "confus",
    "delay",
    "error",
    "extra",
    "fail",
    "longer",
    "wrong",
)
_ALTERNATE_PATH_MARKERS = (
    "alternate path",
    "alternative path",
    "different path",
    "path deviation",
    "reference path",
    "did not follow",
    "does not follow",
)
_OPERATIONAL_FAILURE_NAMES = {
    "ModelConfigurationError",
    "ModelFailureError",
    "TimeoutError",
    "ConnectionError",
    "OSError",
}
_SEVERITY_ORDER = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}
_REVIEWER_ROLE_ALIASES = {
    "report-evidence-auditor": ModelRole.REPORT_EVIDENCE_AUDITOR,
    "evidence-auditor": ModelRole.REPORT_EVIDENCE_AUDITOR,
    "report-pattern-reviewer": ModelRole.REPORT_PATTERN_REVIEWER,
    "pattern-reviewer": ModelRole.REPORT_PATTERN_REVIEWER,
    "report-analyst": ModelRole.REPORT_ANALYST,
    "ux-analyst": ModelRole.REPORT_ANALYST,
}

_Response = (
    AnalystResponse
    | EvidenceAuditResponse
    | PatternReviewResponse
    | AdjudicationResponse
)


class _BenignCandidate(ValueError):
    """Candidate rejected because it describes a harmless valid alternative."""


@dataclass(frozen=True, slots=True)
class _RoleRun:
    response: _Response | None
    retrieval_log: tuple[Mapping[str, object], ...]
    unavailable: bool = False
    invalid: bool = False
    limitation: str | None = None


def _json_safe(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump(mode="python"))
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in cast(Mapping[object, object], value).items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in cast(Sequence[object], value)]
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _canonical_json(value: object) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _redact_structured(value: object, *, key: str = "") -> object:
    if key and contains_forbidden_narrative(key):
        return "[redacted]"
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {
            str(name): _redact_structured(item, key=str(name))
            for name, item in mapping.items()
        }
    if isinstance(value, list):
        items = cast(list[object], value)
        return [_redact_structured(item) for item in items]
    if isinstance(value, tuple):
        items = cast(tuple[object, ...], value)
        return tuple(_redact_structured(item) for item in items)
    if isinstance(value, str):
        return redact_forbidden_narrative(value)
    return value


def _response_payload(response: _Response) -> Mapping[str, object]:
    payload = _redact_structured(_json_safe(response))
    if not isinstance(payload, Mapping):
        return {"schema": type(response).__name__}
    return cast(Mapping[str, object], payload)


def _error_category(error: BaseException) -> tuple[bool, str]:
    name = type(error).__name__
    if name in _OPERATIONAL_FAILURE_NAMES or isinstance(error, RuntimeError):
        return True, "model transport or configuration failure"
    if isinstance(
        error, (ModelResponseValidationError, ValidationError, ValueError, TypeError)
    ):
        return False, "invalid structured synthesis output"
    return False, "synthesis role failure"


def _role_schema(role: ModelRole) -> type[_Response]:
    if role is ModelRole.REPORT_ANALYST:
        return AnalystResponse
    if role is ModelRole.REPORT_EVIDENCE_AUDITOR:
        return EvidenceAuditResponse
    if role is ModelRole.REPORT_PATTERN_REVIEWER:
        return PatternReviewResponse
    return AdjudicationResponse


def _finding_sort_key(finding: SynthesisFinding) -> tuple[int, str]:
    severity = finding.severity
    severity_value = severity.value if isinstance(severity, Enum) else str(severity)
    return _SEVERITY_ORDER.get(severity_value, len(_SEVERITY_ORDER)), finding.finding_id


def _normalize_response(role: ModelRole, value: object) -> _Response:
    schema = _role_schema(role)
    if isinstance(value, schema):
        value = value.model_dump(mode="python")
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    if not isinstance(value, Mapping):
        raise TypeError("synthesis role response must be a structured object")
    return schema.model_validate(value)


def _manifest_payload(provider: object, role: ModelRole) -> Mapping[str, object]:
    manifest = getattr(provider, "manifest", None)
    if isinstance(manifest, ModelManifest):
        return cast(Mapping[str, object], _json_safe(manifest))
    if is_dataclass(manifest) and not isinstance(manifest, type):
        return cast(Mapping[str, object], _json_safe(manifest))
    return {
        "provider_id": str(getattr(provider, "provider_id", "unavailable")),
        "role": role.value,
        "model_id": str(getattr(provider, "model", "unavailable")),
        "endpoint_origin": str(getattr(provider, "endpoint_origin", "unavailable")),
        "prompt_version": str(getattr(provider, "prompt_version", "unavailable")),
        "schema_version": _role_schema(role).schema_version,
        "provider_version": str(getattr(provider, "provider_version", "unavailable")),
    }


def _provider_roles(
    roles: Mapping[object, object],
) -> dict[ModelRole, object]:
    aliases = {
        "analyst": ModelRole.REPORT_ANALYST,
        "report-analyst": ModelRole.REPORT_ANALYST,
        "evidence-auditor": ModelRole.REPORT_EVIDENCE_AUDITOR,
        "report-evidence-auditor": ModelRole.REPORT_EVIDENCE_AUDITOR,
        "pattern-reviewer": ModelRole.REPORT_PATTERN_REVIEWER,
        "report-pattern-reviewer": ModelRole.REPORT_PATTERN_REVIEWER,
        "adjudicator": ModelRole.REPORT_ADJUDICATOR,
        "report-adjudicator": ModelRole.REPORT_ADJUDICATOR,
    }
    normalized: dict[ModelRole, object] = {}
    for key, provider in roles.items():
        if isinstance(key, ModelRole):
            role = key
        else:
            role = aliases.get(str(key).casefold())
            if role is None:
                try:
                    role = ModelRole(str(key))
                except ValueError:
                    continue
        if role in {
            ModelRole.REPORT_ANALYST,
            ModelRole.REPORT_EVIDENCE_AUDITOR,
            ModelRole.REPORT_PATTERN_REVIEWER,
            ModelRole.REPORT_ADJUDICATOR,
        }:
            normalized[role] = provider
    return normalized


class ReportSynthesisService:
    """Orchestrate isolated role calls behind one bounded retrieval boundary."""

    def __init__(
        self,
        analyst: ReportAnalystPort | None = None,
        evidence_auditor: ReportEvidenceAuditorPort | None = None,
        pattern_reviewer: ReportPatternReviewerPort | None = None,
        adjudicator: ReportAdjudicatorPort | None = None,
        *,
        roles: Mapping[object, object] | None = None,
        resolver: EvidenceResolver | None = None,
        principles: Sequence[UxPrinciple] | None = None,
        max_retrieval_rounds: int = MAX_RETRIEVAL_ROUNDS,
        max_retrieval_entries: int = DEFAULT_MAX_RETRIEVAL_ENTRIES,
        max_attachment_bytes: int = DEFAULT_MAX_ATTACHMENT_BYTES,
        max_adjudication_revisions: int = 1,
        max_final_verifications: int = 1,
        model_record_source: object | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        supplied_roles = _provider_roles(roles or {})
        self.analyst = analyst or supplied_roles.get(ModelRole.REPORT_ANALYST)
        self.evidence_auditor = evidence_auditor or supplied_roles.get(
            ModelRole.REPORT_EVIDENCE_AUDITOR
        )
        self.pattern_reviewer = pattern_reviewer or supplied_roles.get(
            ModelRole.REPORT_PATTERN_REVIEWER
        )
        self.adjudicator = adjudicator or supplied_roles.get(
            ModelRole.REPORT_ADJUDICATOR
        )
        if type(max_retrieval_rounds) is not int or not 1 <= max_retrieval_rounds <= 5:
            raise ValueError("max_retrieval_rounds must be between 1 and 5")
        if type(max_retrieval_entries) is not int or max_retrieval_entries <= 0:
            raise ValueError("max_retrieval_entries must be greater than zero")
        if type(max_attachment_bytes) is not int or max_attachment_bytes <= 0:
            raise ValueError("max_attachment_bytes must be greater than zero")
        if (
            type(max_adjudication_revisions) is not int
            or not 0 <= max_adjudication_revisions <= 2
        ):
            raise ValueError("max_adjudication_revisions must be between 0 and 2")
        if (
            type(max_final_verifications) is not int
            or not 1 <= max_final_verifications <= 2
        ):
            raise ValueError("max_final_verifications must be between 1 and 2")
        self.resolver = resolver or EvidenceResolver()
        raw_principles: Sequence[object] = cast(Sequence[object], principles or ())
        if any(not isinstance(item, UxPrinciple) for item in raw_principles):
            raise TypeError("principles must contain UxPrinciple values")
        self.principles = tuple(cast(Sequence[UxPrinciple], raw_principles))
        self.max_retrieval_rounds = max_retrieval_rounds
        self.max_retrieval_entries = max_retrieval_entries
        self.max_attachment_bytes = max_attachment_bytes
        self.max_adjudication_revisions = max_adjudication_revisions
        self.max_final_verifications = max_final_verifications
        self._model_record_source = model_record_source
        self._clock = clock
        self._sequence = 0

    async def synthesize(self, corpus: EvidenceCorpus) -> SynthesisAttempt:
        """Run all synthesis roles and return an immutable synthesis outcome."""

        corpus_input: Any = corpus
        if not isinstance(corpus_input, EvidenceCorpus):
            raise TypeError("synthesize requires an EvidenceCorpus")
        corpus = corpus_input
        attempt_id, created_at = self._attempt_identity(corpus)
        retrieval_log: list[Mapping[str, object]] = []
        limitations = self._base_limitations(corpus)
        role_manifest = self._role_manifests()
        missing_roles = self._missing_roles()
        if missing_roles:
            return self._attempt(
                corpus,
                attempt_id=attempt_id,
                created_at=created_at,
                status=SynthesisStatus.UNAVAILABLE,
                limitations=limitations
                + [
                    "Report synthesis is unavailable because role configuration is incomplete."
                ],
                retrieval_log=retrieval_log,
                role_manifest=role_manifest,
            )

        analyst_run = await self._run_role(
            ModelRole.REPORT_ANALYST,
            cast(object, self.analyst),
            corpus,
            candidate_findings=(),
        )
        retrieval_log.extend(analyst_run.retrieval_log)
        if analyst_run.response is None:
            return self._attempt(
                corpus,
                attempt_id=attempt_id,
                created_at=created_at,
                status=(
                    SynthesisStatus.UNAVAILABLE
                    if analyst_run.unavailable
                    else SynthesisStatus.REJECTED
                ),
                limitations=limitations
                + [
                    analyst_run.limitation
                    or "Report analyst produced no usable output."
                ],
                retrieval_log=retrieval_log,
                role_manifest=role_manifest,
            )

        analyst_response = cast(AnalystResponse, analyst_run.response)
        candidate_models = tuple(analyst_response.candidate_findings)
        candidate_findings: list[SynthesisFinding] = []
        benign_candidate_count = 0
        candidate_limitations: list[str] = []
        duplicate_candidate_ids: set[str] = set()
        seen_candidate_ids: set[str] = set()
        for candidate in candidate_models:
            if candidate.finding_id in seen_candidate_ids:
                duplicate_candidate_ids.add(candidate.finding_id)
            seen_candidate_ids.add(candidate.finding_id)
            try:
                finding = self._validated_finding(
                    corpus,
                    candidate,
                    reviewer_state="candidate",
                )
            except _BenignCandidate:
                benign_candidate_count += 1
                candidate_limitations.append(
                    f"Candidate {candidate.finding_id} described a valid alternate path and was not treated as an issue."
                )
            except (TypeError, ValueError) as error:
                candidate_limitations.append(
                    f"Candidate {candidate.finding_id} failed deterministic publication validation: {self._safe_validation_reason(error)}."
                )
            else:
                candidate_findings.append(finding)
        limitations.extend(candidate_limitations)

        candidate_ids = {finding.finding_id for finding in candidate_findings}
        candidate_input = tuple(
            candidate
            for candidate in candidate_models
            if candidate.finding_id in candidate_ids
        )

        candidate_evidence_ids = self._evidence_ids(candidate_findings)
        auditor_run = await self._run_role(
            ModelRole.REPORT_EVIDENCE_AUDITOR,
            cast(object, self.evidence_auditor),
            corpus,
            candidate_findings=candidate_input,
            initial_evidence_ids=candidate_evidence_ids,
        )
        retrieval_log.extend(auditor_run.retrieval_log)
        if auditor_run.response is None:
            return self._attempt(
                corpus,
                attempt_id=attempt_id,
                created_at=created_at,
                status=(
                    SynthesisStatus.UNAVAILABLE
                    if auditor_run.unavailable
                    else SynthesisStatus.REJECTED
                ),
                limitations=limitations
                + [
                    auditor_run.limitation
                    or "Evidence auditor produced no usable output."
                ],
                retrieval_log=retrieval_log,
                role_manifest=role_manifest,
                candidate_findings=tuple(candidate_findings),
                rejected_findings=self._not_established(
                    candidate_findings,
                    "Evidence auditor output was not trustworthy.",
                ),
            )

        pattern_run = await self._run_role(
            ModelRole.REPORT_PATTERN_REVIEWER,
            cast(object, self.pattern_reviewer),
            corpus,
            candidate_findings=candidate_input,
            initial_evidence_ids=candidate_evidence_ids,
        )
        retrieval_log.extend(pattern_run.retrieval_log)
        if pattern_run.response is None:
            return self._attempt(
                corpus,
                attempt_id=attempt_id,
                created_at=created_at,
                status=(
                    SynthesisStatus.UNAVAILABLE
                    if pattern_run.unavailable
                    else SynthesisStatus.REJECTED
                ),
                limitations=limitations
                + [
                    pattern_run.limitation
                    or "Pattern reviewer produced no usable output."
                ],
                retrieval_log=retrieval_log,
                role_manifest=role_manifest,
                candidate_findings=tuple(candidate_findings),
                rejected_findings=self._not_established(
                    candidate_findings,
                    "Pattern reviewer output was not trustworthy.",
                ),
            )

        try:
            auditor_objections = self._reviewer_objections(
                corpus,
                cast(EvidenceAuditResponse, auditor_run.response),
                ModelRole.REPORT_EVIDENCE_AUDITOR,
            )
            pattern_objections = self._reviewer_objections(
                corpus,
                cast(PatternReviewResponse, pattern_run.response),
                ModelRole.REPORT_PATTERN_REVIEWER,
            )
            objections = auditor_objections + pattern_objections
            self._validate_reviewer_objections(objections, candidate_ids)
        except (TypeError, ValueError) as error:
            limitations.append(
                f"Reviewer objections failed deterministic validation: {self._safe_validation_reason(error)}."
            )
            return self._attempt(
                corpus,
                attempt_id=attempt_id,
                created_at=created_at,
                status=SynthesisStatus.REJECTED,
                limitations=limitations,
                retrieval_log=retrieval_log,
                role_manifest=role_manifest,
                candidate_findings=tuple(candidate_findings),
                rejected_findings=self._not_established(
                    candidate_findings, "reviewer output was not trustworthy"
                ),
            )

        adjudication_evidence_ids = self._evidence_ids(candidate_findings) + tuple(
            ref.evidence_id
            for objection in objections
            for ref in objection.evidence_refs
        )
        adjudication_run = await self._run_role(
            ModelRole.REPORT_ADJUDICATOR,
            cast(object, self.adjudicator),
            corpus,
            candidate_findings=candidate_input,
            objections=objections,
            initial_evidence_ids=tuple(dict.fromkeys(adjudication_evidence_ids)),
        )
        retrieval_log.extend(adjudication_run.retrieval_log)
        if adjudication_run.response is None:
            return self._attempt(
                corpus,
                attempt_id=attempt_id,
                created_at=created_at,
                status=(
                    SynthesisStatus.UNAVAILABLE
                    if adjudication_run.unavailable
                    else SynthesisStatus.REJECTED
                ),
                limitations=limitations
                + [
                    adjudication_run.limitation
                    or "Report adjudicator produced no usable output."
                ],
                retrieval_log=retrieval_log,
                role_manifest=role_manifest,
                candidate_findings=tuple(candidate_findings),
                objections=objections,
                rejected_findings=self._not_established(
                    candidate_findings,
                    "Adjudicator output was not trustworthy.",
                ),
            )

        adjudication_response = cast(AdjudicationResponse, adjudication_run.response)
        final_models = tuple(adjudication_response.final_findings)
        try:
            resolved_objections = self._apply_resolutions(
                corpus,
                objections,
                adjudication_response.objection_resolutions,
                limitations,
            )
        except (TypeError, ValueError) as error:
            limitations.append(
                f"Resolution validation failed: {self._safe_validation_reason(error)}."
            )
            return self._attempt(
                corpus,
                attempt_id=attempt_id,
                created_at=created_at,
                status=SynthesisStatus.REJECTED,
                limitations=limitations,
                retrieval_log=retrieval_log,
                role_manifest=role_manifest,
                candidate_findings=tuple(candidate_findings),
                objections=objections,
                rejected_findings=self._not_established(
                    candidate_findings,
                    "Resolution validation rejected the adjudication output.",
                ),
            )
        accepted, rejected, final_limitations = self._validated_final_findings(
            corpus,
            final_models,
            candidate_findings,
            resolved_objections,
        )
        limitations.extend(final_limitations)

        if (
            self._has_unresolved_blocking(resolved_objections)
            and self.max_adjudication_revisions
        ):
            revision_run = await self._run_role(
                ModelRole.REPORT_ADJUDICATOR,
                cast(object, self.adjudicator),
                corpus,
                candidate_findings=tuple(final_models) or candidate_input,
                objections=resolved_objections,
                initial_evidence_ids=tuple(dict.fromkeys(adjudication_evidence_ids)),
                previous_output=adjudication_response,
                phase="adjudication-revision",
                max_rounds=2,
            )
            retrieval_log.extend(revision_run.retrieval_log)
            if revision_run.response is None:
                limitations.append(
                    revision_run.limitation
                    or "Adjudication repair produced no usable output."
                )
            else:
                adjudication_response = cast(
                    AdjudicationResponse, revision_run.response
                )
                try:
                    resolved_objections = self._apply_resolutions(
                        corpus,
                        resolved_objections,
                        adjudication_response.objection_resolutions,
                        limitations,
                    )
                except (TypeError, ValueError) as error:
                    limitations.append(
                        f"Resolution validation failed during adjudication repair: {self._safe_validation_reason(error)}."
                    )
                    accepted = []
                    rejected = self._not_established(
                        candidate_findings,
                        "Resolution validation rejected the adjudication repair.",
                    )
                else:
                    accepted, rejected, final_limitations = (
                        self._validated_final_findings(
                            corpus,
                            tuple(adjudication_response.final_findings),
                            candidate_findings,
                            resolved_objections,
                        )
                    )
                    limitations.extend(final_limitations)

        accepted, rejected, verification_limitations = self._final_verification(
            corpus,
            accepted,
            rejected,
            resolved_objections,
        )
        limitations.extend(verification_limitations)
        unresolved_blocking = self._has_unresolved_blocking(resolved_objections)
        if unresolved_blocking and accepted:
            limitations.append(
                "Publication validation rejected all final findings because a blocking objection remained unresolved."
            )
            rejected.extend(
                self._not_established(
                    accepted, "An unresolved blocking objection prevented publication."
                )
            )
            accepted = []

        if duplicate_candidate_ids:
            limitations.append("Publication validation rejected duplicate finding IDs.")
            rejected = [
                replace(
                    finding,
                    reviewer_state="not-established",
                    reviewer_notes=(
                        *finding.reviewer_notes,
                        "Publication validation rejected duplicate finding ID.",
                    ),
                )
                for finding in candidate_findings
            ]
            accepted = []

        if accepted:
            status = SynthesisStatus.ACCEPTED
        elif unresolved_blocking or final_models:
            status = SynthesisStatus.REJECTED
        elif candidate_models and not (
            benign_candidate_count == len(candidate_models) and not rejected
        ):
            status = SynthesisStatus.REJECTED
        else:
            status = SynthesisStatus.NO_ISSUES

        return self._attempt(
            corpus,
            attempt_id=attempt_id,
            created_at=created_at,
            status=status,
            limitations=limitations,
            retrieval_log=retrieval_log,
            role_manifest=role_manifest,
            candidate_findings=tuple(candidate_findings),
            objections=resolved_objections,
            rejected_findings=tuple(rejected),
            findings=tuple(accepted),
        )

    async def _run_role(
        self,
        role: ModelRole,
        provider: object,
        corpus: EvidenceCorpus,
        *,
        candidate_findings: Sequence[CandidateFinding | SynthesisFinding],
        objections: Sequence[SynthesisObjection] = (),
        initial_evidence_ids: Sequence[str] = (),
        previous_output: _Response | None = None,
        phase: str = "retrieval",
        max_rounds: int | None = None,
    ) -> _RoleRun:
        logs: list[Mapping[str, object]] = []
        resolved: ResolvedEvidence | None = None
        if initial_evidence_ids:
            try:
                resolved = self.resolver.resolve(
                    corpus,
                    tuple(dict.fromkeys(initial_evidence_ids)),
                    max_entries=self.max_retrieval_entries,
                    max_attachment_bytes=self.max_attachment_bytes,
                )
            except (OSError, RuntimeError, ValueError, TypeError) as error:
                return _RoleRun(
                    response=None,
                    retrieval_log=(
                        {
                            "role": role.value,
                            "phase": "context",
                            "round": 0,
                            "request": tuple(initial_evidence_ids),
                            "response": {"status": "unavailable"},
                            "error": self._safe_validation_reason(error),
                        },
                    ),
                    unavailable=False,
                    invalid=True,
                    limitation="Relevant evidence could not be resolved through the evidence boundary.",
                )
            logs.append(
                {
                    "role": role.value,
                    "phase": "context",
                    "round": 0,
                    "request": tuple(initial_evidence_ids),
                    "resolved_evidence_ids": resolved.evidence_ids,
                    "response": {"status": "resolved"},
                }
            )

        prior = previous_output
        rounds = max_rounds or self.max_retrieval_rounds
        for round_number in range(1, rounds + 1):
            try:
                raw_response = await self._invoke_role(
                    role,
                    provider,
                    corpus,
                    candidate_findings=candidate_findings,
                    objections=objections,
                    resolved_evidence=resolved,
                    previous_output=prior,
                )
                response = _normalize_response(role, raw_response)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                operational, category = _error_category(error)
                logs.append(
                    {
                        "role": role.value,
                        "phase": phase,
                        "round": round_number,
                        "request": (),
                        "response": {"status": "error"},
                        "error": category,
                    }
                )
                return _RoleRun(
                    response=None,
                    retrieval_log=tuple(logs),
                    unavailable=operational,
                    invalid=not operational,
                    limitation=(
                        "Synthesis model transport or configuration failed."
                        if operational
                        else "A synthesis role returned invalid structured output."
                    ),
                )

            log: dict[str, object] = {
                "role": role.value,
                "phase": phase,
                "round": round_number,
                "request": tuple(response.evidence_requests),
                "resolved_evidence_ids": resolved.evidence_ids if resolved else (),
                "response": _response_payload(response),
            }
            if response.complete:
                logs.append(log)
                return _RoleRun(response, tuple(logs))

            try:
                resolved = self.resolver.resolve(
                    corpus,
                    response.evidence_requests,
                    max_entries=self.max_retrieval_entries,
                    max_attachment_bytes=self.max_attachment_bytes,
                )
            except (OSError, RuntimeError, ValueError, TypeError) as error:
                log["response"] = {
                    "status": "unavailable",
                    "reason": self._safe_validation_reason(error),
                }
                logs.append(log)
                return _RoleRun(
                    response=None,
                    retrieval_log=tuple(logs),
                    invalid=True,
                    limitation="A synthesis retrieval request could not be resolved through the evidence boundary.",
                )
            log["resolved_evidence_ids"] = resolved.evidence_ids
            logs.append(log)
            if round_number >= rounds:
                return _RoleRun(
                    response=None,
                    retrieval_log=tuple(logs),
                    unavailable=True,
                    limitation="A synthesis role exceeded its bounded retrieval budget without usable output.",
                )
            prior = response

        return _RoleRun(
            response=None,
            retrieval_log=tuple(logs),
            unavailable=True,
            limitation="A synthesis role exceeded its bounded retrieval budget without usable output.",
        )

    async def _invoke_role(
        self,
        role: ModelRole,
        provider: object,
        corpus: EvidenceCorpus,
        *,
        candidate_findings: Sequence[CandidateFinding | SynthesisFinding],
        objections: Sequence[SynthesisObjection],
        resolved_evidence: ResolvedEvidence | None,
        previous_output: _Response | None,
    ) -> object:
        common = {
            "resolved_evidence": resolved_evidence,
            "previous_output": previous_output,
        }
        if role is ModelRole.REPORT_ANALYST:
            method = getattr(provider, "analyze")
            value = method(corpus, self.principles, **common)
        elif role is ModelRole.REPORT_EVIDENCE_AUDITOR:
            method = getattr(provider, "audit", None) or getattr(provider, "review")
            value = method(corpus, self.principles, candidate_findings, **common)
        elif role is ModelRole.REPORT_PATTERN_REVIEWER:
            method = getattr(provider, "review")
            value = method(corpus, self.principles, candidate_findings, **common)
        else:
            method = getattr(provider, "adjudicate", None) or getattr(
                provider, "resolve"
            )
            value = method(
                corpus,
                self.principles,
                candidate_findings,
                objections,
                **common,
            )
        if isawaitable(value):
            return await cast(Any, value)
        return value

    def _validated_finding(
        self,
        corpus: EvidenceCorpus,
        candidate: CandidateFinding,
        *,
        reviewer_state: str,
    ) -> SynthesisFinding:
        normalized = CandidateFinding.model_validate(
            candidate.model_dump(mode="python")
        )
        finding = normalized.to_domain(reviewer_state=reviewer_state)
        self._validate_finding(corpus, finding)
        if self._is_harmless_alternate(corpus, finding):
            raise _BenignCandidate("harmless alternate path")
        return finding

    def _validate_finding(
        self, corpus: EvidenceCorpus, finding: SynthesisFinding
    ) -> None:
        textual_fields = (
            finding.title,
            finding.issue,
            finding.impact,
            finding.root_cause,
            finding.severity_justification,
            *finding.fixes,
            *finding.affected_surfaces,
            *finding.principles,
            *finding.limitations,
            *finding.reviewer_notes,
            *(item for item in finding.counterevidence if isinstance(item, str)),
        )
        lowered = " ".join(textual_fields).casefold()
        if contains_forbidden_narrative(" ".join(textual_fields)):
            raise ValueError("finding contains forbidden narrative input")
        if not finding.severity_justification.strip():
            raise ValueError("finding requires severity justification")
        evidence_refs = self._finding_evidence_refs(finding)
        if any(marker in lowered for marker in _PRINCIPLE_AUTHORITY_MARKERS):
            raise ValueError("UX principles cannot justify severity")
        if any(
            marker in finding.severity_justification.casefold()
            for marker in ("principle", "heuristic", "guideline")
        ):
            raise ValueError("UX principles cannot justify severity")
        if finding.evidence_class is EvidenceClass.UNSUPPORTED_HUMAN_CLAIM:
            raise ValueError("unsupported human claim cannot become finding")
        validate_evidence_refs(corpus, evidence_refs)
        self.resolver.resolve(
            corpus,
            tuple(ref.evidence_id for ref in evidence_refs),
            max_entries=self.max_retrieval_entries,
            max_attachment_bytes=self.max_attachment_bytes,
        )
        entries = tuple(corpus.require(ref.evidence_id) for ref in evidence_refs)
        if any(
            entry.evidence_class is EvidenceClass.UNSUPPORTED_HUMAN_CLAIM
            for entry in entries
        ):
            raise ValueError("unsupported human claim cannot support a finding")
        if finding.evidence_class is EvidenceClass.DETERMINISTIC_FACT and any(
            entry.evidence_class is not EvidenceClass.DETERMINISTIC_FACT
            for entry in entries
        ):
            raise ValueError(
                "finding evidence class is incompatible with referenced evidence"
            )
        self._validate_verifier_consistency(corpus, finding)
        if (
            _CAUSAL_MARKERS.search(finding.root_cause)
            and len({(ref.run_id, ref.kind) for ref in evidence_refs}) < 2
        ):
            raise ValueError("causal language is not supported by enough evidence")

    @staticmethod
    def _finding_evidence_refs(finding: SynthesisFinding) -> tuple[EvidenceRef, ...]:
        return tuple(finding.evidence_refs) + tuple(
            item for item in finding.counterevidence if isinstance(item, EvidenceRef)
        )

    @staticmethod
    def _validate_verifier_consistency(
        corpus: EvidenceCorpus,
        finding: SynthesisFinding,
    ) -> None:
        runs = {ref.run_id for ref in finding.evidence_refs}
        verification: dict[str, bool] = {}
        for entry in corpus.entries:
            if entry.ref.kind != "verification" or entry.ref.run_id not in runs:
                continue
            value = entry.payload.get("verified")
            if isinstance(value, bool):
                verification[entry.ref.run_id] = value
        text = " ".join((finding.issue, finding.impact, finding.root_cause)).casefold()
        for run_id, verified in verification.items():
            if verified and any(
                marker in text
                for marker in (
                    "failed",
                    "did not complete",
                    "not verified",
                    "abandoned",
                )
            ):
                raise ValueError(
                    f"finding conflicts with verifier outcome for {run_id}"
                )
            if not verified and any(
                marker in text
                for marker in ("completed successfully", "verified successfully")
            ):
                raise ValueError(
                    f"finding conflicts with verifier outcome for {run_id}"
                )

    @staticmethod
    def _is_harmless_alternate(
        corpus: EvidenceCorpus,
        finding: SynthesisFinding,
    ) -> bool:
        text = " ".join(
            (finding.title, finding.issue, finding.impact, finding.root_cause)
        ).casefold()
        if not any(marker in text for marker in _ALTERNATE_PATH_MARKERS):
            return False
        finding_runs = {ref.run_id for ref in finding.evidence_refs}
        expectation_has_alternative = any(
            entry.ref.kind == "expectation"
            and entry.ref.run_id in finding_runs
            and bool(entry.payload.get("acceptable_alternatives"))
            for entry in corpus.entries
        )
        if not expectation_has_alternative:
            return False
        return not any(marker in text for marker in _HARM_MARKERS)

    def _reviewer_objections(
        self,
        corpus: EvidenceCorpus,
        response: EvidenceAuditResponse | PatternReviewResponse,
        role: ModelRole,
    ) -> tuple[SynthesisObjection, ...]:
        objections: list[SynthesisObjection] = []
        seen_ids: set[str] = set()
        for typed in response.objections:
            normalized = TypedObjection.model_validate(typed.model_dump(mode="python"))
            if normalized.objection_id in seen_ids:
                raise ValueError("duplicate reviewer objection ID")
            seen_ids.add(normalized.objection_id)
            domain = normalized.to_domain()
            if contains_forbidden_narrative(
                " ".join(
                    item
                    for item in (
                        domain.message,
                        domain.reviewer_role,
                        domain.resolution or "",
                    )
                    if item
                )
            ):
                raise ValueError(
                    "reviewer objection contains forbidden narrative input"
                )
            validate_evidence_refs(corpus, domain.evidence_refs)
            if domain.evidence_refs:
                self.resolver.resolve(
                    corpus,
                    tuple(ref.evidence_id for ref in domain.evidence_refs),
                    max_entries=self.max_retrieval_entries,
                    max_attachment_bytes=self.max_attachment_bytes,
                )
            reviewer_role = domain.reviewer_role.strip().casefold()
            if not reviewer_role:
                canonical_role = role
            else:
                canonical_role = _REVIEWER_ROLE_ALIASES.get(reviewer_role)
                if canonical_role is None:
                    raise ValueError("reviewer role is not a canonical report role")
                if canonical_role is not role:
                    raise ValueError("reviewer role does not match the reviewing role")
            domain = replace(domain, reviewer_role=canonical_role.value)
            objections.append(domain)
        return tuple(objections)

    @staticmethod
    def _validate_reviewer_objections(
        objections: Sequence[SynthesisObjection],
        candidate_ids: set[str],
    ) -> None:
        seen_ids: set[str] = set()
        for objection in objections:
            if objection.objection_id in seen_ids:
                raise ValueError("duplicate reviewer objection ID")
            seen_ids.add(objection.objection_id)
            if objection.finding_id not in candidate_ids:
                raise ValueError("reviewer objection references unknown finding ID")

    def _apply_resolutions(
        self,
        corpus: EvidenceCorpus,
        objections: Sequence[SynthesisObjection],
        resolutions: Sequence[ObjectionResolution],
        limitations: list[str],
    ) -> tuple[SynthesisObjection, ...]:
        objections_by_id = {item.objection_id: item for item in objections}
        by_id: dict[str, ObjectionResolution] = {}
        for resolution in resolutions:
            if resolution.objection_id in by_id:
                raise ValueError(
                    f"resolution {resolution.objection_id} was provided more than once"
                )
            objection = objections_by_id.get(resolution.objection_id)
            if objection is None:
                raise ValueError(
                    f"resolution {resolution.objection_id} references an unknown objection"
                )
            if resolution.finding_id != objection.finding_id:
                raise ValueError(
                    f"resolution {resolution.objection_id} finding ID does not match objection"
                )
            by_id[resolution.objection_id] = resolution
        result: list[SynthesisObjection] = []
        for objection in objections:
            resolution = by_id.get(objection.objection_id)
            if resolution is None:
                result.append(objection)
                continue
            refs = tuple(ref.to_domain() for ref in resolution.evidence_refs)
            valid_resolution = True
            if contains_forbidden_narrative(resolution.resolution):
                valid_resolution = False
                limitations.append(
                    f"Resolution {resolution.objection_id} contained forbidden narrative input."
                )
            if refs:
                try:
                    validate_evidence_refs(corpus, refs)
                    self.resolver.resolve(
                        corpus,
                        tuple(ref.evidence_id for ref in refs),
                        max_entries=self.max_retrieval_entries,
                        max_attachment_bytes=self.max_attachment_bytes,
                    )
                except (OSError, RuntimeError, TypeError, ValueError):
                    valid_resolution = False
            elif (
                resolution.resolved and objection.severity is ObjectionSeverity.BLOCKING
            ):
                valid_resolution = False
            if not valid_resolution:
                limitations.append(
                    f"Resolution {resolution.objection_id} lacked valid supporting evidence."
                )
                result.append(objection)
                continue
            result.append(
                replace(
                    objection,
                    resolved=resolution.resolved,
                    resolution=resolution.resolution
                    if resolution.resolved or resolution.resolution
                    else None,
                    evidence_refs=refs or objection.evidence_refs,
                )
            )
        return tuple(result)

    def _validated_final_findings(
        self,
        corpus: EvidenceCorpus,
        final_models: Sequence[CandidateFinding],
        candidate_findings: Sequence[SynthesisFinding],
        objections: Sequence[SynthesisObjection],
    ) -> tuple[list[SynthesisFinding], list[SynthesisFinding], list[str]]:
        candidate_ids = {finding.finding_id for finding in candidate_findings}
        accepted: list[SynthesisFinding] = []
        rejected: list[SynthesisFinding] = []
        limitations: list[str] = []
        final_ids = [model.finding_id for model in final_models]
        duplicate_ids = {
            finding_id for finding_id in final_ids if final_ids.count(finding_id) > 1
        }
        if duplicate_ids:
            limitations.extend(
                f"Final finding {finding_id} was duplicated."
                for finding_id in sorted(duplicate_ids)
            )
            return (
                [],
                self._not_established(
                    candidate_findings,
                    "Publication validation rejected duplicate final finding IDs.",
                ),
                limitations,
            )
        for model in final_models:
            if model.finding_id not in candidate_ids:
                limitations.append(
                    f"Final finding {model.finding_id} was not present in analyst candidates."
                )
                continue
            try:
                finding = self._validated_finding(
                    corpus,
                    model,
                    reviewer_state="accepted",
                )
            except _BenignCandidate:
                limitations.append(
                    f"Final finding {model.finding_id} described a valid alternate path and was not published."
                )
                continue
            except (TypeError, ValueError) as error:
                limitations.append(
                    f"Final finding {model.finding_id} failed deterministic publication validation: {self._safe_validation_reason(error)}."
                )
                continue
            if self._blocking_for(finding.finding_id, objections):
                rejected.append(
                    replace(
                        finding,
                        reviewer_state="not-established",
                        reviewer_notes=tuple(finding.reviewer_notes)
                        + ("An unresolved blocking objection prevented publication.",),
                    )
                )
            else:
                accepted.append(finding)
        if not final_models and candidate_findings:
            rejected.extend(
                self._not_established(
                    candidate_findings, "adjudicator published no surviving finding"
                )
            )
        accepted.sort(key=_finding_sort_key)
        return accepted, rejected, limitations

    def _final_verification(
        self,
        corpus: EvidenceCorpus,
        accepted: Sequence[SynthesisFinding],
        rejected: Sequence[SynthesisFinding],
        objections: Sequence[SynthesisObjection],
    ) -> tuple[list[SynthesisFinding], list[SynthesisFinding], list[str]]:
        verified: list[SynthesisFinding] = []
        rejected_values = list(rejected)
        limitations: list[str] = []
        for finding in accepted:
            try:
                self._validate_finding(corpus, finding)
                if self._blocking_for(finding.finding_id, objections):
                    raise ValueError("unresolved blocking objection")
            except (TypeError, ValueError) as error:
                rejected_values.append(
                    replace(
                        finding,
                        reviewer_state="not-established",
                        reviewer_notes=tuple(finding.reviewer_notes)
                        + ("Final deterministic verification failed.",),
                    )
                )
                limitations.append(
                    f"Final finding {finding.finding_id} failed the final verification pass: {self._safe_validation_reason(error)}."
                )
            else:
                verified.append(finding)
        verified.sort(key=_finding_sort_key)
        return verified, rejected_values, limitations

    @staticmethod
    def _blocking_for(
        finding_id: str,
        objections: Sequence[SynthesisObjection],
    ) -> bool:
        return any(
            item.finding_id == finding_id
            and item.severity is ObjectionSeverity.BLOCKING
            and not item.resolved
            for item in objections
        )

    @staticmethod
    def _has_unresolved_blocking(objections: Sequence[SynthesisObjection]) -> bool:
        return any(
            item.severity is ObjectionSeverity.BLOCKING and not item.resolved
            for item in objections
        )

    @staticmethod
    def _not_established(
        findings: Sequence[SynthesisFinding], reason: str
    ) -> list[SynthesisFinding]:
        return [
            replace(
                finding,
                reviewer_state="not-established",
                reviewer_notes=tuple(finding.reviewer_notes) + (reason,),
            )
            for finding in findings
        ]

    @staticmethod
    def _evidence_ids(findings: Sequence[SynthesisFinding]) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                ref.evidence_id
                for finding in findings
                for ref in ReportSynthesisService._finding_evidence_refs(finding)
            )
        )

    def _role_manifests(self) -> Mapping[str, object]:
        values: dict[str, object] = {}
        providers = (
            (ModelRole.REPORT_ANALYST, self.analyst),
            (ModelRole.REPORT_EVIDENCE_AUDITOR, self.evidence_auditor),
            (ModelRole.REPORT_PATTERN_REVIEWER, self.pattern_reviewer),
            (ModelRole.REPORT_ADJUDICATOR, self.adjudicator),
        )
        for role, provider in providers:
            if provider is not None:
                values[role.value] = _manifest_payload(provider, role)
        return values

    def _missing_roles(self) -> tuple[str, ...]:
        missing: list[str] = []
        for role, provider in (
            (ModelRole.REPORT_ANALYST, self.analyst),
            (ModelRole.REPORT_EVIDENCE_AUDITOR, self.evidence_auditor),
            (ModelRole.REPORT_PATTERN_REVIEWER, self.pattern_reviewer),
            (ModelRole.REPORT_ADJUDICATOR, self.adjudicator),
        ):
            if provider is None:
                missing.append(role.value)
        return tuple(missing)

    @staticmethod
    def _base_limitations(corpus: EvidenceCorpus) -> list[str]:
        matched_expectation = any(
            entry.ref.kind == "expectation" and bool(entry.payload.get("matched"))
            for entry in corpus.entries
        )
        if matched_expectation:
            return []
        return [
            "No frozen expectation matched the tested scope; synthesis is limited to observed outcomes and interactions."
        ]

    def _usage(self) -> Mapping[str, float]:
        source = self._model_record_source
        if source is None:
            return {"role_calls": 0.0, "usage_available": 0.0}
        try:
            raw_records = getattr(source, "records")
            records = tuple(raw_records)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return {"role_calls": 0.0, "usage_available": 0.0}
        if not records or any(
            not isinstance(item, ModelCallRecord) for item in records
        ):
            return {"role_calls": 0.0, "usage_available": 0.0}
        return {
            "role_calls": float(len(records)),
            "model_attempts": float(sum(item.attempts for item in records)),
            "prompt_tokens": float(
                sum(item.token_usage.prompt_tokens for item in records)
            ),
            "completion_tokens": float(
                sum(item.token_usage.completion_tokens for item in records)
            ),
            "total_tokens": float(
                sum(item.token_usage.total_tokens for item in records)
            ),
            "latency_ms": float(sum(item.latency_ms for item in records)),
            "usage_available": 1.0,
        }

    def _attempt_identity(self, corpus: EvidenceCorpus) -> tuple[str, str]:
        if self._clock is not None:
            created_at = self._clock()
        else:
            created_at = datetime.now(UTC).isoformat(timespec="seconds")
        self._sequence += 1
        safe_created = created_at.replace(":", "").replace("+", "-")
        digest_prefix = corpus.digest[:12]
        attempt_id = f"synthesis-{safe_created}-{digest_prefix}-{self._sequence}-{time.time_ns()}"
        return attempt_id, created_at

    def _attempt(
        self,
        corpus: EvidenceCorpus,
        *,
        attempt_id: str,
        created_at: str,
        status: SynthesisStatus,
        limitations: Sequence[str],
        retrieval_log: Sequence[Mapping[str, object]],
        role_manifest: Mapping[str, object],
        candidate_findings: Sequence[SynthesisFinding] = (),
        objections: Sequence[SynthesisObjection] = (),
        rejected_findings: Sequence[SynthesisFinding] = (),
        findings: Sequence[SynthesisFinding] = (),
    ) -> SynthesisAttempt:
        expectation_payload = [
            entry.payload for entry in corpus.entries if entry.ref.kind == "expectation"
        ]
        expectation_digest = hashlib.sha256(
            _canonical_json(expectation_payload).encode("utf-8")
        ).hexdigest()
        return SynthesisAttempt(
            attempt_id=attempt_id,
            status=status,
            corpus_digest=corpus.digest,
            expectation_digest=expectation_digest,
            principle_pack_digest=corpus.principle_pack_digest,
            model_manifest={"roles": role_manifest},
            role_manifest=role_manifest,
            prompt_version=REPORT_SYNTHESIS_PROMPT_VERSION,
            schema_version=REPORT_SYNTHESIS_APPLICATION_SCHEMA_VERSION,
            retrieval_log=tuple(retrieval_log),
            usage=self._usage(),
            candidate_findings=tuple(candidate_findings),
            objections=tuple(objections),
            rejected_findings=tuple(rejected_findings),
            findings=tuple(findings),
            limitations=tuple(
                dict.fromkeys(item for item in limitations if item.strip())
            ),
            fallback_available=True,
            created_at=created_at,
        )

    @staticmethod
    def _safe_validation_reason(error: BaseException) -> str:
        name = type(error).__name__
        if name in {"ValueError", "TypeError", "ValidationError"}:
            return "evidence, schema, or publication contract failed"
        return "deterministic validation failed"


__all__ = [
    "DEFAULT_MAX_ATTACHMENT_BYTES",
    "DEFAULT_MAX_RETRIEVAL_ENTRIES",
    "MAX_RETRIEVAL_ROUNDS",
    "REPORT_SYNTHESIS_APPLICATION_SCHEMA_VERSION",
    "REPORT_SYNTHESIS_PROMPT_VERSION",
    "ReportSynthesisService",
]
