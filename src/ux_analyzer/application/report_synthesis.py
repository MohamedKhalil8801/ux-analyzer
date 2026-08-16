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
    CANONICAL_SYNTHESIS_ROLES,
    REPORT_ADJUDICATOR_ROLE,
    EvidenceRef,
    ObjectionSeverity,
    RejectedCandidateAudit,
    SynthesisAttempt,
    SynthesisFinding,
    SynthesisObjection,
    SynthesisRoleReceipt,
    SynthesisStatus,
    final_finding_preserves_candidate,
)
from ux_analyzer.ports.model_transport import (
    MODEL_ATTACHMENT_MAX_BYTES,
    TransportBudgetError,
    TransportEvidenceUnavailableError,
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

REPORT_SYNTHESIS_APPLICATION_SCHEMA_VERSION = "synthesis-v2"
REPORT_SYNTHESIS_PROMPT_VERSION = "report-synthesis-orchestrator-v1"
MAX_RETRIEVAL_ROUNDS = 3
DEFAULT_MAX_RETRIEVAL_ENTRIES = 16
MAX_ROLE_RETRIEVAL_ENTRIES = 32
DEFAULT_MAX_ATTACHMENT_BYTES = MODEL_ATTACHMENT_MAX_BYTES

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
_PRIMARY_OBSERVED_EVIDENCE_KINDS = frozenset(
    {
        "event",
        "replay",
        "verification",
        "metric",
        "viewport",
        "element",
        "screenshot",
        "heatmap",
    }
)
_UI_STATE_EVIDENCE_KINDS = frozenset({"viewport", "element", "screenshot", "heatmap"})
_BEHAVIOR_OR_OUTCOME_EVIDENCE_KINDS = frozenset(
    {"event", "replay", "verification", "metric"}
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
    "evidence auditor": ModelRole.REPORT_EVIDENCE_AUDITOR,
    "report-pattern-reviewer": ModelRole.REPORT_PATTERN_REVIEWER,
    "pattern-reviewer": ModelRole.REPORT_PATTERN_REVIEWER,
    "pattern reviewer": ModelRole.REPORT_PATTERN_REVIEWER,
    "report-analyst": ModelRole.REPORT_ANALYST,
    "ux-analyst": ModelRole.REPORT_ANALYST,
}


def _scoped_objection_id(role: ModelRole, objection_id: str) -> str:
    """Make reviewer-local IDs unique without changing their model-authored meaning."""

    prefix = f"{role.value}:"
    scoped = f"{prefix}{objection_id}"
    if len(scoped) <= 256:
        return scoped
    digest = hashlib.sha256(objection_id.encode("utf-8")).hexdigest()[:16]
    available = 256 - len(prefix) - len(digest) - 1
    return f"{prefix}{objection_id[:available]}-{digest}"

_Response = (
    AnalystResponse
    | EvidenceAuditResponse
    | PatternReviewResponse
    | AdjudicationResponse
)


class _BenignCandidate(ValueError):
    """Candidate rejected because it describes a harmless valid alternative."""


class _EvidenceBatchResolutionError(ValueError):
    """Carry per-batch audit records when bounded evidence resolution fails."""

    def __init__(
        self,
        cause: BaseException,
        batch_logs: Sequence[Mapping[str, object]],
    ) -> None:
        self.cause = cause
        self.batch_logs = tuple(batch_logs)
        super().__init__(str(cause))


@dataclass(frozen=True, slots=True)
class _RoleRun:
    response: _Response | None
    retrieval_log: tuple[Mapping[str, object], ...]
    receipt: SynthesisRoleReceipt | None = None
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


def _retain_response_limitations(target: list[str], response: _Response) -> None:
    for limitation in response.limitations:
        if limitation not in target:
            target.append(limitation)


def _error_category(error: BaseException) -> tuple[bool, str]:
    name = type(error).__name__
    reason = getattr(error, "reason", None)
    if isinstance(error, TransportEvidenceUnavailableError):
        return True, "visual evidence unavailable"
    if isinstance(error, TransportBudgetError):
        return True, "report request exceeded transport budget"
    if name == "ModelFailureError" and reason == "invalid structured output":
        return False, "invalid structured synthesis output"
    if name == "ModelFailureError" and reason == "model unavailable":
        return True, "model provider unavailable"
    if name == "ModelFailureError" and reason in {
        "request rejected",
        "safety rejection",
        "authentication failure",
    }:
        return True, "model provider rejected request"
    if name in _OPERATIONAL_FAILURE_NAMES or isinstance(error, RuntimeError):
        return True, "model transport or configuration failure"
    if isinstance(
        error, (ModelResponseValidationError, ValidationError, ValueError, TypeError)
    ):
        return False, "invalid structured synthesis output"
    return False, "synthesis role failure"


def _provider_failure_details(error: BaseException) -> dict[str, object]:
    details: dict[str, object] = {}
    status_code = getattr(error, "status_code", None)
    if type(status_code) is int:
        details["status_code"] = status_code
    for attribute, key in (
        ("error_code", "error_code"),
        ("error_type", "error_type"),
        ("request_id", "request_id"),
    ):
        value = getattr(error, attribute, None)
        if (
            isinstance(value, str)
            and 0 < len(value) <= 256
            and "\r" not in value
            and "\n" not in value
        ):
            details[key] = value
    diagnostics = getattr(error, "diagnostics", None)
    if isinstance(diagnostics, Mapping):
        safe_diagnostics = _safe_structural_diagnostics(
            cast(Mapping[object, object], diagnostics)
        )
        if safe_diagnostics:
            details["diagnostics"] = safe_diagnostics
    elif isinstance(error, ModelResponseValidationError):
        reason_code, reason = _safe_role_validation_reason(error.reason)
        details["diagnostics"] = _safe_structural_diagnostics(
            {
                "role": error.role.value,
                "stage": "role_validation",
                "error_type": type(error).__name__,
                "validation_reason_code": reason_code,
                "validation_reason": reason,
                "response_summary": error.response_summary,
            }
        )
    return details


def _safe_role_validation_reason(reason: str) -> tuple[str, str]:
    normalized = reason.strip().casefold()
    known_reasons = (
        (
            "response exceeds bounded output limits",
            "bounded-output",
            "response exceeds bounded output limits",
        ),
        (
            "response limitation contains forbidden narrative",
            "forbidden-narrative",
            "response limitation contains forbidden narrative",
        ),
        (
            "undelivered evidence id",
            "undelivered-evidence-id",
            "undelivered evidence ID",
        ),
        (
            "finding references an unknown ux principle",
            "unknown-principle",
            "finding references an unknown UX principle",
        ),
        ("incomplete response", "incomplete-response", "incomplete response"),
        (
            "invalid evidence request id",
            "invalid-evidence-id",
            "invalid evidence request ID",
        ),
        (
            "evidence request cannot be validated",
            "missing-corpus-evidence",
            "evidence request cannot be validated without corpus IDs",
        ),
        ("unknown evidence id", "unknown-evidence-id", "unknown evidence ID"),
        ("duplicate evidence id", "duplicate-evidence-id", "duplicate evidence ID"),
        (
            "principles are not evidence",
            "principle-used-as-evidence",
            "principles are not evidence",
        ),
        (
            "unknown ux principle",
            "unknown-principle",
            "finding references an unknown UX principle",
        ),
    )
    for prefix, code, safe_message in known_reasons:
        if normalized.startswith(prefix):
            return code, safe_message
    if normalized == "response schema validation failed":
        return "schema-validation", "response schema validation failed"
    return "role-validation-failed", "role validation failed"


def _safe_response_summary(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    summary = cast(Mapping[object, object], value)
    result: dict[str, object] = {}
    for name in ("schema", "parsed_type"):
        item = summary.get(name)
        if isinstance(item, str) and 0 < len(item) <= 128 and "\n" not in item:
            result[name] = item
    for name in ("top_level_key_count", "candidate_count", "objection_count"):
        item = summary.get(name)
        if type(item) is int and 0 <= item <= 10_000:
            result[name] = item
    return result


def _safe_structural_diagnostics(value: Mapping[object, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    enum_fields = {
        "role": {role.value for role in ModelRole if role.value.startswith("report-")},
        "response_mode": {"plain", "strict", "json-object"},
        "stage": {
            "content_parsing",
            "normalization",
            "schema_validation",
            "request_budget",
            "role_validation",
        },
        "response_content_type": {
            "missing",
            "object",
            "list",
            "tuple",
            "bool",
            "str",
            "int",
            "float",
            "null",
            "other",
        },
        "parsed_type": {
            "object",
            "list",
            "tuple",
            "bool",
            "str",
            "int",
            "float",
            "null",
            "other",
        },
        "finish_reason": {
            "stop",
            "length",
            "tool_calls",
            "function_call",
            "content_filter",
        },
        "error_type": {
            "ValueError",
            "TypeError",
            "ValidationError",
            "ModelResponseValidationError",
        },
    }
    for name, allowed in enum_fields.items():
        item = value.get(name)
        if isinstance(item, str) and item in allowed:
            result[name] = item
    for name in (
        "attempt_count",
        "response_content_length",
        "response_content_part_count",
        "request_bytes",
        "budget_bytes",
        "attachment_bytes",
        "attachment_bytes_deferred",
        "attachment_count",
        "resolved_evidence_count",
        "resolved_evidence_deferred",
    ):
        item = value.get(name)
        if type(item) is int and 0 <= item <= 1_000_000_000:
            result[name] = item
    markers = value.get("response_content_markers")
    allowed_markers = {
        "missing",
        "content_parts",
        "text_only_parts",
        "mixed_parts",
        "multiple_parts",
        "empty",
        "text_part",
        "structured_object",
        "object",
        "list",
        "tuple",
        "bool",
        "str",
        "int",
        "float",
        "null",
        "other",
        "markdown_fence",
        "object_candidate",
        "prose_prefix",
        "raw_object_prefix",
        "array_prefix",
        "scalar_prefix",
        "prose_suffix",
        "prose",
    }
    if isinstance(markers, Sequence) and not isinstance(markers, (str, bytes)):
        safe_markers = [
            item
            for item in cast(Sequence[object], markers)[:16]
            if isinstance(item, str) and item in allowed_markers
        ]
        if safe_markers:
            result["response_content_markers"] = safe_markers
    names = value.get("top_level_keys")
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        safe_names = [
            item
            for item in cast(Sequence[object], names)[:64]
            if _safe_diagnostic_name(item)
        ]
        if safe_names:
            result["top_level_keys"] = safe_names
    value_types = value.get("top_level_value_types")
    if isinstance(value_types, Mapping):
        safe_types = {
            str(key): item
            for key, item in cast(Mapping[object, object], value_types).items()
            if _safe_diagnostic_name(key)
            and isinstance(item, str)
            and item
            in {
                "object",
                "list",
                "tuple",
                "bool",
                "str",
                "int",
                "float",
                "null",
                "other",
            }
        }
        if safe_types:
            result["top_level_value_types"] = safe_types
    errors = value.get("validation_errors")
    if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes)):
        safe_errors = [
            safe
            for item in cast(Sequence[object], errors)[:16]
            if (safe := _safe_validation_error(item)) is not None
        ]
        if safe_errors:
            result["validation_errors"] = safe_errors
    for name in ("validation_reason_code", "validation_reason"):
        item = value.get(name)
        if isinstance(item, str) and re.fullmatch(r"[A-Za-z0-9 ._-]{1,128}", item):
            result[name] = item
    summary = _safe_response_summary(value.get("response_summary"))
    if summary:
        result["response_summary"] = summary
    return result


_DIAGNOSTIC_SCHEMA_FIELDS = frozenset(
    {
        "complete",
        "evidence_requests",
        "unavailable_evidence_ids",
        "candidate_findings",
        "objections",
        "final_findings",
        "objection_resolutions",
        "finding_id",
        "title",
        "issue",
        "impact",
        "root_cause",
        "fixes",
        "severity",
        "confidence",
        "evidence_refs",
        "affected_surfaces",
        "principles",
        "counterevidence",
        "limitations",
        "reviewer_state",
        "evidence_class",
        "reproducibility",
        "severity_justification",
        "reviewer_notes",
        "objection_id",
        "objection_type",
        "message",
        "reviewer_role",
        "resolved",
        "resolution",
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
    }
)
_DIAGNOSTIC_HASHED_NAME = re.compile(r"^unknown-[0-9a-f]{12}$")


def _safe_diagnostic_name(value: object) -> bool:
    return isinstance(value, str) and (
        value in _DIAGNOSTIC_SCHEMA_FIELDS
        or _DIAGNOSTIC_HASHED_NAME.fullmatch(value) is not None
    )


def _safe_validation_error(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    item = cast(Mapping[object, object], value)
    path = item.get("path")
    error_type = item.get("type")
    if not isinstance(path, Sequence) or isinstance(path, (str, bytes)):
        return None
    safe_path: list[object] = []
    for component in cast(Sequence[object], path)[:16]:
        if type(component) is int and 0 <= component <= 10_000:
            safe_path.append(component)
        elif _safe_diagnostic_name(component):
            safe_path.append(component)
        else:
            return None
    if (
        not isinstance(error_type, str)
        or re.fullmatch(r"[a-z0-9_]{1,64}", error_type) is None
    ):
        return None
    return {"path": safe_path, "type": error_type}


def _operational_limitation(error: BaseException, category: str) -> str:
    if category == "visual evidence unavailable":
        limitation = (
            "Visual evidence was unavailable within the bounded transport request."
        )
    elif category == "report request exceeded transport budget":
        limitation = "The synthesis request exceeded its bounded transport budget."
    elif category == "model provider unavailable":
        limitation = (
            "The synthesis provider reports that the configured model is unavailable."
        )
    elif category == "model provider rejected request":
        limitation = "The synthesis provider rejected the report request."
    else:
        limitation = "Synthesis model transport or configuration failed."
    details = _provider_failure_details(error)
    status_code = details.get("status_code")
    if type(status_code) is int:
        limitation += f" HTTP {status_code}."
    error_code = details.get("error_code")
    if isinstance(error_code, str):
        limitation += f" Provider code: {error_code}."
    request_id = details.get("request_id")
    if isinstance(request_id, str):
        limitation += f" Request ID: {request_id}."
    return limitation


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
        role_receipts: dict[str, SynthesisRoleReceipt] = {}
        rejected_candidate_audits: list[RejectedCandidateAudit] = []
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
        if analyst_run.receipt is not None:
            role_receipts[analyst_run.receipt.role] = analyst_run.receipt
        _retain_response_limitations(limitations, analyst_response)
        candidate_models = tuple(analyst_response.candidate_findings)
        candidate_findings: list[SynthesisFinding] = []
        benign_candidate_count = 0
        candidate_limitations: list[str] = []
        duplicate_candidate_ids: set[str] = set()
        seen_candidate_ids: set[str] = set()
        retained_candidate_ids: set[str] = set()
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
                rejected_candidate_audits.append(
                    self._rejected_candidate_audit(candidate, error)
                )
                candidate_limitations.append(
                    f"Candidate {candidate.finding_id} failed deterministic publication validation: {self._safe_validation_reason(error)}."
                )
            else:
                if finding.finding_id not in retained_candidate_ids:
                    candidate_findings.append(finding)
                    retained_candidate_ids.add(finding.finding_id)
        limitations.extend(candidate_limitations)

        candidate_ids = {finding.finding_id for finding in candidate_findings}
        candidate_input_by_id = {
            candidate.finding_id: candidate
            for candidate in reversed(candidate_models)
            if candidate.finding_id in candidate_ids
        }
        candidate_input = tuple(
            candidate_input_by_id[finding.finding_id] for finding in candidate_findings
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
                role_receipts=tuple(role_receipts.values()),
                rejected_candidate_audits=tuple(rejected_candidate_audits),
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
                role_receipts=tuple(role_receipts.values()),
                rejected_candidate_audits=tuple(rejected_candidate_audits),
                candidate_findings=tuple(candidate_findings),
                rejected_findings=self._not_established(
                    candidate_findings,
                    "Pattern reviewer output was not trustworthy.",
                ),
            )

        auditor_response = cast(EvidenceAuditResponse, auditor_run.response)
        pattern_response = cast(PatternReviewResponse, pattern_run.response)
        for role_run in (auditor_run, pattern_run):
            if role_run.receipt is not None:
                role_receipts[role_run.receipt.role] = role_run.receipt
        _retain_response_limitations(limitations, auditor_response)
        _retain_response_limitations(limitations, pattern_response)
        try:
            auditor_objections = self._reviewer_objections(
                corpus,
                auditor_response,
                ModelRole.REPORT_EVIDENCE_AUDITOR,
            )
            pattern_objections = self._reviewer_objections(
                corpus,
                pattern_response,
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
                role_receipts=tuple(role_receipts.values()),
                rejected_candidate_audits=tuple(rejected_candidate_audits),
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
                role_receipts=tuple(role_receipts.values()),
                rejected_candidate_audits=tuple(rejected_candidate_audits),
                candidate_findings=tuple(candidate_findings),
                objections=objections,
                rejected_findings=self._not_established(
                    candidate_findings,
                    "Adjudicator output was not trustworthy.",
                ),
            )

        adjudication_response = cast(AdjudicationResponse, adjudication_run.response)
        if adjudication_run.receipt is not None:
            role_receipts[adjudication_run.receipt.role] = adjudication_run.receipt
        _retain_response_limitations(limitations, adjudication_response)
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
                role_receipts=tuple(role_receipts.values()),
                rejected_candidate_audits=tuple(rejected_candidate_audits),
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

        for _revision_number in range(self.max_adjudication_revisions):
            if not self._has_unresolved_blocking(resolved_objections):
                break
            revision_run = await self._run_role(
                ModelRole.REPORT_ADJUDICATOR,
                cast(object, self.adjudicator),
                corpus,
                candidate_findings=candidate_input,
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
                break
            else:
                adjudication_response = cast(
                    AdjudicationResponse, revision_run.response
                )
                if revision_run.receipt is not None:
                    role_receipts[revision_run.receipt.role] = revision_run.receipt
                _retain_response_limitations(limitations, adjudication_response)
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
                    break
                else:
                    final_models = tuple(adjudication_response.final_findings)
                    accepted, rejected, final_limitations = (
                        self._validated_final_findings(
                            corpus,
                            final_models,
                            candidate_findings,
                            resolved_objections,
                        )
                    )
                    limitations.extend(final_limitations)

        for _verification_number in range(self.max_final_verifications):
            accepted, rejected, verification_limitations = self._final_verification(
                corpus,
                accepted,
                rejected,
                resolved_objections,
            )
            limitations.extend(verification_limitations)
            if not accepted:
                break
        unresolved_blocking = self._has_unresolved_blocking(resolved_objections)
        publication_invalid = False

        undispositioned = tuple(
            objection
            for objection in resolved_objections
            if objection.resolution is None
            or objection.resolved_by_role != REPORT_ADJUDICATOR_ROLE
        )
        if undispositioned:
            publication_invalid = True
            limitations.append(
                "Publication validation requires an explicit disposition from the report adjudicator for every objection."
            )
            rejected = self._not_established(
                candidate_findings,
                "One or more reviewer objections lacked an explicit adjudicator disposition.",
            )
            accepted = []

        if set(role_receipts) != set(CANONICAL_SYNTHESIS_ROLES):
            publication_invalid = True
            limitations.append(
                "Publication validation requires validated completion receipts for all four synthesis roles."
            )
            rejected = self._not_established(
                candidate_findings,
                "The synthesis role completion provenance was incomplete.",
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

        if publication_invalid:
            status = SynthesisStatus.REJECTED
        elif accepted:
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
            role_receipts=tuple(role_receipts.values()),
            rejected_candidate_audits=tuple(rejected_candidate_audits),
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
        cumulative_requested: set[str] = set()
        if initial_evidence_ids:
            cumulative_requested.update(initial_evidence_ids)
            try:
                resolved, batches = self._resolve_evidence_batches(
                    corpus,
                    tuple(dict.fromkeys(initial_evidence_ids)),
                    role=role,
                    phase="context",
                    round_number=0,
                )
            except (OSError, RuntimeError, ValueError, TypeError) as error:
                cause = getattr(error, "cause", error)
                batches = getattr(error, "batch_logs", ())
                return _RoleRun(
                    response=None,
                    retrieval_log=(
                        {
                            "role": role.value,
                            "phase": "context",
                            "round": 0,
                            "request": tuple(initial_evidence_ids),
                            "batches": tuple(batches),
                            "response": {"status": "rejected"},
                            "error": self._safe_validation_reason(cause),
                        },
                    ),
                    unavailable=False,
                    invalid=True,
                    limitation=(
                        "The synthesis evidence boundary rejected the initial role "
                        "context."
                    ),
                )
            logs.append(
                {
                    "role": role.value,
                    "phase": "context",
                    "round": 0,
                    "request": tuple(initial_evidence_ids),
                    "batches": batches,
                    "resolved_evidence_ids": resolved.evidence_ids,
                    "response": {"status": "resolved"},
                }
            )

        prior = previous_output
        rounds = max_rounds or self.max_retrieval_rounds
        for round_number in range(1, rounds + 1):
            prior_role_record_count = self._role_record_count(role)
            try:
                raw_response = await self._invoke_role(
                    role,
                    provider,
                    corpus,
                    candidate_findings=candidate_findings,
                    objections=objections,
                    resolved_evidence=resolved,
                    previous_output=prior,
                    retrieval_round=round_number,
                    max_retrieval_rounds=rounds,
                )
                response = _normalize_response(role, raw_response)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                operational, category = _error_category(error)
                response_payload: dict[str, object] = {"status": "error"}
                provider_details = _provider_failure_details(error)
                if provider_details:
                    response_payload["provider"] = provider_details
                logs.append(
                    {
                        "role": role.value,
                        "phase": phase,
                        "round": round_number,
                        "request": (),
                        "response": response_payload,
                        "error": category,
                    }
                )
                return _RoleRun(
                    response=None,
                    retrieval_log=tuple(logs),
                    unavailable=operational,
                    invalid=not operational,
                    limitation=(
                        _operational_limitation(error, category)
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
                return _RoleRun(
                    response,
                    tuple(logs),
                    receipt=self._completion_receipt(
                        role,
                        provider,
                        response,
                        prior_role_record_count=prior_role_record_count,
                    ),
                )

            try:
                requested_ids = tuple(dict.fromkeys(response.evidence_requests))
                if cumulative_requested.intersection(requested_ids):
                    raise ValueError("repeated evidence request")
                if (
                    len(cumulative_requested | set(requested_ids))
                    > MAX_ROLE_RETRIEVAL_ENTRIES
                ):
                    raise ValueError("cumulative evidence request limit exceeded")
                cumulative_requested.update(requested_ids)
                newly_resolved, batches = self._resolve_evidence_batches(
                    corpus,
                    response.evidence_requests,
                    role=role,
                    phase=phase,
                    round_number=round_number,
                )
                if resolved is None:
                    resolved = newly_resolved
                else:
                    combined_entries = {
                        entry.ref.evidence_id: entry for entry in resolved.entries
                    }
                    combined_entries.update(
                        {
                            entry.ref.evidence_id: entry
                            for entry in newly_resolved.entries
                        }
                    )
                    resolved = ResolvedEvidence.from_entries(
                        corpus,
                        tuple(combined_entries.values()),
                        max_entries=MAX_ROLE_RETRIEVAL_ENTRIES,
                        max_attachment_bytes=self.max_attachment_bytes,
                    )
            except (OSError, RuntimeError, ValueError, TypeError) as error:
                cause = getattr(error, "cause", error)
                log["response"] = {
                    "status": "rejected",
                    "reason": self._safe_validation_reason(cause),
                }
                log["batches"] = getattr(error, "batch_logs", ())
                logs.append(log)
                return _RoleRun(
                    response=None,
                    retrieval_log=tuple(logs),
                    invalid=True,
                    limitation=(
                        "The synthesis evidence boundary rejected a retrieval request."
                    ),
                )
            log["resolved_evidence_ids"] = resolved.evidence_ids
            log["batches"] = batches
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

    def _resolve_evidence_batches(
        self,
        corpus: EvidenceCorpus,
        evidence_ids: Sequence[str],
        *,
        role: ModelRole,
        phase: str,
        round_number: int,
    ) -> tuple[ResolvedEvidence, tuple[Mapping[str, object], ...]]:
        """Resolve one logical request through resolver-sized, audited batches."""

        requested = tuple(dict.fromkeys(evidence_ids))
        if not requested:
            raise ValueError("evidence request must not be empty")
        if len(requested) > MAX_ROLE_RETRIEVAL_ENTRIES:
            raise ValueError("evidence request exceeds role retrieval limit")
        batch_count = (
            len(requested) + self.max_retrieval_entries - 1
        ) // self.max_retrieval_entries
        entries: list[object] = []
        attachment_bytes = 0
        batch_logs: list[dict[str, object]] = []
        try:
            for batch_index in range(batch_count):
                start = batch_index * self.max_retrieval_entries
                batch = requested[start : start + self.max_retrieval_entries]
                batch_log: dict[str, object] = {
                    "batch_index": batch_index,
                    "batch_count": batch_count,
                    "request": batch,
                    "response": {"status": "resolving"},
                }
                batch_logs.append(batch_log)
                remaining_attachment_bytes = max(
                    1, self.max_attachment_bytes - attachment_bytes
                )
                resolved_batch = self.resolver.resolve(
                    corpus,
                    batch,
                    max_entries=self.max_retrieval_entries,
                    max_attachment_bytes=remaining_attachment_bytes,
                )
                entries.extend(resolved_batch.entries)
                attachment_bytes += resolved_batch.attachment_bytes
                if attachment_bytes > self.max_attachment_bytes:
                    raise ValueError("cumulative attachment byte limit exceeded")
                batch_log["resolved_evidence_ids"] = resolved_batch.evidence_ids
                batch_log["attachment_bytes"] = resolved_batch.attachment_bytes
                batch_log["cumulative_attachment_bytes"] = attachment_bytes
                batch_log["response"] = {"status": "resolved"}
            resolved = ResolvedEvidence.from_entries(
                corpus,
                entries,
                max_entries=len(requested),
                max_attachment_bytes=self.max_attachment_bytes,
            )
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            batch_logs[-1]["response"] = {
                "status": "rejected",
                "reason": self._safe_validation_reason(error),
            }
            raise _EvidenceBatchResolutionError(error, batch_logs) from error
        for batch_log in batch_logs:
            batch_log.update(
                {
                    "role": role.value,
                    "phase": phase,
                    "round": round_number,
                }
            )
        return resolved, tuple(batch_logs)

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
        retrieval_round: int,
        max_retrieval_rounds: int,
    ) -> object:
        common = {
            "resolved_evidence": resolved_evidence,
            "previous_output": previous_output,
            "retrieval_round": retrieval_round,
            "max_retrieval_rounds": max_retrieval_rounds,
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
        evidence_refs = tuple(finding.evidence_refs)
        counterevidence_refs = tuple(
            item for item in finding.counterevidence if isinstance(item, EvidenceRef)
        )
        if any(marker in lowered for marker in _PRINCIPLE_AUTHORITY_MARKERS):
            raise ValueError("UX principles cannot justify severity")
        if any(
            marker in finding.severity_justification.casefold()
            for marker in ("principle", "heuristic", "guideline")
        ):
            raise ValueError("UX principles cannot justify severity")
        if finding.evidence_class is EvidenceClass.UNSUPPORTED_HUMAN_CLAIM:
            raise ValueError("unsupported human claim cannot become finding")
        self._validate_and_resolve_references(
            corpus,
            (*evidence_refs, *counterevidence_refs),
            role=ModelRole.REPORT_ANALYST,
            phase="publication-validation",
        )
        entries = tuple(corpus.require(ref.evidence_id) for ref in evidence_refs)
        counter_entries = tuple(
            corpus.require(ref.evidence_id) for ref in counterevidence_refs
        )
        if any(
            entry.evidence_class is EvidenceClass.UNSUPPORTED_HUMAN_CLAIM
            for entry in (*entries, *counter_entries)
        ):
            raise ValueError("unsupported human claim cannot support a finding")
        if finding.evidence_class is EvidenceClass.DETERMINISTIC_FACT and any(
            entry.evidence_class is not EvidenceClass.DETERMINISTIC_FACT
            for entry in entries
        ):
            raise ValueError(
                "finding evidence class is incompatible with referenced evidence"
            )
        evidence_kinds = {entry.ref.kind for entry in entries}
        if not evidence_kinds.intersection(_PRIMARY_OBSERVED_EVIDENCE_KINDS):
            raise ValueError("finding has no primary observed evidence")
        if finding.affected_surfaces:
            supported_surfaces = {
                value.casefold()
                for entry in entries
                for value in self._surface_tokens(entry.payload)
            }
            missing_surfaces = tuple(
                surface
                for surface in finding.affected_surfaces
                if surface.casefold() not in supported_surfaces
            )
            if missing_surfaces:
                raise ValueError(
                    "affected surfaces are not named by supporting evidence"
                )
        self._validate_verifier_consistency(corpus, finding)
        if _CAUSAL_MARKERS.search(finding.root_cause) and not (
            {entry.ref.kind for entry in entries}.intersection(_UI_STATE_EVIDENCE_KINDS)
            and {entry.ref.kind for entry in entries}.intersection(
                _BEHAVIOR_OR_OUTCOME_EVIDENCE_KINDS
            )
        ):
            raise ValueError("causal language is not supported by enough evidence")

    @staticmethod
    def _finding_evidence_refs(finding: SynthesisFinding) -> tuple[EvidenceRef, ...]:
        return tuple(finding.evidence_refs) + tuple(
            item for item in finding.counterevidence if isinstance(item, EvidenceRef)
        )

    @staticmethod
    def _surface_tokens(payload: Mapping[str, object]) -> tuple[str, ...]:
        values: list[str] = []
        for key, raw in payload.items():
            if str(key).casefold() not in {
                "surface",
                "surface_id",
                "surface_ids",
                "surfaces",
            }:
                continue
            if isinstance(raw, str) and raw.strip():
                values.append(raw.strip())
            elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                raw_values = cast(Sequence[object], raw)
                values.extend(
                    item.strip()
                    for item in raw_values
                    if isinstance(item, str) and item.strip()
                )
        return tuple(values)

    def _validate_and_resolve_references(
        self,
        corpus: EvidenceCorpus,
        refs: Sequence[EvidenceRef],
        *,
        role: ModelRole,
        phase: str,
    ) -> None:
        validate_evidence_refs(corpus, refs)
        if refs:
            self._resolve_evidence_batches(
                corpus,
                tuple(ref.evidence_id for ref in refs),
                role=role,
                phase=phase,
                round_number=0,
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
            domain = replace(
                normalized.to_domain(),
                objection_id=_scoped_objection_id(role, normalized.objection_id),
                objection_type=normalized.objection_type,
                resolved=False,
                resolution=None,
                resolved_by_role=None,
                resolution_evidence_refs=(),
            )
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
            self._validate_and_resolve_references(
                corpus,
                domain.evidence_refs,
                role=role,
                phase="review-validation",
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
            resolution_id = self._resolution_objection_id(
                objections_by_id, resolution.objection_id
            )
            if resolution_id in by_id:
                raise ValueError(
                    f"resolution {resolution_id} was provided more than once"
                )
            objection = objections_by_id.get(resolution_id)
            if objection is None:
                raise ValueError(
                    f"resolution {resolution.objection_id} references an unknown objection"
                )
            if resolution.finding_id != objection.finding_id:
                raise ValueError(
                    f"resolution {resolution.objection_id} finding ID does not match objection"
                )
            by_id[resolution_id] = resolution.model_copy(
                update={"objection_id": resolution_id}
            )
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
                    self._validate_and_resolve_references(
                        corpus,
                        refs,
                        role=ModelRole.REPORT_ADJUDICATOR,
                        phase="resolution-validation",
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
                    resolved_by_role=REPORT_ADJUDICATOR_ROLE,
                    resolution_evidence_refs=refs,
                )
            )
        return tuple(result)

    @staticmethod
    def _resolution_objection_id(
        objections_by_id: Mapping[str, SynthesisObjection], objection_id: str
    ) -> str:
        if objection_id in objections_by_id:
            return objection_id
        suffix = f":{objection_id}"
        matches = tuple(
            candidate for candidate in objections_by_id if candidate.endswith(suffix)
        )
        if len(matches) == 1:
            return matches[0]
        return objection_id

    def _validated_final_findings(
        self,
        corpus: EvidenceCorpus,
        final_models: Sequence[CandidateFinding],
        candidate_findings: Sequence[SynthesisFinding],
        objections: Sequence[SynthesisObjection],
    ) -> tuple[list[SynthesisFinding], list[SynthesisFinding], list[str]]:
        candidates_by_id = {
            finding.finding_id: finding for finding in candidate_findings
        }
        candidate_ids = set(candidates_by_id)
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
            reviewed = candidates_by_id[model.finding_id]
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
            if not final_finding_preserves_candidate(
                finding,
                reviewed,
                objections=objections,
            ):
                limitations.append(
                    f"Final finding {model.finding_id} failed publication validation because it changed the reviewed core claim."
                )
                rejected.append(
                    replace(
                        reviewed,
                        reviewer_state="not-established",
                        reviewer_notes=tuple(reviewed.reviewer_notes)
                        + ("The adjudicator changed the reviewed core claim.",),
                    )
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
        disposed_ids = {finding.finding_id for finding in (*accepted, *rejected)}
        rejected.extend(
            self._not_established(
                tuple(
                    finding
                    for finding in candidate_findings
                    if finding.finding_id not in disposed_ids
                ),
                "adjudicator published no surviving finding",
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

    def _completion_receipt(
        self,
        role: ModelRole,
        provider: object,
        response: _Response,
        *,
        prior_role_record_count: int | None,
    ) -> SynthesisRoleReceipt | None:
        manifest = _manifest_payload(provider, role)
        if str(manifest.get("role", "")) != role.value:
            return None
        if str(manifest.get("schema_version", "")) != response.schema_version:
            return None
        prompt_digest: str | None = None
        source = self._model_record_source
        if source is None:
            return None
        matching = self._role_call_records(role)
        if prior_role_record_count is None or len(matching) <= prior_role_record_count:
            return None
        record = matching[-1]
        if record.schema_version != response.schema_version:
            return None
        if re.fullmatch(r"[0-9a-f]{64}", record.prompt_digest) is None:
            return None
        prompt_digest = record.prompt_digest
        provider_id = str(manifest.get("provider_id", ""))
        model_id = str(manifest.get("model_id", ""))
        if (
            not provider_id
            or not model_id
            or provider_id == "unavailable"
            or model_id == "unavailable"
        ):
            return None
        return SynthesisRoleReceipt(
            role=role.value,
            provider_id=provider_id,
            model_id=model_id,
            prompt_digest=prompt_digest,
            schema_digest=hashlib.sha256(
                _canonical_json(_role_schema(role).model_json_schema()).encode("utf-8")
            ).hexdigest(),
            output_digest=hashlib.sha256(
                _canonical_json(response.model_dump(mode="python")).encode("utf-8")
            ).hexdigest(),
        )

    def _role_call_records(self, role: ModelRole) -> tuple[ModelCallRecord, ...]:
        source = self._model_record_source
        if source is None:
            return ()
        try:
            records = tuple(getattr(source, "records"))
        except (AttributeError, TypeError, RuntimeError, ValueError):
            return ()
        return tuple(
            record
            for record in records
            if isinstance(record, ModelCallRecord) and record.role is role
        )

    def _role_record_count(self, role: ModelRole) -> int | None:
        if self._model_record_source is None:
            return None
        return len(self._role_call_records(role))

    @staticmethod
    def _rejected_candidate_audit(
        candidate: CandidateFinding,
        error: BaseException,
    ) -> RejectedCandidateAudit:
        finding_id = candidate.finding_id
        if re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", finding_id) is None:
            finding_id = "candidate-" + hashlib.sha256(
                finding_id.encode("utf-8")
            ).hexdigest()[:12]
        reason_code, _ = _safe_role_validation_reason(str(error))
        return RejectedCandidateAudit(
            finding_id=finding_id,
            source_role=ModelRole.REPORT_ANALYST.value,
            reason_code=reason_code,
            output_digest=hashlib.sha256(
                _canonical_json(candidate.model_dump(mode="python")).encode("utf-8")
            ).hexdigest(),
        )

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
        role_receipts: Sequence[SynthesisRoleReceipt] = (),
        rejected_candidate_audits: Sequence[RejectedCandidateAudit] = (),
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
            role_receipts=tuple(role_receipts),
            rejected_candidate_audits=tuple(rejected_candidate_audits),
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
        reason = str(error).strip().casefold()
        known_reasons = (
            (
                "evidence reference does not match corpus entry",
                "evidence reference does not match corpus entry",
            ),
            (
                "evidence reference",
                "evidence reference validation failed",
            ),
            (
                "evidence id namespace",
                "evidence ID namespace validation failed",
            ),
            (
                "finding evidence class is incompatible",
                "finding evidence class is incompatible with referenced evidence",
            ),
            (
                "finding conflicts with verifier outcome",
                "finding conflicts with verifier outcome",
            ),
            (
                "finding contains forbidden narrative input",
                "finding contains forbidden narrative input",
            ),
            (
                "finding requires severity justification",
                "finding requires severity justification",
            ),
            (
                "affected surfaces are not named",
                "affected surfaces are not named by supporting evidence",
            ),
            (
                "ux principles cannot justify severity",
                "UX principles cannot justify severity",
            ),
            (
                "unsupported human claim",
                "unsupported human claims cannot support a finding",
            ),
            (
                "finding conflicts with verifier outcome",
                "finding conflicts with verifier outcome",
            ),
            (
                "finding has no primary observed evidence",
                "finding has no primary observed evidence",
            ),
            (
                "causal language is not supported",
                "causal language is not supported by evidence",
            ),
            (
                "finding has no primary observed evidence",
                "finding has no primary observed evidence",
            ),
            ("unknown evidence id", "unknown evidence ID"),
            ("duplicate evidence id", "duplicate evidence ID"),
        )
        for prefix, safe_message in known_reasons:
            if reason.startswith(prefix):
                return safe_message
        name = type(error).__name__
        if name in {"ValueError", "TypeError", "ValidationError"}:
            return "evidence, schema, or publication contract failed"
        return "deterministic validation failed"


__all__ = [
    "DEFAULT_MAX_ATTACHMENT_BYTES",
    "DEFAULT_MAX_RETRIEVAL_ENTRIES",
    "MAX_ROLE_RETRIEVAL_ENTRIES",
    "MAX_RETRIEVAL_ROUNDS",
    "REPORT_SYNTHESIS_APPLICATION_SCHEMA_VERSION",
    "REPORT_SYNTHESIS_PROMPT_VERSION",
    "ReportSynthesisService",
]
