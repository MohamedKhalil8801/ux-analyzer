from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from ux_analyzer.adapters.openai import ModelFailureError
from ux_analyzer.application.evidence_corpus import (
    EvidenceCorpus,
    EvidenceEntry,
    EvidenceResolver,
)
from ux_analyzer.application.report_synthesis import (
    DEFAULT_MAX_ATTACHMENT_BYTES,
    REPORT_SYNTHESIS_PROMPT_VERSION,
    ReportSynthesisService,
    _safe_role_validation_reason,  # pyright: ignore[reportPrivateUsage]
)
from ux_analyzer.domain.findings import EvidenceClass, FindingSeverity
from ux_analyzer.domain.synthesis import (
    EvidenceRef,
    FindingKind,
    ObjectionSeverity,
    SynthesisAttempt,
    SynthesisFinding,
    SynthesisRoleReceipt,
    SynthesisStatus,
)
from ux_analyzer.ports.model_transport import (
    MODEL_ATTACHMENT_MAX_BYTES,
    MODEL_REQUEST_MAX_BYTES,
    TransportEvidenceUnavailableError,
)
from ux_analyzer.ports.models import (
    ModelCallRecord,
    ModelResponseValidationError,
    ModelRole,
    TokenUsage,
)
from ux_analyzer.providers.report_synthesis import (
    AdjudicationResponse,
    AnalystResponse,
    CandidateFinding,
    EvidenceAuditResponse,
    EvidenceReference,
    ObjectionResolution,
    PatternReviewResponse,
    ReportTransportBudgetError,
    TypedObjection,
)

EVIDENCE_ID = "event:run-a:1"
SECOND_EVIDENCE_ID = "event:run-a:2"
HEATMAP_ID = "heatmap:run-a:viewport-1:1s"


def test_attachment_resolution_and_request_ceilings_are_distinct() -> None:
    assert DEFAULT_MAX_ATTACHMENT_BYTES == MODEL_ATTACHMENT_MAX_BYTES
    assert DEFAULT_MAX_ATTACHMENT_BYTES > MODEL_REQUEST_MAX_BYTES


@pytest.mark.parametrize(
    ("reason", "expected"),
    (
        ("response exceeds bounded output limits", "bounded-output"),
        (
            "response limitation contains forbidden narrative",
            "forbidden-narrative",
        ),
        ("undelivered evidence ID", "undelivered-evidence-id"),
        (
            "finding references an unknown UX principle",
            "unknown-principle",
        ),
    ),
)
def test_role_validation_reason_codes_are_safe_and_specific(
    reason: str,
    expected: str,
) -> None:
    assert _safe_role_validation_reason(reason) == (expected, reason)


def _corpus(
    tmp_path: Path,
    *,
    extra_entries: Sequence[EvidenceEntry] = (),
) -> EvidenceCorpus:
    return EvidenceCorpus(
        output_root=tmp_path,
        entries=(
            EvidenceEntry(
                ref=EvidenceRef(EVIDENCE_ID, "event", "run-a", replay_sequence=1),
                evidence_class=EvidenceClass.DETERMINISTIC_FACT,
                summary="The user searched outside the expected task area.",
                payload={
                    "sequence": 1,
                    "succeeded": True,
                    "surface_ids": ("settings", "billing", "profile"),
                },
            ),
            *extra_entries,
        ),
    )


def _candidate(
    *,
    finding_id: str = "invite-control",
    evidence_id: str = EVIDENCE_ID,
    severity: str = "high",
    title: str = "The invite control is hard to find",
    issue: str = "The user searches outside the expected task area before finding the invite control.",
    impact: str = "An important collaboration task takes longer to complete.",
    root_cause: str = "The entry point is labeled around internal product structure.",
    severity_justification: str = "The evidence shows extra navigation on an important task.",
    affected_surfaces: Sequence[str] = (),
    evidence_ref: dict[str, object] | None = None,
) -> CandidateFinding:
    reference = evidence_ref or {
        "evidence_id": evidence_id,
        "kind": "event",
        "run_id": "run-a",
        "replay_sequence": 2 if evidence_id.endswith(":2") else 1,
    }
    return CandidateFinding.model_validate(
        {
            "finding_id": finding_id,
            "title": title,
            "issue": issue,
            "impact": impact,
            "root_cause": root_cause,
            "fixes": ["Label the entry point around the user's goal."],
            "severity": severity,
            "confidence": 0.9,
            "evidence_refs": [reference],
            "affected_surfaces": list(affected_surfaces),
            "severity_justification": severity_justification,
        }
    )


def _expectation_entry(
    *, alternatives: Sequence[str] = ("Use the team page.",)
) -> EvidenceEntry:
    return EvidenceEntry(
        ref=EvidenceRef("expectation:run-a", "expectation", "run-a"),
        evidence_class=EvidenceClass.DETERMINISTIC_FACT,
        summary="A frozen expectation was selected for the run.",
        payload={
            "matched": True,
            "acceptable_alternatives": tuple(alternatives),
        },
    )


def _scenario_entry() -> EvidenceEntry:
    return EvidenceEntry(
        ref=EvidenceRef("scenario:run-a", "scenario", "run-a"),
        evidence_class=EvidenceClass.DETERMINISTIC_FACT,
        summary="The scenario scope was recorded.",
        payload={"id": "invite"},
    )


def _verification_entry(*, verified: bool) -> EvidenceEntry:
    return EvidenceEntry(
        ref=EvidenceRef("verification:run-a", "verification", "run-a"),
        evidence_class=EvidenceClass.DETERMINISTIC_FACT,
        summary="The independent verifier recorded a result.",
        payload={"verified": verified},
    )


def _second_event_entry() -> EvidenceEntry:
    return EvidenceEntry(
        ref=EvidenceRef(SECOND_EVIDENCE_ID, "event", "run-a", replay_sequence=2),
        evidence_class=EvidenceClass.DETERMINISTIC_FACT,
        summary="The user reached the second recorded state.",
        payload={"sequence": 2, "succeeded": True},
    )


def _surface_event_entry() -> EvidenceEntry:
    return EvidenceEntry(
        ref=EvidenceRef(SECOND_EVIDENCE_ID, "event", "run-a", replay_sequence=2),
        evidence_class=EvidenceClass.DETERMINISTIC_FACT,
        summary="The user opened the settings surface.",
        payload={"sequence": 2, "succeeded": True, "surface_id": "settings"},
    )


def _heatmap_entry() -> EvidenceEntry:
    return EvidenceEntry(
        ref=EvidenceRef(
            HEATMAP_ID,
            "heatmap",
            "run-a",
            viewport_id="viewport-1",
            sha256="a" * 64,
        ),
        evidence_class=EvidenceClass.MODEL_ESTIMATE,
        summary="A validated heatmap reference.",
        payload={"media_type": "image/png"},
    )


def _large_mixed_reference_entries() -> tuple[EvidenceEntry, ...]:
    return tuple(
        EvidenceEntry(
            ref=EvidenceRef(
                f"event:run-a:{index}",
                "event",
                "run-a",
                replay_sequence=index,
            ),
            evidence_class=EvidenceClass.DETERMINISTIC_FACT,
            summary=f"Recorded event {index}.",
            payload={"sequence": index},
        )
        for index in range(2, 34)
    ) + (
        EvidenceEntry(
            ref=EvidenceRef(
                "metric:run-a:outcome",
                "metric",
                "run-a",
                metric_id="outcome",
            ),
            evidence_class=EvidenceClass.DETERMINISTIC_FACT,
            summary="The task outcome was recorded.",
            payload={"value": "verified-success"},
        ),
    )


def _large_mixed_evidence_references() -> list[EvidenceReference]:
    references = [
        EvidenceReference(
            evidence_id=f"event:run-a:{index}",
            kind="event",
            run_id="run-a",
            replay_sequence=index,
        )
        for index in range(1, 34)
    ]
    references.append(
        EvidenceReference(
            evidence_id="metric:run-a:outcome",
            kind="metric",
            run_id="run-a",
            metric_id="outcome",
        )
    )
    return references


class _ScriptedRole:
    def __init__(
        self,
        role: str,
        responses: Sequence[object],
        record_source: _RecordingModelSource | None = None,
    ) -> None:
        self.role = role
        self.provider_id = "fixture-provider"
        self.model = "fixture-model"
        self.endpoint_origin = "https://fixture.invalid"
        self.prompt_version = f"fixture-{role}-v1"
        self.provider_version = "fixture-v1"
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.record_source = record_source

    def _next(self, **call: Any) -> object:
        self.calls.append(call)
        response = self.responses.pop(0) if self.responses else None
        if isinstance(response, BaseException):
            raise response
        if response is not None:
            result = response
        elif self.role == "analyst":
            result = AnalystResponse(complete=True)
        elif self.role == "auditor":
            result = EvidenceAuditResponse(complete=True)
        elif self.role == "pattern":
            result = PatternReviewResponse(complete=True)
        else:
            result = AdjudicationResponse(complete=True)
        if self.record_source is not None:
            self.record_source.record(self.role, call, result)
        return result

    async def analyze(self, *args: Any, **kwargs: Any) -> object:
        return self._next(args=args, kwargs=kwargs)

    async def audit(self, *args: Any, **kwargs: Any) -> object:
        return self._next(args=args, kwargs=kwargs)

    async def review(self, *args: Any, **kwargs: Any) -> object:
        return self._next(args=args, kwargs=kwargs)

    async def adjudicate(self, *args: Any, **kwargs: Any) -> object:
        return self._next(args=args, kwargs=kwargs)


class _RecordingResolver(EvidenceResolver):
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.limits: list[dict[str, int]] = []

    def resolve(
        self, corpus: EvidenceCorpus, evidence_ids: Sequence[str], **kwargs: Any
    ):
        self.calls.append(tuple(evidence_ids))
        self.limits.append(
            {
                "max_entries": kwargs["max_entries"],
                "max_attachment_bytes": kwargs["max_attachment_bytes"],
            }
        )
        return super().resolve(corpus, evidence_ids, **kwargs)


class _ModelRecordSource:
    def __init__(self, records: Sequence[ModelCallRecord]) -> None:
        self.records = tuple(records)


class _RecordingModelSource:
    def __init__(self) -> None:
        self.records: list[ModelCallRecord] = []

    def record(self, role: str, call: Mapping[str, Any], response: object) -> None:
        roles = {
            "analyst": ModelRole.REPORT_ANALYST,
            "auditor": ModelRole.REPORT_EVIDENCE_AUDITOR,
            "pattern": ModelRole.REPORT_PATTERN_REVIEWER,
            "adjudicator": ModelRole.REPORT_ADJUDICATOR,
        }
        schemas = {
            "analyst": AnalystResponse,
            "auditor": EvidenceAuditResponse,
            "pattern": PatternReviewResponse,
            "adjudicator": AdjudicationResponse,
        }
        self.records.append(
            ModelCallRecord(
                role=roles[role],
                model="fixture-model",
                endpoint_origin="https://fixture.invalid",
                prompt_digest=hashlib.sha256(
                    repr(dict(call)).encode("utf-8")
                ).hexdigest(),
                schema_version=schemas[role].schema_version,
                attempts=1,
                latency_ms=0,
                token_usage=TokenUsage(0, 0, 0),
                request={},
                response={"type": type(response).__name__},
            )
        )


def _scripted_service(
    *,
    analyst: Sequence[object] = (),
    auditor: Sequence[object] = (),
    pattern: Sequence[object] = (),
    adjudicator: Sequence[object] = (),
    resolver: EvidenceResolver | None = None,
    max_adjudication_revisions: int = 1,
    max_final_verifications: int = 1,
    model_record_source: object | None = None,
    resume_analyst_receipt: SynthesisRoleReceipt | None = None,
    resume_candidate_findings: Sequence[SynthesisFinding] = (),
    resume_attempt_id: str = "",
) -> tuple[ReportSynthesisService, tuple[_ScriptedRole, ...]]:
    recording_source = None
    effective_source = model_record_source
    if effective_source is None:
        recording_source = _RecordingModelSource()
        effective_source = recording_source
    roles = (
        _ScriptedRole("analyst", analyst, recording_source),
        _ScriptedRole("auditor", auditor, recording_source),
        _ScriptedRole("pattern", pattern, recording_source),
        _ScriptedRole("adjudicator", adjudicator, recording_source),
    )
    return (
        ReportSynthesisService(
            analyst=roles[0],
            evidence_auditor=roles[1],
            pattern_reviewer=roles[2],
            adjudicator=roles[3],
            resolver=resolver,
            max_adjudication_revisions=max_adjudication_revisions,
            max_final_verifications=max_final_verifications,
            model_record_source=effective_source,
            resume_analyst_receipt=resume_analyst_receipt,
            resume_candidate_findings=resume_candidate_findings,
            resume_attempt_id=resume_attempt_id,
        ),
        roles,
    )


def _analyst_role_manifest(*, prompt_version: str) -> dict[str, object]:
    return {
        "provider_id": "fixture-provider",
        "role": "report-analyst",
        "model_id": "fixture-model",
        "endpoint_origin": "https://fixture.invalid",
        "prompt_version": prompt_version,
        "schema_version": AnalystResponse.schema_version,
        "provider_version": "fixture-v1",
    }


class _SlowGateRole(_ScriptedRole):
    """Role whose calls block until an asyncio event is set, for overlap tests."""

    def __init__(self, role: str, response: object) -> None:
        super().__init__(role, [response])
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def _gate(self) -> None:
        self.entered.set()
        await self.release.wait()

    async def audit(self, *args: Any, **kwargs: Any) -> object:
        await self._gate()
        return self._next(args=args, kwargs=kwargs)

    async def review(self, *args: Any, **kwargs: Any) -> object:
        await self._gate()
        return self._next(args=args, kwargs=kwargs)


@pytest.mark.asyncio
async def test_reviewers_run_concurrently(tmp_path: Path) -> None:
    """Auditor and pattern reviewer overlap; neither waits for the other."""

    candidate = _candidate()
    auditor = _SlowGateRole(
        "auditor", EvidenceAuditResponse(complete=True)
    )
    pattern = _SlowGateRole(
        "pattern", PatternReviewResponse(complete=True)
    )
    roles = (
        _ScriptedRole(
            "analyst",
            [AnalystResponse(complete=True, candidate_findings=[candidate])],
        ),
        auditor,
        pattern,
        _ScriptedRole(
            "adjudicator",
            [AdjudicationResponse(complete=True, final_findings=[candidate])],
        ),
    )
    recording_source = _RecordingModelSource()
    for role in roles:
        role.record_source = recording_source
    service = ReportSynthesisService(
        analyst=roles[0],
        evidence_auditor=roles[1],
        pattern_reviewer=roles[2],
        adjudicator=roles[3],
        model_record_source=recording_source,
    )

    async def run() -> object:
        return await service.synthesize(_corpus(tmp_path))

    task = asyncio.create_task(run())
    # Both reviewers must be entered while neither has released the gate.
    await asyncio.wait_for(auditor.entered.wait(), timeout=5)
    await asyncio.wait_for(pattern.entered.wait(), timeout=5)
    assert auditor.calls == []
    assert pattern.calls == []
    auditor.release.set()
    pattern.release.set()
    attempt = cast(SynthesisAttempt, await asyncio.wait_for(task, timeout=10))

    assert attempt.status is SynthesisStatus.ACCEPTED


@pytest.mark.asyncio
async def test_analyst_receipt_for_resume_accepts_validated_prior_attempt(
    tmp_path: Path,
) -> None:
    """A same-corpus, schema-matched prior attempt yields its analyst receipt."""

    service, _roles = _scripted_service()
    corpus = _corpus(tmp_path)
    candidate = _candidate()
    finding = candidate.to_domain(reviewer_state="candidate")
    receipt = SynthesisRoleReceipt(
        role="report-analyst",
        provider_id="fixture-provider",
        model_id="fixture-model",
        prompt_digest="a" * 64,
        schema_digest=hashlib.sha256(
            json.dumps(
                AnalystResponse.model_json_schema(),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        output_digest="b" * 64,
    )
    prior = SynthesisAttempt(
        attempt_id="synthesis-prior",
        status=SynthesisStatus.UNAVAILABLE,
        corpus_digest=corpus.digest,
        role_manifest={
            "report-analyst": _analyst_role_manifest(
                prompt_version="fixture-analyst-v1"
            )
        },
        prompt_version=REPORT_SYNTHESIS_PROMPT_VERSION,
        role_receipts=(receipt,),
        candidate_findings=(finding,),
    )

    resumed = service.analyst_receipt_for_resume(prior, corpus)

    assert resumed is receipt


@pytest.mark.asyncio
async def test_analyst_receipt_for_resume_rejects_prompt_version_mismatch(
    tmp_path: Path,
) -> None:
    service, _roles = _scripted_service()
    corpus = _corpus(tmp_path)
    finding = _candidate().to_domain(reviewer_state="candidate")
    receipt = SynthesisRoleReceipt(
        role="report-analyst",
        provider_id="fixture-provider",
        model_id="fixture-model",
        prompt_digest="a" * 64,
        schema_digest=hashlib.sha256(
            json.dumps(
                AnalystResponse.model_json_schema(),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        output_digest="b" * 64,
    )
    prior = SynthesisAttempt(
        attempt_id="synthesis-prior",
        status=SynthesisStatus.UNAVAILABLE,
        corpus_digest=corpus.digest,
        role_manifest={
            "report-analyst": _analyst_role_manifest(
                prompt_version="fixture-analyst-v0"
            )
        },
        prompt_version=REPORT_SYNTHESIS_PROMPT_VERSION,
        role_receipts=(receipt,),
        candidate_findings=(finding,),
    )

    assert service.analyst_receipt_for_resume(prior, corpus) is None


@pytest.mark.asyncio
async def test_analyst_receipt_for_resume_rejects_mismatched_corpus(
    tmp_path: Path,
) -> None:
    """A prior attempt from a different corpus digest must not be reused."""

    service, _roles = _scripted_service()
    corpus = _corpus(tmp_path)
    candidate = _candidate()
    finding = candidate.to_domain(reviewer_state="candidate")
    receipt = SynthesisRoleReceipt(
        role="report-analyst",
        provider_id="fixture-provider",
        model_id="fixture-model",
        prompt_digest="a" * 64,
        schema_digest=hashlib.sha256(
            json.dumps(
                AnalystResponse.model_json_schema(),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        output_digest="b" * 64,
    )
    prior = SynthesisAttempt(
        attempt_id="synthesis-prior",
        status=SynthesisStatus.UNAVAILABLE,
        corpus_digest="f" * 64,
        prompt_version=REPORT_SYNTHESIS_PROMPT_VERSION,
        role_receipts=(receipt,),
        candidate_findings=(finding,),
    )

    assert service.analyst_receipt_for_resume(prior, corpus) is None


@pytest.mark.asyncio
async def test_analyst_receipt_for_resume_rejects_invalidated_finding(
    tmp_path: Path,
) -> None:
    """A candidate that fails deterministic validation against the corpus
    (here: a bad evidence reference) forces a fresh analyst run."""

    service, _roles = _scripted_service()
    corpus = _corpus(tmp_path)
    raw_candidate = _candidate().model_dump(mode="python")
    raw_candidate["evidence_refs"] = [
        {"evidence_id": "event:run-z:999", "kind": "event", "run_id": "run-z"}
    ]
    finding = CandidateFinding.model_validate(raw_candidate).to_domain(
        reviewer_state="candidate"
    )
    receipt = SynthesisRoleReceipt(
        role="report-analyst",
        provider_id="fixture-provider",
        model_id="fixture-model",
        prompt_digest="a" * 64,
        schema_digest=hashlib.sha256(
            json.dumps(
                AnalystResponse.model_json_schema(),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        output_digest="b" * 64,
    )
    prior = SynthesisAttempt(
        attempt_id="synthesis-prior",
        status=SynthesisStatus.UNAVAILABLE,
        corpus_digest=corpus.digest,
        role_manifest={
            "report-analyst": _analyst_role_manifest(
                prompt_version="fixture-analyst-v1"
            )
        },
        prompt_version=REPORT_SYNTHESIS_PROMPT_VERSION,
        role_receipts=(receipt,),
        candidate_findings=(finding,),
    )

    assert service.analyst_receipt_for_resume(prior, corpus) is None


@pytest.mark.asyncio
async def test_resume_skips_analyst_call_and_publishes(tmp_path: Path) -> None:
    """With a reusable receipt, no analyst model call happens and the analyst
    receipt flows through to publication."""

    candidate = _candidate()
    resumed_receipt = SynthesisRoleReceipt(
        role="report-analyst",
        provider_id="fixture-provider",
        model_id="fixture-model",
        prompt_digest="a" * 64,
        schema_digest=hashlib.sha256(
            json.dumps(
                AnalystResponse.model_json_schema(),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        output_digest="b" * 64,
    )
    service, roles = _scripted_service(
        adjudicator=[
            AdjudicationResponse(complete=True, final_findings=[candidate])
        ],
        resume_analyst_receipt=resumed_receipt,
        resume_candidate_findings=(
            CandidateFinding.model_validate(
                candidate.model_dump(mode="python")
            ).to_domain(reviewer_state="candidate"),
        ),
        resume_attempt_id="synthesis-prior",
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.ACCEPTED
    assert roles[0].calls == []  # analyst never invoked
    assert any(
        entry.get("phase") == "resume" and entry.get("role") == "report-analyst"
        for entry in attempt.retrieval_log
    )
    assert any(
        receipt.role == "report-analyst" for receipt in attempt.role_receipts
    )


def _blocking_objection(finding_id: str) -> TypedObjection:
    return TypedObjection(
        objection_id="objection-1",
        finding_id=finding_id,
        objection_type="factual-support",
        severity=ObjectionSeverity.BLOCKING,
        message="The cited evidence does not establish the stated claim.",
        evidence_refs=[
            EvidenceReference(
                evidence_id=EVIDENCE_ID,
                kind="event",
                run_id="run-a",
                replay_sequence=1,
            )
        ],
        reviewer_role="report-evidence-auditor",
    )


@pytest.mark.asyncio
async def test_synthesis_publishes_only_after_review_consensus(tmp_path: Path) -> None:
    candidate = _candidate()
    service, roles = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        adjudicator=[
            AdjudicationResponse(complete=True, final_findings=[candidate])
        ],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.ACCEPTED
    assert attempt.findings[0].evidence_refs
    assert not [
        objection
        for objection in attempt.objections
        if objection.severity is ObjectionSeverity.BLOCKING and not objection.resolved
    ]
    assert all(role.calls for role in roles)
    assert {str(entry["role"]) for entry in attempt.retrieval_log} == {
        "report-analyst",
        "report-evidence-auditor",
        "report-pattern-reviewer",
        "report-adjudicator",
    }
    assert {receipt.role for receipt in attempt.role_receipts} == {
        "report-analyst",
        "report-evidence-auditor",
        "report-pattern-reviewer",
        "report-adjudicator",
    }
    assert all(
        len(digest) == 64
        for receipt in attempt.role_receipts
        for digest in (
            receipt.prompt_digest,
            receipt.schema_digest,
            receipt.output_digest,
        )
    )
    schemas = {
        "report-analyst": AnalystResponse,
        "report-evidence-auditor": EvidenceAuditResponse,
        "report-pattern-reviewer": PatternReviewResponse,
        "report-adjudicator": AdjudicationResponse,
    }
    outputs = {
        "report-analyst": AnalystResponse(
            complete=True,
            candidate_findings=[candidate],
        ),
        "report-evidence-auditor": EvidenceAuditResponse(complete=True),
        "report-pattern-reviewer": PatternReviewResponse(complete=True),
        "report-adjudicator": AdjudicationResponse(
            complete=True,
            final_findings=[candidate],
        ),
    }
    recorded = {
        record.role.value: record
        for record in service._model_record_source.records  # pyright: ignore[reportOptionalMemberAccess, reportPrivateUsage]
    }
    for receipt in attempt.role_receipts:
        expected_schema_digest = hashlib.sha256(
            json.dumps(
                schemas[receipt.role].model_json_schema(),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        assert receipt.schema_digest == expected_schema_digest
        assert receipt.prompt_digest == recorded[receipt.role].prompt_digest
        expected_output_digest = hashlib.sha256(
            json.dumps(
                outputs[receipt.role].model_dump(mode="python"),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        assert receipt.output_digest == expected_output_digest


@pytest.mark.asyncio
async def test_manifest_only_role_completions_cannot_publish_no_issues(
    tmp_path: Path,
) -> None:
    roles = (
        _ScriptedRole("analyst", ()),
        _ScriptedRole("auditor", ()),
        _ScriptedRole("pattern", ()),
        _ScriptedRole("adjudicator", ()),
    )
    service = ReportSynthesisService(
        analyst=roles[0],
        evidence_auditor=roles[1],
        pattern_reviewer=roles[2],
        adjudicator=roles[3],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.role_receipts
    assert any("completion receipts" in item for item in attempt.limitations)


@pytest.mark.asyncio
async def test_retrieval_stops_after_complete_and_uses_resolver(tmp_path: Path) -> None:
    resolver = _RecordingResolver()
    service, roles = _scripted_service(
        analyst=[
            AnalystResponse(complete=False, evidence_requests=[EVIDENCE_ID]),
            AnalystResponse(complete=True),
        ],
        resolver=resolver,
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.NO_ISSUES
    assert len(roles[0].calls) == 2
    assert resolver.calls == [(EVIDENCE_ID,)]
    analyst_logs = [
        entry for entry in attempt.retrieval_log if entry["role"] == "report-analyst"
    ]
    assert len(analyst_logs) == 2
    assert analyst_logs[0]["request"] == (EVIDENCE_ID,)
    assert analyst_logs[1]["request"] == ()


@pytest.mark.asyncio
async def test_role_retrieval_preserves_prior_resolved_evidence_across_rounds(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    service, roles = _scripted_service(
        analyst=[
            AnalystResponse(complete=False, evidence_requests=[EVIDENCE_ID]),
            AnalystResponse(complete=False, evidence_requests=[SECOND_EVIDENCE_ID]),
            AnalystResponse(complete=True, candidate_findings=[candidate]),
        ],
        adjudicator=[AdjudicationResponse(complete=True, final_findings=[candidate])],
    )

    attempt = await service.synthesize(
        _corpus(tmp_path, extra_entries=(_second_event_entry(),))
    )

    assert attempt.status is SynthesisStatus.ACCEPTED
    resolved = roles[0].calls[2]["kwargs"]["resolved_evidence"]
    assert resolved.evidence_ids == (EVIDENCE_ID, SECOND_EVIDENCE_ID)


def test_retrieval_rejects_requests_above_total_role_limit(
    tmp_path: Path,
) -> None:
    evidence_ids = [f"event:run-a:{index}" for index in range(1, 61)]
    extra_entries = tuple(
        EvidenceEntry(
            ref=EvidenceRef(evidence_id, "event", "run-a", replay_sequence=index),
            evidence_class=EvidenceClass.DETERMINISTIC_FACT,
            summary=f"Recorded event {index}.",
            payload={"sequence": index},
        )
        for index, evidence_id in enumerate(evidence_ids[1:], start=2)
    )
    resolver = _RecordingResolver()
    service, _ = _scripted_service(
        resolver=resolver,
    )
    corpus = _corpus(tmp_path, extra_entries=extra_entries)

    with pytest.raises(ValueError, match="role retrieval limit"):
        service._resolve_evidence_batches(  # pyright: ignore[reportPrivateUsage]
            corpus,
            evidence_ids,
            role=ModelRole.REPORT_ANALYST,
            phase="retrieval",
            round_number=1,
        )
    assert resolver.calls == []


@pytest.mark.asyncio
async def test_retrieval_rejects_cumulative_role_requests_above_total_limit(
    tmp_path: Path,
) -> None:
    evidence_ids = [f"event:run-a:{index}" for index in range(1, 34)]
    entries = tuple(
        EvidenceEntry(
            ref=EvidenceRef(evidence_id, "event", "run-a", replay_sequence=index),
            evidence_class=EvidenceClass.DETERMINISTIC_FACT,
            summary=f"Recorded event {index}.",
            payload={"sequence": index},
        )
        for index, evidence_id in enumerate(evidence_ids[1:], start=2)
    )
    resolver = _RecordingResolver()
    service, _ = _scripted_service(
        analyst=[
            AnalystResponse(complete=False, evidence_requests=evidence_ids[:16]),
            AnalystResponse(complete=False, evidence_requests=evidence_ids[16:32]),
            AnalystResponse(complete=False, evidence_requests=evidence_ids[32:]),
        ],
        resolver=resolver,
    )

    attempt = await service.synthesize(_corpus(tmp_path, extra_entries=entries))

    assert attempt.status is SynthesisStatus.REJECTED
    assert resolver.calls == [tuple(evidence_ids[:16]), tuple(evidence_ids[16:32])]
    assert any("retrieval request" in item for item in attempt.limitations)


@pytest.mark.asyncio
async def test_review_rejects_large_mixed_reference_collections(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    references = _large_mixed_evidence_references()
    objection = TypedObjection.model_construct(
        objection_id="large-objection",
        finding_id=candidate.finding_id,
        objection_type="factual-support",
        severity=ObjectionSeverity.MATERIAL,
        message="The broader evidence set should be checked.",
        evidence_refs=references,
        reviewer_role="report-evidence-auditor",
    )
    resolution = ObjectionResolution.model_construct(
        objection_id=objection.objection_id,
        finding_id=candidate.finding_id,
        resolved=True,
        resolution="The broader evidence set supports the reviewed claim.",
        evidence_refs=references,
    )
    resolver = _RecordingResolver()
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        auditor=[EvidenceAuditResponse(complete=True, objections=[objection])],
        pattern=[PatternReviewResponse(complete=True)],
        adjudicator=[
            AdjudicationResponse(
                complete=True,
                final_findings=[candidate],
                objection_resolutions=[resolution],
            )
        ],
        resolver=resolver,
    )

    attempt = await service.synthesize(
        _corpus(tmp_path, extra_entries=_large_mixed_reference_entries())
    )

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings
    assert all(len(call) <= 16 for call in resolver.calls)


@pytest.mark.asyncio
async def test_repeated_evidence_request_is_rejected_without_third_call(
    tmp_path: Path,
) -> None:
    resolver = _RecordingResolver()
    incomplete = AnalystResponse(complete=False, evidence_requests=[EVIDENCE_ID])
    service, roles = _scripted_service(
        analyst=[incomplete, incomplete, incomplete],
        resolver=resolver,
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.REJECTED
    assert len(roles[0].calls) == 2
    assert len(resolver.calls) == 1
    assert any("retrieval request" in limitation for limitation in attempt.limitations)
    assert all(entry["role"] == "report-analyst" for entry in attempt.retrieval_log)


@pytest.mark.asyncio
async def test_completed_role_unavailable_evidence_limitation_is_retained(
    tmp_path: Path,
) -> None:
    limitations = (
        "Analyst visual evidence was unavailable within the transport budget.",
        "Auditor visual evidence was unavailable within the transport budget.",
        "Pattern visual evidence was unavailable within the transport budget.",
        "Adjudicator visual evidence was unavailable within the transport budget.",
    )
    service, _ = _scripted_service(
        analyst=[
            AnalystResponse(
                complete=True,
                unavailable_evidence_ids=[EVIDENCE_ID],
                limitations=[limitations[0]],
            )
        ],
        auditor=[EvidenceAuditResponse(complete=True, limitations=[limitations[1]])],
        pattern=[PatternReviewResponse(complete=True, limitations=[limitations[2]])],
        adjudicator=[AdjudicationResponse(complete=True, limitations=[limitations[3]])],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.NO_ISSUES
    assert set(limitations).issubset(attempt.limitations)


@pytest.mark.asyncio
async def test_re_requested_transport_unavailable_evidence_is_operational(
    tmp_path: Path,
) -> None:
    service, _ = _scripted_service(
        analyst=[TransportEvidenceUnavailableError(1)],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.UNAVAILABLE
    assert any(
        "visual evidence was unavailable within the bounded transport request"
        in limitation.casefold()
        for limitation in attempt.limitations
    )
    analyst_log = next(
        entry for entry in attempt.retrieval_log if entry["role"] == "report-analyst"
    )
    assert analyst_log["error"] == "visual evidence unavailable"
    assert "event:run-a:1" not in repr(analyst_log)


@pytest.mark.asyncio
async def test_report_transport_budget_is_unavailable_with_safe_diagnostics(
    tmp_path: Path,
) -> None:
    budget_error = ReportTransportBudgetError(
        {
            "stage": "request_budget",
            "request_bytes": 750_001,
            "budget_bytes": 750_000,
            "attachment_bytes": 588_028,
            "attachment_bytes_deferred": 0,
            "attachment_count": 1,
            "response_content_length": -1,
            "top_level_keys": ["privateProviderField"],
        }
    )
    service, _ = _scripted_service(analyst=[budget_error])

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.UNAVAILABLE
    analyst_log = next(
        entry for entry in attempt.retrieval_log if entry["role"] == "report-analyst"
    )
    assert analyst_log["error"] == "report request exceeded transport budget"
    assert analyst_log["response"]["provider"]["diagnostics"] == {
        "stage": "request_budget",
        "request_bytes": 750_001,
        "budget_bytes": 750_000,
        "attachment_bytes": 588_028,
        "attachment_bytes_deferred": 0,
        "attachment_count": 1,
    }
    assert "invalid structured output" not in repr(attempt.retrieval_log)


@pytest.mark.asyncio
async def test_role_validation_failure_keeps_safe_stage_diagnostics(
    tmp_path: Path,
) -> None:
    service, _ = _scripted_service(
        analyst=[
            ModelResponseValidationError(
                ModelRole.REPORT_ANALYST,
                "unknown evidence ID: private-provider-content-must-not-be-recorded",
                response_summary={
                    "schema": "AnalystResponse",
                    "top_level_keys": ["private-provider-content-must-not-be-recorded"],
                    "private": "private-provider-content-must-not-be-recorded",
                },
            ),
            ModelResponseValidationError(
                ModelRole.REPORT_ANALYST,
                "unknown evidence ID: private-provider-content-must-not-be-recorded",
                response_summary={"schema": "AnalystResponse"},
            ),
        ]
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings
    assert any(
        "invalid structured output" in limitation for limitation in attempt.limitations
    )
    assert "private-provider-content-must-not-be-recorded" not in repr(
        attempt.limitations
    )
    analyst_log = next(
        entry for entry in attempt.retrieval_log if entry["role"] == "report-analyst"
    )
    assert analyst_log["response"]["provider"]["diagnostics"] == {
        "role": "report-analyst",
        "stage": "role_validation",
        "error_type": "ModelResponseValidationError",
        "validation_reason_code": "unknown-evidence-id",
        "validation_reason": "unknown evidence ID",
        "response_summary": {
            "schema": "AnalystResponse",
        },
    }
    assert "private-provider-content-must-not-be-recorded" not in repr(
        attempt.retrieval_log
    )


@pytest.mark.asyncio
async def test_hallucinated_reference_is_rejected_before_publication(
    tmp_path: Path,
) -> None:
    candidate = _candidate(evidence_id="event:run-a:999")
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        adjudicator=[AdjudicationResponse(complete=True, final_findings=[candidate])],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings
    assert any(
        "publication validation" in limitation for limitation in attempt.limitations
    )
    assert any(
        "unknown evidence ID" in limitation for limitation in attempt.limitations
    )
    assert len(attempt.rejected_candidate_audits) == 1
    assert attempt.rejected_candidate_audits[0].source_role == "report-analyst"
    assert attempt.rejected_candidate_audits[0].reason_code == "unknown-evidence-id"
    assert "event:run-a:999" not in repr(attempt.rejected_candidate_audits)


@pytest.mark.asyncio
async def test_unsupported_human_claim_is_rejected_before_publication(
    tmp_path: Path,
) -> None:
    unsupported = EvidenceEntry(
        ref=EvidenceRef("limitation:run-a:claim", "limitation", "run-a"),
        evidence_class=EvidenceClass.UNSUPPORTED_HUMAN_CLAIM,
        summary="Unsupported human claim from an external narrative.",
        payload={"text": "The user disliked the interface."},
    )
    candidate = _candidate(evidence_id=unsupported.ref.evidence_id)
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])]
    )

    attempt = await service.synthesize(_corpus(tmp_path, extra_entries=(unsupported,)))

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings
    assert any(
        "publication validation" in limitation for limitation in attempt.limitations
    )


@pytest.mark.asyncio
async def test_duplicate_candidate_ids_are_rejected_before_publication(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    duplicate = _candidate(title="The same finding was emitted twice.")
    service, _ = _scripted_service(
        analyst=[
            AnalystResponse(
                complete=True,
                candidate_findings=[candidate, duplicate],
            )
        ],
        adjudicator=[AdjudicationResponse(complete=True, final_findings=[candidate])],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings
    assert attempt.rejected_findings
    assert all(
        finding.reviewer_state == "not-established"
        for finding in attempt.rejected_findings
    )
    assert any("duplicate finding IDs" in item for item in attempt.limitations)


@pytest.mark.asyncio
async def test_duplicate_final_finding_ids_reject_final_output(tmp_path: Path) -> None:
    candidate = _candidate()
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        adjudicator=[
            AdjudicationResponse(
                complete=True,
                final_findings=[candidate, candidate],
            )
        ],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings
    assert any("duplicated" in item for item in attempt.limitations)


@pytest.mark.asyncio
async def test_adjudicator_only_finding_cannot_be_published(tmp_path: Path) -> None:
    candidate = _candidate()
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True)],
        adjudicator=[AdjudicationResponse(complete=True, final_findings=[candidate])],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings
    assert any("analyst candidates" in item for item in attempt.limitations)


@pytest.mark.asyncio
async def test_forged_counterevidence_reference_is_rejected_before_publication(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    forged = CandidateFinding.model_validate(
        {
            **candidate.model_dump(mode="python"),
            "counterevidence": [
                {
                    "evidence_id": "event:run-a:999",
                    "kind": "event",
                    "run_id": "run-a",
                    "replay_sequence": 999,
                }
            ],
        }
    )
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[forged])],
        adjudicator=[AdjudicationResponse(complete=True, final_findings=[forged])],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings
    assert any("publication validation" in item for item in attempt.limitations)


@pytest.mark.asyncio
async def test_cross_reviewer_duplicate_objection_ids_are_namespaced(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    objection = TypedObjection(
        objection_id="shared-objection",
        finding_id=candidate.finding_id,
        objection_type="severity",
        severity=ObjectionSeverity.MATERIAL,
        message="The severity needs more support.",
    )
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        auditor=[EvidenceAuditResponse(complete=True, objections=[objection])],
        pattern=[PatternReviewResponse(complete=True, objections=[objection])],
        adjudicator=[
            AdjudicationResponse(
                complete=True,
                final_findings=[candidate],
                objection_resolutions=[
                    ObjectionResolution(
                        objection_id=(
                            "report-evidence-auditor:shared-objection"
                        ),
                        finding_id=candidate.finding_id,
                        resolved=False,
                        resolution="The objection remains as a qualification.",
                    ),
                    ObjectionResolution(
                        objection_id=(
                            "report-pattern-reviewer:shared-objection"
                        ),
                        finding_id=candidate.finding_id,
                        resolved=False,
                        resolution="The objection remains as a qualification.",
                    ),
                ],
            )
        ],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.ACCEPTED
    assert attempt.findings
    assert len(attempt.objections) == 2
    assert len({item.objection_id for item in attempt.objections}) == 2
    assert {
        item.reviewer_role for item in attempt.objections
    } == {"report-evidence-auditor", "report-pattern-reviewer"}


@pytest.mark.asyncio
async def test_identical_reviewer_objection_is_deduplicated(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    objection = TypedObjection(
        objection_id="repeated-objection",
        finding_id=candidate.finding_id,
        objection_type="severity",
        severity=ObjectionSeverity.MATERIAL,
        message="The severity needs more support.",
    )
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        auditor=[
            EvidenceAuditResponse(
                complete=True,
                objections=[objection, objection],
            )
        ],
        adjudicator=[
            AdjudicationResponse(
                complete=True,
                final_findings=[candidate],
                objection_resolutions=[
                    ObjectionResolution(
                        objection_id="report-evidence-auditor:repeated-objection",
                        finding_id=candidate.finding_id,
                        resolved=True,
                        resolution="The severity remains low and supported.",
                    )
                ],
            )
        ],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.ACCEPTED
    assert len(attempt.objections) == 1


@pytest.mark.asyncio
async def test_conflicting_reviewer_objection_ids_are_rejected(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    objection = TypedObjection(
        objection_id="conflicting-objection",
        finding_id=candidate.finding_id,
        objection_type="severity",
        severity=ObjectionSeverity.MATERIAL,
        message="The severity needs more support.",
    )
    conflicting = objection.model_copy(
        update={"message": "A different objection uses the same ID."}
    )
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        auditor=[
            EvidenceAuditResponse(
                complete=True,
                objections=[objection, conflicting],
            )
        ],
        adjudicator=[AdjudicationResponse(complete=True, final_findings=[candidate])],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.REJECTED


@pytest.mark.asyncio
async def test_orphan_blocking_objection_prevents_publication(tmp_path: Path) -> None:
    candidate = _candidate()
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        auditor=[
            EvidenceAuditResponse(
                complete=True,
                objections=[_blocking_objection("ghost-finding")],
            )
        ],
        adjudicator=[AdjudicationResponse(complete=True, final_findings=[candidate])],
        max_adjudication_revisions=0,
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings


@pytest.mark.asyncio
async def test_reviewer_validation_failure_is_not_no_issues(tmp_path: Path) -> None:
    objection = _blocking_objection("ghost-finding")
    service, _ = _scripted_service(
        auditor=[
            EvidenceAuditResponse(complete=True, objections=[objection, objection])
        ]
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.REJECTED
    assert attempt.status is not SynthesisStatus.NO_ISSUES


@pytest.mark.asyncio
async def test_underscore_narrative_sentinels_are_redacted_and_rejected(
    tmp_path: Path,
) -> None:
    candidate = _candidate(
        title="PRIOR_AGENT_PRIVATE_REASONING_SENTINEL",
        issue="PRIOR_FINDING_PROSE_SENTINEL",
    )
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        adjudicator=[AdjudicationResponse(complete=True, final_findings=[candidate])],
    )

    attempt = await service.synthesize(_corpus(tmp_path))
    serialized = repr(attempt.retrieval_log)

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings
    assert "PRIOR_AGENT_PRIVATE_REASONING_SENTINEL" not in serialized
    assert "PRIOR_FINDING_PROSE_SENTINEL" not in serialized


@pytest.mark.asyncio
async def test_underscore_objection_sentinel_is_redacted_and_rejected(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    objection = TypedObjection(
        objection_id="objection-1",
        finding_id=candidate.finding_id,
        objection_type="factual-support",
        severity=ObjectionSeverity.BLOCKING,
        message="PRIOR_AGENT_PRIVATE_REASONING_SENTINEL",
    )
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        auditor=[EvidenceAuditResponse(complete=True, objections=[objection])],
    )

    attempt = await service.synthesize(_corpus(tmp_path))
    serialized = repr(attempt.retrieval_log)

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings
    assert "PRIOR_AGENT_PRIVATE_REASONING_SENTINEL" not in serialized


@pytest.mark.asyncio
async def test_underscore_resolution_sentinel_is_redacted_and_cannot_resolve_blocker(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    objection = _blocking_objection(candidate.finding_id)
    resolution = ObjectionResolution(
        objection_id=objection.objection_id,
        finding_id=candidate.finding_id,
        resolved=True,
        resolution="PRIOR_FINDING_PROSE_SENTINEL",
        evidence_refs=objection.evidence_refs,
    )
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        auditor=[EvidenceAuditResponse(complete=True, objections=[objection])],
        adjudicator=[
            AdjudicationResponse(
                complete=True,
                final_findings=[candidate],
                objection_resolutions=[resolution],
            )
        ],
        max_adjudication_revisions=0,
    )

    attempt = await service.synthesize(_corpus(tmp_path))
    serialized = repr(attempt.retrieval_log)

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings
    assert not attempt.objections[0].resolved
    assert "PRIOR_FINDING_PROSE_SENTINEL" not in serialized


@pytest.mark.asyncio
async def test_forged_heatmap_digest_is_rejected(tmp_path: Path) -> None:
    candidate = _candidate(
        evidence_ref={
            "evidence_id": HEATMAP_ID,
            "kind": "heatmap",
            "run_id": "run-a",
            "viewport_id": "viewport-1",
            "sha256": "b" * 64,
        }
    )
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])]
    )

    attempt = await service.synthesize(
        _corpus(tmp_path, extra_entries=(_heatmap_entry(),))
    )

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings


@pytest.mark.asyncio
async def test_conflicting_verifier_outcome_is_not_published(tmp_path: Path) -> None:
    candidate = _candidate(
        issue="The task failed even though the user completed it.",
        impact="The failed task would block the user.",
    )
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])]
    )

    attempt = await service.synthesize(
        _corpus(tmp_path, extra_entries=(_verification_entry(verified=True),))
    )

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings
    assert any(
        "publication validation" in limitation for limitation in attempt.limitations
    )


@pytest.mark.asyncio
async def test_negated_failure_phrase_does_not_conflict_with_verifier(
    tmp_path: Path,
) -> None:
    candidate = _candidate(
        issue="No run failed, but the goal control is crowded.",
        impact="The task still takes longer to complete.",
    )
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        adjudicator=[
            AdjudicationResponse(complete=True, final_findings=[candidate])
        ],
    )

    attempt = await service.synthesize(
        _corpus(tmp_path, extra_entries=(_verification_entry(verified=True),))
    )

    assert attempt.status is SynthesisStatus.ACCEPTED
    assert attempt.findings


@pytest.mark.asyncio
async def test_heuristic_metric_reference_in_severity_is_published(
    tmp_path: Path,
) -> None:
    candidate = _candidate(
        severity_justification=(
            "The signal is a single heuristic ambiguity flag measured at 1.0."
        )
    )
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        adjudicator=[
            AdjudicationResponse(complete=True, final_findings=[candidate])
        ],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.ACCEPTED
    assert attempt.findings


@pytest.mark.asyncio
async def test_principle_authority_severity_is_not_published(tmp_path: Path) -> None:
    candidate = _candidate(
        severity_justification="The severity is justified because heuristic #3 says so."
    )
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])]
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings
    assert any(
        "publication validation" in limitation for limitation in attempt.limitations
    )


@pytest.mark.asyncio
async def test_harmless_alternate_path_is_no_issue(tmp_path: Path) -> None:
    candidate = _candidate(
        title="The user chose an alternate path",
        issue="The user followed a different path from the reference path.",
        impact="The task completed successfully.",
        root_cause="The chosen path differs from the reference path.",
    )
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])]
    )

    attempt = await service.synthesize(
        _corpus(tmp_path, extra_entries=(_expectation_entry(),))
    )

    assert attempt.status is SynthesisStatus.NO_ISSUES
    assert not attempt.findings
    assert any("alternate path" in limitation for limitation in attempt.limitations)


@pytest.mark.asyncio
async def test_repeated_low_impact_inconsistency_keeps_low_severity(
    tmp_path: Path,
) -> None:
    candidate = _candidate(
        finding_id="minor-label-inconsistency",
        severity="low",
        title="A minor label differs between states",
        issue="A label is inconsistent in one low-impact state.",
        impact="The wording mismatch is easy to recover from.",
        root_cause="The label differs from another state.",
        severity_justification="The observed mismatch is minor and recoverable.",
    )
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        adjudicator=[AdjudicationResponse(complete=True, final_findings=[candidate])],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.ACCEPTED
    assert attempt.findings[0].severity is FindingSeverity.LOW


@pytest.mark.asyncio
async def test_isolated_blocker_outranks_broad_low_impact_root_cause(
    tmp_path: Path,
) -> None:
    broad = _candidate(
        finding_id="shared-label-inconsistency",
        evidence_id=EVIDENCE_ID,
        severity="low",
        title="Labels vary across several surfaces",
        issue="A low-impact label inconsistency appears across several surfaces.",
        impact="The wording mismatch is recoverable.",
        root_cause="A shared label is inconsistent across surfaces.",
        severity_justification="The evidence shows a broad but low-impact mismatch.",
        affected_surfaces=("settings", "billing", "profile"),
    )
    blocker = _candidate(
        finding_id="task-blocker",
        evidence_id=SECOND_EVIDENCE_ID,
        severity="critical",
        title="The task cannot be completed",
        issue="The user cannot complete the required task.",
        impact="A critical task is blocked.",
        root_cause="The submit control is unavailable.",
        severity_justification="The evidence shows an unrecoverable blocker on a critical task.",
    )
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[broad, blocker])],
        adjudicator=[
            AdjudicationResponse(
                complete=True,
                final_findings=[broad, blocker],
            )
        ],
    )

    attempt = await service.synthesize(
        _corpus(tmp_path, extra_entries=(_second_event_entry(),))
    )

    assert attempt.status is SynthesisStatus.ACCEPTED
    assert [finding.finding_id for finding in attempt.findings] == [
        "task-blocker",
        "shared-label-inconsistency",
    ]
    assert len(attempt.findings[1].affected_surfaces) == 3


@pytest.mark.asyncio
async def test_unsupported_causal_language_is_rejected(tmp_path: Path) -> None:
    candidate = _candidate(
        root_cause="The label causes users to abandon the task.",
        impact="The task takes longer.",
    )
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])]
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.REJECTED


@pytest.mark.asyncio
async def test_counterevidence_cannot_supply_primary_support(tmp_path: Path) -> None:
    payload = _candidate().model_dump(mode="python")
    payload["evidence_refs"] = [
        {
            "evidence_id": "expectation:run-a",
            "kind": "expectation",
            "run_id": "run-a",
        }
    ]
    payload["counterevidence"] = [
        {
            "evidence_id": EVIDENCE_ID,
            "kind": "event",
            "run_id": "run-a",
            "replay_sequence": 1,
        }
    ]
    candidate = CandidateFinding.model_validate(payload)
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        adjudicator=[AdjudicationResponse(complete=True, final_findings=[candidate])],
    )

    attempt = await service.synthesize(
        _corpus(tmp_path, extra_entries=(_expectation_entry(),))
    )

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings


@pytest.mark.asyncio
async def test_counterevidence_cannot_supply_causal_ui_state_support(
    tmp_path: Path,
) -> None:
    payload = _candidate(
        root_cause="The layout causes the extra navigation."
    ).model_dump(mode="python")
    payload["counterevidence"] = [
        {
            "evidence_id": HEATMAP_ID,
            "kind": "heatmap",
            "run_id": "run-a",
            "viewport_id": "viewport-1",
            "sha256": "a" * 64,
        }
    ]
    candidate = CandidateFinding.model_validate(payload)
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        adjudicator=[AdjudicationResponse(complete=True, final_findings=[candidate])],
    )

    attempt = await service.synthesize(
        _corpus(tmp_path, extra_entries=(_heatmap_entry(),))
    )

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings


@pytest.mark.asyncio
async def test_affected_surfaces_must_be_named_by_supporting_evidence(
    tmp_path: Path,
) -> None:
    unsupported = _candidate(affected_surfaces=("admin",))
    supported = _candidate(
        finding_id="settings-surface",
        evidence_id=SECOND_EVIDENCE_ID,
        affected_surfaces=("settings",),
    )
    service, _ = _scripted_service(
        analyst=[
            AnalystResponse(
                complete=True,
                candidate_findings=[unsupported, supported],
            )
        ],
        adjudicator=[AdjudicationResponse(complete=True, final_findings=[supported])],
    )

    attempt = await service.synthesize(
        _corpus(tmp_path, extra_entries=(_surface_event_entry(),))
    )

    assert attempt.status is SynthesisStatus.ACCEPTED
    assert [finding.finding_id for finding in attempt.findings] == ["settings-surface"]
    assert [finding.finding_id for finding in attempt.candidate_findings] == [
        "settings-surface"
    ]


@pytest.mark.asyncio
async def test_scope_and_expectation_metadata_cannot_establish_a_finding(
    tmp_path: Path,
) -> None:
    candidate = _candidate(root_cause="The expectation is not met.")
    candidate = candidate.model_copy(
        update={
            "evidence_refs": [
                EvidenceReference(
                    evidence_id="scenario:run-a",
                    kind="scenario",
                    run_id="run-a",
                ),
                EvidenceReference(
                    evidence_id="expectation:run-a",
                    kind="expectation",
                    run_id="run-a",
                ),
            ]
        }
    )
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])]
    )

    attempt = await service.synthesize(
        _corpus(
            tmp_path,
            extra_entries=(_scenario_entry(), _expectation_entry()),
        )
    )

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings
    assert any(
        "publication validation" in limitation for limitation in attempt.limitations
    )


@pytest.mark.asyncio
async def test_adjudicator_cannot_replace_reviewed_core_claim(tmp_path: Path) -> None:
    reviewed = _candidate(severity="low")
    replacement = _candidate(
        finding_id=reviewed.finding_id,
        severity="critical",
        title="An unrelated account deletion blocker",
        issue="The user cannot delete an account.",
        impact="A separate critical workflow is blocked.",
        root_cause="The account deletion control is missing.",
        severity_justification="A critical account workflow is unavailable.",
    )
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[reviewed])],
        adjudicator=[AdjudicationResponse(complete=True, final_findings=[replacement])],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings
    assert attempt.rejected_findings
    assert any(
        "publication validation" in limitation for limitation in attempt.limitations
    )


@pytest.mark.asyncio
async def test_adjudicator_cannot_change_reviewed_severity_without_resolution(
    tmp_path: Path,
) -> None:
    reviewed = _candidate(severity="high")
    changed = reviewed.model_copy(
        update={
            "severity": FindingSeverity.MEDIUM,
            "confidence": 0.72,
            "severity_justification": "Recovery is immediate.",
        }
    )
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[reviewed])],
        adjudicator=[AdjudicationResponse(complete=True, final_findings=[changed])],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings
    assert attempt.rejected_findings[0].finding_id == reviewed.finding_id


@pytest.mark.asyncio
async def test_evidence_backed_severity_resolution_authorizes_reviewed_change(
    tmp_path: Path,
) -> None:
    reviewed = _candidate(severity="high")
    changed = reviewed.model_copy(
        update={
            "severity": FindingSeverity.MEDIUM,
            "confidence": 0.72,
            "severity_justification": "Recovery is immediate.",
        }
    )
    objection = TypedObjection(
        objection_id="severity-review",
        finding_id=reviewed.finding_id,
        objection_type="severity",
        severity=ObjectionSeverity.MATERIAL,
        message="The evidence shows immediate recovery, so high severity is not established.",
        evidence_refs=[
            EvidenceReference(
                evidence_id=EVIDENCE_ID,
                kind="event",
                run_id="run-a",
                replay_sequence=1,
            )
        ],
        reviewer_role="report-pattern-reviewer",
    )
    resolution = ObjectionResolution(
        objection_id=objection.objection_id,
        finding_id=reviewed.finding_id,
        resolved=True,
        resolution="The final severity and confidence were reduced.",
        evidence_refs=objection.evidence_refs,
    )
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[reviewed])],
        pattern=[PatternReviewResponse(complete=True, objections=[objection])],
        adjudicator=[
            AdjudicationResponse(
                complete=True,
                final_findings=[changed],
                objection_resolutions=[resolution],
            )
        ],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.ACCEPTED
    assert attempt.findings[0].severity is FindingSeverity.MEDIUM
    assert attempt.objections[0].objection_type == "severity"
    assert attempt.objections[0].resolved_by_role == "report-adjudicator"


@pytest.mark.asyncio
async def test_every_objection_requires_explicit_adjudicator_disposition(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    objection = TypedObjection(
        objection_id="editorial-review",
        finding_id=candidate.finding_id,
        objection_type="other",
        severity=ObjectionSeverity.EDITORIAL,
        message="The title could be shorter.",
        evidence_refs=[],
        reviewer_role="report-pattern-reviewer",
    )
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        pattern=[PatternReviewResponse(complete=True, objections=[objection])],
        adjudicator=[
            AdjudicationResponse(complete=True, final_findings=[candidate])
        ],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings
    assert any("explicit disposition" in item for item in attempt.limitations)


@pytest.mark.asyncio
async def test_unresolved_blocking_objection_rejects_finding(tmp_path: Path) -> None:
    candidate = _candidate()
    objection = _blocking_objection(candidate.finding_id)
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        auditor=[EvidenceAuditResponse(complete=True, objections=[objection])],
        adjudicator=[
            AdjudicationResponse(complete=True, final_findings=[candidate]),
            AdjudicationResponse(complete=True, final_findings=[candidate]),
        ],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings
    assert attempt.rejected_findings[0].reviewer_state == "not-established"
    assert attempt.objections[0].severity is ObjectionSeverity.BLOCKING
    assert not attempt.objections[0].resolved


@pytest.mark.asyncio
async def test_unresolved_blocker_rejects_only_its_finding(tmp_path: Path) -> None:
    blocked = _candidate()
    independent = _candidate(
        finding_id="independent-finding",
        evidence_id=SECOND_EVIDENCE_ID,
        title="The settings entry point is hard to find",
        issue="The user searches outside the settings area before finding the entry point.",
        impact="A separate settings task takes longer to complete.",
        root_cause="The entry point is labeled around internal product structure.",
    )
    objection = _blocking_objection(blocked.finding_id)
    disposition = ObjectionResolution(
        objection_id=objection.objection_id,
        finding_id=blocked.finding_id,
        resolved=False,
        resolution="The objection is upheld and the blocked finding must not publish.",
        evidence_refs=objection.evidence_refs,
    )
    service, _ = _scripted_service(
        analyst=[
            AnalystResponse(
                complete=True,
                candidate_findings=[blocked, independent],
            )
        ],
        auditor=[EvidenceAuditResponse(complete=True, objections=[objection])],
        adjudicator=[
            AdjudicationResponse(
                complete=True,
                final_findings=[blocked, independent],
                objection_resolutions=[disposition],
            ),
            AdjudicationResponse(
                complete=True,
                final_findings=[blocked, independent],
            ),
        ],
    )

    attempt = await service.synthesize(
        _corpus(tmp_path, extra_entries=(_second_event_entry(),))
    )

    assert attempt.status is SynthesisStatus.ACCEPTED
    assert [finding.finding_id for finding in attempt.findings] == [
        "independent-finding"
    ]
    assert [finding.finding_id for finding in attempt.rejected_findings] == [
        "invite-control"
    ]
    assert not attempt.objections[0].resolved


@pytest.mark.asyncio
async def test_one_adjudication_revision_can_resolve_blocking_objection(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    objection = _blocking_objection(candidate.finding_id)
    resolution_ref = EvidenceReference(
        evidence_id=SECOND_EVIDENCE_ID,
        kind="event",
        run_id="run-a",
        replay_sequence=2,
    )
    resolution = ObjectionResolution(
        objection_id=objection.objection_id,
        finding_id=candidate.finding_id,
        resolved=True,
        resolution="The event plus the reviewed result supports the final claim.",
        evidence_refs=[resolution_ref],
    )
    service, roles = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        auditor=[EvidenceAuditResponse(complete=True, objections=[objection])],
        adjudicator=[
            AdjudicationResponse(complete=True, final_findings=[candidate]),
            AdjudicationResponse(
                complete=True,
                final_findings=[candidate],
                objection_resolutions=[resolution],
            ),
        ],
    )

    attempt = await service.synthesize(
        _corpus(tmp_path, extra_entries=(_second_event_entry(),))
    )

    assert attempt.status is SynthesisStatus.ACCEPTED
    assert not [
        item
        for item in attempt.objections
        if item.severity is ObjectionSeverity.BLOCKING and not item.resolved
    ]
    assert attempt.objections[0].evidence_refs == tuple(
        ref.to_domain() for ref in objection.evidence_refs
    )
    assert attempt.objections[0].resolved_by_role == "report-adjudicator"
    assert attempt.objections[0].resolution_evidence_refs == (
        resolution_ref.to_domain(),
    )
    assert len(roles[3].calls) == 2


@pytest.mark.asyncio
async def test_one_adjudication_revision_dispositions_material_objection(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    objection = _blocking_objection(candidate.finding_id).model_copy(
        update={"severity": ObjectionSeverity.MATERIAL}
    )
    resolution = ObjectionResolution(
        objection_id=objection.objection_id,
        finding_id=candidate.finding_id,
        resolved=True,
        resolution="The cited event supports the reviewed claim.",
        evidence_refs=objection.evidence_refs,
    )
    service, roles = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        auditor=[EvidenceAuditResponse(complete=True, objections=[objection])],
        adjudicator=[
            AdjudicationResponse(complete=True, final_findings=[candidate]),
            AdjudicationResponse(
                complete=True,
                final_findings=[candidate],
                objection_resolutions=[resolution],
            ),
        ],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.ACCEPTED
    assert attempt.objections[0].resolved_by_role == "report-adjudicator"
    assert len(roles[3].calls) == 2


@pytest.mark.asyncio
async def test_adjudication_revision_repairs_unauthorized_evidence_contraction(
    tmp_path: Path,
) -> None:
    candidate = _candidate().model_copy(
        update={
            "evidence_refs": [
                *_candidate().evidence_refs,
                EvidenceReference(
                    evidence_id=SECOND_EVIDENCE_ID,
                    kind="event",
                    run_id="run-a",
                    replay_sequence=2,
                ),
            ]
        }
    )
    narrowed = candidate.model_copy(
        update={"evidence_refs": [candidate.evidence_refs[0]]}
    )
    service, roles = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        adjudicator=[
            AdjudicationResponse(complete=True, final_findings=[narrowed]),
            AdjudicationResponse(complete=True, final_findings=[candidate]),
        ],
    )

    attempt = await service.synthesize(
        _corpus(tmp_path, extra_entries=(_second_event_entry(),))
    )

    assert attempt.status is SynthesisStatus.ACCEPTED
    assert len(roles[3].calls) == 2


@pytest.mark.asyncio
async def test_configured_adjudication_revision_limit_is_honored(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    objection = _blocking_objection(candidate.finding_id)
    resolution = ObjectionResolution(
        objection_id=objection.objection_id,
        finding_id=candidate.finding_id,
        resolved=True,
        resolution="The cited event resolves the reviewed objection.",
        evidence_refs=objection.evidence_refs,
    )
    service, roles = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        auditor=[EvidenceAuditResponse(complete=True, objections=[objection])],
        adjudicator=[
            AdjudicationResponse(complete=True, final_findings=[candidate]),
            AdjudicationResponse(complete=True, final_findings=[candidate]),
            AdjudicationResponse(
                complete=True,
                final_findings=[candidate],
                objection_resolutions=[resolution],
            ),
        ],
        max_adjudication_revisions=2,
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.ACCEPTED
    assert len(roles[3].calls) == 3


@pytest.mark.asyncio
async def test_configured_final_verification_count_is_honored(
    tmp_path: Path,
) -> None:
    candidate = _candidate()

    class CountingService(ReportSynthesisService):
        final_verification_calls = 0

        def _final_verification(self, *args: Any, **kwargs: Any):
            self.final_verification_calls += 1
            return super()._final_verification(*args, **kwargs)

    record_source = _RecordingModelSource()
    roles = (
        _ScriptedRole(
            "analyst",
            [AnalystResponse(complete=True, candidate_findings=[candidate])],
            record_source,
        ),
        _ScriptedRole(
            "auditor", [EvidenceAuditResponse(complete=True)], record_source
        ),
        _ScriptedRole(
            "pattern", [PatternReviewResponse(complete=True)], record_source
        ),
        _ScriptedRole(
            "adjudicator",
            [AdjudicationResponse(complete=True, final_findings=[candidate])],
            record_source,
        ),
    )
    service = CountingService(
        analyst=roles[0],
        evidence_auditor=roles[1],
        pattern_reviewer=roles[2],
        adjudicator=roles[3],
        max_final_verifications=2,
        model_record_source=record_source,
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.ACCEPTED
    assert service.final_verification_calls == 2


@pytest.mark.asyncio
async def test_missing_expectation_is_explicit_limitation(tmp_path: Path) -> None:
    service, _ = _scripted_service()

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.NO_ISSUES
    assert not attempt.findings
    assert any(
        "No frozen expectation matched" in limitation
        for limitation in attempt.limitations
    )


@pytest.mark.asyncio
async def test_transport_failure_returns_safe_unavailable_state(tmp_path: Path) -> None:
    service, _ = _scripted_service(analyst=[RuntimeError("secret endpoint details")])

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.UNAVAILABLE
    assert attempt.fallback_available
    assert all(
        "secret endpoint details" not in limitation
        for limitation in attempt.limitations
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_kind", ["duplicate", "orphan", "mismatch"])
async def test_invalid_objection_resolutions_are_rejected_without_clearing_objection(
    tmp_path: Path,
    invalid_kind: str,
) -> None:
    candidate = _candidate()
    objection = _blocking_objection(candidate.finding_id)
    resolution = ObjectionResolution(
        objection_id=("orphan" if invalid_kind == "orphan" else objection.objection_id),
        finding_id=(
            "other-finding" if invalid_kind == "mismatch" else candidate.finding_id
        ),
        resolved=True,
        resolution="The evidence supports this resolution.",
        evidence_refs=objection.evidence_refs,
    )
    resolutions = (
        [resolution, resolution] if invalid_kind == "duplicate" else [resolution]
    )
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        auditor=[EvidenceAuditResponse(complete=True, objections=[objection])],
        adjudicator=[
            AdjudicationResponse(
                complete=True,
                final_findings=[candidate],
                objection_resolutions=resolutions,
            )
        ],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings
    assert attempt.objections[0].resolved is False
    assert attempt.rejected_findings[0].reviewer_state == "not-established"
    assert any("resolution" in limitation.lower() for limitation in attempt.limitations)


@pytest.mark.asyncio
async def test_invalid_structured_role_output_is_rejected_not_unavailable(
    tmp_path: Path,
) -> None:
    service, _ = _scripted_service(
        analyst=[
            ModelResponseValidationError(
                ModelRole.REPORT_ANALYST,
                "invalid structured output",
            ),
            ModelResponseValidationError(
                ModelRole.REPORT_ANALYST,
                "invalid structured output",
            ),
        ]
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.REJECTED
    assert attempt.status is not SynthesisStatus.UNAVAILABLE
    assert any("invalid" in limitation.lower() for limitation in attempt.limitations)


@pytest.mark.asyncio
async def test_invalid_structured_role_output_retries_once_without_fallback(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    service, roles = _scripted_service(
        analyst=[
            ModelResponseValidationError(
                ModelRole.REPORT_ANALYST,
                "invalid structured output",
            ),
            AnalystResponse(complete=True, candidate_findings=[candidate]),
        ],
        adjudicator=[AdjudicationResponse(complete=True, final_findings=[candidate])],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.ACCEPTED
    assert len(roles[0].calls) == 2
    assert any(
        log["response"].get("retrying") is True
        for log in attempt.retrieval_log
        if log["role"] == ModelRole.REPORT_ANALYST.value
    )


@pytest.mark.asyncio
async def test_transport_reason_with_structural_diagnostics_is_retried(
    tmp_path: Path,
) -> None:
    """A connect error must not mask an earlier structural rejection.

    The adapter can end a bounded call on a transport error while still
    reporting the stage of an attempt that failed schema validation. That is
    an output problem, not an outage, so the role is retried and the attempt
    is rejected rather than declared unavailable.
    """

    structural = ModelFailureError(
        "connect-error",
        diagnostics={"stage": "schema_validation", "attempt_count": 1},
    )
    service, roles = _scripted_service(analyst=[structural, structural])

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.REJECTED
    assert attempt.status is not SynthesisStatus.UNAVAILABLE
    assert len(roles[0].calls) == 2


@pytest.mark.asyncio
async def test_provider_outage_with_stale_diagnostics_stays_unavailable(
    tmp_path: Path,
) -> None:
    service, roles = _scripted_service(
        analyst=[
            ModelFailureError(
                "rate limit",
                status_code=429,
                diagnostics={"stage": "schema_validation"},
            )
        ]
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.UNAVAILABLE
    assert len(roles[0].calls) == 1


@pytest.mark.asyncio
async def test_reviewer_alias_is_normalized_to_canonical_role(tmp_path: Path) -> None:
    candidate = _candidate()
    objection = _blocking_objection(candidate.finding_id).model_copy(
        update={"reviewer_role": "evidence-auditor"}
    )
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        auditor=[EvidenceAuditResponse(complete=True, objections=[objection])],
        adjudicator=[AdjudicationResponse(complete=True)],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.objections[0].reviewer_role == "report-evidence-auditor"
    assert all(
        item.reviewer_role in {"report-evidence-auditor", "report-pattern-reviewer"}
        for item in attempt.objections
    )


@pytest.mark.asyncio
async def test_spaced_reviewer_alias_is_normalized_to_canonical_role(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    objection = _blocking_objection(candidate.finding_id).model_copy(
        update={"reviewer_role": "evidence auditor"}
    )
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        auditor=[EvidenceAuditResponse(complete=True, objections=[objection])],
        adjudicator=[AdjudicationResponse(complete=True)],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.objections[0].reviewer_role == "report-evidence-auditor"


@pytest.mark.asyncio
async def test_unknown_reviewer_alias_rejects_synthesis(tmp_path: Path) -> None:
    candidate = _candidate()
    objection = _blocking_objection(candidate.finding_id).model_copy(
        update={"reviewer_role": "arbitrary-reviewer"}
    )
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        auditor=[EvidenceAuditResponse(complete=True, objections=[objection])],
        adjudicator=[AdjudicationResponse(complete=True)],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.objections
    assert any("Reviewer objections failed" in item for item in attempt.limitations)


@pytest.mark.asyncio
async def test_attempt_usage_aggregates_model_call_records(tmp_path: Path) -> None:
    source = _ModelRecordSource(
        (
            ModelCallRecord(
                role=ModelRole.REPORT_ANALYST,
                model="report-model",
                endpoint_origin="https://llm.example.test",
                prompt_digest="a" * 64,
                schema_version="report-analyst-v1",
                attempts=1,
                latency_ms=17,
                token_usage=TokenUsage(4, 3, 7),
                request={},
                response={},
            ),
            ModelCallRecord(
                role=ModelRole.REPORT_ADJUDICATOR,
                model="report-model",
                endpoint_origin="https://llm.example.test",
                prompt_digest="b" * 64,
                schema_version="report-adjudicator-v1",
                attempts=2,
                latency_ms=23,
                token_usage=TokenUsage(8, 5, 13),
                request={},
                response={},
            ),
        )
    )
    service, _ = _scripted_service(model_record_source=source)

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.usage["role_calls"] == 2
    assert attempt.usage["prompt_tokens"] == 12
    assert attempt.usage["completion_tokens"] == 8
    assert attempt.usage["total_tokens"] == 20
    assert attempt.usage["latency_ms"] == 40
    assert attempt.usage["usage_available"] == 1


@pytest.mark.asyncio
async def test_attempt_per_role_usage_breaks_down_by_role(tmp_path: Path) -> None:
    source = _ModelRecordSource(
        (
            ModelCallRecord(
                role=ModelRole.REPORT_ANALYST,
                model="report-model",
                endpoint_origin="https://llm.example.test",
                prompt_digest="a" * 64,
                schema_version="report-analyst-v1",
                attempts=1,
                latency_ms=17,
                token_usage=TokenUsage(4, 3, 7, reasoning_tokens=1, cached_tokens=2),
                request={},
                response={},
            ),
            ModelCallRecord(
                role=ModelRole.REPORT_ADJUDICATOR,
                model="report-model",
                endpoint_origin="https://llm.example.test",
                prompt_digest="b" * 64,
                schema_version="report-adjudicator-v1",
                attempts=2,
                latency_ms=23,
                token_usage=TokenUsage(8, 5, 13),
                request={},
                response={},
            ),
            ModelCallRecord(
                role=ModelRole.REPORT_ADJUDICATOR,
                model="report-model",
                endpoint_origin="https://llm.example.test",
                prompt_digest="c" * 64,
                schema_version="report-adjudicator-v1",
                attempts=1,
                latency_ms=10,
                token_usage=TokenUsage(2, 2, 4),
                request={},
                response={},
            ),
        )
    )
    service, _ = _scripted_service(model_record_source=source)

    attempt = await service.synthesize(_corpus(tmp_path))

    analyst = attempt.per_role_usage[ModelRole.REPORT_ANALYST.value]
    adjudicator = attempt.per_role_usage[ModelRole.REPORT_ADJUDICATOR.value]
    assert analyst["role_calls"] == 1
    assert analyst["prompt_tokens"] == 4
    assert analyst["reasoning_tokens"] == 1
    assert analyst["cached_tokens"] == 2
    assert analyst["latency_ms"] == 17
    assert adjudicator["role_calls"] == 2
    assert adjudicator["model_attempts"] == 3
    assert adjudicator["prompt_tokens"] == 10
    assert adjudicator["total_tokens"] == 17
    assert adjudicator["latency_ms"] == 33
    # Totals agree with the aggregate usage block.
    assert attempt.usage["prompt_tokens"] == sum(
        entry["prompt_tokens"] for entry in attempt.per_role_usage.values()
    )
    assert attempt.usage["latency_ms"] == sum(
        entry["latency_ms"] for entry in attempt.per_role_usage.values()
    )
    assert attempt.attempt_wall_ms >= 0.0


@pytest.mark.asyncio
async def test_no_issues_requires_valid_completion_provenance_for_every_role(
    tmp_path: Path,
) -> None:
    source = _ModelRecordSource(
        (
            ModelCallRecord(
                role=ModelRole.REPORT_ANALYST,
                model="report-model",
                endpoint_origin="https://llm.example.test",
                prompt_digest="a" * 64,
                schema_version="wrong-schema-version",
                attempts=1,
                latency_ms=17,
                token_usage=TokenUsage(4, 3, 7),
                request={},
                response={},
            ),
        )
    )
    service, _ = _scripted_service(model_record_source=source)

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings
    assert any("completion receipts" in item for item in attempt.limitations)


@pytest.mark.asyncio
async def test_adjudicator_cannot_reclassify_a_scenario_defect(tmp_path: Path) -> None:
    """The analyst owns the classification, so a scenario defect that the
    adjudicator rewrites as a product issue still publishes as a defect."""

    candidate = _candidate().model_copy(
        update={"finding_kind": FindingKind.SCENARIO_DEFECT}
    )
    drifted = candidate.model_copy(update={"finding_kind": FindingKind.UX_ISSUE})
    service, _ = _scripted_service(
        analyst=[AnalystResponse(complete=True, candidate_findings=[candidate])],
        adjudicator=[AdjudicationResponse(complete=True, final_findings=[drifted])],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.ACCEPTED
    assert attempt.findings[0].finding_kind is FindingKind.SCENARIO_DEFECT


@pytest.mark.asyncio
async def test_invalid_output_retry_carries_validation_feedback(
    tmp_path: Path,
) -> None:
    """A retried role must receive the safe validation reason, not a blind re-send."""

    candidate = _candidate()
    invalid = ModelResponseValidationError(
        ModelRole.REPORT_ANALYST,
        "finding references an unknown UX principle: accessibility-contrast",
    )
    service, roles = _scripted_service(
        analyst=[invalid, AnalystResponse(complete=True, candidate_findings=[candidate])],
        adjudicator=[AdjudicationResponse(complete=True, final_findings=[candidate])],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.ACCEPTED
    assert len(roles[0].calls) == 2
    first_kwargs = roles[0].calls[0]["kwargs"]
    second_kwargs = roles[0].calls[1]["kwargs"]
    assert first_kwargs.get("validation_feedback") is None
    feedback = second_kwargs.get("validation_feedback")
    assert feedback is not None
    assert "unknown UX principle" in feedback
    # The feedback must be the bounded safe phrase, never raw attacker text.
    assert "accessibility-contrast" not in feedback
    retrieval_logs = [
        log
        for log in attempt.retrieval_log
        if log.get("role") == ModelRole.REPORT_ANALYST.value
        and log.get("response", {}).get("retrying") is True
    ]
    assert retrieval_logs
    assert all(
        "validation_feedback" in log["response"] for log in retrieval_logs
    )
