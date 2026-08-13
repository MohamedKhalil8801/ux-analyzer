from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from ux_analyzer.application.evidence_corpus import (
    EvidenceCorpus,
    EvidenceEntry,
    EvidenceResolver,
)
from ux_analyzer.application.report_synthesis import ReportSynthesisService
from ux_analyzer.domain.findings import EvidenceClass, FindingSeverity
from ux_analyzer.domain.synthesis import (
    EvidenceRef,
    ObjectionSeverity,
    SynthesisStatus,
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
    TypedObjection,
)

EVIDENCE_ID = "event:run-a:1"
SECOND_EVIDENCE_ID = "event:run-a:2"
HEATMAP_ID = "heatmap:run-a:viewport-1:1s"


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
                payload={"sequence": 1, "succeeded": True},
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


class _HappyAnalyst:
    def __init__(self, candidate: CandidateFinding) -> None:
        self.candidate = candidate
        self.calls: list[dict[str, Any]] = []

    async def analyze(self, *args: Any, **kwargs: Any) -> AnalystResponse:
        self.calls.append({"args": args, "kwargs": kwargs})
        return AnalystResponse(
            complete=True,
            candidate_findings=[self.candidate],
        )


class _HappyAuditor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def audit(self, *args: Any, **kwargs: Any) -> EvidenceAuditResponse:
        self.calls.append({"args": args, "kwargs": kwargs})
        return EvidenceAuditResponse(complete=True)


class _HappyPatternReviewer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def review(self, *args: Any, **kwargs: Any) -> PatternReviewResponse:
        self.calls.append({"args": args, "kwargs": kwargs})
        return PatternReviewResponse(complete=True)


class _HappyAdjudicator:
    def __init__(self, candidate: CandidateFinding) -> None:
        self.candidate = candidate
        self.calls: list[dict[str, Any]] = []

    async def adjudicate(self, *args: Any, **kwargs: Any) -> AdjudicationResponse:
        self.calls.append({"args": args, "kwargs": kwargs})
        return AdjudicationResponse(
            complete=True,
            final_findings=[self.candidate],
        )


class _ScriptedRole:
    def __init__(self, role: str, responses: Sequence[object]) -> None:
        self.role = role
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def _next(self, **call: Any) -> object:
        self.calls.append(call)
        response = self.responses.pop(0) if self.responses else None
        if isinstance(response, BaseException):
            raise response
        if response is not None:
            return response
        if self.role == "analyst":
            return AnalystResponse(complete=True)
        if self.role == "auditor":
            return EvidenceAuditResponse(complete=True)
        if self.role == "pattern":
            return PatternReviewResponse(complete=True)
        return AdjudicationResponse(complete=True)

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


def _scripted_service(
    *,
    analyst: Sequence[object] = (),
    auditor: Sequence[object] = (),
    pattern: Sequence[object] = (),
    adjudicator: Sequence[object] = (),
    resolver: EvidenceResolver | None = None,
    max_adjudication_revisions: int = 1,
    model_record_source: object | None = None,
) -> tuple[ReportSynthesisService, tuple[_ScriptedRole, ...]]:
    roles = (
        _ScriptedRole("analyst", analyst),
        _ScriptedRole("auditor", auditor),
        _ScriptedRole("pattern", pattern),
        _ScriptedRole("adjudicator", adjudicator),
    )
    return (
        ReportSynthesisService(
            analyst=roles[0],
            evidence_auditor=roles[1],
            pattern_reviewer=roles[2],
            adjudicator=roles[3],
            resolver=resolver,
            max_adjudication_revisions=max_adjudication_revisions,
            model_record_source=model_record_source,
        ),
        roles,
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
    analyst = _HappyAnalyst(candidate)
    auditor = _HappyAuditor()
    pattern_reviewer = _HappyPatternReviewer()
    adjudicator = _HappyAdjudicator(candidate)
    service = ReportSynthesisService(
        analyst=analyst,
        evidence_auditor=auditor,
        pattern_reviewer=pattern_reviewer,
        adjudicator=adjudicator,
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.ACCEPTED
    assert attempt.findings[0].evidence_refs
    assert not [
        objection
        for objection in attempt.objections
        if objection.severity is ObjectionSeverity.BLOCKING and not objection.resolved
    ]
    assert analyst.calls
    assert auditor.calls
    assert pattern_reviewer.calls
    assert adjudicator.calls
    assert {str(entry["role"]) for entry in attempt.retrieval_log} == {
        "report-analyst",
        "report-evidence-auditor",
        "report-pattern-reviewer",
        "report-adjudicator",
    }


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
async def test_retrieval_chunks_large_requests_before_resolver_boundary(
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
        analyst=[
            AnalystResponse(complete=False, evidence_requests=evidence_ids),
            AnalystResponse(complete=True),
        ],
        resolver=resolver,
    )

    attempt = await service.synthesize(
        _corpus(tmp_path, extra_entries=extra_entries)
    )

    assert attempt.status is SynthesisStatus.NO_ISSUES
    assert resolver.calls == [tuple(evidence_ids[:32]), tuple(evidence_ids[32:])]
    assert resolver.limits == [
        {"max_entries": 32, "max_attachment_bytes": 16 * 1024 * 1024},
        {"max_entries": 32, "max_attachment_bytes": 16 * 1024 * 1024},
    ]
    retrieval_log = next(
        entry
        for entry in attempt.retrieval_log
        if entry["request"] == tuple(evidence_ids)
    )
    batch_logs = retrieval_log["batches"]
    assert [(entry["batch_index"], entry["batch_count"]) for entry in batch_logs] == [
        (0, 2),
        (1, 2),
    ]
    assert batch_logs[0]["request"] == tuple(evidence_ids[:32])
    assert batch_logs[1]["request"] == tuple(evidence_ids[32:])
    assert batch_logs[0]["resolved_evidence_ids"] == tuple(evidence_ids[:32])
    assert batch_logs[1]["resolved_evidence_ids"] == tuple(evidence_ids[32:])


@pytest.mark.asyncio
async def test_retrieval_budget_returns_unavailable_without_fourth_call(
    tmp_path: Path,
) -> None:
    resolver = _RecordingResolver()
    incomplete = AnalystResponse(complete=False, evidence_requests=[EVIDENCE_ID])
    service, roles = _scripted_service(
        analyst=[incomplete, incomplete, incomplete],
        resolver=resolver,
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.UNAVAILABLE
    assert len(roles[0].calls) == 3
    assert len(resolver.calls) == 3
    assert any("retrieval budget" in limitation for limitation in attempt.limitations)
    assert all(entry["role"] == "report-analyst" for entry in attempt.retrieval_log)


@pytest.mark.asyncio
async def test_hallucinated_reference_is_rejected_before_publication(
    tmp_path: Path,
) -> None:
    candidate = _candidate(evidence_id="event:run-a:999")
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
async def test_cross_reviewer_duplicate_objection_ids_reject_publication(
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
        adjudicator=[AdjudicationResponse(complete=True, final_findings=[candidate])],
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings
    assert any("objection" in item.lower() for item in attempt.limitations)


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
    assert not attempt.findings
    assert any(
        "publication validation" in limitation for limitation in attempt.limitations
    )


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
    resolution = ObjectionResolution(
        objection_id=objection.objection_id,
        finding_id=candidate.finding_id,
        resolved=True,
        resolution="The event plus the reviewed result supports the final claim.",
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
    assert not [
        item
        for item in attempt.objections
        if item.severity is ObjectionSeverity.BLOCKING and not item.resolved
    ]
    assert len(roles[3].calls) == 2


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
            )
        ]
    )

    attempt = await service.synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.REJECTED
    assert attempt.status is not SynthesisStatus.UNAVAILABLE
    assert any("invalid" in limitation.lower() for limitation in attempt.limitations)


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
