from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeVar, cast

import pytest
from pydantic import BaseModel

from ux_analyzer.adapters.openai import ModelFailureError
from ux_analyzer.application.evidence_corpus import EvidenceCorpus, EvidenceEntry
from ux_analyzer.application.report_synthesis import ReportSynthesisService
from ux_analyzer.domain.findings import EvidenceClass
from ux_analyzer.domain.synthesis import EvidenceRef, SynthesisStatus
from ux_analyzer.ports.models import ChatMessage, ModelRole
from ux_analyzer.providers.report_synthesis import (
    EvidenceAuditor,
    PatternReviewer,
    ReportAdjudicator,
    ReportAnalyst,
)

EVIDENCE_ID = "event:run-a:1"
SchemaT = TypeVar("SchemaT", bound=BaseModel)


def _corpus(tmp_path: Path) -> EvidenceCorpus:
    return EvidenceCorpus(
        output_root=tmp_path,
        entries=(
            EvidenceEntry(
                ref=EvidenceRef(EVIDENCE_ID, "event", "run-a", replay_sequence=1),
                evidence_class=EvidenceClass.DETERMINISTIC_FACT,
                summary="The user opened the invite control after searching.",
                payload={"sequence": 1, "succeeded": True},
            ),
        ),
    )


def _candidate_payload() -> dict[str, object]:
    evidence_refs: list[dict[str, object]] = [
        {
            "evidence_id": EVIDENCE_ID,
            "kind": "event",
            "run_id": "run-a",
            "replay_sequence": 1,
        }
    ]
    return {
        "finding_id": "invite-control",
        "title": "The invite control is hard to find",
        "issue": "The user searches outside the expected task area before finding the invite control.",
        "impact": "An important collaboration task takes longer to complete.",
        "root_cause": "The entry point is labeled around internal product structure.",
        "fixes": ["Label the entry point around the user's goal."],
        "severity": "high",
        "confidence": 0.9,
        "evidence_refs": evidence_refs,
        "severity_justification": "The evidence shows extra navigation on an important task.",
    }


class _StructuredClient:
    endpoint_origin = "https://llm.example.test/v1"
    provider_id = "integration-client"
    provider_version = "integration-v1"

    def __init__(self, scripts: Mapping[ModelRole, Sequence[object]]) -> None:
        self.scripts = {role: list(values) for role, values in scripts.items()}
        self.calls: list[
            tuple[type[BaseModel], tuple[ChatMessage, ...], ModelRole]
        ] = []

    async def complete(
        self,
        schema: type[SchemaT],
        messages: Sequence[ChatMessage],
        model: str,
        role: ModelRole,
    ) -> SchemaT:
        del model
        message_tuple = tuple(messages)
        self.calls.append((schema, message_tuple, role))
        response = self.scripts[role].pop(0)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, BaseModel):
            return cast(SchemaT, response)
        return schema.model_validate(response)


def _providers(
    client: _StructuredClient,
) -> ReportSynthesisService:
    return ReportSynthesisService(
        analyst=ReportAnalyst(client, model="gpt-report"),
        evidence_auditor=EvidenceAuditor(client, model="gpt-report"),
        pattern_reviewer=PatternReviewer(client, model="gpt-report"),
        adjudicator=ReportAdjudicator(client, model="gpt-report"),
    )


@pytest.mark.asyncio
async def test_integration_orchestrates_real_role_providers_and_retrieval(
    tmp_path: Path,
) -> None:
    candidate = _candidate_payload()
    client = _StructuredClient(
        {
            ModelRole.REPORT_ANALYST: [
                {
                    "complete": False,
                    "evidence_requests": [EVIDENCE_ID],
                },
                {
                    "complete": True,
                    "candidate_findings": [candidate],
                },
            ],
            ModelRole.REPORT_EVIDENCE_AUDITOR: [{"complete": True, "objections": []}],
            ModelRole.REPORT_PATTERN_REVIEWER: [{"complete": True, "objections": []}],
            ModelRole.REPORT_ADJUDICATOR: [
                {"complete": True, "final_findings": [candidate]}
            ],
        }
    )

    attempt = await _providers(client).synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.ACCEPTED
    assert attempt.findings[0].finding_id == "invite-control"
    assert (
        len([call for call in client.calls if call[2] is ModelRole.REPORT_ANALYST]) == 2
    )
    assert {call[2] for call in client.calls} == {
        ModelRole.REPORT_ANALYST,
        ModelRole.REPORT_EVIDENCE_AUDITOR,
        ModelRole.REPORT_PATTERN_REVIEWER,
        ModelRole.REPORT_ADJUDICATOR,
    }
    assert any(
        entry["role"] == ModelRole.REPORT_ANALYST.value
        and entry["resolved_evidence_ids"] == (EVIDENCE_ID,)
        for entry in attempt.retrieval_log
    )


@pytest.mark.asyncio
async def test_integration_unresolved_blocking_review_is_rejected(
    tmp_path: Path,
) -> None:
    candidate = _candidate_payload()
    evidence_ref = cast(list[dict[str, object]], candidate["evidence_refs"])
    objection: dict[str, object] = {
        "objection_id": "objection-1",
        "finding_id": "invite-control",
        "objection_type": "factual-support",
        "severity": "blocking",
        "message": "The cited event does not establish the stated cause.",
        "evidence_refs": [evidence_ref[0]],
        "reviewer_role": "report-evidence-auditor",
    }
    client = _StructuredClient(
        {
            ModelRole.REPORT_ANALYST: [
                {"complete": False, "evidence_requests": [EVIDENCE_ID]},
                {"complete": True, "candidate_findings": [candidate]},
            ],
            ModelRole.REPORT_EVIDENCE_AUDITOR: [
                {"complete": True, "objections": [objection]}
            ],
            ModelRole.REPORT_PATTERN_REVIEWER: [{"complete": True, "objections": []}],
            ModelRole.REPORT_ADJUDICATOR: [
                {"complete": True, "final_findings": [candidate]},
                {"complete": True, "final_findings": [candidate]},
            ],
        }
    )

    attempt = await _providers(client).synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings
    assert attempt.rejected_findings[0].reviewer_state == "not-established"
    assert (
        len([call for call in client.calls if call[2] is ModelRole.REPORT_ANALYST]) == 2
    )
    assert (
        len([call for call in client.calls if call[2] is ModelRole.REPORT_ADJUDICATOR])
        == 2
    )


@pytest.mark.asyncio
async def test_integration_reviewer_cannot_pre_resolve_its_blocking_objection(
    tmp_path: Path,
) -> None:
    candidate = _candidate_payload()
    evidence_ref = cast(list[dict[str, object]], candidate["evidence_refs"])
    objection = {
        "objection_id": "objection-1",
        "finding_id": "invite-control",
        "objection_type": "factual-support",
        "severity": "blocking",
        "message": "The cited event does not establish the stated cause.",
        "evidence_refs": [evidence_ref[0]],
        "reviewer_role": "report-evidence-auditor",
        "resolved": True,
        "resolution": "The auditor considers its own objection resolved.",
    }
    client = _StructuredClient(
        {
            ModelRole.REPORT_ANALYST: [
                {"complete": False, "evidence_requests": [EVIDENCE_ID]},
                {"complete": True, "candidate_findings": [candidate]},
            ],
            ModelRole.REPORT_EVIDENCE_AUDITOR: [
                {"complete": True, "objections": [objection]}
            ],
            ModelRole.REPORT_PATTERN_REVIEWER: [{"complete": True, "objections": []}],
            ModelRole.REPORT_ADJUDICATOR: [
                {"complete": True, "final_findings": [candidate]},
                {"complete": True, "final_findings": [candidate]},
            ],
        }
    )

    attempt = await _providers(client).synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings
    assert not attempt.objections[0].resolved
    assert attempt.objections[0].resolution is None


@pytest.mark.asyncio
async def test_integration_transport_failure_keeps_fallback_available(
    tmp_path: Path,
) -> None:
    client = _StructuredClient(
        {
            ModelRole.REPORT_ANALYST: [RuntimeError("do not persist this detail")],
            ModelRole.REPORT_EVIDENCE_AUDITOR: [],
            ModelRole.REPORT_PATTERN_REVIEWER: [],
            ModelRole.REPORT_ADJUDICATOR: [],
        }
    )

    attempt = await _providers(client).synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.UNAVAILABLE
    assert attempt.fallback_available
    assert all("do not persist" not in item for item in attempt.limitations)


@pytest.mark.asyncio
async def test_integration_provider_failure_keeps_safe_diagnostics_in_fallback(
    tmp_path: Path,
) -> None:
    client = _StructuredClient(
        {
            ModelRole.REPORT_ANALYST: [
                ModelFailureError(
                    "model unavailable",
                    status_code=503,
                    error_code="MODEL_UNAVAILABLE",
                    error_type="server_error",
                    request_id="request-123",
                )
            ],
            ModelRole.REPORT_EVIDENCE_AUDITOR: [],
            ModelRole.REPORT_PATTERN_REVIEWER: [],
            ModelRole.REPORT_ADJUDICATOR: [],
        }
    )

    attempt = await _providers(client).synthesize(_corpus(tmp_path))

    assert attempt.status is SynthesisStatus.UNAVAILABLE
    assert attempt.fallback_available
    provider_metadata = attempt.retrieval_log[0]["response"]["provider"]
    assert provider_metadata == {
        "status_code": 503,
        "error_code": "MODEL_UNAVAILABLE",
        "error_type": "server_error",
        "request_id": "request-123",
    }
    assert attempt.retrieval_log[0]["error"] == "model provider unavailable"
    assert "HTTP 503" in attempt.limitations[-1]
    assert "request-123" in attempt.limitations[-1]
