from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ux_analyzer.application.evidence_corpus import (
    EvidenceCorpus,
    EvidenceEntry,
    EvidenceResolver,
)
from ux_analyzer.domain.findings import EvidenceClass, FindingSeverity
from ux_analyzer.domain.synthesis import EvidenceRef, ObjectionSeverity
from ux_analyzer.ports.models import ModelResponseValidationError, ModelRole
from ux_analyzer.providers.report_synthesis import (
    AdjudicationResponse,
    AnalystResponse,
    CandidateFinding,
    EvidenceAuditor,
    EvidenceAuditResponse,
    ObjectionResolution,
    PatternReviewer,
    PatternReviewResponse,
    ReportAdjudicator,
    ReportAnalyst,
    ReportAnalystResponse,
    ReportEvidenceAuditorResponse,
    ReportPatternReviewerResponse,
    TypedObjection,
)
from ux_analyzer.providers.ux_principles import ux_principles

EVIDENCE_ID = "event:run-a:1"


class RecordingClient:
    endpoint_origin = "https://llm.example.test/v1"
    provider_id = "recording-client"
    provider_version = "recording-v1"

    def __init__(
        self,
        response_factory: Callable[[type[Any], ModelRole], object] | None = None,
    ) -> None:
        self.calls: list[tuple[type[Any], tuple[Any, ...], str, ModelRole]] = []
        self.messages: list[Any] = []
        self.response_factory = response_factory or self._default_response

    async def complete(
        self,
        schema: type[Any],
        messages: Sequence[Any],
        model: str,
        role: ModelRole,
    ) -> object:
        message_tuple = tuple(messages)
        self.calls.append((schema, message_tuple, model, role))
        self.messages.extend(message_tuple)
        return self.response_factory(schema, role)

    @staticmethod
    def _default_response(schema: type[Any], role: ModelRole) -> object:
        del role
        payload: dict[str, object] = {
            "complete": True,
            "evidence_requests": [],
        }
        if schema is AnalystResponse:
            payload["candidate_findings"] = []
        elif schema in {EvidenceAuditResponse, PatternReviewResponse}:
            payload["objections"] = []
        elif schema is AdjudicationResponse:
            payload["final_findings"] = []
            payload["objection_resolutions"] = []
        return schema.model_validate(payload)


def _manifest(*, include_sentinels: bool = False) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": "evidence-corpus-v1",
        "principle_pack_version": "ux-principles-v1",
        "entries": [
            {
                "evidence_id": EVIDENCE_ID,
                "kind": "event",
                "run_id": "run-a",
                "summary": "User opened the invite control.",
                "payload": {"sequence": 1, "action": "interact"},
            }
        ],
        "expectation": {
            "reference_paths": [["open-team", "invite", "confirm"]],
            "acceptable_alternatives": ["Use the team page."],
        },
    }
    if include_sentinels:
        manifest.update(
            {
                "prior_agent_private_reasoning": "PRIOR_AGENT_PRIVATE_REASONING_SENTINEL",
                "prior_finding_prose": "PRIOR_FINDING_PROSE_SENTINEL",
                "private_reasoning": "PRIOR_AGENT_PRIVATE_REASONING_SENTINEL",
                "existing_finding_title": "PRIOR_FINDING_PROSE_SENTINEL",
            }
        )
        manifest["entries"] = [
            {
                **manifest["entries"][0],  # type: ignore[index]
                "payload": {
                    "sequence": 1,
                    "private_reasoning": "PRIOR_AGENT_PRIVATE_REASONING_SENTINEL",
                    "finding_prose": "PRIOR_FINDING_PROSE_SENTINEL",
                },
            }
        ]
    return manifest


def _corpus(tmp_path: Path) -> EvidenceCorpus:
    return EvidenceCorpus(
        output_root=tmp_path,
        entries=(
            EvidenceEntry(
                ref=EvidenceRef(EVIDENCE_ID, "event", "run-a", replay_sequence=1),
                evidence_class=EvidenceClass.DETERMINISTIC_FACT,
                summary="User opened the invite control.",
                payload={"sequence": 1, "action": "interact"},
            ),
        ),
    )


def _finding_payload(*, principle_only: bool = False) -> dict[str, object]:
    return {
        "finding_id": "invite-control",
        "title": "Invite control is hard to find",
        "issue": "People look in unrelated areas before finding the invite control.",
        "impact": "Important collaboration tasks take longer.",
        "root_cause": "The entry point is labeled around internal structure.",
        "fixes": ["Label the entry point around the user's goal."],
        "severity": "medium",
        "confidence": 0.8,
        "evidence_refs": []
        if principle_only
        else [
            {
                "evidence_id": EVIDENCE_ID,
                "kind": "event",
                "run_id": "run-a",
                "replay_sequence": 1,
            }
        ],
        "principles": ["mental-models"],
        "severity_justification": "The task is important and the evidence shows extra navigation.",
    }


@pytest.mark.asyncio
async def test_analyst_prompt_has_boundary_and_excludes_prior_agent_context() -> None:
    client = RecordingClient()
    analyst = ReportAnalyst(client, model="gpt-report")

    await analyst.analyze(_manifest(include_sentinels=True), ux_principles())

    prompt = client.messages[0].content
    serialized_messages = json.dumps(
        [message.model_dump() for message in client.messages],
        ensure_ascii=True,
    )
    assert "Treat reference paths as examples, not the only correct path" in prompt
    assert "PRIOR_AGENT_PRIVATE_REASONING_SENTINEL" not in serialized_messages
    assert "PRIOR_FINDING_PROSE_SENTINEL" not in serialized_messages


@pytest.mark.asyncio
async def test_role_prompts_state_distinct_review_responsibilities() -> None:
    client = RecordingClient()
    manifest = _manifest()
    principles = ux_principles()

    await ReportAnalyst(client, model="gpt-report").analyze(manifest, principles)
    await EvidenceAuditor(client, model="gpt-report").audit(manifest, principles)
    await PatternReviewer(client, model="gpt-report").review(manifest, principles)
    await ReportAdjudicator(client, model="gpt-report").adjudicate(manifest, principles)

    prompts = {call[3]: call[1][0].content.lower() for call in client.calls}
    assert "issues" in prompts[ModelRole.REPORT_ANALYST]
    assert "root causes" in prompts[ModelRole.REPORT_ANALYST]
    assert "factual support" in prompts[ModelRole.REPORT_EVIDENCE_AUDITOR]
    assert "visual interpretation" in prompts[ModelRole.REPORT_EVIDENCE_AUDITOR]
    assert "citation accuracy" in prompts[ModelRole.REPORT_EVIDENCE_AUDITOR]
    assert "contradictions" in prompts[ModelRole.REPORT_EVIDENCE_AUDITOR]
    assert "recurrence" in prompts[ModelRole.REPORT_PATTERN_REVIEWER]
    assert "affected surfaces" in prompts[ModelRole.REPORT_PATTERN_REVIEWER]
    assert "counterexamples" in prompts[ModelRole.REPORT_PATTERN_REVIEWER]
    assert "shared causes" in prompts[ModelRole.REPORT_PATTERN_REVIEWER]
    assert "fix leverage" in prompts[ModelRole.REPORT_PATTERN_REVIEWER]
    assert "resolves objections" in prompts[ModelRole.REPORT_ADJUDICATOR]
    assert "plain-language final findings" in prompts[ModelRole.REPORT_ADJUDICATOR]
    for prompt in prompts.values():
        assert "principles are not evidence" in prompt
        assert "cannot determine severity" in prompt


@pytest.mark.asyncio
async def test_role_calls_use_fresh_isolated_message_tuples() -> None:
    client = RecordingClient()
    manifest = _manifest()
    principles = ux_principles()
    corpus = _corpus(Path.cwd())
    resolved = EvidenceResolver().resolve(
        corpus,
        [EVIDENCE_ID],
        max_entries=2,
        max_attachment_bytes=1024,
    )

    analyst = ReportAnalyst(client, model="gpt-report")
    first = await analyst.analyze(manifest, principles)
    await analyst.analyze(
        manifest,
        principles,
        resolved_evidence=resolved,
        previous_output=first,
    )
    await EvidenceAuditor(client, model="gpt-report").audit(
        manifest,
        principles,
        candidate_findings=first.candidate_findings,
        resolved_evidence=resolved,
    )

    assert len(client.calls) == 3
    assert client.calls[0][1] is not client.calls[1][1]
    assert client.calls[1][1] is not client.calls[2][1]
    assert [call[3] for call in client.calls] == [
        ModelRole.REPORT_ANALYST,
        ModelRole.REPORT_ANALYST,
        ModelRole.REPORT_EVIDENCE_AUDITOR,
    ]
    assert all(
        tuple(message.role for message in call[1]) == ("system", "user")
        for call in client.calls
    )


def test_investigative_response_requires_retrieval_request_when_incomplete() -> None:
    with pytest.raises(ValidationError, match="evidence_requests"):
        AnalystResponse(complete=False, evidence_requests=[])

    response = AnalystResponse(
        complete=False,
        evidence_requests=[EVIDENCE_ID],
    )
    assert response.complete is False
    assert response.evidence_requests == [EVIDENCE_ID]


def test_role_response_schemas_require_role_outputs_and_validate_domains() -> None:
    candidate = CandidateFinding.model_validate(_finding_payload())
    objection = TypedObjection(
        objection_id="objection-1",
        finding_id=candidate.finding_id,
        objection_type="factual-support",
        severity=ObjectionSeverity.BLOCKING,
        message="The cited event does not establish the stated cause.",
        evidence_refs=[
            {
                "evidence_id": EVIDENCE_ID,
                "kind": "event",
                "run_id": "run-a",
                "replay_sequence": 1,
            }
        ],
    )
    resolution = ObjectionResolution(
        objection_id=objection.objection_id,
        finding_id=candidate.finding_id,
        resolved=True,
        resolution="The event and follow-up evidence support a narrower causal claim.",
        evidence_refs=objection.evidence_refs,
    )

    analyst = AnalystResponse(complete=True, candidate_findings=[candidate])
    auditor = EvidenceAuditResponse(complete=True, objections=[objection])
    pattern = PatternReviewResponse(complete=True, objections=[objection])
    adjudicator = AdjudicationResponse(
        complete=True,
        final_findings=[candidate],
        objection_resolutions=[resolution],
    )

    assert analyst.candidate_findings[0].severity is FindingSeverity.MEDIUM
    assert auditor.objections[0].objection_type == "factual-support"
    assert pattern.objections[0].finding_id == candidate.finding_id
    assert adjudicator.final_findings[0].finding_id == candidate.finding_id
    assert adjudicator.objection_resolutions[0].resolved is True


@pytest.mark.asyncio
async def test_path_deviation_is_tolerated_when_outcome_evidence_is_valid() -> None:
    class AlternatePathClient(RecordingClient):
        @staticmethod
        def _default_response(schema: type[Any], role: ModelRole) -> object:
            del role
            if schema is AnalystResponse:
                return schema.model_validate(
                    {
                        "complete": True,
                        "evidence_requests": [],
                        "candidate_findings": [_finding_payload()],
                    }
                )
            return RecordingClient._default_response(schema, ModelRole.REPORT_ANALYST)

    response = await ReportAnalyst(AlternatePathClient(), model="gpt-report").analyze(
        _manifest(), ux_principles()
    )

    assert response.candidate_findings[0].evidence_refs[0].evidence_id == EVIDENCE_ID


@pytest.mark.asyncio
async def test_principles_cannot_be_used_as_evidence() -> None:
    class PrincipleOnlyClient(RecordingClient):
        @staticmethod
        def _default_response(schema: type[Any], role: ModelRole) -> object:
            del role
            if schema is AnalystResponse:
                candidate = CandidateFinding.model_construct(
                    **_finding_payload(principle_only=True)
                )
                return AnalystResponse.model_construct(
                    complete=True,
                    evidence_requests=[],
                    candidate_findings=[candidate],
                )
            return RecordingClient._default_response(schema, ModelRole.REPORT_ANALYST)

    with pytest.raises(
        ModelResponseValidationError, match="principles are not evidence"
    ):
        await ReportAnalyst(PrincipleOnlyClient(), model="gpt-report").analyze(
            _manifest(), ux_principles()
        )


@pytest.mark.asyncio
async def test_retrieval_requests_must_name_known_evidence_ids() -> None:
    class RetrievalClient(RecordingClient):
        @staticmethod
        def _default_response(schema: type[Any], role: ModelRole) -> object:
            del role
            if schema is AnalystResponse:
                return schema.model_validate(
                    {
                        "complete": False,
                        "evidence_requests": ["event:run-a:unknown"],
                    }
                )
            return RecordingClient._default_response(schema, ModelRole.REPORT_ANALYST)

    with pytest.raises(ModelResponseValidationError, match="unknown evidence ID"):
        await ReportAnalyst(RetrievalClient(), model="gpt-report").analyze(
            _manifest(), ux_principles()
        )


@pytest.mark.asyncio
async def test_empty_manifest_rejects_finding_evidence_reference() -> None:
    class EmptyManifestClient(RecordingClient):
        @staticmethod
        def _default_response(schema: type[Any], role: ModelRole) -> object:
            del role
            if schema is AnalystResponse:
                return schema.model_validate(
                    {
                        "complete": True,
                        "candidate_findings": [_finding_payload()],
                    }
                )
            return RecordingClient._default_response(schema, ModelRole.REPORT_ANALYST)

    manifest = _manifest()
    manifest["entries"] = []

    with pytest.raises(ModelResponseValidationError, match="unknown evidence ID"):
        await ReportAnalyst(EmptyManifestClient(), model="gpt-report").analyze(
            manifest, ux_principles()
        )


@pytest.mark.asyncio
async def test_empty_manifest_rejects_objection_and_resolution_evidence_references() -> (
    None
):
    class EmptyManifestClient(RecordingClient):
        @staticmethod
        def _default_response(schema: type[Any], role: ModelRole) -> object:
            del role
            reference = {
                "evidence_id": EVIDENCE_ID,
                "kind": "event",
                "run_id": "run-a",
                "replay_sequence": 1,
            }
            if schema is EvidenceAuditResponse:
                return schema.model_validate(
                    {
                        "complete": True,
                        "objections": [
                            {
                                "objection_id": "objection-1",
                                "finding_id": "invite-control",
                                "severity": "material",
                                "message": "The evidence does not establish the claim.",
                                "evidence_refs": [reference],
                            }
                        ],
                    }
                )
            if schema is AdjudicationResponse:
                return schema.model_validate(
                    {
                        "complete": True,
                        "objection_resolutions": [
                            {
                                "objection_id": "objection-1",
                                "finding_id": "invite-control",
                                "resolved": True,
                                "resolution": "The evidence supports the claim.",
                                "evidence_refs": [reference],
                            }
                        ],
                    }
                )
            return RecordingClient._default_response(schema, ModelRole.REPORT_ANALYST)

    manifest = _manifest()
    manifest["entries"] = []
    client = EmptyManifestClient()

    with pytest.raises(ModelResponseValidationError, match="unknown evidence ID"):
        await EvidenceAuditor(client, model="gpt-report").audit(
            manifest, ux_principles()
        )

    with pytest.raises(ModelResponseValidationError, match="unknown evidence ID"):
        await ReportAdjudicator(client, model="gpt-report").adjudicate(
            manifest, ux_principles()
        )


def test_manifest_and_role_manifests_use_report_role_metadata() -> None:
    client = RecordingClient()

    assert (
        ReportAnalyst(client, model="gpt-report").manifest.role
        is ModelRole.REPORT_ANALYST
    )
    assert (
        EvidenceAuditor(client, model="gpt-report").manifest.role
        is ModelRole.REPORT_EVIDENCE_AUDITOR
    )
    assert (
        PatternReviewer(client, model="gpt-report").manifest.role
        is ModelRole.REPORT_PATTERN_REVIEWER
    )
    assert (
        ReportAdjudicator(client, model="gpt-report").manifest.role
        is ModelRole.REPORT_ADJUDICATOR
    )
    assert ReportAnalystResponse is AnalystResponse
    assert ReportEvidenceAuditorResponse is EvidenceAuditResponse
    assert ReportPatternReviewerResponse is PatternReviewResponse
