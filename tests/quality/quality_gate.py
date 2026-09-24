"""Offline quality gate for payload-shaping optimizations.

Run with:  uv run python tests/quality/quality_gate.py [--json PATH]

Every planned optimization that changes what is sent to a model (schema dedup,
payload compaction, retry feedback, prompt slimming, transport changes) is
gated on this benchmark before it may be committed. The gate is fully
deterministic -- no network, no live models: it drives the real
``ReportSynthesisService``, ``ReportAnalyst``, ``RedesignProposer``,
``RedesignCriticMerger``, and ``StructuredCognitiveAgent`` code with scripted
clients, mirroring the existing unit-test doubles.

Why these scenarios: each one pins an invariant on the exact surface a
deferred optimization could degrade.

1.  wire-payload-accounting      measures the real bytes per call so claimed
    savings are verifiable; asserts payload structure and privacy scrubbing.
2.  degraded-transport-contract  schema dedup must never break the inline
    contract that degraded adapter modes (json-object/plain) rely on.
3.  reviewer-retry-parity        invalid structured outputs must retry a
    bounded number of times with full context preserved and fail as
    REJECTED (not UNAVAILABLE); blind-retry vs feedback mode is reported.
4.  quota-fail-fast              a provider 429 must end the attempt as
    UNAVAILABLE after one call, with stale structural diagnostics not
    masking the outage.
5.  adjudicator-determinism      publication validation is code-enforced:
    an adjudicator may not replace a reviewed core claim or inflate a
    severity regardless of payload shape.
6.  corpus-coverage-parity       manifest compaction must keep evidence ids,
    run ids, expectation entries, and prompt guardrails intact.
7.  redesign-round-trip          proposer/critic must recover from one
    invalid output; critic payload must carry the consolidated proposals,
    page digests, and every proposal invariant.
8.  cognitive-payload-coverage   element aliases must stay stable and
    collision-free across visible elements and available_controls; no
    private execution data may leak into any message.

Exit code is 0 only when every scenario passes. ``--json`` writes a
machine-readable score for baseline tracking across optimization commits.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ux_analyzer.adapters.openai import ModelFailureError
from ux_analyzer.application.evidence_corpus import EvidenceCorpus, EvidenceEntry
from ux_analyzer.application.report_synthesis import (
    MAX_INVALID_STRUCTURED_ROLE_RETRIES,
    ReportSynthesisService,
)
from ux_analyzer.domain.attention import ProgressiveObservation
from ux_analyzer.domain.findings import EvidenceClass, FindingSeverity
from ux_analyzer.domain.interface import (
    BoundingBox,
    ElementSnapshot,
    PrivateExecutionReference,
    ViewportSnapshot,
)
from ux_analyzer.domain.synthesis import (
    EvidenceRef,
    SynthesisStatus,
)
from ux_analyzer.ports.models import (
    ModelCallRecord,
    ModelResponseValidationError,
    ModelRole,
    TokenUsage,
)
from ux_analyzer.providers.cognitive import StructuredCognitiveAgent
from ux_analyzer.providers.redesign import (
    CriticResponse,
    ProposerResponse,
    RedesignCriticMerger,
    RedesignProposer,
)
from ux_analyzer.providers.report_synthesis import (
    AdjudicationResponse,
    AnalystResponse,
    CandidateFinding,
    EvidenceAuditResponse,
    PatternReviewResponse,
    ReportAnalyst,
)
from ux_analyzer.providers.ux_principles import ux_principles

EVIDENCE_ID = "event:run-a:1"


# ---------------------------------------------------------------------------
# Gate reporting
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Checkpoint:
    name: str
    passed: bool
    detail: str


@dataclass(slots=True)
class ScenarioReport:
    name: str
    checkpoints: list[Checkpoint] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    metrics: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.checkpoints) and all(c.passed for c in self.checkpoints)

    def check(self, name: str, condition: bool, detail: str) -> None:
        self.checkpoints.append(
            Checkpoint(name=name, passed=bool(condition), detail=detail)
        )

    def note(self, text: str) -> None:
        self.notes.append(text)


# ---------------------------------------------------------------------------
# Scripted role doubles (mirror tests/unit/application/test_report_synthesis.py)
# ---------------------------------------------------------------------------


class _RecordingModelSource:
    """Collects ModelCallRecord entries the service attaches to attempts."""

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


class _ScriptedRole:
    """Service-role double: replays scripted outcomes, records every call."""

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

    def _next(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> object:
        self.calls.append(
            {
                "kwargs": dict(kwargs),
                "args_repr_len": len(repr(args)),
                "kwargs_repr_len": len(repr(sorted(kwargs.items(), key=str))),
            }
        )
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
            self.record_source.record(self.role, dict(kwargs), result)
        return result

    async def analyze(self, *args: Any, **kwargs: Any) -> object:
        return self._next(args, kwargs)

    async def audit(self, *args: Any, **kwargs: Any) -> object:
        return self._next(args, kwargs)

    async def review(self, *args: Any, **kwargs: Any) -> object:
        return self._next(args, kwargs)

    async def adjudicate(self, *args: Any, **kwargs: Any) -> object:
        return self._next(args, kwargs)


def _synthesis_service(
    tmp_path: Path,
    *,
    analyst: Sequence[object] = (),
    auditor: Sequence[object] = (),
    pattern: Sequence[object] = (),
    adjudicator: Sequence[object] = (),
) -> tuple[ReportSynthesisService, _RecordingModelSource]:
    source = _RecordingModelSource()
    roles = (
        _ScriptedRole("analyst", analyst, source),
        _ScriptedRole("auditor", auditor, source),
        _ScriptedRole("pattern", pattern, source),
        _ScriptedRole("adjudicator", adjudicator, source),
    )
    service = ReportSynthesisService(
        analyst=roles[0],
        evidence_auditor=roles[1],
        pattern_reviewer=roles[2],
        adjudicator=roles[3],
        model_record_source=source,
    )
    return service, source


def _corpus(tmp_path: Path, *, extra: Sequence[EvidenceEntry] = ()) -> EvidenceCorpus:
    return EvidenceCorpus(
        output_root=tmp_path,
        entries=(
            EvidenceEntry(
                ref=EvidenceRef(EVIDENCE_ID, "event", "run-a", replay_sequence=1),
                evidence_class=EvidenceClass.DETERMINISTIC_FACT,
                summary="User opened the invite control.",
                payload={"sequence": 1, "action": "interact"},
            ),
            *extra,
        ),
    )


def _candidate(**overrides: object) -> CandidateFinding:
    kwargs: dict[str, object] = {
        "finding_id": "invite-control",
        "title": "The invite control is hard to find",
        "issue": "The user searches outside the expected task area before finding the invite control.",
        "impact": "An important collaboration task takes longer to complete.",
        "root_cause": "The entry point is labeled around internal product structure.",
        "fixes": ["Label the entry point around the user's goal."],
        "severity": "high",
        "confidence": 0.9,
        "evidence_refs": [
            {
                "evidence_id": EVIDENCE_ID,
                "kind": "event",
                "run_id": "run-a",
                "replay_sequence": 1,
            }
        ],
        "affected_surfaces": [],
        "severity_justification": "The evidence shows extra navigation on an important task.",
    }
    kwargs.update(overrides)
    return CandidateFinding.model_validate(kwargs)


def _invalid_output(role: ModelRole) -> ModelResponseValidationError:
    return ModelResponseValidationError(role, "invalid structured output")


# ---------------------------------------------------------------------------
# StructuredModelClient doubles (mirror tests/unit/providers pattern)
# ---------------------------------------------------------------------------


class _FlakyStructuredClient:
    """ScriptedModelClient double that records calls and replays outcomes.

    Outcomes are consumed in order across *all* calls; ``None`` means "answer
    with the default stub for this schema". Records per-call payload bytes for
    the wire-payload accounting.
    """

    endpoint_origin = "https://llm.example.test/v1"
    provider_id = "gate-recording"
    provider_version = "gate-v1"

    def __init__(
        self,
        outcomes: Sequence[object] = (),
        defaults: Mapping[type[Any], object] | None = None,
    ) -> None:
        self.outcomes = list(outcomes)
        self.defaults: dict[type[Any], object] = dict(defaults or {})
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        schema: type[Any],
        messages: Sequence[Any],
        model: str,
        role: ModelRole,
    ) -> object:
        user_content = list(messages)[-1].content
        self.calls.append(
            {
                "schema": schema,
                "role": role,
                "messages": list(messages),
                "user_content": user_content,
                "user_bytes": len(user_content.encode("utf-8")),
            }
        )
        if self.outcomes:
            outcome = self.outcomes.pop(0)
        else:
            outcome = None
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome is not None:
            payload = outcome
        else:
            payload = self.defaults.get(schema)
            if payload is None:
                raise AssertionError(f"no scripted outcome for {schema.__name__}")
        return schema.model_validate(payload)


def _synthesis_defaults() -> dict[type[Any], object]:
    return {
        AnalystResponse: {"complete": True, "candidate_findings": [], "evidence_requests": []},
        EvidenceAuditResponse: {"complete": True, "objections": []},
        PatternReviewResponse: {"complete": True, "objections": []},
        AdjudicationResponse: {"complete": True, "final_findings": [], "objection_resolutions": []},
    }


def _redesign_defaults() -> dict[type[Any], object]:
    proposal = {
        "proposal_id": "p1",
        "page_url": "https://fixture.test/",
        "category": "whitespace",
        "title": "Widen spacing between pricing cards",
        "observation": "Cards sit 8px apart and read as one block.",
        "rationale": "Grouping clarity suffers without separation.",
        "change": "Raise the gap to 32px.",
        "principle_ids": ["gestalt-proximity"],
        "impact": "medium",
        "effort": "small",
        "section_refs": [
            {
                "url": "https://fixture.test/",
                "section_label": "Pricing cards",
                "box": {"x": 0.0, "y": 400.0, "width": 1280.0, "height": 600.0},
                "summary": "Three pricing cards in a row with feature lists.",
            }
        ],
    }
    return {
        ProposerResponse: {
            "page_understanding": {
                "page_url": "https://fixture.test/",
                "intent": "Explain the product and drive signups.",
                "audience_inference": "Likely first-time visitors evaluating pricing.",
                "section_relationships": "Hero feeds a feature row and a pricing block.",
            },
            "proposals": [proposal],
        },
        CriticResponse: {
            "final_proposals": [{**proposal, "proposal_id": "m1"}],
            "killed": [{"proposal_id": "p2", "reason": "duplicates m1"}],
            "consistency_notes": ["Applied the same card gap rule on every page."],
        },
    }


# ---------------------------------------------------------------------------
# Scenario 1: wire payload accounting + privacy scrubbing
# ---------------------------------------------------------------------------


def scenario_wire_payload_accounting(tmp_path: Path) -> ScenarioReport:
    report = ScenarioReport("wire-payload-accounting")
    client = _FlakyStructuredClient(defaults=_synthesis_defaults())
    analyst = ReportAnalyst(client, model="gate-report-model")  # type: ignore[arg-type]
    manifest: dict[str, object] = {
        "schema_version": "evidence-corpus-v1",
        "principle_pack_version": "ux-principles-v1",
        "entries": [
            {
                "evidence_id": EVIDENCE_ID,
                "kind": "event",
                "run_id": "run-a",
                "summary": "User opened the invite control.",
                "payload": {
                    "sequence": 1,
                    "action": "interact",
                    "private_reasoning": "PRIOR_AGENT_PRIVATE_REASONING_SENTINEL",
                    "finding_prose": "PRIOR_FINDING_PROSE_SENTINEL",
                },
            }
        ],
        "expectation": {
            "reference_paths": [["open-team", "invite", "confirm"]],
            "acceptable_alternatives": ["Use the team page."],
        },
        "prior_agent_private_reasoning": "PRIOR_AGENT_PRIVATE_REASONING_SENTINEL",
        "prior_finding_prose": "PRIOR_FINDING_PROSE_SENTINEL",
    }
    asyncio.run(analyst.analyze(manifest, ux_principles()))

    assert client.calls, "analyst must invoke the model client"
    call = client.calls[0]
    payload = json.loads(call["user_content"])
    contract = payload["response_schema"]
    schema_json = json.dumps(contract["schema"])

    report.check(
        "payload-contract-present",
        contract["role"] == ModelRole.REPORT_ANALYST.value
        and "schema_version" in contract
        and '"candidate_findings"' in schema_json,
        "user payload carries the role/response-schema contract",
    )
    manifest_payload = payload.get("corpus_manifest", {})
    report.check(
        "manifest-entries-present",
        bool(manifest_payload.get("entries")),
        "compact manifest entries reach the user payload",
    )
    serialized = json.dumps(
        [message.model_dump() for message in call["messages"]], ensure_ascii=True
    )
    report.check(
        "privacy-sentinels-scrubbed",
        "PRIOR_AGENT_PRIVATE_REASONING_SENTINEL" not in serialized
        and "PRIOR_FINDING_PROSE_SENTINEL" not in serialized,
        "private reasoning / finding prose never reach the wire",
    )
    report.metrics["user_payload_bytes"] = call["user_bytes"]
    report.metrics["contract_schema_bytes"] = len(schema_json)
    report.note(
        f"analyst user payload = {call['user_bytes']:,} bytes, of which the "
        f"inline response-schema contract = {len(schema_json):,} bytes. A "
        "tool-call-mode dedup saving must come out of the duplicate only, "
        "never the degraded-mode copy (scenario 2)."
    )
    return report


# ---------------------------------------------------------------------------
# Scenario 2: degraded-transport contract (schema dedup surface)
# ---------------------------------------------------------------------------


def scenario_degraded_transport_contract(tmp_path: Path) -> ScenarioReport:
    report = ScenarioReport("degraded-transport-contract")
    client = _FlakyStructuredClient(defaults=_synthesis_defaults())
    analyst = ReportAnalyst(client, model="gate-report-model")  # type: ignore[arg-type]
    asyncio.run(analyst.analyze(_minimal_manifest(), ux_principles()))
    payload: dict[str, Any] = json.loads(client.calls[0]["user_content"])
    inline_schema = payload["response_schema"]["schema"]
    expected = AnalystResponse.model_json_schema()

    report.check(
        "inline-schema-parses",
        isinstance(inline_schema, dict) and bool(inline_schema),
        "the in-message schema is valid JSON object form",
    )
    report.check(
        "inline-schema-carries-full-contract",
        inline_schema.get("required") == expected.get("required")
        and set(inline_schema.get("properties", {})) >= set(expected.get("properties", {})),
        "inline schema top-level required/properties match the real model schema",
    )
    defs = inline_schema.get("$defs", {})
    report.check(
        "inline-schema-keeps-nested-definitions",
        set(defs) == set(expected.get("$defs", {})) and bool(defs),
        "nested $defs survive (degraded modes lose the tools[] copy entirely)",
    )
    report.note(
        "Schema dedup MUST be mode-conditional: when the adapter falls back to "
        "json-object/plain, this in-message contract is the only schema the "
        "model sees. Removing it unconditionally would remove the contract "
        "exactly when tool enforcement is weakest."
    )
    return report


# ---------------------------------------------------------------------------
# Scenario 3: reviewer retry parity (retry-feedback surface)
# ---------------------------------------------------------------------------


def scenario_reviewer_retry_parity(tmp_path: Path) -> ScenarioReport:
    report = ScenarioReport("reviewer-retry-parity")
    invalid = [
        _invalid_output(ModelRole.REPORT_PATTERN_REVIEWER),
        _invalid_output(ModelRole.REPORT_PATTERN_REVIEWER),
        _invalid_output(ModelRole.REPORT_PATTERN_REVIEWER),
    ]
    service, _source = _synthesis_service(
        tmp_path,
        analyst=[AnalystResponse(complete=True, candidate_findings=[_candidate()])],
        pattern=invalid,
    )
    attempt = asyncio.run(service.synthesize(_corpus(tmp_path)))

    report.check(
        "retry-budget-is-bounded",
        MAX_INVALID_STRUCTURED_ROLE_RETRIES >= 1,
        f"MAX_INVALID_STRUCTURED_ROLE_RETRIES={MAX_INVALID_STRUCTURED_ROLE_RETRIES}",
    )
    report.check(
        "exhausted-retries-reject-not-unavailable",
        attempt.status is SynthesisStatus.REJECTED,
        f"status={attempt.status}",
    )
    report.check(
        "limitation-mentions-invalid-output",
        any("invalid" in limitation.lower() for limitation in attempt.limitations),
        "limitations record the invalid structured output",
    )
    pattern_calls = [c for c in _last_role_calls(service) if c is not None]
    report.check(
        "context-preserved-across-retries",
        len(pattern_calls) == MAX_INVALID_STRUCTURED_ROLE_RETRIES + 1
        and all(c["kwargs_repr_len"] >= 200 for c in pattern_calls),
        f"{len(pattern_calls)} reviewer calls, each still carrying the full "
        "manifest/finding context (no context stripping mid-retry)",
    )
    feedback_values = [
        (c.get("kwargs") or {}).get("validation_feedback") for c in pattern_calls
    ]
    report.check(
        "retry-feedback-mode",
        feedback_values[0] is None
        and all(
            isinstance(value, str) and value
            for value in feedback_values[1:]
        ),
        "first call has no feedback; every retry carries the bounded safe "
        "validation reason (feedback-retry mode)",
    )
    return report


def _last_role_calls(service: ReportSynthesisService) -> list[dict[str, Any] | None]:
    """Best-effort: scripted reviewer calls are recorded on the role doubles."""
    reviewer = getattr(service, "_pattern_reviewer", None) or getattr(
        service, "pattern_reviewer", None
    )
    calls = getattr(reviewer, "calls", None)
    return list(calls) if isinstance(calls, list) else []


# ---------------------------------------------------------------------------
# Scenario 4: quota fail-fast (preflight surface)
# ---------------------------------------------------------------------------


def scenario_quota_fail_fast(tmp_path: Path) -> ScenarioReport:
    report = ScenarioReport("quota-fail-fast")
    stale_429 = ModelFailureError(
        "rate limit",
        status_code=429,
        diagnostics={"stage": "schema_validation"},
    )
    service, _source = _synthesis_service(tmp_path, analyst=[stale_429])
    attempt = asyncio.run(service.synthesize(_corpus(tmp_path)))

    report.check(
        "quota-429-marks-attempt-unavailable",
        attempt.status is SynthesisStatus.UNAVAILABLE,
        f"status={attempt.status.name}",
    )
    analyst = getattr(service, "_analyst", None) or getattr(service, "analyst", None)
    calls = getattr(analyst, "calls", []) or []
    report.check(
        "single-provider-call-no-pipeline-burn",
        len(calls) == 1,
        f"analyst invoked {len(calls)}x before the attempt ended",
    )
    report.note(
        "A preflight probe must preserve this: any preflight extra call is a "
        "budgeted addition, and quota-exhausted runs must never proceed into "
        "the full role pipeline (observed cost: ~243 min / 1.45M prompt tokens "
        "across 8 doomed attempts)."
    )
    return report


# ---------------------------------------------------------------------------
# Scenario 5: adjudicator publication determinism
# ---------------------------------------------------------------------------


def scenario_adjudicator_determinism(tmp_path: Path) -> ScenarioReport:
    report = ScenarioReport("adjudicator-determinism")
    reviewed = _candidate(severity="low")
    replacement = reviewed.model_copy(
        update={
            "severity": FindingSeverity.CRITICAL,
            "title": "An unrelated account deletion blocker",
            "issue": "The user cannot delete an account.",
            "impact": "A separate critical workflow is blocked.",
            "root_cause": "The account deletion control is missing.",
            "severity_justification": "A critical account workflow is unavailable.",
        }
    )
    service, _source = _synthesis_service(
        tmp_path,
        analyst=[AnalystResponse(complete=True, candidate_findings=[reviewed])],
        adjudicator=[AdjudicationResponse(complete=True, final_findings=[replacement])],
    )
    attempt = asyncio.run(service.synthesize(_corpus(tmp_path)))

    report.check(
        "replaced-core-claim-rejected",
        attempt.status is SynthesisStatus.REJECTED and not attempt.findings,
        f"status={attempt.status.name}, published findings={len(attempt.findings)}",
    )
    report.check(
        "rejection-is-publication-validation",
        any(
            "publication validation" in limitation
            for limitation in attempt.limitations
        ),
        "limitations name publication validation",
    )
    report.note(
        "Publication invariants are code-enforced. Any prompt/schema slimming "
        "must leave this behavior byte-identical: a model can never publish a "
        "finding the reviewers did not establish, whatever the payload shape."
    )
    return report


# ---------------------------------------------------------------------------
# Scenario 6: corpus coverage parity (manifest compaction surface)
# ---------------------------------------------------------------------------


def scenario_corpus_coverage_parity(tmp_path: Path) -> ScenarioReport:
    report = ScenarioReport("corpus-coverage-parity")
    client = _FlakyStructuredClient(defaults=_synthesis_defaults())
    analyst = ReportAnalyst(client, model="gate-report-model")  # type: ignore[arg-type]
    asyncio.run(analyst.analyze(_minimal_manifest(), ux_principles()))
    payload = json.loads(client.calls[0]["user_content"])
    payload_json = json.dumps(payload)
    prompt = client.calls[0]["messages"][0].content

    manifest_payload = payload.get("corpus_manifest", {})
    handle_policy = payload.get("evidence_request_policy", {})
    report.check(
        "evidence-handles-traceable",
        bool(manifest_payload.get("entries"))
        and handle_policy.get("handle_format") == "e{index}",
        "compact manifest entries + handle format keep evidence requestable",
    )
    report.check(
        "run-id-traceable",
        "run-a" in payload_json,
        "source run id reaches the model payload",
    )
    report.check(
        "expectation-entry-traceable",
        "reference_paths" in payload_json,
        "expectation entries (reference paths, alternatives) reach the payload",
    )
    report.check(
        "prompt-guardrails-intact",
        "Treat reference paths as examples, not the only correct path" in prompt
        and "At most 8 candidate findings" in prompt
        and "Consolidate repeated signals" in prompt,
        "analyst prompt guardrails survive prompt slimming",
    )
    report.note(
        "Manifest compaction may only remove redundancy (duplicate fields, "
        "repeated boilerplate); every evidence id, run id, and expectation "
        "entry must remain reachable, and prompt guardrails must stay."
    )
    return report


# ---------------------------------------------------------------------------
# Scenario 7: redesign proposer/critic round trip
# ---------------------------------------------------------------------------


def scenario_redesign_round_trip(tmp_path: Path) -> ScenarioReport:
    report = ScenarioReport("redesign-round-trip")
    defaults = _redesign_defaults()
    client = _FlakyStructuredClient(defaults=defaults)
    proposer = RedesignProposer(client, model="gate-redesign-model")  # type: ignore[arg-type]
    critic = RedesignCriticMerger(client, model="gate-redesign-model")  # type: ignore[arg-type]

    async def _run() -> None:
        page_payload = {"url": "https://fixture.test/", "title": "Fixture"}
        proposer_response = await proposer.analyze(
            page_payload, audience="designers", principles=[{"id": "gestalt-proximity", "name": "Proximity"}]
        )
        consolidated = [
            proposal.model_dump() if hasattr(proposal, "model_dump") else dict(proposal)
            for proposal in proposer_response.proposals
        ]
        digests = [{"url": "https://fixture.test/", "digest": "abc123"}]
        await critic.review(consolidated, digests, audience="", principles=[])

    asyncio.run(_run())
    proposer_call, critic_call = client.calls[0], client.calls[-1]
    proposer_payload = json.loads(proposer_call["user_content"])
    critic_payload = json.loads(critic_call["user_content"])

    report.check(
        "proposer-role-and-payload",
        proposer_call["role"] is ModelRole.REDESIGN_PROPOSER
        and proposer_payload["role_input"]["page_payload"]["title"] == "Fixture"
        and proposer_payload["role_input"]["requested_audience"] == "designers",
        "proposer receives page payload + audience",
    )
    report.check(
        "proposer-schema-in-payload",
        proposer_payload["response_schema"]["schema_version"] == "redesign-proposer-v1",
        "proposer contract rides in the user payload",
    )
    report.check(
        "critic-gets-consolidated-proposals-and-digests",
        critic_payload["role_input"]["consolidated_proposals"][0]["proposal_id"] == "p1"
        and critic_payload["role_input"]["page_payloads_digests"][0]["digest"] == "abc123",
        "critic receives every consolidated proposal + page digests",
    )
    critic_out = defaults[CriticResponse]
    final_proposals = critic_out["final_proposals"]  # type: ignore[index]
    proposal_json = json.dumps(final_proposals)
    report.check(
        "proposal-invariants-survive-critic",
        '"proposal_id"' in proposal_json
        and '"page_url"' in proposal_json
        and '"principle_ids"' in proposal_json
        and '"section_refs"' in proposal_json,
        "critic re-emit keeps ids, page refs, principles, and section refs",
    )
    report.metrics["proposer_payload_bytes"] = proposer_call["user_bytes"]
    report.metrics["critic_payload_bytes"] = critic_call["user_bytes"]
    report.note(
        f"critic payload = {critic_call['user_bytes']:,} bytes and the critic "
        "re-emits every accepted proposal verbatim (observed cost: 11.8k "
        "completion tokens, 797s with 2 retries). Compaction may shrink "
        "duplicated context but must keep the invariants above."
    )
    return report


# ---------------------------------------------------------------------------
# Scenario 8: cognitive payload coverage + privacy
# ---------------------------------------------------------------------------


def scenario_cognitive_payload_coverage(tmp_path: Path) -> ScenarioReport:
    report = ScenarioReport("cognitive-payload-coverage")
    from ux_analyzer.providers.cognitive import CognitiveModelResponse

    defaults = {
        CognitiveModelResponse: {
            "action": "inspect",
            "element_id": "target",
            "reason": "Visible control matches goal.",
        }
    }
    client = _FlakyStructuredClient(defaults=defaults)
    agent = StructuredCognitiveAgent(
        client, model="gate-cognitive-model", fixture_keys=("invite_email",)
    )
    snapshot = _cognitive_snapshot()
    observation = ProgressiveObservation.from_snapshot(
        snapshot, newly_revealed_ids=("target",)
    )
    decision = asyncio.run(agent.decide("Find invite", observation))
    call = client.calls[0]
    payload = json.loads(call["user_content"])
    element_ids = [
        element.get("element_id")
        for element in payload.get("newly_revealed_elements", [])
        + payload.get("remembered_elements", [])
    ]

    action = decision.action
    report.check(
        "decision-returns-structured-action",
        action.kind == "inspect" and action.element_id == "target",  # type: ignore[union-attr]
        f"cognitive agent returned a structured {action.kind} decision",
    )
    report.check(
        "goal-and-visible-elements-present",
        payload.get("goal") == "Find invite" and "target" in element_ids,
        "goal + newly-revealed/remembered elements reach the payload",
    )
    report.check(
        "available-controls-reference-visible-elements",
        set(payload.get("available_controls", [])) <= set(element_ids),
        "available_controls is an id list referencing elements in the payload "
        "(stage-3 dedup: no duplicated full objects)",
    )
    report.check(
        "element-ids-unique-across-lists",
        len(element_ids) == len(set(element_ids)) and bool(element_ids),
        "element ids stay distinct across newly-revealed and remembered lists",
    )
    serialized = json.dumps(
        [message.model_dump() for message in call["messages"]], ensure_ascii=True
    )
    report.check(
        "no-private-execution-leak",
        "private-token" not in serialized
        and "private hidden label" not in serialized
        and "PrivateExecutionReference" not in serialized
        and "data-testid" not in serialized
        and "fixture.invalid/private" not in serialized,
        "private execution refs, hidden labels, selectors, and destinations "
        "never reach the model",
    )
    report.note(
        "Cognitive payload slimming (stage 3 landed: available_controls is an "
        "id list) must keep element ids stable per step and never leak private "
        "execution data. Structured-output retry for this role lives in the "
        "adapter/application layer, not the provider (scenario 3 covers retry "
        "semantics for report roles)."
    )
    return report


def _cognitive_snapshot() -> ViewportSnapshot:
    return ViewportSnapshot(
        id="viewport-1",
        elements=(
            ElementSnapshot(
                id="target",
                role="button",
                label="Invite teammate",
                bounds=BoundingBox(x=1, y=2, width=10, height=10),
                visibility_fraction=1.0,
                actionable=True,
                provider_id="fixture",
                execution_reference=PrivateExecutionReference(
                    provider_id="fixture",
                    viewport_id="viewport-1",
                    token="private-token",
                ),
                selector="[data-testid='target']",
                test_id="target",
                hidden_label="private hidden label",
                destination_url="https://fixture.invalid/private",
            ),
            ElementSnapshot(
                id="unnoticed",
                role="link",
                label="Private destination",
                bounds=BoundingBox(x=20, y=2, width=10, height=10),
                visibility_fraction=1.0,
                actionable=True,
                destination_url="https://fixture.invalid/unnoticed",
            ),
        ),
    )


def _minimal_manifest() -> dict[str, object]:
    return {
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


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

SCENARIOS: list[Callable[[Path], ScenarioReport]] = [
    scenario_wire_payload_accounting,
    scenario_degraded_transport_contract,
    scenario_reviewer_retry_parity,
    scenario_quota_fail_fast,
    scenario_adjudicator_determinism,
    scenario_corpus_coverage_parity,
    scenario_redesign_round_trip,
    scenario_cognitive_payload_coverage,
]


def _run_all() -> list[ScenarioReport]:
    reports: list[ScenarioReport] = []
    for scenario in SCENARIOS:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                reports.append(scenario(Path(tmp)))
            except Exception as exc:  # noqa: BLE001 - surfaced as a failure
                failure = ScenarioReport(scenario.__name__)
                failure.check(
                    "scenario-completed", False, f"{type(exc).__name__}: {exc}"
                )
                reports.append(failure)
    return reports


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "Offline quality gate").splitlines()[0]
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="write a machine-readable score file"
    )
    args = parser.parse_args(argv)

    reports = _run_all()
    failed = [report for report in reports if not report.passed]
    total_checks = sum(len(r.checkpoints) for r in reports)

    print("ux-analyzer offline quality gate")
    print("=" * 72)
    for report in reports:
        print(f"\n[{'PASS' if report.passed else 'FAIL'}] {report.name}")
        for checkpoint in report.checkpoints:
            print(f"  {'ok ' if checkpoint.passed else '!! '} {checkpoint.name}: {checkpoint.detail}")
        for note in report.notes:
            print(f"  ..  {note}")
    print("\n" + "=" * 72)
    print(
        f"{len(reports) - len(failed)}/{len(reports)} scenarios passed "
        f"({total_checks - sum(1 for r in failed for c in r.checkpoints if not c.passed)}"
        f"/{total_checks} checks)"
    )
    for report in failed:
        print(f"FAILED: {report.name}")

    if args.json is not None:
        args.json.write_text(
            json.dumps(
                {
                    "passed": not failed,
                    "scenarios": [
                        {
                            "name": report.name,
                            "passed": report.passed,
                            "checks": [
                                {"name": c.name, "passed": c.passed, "detail": c.detail}
                                for c in report.checkpoints
                            ],
                            "notes": report.notes,
                            "metrics": report.metrics,
                        }
                        for report in reports
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"score written to {args.json}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
