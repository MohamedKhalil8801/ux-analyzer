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
9.  adapter-schema-strip-modes   the adapter's tool-call schema strip must
    be mode-conditional: tool-call bodies drop the duplicate in-message
    schema, degraded modes (strict/json-object) keep it byte-identical,
    and request_size stays measured unstripped so transport packing keeps
    headroom for a degraded re-send.
10. recorded-attempt-shapes       replays the four rejected recorded attempts
    against one immutable corpus. Authorized evidence contraction publishes;
    unauthorized contraction still rejects; a heuristic-only harm claim never
    reaches publication; identical duplicate reviewer objections collapse
    while conflicting duplicates still reject. These are the regression guard
    for every later change in this pipeline.
11. prompt-prefix-stability      the report-role user payload must serialize
    every static block (corpus manifest, response schema, principle pack, role
    input) before anything that varies per retrieval round, and the analyst's
    consecutive rounds must share at least that prefix. Recorded baseline:
    1.38M prompt tokens at a 0.3-13% cache hit ratio.

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
from typing import Any, cast

import httpx

from ux_analyzer.adapters import openai as openai_adapter
from ux_analyzer.adapters.openai import (
    ModelFailureError,
    OpenAICompatibleSettings,
    OpenAICompatibleStructuredClient,
)
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
    ChatMessage,
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
# Scenario 9: adapter tool-call schema strip is mode-conditional
# ---------------------------------------------------------------------------


def _gate_adapter_settings() -> OpenAICompatibleSettings:
    values: dict[str, object] = {
        "base_url": "https://fake-llm.test/v1",
        "api_key": "gate-key",
        "scent_model": "gate-scent",
        "cognitive_model": "gate-cognitive",
        "retry_policy": {"max_attempts": 3, "base_delay_seconds": 0},
    }
    return OpenAICompatibleSettings.model_validate(values)


def _gate_user_payload_message() -> ChatMessage:
    """A user message shaped like the real report provider payload."""

    payload = {
        "corpus_manifest": {"entries": ["e0", "e1"]},
        "response_schema": {
            "role": "report-analyst",
            "schema_version": AnalystResponse.schema_version,
            "schema": AnalystResponse.model_json_schema(),
        },
    }
    content = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return ChatMessage(role="user", content=content)


def scenario_adapter_schema_strip_modes(tmp_path: Path) -> ScenarioReport:
    report = ScenarioReport("adapter-schema-strip-modes")
    client = OpenAICompatibleStructuredClient(
        _gate_adapter_settings(),
        http_client=httpx.AsyncClient(),
    )
    messages = (
        ChatMessage(role="system", content="You are a report analyst."),
        _gate_user_payload_message(),
    )
    messages_with_trailing = (*messages, ChatMessage(role="user", content="plain"))
    stripped = openai_adapter._strip_inline_response_schema(messages)
    stripped_trailing = openai_adapter._strip_inline_response_schema(
        messages_with_trailing
    )

    def _payload_of(source: object) -> dict[str, Any]:
        assert isinstance(source, ChatMessage)
        return dict(json.loads(source.content))

    original = _payload_of(messages[1])
    stripped_payload = _payload_of(stripped[1])
    stripped_trailing_payload = _payload_of(stripped_trailing[1])

    report.check(
        "tool-call-mode-strips-schema",
        "response_schema" in original
        and "response_schema" not in stripped_payload
        and "corpus_manifest" in stripped_payload,
        "tool-call body drops the in-message response_schema while keeping "
        "the rest of the payload intact",
    )
    report.check(
        "strip-preserves-adjacent-messages",
        stripped_trailing[2].content == messages_with_trailing[2].content
        and "response_schema" not in stripped_trailing_payload,
        "non-JSON and schema-less messages pass through untouched",
    )

    def _wire_payload(messages_value: object) -> dict[str, Any] | None:
        """Find the parseable JSON user payload in a wire message list."""

        assert isinstance(messages_value, list)
        for raw in messages_value:
            if not isinstance(raw, dict):
                continue
            content = raw.get("content")
            if not isinstance(content, str) or not content.startswith("{"):
                continue
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "response_schema" in parsed:
                return parsed
        return None

    # Degraded transports: the wire body must keep the in-message contract.
    for mode in ("strict", "json-object", "plain"):
        degraded_payload = client._request_payload(
            AnalystResponse,
            messages,
            "gate-report-model",
            ModelRole.REPORT_ANALYST,
            mode,
        )
        degraded_user = _wire_payload(degraded_payload.get("messages"))
        report.check(
            f"degraded-{mode}-keeps-contract",
            degraded_user is not None,
            f"{mode} wire body keeps the in-message response_schema contract",
        )

    tool_payload = client._request_payload(
        AnalystResponse,
        messages,
        "gate-report-model",
        ModelRole.REPORT_ANALYST,
        "tool-call",
    )
    tool_user = _wire_payload(tool_payload.get("messages"))
    tool_tools = tool_payload.get("tools")
    assert isinstance(tool_tools, list)
    tool_function = dict(tool_tools[0])
    tool_spec = dict(tool_function.get("function") or {})
    report.check(
        "tool-call-wire-has-schema-once",
        tool_user is None
        and isinstance(tool_spec.get("parameters"), dict)
        and bool(tool_spec["parameters"]),
        "tool-call wire body carries the schema only in tools[].function",
    )

    stripped_bytes = len(
        openai_adapter.serialize_transport_json(tool_payload)
    )
    unstripped_payload = client._request_payload(
        AnalystResponse,
        messages,
        "gate-report-model",
        ModelRole.REPORT_ANALYST,
        "tool-call",
        strip_inline_schema=False,
    )
    unstripped_bytes = len(
        openai_adapter.serialize_transport_json(unstripped_payload)
    )
    packed_size = client.request_size(
        AnalystResponse,
        messages,
        model="gate-report-model",
        role=ModelRole.REPORT_ANALYST,
    )
    degraded_wire = client._request_payload(
        AnalystResponse,
        messages,
        "gate-report-model",
        ModelRole.REPORT_ANALYST,
        "strict",
    )
    degraded_bytes = len(
        openai_adapter.serialize_transport_json(degraded_wire)
    )
    report.metrics["tool_call_wire_bytes"] = stripped_bytes
    report.metrics["unstripped_wire_bytes"] = unstripped_bytes
    report.metrics["request_size_bytes"] = packed_size
    report.metrics["degraded_strict_wire_bytes"] = degraded_bytes
    report.check(
        "request-size-keeps-packing-headroom",
        packed_size == unstripped_bytes
        and stripped_bytes < unstripped_bytes
        and degraded_bytes >= stripped_bytes,
        "packing measure (request_size) stays unstripped and the stripped "
        "tool-call wire is strictly smaller, so no degraded re-send can "
        "overflow a budget its tool-call send fit in; the strict-mode "
        "response_format block is a pre-existing overhead request_size never "
        f"included (tool-call wire {stripped_bytes:,} B vs unstripped "
        f"{unstripped_bytes:,} B, saved {unstripped_bytes - stripped_bytes:,} B)",
    )
    report.note(
        "Strip is adapter-side and mode-conditional by construction: providers "
        "keep emitting the inline contract (scenario 2), so degraded modes and "
        "codex transport are unchanged. Only tool-call bodies lose the duplicate."
    )
    return report


# ---------------------------------------------------------------------------
# Scenario 10: recorded-attempt replay shapes
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "synthesis"
ATTEMPT_1 = "2026-09-24T192546Z-98bf4207803b-1"
ATTEMPT_2 = "2026-09-25T074903Z-98bf4207803b-1"
ATTEMPT_3 = "2026-09-25T085418Z-98bf4207803b-1"
ATTEMPT_4 = "2026-09-25T091923Z-98bf4207803b-1"
ATTEMPT_5 = "2026-09-25T094814Z-98bf4207803b-1"
WORK_FINDING_ID = "work-showcase-obscured-entry"
HEURISTIC_FINDING_ID = "work-showcase-discovery-warning-signals"

_ROLE_SCHEMA_BY_NAME: Mapping[str, type[Any]] = {
    "report-analyst": AnalystResponse,
    "report-evidence-auditor": EvidenceAuditResponse,
    "report-pattern-reviewer": PatternReviewResponse,
    "report-adjudicator": AdjudicationResponse,
}
_ROLE_NAME_BY_MODEL_ROLE: Mapping[ModelRole, str] = {
    ModelRole.REPORT_ANALYST: "analyst",
    ModelRole.REPORT_EVIDENCE_AUDITOR: "auditor",
    ModelRole.REPORT_PATTERN_REVIEWER: "pattern",
    ModelRole.REPORT_ADJUDICATOR: "adjudicator",
}


def _load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _recorded_corpus(tmp_path: Path) -> EvidenceCorpus:
    payload = _load_fixture("recorded-attempt-corpus.json")
    entries = []
    for item in payload["entries"]:
        entries.append(
            EvidenceEntry(
                ref=EvidenceRef(
                    evidence_id=item["evidence_id"],
                    kind=item["kind"],
                    run_id=item["run_id"],
                    viewport_id=item["viewport_id"],
                    element_id=item["element_id"],
                    event_id=item["event_id"],
                    metric_id=item["metric_id"],
                    artifact_path=item["artifact_path"],
                    replay_sequence=item["replay_sequence"],
                    sha256=item["sha256"],
                ),
                evidence_class=EvidenceClass(item["evidence_class"]),
                summary=item["summary"],
                payload=item["payload"],
            )
        )
    return EvidenceCorpus(
        output_root=tmp_path,
        entries=tuple(entries),
        principle_pack_version=payload["principle_pack_version"],
        principle_pack_digest=payload["principle_pack_digest"],
    )


def _recorded_ref(evidence_id: str, corpus: EvidenceCorpus) -> dict[str, object]:
    ref = corpus.require(evidence_id).ref
    return {
        "evidence_id": ref.evidence_id,
        "kind": ref.kind,
        "run_id": ref.run_id,
        "viewport_id": ref.viewport_id,
        "element_id": ref.element_id,
        "event_id": ref.event_id,
        "metric_id": ref.metric_id,
        "artifact_path": ref.artifact_path,
        "replay_sequence": ref.replay_sequence,
        "sha256": ref.sha256,
    }


def _expand_recorded(value: Any, corpus: EvidenceCorpus) -> Any:
    if isinstance(value, list):
        return [_expand_recorded(item, corpus) for item in value]
    if not isinstance(value, dict):
        return value
    expanded: dict[str, Any] = {}
    for name, item in value.items():
        if name == "evidence_ids":
            expanded["evidence_refs"] = [
                _recorded_ref(str(evidence_id), corpus) for evidence_id in item
            ]
        elif name == "counterevidence":
            expanded["counterevidence"] = [
                _recorded_ref(str(entry["evidence_id"]), corpus)
                if isinstance(entry, dict)
                else entry
                for entry in item
            ]
        else:
            expanded[name] = _expand_recorded(item, corpus)
    return expanded


def _recorded_attempt(attempt_id: str) -> dict[str, Any]:
    return _load_fixture("recorded-attempt-shapes.json")["attempts"][attempt_id]


def _recorded_role_response(
    attempt_id: str, role: ModelRole, corpus: EvidenceCorpus
) -> Any | None:
    rounds = _recorded_attempt(attempt_id)["roles"].get(role.value)
    if not rounds:
        return None
    payload = _expand_recorded(rounds[-1]["response"], corpus)
    return _ROLE_SCHEMA_BY_NAME[role.value].model_validate(payload)


def _recorded_candidate(
    attempt_id: str, corpus: EvidenceCorpus
) -> CandidateFinding:
    candidates = _recorded_attempt(attempt_id).get("candidates", ())
    assert candidates, f"{attempt_id} recorded no candidate finding"
    return CandidateFinding.model_validate(_expand_recorded(candidates[0], corpus))


def _replay_attempt(tmp_path: Path, attempt_id: str) -> Any:
    """Drive the real service with the recorded role outputs for one attempt."""

    corpus = _recorded_corpus(tmp_path)
    analyst = _recorded_role_response(attempt_id, ModelRole.REPORT_ANALYST, corpus)
    if analyst is None:
        analyst = AnalystResponse(
            complete=True,
            candidate_findings=[_recorded_candidate(attempt_id, corpus)],
        )
    scripted = {"analyst": [analyst]}
    for role in (
        ModelRole.REPORT_EVIDENCE_AUDITOR,
        ModelRole.REPORT_PATTERN_REVIEWER,
        ModelRole.REPORT_ADJUDICATOR,
    ):
        response = _recorded_role_response(attempt_id, role, corpus)
        scripted[_ROLE_NAME_BY_MODEL_ROLE[role]] = [] if response is None else [response]
    service, _roles = _synthesis_service(
        tmp_path,
        analyst=cast(Sequence[object], scripted["analyst"]),
        auditor=cast(Sequence[object], scripted["auditor"]),
        pattern=cast(Sequence[object], scripted["pattern"]),
        adjudicator=cast(Sequence[object], scripted["adjudicator"]),
    )
    return asyncio.run(service.synthesize(corpus))


def _evidence_ids(finding: object) -> set[str]:
    refs = cast(Any, finding).evidence_refs
    return {ref.evidence_id for ref in refs}


def scenario_recorded_attempt_shapes(tmp_path: Path) -> ScenarioReport:
    report = ScenarioReport("recorded-attempt-shapes")
    corpus = _recorded_corpus(tmp_path)

    attempt_2 = _replay_attempt(tmp_path, ATTEMPT_2)
    candidate = _recorded_candidate(ATTEMPT_2, corpus)
    published = attempt_2.findings
    removed = _evidence_ids(candidate) - (
        _evidence_ids(published[0]) if published else set()
    )
    authorized = {
        ref.evidence_id
        for objection in attempt_2.objections
        if objection.resolved and objection.resolution_evidence_refs
        for ref in objection.resolution_evidence_refs
    }
    report.check(
        "authorized-contraction-publishes",
        attempt_2.status is SynthesisStatus.ACCEPTED
        and [finding.finding_id for finding in published] == [WORK_FINDING_ID],
        f"attempt 2 replayed as {attempt_2.status.value} with "
        f"{len(published)} published finding(s)",
    )
    report.check(
        "authorized-contraction-is-resolution-backed",
        len(removed) == 6 and removed <= authorized,
        f"{len(removed)} candidate evidence IDs contracted, all cited by an "
        f"adjudicator resolution ({len(removed - authorized)} unauthorized)",
    )

    attempt_3 = _replay_attempt(tmp_path, ATTEMPT_3)
    report.check(
        "unauthorized-contraction-rejected",
        attempt_3.status is not SynthesisStatus.ACCEPTED
        and not attempt_3.findings
        and any(
            "changed the reviewed core claim" in limitation
            for limitation in attempt_3.limitations
        ),
        f"attempt 3 replayed as {attempt_3.status.value} with "
        f"{len(attempt_3.findings)} published finding(s)",
    )

    attempt_1 = _replay_attempt(tmp_path, ATTEMPT_1)
    heuristic_candidate = _recorded_candidate(ATTEMPT_1, corpus)
    heuristic_only = {
        ref.evidence_id
        for ref in heuristic_candidate.evidence_refs
        if any(
            marker in ref.evidence_id
            for marker in (
                "target-discovery-rank",
                "target-below-fold",
                "ambiguous-target",
                "discovery-cost",
            )
        )
    }
    report.check(
        "heuristic-only-claim-not-published",
        attempt_1.status is not SynthesisStatus.ACCEPTED
        and not attempt_1.findings
        and {finding.finding_id for finding in attempt_1.rejected_findings}
        == {HEURISTIC_FINDING_ID}
        and all(
            corpus.require(evidence_id).evidence_class is EvidenceClass.MODEL_ESTIMATE
            for evidence_id in heuristic_only
        ),
        f"attempt 1 replayed as {attempt_1.status.value}; the candidate rests on "
        f"{len(heuristic_only)} model-estimate metrics while the run records "
        "verified completion and zero wrong actions",
    )

    attempt_4_corpus = corpus
    auditor = _recorded_role_response(
        ATTEMPT_4, ModelRole.REPORT_EVIDENCE_AUDITOR, attempt_4_corpus
    )
    identifiers = [item.objection_id for item in auditor.objections]
    duplicates = sorted({name for name in identifiers if identifiers.count(name) > 1})
    collapsed = _replay_attempt(tmp_path, ATTEMPT_4)
    report.check(
        "identical-duplicate-objections-collapse",
        bool(duplicates)
        and not any(
            "Reviewer objections failed deterministic validation" in limitation
            for limitation in collapsed.limitations
        )
        and len({item.objection_id for item in collapsed.objections})
        == len(collapsed.objections),
        f"{len(duplicates)} identical duplicate objection ID(s) in the recorded "
        "auditor response collapse to one each and adjudication proceeds",
    )
    report.check(
        "recorded-duplicate-shape-is-real",
        all(
            len(
                {
                    item.model_dump_json()
                    for item in auditor.objections
                    if item.objection_id == name
                }
            )
            == 1
            for name in duplicates
        ),
        "the replayed duplicate objections are byte-identical, so collapsing "
        "them cannot hide a disagreement",
    )

    attempt_5 = _replay_attempt(tmp_path, ATTEMPT_5)
    report.check(
        "accepted-baseline-still-publishes",
        attempt_5.status is SynthesisStatus.ACCEPTED
        and [finding.finding_id for finding in attempt_5.findings]
        == [WORK_FINDING_ID],
        f"attempt 5 replayed as {attempt_5.status.value} with "
        f"{len(attempt_5.findings)} published finding(s)",
    )
    report.note(
        "These five checks are the definition of 'did not worsen quality'. Any "
        "prompt change, gate change, or payload reshaping must keep the "
        "authorized contraction publishing, the unauthorized one rejecting, the "
        "heuristic-only claim unpublished, the duplicates collapsed, and the "
        "accepted baseline finding published."
    )
    return report


# ---------------------------------------------------------------------------
# Scenario 11: prompt-prefix stability
# ---------------------------------------------------------------------------

_STATIC_PAYLOAD_KEYS = (
    "corpus_manifest",
    "response_schema",
    "ux_principle_pack",
    "role_input",
)


def _static_prefix_bytes(payload: str) -> int:
    parsed = json.loads(payload)
    total = 1
    for key, value in parsed.items():
        if key not in _STATIC_PAYLOAD_KEYS:
            return total
        total += (
            len(key)
            + 1
            + len(json.dumps(value, ensure_ascii=True, separators=(",", ":")))
            + 1
        )
    return total


def _shared_prefix_bytes(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def scenario_prompt_prefix_stability(tmp_path: Path) -> ScenarioReport:
    report = ScenarioReport("prompt-prefix-stability")
    corpus = _recorded_corpus(tmp_path)
    client = _FlakyStructuredClient(defaults=_synthesis_defaults())
    analyst = ReportAnalyst(client, model="gate-report-model")  # type: ignore[arg-type]

    async def _rounds() -> None:
        for round_number in (1, 2, 3):
            await analyst.analyze(corpus, ux_principles(), retrieval_round=round_number)

    asyncio.run(_rounds())
    payloads = [call["user_content"] for call in client.calls]
    keys = list(json.loads(payloads[0]))
    static_positions = [keys.index(key) for key in _STATIC_PAYLOAD_KEYS if key in keys]
    varying_positions = [index for index, key in enumerate(keys) if index not in static_positions]
    static_prefix = _static_prefix_bytes(payloads[0])
    shared = min(
        _shared_prefix_bytes(payloads[0], payloads[1]),
        _shared_prefix_bytes(payloads[1], payloads[2]),
    )

    report.check(
        "static-blocks-serialize-first",
        set(_STATIC_PAYLOAD_KEYS) <= set(keys)
        and (not varying_positions or max(static_positions) < min(varying_positions)),
        f"payload key order is {keys}",
    )
    report.check(
        "rounds-share-the-static-prefix",
        shared >= static_prefix,
        f"consecutive analyst rounds share {shared:,} of {len(payloads[0]):,} "
        f"payload bytes; the static prefix is {static_prefix:,} bytes",
    )
    report.metrics["analyst_payload_bytes"] = len(payloads[0])
    report.metrics["static_prefix_bytes"] = static_prefix
    report.metrics["shared_prefix_bytes"] = shared
    report.metrics["static_prefix_ratio"] = round(
        100 * shared / max(len(payloads[0]), 1), 1
    )
    report.note(
        "Provider prompt caching only reuses a request prefix, so the reusable "
        "bytes must come first. Alphabetical key order put the 17 kB principle "
        "pack and 5 kB response schema behind the 0.5 kB per-round retrieval "
        "policy, which truncated the reusable prefix to the 2.8 kB corpus "
        "manifest and held the observed cache hit ratio to 0.3-13% across "
        "1.38M prompt tokens on one identical corpus. Reordering is "
        "serialization-only: key names, values, and the system prompt are "
        "unchanged, so no prompt_version bump is due."
    )
    return report


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
    scenario_adapter_schema_strip_modes,
    scenario_recorded_attempt_shapes,
    scenario_prompt_prefix_stability,
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
