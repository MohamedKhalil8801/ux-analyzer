"""Replay fixtures for the five recorded report-synthesis attempts.

Every attempt under ``reports/exploration-generated/synthesis/attempts/`` ran
against one identical corpus (digest ``98bf4207803ba00e...``). Four of the five
were rejected: two rejections were mechanical bugs in the pipeline, two were
correct verdicts, and the fixes for the two bugs landed together in commit
``130cc24``. Those fixes are only trustworthy while a deterministic replay of
the same role outputs keeps producing the same verdicts, which is what this
module pins.

``/reports/`` is gitignored, so the replay inputs were extracted into two
committed fixtures:

* ``tests/fixtures/synthesis/recorded-attempt-corpus.json`` - the 36 evidence
  entries the recorded role outputs referenced. IDs, kinds, evidence classes,
  and summaries are verbatim; viewport element/region arrays are trimmed
  because publication validation reads only the reference namespace, the
  evidence class, and the surface tokens.
* ``tests/fixtures/synthesis/recorded-attempt-shapes.json`` - the redacted role
  responses as recorded in each attempt's ``retrieval_log``. Evidence references
  are stored as IDs and rebuilt from the corpus entry; objection and resolution
  prose is excerpted. Finding prose is verbatim.

Shape mapping:

======  =============================  ========================================
Shape   Recorded attempt              Asserted behaviour
======  =============================  ========================================
A       attempt 2 (2026-09-25T074903Z) A final that narrows a candidate by
                                       removing disproven evidence *with*
                                       resolution authorization publishes.
B       attempt 3 (2026-09-25T085418Z) The same removal *without* resolution
                                       authorization is still rejected.
C       attempt 1 (2026-09-24T192546Z) A candidate whose issue and impact rest
                                       on heuristic signals that delivered
                                       observed evidence contradicts never
                                       reaches publication.
D       attempt 4 (2026-09-25T091923Z) Identical duplicate reviewer objections
                                       collapse to one and adjudication runs;
                                       conflicting duplicates still reject.
======  =============================  ========================================

Shape E replays the accepted attempt 5 finding as a negative control: the
deterministic guards must keep publishing it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from ux_analyzer.application.evidence_corpus import EvidenceCorpus, EvidenceEntry
from ux_analyzer.application.report_synthesis import ReportSynthesisService
from ux_analyzer.domain.findings import EvidenceClass
from ux_analyzer.domain.synthesis import (
    REPORT_ADJUDICATOR_ROLE,
    EvidenceRef,
    ReviewDisposition,
    SynthesisAttempt,
    SynthesisStatus,
)
from ux_analyzer.ports.models import ModelCallRecord, ModelRole, TokenUsage
from ux_analyzer.ports.report_synthesis import ScenarioReviewReport
from ux_analyzer.providers.report_synthesis import (
    AdjudicationResponse,
    AnalystResponse,
    CandidateFinding,
    EvidenceAuditResponse,
    EvidenceReference,
    PatternReviewResponse,
)

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "synthesis"

ATTEMPT_1 = "2026-09-24T192546Z-98bf4207803b-1"
ATTEMPT_2 = "2026-09-25T074903Z-98bf4207803b-1"
ATTEMPT_3 = "2026-09-25T085418Z-98bf4207803b-1"
ATTEMPT_4 = "2026-09-25T091923Z-98bf4207803b-1"
ATTEMPT_5 = "2026-09-25T094814Z-98bf4207803b-1"
RECORDED_ATTEMPTS = (ATTEMPT_1, ATTEMPT_2, ATTEMPT_3, ATTEMPT_4, ATTEMPT_5)

WORK_FINDING_ID = "work-showcase-obscured-entry"
HEURISTIC_FINDING_ID = "work-showcase-discovery-warning-signals"
WORK_RUN_SUFFIX = "d68bc046b7b"

_RESPONSE_SCHEMA = {
    ModelRole.REPORT_ANALYST: AnalystResponse,
    ModelRole.REPORT_EVIDENCE_AUDITOR: EvidenceAuditResponse,
    ModelRole.REPORT_PATTERN_REVIEWER: PatternReviewResponse,
    ModelRole.REPORT_ADJUDICATOR: AdjudicationResponse,
}


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------


def _load(name: str) -> Mapping[str, Any]:
    return cast(
        Mapping[str, Any],
        json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8")),
    )


def _shapes_document() -> Mapping[str, Any]:
    return cast(Mapping[str, Any], _load("recorded-attempt-shapes.json"))


def _ref_fields(evidence_id: str, corpus: EvidenceCorpus) -> dict[str, object]:
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


def _reference(evidence_id: str, corpus: EvidenceCorpus) -> EvidenceReference:
    return EvidenceReference.model_validate(_ref_fields(evidence_id, corpus))


def _expand(value: object, corpus: EvidenceCorpus) -> object:
    """Rebuild a recorded payload's evidence references from the corpus."""

    if isinstance(value, list):
        return [_expand(item, corpus) for item in value]
    if not isinstance(value, dict):
        return value
    expanded: dict[str, object] = {}
    for name, item in value.items():
        if name == "evidence_ids":
            expanded["evidence_refs"] = [
                _reference(str(evidence_id), corpus)
                for evidence_id in cast(list[str], item)
            ]
        elif name == "counterevidence":
            expanded["counterevidence"] = [
                _reference(str(cast(Mapping[str, Any], entry)["evidence_id"]), corpus)
                if isinstance(entry, dict)
                else entry
                for entry in cast(list[object], item)
            ]
        else:
            expanded[name] = _expand(item, corpus)
    return expanded


def _expanded_record(
    record: Mapping[str, object], corpus: EvidenceCorpus
) -> dict[str, object]:
    return cast(dict[str, object], _expand(dict(record), corpus))


def _fixture_corpus(tmp_path: Path) -> EvidenceCorpus:
    payload = _load("recorded-attempt-corpus.json")
    entries = tuple(
        EvidenceEntry(
            ref=EvidenceRef(
                evidence_id=cast(str, item["evidence_id"]),
                kind=cast(str, item["kind"]),
                run_id=cast(str, item["run_id"]),
                viewport_id=cast(str | None, item["viewport_id"]),
                element_id=cast(str | None, item["element_id"]),
                event_id=cast(str | None, item["event_id"]),
                metric_id=cast(str | None, item["metric_id"]),
                artifact_path=cast(str | None, item["artifact_path"]),
                replay_sequence=cast(int | None, item["replay_sequence"]),
                sha256=cast(str | None, item["sha256"]),
            ),
            evidence_class=EvidenceClass(cast(str, item["evidence_class"])),
            summary=cast(str, item["summary"]),
            payload=cast(Mapping[str, object], item["payload"]),
        )
        for item in cast(list[Mapping[str, Any]], payload["entries"])
    )
    return EvidenceCorpus(
        output_root=tmp_path,
        entries=entries,
        principle_pack_version=cast(str, payload["principle_pack_version"]),
        principle_pack_digest=cast(str, payload["principle_pack_digest"]),
    )


def _attempt_record(attempt_id: str) -> Mapping[str, Any]:
    attempts = cast(Mapping[str, Any], _shapes_document()["attempts"])
    return cast(Mapping[str, Any], attempts[attempt_id])


def _recorded_response(
    attempt_id: str, role: ModelRole, corpus: EvidenceCorpus
) -> object | None:
    """The recorded response for *role*, or None when the stage was reused."""

    rounds = cast(
        Mapping[str, list[Mapping[str, Any]]], _attempt_record(attempt_id)["roles"]
    )
    entries = rounds.get(role.value)
    if not entries:
        return None
    payload = cast(Mapping[str, object], entries[-1]["response"])
    return _RESPONSE_SCHEMA[role].model_validate(
        cast(Mapping[str, object], _expand(payload, corpus))
    )


def _recorded_candidate(attempt_id: str, corpus: EvidenceCorpus) -> CandidateFinding:
    """The analyst candidate the attempt actually started from.

    Attempts 3 and 4 resumed the analyst stage, so their recorded artifact
    carries the reused candidate rather than a fresh analyst response.
    """

    candidates = cast(
        list[Mapping[str, object]], _attempt_record(attempt_id).get("candidates", ())
    )
    assert candidates, f"{attempt_id} recorded no candidate finding"
    return CandidateFinding.model_validate(_expanded_record(candidates[0], corpus))


def _recorded_accepted_finding(
    attempt_id: str, corpus: EvidenceCorpus
) -> CandidateFinding:
    """The finding the recorded attempt published."""

    findings = cast(
        list[Mapping[str, object]],
        _attempt_record(attempt_id).get("final_findings", ()),
    )
    assert findings, f"{attempt_id} recorded no published finding"
    return CandidateFinding.model_validate(_expanded_record(findings[0], corpus))


def _scenario_reviews(corpus: EvidenceCorpus) -> list[ScenarioReviewReport]:
    """One examined review per corpus scenario.

    Publication validation rejects an attempt that leaves any scenario
    unexamined, so every replayed analyst response has to account for every
    scenario the corpus holds evidence for.
    """

    return [
        ScenarioReviewReport(
            scenario_id=scenario_id,
            disposition=ReviewDisposition.NO_ISSUE_FOUND,
            evidence_ids=[f"scenario:{entry.ref.run_id}"],
            signals_weighed=["attention cost", "action count"],
            note="Replayed attempt: the recorded record did not examine this "
            "scenario for improvements.",
        )
        for scenario_id in corpus.scenario_ids()
        for entry in corpus.entries
        if entry.ref.kind == "scenario" and entry.payload.get("id") == scenario_id
    ]


def _candidate_response(attempt_id: str, corpus: EvidenceCorpus) -> AnalystResponse:
    recorded = _recorded_response(attempt_id, ModelRole.REPORT_ANALYST, corpus)
    if isinstance(recorded, AnalystResponse):
        assert recorded.complete
        return recorded.model_copy(
            update={"scenario_reviews": _scenario_reviews(corpus)}
        )
    return AnalystResponse(
        complete=True,
        candidate_findings=[_recorded_candidate(attempt_id, corpus)],
        scenario_reviews=_scenario_reviews(corpus),
    )


# ---------------------------------------------------------------------------
# Role doubles
# ---------------------------------------------------------------------------


class _RecordingModelSource:
    def __init__(self) -> None:
        self.records: list[ModelCallRecord] = []

    def record(self, role: str, call: Mapping[str, Any], response: object) -> None:
        self.records.append(
            ModelCallRecord(
                role=ModelRole(role),
                model="fixture-model",
                endpoint_origin="https://fixture.invalid",
                prompt_digest=hashlib.sha256(
                    repr(sorted(call.items(), key=str)).encode("utf-8")
                ).hexdigest(),
                schema_version=getattr(response, "schema_version", "unknown"),
                attempts=1,
                latency_ms=0,
                token_usage=TokenUsage(0, 0, 0),
                request={},
                response={"type": type(response).__name__},
            )
        )


class _ScriptedRole:
    """Service-role double: replays scripted outcomes and records every call."""

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

    def _next(self, call: Mapping[str, Any]) -> object:
        self.calls.append(dict(call))
        response = self.responses.pop(0) if self.responses else None
        if isinstance(response, BaseException):
            raise response
        if response is not None:
            result = response
        elif self.role == "report-analyst":
            result = AnalystResponse(complete=True)
        elif self.role == "report-evidence-auditor":
            result = EvidenceAuditResponse(complete=True)
        elif self.role == "report-pattern-reviewer":
            result = PatternReviewResponse(complete=True)
        else:
            result = AdjudicationResponse(complete=True)
        if self.record_source is not None:
            self.record_source.record(self.role, call, result)
        return result

    async def analyze(self, *args: Any, **kwargs: Any) -> object:
        return self._next(kwargs)

    async def audit(self, *args: Any, **kwargs: Any) -> object:
        return self._next(kwargs)

    async def review(self, *args: Any, **kwargs: Any) -> object:
        return self._next(kwargs)

    async def adjudicate(self, *args: Any, **kwargs: Any) -> object:
        return self._next(kwargs)


def _service(
    analyst: Sequence[object] = (),
    auditor: Sequence[object] = (),
    pattern: Sequence[object] = (),
    adjudicator: Sequence[object] = (),
) -> tuple[ReportSynthesisService, tuple[_ScriptedRole, ...]]:
    source = _RecordingModelSource()
    roles = (
        _ScriptedRole("report-analyst", analyst, source),
        _ScriptedRole("report-evidence-auditor", auditor, source),
        _ScriptedRole("report-pattern-reviewer", pattern, source),
        _ScriptedRole("report-adjudicator", adjudicator, source),
    )
    return (
        ReportSynthesisService(
            analyst=roles[0],
            evidence_auditor=roles[1],
            pattern_reviewer=roles[2],
            adjudicator=roles[3],
            model_record_source=source,
        ),
        roles,
    )


def _replay(
    attempt_id: str,
    tmp_path: Path,
    *,
    adjudicator: Sequence[object] | None = None,
    auditor: object | None = None,
) -> SynthesisAttempt:
    """Drive the service with the recorded role outputs for *attempt_id*."""

    corpus = _fixture_corpus(tmp_path)
    analyst = _candidate_response(attempt_id, corpus)
    auditor_response = (
        auditor
        if auditor is not None
        else _recorded_response(attempt_id, ModelRole.REPORT_EVIDENCE_AUDITOR, corpus)
    )
    pattern = _recorded_response(attempt_id, ModelRole.REPORT_PATTERN_REVIEWER, corpus)
    adjudication = _recorded_response(attempt_id, ModelRole.REPORT_ADJUDICATOR, corpus)
    scripted_adjudicator = (
        adjudicator
        if adjudicator is not None
        else (() if adjudication is None else (adjudication,))
    )
    service, _roles = _service(
        analyst=[analyst],
        auditor=() if auditor_response is None else (auditor_response,),
        pattern=() if pattern is None else (pattern,),
        adjudicator=scripted_adjudicator,
    )
    return asyncio.run(service.synthesize(corpus))


def _evidence_ids(finding: object) -> set[str]:
    refs = cast(Any, finding).evidence_refs
    return {ref.evidence_id for ref in refs}


def _authorized_evidence_ids(attempt: SynthesisAttempt) -> set[str]:
    return {
        ref.evidence_id
        for objection in attempt.objections
        if objection.resolved and objection.resolved_by_role == REPORT_ADJUDICATOR_ROLE
        for ref in objection.resolution_evidence_refs
    }


def _work_run_id(corpus: EvidenceCorpus) -> str:
    for entry in corpus.entries:
        if entry.ref.run_id.endswith(WORK_RUN_SUFFIX):
            return entry.ref.run_id
    raise AssertionError("recorded corpus has no work run")


def _unauthorized_attempt_three_removals() -> tuple[str, ...]:
    """Candidate evidence the attempt-3 final drops without authorization.

    Derived from the fixture so the anti-loosening test stays pinned to the
    recorded shape instead of a hand-copied ID list.
    """

    record = _attempt_record(ATTEMPT_3)
    candidate = {
        str(value)
        for value in cast(Mapping[str, Any], record["candidates"][0])["evidence_ids"]
    }
    adjudication = cast(
        Mapping[str, Any], record["roles"]["report-adjudicator"][-1]["response"]
    )
    final = {
        str(value)
        for value in cast(Mapping[str, Any], adjudication["final_findings"][0])[
            "evidence_ids"
        ]
    }
    authorized = {
        str(value)
        for resolution in cast(
            list[Mapping[str, Any]], adjudication["objection_resolutions"]
        )
        for value in resolution["evidence_ids"]
    }
    return tuple(sorted((candidate - final) - authorized))


_UNAUTHORIZED_REMOVALS = _unauthorized_attempt_three_removals()


# ---------------------------------------------------------------------------
# Fixture integrity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("attempt_id", RECORDED_ATTEMPTS)
def test_every_recorded_attempt_fixture_is_replayable(
    tmp_path: Path, attempt_id: str
) -> None:
    corpus = _fixture_corpus(tmp_path)
    record = _attempt_record(attempt_id)
    assert record["status"] in {"accepted", "rejected"}
    published = _recorded_response(attempt_id, ModelRole.REPORT_ADJUDICATOR, corpus)
    if published is None:
        assert _recorded_candidate(attempt_id, corpus).finding_id
    else:
        assert cast(AdjudicationResponse, published).complete
    for role in cast(Mapping[str, Any], record["roles"]):
        model_role = ModelRole(role)
        response = _recorded_response(attempt_id, model_role, corpus)
        assert isinstance(response, _RESPONSE_SCHEMA[model_role])


def test_recorded_corpus_covers_every_referenced_evidence_id(
    tmp_path: Path,
) -> None:
    corpus = _fixture_corpus(tmp_path)
    known = {entry.ref.evidence_id for entry in corpus.entries}
    referenced: set[str] = set()
    for record in cast(Mapping[str, Any], _shapes_document()["attempts"]).values():
        for key in ("candidates", "final_findings"):
            for finding in cast(list[Mapping[str, Any]], record.get(key, ())):
                referenced.update(
                    str(value) for value in finding.get("evidence_ids", ())
                )
        for rounds in cast(Mapping[str, Any], record["roles"]).values():
            for round_record in cast(list[Mapping[str, Any]], rounds):
                response = cast(Mapping[str, Any], round_record["response"])
                for key in (
                    "candidate_findings",
                    "final_findings",
                    "objections",
                    "objection_resolutions",
                ):
                    for item in cast(list[Mapping[str, Any]], response.get(key, ())):
                        referenced.update(
                            str(value) for value in item.get("evidence_ids", ())
                        )
    assert len(referenced) >= 30
    assert referenced <= known


def test_attempt_three_and_four_reused_the_attempt_two_analyst_stage(
    tmp_path: Path,
) -> None:
    corpus = _fixture_corpus(tmp_path)
    baseline = _recorded_candidate(ATTEMPT_2, corpus)
    for attempt_id in (ATTEMPT_3, ATTEMPT_4):
        assert _recorded_candidate(attempt_id, corpus) == baseline
        assert _recorded_response(attempt_id, ModelRole.REPORT_ANALYST, corpus) is None


def test_accepted_attempt_five_finding_still_publishes(tmp_path: Path) -> None:
    corpus = _fixture_corpus(tmp_path)
    published = _recorded_accepted_finding(ATTEMPT_5, corpus)
    candidate = _recorded_candidate(ATTEMPT_2, corpus)
    assert published.finding_id == candidate.finding_id == WORK_FINDING_ID
    # The published baseline rests on deterministic records, not heuristics.
    assert published.evidence_class == EvidenceClass.DETERMINISTIC_FACT
    assert all(
        corpus.require(ref.evidence_id).evidence_class
        is EvidenceClass.DETERMINISTIC_FACT
        for ref in published.evidence_refs
    )

    attempt = _replay(ATTEMPT_5, tmp_path)

    assert attempt.status is SynthesisStatus.ACCEPTED
    assert [finding.finding_id for finding in attempt.findings] == [WORK_FINDING_ID]
    assert _evidence_ids(attempt.findings[0]) == _evidence_ids(published)
    assert attempt.findings[0].reviewer_state == "accepted"


# ---------------------------------------------------------------------------
# Shape A (attempt 2): authorized contraction must publish
# ---------------------------------------------------------------------------


def test_attempt_two_shape_publishes_resolution_backed_contraction(
    tmp_path: Path,
) -> None:
    corpus = _fixture_corpus(tmp_path)
    candidate = _recorded_candidate(ATTEMPT_2, corpus)

    attempt = _replay(ATTEMPT_2, tmp_path)

    assert attempt.status is SynthesisStatus.ACCEPTED
    assert [finding.finding_id for finding in attempt.findings] == [WORK_FINDING_ID]
    published = attempt.findings[0]
    assert published.reviewer_state == "accepted"
    removed = _evidence_ids(candidate) - _evidence_ids(published)
    assert len(removed) == 6
    assert removed <= _authorized_evidence_ids(attempt)
    assert not any(
        "failed publication validation" in limitation
        or "changed the reviewed core claim" in limitation
        for limitation in attempt.limitations
    )


def test_pre_fix_superset_guard_rejected_every_narrowed_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Why attempt 2 was a mechanical failure, not a quality verdict.

    Before commit ``130cc24`` the guard was a plain subset test - the final
    evidence set had to be a superset of the candidate's - so *no* narrowed
    finding could ever publish, authorized or not. Emulating that rule here
    pins the two things the rewrite must keep true: an authorized contraction
    now publishes, and the guard is no longer the only thing that stood between
    the adjudicator and a valid finding.
    """

    def superset_only_guard(
        final: object, candidate: object, *, objections: object = ()
    ) -> bool:
        final_refs = cast(Any, final).evidence_refs
        candidate_refs = cast(Any, candidate).evidence_refs
        candidate_ids = {ref.evidence_id for ref in candidate_refs}
        return {ref.evidence_id for ref in final_refs} >= candidate_ids

    from ux_analyzer.application import report_synthesis as application

    monkeypatch.setattr(
        application, "final_finding_preserves_candidate", superset_only_guard
    )

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        replayed = _fixture_corpus(tmp)
        for attempt_id in (ATTEMPT_2, ATTEMPT_5):
            candidate = _recorded_candidate(attempt_id, replayed)
            adjudication = cast(
                AdjudicationResponse,
                _recorded_response(attempt_id, ModelRole.REPORT_ADJUDICATOR, replayed),
            )
            final = adjudication.final_findings[0]
            assert _evidence_ids(candidate) - _evidence_ids(final), (
                f"{attempt_id} must narrow the candidate"
            )
            assert superset_only_guard(final, candidate) is False
            attempt = _replay(attempt_id, tmp)
            assert not attempt.findings
            assert attempt.status is not SynthesisStatus.ACCEPTED

        # The live guard accepts the same shapes.
        monkeypatch.undo()
        for attempt_id in (ATTEMPT_2, ATTEMPT_5):
            attempt = _replay(attempt_id, tmp)
            assert attempt.status is SynthesisStatus.ACCEPTED
            assert [finding.finding_id for finding in attempt.findings] == [
                WORK_FINDING_ID
            ]


def test_attempt_two_shape_needs_the_resolution_authorization(
    tmp_path: Path,
) -> None:
    """The attempt-2 rejection was mechanical, not a quality verdict.

    ``final_finding_preserves_candidate`` used to require the final evidence set
    to be a superset of the candidate's, so *any* contraction failed - including
    one the adjudicator had authorized with resolution evidence. Dropping the
    recorded resolutions must put the same final back into rejection.
    """

    corpus = _fixture_corpus(tmp_path)
    adjudication = cast(
        AdjudicationResponse,
        _recorded_response(ATTEMPT_2, ModelRole.REPORT_ADJUDICATOR, corpus),
    )
    stripped = adjudication.model_copy(update={"objection_resolutions": []})

    attempt = _replay(ATTEMPT_2, tmp_path, adjudicator=[stripped])

    assert attempt.status is not SynthesisStatus.ACCEPTED
    assert not attempt.findings
    assert any(
        "changed the reviewed core claim" in limitation
        for limitation in attempt.limitations
    )


# ---------------------------------------------------------------------------
# Shape B (attempt 3): unauthorized contraction must still be rejected
# ---------------------------------------------------------------------------


def test_attempt_three_shape_rejects_unauthorized_contraction(
    tmp_path: Path,
) -> None:
    corpus = _fixture_corpus(tmp_path)
    candidate = _recorded_candidate(ATTEMPT_3, corpus)
    adjudication = cast(
        AdjudicationResponse,
        _recorded_response(ATTEMPT_3, ModelRole.REPORT_ADJUDICATOR, corpus),
    )
    final = adjudication.final_findings[0]
    removed = _evidence_ids(candidate) - _evidence_ids(final)
    authorized = {
        ref.evidence_id
        for resolution in adjudication.objection_resolutions
        for ref in resolution.evidence_refs
    }
    assert removed
    assert removed - authorized, "the replayed final must drop unauthorized evidence"

    attempt = _replay(ATTEMPT_3, tmp_path)

    assert attempt.status is not SynthesisStatus.ACCEPTED
    assert not attempt.findings
    assert any(
        "changed the reviewed core claim" in limitation
        for limitation in attempt.limitations
    )
    assert WORK_FINDING_ID in {
        finding.finding_id for finding in attempt.rejected_findings
    }


@pytest.mark.parametrize("dropped_evidence_id", _UNAUTHORIZED_REMOVALS)
def test_contraction_guard_needs_each_removed_id_named_by_a_resolution(
    tmp_path: Path, dropped_evidence_id: str
) -> None:
    """Anti-loosening guard for the contraction rewrite.

    The attempt-3 shape is rejected because two of the four heuristic evidence
    IDs its final drops are not named by any adjudicator resolution. For each
    such ID, withdrawing the resolution that authorizes it must keep the attempt
    rejected: authorizing one removed ID never authorizes a different one.
    """

    corpus = _fixture_corpus(tmp_path)
    candidate = _recorded_candidate(ATTEMPT_3, corpus)
    adjudication = cast(
        AdjudicationResponse,
        _recorded_response(ATTEMPT_3, ModelRole.REPORT_ADJUDICATOR, corpus),
    )
    final = adjudication.final_findings[0]
    assert dropped_evidence_id in _evidence_ids(candidate)
    assert dropped_evidence_id not in _evidence_ids(final)
    stripped = adjudication.model_copy(
        update={
            "objection_resolutions": [
                resolution.model_copy(update={"evidence_refs": []})
                if any(
                    ref.evidence_id == dropped_evidence_id
                    for ref in resolution.evidence_refs
                )
                else resolution
                for resolution in adjudication.objection_resolutions
            ]
        }
    )

    attempt = _replay(ATTEMPT_3, tmp_path, adjudicator=[stripped])

    assert attempt.status is not SynthesisStatus.ACCEPTED
    assert not attempt.findings
    assert any(
        "changed the reviewed core claim" in limitation
        for limitation in attempt.limitations
    )


def test_contraction_guard_fails_closed_without_any_resolution_evidence(
    tmp_path: Path,
) -> None:
    """Even a fully resolved objection set cannot authorize a contraction."""

    corpus = _fixture_corpus(tmp_path)
    adjudication = cast(
        AdjudicationResponse,
        _recorded_response(ATTEMPT_3, ModelRole.REPORT_ADJUDICATOR, corpus),
    )
    stripped = adjudication.model_copy(
        update={
            "objection_resolutions": [
                resolution.model_copy(update={"evidence_refs": []})
                for resolution in adjudication.objection_resolutions
            ]
        }
    )

    attempt = _replay(ATTEMPT_3, tmp_path, adjudicator=[stripped])

    assert not attempt.findings
    assert attempt.status is not SynthesisStatus.ACCEPTED


# ---------------------------------------------------------------------------
# Shape C (attempt 1): heuristic-only harm claims must not publish
# ---------------------------------------------------------------------------


def test_attempt_one_shape_heuristic_claim_never_publishes(
    tmp_path: Path,
) -> None:
    corpus = _fixture_corpus(tmp_path)
    candidate = _recorded_candidate(ATTEMPT_1, corpus)
    heuristic_only = {
        ref.evidence_id
        for ref in candidate.evidence_refs
        if "target-discovery-rank" in ref.evidence_id
        or "target-below-fold" in ref.evidence_id
        or "ambiguous-target" in ref.evidence_id
        or "discovery-cost" in ref.evidence_id
    }
    assert len(heuristic_only) == 4
    assert all(
        corpus.require(evidence_id).evidence_class is EvidenceClass.MODEL_ESTIMATE
        for evidence_id in heuristic_only
    )
    work_run = _work_run_id(corpus)
    assert corpus.require(f"metric:{work_run}:wrong-actions").payload["value"] == 0
    assert corpus.require(f"metric:{work_run}:verified-completion").payload["value"]

    attempt = _replay(ATTEMPT_1, tmp_path)

    assert attempt.status is not SynthesisStatus.ACCEPTED
    assert not attempt.findings
    assert {finding.finding_id for finding in attempt.rejected_findings} == {
        HEURISTIC_FINDING_ID
    }


def test_publish_everything_adjudicator_cannot_rescue_the_attempt_one_claim(
    tmp_path: Path,
) -> None:
    corpus = _fixture_corpus(tmp_path)
    candidate = _recorded_candidate(ATTEMPT_1, corpus)

    attempt = _replay(
        ATTEMPT_1,
        tmp_path,
        adjudicator=[AdjudicationResponse(complete=True, final_findings=[candidate])],
    )

    assert not attempt.findings
    assert attempt.status is not SynthesisStatus.ACCEPTED


# ---------------------------------------------------------------------------
# Shape D (attempt 4): duplicate reviewer objections
# ---------------------------------------------------------------------------


def test_attempt_four_auditor_response_carries_identical_duplicate_objections(
    tmp_path: Path,
) -> None:
    corpus = _fixture_corpus(tmp_path)
    auditor = cast(
        EvidenceAuditResponse,
        _recorded_response(ATTEMPT_4, ModelRole.REPORT_EVIDENCE_AUDITOR, corpus),
    )
    identifiers = [item.objection_id for item in auditor.objections]
    duplicates = sorted({name for name in identifiers if identifiers.count(name) > 1})
    assert duplicates, "the recorded response must carry identical duplicate IDs"
    for name in duplicates:
        repeated = [item for item in auditor.objections if item.objection_id == name]
        serialized = {item.model_dump_json() for item in repeated}
        assert len(serialized) == 1


def test_identical_duplicate_objections_collapse_and_adjudication_proceeds(
    tmp_path: Path,
) -> None:
    corpus = _fixture_corpus(tmp_path)
    auditor = cast(
        EvidenceAuditResponse,
        _recorded_response(ATTEMPT_4, ModelRole.REPORT_EVIDENCE_AUDITOR, corpus),
    )
    identifiers = [item.objection_id for item in auditor.objections]
    scoped = {f"report-evidence-auditor:{name}" for name in identifiers}
    work_run = _work_run_id(corpus)
    disposition = AdjudicationResponse(
        complete=True,
        objection_resolutions=[
            {
                "objection_id": objection_id,
                "finding_id": WORK_FINDING_ID,
                "resolved": True,
                "resolution": "Sustained. The delivered record answers the challenge.",
                "evidence_refs": [_ref_fields(f"verification:{work_run}", corpus)],
            }
            for objection_id in sorted(scoped)
        ],
    )
    service, roles = _service(
        analyst=[_candidate_response(ATTEMPT_4, corpus)],
        auditor=[auditor],
        pattern=[
            cast(
                PatternReviewResponse,
                _recorded_response(
                    ATTEMPT_4, ModelRole.REPORT_PATTERN_REVIEWER, corpus
                ),
            )
        ],
        adjudicator=[disposition, AdjudicationResponse(complete=True)],
    )
    attempt = asyncio.run(service.synthesize(_fixture_corpus(tmp_path)))

    assert len(roles[1].calls) == 1
    assert not any(
        "Reviewer objections failed deterministic validation" in limitation
        for limitation in attempt.limitations
    )
    objection_ids = [item.objection_id for item in attempt.objections]
    assert len(objection_ids) == len(set(objection_ids))
    assert scoped <= set(objection_ids)
    assert roles[3].calls, "adjudication must run after the reviewer output"


def test_conflicting_duplicate_objections_still_reject_the_role_output(
    tmp_path: Path,
) -> None:
    corpus = _fixture_corpus(tmp_path)
    auditor = cast(
        EvidenceAuditResponse,
        _recorded_response(ATTEMPT_4, ModelRole.REPORT_EVIDENCE_AUDITOR, corpus),
    )
    payload = auditor.model_dump(mode="python")
    first = dict(cast(Mapping[str, Any], cast(list[Any], payload["objections"])[0]))
    payload["objections"] = [
        first,
        {**first, "message": "A different challenge under the same objection ID."},
    ]
    conflicting = EvidenceAuditResponse.model_validate(payload)

    service, _roles = _service(
        analyst=[_candidate_response(ATTEMPT_4, corpus)],
        auditor=[conflicting],
    )
    attempt = asyncio.run(service.synthesize(_fixture_corpus(tmp_path)))

    assert attempt.status is SynthesisStatus.REJECTED
    assert not attempt.findings
    assert any(
        "Reviewer objections failed deterministic validation" in limitation
        for limitation in attempt.limitations
    )
