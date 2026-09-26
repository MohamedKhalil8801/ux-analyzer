"""Analyst evidence-class discipline and prior-rejection feed-forward.

Covers the two accuracy workstreams that sit upstream of the recorded-attempt
replay fixtures:

* **Evidence-class discipline.** A heuristic prominence signal is a model
  estimate. A claim that a person could not do something, had to hunt for it, or
  was delayed has to stand on an observed record. The analyst prompt carries
  the discipline in words (v12) and publication validation carries it in code,
  so a heuristic-only harm claim is stopped at the analyst stage instead of
  spending auditor, reviewer, and adjudicator work on it.
* **Rejection feed-forward.** Attempts 2 and 3 re-derived and re-rejected the
  same "work showcase" candidate blind. Prior rejections are now handed to the
  analyst as context - never a prohibition - with every referenced evidence ID
  checked against the live corpus first.

The recorded-attempt replay fixtures in
``tests/unit/application/test_recorded_attempt_regressions.py`` remain the
quality guard: nothing here may make those verdicts change.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from ux_analyzer.application.evidence_corpus import EvidenceCorpus, EvidenceEntry
from ux_analyzer.application.report_synthesis import (
    MAX_PRIOR_REJECTIONS,
    ReportSynthesisService,
    _heuristic_only_harm_claim,  # pyright: ignore[reportPrivateUsage]
)
from ux_analyzer.domain.findings import EvidenceClass
from ux_analyzer.domain.synthesis import (
    EvidenceRef,
    PriorRejection,
    SynthesisAttempt,
    SynthesisFinding,
    SynthesisObjection,
    SynthesisStatus,
)
from ux_analyzer.providers.report_synthesis import (
    AnalystResponse,
    CandidateFinding,
    EvidenceAuditor,
    ReportAnalyst,
)

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "synthesis"
WORK_RUN = "run-138e9c5ae1b43547a7d6a57dd11c7ad9154522eea7a547dccd697d68bc046b7b"

HEURISTIC_METRIC_IDS = (
    "target-discovery-rank",
    "target-below-fold",
    "ambiguous-target",
    "discovery-cost",
)


def _load(name: str) -> Mapping[str, Any]:
    return cast(
        Mapping[str, Any],
        json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8")),
    )


def _corpus(tmp_path: Path) -> EvidenceCorpus:
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
    return EvidenceCorpus(output_root=tmp_path, entries=entries)


def _ref(evidence_id: str, corpus: EvidenceCorpus) -> dict[str, object]:
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


def _heuristic_only_candidate(corpus: EvidenceCorpus) -> CandidateFinding:
    """Attempt 1's claim with every deterministic citation removed.

    Same prose, same harm assertion, but the only evidence left is the four
    heuristic discovery signals: prominence rank, below-fold, ambiguity, and
    discovery cost.
    """

    return CandidateFinding.model_validate(
        {
            "finding_id": "work-showcase-discovery-warning-signals",
            "title": "Heuristic discovery signals warn about the work showcase",
            "issue": (
                "The work-showcase run records heuristic discovery warnings: "
                "ambiguity 1.0, below-fold placement 1.0, target-discovery rank "
                "12, and discovery cost 1.35. Users cannot find the work "
                "showcase entry from the initial state."
            ),
            "impact": (
                "People may spend extra effort identifying the destination and "
                "have to scroll before the work showcase is reachable."
            ),
            "root_cause": (
                "The scored target and the visible link label may not refer to "
                "the same element."
            ),
            "fixes": ["Align the work-showcase link label with the goal wording."],
            "severity": "low",
            "confidence": 0.4,
            "evidence_refs": [
                _ref(f"metric:{WORK_RUN}:{metric_id}", corpus)
                for metric_id in HEURISTIC_METRIC_IDS
            ],
            "severity_justification": (
                "Four heuristic discovery warnings fire for this run and the "
                "goal depends on reaching the work showcase."
            ),
        }
    )


class _RecordingAnalyst:
    def __init__(self, responses: Sequence[object]) -> None:
        self.provider_id = "fixture-provider"
        self.model = "fixture-model"
        self.endpoint_origin = "https://fixture.invalid"
        self.prompt_version = "fixture-analyst-v1"
        self.provider_version = "fixture-v1"
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def analyze(self, *args: Any, **kwargs: Any) -> object:
        self.calls.append(dict(kwargs))
        return (
            self.responses.pop(0) if self.responses else AnalystResponse(complete=True)
        )


class _CountingRole:
    def __init__(self, role: str) -> None:
        self.role = role
        self.provider_id = "fixture-provider"
        self.model = "fixture-model"
        self.endpoint_origin = "https://fixture.invalid"
        self.prompt_version = f"fixture-{role}-v1"
        self.provider_version = "fixture-v1"
        self.calls: list[dict[str, Any]] = []

    def _record(self, args: tuple[Any, ...], **kwargs: Any) -> object:
        from ux_analyzer.providers.report_synthesis import (
            AdjudicationResponse,
            EvidenceAuditResponse,
            PatternReviewResponse,
        )

        self.calls.append({"args": args, **kwargs})
        return {
            "auditor": EvidenceAuditResponse,
            "pattern": PatternReviewResponse,
            "adjudicator": AdjudicationResponse,
        }[self.role].model_validate({"complete": True})

    async def audit(self, *args: Any, **kwargs: Any) -> object:
        return self._record(args, **kwargs)

    async def review(self, *args: Any, **kwargs: Any) -> object:
        return self._record(args, **kwargs)

    async def adjudicate(self, *args: Any, **kwargs: Any) -> object:
        return self._record(args, **kwargs)


def _role_prompt(role: type[Any]) -> str:
    """The composed system prompt for a role, without a transport client."""

    return cast(str, role.__new__(role).prompt)


# ---------------------------------------------------------------------------
# Deterministic analyst-stage gate
# ---------------------------------------------------------------------------


def _domain(candidate: CandidateFinding) -> SynthesisFinding:
    return candidate.to_domain(reviewer_state="candidate")


def test_heuristic_only_harm_claim_is_detected(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    assert _heuristic_only_harm_claim(
        corpus, _domain(_heuristic_only_candidate(corpus))
    )


def test_heuristic_only_gate_needs_a_harm_assertion(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    candidate = _heuristic_only_candidate(corpus)
    descriptive = CandidateFinding.model_validate(
        {
            **candidate.model_dump(mode="json"),
            "issue": (
                "The work-showcase run records ambiguity 1.0, below-fold "
                "placement 1.0, target-discovery rank 12, and discovery cost "
                "1.35 for the scored target."
            ),
            "impact": (
                "Those four values are heuristic estimates produced by the "
                "configured prominence provider."
            ),
        }
    )
    assert _heuristic_only_harm_claim(corpus, _domain(descriptive)) is False


def test_heuristic_only_gate_ignores_negated_harm_language(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    candidate = _heuristic_only_candidate(corpus)
    negated = CandidateFinding.model_validate(
        {
            **candidate.model_dump(mode="json"),
            "issue": (
                "The work-showcase run records ambiguity 1.0, below-fold "
                "placement 1.0, target-discovery rank 12, and discovery cost "
                "1.35. No user was blocked and the run did not fail."
            ),
            "impact": (
                "No user was delayed: the run recorded zero wrong actions, and "
                "the heuristic signals did not cause extra effort."
            ),
        }
    )
    assert _heuristic_only_harm_claim(corpus, _domain(negated)) is False


def test_heuristic_only_gate_ignores_findings_with_deterministic_support(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    candidate = _heuristic_only_candidate(corpus)
    with_observation = CandidateFinding.model_validate(
        {
            **candidate.model_dump(mode="json"),
            "evidence_refs": [
                *[
                    _ref(f"metric:{WORK_RUN}:{metric_id}", corpus)
                    for metric_id in HEURISTIC_METRIC_IDS
                ],
                _ref(f"event:{WORK_RUN}:9", corpus),
            ],
        }
    )
    assert _heuristic_only_harm_claim(corpus, _domain(with_observation)) is False


def test_heuristic_only_claim_is_stopped_before_the_reviewers(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    candidate = _heuristic_only_candidate(corpus)
    analyst = _RecordingAnalyst(
        [AnalystResponse(complete=True, candidate_findings=[candidate])]
    )
    auditor = _CountingRole("auditor")
    pattern = _CountingRole("pattern")
    adjudicator = _CountingRole("adjudicator")
    service = ReportSynthesisService(
        analyst=cast(Any, analyst),
        evidence_auditor=cast(Any, auditor),
        pattern_reviewer=cast(Any, pattern),
        adjudicator=cast(Any, adjudicator),
    )

    attempt = asyncio.run(service.synthesize(corpus))

    assert not attempt.findings
    assert not attempt.candidate_findings
    assert attempt.status is SynthesisStatus.REJECTED
    assert any(
        "heuristic-only harm claim" in limitation for limitation in attempt.limitations
    )
    assert [
        (audit.finding_id, audit.source_role)
        for audit in attempt.rejected_candidate_audits
    ] == [("work-showcase-discovery-warning-signals", "report-analyst")]
    # No reviewer or adjudicator work is spent on the weak claim: both reviewers
    # are handed an empty candidate list and nothing is ever adjudicated.
    assert auditor.calls, "reviewer roles still run their bounded stage"
    assert all(not call["args"][2] for call in auditor.calls), (
        "the evidence auditor must not be handed the weak candidate"
    )
    assert all(not call["args"][2] for call in pattern.calls), (
        "the pattern reviewer must not be handed the weak candidate"
    )


def test_one_observed_record_lets_the_same_claim_through_the_gate(
    tmp_path: Path,
) -> None:
    """The gate is narrow: one deterministic observation is enough support."""

    corpus = _corpus(tmp_path)
    candidate = _heuristic_only_candidate(corpus)
    with_observation = CandidateFinding.model_validate(
        {
            **candidate.model_dump(mode="json"),
            "evidence_refs": [
                *[
                    _ref(f"metric:{WORK_RUN}:{metric_id}", corpus)
                    for metric_id in HEURISTIC_METRIC_IDS
                ],
                _ref(f"event:{WORK_RUN}:9", corpus),
            ],
        }
    )
    analyst = _RecordingAnalyst(
        [AnalystResponse(complete=True, candidate_findings=[with_observation])]
    )
    auditor = _CountingRole("auditor")
    service = ReportSynthesisService(
        analyst=cast(Any, analyst),
        evidence_auditor=cast(Any, auditor),
        pattern_reviewer=cast(Any, _CountingRole("pattern")),
        adjudicator=cast(Any, _CountingRole("adjudicator")),
    )

    attempt = asyncio.run(service.synthesize(corpus))

    assert [finding.finding_id for finding in attempt.candidate_findings] == [
        candidate.finding_id
    ]
    assert auditor.calls, "an observed record moves the claim to the reviewers"
    assert not any(
        "heuristic-only harm claim" in limitation for limitation in attempt.limitations
    )


# ---------------------------------------------------------------------------
# Analyst prompt discipline
# ---------------------------------------------------------------------------


def test_analyst_prompt_requires_observed_behavior_and_names_heuristics() -> None:
    assert ReportAnalyst.prompt_version == "report-analyst-v13"
    prompt = _role_prompt(ReportAnalyst)
    for clause in (
        "Rest every issue and impact claim on observed behavior",
        "Heuristic signals - prominence rank, below-fold counts, ambiguity",
        "they can never be the reason a person was harmed",
        "wrong-actions, backtracks, recovery actions, false success, verified",
        "Zero wrong actions together with verified completion",
        "One required-looking scroll or a required traversal step is expected",
        "Treat it as context, not a prohibition",
    ):
        assert clause in prompt, clause


def test_analyst_prompt_requires_examining_every_scenario() -> None:
    """Examination cannot be skipped when runs succeed.

    This is the clause that stops a clean, fast, verified-success run from
    ending the analysis. Publication validation enforces the coverage; this
    prompt is what tells the analyst to spend its budget on it.
    """

    prompt = _role_prompt(ReportAnalyst)
    for clause in (
        "Examine every scenario",
        "Success is not a reason to stop looking",
        "a fast path is not automatically a good one",
        "Return exactly one scenario_review per scenario_id",
        "the attempt is rejected if any scenario is left unreviewed",
        "no-issue-found is a legitimate and expected outcome",
    ):
        assert clause in prompt, clause


def test_analyst_prompt_forbids_using_success_as_a_reason_to_stop() -> None:
    """The old clause told the model success meant no friction. That is gone.

    It is the single sentence that produced an empty report on a run where
    every scenario verified successfully, so its removal is a behavior
    contract, not copy editing.
    """

    prompt = _role_prompt(ReportAnalyst)
    assert (
        "is evidence that the task completed without observed friction"
        not in prompt
    )
    assert "does not mean the interaction was well designed" in prompt


def test_analyst_prompt_requires_balancing_competing_signals() -> None:
    """No single metric may stand in for design judgment.

    Step count is the obvious offender: minimizing it rewards a dense screen
    of small targets, which is cheap to click and expensive to understand.
    """

    prompt = _role_prompt(ReportAnalyst)
    for clause in (
        "Weigh several signals against each other",
        "any one of them alone can point the wrong way",
        "how much attention the persona spent before acting",
        "how many competing controls were on screen at the moment of decision",
        "say in the finding which signals you balanced",
    ):
        assert clause in prompt, clause


def test_analyst_prompt_separates_improvement_from_harm() -> None:
    """Improvement claims the task worked; they may not smuggle in harm."""

    prompt = _role_prompt(ReportAnalyst)
    for clause in (
        "Report a balance of trade-offs as finding_kind improvement",
        "It must not claim the persona was harmed, blocked, delayed, or misled",
        "if it does, it is a ux-issue",
    ):
        assert clause in prompt, clause


def test_auditor_prompt_objects_to_heuristic_only_contradicted_claims() -> None:
    assert EvidenceAuditor.prompt_version == "report-evidence-auditor-v6"
    prompt = _role_prompt(EvidenceAuditor)
    for clause in (
        "rests only on heuristic signals",
        "zero wrong actions, zero backtracks, zero recovery actions, verified",
        "Cite the counterevidence that decides it",
        "A required scroll or a required traversal step is expected behavior",
    ):
        assert clause in prompt, clause


def test_analyst_receives_prior_rejections_in_its_role_input(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    rejections = (
        PriorRejection(
            finding_id="work-showcase-discovery-warning-signals",
            title="Heuristic discovery signals warn about the work showcase",
            reasons=(
                "The heuristic signals are model estimates and the run records "
                "verified completion with zero wrong actions.",
            ),
            evidence_ids=(
                f"metric:{WORK_RUN}:target-below-fold",
                f"metric:{WORK_RUN}:discovery-cost",
            ),
            attempt_id="synthesis-2026-09-24T192500Z-98bf4207803b-1",
        ),
    )
    analyst = _RecordingAnalyst([AnalystResponse(complete=True)])
    service = ReportSynthesisService(
        analyst=cast(Any, analyst),
        evidence_auditor=cast(Any, _CountingRole("auditor")),
        pattern_reviewer=cast(Any, _CountingRole("pattern")),
        adjudicator=cast(Any, _CountingRole("adjudicator")),
        prior_rejections=rejections,
    )

    asyncio.run(service.synthesize(corpus))

    payload = analyst.calls[0]["prior_rejections"]
    assert len(payload) == 1
    entry = cast(Mapping[str, Any], payload[0])
    assert entry["finding_id"] == "work-showcase-discovery-warning-signals"
    assert list(entry["evidence_ids"]) == [
        f"metric:{WORK_RUN}:target-below-fold",
        f"metric:{WORK_RUN}:discovery-cost",
    ]
    assert "not a prohibition" in cast(str, entry["policy"])


def test_prior_rejection_context_drops_evidence_the_corpus_lacks(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    rejections = (
        PriorRejection(
            finding_id="work-showcase-discovery-warning-signals",
            title="Heuristic discovery signals warn about the work showcase",
            reasons=("The heuristic signals could not establish user harm.",),
            evidence_ids=(
                f"metric:{WORK_RUN}:target-below-fold",
                "metric:run-that-no-longer-exists:target-below-fold",
            ),
            attempt_id="synthesis-2026-09-24T192500Z-98bf4207803b-1",
        ),
    )
    analyst = _RecordingAnalyst([AnalystResponse(complete=True)])
    service = ReportSynthesisService(
        analyst=cast(Any, analyst),
        evidence_auditor=cast(Any, _CountingRole("auditor")),
        pattern_reviewer=cast(Any, _CountingRole("pattern")),
        adjudicator=cast(Any, _CountingRole("adjudicator")),
        prior_rejections=rejections,
    )

    asyncio.run(service.synthesize(corpus))

    entry = cast(Mapping[str, Any], analyst.calls[0]["prior_rejections"][0])
    assert list(entry["evidence_ids"]) == [f"metric:{WORK_RUN}:target-below-fold"]


def test_prior_rejection_without_a_reason_is_not_forwarded(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    analyst = _RecordingAnalyst([AnalystResponse(complete=True)])
    service = ReportSynthesisService(
        analyst=cast(Any, analyst),
        evidence_auditor=cast(Any, _CountingRole("auditor")),
        pattern_reviewer=cast(Any, _CountingRole("pattern")),
        adjudicator=cast(Any, _CountingRole("adjudicator")),
        prior_rejections=(
            PriorRejection(
                finding_id="silent-rejection",
                title="A finding that was rejected without a recorded reason",
            ),
        ),
    )

    asyncio.run(service.synthesize(corpus))

    assert analyst.calls[0]["prior_rejections"] == ()


def test_prior_rejection_is_not_a_hard_blacklist(tmp_path: Path) -> None:
    """A rejected subject stays proposeable when the analyst strengthens it."""

    corpus = _corpus(tmp_path)
    rejections = (
        PriorRejection(
            finding_id="work-showcase-discovery-warning-signals",
            title="Heuristic discovery signals warn about the work showcase",
            reasons=("The claim rested on heuristic signals alone.",),
            evidence_ids=(f"metric:{WORK_RUN}:target-below-fold",),
            attempt_id="synthesis-2026-09-24T192500Z-98bf4207803b-1",
        ),
    )
    candidate = CandidateFinding.model_validate(
        {
            "finding_id": "work-showcase-discovery-warning-signals",
            "title": "Initial captures record the work-showcase controls as occluded",
            "issue": (
                "In the initial viewport of three runs every listed actionable "
                "control is recorded at zero visibility and full occlusion while "
                "a non-actionable message keeps full visibility."
            ),
            "impact": (
                "The captured state removes every visible route to the "
                "work-showcase entry for the duration of that state."
            ),
            "root_cause": "The covering element is not identified by the record.",
            "fixes": ["Expose one actionable route in the first settled state."],
            "severity": "low",
            "confidence": 0.6,
            "evidence_refs": [_ref(f"event:{WORK_RUN}:9", corpus)],
            "severity_justification": (
                "Three deterministic initial captures record the same state and "
                "one run later completes the task."
            ),
        }
    )
    analyst = _RecordingAnalyst(
        [AnalystResponse(complete=True, candidate_findings=[candidate])]
    )
    service = ReportSynthesisService(
        analyst=cast(Any, analyst),
        evidence_auditor=cast(Any, _CountingRole("auditor")),
        pattern_reviewer=cast(Any, _CountingRole("pattern")),
        adjudicator=cast(Any, _CountingRole("adjudicator")),
        prior_rejections=rejections,
    )

    attempt = asyncio.run(service.synthesize(corpus))

    assert [finding.finding_id for finding in attempt.candidate_findings] == [
        "work-showcase-discovery-warning-signals"
    ]


# ---------------------------------------------------------------------------
# Prior rejections derived from recorded attempts
# ---------------------------------------------------------------------------


def _attempt(
    attempt_id: str,
    *,
    rejected: Sequence[SynthesisFinding] = (),
    objections: Sequence[SynthesisObjection] = (),
) -> SynthesisAttempt:
    return SynthesisAttempt(
        attempt_id=attempt_id,
        status=SynthesisStatus.REJECTED,
        corpus_digest="a" * 64,
        rejected_findings=tuple(rejected),
        objections=tuple(objections),
    )


def _finding(finding_id: str, *, notes: Sequence[str] = ()) -> SynthesisFinding:
    return SynthesisFinding(
        finding_id=finding_id,
        title=f"Rejected finding {finding_id}",
        issue="The issue as recorded.",
        impact="The impact as recorded.",
        root_cause="The root cause as recorded.",
        fixes=("A recorded fix.",),
        severity="low",
        confidence=0.4,
        evidence_refs=(
            EvidenceRef(f"metric:{WORK_RUN}:target-below-fold", "metric", WORK_RUN),
        ),
        reviewer_state="not-established",
        severity_justification="The recorded severity justification.",
        reviewer_notes=tuple(notes),
    )


def _objection(finding_id: str, message: str) -> SynthesisObjection:
    return SynthesisObjection(
        objection_id=f"report-evidence-auditor:{finding_id}-001",
        finding_id=finding_id,
        severity="material",
        objection_type="factual-support",
        message=message,
        reviewer_role="report-evidence-auditor",
    )


def test_prior_rejections_are_derived_from_reviewed_reasons() -> None:
    attempts = (
        _attempt(
            "synthesis-a",
            rejected=(
                _finding(
                    "work-showcase",
                    notes=("adjudicator published no surviving finding",),
                ),
            ),
            objections=(
                _objection(
                    "work-showcase",
                    "The heuristic signals are model estimates, not observed harm.",
                ),
            ),
        ),
        _attempt("synthesis-b", rejected=(_finding("silent-rejection"),)),
    )

    rejections = ReportSynthesisService.prior_rejections_from_attempts(attempts)

    assert [item.finding_id for item in rejections] == ["work-showcase"]
    entry = rejections[0]
    assert entry.attempt_id == "synthesis-a"
    assert entry.reasons[0] == (
        "The heuristic signals are model estimates, not observed harm."
    )
    assert "adjudicator published no surviving finding" in entry.reasons
    assert entry.evidence_ids == (f"metric:{WORK_RUN}:target-below-fold",)


def test_prior_rejections_prefer_the_newest_attempt_per_finding() -> None:
    attempts = (
        _attempt(
            "synthesis-a",
            rejected=(_finding("work-showcase", notes=("stale reason",)),),
        ),
        _attempt(
            "synthesis-b",
            rejected=(_finding("work-showcase", notes=("fresh reason",)),),
        ),
    )

    rejections = ReportSynthesisService.prior_rejections_from_attempts(attempts)

    assert len(rejections) == 1
    assert rejections[0].attempt_id == "synthesis-b"
    assert rejections[0].reasons == ("fresh reason",)


def test_prior_rejections_are_bounded() -> None:
    attempts = tuple(
        _attempt(
            f"synthesis-{index}",
            rejected=(_finding(f"finding-{index}", notes=("a reason",)),),
        )
        for index in range(MAX_PRIOR_REJECTIONS + 5)
    )

    rejections = ReportSynthesisService.prior_rejections_from_attempts(attempts)

    assert len(rejections) == MAX_PRIOR_REJECTIONS
    assert rejections[0].finding_id == f"finding-{MAX_PRIOR_REJECTIONS + 4}"


def test_prior_rejection_rejects_malformed_records() -> None:
    with pytest.raises(ValueError, match="rejected finding title"):
        PriorRejection(finding_id="a", title="t" * 241, reasons=("r",))
    with pytest.raises(ValueError, match="duplicate evidence ID"):
        PriorRejection(
            finding_id="a",
            title="t",
            evidence_ids=("metric:x:y", "metric:x:y"),
        )


def test_service_rejects_non_prior_rejection_context() -> None:
    with pytest.raises(TypeError, match="prior_rejections"):
        ReportSynthesisService(
            analyst=cast(Any, _RecordingAnalyst([])),
            prior_rejections=cast(Sequence[PriorRejection], [{"finding_id": "a"}]),
        )


def test_only_the_analyst_receives_rejection_context(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    analyst = _RecordingAnalyst([AnalystResponse(complete=True)])
    auditor = _CountingRole("auditor")
    service = ReportSynthesisService(
        analyst=cast(Any, analyst),
        evidence_auditor=cast(Any, auditor),
        pattern_reviewer=cast(Any, _CountingRole("pattern")),
        adjudicator=cast(Any, _CountingRole("adjudicator")),
        prior_rejections=(
            PriorRejection(
                finding_id="work-showcase",
                title="Rejected finding",
                reasons=("A recorded reason.",),
            ),
        ),
    )

    asyncio.run(service.synthesize(corpus))

    assert "prior_rejections" in analyst.calls[0]
    assert auditor.calls and all(
        "prior_rejections" not in call for call in auditor.calls
    )
