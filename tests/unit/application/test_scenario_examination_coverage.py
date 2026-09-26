"""Examination coverage is a publication invariant, not a prompt request.

The walkthrough must examine every scenario whatever the runs did, and must not
let one metric stand in for design judgment. These tests pin both, plus the
provenance rules that keep a review from asserting something it did not look at.
"""

from __future__ import annotations

import pytest

from ux_analyzer.domain.synthesis import (
    FindingKind,
    ReviewDisposition,
    ScenarioReview,
    SynthesisAttempt,
    SynthesisStatus,
)
from ux_analyzer.ports.report_synthesis import ScenarioReviewReport


def test_improvement_is_a_distinct_finding_kind() -> None:
    """Improvement is not harm and not a scenario defect."""

    assert FindingKind.IMPROVEMENT == "improvement"
    assert FindingKind.IMPROVEMENT is not FindingKind.UX_ISSUE
    assert FindingKind.IMPROVEMENT is not FindingKind.SCENARIO_DEFECT
    assert ReviewDisposition.IMPROVEMENT is ReviewDisposition("improvement")


def test_review_with_a_finding_must_name_examined_evidence() -> None:
    """A review that found something has to say what it looked at."""

    with pytest.raises(ValueError, match="must name examined evidence"):
        ScenarioReview(
            scenario_id="discover-ratings",
            disposition=ReviewDisposition.IMPROVEMENT,
        )


def test_no_issue_found_review_may_carry_no_evidence() -> None:
    """'Examined and found nothing' is a legitimate, evidence-free outcome."""

    review = ScenarioReview(
        scenario_id="discover-ratings",
        disposition=ReviewDisposition.NO_ISSUE_FOUND,
        signals_weighed=("attention cost", "action count"),
        note="The target was the first element observed on the first viewport.",
    )
    assert review.evidence_ids == ()
    assert review.disposition is ReviewDisposition.NO_ISSUE_FOUND


def test_review_must_name_the_signals_it_balanced() -> None:
    """Weighing nothing is not a review.

    Without the balance discipline a single metric - step count, most often -
    stands in for design judgment, and step count rewards exactly the dense,
    low-attention-target layouts that are hardest to use.
    """

    with pytest.raises(ValueError, match="signals_weighed"):
        ScenarioReviewReport(
            scenario_id="discover-ratings",
            disposition=ReviewDisposition.NO_ISSUE_FOUND,
            evidence_ids=["e1"],
            signals_weighed=[],
        )


def test_attempt_carries_one_review_per_scenario() -> None:
    """Two reviews for one scenario is an inconsistent record."""

    review = ScenarioReview(
        scenario_id="discover-ratings",
        disposition=ReviewDisposition.NO_ISSUE_FOUND,
        signals_weighed=("attention cost",),
    )
    with pytest.raises(ValueError, match="one review per scenario"):
        SynthesisAttempt(
            attempt_id="a-1",
            status=SynthesisStatus.NO_ISSUES,
            scenario_reviews=(review, review),
        )


def test_attempt_reviews_must_be_review_objects() -> None:
    with pytest.raises(TypeError, match="ScenarioReview values"):
        SynthesisAttempt(
            attempt_id="a-1",
            status=SynthesisStatus.NO_ISSUES,
            scenario_reviews=("not-a-review",),  # type: ignore[arg-type]
        )


def test_every_scenario_appears_in_a_healthy_attempt() -> None:
    """The shape publication validation requires: full coverage, no gaps."""

    attempt = SynthesisAttempt(
        attempt_id="a-1",
        status=SynthesisStatus.NO_ISSUES,
        scenario_reviews=(
            ScenarioReview(
                scenario_id="discover-ratings",
                disposition=ReviewDisposition.NO_ISSUE_FOUND,
                signals_weighed=("attention cost", "action count"),
                note="Nothing better than the recorded path is available.",
            ),
            ScenarioReview(
                scenario_id="contact-author",
                disposition=ReviewDisposition.IMPROVEMENT,
                evidence_ids=("event:run-a:14",),
                signals_weighed=("attention cost", "competing-control density"),
                note="Reachable in one click, but only after scanning nine "
                "viewports of competing controls.",
            ),
        ),
    )
    corpus_scenarios = {"discover-ratings", "contact-author"}
    reviewed = {review.scenario_id for review in attempt.scenario_reviews}
    assert reviewed == corpus_scenarios
    assert attempt.scenario_reviews[1].disposition is (
        ReviewDisposition.IMPROVEMENT
    )


def test_review_note_length_is_bounded() -> None:
    with pytest.raises(ValueError, match="note is too long"):
        ScenarioReview(
            scenario_id="discover-ratings",
            disposition=ReviewDisposition.NO_ISSUE_FOUND,
            signals_weighed=("attention cost",),
            note="x" * 1025,
        )
