from __future__ import annotations

from dataclasses import replace

import pytest

from ux_analyzer.application.evaluation import (
    DiscoveryCostBreakdown,
    EvaluationTarget,
    RunMetrics,
)
from ux_analyzer.domain.benchmark import ExperimentPolicy
from ux_analyzer.domain.findings import (
    Evidence,
    EvidenceClass,
    UnsupportedHumanClaimError,
)
from ux_analyzer.domain.run import RunOutcomeKind
from ux_analyzer.providers.finding_rules import (
    FindingCategory,
    FindingRule,
    FindingRuleSet,
    findings_for_run,
)


def _metrics():
    return RunMetrics(
        run_id="run-1",
        seed=1,
        scenario_id="invite",
        application_version_id="defective",
        persona_id="persona",
        policy=ExperimentPolicy.PROGRESSIVE_PROMINENCE_SCENT,
        target=EvaluationTarget("target"),
        target_discovery_rank=6,
        inspected_elements=5,
        inspected_regions=4,
        scrolls=3,
        wrong_actions=3,
        backtracks=2,
        verified_completion=False,
        claimed_completion=True,
        false_success=True,
        abandoned=True,
        target_prominence=0.1,
        target_scent=0.2,
        strongest_competing_scent=0.9,
        target_below_fold=True,
        unexpected_hierarchy=True,
        ambiguous_target=True,
        navigation_depth=5,
        feedback_observed=False,
        recovery_actions=0,
        recovery_success=False,
        outcome=RunOutcomeKind.AGENT_ABANDONED,
        evidence=(),
        metrics=(),
        discovery_cost=DiscoveryCostBreakdown(
            inspection_cost=5.0,
            region_cost=4.0,
            scroll_cost=3.0,
            wrong_action_cost=3.0,
            backtrack_cost=2.0,
            uncertainty_cost=1.0,
            abandonment_penalty=1.0,
        ),
    )


def test_finding_rules_emit_required_typed_categories_with_evidence() -> None:
    findings = findings_for_run(_metrics())
    categories = {finding.category for finding in findings}

    assert {
        FindingCategory.WEAK_TARGET_PROMINENCE.value,
        FindingCategory.WEAK_SCENT.value,
        FindingCategory.STRONG_MISLEADING_ALTERNATIVE.value,
        FindingCategory.UNEXPECTED_HIERARCHY.value,
        FindingCategory.AMBIGUOUS_ICON_LABEL.value,
        FindingCategory.EXCESSIVE_DEPTH.value,
        FindingCategory.TARGET_BELOW_FOLD.value,
        FindingCategory.MISSING_FEEDBACK.value,
        FindingCategory.WRONG_ACTION_BURDEN.value,
        FindingCategory.POOR_RECOVERY.value,
    } <= categories
    assert all(finding.evidence_ids for finding in findings)
    assert all(
        finding.evidence_class
        in {EvidenceClass.DETERMINISTIC_FACT, EvidenceClass.MODEL_ESTIMATE}
        for finding in findings
    )
    by_category = {finding.category: finding for finding in findings}
    for category in (
        FindingCategory.WEAK_TARGET_PROMINENCE,
        FindingCategory.UNEXPECTED_HIERARCHY,
        FindingCategory.AMBIGUOUS_ICON_LABEL,
        FindingCategory.EXCESSIVE_DEPTH,
        FindingCategory.TARGET_BELOW_FOLD,
        FindingCategory.MISSING_FEEDBACK,
        FindingCategory.WRONG_ACTION_BURDEN,
        FindingCategory.POOR_RECOVERY,
    ):
        assert (
            by_category[category.value].evidence_class is EvidenceClass.MODEL_ESTIMATE
        )


def test_finding_rule_rejects_unsupported_human_claim_class() -> None:
    with pytest.raises(UnsupportedHumanClaimError):
        FindingRule(
            category=FindingCategory.WEAK_SCENT,
            severity="high",
            evidence_class=EvidenceClass.UNSUPPORTED_HUMAN_CLAIM,
            predicate=lambda _: True,
        )


def test_finding_rule_set_does_not_put_human_claim_in_scorecard() -> None:
    with pytest.raises(UnsupportedHumanClaimError):
        FindingRuleSet.from_rules(
            (
                FindingRule(
                    category=FindingCategory.WEAK_SCENT,
                    severity="high",
                    evidence_class=EvidenceClass.UNSUPPORTED_HUMAN_CLAIM,
                    predicate=lambda _: True,
                ),
            )
        )


def test_findings_reference_emitted_metric_evidence() -> None:
    metrics = _metrics()
    metrics = replace(
        metrics,
        evidence=(
            Evidence(
                evidence_id="run-1:target-prominence",
                evidence_class=EvidenceClass.MODEL_ESTIMATE,
                description="Target prominence evidence.",
            ),
        ),
    )

    finding = next(
        item
        for item in findings_for_run(metrics)
        if item.category == FindingCategory.WEAK_TARGET_PROMINENCE.value
    )

    assert finding.evidence_ids == ("run-1:target-prominence",)


def test_findings_explain_cause_metrics_references_actions_and_replay() -> None:
    metrics = replace(
        _metrics(),
        viewport_ids=("viewport-1", "viewport-2"),
        element_ids=("target",),
        action_sequence=("interact-with-element target: failed", "back: succeeded"),
    )

    finding = next(
        item
        for item in findings_for_run(metrics)
        if item.category == FindingCategory.WEAK_TARGET_PROMINENCE.value
    )

    assert finding.title == "Target is visually easy to miss"
    assert "0.1" in finding.cause
    assert finding.run_ids == ("run-1",)
    assert finding.viewport_ids == ("viewport-1", "viewport-2")
    assert finding.element_ids == ("target",)
    assert finding.supporting_metrics == {"target-prominence": 0.1}
    assert finding.action_sequence == (
        "interact-with-element target: failed",
        "back: succeeded",
    )
    assert finding.replay_links == ("#run=run-1&element=target",)


def test_prominence_finding_cites_provider_stage_profiles_and_limitations() -> None:
    metrics = replace(
        _metrics(),
        prominence_provider_id="foveacast",
        active_search_stage="persistent",
        prominence_model_id="foveacast-v0.2.0",
        prominence_model_version="v0.2.0",
        prominence_profile_event_ids=("event-4",),
        prominence_operational_event_ids=("event-5",),
        target_prominence_profiles={
            "immediate": 0.1,
            "early": 0.2,
            "eventual": 0.3,
        },
        target_prominence_sources={
            "immediate": "foveacast-v0.2.0",
            "early": "foveacast-v0.2.0",
            "eventual": "foveacast-v0.2.0",
        },
    )

    finding = next(
        item
        for item in findings_for_run(metrics)
        if item.category == FindingCategory.WEAK_TARGET_PROMINENCE.value
    )

    text = f"{finding.generated_explanation} {finding.cause}"
    assert "foveacast" in text
    assert "persistent" in text
    assert "event-4" in text
    assert "event-5" in text
    assert "foveacast-v0.2.0" in text
    assert "1s" in text
    assert "3s" in text
    assert "7s" in text
    assert "Heuristic prominence estimate for target." not in text
    assert finding.limitations


def test_prominence_finding_redacts_secret_like_provenance_identifiers() -> None:
    metrics = replace(
        _metrics(),
        prominence_provider_id="foveacast",
        prominence_model_id="api_key=provider-secret",
        prominence_model_version="secret-token",
        prominence_profile_event_ids=("event-4",),
        prominence_operational_event_ids=("event-5",),
        target_prominence_profiles={"immediate": 0.1},
        target_prominence_sources={"immediate": "token=provider-secret"},
    )

    finding = next(
        item
        for item in findings_for_run(metrics)
        if item.category == FindingCategory.WEAK_TARGET_PROMINENCE.value
    )

    text = f"{finding.generated_explanation} {finding.cause}"
    assert "provider-secret" not in text
    assert "api_key=" not in text


def test_prominence_finding_redacts_secret_like_fallback_reason() -> None:
    metrics = replace(
        _metrics(),
        prominence_fallback=True,
        prominence_fallback_reason="api_key=provider-secret",
    )

    finding = next(
        item
        for item in findings_for_run(metrics)
        if item.category == FindingCategory.WEAK_TARGET_PROMINENCE.value
    )

    text = f"{finding.generated_explanation} {finding.cause}"
    assert "provider-secret" not in text
    assert "api_key=" not in text
