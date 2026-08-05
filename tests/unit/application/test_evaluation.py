from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from ux_analyzer.application.evaluation import (
    DiscoveryCostBreakdown,
    DiscoveryCostConfig,
    EvaluationEvidenceUnavailable,
    EvaluationTarget,
    RunEvaluationInputs,
    aggregate_cell,
    compare_variants,
    evaluate_experiment_results,
    evaluate_run,
    evaluation_inputs_for,
    evaluation_target_for,
    interval_summary,
    persisted_comparison_sample_is_valid,
)
from ux_analyzer.application.run_agent import (
    ProminenceEvidence,
    RunEvidence,
    RunResult,
    ScentEvidence,
)
from ux_analyzer.domain.attention import (
    Back,
    FullScent,
    InteractWithElement,
    ProgressiveObservation,
    Scroll,
)
from ux_analyzer.domain.benchmark import (
    ApplicationVersion,
    ApplicationVersionKind,
    Budget,
    ExperimentPolicy,
    FixtureInputs,
    Persona,
    Scenario,
    ScenarioEvaluationTarget,
    VisibleResultVerifierSpec,
)
from ux_analyzer.domain.findings import (
    Evidence,
    EvidenceClass,
    UnsupportedHumanClaimError,
)
from ux_analyzer.domain.interface import (
    BoundingBox,
    ElementSnapshot,
    PrivateExecutionReference,
    ViewportSnapshot,
)
from ux_analyzer.domain.run import (
    ActionExecuted,
    ActionProposed,
    AgentAbandoned,
    ArtifactChecksum,
    ObservationRecorded,
    ProviderManifest,
    RunSpec,
    RunStarted,
    RunState,
    RunTerminated,
    VerificationResult,
    VerifiedSuccess,
    ViewportCaptured,
)
from ux_analyzer.providers.prominence import ProminenceResult


def _spec(version: ApplicationVersion) -> RunSpec:
    scenario = Scenario(
        id="invite",
        name="Invite",
        goal="Invite teammate",
        application_version_ids=("defective", "improved"),
        start_state="dashboard",
        fixture_inputs=FixtureInputs(values={"email": "person@example.com"}),
        budget=Budget(
            max_steps=20,
            max_observations=5,
            max_interactions=5,
            timeout_seconds=10,
        ),
        verifier=VisibleResultVerifierSpec(type="visible-result", text="Sent"),
        safeguards=(),
        eligible_persona_ids=("persona",),
        expected_evidence=(),
        evaluation_target=ScenarioEvaluationTarget(
            labels_by_version={"defective": "Target", "improved": "Target"},
            role="button",
        ),
    )
    return RunSpec(
        run_id=f"run-{version.id}",
        seed=7,
        scenario=scenario,
        application_version=version,
        persona=Persona(
            id="persona",
            name="Persona",
            working_memory_capacity=3,
            initial_confidence=0.5,
            initial_frustration=0.0,
            abandonment_threshold=0.9,
            attention_temperature=1.0,
        ),
        policy=ExperimentPolicy.PROGRESSIVE_PROMINENCE_SCENT,
        config_digest="config-sha",
    )


def _element(element_id: str, *, viewport_id: str = "viewport-1") -> ElementSnapshot:
    return ElementSnapshot(
        id=element_id,
        role="button",
        label=element_id.title(),
        bounds=BoundingBox(x=10, y=10, width=100, height=40),
        visibility_fraction=1.0,
        actionable=True,
        provider_id="fixture",
        execution_reference=PrivateExecutionReference(
            provider_id="fixture", viewport_id=viewport_id, token=element_id
        ),
    )


def _result(version: ApplicationVersion) -> RunResult:
    spec = _spec(version)
    snapshot = ViewportSnapshot(
        id="viewport-1",
        provider_id="fixture",
        elements=(_element("competitor"), _element("target")),
    )
    state = RunState.initial(spec).apply(RunStarted(run_id=spec.run_id))
    state = state.apply(ViewportCaptured(snapshot=snapshot))
    state = state.apply(
        ObservationRecorded(
            observation=ProgressiveObservation.from_snapshot(
                snapshot, newly_revealed_ids=("competitor", "target"), region_id=None
            )
        )
    )
    state = state.apply(ActionProposed(action=Scroll()))
    state = state.apply(
        ActionExecuted(action=Scroll(), viewport_id=snapshot.id, succeeded=True)
    )
    state = state.apply(ActionProposed(action=Back()))
    state = state.apply(
        ActionExecuted(action=Back(), viewport_id=snapshot.id, succeeded=True)
    )
    verification = VerificationResult(verified=True, evidence_ids=("verify-1",))
    state = state.apply(
        RunTerminated(
            outcome=VerifiedSuccess(),
            verification=verification,
            provider_manifests=(
                ProviderManifest(
                    provider_id="fixture",
                    role="observation",
                    model_id=None,
                    endpoint_origin="fixture",
                    version="1",
                ),
            ),
            configuration_digest=spec.config_digest,
            artifact_checksums=(ArtifactChecksum(path="timeline", sha256="sha"),),
        )
    )
    return RunResult(
        run_id=spec.run_id,
        outcome=VerifiedSuccess(),
        verification=verification,
        agent_claimed_success=False,
        state=state,
    )


def test_evaluate_run_reports_metrics_and_preserves_cost_components() -> None:
    result = _result(
        ApplicationVersion(
            id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
        )
    )
    metrics = evaluate_run(
        result,
        EvaluationTarget(element_id="target"),
        inputs=RunEvaluationInputs(
            prominence_scores={"target": 0.2, "competitor": 0.8},
            scent_scores={"target": 0.3, "competitor": 0.9},
            target_below_fold=True,
            feedback_observed=False,
            navigation_depth=4,
        ),
        cost_config=DiscoveryCostConfig(
            inspection_cost=2.0,
            region_cost=3.0,
            scroll_cost=4.0,
            wrong_action_cost=5.0,
            backtrack_cost=6.0,
            uncertainty_cost=7.0,
            abandonment_penalty=8.0,
        ),
    )

    assert metrics.target_discovery_rank == 2
    assert metrics.scrolls == 1
    assert metrics.backtracks == 1
    assert metrics.verified_completion is True
    assert metrics.target_scent == 0.3
    assert metrics.strongest_competing_scent == 0.9
    assert metrics.discovery_cost.components == pytest.approx(
        {
            "inspection_cost": 0.0,
            "region_cost": 0.0,
            "scroll_cost": 4.0,
            "wrong_action_cost": 0.0,
            "backtrack_cost": 6.0,
            "uncertainty_cost": 3.5,
            "abandonment_penalty": 0.0,
        }
    )
    assert metrics.discovery_cost.total == pytest.approx(13.5)
    assert metrics.evidence
    assert metrics.metric("discovery-cost").evidence_ids
    assert metrics.metric("target-discovery-rank").evidence_class is (
        EvidenceClass.MODEL_ESTIMATE
    )
    assert metrics.metric("target-prominence").evidence_class is (
        EvidenceClass.MODEL_ESTIMATE
    )
    assert metrics.metric("target-below-fold").evidence_class is (
        EvidenceClass.MODEL_ESTIMATE
    )


def test_aggregate_cell_provides_median_interval_and_reproducibility() -> None:
    version = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )
    first = evaluate_run(_result(version), EvaluationTarget("target"))
    second = replace(first, run_id="run-second", seed=8, scrolls=3)

    aggregate = aggregate_cell((first, second))

    assert aggregate.run_count == 2
    assert aggregate.metric_summaries["scrolls"].median == 2.0
    assert aggregate.metric_summaries["scrolls"].interval.lower == pytest.approx(1.05)
    assert aggregate.metric_summaries["scrolls"].interval.upper == pytest.approx(2.95)
    assert aggregate.reproducibility.value in {"seeded", "reproducible"}


def test_interval_summary_uses_requested_central_interval() -> None:
    summary = interval_summary(
        tuple(float(value) for value in range(11)), confidence=0.8
    )

    assert summary.median == pytest.approx(5)
    assert summary.lower == pytest.approx(1)
    assert summary.upper == pytest.approx(9)


@pytest.mark.parametrize(
    "target",
    (
        EvaluationTarget("missing"),
        EvaluationTarget("target", role="link"),
        EvaluationTarget("target", region_id="missing-region"),
    ),
)
def test_invalid_target_references_are_explicitly_unavailable(
    target: EvaluationTarget,
) -> None:
    version = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )

    with pytest.raises(
        EvaluationEvidenceUnavailable, match="evaluation evidence unavailable"
    ):
        evaluate_run(_result(version), target)


def test_compare_variants_requires_paired_seeds_and_applies_directional_gate() -> None:
    defective = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )
    improved = ApplicationVersion(
        id="improved", kind=ApplicationVersionKind.IMPROVED, label="Improved"
    )
    baseline = evaluate_run(_result(defective), EvaluationTarget("target"))
    candidate = evaluate_run(_result(improved), EvaluationTarget("target"))
    candidate = replace(
        candidate,
        run_id="run-improved",
        discovery_cost=DiscoveryCostBreakdown(
            inspection_cost=0.0,
            region_cost=0.0,
            scroll_cost=0.0,
            wrong_action_cost=0.0,
            backtrack_cost=0.0,
            uncertainty_cost=0.0,
            abandonment_penalty=0.0,
        ),
    )

    comparison = compare_variants((baseline,), (candidate,))

    assert comparison.gate.passed is True
    assert comparison.gate.discovery_cost_decreased is True
    assert comparison.paired_seeds == (7,)
    assert comparison.paired_model_trials == (0,)


def test_model_trials_form_distinct_evaluation_cells() -> None:
    version = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )
    first = evaluate_run(_result(version), EvaluationTarget("target"))
    second = replace(first, run_id="run-trial-1", model_trial=1)

    cells = aggregate_cell((first,))
    trial_cells = evaluate_experiment_results(
        (
            replace(_result(version), metrics=first),
            replace(_result(version), run_id="run-trial-1", metrics=second),
        )
    ).cell_aggregates

    assert cells.model_trial == 0
    assert len(trial_cells) == 2
    assert {cell.model_trial for cell in trial_cells} == {0, 1}


def test_completion_improvement_dominates_effort_of_early_abandonment() -> None:
    defective = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )
    improved = ApplicationVersion(
        id="improved", kind=ApplicationVersionKind.IMPROVED, label="Improved"
    )
    baseline = replace(
        evaluate_run(_result(defective), EvaluationTarget("target")),
        verified_completion=False,
        abandoned=True,
        outcome="agent-abandoned",
        wrong_actions=0,
        backtracks=0,
        discovery_cost=DiscoveryCostBreakdown(0, 0, 0, 0, 0, 0, 1),
    )
    candidate = replace(
        evaluate_run(_result(improved), EvaluationTarget("target")),
        run_id="run-improved",
        verified_completion=True,
        wrong_actions=2,
        backtracks=1,
        discovery_cost=DiscoveryCostBreakdown(1, 1, 0, 2, 1, 0, 0),
    )

    comparison = compare_variants((baseline,), (candidate,))

    assert comparison.gate.passed
    assert comparison.gate.verified_completion_rate_not_regressed
    assert not comparison.gate.discovery_cost_decreased
    assert not comparison.gate.wrong_action_burden_not_increased
    assert not comparison.gate.backtrack_burden_not_increased
    assert comparison.gate.reasons == ()


def test_completion_dominance_does_not_hide_jointly_completed_regression() -> None:
    defective = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )
    improved = ApplicationVersion(
        id="improved", kind=ApplicationVersionKind.IMPROVED, label="Improved"
    )
    baseline_abandoned = replace(
        evaluate_run(_result(defective), EvaluationTarget("target")),
        verified_completion=False,
        abandoned=True,
        outcome="agent-abandoned",
        discovery_cost=DiscoveryCostBreakdown(0, 0, 0, 0, 0, 0, 1),
    )
    improved_completed = replace(
        evaluate_run(_result(improved), EvaluationTarget("target")),
        run_id="improved-seed-7",
        discovery_cost=DiscoveryCostBreakdown(2, 0, 0, 0, 0, 0, 0),
    )
    baseline_completed = replace(
        evaluate_run(_result(defective), EvaluationTarget("target")),
        run_id="baseline-seed-8",
        seed=8,
        backtracks=0,
        discovery_cost=DiscoveryCostBreakdown(1, 0, 0, 0, 0, 0, 0),
    )
    improved_regressed = replace(
        evaluate_run(_result(improved), EvaluationTarget("target")),
        run_id="improved-seed-8",
        seed=8,
        wrong_actions=1,
        backtracks=1,
        discovery_cost=DiscoveryCostBreakdown(2, 0, 0, 1, 1, 0, 0),
    )

    comparison = compare_variants(
        (baseline_abandoned, baseline_completed),
        (improved_completed, improved_regressed),
    )

    assert not comparison.gate.passed
    assert comparison.gate.reasons == (
        "paired median discovery cost did not decrease",
        "paired median wrong-action burden increased",
        "paired median backtrack burden increased",
    )


def test_model_dependent_inputs_are_not_reported_as_seed_only() -> None:
    version = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )

    result = _result(version)
    result = replace(
        result,
        state=replace(
            result.state,
            spec=replace(result.state.spec, model_trial=3),
        ),
    )
    metrics = evaluate_run(
        result,
        EvaluationTarget("target"),
        inputs=RunEvaluationInputs(
            prominence_scores={"target": 0.2},
            model_dependent=True,
        ),
    )

    assert metrics.reproducibility.value == "model-dependent"
    assert metrics.model_trial == 3
    assert metrics.reproducibility_label == (
        "model-dependent (attention seed 7, model trial 3)"
    )
    prominence = metrics.metric("target-prominence")
    evidence = next(
        item for item in metrics.evidence if item.evidence_id in prominence.evidence_ids
    )
    assert "attention seed 7 and model trial 3" in evidence.description


def test_heuristic_prominence_persists_provider_version_without_model_id() -> None:
    version = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )
    base = _result(version)
    result = replace(
        base,
        state=replace(
            base.state,
            provider_manifests=(
                ProviderManifest(
                    provider_id="heuristic-prominence",
                    role="prominence",
                    model_id=None,
                    endpoint_origin="internal",
                    version="heuristic-prominence-v1",
                ),
            ),
        ),
    )

    inputs = evaluation_inputs_for(result)
    metrics = evaluate_run(result, EvaluationTarget("target"), inputs=inputs)

    assert inputs.prominence_model_id is None
    assert inputs.prominence_provider_version == "heuristic-prominence-v1"
    assert metrics.prominence_provider_version == "heuristic-prominence-v1"


def test_production_inputs_derive_scores_and_fold_facts_from_recorded_evidence() -> (
    None
):
    version = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )
    result = _result(version)
    result = replace(
        result,
        evidence=RunEvidence(
            prominence=(
                ProminenceEvidence(
                    viewport_id="viewport-1",
                    scores=(
                        ProminenceResult(
                            element_id="target",
                            raw_score=0.2,
                            normalized_probability=0.2,
                            first_notice_probability=0.2,
                            notice_within_budget_probability=0.2,
                            feature_contributions={},
                            raw_values={},
                            normalized_values={},
                        ),
                    ),
                ),
            ),
            scent=(
                ScentEvidence(
                    kind="full-scent",
                    viewport_id="viewport-1",
                    scores=(
                        FullScent(
                            element_id="target",
                            viewport_id="viewport-1",
                            score=0.25,
                        ),
                    ),
                ),
            ),
        ),
    )

    inputs = evaluation_inputs_for(result)

    assert inputs.target_prominence == pytest.approx(0.2)
    assert inputs.scent_scores["target"] == pytest.approx(0.25)
    assert inputs.target_below_fold is False
    assert inputs.feedback_observed is None
    assert inputs.model_dependent is True


@pytest.mark.parametrize("provider_id", ("heuristic", "foveacast"))
def test_evaluation_buckets_stage_from_record_when_score_has_no_stage(
    provider_id: str,
) -> None:
    version = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )
    base = _result(version)
    result = replace(
        base,
        state=replace(
            base.state,
            spec=replace(base.state.spec, prominence_provider_id=provider_id),
        ),
        evidence=RunEvidence(
            prominence=(
                ProminenceEvidence(
                    viewport_id="viewport-1",
                    stage="persistent",
                    scores=(
                        ProminenceResult(
                            element_id="target",
                            raw_score=0.7,
                            normalized_probability=0.7,
                            provider_id=(
                                "foveacast"
                                if provider_id == "foveacast"
                                else "heuristic-prominence"
                            ),
                        ),
                    ),
                ),
            )
        ),
    )

    inputs = evaluation_inputs_for(result)

    assert inputs.active_search_stage == "persistent"
    assert inputs.prominence_scores_by_stage["persistent"]["target"] == pytest.approx(
        0.7
    )
    assert inputs.target_prominence == pytest.approx(0.7)


def test_evaluation_uses_active_operational_score_and_retains_target_profiles() -> None:
    version = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )
    result = _result(version)
    result = replace(
        result,
        state=replace(
            result.state,
            spec=replace(result.state.spec, prominence_provider_id="foveacast"),
        ),
        evidence=RunEvidence(
            prominence=(
                ProminenceEvidence(
                    viewport_id="viewport-1",
                    source_event_id="event-3",
                    operational_event_id="event-3",
                    profile_event_id="event-2",
                    scores=(
                        ProminenceResult(
                            element_id="target",
                            raw_score=0.2,
                            normalized_probability=0.2,
                            raw_values={"stage_1s": 0.2},
                            provider_id="foveacast",
                            stage="initial",
                            evidence_kind="predicted",
                            evidence_source="foveacast-v0.2.0",
                        ),
                    ),
                ),
                ProminenceEvidence(
                    viewport_id="viewport-1",
                    source_event_id="event-5",
                    operational_event_id="event-5",
                    profile_event_id="event-4",
                    scores=(
                        ProminenceResult(
                            element_id="target",
                            raw_score=0.8,
                            normalized_probability=0.8,
                            raw_values={"stage_3s": 0.8, "stage_7s": 0.6},
                            provider_id="foveacast",
                            stage="persistent",
                            evidence_kind="predicted",
                            evidence_source="foveacast-v0.2.0",
                        ),
                    ),
                ),
            )
        ),
    )

    inputs = evaluation_inputs_for(result)

    assert inputs.target_prominence == pytest.approx(0.8)
    assert inputs.active_search_stage == "persistent"
    assert inputs.prominence_provider_id == "foveacast"
    assert inputs.prominence_profile_event_ids == ("event-2", "event-4")
    assert inputs.prominence_operational_event_ids == ("event-3", "event-5")
    assert inputs.target_prominence_profiles == {
        "immediate": pytest.approx(0.2),
        "early": pytest.approx(0.8),
        "eventual": pytest.approx(0.6),
    }
    assert inputs.target_prominence_sources == {
        "immediate": "foveacast-v0.2.0",
        "early": "foveacast-v0.2.0",
        "eventual": "foveacast-v0.2.0",
    }

    metrics = evaluate_run(result, EvaluationTarget("target"), inputs=inputs)
    evidence = next(
        item
        for item in metrics.evidence
        if item.evidence_id in metrics.metric("target-prominence").evidence_ids
    )
    assert metrics.target_prominence == pytest.approx(0.8)
    assert metrics.prominence_provider_id == "foveacast"
    assert metrics.active_search_stage == "persistent"
    assert "event-3" in evidence.description
    assert "event-5" in evidence.description
    assert "1s" in evidence.description
    assert "3s" in evidence.description
    assert "7s" in evidence.description


def test_evaluation_returns_unavailable_when_active_stage_has_no_target_score() -> None:
    version = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )
    base = _result(version)
    result = replace(
        base,
        state=replace(
            base.state,
            spec=replace(base.state.spec, prominence_provider_id="foveacast"),
        ),
        evidence=RunEvidence(
            prominence=(
                ProminenceEvidence(
                    viewport_id="viewport-1",
                    source_event_id="event-3",
                    scores=(
                        ProminenceResult(
                            element_id="target",
                            raw_score=0.2,
                            normalized_probability=0.2,
                            provider_id="foveacast",
                            stage="initial",
                        ),
                    ),
                ),
                ProminenceEvidence(
                    viewport_id="viewport-1",
                    source_event_id="event-5",
                    scores=(
                        ProminenceResult(
                            element_id="competitor",
                            raw_score=0.9,
                            normalized_probability=0.9,
                            provider_id="foveacast",
                            stage="exploration",
                        ),
                    ),
                ),
            )
        ),
    )

    inputs = evaluation_inputs_for(result)

    assert inputs.active_search_stage == "exploration"
    assert inputs.target_prominence is None


def test_empty_latest_prominence_record_advances_stage_and_clears_target_score() -> (
    None
):
    version = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )
    base = _result(version)
    result = replace(
        base,
        state=replace(
            base.state,
            spec=replace(base.state.spec, prominence_provider_id="foveacast"),
        ),
        evidence=RunEvidence(
            prominence=(
                ProminenceEvidence(
                    viewport_id="viewport-1",
                    scores=(
                        ProminenceResult(
                            element_id="target",
                            raw_score=0.8,
                            normalized_probability=0.8,
                            provider_id="foveacast",
                            stage="initial",
                        ),
                    ),
                    stage="initial",
                ),
                ProminenceEvidence(
                    viewport_id="viewport-1",
                    scores=(),
                    stage="persistent",
                ),
            )
        ),
    )

    inputs = evaluation_inputs_for(result)

    assert inputs.active_search_stage == "persistent"
    assert inputs.target_prominence is None


def test_evaluate_run_marks_fallback_learned_sample_invalid_without_manual_override() -> (
    None
):
    version = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )
    base = _result(version)
    result = replace(
        base,
        state=replace(
            base.state,
            spec=replace(base.state.spec, prominence_provider_id="foveacast"),
        ),
        ux_sample_valid=False,
        ux_sample_invalid_reason="saliency-fallback: runtime unavailable",
    )

    metrics = evaluate_run(result, EvaluationTarget("target"))

    assert metrics.prominence_fallback is True
    assert metrics.comparison_valid is False


def test_evaluation_uses_scenario_target_after_wrong_action_and_abandonment() -> None:
    version = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )
    result = _result(version)
    events = list(result.state.events)
    wrong = InteractWithElement(element_id="competitor")
    events.insert(-1, ActionProposed(action=wrong))
    events.insert(
        -1,
        ActionExecuted(
            action=wrong,
            viewport_id="viewport-1",
            succeeded=False,
            error="wrong action",
        ),
    )
    abandoned = AgentAbandoned(reason="user gave up")
    result = replace(
        result,
        outcome=abandoned,
        verification=VerificationResult(verified=False),
        state=replace(
            result.state,
            events=tuple(events[:-1])
            + (
                RunTerminated(
                    outcome=abandoned,
                    verification=VerificationResult(verified=False),
                    configuration_digest=result.state.spec.config_digest,
                    artifact_checksums=(
                        ArtifactChecksum(path="timeline", sha256="sha"),
                    ),
                ),
            ),
        ),
    )

    metrics = evaluate_run(result, evaluation_target_for(result))

    assert metrics.target.element_id is not None
    assert metrics.target.element_id.endswith("target")
    assert metrics.wrong_actions == 1
    assert metrics.abandoned is True


def test_evaluation_cells_keep_prominence_provider_in_cell_identity() -> None:
    version = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )
    heuristic = evaluate_run(_result(version), EvaluationTarget("target"))
    learned_result = _result(version)
    learned_result = replace(
        learned_result,
        run_id="run-foveacast",
        state=replace(
            learned_result.state,
            spec=replace(
                learned_result.state.spec,
                run_id="run-foveacast",
                prominence_provider_id="foveacast",
            ),
        ),
    )
    learned = evaluate_run(
        learned_result,
        EvaluationTarget("target"),
        inputs=RunEvaluationInputs(
            prominence_provider_id="foveacast",
            active_search_stage="initial",
            model_dependent=True,
        ),
    )

    evaluation = evaluate_experiment_results(
        (
            replace(_result(version), metrics=heuristic),
            replace(learned_result, metrics=learned),
        )
    )

    assert {cell.prominence_provider_id for cell in evaluation.cell_aggregates} == {
        "heuristic",
        "foveacast",
    }


def test_evaluation_canonicalizes_prominence_provider_aliases() -> None:
    version = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )
    result = _result(version)

    metrics = evaluate_run(
        result,
        EvaluationTarget("target"),
        inputs=RunEvaluationInputs(
            prominence_provider_id="foveacast-prominence",
        ),
    )

    assert metrics.prominence_provider_id == "foveacast"


def test_prominence_event_ids_keep_target_lineage_separate_from_global_lineage() -> (
    None
):
    version = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )
    base = _result(version)
    result = replace(
        base,
        state=replace(
            base.state,
            spec=replace(base.state.spec, prominence_provider_id="foveacast"),
        ),
        evidence=RunEvidence(
            prominence=(
                ProminenceEvidence(
                    viewport_id="viewport-1",
                    scores=(
                        ProminenceResult(
                            element_id="target",
                            raw_score=0.2,
                            normalized_probability=0.2,
                            provider_id="foveacast",
                            stage="initial",
                        ),
                    ),
                    source_event_id="event-target",
                    profile_event_id="event-profile-target",
                    stage="initial",
                ),
                ProminenceEvidence(
                    viewport_id="viewport-1",
                    scores=(
                        ProminenceResult(
                            element_id="competitor",
                            raw_score=0.9,
                            normalized_probability=0.9,
                            provider_id="foveacast",
                            stage="exploration",
                        ),
                    ),
                    source_event_id="event-competitor",
                    profile_event_id="event-profile-competitor",
                    stage="exploration",
                ),
            )
        ),
    )

    inputs = evaluation_inputs_for(result)

    assert inputs.prominence_profile_event_ids == ("event-profile-target",)
    assert inputs.prominence_operational_event_ids == ("event-target",)
    assert inputs.prominence_lineage_event_ids == (
        "event-profile-target",
        "event-target",
    )


def test_profile_target_does_not_make_competitor_operational_event_target_bearing() -> (
    None
):
    version = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )
    base = _result(version)
    result = replace(
        base,
        state=replace(
            base.state,
            spec=replace(base.state.spec, prominence_provider_id="foveacast"),
        ),
        evidence=RunEvidence(
            prominence=(
                ProminenceEvidence(
                    viewport_id="viewport-1",
                    stage="exploration",
                    profile_event_id="event-profile",
                    operational_event_id="event-operational",
                    profiles=(
                        SimpleNamespace(
                            viewport_id="inference-1",
                            element_id="target",
                            immediate=None,
                            early=None,
                            eventual=None,
                        ),
                    ),
                    scores=(
                        ProminenceResult(
                            element_id="competitor",
                            raw_score=0.8,
                            normalized_probability=0.8,
                            provider_id="foveacast",
                        ),
                    ),
                ),
            )
        ),
    )

    inputs = evaluation_inputs_for(result)

    assert inputs.prominence_profile_event_ids == ("event-profile",)
    assert inputs.prominence_operational_event_ids == ()
    assert inputs.prominence_lineage_event_ids == ("event-profile",)


def test_run_evaluation_inputs_copy_score_mappings() -> None:
    scores = {"target": 0.2}
    stage_scores = {"initial": {"target": 0.2}}
    inputs = RunEvaluationInputs(
        prominence_scores=scores,
        prominence_scores_by_stage=stage_scores,
    )

    scores["target"] = 0.9
    stage_scores["initial"]["target"] = 0.9

    assert inputs.prominence_scores["target"] == pytest.approx(0.2)
    assert inputs.prominence_scores_by_stage["initial"]["target"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("result", "metrics", "manifest", "expected"),
    (
        (
            {"run_id": "run-1", "ux_sample_valid": True},
            {
                "run_id": "run-1",
                "prominence_provider_id": "foveacast",
                "comparison_valid": True,
                "prominence_fallback": False,
            },
            {"run_id": "run-1", "prominence_provider_id": "foveacast"},
            True,
        ),
        (
            {"run_id": "run-1", "ux_sample_valid": True},
            {"run_id": "run-1", "prominence_provider_id": "foveacast"},
            {"run_id": "run-1", "prominence_provider_id": "foveacast"},
            False,
        ),
        (
            {"run_id": "run-1", "ux_sample_valid": False},
            {
                "run_id": "run-1",
                "prominence_provider_id": "foveacast",
                "comparison_valid": True,
                "prominence_fallback": False,
            },
            {"run_id": "run-1", "prominence_provider_id": "foveacast"},
            False,
        ),
        (
            {"run_id": "run-1", "ux_sample_valid": True},
            {
                "run_id": "run-1",
                "prominence_provider_id": "heuristic",
                "comparison_valid": True,
                "prominence_fallback": False,
            },
            {"run_id": "run-1", "prominence_provider_id": "foveacast"},
            False,
        ),
        (
            {
                "run_id": "run-1",
                "ux_sample_valid": True,
                "ux_sample_invalid_reason": "saliency-fallback: runtime unavailable",
            },
            {
                "run_id": "run-1",
                "prominence_provider_id": "foveacast",
                "comparison_valid": True,
                "prominence_fallback": True,
                "prominence_fallback_reason": "runtime unavailable",
            },
            {"run_id": "run-1", "prominence_provider_id": "foveacast"},
            False,
        ),
    ),
)
def test_persisted_comparison_gate_cross_checks_all_validity_fields(
    result: dict[str, object],
    metrics: dict[str, object],
    manifest: dict[str, object],
    expected: bool,
) -> None:
    from ux_analyzer.application.evaluation import persisted_comparison_sample_is_valid

    assert (
        persisted_comparison_sample_is_valid(
            result,
            metrics,
            manifest=manifest,
            expected_run_id="run-1",
            expected_prominence_provider_id="foveacast",
        )
        is expected
    )


def test_persisted_comparison_gate_rejects_typed_fallback_event_even_with_true_flags() -> (
    None
):
    assert not persisted_comparison_sample_is_valid(
        {"run_id": "run-1", "ux_sample_valid": True},
        {
            "run_id": "run-1",
            "prominence_provider_id": "foveacast",
            "comparison_valid": True,
            "prominence_fallback": False,
            "prominence_fallback_reason": None,
        },
        manifest={"run_id": "run-1", "prominence_provider_id": "foveacast"},
        expected_run_id="run-1",
        expected_prominence_provider_id="foveacast",
        timeline_events=({"sequence": 4, "kind": "saliency-fallback-recorded"},),
    )


def test_cell_aggregate_reports_all_active_stages() -> None:
    version = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )
    first = evaluate_run(
        _result(version),
        EvaluationTarget("target"),
        inputs=RunEvaluationInputs(active_search_stage="initial"),
    )
    second = replace(first, run_id="run-persistent", active_search_stage="persistent")

    aggregate = aggregate_cell((first, second))

    assert aggregate.active_search_stages == ("initial", "persistent")


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("raw_score", True),
        ("normalized_probability", False),
        ("first_notice_probability", True),
        ("notice_within_budget_probability", False),
    ),
)
def test_prominence_result_rejects_boolean_scores(field_name: str, value: bool) -> None:
    with pytest.raises(ValueError):
        ProminenceResult(
            element_id="target",
            raw_score=value if field_name == "raw_score" else 0.2,
            normalized_probability=value
            if field_name == "normalized_probability"
            else 0.2,
            first_notice_probability=value
            if field_name == "first_notice_probability"
            else 0.2,
            notice_within_budget_probability=value
            if field_name == "notice_within_budget_probability"
            else 0.2,
        )


def test_prominence_result_rejects_unknown_search_stage() -> None:
    with pytest.raises(ValueError, match="stage"):
        ProminenceResult(
            element_id="target",
            raw_score=0.2,
            normalized_probability=0.2,
            stage="later",
        )


def test_fallback_learned_sample_is_excluded_from_foveacast_scorecards() -> None:
    version = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )
    result = _result(version)
    learned_result = replace(
        result,
        run_id="run-fallback",
        state=replace(
            result.state,
            spec=replace(
                result.state.spec,
                run_id="run-fallback",
                prominence_provider_id="foveacast",
            ),
        ),
        ux_sample_valid=False,
    )
    fallback_metrics = replace(
        evaluate_run(
            learned_result,
            EvaluationTarget("target"),
            inputs=RunEvaluationInputs(
                prominence_provider_id="foveacast",
                active_search_stage="initial",
                prominence_fallback=True,
                prominence_fallback_reason="runtime unavailable",
                model_dependent=True,
            ),
        ),
        comparison_valid=False,
    )

    evaluation = evaluate_experiment_results(
        (replace(learned_result, metrics=fallback_metrics),)
    )

    assert evaluation.run_metrics == ()
    assert evaluation.cell_aggregates == ()
    assert evaluation.variant_comparisons == ()


def test_production_inputs_mark_target_below_fold_after_recorded_scroll() -> None:
    version = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )
    result = _result(version)
    events = list(result.state.events)
    observation = next(
        event for event in events if isinstance(event, ObservationRecorded)
    )
    events.remove(observation)
    scroll_index = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, ActionExecuted) and isinstance(event.action, Scroll)
    )
    events.insert(scroll_index + 1, observation)
    result = replace(result, state=replace(result.state, events=tuple(events)))

    assert evaluation_inputs_for(result).target_below_fold is True


@pytest.mark.parametrize(
    ("persona_id", "policy"),
    (
        ("first-time-nontechnical", ExperimentPolicy.FULL_LIST),
        (
            "first-time-nontechnical",
            ExperimentPolicy.PROGRESSIVE_PROMINENCE_SCENT,
        ),
        ("impatient", ExperimentPolicy.FULL_LIST),
        ("impatient", ExperimentPolicy.PROGRESSIVE_PROMINENCE_SCENT),
    ),
)
@pytest.mark.parametrize(
    ("version_kind", "feedback_label", "expected"),
    (
        (ApplicationVersionKind.DEFECTIVE, "Protection enabled.", False),
        (
            ApplicationVersionKind.IMPROVED,
            "Two-factor authentication enabled.",
            True,
        ),
    ),
)
def test_feedback_uses_first_post_action_snapshot_across_matrix(
    persona_id: str,
    policy: ExperimentPolicy,
    version_kind: ApplicationVersionKind,
    feedback_label: str,
    expected: bool,
) -> None:
    result = _feedback_result(
        persona_id=persona_id,
        policy=policy,
        version_kind=version_kind,
        feedback_label=feedback_label,
    )

    assert evaluation_inputs_for(result).feedback_observed is expected


def test_feedback_is_unknown_when_post_action_capture_failed() -> None:
    result = _feedback_result(
        persona_id="first-time-nontechnical",
        policy=ExperimentPolicy.PROGRESSIVE_PROMINENCE_SCENT,
        version_kind=ApplicationVersionKind.IMPROVED,
        feedback_label=None,
    )

    assert evaluation_inputs_for(result).feedback_observed is None


def test_scorecard_rejects_unsupported_human_evidence() -> None:
    version = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )
    metrics = evaluate_run(_result(version), EvaluationTarget("target"))
    unsupported = replace(
        metrics,
        evidence=(
            Evidence(
                evidence_id="human-claim",
                evidence_class=EvidenceClass.UNSUPPORTED_HUMAN_CLAIM,
                description="Real users will be satisfied.",
            ),
        ),
    )

    with pytest.raises(UnsupportedHumanClaimError):
        aggregate_cell((unsupported,))


def _feedback_result(
    *,
    persona_id: str,
    policy: ExperimentPolicy,
    version_kind: ApplicationVersionKind,
    feedback_label: str | None,
) -> RunResult:
    version = ApplicationVersion(
        id=f"fixture-app-{version_kind.value}",
        kind=version_kind,
        label=version_kind.value.title(),
    )
    scenario = Scenario(
        id="enable-2fa",
        name="Enable two-factor authentication",
        goal="Enable two-factor authentication",
        application_version_ids=(version.id,),
        start_state="settings",
        fixture_inputs=FixtureInputs(values={"totp_code": "246810"}),
        budget=Budget(20, 20, 10, 30),
        verifier=VisibleResultVerifierSpec(
            type="visible-result", text="Two-factor authentication enabled"
        ),
        safeguards=(),
        eligible_persona_ids=(persona_id,),
        expected_evidence=(),
        evaluation_target=ScenarioEvaluationTarget(
            labels_by_version={
                "defective": "Enable two-factor authentication",
                "improved": "Enable two-factor authentication",
            },
            role="button",
        ),
    )
    spec = RunSpec(
        run_id=f"run-{version_kind.value}-{persona_id}-{policy.value}",
        seed=7,
        scenario=scenario,
        application_version=version,
        persona=Persona(
            id=persona_id,
            name=persona_id,
            working_memory_capacity=3,
            initial_confidence=0.5,
            initial_frustration=0,
            abandonment_threshold=0.9,
            attention_temperature=1,
        ),
        policy=policy,
        config_digest="config-sha",
    )
    target = ElementSnapshot(
        id="viewport-before-target",
        role="button",
        label="Enable two-factor authentication",
        bounds=BoundingBox(x=10, y=10, width=200, height=40),
        visibility_fraction=1,
        actionable=True,
        provider_id="fixture",
        execution_reference=PrivateExecutionReference(
            provider_id="fixture",
            viewport_id="viewport-before",
            token="target-token",
        ),
        lineage_id="two-factor-submit",
    )
    before = ViewportSnapshot(
        id="viewport-before",
        provider_id="fixture",
        elements=(target,),
    )
    state = RunState.initial(spec).apply(RunStarted(run_id=spec.run_id))
    state = state.apply(ViewportCaptured(snapshot=before))
    state = state.apply(
        ObservationRecorded(
            observation=ProgressiveObservation.from_snapshot(
                before, newly_revealed_ids=(target.id,)
            )
        )
    )
    action = InteractWithElement(element_id=target.id)
    state = state.apply(ActionProposed(action=action))
    state = state.apply(
        ActionExecuted(
            action=action,
            viewport_id=before.id,
            execution_reference=target.execution_reference,
            succeeded=True,
            platform_action_kind="click",
            state_changed=True,
        )
    )
    if feedback_label is not None:
        feedback = ElementSnapshot(
            id="viewport-after-feedback",
            role="text",
            label=feedback_label,
            bounds=BoundingBox(x=10, y=10, width=260, height=40),
            visibility_fraction=1,
            actionable=False,
            lineage_id="two-factor-feedback",
        )
        state = state.apply(
            ViewportCaptured(
                snapshot=ViewportSnapshot(
                    id="viewport-after",
                    provider_id="fixture",
                    elements=(feedback,),
                )
            )
        )
    verification = VerificationResult(verified=True, evidence_ids=("verify-1",))
    state = state.apply(
        RunTerminated(
            outcome=VerifiedSuccess(),
            verification=verification,
            provider_manifests=(
                ProviderManifest(
                    provider_id="fixture",
                    role="observation",
                    model_id=None,
                    endpoint_origin="fixture",
                    version="1",
                ),
            ),
            configuration_digest=spec.config_digest,
            artifact_checksums=(ArtifactChecksum(path="timeline", sha256="sha"),),
        )
    )
    return RunResult(
        run_id=spec.run_id,
        outcome=VerifiedSuccess(),
        verification=verification,
        agent_claimed_success=False,
        state=state,
    )


def test_evaluate_experiment_results_aggregates_cells_and_paired_gate() -> None:
    defective = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )
    improved = ApplicationVersion(
        id="improved", kind=ApplicationVersionKind.IMPROVED, label="Improved"
    )
    baseline_result = _result(defective)
    improved_result = _result(improved)
    baseline_metrics = evaluate_run(baseline_result, EvaluationTarget("target"))
    improved_metrics = replace(
        evaluate_run(improved_result, EvaluationTarget("target")),
        discovery_cost=DiscoveryCostBreakdown(0, 0, 0, 0, 0, 0, 0),
    )

    evaluation = evaluate_experiment_results(
        (
            replace(baseline_result, metrics=baseline_metrics),
            replace(improved_result, metrics=improved_metrics),
        )
    )

    assert len(evaluation.run_metrics) == 2
    assert len(evaluation.cell_aggregates) == 2
    assert len(evaluation.variant_comparisons) == 1
    assert evaluation.variant_comparisons[0].gate.passed


def test_experiment_evaluation_retains_partial_cells_without_unpaired_gate() -> None:
    defective = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )
    improved = ApplicationVersion(
        id="improved", kind=ApplicationVersionKind.IMPROVED, label="Improved"
    )
    baseline_result = _result(defective)
    improved_result = _result(improved)
    baseline_metrics = evaluate_run(baseline_result, EvaluationTarget("target"))
    improved_metrics = replace(
        evaluate_run(improved_result, EvaluationTarget("target")),
        seed=8,
    )

    evaluation = evaluate_experiment_results(
        (
            replace(baseline_result, metrics=baseline_metrics),
            replace(improved_result, metrics=improved_metrics),
        )
    )

    assert len(evaluation.cell_aggregates) == 2
    assert evaluation.variant_comparisons == ()
