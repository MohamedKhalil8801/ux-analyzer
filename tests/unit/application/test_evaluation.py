from __future__ import annotations

from dataclasses import replace

import pytest

from ux_analyzer.application.evaluation import (
    DiscoveryCostBreakdown,
    DiscoveryCostConfig,
    EvaluationTarget,
    RunEvaluationInputs,
    aggregate_cell,
    compare_variants,
    evaluate_experiment_results,
    evaluate_run,
)
from ux_analyzer.application.run_agent import RunResult
from ux_analyzer.domain.attention import Back, ProgressiveObservation, Scroll
from ux_analyzer.domain.benchmark import (
    ApplicationVersion,
    ApplicationVersionKind,
    Budget,
    ExperimentPolicy,
    FixtureInputs,
    Persona,
    Scenario,
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


def test_aggregate_cell_provides_median_interval_and_reproducibility() -> None:
    version = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )
    first = evaluate_run(_result(version), EvaluationTarget("target"))
    second = replace(first, run_id="run-second", seed=8, scrolls=3)

    aggregate = aggregate_cell((first, second))

    assert aggregate.run_count == 2
    assert aggregate.metric_summaries["scrolls"].median == 2.0
    assert aggregate.metric_summaries["scrolls"].interval.lower <= 1.0
    assert aggregate.metric_summaries["scrolls"].interval.upper >= 3.0
    assert aggregate.reproducibility.value in {"seeded", "reproducible"}


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


def test_model_dependent_inputs_are_not_reported_as_seed_only() -> None:
    version = ApplicationVersion(
        id="defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )

    metrics = evaluate_run(
        _result(version),
        EvaluationTarget("target"),
        inputs=RunEvaluationInputs(model_dependent=True),
    )

    assert metrics.reproducibility.value == "model-dependent"


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
