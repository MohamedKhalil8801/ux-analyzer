from __future__ import annotations

import random

import pytest

from ux_analyzer.application.experiment import (
    ExperimentContext,
    deterministic_run_id,
    expand_experiment,
)
from ux_analyzer.domain.attention import AttentionState
from ux_analyzer.domain.benchmark import (
    Application,
    ApplicationVersion,
    ApplicationVersionKind,
    BenchmarkProject,
    Budget,
    ExperimentDefinition,
    ExperimentPolicy,
    FixtureInputs,
    Persona,
    Scenario,
    ScenarioEvaluationTarget,
    VisibleResultVerifierSpec,
)
from ux_analyzer.domain.interface import BoundingBox, ElementSnapshot, ViewportSnapshot
from ux_analyzer.providers.full_list_policy import FullListPolicy
from ux_analyzer.providers.prominence import ProminenceResult
from ux_analyzer.providers.ranked_list_policy import ProminenceRankedListPolicy


def _project() -> BenchmarkProject:
    defective = ApplicationVersion(
        id="app-defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )
    improved = ApplicationVersion(
        id="app-improved", kind=ApplicationVersionKind.IMPROVED, label="Improved"
    )
    scenario = Scenario(
        id="invite",
        name="Invite teammate",
        goal="Invite teammate",
        application_version_ids=(defective.id, improved.id),
        start_state="dashboard",
        fixture_inputs=FixtureInputs(
            values={"invite_email": "person@example.com"},
            sensitive_keys=frozenset({"invite_email"}),
        ),
        budget=Budget(
            max_steps=10,
            max_observations=10,
            max_interactions=5,
            timeout_seconds=10,
        ),
        verifier=VisibleResultVerifierSpec(type="visible-result", text="Sent"),
        safeguards=("fixture-only",),
        eligible_persona_ids=("new-user",),
        expected_evidence=("discovery-cost",),
        evaluation_target=ScenarioEvaluationTarget(
            labels_by_version={
                "defective": "Invite teammate",
                "improved": "Invite teammate",
            },
            role="button",
        ),
    )
    persona = Persona(
        id="new-user",
        name="New user",
        working_memory_capacity=3,
        initial_confidence=0.5,
        initial_frustration=0.0,
        abandonment_threshold=0.8,
        attention_temperature=1.0,
    )
    return BenchmarkProject(
        id="demo",
        name="Demo",
        applications=(
            Application(id="app", name="App", versions=(defective, improved)),
        ),
        scenarios=(scenario,),
        personas=(persona,),
        experiments=(),
    )


def _definition(
    policies: tuple[ExperimentPolicy, ...],
    seeds: tuple[int, ...] = (),
    model_trials: tuple[int, ...] = (0,),
) -> ExperimentDefinition:
    return ExperimentDefinition(
        id="core",
        name="Core pair",
        scenario_ids=("invite",),
        application_version_ids=("app-defective", "app-improved"),
        persona_ids=("new-user",),
        policies=policies,
        seeds=seeds,
        model_trials=model_trials,
        run_count=10,
    )


def test_expand_experiment_is_stable_and_preserves_comparison_inputs() -> None:
    project = _project()
    definition = _definition(
        (
            ExperimentPolicy.FULL_LIST,
            ExperimentPolicy.PROGRESSIVE_PROMINENCE_SCENT,
        )
    )
    context = ExperimentContext(
        definition=definition,
        project=project,
        config_digest="config-sha",
        model_config={"cognitive_model": "model-a", "scent_model": "model-b"},
    )

    first = expand_experiment(context)
    repeat = expand_experiment(context)

    assert len(first) == 22
    assert [spec.run_id for spec in first] == [spec.run_id for spec in repeat]
    assert len({spec.run_id for spec in first}) == len(first)
    assert first[0].seed == 0
    assert first[0].policy is ExperimentPolicy.FULL_LIST
    assert [spec.seed for spec in first[1:11]] == list(range(10))
    assert [spec.policy for spec in first[1:11]] == [
        ExperimentPolicy.PROGRESSIVE_PROMINENCE_SCENT
    ] * 10
    assert first[0].scenario is first[1].scenario
    assert first[0].persona is first[1].persona
    assert first[0].scenario.fixture_inputs == first[1].scenario.fixture_inputs
    assert first[0].config_digest == first[1].config_digest == "config-sha"
    assert context.model_config == {
        "cognitive_model": "model-a",
        "scent_model": "model-b",
    }
    assert first[0].run_id != first[10].run_id


def test_deterministic_policies_use_only_first_explicit_seed() -> None:
    specs = expand_experiment(
        _definition(
            (
                ExperimentPolicy.FULL_LIST,
                ExperimentPolicy.PROMINENCE_RANKED_LIST,
            ),
            seeds=(17, 23),
        ),
        project=_project(),
        config_digest="config-sha",
    )

    assert len(specs) == 4
    assert {spec.seed for spec in specs} == {17}


def test_model_trials_expand_independently_from_attention_seeds() -> None:
    specs = expand_experiment(
        _definition(
            (ExperimentPolicy.PROGRESSIVE_PROMINENCE,),
            seeds=(7,),
            model_trials=(0, 1, 2),
        ),
        project=_project(),
        config_digest="config-sha",
    )

    assert {(spec.seed, spec.model_trial) for spec in specs} == {
        (7, 0),
        (7, 1),
        (7, 2),
    }
    assert len(specs) == 6
    assert len({spec.run_id for spec in specs}) == 6


def test_zero_model_trial_preserves_legacy_run_id_payload() -> None:
    kwargs = {
        "experiment_id": "core",
        "scenario_id": "invite",
        "application_version_id": "app-defective",
        "persona_id": "new-user",
        "policy": ExperimentPolicy.PROGRESSIVE_PROMINENCE,
        "seed": 7,
        "config_digest": "config-sha",
    }

    assert deterministic_run_id(**kwargs, model_trial=0) == (
        "run-6efda27bcc2d4245830ceae8ca1c030f42d90153e47ebea15d740593993cfadf"
    )
    assert deterministic_run_id(**kwargs, model_trial=1) != deterministic_run_id(
        **kwargs, model_trial=0
    )


def test_explicit_seed_matrix_replaces_default_run_count_seeds() -> None:
    specs = expand_experiment(
        _definition((ExperimentPolicy.PROGRESSIVE_PROMINENCE,), seeds=(17, 23)),
        project=_project(),
        config_digest="config-sha",
    )

    assert [spec.seed for spec in specs] == [17, 23, 17, 23]
    assert len(specs) == 4


def test_expand_experiment_rejects_missing_resolution_context() -> None:
    with pytest.raises(ValueError, match="project or resolution context"):
        expand_experiment(_definition((ExperimentPolicy.FULL_LIST,)))


def _snapshot() -> ViewportSnapshot:
    return ViewportSnapshot(
        id="viewport-1",
        elements=tuple(
            ElementSnapshot(
                id=element_id,
                role="button",
                label=element_id.title(),
                bounds=BoundingBox(x=index * 100, y=10, width=80, height=30),
                visibility_fraction=1.0,
                actionable=True,
            )
            for index, element_id in enumerate(
                ("first", "second", "third", "fourth", "fifth")
            )
        ),
    )


def test_full_list_policy_reveals_all_visible_persona_safe_elements() -> None:
    selection = FullListPolicy().next_observation(
        AttentionState.initial(
            _project().scenarios[0].budget,
            confidence=0.5,
            frustration=0.0,
        ),
        _snapshot(),
        (),
        (),
        random.Random(1),
    )

    assert selection.selected_ids == ("first", "second", "third", "fourth", "fifth")
    assert len(selection.observation.newly_revealed_elements) == 5
    assert (
        "execution_reference"
        not in selection.observation.newly_revealed_elements[0].model_dump()
    )


def test_ranked_list_policy_sorts_without_exposing_numeric_scores() -> None:
    snapshot = _snapshot()
    scores = (
        ProminenceResult("first", 0.1, 0.1),
        ProminenceResult("second", 0.9, 0.8),
        ProminenceResult("third", 0.5, 0.4),
        ProminenceResult("fourth", 0.3, 0.3),
        ProminenceResult("fifth", 0.2, 0.2),
    )

    selection = ProminenceRankedListPolicy().next_observation(
        AttentionState.initial(
            _project().scenarios[0].budget,
            confidence=0.5,
            frustration=0.0,
        ),
        snapshot,
        scores,
        (),
        random.Random(1),
    )

    assert selection.selected_ids == (
        "second",
        "third",
        "fourth",
        "fifth",
        "first",
    )
    assert all(
        "normalized_probability" not in item.model_dump()
        for item in selection.observation.newly_revealed_elements
    )
