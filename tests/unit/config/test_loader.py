from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
import yaml

from ux_analyzer.application.experiment import (
    ExperimentContext,
    deterministic_run_id,
    expand_experiment,
)
from ux_analyzer.config.loader import ProjectConfigError, load_project
from ux_analyzer.domain.benchmark import (
    ApplicationVersion,
    ApplicationVersionKind,
    ExperimentPolicy,
    FixtureStateVerifierSpec,
    VisibleResultVerifierSpec,
)

FIXTURE_PATH = (
    Path(__file__).parents[2] / "fixtures" / "config" / "minimal-project.yaml"
)
LEGACY_CONFIG_DIGEST = (
    "8d89b1483b2bd968a2b8b5c6dd2ce37a3749537899a0b88aceb8abf9f4fecb08"
)
LEGACY_RUN_ID = "run-bddad93657faaf94d4317372c772d125ef5b62e1bc30f6e61b2aa36c5355c264"
LIVE_CONFIG_DIGEST = "3fdc33d0c9ed994a924db0a0bd640899477dfd2841666eb72350a138fb5b92a6"


def _read_project() -> dict[str, Any]:
    with FIXTURE_PATH.open(encoding="utf-8") as project_file:
        loaded = yaml.safe_load(project_file)
    assert isinstance(loaded, dict)
    return loaded


def _write_project(
    tmp_path: Path, project: dict[str, Any], name: str = "project.yaml"
) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    return path


def _live_project(
    *,
    start_url: str = "https://portfolio.example/work",
    allowed_origins: list[str] | None = None,
) -> dict[str, Any]:
    project = _read_project()
    resource_origins = (
        allowed_origins if allowed_origins is not None else ["https://fonts.example"]
    )
    project["applications"][0]["versions"] = [
        {
            "id": "portfolio-live",
            "kind": "live",
            "label": "Production",
            "start_url": start_url,
            "allowed_origins": list(resource_origins),
            "navigation_settle_ms": 2000,
            "action_settle_ms": 900,
        }
    ]
    project["scenarios"][0]["application_version_ids"] = ["portfolio-live"]
    project["scenarios"][0]["fixture_inputs"] = {}
    project["scenarios"][0]["verifier"] = {
        "type": "visible-result",
        "text": "Work",
    }
    project["scenarios"][0]["evaluation_target"]["labels_by_version"] = {"live": "Work"}
    project["experiments"][0]["application_version_ids"] = ["portfolio-live"]
    return project


def test_load_valid_project_into_frozen_domain_contracts() -> None:
    loaded = load_project(FIXTURE_PATH)

    assert loaded.project.id == "minimal-project"
    fixture_version = loaded.project.applications[0].versions[0]
    assert fixture_version.kind is ApplicationVersionKind.DEFECTIVE
    assert fixture_version.start_url is None
    assert fixture_version.allowed_origins == ()
    assert fixture_version.navigation_settle_ms == 0
    assert fixture_version.action_settle_ms == 0
    assert loaded.project.experiments[0].policies == (ExperimentPolicy.FULL_LIST,)
    assert isinstance(loaded.project.scenarios[0].verifier, FixtureStateVerifierSpec)
    assert loaded.project.scenarios[0].fixture_inputs.values["invite_email"] == (
        "person@example.com"
    )
    assert loaded.project.scenarios[0].fixture_inputs.sensitive_keys == frozenset(
        {"invite_email"}
    )
    assert len(loaded.config_digest) == 64


def test_loads_single_live_application_version(tmp_path: Path) -> None:
    project = _read_project()
    project["applications"][0]["versions"] = [
        {
            "id": "portfolio-live",
            "kind": "live",
            "label": "Production",
            "start_url": "https://portfolio.example/work",
            "allowed_origins": ["https://fonts.example"],
            "navigation_settle_ms": 2000,
            "action_settle_ms": 900,
        }
    ]
    project["scenarios"][0]["application_version_ids"] = ["portfolio-live"]
    project["scenarios"][0]["fixture_inputs"] = {}
    project["scenarios"][0]["verifier"] = {
        "type": "visible-result",
        "text": "Work",
    }
    project["scenarios"][0]["evaluation_target"]["labels_by_version"] = {"live": "Work"}
    project["experiments"][0]["application_version_ids"] = ["portfolio-live"]

    version = (
        load_project(_write_project(tmp_path, project))
        .project.applications[0]
        .versions[0]
    )

    assert version.kind is ApplicationVersionKind.LIVE
    assert version.start_url == "https://portfolio.example/work"
    assert version.allowed_origins == ("https://fonts.example",)
    assert version.navigation_settle_ms == 2000
    assert version.action_settle_ms == 900


def test_live_application_version_requires_start_url(tmp_path: Path) -> None:
    project = _read_project()
    project["applications"][0]["versions"] = [
        {"id": "portfolio-live", "kind": "live", "label": "Production"}
    ]
    project["scenarios"][0]["application_version_ids"] = ["portfolio-live"]
    project["scenarios"][0]["evaluation_target"]["labels_by_version"] = {"live": "Work"}
    project["experiments"][0]["application_version_ids"] = ["portfolio-live"]

    with pytest.raises(ProjectConfigError, match="start_url"):
        load_project(_write_project(tmp_path, project))


def test_loads_visible_result_verifier_all_of_constraints(tmp_path: Path) -> None:
    project = _live_project()
    project["scenarios"][0]["verifier"]["all_of"] = ["PAIR Systems", "React"]

    verifier = (
        load_project(_write_project(tmp_path, project)).project.scenarios[0].verifier
    )

    assert isinstance(verifier, VisibleResultVerifierSpec)
    assert verifier.text == "Work"
    assert verifier.all_of == ("PAIR Systems", "React")


@pytest.mark.parametrize("all_of", [[""], ["React", "React"]])
def test_visible_result_verifier_rejects_invalid_all_of(
    tmp_path: Path, all_of: list[str]
) -> None:
    project = _live_project()
    project["scenarios"][0]["verifier"]["all_of"] = all_of

    with pytest.raises(ProjectConfigError, match="all_of"):
        load_project(_write_project(tmp_path, project))


@pytest.mark.parametrize("timing_field", ["navigation_settle_ms", "action_settle_ms"])
def test_live_settle_timings_must_not_be_negative(
    tmp_path: Path, timing_field: str
) -> None:
    project = _read_project()
    project["applications"][0]["versions"] = [
        {
            "id": "portfolio-live",
            "kind": "live",
            "label": "Production",
            "start_url": "https://portfolio.example/work",
            timing_field: -1,
        }
    ]
    project["scenarios"][0]["application_version_ids"] = ["portfolio-live"]
    project["scenarios"][0]["evaluation_target"]["labels_by_version"] = {"live": "Work"}
    project["experiments"][0]["application_version_ids"] = ["portfolio-live"]

    with pytest.raises(ProjectConfigError, match=timing_field):
        load_project(_write_project(tmp_path, project))


def test_live_configuration_canonicalizes_url_and_resource_origins(
    tmp_path: Path,
) -> None:
    project = _live_project(
        start_url="HTTPS://Portfolio.Example:443/work",
        allowed_origins=["HTTPS://Fonts.Example:443/", "http://CDN.Example:80"],
    )

    version = (
        load_project(_write_project(tmp_path, project))
        .project.applications[0]
        .versions[0]
    )

    assert version.start_url == "https://portfolio.example/work"
    assert version.allowed_origins == (
        "https://fonts.example",
        "http://cdn.example",
    )


@pytest.mark.parametrize(
    "start_url",
    [
        "http://portfolio.example/work",
        "https://user:password@portfolio.example/work",
        "https://portfolio.example:not-a-port/work",
        "https://portfolio.example:0/work",
        "https://portfolio.example:65536/work",
    ],
)
def test_live_configuration_rejects_unsafe_start_url(
    tmp_path: Path, start_url: str
) -> None:
    with pytest.raises(ProjectConfigError, match="start_url"):
        load_project(_write_project(tmp_path, _live_project(start_url=start_url)))


@pytest.mark.parametrize(
    "allowed_origins",
    [
        ["ftp://fonts.example"],
        ["https://user:password@fonts.example"],
        ["https://fonts.example/assets"],
        ["https://fonts.example?query=1"],
        ["https://fonts.example#fragment"],
        ["https://fonts.example:not-a-port"],
        ["https://fonts.example:0"],
        ["https://fonts.example:65536"],
    ],
)
def test_live_configuration_rejects_unsafe_resource_origins(
    tmp_path: Path, allowed_origins: list[str]
) -> None:
    with pytest.raises(ProjectConfigError, match="origin"):
        load_project(
            _write_project(
                tmp_path,
                _live_project(allowed_origins=allowed_origins),
            )
        )


def test_live_configuration_rejects_duplicate_canonical_resource_origins(
    tmp_path: Path,
) -> None:
    project = _live_project(
        allowed_origins=["https://fonts.example", "HTTPS://Fonts.Example:443/"]
    )

    with pytest.raises(ProjectConfigError, match="origin"):
        load_project(_write_project(tmp_path, project))


@pytest.mark.parametrize(
    ("start_url", "allowed_origins"),
    [
        ("http://portfolio.example/work", ()),
        ("https://user:password@portfolio.example/work", ()),
        ("https://portfolio.example:not-a-port/work", ()),
        ("https://portfolio.example:0/work", ()),
        ("https://portfolio.example/work", ("https://fonts.example/assets",)),
        (
            "https://portfolio.example/work",
            ("https://fonts.example", "HTTPS://Fonts.Example:443/"),
        ),
        ("https://portfolio.example/work", ("https://fonts.example:0",)),
    ],
)
def test_application_version_rejects_unsafe_network_values_directly(
    start_url: str, allowed_origins: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError):
        ApplicationVersion(
            id="portfolio-live",
            kind=ApplicationVersionKind.LIVE,
            label="Production",
            start_url=start_url,
            allowed_origins=allowed_origins,
        )


def test_live_scenario_rejects_fixture_inputs(tmp_path: Path) -> None:
    project = _live_project()
    project["scenarios"][0]["fixture_inputs"] = {"secret": {"value": "not-for-live"}}

    with pytest.raises(ProjectConfigError, match="live.*fixture_inputs"):
        load_project(_write_project(tmp_path, project))


def test_live_scenario_rejects_fixture_state_verifier(tmp_path: Path) -> None:
    project = _live_project()
    project["scenarios"][0]["fixture_inputs"] = {}
    project["scenarios"][0]["verifier"] = {
        "type": "fixture-state",
        "resource": "result",
        "field": "text",
        "operator": "equals",
        "expected_fixture_key": "expected",
    }

    with pytest.raises(ProjectConfigError, match="live.*fixture-state"):
        load_project(_write_project(tmp_path, project))


def test_legacy_fixture_digest_and_run_id_ignore_explicit_new_defaults(
    tmp_path: Path,
) -> None:
    omitted_defaults = load_project(
        _write_project(tmp_path, _read_project(), "omitted.yaml")
    )
    explicit_defaults = _read_project()
    for version in explicit_defaults["applications"][0]["versions"]:
        version.update(
            {
                "start_url": None,
                "allowed_origins": [],
                "navigation_settle_ms": 0,
                "action_settle_ms": 0,
            }
        )
    explicit = load_project(
        _write_project(tmp_path, explicit_defaults, "explicit.yaml")
    )

    assert (
        omitted_defaults.config_digest == explicit.config_digest == LEGACY_CONFIG_DIGEST
    )
    assert (
        deterministic_run_id(
            experiment_id="smoke",
            scenario_id="invite-teammate",
            application_version_id="fixture-app-defective",
            persona_id="first-time-user",
            policy=ExperimentPolicy.FULL_LIST,
            seed=11,
            config_digest=explicit.config_digest,
        )
        == LEGACY_RUN_ID
    )


def test_non_default_live_version_fields_affect_digest(tmp_path: Path) -> None:
    first = load_project(_write_project(tmp_path, _live_project(), "first.yaml"))
    changed_project = _live_project()
    changed_project["applications"][0]["versions"][0]["action_settle_ms"] = 901
    changed = load_project(_write_project(tmp_path, changed_project, "changed.yaml"))

    assert first.config_digest == LIVE_CONFIG_DIGEST
    assert changed.config_digest != first.config_digest


def test_missing_model_trials_defaults_to_zero() -> None:
    loaded = load_project(FIXTURE_PATH)

    assert loaded.project.experiments[0].model_trials == (0,)


def test_missing_prominence_provider_ids_defaults_to_heuristic() -> None:
    loaded = load_project(FIXTURE_PATH)

    assert loaded.project.experiments[0].prominence_provider_ids == ("heuristic",)
    assert loaded.runtime.saliency.model_set == ("foveacast-v0.2.0",)
    assert loaded.runtime.saliency.execution_provider_preference == "auto"
    assert loaded.runtime.saliency.fallback.provider_id == "heuristic"


def test_loads_saliency_provider_configuration_and_provider_axis(
    tmp_path: Path,
) -> None:
    project = _read_project()
    project["providers"] = {
        "prominence": {
            "version": "heuristic-project-v2",
            "weights": {"contrast": 0.7},
            "temperature": 0.75,
        },
        "saliency": {
            "model_set": ["foveacast-v0.2.0"],
            "precision": "fp16",
            "execution_provider_preference": "cpu",
            "cache": {"enabled": False, "scope": "experiment"},
            "aggregation": {
                "version": "aggregation-project-v2",
                "density_weight": 0.5,
                "robust_peak_weight": 0.3,
                "mass_share_weight": 0.2,
                "temperature": 0.8,
            },
            "stage_selector": {
                "version": "stage-project-v2",
                "temperature": 0.9,
                "stage_mixtures": {"initial": {"1s": 1.0}},
            },
            "fallback": {"enabled": True, "provider_id": "heuristic"},
        },
    }
    project["experiments"][0]["prominence_provider_ids"] = [
        "heuristic",
        "foveacast",
    ]

    loaded = load_project(_write_project(tmp_path, project))

    assert loaded.project.experiments[0].prominence_provider_ids == (
        "heuristic",
        "foveacast",
    )
    assert loaded.runtime.prominence.version == "heuristic-project-v2"
    assert loaded.runtime.saliency.model_set == ("foveacast-v0.2.0",)
    assert loaded.runtime.saliency.execution_provider_preference == "cpu"
    assert loaded.runtime.saliency.cache.enabled is False
    assert loaded.runtime.saliency.aggregation.version == "aggregation-project-v2"
    assert loaded.runtime.saliency.stage_selector.version == "stage-project-v2"
    assert loaded.runtime.saliency.stage_selector.mixtures == {"initial": {"1s": 1.0}}


def test_unknown_prominence_provider_id_is_rejected_before_execution(
    tmp_path: Path,
) -> None:
    project = _read_project()
    project["experiments"][0]["prominence_provider_ids"] = ["unknown"]

    with pytest.raises(ProjectConfigError, match="unknown prominence provider"):
        load_project(_write_project(tmp_path, project))


def test_duplicate_prominence_provider_ids_are_rejected_as_project_config_error(
    tmp_path: Path,
) -> None:
    project = _read_project()
    project["experiments"][0]["prominence_provider_ids"] = ["heuristic", "heuristic"]

    with pytest.raises(ProjectConfigError, match="duplicate prominence provider"):
        load_project(_write_project(tmp_path, project))


@pytest.mark.parametrize(
    "model_set",
    ([""], ["foveacast-v0.2.0", "foveacast-v0.2.0"]),
)
def test_saliency_model_ids_must_be_non_empty_and_unique(
    tmp_path: Path, model_set: list[str]
) -> None:
    project = _read_project()
    project["providers"] = {"saliency": {"model_set": model_set}}

    with pytest.raises(ProjectConfigError, match="saliency model ID"):
        load_project(_write_project(tmp_path, project))


def test_invalid_heuristic_config_is_not_labeled_as_saliency_config(
    tmp_path: Path,
) -> None:
    project = _read_project()
    project["providers"] = {"prominence": {"weights": {"unknown": 1.0}}}

    with pytest.raises(ProjectConfigError, match="invalid prominence config") as error:
        load_project(_write_project(tmp_path, project))

    assert "invalid saliency config" not in str(error.value)


def test_scenario_timeout_can_be_null_or_omitted(tmp_path: Path) -> None:
    project = _read_project()
    project["scenarios"][0]["budget"]["timeout_seconds"] = None

    loaded_with_null = load_project(_write_project(tmp_path, project, "null.yaml"))

    assert loaded_with_null.project.scenarios[0].budget.timeout_seconds is None

    del project["scenarios"][0]["budget"]["timeout_seconds"]
    loaded_without_field = load_project(
        _write_project(tmp_path, project, "omitted.yaml")
    )

    assert loaded_without_field.project.scenarios[0].budget.timeout_seconds is None


def test_scenario_timeout_rejects_non_positive_finite_value(tmp_path: Path) -> None:
    project = _read_project()
    project["scenarios"][0]["budget"]["timeout_seconds"] = 0

    with pytest.raises(ProjectConfigError, match="timeout_seconds"):
        load_project(_write_project(tmp_path, project))


def test_scenario_budget_loads_max_model_calls(tmp_path: Path) -> None:
    project = _read_project()
    project["scenarios"][0]["budget"]["max_model_calls"] = 7

    loaded = load_project(_write_project(tmp_path, project))

    assert loaded.project.scenarios[0].budget.max_model_calls == 7


def test_progressive_attention_defaults_to_two_element_batch(tmp_path: Path) -> None:
    project = _read_project()
    project.setdefault("providers", {}).setdefault("attention", {}).pop(
        "batch_size", None
    )

    loaded = load_project(_write_project(tmp_path, project))

    assert loaded.runtime.attention.batch_size == 2
    assert loaded.runtime.attention.cross_region_exploration == 1
    assert loaded.runtime.attention.version == "progressive-attention-v4"
    assert loaded.runtime.attention.recovery_scent_threshold == pytest.approx(0.9)
    assert loaded.runtime.attention.recovery_after_misses == 2


def test_demo_calibrates_strong_scent_weight_for_progressive_attention() -> None:
    demo_path = Path(__file__).parents[3] / "benchmarks" / "demo" / "project.yaml"

    loaded = load_project(demo_path)

    assert loaded.runtime.attention.batch_size == 2
    assert loaded.runtime.attention.coarse_scent_weight == pytest.approx(1.0)


def test_demo_loads_four_cell_focused_validation_experiment() -> None:
    demo_path = Path(__file__).parents[3] / "benchmarks" / "demo" / "project.yaml"

    loaded = load_project(demo_path)
    experiment = next(
        item for item in loaded.project.experiments if item.id == "focused-validation"
    )

    assert experiment.scenario_ids == ("invite-teammate",)
    assert experiment.application_version_ids == (
        "fixture-app-defective",
        "fixture-app-improved",
    )
    assert experiment.persona_ids == ("first-time-nontechnical",)
    assert experiment.run_count == 1


def test_demo_uses_reduced_local_budget_matrices() -> None:
    demo_path = Path(__file__).parents[3] / "benchmarks" / "demo" / "project.yaml"

    loaded = load_project(demo_path)
    experiments = {item.id: item for item in loaded.project.experiments}

    expected_scenarios = ("invite-teammate", "enable-2fa")
    expected_versions = ("fixture-app-defective", "fixture-app-improved")
    expected_personas = ("first-time-nontechnical",)
    expected_policies = {
        "core-pair": (
            ExperimentPolicy.FULL_LIST,
            ExperimentPolicy.PROGRESSIVE_PROMINENCE_SCENT,
        ),
        "ablations": (
            ExperimentPolicy.PROMINENCE_RANKED_LIST,
            ExperimentPolicy.PROGRESSIVE_PROMINENCE,
        ),
    }

    for experiment_id, policies in expected_policies.items():
        experiment = experiments[experiment_id]
        assert experiment.scenario_ids == expected_scenarios
        assert experiment.application_version_ids == expected_versions
        assert experiment.persona_ids == expected_personas
        assert experiment.policies == policies
        assert experiment.run_count == 1
        assert experiment.model_trials == (0,)
        assert (
            len(
                expand_experiment(
                    ExperimentContext(
                        definition=experiment,
                        project=loaded.project,
                        config_digest=loaded.config_digest,
                    )
                )
            )
            == 8
        )

    baseline = experiments["baseline-model-trials"]
    assert baseline.scenario_ids == ("invite-teammate",)
    assert baseline.application_version_ids == expected_versions
    assert baseline.persona_ids == expected_personas
    assert baseline.policies == (
        ExperimentPolicy.FULL_LIST,
        ExperimentPolicy.PROGRESSIVE_PROMINENCE_SCENT,
    )
    assert baseline.seeds == (0,)
    assert baseline.model_trials == (0, 1)
    assert baseline.run_count == 1
    assert (
        len(
            expand_experiment(
                ExperimentContext(
                    definition=baseline,
                    project=loaded.project,
                    config_digest=loaded.config_digest,
                )
            )
        )
        == 8
    )


def test_portfolio_benchmark_is_small_visual_only_matrix() -> None:
    path = Path(__file__).parents[3] / "benchmarks" / "portfolio" / "project.yaml"
    loaded = load_project(path)
    experiment = loaded.project.experiments[0]
    application = loaded.project.applications[0]
    version = application.versions[0]

    assert len(loaded.project.scenarios) == 3
    assert len(application.versions) == 1
    assert version.kind is ApplicationVersionKind.LIVE
    assert version.start_url == "https://mohamed-khalil.vercel.app"
    assert version.allowed_origins == (
        "https://fonts.googleapis.com",
        "https://fonts.gstatic.com",
    )
    assert version.navigation_settle_ms == 2400
    assert version.action_settle_ms == 2800
    assert loaded.runtime.saliency.fallback.enabled is False
    assert loaded.runtime.attention.batch_size == 3
    assert loaded.runtime.attention.cross_region_exploration == 1
    assert experiment.id == "portfolio-foveacast-vs-heuristic"
    assert experiment.application_version_ids == ("portfolio-live",)
    assert experiment.prominence_provider_ids == ("heuristic", "foveacast")
    assert experiment.policies == (ExperimentPolicy.PROGRESSIVE_PROMINENCE,)
    assert experiment.seeds == (0,)
    assert experiment.model_trials == (0,)
    assert experiment.run_count == 1
    expected_scenarios = (
        (
            "Find Mohamed's current employer, role, and primary web frontend technologies.",
            "PAIR Systems",
            "Frontend Engineer",
            ("PAIR Systems", "React", "TypeScript"),
        ),
        (
            "Find stated city and country coverage for Mohamed's open-source prayer-time project.",
            "Open Prayer Times",
            "48,000+",
            ("cities calibrated", "130+", "countries"),
        ),
        (
            "Find Mohamed's public email address and whether he is open to remote frontend work.",
            "Contact",
            "m.khalil.bus@gmail.com",
            ("Open to mobile and web roles", "open to remote"),
        ),
    )
    for scenario, (goal, target_label, text, all_of) in zip(
        loaded.project.scenarios, expected_scenarios, strict=True
    ):
        assert scenario.goal == goal
        assert scenario.fixture_inputs.values == {}
        assert scenario.viewport_width == 1440
        assert scenario.viewport_height == 900
        assert scenario.budget.max_steps == 60
        assert scenario.budget.max_observations == 30
        assert scenario.budget.max_interactions == 24
        assert scenario.budget.max_model_calls == 40
        assert scenario.budget.timeout_seconds is None
        assert isinstance(scenario.verifier, VisibleResultVerifierSpec)
        assert scenario.verifier.text == text
        assert scenario.verifier.all_of == all_of
        assert scenario.verifier.role is None
        assert scenario.evaluation_target.labels_by_version == {"live": target_label}
        assert scenario.evaluation_target.role == "link"
    specs = expand_experiment(
        ExperimentContext(
            definition=experiment,
            project=loaded.project,
            config_digest=loaded.config_digest,
        )
    )
    assert len(specs) == 6
    assert tuple(
        (
            spec.scenario.id,
            spec.prominence_provider_id,
            spec.persona.id,
            spec.application_version.id,
            spec.policy.value,
            spec.seed,
            spec.model_trial,
        )
        for spec in specs
    ) == (
        (
            "current-pair-role",
            "heuristic",
            "first-time-nontechnical",
            "portfolio-live",
            "progressive-prominence",
            0,
            0,
        ),
        (
            "current-pair-role",
            "foveacast",
            "first-time-nontechnical",
            "portfolio-live",
            "progressive-prominence",
            0,
            0,
        ),
        (
            "open-prayer-times-scale",
            "heuristic",
            "first-time-nontechnical",
            "portfolio-live",
            "progressive-prominence",
            0,
            0,
        ),
        (
            "open-prayer-times-scale",
            "foveacast",
            "first-time-nontechnical",
            "portfolio-live",
            "progressive-prominence",
            0,
            0,
        ),
        (
            "contact-remote-role",
            "heuristic",
            "first-time-nontechnical",
            "portfolio-live",
            "progressive-prominence",
            0,
            0,
        ),
        (
            "contact-remote-role",
            "foveacast",
            "first-time-nontechnical",
            "portfolio-live",
            "progressive-prominence",
            0,
            0,
        ),
    )


def test_loads_versioned_runtime_provider_and_evaluation_formulas(
    tmp_path: Path,
) -> None:
    project = _read_project()
    project["providers"] = {
        "prominence": {
            "version": "prominence-project-v2",
            "weights": {"contrast": 0.7, "occlusion": -0.3},
            "temperature": 0.75,
        },
        "attention": {
            "version": "attention-project-v2",
            "batch_size": 2,
            "cross_region_exploration": 1,
            "prominence_weight": 1.4,
            "coarse_scent_weight": 0.6,
            "novelty_penalty": 0.2,
            "failure_penalty": 0.4,
            "recovery_scent_threshold": 0.85,
            "recovery_after_misses": 3,
        },
        "expectation": {"enabled": False},
    }
    project["scenarios"][0]["viewport"] = {"width": 900, "height": 700}
    project["scenarios"][0]["evaluation_target"] = {
        "labels_by_version": {
            "defective": "Share",
            "improved": "Invite teammate",
        },
        "roles_by_version": {"defective": "button", "improved": "link"},
    }
    project["evaluation"] = {
        "discovery_cost": {
            "version": "discovery-project-v2",
            "inspection_cost": 2.0,
            "region_cost": 3.0,
            "scroll_cost": 4.0,
            "wrong_action_cost": 5.0,
            "backtrack_cost": 6.0,
            "uncertainty_cost": 7.0,
            "abandonment_penalty": 8.0,
        },
        "findings": {
            "version": "finding-project-v2",
            "weak_target_prominence_below": 0.2,
            "weak_scent_below": 0.25,
            "misleading_scent_margin": 0.15,
            "excessive_navigation_depth_at_least": 3,
            "wrong_action_count_at_least": 1,
        },
        "state_updates": {
            "version": "state-project-v2",
            "success_confidence_delta": 0.1,
            "success_frustration_delta": -0.2,
            "failure_confidence_delta": -0.2,
            "failure_frustration_delta": 0.3,
        },
    }

    loaded = load_project(_write_project(tmp_path, project))

    assert loaded.runtime.prominence.version == "prominence-project-v2"
    assert loaded.runtime.prominence.weights["contrast"] == pytest.approx(0.7)
    assert loaded.runtime.attention.version == "attention-project-v2"
    assert loaded.runtime.attention.batch_size == 2
    assert loaded.runtime.attention.cross_region_exploration == 1
    assert loaded.runtime.attention.recovery_scent_threshold == pytest.approx(0.85)
    assert loaded.runtime.attention.recovery_after_misses == 3
    assert loaded.runtime.discovery_cost.version == "discovery-project-v2"
    assert loaded.runtime.findings.version == "finding-project-v2"
    assert loaded.runtime.state_updates.version == "state-project-v2"
    assert loaded.runtime.expectation_enabled is False
    assert loaded.project.scenarios[0].viewport_width == 900
    assert loaded.project.scenarios[0].viewport_height == 700
    target = loaded.project.scenarios[0].evaluation_target
    assert target.roles_by_version == {"defective": "button", "improved": "link"}


def test_expectation_provider_cannot_be_enabled_in_poc(tmp_path: Path) -> None:
    project = _read_project()
    project["providers"] = {"expectation": {"enabled": True}}

    with pytest.raises(ProjectConfigError, match="expectation"):
        load_project(_write_project(tmp_path, project))


def test_domain_project_is_immutable() -> None:
    project = load_project(FIXTURE_PATH).project

    with pytest.raises(FrozenInstanceError):
        project.name = "changed"  # type: ignore[misc]


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    project = _read_project()
    project["personas"].append(dict(project["personas"][0]))

    with pytest.raises(ProjectConfigError, match="duplicate persona id"):
        load_project(_write_project(tmp_path, project))


def test_scenario_references_existing_application_versions(tmp_path: Path) -> None:
    project = _read_project()
    project["scenarios"][0]["application_version_ids"] = ["missing"]

    with pytest.raises(ProjectConfigError, match="unknown application version"):
        load_project(_write_project(tmp_path, project))


@pytest.mark.parametrize("missing_kind", ["defective", "improved"])
def test_application_requires_defective_and_improved_versions(
    tmp_path: Path, missing_kind: str
) -> None:
    project = _read_project()
    project["applications"][0]["versions"] = [
        version
        for version in project["applications"][0]["versions"]
        if version["kind"] != missing_kind
    ]

    with pytest.raises(
        ProjectConfigError, match=f"missing {missing_kind} application version"
    ):
        load_project(_write_project(tmp_path, project))


def test_unsupported_verifier_type_is_rejected(tmp_path: Path) -> None:
    project = _read_project()
    project["scenarios"][0]["verifier"]["type"] = "unsupported"

    with pytest.raises(ProjectConfigError, match="unsupported verifier type"):
        load_project(_write_project(tmp_path, project))


def test_invalid_persona_parameter_range_is_rejected(tmp_path: Path) -> None:
    project = _read_project()
    project["personas"][0]["working_memory_capacity"] = 0

    with pytest.raises(ProjectConfigError, match="working_memory_capacity"):
        load_project(_write_project(tmp_path, project))


def test_zero_run_count_is_rejected(tmp_path: Path) -> None:
    project = _read_project()
    project["experiments"][0]["run_count"] = 0

    with pytest.raises(ProjectConfigError, match="run_count"):
        load_project(_write_project(tmp_path, project))


def test_digest_is_stable_when_yaml_key_order_changes(tmp_path: Path) -> None:
    project = _read_project()
    reordered = {key: project[key] for key in reversed(list(project))}
    reordered["applications"] = [
        {key: application[key] for key in reversed(list(application))}
        for application in project["applications"]
    ]

    first = load_project(_write_project(tmp_path, project, "first.yaml"))
    second = load_project(_write_project(tmp_path, reordered, "second.yaml"))

    assert first.config_digest == second.config_digest


def test_additive_saliency_experiment_keeps_legacy_selected_digest(
    tmp_path: Path,
) -> None:
    base_project = _read_project()
    additive_project = copy.deepcopy(base_project)
    additive_experiment = copy.deepcopy(additive_project["experiments"][0])
    additive_experiment["id"] = "saliency-focused-validation"
    additive_experiment["name"] = "Focused saliency validation"
    additive_project["experiments"].append(additive_experiment)

    base = load_project(_write_project(tmp_path, base_project, "base.yaml"))
    additive = load_project(_write_project(tmp_path, additive_project, "additive.yaml"))

    assert base.config_digest_for("smoke") == additive.config_digest_for("smoke")
