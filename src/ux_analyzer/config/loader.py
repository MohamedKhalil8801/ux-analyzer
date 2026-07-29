"""Load, validate, resolve, and fingerprint benchmark YAML configuration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml
from pydantic import ValidationError

from ux_analyzer.application.evaluation import DiscoveryCostConfig
from ux_analyzer.application.state_updates import StateUpdateConfig
from ux_analyzer.config.models import (
    ApplicationModel,
    ExperimentModel,
    FixtureStateVerifierModel,
    ProjectModel,
    ScenarioModel,
)
from ux_analyzer.domain.benchmark import (
    Application,
    ApplicationVersion,
    ApplicationVersionKind,
    BenchmarkProject,
    Budget,
    ExperimentDefinition,
    ExperimentPolicy,
    FixtureInputs,
    FixtureStateVerifierSpec,
    Persona,
    Scenario,
    VerifierOperator,
    VerifierSpec,
    VisibleResultVerifierSpec,
)
from ux_analyzer.providers.attention_policy import AttentionPolicyConfig
from ux_analyzer.providers.finding_rules import FindingRuleConfig
from ux_analyzer.providers.prominence import (
    DEFAULT_PROMINENCE_WEIGHTS,
    HeuristicProminenceConfig,
)


class ProjectConfigError(ValueError):
    """Raised when project YAML is invalid or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Resolved versioned provider and evaluation formulas for composition."""

    prominence: HeuristicProminenceConfig
    attention: AttentionPolicyConfig
    discovery_cost: DiscoveryCostConfig
    findings: FindingRuleConfig
    state_updates: StateUpdateConfig


@dataclass(frozen=True, slots=True)
class LoadedProject:
    """Validated domain project plus canonical configuration fingerprint."""

    project: BenchmarkProject
    config_digest: str
    runtime: RuntimeConfig

    @property
    def digest(self) -> str:
        """Return canonical SHA-256 digest for callers using the short name."""

        return self.config_digest


def load_project(path: Path) -> LoadedProject:
    """Load one YAML project into immutable domain objects."""

    try:
        raw_config_value: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ProjectConfigError(
            f"cannot read project config {path}: {error}"
        ) from error
    except yaml.YAMLError as error:
        raise ProjectConfigError(f"invalid YAML in {path}: {error}") from error

    if not isinstance(raw_config_value, dict):
        raise ProjectConfigError("project config root must be a YAML mapping")
    raw_config = cast(dict[object, object], raw_config_value)
    _reject_unsupported_verifier_types(raw_config)

    try:
        config = ProjectModel.model_validate(raw_config)
    except ValidationError as error:
        raise ProjectConfigError(_format_validation_error(error)) from error

    _validate_references(config)
    project = _to_domain(config)
    digest = _canonical_digest(config.model_dump(mode="json"))
    return LoadedProject(
        project=project,
        config_digest=digest,
        runtime=_to_runtime(config),
    )


def _to_runtime(config: ProjectModel) -> RuntimeConfig:
    prominence = config.providers.prominence
    attention = config.providers.attention
    discovery = config.evaluation.discovery_cost
    findings = config.evaluation.findings
    state_updates = config.evaluation.state_updates
    return RuntimeConfig(
        prominence=HeuristicProminenceConfig(
            version=prominence.version,
            weights=prominence.weights or dict(DEFAULT_PROMINENCE_WEIGHTS),
            temperature=prominence.temperature,
        ),
        attention=AttentionPolicyConfig(
            version=attention.version,
            batch_size=attention.batch_size,
            temperature=1.0,
            prominence_weight=attention.prominence_weight,
            coarse_scent_weight=attention.coarse_scent_weight,
            novelty_penalty=attention.novelty_penalty,
            failure_penalty=attention.failure_penalty,
        ),
        discovery_cost=DiscoveryCostConfig(
            version=discovery.version,
            inspection_cost=discovery.inspection_cost,
            region_cost=discovery.region_cost,
            scroll_cost=discovery.scroll_cost,
            wrong_action_cost=discovery.wrong_action_cost,
            backtrack_cost=discovery.backtrack_cost,
            uncertainty_cost=discovery.uncertainty_cost,
            abandonment_penalty=discovery.abandonment_penalty,
        ),
        findings=FindingRuleConfig(
            version=findings.version,
            weak_target_prominence_below=findings.weak_target_prominence_below,
            weak_scent_below=findings.weak_scent_below,
            misleading_scent_margin=findings.misleading_scent_margin,
            excessive_navigation_depth_at_least=(
                findings.excessive_navigation_depth_at_least
            ),
            wrong_action_count_at_least=findings.wrong_action_count_at_least,
        ),
        state_updates=StateUpdateConfig(
            version=state_updates.version,
            success_confidence_delta=state_updates.success_confidence_delta,
            success_frustration_delta=state_updates.success_frustration_delta,
            failure_confidence_delta=state_updates.failure_confidence_delta,
            failure_frustration_delta=state_updates.failure_frustration_delta,
        ),
    )


def _format_validation_error(error: ValidationError) -> str:
    first_error = error.errors()[0]
    location = ".".join(str(part) for part in first_error["loc"])
    message = str(first_error["msg"])
    return f"invalid project config at {location}: {message}"


def _reject_unsupported_verifier_types(raw_config: dict[object, object]) -> None:
    scenarios_value = raw_config.get("scenarios", [])
    if not isinstance(scenarios_value, list):
        return
    scenarios = cast(list[object], scenarios_value)
    for scenario_value in scenarios:
        scenario = scenario_value
        if not isinstance(scenario, dict):
            continue
        scenario_mapping = cast(dict[object, object], scenario)
        verifier = scenario_mapping.get("verifier")
        if not isinstance(verifier, dict):
            continue
        verifier_mapping = cast(dict[object, object], verifier)
        verifier_type = verifier_mapping.get("type")
        if verifier_type not in {"fixture-state", "visible-result"}:
            raise ProjectConfigError(f"unsupported verifier type: {verifier_type!r}")


def _validate_references(config: ProjectModel) -> None:
    _assert_unique_ids(
        "application", (application.id for application in config.applications)
    )
    _assert_unique_ids("scenario", (scenario.id for scenario in config.scenarios))
    _assert_unique_ids("persona", (persona.id for persona in config.personas))
    _assert_unique_ids(
        "experiment", (experiment.id for experiment in config.experiments)
    )

    version_ids: set[str] = set()
    versions_by_id: dict[str, ApplicationVersionKind] = {}
    for application in config.applications:
        kinds = {version.kind for version in application.versions}
        for required_kind in ("defective", "improved"):
            if required_kind not in kinds:
                raise ProjectConfigError(
                    f"application {application.id!r} missing {required_kind} application version"
                )
        for version in application.versions:
            if version.id in version_ids:
                raise ProjectConfigError(
                    f"duplicate application version id: {version.id}"
                )
            version_ids.add(version.id)
            versions_by_id[version.id] = ApplicationVersionKind(version.kind)

    persona_ids = {persona.id for persona in config.personas}
    scenario_ids = {scenario.id for scenario in config.scenarios}
    for scenario in config.scenarios:
        for version_id in scenario.application_version_ids:
            if version_id not in versions_by_id:
                raise ProjectConfigError(
                    f"scenario {scenario.id!r} references unknown application version "
                    f"{version_id!r}"
                )
        for persona_id in scenario.eligible_persona_ids:
            if persona_id not in persona_ids:
                raise ProjectConfigError(
                    f"scenario {scenario.id!r} references unknown persona {persona_id!r}"
                )
        if isinstance(scenario.verifier, FixtureStateVerifierModel):
            if scenario.verifier.expected_fixture_key not in scenario.fixture_inputs:
                raise ProjectConfigError(
                    f"scenario {scenario.id!r} references unknown fixture input "
                    f"{scenario.verifier.expected_fixture_key!r}"
                )

    for experiment in config.experiments:
        for scenario_id in experiment.scenario_ids:
            if scenario_id not in scenario_ids:
                raise ProjectConfigError(
                    f"experiment {experiment.id!r} references unknown scenario {scenario_id!r}"
                )
        for version_id in experiment.application_version_ids:
            if version_id not in versions_by_id:
                raise ProjectConfigError(
                    f"experiment {experiment.id!r} references unknown application version "
                    f"{version_id!r}"
                )
        for persona_id in experiment.persona_ids:
            if persona_id not in persona_ids:
                raise ProjectConfigError(
                    f"experiment {experiment.id!r} references unknown persona {persona_id!r}"
                )


def _assert_unique_ids(kind: str, ids: Iterable[str]) -> None:
    values = list(ids)
    if len(values) != len(set(values)):
        raise ProjectConfigError(f"duplicate {kind} id")


def _to_domain(config: ProjectModel) -> BenchmarkProject:
    applications = tuple(
        _to_application(application) for application in config.applications
    )
    scenarios = tuple(_to_scenario(scenario) for scenario in config.scenarios)
    personas = tuple(
        Persona(
            id=persona.id,
            name=persona.name,
            working_memory_capacity=persona.working_memory_capacity,
            initial_confidence=persona.initial_confidence,
            initial_frustration=persona.initial_frustration,
            abandonment_threshold=persona.abandonment_threshold,
            attention_temperature=persona.attention_temperature,
        )
        for persona in config.personas
    )
    experiments = tuple(_to_experiment(experiment) for experiment in config.experiments)
    return BenchmarkProject(
        id=config.id,
        name=config.name,
        applications=applications,
        scenarios=scenarios,
        personas=personas,
        experiments=experiments,
    )


def _to_application(application: ApplicationModel) -> Application:
    return Application(
        id=application.id,
        name=application.name,
        versions=tuple(
            ApplicationVersion(
                id=version.id,
                kind=ApplicationVersionKind(version.kind),
                label=version.label,
            )
            for version in application.versions
        ),
    )


def _to_scenario(scenario: ScenarioModel) -> Scenario:
    verifier: VerifierSpec
    if isinstance(scenario.verifier, FixtureStateVerifierModel):
        verifier = FixtureStateVerifierSpec(
            type=scenario.verifier.type,
            resource=scenario.verifier.resource,
            field=scenario.verifier.field,
            operator=VerifierOperator(scenario.verifier.operator),
            expected_fixture_key=scenario.verifier.expected_fixture_key,
        )
    else:
        verifier = VisibleResultVerifierSpec(
            type=scenario.verifier.type,
            text=scenario.verifier.text,
            role=scenario.verifier.role,
        )
    return Scenario(
        id=scenario.id,
        name=scenario.name,
        goal=scenario.goal,
        application_version_ids=tuple(scenario.application_version_ids),
        start_state=scenario.start_state,
        fixture_inputs=FixtureInputs(
            values={key: value.value for key, value in scenario.fixture_inputs.items()},
            sensitive_keys=frozenset(
                key for key, value in scenario.fixture_inputs.items() if value.sensitive
            ),
        ),
        budget=Budget(
            max_steps=scenario.budget.max_steps,
            max_observations=scenario.budget.max_observations,
            max_interactions=scenario.budget.max_interactions,
            timeout_seconds=scenario.budget.timeout_seconds,
        ),
        verifier=verifier,
        safeguards=tuple(scenario.safeguards),
        eligible_persona_ids=tuple(scenario.eligible_persona_ids),
        expected_evidence=tuple(scenario.expected_evidence),
    )


def _to_experiment(experiment: ExperimentModel) -> ExperimentDefinition:
    return ExperimentDefinition(
        id=experiment.id,
        name=experiment.name,
        scenario_ids=tuple(experiment.scenario_ids),
        application_version_ids=tuple(experiment.application_version_ids),
        persona_ids=tuple(experiment.persona_ids),
        policies=tuple(ExperimentPolicy(policy) for policy in experiment.policies),
        seeds=tuple(experiment.seeds),
        run_count=experiment.run_count,
    )


def _canonical_digest(payload: dict[str, object]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
