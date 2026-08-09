"""Load, validate, resolve, and fingerprint benchmark YAML configuration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

import yaml
from pydantic import ValidationError

from ux_analyzer.application.evaluation import DiscoveryCostConfig
from ux_analyzer.application.state_updates import StateUpdateConfig
from ux_analyzer.config.models import (
    ApplicationModel,
    ExperimentModel,
    FixtureStateVerifierModel,
    FrozenExpectationDocumentModel,
    ProjectModel,
    SaliencyProviderModel,
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
    ScenarioEvaluationTarget,
    VerifierOperator,
    VerifierSpec,
    VisibleResultVerifierSpec,
    resolve_prominence_provider_id,
)
from ux_analyzer.domain.expectations import ExpectationKey, FrozenExpectation
from ux_analyzer.providers.attention_policy import AttentionPolicyConfig
from ux_analyzer.providers.finding_rules import FindingRuleConfig
from ux_analyzer.providers.prominence import (
    DEFAULT_PROMINENCE_WEIGHTS,
    HeuristicProminenceConfig,
)
from ux_analyzer.providers.saliency_aggregation import SaliencyAggregationConfig


class ProjectConfigError(ValueError):
    """Raised when project YAML is invalid or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class SaliencyCacheConfig:
    """Experiment-scoped cache policy for learned saliency evidence."""

    enabled: bool = True
    scope: str = "experiment"


@dataclass(frozen=True, slots=True)
class SaliencyStageSelectorConfig:
    """Versioned stage mixture and temperature configuration."""

    version: str = "saliency-stage-selector-v1"
    temperature: float = 1.0
    mixtures: Mapping[str, Mapping[str, float]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class SaliencyFallbackConfig:
    """Operational fallback policy when learned prominence is unavailable."""

    enabled: bool = True
    provider_id: str = "heuristic"


@dataclass(frozen=True, slots=True)
class SaliencyRuntimeConfig:
    """Resolved learned saliency provider configuration."""

    model_set: tuple[str, ...]
    precision: str
    execution_provider_preference: str
    cache: SaliencyCacheConfig
    aggregation: SaliencyAggregationConfig
    stage_selector: SaliencyStageSelectorConfig
    fallback: SaliencyFallbackConfig


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Resolved versioned provider and evaluation formulas for composition."""

    prominence: HeuristicProminenceConfig
    attention: AttentionPolicyConfig
    discovery_cost: DiscoveryCostConfig
    findings: FindingRuleConfig
    state_updates: StateUpdateConfig
    expectations: tuple[FrozenExpectation, ...]
    expectation_enabled: bool
    saliency: SaliencyRuntimeConfig


@dataclass(frozen=True, slots=True)
class LoadedProject:
    """Validated domain project plus canonical configuration fingerprint."""

    project: BenchmarkProject
    config_digest: str
    runtime: RuntimeConfig
    experiment_digests: Mapping[str, str]

    @property
    def digest(self) -> str:
        """Return canonical SHA-256 digest for callers using the short name."""

        return self.config_digest

    def config_digest_for(self, experiment_id: str) -> str:
        """Return identity digest scoped to one selected experiment."""

        try:
            return self.experiment_digests[experiment_id]
        except KeyError as error:
            raise ValueError(f"unknown experiment ID: {experiment_id}") from error


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
    digest_payload = config.model_dump(mode="json")
    digest = _canonical_digest(digest_payload)
    experiment_digests = MappingProxyType(
        {
            experiment.id: _canonical_digest_for_experiment(
                digest_payload, experiment.id
            )
            for experiment in config.experiments
        }
    )
    runtime = _to_runtime(config)
    return LoadedProject(
        project=project,
        config_digest=digest,
        runtime=runtime,
        experiment_digests=experiment_digests,
    )


def _to_runtime(config: ProjectModel) -> RuntimeConfig:
    prominence = config.providers.prominence
    attention = config.providers.attention
    discovery = config.evaluation.discovery_cost
    findings = config.evaluation.findings
    state_updates = config.evaluation.state_updates
    saliency = config.providers.saliency or SaliencyProviderModel()
    try:
        prominence_runtime = HeuristicProminenceConfig(
            version=prominence.version,
            weights=prominence.weights or dict(DEFAULT_PROMINENCE_WEIGHTS),
            temperature=prominence.temperature,
        )
    except ValueError as error:
        raise ProjectConfigError(f"invalid prominence config: {error}") from error
    try:
        saliency_runtime = _to_saliency_runtime(saliency)
    except ValueError as error:
        raise ProjectConfigError(f"invalid saliency config: {error}") from error
    return RuntimeConfig(
        prominence=prominence_runtime,
        attention=AttentionPolicyConfig(
            version=attention.version,
            batch_size=attention.batch_size,
            cross_region_exploration=attention.cross_region_exploration,
            temperature=1.0,
            prominence_weight=attention.prominence_weight,
            coarse_scent_weight=attention.coarse_scent_weight,
            novelty_penalty=attention.novelty_penalty,
            failure_penalty=attention.failure_penalty,
            recovery_scent_threshold=attention.recovery_scent_threshold,
            recovery_after_misses=attention.recovery_after_misses,
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
        expectations=_to_expectations(config.providers.expectation.documents),
        expectation_enabled=config.providers.expectation.enabled,
        saliency=saliency_runtime,
    )


def _to_expectations(
    documents: Iterable[FrozenExpectationDocumentModel],
) -> tuple[FrozenExpectation, ...]:
    try:
        return tuple(
            FrozenExpectation(
                expectation_id=document.id,
                schema_version=document.schema_version,
                key=ExpectationKey(
                    application_version_id=document.application_version_id,
                    scenario_id=document.scenario_id,
                    persona_id=document.persona_id,
                ),
                desired_outcomes=tuple(document.desired_outcomes),
                required_invariants=tuple(document.required_invariants),
                acceptable_alternatives=tuple(document.acceptable_alternatives),
                reference_paths=tuple(tuple(path) for path in document.reference_paths),
                effort_bounds=dict(document.effort_bounds),
                warning_signals=tuple(document.warning_signals),
            )
            for document in documents
        )
    except (TypeError, ValueError) as error:
        raise ProjectConfigError(
            f"invalid frozen expectation document: {error}"
        ) from error


def _to_saliency_runtime(config: SaliencyProviderModel) -> SaliencyRuntimeConfig:
    aggregation = config.aggregation
    stage_selector = config.stage_selector
    mixtures = {
        stage: MappingProxyType(dict(values))
        for stage, values in stage_selector.mixtures.items()
    }
    resolve_prominence_provider_id(config.fallback.provider_id)
    return SaliencyRuntimeConfig(
        model_set=tuple(config.model_set),
        precision=config.precision,
        execution_provider_preference=config.execution_provider_preference,
        cache=SaliencyCacheConfig(
            enabled=config.cache.enabled,
            scope=config.cache.scope,
        ),
        aggregation=SaliencyAggregationConfig(
            version=aggregation.version,
            density_weight=aggregation.density_weight,
            robust_peak_weight=aggregation.robust_peak_weight,
            mass_share_weight=aggregation.mass_share_weight,
            temperature=aggregation.temperature,
            meaningful_score_threshold=aggregation.meaningful_score_threshold,
            semantic_roles=tuple(aggregation.semantic_roles),
            structural_roles=tuple(aggregation.structural_roles),
        ),
        stage_selector=SaliencyStageSelectorConfig(
            version=stage_selector.version,
            temperature=stage_selector.temperature,
            mixtures=MappingProxyType(mixtures),
        ),
        fallback=SaliencyFallbackConfig(
            enabled=config.fallback.enabled,
            provider_id=config.fallback.provider_id,
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
        if kinds != {"live"}:
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
    _validate_expectation_references(
        config,
        versions_by_id=versions_by_id,
        scenario_ids=scenario_ids,
        persona_ids=persona_ids,
    )
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
        live_version_ids = tuple(
            version_id
            for version_id in scenario.application_version_ids
            if versions_by_id[version_id] is ApplicationVersionKind.LIVE
        )
        if live_version_ids and scenario.fixture_inputs:
            raise ProjectConfigError(
                f"scenario {scenario.id!r} referencing live application versions "
                "must not define fixture_inputs"
            )
        if live_version_ids and isinstance(
            scenario.verifier, FixtureStateVerifierModel
        ):
            raise ProjectConfigError(
                f"scenario {scenario.id!r} referencing live application versions "
                "must not use fixture-state verifier"
            )
        if isinstance(scenario.verifier, FixtureStateVerifierModel):
            if scenario.verifier.expected_fixture_key not in scenario.fixture_inputs:
                raise ProjectConfigError(
                    f"scenario {scenario.id!r} references unknown fixture input "
                    f"{scenario.verifier.expected_fixture_key!r}"
                )
        for version_id in scenario.application_version_ids:
            version_kind = versions_by_id[version_id].value
            labels = scenario.evaluation_target.labels_by_version
            if version_id not in labels and version_kind not in labels:
                raise ProjectConfigError(
                    f"scenario {scenario.id!r} evaluation target has no label for "
                    f"application version {version_id!r}"
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
        if len(experiment.prominence_provider_ids) != len(
            set(experiment.prominence_provider_ids)
        ):
            raise ProjectConfigError(
                f"duplicate prominence provider ID in experiment {experiment.id!r}"
            )
        for provider_id in experiment.prominence_provider_ids:
            try:
                resolve_prominence_provider_id(provider_id)
            except ValueError as error:
                raise ProjectConfigError(str(error)) from error


def _assert_unique_ids(kind: str, ids: Iterable[str]) -> None:
    values = list(ids)
    if len(values) != len(set(values)):
        raise ProjectConfigError(f"duplicate {kind} id")


def _validate_expectation_references(
    config: ProjectModel,
    *,
    versions_by_id: Mapping[str, ApplicationVersionKind],
    scenario_ids: set[str],
    persona_ids: set[str],
) -> None:
    scenarios_by_id = {scenario.id: scenario for scenario in config.scenarios}
    seen_keys: set[tuple[str, str, str]] = set()
    for document in config.providers.expectation.documents:
        key = (
            document.application_version_id,
            document.scenario_id,
            document.persona_id,
        )
        if key in seen_keys:
            raise ProjectConfigError(
                "duplicate frozen expectation key: "
                f"{document.application_version_id!r}, "
                f"{document.scenario_id!r}, {document.persona_id!r}"
            )
        seen_keys.add(key)

        if document.application_version_id not in versions_by_id:
            raise ProjectConfigError(
                f"frozen expectation {document.id!r} references unknown "
                f"application version {document.application_version_id!r}"
            )
        scenario = scenarios_by_id.get(document.scenario_id)
        if scenario is None or document.scenario_id not in scenario_ids:
            raise ProjectConfigError(
                f"frozen expectation {document.id!r} references unknown scenario "
                f"{document.scenario_id!r}"
            )
        if document.persona_id != "*" and document.persona_id not in persona_ids:
            raise ProjectConfigError(
                f"frozen expectation {document.id!r} references unknown persona "
                f"{document.persona_id!r}"
            )
        if document.application_version_id not in scenario.application_version_ids:
            raise ProjectConfigError(
                f"frozen expectation {document.id!r} references application version "
                f"{document.application_version_id!r} outside scenario "
                f"{document.scenario_id!r}"
            )


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
                start_url=version.start_url,
                allowed_origins=tuple(version.allowed_origins),
                navigation_settle_ms=version.navigation_settle_ms,
                action_settle_ms=version.action_settle_ms,
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
            all_of=tuple(scenario.verifier.all_of),
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
            max_model_calls=scenario.budget.max_model_calls,
        ),
        verifier=verifier,
        safeguards=tuple(scenario.safeguards),
        eligible_persona_ids=tuple(scenario.eligible_persona_ids),
        expected_evidence=tuple(scenario.expected_evidence),
        evaluation_target=ScenarioEvaluationTarget(
            labels_by_version=scenario.evaluation_target.labels_by_version,
            role=scenario.evaluation_target.role,
            roles_by_version=scenario.evaluation_target.roles_by_version,
            region_label=scenario.evaluation_target.region_label,
        ),
        viewport_width=scenario.viewport.width,
        viewport_height=scenario.viewport.height,
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
        model_trials=tuple(experiment.model_trials),
        prominence_provider_ids=tuple(experiment.prominence_provider_ids),
    )


def _canonical_digest(payload: dict[str, object]) -> str:
    normalized = _digest_compatibility_payload(payload)
    return _digest_normalized_payload(normalized)


def _canonical_digest_for_experiment(
    payload: dict[str, object], experiment_id: str
) -> str:
    normalized = _digest_compatibility_payload(payload)
    experiments_value = normalized.get("experiments")
    if not isinstance(experiments_value, list):
        raise ProjectConfigError("project config has no experiments")
    selected: list[object] = []
    for item in cast(list[object], experiments_value):
        if (
            isinstance(item, dict)
            and cast(dict[object, object], item).get("id") == experiment_id
        ):
            selected.append(cast(dict[object, object], item))
    if not selected:
        raise ProjectConfigError(f"unknown experiment ID: {experiment_id}")
    normalized["experiments"] = selected
    return _digest_normalized_payload(normalized)


def _digest_normalized_payload(normalized: dict[str, object]) -> str:
    canonical = json.dumps(
        normalized,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _digest_compatibility_payload(payload: dict[str, object]) -> dict[str, object]:
    """Keep omitted default settings out of legacy config identity."""

    normalized = dict(payload)
    applications_value = normalized.get("applications")
    if isinstance(applications_value, list):
        applications: list[object] = []
        for item in cast(list[object], applications_value):
            if isinstance(item, dict):
                application = dict(cast(dict[str, object], item))
                versions_value = application.get("versions")
                if isinstance(versions_value, list):
                    versions: list[object] = []
                    for version_value in cast(list[object], versions_value):
                        if isinstance(version_value, dict):
                            version = dict(cast(dict[str, object], version_value))
                            if version.get("start_url") is None:
                                version.pop("start_url", None)
                            if version.get("allowed_origins") == []:
                                version.pop("allowed_origins", None)
                            if version.get("navigation_settle_ms") == 0:
                                version.pop("navigation_settle_ms", None)
                            if version.get("action_settle_ms") == 0:
                                version.pop("action_settle_ms", None)
                            versions.append(version)
                        else:
                            versions.append(version_value)
                    application["versions"] = versions
                applications.append(application)
            else:
                applications.append(item)
        normalized["applications"] = applications
    scenarios_value = normalized.get("scenarios")
    if isinstance(scenarios_value, list):
        scenarios: list[object] = []
        for item in cast(list[object], scenarios_value):
            if isinstance(item, dict):
                scenario = dict(cast(dict[str, object], item))
                verifier_value = scenario.get("verifier")
                if isinstance(verifier_value, dict):
                    verifier = dict(cast(dict[str, object], verifier_value))
                    if verifier.get("all_of") == []:
                        verifier.pop("all_of", None)
                    scenario["verifier"] = verifier
                scenarios.append(scenario)
            else:
                scenarios.append(item)
        normalized["scenarios"] = scenarios
    providers_value = normalized.get("providers")
    if isinstance(providers_value, dict):
        providers = dict(cast(dict[str, object], providers_value))
        if providers.get("saliency") is None:
            providers.pop("saliency", None)
        expectation_value = providers.get("expectation")
        if isinstance(expectation_value, dict):
            expectation = dict(cast(dict[str, object], expectation_value))
            documents_value = expectation.get("documents")
            if expectation.get("enabled") is False and documents_value == []:
                expectation.pop("provider_id", None)
                expectation.pop("documents", None)
            elif isinstance(documents_value, list):
                documents = [
                    dict(cast(dict[str, object], document))
                    for document in cast(list[object], documents_value)
                    if isinstance(document, dict)
                ]
                documents.sort(
                    key=lambda document: (
                        str(document.get("application_version_id", "")),
                        str(document.get("scenario_id", "")),
                        str(document.get("persona_id", "")),
                        str(document.get("id", "")),
                    )
                )
                expectation["documents"] = documents
            providers["expectation"] = expectation
        normalized["providers"] = providers
    experiments_value = normalized.get("experiments")
    if isinstance(experiments_value, list):
        experiments: list[object] = []
        for item in cast(list[object], experiments_value):
            if isinstance(item, dict):
                experiment = dict(cast(dict[str, object], item))
                if experiment.get("prominence_provider_ids") == ["heuristic"]:
                    experiment.pop("prominence_provider_ids", None)
                experiments.append(experiment)
            else:
                experiments.append(item)
        normalized["experiments"] = experiments
    return normalized
