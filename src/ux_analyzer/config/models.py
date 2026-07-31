"""Pydantic models for the YAML configuration boundary."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


def _empty_int_list() -> list[int]:
    return []


def _empty_float_mapping() -> dict[str, float]:
    return {}


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ApplicationVersionModel(_ConfigModel):
    id: str = Field(min_length=1)
    kind: Literal["defective", "improved"]
    label: str = Field(min_length=1)


class ApplicationModel(_ConfigModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    versions: list[ApplicationVersionModel] = Field(min_length=1)


class FixtureInputModel(_ConfigModel):
    value: str
    sensitive: bool = False


class FixtureStateVerifierModel(_ConfigModel):
    type: Literal["fixture-state"]
    resource: str = Field(min_length=1)
    field: str = Field(min_length=1)
    operator: Literal["equals", "not-equals", "contains", "truthy", "falsy"]
    expected_fixture_key: str = Field(min_length=1)


class VisibleResultVerifierModel(_ConfigModel):
    type: Literal["visible-result"]
    text: str = Field(min_length=1)
    role: str | None = Field(default=None, min_length=1)


VerifierModel = Annotated[
    FixtureStateVerifierModel | VisibleResultVerifierModel,
    Field(discriminator="type"),
]


class BudgetModel(_ConfigModel):
    max_steps: int = Field(gt=0)
    max_observations: int = Field(gt=0)
    max_interactions: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)


class ViewportModel(_ConfigModel):
    width: int = Field(default=1280, gt=0)
    height: int = Field(default=800, gt=0)


class ScenarioEvaluationTargetModel(_ConfigModel):
    labels_by_version: dict[str, str] = Field(min_length=1)
    role: str | None = Field(default=None, min_length=1)
    region_label: str | None = Field(default=None, min_length=1)


class ScenarioModel(_ConfigModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    application_version_ids: list[str] = Field(min_length=1)
    start_state: str = Field(min_length=1)
    fixture_inputs: dict[str, FixtureInputModel] = Field(default_factory=dict)
    budget: BudgetModel
    verifier: VerifierModel
    safeguards: list[str] = Field(default_factory=list)
    eligible_persona_ids: list[str] = Field(min_length=1)
    expected_evidence: list[str] = Field(default_factory=list)
    evaluation_target: ScenarioEvaluationTargetModel
    viewport: ViewportModel = Field(default_factory=ViewportModel)


class PersonaModel(_ConfigModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    working_memory_capacity: int = Field(gt=0, le=100)
    initial_confidence: float = Field(ge=0, le=1)
    initial_frustration: float = Field(ge=0, le=1)
    abandonment_threshold: float = Field(ge=0, le=1)
    attention_temperature: float = Field(gt=0)


class ExperimentModel(_ConfigModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    scenario_ids: list[str] = Field(min_length=1)
    application_version_ids: list[str] = Field(min_length=1)
    persona_ids: list[str] = Field(min_length=1)
    policies: list[
        Literal[
            "full-list",
            "prominence-ranked-list",
            "progressive-prominence",
            "progressive-prominence-scent",
        ]
    ] = Field(min_length=1)
    seeds: list[int] = Field(default_factory=_empty_int_list)
    run_count: int = Field(gt=0)


class ProminenceProviderModel(_ConfigModel):
    version: str = Field(default="heuristic-prominence-v1", min_length=1)
    weights: dict[str, float] = Field(default_factory=_empty_float_mapping)
    temperature: float = Field(default=1.0, gt=0)


class AttentionProviderModel(_ConfigModel):
    version: str = Field(default="progressive-attention-v1", min_length=1)
    batch_size: int = Field(default=1, ge=1, le=3)
    prominence_weight: float = Field(default=1.0, ge=0)
    coarse_scent_weight: float = Field(default=0.5, ge=0)
    novelty_penalty: float = Field(default=0.25, ge=0, le=1)
    failure_penalty: float = Field(default=0.5, ge=0, le=1)


class ExpectationProviderModel(_ConfigModel):
    enabled: Literal[False] = False


class ProvidersModel(_ConfigModel):
    prominence: ProminenceProviderModel = Field(default_factory=ProminenceProviderModel)
    attention: AttentionProviderModel = Field(default_factory=AttentionProviderModel)
    expectation: ExpectationProviderModel = Field(
        default_factory=ExpectationProviderModel
    )


class DiscoveryCostModel(_ConfigModel):
    version: str = Field(default="discovery-cost-v1", min_length=1)
    inspection_cost: float = Field(default=1.0, ge=0)
    region_cost: float = Field(default=1.0, ge=0)
    scroll_cost: float = Field(default=1.0, ge=0)
    wrong_action_cost: float = Field(default=1.0, ge=0)
    backtrack_cost: float = Field(default=1.0, ge=0)
    uncertainty_cost: float = Field(default=1.0, ge=0)
    abandonment_penalty: float = Field(default=1.0, ge=0)


class FindingRulesModel(_ConfigModel):
    version: str = Field(default="finding-rules-v1", min_length=1)
    weak_target_prominence_below: float = Field(default=0.25, ge=0, le=1)
    weak_scent_below: float = Field(default=0.30, ge=0, le=1)
    misleading_scent_margin: float = Field(default=0.20, ge=0, le=1)
    excessive_navigation_depth_at_least: int = Field(default=4, ge=1)
    wrong_action_count_at_least: int = Field(default=2, ge=1)


class StateUpdatesModel(_ConfigModel):
    version: str = Field(default="state-updates-v1", min_length=1)
    success_confidence_delta: float = 0.05
    success_frustration_delta: float = -0.1
    failure_confidence_delta: float = -0.1
    failure_frustration_delta: float = 0.2


class EvaluationModel(_ConfigModel):
    discovery_cost: DiscoveryCostModel = Field(default_factory=DiscoveryCostModel)
    findings: FindingRulesModel = Field(default_factory=FindingRulesModel)
    state_updates: StateUpdatesModel = Field(default_factory=StateUpdatesModel)


class ProjectModel(_ConfigModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    applications: list[ApplicationModel] = Field(min_length=1)
    scenarios: list[ScenarioModel] = Field(min_length=1)
    personas: list[PersonaModel] = Field(min_length=1)
    experiments: list[ExperimentModel] = Field(min_length=1)
    providers: ProvidersModel = Field(default_factory=ProvidersModel)
    evaluation: EvaluationModel = Field(default_factory=EvaluationModel)
