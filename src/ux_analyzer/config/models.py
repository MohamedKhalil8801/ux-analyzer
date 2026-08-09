"""Pydantic models for the YAML configuration boundary."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ux_analyzer.domain.benchmark import (
    canonicalize_http_origin,
    canonicalize_https_url,
)


def _empty_int_list() -> list[int]:
    return []


def _default_model_trials() -> list[int]:
    return [0]


def _default_prominence_provider_ids() -> list[str]:
    return ["heuristic"]


def _default_saliency_model_set() -> list[str]:
    return ["foveacast-v0.2.0"]


def _default_saliency_semantic_roles() -> list[str]:
    return ["button", "checkbox", "input", "link", "menu", "tab", "text"]


def _default_saliency_structural_roles() -> list[str]:
    return ["other"]


def _default_saliency_stage_mixtures() -> dict[str, dict[str, float]]:
    return {
        "initial": {"1s": 1.0},
        "exploration": {"3s": 1.0},
        "persistent": {"3s": 0.25, "7s": 0.75},
    }


def _empty_float_mapping() -> dict[str, float]:
    return {}


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ApplicationVersionModel(_ConfigModel):
    id: str = Field(min_length=1)
    kind: Literal["defective", "improved", "live"]
    label: str = Field(min_length=1)
    start_url: str | None = None
    allowed_origins: list[str] = Field(default_factory=list)
    navigation_settle_ms: int = Field(default=0, ge=0)
    action_settle_ms: int = Field(default=0, ge=0)

    @field_validator("start_url")
    @classmethod
    def _canonicalize_start_url(cls, start_url: str | None) -> str | None:
        if start_url is None:
            return None
        return canonicalize_https_url(start_url)

    @field_validator("allowed_origins")
    @classmethod
    def _canonicalize_allowed_origins(cls, origins: list[str]) -> list[str]:
        canonical_origins = [canonicalize_http_origin(origin) for origin in origins]
        if len(canonical_origins) != len(set(canonical_origins)):
            raise ValueError("allowed origins must be unique")
        return canonical_origins

    @model_validator(mode="after")
    def _validate_live_start_url(self) -> ApplicationVersionModel:
        if self.kind == "live" and self.start_url is None:
            raise ValueError("live application version requires start_url")
        return self


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
    all_of: list[str] = Field(default_factory=list)

    @field_validator("all_of")
    @classmethod
    def _validate_all_of(cls, all_of: list[str]) -> list[str]:
        if any(not value.strip() for value in all_of):
            raise ValueError("all_of strings must not be empty")
        if len(all_of) != len(set(all_of)):
            raise ValueError("all_of strings must be unique")
        return all_of


VerifierModel = Annotated[
    FixtureStateVerifierModel | VisibleResultVerifierModel,
    Field(discriminator="type"),
]


class BudgetModel(_ConfigModel):
    max_steps: int = Field(gt=0)
    max_observations: int = Field(gt=0)
    max_interactions: int = Field(gt=0)
    max_model_calls: int = Field(default=64, gt=0)
    timeout_seconds: float | None = Field(default=None, gt=0)


class ViewportModel(_ConfigModel):
    width: int = Field(default=1280, gt=0)
    height: int = Field(default=800, gt=0)


class ScenarioEvaluationTargetModel(_ConfigModel):
    labels_by_version: dict[str, str] = Field(min_length=1)
    role: str | None = Field(default=None, min_length=1)
    roles_by_version: dict[str, str] = Field(default_factory=dict)
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
    model_trials: list[int] = Field(default_factory=_default_model_trials, min_length=1)
    prominence_provider_ids: list[str] = Field(
        default_factory=_default_prominence_provider_ids, min_length=1
    )
    run_count: int = Field(gt=0)


class ProminenceProviderModel(_ConfigModel):
    version: str = Field(default="heuristic-prominence-v1", min_length=1)
    weights: dict[str, float] = Field(default_factory=_empty_float_mapping)
    temperature: float = Field(default=1.0, gt=0)


class AttentionProviderModel(_ConfigModel):
    version: str = Field(default="progressive-attention-v4", min_length=1)
    batch_size: int = Field(default=2, ge=1, le=3)
    cross_region_exploration: int = Field(default=1, ge=0, le=2)
    prominence_weight: float = Field(default=1.0, ge=0)
    coarse_scent_weight: float = Field(default=0.5, ge=0)
    novelty_penalty: float = Field(default=0.25, ge=0, le=1)
    failure_penalty: float = Field(default=0.5, ge=0, le=1)
    recovery_scent_threshold: float = Field(default=0.9, ge=0, le=1)
    recovery_after_misses: int = Field(default=2, ge=1)


class FrozenExpectationDocumentModel(_ConfigModel):
    id: str = Field(min_length=1)
    schema_version: Literal["frozen-expectation-v1"]
    application_version_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    persona_id: str = Field(min_length=1)
    desired_outcomes: list[str] = Field(min_length=1)
    required_invariants: list[str] = Field(default_factory=list)
    acceptable_alternatives: list[str] = Field(default_factory=list)
    reference_paths: list[list[str]] = Field(default_factory=list)
    effort_bounds: dict[str, float] = Field(default_factory=_empty_float_mapping)
    warning_signals: list[str] = Field(default_factory=list)


class ExpectationProviderModel(_ConfigModel):
    enabled: bool = False
    provider_id: Literal["frozen-expectation-v1"] = "frozen-expectation-v1"
    documents: list[FrozenExpectationDocumentModel] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_documents_when_enabled(self) -> ExpectationProviderModel:
        if self.enabled and not self.documents:
            raise ValueError(
                "enabled expectation provider requires at least one document"
            )
        return self


class SaliencyCacheModel(_ConfigModel):
    enabled: bool = True
    scope: Literal["experiment"] = "experiment"


class SaliencyAggregationModel(_ConfigModel):
    version: str = Field(default="element-saliency-aggregation-v1", min_length=1)
    density_weight: float = Field(default=0.60, ge=0)
    robust_peak_weight: float = Field(default=0.25, ge=0)
    mass_share_weight: float = Field(default=0.15, ge=0)
    temperature: float = Field(default=1.0, gt=0)
    meaningful_score_threshold: float = Field(default=1e-9, ge=0)
    semantic_roles: list[str] = Field(
        default_factory=_default_saliency_semantic_roles, min_length=1
    )
    structural_roles: list[str] = Field(
        default_factory=_default_saliency_structural_roles, min_length=1
    )


class SaliencyStageSelectorModel(_ConfigModel):
    version: str = Field(default="saliency-stage-selector-v1", min_length=1)
    temperature: float = Field(default=1.0, gt=0)
    mixtures: dict[str, dict[str, float]] = Field(
        default_factory=_default_saliency_stage_mixtures,
        validation_alias=AliasChoices("mixtures", "stage_mixtures"),
    )


class SaliencyFallbackModel(_ConfigModel):
    enabled: bool = True
    provider_id: str = Field(default="heuristic", min_length=1)


class SaliencyProviderModel(_ConfigModel):
    model_set: list[str] = Field(
        default_factory=_default_saliency_model_set, min_length=1
    )
    precision: Literal["fp16"] = "fp16"
    execution_provider_preference: Literal["auto", "cpu", "directml"] = Field(
        default="auto",
        validation_alias=AliasChoices(
            "execution_provider_preference", "execution_preference"
        ),
    )
    cache: SaliencyCacheModel = Field(default_factory=SaliencyCacheModel)
    aggregation: SaliencyAggregationModel = Field(
        default_factory=SaliencyAggregationModel
    )
    stage_selector: SaliencyStageSelectorModel = Field(
        default_factory=SaliencyStageSelectorModel
    )
    fallback: SaliencyFallbackModel = Field(default_factory=SaliencyFallbackModel)

    @field_validator("model_set")
    @classmethod
    def _validate_model_set(cls, model_set: list[str]) -> list[str]:
        if any(not model_id.strip() for model_id in model_set):
            raise ValueError("saliency model IDs must not be empty")
        if len(model_set) != len(set(model_set)):
            raise ValueError("saliency model IDs must be unique")
        return model_set


class ProvidersModel(_ConfigModel):
    prominence: ProminenceProviderModel = Field(default_factory=ProminenceProviderModel)
    attention: AttentionProviderModel = Field(default_factory=AttentionProviderModel)
    expectation: ExpectationProviderModel = Field(
        default_factory=ExpectationProviderModel
    )
    saliency: SaliencyProviderModel | None = None


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
