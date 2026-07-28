"""Pydantic models for the YAML configuration boundary."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


def _empty_int_list() -> list[int]:
    return []


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


class ProjectModel(_ConfigModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    applications: list[ApplicationModel] = Field(min_length=1)
    scenarios: list[ScenarioModel] = Field(min_length=1)
    personas: list[PersonaModel] = Field(min_length=1)
    experiments: list[ExperimentModel] = Field(min_length=1)
