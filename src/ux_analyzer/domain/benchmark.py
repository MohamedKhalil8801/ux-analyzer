"""Platform-neutral benchmark configuration domain contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class ApplicationVersionKind(StrEnum):
    """Presentation variant under evaluation."""

    DEFECTIVE = "defective"
    IMPROVED = "improved"


class ExperimentPolicy(StrEnum):
    """Policy used to expose interface elements to an agent."""

    FULL_LIST = "full-list"
    PROMINENCE_RANKED_LIST = "prominence-ranked-list"
    PROGRESSIVE_PROMINENCE = "progressive-prominence"
    PROGRESSIVE_PROMINENCE_SCENT = "progressive-prominence-scent"


class VerifierOperator(StrEnum):
    """Supported comparison operations for fixture-state verification."""

    EQUALS = "equals"
    NOT_EQUALS = "not-equals"
    CONTAINS = "contains"
    TRUTHY = "truthy"
    FALSY = "falsy"


@dataclass(frozen=True, slots=True)
class FixtureInputs:
    """Typed scenario values and exact values that require redaction."""

    values: Mapping[str, str]
    sensitive_keys: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        copied_values = dict(self.values)
        if not self.sensitive_keys.issubset(copied_values):
            unknown_keys = sorted(self.sensitive_keys.difference(copied_values))
            raise ValueError(f"sensitive fixture keys are unknown: {unknown_keys}")
        object.__setattr__(self, "values", MappingProxyType(copied_values))
        object.__setattr__(self, "sensitive_keys", frozenset(self.sensitive_keys))


@dataclass(frozen=True, slots=True)
class Budget:
    """Finite resources available to one benchmark run."""

    max_steps: int
    max_observations: int
    max_interactions: int
    timeout_seconds: float

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")
        if self.max_observations <= 0:
            raise ValueError("max_observations must be greater than zero")
        if self.max_interactions <= 0:
            raise ValueError("max_interactions must be greater than zero")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")


@dataclass(frozen=True, slots=True)
class VerifierSpecBase:
    """Common identity for supported verifier specifications."""

    type: str


@dataclass(frozen=True, slots=True)
class FixtureStateVerifierSpec(VerifierSpecBase):
    """Verifier that compares private fixture state with a scenario input."""

    resource: str
    field: str
    operator: VerifierOperator
    expected_fixture_key: str


@dataclass(frozen=True, slots=True)
class VisibleResultVerifierSpec(VerifierSpecBase):
    """Verifier that checks persona-visible result text."""

    text: str
    role: str | None = None


type VerifierSpec = FixtureStateVerifierSpec | VisibleResultVerifierSpec


@dataclass(frozen=True, slots=True)
class ApplicationVersion:
    """Immutable presentation version of an application."""

    id: str
    kind: ApplicationVersionKind
    label: str


@dataclass(frozen=True, slots=True)
class Application:
    """Application and its defective/improved presentation variants."""

    id: str
    name: str
    versions: tuple[ApplicationVersion, ...]

    def __post_init__(self) -> None:
        versions = tuple(self.versions)
        if not versions:
            raise ValueError("application must define at least one version")
        if len({version.id for version in versions}) != len(versions):
            raise ValueError(f"application {self.id!r} has duplicate version IDs")
        kinds = {version.kind for version in versions}
        missing = [kind.value for kind in ApplicationVersionKind if kind not in kinds]
        if missing:
            raise ValueError(
                f"application {self.id!r} is missing {', '.join(missing)} version"
            )
        object.__setattr__(self, "versions", versions)


@dataclass(frozen=True, slots=True)
class Scenario:
    """Goal, inputs, constraints, and verification contract for one task."""

    id: str
    name: str
    goal: str
    application_version_ids: tuple[str, ...]
    start_state: str
    fixture_inputs: FixtureInputs
    budget: Budget
    verifier: VerifierSpec
    safeguards: tuple[str, ...]
    eligible_persona_ids: tuple[str, ...]
    expected_evidence: tuple[str, ...]
    viewport_width: int = 1280
    viewport_height: int = 800

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "application_version_ids", tuple(self.application_version_ids)
        )
        object.__setattr__(self, "safeguards", tuple(self.safeguards))
        object.__setattr__(
            self, "eligible_persona_ids", tuple(self.eligible_persona_ids)
        )
        object.__setattr__(self, "expected_evidence", tuple(self.expected_evidence))
        if self.viewport_width <= 0 or self.viewport_height <= 0:
            raise ValueError("scenario viewport dimensions must be greater than zero")


@dataclass(frozen=True, slots=True)
class Persona:
    """Deterministic simulated-user parameters."""

    id: str
    name: str
    working_memory_capacity: int
    initial_confidence: float
    initial_frustration: float
    abandonment_threshold: float
    attention_temperature: float

    def __post_init__(self) -> None:
        if self.working_memory_capacity <= 0:
            raise ValueError("working_memory_capacity must be greater than zero")
        for parameter_name, parameter in (
            ("initial_confidence", self.initial_confidence),
            ("initial_frustration", self.initial_frustration),
            ("abandonment_threshold", self.abandonment_threshold),
        ):
            if not 0 <= parameter <= 1:
                raise ValueError(f"{parameter_name} must be between 0 and 1")
        if self.attention_temperature <= 0:
            raise ValueError("attention_temperature must be greater than zero")


@dataclass(frozen=True, slots=True)
class ExperimentDefinition:
    """Stable Cartesian-product definition for benchmark run expansion."""

    id: str
    name: str
    scenario_ids: tuple[str, ...]
    application_version_ids: tuple[str, ...]
    persona_ids: tuple[str, ...]
    policies: tuple[ExperimentPolicy, ...]
    seeds: tuple[int, ...]
    run_count: int

    def __post_init__(self) -> None:
        if self.run_count <= 0:
            raise ValueError("run_count must be greater than zero")
        object.__setattr__(self, "scenario_ids", tuple(self.scenario_ids))
        object.__setattr__(
            self, "application_version_ids", tuple(self.application_version_ids)
        )
        object.__setattr__(self, "persona_ids", tuple(self.persona_ids))
        object.__setattr__(self, "policies", tuple(self.policies))
        object.__setattr__(self, "seeds", tuple(self.seeds))


@dataclass(frozen=True, slots=True)
class BenchmarkProject:
    """Complete immutable benchmark configuration."""

    id: str
    name: str
    applications: tuple[Application, ...]
    scenarios: tuple[Scenario, ...]
    personas: tuple[Persona, ...]
    experiments: tuple[ExperimentDefinition, ...]

    def __post_init__(self) -> None:
        applications = tuple(self.applications)
        scenarios = tuple(self.scenarios)
        personas = tuple(self.personas)
        experiments = tuple(self.experiments)
        for collection_name, items in (
            ("application", applications),
            ("scenario", scenarios),
            ("persona", personas),
            ("experiment", experiments),
        ):
            ids = [item.id for item in items]
            if len(set(ids)) != len(ids):
                raise ValueError(f"duplicate {collection_name} ID")
        object.__setattr__(self, "applications", applications)
        object.__setattr__(self, "scenarios", scenarios)
        object.__setattr__(self, "personas", personas)
        object.__setattr__(self, "experiments", experiments)
