"""Evidence-preserving benchmark metrics and variant comparison gates."""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from ux_analyzer.domain.attention import (
    Back,
    InteractWithElement,
    PersonaObservation,
    Scroll,
)
from ux_analyzer.domain.benchmark import (
    ApplicationVersionKind,
    ScenarioEvaluationTarget,
)
from ux_analyzer.domain.findings import (
    Evidence,
    EvidenceClass,
    Metric,
    Reproducibility,
    UnsupportedHumanClaimError,
)
from ux_analyzer.domain.interface import ElementSnapshot
from ux_analyzer.domain.run import (
    ActionExecuted,
    AgentAbandoned,
    ObservationRecorded,
    ProviderManifest,
    RunOutcome,
    RunOutcomeKind,
    ViewportCaptured,
)

if TYPE_CHECKING:
    from ux_analyzer.application.run_agent import (
        ProminenceEvidence,
        RunResult,
        ScentEvidence,
    )
    from ux_analyzer.application.saliency import ProminenceResult


def _empty_float_mapping() -> dict[str, float]:
    return {}


def _empty_text_mapping() -> dict[str, str]:
    return {}


def _empty_stage_score_mapping() -> dict[str, dict[str, float]]:
    return {}


_SAFE_PROVENANCE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_SENSITIVE_PROVENANCE_MARKERS = (
    "access_token",
    "api_key",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)
_REDACTED_PROVENANCE = "[REDACTED]"


def _safe_provenance_identifier(value: object) -> str:
    """Keep provider evidence identifiers within public summary allowlists."""

    text = str(value).strip()
    lowered = text.casefold()
    if (
        not text
        or ".." in text
        or any(marker in lowered for marker in _SENSITIVE_PROVENANCE_MARKERS)
        or _SAFE_PROVENANCE_PATTERN.fullmatch(text) is None
    ):
        return _REDACTED_PROVENANCE
    return text


def _safe_provenance_reason(value: object) -> str:
    """Keep fallback diagnostics bounded and free of credential-like text."""

    text = str(value).strip()
    lowered = text.casefold()
    if (
        not text
        or len(text) > 512
        or any(character in text for character in "\x00\r\n")
        or any(marker in lowered for marker in _SENSITIVE_PROVENANCE_MARKERS)
    ):
        return _REDACTED_PROVENANCE
    return text


class EvaluationMetric(StrEnum):
    """Stable names for metrics emitted by this evaluator."""

    TARGET_DISCOVERY_RANK = "target-discovery-rank"
    INSPECTED_ELEMENTS = "inspected-elements"
    INSPECTED_REGIONS = "inspected-regions"
    SCROLLS = "scrolls"
    WRONG_ACTIONS = "wrong-actions"
    BACKTRACKS = "backtracks"
    VERIFIED_COMPLETION = "verified-completion"
    CLAIMED_COMPLETION = "claimed-completion"
    FALSE_SUCCESS = "false-success"
    TARGET_PROMINENCE = "target-prominence"
    TARGET_SCENT = "target-scent"
    STRONGEST_COMPETING_SCENT = "strongest-competing-scent"
    TARGET_BELOW_FOLD = "target-below-fold"
    UNEXPECTED_HIERARCHY = "unexpected-hierarchy"
    AMBIGUOUS_TARGET = "ambiguous-target"
    NAVIGATION_DEPTH = "navigation-depth"
    FEEDBACK_OBSERVED = "feedback-observed"
    RECOVERY_ACTIONS = "recovery-actions"
    RECOVERY_SUCCESS = "recovery-success"
    DISCOVERY_COST = "discovery-cost"


class EvaluationEvidenceUnavailable(ValueError):
    """Raised when declared target evidence cannot be resolved safely."""

    def __init__(self, detail: str) -> None:
        self.safe_reason = f"evaluation evidence unavailable: {detail}"
        super().__init__(self.safe_reason)


@dataclass(frozen=True, slots=True)
class DiscoveryCostConfig:
    """Versioned weights for the public discovery-cost formula."""

    version: str = "discovery-cost-v1"
    inspection_cost: float = 1.0
    region_cost: float = 1.0
    scroll_cost: float = 1.0
    wrong_action_cost: float = 1.0
    backtrack_cost: float = 1.0
    uncertainty_cost: float = 1.0
    abandonment_penalty: float = 1.0

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("discovery cost version must not be empty")
        for name in (
            "inspection_cost",
            "region_cost",
            "scroll_cost",
            "wrong_action_cost",
            "backtrack_cost",
            "uncertainty_cost",
            "abandonment_penalty",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class DiscoveryCostBreakdown:
    """Every component of simulated discovery cost, kept separately."""

    inspection_cost: float
    region_cost: float
    scroll_cost: float
    wrong_action_cost: float
    backtrack_cost: float
    uncertainty_cost: float
    abandonment_penalty: float

    def __post_init__(self) -> None:
        for name in self.components:
            value = self.components[name]
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def components(self) -> Mapping[str, float]:
        return MappingProxyType(
            {
                "inspection_cost": self.inspection_cost,
                "region_cost": self.region_cost,
                "scroll_cost": self.scroll_cost,
                "wrong_action_cost": self.wrong_action_cost,
                "backtrack_cost": self.backtrack_cost,
                "uncertainty_cost": self.uncertainty_cost,
                "abandonment_penalty": self.abandonment_penalty,
            }
        )

    @property
    def total(self) -> float:
        """Return exact sum of named formula components."""

        return sum(self.components.values())


@dataclass(frozen=True, slots=True)
class EvaluationTarget:
    """Persona-visible target identity used to interpret run events."""

    element_id: str | None
    region_id: str | None = None
    expected_label: str | None = None
    role: str | None = None

    def __post_init__(self) -> None:
        if self.element_id is not None and not self.element_id:
            raise ValueError("evaluation target element ID must not be empty")
        if self.expected_label is not None and not self.expected_label:
            raise ValueError("evaluation target label must not be empty")
        if self.element_id is None and self.expected_label is None:
            raise ValueError("evaluation target needs element ID or expected label")


@dataclass(frozen=True, slots=True)
class RunEvaluationInputs:
    """Optional evidence unavailable in the current typed run event model."""

    prominence_scores: Mapping[str, float] = field(default_factory=_empty_float_mapping)
    prominence_scores_by_stage: Mapping[str, Mapping[str, float]] = field(
        default_factory=_empty_stage_score_mapping
    )
    scent_scores: Mapping[str, float] = field(default_factory=_empty_float_mapping)
    target_prominence: float | None = None
    target_below_fold: bool | None = None
    unexpected_hierarchy: bool | None = None
    ambiguous_target: bool | None = None
    navigation_depth: int = 0
    feedback_observed: bool | None = None
    recovery_actions: int | None = None
    recovery_success: bool | None = None
    uncertainty: float | None = None
    model_dependent: bool = False
    prominence_provider_id: str | None = None
    active_search_stage: str | None = None
    target_prominence_profiles: Mapping[str, float] = field(
        default_factory=_empty_float_mapping
    )
    target_prominence_sources: Mapping[str, str] = field(
        default_factory=_empty_text_mapping
    )
    prominence_profile_event_ids: tuple[str, ...] = ()
    prominence_operational_event_ids: tuple[str, ...] = ()
    prominence_lineage_event_ids: tuple[str, ...] = ()
    prominence_model_id: str | None = None
    prominence_model_version: str | None = None
    prominence_provider_version: str | None = None
    prominence_fallback: bool = False
    prominence_fallback_reason: str | None = None

    def __post_init__(self) -> None:
        for name in ("prominence_scores", "scent_scores"):
            source = getattr(self, name)
            copied = {str(key): float(value) for key, value in source.items()}
            for key, value in copied.items():
                if not key or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"{name} must contain finite scores in [0, 1]")
            object.__setattr__(self, name, MappingProxyType(copied))
        stage_scores: dict[str, Mapping[str, float]] = {}
        for stage, scores in self.prominence_scores_by_stage.items():
            stage_name = _safe_provenance_identifier(stage)
            copied_scores = {str(key): float(value) for key, value in scores.items()}
            if any(
                not key.strip() or not math.isfinite(value) or not 0 <= value <= 1
                for key, value in copied_scores.items()
            ):
                raise ValueError("prominence stage scores must be finite in [0, 1]")
            stage_scores[stage_name] = MappingProxyType(copied_scores)
        object.__setattr__(
            self, "prominence_scores_by_stage", MappingProxyType(stage_scores)
        )
        profiles = {
            str(key): float(value)
            for key, value in self.target_prominence_profiles.items()
        }
        if any(
            not key.strip() or not math.isfinite(value) or not 0 <= value <= 1
            for key, value in profiles.items()
        ):
            raise ValueError("target prominence profiles must contain scores in [0, 1]")
        sources = {
            str(key): _safe_provenance_identifier(value)
            for key, value in self.target_prominence_sources.items()
        }
        if any(not key.strip() or not value.strip() for key, value in sources.items()):
            raise ValueError("target prominence sources need non-empty text")
        object.__setattr__(
            self, "target_prominence_profiles", MappingProxyType(profiles)
        )
        object.__setattr__(self, "target_prominence_sources", MappingProxyType(sources))
        for name in (
            "prominence_provider_id",
            "active_search_stage",
            "prominence_model_id",
            "prominence_model_version",
            "prominence_provider_version",
            "prominence_fallback_reason",
        ):
            value = getattr(self, name)
            if value is not None and not value.strip():
                raise ValueError(f"{name} must not be empty")
        for name in (
            "prominence_provider_id",
            "prominence_model_id",
            "prominence_model_version",
        ):
            value = getattr(self, name)
            if value is not None:
                if name == "prominence_provider_id":
                    value = _canonical_prominence_provider_id(value)
                object.__setattr__(self, name, _safe_provenance_identifier(value))
        if self.active_search_stage is not None:
            object.__setattr__(
                self,
                "active_search_stage",
                _safe_provenance_identifier(self.active_search_stage),
            )
        if self.prominence_fallback_reason is not None:
            object.__setattr__(
                self,
                "prominence_fallback_reason",
                _safe_provenance_reason(self.prominence_fallback_reason),
            )
        object.__setattr__(
            self,
            "prominence_profile_event_ids",
            tuple(self.prominence_profile_event_ids),
        )
        object.__setattr__(
            self,
            "prominence_operational_event_ids",
            tuple(self.prominence_operational_event_ids),
        )
        object.__setattr__(
            self,
            "prominence_lineage_event_ids",
            tuple(self.prominence_lineage_event_ids),
        )
        for name in ("target_prominence", "uncertainty"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be between 0 and 1")
        if self.navigation_depth < 0:
            raise ValueError("navigation depth must not be negative")
        if self.recovery_actions is not None and self.recovery_actions < 0:
            raise ValueError("recovery actions must not be negative")


@dataclass(frozen=True, slots=True)
class RunMetrics:
    """All per-run measurements with their evidence references."""

    run_id: str
    seed: int
    scenario_id: str
    application_version_id: str
    persona_id: str
    policy: str
    target: EvaluationTarget
    target_discovery_rank: int | None
    inspected_elements: int
    inspected_regions: int
    scrolls: int
    wrong_actions: int
    backtracks: int
    verified_completion: bool
    claimed_completion: bool
    false_success: bool
    abandoned: bool
    model_trial: int = 0
    prominence_provider_id: str = "heuristic"
    active_search_stage: str | None = None
    target_prominence: float | None = None
    target_scent: float | None = None
    strongest_competing_scent: float | None = None
    target_below_fold: bool | None = None
    unexpected_hierarchy: bool | None = None
    ambiguous_target: bool | None = None
    navigation_depth: int = 0
    feedback_observed: bool | None = None
    recovery_actions: int = 0
    recovery_success: bool | None = None
    outcome: str = ""
    reproducibility: Reproducibility = Reproducibility.SEEDED
    evidence: tuple[Evidence, ...] = ()
    metrics: tuple[Metric, ...] = ()
    discovery_cost: DiscoveryCostBreakdown = field(
        default_factory=lambda: DiscoveryCostBreakdown(0, 0, 0, 0, 0, 0, 0)
    )
    config_digest: str | None = None
    viewport_ids: tuple[str, ...] = ()
    element_ids: tuple[str, ...] = ()
    action_sequence: tuple[str, ...] = ()
    target_prominence_profiles: Mapping[str, float] = field(
        default_factory=_empty_float_mapping
    )
    target_prominence_sources: Mapping[str, str] = field(
        default_factory=_empty_text_mapping
    )
    prominence_profile_event_ids: tuple[str, ...] = ()
    prominence_operational_event_ids: tuple[str, ...] = ()
    prominence_lineage_event_ids: tuple[str, ...] = ()
    prominence_model_id: str | None = None
    prominence_model_version: str | None = None
    prominence_provider_version: str | None = None
    prominence_fallback: bool = False
    prominence_fallback_reason: str | None = None
    comparison_valid: bool = True

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "scenario_id",
            "application_version_id",
            "persona_id",
            "policy",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        for name in (
            "inspected_elements",
            "inspected_regions",
            "scrolls",
            "wrong_actions",
            "backtracks",
            "navigation_depth",
            "recovery_actions",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")
        if not self.prominence_provider_id.strip():
            raise ValueError("prominence provider ID must not be empty")
        object.__setattr__(
            self,
            "prominence_provider_id",
            _safe_provenance_identifier(
                _canonical_prominence_provider_id(self.prominence_provider_id)
            ),
        )
        if (
            self.active_search_stage is not None
            and not self.active_search_stage.strip()
        ):
            raise ValueError("active search stage must not be empty")
        if self.active_search_stage is not None:
            object.__setattr__(
                self,
                "active_search_stage",
                _safe_provenance_identifier(self.active_search_stage),
            )
        profiles = {
            str(key): float(value)
            for key, value in self.target_prominence_profiles.items()
        }
        if any(
            not key.strip() or not math.isfinite(value) or not 0 <= value <= 1
            for key, value in profiles.items()
        ):
            raise ValueError("target prominence profiles must contain scores in [0, 1]")
        sources = {
            str(key): _safe_provenance_identifier(value)
            for key, value in self.target_prominence_sources.items()
        }
        if any(not key.strip() or not value.strip() for key, value in sources.items()):
            raise ValueError("target prominence sources need non-empty text")
        for name in (
            "prominence_model_id",
            "prominence_model_version",
            "prominence_provider_version",
            "prominence_fallback_reason",
        ):
            value = getattr(self, name)
            if value is not None and not value.strip():
                raise ValueError(f"{name} must not be empty")
            if name != "prominence_fallback_reason" and value is not None:
                object.__setattr__(self, name, _safe_provenance_identifier(value))
        if self.prominence_fallback_reason is not None:
            object.__setattr__(
                self,
                "prominence_fallback_reason",
                _safe_provenance_reason(self.prominence_fallback_reason),
            )
        object.__setattr__(
            self, "target_prominence_profiles", MappingProxyType(profiles)
        )
        object.__setattr__(self, "target_prominence_sources", MappingProxyType(sources))
        object.__setattr__(
            self,
            "prominence_profile_event_ids",
            tuple(self.prominence_profile_event_ids),
        )
        object.__setattr__(
            self,
            "prominence_operational_event_ids",
            tuple(self.prominence_operational_event_ids),
        )
        object.__setattr__(
            self,
            "prominence_lineage_event_ids",
            tuple(self.prominence_lineage_event_ids),
        )
        for name in (
            "target_prominence",
            "target_scent",
            "strongest_competing_scent",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be between 0 and 1")
        object.__setattr__(
            self, "reproducibility", Reproducibility(self.reproducibility)
        )
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "viewport_ids", tuple(self.viewport_ids))
        object.__setattr__(self, "element_ids", tuple(self.element_ids))
        object.__setattr__(self, "action_sequence", tuple(self.action_sequence))

    @property
    def inspected_element_count(self) -> int:
        return self.inspected_elements

    @property
    def inspected_region_count(self) -> int:
        return self.inspected_regions

    @property
    def completion(self) -> bool:
        return self.verified_completion

    @property
    def reproducibility_label(self) -> str:
        """Describe model-dependent evidence with both identity axes."""

        if self.reproducibility is Reproducibility.MODEL_DEPENDENT:
            return (
                "model-dependent "
                f"(attention seed {self.seed}, model trial {self.model_trial})"
            )
        return self.reproducibility.value

    def metric(self, name: EvaluationMetric | str) -> Metric:
        selected = str(name)
        standard = _standard_metric_values(self).get(selected)
        if standard is not None:
            value, evidence_class = standard
            existing = next(
                (metric for metric in self.metrics if metric.name == selected), None
            )
            return Metric(
                name=selected,
                value=value,
                evidence_class=evidence_class,
                evidence_ids=existing.evidence_ids if existing is not None else (),
            )
        for metric in self.metrics:
            if metric.name == selected:
                return metric
        raise KeyError(selected)


@dataclass(frozen=True, slots=True)
class IntervalSummary:
    """Empirical median and central interval for repeated seeded runs."""

    count: int
    median: float
    lower: float
    upper: float
    confidence: float = 0.95

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("interval count must be positive")
        if not 0 < self.confidence < 1:
            raise ValueError("interval confidence must be between zero and one")
        values = (self.median, self.lower, self.upper)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("interval values must be finite")
        if not self.lower <= self.median <= self.upper:
            raise ValueError("interval must contain median")

    @property
    def interval(self) -> tuple[float, float]:
        return self.lower, self.upper


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Aggregate for one named metric and its trust class."""

    name: str
    interval: IntervalSummary
    evidence_class: EvidenceClass
    evidence_ids: tuple[str, ...]

    @property
    def median(self) -> float:
        return self.interval.median


@dataclass(frozen=True, slots=True)
class CellAggregate:
    """Per scenario/version/persona/policy/model-trial aggregate."""

    scenario_id: str
    application_version_id: str
    persona_id: str
    policy: str
    run_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    metric_summaries: Mapping[str, MetricSummary]
    reproducibility: Reproducibility
    reproducibility_summary: Mapping[str, int]
    evidence_ids: tuple[str, ...]
    model_trial: int = 0
    prominence_provider_id: str = "heuristic"
    active_search_stages: tuple[str, ...] = ()

    @property
    def run_count(self) -> int:
        return len(self.run_ids)

    @property
    def metrics(self) -> Mapping[str, MetricSummary]:
        return self.metric_summaries


@dataclass(frozen=True, slots=True)
class DirectionalGateResult:
    """Paired-seed directional acceptance result."""

    passed: bool
    paired_seed_count: int
    discovery_cost_decreased: bool
    wrong_action_burden_not_increased: bool
    backtrack_burden_not_increased: bool
    verified_completion_rate_not_regressed: bool
    baseline_discovery_cost_median: float
    improved_discovery_cost_median: float
    baseline_wrong_action_median: float
    improved_wrong_action_median: float
    baseline_backtrack_median: float
    improved_backtrack_median: float
    baseline_completion_rate: float
    improved_completion_rate: float
    reasons: tuple[str, ...] = ()

    @property
    def gate_passed(self) -> bool:
        return self.passed


@dataclass(frozen=True, slots=True)
class VariantComparison:
    """Comparison of two otherwise identical fixed-seed variant cells."""

    baseline: CellAggregate
    improved: CellAggregate
    paired_seeds: tuple[int, ...]
    gate: DirectionalGateResult
    paired_model_trials: tuple[int, ...] = ()

    @property
    def passed(self) -> bool:
        return self.gate.passed


@dataclass(frozen=True, slots=True)
class ExperimentEvaluation:
    """Persistable run metrics, cell aggregates, and paired variant gates."""

    run_metrics: tuple[RunMetrics, ...]
    cell_aggregates: tuple[CellAggregate, ...]
    variant_comparisons: tuple[VariantComparison, ...]


def evaluate_run(
    result: RunResult,
    target: EvaluationTarget | str | None = None,
    *,
    target_element_id: str | None = None,
    inputs: RunEvaluationInputs | None = None,
    cost_config: DiscoveryCostConfig | None = None,
) -> RunMetrics:
    """Derive metrics from typed run state and explicitly supplied evidence."""

    if target is not None and target_element_id is not None:
        raise ValueError("provide target or target_element_id, not both")
    selected_target = (
        target
        if isinstance(target, EvaluationTarget)
        else EvaluationTarget(target or target_element_id)
    )
    settings = inputs or RunEvaluationInputs()
    config = cost_config or DiscoveryCostConfig()
    state = result.state
    prominence_provider_id = _canonical_prominence_provider_id(
        settings.prominence_provider_id or state.spec.prominence_provider_id
    )
    active_search_stage = settings.active_search_stage
    prominence_fallback = settings.prominence_fallback
    if (
        _is_learned_prominence_provider(prominence_provider_id)
        and not result.ux_sample_valid
    ):
        prominence_fallback = True
    comparison_valid = not (
        _is_learned_prominence_provider(prominence_provider_id) and prominence_fallback
    )
    observations = tuple(
        event.observation
        for event in state.events
        if isinstance(event, ObservationRecorded)
    )
    target_elements = _target_elements(result, selected_target)
    target_ids = {element.id for element in target_elements}
    target_rank = _target_rank(observations, target_ids)
    inspected_regions = len(
        {
            observation.region_context.id
            for observation in observations
            if observation.region_context is not None
        }
    )
    executed_actions = tuple(
        event for event in state.events if isinstance(event, ActionExecuted)
    )
    scrolls = sum(isinstance(event.action, Scroll) for event in executed_actions)
    backtracks = sum(isinstance(event.action, Back) for event in executed_actions)
    wrong_actions = sum(
        isinstance(event.action, InteractWithElement)
        and (not event.succeeded or event.action.element_id not in target_ids)
        for event in executed_actions
    )
    outcome = _outcome_kind(result.outcome)
    abandoned = isinstance(result.outcome, AgentAbandoned)
    uncertainty = (
        settings.uncertainty
        if settings.uncertainty is not None
        else max(0.0, 1.0 - state.attention.confidence)
    )
    cost = DiscoveryCostBreakdown(
        inspection_cost=len(state.attention.inspected_ids) * config.inspection_cost,
        region_cost=inspected_regions * config.region_cost,
        scroll_cost=scrolls * config.scroll_cost,
        wrong_action_cost=wrong_actions * config.wrong_action_cost,
        backtrack_cost=backtracks * config.backtrack_cost,
        uncertainty_cost=uncertainty * config.uncertainty_cost,
        abandonment_penalty=config.abandonment_penalty if abandoned else 0.0,
    )
    target_prominence = settings.target_prominence
    if target_prominence is None:
        target_prominence = _active_target_prominence(settings, target_ids)
    target_scent = next(
        (
            settings.scent_scores[element_id]
            for element_id in reversed(tuple(settings.scent_scores))
            if element_id in target_ids
        ),
        None,
    )
    competitor_scores = [
        score
        for element_id, score in settings.scent_scores.items()
        if element_id not in target_ids
    ]
    strongest_competing_scent = max(competitor_scores, default=None)
    verified = result.verification.verified
    claimed = result.agent_claimed_success
    model_dependent = (
        settings.model_dependent
        or _has_model_manifest(state.provider_manifests)
        or _is_learned_prominence_provider(prominence_provider_id)
    )
    reproducibility = (
        Reproducibility.MODEL_DEPENDENT if model_dependent else Reproducibility.SEEDED
    )
    event_ids = _event_ids(result)
    prominence_event_ids = _unique_event_ids(
        (
            *settings.prominence_profile_event_ids,
            *settings.prominence_operational_event_ids,
            *_prominence_event_ids(result.evidence.prominence, target_ids),
        )
    )
    scent_event_ids = _scent_event_ids(result.evidence.scent, target_ids)
    evidence_records: list[Evidence] = []
    metric_records: list[Metric] = []

    def record(
        name: EvaluationMetric | str,
        value: float,
        evidence_class: EvidenceClass,
        description: str,
        source_ids: tuple[str, ...] = (),
    ) -> None:
        metric_name = str(name)
        evidence_id = f"{result.run_id}:{metric_name}"
        evidence_records.append(
            Evidence(
                evidence_id=evidence_id,
                evidence_class=evidence_class,
                description=description,
                source_event_ids=source_ids,
            )
        )
        metric_records.append(
            Metric(
                name=metric_name,
                value=value,
                evidence_class=evidence_class,
                evidence_ids=(evidence_id,),
            )
        )

    deterministic = EvidenceClass.DETERMINISTIC_FACT
    estimated = EvidenceClass.MODEL_ESTIMATE
    if target_rank is not None:
        record(
            EvaluationMetric.TARGET_DISCOVERY_RANK,
            float(target_rank),
            estimated,
            "Policy-dependent target rank in recorded observation order.",
            event_ids,
        )
    record(
        EvaluationMetric.INSPECTED_ELEMENTS,
        float(len(state.attention.inspected_ids)),
        deterministic,
        "Unique element IDs recorded in attention inspected state.",
        event_ids,
    )
    record(
        EvaluationMetric.INSPECTED_REGIONS,
        float(inspected_regions),
        deterministic,
        "Unique region IDs present in progressive observations.",
        event_ids,
    )
    record(
        EvaluationMetric.SCROLLS,
        float(scrolls),
        deterministic,
        "Executed scroll actions.",
        event_ids,
    )
    record(
        EvaluationMetric.WRONG_ACTIONS,
        float(wrong_actions),
        deterministic,
        "Failed or non-target interactions recorded by the run.",
        event_ids,
    )
    record(
        EvaluationMetric.BACKTRACKS,
        float(backtracks),
        deterministic,
        "Executed back actions.",
        event_ids,
    )
    record(
        EvaluationMetric.VERIFIED_COMPLETION,
        float(verified),
        deterministic,
        "Independent verifier result.",
    )
    record(
        EvaluationMetric.CLAIMED_COMPLETION,
        float(claimed),
        estimated,
        "Agent success claim.",
    )
    record(
        EvaluationMetric.FALSE_SUCCESS,
        float(claimed and not verified),
        deterministic,
        "Agent claim disagreed with verifier.",
    )
    if target_prominence is not None:
        prominence_description = _prominence_evidence_description(
            settings=settings,
            provider_id=prominence_provider_id,
            active_search_stage=active_search_stage,
            model_dependent=model_dependent,
            seed=state.spec.seed,
            model_trial=state.spec.model_trial,
        )
        record(
            EvaluationMetric.TARGET_PROMINENCE,
            target_prominence,
            estimated,
            prominence_description,
            prominence_event_ids,
        )
    if target_scent is not None:
        record(
            EvaluationMetric.TARGET_SCENT,
            target_scent,
            estimated,
            "Configured scent evidence for target.",
            scent_event_ids,
        )
    if strongest_competing_scent is not None:
        record(
            EvaluationMetric.STRONGEST_COMPETING_SCENT,
            strongest_competing_scent,
            estimated,
            "Strongest configured non-target scent.",
            _scent_event_ids(
                result.evidence.scent,
                frozenset(settings.scent_scores) - target_ids,
            ),
        )
    if settings.target_below_fold is not None:
        record(
            EvaluationMetric.TARGET_BELOW_FOLD,
            float(settings.target_below_fold),
            estimated,
            "Policy-dependent inference that target required scrolling before reveal.",
        )
    if settings.unexpected_hierarchy is not None:
        record(
            EvaluationMetric.UNEXPECTED_HIERARCHY,
            float(settings.unexpected_hierarchy),
            estimated,
            "Configured navigation hierarchy evidence.",
        )
    if settings.ambiguous_target is not None:
        record(
            EvaluationMetric.AMBIGUOUS_TARGET,
            float(settings.ambiguous_target),
            estimated,
            "Heuristic goal-label ambiguity estimate.",
        )
    record(
        EvaluationMetric.NAVIGATION_DEPTH,
        float(settings.navigation_depth),
        deterministic,
        "Configured navigation depth.",
    )
    if settings.feedback_observed is not None:
        record(
            EvaluationMetric.FEEDBACK_OBSERVED,
            float(settings.feedback_observed),
            estimated,
            "Heuristic goal-matching feedback estimate.",
        )
    record(
        EvaluationMetric.RECOVERY_ACTIONS,
        float(settings.recovery_actions or 0),
        deterministic,
        "Recorded recovery actions.",
    )
    if settings.recovery_success is not None:
        record(
            EvaluationMetric.RECOVERY_SUCCESS,
            float(settings.recovery_success),
            estimated,
            "Policy-dependent recovery outcome estimate.",
        )
    for component_name, component_value in cost.components.items():
        record(
            component_name,
            component_value,
            estimated,
            f"Discovery cost component from {config.version}.",
        )
    record(
        EvaluationMetric.DISCOVERY_COST,
        cost.total,
        estimated,
        f"Sum of discovery cost components from {config.version}.",
    )

    return RunMetrics(
        run_id=result.run_id,
        seed=state.spec.seed,
        scenario_id=state.spec.scenario.id,
        application_version_id=state.spec.application_version.id,
        persona_id=state.spec.persona.id,
        policy=state.spec.policy.value,
        target=selected_target,
        target_discovery_rank=target_rank,
        inspected_elements=len(state.attention.inspected_ids),
        inspected_regions=inspected_regions,
        scrolls=scrolls,
        wrong_actions=wrong_actions,
        backtracks=backtracks,
        verified_completion=verified,
        claimed_completion=claimed,
        false_success=claimed and not verified,
        abandoned=abandoned,
        model_trial=state.spec.model_trial,
        prominence_provider_id=prominence_provider_id,
        active_search_stage=active_search_stage,
        target_prominence=target_prominence,
        target_scent=target_scent,
        strongest_competing_scent=strongest_competing_scent,
        target_below_fold=settings.target_below_fold,
        unexpected_hierarchy=settings.unexpected_hierarchy,
        ambiguous_target=settings.ambiguous_target,
        navigation_depth=settings.navigation_depth,
        feedback_observed=settings.feedback_observed,
        recovery_actions=settings.recovery_actions or 0,
        recovery_success=settings.recovery_success,
        outcome=outcome,
        reproducibility=reproducibility,
        evidence=tuple(evidence_records),
        metrics=tuple(metric_records),
        discovery_cost=cost,
        config_digest=state.spec.config_digest,
        viewport_ids=tuple(snapshot.id for snapshot in state.snapshots),
        element_ids=tuple(element.id for element in target_elements),
        action_sequence=_action_sequence(executed_actions),
        target_prominence_profiles=settings.target_prominence_profiles,
        target_prominence_sources=settings.target_prominence_sources,
        prominence_profile_event_ids=settings.prominence_profile_event_ids,
        prominence_operational_event_ids=settings.prominence_operational_event_ids,
        prominence_lineage_event_ids=settings.prominence_lineage_event_ids,
        prominence_model_id=settings.prominence_model_id,
        prominence_model_version=settings.prominence_model_version,
        prominence_provider_version=settings.prominence_provider_version,
        prominence_fallback=prominence_fallback,
        prominence_fallback_reason=settings.prominence_fallback_reason,
        comparison_valid=comparison_valid,
    )


def calculate_run_metrics(*args: object, **kwargs: object) -> RunMetrics:
    """Compatibility name for callers using calculation vocabulary."""

    return evaluate_run(*args, **kwargs)  # type: ignore[arg-type]


def evaluation_target_for(result: RunResult) -> EvaluationTarget:
    """Resolve target only from immutable scenario contract and recorded snapshots."""

    spec = result.state.spec
    contract = spec.scenario.evaluation_target
    expected_label = contract.label_for(spec.application_version)
    expected_role = contract.role_for(spec.application_version)
    for snapshot in reversed(result.state.snapshots):
        regions = {region.id: region.label for region in snapshot.regions}
        for element in reversed(snapshot.elements):
            if _matches_target(
                element,
                regions,
                contract,
                expected_label,
                expected_role,
            ):
                return EvaluationTarget(
                    element_id=element.id,
                    region_id=element.region_id,
                    expected_label=expected_label,
                    role=expected_role,
                )
    role = f" role {expected_role!r}" if expected_role is not None else ""
    region = (
        f" region {contract.region_label!r}"
        if contract.region_label is not None
        else ""
    )
    raise EvaluationEvidenceUnavailable(
        f"target label {expected_label!r}{role}{region} not found in recorded snapshots"
    )


def evaluation_inputs_for(result: RunResult) -> RunEvaluationInputs:
    """Build evaluator inputs from evidence recorded by production RunAgent."""

    target = evaluation_target_for(result)
    target_elements = _target_elements(result, target)
    target_ids = {element.id for element in target_elements}
    visible_element_ids_by_viewport = {
        snapshot.id: {element.id for element in snapshot.elements}
        for snapshot in result.state.snapshots
    }
    target_viewport_ids = {
        viewport_id
        for viewport_id, element_ids in visible_element_ids_by_viewport.items()
        if element_ids & target_ids
    }
    prominence_scores: dict[str, float] = {}
    prominence_scores_by_stage: dict[str, dict[str, float]] = {}
    target_prominence_profiles: dict[str, float] = {}
    target_prominence_sources: dict[str, str] = {}
    active_search_stage: str | None = None
    prominence_profile_event_ids: list[str] = []
    prominence_operational_event_ids: list[str] = []
    prominence_lineage_event_ids: list[str] = []
    prominence_fallback = False
    for record in result.evidence.prominence:
        operational_event_id = record.operational_event_id or record.source_event_id
        visible_ids = visible_element_ids_by_viewport.get(record.viewport_id, set())
        scoped_scores = tuple(
            score for score in record.scores if score.element_id in visible_ids
        )
        target_score_bearing = any(
            score.element_id in target_ids for score in scoped_scores
        )
        target_profile_bearing = (
            any(profile.element_id in target_ids for profile in record.profiles)
            and record.viewport_id in target_viewport_ids
        )
        if (
            target_score_bearing or target_profile_bearing
        ) and record.profile_event_id is not None:
            prominence_profile_event_ids.append(record.profile_event_id)
            prominence_lineage_event_ids.append(record.profile_event_id)
        if target_score_bearing and operational_event_id is not None:
            prominence_operational_event_ids.append(operational_event_id)
            prominence_lineage_event_ids.append(operational_event_id)
        if record.stage is not None:
            active_search_stage = record.stage
        for score in scoped_scores:
            prominence_scores[score.element_id] = score.normalized_probability
            score_stage = score.stage or record.stage
            if score_stage is not None:
                stage_scores = prominence_scores_by_stage.setdefault(score_stage, {})
                stage_scores[score.element_id] = score.normalized_probability
                if record.stage is None:
                    active_search_stage = score_stage
            if score.element_id in target_ids:
                if score.evidence_kind == "fallback":
                    prominence_fallback = True
                for name, value in _profile_values(score).items():
                    target_prominence_profiles.setdefault(name, value)
                    target_prominence_sources.setdefault(
                        name,
                        _safe_provenance_identifier(
                            score.evidence_source or score.provider_id
                        ),
                    )
        for profile in record.profiles:
            if (
                profile.element_id not in target_ids
                or record.viewport_id not in target_viewport_ids
            ):
                continue
            for name, estimate in (
                ("immediate", profile.immediate),
                ("early", profile.early),
                ("eventual", profile.eventual),
            ):
                if estimate is None or estimate.score is None:
                    continue
                target_prominence_profiles.setdefault(name, estimate.score)
                target_prominence_sources.setdefault(
                    name,
                    _safe_provenance_identifier(estimate.source or "unavailable"),
                )
    provider_id = result.state.spec.prominence_provider_id
    if _is_learned_prominence_provider(provider_id):
        prominence_fallback = prominence_fallback or any(
            score.provider_id not in {"foveacast", "foveacast-prominence"}
            or score.evidence_kind == "fallback"
            for record in result.evidence.prominence
            for score in record.scores
        )
        invalid_reason = result.ux_sample_invalid_reason or ""
        prominence_fallback = prominence_fallback or invalid_reason.startswith(
            "saliency-fallback:"
        )
    target_prominence = _active_target_score(
        prominence_scores,
        prominence_scores_by_stage,
        active_search_stage,
        target_ids,
    )
    model_id, model_version, provider_version = _prominence_model_details(
        result.state.provider_manifests, provider_id
    )
    if model_id is None:
        model_id = next(
            (
                _safe_provenance_identifier(score.evidence_source)
                for record in reversed(result.evidence.prominence)
                for score in reversed(record.scores)
                if score.evidence_source
            ),
            None,
        )
    scent_scores: dict[str, float] = {}
    for record in result.evidence.scent:
        for score in record.scores:
            element_id = getattr(score, "element_id", None)
            value = getattr(score, "score", None)
            if isinstance(element_id, str) and isinstance(value, int | float):
                scent_scores[element_id] = float(value)
    navigation_depth = sum(
        isinstance(event, ActionExecuted)
        and event.succeeded
        and event.navigation_occurred
        for event in result.state.events
    )
    recovery_actions = sum(
        isinstance(event, ActionExecuted) and isinstance(event.action, Back)
        for event in result.state.events
    )
    target_below_fold = _target_below_fold(result, target_elements)
    path_labels = (
        _pretarget_interaction_labels(result, target_elements)
        if target_elements
        else ()
    )
    ambiguous_target = (
        not _label_matches_goal(
            target_elements[-1].label, result.state.spec.scenario.goal
        )
        or any(
            not _label_matches_goal(label, result.state.spec.scenario.goal)
            for label in path_labels
        )
        if target_elements
        else None
    )
    unexpected_hierarchy = (
        bool(
            len(path_labels) >= 2
            or (
                path_labels
                and target_elements[-1].id
                not in {element.id for element in result.state.snapshots[0].elements}
                and ambiguous_target
            )
        )
        if target_elements
        else None
    )
    return RunEvaluationInputs(
        prominence_scores=prominence_scores,
        prominence_scores_by_stage=prominence_scores_by_stage,
        scent_scores=scent_scores,
        target_prominence=target_prominence,
        target_below_fold=target_below_fold,
        unexpected_hierarchy=unexpected_hierarchy,
        ambiguous_target=ambiguous_target,
        navigation_depth=navigation_depth,
        feedback_observed=_feedback_observed(result, target_elements),
        recovery_actions=recovery_actions,
        recovery_success=(result.verification.verified if recovery_actions else None),
        uncertainty=max(0.0, 1.0 - result.state.attention.confidence),
        model_dependent=bool(
            result.evidence.model_calls
            or result.evidence.scent
            or _is_learned_prominence_provider(provider_id)
        ),
        prominence_provider_id=provider_id,
        active_search_stage=active_search_stage,
        target_prominence_profiles=target_prominence_profiles,
        target_prominence_sources=target_prominence_sources,
        prominence_operational_event_ids=_unique_event_ids(
            prominence_operational_event_ids
        ),
        prominence_profile_event_ids=_unique_event_ids(prominence_profile_event_ids),
        prominence_lineage_event_ids=_unique_event_ids(prominence_lineage_event_ids),
        prominence_model_id=model_id,
        prominence_model_version=model_version,
        prominence_provider_version=provider_version,
        prominence_fallback=prominence_fallback,
        prominence_fallback_reason=(
            result.ux_sample_invalid_reason.removeprefix("saliency-fallback: ").strip()
            if prominence_fallback and result.ux_sample_invalid_reason
            else None
        ),
    )


def _target_elements(
    result: RunResult, target: EvaluationTarget
) -> tuple[ElementSnapshot, ...]:
    identity_candidates = [
        element
        for snapshot in result.state.snapshots
        for element in snapshot.elements
        if target.element_id is None or element.id == target.element_id
    ]
    selected = [
        element
        for element in identity_candidates
        if target.expected_label is None
        or element.label.strip().casefold() == target.expected_label.strip().casefold()
        if target.role is None
        or str(getattr(element.role, "value", element.role)) == target.role
        if target.region_id is None or element.region_id == target.region_id
    ]
    if not selected:
        reference = target.element_id or target.expected_label or "target"
        constraints = tuple(
            value
            for value in (
                f"role {target.role!r}" if target.role is not None else "",
                f"region {target.region_id!r}" if target.region_id is not None else "",
            )
            if value
        )
        suffix = f" with {' and '.join(constraints)}" if constraints else ""
        raise EvaluationEvidenceUnavailable(
            f"target reference {reference!r}{suffix} not found in recorded snapshots"
        )
    if target.element_id is not None and identity_candidates[-1] not in selected:
        raise EvaluationEvidenceUnavailable(
            f"latest target reference {target.element_id!r} does not match declared constraints"
        )
    lineage_id = selected[-1].lineage_id
    if lineage_id is None:
        return tuple(selected)
    return tuple(
        element
        for snapshot in result.state.snapshots
        for element in snapshot.elements
        if element.lineage_id == lineage_id
    )


def _target_below_fold(
    result: RunResult, target_elements: Sequence[ElementSnapshot]
) -> bool | None:
    if not target_elements:
        return None
    target_ids = {element.id for element in target_elements}
    scrolled = False
    for event in result.state.events:
        if isinstance(event, ActionExecuted) and isinstance(event.action, Scroll):
            scrolled = True
        if isinstance(event, ObservationRecorded) and any(
            element.id in target_ids
            for element in event.observation.newly_revealed_elements
        ):
            return scrolled
    return False


def _pretarget_interaction_labels(
    result: RunResult, target_elements: Sequence[ElementSnapshot]
) -> tuple[str, ...]:
    target_ids = {element.id for element in target_elements}
    snapshots = {snapshot.id: snapshot for snapshot in result.state.snapshots}
    labels: list[str] = []
    for event in result.state.events:
        if not isinstance(event, ActionExecuted) or not isinstance(
            event.action, InteractWithElement
        ):
            continue
        if event.action.element_id in target_ids:
            break
        if not event.succeeded:
            continue
        if event.platform_action_kind in {"clear-text", "type-text"}:
            continue
        labels.append(
            snapshots[event.viewport_id].element(event.action.element_id).label
        )
    return tuple(labels)


def _feedback_observed(
    result: RunResult, target_elements: Sequence[ElementSnapshot]
) -> bool | None:
    if not target_elements:
        return None
    target_ids = {element.id for element in target_elements}
    goal = result.state.spec.scenario.goal
    events = result.state.events
    for index, event in enumerate(events):
        if not (
            isinstance(event, ActionExecuted)
            and isinstance(event.action, InteractWithElement)
            and event.action.element_id in target_ids
            and event.succeeded
            and event.state_changed
        ):
            continue
        for later in events[index + 1 :]:
            if isinstance(later, ActionExecuted) and later.state_changed:
                break
            if not isinstance(later, ViewportCaptured):
                continue
            if later.snapshot.id == event.viewport_id:
                continue
            return any(
                _is_success_feedback(element.label, goal)
                for element in later.snapshot.elements
            )
        return None
    return None


def _action_sequence(actions: Sequence[ActionExecuted]) -> tuple[str, ...]:
    sequence: list[str] = []
    for event in actions:
        action_kind = str(getattr(event.action, "kind", "action"))
        element_id = getattr(event.action, "element_id", None)
        target = f" {element_id}" if isinstance(element_id, str) else ""
        result = "succeeded" if event.succeeded else "failed"
        sequence.append(f"{action_kind}{target}: {result}")
    return tuple(sequence)


def _is_success_feedback(label: str, goal: str) -> bool:
    lowered = label.lower()
    return any(
        marker in lowered for marker in ("sent", "enabled", "completed", "success")
    ) and _label_matches_goal(label, goal)


def _label_matches_goal(label: str, goal: str) -> bool:
    ignored = {"a", "an", "authentication", "enable", "send", "the", "to"}
    label_tokens = {_token_root(token) for token in _words(label)} - ignored
    goal_tokens = {_token_root(token) for token in _words(goal)} - ignored
    return bool(label_tokens & goal_tokens)


def _words(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in "".join(
            character.lower() if character.isalnum() else " " for character in value
        ).split()
        if token
    )


def _token_root(value: str) -> str:
    if value.startswith("invit"):
        return "invite"
    return value


def evaluate_experiment_results(results: Sequence[RunResult]) -> ExperimentEvaluation:
    """Aggregate evaluated production runs and apply exact paired-seed gates."""

    run_results = tuple(results)
    all_metrics = tuple(
        result.metrics for result in run_results if result.metrics is not None
    )
    if len(all_metrics) != len(run_results):
        raise ValueError("every completed run needs persisted metrics")
    metrics = tuple(
        metric
        for result, metric in zip(run_results, all_metrics, strict=True)
        if _comparison_sample_is_valid(result, metric)
    )
    cells = aggregate_cells(metrics)
    grouped: dict[
        tuple[str, str, str, str | None, int, str],
        dict[ApplicationVersionKind, list[RunMetrics]],
    ] = {}
    by_run_id = {
        result.run_id: result
        for result in run_results
        if _comparison_sample_is_valid(result, result.metrics)
    }
    for metric in metrics:
        result = by_run_id[metric.run_id]
        key = (
            metric.scenario_id,
            metric.persona_id,
            metric.policy,
            metric.config_digest,
            metric.model_trial,
            metric.prominence_provider_id,
        )
        grouped.setdefault(key, {}).setdefault(
            result.state.spec.application_version.kind, []
        ).append(metric)
    comparisons: list[VariantComparison] = []
    for key in sorted(grouped):
        variants = grouped[key]
        baseline = variants.get(ApplicationVersionKind.DEFECTIVE)
        improved = variants.get(ApplicationVersionKind.IMPROVED)
        if (
            baseline
            and improved
            and {run.seed for run in baseline} == {run.seed for run in improved}
        ):
            comparisons.append(compare_variants(baseline, improved))
    return ExperimentEvaluation(metrics, cells, tuple(comparisons))


def _comparison_sample_is_valid(result: RunResult, metric: RunMetrics | None) -> bool:
    return comparison_sample_is_valid(result, metric)


def comparison_sample_is_valid(result: RunResult, metric: RunMetrics | None) -> bool:
    """Central validity gate for scorecard and comparison aggregation."""

    if metric is None or metric.comparison_valid is not True:
        return False
    if metric.run_id != result.run_id:
        return False
    if metric.prominence_provider_id != _canonical_prominence_provider_id(
        result.state.spec.prominence_provider_id
    ):
        return False
    if result.ux_sample_valid is not True:
        return False
    if metric.prominence_fallback or metric.prominence_fallback_reason:
        return False
    invalid_reason = result.ux_sample_invalid_reason or ""
    return not invalid_reason.startswith("saliency-fallback:")


def persisted_comparison_sample_is_valid(
    result: Mapping[str, object],
    metrics: Mapping[str, object],
    *,
    manifest: Mapping[str, object] | None,
    expected_run_id: str,
    expected_prominence_provider_id: str,
    timeline_events: Sequence[Mapping[str, object]] = (),
) -> bool:
    """Cross-check persisted validity fields before resume or scorecard use."""

    expected_provider = _canonical_prominence_provider_id(
        expected_prominence_provider_id
    )
    if (
        result.get("run_id") != expected_run_id
        or metrics.get("run_id") != expected_run_id
    ):
        return False
    if manifest is not None and manifest.get("run_id") != expected_run_id:
        return False
    provider_values = (
        metrics.get("prominence_provider_id"),
        manifest.get("prominence_provider_id") if manifest is not None else None,
    )
    if any(
        _canonical_prominence_provider_id(value) != expected_provider
        for value in provider_values
    ):
        return False
    state = result.get("state")
    state_mapping: Mapping[str, object] = (
        cast(Mapping[str, object], state) if isinstance(state, Mapping) else {}
    )
    spec = state_mapping.get("spec")
    spec_mapping: Mapping[str, object] = (
        cast(Mapping[str, object], spec) if isinstance(spec, Mapping) else {}
    )
    if "prominence_provider_id" in spec_mapping and (
        _canonical_prominence_provider_id(spec_mapping.get("prominence_provider_id"))
        != expected_provider
    ):
        return False
    if (
        type(result.get("ux_sample_valid")) is not bool
        or result.get("ux_sample_valid") is not True
    ):
        return False
    if (
        type(metrics.get("comparison_valid")) is not bool
        or metrics.get("comparison_valid") is not True
    ):
        return False
    if (
        type(metrics.get("prominence_fallback")) is not bool
        or metrics.get("prominence_fallback") is not False
    ):
        return False
    if metrics.get("prominence_fallback_reason") not in (None, ""):
        return False
    if _timeline_contains_prominence_fallback(timeline_events):
        return False
    invalid_reason = result.get("ux_sample_invalid_reason")
    if invalid_reason not in (None, ""):
        return False
    return True


def _timeline_contains_prominence_fallback(
    events: Sequence[Mapping[str, object]],
) -> bool:
    return any(
        event.get("kind") == "saliency-fallback-recorded"
        or (
            event.get("kind") == "prominence-recorded"
            and event.get("cache_state") == "fallback"
        )
        for event in events
    )


def aggregate_cell(runs: Sequence[RunMetrics]) -> CellAggregate:
    """Aggregate repeated runs sharing one scenario/version/persona/policy cell."""

    ordered = tuple(runs)
    if not ordered:
        raise ValueError("cannot aggregate empty cell")
    first = ordered[0]
    identity = (
        first.scenario_id,
        first.application_version_id,
        first.persona_id,
        first.policy,
        first.model_trial,
        first.prominence_provider_id,
    )
    if any(
        (
            item.scenario_id,
            item.application_version_id,
            item.persona_id,
            item.policy,
            item.model_trial,
            item.prominence_provider_id,
        )
        != identity
        for item in ordered
    ):
        raise ValueError(
            "cell runs must share scenario, version, persona, policy, model trial, "
            "and prominence provider"
        )
    if any(not run.comparison_valid for run in ordered):
        raise ValueError(
            "invalid prominence comparison sample cannot enter evaluation scorecard"
        )
    _reject_unsupported_metrics(ordered)
    metric_names = sorted(
        {
            metric.name
            for run in ordered
            for metric in (*run.metrics, *_standard_metrics(run))
        }
    )
    summaries: dict[str, MetricSummary] = {}
    for name in metric_names:
        available = [
            run.metric(name)
            for run in ordered
            if _has_metric(run, name) or name in _standard_metric_values(run)
        ]
        values = [metric.value for metric in available]
        classes = {EvidenceClass(metric.evidence_class) for metric in available}
        evidence_class = (
            EvidenceClass.MODEL_ESTIMATE
            if EvidenceClass.MODEL_ESTIMATE in classes
            else EvidenceClass.DETERMINISTIC_FACT
        )
        summaries[name] = MetricSummary(
            name=name,
            interval=interval_summary(values),
            evidence_class=evidence_class,
            evidence_ids=tuple(
                evidence_id
                for metric in available
                for evidence_id in metric.evidence_ids
            ),
        )
    reproducibilities = [Reproducibility(run.reproducibility) for run in ordered]
    reproducibility = _aggregate_reproducibility(reproducibilities)
    reproducibility_summary = {
        value.value: sum(item is value for item in reproducibilities)
        for value in Reproducibility
        if any(item is value for item in reproducibilities)
    }
    return CellAggregate(
        scenario_id=first.scenario_id,
        application_version_id=first.application_version_id,
        persona_id=first.persona_id,
        policy=first.policy,
        run_ids=tuple(run.run_id for run in ordered),
        seeds=tuple(run.seed for run in ordered),
        metric_summaries=MappingProxyType(summaries),
        reproducibility=reproducibility,
        reproducibility_summary=MappingProxyType(reproducibility_summary),
        evidence_ids=tuple(
            evidence.evidence_id for run in ordered for evidence in run.evidence
        ),
        model_trial=first.model_trial,
        prominence_provider_id=first.prominence_provider_id,
        active_search_stages=tuple(
            sorted(
                {
                    run.active_search_stage
                    for run in ordered
                    if run.active_search_stage is not None
                }
            )
        ),
    )


def aggregate_cells(runs: Sequence[RunMetrics]) -> tuple[CellAggregate, ...]:
    """Aggregate all cells in stable identity order."""

    groups: dict[tuple[str, str, str, str, int, str], list[RunMetrics]] = {}
    for run in runs:
        key = (
            run.scenario_id,
            run.application_version_id,
            run.persona_id,
            run.policy,
            run.model_trial,
            run.prominence_provider_id,
        )
        groups.setdefault(key, []).append(run)
    return tuple(aggregate_cell(groups[key]) for key in sorted(groups))


def build_scorecard(runs: Sequence[RunMetrics]) -> CellAggregate:
    """Build supported evidence scorecard; human claims are rejected."""

    return aggregate_cell(runs)


def interval_summary(
    values: Sequence[float], confidence: float = 0.95
) -> IntervalSummary:
    """Return empirical median and central interval using linear quantiles."""

    if not values:
        raise ValueError("interval requires at least one value")
    if not 0 < confidence < 1:
        raise ValueError("interval confidence must be between zero and one")
    ordered = sorted(float(value) for value in values)
    return IntervalSummary(
        count=len(ordered),
        median=_quantile(ordered, 0.5),
        lower=_quantile(ordered, (1 - confidence) / 2),
        upper=_quantile(ordered, 1 - ((1 - confidence) / 2)),
        confidence=confidence,
    )


def compare_variants(
    baseline_runs: Sequence[RunMetrics],
    improved_runs: Sequence[RunMetrics],
) -> VariantComparison:
    """Compare variants with exact paired seeds and directional acceptance rules."""

    baseline = tuple(baseline_runs)
    improved = tuple(improved_runs)
    if not baseline or not improved:
        raise ValueError("variant comparison needs both baseline and improved runs")
    baseline_by_assignment = _by_seed_and_model_trial(baseline)
    improved_by_assignment = _by_seed_and_model_trial(improved)
    if set(baseline_by_assignment) != set(improved_by_assignment):
        raise ValueError(
            "variant comparison requires identical paired attention seeds and "
            "model trials"
        )
    _validate_variant_identity(baseline, improved)
    paired_assignments = tuple(sorted(baseline_by_assignment))
    paired_seeds = tuple(seed for seed, _ in paired_assignments)
    paired_model_trials = tuple(trial for _, trial in paired_assignments)
    baseline_cell = aggregate_cell(baseline)
    improved_cell = aggregate_cell(improved)
    cost_deltas = tuple(
        improved_by_assignment[assignment].discovery_cost.total
        - baseline_by_assignment[assignment].discovery_cost.total
        for assignment in paired_assignments
    )
    wrong_deltas = tuple(
        improved_by_assignment[assignment].wrong_actions
        - baseline_by_assignment[assignment].wrong_actions
        for assignment in paired_assignments
    )
    backtrack_deltas = tuple(
        improved_by_assignment[assignment].backtracks
        - baseline_by_assignment[assignment].backtracks
        for assignment in paired_assignments
    )
    baseline_completion = sum(run.verified_completion for run in baseline) / len(
        baseline
    )
    improved_completion = sum(run.verified_completion for run in improved) / len(
        improved
    )
    cost_decreased = _median(cost_deltas) < 0
    wrong_not_increased = _median(wrong_deltas) <= 0
    backtrack_not_increased = _median(backtrack_deltas) <= 0
    completion_not_regressed = improved_completion >= baseline_completion
    gated_assignments = tuple(
        assignment
        for assignment in paired_assignments
        if not (
            not baseline_by_assignment[assignment].verified_completion
            and improved_by_assignment[assignment].verified_completion
        )
    )
    gated_cost_decreased = (
        not gated_assignments
        or _median(
            tuple(
                improved_by_assignment[assignment].discovery_cost.total
                - baseline_by_assignment[assignment].discovery_cost.total
                for assignment in gated_assignments
            )
        )
        < 0
    )
    gated_wrong_not_increased = (
        not gated_assignments
        or _median(
            tuple(
                improved_by_assignment[assignment].wrong_actions
                - baseline_by_assignment[assignment].wrong_actions
                for assignment in gated_assignments
            )
        )
        <= 0
    )
    gated_backtrack_not_increased = (
        not gated_assignments
        or _median(
            tuple(
                improved_by_assignment[assignment].backtracks
                - baseline_by_assignment[assignment].backtracks
                for assignment in gated_assignments
            )
        )
        <= 0
    )
    reasons = tuple(
        reason
        for reason, passed in (
            ("paired median discovery cost did not decrease", gated_cost_decreased),
            (
                "paired median wrong-action burden increased",
                gated_wrong_not_increased,
            ),
            (
                "paired median backtrack burden increased",
                gated_backtrack_not_increased,
            ),
            ("verified completion rate regressed", completion_not_regressed),
        )
        if not passed
    )
    gate = DirectionalGateResult(
        passed=not reasons,
        paired_seed_count=len(paired_seeds),
        discovery_cost_decreased=cost_decreased,
        wrong_action_burden_not_increased=wrong_not_increased,
        backtrack_burden_not_increased=backtrack_not_increased,
        verified_completion_rate_not_regressed=completion_not_regressed,
        baseline_discovery_cost_median=_median(
            tuple(run.discovery_cost.total for run in baseline)
        ),
        improved_discovery_cost_median=_median(
            tuple(run.discovery_cost.total for run in improved)
        ),
        baseline_wrong_action_median=_median(
            tuple(run.wrong_actions for run in baseline)
        ),
        improved_wrong_action_median=_median(
            tuple(run.wrong_actions for run in improved)
        ),
        baseline_backtrack_median=_median(tuple(run.backtracks for run in baseline)),
        improved_backtrack_median=_median(tuple(run.backtracks for run in improved)),
        baseline_completion_rate=baseline_completion,
        improved_completion_rate=improved_completion,
        reasons=reasons,
    )
    return VariantComparison(
        baseline=baseline_cell,
        improved=improved_cell,
        paired_seeds=paired_seeds,
        gate=gate,
        paired_model_trials=paired_model_trials,
    )


def _target_rank(
    observations: Sequence[PersonaObservation], target_ids: set[str]
) -> int | None:
    if not target_ids:
        return None
    rank = 0
    for observation in observations:
        for element in observation.newly_revealed_elements:
            rank += 1
            if element.id in target_ids:
                return rank
    return None


def _matches_target(
    element: ElementSnapshot,
    region_labels: Mapping[str, str],
    contract: ScenarioEvaluationTarget,
    expected_label: str,
    expected_role: str | None,
) -> bool:
    def _norm(value: str) -> str:
        return " ".join(value.strip().split()).casefold()

    # Substring match so "Cairo, Egypt" finds "محمد خليل Cairo, Egypt · Open to remote"
    # and whitespace differences (newlines, multiple spaces, NBSP) do not break it.
    # The Arabic prefix and middle-dot suffix are preserved in the element label.
    if _norm(expected_label) not in _norm(element.label):
        return False
    if (
        expected_role is not None
        and str(getattr(element.role, "value", element.role)) != expected_role
    ):
        return False
    if contract.region_label is None:
        return True
    if element.region_id is None:
        return False
    return (
        region_labels.get(element.region_id, "").strip().casefold()
        == contract.region_label.strip().casefold()
    )


def _outcome_kind(outcome: RunOutcome) -> str:
    return str(getattr(outcome, "kind", RunOutcomeKind.INTERNAL_ERROR.value))


def _has_model_manifest(manifests: Sequence[object]) -> bool:
    return any(
        getattr(manifest, "model_id", None)
        and str(getattr(manifest, "role", ""))
        in {"cognitive", "coarse-scent", "full-scent"}
        for manifest in manifests
    )


def _is_learned_prominence_provider(provider_id: str) -> bool:
    return _canonical_prominence_provider_id(provider_id) == "foveacast"


def _profile_values(score: ProminenceResult) -> dict[str, float]:
    raw_values = score.raw_values
    values: dict[str, float] = {}
    duration_names = {
        "stage_1s": "immediate",
        "stage_3s": "early",
        "stage_7s": "eventual",
    }
    for raw_name, profile_name in duration_names.items():
        raw_value = raw_values.get(raw_name)
        if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
            if math.isfinite(float(raw_value)) and 0 <= raw_value <= 1:
                values[profile_name] = float(raw_value)
    if values:
        return values
    stage = score.stage
    raw_score = score.raw_score
    profile_name = {
        "initial": "immediate",
        "exploration": "early",
        "persistent": "eventual",
    }.get(str(stage))
    if (
        profile_name is not None
        and math.isfinite(float(raw_score))
        and 0 <= raw_score <= 1
    ):
        return {profile_name: float(raw_score)}
    return {}


def _active_target_score(
    scores: Mapping[str, float],
    scores_by_stage: Mapping[str, Mapping[str, float]],
    active_stage: str | None,
    target_ids: Collection[str],
) -> float | None:
    if active_stage is not None:
        active_scores = scores_by_stage.get(active_stage, {})
        for element_id in reversed(tuple(active_scores)):
            if element_id in target_ids:
                return active_scores[element_id]
        return None
    return next(
        (
            scores[element_id]
            for element_id in reversed(tuple(scores))
            if element_id in target_ids
        ),
        None,
    )


def _active_target_prominence(
    settings: RunEvaluationInputs, target_ids: Collection[str]
) -> float | None:
    if settings.active_search_stage is not None:
        return next(
            (
                settings.prominence_scores_by_stage[settings.active_search_stage][
                    element_id
                ]
                for element_id in reversed(
                    tuple(
                        settings.prominence_scores_by_stage.get(
                            settings.active_search_stage, {}
                        )
                    )
                )
                if element_id in target_ids
            ),
            None,
        )
    return next(
        (
            settings.prominence_scores[element_id]
            for element_id in reversed(tuple(settings.prominence_scores))
            if element_id in target_ids
        ),
        None,
    )


def _prominence_model_details(
    manifests: Sequence[ProviderManifest], provider_id: str
) -> tuple[str | None, str | None, str | None]:
    canonical_provider = _canonical_prominence_provider_id(provider_id)
    for manifest in manifests:
        if manifest.role != "prominence":
            continue
        if (
            _canonical_prominence_provider_id(manifest.provider_id)
            != canonical_provider
        ):
            continue
        if canonical_provider == "heuristic":
            return None, None, _safe_provenance_identifier(manifest.version)
        return (
            _safe_provenance_identifier(manifest.model_id)
            if manifest.model_id
            else None,
            _safe_provenance_identifier(manifest.version) if manifest.version else None,
            None,
        )
    return None, None, None


def _canonical_prominence_provider_id(provider_id: object) -> str:
    return {
        "heuristic": "heuristic",
        "heuristic-prominence": "heuristic",
        "foveacast": "foveacast",
        "foveacast-prominence": "foveacast",
    }.get(provider_id, provider_id)


def format_prominence_provenance(
    *,
    provider_id: str,
    active_search_stage: str | None,
    target_prominence_profiles: Mapping[str, float],
    target_prominence_sources: Mapping[str, str],
    profile_event_ids: Sequence[str],
    operational_event_ids: Sequence[str],
    model_id: str | None,
    model_version: str | None,
    provider_version: str | None,
) -> str:
    """Format stable provider-aware prominence evidence provenance."""

    profile_sources = (
        ", ".join(
            f"{name}={target_prominence_sources.get(name, 'source unavailable')}"
            for name in ("immediate", "early", "eventual")
            if name in target_prominence_profiles
        )
        or "profile source unavailable"
    )
    duration_labels = {
        "immediate": "1s immediate",
        "early": "3s early",
        "eventual": "7s eventual",
    }
    durations = (
        ", ".join(
            duration_labels[name]
            for name in ("immediate", "early", "eventual")
            if name in target_prominence_profiles
        )
        or "unavailable"
    )
    return (
        f"provider {provider_id}, active search stage "
        f"{active_search_stage or 'unavailable'}, duration sources {durations} "
        f"({profile_sources}); profile event(s) "
        f"{', '.join(profile_event_ids) or 'unavailable'}; operational "
        f"prominence event(s) {', '.join(operational_event_ids) or 'unavailable'}; "
        f"provider model/version {model_id or 'not applicable'}/"
        f"{model_version or 'unavailable'}; provider version "
        f"{provider_version or 'unavailable'}"
    )


def _prominence_evidence_description(
    *,
    settings: RunEvaluationInputs,
    provider_id: str,
    active_search_stage: str | None,
    model_dependent: bool,
    seed: int,
    model_trial: int,
) -> str:
    provenance = format_prominence_provenance(
        provider_id=provider_id,
        active_search_stage=active_search_stage,
        target_prominence_profiles=settings.target_prominence_profiles,
        target_prominence_sources=settings.target_prominence_sources,
        profile_event_ids=settings.prominence_profile_event_ids,
        operational_event_ids=settings.prominence_operational_event_ids,
        model_id=settings.prominence_model_id,
        model_version=settings.prominence_model_version,
        provider_version=settings.prominence_provider_version,
    )
    limitation = "model-estimate evidence from simulated benchmark replay; not calibrated to human attention"
    reproducibility = (
        f"attention seed {seed} and model trial {model_trial}"
        if model_dependent
        else "deterministic configuration"
    )
    return (
        "Operational prominence model estimate for target: "
        f"{provenance}; "
        f"{reproducibility}; limitation: {limitation}."
    )


def _event_ids(result: RunResult) -> tuple[str, ...]:
    event_ids = tuple(result.evidence.state_event_ids)
    if len(event_ids) != len(result.state.events):
        return ()
    return event_ids


def _prominence_event_ids(
    records: Sequence[ProminenceEvidence], target_ids: Collection[str]
) -> tuple[str, ...]:
    return _unique_event_ids(
        record.source_event_id
        for record in records
        if any(score.element_id in target_ids for score in record.scores)
    )


def _scent_event_ids(
    records: Sequence[ScentEvidence], target_ids: Collection[str]
) -> tuple[str, ...]:
    return _unique_event_ids(
        record.source_event_id
        for record in records
        if any(
            getattr(score, "element_id", None) in target_ids for score in record.scores
        )
    )


def _unique_event_ids(values: Iterable[str | None]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if value is not None and value not in result:
            result.append(value)
    return tuple(result)


def _has_metric(run: RunMetrics, name: str) -> bool:
    return any(metric.name == name for metric in run.metrics)


def _standard_metrics(run: RunMetrics) -> tuple[Metric, ...]:
    return tuple(
        Metric(name=name, value=value, evidence_class=evidence_class)
        for name, (value, evidence_class) in _standard_metric_values(run).items()
    )


def _standard_metric_values(run: RunMetrics) -> dict[str, tuple[float, EvidenceClass]]:
    deterministic = EvidenceClass.DETERMINISTIC_FACT
    estimated = EvidenceClass.MODEL_ESTIMATE
    values: dict[str, tuple[float, EvidenceClass]] = {
        EvaluationMetric.INSPECTED_ELEMENTS.value: (
            float(run.inspected_elements),
            deterministic,
        ),
        EvaluationMetric.INSPECTED_REGIONS.value: (
            float(run.inspected_regions),
            deterministic,
        ),
        EvaluationMetric.SCROLLS.value: (float(run.scrolls), deterministic),
        EvaluationMetric.WRONG_ACTIONS.value: (float(run.wrong_actions), deterministic),
        EvaluationMetric.BACKTRACKS.value: (float(run.backtracks), deterministic),
        EvaluationMetric.VERIFIED_COMPLETION.value: (
            float(run.verified_completion),
            deterministic,
        ),
        EvaluationMetric.CLAIMED_COMPLETION.value: (
            float(run.claimed_completion),
            estimated,
        ),
        EvaluationMetric.FALSE_SUCCESS.value: (float(run.false_success), deterministic),
        EvaluationMetric.NAVIGATION_DEPTH.value: (
            float(run.navigation_depth),
            deterministic,
        ),
        EvaluationMetric.RECOVERY_ACTIONS.value: (
            float(run.recovery_actions),
            deterministic,
        ),
        EvaluationMetric.DISCOVERY_COST.value: (run.discovery_cost.total, estimated),
    }
    optional: tuple[tuple[EvaluationMetric, float | None, EvidenceClass], ...] = (
        (
            EvaluationMetric.TARGET_DISCOVERY_RANK,
            run.target_discovery_rank,
            estimated,
        ),
        (EvaluationMetric.TARGET_PROMINENCE, run.target_prominence, estimated),
        (EvaluationMetric.TARGET_SCENT, run.target_scent, estimated),
        (
            EvaluationMetric.STRONGEST_COMPETING_SCENT,
            run.strongest_competing_scent,
            estimated,
        ),
        (
            EvaluationMetric.TARGET_BELOW_FOLD,
            float(run.target_below_fold) if run.target_below_fold is not None else None,
            estimated,
        ),
        (
            EvaluationMetric.UNEXPECTED_HIERARCHY,
            float(run.unexpected_hierarchy)
            if run.unexpected_hierarchy is not None
            else None,
            estimated,
        ),
        (
            EvaluationMetric.AMBIGUOUS_TARGET,
            float(run.ambiguous_target) if run.ambiguous_target is not None else None,
            estimated,
        ),
        (
            EvaluationMetric.FEEDBACK_OBSERVED,
            float(run.feedback_observed) if run.feedback_observed is not None else None,
            estimated,
        ),
        (
            EvaluationMetric.RECOVERY_SUCCESS,
            float(run.recovery_success) if run.recovery_success is not None else None,
            estimated,
        ),
    )
    for name, value, evidence_class in optional:
        if value is not None:
            values[name.value] = (float(value), evidence_class)
    for name, value in run.discovery_cost.components.items():
        values[name] = (value, estimated)
    return values


def _reject_unsupported_metrics(runs: Sequence[RunMetrics]) -> None:
    if any(
        EvidenceClass(metric.evidence_class) is EvidenceClass.UNSUPPORTED_HUMAN_CLAIM
        for run in runs
        for metric in run.metrics
    ):
        raise UnsupportedHumanClaimError(
            "unsupported human claim cannot enter evaluation scorecard"
        )
    if any(
        EvidenceClass(evidence.evidence_class) is EvidenceClass.UNSUPPORTED_HUMAN_CLAIM
        for run in runs
        for evidence in run.evidence
    ):
        raise UnsupportedHumanClaimError(
            "unsupported human claim cannot enter evaluation scorecard"
        )


def _aggregate_reproducibility(values: Sequence[Reproducibility]) -> Reproducibility:
    if Reproducibility.NOT_REPRODUCIBLE in values:
        return Reproducibility.NOT_REPRODUCIBLE
    if Reproducibility.MODEL_DEPENDENT in values:
        return Reproducibility.MODEL_DEPENDENT
    if Reproducibility.SEEDED in values:
        return Reproducibility.SEEDED
    return Reproducibility.REPRODUCIBLE


def _quantile(values: Sequence[float], probability: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _median(values: Sequence[float]) -> float:
    return interval_summary(values).median


def _by_seed_and_model_trial(
    runs: Sequence[RunMetrics],
) -> dict[tuple[int, int], RunMetrics]:
    result: dict[tuple[int, int], RunMetrics] = {}
    for run in runs:
        assignment = (run.seed, run.model_trial)
        if assignment in result:
            raise ValueError(
                "duplicate attention seed and model trial in variant cell: "
                f"{run.seed}, {run.model_trial}"
            )
        result[assignment] = run
    return result


def _validate_variant_identity(
    baseline: Sequence[RunMetrics], improved: Sequence[RunMetrics]
) -> None:
    baseline_identity = (
        baseline[0].scenario_id,
        baseline[0].persona_id,
        baseline[0].policy,
        baseline[0].config_digest,
        baseline[0].prominence_provider_id,
    )
    improved_identity = (
        improved[0].scenario_id,
        improved[0].persona_id,
        improved[0].policy,
        improved[0].config_digest,
        improved[0].prominence_provider_id,
    )
    if baseline_identity != improved_identity:
        raise ValueError(
            "paired variants must share scenario, persona, policy, and config"
        )
