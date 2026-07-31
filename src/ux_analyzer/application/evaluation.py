"""Evidence-preserving benchmark metrics and variant comparison gates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING

from ux_analyzer.domain.attention import (
    Back,
    InteractWithElement,
    PersonaObservation,
    Scroll,
)
from ux_analyzer.domain.benchmark import ApplicationVersionKind
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
    RunOutcome,
    RunOutcomeKind,
)

if TYPE_CHECKING:
    from ux_analyzer.application.run_agent import RunResult


def _empty_float_mapping() -> dict[str, float]:
    return {}


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

    element_id: str
    region_id: str | None = None

    def __post_init__(self) -> None:
        if not self.element_id:
            raise ValueError("evaluation target element ID must not be empty")


@dataclass(frozen=True, slots=True)
class RunEvaluationInputs:
    """Optional evidence unavailable in the current typed run event model."""

    prominence_scores: Mapping[str, float] = field(default_factory=_empty_float_mapping)
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

    def __post_init__(self) -> None:
        for name in ("prominence_scores", "scent_scores"):
            source = getattr(self, name)
            copied = {str(key): float(value) for key, value in source.items()}
            for key, value in copied.items():
                if not key or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"{name} must contain finite scores in [0, 1]")
            object.__setattr__(self, name, MappingProxyType(copied))
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

    @property
    def inspected_element_count(self) -> int:
        return self.inspected_elements

    @property
    def inspected_region_count(self) -> int:
        return self.inspected_regions

    @property
    def completion(self) -> bool:
        return self.verified_completion

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
    """Per scenario/version/persona/policy aggregate."""

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
        else EvaluationTarget(target or target_element_id or "")
    )
    settings = inputs or RunEvaluationInputs()
    config = cost_config or DiscoveryCostConfig()
    state = result.state
    observations = tuple(
        event.observation
        for event in state.events
        if isinstance(event, ObservationRecorded)
    )
    target_rank = _target_rank(observations, selected_target.element_id)
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
        and (
            not event.succeeded or event.action.element_id != selected_target.element_id
        )
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
        target_prominence = settings.prominence_scores.get(selected_target.element_id)
    target_scent = settings.scent_scores.get(selected_target.element_id)
    competitor_scores = [
        score
        for element_id, score in settings.scent_scores.items()
        if element_id != selected_target.element_id
    ]
    strongest_competing_scent = max(competitor_scores, default=None)
    verified = result.verification.verified
    claimed = result.agent_claimed_success
    model_dependent = settings.model_dependent or _has_model_manifest(
        state.provider_manifests
    )
    reproducibility = (
        Reproducibility.MODEL_DEPENDENT if model_dependent else Reproducibility.SEEDED
    )
    event_ids = _event_ids(state.events)
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
            deterministic,
            "Target first appeared in recorded observation order.",
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
        record(
            EvaluationMetric.TARGET_PROMINENCE,
            target_prominence,
            deterministic,
            "Recorded heuristic prominence evidence for target.",
        )
    if target_scent is not None:
        record(
            EvaluationMetric.TARGET_SCENT,
            target_scent,
            estimated,
            "Configured scent evidence for target.",
        )
    if strongest_competing_scent is not None:
        record(
            EvaluationMetric.STRONGEST_COMPETING_SCENT,
            strongest_competing_scent,
            estimated,
            "Strongest configured non-target scent.",
        )
    if settings.target_below_fold is not None:
        record(
            EvaluationMetric.TARGET_BELOW_FOLD,
            float(settings.target_below_fold),
            deterministic,
            "Configured viewport fold fact.",
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
            deterministic,
            "Configured target label or icon ambiguity fact.",
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
            deterministic,
            "Configured visible feedback fact.",
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
            deterministic,
            "Configured recovery outcome.",
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
    )


def calculate_run_metrics(*args: object, **kwargs: object) -> RunMetrics:
    """Compatibility name for callers using calculation vocabulary."""

    return evaluate_run(*args, **kwargs)  # type: ignore[arg-type]


def evaluation_target_for(result: RunResult) -> EvaluationTarget:
    """Infer best recorded target without hidden verifier or selector data."""

    interactions: list[ActionExecuted] = []
    for event in result.state.events:
        if isinstance(event, ActionExecuted) and isinstance(
            event.action, InteractWithElement
        ):
            interactions.append(event)
    successful = [event for event in interactions if event.succeeded]
    selected = successful[-1:] or interactions[-1:]
    if selected:
        action = selected[0].action
        if not isinstance(action, InteractWithElement):
            raise TypeError("recorded interaction lost typed action")
        return EvaluationTarget(action.element_id)
    observations = [
        event.observation
        for event in result.state.events
        if isinstance(event, ObservationRecorded)
    ]
    visible = [
        element
        for observation in observations
        for element in observation.newly_revealed_elements
    ]
    candidate = next(
        (element for element in reversed(visible) if element.actionable), None
    )
    if candidate is None and visible:
        candidate = visible[-1]
    if candidate is None:
        raise ValueError(
            f"run {result.run_id} has no persona-visible evaluation target"
        )
    return EvaluationTarget(candidate.id, candidate.region_id)


def evaluation_inputs_for(result: RunResult) -> RunEvaluationInputs:
    """Build evaluator inputs from evidence recorded by production RunAgent."""

    target = evaluation_target_for(result)
    target_elements = _target_elements(result, target)
    target_ids = {element.id for element in target_elements}
    prominence_scores: dict[str, float] = {}
    for record in result.evidence.prominence:
        for score in record.scores:
            prominence_scores[score.element_id] = score.normalized_probability
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
    target_prominence = next(
        (
            prominence_scores[element_id]
            for element_id in reversed(tuple(prominence_scores))
            if element_id in target_ids
        ),
        None,
    )
    target_below_fold = _target_below_fold(result, target_elements)
    path_labels = _pretarget_interaction_labels(result, target_elements)
    ambiguous_target = not _label_matches_goal(
        target_elements[-1].label, result.state.spec.scenario.goal
    ) or any(
        not _label_matches_goal(label, result.state.spec.scenario.goal)
        for label in path_labels
    )
    unexpected_hierarchy = bool(
        len(path_labels) >= 2
        or (
            path_labels
            and target_elements[-1].id
            not in {element.id for element in result.state.snapshots[0].elements}
            and ambiguous_target
        )
    )
    return RunEvaluationInputs(
        prominence_scores=prominence_scores,
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
        model_dependent=bool(result.evidence.model_calls or result.evidence.scent),
    )


def _target_elements(
    result: RunResult, target: EvaluationTarget
) -> tuple[ElementSnapshot, ...]:
    selected = [
        element
        for snapshot in result.state.snapshots
        for element in snapshot.elements
        if element.id == target.element_id
    ]
    if not selected:
        raise ValueError(f"target {target.element_id!r} has no recorded snapshot")
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
) -> bool:
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
) -> bool:
    target_ids = {element.id for element in target_elements}
    target_executed = False
    goal = result.state.spec.scenario.goal
    for event in result.state.events:
        if isinstance(event, ActionExecuted) and isinstance(
            event.action, InteractWithElement
        ):
            target_executed = target_executed or event.action.element_id in target_ids
            continue
        if target_executed and hasattr(event, "snapshot"):
            snapshot = getattr(event, "snapshot")
            if any(
                _is_success_feedback(element.label, goal)
                for element in snapshot.elements
            ):
                return True
    return False


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
    metrics = tuple(
        result.metrics for result in run_results if result.metrics is not None
    )
    if len(metrics) != len(run_results):
        raise ValueError("every completed run needs persisted metrics")
    cells = aggregate_cells(metrics)
    grouped: dict[
        tuple[str, str, str, str | None],
        dict[ApplicationVersionKind, list[RunMetrics]],
    ] = {}
    by_run_id = {result.run_id: result for result in run_results}
    for metric in metrics:
        result = by_run_id[metric.run_id]
        key = (
            metric.scenario_id,
            metric.persona_id,
            metric.policy,
            metric.config_digest,
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
    )
    if any(
        (
            item.scenario_id,
            item.application_version_id,
            item.persona_id,
            item.policy,
        )
        != identity
        for item in ordered
    ):
        raise ValueError("cell runs must share scenario, version, persona, and policy")
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
    )


def aggregate_cells(runs: Sequence[RunMetrics]) -> tuple[CellAggregate, ...]:
    """Aggregate all cells in stable identity order."""

    groups: dict[tuple[str, str, str, str], list[RunMetrics]] = {}
    for run in runs:
        key = (
            run.scenario_id,
            run.application_version_id,
            run.persona_id,
            run.policy,
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
        lower=ordered[0],
        upper=ordered[-1],
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
    baseline_by_seed = _by_seed(baseline)
    improved_by_seed = _by_seed(improved)
    if set(baseline_by_seed) != set(improved_by_seed):
        raise ValueError("variant comparison requires identical paired seeds")
    _validate_variant_identity(baseline, improved)
    paired_seeds = tuple(sorted(baseline_by_seed))
    baseline_cell = aggregate_cell(baseline)
    improved_cell = aggregate_cell(improved)
    cost_deltas = tuple(
        improved_by_seed[seed].discovery_cost.total
        - baseline_by_seed[seed].discovery_cost.total
        for seed in paired_seeds
    )
    wrong_deltas = tuple(
        improved_by_seed[seed].wrong_actions - baseline_by_seed[seed].wrong_actions
        for seed in paired_seeds
    )
    backtrack_deltas = tuple(
        improved_by_seed[seed].backtracks - baseline_by_seed[seed].backtracks
        for seed in paired_seeds
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
    reasons = tuple(
        reason
        for reason, passed in (
            ("paired median discovery cost did not decrease", cost_decreased),
            ("paired median wrong-action burden increased", wrong_not_increased),
            ("paired median backtrack burden increased", backtrack_not_increased),
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
    )


def _target_rank(
    observations: Sequence[PersonaObservation], target_id: str
) -> int | None:
    rank = 0
    for observation in observations:
        for element in observation.newly_revealed_elements:
            rank += 1
            if element.id == target_id:
                return rank
    return None


def _outcome_kind(outcome: RunOutcome) -> str:
    return str(getattr(outcome, "kind", RunOutcomeKind.INTERNAL_ERROR.value))


def _has_model_manifest(manifests: Sequence[object]) -> bool:
    return any(
        getattr(manifest, "model_id", None)
        and str(getattr(manifest, "role", ""))
        in {"cognitive", "coarse-scent", "full-scent"}
        for manifest in manifests
    )


def _event_ids(events: Sequence[object]) -> tuple[str, ...]:
    return tuple(f"event-{index}" for index, _ in enumerate(events, start=1))


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
            deterministic,
        ),
        (EvaluationMetric.TARGET_PROMINENCE, run.target_prominence, deterministic),
        (EvaluationMetric.TARGET_SCENT, run.target_scent, estimated),
        (
            EvaluationMetric.STRONGEST_COMPETING_SCENT,
            run.strongest_competing_scent,
            estimated,
        ),
        (
            EvaluationMetric.TARGET_BELOW_FOLD,
            float(run.target_below_fold) if run.target_below_fold is not None else None,
            deterministic,
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
            deterministic,
        ),
        (
            EvaluationMetric.FEEDBACK_OBSERVED,
            float(run.feedback_observed) if run.feedback_observed is not None else None,
            deterministic,
        ),
        (
            EvaluationMetric.RECOVERY_SUCCESS,
            float(run.recovery_success) if run.recovery_success is not None else None,
            deterministic,
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


def _by_seed(runs: Sequence[RunMetrics]) -> dict[int, RunMetrics]:
    result: dict[int, RunMetrics] = {}
    for run in runs:
        if run.seed in result:
            raise ValueError(f"duplicate seed in variant cell: {run.seed}")
        result[run.seed] = run
    return result


def _validate_variant_identity(
    baseline: Sequence[RunMetrics], improved: Sequence[RunMetrics]
) -> None:
    baseline_identity = (
        baseline[0].scenario_id,
        baseline[0].persona_id,
        baseline[0].policy,
        baseline[0].config_digest,
    )
    improved_identity = (
        improved[0].scenario_id,
        improved[0].persona_id,
        improved[0].policy,
        improved[0].config_digest,
    )
    if baseline_identity != improved_identity:
        raise ValueError(
            "paired variants must share scenario, persona, policy, and config"
        )
