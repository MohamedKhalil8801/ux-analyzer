"""Typed, evidence-backed usability finding rules."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from ux_analyzer.application.evaluation import (
    RunMetrics,
    format_prominence_provenance,
)
from ux_analyzer.domain.findings import (
    EvidenceClass,
    Finding,
    FindingSeverity,
    Reproducibility,
    UnsupportedHumanClaimError,
)


class FindingCategory(StrEnum):
    """Supported POC finding categories."""

    WEAK_TARGET_PROMINENCE = "weak-target-prominence"
    WEAK_SCENT = "weak-scent"
    STRONG_MISLEADING_ALTERNATIVE = "strong-misleading-alternative"
    UNEXPECTED_HIERARCHY = "unexpected-hierarchy"
    AMBIGUOUS_ICON_LABEL = "ambiguous-icon-label"
    EXCESSIVE_DEPTH = "excessive-depth"
    TARGET_BELOW_FOLD = "target-below-fold"
    MISSING_FEEDBACK = "missing-feedback"
    WRONG_ACTION_BURDEN = "wrong-action-burden"
    POOR_RECOVERY = "poor-recovery"


@dataclass(frozen=True, slots=True)
class FindingRuleConfig:
    """Thresholds for typed finding predicates."""

    version: str = "finding-rules-v1"
    weak_target_prominence_below: float = 0.25
    weak_scent_below: float = 0.30
    misleading_scent_margin: float = 0.20
    excessive_navigation_depth_at_least: int = 4
    wrong_action_count_at_least: int = 2

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("finding rule version must not be empty")
        for name in (
            "weak_target_prominence_below",
            "weak_scent_below",
            "misleading_scent_margin",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.excessive_navigation_depth_at_least < 1:
            raise ValueError("excessive depth threshold must be positive")
        if self.wrong_action_count_at_least < 1:
            raise ValueError("wrong-action threshold must be positive")


Predicate = Callable[[RunMetrics], bool]


@dataclass(frozen=True, slots=True)
class FindingRule:
    """One typed predicate that can emit one evidence-backed finding."""

    category: FindingCategory | str
    severity: FindingSeverity | str
    evidence_class: EvidenceClass | str
    predicate: Predicate
    limitations: tuple[str, ...] = ("simulated benchmark evidence",)

    def __post_init__(self) -> None:
        category = FindingCategory(self.category)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "severity", FindingSeverity(self.severity))
        evidence_class = EvidenceClass(self.evidence_class)
        if evidence_class is EvidenceClass.UNSUPPORTED_HUMAN_CLAIM:
            raise UnsupportedHumanClaimError(
                "unsupported human claim cannot become finding rule"
            )
        object.__setattr__(self, "evidence_class", evidence_class)
        if not callable(self.predicate):
            raise TypeError("finding rule predicate must be callable")
        object.__setattr__(self, "limitations", tuple(self.limitations))

    def apply(self, metrics: RunMetrics) -> Finding | None:
        """Apply predicate and create finding with stable evidence references."""

        if not self.predicate(metrics):
            return None
        category = FindingCategory(self.category)
        evidence_ids = _evidence_ids(metrics, category.value)
        source_event_ids = _source_event_ids(metrics, evidence_ids)
        reproducibility = (
            metrics.reproducibility
            if self.evidence_class is EvidenceClass.MODEL_ESTIMATE
            else Reproducibility.REPRODUCIBLE
        )
        return Finding(
            finding_id=f"{metrics.run_id}:{category.value}",
            category=category.value,
            severity=self.severity,
            reproducibility=reproducibility,
            evidence_class=self.evidence_class,
            evidence_ids=evidence_ids,
            limitations=self.limitations,
            generated_explanation=(
                f"Rule {category.value} triggered for run {metrics.run_id}. "
                f"{_prominence_context(metrics)}"
            ),
            title=_title(category),
            cause=_cause(category, metrics),
            run_ids=(metrics.run_id,),
            viewport_ids=metrics.viewport_ids,
            element_ids=(
                metrics.element_ids
                or ((metrics.target.element_id,) if metrics.target.element_id else ())
            ),
            supporting_metrics=_metric_values(metrics, category.value),
            action_sequence=metrics.action_sequence,
            replay_links=(
                *(
                    (
                        _replay_link(
                            metrics.run_id,
                            metrics.target.element_id,
                            source_event_ids[0] if source_event_ids else None,
                        ),
                    )
                    if metrics.target.element_id
                    else ()
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class FindingRuleSet:
    """Immutable collection of supported finding rules."""

    rules: tuple[FindingRule, ...]

    def __post_init__(self) -> None:
        rules = tuple(self.rules)
        if len({rule.category for rule in rules}) != len(rules):
            raise ValueError("finding rule categories must be unique")
        if any(
            EvidenceClass(rule.evidence_class) is EvidenceClass.UNSUPPORTED_HUMAN_CLAIM
            for rule in rules
        ):
            raise UnsupportedHumanClaimError(
                "unsupported human claim cannot enter finding rule set"
            )
        object.__setattr__(self, "rules", rules)

    @classmethod
    def default(cls, config: FindingRuleConfig | None = None) -> FindingRuleSet:
        """Build all required POC finding categories."""

        settings = config or FindingRuleConfig()
        model = EvidenceClass.MODEL_ESTIMATE
        return cls(
            rules=(
                FindingRule(
                    FindingCategory.WEAK_TARGET_PROMINENCE,
                    FindingSeverity.MEDIUM,
                    model,
                    lambda item: (
                        item.target_prominence is not None
                        and item.target_prominence
                        < settings.weak_target_prominence_below
                    ),
                ),
                FindingRule(
                    FindingCategory.WEAK_SCENT,
                    FindingSeverity.MEDIUM,
                    model,
                    lambda item: (
                        item.target_scent is not None
                        and item.target_scent < settings.weak_scent_below
                    ),
                ),
                FindingRule(
                    FindingCategory.STRONG_MISLEADING_ALTERNATIVE,
                    FindingSeverity.HIGH,
                    model,
                    lambda item: (
                        item.target_scent is not None
                        and item.strongest_competing_scent is not None
                        and item.strongest_competing_scent - item.target_scent
                        >= settings.misleading_scent_margin
                    ),
                ),
                FindingRule(
                    FindingCategory.UNEXPECTED_HIERARCHY,
                    FindingSeverity.MEDIUM,
                    model,
                    lambda item: item.unexpected_hierarchy is True,
                ),
                FindingRule(
                    FindingCategory.AMBIGUOUS_ICON_LABEL,
                    FindingSeverity.MEDIUM,
                    model,
                    lambda item: item.ambiguous_target is True,
                ),
                FindingRule(
                    FindingCategory.EXCESSIVE_DEPTH,
                    FindingSeverity.MEDIUM,
                    model,
                    lambda item: (
                        item.navigation_depth
                        >= settings.excessive_navigation_depth_at_least
                    ),
                ),
                FindingRule(
                    FindingCategory.TARGET_BELOW_FOLD,
                    FindingSeverity.MEDIUM,
                    model,
                    lambda item: item.target_below_fold is True,
                ),
                FindingRule(
                    FindingCategory.MISSING_FEEDBACK,
                    FindingSeverity.HIGH,
                    model,
                    lambda item: item.feedback_observed is False,
                ),
                FindingRule(
                    FindingCategory.WRONG_ACTION_BURDEN,
                    FindingSeverity.HIGH,
                    model,
                    lambda item: (
                        item.wrong_actions >= settings.wrong_action_count_at_least
                    ),
                ),
                FindingRule(
                    FindingCategory.POOR_RECOVERY,
                    FindingSeverity.HIGH,
                    model,
                    lambda item: (
                        item.recovery_success is False
                        or (
                            item.recovery_success is None
                            and item.recovery_actions == 0
                            and item.wrong_actions > 0
                        )
                    ),
                ),
            )
        )

    @classmethod
    def from_rules(cls, rules: Sequence[FindingRule]) -> FindingRuleSet:
        """Create rule set after validating evidence classes."""

        return cls(tuple(rules))

    def evaluate(self, metrics: RunMetrics) -> tuple[Finding, ...]:
        return tuple(
            finding
            for rule in self.rules
            if (finding := rule.apply(metrics)) is not None
        )


def findings_for_run(
    metrics: RunMetrics, rules: FindingRuleSet | None = None
) -> tuple[Finding, ...]:
    """Evaluate default typed rules for one run."""

    return (rules or FindingRuleSet.default()).evaluate(metrics)


def build_findings(
    metrics: RunMetrics, rules: FindingRuleSet | None = None
) -> tuple[Finding, ...]:
    """Compatibility name for finding generation callers."""

    return findings_for_run(metrics, rules)


def _evidence_ids(metrics: RunMetrics, category: str) -> tuple[str, ...]:
    metric_names = {
        FindingCategory.WEAK_TARGET_PROMINENCE.value: ("target-prominence",),
        FindingCategory.WEAK_SCENT.value: ("target-scent",),
        FindingCategory.STRONG_MISLEADING_ALTERNATIVE.value: (
            "target-scent",
            "strongest-competing-scent",
        ),
        FindingCategory.UNEXPECTED_HIERARCHY.value: ("unexpected-hierarchy",),
        FindingCategory.AMBIGUOUS_ICON_LABEL.value: ("ambiguous-target",),
        FindingCategory.EXCESSIVE_DEPTH.value: ("navigation-depth",),
        FindingCategory.TARGET_BELOW_FOLD.value: ("target-below-fold",),
        FindingCategory.MISSING_FEEDBACK.value: ("feedback-observed",),
        FindingCategory.WRONG_ACTION_BURDEN.value: ("wrong-actions",),
        FindingCategory.POOR_RECOVERY.value: (
            "recovery-success",
            "recovery-actions",
        ),
    }.get(category, ())
    related = tuple(
        evidence.evidence_id
        for evidence in metrics.evidence
        if any(name in evidence.evidence_id for name in metric_names)
    )
    if related:
        return related
    return (f"{metrics.run_id}:{category}",)


def _source_event_ids(
    metrics: RunMetrics, evidence_ids: tuple[str, ...]
) -> tuple[str, ...]:
    result: list[str] = []
    for evidence in metrics.evidence:
        if evidence.evidence_id not in evidence_ids:
            continue
        for event_id in evidence.source_event_ids:
            if event_id not in result:
                result.append(event_id)
    return tuple(result)


def _replay_link(run_id: str, element_id: str, event_id: str | None) -> str:
    event = f"&event={event_id}" if event_id else ""
    return f"#run={run_id}{event}&element={element_id}"


def _metric_names(category: str) -> tuple[str, ...]:
    return {
        FindingCategory.WEAK_TARGET_PROMINENCE.value: ("target-prominence",),
        FindingCategory.WEAK_SCENT.value: ("target-scent",),
        FindingCategory.STRONG_MISLEADING_ALTERNATIVE.value: (
            "target-scent",
            "strongest-competing-scent",
        ),
        FindingCategory.UNEXPECTED_HIERARCHY.value: ("unexpected-hierarchy",),
        FindingCategory.AMBIGUOUS_ICON_LABEL.value: ("ambiguous-target",),
        FindingCategory.EXCESSIVE_DEPTH.value: ("navigation-depth",),
        FindingCategory.TARGET_BELOW_FOLD.value: ("target-below-fold",),
        FindingCategory.MISSING_FEEDBACK.value: ("feedback-observed",),
        FindingCategory.WRONG_ACTION_BURDEN.value: ("wrong-actions",),
        FindingCategory.POOR_RECOVERY.value: (
            "recovery-success",
            "recovery-actions",
        ),
    }.get(category, ())


def _metric_values(metrics: RunMetrics, category: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in _metric_names(category):
        try:
            values[name] = metrics.metric(name).value
        except KeyError:
            continue
    return values


def _title(category: FindingCategory) -> str:
    return {
        FindingCategory.WEAK_TARGET_PROMINENCE: "Target is visually easy to miss",
        FindingCategory.WEAK_SCENT: "Target wording gives weak goal cues",
        FindingCategory.STRONG_MISLEADING_ALTERNATIVE: (
            "Another control looks more relevant than target"
        ),
        FindingCategory.UNEXPECTED_HIERARCHY: "Target sits in an unexpected hierarchy",
        FindingCategory.AMBIGUOUS_ICON_LABEL: "Target label or icon is ambiguous",
        FindingCategory.EXCESSIVE_DEPTH: "Target requires too many navigation steps",
        FindingCategory.TARGET_BELOW_FOLD: "Target starts below visible viewport",
        FindingCategory.MISSING_FEEDBACK: "Action lacks visible success feedback",
        FindingCategory.WRONG_ACTION_BURDEN: "Task attracts repeated wrong actions",
        FindingCategory.POOR_RECOVERY: "Task provides weak recovery after errors",
    }[category]


def _cause(category: FindingCategory, metrics: RunMetrics) -> str:
    values = _metric_values(metrics, category.value)
    details = ", ".join(f"{name}={value:g}" for name, value in values.items())
    descriptions = {
        FindingCategory.WEAK_TARGET_PROMINENCE: (
            "Recorded target prominence is low relative to configured rule."
        ),
        FindingCategory.WEAK_SCENT: (
            "Model-estimated target scent is weak for configured goal."
        ),
        FindingCategory.STRONG_MISLEADING_ALTERNATIVE: (
            "Recorded competitor scent exceeds target scent by configured margin."
        ),
        FindingCategory.UNEXPECTED_HIERARCHY: (
            "Recorded navigation path places target behind unexpected labels or levels."
        ),
        FindingCategory.AMBIGUOUS_ICON_LABEL: (
            "Recorded target or path labels do not clearly match task goal."
        ),
        FindingCategory.EXCESSIVE_DEPTH: (
            "Recorded successful navigation depth reached configured limit."
        ),
        FindingCategory.TARGET_BELOW_FOLD: (
            "Target first appeared only after recorded scroll action."
        ),
        FindingCategory.MISSING_FEEDBACK: (
            "Post-action snapshot contained no goal-matching visible success message."
        ),
        FindingCategory.WRONG_ACTION_BURDEN: (
            "Recorded run crossed configured wrong-action count."
        ),
        FindingCategory.POOR_RECOVERY: (
            "Recorded recovery actions did not restore verified progress."
        ),
    }[category]
    context = _prominence_context(metrics)
    if details:
        return f"{descriptions} {context} Supporting metrics: {details}."
    return f"{descriptions} {context}"


def _prominence_context(metrics: RunMetrics) -> str:
    provenance = format_prominence_provenance(
        provider_id=metrics.prominence_provider_id,
        active_search_stage=metrics.active_search_stage,
        target_prominence_profiles=metrics.target_prominence_profiles,
        target_prominence_sources=metrics.target_prominence_sources,
        profile_event_ids=metrics.prominence_profile_event_ids,
        operational_event_ids=metrics.prominence_operational_event_ids,
        model_id=metrics.prominence_model_id,
        model_version=metrics.prominence_model_version,
        provider_version=metrics.prominence_provider_version,
    )
    fallback = (
        f" Fallback warning: {metrics.prominence_fallback_reason}."
        if metrics.prominence_fallback_reason
        else ""
    )
    return (
        "Prominence evidence: "
        f"{provenance}; limitation=simulated model-estimate evidence, "
        f"not human-attention calibration.{fallback}"
    )
