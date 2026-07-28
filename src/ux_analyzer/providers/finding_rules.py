"""Typed, evidence-backed usability finding rules."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from ux_analyzer.application.evaluation import RunMetrics
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

    weak_target_prominence_below: float = 0.25
    weak_scent_below: float = 0.30
    misleading_scent_margin: float = 0.20
    excessive_navigation_depth_at_least: int = 4
    wrong_action_count_at_least: int = 2

    def __post_init__(self) -> None:
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
                f"Rule {category.value} triggered for run {metrics.run_id}."
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
        fact = EvidenceClass.DETERMINISTIC_FACT
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
                    fact,
                    lambda item: item.ambiguous_target is True,
                ),
                FindingRule(
                    FindingCategory.EXCESSIVE_DEPTH,
                    FindingSeverity.MEDIUM,
                    fact,
                    lambda item: (
                        item.navigation_depth
                        >= settings.excessive_navigation_depth_at_least
                    ),
                ),
                FindingRule(
                    FindingCategory.TARGET_BELOW_FOLD,
                    FindingSeverity.MEDIUM,
                    fact,
                    lambda item: item.target_below_fold is True,
                ),
                FindingRule(
                    FindingCategory.MISSING_FEEDBACK,
                    FindingSeverity.HIGH,
                    fact,
                    lambda item: item.feedback_observed is False,
                ),
                FindingRule(
                    FindingCategory.WRONG_ACTION_BURDEN,
                    FindingSeverity.HIGH,
                    fact,
                    lambda item: (
                        item.wrong_actions >= settings.wrong_action_count_at_least
                    ),
                ),
                FindingRule(
                    FindingCategory.POOR_RECOVERY,
                    FindingSeverity.HIGH,
                    fact,
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
