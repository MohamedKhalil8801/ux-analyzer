"""Application-facing prominence contracts and learned evidence batches."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

from ux_analyzer.domain.interface import ViewportSnapshot
from ux_analyzer.domain.saliency import ElementAttentionProfile, SearchStage
from ux_analyzer.ports.artifacts import RunBundleWriter
from ux_analyzer.ports.observation import ObservationCapture


def _empty_float_map() -> dict[str, float]:
    return {}


def _empty_component_provenance_map() -> dict[str, str]:
    return {}


@dataclass(frozen=True, slots=True)
class FeatureMeasurement:
    """Raw feature, normalized feature, and its weighted contribution."""

    raw: float
    normalized: float
    contribution: float


@dataclass(frozen=True, slots=True)
class ProminenceResult:
    """Application-facing prominence evidence for one element."""

    element_id: str
    raw_score: float
    normalized_probability: float
    feature_contributions: Mapping[str, float] = field(default_factory=_empty_float_map)
    raw_values: Mapping[str, float] = field(default_factory=_empty_float_map)
    normalized_values: Mapping[str, float] = field(default_factory=_empty_float_map)
    first_notice_probability: float | None = None
    notice_within_budget_probability: float | None = None
    provider_id: str = "heuristic-prominence"
    stage: str | None = None
    evidence_kind: str = "derived"
    evidence_source: str | None = None
    component_provenance: Mapping[str, str] = field(
        default_factory=_empty_component_provenance_map
    )

    def __post_init__(self) -> None:
        if not self.element_id:
            raise ValueError("prominence element ID must not be empty")
        for name, value in (
            ("raw_score", self.raw_score),
            ("normalized_probability", self.normalized_probability),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.normalized_probability <= 1:
            raise ValueError("normalized_probability must be between 0 and 1")
        feature_contributions: dict[str, float] = dict(self.feature_contributions)
        raw_values: dict[str, float] = dict(self.raw_values)
        normalized_values: dict[str, float] = dict(self.normalized_values)
        component_provenance = dict(self.component_provenance)
        if any(
            not name.strip() or not source.strip()
            for name, source in component_provenance.items()
        ):
            raise ValueError("component provenance needs non-empty text")
        object.__setattr__(
            self, "feature_contributions", MappingProxyType(feature_contributions)
        )
        object.__setattr__(self, "raw_values", MappingProxyType(raw_values))
        object.__setattr__(
            self, "normalized_values", MappingProxyType(normalized_values)
        )
        object.__setattr__(
            self, "component_provenance", MappingProxyType(component_provenance)
        )
        first_notice = self.first_notice_probability
        if first_notice is None:
            first_notice = self.normalized_probability
            object.__setattr__(self, "first_notice_probability", first_notice)
        within_budget = self.notice_within_budget_probability
        if within_budget is None:
            within_budget = self.normalized_probability
            object.__setattr__(self, "notice_within_budget_probability", within_budget)
        for name, value in (
            ("first_notice_probability", first_notice),
            ("notice_within_budget_probability", within_budget),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not self.provider_id.strip():
            raise ValueError("prominence provider ID must not be empty")
        if self.stage is not None and not self.stage.strip():
            raise ValueError("prominence stage must not be empty")
        if self.evidence_kind not in {
            "predicted",
            "derived",
            "fallback",
            "unavailable",
        }:
            raise ValueError("prominence evidence kind is invalid")
        if self.evidence_source is not None and not self.evidence_source.strip():
            raise ValueError("prominence evidence source must not be empty")

    @property
    def score(self) -> float:
        """Compatibility alias for the unnormalized weighted score."""

        return self.raw_score

    @property
    def probability(self) -> float:
        """Compatibility alias for the normalized notice probability."""

        return self.normalized_probability

    @property
    def feature_values(self) -> Mapping[str, FeatureMeasurement]:
        """Combine raw, normalized, and contribution maps by feature."""

        return MappingProxyType(
            {
                name: FeatureMeasurement(
                    raw=self.raw_values.get(name, 0.0),
                    normalized=self.normalized_values.get(name, 0.0),
                    contribution=self.feature_contributions.get(name, 0.0),
                )
                for name in set(self.raw_values)
                | set(self.normalized_values)
                | set(self.feature_contributions)
            }
        )


@dataclass(frozen=True, slots=True)
class ProminenceBatch:
    """Selected operational scores plus model evidence and failure provenance."""

    scores: tuple[ProminenceResult, ...]
    provider_id: str
    active_provider_id: str
    stage: SearchStage | str
    learned_profiles: tuple[ElementAttentionProfile, ...] = ()
    learned_scores: tuple[ProminenceResult, ...] = ()
    heuristic_scores: tuple[ProminenceResult, ...] = ()
    hybrid_scores: tuple[ProminenceResult, ...] = ()
    learned_available: bool = True
    learned_unavailable_reason: str | None = None
    fallback_reason: str | None = None
    cache_state: str = "miss"
    unavailable_learned_profiles: tuple[ElementAttentionProfile, ...] = ()

    def __post_init__(self) -> None:
        scores = tuple(self.scores)
        learned_profiles = tuple(self.learned_profiles)
        unavailable_learned_profiles = tuple(self.unavailable_learned_profiles)
        learned_scores = tuple(self.learned_scores)
        heuristic_scores = tuple(self.heuristic_scores)
        hybrid_scores = tuple(self.hybrid_scores)
        if not self.provider_id.strip() or not self.active_provider_id.strip():
            raise ValueError("prominence batch provider IDs must not be empty")
        stage = SearchStage(self.stage)
        if self.cache_state not in {"hit", "miss", "fallback"}:
            raise ValueError("prominence batch cache state is invalid")
        if self.learned_available and self.learned_unavailable_reason is not None:
            raise ValueError(
                "available learned prominence must not have failure reason"
            )
        if not self.learned_available and not self.learned_unavailable_reason:
            raise ValueError("unavailable learned prominence needs failure reason")
        for name, values in (
            ("scores", scores),
            ("learned_scores", learned_scores),
            ("heuristic_scores", heuristic_scores),
            ("hybrid_scores", hybrid_scores),
            ("unavailable_learned_profiles", unavailable_learned_profiles),
        ):
            ids = tuple(item.element_id for item in values)
            if len(ids) != len(set(ids)):
                raise ValueError(f"{name} must not contain duplicate elements")
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "learned_profiles", learned_profiles)
        object.__setattr__(
            self, "unavailable_learned_profiles", unavailable_learned_profiles
        )
        object.__setattr__(self, "learned_scores", learned_scores)
        object.__setattr__(self, "heuristic_scores", heuristic_scores)
        object.__setattr__(self, "hybrid_scores", hybrid_scores)
        object.__setattr__(self, "stage", stage)

    @property
    def results(self) -> tuple[ProminenceResult, ...]:
        """Compatibility alias for callers that call scores results."""

        return self.scores


class ProminenceProvider(Protocol):
    """Application boundary for capture-scoped prominence selection."""

    def score(
        self,
        capture: ObservationCapture,
        snapshot: ViewportSnapshot,
        stage: SearchStage | str,
        artifacts: RunBundleWriter | None,
    ) -> ProminenceBatch: ...


__all__ = [
    "FeatureMeasurement",
    "ProminenceBatch",
    "ProminenceProvider",
    "ProminenceResult",
]
