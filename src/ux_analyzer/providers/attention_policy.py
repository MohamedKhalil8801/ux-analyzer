"""Seeded progressive attention selection over prominence-scored snapshots."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from ux_analyzer.domain.attention import (
    AttentionRecoveryMiss,
    AttentionState,
    CoarseScent,
    PersonaObservation,
    ProgressiveObservation,
)
from ux_analyzer.domain.interface import ViewportSnapshot
from ux_analyzer.providers.prominence import ProminenceResult

CoarseScentInput = Sequence[CoarseScent] | Mapping[str, float] | None


def _empty_region_priors() -> dict[str, float]:
    return {}


@dataclass(frozen=True, slots=True)
class AttentionPolicyConfig:
    """Versioned, configurable policy weights and reveal limits."""

    version: str = "progressive-attention-v4"
    batch_size: int = 2
    cross_region_exploration: int = 1
    temperature: float = 1.0
    prominence_weight: float = 1.0
    coarse_scent_weight: float = 0.5
    novelty_penalty: float = 0.25
    failure_penalty: float = 0.5
    recovery_scent_threshold: float = 0.9
    recovery_after_misses: int = 2
    region_priors: Mapping[str, float] = field(default_factory=_empty_region_priors)

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("attention policy version must not be empty")
        if not 1 <= self.batch_size <= 3:
            raise ValueError("attention batch_size must be between 1 and 3")
        if not 0 <= self.cross_region_exploration <= 2:
            raise ValueError("cross_region_exploration must be between 0 and 2")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("attention temperature must be greater than zero")
        for name, value in (
            ("prominence_weight", self.prominence_weight),
            ("coarse_scent_weight", self.coarse_scent_weight),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must not be negative")
        for name, value in (
            ("novelty_penalty", self.novelty_penalty),
            ("failure_penalty", self.failure_penalty),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not 0 <= self.recovery_scent_threshold <= 1:
            raise ValueError("recovery_scent_threshold must be between 0 and 1")
        if self.recovery_after_misses < 1:
            raise ValueError("recovery_after_misses must be positive")
        for region_id, value in self.region_priors.items():
            if not region_id or not math.isfinite(value):
                raise ValueError("region priors need non-empty IDs and finite values")
        object.__setattr__(
            self, "region_priors", MappingProxyType(dict(self.region_priors))
        )


@dataclass(frozen=True, slots=True)
class ObservationSelection:
    """Sampled observation plus probabilities used to make the choice."""

    observation: PersonaObservation
    region_id: str | None
    element_probabilities: Mapping[str, float]
    region_probabilities: Mapping[str | None, float]
    selection_mode: str
    recovery_selected_ids: tuple[str, ...] = ()
    next_recovery_state: tuple[AttentionRecoveryMiss, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "element_probabilities",
            MappingProxyType(dict(self.element_probabilities)),
        )
        object.__setattr__(
            self,
            "region_probabilities",
            MappingProxyType(dict(self.region_probabilities)),
        )
        object.__setattr__(
            self, "recovery_selected_ids", tuple(self.recovery_selected_ids)
        )
        object.__setattr__(
            self, "next_recovery_state", tuple(self.next_recovery_state)
        )

    @property
    def selected_ids(self) -> tuple[str, ...]:
        """IDs revealed by this bounded observation, in sample order."""

        return tuple(element.id for element in self.observation.newly_revealed_elements)

    @property
    def newly_revealed_elements(self):
        """Convenience view matching the domain observation contract."""

        return self.observation.newly_revealed_elements


@dataclass(frozen=True, slots=True)
class _Candidate:
    element_id: str
    priority: float
    logit: float


class ProgressiveAttentionPolicy:
    """Select regions first, then sample a small batch of child elements."""

    id = "progressive-attention"

    def __init__(self, config: AttentionPolicyConfig | None = None) -> None:
        self.config = config or AttentionPolicyConfig()

    @property
    def version(self) -> str:
        return self.config.version

    def next_observation(
        self,
        state: AttentionState,
        snapshot: ViewportSnapshot,
        scores: Sequence[ProminenceResult],
        coarse_scent: CoarseScentInput,
        rng: random.Random,
        *,
        recovery_level: int = 0,
    ) -> ObservationSelection:
        """Return one seeded, bounded progressive observation."""

        if recovery_level < 0:
            raise ValueError("recovery_level must not be negative")
        score_by_id = _score_map(scores, snapshot)
        scent_by_id = _scent_map(coarse_scent, snapshot)
        visible_ids = {
            element.id
            for element in snapshot.elements
            if element.visibility_fraction > 0
        }
        candidates = tuple(
            _candidate(
                element_id=element.id,
                prominence=score_by_id.get(element.id),
                scent=scent_by_id.get(element.id, 0.0),
                state=state,
                config=self.config,
            )
            for element in snapshot.elements
            if element.id in visible_ids and element.id not in state.remembered_ids
        )
        if not candidates:
            raise ValueError("no unobserved visible elements remain")
        element_probabilities = _probabilities(candidates, self.config.temperature)
        grouped = _group_candidates(snapshot, candidates)
        has_region_candidates = any(region_id is not None for region_id in grouped)
        region_probabilities = _region_probabilities(grouped, self.config)
        if has_region_candidates:
            region_id = _sample(
                tuple(region_probabilities), tuple(region_probabilities.values()), rng
            )
            pool = grouped[region_id]
            mode = "region-first"
            selected = _select_region_batch(
                snapshot,
                candidates,
                pool,
                region_id,
                self.config,
                rng,
            )
            if _selected_region(snapshot, selected) is None:
                mode = f"{mode}+cross-region"
                region_id = None
        else:
            region_id = None
            pool = list(candidates)
            mode = "element-fallback"
            selected = _sample_without_replacement(
                pool,
                self.config.batch_size,
                self.config.temperature,
                rng,
            )
        selected, recovery_ids, recovery_state = _apply_recovery(
            state,
            snapshot,
            candidates,
            selected,
            scent_by_id,
            self.config,
        )
        if recovery_ids:
            mode = f"{mode}+recovery"
            region_id = _selected_region(snapshot, selected)
        selected_ids = tuple(candidate.element_id for candidate in selected)
        remembered_ids = tuple(
            item.element_id
            for item in state.memory
            if item.element_id in {element.id for element in snapshot.elements}
            and item.element_id not in selected_ids
        )
        observation = ProgressiveObservation.from_snapshot(
            snapshot,
            newly_revealed_ids=selected_ids,
            remembered_ids=remembered_ids,
            region_id=region_id,
        )
        return ObservationSelection(
            observation=observation,
            region_id=region_id,
            element_probabilities=element_probabilities,
            region_probabilities=region_probabilities,
            selection_mode=mode,
            recovery_selected_ids=recovery_ids,
            next_recovery_state=recovery_state,
        )


def _score_map(
    scores: Sequence[ProminenceResult], snapshot: ViewportSnapshot
) -> dict[str, ProminenceResult]:
    result: dict[str, ProminenceResult] = {}
    known_ids = {element.id for element in snapshot.elements}
    for score in scores:
        if score.element_id not in known_ids:
            raise ValueError(
                f"prominence score references unknown element {score.element_id!r}"
            )
        if score.element_id in result:
            raise ValueError(f"duplicate prominence score for {score.element_id!r}")
        result[score.element_id] = score
    return result


def _scent_map(
    coarse_scent: CoarseScentInput, snapshot: ViewportSnapshot
) -> dict[str, float]:
    if coarse_scent is None:
        return {}
    if isinstance(coarse_scent, Mapping):
        values = dict(coarse_scent)
    else:
        values = {}
        for item in coarse_scent:
            if item.viewport_id != snapshot.id:
                raise ValueError("coarse scent belongs to a different viewport")
            values[item.element_id] = item.score
    for element_id, score in values.items():
        if element_id not in {element.id for element in snapshot.elements}:
            raise ValueError(f"coarse scent references unknown element {element_id!r}")
        if not 0 <= score <= 1:
            raise ValueError("coarse scent score must be between 0 and 1")
    return values


def _candidate(
    *,
    element_id: str,
    prominence: ProminenceResult | None,
    scent: float,
    state: AttentionState,
    config: AttentionPolicyConfig,
) -> _Candidate:
    prominence_probability = prominence.normalized_probability if prominence else 0.0
    prominence_probability = max(prominence_probability, 1e-12)
    logit = config.prominence_weight * math.log(prominence_probability)
    logit += config.coarse_scent_weight * scent
    if element_id in state.noticed_ids or element_id in state.inspected_ids:
        logit += math.log(max(1 - config.novelty_penalty, 1e-12))
    if element_id in state.failed_candidates:
        logit += math.log(max(1 - config.failure_penalty, 1e-12))
    return _Candidate(element_id=element_id, priority=math.exp(logit), logit=logit)


def _apply_recovery(
    state: AttentionState,
    snapshot: ViewportSnapshot,
    candidates: Sequence[_Candidate],
    selected: tuple[_Candidate, ...],
    scent_by_id: Mapping[str, float],
    config: AttentionPolicyConfig,
) -> tuple[
    tuple[_Candidate, ...],
    tuple[str, ...],
    tuple[AttentionRecoveryMiss, ...],
]:
    """Force one overdue target after the normal seeded draw has completed."""

    order = {element.id: index for index, element in enumerate(snapshot.elements)}
    unique_keys = _unique_recovery_keys(snapshot)
    candidate_by_id = {candidate.element_id: candidate for candidate in candidates}
    selected_ids = {candidate.element_id for candidate in selected}
    previous_misses = {miss.key: miss.consecutive_misses for miss in state.recovery_misses}
    misses: list[AttentionRecoveryMiss] = []
    overdue: list[tuple[int, AttentionRecoveryMiss, _Candidate]] = []

    for element in snapshot.elements:
        candidate = candidate_by_id.get(element.id)
        key = _recovery_key(snapshot, element)
        if (
            candidate is None
            or key is None
            or key not in unique_keys
            or not element.actionable
            or element.disabled
            or scent_by_id.get(element.id, 0.0)
            < config.recovery_scent_threshold
            or element.id in selected_ids
        ):
            continue
        previous_count = previous_misses.get(key, 0)
        miss = AttentionRecoveryMiss(
            provider_id=key[0],
            lineage_id=key[1],
            consecutive_misses=min(
                config.recovery_after_misses,
                previous_count + 1,
            ),
        )
        misses.append(miss)
        if previous_count >= config.recovery_after_misses:
            overdue.append((previous_count, miss, candidate))

    if not overdue:
        return selected, (), tuple(misses)

    _, forced_miss, forced = min(
        overdue,
        key=lambda item: (
            -item[0],
            -scent_by_id.get(item[2].element_id, 0.0),
            -item[2].logit,
            order[item[2].element_id],
        ),
    )
    retained = selected[:1]
    recovery_state = tuple(miss for miss in misses if miss.key != forced_miss.key)
    return (forced, *retained), (forced.element_id,), recovery_state


def _unique_recovery_keys(
    snapshot: ViewportSnapshot,
) -> frozenset[tuple[str, str]]:
    counts: dict[tuple[str, str], int] = {}
    for element in snapshot.elements:
        key = _recovery_key(snapshot, element)
        if key is not None:
            counts[key] = counts.get(key, 0) + 1
    return frozenset(key for key, count in counts.items() if count == 1)


def _recovery_key(
    snapshot: ViewportSnapshot,
    element: object,
) -> tuple[str, str] | None:
    provider_id = getattr(element, "provider_id", None) or snapshot.provider_id
    lineage_id = getattr(element, "lineage_id", None)
    if (
        not isinstance(provider_id, str)
        or not isinstance(lineage_id, str)
        or (
            snapshot.provider_id is not None
            and provider_id != snapshot.provider_id
        )
    ):
        return None
    return provider_id, lineage_id


def _selected_region(
    snapshot: ViewportSnapshot,
    selected: Sequence[_Candidate],
) -> str | None:
    known_region_ids = {region.id for region in snapshot.regions}
    region_ids = {
        snapshot.element(candidate.element_id).region_id for candidate in selected
    }
    if len(region_ids) != 1:
        return None
    region_id = next(iter(region_ids))
    return region_id if region_id in known_region_ids else None


def _group_candidates(
    snapshot: ViewportSnapshot, candidates: Sequence[_Candidate]
) -> dict[str | None, list[_Candidate]]:
    known_regions = {region.id for region in snapshot.regions}
    by_id = {element.id: element for element in snapshot.elements}
    grouped: dict[str | None, list[_Candidate]] = {}
    for candidate in candidates:
        region_id = by_id[candidate.element_id].region_id
        key = region_id if region_id in known_regions else None
        grouped.setdefault(key, []).append(candidate)
    return grouped


def _probabilities(
    candidates: Sequence[_Candidate], temperature: float
) -> dict[str, float]:
    values = _softmax([candidate.logit for candidate in candidates], temperature)
    return {
        candidate.element_id: values[index]
        for index, candidate in enumerate(candidates)
    }


def _region_probabilities(
    grouped: Mapping[str | None, Sequence[_Candidate]], config: AttentionPolicyConfig
) -> dict[str | None, float]:
    region_ids = tuple(grouped)
    logits: tuple[float, ...] = tuple(
        math.log(sum(candidate.priority for candidate in grouped[region_id]))
        + (config.region_priors.get(region_id, 0.0) if region_id is not None else 0.0)
        for region_id in region_ids
    )
    values = _softmax(logits, config.temperature)
    return {region_id: values[index] for index, region_id in enumerate(region_ids)}


def _sample(
    values: Sequence[str | None], probabilities: Sequence[float], rng: random.Random
) -> str | None:
    point = rng.random()
    cumulative = 0.0
    for value, probability in zip(values, probabilities, strict=True):
        cumulative += probability
        if point < cumulative:
            return value
    return values[-1]


def _sample_without_replacement(
    candidates: Sequence[_Candidate],
    batch_size: int,
    temperature: float,
    rng: random.Random,
) -> tuple[_Candidate, ...]:
    remaining = list(candidates)
    selected: list[_Candidate] = []
    for _ in range(min(batch_size, len(remaining))):
        probabilities = _softmax(
            [candidate.logit for candidate in remaining], temperature
        )
        index = _sample_index(probabilities, rng)
        selected.append(remaining.pop(index))
    return tuple(selected)


def _select_region_batch(
    snapshot: ViewportSnapshot,
    candidates: Sequence[_Candidate],
    region_pool: Sequence[_Candidate],
    region_id: str | None,
    config: AttentionPolicyConfig,
    rng: random.Random,
) -> tuple[_Candidate, ...]:
    exploration_count = min(
        config.cross_region_exploration,
        max(config.batch_size - 1, 0),
    )
    local_count = config.batch_size - exploration_count
    selected = list(
        _sample_without_replacement(
            region_pool,
            local_count,
            config.temperature,
            rng,
        )
    )
    selected_ids = {candidate.element_id for candidate in selected}
    outside = tuple(
        candidate
        for candidate in candidates
        if candidate.element_id not in selected_ids
        and snapshot.element(candidate.element_id).region_id != region_id
    )
    explored = _sample_without_replacement(
        outside,
        exploration_count,
        config.temperature,
        rng,
    )
    selected.extend(explored)
    selected_ids.update(candidate.element_id for candidate in explored)
    if len(selected) < config.batch_size:
        remaining = tuple(
            candidate
            for candidate in candidates
            if candidate.element_id not in selected_ids
        )
        selected.extend(
            _sample_without_replacement(
                remaining,
                config.batch_size - len(selected),
                config.temperature,
                rng,
            )
        )
    return tuple(selected)


def _sample_index(probabilities: Sequence[float], rng: random.Random) -> int:
    point = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if point < cumulative:
            return index
    return len(probabilities) - 1


def _softmax(logits: Sequence[float], temperature: float) -> tuple[float, ...]:
    scaled = [value / temperature for value in logits]
    maximum = max(scaled)
    exponentials = [math.exp(value - maximum) for value in scaled]
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)
