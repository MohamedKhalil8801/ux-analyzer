"""Provider-neutral ranking metrics and calibrated late fusion."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    top1_accuracy: float
    hit_rate_at_3: float
    mean_reciprocal_rank: float
    ndcg_at_3: float
    pairwise_accuracy: float


@dataclass(frozen=True, slots=True)
class LabeledDistribution:
    labels: Mapping[str, int]
    heuristic: Mapping[str, float]
    foveacast: Mapping[str, float]


def score_ranking(
    labels: Mapping[str, int], probabilities: Mapping[str, float]
) -> RankingMetrics:
    _validate_element_ids(labels, probabilities)
    if not labels:
        raise ValueError("ranking needs at least one element")
    for value in labels.values():
        if not _is_nonnegative_integer(value):
            raise ValueError("relevance labels must be non-negative integers")
    normalized = _normalized_distribution(probabilities)
    ranked = sorted(normalized, key=lambda item: (-normalized[item], item))
    maximum_relevance = max(labels.values())
    relevant = {item for item, value in labels.items() if value == maximum_relevance}
    first_relevant_rank = next(
        index
        for index, element_id in enumerate(ranked, start=1)
        if element_id in relevant
    )
    return RankingMetrics(
        top1_accuracy=float(ranked[0] in relevant),
        hit_rate_at_3=float(any(item in relevant for item in ranked[:3])),
        mean_reciprocal_rank=1.0 / first_relevant_rank,
        ndcg_at_3=_ndcg(labels, ranked, cutoff=3),
        pairwise_accuracy=_pairwise_accuracy(labels, normalized),
    )


def fuse_probabilities(
    heuristic: Mapping[str, float],
    foveacast: Mapping[str, float],
    *,
    foveacast_weight: float,
) -> dict[str, float]:
    _validate_element_ids(heuristic, foveacast)
    if not math.isfinite(foveacast_weight) or not 0.0 <= foveacast_weight <= 1.0:
        raise ValueError("foveacast_weight must be between zero and one")
    heuristic_normalized = _normalized_distribution(heuristic)
    foveacast_normalized = _normalized_distribution(foveacast)
    return {
        element_id: (1.0 - foveacast_weight) * heuristic_normalized[element_id]
        + foveacast_weight * foveacast_normalized[element_id]
        for element_id in heuristic
    }


def select_fusion_weight(cases: Sequence[LabeledDistribution]) -> float:
    if not cases:
        raise ValueError("fusion calibration needs at least one case")
    best_weight = 0.0
    best_score = (-1.0, -1.0, -1.0)
    for step in range(21):
        weight = step / 20
        metrics = [
            score_ranking(
                case.labels,
                fuse_probabilities(
                    case.heuristic,
                    case.foveacast,
                    foveacast_weight=weight,
                ),
            )
            for case in cases
        ]
        score = (
            _mean(item.ndcg_at_3 for item in metrics),
            _mean(item.mean_reciprocal_rank for item in metrics),
            _mean(item.pairwise_accuracy for item in metrics),
        )
        if score > best_score:
            best_score = score
            best_weight = weight
    return best_weight


def _validate_element_ids(
    left: Mapping[str, object], right: Mapping[str, object]
) -> None:
    if set(left) != set(right):
        raise ValueError("distributions must contain the same element IDs")


def _normalized_distribution(values: Mapping[str, float]) -> dict[str, float]:
    if not values:
        raise ValueError("probability distribution must not be empty")
    converted: dict[str, float] = {}
    for element_id, value in values.items():
        if not element_id or not _is_finite_nonnegative_number(value):
            raise ValueError("probabilities must be finite and non-negative")
        converted[element_id] = float(value)
    total = math.fsum(converted.values())
    if total <= 0:
        raise ValueError("probabilities need positive total mass")
    return {element_id: value / total for element_id, value in converted.items()}


def _ndcg(labels: Mapping[str, int], ranked: Sequence[str], *, cutoff: int) -> float:
    def discounted_gain(relevances: Sequence[int]) -> float:
        return math.fsum(
            (2**relevance - 1) / math.log2(index + 2)
            for index, relevance in enumerate(relevances[:cutoff])
        )

    actual = discounted_gain([labels[item] for item in ranked])
    ideal = discounted_gain(sorted(labels.values(), reverse=True))
    return actual / ideal if ideal > 0 else 1.0


def _pairwise_accuracy(
    labels: Mapping[str, int], probabilities: Mapping[str, float]
) -> float:
    element_ids = tuple(labels)
    correct = 0.0
    comparisons = 0
    for left_index, left_id in enumerate(element_ids):
        for right_id in element_ids[left_index + 1 :]:
            relevance_delta = labels[left_id] - labels[right_id]
            if relevance_delta == 0:
                continue
            comparisons += 1
            probability_delta = probabilities[left_id] - probabilities[right_id]
            if probability_delta == 0:
                correct += 0.5
            elif (probability_delta > 0) == (relevance_delta > 0):
                correct += 1.0
    return correct / comparisons if comparisons else 1.0


def _mean(values: Iterable[float]) -> float:
    items = tuple(values)
    return math.fsum(items) / len(items)


def _is_nonnegative_integer(value: object) -> bool:
    return type(value) is not bool and isinstance(value, int) and value >= 0


def _is_finite_nonnegative_number(value: object) -> bool:
    return (
        type(value) is not bool
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


__all__ = [
    "LabeledDistribution",
    "RankingMetrics",
    "fuse_probabilities",
    "score_ranking",
    "select_fusion_weight",
]
