from __future__ import annotations

import pytest

from ux_analyzer.benchmarking.prominence_comparison import (
    LabeledDistribution,
    fuse_probabilities,
    score_ranking,
    select_fusion_weight,
)


def test_perfect_ranking_scores_one_on_all_rank_metrics() -> None:
    metrics = score_ranking(
        {"target": 3, "secondary": 2, "noise": 0},
        {"target": 0.7, "secondary": 0.2, "noise": 0.1},
    )

    assert metrics.top1_accuracy == 1.0
    assert metrics.hit_rate_at_3 == 1.0
    assert metrics.mean_reciprocal_rank == 1.0
    assert metrics.ndcg_at_3 == 1.0
    assert metrics.pairwise_accuracy == 1.0


def test_reversed_ranking_penalizes_order_but_still_hits_target_at_three() -> None:
    metrics = score_ranking(
        {"target": 3, "secondary": 2, "noise": 0},
        {"target": 0.1, "secondary": 0.2, "noise": 0.7},
    )

    assert metrics.top1_accuracy == 0.0
    assert metrics.hit_rate_at_3 == 1.0
    assert metrics.mean_reciprocal_rank == pytest.approx(1 / 3)
    assert metrics.ndcg_at_3 < 0.7
    assert metrics.pairwise_accuracy == 0.0


def test_tied_scores_receive_half_credit_for_unequal_relevance_pairs() -> None:
    metrics = score_ranking(
        {"target": 3, "secondary": 1, "noise": 0},
        {"target": 1.0, "secondary": 1.0, "noise": 0.0},
    )

    assert metrics.pairwise_accuracy == pytest.approx(5 / 6)


def test_missing_probability_is_rejected_instead_of_silently_ignored() -> None:
    with pytest.raises(ValueError, match="same element IDs"):
        score_ranking(
            {"target": 3, "noise": 0},
            {"target": 1.0},
        )


def test_fusion_endpoints_reproduce_pure_provider_distributions() -> None:
    heuristic = {"target": 0.8, "noise": 0.2}
    foveacast = {"target": 0.3, "noise": 0.7}

    assert fuse_probabilities(heuristic, foveacast, foveacast_weight=0.0) == heuristic
    assert fuse_probabilities(heuristic, foveacast, foveacast_weight=1.0) == foveacast


def test_fusion_weight_is_selected_only_from_supplied_calibration_cases() -> None:
    calibration = (
        LabeledDistribution(
            labels={"target": 3, "noise": 0},
            heuristic={"target": 0.9, "noise": 0.1},
            foveacast={"target": 0.1, "noise": 0.9},
        ),
        LabeledDistribution(
            labels={"target": 3, "noise": 0},
            heuristic={"target": 0.8, "noise": 0.2},
            foveacast={"target": 0.2, "noise": 0.8},
        ),
    )

    assert select_fusion_weight(calibration) == 0.0


def test_fusion_rejects_non_matching_or_invalid_distributions() -> None:
    with pytest.raises(ValueError, match="same element IDs"):
        fuse_probabilities(
            {"target": 1.0},
            {"other": 1.0},
            foveacast_weight=0.5,
        )
    with pytest.raises(ValueError, match="between zero and one"):
        fuse_probabilities(
            {"target": 1.0},
            {"target": 1.0},
            foveacast_weight=1.1,
        )
