"""Scoring + tier math parity tests (reference index.ts / verdict.ts)."""

from __future__ import annotations

from ux_analyzer.analysis.slop.scoring import (
    combine_axes,
    grade_for_score,
    score_copy,
    score_patterns,
    tier_for,
)


def _row(weight: int, triggered: bool) -> dict:
    return {"id": f"p{weight}", "weight": weight, "triggered": triggered}


def test_score_patterns_sums_weights_and_clamps() -> None:
    rows = [_row(8, True), _row(4, True), _row(3, False)]
    result = score_patterns(rows)
    assert result["score"] == 12
    assert result["tier"] == "Mild"
    assert result["patternsFlagged"] == 2
    assert result["patternsTotal"] == 3
    # Clamped at 100
    rows = [_row(8, True)] * 14
    assert score_patterns(rows)["score"] == 100


def test_tier_bands() -> None:
    assert tier_for("design", 0) == "Clean"
    assert tier_for("design", 9) == "Clean"
    assert tier_for("design", 10) == "Mild"
    assert tier_for("design", 27) == "Mild"
    assert tier_for("design", 28) == "Heavy"
    assert tier_for("design", 100) == "Heavy"
    assert tier_for("copy", 7) == "Clean"
    assert tier_for("copy", 8) == "Mild"
    assert tier_for("copy", 20) == "Heavy"


def test_grade_bands() -> None:
    assert grade_for_score(0) == "A+"
    assert grade_for_score(9) == "A-"
    assert grade_for_score(27) == "C"
    assert grade_for_score(28) == "D+"
    assert grade_for_score(31) == "D+"
    assert grade_for_score(100) == "F"


def test_score_copy_thin_page_stays_clean() -> None:
    rows = [
        {"id": "buzzword_density", "weight": 7, "triggered": True},
    ]
    result = score_copy(rows, word_count=30)  # thin page
    assert result["thin"] is True
    assert result["score"] == 0
    assert result["tier"] == "Clean"
    assert rows[0]["triggered"] is False


def test_score_copy_normal_page_scores() -> None:
    rows = [
        {"id": "em_dash_overload", "weight": 5, "triggered": True},
        {"id": "filler_openers", "weight": 5, "triggered": True},
    ]
    result = score_copy(rows, word_count=500)
    assert result["thin"] is False
    assert result["score"] == 10
    assert result["tier"] == "Mild"


def test_combine_axes_max_plus_penalty() -> None:
    design = {"score": 20, "tier": "Mild"}
    copy = {"score": 5, "tier": "Clean"}
    result = combine_axes({"design": design, "copy": copy})
    assert result["unifiedScore"] == 20  # one dirty axis: no penalty
    assert result["unifiedTier"] == "Mild"

    dirty_copy = {"score": 12, "tier": "Mild"}
    result = combine_axes({"design": design, "copy": dirty_copy})
    assert result["unifiedScore"] == 26  # 20 + 6 for the second dirty axis
    assert result["unifiedTier"] == "Mild"

    heavy = {"score": 30, "tier": "Heavy"}
    result = combine_axes({"design": heavy, "copy": dirty_copy})
    assert result["unifiedScore"] == 36


def test_combine_axes_empty() -> None:
    result = combine_axes({})
    assert result["unifiedScore"] == 0
    assert result["unifiedTier"] == "Clean"
