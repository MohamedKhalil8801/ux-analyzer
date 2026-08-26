"""Score + tier math (port of @slop-detect/core index.ts / verdict.ts).

0-100 score = sum of triggered pattern weights, clamped. Tiers:
design Clean(0-9) Mild(10-27) Heavy(>=28); copy Clean(0-7) Mild(8-19)
Heavy(>=20). Unified = max axis + 6 per additional dirty axis.
"""

from __future__ import annotations

from typing import Any

DEFINITIONS_VERSION = "2026.08"

AXIS_TIERS = {"design": {"mild": 10, "heavy": 28}, "copy": {"mild": 8, "heavy": 20}}

GRADE_BANDS = [
    (2, "A+"),
    (5, "A"),
    (9, "A-"),
    (14, "B+"),
    (19, "B"),
    (23, "B-"),
    (27, "C"),
    (33, "D+"),
    (39, "D"),
    (100, "F"),
]


def tier_for(axis: str, score: int) -> str:
    t = AXIS_TIERS.get(axis, AXIS_TIERS["design"])
    return "Heavy" if score >= t["heavy"] else "Mild" if score >= t["mild"] else "Clean"


def grade_for_score(score: int) -> str:
    s = max(0, min(100, score))
    for band_max, grade in GRADE_BANDS:
        if s <= band_max:
            return grade
    return "F"


_VERDICTS = {
    "Clean": [
        "Crafted, not generated. This page has a point of view.",
        "Clean. Reads like a human made deliberate choices.",
        "No template tells here — this one earned its look.",
        "Premium-feeling. The AI-slop fingerprint is absent.",
    ],
    "Mild": [
        "Mostly clean, with a few template tells creeping in.",
        "Decent bones, but the slop is starting to show.",
        "A handful of AI-default choices away from genuinely sharp.",
        "Good page wearing a couple of borrowed clichés.",
    ],
    "Heavy": [
        "Heavy slop. This wears the Cursor/v0/Bolt starter kit head to toe.",
        "Straight off the AI assembly line — gradients, Inter, the works.",
        "You can smell the default template from here.",
        "Maximum genericness. Every tell is firing at once.",
    ],
}


def _pick(pool: list[str], seed: str) -> str:
    h = 2166136261
    for ch in seed:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return pool[abs(h) % len(pool)]


def verdict_for(score: int, tier: str, triggered: list[dict[str, Any]]) -> str:
    pool = _VERDICTS.get(tier, _VERDICTS["Mild"])
    sig = ",".join(sorted(str(p.get("id", p)) for p in triggered))
    return _pick(pool, f"{tier}:{sig}")


def score_patterns(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Rows: [{weight, triggered, ...}] → {score, tier, grade, verdict, ...}."""
    score = sum(p.get("weight", 0) for p in rows if p.get("triggered"))
    clamped = min(100, score)
    tier = tier_for("design", clamped)
    triggered = [p for p in rows if p.get("triggered")]
    return {
        "score": clamped,
        "tier": tier,
        "grade": grade_for_score(clamped),
        "verdict": verdict_for(clamped, tier, triggered),
        "patternsFlagged": len(triggered),
        "patternsTotal": len(rows),
        "definitionsVersion": DEFINITIONS_VERSION,
    }


def score_copy(patterns: list[dict[str, Any]], word_count: int) -> dict[str, Any]:
    """Copy-axis summary; word_count < 40 is too thin to judge (Clean + flag)."""
    thin = (word_count or 0) < 40
    for p in patterns:
        p["triggered"] = not thin and bool(p.get("triggered"))
    raw = sum(p.get("weight", 0) for p in patterns if p.get("triggered"))
    score = min(100, raw)
    tier = tier_for("copy", score)
    triggered = [p for p in patterns if p.get("triggered")]
    return {
        "axis": "copy",
        "score": score,
        "tier": tier,
        "grade": grade_for_score(score),
        "patternsFlagged": len(triggered),
        "patternsTotal": len(patterns),
        "wordCount": word_count or 0,
        "thin": thin,
    }


def combine_axes(axis_summaries: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    present = [k for k, v in axis_summaries.items() if v is not None]
    scores = [int(axis_summaries[k]["score"]) for k in present]  # type: ignore[index]
    if not scores:
        return {
            "unifiedScore": 0,
            "unifiedTier": "Clean",
            "unifiedGrade": grade_for_score(0),
            "axesScored": [],
        }
    mx = max(scores)
    dirty = sum(1 for k in present if axis_summaries[k]["tier"] != "Clean")  # type: ignore[index]
    penalty = max(0, dirty - 1) * 6
    unified = min(100, mx + penalty)
    return {
        "unifiedScore": unified,
        "unifiedTier": tier_for("design", unified),
        "unifiedGrade": grade_for_score(unified),
        "axesScored": present,
        "dirtyAxes": dirty,
    }
