from __future__ import annotations

import json
from pathlib import Path

CORPUS = Path(__file__).parents[3] / "benchmarks" / "prominence" / "cases.json"
REQUIRED_TAGS = {
    "single-target",
    "competing-cta",
    "dense-navigation",
    "form",
    "heading",
    "low-contrast",
    "occlusion",
    "center-bias",
    "decorative-container",
    "responsive",
    "semantic-vs-visual",
    "reading-order",
}


def test_prominence_corpus_is_complete_and_well_formed() -> None:
    payload = json.loads(CORPUS.read_text(encoding="utf-8"))
    cases = payload["cases"]

    assert payload["version"] == "prominence-corpus-v1"
    assert len(cases) >= 24
    assert {case["split"] for case in cases} == {"calibration", "holdout"}
    assert {case["split"] for case in cases}.issubset({"calibration", "holdout"})
    assert len({case["id"] for case in cases}) == len(cases)
    assert REQUIRED_TAGS.issubset({tag for case in cases for tag in case["tags"]})

    split_counts = {
        split: sum(case["split"] == split for case in cases)
        for split in ("calibration", "holdout")
    }
    assert split_counts["calibration"] >= 12
    assert split_counts["holdout"] >= 12

    for case in cases:
        screenshot = CORPUS.parent / case["screenshot"]
        assert screenshot.is_file(), case["id"]
        assert screenshot.stat().st_size > 100, case["id"]
        width = case["viewport"]["width"]
        height = case["viewport"]["height"]
        assert width > 0 and height > 0
        assert len(case["elements"]) >= 3
        assert max(element["relevance"] for element in case["elements"]) == 3
        assert any(element["relevance"] == 0 for element in case["elements"])
        assert len({element["id"] for element in case["elements"]}) == len(
            case["elements"]
        )
        for element in case["elements"]:
            assert element["role"] in {
                "button",
                "checkbox",
                "input",
                "link",
                "menu",
                "tab",
                "text",
                "other",
            }
            assert 0 <= element["relevance"] <= 3
            bounds = element["bounds"]
            assert bounds["width"] > 0 and bounds["height"] > 0
            assert 0 <= bounds["x"] < width
            assert 0 <= bounds["y"] < height
            assert bounds["x"] + bounds["width"] <= width
            assert bounds["y"] + bounds["height"] <= height
