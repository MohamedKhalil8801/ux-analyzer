"""Renderer slop-pass-through tests: ux-audit.json -> report context."""

from __future__ import annotations

import json

from ux_analyzer.reporting.renderer import _load_ux_audit


def _audit_payload(slop: dict | None) -> dict:
    return {
        "schema_version": "ux-audit-v1",
        "total_issues": 1,
        "urls": [
            {
                "url": "https://site.test/",
                "total": 1,
                "issues": [
                    {
                        "category": "visual",
                        "check_id": "spacing",
                        "title": "Tight spacing",
                        "severity": "medium",
                        "evidence": {"gap": "4px"},
                    }
                ],
                "slop": slop,
            }
        ],
        "errors": [],
    }


def test_load_ux_audit_passes_slop_through(tmp_path) -> None:
    slop = {
        "score": 31,
        "tier": "Heavy",
        "grade": "D+",
        "verdict": "Heavy slop.",
        "patternsFlagged": 1,
        "patternsTotal": 27,
        "patterns": [
            {
                "id": "slop_fonts",
                "label": "AI-default font stack (Inter / Geist / Space Grotesk)",
                "short": "Slop fonts",
                "category": "fonts",
                "weight": 8,
                "triggered": True,
                "evidence": {"ratio": 0.985, "heroIsSlop": True},
            },
            {
                "id": "cream_default_bg",
                "label": "Cream / beige default page background",
                "short": "Cream bg",
                "category": "colors",
                "weight": 7,
                "triggered": False,
                "evidence": {"surface": "#171719"},
            },
        ],
        "copy": {
            "score": 5,
            "tier": "Clean",
            "grade": "A",
            "patternsFlagged": 1,
            "patternsTotal": 9,
            "patterns": [
                {
                    "id": "em_dash_overload",
                    "label": "Em-dash overload (— used as the default connective)",
                    "short": "Em-dashes",
                    "category": "copy",
                    "weight": 5,
                    "triggered": True,
                    "evidence": {"total": 5},
                }
            ],
        },
        "unifiedScore": 31,
        "unifiedTier": "Heavy",
    }
    audit_file = tmp_path / "ux-audit.json"
    audit_file.write_text(json.dumps(_audit_payload(slop)), encoding="utf-8")
    loaded = _load_ux_audit(tmp_path)
    assert loaded is not None
    url_report = loaded["url_reports"][0]
    assert url_report["slop"]["score"] == 31
    assert url_report["slop"]["tier"] == "Heavy"
    assert url_report["slop"]["patternsFlagged"] == 1
    assert len(url_report["slop"]["patterns"]) == 2  # all rows pass through
    assert url_report["slop"]["copy"]["score"] == 5
    assert url_report["slop"]["unifiedScore"] == 31


def test_load_ux_audit_omits_malformed_slop(tmp_path) -> None:
    audit_file = tmp_path / "ux-audit.json"
    audit_file.write_text(
        json.dumps(_audit_payload({"score": "not-an-int", "tier": "Heavy"})),
        encoding="utf-8",
    )
    loaded = _load_ux_audit(tmp_path)
    assert loaded is not None
    assert loaded["url_reports"][0]["slop"] is None
