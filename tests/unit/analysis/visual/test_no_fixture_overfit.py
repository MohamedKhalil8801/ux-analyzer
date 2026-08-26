"""Guard against benchmark overfitting in visual detectors.

The UEye fixtures are named after real challenge slugs and contain
challenge-specific text. Detectors must generalize, so this test fails if
detector source references fixture slugs or challenge-specific strings.
"""

from __future__ import annotations

import pathlib

VISUAL_DIR = pathlib.Path(__file__).resolve().parents[3] / "src" / "ux_analyzer" / "analysis" / "visual"

FORBIDDEN = [
    # fixture slugs
    "activity-feed", "feature-blip", "simple-login-form", "dropdown-menu",
    "url-shortener", "mobile-app-onboarding", "ueye-footer-design",
    "reporting-statistics", "user-testimonials", "pricing-comparison",
    "sports-stats", "food-ingredients", "toggle-settings", "news-listings",
    "user-support-popup",
    # challenge-specific content strings
    "cryptocompany", "james opened", "alisa created", "kirk canceled",
    "claire failed", "brian reached",
]


def test_detectors_do_not_reference_fixtures() -> None:
    offenders: list[str] = []
    for path in sorted(VISUAL_DIR.glob("*.py")):
        src = path.read_text(encoding="utf-8").lower()
        for needle in FORBIDDEN:
            if needle in src:
                offenders.append(f"{path.name}: {needle}")
    assert not offenders, f"detector source references benchmark fixtures: {offenders}"
