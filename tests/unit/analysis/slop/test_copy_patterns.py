"""Copy-axis pattern tests (reference copyPatterns.ts parity)."""

from __future__ import annotations

from ux_analyzer.analysis.slop.copy_patterns import COPY_PATTERNS, run_copy_patterns


def _match(pid: str, ctx: dict):
    return next(p["match"] for p in COPY_PATTERNS if p["id"] == pid)(ctx)


def test_buzzword_density_breadth_and_density() -> None:
    ctx = {"text": "We leverage AI for our teams. It is a robust solution.", "wordCount": 40}
    ev = _match("buzzword_density", ctx)
    assert ev["triggered"] is False  # 2 distinct, total 2 below the breadth bar
    assert ev["distinct"] == 2
    dense = {
        "text": "Leverage. Leverage. Leverage. Leverage. Leverage. Seamless. Seamless. Robust. Unlock.",
        "wordCount": 40,
    }
    assert _match("buzzword_density", dense)["triggered"] is True  # distinct >= 4


def test_em_dash_overload_thresholds() -> None:
    light = {"text": "Hello — world. Fine — writing.", "wordCount": 40}
    assert _match("em_dash_overload", light)["triggered"] is False
    heavy = {"text": "a — b — c — d — e — f — g — h — i", "wordCount": 60}
    assert _match("em_dash_overload", heavy)["triggered"] is True  # 8+ em-dashes


def test_antithesis_requires_two() -> None:
    one = {"text": "More than just a tool.", "wordCount": 100}
    assert _match("antithesis_construction", one)["triggered"] is False
    # Note: the reference double-counts "It's not just a tool." (both "it's not
    # just" and "not just a" patterns fire), so it triggers on a single phrase.
    double = {"text": "It's not just a tool.", "wordCount": 100}
    assert _match("antithesis_construction", double)["triggered"] is True
    two = {"text": "It's not just a tool. It is not only fast but also reliable.", "wordCount": 100}
    assert _match("antithesis_construction", two)["triggered"] is True


def test_filler_openers_single_fires() -> None:
    ctx = {"text": "In today's fast-paced world, teams need better tools.", "wordCount": 100}
    assert _match("filler_openers", ctx)["triggered"] is True


def test_rule_of_three_tricolon() -> None:
    two = {"text": "fast, reliable, and scalable. simple, clear, and honest.", "wordCount": 100}
    assert _match("rule_of_three", two)["triggered"] is False  # needs >= 3
    three = {"text": "fast, reliable, and scalable. simple, clear, and honest. quick, smooth, and friendly.", "wordCount": 100}
    assert _match("rule_of_three", three)["triggered"] is True


def test_unicode_artifacts_zero_width() -> None:
    ctx = {"text": "invisible\u200Bspace in copy", "wordCount": 100}
    assert _match("unicode_artifacts", ctx)["triggered"] is True


def test_emoji_bullets_need_three_lines() -> None:
    ctx = {"headings": ["🚀 Launch", "✅ Verified", "✨ New"], "paragraphs": [], "wordCount": 100}
    assert _match("emoji_bullet_headers", ctx)["triggered"] is True
    ctx = {"headings": ["🚀 Launch"], "paragraphs": [], "wordCount": 100}
    assert _match("emoji_bullet_headers", ctx)["triggered"] is False


def test_run_copy_patterns_shape_and_thin_suppression() -> None:
    rows = run_copy_patterns(
        {"text": "x", "headings": [], "paragraphs": [], "wordCount": 10}
    )
    assert len(rows) == 9
    assert all(not r["triggered"] for r in rows)  # thin page: all suppressed
