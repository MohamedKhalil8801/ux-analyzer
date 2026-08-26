"""Color parsing + classification (reference-color parity tests)."""

from __future__ import annotations

from ux_analyzer.analysis.slop.color import (
    channel_spread,
    contrast_ratio,
    is_dark,
    is_mid_grey,
    is_neutral,
    is_purple,
    parse_color,
    relative_luminance,
    rgb_to_hsl,
)


def test_parse_color_rgb_and_hex() -> None:
    assert parse_color("rgb(255, 0, 0)") is not None
    c = parse_color("rgb(24, 164, 111)")
    assert c is not None and (c.r, c.g, c.b) == (24, 164, 111) and c.a == 1.0
    c = parse_color("#f6f6ef")
    assert c is not None and (c.r, c.g, c.b) == (246, 246, 239)
    c = parse_color("rgba(1, 2, 3, 0.5)")
    assert c is not None and abs(c.a - 0.5) < 1e-9


def test_parse_color_transparent_and_invalid() -> None:
    assert parse_color(None) is None
    assert parse_color("transparent") is None
    assert parse_color("none") is None
    assert parse_color("rgba(0, 0, 0, 0)") is None
    assert parse_color("") is None
    assert parse_color("rgb(0, 0, 0, 0)") is None


def test_is_purple_vibe_zone() -> None:
    # Tailwind indigo-600-ish: rgb(79, 70, 229)
    assert is_purple(parse_color("rgb(79, 70, 229)"))
    # Red is not purple
    assert not is_purple(parse_color("rgb(255, 0, 0)"))
    # Washed-out purple (low saturation) is not VibeCode purple
    assert not is_purple(parse_color("rgb(150, 145, 165)"))
    # Near-transparent purple does not count
    assert not is_purple(parse_color("rgba(79, 70, 229, 0.1)"))


def test_is_dark_and_mid_grey() -> None:
    assert is_dark(parse_color("rgb(17, 17, 20)"))
    assert not is_dark(parse_color("rgb(255, 255, 255)"))
    assert is_mid_grey(parse_color("rgb(130, 130, 130)"))
    assert not is_mid_grey(parse_color("rgb(255, 255, 255)"))
    assert not is_mid_grey(parse_color("rgb(79, 70, 229)"))


def test_contrast_and_luminance() -> None:
    white = parse_color("#ffffff")
    black = parse_color("#000000")
    assert white is not None and black is not None
    assert abs(contrast_ratio(white, black) - 21.0) < 1e-9
    assert abs(contrast_ratio(black, black) - 1.0) < 1e-9
    assert relative_luminance(black) == 0.0
    assert abs(relative_luminance(white) - 1.0) < 1e-9


def test_hsl_and_spread_helpers() -> None:
    c = parse_color("rgb(255, 0, 0)")
    hsl = rgb_to_hsl(c)
    assert hsl is not None and abs(hsl.h - 0.0) < 1e-6 and hsl.s > 0.99
    assert channel_spread(parse_color("rgb(130, 130, 130)")) < 1
    assert channel_spread(parse_color("rgb(255, 0, 0)")) == 255
    assert is_neutral(parse_color("rgb(128, 128, 128)"))
    assert not is_neutral(parse_color("rgb(79, 70, 229)"))
