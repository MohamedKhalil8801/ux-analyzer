"""Design-pattern detection tests over synthetic snapshots.

Covers the reference-fidelity edges that cost parity in the bench: full-text
matching for numbered steps, the gradient opaque-stop quirk, multi-value
border-radius parsing, negative letter-spacing, and html-surface background
resolution.
"""

from __future__ import annotations

from ux_analyzer.analysis.slop.context import build_context
from ux_analyzer.analysis.slop.patterns import (
    _crushed_tracking,
    _flat_type_hierarchy,
    _gradient_backgrounds,
    _nested_cards,
    _numbered_steps,
    _perma_dark_mode,
    _slop_fonts,
)
from ux_analyzer.analysis.visual.snapshot import snapshot_from_dict


def _node(
    i: int,
    parent: int,
    tag: str,
    cls: str = "",
    text: str = "",
    box=(0.0, 0.0, 100.0, 100.0),
    styles: dict | None = None,
) -> dict:
    x, y, w, h = box
    return {
        "i": i,
        "parent": parent,
        "depth": 0,
        "tag": tag,
        "cls": cls,
        "id": "",
        "text": text,
        "box": {"x": x, "y": y, "w": w, "h": h},
        "styles": styles or {},
    }


def _snap(nodes: list[dict]) -> object:
    return snapshot_from_dict({"rootBox": {"x": 0, "y": 0, "w": 1280, "h": 800}, "nodes": nodes})


def test_slop_fonts_ratio_threshold() -> None:
    nodes = [
        _node(0, -1, "body", box=(0, 0, 1280, 800)),
        _node(1, 0, "div", text="Hello", styles={"font-family": "Inter, sans-serif"}),
        _node(2, 0, "div", text="World", styles={"font-family": "Verdana"}),
        _node(3, 0, "div", text="Third", styles={"font-family": "Inter"}),
    ]
    ctx = build_context(_snap(nodes))
    ev = _slop_fonts(ctx)
    # Body's textContent includes descendants, so it is counted too (reference
    # behavior); 2 of 4 elements use slop fonts -> ratio 0.5, not triggered.
    assert ev["slopCount"] == 2
    assert ev["total"] == 4
    assert ev["ratio"] == 0.5
    assert ev["triggered"] is False
    # Add a fourth Inter element: 3/5 = 0.6 -> triggers.
    nodes.append(
        _node(4, 0, "div", text="Fourth", styles={"font-family": "Inter"})
    )
    ev = _slop_fonts(build_context(_snap(nodes)))
    assert ev["triggered"] is True


def test_gradient_backgrounds_opaque_stop_quirk() -> None:
    """Reference treats a stop whose final channel is 0 as transparent."""
    nodes = [
        _node(0, -1, "body", box=(0, 0, 1280, 800)),
        _node(
            1, 0, "div",
            styles={"background-image": "linear-gradient(rgb(24, 164, 111), rgb(24, 164, 111))"},
        ),
        _node(
            2, 0, "div",
            styles={"background-image": "linear-gradient(rgb(245, 87, 0), rgb(245, 87, 0))"},
        ),
        _node(
            3, 0, "div",
            styles={"background-image": "linear-gradient(rgba(0, 0, 0, 0.1), rgba(0, 0, 0, 0))"},
        ),
        _node(
            4, 0, "div",
            styles={"background-image": "radial-gradient(circle, rgb(255, 159, 255), rgb(255, 159, 255))"},
        ),
    ]
    ctx = build_context(_snap(nodes))
    ev = _gradient_backgrounds(ctx)
    # rgb(245,87,0) reads transparent (blue channel 0); the rgba stack counts
    # because its 0.1 alpha stop is above the 0.05 floor. Green + rgba + pink.
    assert ev["bgElements"] == 3


def test_numbered_steps_matches_full_text() -> None:
    nodes = [
        _node(0, -1, "body", box=(0, 0, 1280, 800)),
        _node(1, 0, "div", cls="steps"),
        _node(2, 1, "span", text="1"),
        _node(3, 1, "button", text="", box=(0, 0, 100, 30)),
        _node(4, 1, "span", text="2"),
        _node(5, 1, "span", text="3"),
        _node(6, 1, "span", text="4"),
    ]
    ctx = build_context(_snap(nodes))
    ev = _numbered_steps(ctx)
    assert ev["bestRun"] == 4
    assert ev["triggered"] is True


def test_crushed_tracking_negative_letter_spacing() -> None:
    nodes = [
        _node(0, -1, "body", box=(0, 0, 1280, 800)),
        _node(
            1, 0, "h1", text="Agentic Infrastructure",
            styles={"font-size": "64px", "letter-spacing": "-3.84px"},
        ),
    ]
    ctx = build_context(_snap(nodes))
    ev = _crushed_tracking(ctx)
    assert ev["count"] == 1
    assert ev["triggered"] is True


def test_multi_value_border_radius_parses_leading_value() -> None:
    """nested_cards depends on parseFloat semantics for '16px 16px 0px 0px'."""
    nodes = [
        _node(0, -1, "body", box=(0, 0, 1280, 800)),
        # outer card: full class carries "border" token, radius is multi-value
        _node(
            1, 0, "div", cls="relative overflow-hidden rounded-2xl rounded-b-none border",
            box=(16, 67, 1248, 748),
            styles={
                "border-radius": "16px 16px 0px 0px",
                "background-color": "rgba(0, 0, 0, 0)",
                "box-shadow": "none",
            },
        ),
        # inner chat prompt card
        _node(
            2, 1, "div", cls="bg-chat-prompt-bg rounded-3xl border",
            box=(320, 332, 640, 141),
            styles={
                "border-radius": "24px",
                "border-top-width": "1px",
                "background-color": "rgb(30, 30, 33)",
                "box-shadow": "rgba(0, 0, 0, 0) 0px 0px 0px 0px",
            },
        ),
        _node(3, 2, "div", text="Type your prompt here"),
        _node(4, 2, "div", text="More prompt content"),
    ]
    ctx = build_context(_snap(nodes))
    ev = _nested_cards(ctx)
    assert ev["nested"] == 1
    assert ev["triggered"] is False  # one nested card is legitimate per reference


def test_perma_dark_mode_html_surface() -> None:
    nodes = [
        _node(0, -1, "body", box=(0, 0, 1280, 800), styles={"background-color": "rgba(0, 0, 0, 0)"}),
        _node(1, 0, "p", text="Body copy with enough text to count."),
        _node(2, 0, "p", text="Another paragraph of reading text."),
        _node(3, 0, "p", text="A third paragraph with more words."),
        _node(4, 0, "p", text="Fourth paragraph of body text here."),
        _node(5, 0, "p", text="Fifth paragraph to cross the threshold."),
    ]
    for n in nodes:
        n["styles"].setdefault("color", "rgb(207, 207, 207)")
    ctx = build_context(
        _snap(nodes), surface={"htmlBg": "rgb(33, 33, 33)", "bodyBg": "rgba(0, 0, 0, 0)"}
    )
    ev = _perma_dark_mode(ctx)
    assert ev["bodyDark"] is True
    assert ev["triggered"] is True


def test_flat_type_hierarchy_ratio() -> None:
    nodes = [
        _node(0, -1, "body", box=(0, 0, 1280, 800)),
        _node(1, 0, "p", text="aa", styles={"font-size": "9px"}),
        _node(2, 0, "p", text="bb", styles={"font-size": "11px"}),
        _node(3, 0, "p", text="cc", styles={"font-size": "13px"}),
    ]
    ctx = build_context(_snap(nodes))
    ev = _flat_type_hierarchy(ctx)
    assert ev["distinct"] == 3
    assert ev["triggered"] is True
