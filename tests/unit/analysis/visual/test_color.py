"""Unit tests for the color & contrast visual detector."""

from __future__ import annotations

from ux_analyzer.analysis.visual.color import analyze_color
from ux_analyzer.analysis.visual.snapshot import Box, Snapshot, SNode


def _node(
    i: int,
    parent: int,
    depth: int,
    tag: str,
    *,
    cls: str = "",
    text: str = "",
    box: tuple[float, float, float, float] = (0.0, 0.0, 200.0, 60.0),
    **styles: str,
) -> SNode:
    base: dict[str, str] = {
        "display": "block",
        "position": "static",
        "opacity": "1",
        "visibility": "visible",
        "background-color": "rgba(0, 0, 0, 0)",
        "color": "rgb(0, 0, 0)",
        "font-size": "16px",
        "font-weight": "normal",
        "font-style": "normal",
        "text-decoration-line": "none",
    }
    base.update(styles)
    x, y, w, h = box
    return SNode(
        i=i,
        parent=parent,
        depth=depth,
        tag=tag,
        cls=cls,
        id="",
        text=text,
        box=Box(x=x, y=y, w=w, h=h),
        styles=base,
    )


def _snapshot(nodes: list[SNode]) -> Snapshot:
    return Snapshot(root_box=Box(x=0, y=0, w=1280, h=900), nodes=tuple(nodes))


def test_777_on_white_fails_with_boundary_math() -> None:
    snap = _snapshot(
        [
            _node(
                0, -1, 0, "section",
                **{"background-color": "rgb(255, 255, 255)"},
            ),
            _node(
                1, 0, 1, "p",
                text="Payment declined recently",
                **{"color": "rgb(119, 119, 119)"},
            ),
        ]
    )
    issues = list(analyze_color(snap))
    assert [issue.check_id for issue in issues] == [
        "contrast.below-threshold"
    ]
    assert issues[0].severity == "medium"
    assert issues[0].fundamental == "contrast"
    assert issues[0].evidence["min_ratio"] == 4.48  # 1.05 / (L(#777)+0.05)
    pair = issues[0].evidence["pairs"][0]
    assert pair["required"] == 4.5
    assert pair["background"] == "rgb(255, 255, 255)"


def test_333_on_white_passes() -> None:
    snap = _snapshot(
        [
            _node(
                0, -1, 0, "section",
                **{"background-color": "rgb(255, 255, 255)"},
            ),
            _node(
                1, 0, 1, "p",
                text="Dark readable copy on white",
                **{"color": "rgb(51, 51, 51)"},
            ),
        ]
    )
    assert list(analyze_color(snap)) == []


def test_rgba_blend_over_dark_ancestor_matches_hand_math() -> None:
    # backdrop = blend(white .5 over rgb(20,20,20)) = rgb(137.5) each channel;
    # rendered text = blend(black .5 over backdrop) = rgb(68.75);
    # ratio = (L(137.5)+0.05)/(L(68.75)+0.05) ~= 2.77 -> critical.
    snap = _snapshot(
        [
            _node(
                0, -1, 0, "main",
                **{"background-color": "rgb(20, 20, 20)"},
            ),
            _node(
                1, 0, 1, "section",
                **{"background-color": "rgba(255, 255, 255, 0.5)"},
            ),
            _node(
                2, 1, 2, "p",
                text="Status line copy",
                **{"color": "rgba(0, 0, 0, 0.5)"},
            ),
        ]
    )
    issues = list(analyze_color(snap))
    assert [issue.check_id for issue in issues] == [
        "contrast.below-threshold"
    ]
    assert issues[0].severity == "critical"
    assert issues[0].evidence["min_ratio"] == 2.77
    pair = issues[0].evidence["pairs"][0]
    assert pair["background"] == "rgb(138, 138, 138)"
    assert pair["foreground"] == "rgba(0, 0, 0, 0.5)"


def test_large_text_thresholds_use_3_to_1() -> None:
    def page(size: str, weight: str) -> Snapshot:
        return _snapshot(
            [
                _node(
                    0, -1, 0, "section",
                    **{"background-color": "rgb(255, 255, 255)"},
                ),
                _node(
                    1, 0, 1, "h2",
                    text="Big display heading",
                    **{
                        "color": "rgb(119, 119, 119)",
                        "font-size": size,
                        "font-weight": weight,
                    },
                ),
            ]
        )

    # 24px counts as large text: 4.48:1 clears the 3:1 bar.
    assert list(analyze_color(page("24px", "normal"))) == []
    # 19px bold also counts as large text.
    assert list(analyze_color(page("19px", "bold"))) == []
    # 23px regular is NOT large: 4.48:1 misses 4.5:1 -> medium failure.
    issues = list(analyze_color(page("23px", "normal")))
    assert [issue.check_id for issue in issues] == ["contrast.below-threshold"]
    assert issues[0].severity == "medium"
    assert issues[0].evidence["pairs"][0]["required"] == 4.5


def test_accent_chaos_fires_at_five_saturated_hues() -> None:
    hues = [
        ("rgb(220, 40, 40)", 0.0),
        ("rgb(40, 200, 60)", 127.5),
        ("rgb(40, 200, 200)", 180.0),
        ("rgb(60, 60, 220)", 240.0),
        ("rgb(200, 60, 220)", 292.5),
    ]
    nodes = [_node(0, -1, 0, "main")]
    for k, (color, _) in enumerate(hues):
        nodes.append(
            _node(
                len(nodes), 0, 1, "div",
                cls="chip",
                box=(20.0 + k * 120, 20, 100, 40),
                **{"background-color": color},
            )
        )
    issues = list(analyze_color(_snapshot(nodes)))
    assert [issue.check_id for issue in issues] == ["color.accent-hue-chaos"]
    assert issues[0].severity == "medium"
    assert issues[0].fundamental == "color"
    assert len(issues[0].element_refs) == 5

    harmonious = ["rgb(30, 80, 220)", "rgb(40, 90, 230)", "rgb(50, 100, 240)"]
    nodes = [_node(0, -1, 0, "main")]
    for k, color in enumerate(harmonious):
        nodes.append(
            _node(
                len(nodes), 0, 1, "div",
                cls="chip",
                box=(20.0 + k * 120, 20, 100, 40),
                **{"background-color": color},
            )
        )
    assert list(analyze_color(_snapshot(nodes))) == []


def test_red_green_state_siblings_fire_low() -> None:
    def statuses(fail_color: str) -> Snapshot:
        return _snapshot(
            [
                _node(0, -1, 0, "div", cls="status-bar"),
                _node(
                    1, 0, 1, "span",
                    cls="state-ok",
                    text="Active",
                    **{"color": "rgb(0, 120, 60)"},
                ),
                _node(
                    2, 0, 1, "span",
                    cls="state-bad",
                    text="Failed",
                    **{"color": fail_color},
                ),
            ]
        )

    issues = list(analyze_color(statuses("rgb(200, 40, 40)")))
    assert [issue.check_id for issue in issues] == ["color.state-by-hue-alone"]
    assert issues[0].severity == "low"
    assert issues[0].fundamental == "color"

    # Amber carries no red/green-only signal even if other detectors differ.
    assert not any(
        issue.check_id == "color.state-by-hue-alone"
        for issue in analyze_color(statuses("rgb(180, 120, 0)"))
    )


def test_size_or_weight_difference_blocks_state_check() -> None:
    snap = _snapshot(
        [
            _node(0, -1, 0, "div"),
            _node(
                1, 0, 1, "span",
                text="Active",
                **{"color": "rgb(0, 120, 60)", "font-size": "16px"},
            ),
            _node(
                2, 0, 1, "span",
                text="Failed",
                **{"color": "rgb(200, 40, 40)", "font-size": "20px"},
            ),
        ]
    )
    assert list(analyze_color(snap)) == []


def test_near_duplicate_grays_fire_low() -> None:
    def labels(a: str, b: str) -> Snapshot:
        return _snapshot(
            [
                _node(0, -1, 0, "div", cls="meta"),
                _node(1, 0, 1, "span", text="Created", **{"color": a}),
                _node(2, 0, 1, "span", text="Updated", **{"color": b}),
            ]
        )

    issues = list(analyze_color(labels("rgb(117, 117, 117)",
                                       "rgb(118, 118, 118)")))
    assert [issue.check_id for issue in issues] == [
        "color.near-duplicate-grays"
    ]
    assert issues[0].severity == "low"
    assert issues[0].fundamental == "color"
    assert len(issues[0].element_refs) == 2

    # A real gray ramp (clear luminance steps, both AA-compliant) stays quiet.
    assert list(analyze_color(labels("rgb(51, 51, 51)",
                                     "rgb(118, 118, 118)"))) == []


def test_opacity_composites_into_effective_colors() -> None:
    # Wrapper at opacity .5 halves the text alpha too: rendered fg becomes
    # blend(#777 .5 over white) = rgb(187); ratio ~= 1.92 -> critical.
    snap = _snapshot(
        [
            _node(
                0, -1, 0, "section",
                **{"background-color": "rgb(255, 255, 255)"},
            ),
            _node(1, 0, 1, "div", **{"opacity": "0.5"}),
            _node(
                2, 1, 2, "p",
                text="Faded overlay copy here",
                **{"color": "rgb(119, 119, 119)"},
            ),
        ]
    )
    issues = list(analyze_color(snap))
    assert [issue.check_id for issue in issues] == ["contrast.below-threshold"]
    assert issues[0].severity == "critical"
    assert issues[0].evidence["min_ratio"] == 1.92


def test_fully_transparent_text_is_skipped() -> None:
    snap = _snapshot(
        [
            _node(0, -1, 0, "section"),
            _node(
                1, 0, 1, "p",
                text="Invisible spacer text",
                **{"color": "rgba(0, 0, 0, 0)"},
            ),
        ]
    )
    assert list(analyze_color(snap)) == []


def test_clean_design_produces_zero_issues() -> None:
    snap = _snapshot(
        [
            _node(
                0, -1, 0, "section",
                box=(0, 0, 900, 600),
                **{"background-color": "rgb(255, 255, 255)"},
            ),
            _node(
                1, 0, 1, "h1",
                text="Billing overview",
                box=(40, 32, 500, 36),
                **{"color": "rgb(17, 17, 17)", "font-size": "28px"},
            ),
            _node(
                2, 0, 1, "p",
                text="Your plan renews on the first of every month.",
                box=(40, 96, 640, 48),
                **{"color": "rgb(51, 51, 51)"},
            ),
            _node(
                3, 0, 1, "p",
                text="Invoices download automatically after each cycle.",
                box=(40, 160, 640, 48),
                **{"color": "rgb(51, 51, 51)"},
            ),
            _node(4, 0, 1, "div", cls="card", box=(40, 240, 400, 160)),
            _node(
                5, 4, 2, "span",
                text="Last charge: August 1",
                box=(64, 264, 320, 24),
                **{"color": "rgb(118, 118, 118)"},
            ),
            _node(
                6, 0, 1, "button",
                box=(480, 250, 180, 48),
                **{"background-color": "rgb(0, 90, 200)"},
            ),
            _node(
                7, 6, 2, "span",
                text="Download invoice",
                box=(496, 264, 148, 24),
                **{"color": "rgb(255, 255, 255)"},
            ),
        ]
    )
    assert list(analyze_color(snap)) == []
