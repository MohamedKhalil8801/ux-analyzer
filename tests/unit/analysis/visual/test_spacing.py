"""Unit tests for the spacing/white-space visual detector."""

from __future__ import annotations

from ux_analyzer.analysis.visual.snapshot import Box, Snapshot, SNode
from ux_analyzer.analysis.visual.spacing import analyze_spacing
from ux_analyzer.analysis.visual.types import VisualIssue

_LONG = (
    "This paragraph carries enough copy to count as a real multi-line "
    "content block in the rendered layout tree."
)


def _node(
    i: int,
    parent: int,
    depth: int,
    tag: str,
    *,
    cls: str = "",
    text: str = "",
    box: tuple[float, float, float, float] = (0.0, 0.0, 100.0, 100.0),
    **styles: str,
) -> SNode:
    base: dict[str, str] = {
        "display": "block",
        "position": "static",
        "opacity": "1",
        "background-color": "rgba(0, 0, 0, 0)",
        "font-size": "16px",
        "line-height": "24px",
    }
    for side in ("top", "right", "bottom", "left"):
        base[f"margin-{side}"] = "0px"
        base[f"padding-{side}"] = "0px"
        base[f"border-{side}-width"] = "0px"
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


def _card_surface(**extra: str) -> dict[str, str]:
    surface = {
        "background-color": "rgb(255, 255, 255)",
        "border-top-width": "1px",
        "border-right-width": "1px",
        "border-bottom-width": "1px",
        "border-left-width": "1px",
    }
    surface.update(extra)
    return surface


def _check_ids(issues: list[VisualIssue]) -> set[str]:
    return {issue.check_id for issue in issues}


def test_cramped_card_fires_but_well_padded_twin_does_not() -> None:
    cramped = _snapshot(
        [
            _node(0, -1, 0, "section", box=(0, 0, 600, 400)),
            _node(
                1,
                0,
                1,
                "div",
                cls="card",
                box=(50, 50, 300, 200),
                **_card_surface(),
            ),
            _node(
                2, 1, 2, "p", text=_LONG, box=(54, 54, 292, 48)
            ),  # 4px inset
            _node(3, 1, 2, "p", text=_LONG, box=(54, 118, 292, 48)),
        ]
    )
    issues = list(analyze_spacing(cramped))
    assert "spacing.cramped-container" in _check_ids(issues)
    assert all(issue.severity == "medium" for issue in issues)

    padded = _snapshot(
        [
            _node(0, -1, 0, "section", box=(0, 0, 600, 400)),
            _node(
                1,
                0,
                1,
                "div",
                cls="card",
                box=(50, 50, 300, 200),
                **_card_surface(),
            ),
            _node(2, 1, 2, "p", text=_LONG, box=(74, 74, 252, 48)),  # 24px
            _node(3, 1, 2, "p", text=_LONG, box=(74, 138, 252, 48)),
        ]
    )
    assert list(analyze_spacing(padded)) == []


def test_edge_touching_card_is_critical() -> None:
    snap = _snapshot(
        [
            _node(0, -1, 0, "section", box=(0, 0, 600, 400)),
            _node(
                1,
                0,
                1,
                "div",
                cls="card",
                box=(50, 50, 300, 200),
                **_card_surface(),
            ),
            _node(2, 1, 2, "p", text=_LONG, box=(51, 60, 290, 48)),  # 1px
            _node(3, 1, 2, "p", text=_LONG, box=(60, 124, 280, 48)),
        ]
    )
    issues = [i for i in analyze_spacing(snap) if i.severity == "critical"]
    assert "spacing.edge-touching" in _check_ids(issues)


def test_inconsistent_sibling_gaps_fire_uniform_do_not() -> None:
    def rows(gaps: list[float]) -> Snapshot:
        nodes = [_node(0, -1, 0, "ul", box=(0, 0, 640, 800))]
        y = 0.0
        for k in range(len(gaps) + 1):
            if k:
                y += 40.0 + gaps[k - 1]
            nodes.append(
                _node(
                    len(nodes),
                    0,
                    1,
                    "li",
                    cls="item",
                    text=f"List item number {k} with some detail text",
                    box=(16, y, 600, 40),
                )
            )
        return _snapshot(nodes)

    erratic = list(analyze_spacing(rows([4.0, 44.0, 4.0])))
    assert "spacing.erratic-sibling-gaps" in _check_ids(erratic)
    rhythm = [i for i in erratic if i.check_id == "spacing.erratic-sibling-gaps"]
    assert all(i.severity == "medium" for i in rhythm)
    assert rhythm[0].evidence["gapsPx"] == [4.0, 44.0, 4.0]

    assert list(analyze_spacing(rows([16.0, 16.0, 16.0]))) == []


def test_zero_gap_sections_is_critical() -> None:
    snap = _snapshot(
        [
            _node(0, -1, 0, "main", box=(0, 0, 700, 700)),
            _node(1, 0, 1, "section", text=_LONG + _LONG, box=(20, 20, 640, 120)),
            _node(
                2, 0, 1, "section", text=_LONG + _LONG, box=(20, 140, 640, 120)
            ),  # 0px gap below node 1
        ]
    )
    issues = [
        i
        for i in analyze_spacing(snap)
        if i.check_id == "spacing.zero-gap-sections"
    ]
    assert len(issues) == 1
    assert issues[0].severity == "critical"


def test_touching_containers_with_inner_padding_do_not_fire() -> None:
    # Boxes meet at y=140 but each section's own padding keeps the rendered
    # text masses 56px apart, so there is no white-space problem.
    snap = _snapshot(
        [
            _node(0, -1, 0, "div", box=(0, 0, 700, 500)),
            _node(
                1,
                0,
                1,
                "section",
                **{"padding-bottom": "32px"},
                box=(20, 20, 640, 120),
            ),
            _node(2, 1, 2, "p", text=_LONG, box=(40, 30, 600, 48)),
            _node(
                3,
                0,
                1,
                "section",
                **{"padding-top": "24px"},
                box=(20, 140, 640, 120),
            ),
            _node(4, 3, 2, "p", text=_LONG, box=(40, 164, 600, 48)),
        ]
    )
    assert list(analyze_spacing(snap)) == []


def test_navigation_region_is_ignored() -> None:
    # Menu links flush to the menu surface and stacked tightly are normal
    # inside nav/header landmarks; nothing here should be flagged.
    nodes = [
        _node(0, -1, 0, "header", box=(0, 0, 800, 220)),
        _node(1, 0, 1, "div", cls="bar", box=(0, 0, 800, 220), **_card_surface()),
        _node(2, 1, 2, "nav", box=(20, 10, 300, 200)),
        _node(3, 2, 3, "ul", cls="menu", box=(20, 80, 300, 100), **_card_surface()),
    ]
    for k in range(3):
        li_y = 82.0 + k * 32.0
        nodes.append(
            _node(len(nodes), 3, 4, "li", box=(20, li_y, 296, 28))
        )
        nodes.append(
            _node(len(nodes), len(nodes) - 1, 5, "a", text=f"Item {k}", box=(20, li_y, 120, 24))
        )  # flush with the menu's left edge
    assert list(analyze_spacing(_snapshot(nodes))) == []


def test_text_far_outside_surface_is_overlay_not_cramped() -> None:
    # A child escaping its container far upward reads as a dropdown/badge,
    # not as cramped padding; the detector must stay quiet.
    snap = _snapshot(
        [
            _node(0, -1, 0, "section", box=(0, 0, 600, 400)),
            _node(
                1,
                0,
                1,
                "div",
                cls="card",
                box=(50, 150, 300, 200),
                **_card_surface(),
            ),
            _node(2, 1, 2, "p", text=_LONG, box=(60, -60, 280, 96)),  # escapes up
            _node(3, 1, 2, "p", text=_LONG, box=(74, 190, 252, 48)),
        ]
    )
    issues = list(analyze_spacing(snap))
    assert not _check_ids(issues) & {"spacing.edge-touching", "spacing.cramped-container"}


def test_packed_records_fire() -> None:
    nodes = [_node(0, -1, 0, "div", box=(0, 0, 640, 500))]
    for k in range(6):
        nodes.append(
            _node(
                len(nodes),
                0,
                1,
                "div",
                cls="row",
                text=_LONG[:60] + f" record {k}",
                box=(8, 8 + k * 36, 600, 32),  # 4px gaps, no dividers
            )
        )
    issues = [
        i for i in analyze_spacing(_snapshot(nodes))
        if i.check_id == "spacing.packed-records"
    ]
    assert len(issues) == 1
    assert issues[0].severity == "medium"
    assert issues[0].evidence["records"] == 6


def test_clean_simple_design_produces_zero_issues() -> None:
    snap = _snapshot(
        [
            # page canvas (no painted surface of its own)
            _node(0, -1, 0, "section", box=(0, 0, 1000, 760)),
            # heading well above following copy
            _node(1, 0, 1, "h1", text="Quarterly report", box=(40, 24, 600, 32)),
            # intro copy: 32px below heading
            _node(2, 0, 1, "p", text=_LONG, box=(40, 88, 700, 48)),
            # a properly padded card with two relaxed paragraphs
            _node(
                3,
                0,
                1,
                "div",
                cls="card",
                box=(40, 184, 420, 240),
                **_card_surface(),
            ),
            _node(4, 3, 2, "p", text=_LONG, box=(64, 208, 372, 48)),
            _node(5, 3, 2, "p", text=_LONG, box=(64, 272, 372, 48)),
            # a plain list with an even rhythm
            _node(6, 0, 1, "ul", cls="list", box=(520, 184, 420, 240)),
            _node(7, 6, 2, "li", cls="row", text="First row of the list", box=(536, 200, 388, 40)),
            _node(8, 6, 2, "li", cls="row", text="Second row of the list", box=(536, 264, 388, 40)),
            _node(9, 6, 2, "li", cls="row", text="Third row of the list", box=(536, 328, 388, 40)),
        ]
    )
    assert list(analyze_spacing(snap)) == []
