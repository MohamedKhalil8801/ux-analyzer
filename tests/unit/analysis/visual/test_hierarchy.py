"""Unit tests for the hierarchy & alignment visual detector."""

from __future__ import annotations

from typing import Any

from ux_analyzer.analysis.visual.hierarchy import analyze_hierarchy
from ux_analyzer.analysis.visual.snapshot import Box, Snapshot, SNode

_BODY = {
    "font-size": "16px",
    "font-weight": "400",
    "color": "rgb(17, 24, 39)",
    "display": "block",
    "position": "static",
}


def _node(
    i: int,
    parent: int,
    depth: int,
    tag: str,
    *,
    text: str = "",
    x: float = 0,
    y: float = 0,
    w: float = 400,
    h: float = 30,
    styles: dict[str, str] | None = None,
) -> SNode:
    merged: dict[str, Any] = dict(_BODY)
    if styles:
        merged.update(styles)
    return SNode(
        i=i,
        parent=parent,
        depth=depth,
        tag=tag,
        cls="",
        id="",
        text=text,
        box=Box(x=x, y=y, w=w, h=h),
        styles={k: v for k, v in merged.items() if v != ""},
    )


def _snap(nodes: list[SNode]) -> Snapshot:
    return Snapshot(
        root_box=Box(x=0, y=0, w=800, h=600), nodes=tuple(nodes)
    )


def _checks(issues: list[Any]) -> set[str]:
    return {i.check_id for i in issues}


def test_drifting_sibling_rows_fire_alignment() -> None:
    nodes = [_node(0, -1, 0, "main")]
    xs = [100.0, 113.0, 126.0]
    for k, x in enumerate(xs):
        nodes.append(
            _node(
                len(nodes),
                0,
                1,
                "div",
                text=f"Row {k} with plenty of content",
                x=x,
                y=100 + k * 40,
            )
        )
    issues = list(analyze_hierarchy(_snap(nodes)))
    checks = _checks(issues)
    assert "alignment.edge-drift" in checks
    drift = next(i for i in issues if i.check_id == "alignment.edge-drift")
    assert drift.fundamental == "alignment"
    assert len(drift.element_refs) == 2  # the two offset rows


def test_aligned_rows_produce_no_alignment_issue() -> None:
    nodes = [_node(0, -1, 0, "main")]
    for k in range(3):
        nodes.append(
            _node(
                len(nodes),
                0,
                1,
                "div",
                text=f"Row {k} content",
                x=100.0,
                y=100 + k * 40,
            )
        )
    issues = [
        i
        for i in analyze_hierarchy(_snap(nodes))
        if i.fundamental == "alignment"
    ]
    assert issues == []


def test_mixed_text_align_fires() -> None:
    nodes = [_node(0, -1, 0, "main")]
    aligns = ["start", "start", "center"]
    for k, ta in enumerate(aligns):
        nodes.append(
            _node(
                len(nodes),
                0,
                1,
                "p",
                text=f"Item {k} text",
                x=50.0,
                y=100 + k * 40,
                styles={"text-align": ta},
            )
        )
    issues = list(analyze_hierarchy(_snap(nodes)))
    assert "alignment.mixed-text-align" in _checks(issues)


def test_flat_title_fires_but_strong_ratio_does_not() -> None:
    def build(title_size: str, title_weight: str) -> Snapshot:
        nodes = [_node(0, -1, 0, "main")]
        nodes.append(
            _node(
                1,
                0,
                1,
                "h1",
                text="Quarterly overview",
                x=40,
                y=20,
                styles={"font-size": title_size, "font-weight": title_weight},
            )
        )
        for k in range(5):
            nodes.append(
                _node(
                    len(nodes),
                    0,
                    1,
                    "p",
                    text=f"Paragraph {k} carries similar body copy here.",
                    x=40,
                    y=80 + k * 40,
                )
            )
        return _snap(nodes)

    flat = [
        i
        for i in analyze_hierarchy(build("16px", "400"))
        if i.check_id == "visual-hierarchy.flat-title"
    ]
    assert len(flat) == 1
    assert flat[0].severity == "medium"

    strong = [
        i
        for i in analyze_hierarchy(build("28px", "700"))
        if i.fundamental == "visual-hierarchy"
    ]
    assert strong == []


def test_inverted_emphasis_fires_and_clean_pair_does_not() -> None:
    def build(meta_size: str, meta_weight: str) -> Snapshot:
        nodes = [_node(0, -1, 0, "main")]
        nodes.append(_node(1, 0, 1, "article", x=40, y=40))
        nodes.append(
            _node(
                2,
                1,
                2,
                "h3",
                text="Headline",
                x=40,
                y=40,
                styles={"font-size": "16px", "font-weight": "700"},
            )
        )
        nodes.append(
            _node(
                3,
                1,
                2,
                "span",
                text="2 hours ago",
                x=40,
                y=70,
                w=120,
                styles={"font-size": meta_size, "font-weight": meta_weight},
            )
        )
        return _snap(nodes)

    inverted = [
        i
        for i in analyze_hierarchy(build("22px", "300"))
        if i.check_id == "visual-hierarchy.inverted-emphasis"
    ]
    assert len(inverted) == 1
    assert inverted[0].severity == "medium"
    refs = " ".join(inverted[0].element_refs)
    assert "span" in refs and "h3" in refs

    clean = analyze_hierarchy(build("12px", "400"))
    assert clean == []


def test_overlapping_siblings_fire_critical() -> None:
    nodes = [_node(0, -1, 0, "main")]
    for k, y in enumerate([100.0, 115.0, 260.0]):
        nodes.append(
            _node(
                len(nodes),
                0,
                1,
                "div",
                text=f"Card {k}",
                x=100.0,
                y=y,
                w=200,
                h=150,
            )
        )
    issues = list(analyze_hierarchy(_snap(nodes)))
    critical = [i for i in issues if i.severity == "critical"]
    assert critical and all(i.fundamental == "alignment" for i in critical)


def test_clean_design_yields_zero_issues() -> None:
    nodes = [_node(0, -1, 0, "main")]
    nodes.append(
        _node(
            1,
            0,
            1,
            "h1",
            text="Billing settings",
            x=60,
            y=24,
            styles={"font-size": "28px", "font-weight": "700"},
        )
    )
    for k in range(4):
        nodes.append(
            _node(
                len(nodes),
                0,
                1,
                "div",
                text=f"Setting row {k} shows its current value.",
                x=60.0,
                y=90 + k * 44,
                styles={"font-size": "14px"},
            )
        )
    issues = list(analyze_hierarchy(_snap(nodes)))
    assert issues == []
