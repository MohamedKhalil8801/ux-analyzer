"""Unit tests for the typography & scale visual piece."""

from __future__ import annotations

from ux_analyzer.analysis.visual.snapshot import Snapshot, snapshot_from_dict
from ux_analyzer.analysis.visual.types import VisualIssue
from ux_analyzer.analysis.visual.typography import analyze_typography

PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog again and again until"
    " the paragraph wraps onto several rendered lines of body copy."
)


def _styles(**overrides: str) -> dict[str, str]:
    base = {
        "font-family": "Inter, sans-serif",
        "font-size": "16px",
        "font-weight": "400",
        "font-style": "normal",
        "line-height": "normal",
        "letter-spacing": "normal",
        "text-transform": "none",
        "color": "rgb(17, 17, 17)",
        "padding-top": "0px",
        "padding-bottom": "0px",
    }
    base.update(overrides)
    return base


def _node(
    i: int,
    parent: int,
    depth: int,
    tag: str,
    cls: str = "",
    text: str = "",
    styles: dict[str, str] | None = None,
    x: float = 0.0,
    y: float = 0.0,
    w: float = 320.0,
    h: float = 24.0,
) -> dict[str, object]:
    return {
        "i": i,
        "parent": parent,
        "depth": depth,
        "tag": tag,
        "cls": cls,
        "id": "",
        "text": text,
        "box": {"x": x, "y": y, "w": w, "h": h},
        "styles": styles if styles is not None else _styles(),
    }


def _snap(nodes: list[dict[str, object]]) -> Snapshot:
    return snapshot_from_dict(
        {
            "rootBox": {"x": 0.0, "y": 0.0, "w": 1200.0, "h": 900.0},
            "nodes": nodes,
        }
    )


def _ids(issues: list[VisualIssue]) -> set[str]:
    return {issue.check_id for issue in issues}


def _scale_ids(issues: list[VisualIssue]) -> set[str]:
    return {i.check_id for i in issues if i.fundamental == "scale"}


def test_flat_scale_fires_and_proper_ratio_does_not() -> None:
    flat = _snap(
        [
            _node(0, -1, 0, "section"),
            _node(
                1, 0, 1, "h1", text="Pricing plans", styles=_styles()
            ),
            _node(
                2,
                0,
                1,
                "p",
                text=PARAGRAPH,
                y=40.0,
                h=72.0,
                styles=_styles(**{"line-height": "24px"}),
            ),
            _node(
                3,
                0,
                1,
                "p",
                text=PARAGRAPH,
                y=120.0,
                h=72.0,
                styles=_styles(**{"line-height": "24px"}),
            ),
        ]
    )
    issues = list(analyze_typography(flat))
    assert "scale-flat-hierarchy" in _scale_ids(issues)

    proper = _snap(
        [
            _node(0, -1, 0, "section"),
            _node(
                1,
                0,
                1,
                "h1",
                text="Pricing plans",
                styles=_styles(
                    **{"font-size": "32px", "font-weight": "700"}
                ),
            ),
            _node(
                2,
                0,
                1,
                "p",
                text=PARAGRAPH,
                y=48.0,
                h=48.0,
                styles=_styles(**{"line-height": "24px"}),
            ),
            _node(
                3,
                0,
                1,
                "p",
                text=PARAGRAPH,
                y=104.0,
                h=48.0,
                styles=_styles(**{"line-height": "24px"}),
            ),
        ]
    )
    assert list(analyze_typography(proper)) == []


def test_balanced_two_family_palette_stays_silent() -> None:
    families = ["Inter, sans-serif", "Merriweather, serif"]
    nodes: list[dict[str, object]] = [_node(0, -1, 0, "section")]
    for k in range(4):
        nodes.append(
            _node(
                k + 1,
                0,
                1,
                "p",
                text=PARAGRAPH,
                y=float(k * 80),
                styles=_styles(
                    **{
                        "font-family": families[k % 2],
                        "line-height": "24px",
                    }
                ),
            )
        )
    issues = list(analyze_typography(_snap(nodes)))
    assert "typography-font-families" not in _ids(issues)
    assert issues == []


def test_one_off_family_on_single_text_run_fires() -> None:
    nodes = [
        _node(0, -1, 0, "section"),
        _node(
            1,
            0,
            1,
            "label",
            text="Shorten your URL",
            y=0.0,
            h=59.0,
            styles=_styles(
                **{
                    "font-family": "Changa, sans-serif",
                    "font-size": "32px",
                    "font-weight": "700",
                }
            ),
        ),
        _node(
            2,
            0,
            1,
            "p",
            text=PARAGRAPH,
            y=70.0,
            h=48.0,
            styles=_styles(**{"line-height": "24px"}),
        ),
    ]
    issues = list(analyze_typography(_snap(nodes)))
    stray = [
        i for i in issues if i.check_id == "typography-family-one-off"
    ]
    assert len(stray) == 1
    assert stray[0].fundamental == "typography"
    assert stray[0].severity == "medium"


def test_controls_split_across_families_fire() -> None:
    plan = [
        ("Alpha, sans-serif", "Home"),
        ("Alpha, sans-serif", "FAQ"),
        ("Beta, serif", "Login"),
        ("Beta, serif", "Docs"),
    ]
    nodes: list[dict[str, object]] = [_node(0, -1, 0, "nav")]
    for k, (fam, label) in enumerate(plan):
        nodes.append(
            _node(
                k + 1,
                0,
                1,
                "a",
                text=label,
                x=float(k * 120),
                w=110.0,
                styles=_styles(**{"font-family": fam}),
            )
        )
    issues = list(analyze_typography(_snap(nodes)))
    assert _ids(issues) == {"typography-mixed-control-families"}


def test_four_families_fire() -> None:
    families = ["Alpha, sans-serif", "Beta, serif", "Gamma, sans-serif", "Delta, mono"]
    nodes: list[dict[str, object]] = [_node(0, -1, 0, "section")]
    for k, fam in enumerate(families):
        nodes.append(
            _node(
                k + 1,
                0,
                1,
                "p",
                text=PARAGRAPH,
                y=float(k * 80),
                styles=_styles(**{"font-family": fam}),
            )
        )
    issues = list(analyze_typography(_snap(nodes)))
    assert "typography-font-families" in _ids(issues)
    family_issue = next(i for i in issues if i.check_id == "typography-font-families")
    assert family_issue.severity == "medium"


def test_tight_line_height_fires_and_normal_does_not() -> None:
    tight = _snap(
        [
            _node(0, -1, 0, "section"),
            _node(
                1,
                0,
                1,
                "p",
                text=PARAGRAPH,
                h=64.0,
                styles=_styles(**{"line-height": "16px"}),
            ),
        ]
    )
    issues = list(analyze_typography(tight))
    assert "typography-leading-tight" in _ids(issues)

    normal = _snap(
        [
            _node(0, -1, 0, "section"),
            _node(
                1,
                0,
                1,
                "p",
                text=PARAGRAPH,
                h=96.0,
                styles=_styles(**{"line-height": "24px"}),
            ),
        ]
    )
    assert list(analyze_typography(normal)) == []


def test_loose_line_height_flags_low() -> None:
    loose = _snap(
        [
            _node(0, -1, 0, "section"),
            _node(
                1,
                0,
                1,
                "p",
                text=PARAGRAPH,
                h=132.0,
                styles=_styles(**{"line-height": "44px"}),
            ),
        ]
    )
    issues = list(analyze_typography(loose))
    ids = _ids(issues)
    assert "typography-leading-loose" in ids
    assert "typography-leading-tight" not in ids


def test_tiny_and_small_body_sizes_fire_with_right_severity() -> None:
    tiny = _snap(
        [
            _node(0, -1, 0, "section"),
            _node(1, 0, 1, "p", text=PARAGRAPH, styles=_styles(**{"font-size": "9px"})),
        ]
    )
    issues = list(analyze_typography(tiny))
    tiny_issue = next(i for i in issues if i.check_id == "scale-tiny-text")
    assert tiny_issue.severity == "critical"

    small = _snap(
        [
            _node(0, -1, 0, "section"),
            _node(
                1,
                0,
                1,
                "p",
                text=PARAGRAPH,
                styles=_styles(**{"font-size": "11px"}),
            ),
        ]
    )
    small_issues = list(analyze_typography(small))
    small_issue = next(i for i in small_issues if i.check_id == "typography-small-body")
    assert small_issue.severity == "medium"

    normal = _snap(
        [
            _node(0, -1, 0, "section"),
            _node(
                1,
                0,
                1,
                "p",
                text=PARAGRAPH,
                styles=_styles(**{"font-size": "15px", "line-height": "23px"}),
            ),
        ]
    )
    assert list(analyze_typography(normal)) == []


def test_inverted_heading_smaller_than_body() -> None:
    nodes: list[dict[str, object]] = [
        _node(0, -1, 0, "section"),
        _node(
            1,
            0,
            1,
            "h1",
            text="Settings",
            styles=_styles(**{"font-size": "12.8px", "font-weight": "700"}),
        ),
        _node(
            2,
            0,
            1,
            "p",
            text=PARAGRAPH,
            y=40.0,
            h=86.4,
            w=280.0,
            styles=_styles(**{"font-size": "19.2px", "line-height": "28.8px"}),
        ),
        _node(
            3,
            0,
            1,
            "p",
            text=PARAGRAPH,
            y=136.0,
            h=86.4,
            w=280.0,
            styles=_styles(**{"font-size": "19.2px", "line-height": "28.8px"}),
        ),
    ]
    issues = list(analyze_typography(_snap(nodes)))
    inverted = [i for i in issues if i.check_id == "scale-inverted-heading"]
    assert len(inverted) == 1
    assert inverted[0].severity == "medium"
    assert inverted[0].element_refs


def test_sibling_size_drift_fires() -> None:
    sizes = ["16px", "17px", "18px"]
    nodes: list[dict[str, object]] = [_node(0, -1, 0, "div")]
    for k, size in enumerate(sizes):
        nodes.append(
            _node(
                k + 1,
                0,
                1,
                "h3",
                cls="card-title",
                text=f"Option {k}",
                y=float(k * 60),
                styles=_styles(**{"font-size": size, "font-weight": "700"}),
            )
        )
    issues = list(analyze_typography(_snap(nodes)))
    assert _ids(issues) == {"scale-sibling-size-drift"}
    drift = issues[0]
    assert drift.severity == "low"
    assert len(drift.element_refs) == 3


def test_caps_wall_and_body_tracking_fire_but_eyebrow_labels_do_not() -> None:
    caps = _snap(
        [
            _node(0, -1, 0, "section"),
            _node(
                1,
                0,
                1,
                "p",
                text=(
                    "THIS ENTIRE SENTENCE IS RENDERED IN FULL CAPITALS FOR"
                    " NO GOOD REASON AT ALL WHATSOEVER FRIEND"
                ),
                styles=_styles(),
            ),
        ]
    )
    assert "typography-caps-string" in _ids(list(analyze_typography(caps)))

    tracked = _snap(
        [
            _node(0, -1, 0, "section"),
            _node(
                1,
                0,
                1,
                "p",
                text=PARAGRAPH,
                styles=_styles(**{"letter-spacing": "3.2px"}),
            ),
        ]
    )
    assert "typography-letter-spacing" in _ids(list(analyze_typography(tracked)))

    eyebrow = _snap(
        [
            _node(0, -1, 0, "section"),
            _node(
                1,
                0,
                1,
                "p",
                cls="subhead",
                text="Gold Package",
                styles=_styles(
                    **{
                        "font-size": "12.8px",
                        "font-weight": "700",
                        "letter-spacing": "2.56px",
                        "text-transform": "uppercase",
                    }
                ),
            ),
        ]
    )
    assert list(analyze_typography(eyebrow)) == []


def test_weight_sprawl_fires() -> None:
    weights = ["300", "400", "500", "600", "700", "800"]
    nodes: list[dict[str, object]] = [_node(0, -1, 0, "section")]
    for k, weight in enumerate(weights):
        nodes.append(
            _node(
                k + 1,
                0,
                1,
                "span",
                text=f"item {k}",
                x=float(k * 90),
                w=80.0,
                styles=_styles(**{"font-weight": weight}),
            )
        )
    issues = list(analyze_typography(_snap(nodes)))
    sprawl = [i for i in issues if i.check_id == "typography-weight-sprawl"]
    assert len(sprawl) == 1
    assert sprawl[0].evidence["weights"] == [300, 400, 500, 600, 700, 800]


def test_clean_design_yields_zero_issues() -> None:
    nodes: list[dict[str, object]] = [
        _node(0, -1, 0, "section"),
        _node(
            1,
            0,
            1,
            "header",
            y=0.0,
            h=56.0,
        ),
        _node(
            2,
            1,
            2,
            "a",
            text="Products",
            w=120.0,
            styles=_styles(**{"font-weight": "700"}),
        ),
        _node(
            3,
            0,
            1,
            "p",
            cls="eyebrow",
            text="Features",
            y=72.0,
            styles=_styles(
                **{
                    "font-size": "12.8px",
                    "font-weight": "700",
                    "letter-spacing": "1.6px",
                    "text-transform": "uppercase",
                }
            ),
        ),
        _node(
            4,
            0,
            1,
            "h1",
            text="Everything you need to ship faster",
            y=96.0,
            h=40.0,
            w=720.0,
            styles=_styles(**{"font-size": "32px", "font-weight": "700"}),
        ),
        _node(
            5,
            0,
            1,
            "h2",
            text="Why teams choose us",
            y=152.0,
            h=32.0,
            styles=_styles(**{"font-size": "24px", "font-weight": "700"}),
        ),
        _node(
            6,
            0,
            1,
            "p",
            text=PARAGRAPH,
            y=200.0,
            h=48.0,
            w=640.0,
            styles=_styles(**{"line-height": "24px"}),
        ),
        _node(
            7,
            0,
            1,
            "p",
            text=PARAGRAPH,
            y=256.0,
            h=48.0,
            w=640.0,
            styles=_styles(**{"line-height": "24px"}),
        ),
        _node(
            8,
            0,
            1,
            "footer",
            text="Small print stays legible at fourteen pixels.",
            y=320.0,
            h=20.0,
            styles=_styles(**{"font-size": "14px"}),
        ),
    ]
    issues = list(analyze_typography(_snap(nodes)))
    assert issues == []


def test_light_body_weight_fires() -> None:
    nodes = [
        _node(0, -1, 0, "section"),
        _node(
            1,
            0,
            1,
            "p",
            text=PARAGRAPH,
            styles=_styles(**{"font-size": "14.4px", "font-weight": "300"}),
        ),
    ]
    issues = list(analyze_typography(_snap(nodes)))
    assert _ids(issues) == {"typography-light-body"}
    assert issues[0].severity == "low"


def test_display_tracking_on_heading_fires() -> None:
    nodes = [
        _node(0, -1, 0, "section"),
        _node(
            1,
            0,
            1,
            "h1",
            text="Welcome!",
            styles=_styles(
                **{
                    "font-size": "25.6px",
                    "font-weight": "700",
                    "letter-spacing": "3.84px",
                }
            ),
        ),
        _node(
            2,
            0,
            1,
            "p",
            text=PARAGRAPH,
            y=48.0,
            h=48.0,
            styles=_styles(**{"line-height": "24px"}),
        ),
    ]
    issues = list(analyze_typography(_snap(nodes)))
    assert _ids(issues) == {"typography-display-tracking"}
    assert issues[0].severity == "low"


def test_uppercase_label_column_exempt_but_disparity_fires() -> None:
    labels = ["Basketball", "Baseball", "Football", "Tennis"]
    scores = ["58", "23", "75", "90"]
    nodes: list[dict[str, object]] = [_node(0, -1, 0, "section")]
    for k, (label, score) in enumerate(zip(labels, scores)):
        card = len(nodes)
        nodes.append(
            _node(
                card,
                0,
                1,
                "div",
                cls="card",
                x=float(k * 140),
                w=120.0,
                h=226.0,
            )
        )
        nodes.append(
            _node(
                card + 1,
                card,
                2,
                "p",
                cls="label",
                text=label,
                x=float(k * 140 + 24),
                y=120.0,
                w=72.8,
                h=15.6,
                styles=_styles(
                    **{
                        "font-size": "10.4px",
                        "font-weight": "700",
                        "text-transform": "uppercase",
                    }
                ),
            )
        )
        nodes.append(
            _node(
                card + 2,
                card,
                2,
                "h3",
                text=score,
                x=float(k * 140 + 24),
                y=140.0,
                w=76.0,
                h=84.0,
                styles=_styles(
                    **{
                        "font-size": "56px",
                        "font-weight": "700",
                        "line-height": "84px",
                    }
                ),
            )
        )
    issues = list(analyze_typography(_snap(nodes)))
    assert _ids(issues) == {"scale-size-disparity"}
    disparity = issues[0]
    assert disparity.severity == "medium"
    assert disparity.evidence["ratio"] >= 4.5


def test_tiny_control_fires_scale_only() -> None:
    nodes = [
        _node(0, -1, 0, "section"),
        _node(
            1,
            0,
            1,
            "h4",
            text="Chat with us",
            styles=_styles(**{"font-size": "20.8px", "font-weight": "700"}),
        ),
        _node(
            2,
            0,
            1,
            "p",
            text=PARAGRAPH,
            y=40.0,
            h=72.0,
            styles=_styles(**{"line-height": "24px"}),
        ),
        _node(
            3,
            0,
            1,
            "a",
            cls="cta",
            text="Chat now",
            y=120.0,
            styles=_styles(**{"font-size": "12px", "font-weight": "700"}),
        ),
    ]
    issues = list(analyze_typography(_snap(nodes)))
    assert _ids(issues) == {"scale-tiny-control"}
    control = issues[0]
    assert control.fundamental == "scale"
    assert control.severity == "medium"
    assert control.element_refs


def test_shrunk_brand_mark_fires() -> None:
    nodes = [
        _node(0, -1, 0, "header"),
        _node(
            1,
            0,
            1,
            "a",
            cls="logo",
            text="Acme Corp",
            styles=_styles(**{"font-size": "11.2px", "font-weight": "700"}),
        ),
        _node(
            2,
            0,
            1,
            "a",
            cls="nav-link",
            text="Home",
            x=200.0,
            styles=_styles(),
        ),
        _node(
            3,
            0,
            1,
            "a",
            cls="nav-link",
            text="Pricing",
            x=280.0,
            styles=_styles(),
        ),
    ]
    issues = list(analyze_typography(_snap(nodes)))
    scale_ids = _scale_ids(issues)
    assert "scale-brand-shrunk" in scale_ids
    brand = next(i for i in issues if i.check_id == "scale-brand-shrunk")
    assert brand.severity == "medium"
