"""Typography & scale piece.

Detects misapplied typography and scale fundamentals in rendered-page
snapshots: unreadable or off-system type sizes, overloaded or one-off font
families, inconsistent faces across controls, harmful leading and
letter-spacing, unscannable all-caps walls, weight problems, flat or
inverted type hierarchies, shrunk brand marks, undersized controls, drifted
sibling sizes, and extreme size disparity. Every threshold is a generic
design floor; nothing is site-specific.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence

from ux_analyzer.analysis.visual.snapshot import Snapshot, SNode
from ux_analyzer.analysis.visual.types import VisualIssue

_HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_PROSE_TAGS = frozenset({"p", "li", "blockquote", "dd", "dt"})
_CONTROLS = frozenset({"a", "button", "input", "select", "textarea"})
_CLICKABLE = frozenset({"a", "button"})
_BRAND_TOKENS = frozenset({"logo", "brand", "wordmark"})
_MAX_REFS = 8

_MIN_TEXT_PX = 10.0  # absolute readability floor for any rendered text
_MIN_LABEL_PX = 11.0  # floor for repeated label patterns
_MIN_BODY_PX = 12.0  # floor for body-length text runs
_MAX_FAMILIES = 3
_MAX_WEIGHTS = 5
_TIGHT_RATIO = 1.15
_LOOSE_RATIO = 2.2
_MIN_WRAP_LINES = 1.8
_TRACK_EM = 0.10
_TRACK_EM_DISPLAY = 0.12
_TRACK_PX = 1.5
_TRACK_CHARS = 40
_TRACK_DISPLAY_CHARS = 4
_CAPS_CHARS = 30
_FLAT_MAX = 1.10  # largest heading may sit at most 10% above body
_INVERT_MIN = 0.95  # heading must be at least 95% of prose body size
_DRIFT_SPAN_PX = 2.5
_DRIFT_MEMBERS = 3
_SPREAD_RATIO = 4.5  # max/min text size within one view
_TINY_CONTROL_PX = 13.0
_TINY_CONTROL_RATIO = 0.85
_BRAND_RATIO = 0.75


def analyze_typography(snapshot: Snapshot) -> Sequence[VisualIssue]:
    """Detect typography and scale misapplication in a visual snapshot."""
    issues: list[VisualIssue] = []
    issues.extend(_family_issues(snapshot))
    issues.extend(_size_issues(snapshot))
    issues.extend(_leading_issues(snapshot))
    issues.extend(_tracking_caps_issues(snapshot))
    issues.extend(_weight_issues(snapshot))
    issues.extend(_scale_issues(snapshot))
    return issues


def _issue(
    snapshot: Snapshot,
    fundamental: str,
    check_id: str,
    title: str,
    description: str,
    severity: str,
    evidence: dict[str, object],
    hits: Sequence[SNode],
) -> VisualIssue:
    return VisualIssue(
        fundamental=fundamental,
        check_id=check_id,
        title=title,
        description=description,
        severity=severity,
        evidence=dict(evidence, hit_count=len(hits)),
        element_refs=tuple(_ref(snapshot, n) for n in hits[:_MAX_REFS]),
    )


def _fs(node: SNode) -> float:
    value = node.style_px("font-size")
    return value if value is not None else 0.0


def _alpha_len(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def _is_offscreen(node: SNode) -> bool:
    """Fully above/left of the initial viewport (hidden skip-links, drawers).

    Below-fold content is NOT offscreen: it renders in the full-page
    capture and is legitimate analysis scope.
    """
    return node.box.y < -20 or node.box.x < -20

def _text_nodes(snapshot: Snapshot) -> list[SNode]:
    return [n for n in snapshot.nodes if _alpha_len(n.text) > 0 and not _is_offscreen(n)]


def _ref(snapshot: Snapshot, node: SNode) -> str:
    return snapshot.selector(node)


def _primary_family(stack: str) -> str:
    return stack.split(",")[0].strip().strip("'\"").lower()


_NAMED_WEIGHTS = {"bold": 700, "normal": 400}


def _weight_int(node: SNode) -> int | None:
    raw = node.style("font-weight").strip().lower()
    if not raw:
        return None
    value = _NAMED_WEIGHTS.get(raw, raw)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _family_issues(snapshot: Snapshot) -> list[VisualIssue]:
    issues: list[VisualIssue] = []
    by_family: dict[str, SNode] = {}
    text_by_family: dict[str, list[SNode]] = defaultdict(list)
    for node in snapshot.nodes:
        if _is_offscreen(node):
            continue
        stack = node.style("font-family")
        if not stack:
            continue
        by_family.setdefault(_primary_family(stack), node)
    for node in _text_nodes(snapshot):
        stack = node.style("font-family")
        if stack:
            text_by_family[_primary_family(stack)].append(node)

    families = sorted(by_family)
    if len(families) > _MAX_FAMILIES:
        issues.append(
            _issue(
                snapshot,
                "typography",
                "typography-font-families",
                "Too many font families in one view",
                (
                    f"{len(families)} distinct font families"
                    f" ({', '.join(families)}) appear in one view; keep the"
                    f" palette to {_MAX_FAMILIES} or fewer so the type system"
                    " reads as deliberate."
                ),
                "medium",
                {"families": families},
                [by_family[fam] for fam in families],
            )
        )

    if len(text_by_family) == 2 and 1 in (
        len(nodes) for nodes in text_by_family.values()
    ):
        loners = [
            fam
            for fam, nodes in text_by_family.items()
            if len(nodes) == 1
        ]
        stray = [text_by_family[fam][0] for fam in loners]
        issues.append(
            _issue(
                snapshot,
                "typography",
                "typography-family-one-off",
                "Single text run breaks the font palette",
                (
                    f"One-off families ({', '.join(sorted(loners))}) appear on"
                    " isolated text while the rest of the view uses a"
                    " different face; accents like this read as accidents"
                    " rather than a two-family system."
                ),
                "medium",
                {"family_counts": {f: len(text_by_family[f]) for f in text_by_family}},
                stray,
            )
        )

    control_families: dict[str, list[SNode]] = defaultdict(list)
    for node in snapshot.nodes:
        if node.tag not in _CONTROLS or _is_offscreen(node):
            continue
        stack = node.style("font-family")
        if stack:
            control_families[_primary_family(stack)].append(node)
    established = {
        fam: nodes
        for fam, nodes in control_families.items()
        if len(nodes) >= 2
    }
    if len(established) >= 2:
        involved = [
            n
            for nodes in established.values()
            for n in nodes
        ]
        issues.append(
            _issue(
                snapshot,
                "typography",
                "typography-mixed-control-families",
                "Controls mix font families at the same role",
                (
                    f"Interactive controls render in {len(established)}"
                    f" different faces ({', '.join(sorted(established))});"
                    " buttons and links at one role should share a family."
                ),
                "medium",
                {
                    "control_family_counts": {
                        f: len(control_families[f]) for f in sorted(established)
                    }
                },
                involved,
            )
        )
    return issues


def _off_grid(value: float) -> bool:
    steps_x10 = abs(value * 10 - round(value * 10)) > 0.15
    steps_int = abs(value - round(value)) > 0.15
    return steps_x10 and steps_int


def _size_issues(snapshot: Snapshot) -> list[VisualIssue]:
    issues: list[VisualIssue] = []
    small_body: list[tuple[SNode, float]] = []
    micro_labels: list[SNode] = []
    odd_sizes: list[tuple[SNode, float]] = []
    label_groups: dict[str, list[SNode]] = defaultdict(list)
    for node in _text_nodes(snapshot):
        size = _fs(node)
        if size <= 0:
            continue
        text = node.text.strip()
        letters = _alpha_len(text)
        if _MIN_TEXT_PX <= size < _MIN_BODY_PX and letters >= 25:
            small_body.append((node, size))
        if size < _MIN_LABEL_PX:
            label_groups[node.tag].append(node)
        if 12 <= size < 16 and letters >= 25 and _off_grid(size):
            odd_sizes.append((node, size))
    for members in label_groups.values():
        if len(members) < 2:
            continue
        uniform = len({round(_fs(m), 2) for m in members}) == 1
        upper = all(
            m.style("text-transform").lower() == "uppercase" for m in members
        )
        if uniform and upper:
            continue
        micro_labels.extend(members)
    if small_body:
        sizes = sorted({round(sz, 2) for _, sz in small_body})
        worst = min(sz for _, sz in small_body)
        nodes = [node for node, _ in small_body]
        issues.append(
            _issue(
                snapshot,
                "typography",
                "typography-small-body",
                "Body copy below the 12px legibility floor",
                (
                    f"Body-length text renders at {sizes} px; the smallest run"
                    f" is {worst} px, under the 12px floor for readable body"
                    " copy."
                ),
                "medium",
                {"font_sizes_px": sizes},
                nodes,
            )
        )
    if micro_labels:
        sizes = sorted({round(_fs(node), 2) for node in micro_labels})
        issues.append(
            _issue(
                snapshot,
                "typography",
                "typography-micro-labels",
                "Repeated labels render below the 11px floor",
                (
                    f"{len(micro_labels)} text elements render at {sizes} px,"
                    " below an 11px floor for labels users are expected to"
                    " read."
                ),
                "low",
                {"font_sizes_px": sizes},
                micro_labels,
            )
        )
    if odd_sizes:
        sizes = sorted({round(sz, 2) for _, sz in odd_sizes})
        nodes = [node for node, _ in odd_sizes]
        issues.append(
            _issue(
                snapshot,
                "typography",
                "typography-odd-body-size",
                "Off-grid font sizes in body text",
                (
                    f"Body text uses arbitrary sizes {sizes} px instead of a"
                    " stepped scale; values like these read as accidents"
                    " rather than decisions."
                ),
                "low",
                {"font_sizes_px": sizes},
                nodes,
            )
        )
    return issues


def _line_height_ratio(raw: str, font_px: float) -> float | None:
    value = raw.strip()
    if not value or value == "normal":
        return None
    try:
        if value.endswith("px"):
            return float(value.removesuffix("px")) / font_px
        if value.endswith("%"):
            return float(value.removesuffix("%")) / 100
        return float(value)
    except ValueError:
        return None


def _vertical_padding(node: SNode) -> float:
    top = node.style_px("padding-top") or 0.0
    bottom = node.style_px("padding-bottom") or 0.0
    return top + bottom


def _leading_issues(snapshot: Snapshot) -> list[VisualIssue]:
    tight: list[tuple[SNode, float]] = []
    loose: list[tuple[SNode, float]] = []
    for node in snapshot.nodes:
        text = node.text.strip()
        if not text:
            continue
        size = _fs(node)
        if size <= 0:
            continue
        ratio = _line_height_ratio(node.style("line-height"), size)
        if ratio is None:
            continue
        if node.tag not in _PROSE_TAGS and _alpha_len(text) < 60:
            continue
        content_h = node.box.h - _vertical_padding(node)
        if content_h <= 0 or ratio * size <= 0:
            continue
        if content_h / (ratio * size) < _MIN_WRAP_LINES:
            continue
        if ratio < _TIGHT_RATIO:
            tight.append((node, round(ratio, 3)))
        elif ratio > _LOOSE_RATIO:
            loose.append((node, round(ratio, 3)))
    issues: list[VisualIssue] = []
    if tight:
        issues.append(
            _issue(
                snapshot,
                "typography",
                "typography-leading-tight",
                "Line height too tight for multi-line text",
                (
                    f"Multi-line text uses {sorted(r for _, r in tight)}x line"
                    " height, under 1.15x; lines crowd together and slow"
                    " reading."
                ),
                "medium",
                {"ratios": sorted(r for _, r in tight)},
                [node for node, _ in tight],
            )
        )
    if loose:
        issues.append(
            _issue(
                snapshot,
                "typography",
                "typography-leading-loose",
                "Line height too loose for multi-line text",
                (
                    f"Multi-line text uses {sorted(r for _, r in loose)}x line"
                    " height, over 2.2x; lines drift apart and lose grouping."
                ),
                "low",
                {"ratios": sorted(r for _, r in loose)},
                [node for node, _ in loose],
            )
        )
    return issues


def _tracking_caps_issues(snapshot: Snapshot) -> list[VisualIssue]:
    tracked_body: list[tuple[SNode, float]] = []
    tracked_display: list[tuple[SNode, float]] = []
    caps_walls: list[SNode] = []
    for node in _text_nodes(snapshot):
        size = _fs(node)
        if size <= 0:
            continue
        text = node.text.strip()
        transform = node.style("text-transform").lower()
        spacing = node.style_px("letter-spacing")
        if (
            spacing is not None
            and spacing >= _TRACK_PX
            and transform not in {"uppercase", "small-caps"}
            and any(ch.islower() for ch in text)
        ):
            em = spacing / size
            if em >= _TRACK_EM and _alpha_len(text) >= _TRACK_CHARS:
                tracked_body.append((node, round(em, 3)))
            elif (
                em >= _TRACK_EM_DISPLAY
                and _alpha_len(text) >= _TRACK_DISPLAY_CHARS
            ):
                tracked_display.append((node, round(em, 3)))
        letters = [ch for ch in text if ch.isalpha()]
        shown_upper = transform == "uppercase" or (
            bool(letters) and text.upper() == text
        )
        if shown_upper and len(letters) > _CAPS_CHARS and " " in text:
            caps_walls.append(node)
    issues: list[VisualIssue] = []
    if tracked_body:
        issues.append(
            _issue(
                snapshot,
                "typography",
                "typography-letter-spacing",
                "Excessive letter spacing on body text",
                (
                    "Long lowercase text carries "
                    f"{sorted(e for _, e in tracked_body)}em letter spacing;"
                    " wide tracking belongs on short display labels, not"
                    " reading text."
                ),
                "low",
                {"em_ratios": sorted(e for _, e in tracked_body)},
                [node for node, _ in tracked_body],
            )
        )
    if tracked_display:
        issues.append(
            _issue(
                snapshot,
                "typography",
                "typography-display-tracking",
                "Heading letter spacing is unusually wide",
                (
                    "Short display text carries "
                    f"{sorted(e for _, e in tracked_display)}em letter"
                    " spacing without an uppercase treatment; tracked-out"
                    " mixed-case headings read as loose and unfocused."
                ),
                "low",
                {"em_ratios": sorted(e for _, e in tracked_display)},
                [node for node, _ in tracked_display],
            )
        )
    if caps_walls:
        issues.append(
            _issue(
                snapshot,
                "typography",
                "typography-caps-string",
                "Long all-caps string hurts scanability",
                (
                    f"{len(caps_walls)} text blocks render more than"
                    f" {_CAPS_CHARS} letters in full capitals; word shape is"
                    " lost and scanning slows."
                ),
                "medium",
                {},
                caps_walls,
            )
        )
    return issues


def _weight_issues(snapshot: Snapshot) -> list[VisualIssue]:
    issues: list[VisualIssue] = []
    weights: set[int] = set()
    light_prose: list[SNode] = []
    for node in _text_nodes(snapshot):
        weight = _weight_int(node)
        if weight is None:
            continue
        weights.add(weight)
        if (
            weight < 400
            and node.tag not in _HEADINGS
            and _alpha_len(node.text.strip()) >= 25
        ):
            light_prose.append(node)
    if len(weights) > _MAX_WEIGHTS:
        issues.append(
            _issue(
                snapshot,
                "typography",
                "typography-weight-sprawl",
                "Too many distinct font weights",
                (
                    f"{len(weights)} distinct font weights ({sorted(weights)})"
                    " appear in one view; more than five signals a missing"
                    " weight system."
                ),
                "medium",
                {"weights": sorted(weights)},
                [],
            )
        )
    if light_prose:
        issues.append(
            _issue(
                snapshot,
                "typography",
                "typography-light-body",
                "Body copy set in a lighter weight than regular",
                (
                    f"{len(light_prose)} body-length text runs use a font"
                    " weight below 400; light strokes drop out at body sizes"
                    " and hurt sustained reading."
                ),
                "low",
                {"weights": sorted({_weight_int(n) or 0 for n in light_prose})},
                light_prose,
            )
        )
    return issues


def _mode_font_size(nodes: Sequence[SNode]) -> float:
    counts = Counter(round(_fs(n), 1) for n in nodes if _fs(n) > 0)
    if not counts:
        return 0.0
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _heading_level_drift(
    snapshot: Snapshot, heads: Sequence[SNode]
) -> list[VisualIssue]:
    levels: dict[str, list[float]] = defaultdict(list)
    for node in heads:
        size = _fs(node)
        if size > 0:
            levels[node.tag].append(size)
    meds = [round(statistics.median(sizes), 2) for sizes in levels.values()]
    steps = sorted(set(meds))
    if len(steps) < 3:
        return []
    gaps = [round(b - a, 2) for a, b in zip(steps, steps[1:])]
    if max(gaps) > _DRIFT_SPAN_PX:
        return []
    return [
        _issue(
            snapshot,
            "scale",
            "scale-heading-step-drift",
            "Heading levels differ by arbitrary slivers",
            (
                f"Heading levels sit at {steps} px, separated by {gaps} px"
                " steps; real scales jump in visible, repeatable increments."
            ),
            "low",
            {"level_sizes_px": steps},
            [],
        )
    ]


def _scale_issues(snapshot: Snapshot) -> list[VisualIssue]:
    issues: list[VisualIssue] = []
    text_nodes = _text_nodes(snapshot)

    tiny = [n for n in text_nodes if 0 < _fs(n) < _MIN_TEXT_PX]
    if tiny:
        sizes = sorted({round(_fs(n), 2) for n in tiny})
        issues.append(
            _issue(
                snapshot,
                "scale",
                "scale-tiny-text",
                "Text renders below the 10px readability floor",
                (
                    f"Text elements render at {sizes} px, under a 10px floor;"
                    " the content is effectively unreadable."
                ),
                "critical",
                {"font_sizes_px": sizes},
                tiny,
            )
        )

    def _is_tiny_box(n: SNode) -> bool:
        return n.box.w <= 2 or n.box.h <= 2 or n.box.w * n.box.h < 80

    sized = [
        (n, _fs(n))
        for n in snapshot.nodes
        if n.text.strip() and _fs(n) > 0 and not _is_offscreen(n) and not _is_tiny_box(n)
    ]
    if sized:
        max_size = max(fs for _, fs in sized)
        min_size = min(fs for _, fs in sized)
        if min_size > 0 and max_size / min_size >= _SPREAD_RATIO:
            floor = min_size * 1.15
            lows = [n for n, fs in sized if fs <= floor]
            peak = [n for n, fs in sized if fs == max_size][:1]
            issues.append(
                _issue(
                    snapshot,
                    "scale",
                    "scale-size-disparity",
                    "Extreme size gap between smallest and largest text",
                    (
                        f"Text sizes span {round(min_size, 1)} px to"
                        f" {round(max_size, 1)} px"
                        f" ({round(max_size / min_size, 1)}x) with no middle"
                        " steps carrying the smallest strings; the tiny text"
                        " drowns next to the display sizes."
                    ),
                    "medium",
                    {
                        "min_px": round(min_size, 1),
                        "max_px": round(max_size, 1),
                        "ratio": round(max_size / min_size, 2),
                        "smallest_count": len(lows),
                    },
                    lows + peak,
                )
            )

    heads = [
        n for n in snapshot.nodes if n.tag in _HEADINGS and n.text.strip()
    ]
    non_heading_text = [n for n in text_nodes if n.tag not in _HEADINGS]
    prose = [n for n in non_heading_text if _alpha_len(n.text.strip()) >= 20]
    prose_body = _mode_font_size(prose) if prose else 0.0
    general_pool = prose if prose else (
        non_heading_text if len(non_heading_text) >= 3 else []
    )
    general_body = _mode_font_size(general_pool) if general_pool else 0.0

    submerged = [
        n
        for n in heads
        if prose_body > 0 and 0 < _fs(n) < _INVERT_MIN * prose_body
    ]
    towering = [
        n
        for n in heads
        if general_body > 0 and _fs(n) > _FLAT_MAX * general_body
    ]
    if submerged:
        sizes = sorted({round(_fs(n), 2) for n in submerged})
        issues.append(
            _issue(
                snapshot,
                "scale",
                "scale-inverted-heading",
                "Heading renders smaller than its body text",
                (
                    f"Headings at {sizes} px sit below the dominant prose size"
                    f" of {round(prose_body, 1)} px, inverting the hierarchy"
                    " the page claims."
                ),
                "medium",
                {"body_px": round(prose_body, 1), "heading_sizes_px": sizes},
                submerged,
            )
        )
    elif heads and general_body > 0 and not towering:
        if len(heads) >= 2 or len(prose) >= 2:
            hsizes = sorted({round(_fs(n), 2) for n in heads})
            issues.append(
                _issue(
                    snapshot,
                    "scale",
                    "scale-flat-hierarchy",
                    "Flat type scale: headings match body size",
                    (
                        f"Every heading renders at {hsizes} px while body text"
                        f" sits at {round(general_body, 1)} px; the largest"
                        " heading is within 10% of body copy, so no level"
                        " reads as more important."
                    ),
                    "medium",
                    {
                        "body_px": round(general_body, 1),
                        "heading_sizes_px": hsizes,
                    },
                    heads,
                )
            )

    clickable = [
        n for n in snapshot.nodes if n.tag in _CLICKABLE and n.text.strip()
    ]
    if clickable:
        ref_size = statistics.median([_fs(n) for n in clickable])
        shrunk_brands = [
            n
            for n in snapshot.nodes
            if _BRAND_TOKENS & set(n.classes)
            and 0 < _fs(n) < _BRAND_RATIO * ref_size
        ]
        if shrunk_brands:
            sizes = sorted({round(_fs(n), 2) for n in shrunk_brands})
            issues.append(
                _issue(
                    snapshot,
                    "scale",
                    "scale-brand-shrunk",
                    "Brand mark rendered far below surrounding controls",
                    (
                        f"Logo/brand text renders at {sizes} px while nearby"
                        f" clickable text medians {round(ref_size, 1)} px;"
                        " the brand anchor loses its place in the hierarchy."
                    ),
                    "medium",
                    {
                        "brand_sizes_px": sizes,
                        "control_median_px": round(ref_size, 1),
                    },
                    shrunk_brands,
                )
            )

    if general_body > 0:
        small_controls = [
            n
            for n in snapshot.nodes
            if n.tag in _CONTROLS
            and n.text.strip()
            and 0 < _fs(n) < _TINY_CONTROL_PX
            and _fs(n) < _TINY_CONTROL_RATIO * general_body
        ]
        if small_controls:
            sizes = sorted({round(_fs(n), 2) for n in small_controls})
            issues.append(
                _issue(
                    snapshot,
                    "scale",
                    "scale-tiny-control",
                    "Action control labeled far below body size",
                    (
                        f"Clickable actions render at {sizes} px, under 13px"
                        f" and well under the {round(general_body, 1)} px"
                        " body size around them; primary actions shrink"
                        " below the text they serve."
                    ),
                    "medium",
                    {
                        "control_sizes_px": sizes,
                        "body_px": round(general_body, 1),
                    },
                    small_controls,
                )
            )

    issues.extend(_heading_level_drift(snapshot, heads))

    groups: dict[tuple[int, str], list[SNode]] = defaultdict(list)
    for node in text_nodes:
        if _fs(node) > 0:
            groups[(node.parent, node.tag)].append(node)
    drift: list[SNode] = []
    for members in groups.values():
        if len(members) < _DRIFT_MEMBERS:
            continue
        sizes = {round(_fs(m), 2) for m in members}
        if len(sizes) >= 2 and max(sizes) - min(sizes) <= _DRIFT_SPAN_PX:
            drift.extend(members)
    if drift:
        sizes = sorted({round(_fs(n), 2) for n in drift})
        issues.append(
            _issue(
                snapshot,
                "scale",
                "scale-sibling-size-drift",
                "Same-role siblings drift by a few pixels of type size",
                (
                    f"Sibling elements of the same role render at {sizes} px;"
                    " the near-identical values look unsystematic rather than"
                    " intentional."
                ),
                "low",
                {"font_sizes_px": sizes},
                drift,
            )
        )
    return issues
