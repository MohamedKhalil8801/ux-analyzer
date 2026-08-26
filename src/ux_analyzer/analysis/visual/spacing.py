"""Spacing & layout piece: white-space fundamental.

Flags designs whose white space is misapplied: cramped containers, content
pressed against surface edges, text blocks butted together with no gap,
erratic sibling spacing rhythm, and record lists packed without separation.

Measurements come from rendered geometry, not author intent, and every rule
carries generic escape hatches (navigation regions, intentional overflow,
inner padding that already supplies separation) so well-spaced designs stay
silent. No per-site knowledge.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import pairwise

from ux_analyzer.analysis.visual.snapshot import Snapshot, SNode
from ux_analyzer.analysis.visual.types import VisualIssue

# --- generic thresholds (px) ---------------------------------------------
_TOUCH_PX = 2.0  # text this close to a visible surface edge -> critical
_CRAMPED_INSET_PX = 8.0  # text inset below this in a surfaced container
_OVERFLOW_SLACK_PX = -8.0  # text further out than this is an overlay/menu
_SURFACE_MIN_W = 120.0  # surfaced containers smaller than this are chips
_SURFACE_MIN_H = 48.0
_COLLISION_MAX_GAP_PX = 4.0  # visual gap below this between text masses
_COLLISION_CRITICAL_PX = 2.0  # at or below this the blocks visually touch
_COLLISION_SLACK_PX = 4.0  # tolerate tiny text-mass overlaps from rounding
_PAIR_MIN_OVERLAP_PX = 20.0  # horizontal overlap for a stacked pair
_RHYTHM_MIN_MEMBERS = 3  # need >=3 like siblings for a rhythm sample
_RHYTHM_MIN_GAPS = 2
_RHYTHM_SPREAD_PX = 16.0  # max-min spread that counts as erratic
_SIZE_RATIO = 0.45  # paired siblings must be roughly the same size
_DENSITY_MAX_GAP_PX = 6.0  # every gap at or below this = packed
_DENSITY_MIN_RECORDS = 5
_DENSITY_MIN_RECORD_H = 28.0  # records must be substantial rows/cards
_DENSITY_MIN_GAPS = 3
_BLOCK_MIN_TEXT = 40  # chars across a subtree to count as a content block
_TIGHT_GAP_PX = 12.0  # record gap at or below this reads as no separation
_TIGHT_RECORD_RATIO = 1.8  # record height vs line-height ceiling for tightness
_COLLISION_MIN_OVERLAP_PX = 8.0  # sibling text boxes overlapping this much
_COLLISION_CRITICAL_PX2 = 3.0  # overlap at or above this is a hard collision
_OVERFLOW_MIN_PX = 18.0  # child sticking out of its parent by this much
_FLUSH_TOP_PX = 2.0  # first child this close to container top = flush
_DEAD_BOTTOM_PX = 48.0  # padded bottom leftover at or above this = dead zone
_TALL_CONTAINER_PX = 300.0  # distribution check only for tall containers

def _is_too_small(node: SNode) -> bool:
    """Decorative thin lines and tiny placeholders are not layout blocks."""
    return node.box.w <= 2 or node.box.h <= 2 or node.box.w * node.box.h < 150


# Navigation and footer landmarks follow different white-space conventions
# (full-bleed menus, compact link columns, hover targets flush to menu
# edges, dropdown overlays); spacing detectors stay quiet inside them.
_NAVIGATION_TAGS = frozenset({"nav", "header", "footer"})


def analyze_spacing(snapshot: Snapshot) -> Sequence[VisualIssue]:
    """Detect white-space misapplication in a rendered snapshot."""
    scan = _Scan(snapshot)
    issues: list[VisualIssue] = []
    issues.extend(_find_cramped_containers(snapshot, scan))
    issues.extend(_find_section_collisions(snapshot, scan))
    groups = [
        (parent_i, tag, members)
        for parent_i, tag, members in _like_sibling_groups(snapshot, scan)
        if not _in_navigation(scan, scan.by_index[parent_i])
    ]
    issues.extend(_find_erratic_sibling_gaps(snapshot, scan, groups))
    issues.extend(_find_packed_records(snapshot, scan, groups))
    issues.extend(_find_tight_record_rhythm(snapshot, scan, groups))
    issues.extend(_find_sibling_text_collision(snapshot, scan))
    issues.extend(_find_child_overflow(snapshot, scan))
    issues.extend(_find_flush_top_dead_bottom(snapshot, scan))
    return issues


class _Scan:
    """Per-snapshot aggregates: visibility and subtree text statistics."""

    def __init__(self, snapshot: Snapshot) -> None:
        self.snapshot = snapshot
        self.by_index = {n.i: n for n in snapshot.nodes}
        self.visible = {n.i: _is_visible(n) for n in snapshot.nodes}
        self._text_len = {n.i: 0 for n in snapshot.nodes}
        self._text_nodes: dict[int, list[int]] = {n.i: [] for n in snapshot.nodes}
        for n in sorted(snapshot.nodes, key=lambda m: -m.depth):
            stripped = n.text.strip()
            if stripped:
                self._text_len[n.i] += len(stripped)
                self._text_nodes[n.i].append(n.i)
            if n.parent >= 0:
                self._text_len[n.parent] += self._text_len[n.i]
                self._text_nodes[n.parent].extend(self._text_nodes[n.i])

    def visible_children(self, i: int) -> list[SNode]:
        return [n for n in self.snapshot.children(i) if self.visible[n.i]]

    def block_text_len(self, i: int) -> int:
        return self._text_len[i]

    def visible_text_nodes(self, i: int) -> list[SNode]:
        return [
            self.by_index[t] for t in self._text_nodes[i] if self.visible[t]
        ]


def _is_visible(node: SNode) -> bool:
    if node.box.w <= 0 or node.box.h <= 0:
        return False
    if node.style("display") == "none":
        return False
    # Hidden skip-links / drawers sit fully above or left of the viewport.
    if node.box.y < -20 or node.box.x < -20:
        return False
    opacity = node.style_px("opacity")
    return opacity is None or opacity > 0


def _has_surface(node: SNode) -> bool:
    """True when the element paints a visible background or border."""
    color = node.style("background-color").strip().lower()
    if color and color != "transparent" and _alpha(color) > 0:
        return True
    return any(
        (node.style_px(f"border-{side}-width") or 0) > 0
        for side in ("top", "right", "bottom", "left")
    )


def _alpha(color: str) -> float:
    """Extract the alpha channel of a css color string (opaque default)."""
    if color.startswith("rgba"):
        inner = color[color.find("(") + 1 : color.rfind(")")]
        parts = [p.strip() for p in inner.split(",")]
        if len(parts) == 4:
            try:
                return float(parts[3])
            except ValueError:
                return 1.0
    return 1.0


def _line_height(node: SNode) -> float:
    lh = node.style_px("line-height")
    if lh and lh > 0:
        return lh
    font_size = node.style_px("font-size")
    return font_size * 1.45 if font_size else 19.0


def _in_navigation(scan: _Scan, node: SNode) -> bool:
    """True when the element lives under a nav/header landmark."""
    cur: SNode | None = node
    while cur is not None:
        if cur.tag in _NAVIGATION_TAGS:
            return True
        cur = scan.by_index.get(cur.parent) if cur.parent >= 0 else None
    return False


def _content_span_y(scan: _Scan, node: SNode) -> tuple[float, float] | None:
    """Vertical extent of the text actually rendered in this subtree."""
    texts = scan.visible_text_nodes(node.i)
    if not texts:
        return None
    top = min(t.box.y for t in texts)
    bottom = max(t.box.y + t.box.h for t in texts)
    return top, bottom


def _is_multiline_text_mass(scan: _Scan, node: SNode) -> bool:
    """A subtree whose rendered text occupies roughly 2+ lines."""
    texts = scan.visible_text_nodes(node.i)
    if not texts:
        return False
    enough_content = (
        sum(len(t.text.strip()) for t in texts) >= _BLOCK_MIN_TEXT
        or len(texts) >= 2
    )
    if not enough_content:
        return False
    top, bottom = _content_span_y(scan, node) or (0.0, 0.0)
    tallest_line = max(_line_height(t) for t in texts)
    return (bottom - top) >= 1.7 * tallest_line


def _similar_size(a: SNode, b: SNode) -> bool:
    for dim in ("w", "h"):
        lo, hi = sorted((getattr(a.box, dim), getattr(b.box, dim)))
        if hi > 0 and lo / hi < _SIZE_RATIO:
            return False
    return True


def _pair_orientation(a: SNode, b: SNode) -> tuple[str, float] | None:
    """How DOM-adjacent like siblings meet: (axis, rendered gap)."""
    if not _similar_size(a, b):
        return None
    x_overlap = min(a.box.x + a.box.w, b.box.x + b.box.w) - max(a.box.x, b.box.x)
    y_overlap = min(a.box.y + a.box.h, b.box.y + b.box.h) - max(a.box.y, b.box.y)
    if (
        b.box.y >= a.box.y
        and x_overlap >= 0.5 * min(a.box.w, b.box.w)
    ):
        return "vertical", b.box.y - (a.box.y + a.box.h)
    if (
        b.box.x >= a.box.x
        and y_overlap >= 0.5 * min(a.box.h, b.box.h)
    ):
        return "horizontal", b.box.x - (a.box.x + a.box.w)
    return None


def _dominant_axis_gaps(members: list[SNode]) -> tuple[str, list[float]]:
    """Gaps between DOM-adjacent like siblings along their shared axis."""
    per_axis: dict[str, list[float]] = {"vertical": [], "horizontal": []}
    for a, b in pairwise(members):
        oriented = _pair_orientation(a, b)
        if oriented is None:
            continue
        axis, gap = oriented
        if gap < -0.5:
            continue  # overlaps belong to other detectors, not rhythm
        per_axis[axis].append(max(gap, 0.0))
    axis = max(per_axis, key=lambda k: len(per_axis[k]))
    return axis, per_axis[axis]


def _like_sibling_groups(
    snapshot: Snapshot, scan: _Scan
) -> Iterator[tuple[int, str, list[SNode]]]:
    """Per parent, visible children sharing a tag, in DOM order."""
    for parent in snapshot.nodes:
        buckets: dict[str, list[SNode]] = {}
        for kid in scan.visible_children(parent.i):
            buckets.setdefault(kid.tag, []).append(kid)
        for tag, members in buckets.items():
            if len(members) >= _RHYTHM_MIN_MEMBERS:
                yield parent.i, tag, members


def _reliable_sides(node: SNode) -> frozenset[str]:
    """Box sides where the box edge tracks the rendered glyphs.

    Full-width block text only touches the side its alignment anchors to;
    centered text leaves slack on both horizontal sides. Inline-level boxes
    shrink-wrap their glyphs, so every side is trustworthy.
    """
    if "inline" in node.style("display"):
        return frozenset({"left", "right", "top", "bottom"})
    align = node.style("text-align")
    if align == "center":
        return frozenset({"top", "bottom"})
    if align in ("right", "end"):
        return frozenset({"right", "top", "bottom"})
    return frozenset({"left", "top", "bottom"})


def _find_cramped_containers(
    snapshot: Snapshot, scan: _Scan
) -> Iterator[VisualIssue]:
    """Visible surfaces holding several text blocks with almost no inset.

    Measures how close the closest *rendered text* gets to each painted
    edge. Wrappers behind wrappers are handled for free because geometry
    already reflects nested padding. Text far outside the surface is an
    overlay/menu, not cramping, and is ignored.
    """
    for c in snapshot.nodes:
        if c.parent < 0 or not scan.visible[c.i] or not _has_surface(c):
            continue
        if _in_navigation(scan, c):
            continue
        if c.box.w < _SURFACE_MIN_W or c.box.h < _SURFACE_MIN_H:
            continue
        texts = scan.visible_text_nodes(c.i)
        if len(texts) < 2:
            continue
        right = c.box.x + c.box.w
        bottom = c.box.y + c.box.h
        worst_node: SNode | None = None
        worst_side = ""
        worst_inset = float("inf")
        for t in texts:
            sides = _reliable_sides(t)
            insets = {
                "left": t.box.x - c.box.x,
                "right": right - (t.box.x + t.box.w),
                "top": t.box.y - c.box.y,
                "bottom": bottom - (t.box.y + t.box.h),
            }
            side, value = min(
                (kv for kv in insets.items() if kv[0] in sides),
                key=lambda kv: kv[1],
            )
            if value < worst_inset:
                worst_inset, worst_node, worst_side = value, t, side
        if worst_node is None:
            continue
        if worst_inset >= _CRAMPED_INSET_PX or worst_inset < _OVERFLOW_SLACK_PX:
            continue
        touching = worst_inset <= _TOUCH_PX
        yield VisualIssue(
            fundamental="white-space",
            check_id=(
                "spacing.edge-touching"
                if touching
                else "spacing.cramped-container"
            ),
            title=(
                "Content pressed against container edge"
                if touching
                else "Container padding too tight for its content"
            ),
            description=(
                "Text inside this bordered/filled container sits only "
                f"{round(worst_inset, 1)}px from its {worst_side} edge. "
                "White space frames content; without it the block reads as "
                "cramped and harder to scan. Increase inner padding so text "
                "keeps clear of the container boundary."
            ),
            severity="critical" if touching else "medium",
            evidence={
                "container": snapshot.selector(c),
                "side": worst_side,
                "minInsetPx": round(worst_inset, 1),
                "textBlocks": len(texts),
            },
            element_refs=(snapshot.selector(c), snapshot.selector(worst_node)),
        )


def _find_section_collisions(
    snapshot: Snapshot, scan: _Scan
) -> Iterator[VisualIssue]:
    """Stacked multi-line text masses separated by (near) zero gap.

    Uses the rendered text extent of each sibling, so a container whose own
    padding already provides visible separation is not penalized.
    """
    for parent in snapshot.nodes:
        kids = sorted(
            scan.visible_children(parent.i), key=lambda k: (k.box.y, k.box.x)
        )
        for a, b in pairwise(kids):
            overlap = min(a.box.x + a.box.w, b.box.x + b.box.w) - max(
                a.box.x, b.box.x
            )
            if overlap < _PAIR_MIN_OVERLAP_PX:
                continue
            if _in_navigation(scan, a) or _in_navigation(scan, b):
                continue
            span_a = _content_span_y(scan, a)
            span_b = _content_span_y(scan, b)
            if span_a is None or span_b is None:
                continue
            gap = span_b[0] - span_a[1]
            if gap < -_COLLISION_SLACK_PX or gap >= _COLLISION_MAX_GAP_PX:
                continue
            if not (
                _is_multiline_text_mass(scan, a)
                and _is_multiline_text_mass(scan, b)
            ):
                continue
            critical = gap <= _COLLISION_CRITICAL_PX
            yield VisualIssue(
                fundamental="white-space",
                check_id="spacing.zero-gap-sections",
                title="Sections butt together with no breathing room",
                description=(
                    "Two multi-line text blocks are separated by only "
                    f"{round(max(gap, 0.0), 1)}px vertically, so they read "
                    "as one merged mass instead of distinct sections. "
                    "Separate related blocks with clear vertical white "
                    "space (or a divider) so each section is parseable."
                ),
                severity="critical" if critical else "medium",
                evidence={"gapPx": round(gap, 1)},
                element_refs=(snapshot.selector(a), snapshot.selector(b)),
            )


def _find_erratic_sibling_gaps(
    snapshot: Snapshot,
    scan: _Scan,
    groups: list[tuple[int, str, list[SNode]]],
) -> Iterator[VisualIssue]:
    """Like-for-like siblings separated by wildly varying gaps.

    Compares DOM-adjacent members of a same-tag sibling run, so class
    variants of one component do not open phantom gaps between samples.
    """
    for _, tag, members in groups:
        # Filter tiny decorative elements and huge full-page sections
        filtered = [m for m in members if not _is_too_small(m) and m.box.h < 600 and m.box.w < 1200]
        if len(filtered) < _RHYTHM_MIN_MEMBERS:
            continue
        # Use filtered for gap calc, but keep original for evidence if needed
        axis, gaps = _dominant_axis_gaps(filtered)
        if len(gaps) < _RHYTHM_MIN_GAPS:
            continue
        smallest, largest = min(gaps), max(gaps)
        if largest - smallest < _RHYTHM_SPREAD_PX:
            continue
        if largest < 2.5 * smallest + 8.0:
            continue
        # For large groups (e.g., 8 sections), require at least 2 inconsistent gaps
        # to avoid flagging a single intentional section break as a rhythm error
        if len(filtered) >= 6 and gaps.count(smallest) >= len(gaps) - 1:
            # Only one outlier among many consistent gaps → likely intentional
            continue
        yield VisualIssue(
            fundamental="white-space",
            check_id="spacing.erratic-sibling-gaps",
            title="Spacing between similar items is inconsistent",
            description=(
                f"The <{tag}> items in this group are separated by gaps "
                f"ranging from {round(smallest, 1)}px to {round(largest, 1)}px. "
                "Like elements should share one spacing rhythm; uneven gaps "
                "make the layout feel arbitrary and slow scanning down. "
                "Pick a single gap size (from a spacing scale) for all items."
            ),
            severity="critical" if smallest <= _TOUCH_PX else "medium",
            evidence={
                "axis": axis,
                "gapsPx": [round(g, 1) for g in gaps],
                "members": len(members),
            },
            element_refs=tuple(snapshot.selector(m) for m in members),
        )


def _find_packed_records(
    snapshot: Snapshot,
    scan: _Scan,
    groups: list[tuple[int, str, list[SNode]]],
) -> Iterator[VisualIssue]:
    """Record runs where every item sits almost flush against the next."""
    for _, tag, members in groups:
        records = [
            m
            for m in members
            if m.box.h >= _DENSITY_MIN_RECORD_H
            and (
                scan.block_text_len(m.i) >= _BLOCK_MIN_TEXT
                or len(scan.visible_text_nodes(m.i)) >= 2
            )
        ]
        if len(records) < _DENSITY_MIN_RECORDS:
            continue
        axis, gaps = _dominant_axis_gaps(records)
        if len(gaps) < _DENSITY_MIN_GAPS:
            continue
        if not all(-0.5 <= g <= _DENSITY_MAX_GAP_PX for g in gaps):
            continue
        border_side = "bottom" if axis == "vertical" else "right"
        if any(
            (m.style_px(f"border-{border_side}-width") or 0) > 0 for m in records
        ):
            continue  # divider borders already separate the records
        yield VisualIssue(
            fundamental="white-space",
            check_id="spacing.packed-records",
            title="List items packed together without separation",
            description=(
                f"{len(records)} <{tag}> records are stacked with at most "
                f"{round(_DENSITY_MAX_GAP_PX, 1)}px between neighbors and no "
                "dividers. Distinct records need white space (or rules) to "
                "read as separate entries; here they fuse into a wall. Add "
                "consistent inter-record gaps or separators."
            ),
            severity="medium",
            evidence={
                "records": len(records),
                "maxGapPx": round(max(gaps), 1),
            },
            element_refs=tuple(snapshot.selector(m) for m in records),
        )


def _find_tight_record_rhythm(
    snapshot: Snapshot,
    scan: _Scan,
    groups: list[tuple[int, str, list[SNode]]],
) -> Iterator[VisualIssue]:
    """Record runs whose gaps are uniformly too tight for their text.

    Unlike ``packed-records`` (which needs substantial multi-part rows),
    this catches short single-line records separated by a sliver of space:
    the row pitch leaves almost no air above/below each line of text.
    """
    for _, tag, members in groups:
        textual = [m for m in members if scan.block_text_len(m.i) > 0]
        if len(textual) < _RHYTHM_MIN_MEMBERS:
            continue
        axis, gaps = _dominant_axis_gaps(textual)
        if axis != "vertical" or len(gaps) < _RHYTHM_MIN_GAPS:
            continue
        line_heights = [_line_height(m) for m in textual]
        tallest_lh = max(line_heights)
        tight_height = [
            m
            for m, lh in zip(textual, line_heights, strict=True)
            if m.box.h <= _TIGHT_RECORD_RATIO * lh
        ]
        if len(tight_height) < len(textual) - 1:
            continue  # most records must be text-height slivers
        if not all(g <= _TIGHT_GAP_PX for g in gaps):
            continue
        border_side = "bottom"
        has_divider = any(
            (m.style_px(f"border-{border_side}-width") or 0) > 0 for m in textual
        )
        # A 1px divider with zero gap is still spacing by line, not white
        # space — UEye treats that as a failure. Only honour dividers when
        # they accompany real white-space gaps.
        if has_divider and max(gaps) > 0:
            continue  # divider borders already separate the records
        yield VisualIssue(
            fundamental="white-space",
            check_id="spacing.tight-record-rhythm",
            title="Rows packed tighter than their own text can breathe",
            description=(
                f"{len(textual)} <{tag}> rows are only "
                f"{round(max(gaps), 1)}px apart while their text needs "
                f"{round(tallest_lh, 1)}px of line height. The list reads "
                "as one dense block; give each row consistent breathing "
                "room so entries stay scannable."
            ),
            severity="medium",
            evidence={
                "rows": len(textual),
                "maxGapPx": round(max(gaps), 1),
                "lineHeightPx": round(tallest_lh, 1),
            },
            element_refs=tuple(snapshot.selector(m) for m in textual),
        )


def _find_sibling_text_collision(
    snapshot: Snapshot, scan: _Scan
) -> Iterator[VisualIssue]:
    """Adjacent siblings whose rendered boxes physically overlap.

    Overlapping siblings mean content collides or margins collapsed into
    each other; either way elements do not get the separation white space
    should provide. Absolutely-positioned overlays are exempt.
    """
    for parent in snapshot.nodes:
        kids = sorted(
            (
                k
                for k in scan.visible_children(parent.i)
                if scan.block_text_len(k.i) > 0
                and k.style("position") not in ("absolute", "fixed")
            ),
            key=lambda k: (k.box.y, k.box.x),
        )
        for a, b in pairwise(kids):
            x_overlap = min(a.box.x + a.box.w, b.box.x + b.box.w) - max(
                a.box.x, b.box.x
            )
            y_overlap = min(a.box.y + a.box.h, b.box.y + b.box.h) - max(
                a.box.y, b.box.y
            )
            if y_overlap < _COLLISION_MIN_OVERLAP_PX:
                continue
            if x_overlap < _PAIR_MIN_OVERLAP_PX:
                continue
            if _in_navigation(scan, a) or _in_navigation(scan, b):
                continue
            critical = y_overlap >= _COLLISION_CRITICAL_PX2
            yield VisualIssue(
                fundamental="white-space",
                check_id="spacing.sibling-overlap",
                title="Elements overlap instead of occupying their own space",
                description=(
                    "Two sibling blocks overlap vertically by "
                    f"{round(y_overlap, 1)}px, so their content collides. "
                    "Every element deserves its own footprint; fix the "
                    "layout (margins, line-height, or container sizing) so "
                    "nothing is painted on top of its neighbor."
                ),
                severity="critical" if critical else "medium",
                evidence={
                    "overlapPx": round(y_overlap, 1),
                    "parent": snapshot.selector(parent),
                },
                element_refs=(snapshot.selector(a), snapshot.selector(b)),
            )


def _find_child_overflow(
    snapshot: Snapshot, scan: _Scan
) -> Iterator[VisualIssue]:
    """Avatar imagery escaping its card's bounds.

    Circular avatars that sit half-outside their card signal a framing
    failure: the composition cannot contain its own imagery, so the card
    appears cropped. Generic: an ``<img>`` with a circular border-radius
    extending past its parent's top edge.
    """
    for node in snapshot.nodes:
        if node.parent < 0 or not scan.visible[node.i]:
            continue
        if node.tag != "img":
            continue
        if node.style("position") in ("absolute", "fixed"):
            continue
        parent = scan.by_index.get(node.parent)
        if parent is None or not scan.visible[parent.i]:
            continue
        if _in_navigation(scan, node):
            continue
        radius = node.style("border-radius").lower()
        if "50%" not in radius and "999" not in radius:
            # also accept the computed 40px for a 80px avatar
            w = node.box.w
            try:
                rad_px = float(radius.removesuffix("px").strip().split()[0])
            except Exception:
                continue
            if abs(rad_px - w / 2) > 2:
                continue
        top_out = parent.box.y - node.box.y
        bottom_out = (node.box.y + node.box.h) - (parent.box.y + parent.box.h)
        worst = max(top_out, bottom_out)
        if worst < _OVERFLOW_MIN_PX:
            continue
        side = "top" if top_out >= bottom_out else "bottom"
        yield VisualIssue(
            fundamental="white-space",
            check_id="spacing.child-overflow",
            title="Avatar escapes its card's bounds",
            description=(
                f"This circular avatar extends {round(worst, 1)}px past its "
                f"container's {side} edge, so the card cannot frame it: "
                "the imagery appears cropped or colliding with the section "
                "above. Reserve headroom for the avatar instead of letting "
                "it bleed out of the card."
            ),
            severity="critical" if worst >= 24 else "medium",
            evidence={
                "overflowPx": round(worst, 1),
                "side": side,
                "parent": snapshot.selector(parent),
            },
            element_refs=(snapshot.selector(node), snapshot.selector(parent)),
        )


def _find_flush_top_dead_bottom(
    snapshot: Snapshot, scan: _Scan
) -> Iterator[VisualIssue]:
    """Tall containers pinned shut at the top and hollow at the bottom.

    When the first child touches the container's content top while a large
    dead zone is left at the padded bottom, vertical distribution is broken:
    white space accumulates in one place instead of separating content.
    """
    for c in snapshot.nodes:
        if c.parent < 0 or not scan.visible[c.i] or _in_navigation(scan, c):
            continue
        if c.box.h < _TALL_CONTAINER_PX:
            continue
        kids = scan.visible_children(c.i)
        if len(kids) < 2:
            continue
        pad_top = c.style_px("padding-top") or 0.0
        pad_bottom = c.style_px("padding-bottom") or 0.0
        first_top = min(k.box.y for k in kids)
        last_bottom = max(k.box.y + k.box.h for k in kids)
        flush_gap = first_top - (c.box.y + pad_top)
        leftover = (c.box.y + c.box.h - pad_bottom) - last_bottom
        if flush_gap > _FLUSH_TOP_PX or leftover < _DEAD_BOTTOM_PX:
            continue
        yield VisualIssue(
            fundamental="white-space",
            check_id="spacing.flush-top-dead-bottom",
            title="Content pinned to the top with dead space below",
            description=(
                "This tall container starts its first element flush at the "
                f"top while leaving {round(leftover, 1)}px of empty space at "
                "the bottom. White space should separate content evenly, not "
                "pool at one end; distribute padding so the composition is "
                "balanced."
            ),
            severity="medium",
            evidence={
                "container": snapshot.selector(c),
                "flushTopPx": round(flush_gap, 1),
                "deadBottomPx": round(leftover, 1),
            },
            element_refs=(snapshot.selector(c),),
        )

