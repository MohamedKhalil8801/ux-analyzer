"""Hierarchy & alignment piece.

Flags broken shared edges between same-role sibling blocks, ragged mixed
text alignment, overlapping blocks, flat visual hierarchies without a focal
point, inverted emphasis (secondary text larger than its primary label),
and undifferentiated uniform text blocks. All thresholds are generic
(no per-site knowledge).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from statistics import median

from ux_analyzer.analysis.visual.snapshot import Snapshot, SNode
from ux_analyzer.analysis.visual.types import VisualIssue

CHECK_EDGE_DRIFT = "alignment.edge-drift"
CHECK_TOP_DRIFT = "alignment.row-top-drift"
CHECK_MIXED_ALIGN = "alignment.mixed-text-align"
CHECK_OVERLAP = "alignment.sibling-overlap"
CHECK_FLAT_TITLE = "visual-hierarchy.flat-title"
CHECK_INVERTED = "visual-hierarchy.inverted-emphasis"
CHECK_UNIFORM = "visual-hierarchy.uniform-block"

_CLUSTER_TOL = 4.0  # px tolerance grouping identical edges
_ROW_DRIFT = 8.0  # px shared-edge tolerance between sibling rows
_SECTION_DRIFT = 12.0  # px tolerance for major full-width sections
_CRITICAL_DRIFT = 80.0  # px offset that means a broken layout
_OVERLAP_FRAC = 0.25  # intersect/min-area fraction meaning real overlap
_TITLE_RATIO = 1.15  # max/body size ratio below which a title is "flat"
_EMPHASIS_RATIO = 1.15  # meta/primary size ratio meaning "larger"
_MIN_GROUP = 3  # smallest sibling group worth judging
_HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_MAX_ISSUES = 12


def _fw(node: SNode) -> int:
    raw = node.style("font-weight").strip().lower()
    if raw == "bold":
        return 700
    if raw == "normal" or not raw:
        return 400
    try:
        return int(float(raw))
    except ValueError:
        return 400


def _visible(node: SNode) -> bool:
    if node.style("display") == "none" or node.style("visibility") == "hidden":
        return False
    if node.box.w <= 0 or node.box.h <= 0:
        return False
    # Hidden skip-links / drawers sit fully above or left of the viewport.
    if node.box.y < -20 or node.box.x < -20:
        return False
    try:
        if float(node.style("opacity") or 1) <= 0:
            return False
    except ValueError:
        pass
    return True


def _inline(node: SNode) -> bool:
    return node.style("display").startswith("inline")


def _abspos(node: SNode) -> bool:
    return node.style("position") in {"absolute", "fixed"}


def _align(node: SNode) -> str:
    a = (node.style("text-align") or "start").strip().lower()
    return {"left": "start", "right": "end"}.get(a, a)


def _color_key(node: SNode) -> str:
    return "".join(node.style("color").lower().split())


def _text_leaves(snapshot: Snapshot) -> list[SNode]:
    parents_with_text = {
        n.parent for n in snapshot.nodes if n.text.strip()
    }
    return [
        n
        for n in snapshot.nodes
        if n.text.strip() and n.i not in parents_with_text and _visible(n)
    ]


def _cluster_edges(
    values: list[float],
) -> list[tuple[float, int]]:  # (mean, count), ascending
    clusters: list[list[float]] = []
    for v in sorted(values):
        if clusters and v - clusters[-1][-1] <= _CLUSTER_TOL:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [(sum(c) / len(c), len(c)) for c in clusters]


def _outliers(
    values: list[float], thr: float
) -> tuple[float | None, list[float]]:
    """Return (dominant edge, values deviating from it by more than thr)."""
    clusters = _cluster_edges(values)
    dominant, _ = max(clusters, key=lambda c: (c[1], -c[0]))
    off = [v for v in values if abs(v - dominant) > thr]
    return dominant, off


def _drift_severity(delta: float, thr: float) -> str:
    if delta >= _CRITICAL_DRIFT:
        return "critical"
    if delta > thr * 2:
        return "medium"
    return "low"


def _role_groups(members: list[SNode]) -> list[list[SNode]]:
    groups: dict[str, list[SNode]] = {}
    for m in members:
        groups.setdefault(m.tag, []).append(m)
    return list(groups.values())


def _y_bands(members: list[SNode]) -> list[list[SNode]]:
    """Split vertically stacked members into visual rows (y bands)."""
    med_h = median(m.box.h for m in members)
    ordered = sorted(members, key=lambda m: m.box.y + m.box.h / 2)
    bands: list[list[SNode]] = []
    centers: list[float] = []
    for m in ordered:
        cy = m.box.y + m.box.h / 2
        if bands and abs(cy - centers[-1]) <= med_h / 2:
            bands[-1].append(m)
            centers[-1] = sum(
                b.box.y + b.box.h / 2 for b in bands[-1]
            ) / len(bands[-1])
        else:
            bands.append([m])
            centers.append(cy)
    return bands


def _overlap_issue(
    snapshot: Snapshot, group: list[SNode]
) -> VisualIssue | None:
    for a_idx, a in enumerate(group):
        for b in group[a_idx + 1 :]:
            ox = min(a.box.x + a.box.w, b.box.x + b.box.w) - max(
                a.box.x, b.box.x
            )
            oy = min(a.box.y + a.box.h, b.box.y + b.box.h) - max(
                a.box.y, b.box.y
            )
            if ox <= 0 or oy <= 0:
                continue
            smaller = min(a.box.w * a.box.h, b.box.w * b.box.h)
            if ox * oy / smaller > _OVERLAP_FRAC:
                return VisualIssue(
                    fundamental="alignment",
                    check_id=CHECK_OVERLAP,
                    title="Same-role blocks overlap",
                    description=(
                        "Sibling elements sharing the same role occupy the "
                        "same space, so content collides instead of stacking "
                        "in a clear layout."
                    ),
                    severity="critical",
                    evidence={
                        "overlap_px": round(min(ox, oy), 1),
                        "overlap_fraction": round(ox * oy / smaller, 2),
                    },
                    element_refs=(
                        snapshot.selector(a),
                        snapshot.selector(b),
                    ),
                )
    return None


def _edge_drift_issues(
    snapshot: Snapshot, parent: SNode, group: list[SNode], axis: str
) -> list[VisualIssue]:
    thr = _ROW_DRIFT
    if parent.depth <= 1:
        widths = [m.box.w for m in group]
        if widths and median(widths) >= 0.5 * snapshot.root_box.w:
            thr = _SECTION_DRIFT
    values = [getattr(m.box, axis) for m in group]
    spread = max(values) - min(values)
    issues: list[VisualIssue] = []
    if spread <= thr:
        return issues
    dominant, off = _outliers(values, thr)
    if dominant is None or not off:
        return issues
    dominant_count = max(c for _, c in _cluster_edges(values))
    if dominant_count < 2 and len(off) < 2:
        return issues
    offenders = [
        m for m in group if abs(getattr(m.box, axis) - dominant) > thr
    ]
    # An outlier that also differs in size is deliberate emphasis (e.g. a
    # raised feature card), not broken alignment.
    kept = [m for m in group if abs(getattr(m.box, axis) - dominant) <= thr]
    ref_w = median([m.box.w for m in kept]) if kept else 0.0
    ref_h = median([m.box.h for m in kept]) if kept else 0.0

    def _resized(m: SNode) -> bool:
        return (
            abs(m.box.w - ref_w) > 0.08 * max(ref_w, 1)
            or abs(m.box.h - ref_h) > 0.08 * max(ref_h, 1)
        )

    if all(_resized(m) for m in offenders):
        return issues
    offenders = [m for m in offenders if not _resized(m)]
    worst = max(abs(getattr(m.box, axis) - dominant) for m in offenders)
    axis_name = "left" if axis == "x" else "top"
    issues.append(
        VisualIssue(
            fundamental="alignment",
            check_id=CHECK_EDGE_DRIFT if axis == "x" else CHECK_TOP_DRIFT,
            title=f"Same-role blocks lose their shared {axis_name} edge",
            description=(
                f"Elements that play the same role start at different "
                f"{axis_name} positions (up to {worst:.0f}px apart), so the "
                "eye cannot follow a single alignment line down the page."
            ),
            severity=_drift_severity(worst, thr),
            evidence={
                "dominant_edge_px": round(dominant, 1),
                "max_offset_px": round(worst, 1),
                "tolerance_px": thr,
            },
            element_refs=tuple(snapshot.selector(m) for m in offenders),
        )
    )
    return issues


def _mixed_align_issue(
    snapshot: Snapshot, group: list[SNode]
) -> VisualIssue | None:
    with_text = [
        m for m in group if any(d.text.strip() for d in _subtree(snapshot, m))
    ]
    if len(with_text) < _MIN_GROUP:
        return None
    # Only "equivalent items" can disagree about alignment: siblings whose
    # rendered heights are in the same league AND small enough to be list
    # items / labels rather than whole page sections. One deliberately
    # centered section among start-aligned sections is design, not a
    # ragged column.
    heights = [m.box.h for m in with_text if m.box.h > 0]
    if not heights or max(heights) > 200.0:
        return None
    if min(heights) * 2.2 < max(heights):
        return None
    counts = Counter(_align(m) for m in with_text)
    if len(counts) < 2:
        return None
    align, top_count = counts.most_common(1)[0]
    if top_count < 2:
        return None
    oddballs = [m for m in with_text if _align(m) != align]
    if not oddballs:
        return None
    offender_align = Counter(_align(m) for m in oddballs).most_common(1)[0][0]
    return VisualIssue(
        fundamental="alignment",
        check_id=CHECK_MIXED_ALIGN,
        title="Mixed text alignment among equivalent items",
        description=(
            f"Equivalent items in the same stack mix '{align}' alignment "
            f"with '{offender_align}' alignment, producing a ragged column "
            "instead of one consistent reading edge."
        ),
        severity="medium",
        evidence={
            "majority_align": align,
            "offender_aligns": sorted({_align(m) for m in oddballs}),
        },
        element_refs=tuple(snapshot.selector(m) for m in oddballs),
    )


def _subtree(snapshot: Snapshot, node: SNode) -> list[SNode]:
    out = [node]
    stack = [node.i]
    while stack:
        cur = stack.pop()
        for n in snapshot.nodes:
            if n.parent == cur:
                out.append(n)
                stack.append(n.i)
    return out


def _is_vertical_stack(group: list[SNode]) -> bool:
    """True when no two members sit side by side on a shared row."""
    ordered = sorted(group, key=lambda m: m.box.y)
    for a, b in zip(ordered, ordered[1:]):
        overlap = (
            min(a.box.y + a.box.h, b.box.y + b.box.h) - max(a.box.y, b.box.y)
        )
        if overlap > 0.5 * min(a.box.h, b.box.h):
            return False
    return True


def _is_in_nav_landmark(snapshot: Snapshot, node: SNode) -> bool:
    """True when the node lives under a nav/header landmark."""
    by_index = {n.i: n for n in snapshot.nodes}
    cur: SNode | None = node
    while cur is not None:
        if cur.tag in {"nav", "header"}:
            return True
        cur = by_index.get(cur.parent) if cur.parent >= 0 else None
    return False


def _is_too_small_for_layout(node: SNode) -> bool:
    """Filter decorative thin lines and tiny placeholders (e.g., 1×54 dividers)."""
    return node.box.w <= 2 or node.box.h <= 2 or node.box.w * node.box.h < 120

def _cross_tag_vertical_drift(snapshot: Snapshot) -> list[VisualIssue]:
    """Major block children of one container that should share a left edge.

    Catches headings, lists and form rows whose left edges drift apart
    inside the same card/section (e.g., a heading 16px offset from the
    list below it, or a label 100px offset from its input).
    """
    issues: list[VisualIssue] = []
    for parent in snapshot.nodes:
        kids = [
            c
            for c in snapshot.children(parent.i)
            if _visible(c) and not _abspos(c) and c.style("display").strip().lower() != "inline" and not _is_too_small_for_layout(c)
        ]
        if len(kids) < 2:
            continue
        # centered stacks naturally have varying left edges (different widths)
        parent_align = (parent.style("text-align") or "start").strip().lower()
        if parent_align in {"center", "centre"}:
            continue
        if any((c.style("text-align") or "start").strip().lower() == "center" for c in kids):
            # if any child is centered, left variance is not meaningful
            continue
        # full-bleed children that are flush with parent are intentional
        # (e.g., a chart spanning the section while text is padded)
        def _is_flush_fullbleed(child: SNode) -> bool:
            return child.box.w >= 0.95 * max(parent.box.w, 1) and abs(child.box.x - parent.box.x) <= 4

        filtered = [c for c in kids if not _is_flush_fullbleed(c)]
        if len(filtered) < 2:
            # need at least two non-full-bleed kids to judge alignment
            # but allow one flush + one inset (e.g., label vs input) as a pair
            # so keep original kids if filtered is too small but original has mix
            if len(kids) >= 2 and any(_is_flush_fullbleed(c) for c in kids) and any(not _is_flush_fullbleed(c) for c in kids):
                # mixed case: keep all kids to detect flush vs inset drift
                pass
            else:
                if len(filtered) < 2:
                    continue
                kids = filtered
        else:
            kids = filtered
        # only vertical stacks (not side-by-side card rows)
        if not _is_vertical_stack(kids):
            continue
        # left edges should be within _ROW_DRIFT for a well-aligned stack
        xs = [c.box.x for c in kids]
        spread = max(xs) - min(xs)
        if spread <= _ROW_DRIFT:
            continue
        # ignore tiny parents and navigation landmarks
        if parent.tag in {"nav", "header"}:
            continue
        if snapshot.root_box.w > 0 and parent.box.w < 0.3 * snapshot.root_box.w:
            # small inner wrappers (form-el) still matter — don't skip
            pass
        dominant, off = _outliers(xs, _ROW_DRIFT)
        if not off or dominant is None:
            continue
        offenders = [c for c in kids if abs(c.box.x - dominant) > _ROW_DRIFT]
        worst = max(abs(c.box.x - dominant) for c in offenders)
        issues.append(
            VisualIssue(
                fundamental="alignment",
                check_id="alignment.cross-tag-drift",
                title="Stacked blocks lose their shared left edge",
                description=(
                    "Blocks that are stacked vertically in the same container"
                    f" start {worst:.0f}px apart horizontally, so the column"
                    " has no single alignment line."
                ),
                severity=_drift_severity(worst, _ROW_DRIFT),
                evidence={"max_offset_px": round(worst, 1)},
                element_refs=tuple(snapshot.selector(c) for c in offenders),
            )
        )
    return issues


def _card_bottom_drift(snapshot: Snapshot) -> list[VisualIssue]:
    """Sibling cards whose bottom edges don't line up *within one row*.

    Catches pricing-style decks where the featured card is taller and its
    bottom drops below its siblings. Cards in DIFFERENT rows of a grid
    naturally end at different heights — comparing across rows produced
    systematic false positives, so each visual row is judged separately.
    """
    issues: list[VisualIssue] = []
    for parent in snapshot.nodes:
        members = [
            c for c in snapshot.children(parent.i) if _visible(c) and not _abspos(c)
        ]
        for group in _role_groups(members):
            if len(group) < 3:
                continue
            if _is_vertical_stack(group):
                continue
            # Judge each visual row separately: bottoms can only drift when
            # the cards actually sit side by side.
            for band in _y_bands(group):
                if len(band) < 2:
                    continue
                bottoms = [c.box.y + c.box.h for c in band]
                spread = max(bottoms) - min(bottoms)
                if spread <= _ROW_DRIFT:
                    continue
                dominant, off = _outliers(bottoms, _ROW_DRIFT)
                if not off or dominant is None:
                    continue
                worst = max(abs(b - dominant) for b in off)
                if worst < 20:
                    continue
                issues.append(
                    VisualIssue(
                        fundamental="alignment",
                        check_id="alignment.card-bottom-drift",
                        title="Sibling cards bottoms don't line up",
                        description=(
                            "Cards that are meant to sit as a row end up to"
                            f" {worst:.0f}px apart at the bottom, so the deck"
                            " looks uneven."
                        ),
                        severity=_drift_severity(worst, _ROW_DRIFT),
                        evidence={"max_offset_px": round(worst, 1)},
                        element_refs=tuple(snapshot.selector(c) for c in band),
                    )
                )
    return issues


def _footer_inset_inconsistency(snapshot: Snapshot) -> list[VisualIssue]:
    """Footer columns where item inset vs column differs.

    Restricted to real footer landmarks: comparing inset across unrelated
    sibling widgets (e.g., a chip grid next to a stacked result list)
    produced false positives on non-footer layouts.
    """
    issues: list[VisualIssue] = []
    for parent in snapshot.nodes:
        # Only judge genuine footer contexts: the same inset-mismatch among
        # unrelated sibling widgets elsewhere is by design.
        if parent.tag != "footer" and not _has_footer_ancestor(snapshot, parent):
            continue
        # find sibling column containers (e.g., footer-left + foot-nav uls)
        cols = [
            c for c in snapshot.children(parent.i) if _visible(c) and not _abspos(c)
        ]
        if len(cols) < 3:
            continue
        # need at least two list-like columns
        lists = [c for c in cols if c.tag in {"ul", "ol"}]
        if len(lists) < 2:
            continue
        # compute left inset of first text item vs its column
        insets: list[float] = []
        for col in lists:
            items = [
                k for k in snapshot.children(col.i) if _visible(k) and k.text.strip() == "" and k.box.w > 0
            ]
            # actual text leaves inside li
            leaves = _text_leaves(snapshot)
            # find leaves whose ancestor is this col
            col_leaves = [leaf for leaf in leaves if _is_descendant(snapshot, leaf, col)]
            if col_leaves:
                first = min(col_leaves, key=lambda n: n.box.y)
                insets.append(first.box.x - col.box.x)
            elif items:
                first = min(items, key=lambda n: n.box.y)
                insets.append(first.box.x - col.box.x)
        if len(insets) < 2:
            continue
        spread = max(insets) - min(insets)
        if spread <= _CLUSTER_TOL:
            continue
        if spread < 20:
            continue
        issues.append(
            VisualIssue(
                fundamental="alignment",
                check_id="alignment.column-inset-drift",
                title="Footer columns indent their items inconsistently",
                description=(
                    "Items inside sibling footer columns are inset"
                    f" {min(insets):.0f}px in one column but"
                    f" {max(insets):.0f}px in another, so the columns"
                    " don't share a common grid."
                ),
                severity="medium",
                evidence={"insets_px": [round(v, 1) for v in insets]},
                element_refs=tuple(snapshot.selector(c) for c in lists),
            )
        )
    return issues


def _has_footer_ancestor(snapshot: Snapshot, node: SNode) -> bool:
    cur: SNode | None = node
    while cur is not None:
        if cur.tag == "footer":
            return True
        cur = (
            next((n for n in snapshot.nodes if n.i == cur.parent), None)
            if cur.parent >= 0
            else None
        )
    return False


def _is_descendant(snapshot: Snapshot, node: SNode, ancestor: SNode) -> bool:
    cur: SNode | None = node
    while cur is not None and cur.parent >= 0:
        if cur.parent == ancestor.i:
            return True
        cur = next((n for n in snapshot.nodes if n.i == cur.parent), None)
    return False


def _deck_uniform_text_issue(snapshot: Snapshot) -> VisualIssue | None:
    """Deck of cards where quote and attribution share identical styling."""
    for parent in snapshot.nodes:
        cards = [
            c for c in snapshot.children(parent.i) if _visible(c) and c.box.w > 80 and c.box.h > 80
        ]
        if len(cards) < 3:
            continue
        # need cards of same tag, similar size
        tags = {c.tag for c in cards}
        if len(tags) != 1:
            continue
        # collect text leaves per card
        leaves_per_card: list[list[SNode]] = []
        all_ok = True
        for card in cards:
            ls = [n for n in _text_leaves(snapshot) if _is_descendant(snapshot, n, card)]
            if len(ls) < 2:
                all_ok = False
                break
            leaves_per_card.append(ls)
        if not all_ok:
            continue
        # compare longest vs shortest leaf per card — they should differ in
        # size/weight/color to show hierarchy, but here they are identical
        # across all cards.
        flat = True
        for ls in leaves_per_card:
            by_len = sorted(ls, key=lambda n: len(n.text.strip()))
            shortest, longest = by_len[0], by_len[-1]
            if (
                _px_or(shortest.style("font-size"), 16.0)
                != _px_or(longest.style("font-size"), 16.0)
                or _fw(shortest) != _fw(longest)
                or _color_key(shortest) != _color_key(longest)
            ):
                flat = False
                break
        if not flat:
            continue
        # also require the deck's leaves all share one style
        all_leaves = [n for ls in leaves_per_card for n in ls]
        first_key = (
            _px_or(all_leaves[0].style("font-size"), 16.0),
            _fw(all_leaves[0]),
            _color_key(all_leaves[0]),
        )
        if any(
            (_px_or(n.style("font-size"), 16.0), _fw(n), _color_key(n)) != first_key
            for n in all_leaves[1:]
        ):
            continue
        return VisualIssue(
            fundamental="visual-hierarchy",
            check_id="visual-hierarchy.deck-uniform",
            title="Card deck text carries equal weight throughout",
            description=(
                "Every card's body and attribution share the same size,"
                " weight and color even though the body is much longer; no"
                " card establishes a clear emphasis, so the deck looks flat."
            ),
            severity="medium",
            evidence={"cards": len(cards), "font_px": first_key[0]},
            element_refs=tuple(snapshot.selector(c) for c in cards),
        )
    return None


def _alignment_issues(snapshot: Snapshot) -> list[VisualIssue]:
    issues: list[VisualIssue] = []
    for parent in snapshot.nodes:
        members = [
            c
            for c in snapshot.children(parent.i)
            if _visible(c) and not _inline(c) and not _abspos(c)
        ]
        for group in _role_groups(members):
            if len(group) < _MIN_GROUP:
                continue
            ov = _overlap_issue(snapshot, group)
            if ov is not None:
                issues.append(ov)
            if _is_vertical_stack(group):
                issues.extend(_edge_drift_issues(snapshot, parent, group, "x"))
                ma = _mixed_align_issue(snapshot, group)
                if ma is not None:
                    issues.append(ma)
                continue
            bands = _y_bands(group)
            if len(bands) <= 1:
                # side-by-side row: judge the shared top edge instead
                issues.extend(_edge_drift_issues(snapshot, parent, group, "y"))
                continue
            # wrapped grid: compare each visual row's starting edge
            starts = [
                min(m.box.x for m in band) for band in bands if len(band) >= 2
            ]
            if len(starts) < 2:
                continue
            spread = max(starts) - min(starts)
            thr = (
                _SECTION_DRIFT
                if parent.depth <= 1
                and median([m.box.w for m in group]) >= 0.5 * snapshot.root_box.w
                else _ROW_DRIFT
            )
            dominant, off = _outliers(starts, thr)
            if spread <= thr or not off or dominant is None:
                continue
            dominant_count = max(c for _, c in _cluster_edges(starts))
            if dominant_count < 2 and len(off) < 2:
                continue
            issues.append(
                VisualIssue(
                    fundamental="alignment",
                    check_id=CHECK_EDGE_DRIFT,
                    title="Wrapped rows start at inconsistent edges",
                    description=(
                        "Visual rows of the same role begin up to "
                        f"{spread:.0f}px apart horizontally, breaking the "
                        "container's alignment grid."
                    ),
                    severity=_drift_severity(spread, thr),
                    evidence={
                        "row_start_edges_px": [round(v, 1) for v in starts],
                        "tolerance_px": thr,
                    },
                    element_refs=tuple(
                        snapshot.selector(min(band, key=lambda m: m.box.x))
                        for band in bands
                        if len(band) >= 2
                    ),
                )
            )
    return issues


def _flat_title_issue(
    snapshot: Snapshot, leaves: list[SNode]
) -> VisualIssue | None:
    if len(leaves) < 5:
        return None
    sizes = [_px_or(n.style("font-size"), 16.0) for n in leaves]
    weights = [_fw(n) for n in leaves]
    size_counts = Counter(sizes)
    body_size = min(
        s for s, c in size_counts.items() if c == max(size_counts.values())
    )
    body_fw = Counter(
        w for w, s in zip(weights, sizes) if s == body_size
    ).most_common(1)[0][0]
    max_size = max(sizes)
    if body_size <= 0 or max_size / body_size >= _TITLE_RATIO:
        return None
    candidates = [
        n
        for n in leaves
        if n.tag in _HEADINGS
        or (_fw(n) >= 600 and len(n.text.strip()) <= 80)
    ]
    if not candidates:
        return None
    strongest = max(
        (
            _fw(n)
            for n in leaves
            if _px_or(n.style("font-size"), 16.0) == max_size
        ),
        default=400,
    )
    if abs(strongest - body_fw) > 100:
        return None
    peers = sum(1 for s in sizes if s >= max_size * 0.93)
    if peers < 3:
        return None
    order = {leaf.i: k for k, leaf in enumerate(leaves)}
    at_body = [
        n for n in candidates if _px_or(n.style("font-size"), 16.0) >= body_size
    ]
    title = min(at_body or candidates, key=lambda n: order[n.i])
    refs = (snapshot.selector(title),)
    return VisualIssue(
        fundamental="visual-hierarchy",
        check_id=CHECK_FLAT_TITLE,
        title="Page lacks a clear focal point (flat title)",
        description=(
            "The largest text on the page is barely bigger than surrounding "
            "body text and carries similar weight, so no element reads as "
            "the primary message."
        ),
        severity="medium",
        evidence={
            "max_font_px": max_size,
            "body_font_px": body_size,
            "ratio": round(max_size / body_size, 3) if body_size else None,
        },
        element_refs=refs,
    )


def _px_or(raw: str, default: float) -> float:
    v = raw.strip().lower()
    if v.endswith("px"):
        try:
            return float(v[:-2])
        except ValueError:
            return default
    if v.endswith("rem") or v.endswith("em"):
        try:
            return float(v.removesuffix("rem").removesuffix("em")) * 16.0
        except ValueError:
            return default
    return default


def _inverted_emphasis_issues(
    snapshot: Snapshot, flagged: set[int]
) -> list[VisualIssue]:
    issues: list[VisualIssue] = []
    leaf_order = {n.i: k for k, n in enumerate(snapshot.nodes)}
    text_parents = {
        n.parent for n in snapshot.nodes if n.text.strip()
    }
    for unit in snapshot.nodes:
        kids = [c for c in snapshot.children(unit.i) if _visible(c)]
        reps: list[tuple[SNode, SNode]] = []
        for kid in kids:
            texts = [
                t
                for t in _subtree(snapshot, kid)
                if t.text.strip() and t.i not in text_parents
            ]
            if texts:
                first = min(texts, key=lambda t: leaf_order[t.i])
                reps.append((kid, first))
        for a_idx in range(len(reps)):
            _, primary = reps[a_idx]
            if _is_in_nav_landmark(snapshot, primary):
                continue
            ps = _px_or(primary.style("font-size"), 16.0)
            for b_idx in range(a_idx + 1, len(reps)):
                _, secondary = reps[b_idx]
                if secondary.i in flagged:
                    continue
                if _is_in_nav_landmark(snapshot, secondary):
                    continue
                ss = _px_or(secondary.style("font-size"), 16.0)
                if ss <= ps * _EMPHASIS_RATIO or ss - ps < 2:
                    continue
                # only metadata-looking text counts as "secondary": short,
                # non-heading support content such as timestamps or bylines.
                sec_words = len(secondary.text.split())
                if (
                    secondary.tag in _HEADINGS
                    or (sec_words > 5 and len(secondary.text.strip()) > 24)
                ):
                    continue
                # If some other element in the unit is both larger and at
                # least as heavy, the unit already has a real focal point
                # and the small bold first node is just an eyebrow label.
                if any(
                    _px_or(t.style("font-size"), 16.0) >= ss
                    and _fw(t) >= _fw(primary)
                    for k, (_, t) in enumerate(reps)
                    if k not in (a_idx, b_idx)
                ):
                    continue
                demoted = (
                    _fw(secondary) < _fw(primary)
                    or _opacity(secondary) <= 0.95
                )
                looks_primary = (
                    _fw(primary) >= 500
                    or primary.tag in _HEADINGS
                    or a_idx == 0
                )
                if not demoted or not looks_primary:
                    continue
                flagged.add(secondary.i)
                issues.append(
                    VisualIssue(
                        fundamental="visual-hierarchy",
                        check_id=CHECK_INVERTED,
                        title="Secondary text outweighs its primary label",
                        description=(
                            "Supporting/meta text is rendered noticeably "
                            "larger than the primary label it belongs to, "
                            "inverting the emphasis order."
                        ),
                        severity="medium",
                        evidence={
                            "primary_font_px": ps,
                            "secondary_font_px": ss,
                            "primary_weight": _fw(primary),
                            "secondary_weight": _fw(secondary),
                        },
                        element_refs=(
                            snapshot.selector(primary),
                            snapshot.selector(secondary),
                        ),
                    )
                )
                break
    return issues


def _opacity(node: SNode) -> float:
    try:
        return float(node.style("opacity") or 1)
    except ValueError:
        return 1.0


def _uniform_block_issues(
    snapshot: Snapshot, leaves: list[SNode]
) -> list[VisualIssue]:
    issues: list[VisualIssue] = []
    groups: dict[int, list[SNode]] = {}
    for leaf in leaves:
        groups.setdefault(leaf.parent, []).append(leaf)
    for members in groups.values():
        if len(members) < 4:
            continue
        first = members[0]
        key = (
            _px_or(first.style("font-size"), 16.0),
            _fw(first),
            _color_key(first),
        )
        if any(
            (_px_or(m.style("font-size"), 16.0), _fw(m), _color_key(m)) != key
            for m in members[1:]
        ):
            continue
        lengths = [len(m.text.strip()) for m in members]
        if min(lengths) < 12 or max(lengths) < 3 * min(lengths):
            continue
        issues.append(
            VisualIssue(
                fundamental="visual-hierarchy",
                check_id=CHECK_UNIFORM,
                title="List items carry equal weight throughout",
                description=(
                    "Four or more sibling text blocks share identical size, "
                    "weight and color while their content lengths differ "
                    "widely, leaving the list without any point of entry."
                ),
                severity="low",
                evidence={
                    "item_count": len(members),
                    "font_px": key[0],
                    "weight": key[1],
                },
                element_refs=tuple(snapshot.selector(m) for m in members),
            )
        )
    return issues


def analyze_hierarchy(snapshot: Snapshot) -> Sequence[VisualIssue]:
    """Detect misapplied alignment and visual hierarchy."""
    leaves = _text_leaves(snapshot)
    issues: list[VisualIssue] = []
    issues.extend(_alignment_issues(snapshot))
    issues.extend(_cross_tag_vertical_drift(snapshot))
    issues.extend(_card_bottom_drift(snapshot))
    issues.extend(_footer_inset_inconsistency(snapshot))
    flat = _flat_title_issue(snapshot, leaves)
    if flat is not None:
        issues.append(flat)
    issues.extend(_inverted_emphasis_issues(snapshot, set()))
    issues.extend(_uniform_block_issues(snapshot, leaves))
    deck = _deck_uniform_text_issue(snapshot)
    if deck is not None:
        issues.append(deck)
    severity_rank = {"critical": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda i: severity_rank.get(i.severity, 3))
    return issues[:_MAX_ISSUES]
