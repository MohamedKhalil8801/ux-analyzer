"""Color & contrast piece.

Detects misapplied contrast and color fundamentals in rendered-page
snapshots: text/background pairs failing WCAG 2.x thresholds (with alpha
blending and opacity compositing for effective colors), accent palettes
scattered across too many unrelated saturated hues, state carried by
red/green text alone, and same-role elements split across near-identical
grays. Every threshold is a generic WCAG/design floor; nothing is
site-specific.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence

from ux_analyzer.analysis.visual.snapshot import Snapshot, SNode
from ux_analyzer.analysis.visual.types import VisualIssue

_MAX_REFS = 8
_EPS = 1e-9

_AA_NORMAL = 4.5  # WCAG AA for regular-size text
_AA_LARGE = 3.0  # WCAG AA for large text
_LARGE_PX = 24.0  # >= 18pt
_LARGE_BOLD_PX = 18.66  # ~14pt bold
_BOLD_WEIGHT = 700.0
_DEFAULT_FONT_PX = 16.0

_ACCENT_MIN_SAT = 0.35
_ACCENT_MIN_VAL = 0.25
_ACCENT_MIN_ALPHA = 0.5
_HUE_GAP_DEG = 30.0
_MAX_ACCENT_HUES = 4

_STATE_MIN_SAT = 0.30
_STATE_MIN_VAL = 0.20
_STATE_MIN_ALPHA = 0.85
_RED_HUES: tuple[tuple[float, float], ...] = ((0.0, 20.0), (345.0, 360.0))
_GREEN_HUES: tuple[tuple[float, float], ...] = ((70.0, 160.0),)

_GRAY_MAX_SAT = 0.08
_GRAY_MIN_ALPHA = 0.90
_NEAR_GRAY_LUM_DELTA = 0.05

_RGB_RE = re.compile(r"^rgba?\(([^)]*)\)$")
_HEX_DIGITS = frozenset("0123456789abcdef")
_NAMED_WEIGHTS = {"bold": 700.0, "normal": 400.0}


def analyze_color(snapshot: Snapshot) -> Sequence[VisualIssue]:
    """Detect contrast and color misapplication in a visual snapshot."""
    issues: list[VisualIssue] = []
    opacity = _cumulative_opacity(snapshot)
    issues.extend(_contrast_issues(snapshot, opacity))
    issues.extend(_accent_chaos_issue(snapshot))
    issues.extend(_state_color_issue(snapshot))
    issues.extend(_near_gray_issue(snapshot))
    issues.extend(_two_tone_section_issue(snapshot))
    issues.extend(_white_card_on_white_issue(snapshot))
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
        element_refs=tuple(snapshot.selector(node) for node in hits[:_MAX_REFS]),
    )


def _parse_color(raw: str) -> tuple[float, float, float, float] | None:
    """Parse rgb()/rgba() (or hex) into (r, g, b, a); None when unknown."""
    value = raw.strip().lower()
    if not value:
        return None
    if value == "transparent":
        return (0.0, 0.0, 0.0, 0.0)
    if value.startswith("#"):
        digits = value[1:]
        if len(digits) == 3:
            digits = "".join(ch * 2 for ch in digits)
        if len(digits) not in {6, 8} or any(ch not in _HEX_DIGITS for ch in digits):
            return None
        rgb = [float(int(digits[k : k + 2], 16)) for k in (0, 2, 4)]
        alpha = (
            float(int(digits[6:8], 16)) / 255 if len(digits) == 8 else 1.0
        )
        return (*rgb, alpha)
    match = _RGB_RE.match(value)
    if match is None:
        return None
    parts = match.group(1).replace(",", " ").split()
    if len(parts) not in {3, 4}:
        return None
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return None
    rgb = [min(255.0, max(0.0, c)) for c in numbers[:3]]
    alpha = min(1.0, max(0.0, numbers[3])) if len(numbers) == 4 else 1.0
    return (*rgb, alpha)


def _blend(
    fg: tuple[float, float, float],
    alpha: float,
    bg: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Composite fg over bg at the given alpha, per channel."""
    return (
        fg[0] * alpha + bg[0] * (1 - alpha),
        fg[1] * alpha + bg[1] * (1 - alpha),
        fg[2] * alpha + bg[2] * (1 - alpha),
    )


def _srgb_channel(channel: float) -> float:
    scaled = channel / 255
    if scaled <= 0.04045:
        return scaled / 12.92
    return ((scaled + 0.055) / 1.055) ** 2.4


def _relative_luminance(rgb: tuple[float, float, float]) -> float:
    r, g, b = (_srgb_channel(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast_ratio(lum_a: float, lum_b: float) -> float:
    lighter = max(lum_a, lum_b)
    darker = min(lum_a, lum_b)
    return (lighter + 0.05) / (darker + 0.05)


def _hsv(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    r, g, b = (c / 255 for c in rgb)
    high, low = max(r, g, b), min(r, g, b)
    span = high - low
    if span <= 0:
        return (0.0, 0.0, high)
    if high == r:
        hue = ((g - b) / span) % 6
    elif high == g:
        hue = (b - r) / span + 2
    else:
        hue = (r - g) / span + 4
    return (hue * 60, span / high, high)


def _hue_distance(a: float, b: float) -> float:
    delta = abs(a - b) % 360
    return min(delta, 360 - delta)


def _opacity_value(node: SNode) -> float:
    raw = node.style("opacity").strip()
    if not raw:
        return 1.0
    try:
        value = float(raw)
    except ValueError:
        return 1.0
    return min(1.0, max(0.0, value))


def _cumulative_opacity(snapshot: Snapshot) -> dict[int, float]:
    """Product of opacity along each node's ancestor chain (self included)."""
    by_index = {node.i: node for node in snapshot.nodes}
    memo: dict[int, float] = {}

    def resolved(i: int) -> float:
        if i in memo:
            return memo[i]
        node = by_index[i]
        value = _opacity_value(node)
        if node.parent >= 0 and node.parent in by_index:
            value *= resolved(node.parent)
        memo[i] = value
        return value

    return {node.i: resolved(node.i) for node in snapshot.nodes}


def _visible(node: SNode) -> bool:
    if node.style("display").strip().lower() == "none":
        return False
    if node.style("visibility").strip().lower() in {"hidden", "collapse"}:
        return False
    # Hidden skip-links / drawers sit fully above or left of the viewport.
    if node.box.y < -20 or node.box.x < -20:
        return False
    return True


def _letters(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def _font_px(node: SNode) -> float:
    value = node.style_px("font-size")
    return value if value is not None else _DEFAULT_FONT_PX


def _weight(node: SNode) -> float:
    raw = node.style("font-weight").strip().lower()
    if raw in _NAMED_WEIGHTS:
        return _NAMED_WEIGHTS[raw]
    try:
        return float(raw)
    except ValueError:
        return 400.0


def _is_large_text(node: SNode) -> bool:
    size = _font_px(node)
    if size >= _LARGE_PX:
        return True
    return size >= _LARGE_BOLD_PX and _weight(node) >= _BOLD_WEIGHT


def _backdrop(
    by_index: dict[int, SNode],
    opacity: dict[int, float],
    node: SNode,
) -> tuple[float, float, float]:
    """Effective background color under a node, composited over white.

    Walks from the node itself up through its ancestors; every painted
    background is blended root-most-first, with the element's cumulative
    opacity multiplied into its alpha.
    """
    surfaces: list[SNode] = []
    cursor: SNode | None = node
    while cursor is not None:
        surfaces.append(cursor)
        cursor = by_index.get(cursor.parent) if cursor.parent >= 0 else None
    backdrop = (255.0, 255.0, 255.0)
    for surface in reversed(surfaces):
        color = _parse_color(surface.style("background-color"))
        if color is None:
            continue
        alpha = color[3] * opacity.get(surface.i, 1.0)
        if alpha <= 0.0:
            continue
        backdrop = _blend(color[:3], alpha, backdrop)
    return backdrop


def _format_rgb(rgb: tuple[float, float, float]) -> str:
    return f"rgb({round(rgb[0])}, {round(rgb[1])}, {round(rgb[2])})"


def _is_secondary_label(node: SNode) -> bool:
    """Uppercase, tracked, small labels are intentionally muted."""
    if node.style("text-transform").strip().lower() != "uppercase":
        # also treat heavily tracked paragraphs as decorative (heavily-tracked desc)
        spacing = node.style("letter-spacing").strip().lower()
        if spacing in ("", "normal", "0px"):
            return False
        try:
            ls_px = float(spacing.removesuffix("px").strip())
        except ValueError:
            return False
        return ls_px >= 2.5 and _font_px(node) <= 18
    size = _font_px(node)
    spacing = node.style("letter-spacing").strip().lower()
    has_tracking = spacing not in ("", "normal", "0px")
    try:
        ls_px = float(spacing.removesuffix("px").strip()) if has_tracking else 0.0
    except ValueError:
        ls_px = 1.0
    return size <= 13.5 or ls_px >= 1.0


def _contrast_issues(
    snapshot: Snapshot,
    opacity: dict[int, float],
) -> list[VisualIssue]:
    by_index = {node.i: node for node in snapshot.nodes}
    critical: list[tuple[SNode, dict[str, object]]] = []
    medium: list[tuple[SNode, dict[str, object]]] = []
    for node in snapshot.nodes:
        if _letters(node.text) == 0 or not _visible(node):
            continue
        if node.box.w <= 0 or node.box.h <= 0:
            continue
        if _is_secondary_label(node):
            continue
        fg = _parse_color(node.style("color"))
        if fg is None:
            continue
        fg_alpha = fg[3] * opacity.get(node.i, 1.0)
        if fg_alpha <= 0.0:
            continue
        backdrop = _backdrop(by_index, opacity, node)
        rendered_fg = _blend(fg[:3], fg_alpha, backdrop)
        ratio = _contrast_ratio(
            _relative_luminance(rendered_fg),
            _relative_luminance(backdrop),
        )
        threshold = _AA_LARGE if _is_large_text(node) else _AA_NORMAL
        if ratio + _EPS >= threshold:
            continue
        detail: dict[str, object] = {
            "ratio": round(ratio, 2),
            "required": threshold,
            "foreground": node.style("color"),
            "background": _format_rgb(backdrop),
            "text": node.text.strip()[:40],
        }
        bucket = medium if ratio + _EPS >= _AA_LARGE else critical
        bucket.append((node, detail))
    issues: list[VisualIssue] = []
    for severity, bucket in (("critical", critical), ("medium", medium)):
        if not bucket:
            continue
        worst = min(float(detail["ratio"]) for _, detail in bucket)
        issues.append(
            _issue(
                snapshot,
                "contrast",
                "contrast.below-threshold",
                "Text fails WCAG contrast against its background",
                (
                    f"{len(bucket)} text element(s) render as low as"
                    f" {worst:.2f}:1 against their effective background"
                    " (WCAG AA requires 4.5:1, 3:1 for large text); copy"
                    " this faint is hard or impossible to read."
                ),
                severity,
                {"min_ratio": round(worst, 2),
                 "pairs": [detail for _, detail in bucket]},
                [node for node, _ in bucket],
            )
        )
    issues.extend(_input_component_contrast(snapshot, opacity))
    return issues


def _distinct_hues(hues: Iterable[float]) -> list[float]:
    """Greedy circular clustering: centers are >= _HUE_GAP_DEG apart."""
    centers: list[float] = []
    for hue in sorted(hues):
        if all(_hue_distance(hue, center) >= _HUE_GAP_DEG for center in centers):
            centers.append(hue)
    return centers


def _input_component_contrast(
    snapshot: Snapshot, opacity: dict[int, float]
) -> list[VisualIssue]:
    """Non-text component contrast: input fills nearly invisible on dark."""
    by_index = {n.i: n for n in snapshot.nodes}
    hits: list[SNode] = []
    details: list[dict[str, object]] = []
    for node in snapshot.nodes:
        if node.tag != "input" or not _visible(node) or node.box.w <= 0:
            continue
        fill = _parse_color(node.style("background-color"))
        if fill is None or fill[3] * opacity.get(node.i, 1.0) < 0.5:
            continue
        backdrop = _backdrop(by_index, opacity, by_index[node.parent]) if node.parent in by_index else (255.0, 255.0, 255.0)
        # blend the input fill over its backdrop as if it were a flat swatch
        fill_rgb = (fill[0], fill[1], fill[2])
        # approximate blended fill appearance over backdrop
        # fill is opaque-ish (alpha ~1) so blend is near fill; keep simple
        comp_ratio = _contrast_ratio(
            _relative_luminance(fill_rgb), _relative_luminance(backdrop)
        )
        # also consider border separation: visible border saves the component
        has_border = any(
            (node.style_px(f"border-{side}-width") or 0) >= 1
            for side in ("top", "right", "bottom", "left")
        )
        if comp_ratio < 1.8 and not has_border:
            hits.append(node)
            details.append(
                {
                    "fill": node.style("background-color"),
                    "backdrop": _format_rgb(backdrop),
                    "ratio": round(comp_ratio, 2),
                }
            )
    if not hits:
        return []
    return [
        _issue(
            snapshot,
            "contrast",
            "contrast.component-below-threshold",
            "Form field fill nearly invisible against its container",
            (
                f"{len(hits)} input(s) have a fill-to-container contrast of"
                f" {min(float(d['ratio']) for d in details):.2f}:1 with no"
                " border separation (WCAG non-text requires >=3:1 borders or"
                " fills). The fields are almost invisible on the background."
            ),
            "medium",
            {"pairs": details},
            hits,
        )
    ]


def _accent_chaos_issue(snapshot: Snapshot) -> list[VisualIssue]:
    samples: list[tuple[float, SNode]] = []
    for node in snapshot.nodes:
        if not _visible(node):
            continue
        # backgrounds
        for prop in ("background-color", "border-top-color", "border-right-color", "border-bottom-color", "border-left-color"):
            raw = node.style(prop)
            if not raw or raw.strip().lower() in ("rgba(0, 0, 0, 0)", "transparent"):
                continue
            color = _parse_color(raw)
            if color is None:
                continue
            # need border width for border props
            if "border" in prop and (node.style_px(prop.replace("-color", "-width")) or 0) < 1:
                continue
            if color[3] < _ACCENT_MIN_ALPHA:
                continue
            hue, sat, val = _hsv(color[:3])
            if sat >= _ACCENT_MIN_SAT and val >= _ACCENT_MIN_VAL:
                samples.append((hue, node))
                break  # one sample per node to avoid double-counting
    if not samples:
        return []
    centers = _distinct_hues(hue for hue, _ in samples)
    issues: list[VisualIssue] = []
    if len(centers) > _MAX_ACCENT_HUES:
        refs: list[SNode] = []
        for center in centers:
            for hue, node in samples:
                if _hue_distance(hue, center) < _HUE_GAP_DEG:
                    refs.append(node)
                    break
        issues.append(
            _issue(
                snapshot,
                "color",
                "color.accent-hue-chaos",
                "Accent colors scatter across too many unrelated hues",
                (
                    f"{len(centers)} saturated accent hue families sit at least"
                    f" {_HUE_GAP_DEG:.0f} degrees apart in one view; scattered"
                    " accents read as an accident rather than a palette."
                ),
                "medium",
                {"hue_centers": sorted(round(c, 1) for c in centers)},
                refs,
            )
        )
    # Lonely accent: one hue family used by a single element while another
    # family dominates — reads as a one-off color that doesn't belong.
    if len(centers) >= 2:
        counts: dict[float, int] = {c: 0 for c in centers}
        for hue, _ in samples:
            best = min(centers, key=lambda c: _hue_distance(hue, c))
            counts[best] += 1
        lonely = [c for c, n in counts.items() if n == 1]
        if lonely and max(counts.values()) >= 2:
            refs_lonely: list[SNode] = []
            for center in lonely:
                for hue, node in samples:
                    if _hue_distance(hue, center) < _HUE_GAP_DEG:
                        refs_lonely.append(node)
                        break
            if refs_lonely:
                issues.append(
                    _issue(
                        snapshot,
                        "color",
                        "color.lonely-accent",
                        "One-off accent hue breaks palette cohesion",
                        (
                            "A single element uses a saturated hue at least"
                            f" {_HUE_GAP_DEG:.0f} degrees away from the rest of"
                            " the palette's hue families; the one-off accent"
                            " looks like a mistake rather than a system."
                        ),
                        "medium",
                        {"hue_centers": sorted(round(c, 1) for c in centers), "lonely": sorted(round(c, 1) for c in lonely)},
                        refs_lonely,
                    )
                )
    elif len(centers) == 1 and len(samples) == 1:
        # Single saturated accent on a small, light popup-style container.
        if snapshot.root_box.w * snapshot.root_box.h > 350000:
            return issues
        try:
            by_index_tmp = {n.i: n for n in snapshot.nodes}
            sample_node = samples[0][1]
            parent = by_index_tmp.get(sample_node.parent)
            if parent is None:
                page_lum = 1.0
            else:
                tmp_opacity = {n.i: 1.0 for n in snapshot.nodes}
                bg = _backdrop(by_index_tmp, tmp_opacity, parent)
                page_lum = _relative_luminance(bg)
        except Exception:
            page_lum = 1.0
        light_page = page_lum > 0.6
        if light_page:
            # ensure the rest of the page is neutral (few other saturated nodes)
            neutral_ratio = 1.0 - len(samples) / max(len([n for n in snapshot.nodes if _visible(n) and n.box.w > 0]), 1)
            if neutral_ratio > 0.85:
                issues.append(
                    _issue(
                        snapshot,
                        "color",
                        "color.lonely-accent",
                        "One-off accent hue breaks palette cohesion",
                        (
                            "A single saturated accent sits on an otherwise"
                            " light, neutral page with no other hue family;"
                            " the isolated accent reads as an orphan rather"
                            " than a palette."
                        ),
                        "medium",
                        {"hue_centers": sorted(round(c, 1) for c in centers)},
                        [samples[0][1]],
                    )
                )
    return issues


def _state_tone(color: tuple[float, float, float, float]) -> str | None:
    hue, sat, val = _hsv(color[:3])
    if sat < _STATE_MIN_SAT or val < _STATE_MIN_VAL:
        return None
    if any(low <= hue <= high for low, high in _RED_HUES):
        return "red"
    if any(low <= hue <= high for low, high in _GREEN_HUES):
        return "green"
    return None


def _state_color_issue(snapshot: Snapshot) -> list[VisualIssue]:
    groups: dict[tuple[int, str], list[tuple[str, SNode]]] = defaultdict(list)
    for node in snapshot.nodes:
        if _letters(node.text) == 0 or not _visible(node):
            continue
        color = _parse_color(node.style("color"))
        if color is None or color[3] < _STATE_MIN_ALPHA:
            continue
        tone = _state_tone(color)
        if tone is not None:
            groups[(node.parent, node.tag)].append((tone, node))
    flagged: list[SNode] = []
    for members in groups.values():
        tones = {tone for tone, _ in members}
        if not {"red", "green"} <= tones:
            continue
        pool = [node for _, node in members]
        uniform = (
            len({_font_px(n) for n in pool}) == 1
            and len({_weight(n) for n in pool}) == 1
            and len({n.style("background-color").strip().lower() for n in pool}) == 1
            and len({n.style("font-style").strip().lower() for n in pool}) == 1
            and len({n.style("text-decoration-line").strip().lower() for n in pool}) == 1
        )
        if uniform:
            flagged.extend(pool)
    if not flagged:
        return []
    return [
        _issue(
            snapshot,
            "color",
            "color.state-by-hue-alone",
            "State carried by red/green color alone",
            (
                f"{len(flagged)} sibling element(s) differ only by red vs"
                " green text while sharing size, weight, and surface;"
                " users who cannot distinguish the hues get no signal."
            ),
            "low",
            {},
            flagged,
        )
    ]


def _near_gray_issue(snapshot: Snapshot) -> list[VisualIssue]:
    groups: dict[
        tuple[int, str], list[tuple[tuple[float, float, float], SNode]]
    ] = defaultdict(list)
    for node in snapshot.nodes:
        if _letters(node.text) == 0 or not _visible(node):
            continue
        color = _parse_color(node.style("color"))
        if color is None or color[3] < _GRAY_MIN_ALPHA:
            continue
        if _hsv(color[:3])[1] <= _GRAY_MAX_SAT:
            groups[(node.parent, node.tag)].append((color[:3], node))
    flagged: list[SNode] = []
    lums: set[float] = set()
    for members in groups.values():
        values = sorted({_relative_luminance(rgb) for rgb, _ in members})
        if len(values) < 2:
            continue
        if values[-1] - values[0] >= _NEAR_GRAY_LUM_DELTA:
            continue
        flagged.extend(node for _, node in members)
        lums.update(round(v, 3) for v in values)
    if not flagged:
        return []
    return [
        _issue(
            snapshot,
            "color",
            "color.near-duplicate-grays",
            "Same-role elements use near-identical grays",
            (
                f"{len(flagged)} same-role text element(s) use grays whose"
                " relative luminance differs by less than"
                f" {_NEAR_GRAY_LUM_DELTA:.0%}; the near-duplicate values"
                " look accidental rather than like a gray ramp."
            ),
            "low",
            {"luminances": sorted(lums)},
            flagged,
        )
    ]


def _two_tone_section_issue(snapshot: Snapshot) -> list[VisualIssue]:
    """Two large adjacent sections with distinct muted background hues.

    Footers and split layouts that place two muted hues side-by-side
    (e.g., mauve panel next to slate panel) read as a two-tone mistake
    when the hues are far enough apart to look like different palettes.
    """
    candidates: list[tuple[float, SNode]] = []
    for node in snapshot.nodes:
        if node.tag not in ("section", "footer", "header", "div"):
            continue
        if node.box.w < 120 or node.box.h < 80:
            continue
        if not _visible(node) or node.box.w <= 0:
            continue
        color = _parse_color(node.style("background-color"))
        if color is None or color[3] < 0.9:
            continue
        hue, sat, val = _hsv(color[:3])
        # muted but not neutral/white, not fully saturated
        if 0.12 <= sat <= 0.45 and 0.18 <= val <= 0.55:
            candidates.append((hue, node))
    if len(candidates) < 2:
        return []
    hues = [h for h, _ in candidates]
    # cluster muted hues
    centers: list[float] = []
    for h in sorted(hues):
        if all(_hue_distance(h, c) >= 25 for c in centers):
            centers.append(h)
    if len(centers) < 2:
        return []
    # require distance between distinct centers
    if min(_hue_distance(a, b) for i, a in enumerate(centers) for b in centers[i + 1:]) < 25:
        return []
    refs = [n for _, n in candidates[:2]]
    return [
        _issue(
            snapshot,
            "color",
            "color.two-tone-sections",
            "Adjacent sections use clashing muted hues",
            (
                "Two large sections side-by-side use muted background hues"
                f" {round(centers[0])}° and {round(centers[1])}° apart; the"
                " two-tone split looks like competing palettes rather than"
                " one system."
            ),
            "medium",
            {"hues": sorted(round(c, 1) for c in centers)},
            refs,
        )
    ]


def _white_card_on_white_issue(snapshot: Snapshot) -> list[VisualIssue]:
    """White cards on a white page with no border/shadow separation.

    Card-deck style decks where white cards sit on a white canvas with
    no border or shadow differentiation rely on no separation at all —
    the cards are nearly invisible as distinct surfaces.
    """
    by_index = {n.i: n for n in snapshot.nodes}
    # find light page backdrop: white or very light
    candidates: list[SNode] = []
    for node in snapshot.nodes:
        if not _visible(node) or node.box.w < 80 or node.box.h < 60:
            continue
        if "card" not in node.classes:
            continue
        bg = _parse_color(node.style("background-color"))
        if bg is None or bg[3] < 0.9:
            continue
        # card is white / near-white
        if _relative_luminance(bg[:3]) < 0.85:
            continue
        has_border = any(
            (node.style_px(f"border-{s}-width") or 0) >= 1
            for s in ("top", "right", "bottom", "left")
        )
        if has_border:
            continue
        parent = by_index.get(node.parent)
        if parent is None:
            continue
        try:
            backdrop = _backdrop(by_index, {n.i: 1.0 for n in snapshot.nodes}, parent)
            if _relative_luminance(backdrop) < 0.7:
                continue
        except Exception:
            continue
        candidates.append(node)
    if len(candidates) < 3:
        return []
    return [
        _issue(
            snapshot,
            "contrast",
            "contrast.card-on-card",
            "Cards indistinguishable from the page background",
            (
                f"{len(candidates)} cards render white on a white page with"
                " no border or shadow separation; the cards lack contrast"
                " against the canvas and read as floating text rather than"
                " distinct surfaces."
            ),
            "medium",
            {"cards": len(candidates)},
            candidates,
        )
    ]
