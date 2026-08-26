"""The 27 deterministic AI-design-slop patterns (port of @slop-detect/core).

Grouped by detection axis so each piece can be built and blind-judged on its
own: fonts, colors/gradients, hero+card layout, glass/glow, images/text tells.
Every pattern returns an evidence dict with a boolean ``triggered`` key using
the same keys as the reference so bench output compares field-for-field.

Reference: ravidsrk/slop-detect `packages/core/src/patterns.ts` (MIT), plus
Impeccable-derived 2026.08 tells (Apache-2.0).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from ux_analyzer.analysis.slop.color import (
    Color,
    channel_spread,
    contrast_ratio,
    is_dark,
    is_mid_grey,
    is_purple,
    parse_color,
    relative_luminance,
    rgb_to_hsl,
)
from ux_analyzer.analysis.slop.context import SlopContext
from ux_analyzer.analysis.slop.fonts import is_accent_serif, is_slop_font
from ux_analyzer.analysis.visual.snapshot import SNode

PatternFn = Callable[[SlopContext], dict[str, Any]]

_PATTERNS: list[dict[str, Any]] = []


def _register(pattern: dict[str, Any]) -> None:
    _PATTERNS.append(pattern)


def patterns() -> list[dict[str, Any]]:
    return list(_PATTERNS)


def pattern_by_id(pid: str) -> dict[str, Any] | None:
    return next((p for p in _PATTERNS if p["id"] == pid), None)


_LEADING_NUM = re.compile(r"^([-+]?\d+(?:\.\d+)?)")


def _px(styles: dict[str, str], prop: str) -> float:
    """parseFloat port: leading numeric prefix of a length value."""
    v = styles.get(prop, "")
    if not v or v in ("none", "auto", "normal"):
        return 0.0
    m = _LEADING_NUM.match(v.strip())
    if not m:
        return 0.0
    return float(m.group(1))


def _text(node: SNode) -> str:
    return (node.text or "").strip()


def _full_text(ctx: SlopContext, node: SNode) -> str:
    """textContent approximation: own text + descendant texts (precomputed)."""
    return ctx.full_text(node)


def _ancestor_tags(ctx: SlopContext, node: SNode, limit: int = 30) -> list[str]:
    tags: list[str] = []
    cur = ctx.parent_of(node)
    guard = 0
    while cur is not None and guard < limit:
        tags.append(cur.tag)
        cur = ctx.parent_of(cur)
        guard += 1
    return tags


def _closest_interactive(ctx: SlopContext, node: SNode) -> bool:
    cur: SNode | None = node
    guard = 0
    while cur is not None and guard < 30:
        if cur.tag in ("a", "button"):
            return True
        if "button" in cur.cls.lower() or "role=button" in cur.cls.lower():
            return True
        cur = ctx.parent_of(cur)
        guard += 1
    return False


# ══════════════════════════════════════════════════════════════════════════════
# PIECE 1 — FONT STACKS (patterns 1, 10, 22, 24, 26, 27)
# ══════════════════════════════════════════════════════════════════════════════


def _slop_fonts(ctx: SlopContext) -> dict[str, Any]:
    slop_count = 0
    total = 0
    accent_serif_words = 0
    for el in ctx.visible:
        if not _full_text(ctx, el):
            continue
        fam = el.style("font-family")
        total += 1
        if is_slop_font(fam):
            slop_count += 1
        if is_accent_serif(fam) and re.search(
            r"italic|oblique", el.style("font-style")
        ):
            accent_serif_words += 1
    hero_fam = ctx.h1.style("font-family") if ctx.h1 else ""
    hero_is_slop = is_slop_font(hero_fam)
    ratio = round(slop_count / total, 3) if total else 0.0
    return {
        "slopCount": slop_count,
        "total": total,
        "ratio": ratio,
        "heroIsSlop": hero_is_slop,
        "heroFam": hero_fam,
        "accentSerifItalicCount": accent_serif_words,
        "triggered": hero_is_slop or ratio >= 0.6 or accent_serif_words > 0,
    }


def _all_caps_labels(ctx: SlopContext) -> dict[str, Any]:
    count = 0
    samples: list[str] = []
    for el in ctx.visible:
        if el.style("text-transform") != "uppercase":
            continue
        txt = _full_text(ctx, el)
        if len(txt) < 3 or len(txt) > 40:
            continue
        ls = _px(el.styles, "letter-spacing")
        if ls < 0.5:
            continue
        count += 1
        if len(samples) < 3:
            samples.append(txt[:30])
    return {"count": count, "samples": samples, "triggered": count >= 2}


def _crushed_tracking(ctx: SlopContext) -> dict[str, Any]:
    count = 0
    samples: list[dict[str, Any]] = []
    for el in ctx.visible:
        txt = _full_text(ctx, el)
        if len(txt) < 3 or len(txt) > 80:
            continue
        font_size = _px(el.styles, "font-size")
        if font_size < 28:
            continue
        ls = _px(el.styles, "letter-spacing")
        if ls == 0.0 and el.style("letter-spacing") in ("normal", ""):
            continue
        if el.style("letter-spacing") == "normal":
            continue
        em = ls / font_size if font_size else 0.0
        if em <= -0.05:
            count += 1
            if len(samples) < 3:
                samples.append({"text": txt[:30], "em": round(em, 3), "px": round(ls, 1)})
    return {"count": count, "samples": samples, "triggered": count >= 1}


def _oversized_hero_h1(ctx: SlopContext) -> dict[str, Any]:
    if not ctx.h1:
        return {"triggered": False}
    font_size = _px(ctx.h1.styles, "font-size")
    text = _full_text(ctx, ctx.h1)
    triggered = font_size >= 72 and len(text) >= 40
    return {
        "fontSize": round(font_size),
        "chars": len(text),
        "text": text[:60],
        "triggered": triggered,
    }


def _wide_body_tracking(ctx: SlopContext) -> dict[str, Any]:
    body = re.compile(r"^(p|li|td|dd|blockquote|figcaption)$")
    count = 0
    samples: list[dict[str, Any]] = []
    for el in ctx.visible:
        if not body.match(el.tag):
            continue
        txt = _full_text(ctx, el)
        if len(txt) < 40:
            continue
        if el.style("text-transform") == "uppercase":
            continue
        font_size = _px(el.styles, "font-size")
        ls = _px(el.styles, "letter-spacing")
        if el.style("letter-spacing") == "normal":
            continue
        em = ls / font_size if font_size else 0.0
        if em > 0.05:
            count += 1
            if len(samples) < 3:
                samples.append({"em": round(em, 3), "text": txt[:30]})
    return {"count": count, "samples": samples, "triggered": count >= 1}


def _flat_type_hierarchy(ctx: SlopContext) -> dict[str, Any]:
    texty = re.compile(r"^(h1|h2|h3|h4|h5|h6|p|span|a|li|td|th|label|button|div)$")
    sizes: set[int] = set()
    for el in ctx.visible:
        if not texty.match(el.tag):
            continue
        txt = _full_text(ctx, el)
        if len(txt) < 2:
            continue
        fs = round(_px(el.styles, "font-size"))
        if 8 <= fs < 200:
            sizes.add(fs)
    arr = sorted(sizes)
    if len(arr) < 3:
        return {"distinct": len(arr), "triggered": False}
    ratio = max(arr) / min(arr)
    return {
        "distinct": len(arr),
        "ratio": round(ratio, 2),
        "min": min(arr),
        "max": max(arr),
        "triggered": ratio < 2.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PIECE 2 — COLORS & GRADIENTS (patterns 2, 3, 4, 11, 20, 21, 23)
# ══════════════════════════════════════════════════════════════════════════════


def _gradient_stops(bg_img: str) -> list[str]:
    return re.findall(r"rgba?\([^)]+\)|#[0-9a-fA-F]{3,8}", bg_img)


def _vibe_purple(ctx: SlopContext) -> dict[str, Any]:
    purple_els = 0
    filled_ctas = 0
    samples: list[dict[str, Any]] = []
    cta_re = re.compile(r"btn|button|cta", re.IGNORECASE)
    outline_re = re.compile(r"outline|ghost", re.IGNORECASE)
    for el in ctx.visible:
        cs = el.styles
        bg = parse_color(cs.get("background-color") or "")
        bg_img = cs.get("background-image") or ""
        border = parse_color(cs.get("border-top-color") or "")
        purp = False
        grad_purp = False
        if is_purple(bg):
            purp = True
        if is_purple(border) and _px(cs, "border-top-width") > 0:
            purp = True
        if "gradient" in bg_img:
            for c in _gradient_stops(bg_img):
                if is_purple(parse_color(c)):
                    grad_purp = True
                    break
            if grad_purp:
                purp = True
        if not purp:
            continue
        purple_els += 1
        cls = el.cls
        is_cta = el.tag in ("a", "button") or bool(cta_re.search(cls))
        if not is_cta:
            continue
        if outline_re.search(cls):
            continue
        filled = (bg is not None and bg.a >= 0.5 and is_purple(bg)) or (
            grad_purp and (bg is None or bg.a < 0.1)
        )
        if filled:
            filled_ctas += 1
            if len(samples) < 2:
                samples.append({"tag": el.tag, "bg": cs.get("background-color", "")})
    return {"purpleEls": purple_els, "filledCtas": filled_ctas, "samples": samples, "triggered": filled_ctas >= 1}


def _gradient_text(ctx: SlopContext) -> dict[str, Any]:
    count = 0
    hero_has_gradient = False
    for el in ctx.visible:
        cs = el.styles
        clip = cs.get("background-clip", "")
        if clip == "text" and re.search(r"gradient\(", cs.get("background-image", "")):
            count += 1
            if ctx.h1 is not None and el.i == ctx.h1.i:
                hero_has_gradient = True
    if ctx.h1 and not hero_has_gradient:
        cs = ctx.h1.styles
        clip = cs.get("background-clip", "")
        if clip == "text" and re.search(r"gradient\(", cs.get("background-image", "")):
            hero_has_gradient = True
    return {"count": count, "heroHasGradient": hero_has_gradient, "triggered": count > 0 or hero_has_gradient}


_RGBA_TOKEN_RE = re.compile(r"rgba?\(\s*\d+\s*,\s*\d+\s*,\s*\d+(?:\s*,\s*([\d.]+))?\s*\)")


def _gradient_backgrounds(ctx: SlopContext) -> dict[str, Any]:
    n = 0
    conic = 0
    for el in ctx.visible:
        bg_img = el.style("background-image")
        if not re.search(r"gradient\(", bg_img):
            continue
        # Reference behavior: alpha is read off each color token's last
        # component via a second regex; a stop whose final channel is 0 reads
        # as transparent and does not count.
        tokens = [m.group(0) for m in _RGBA_TOKEN_RE.finditer(bg_img)]
        has_opaque_stop = not tokens
        if tokens:
            for token in tokens:
                a = re.search(r",\s*([\d.]+)\s*\)", token)
                if not a or float(a.group(1)) > 0.05:
                    has_opaque_stop = True
                    break
        if has_opaque_stop:
            n += 1
            if "conic-gradient" in bg_img:
                conic += 1
    return {"bgElements": n, "conic": conic, "triggered": n >= 5}


def _direct_children(ctx: SlopContext) -> list[SNode]:
    """Direct children of the snapshot root (reference: body.children)."""
    root = ctx.snap.nodes[0] if ctx.snap.nodes else None
    if root is None:
        return []
    return [n for n in ctx.snap.nodes if n.parent == root.i]


def _perma_dark_mode(ctx: SlopContext) -> dict[str, Any]:
    cand: list[Color | None] = [
        parse_color(ctx.surface.get("htmlBg", "")),
        parse_color(ctx.surface.get("bodyBg", "")),
        parse_color(ctx.surface.get("centerBg", "")),
    ]
    for n in _direct_children(ctx)[:8]:
        if n.box.w >= ctx.viewport_w * 0.8 and n.box.h >= ctx.viewport_h * 0.5:
            cand.append(parse_color(n.style("background-color")))
    body_bg = None
    for c in cand:
        if c and c.a >= 0.5:
            body_bg = c
            break
    dark = is_dark(body_bg)
    if not dark:
        return {"triggered": False, "bodyDark": False}
    greys = 0
    total = 0
    for n in ctx.visible:
        if n.tag not in ("p", "li", "span"):
            continue
        txt = _full_text(ctx, n)
        if len(txt) < 6:
            continue
        total += 1
        if is_mid_grey(parse_color(n.style("color"))):
            greys += 1
        if total >= 200:
            break
    ratio = round(greys / total, 2) if total else 0.0
    triggered = dark and (ratio >= 0.3 or total >= 5)
    return {"bodyDark": True, "greyParas": greys, "totalParas": total, "ratio": ratio, "triggered": triggered}


def _cream_default_bg(ctx: SlopContext) -> dict[str, Any]:
    def is_cream(c: Color | None) -> bool:
        if not c or c.a < 0.5:
            return False
        if min(c.r, c.g, c.b) < 209:
            return False
        if not (c.r >= c.g >= c.b):
            return False
        warmth = c.r - c.b
        return 6 <= warmth <= 48

    nodes = ctx.snap.nodes
    body_bg = parse_color(ctx.surface.get("bodyBg", "")) or (
        parse_color(nodes[0].style("background-color")) if nodes else None
    )
    html_bg = parse_color(ctx.surface.get("htmlBg", ""))
    surface = body_bg if body_bg and body_bg.a >= 0.5 else html_bg
    if not surface or (not is_cream(surface) and min(surface.r, surface.g, surface.b) >= 250):
        for n in _direct_children(ctx)[:8]:
            if n.box.w >= ctx.viewport_w * 0.8 and n.box.h >= ctx.viewport_h * 0.5:
                wc = parse_color(n.style("background-color"))
                if wc and wc.a >= 0.5 and is_cream(wc):
                    surface = wc
                    break
    cream = is_cream(surface)
    hexv = (
        "#" + "".join(f"{int(round(v)):02x}" for v in (surface.r, surface.g, surface.b))
        if surface
        else None
    )
    return {"surface": hexv, "triggered": cream}


def _low_contrast_text(ctx: SlopContext) -> dict[str, Any]:
    body = re.compile(r"^(p|li|span|dd|blockquote|figcaption|small)$")
    fails = 0
    checked = 0
    samples: list[dict[str, Any]] = []
    seen: set[int] = set()
    for el in ctx.visible:
        if el.i in seen:
            continue
        if not body.match(el.tag):
            continue
        if _closest_interactive(ctx, el):
            continue
        if any(t in ("nav", "header") for t in _ancestor_tags(ctx, el, 6)):
            continue
        direct = _text(el)
        if len(direct) < 20:
            continue
        seen.add(el.i)
        font_size = _px(el.styles, "font-size")
        if font_size >= 24:
            continue
        fg = parse_color(el.style("color"))
        if not fg or fg.a < 0.5:
            continue
        bg = ctx.effective_bg(el)
        if not bg or bg.approx:
            continue
        if relative_luminance(bg) < 0.6:
            continue
        if channel_spread(fg) >= 40:
            continue
        checked += 1
        ratio = contrast_ratio(fg, bg)
        if ratio < 4.5:
            fails += 1
            if len(samples) < 3:
                samples.append({"text": direct[:30], "ratio": round(ratio, 2), "floor": 4.5})
        if checked >= 400:
            break
    ratio_fail = round(fails / checked, 3) if checked else 0.0
    triggered = checked >= 4 and fails >= 4 and ratio_fail >= 0.25
    return {"fails": fails, "checked": checked, "ratioFail": ratio_fail, "samples": samples, "triggered": triggered}


def _gray_on_color(ctx: SlopContext) -> dict[str, Any]:
    def is_mid_grey_text(c: Color | None) -> bool:
        if not c or c.a < 0.5:
            return False
        if channel_spread(c) >= 20:
            return False
        hsl = rgb_to_hsl(c)
        if not hsl:
            return False
        return 0.3 < hsl.light < 0.75

    count = 0
    samples: list[str] = []
    seen: set[int] = set()
    for el in ctx.visible:
        if el.i in seen:
            continue
        if _closest_interactive(ctx, el):
            continue
        direct = _text(el)
        if len(direct) < 12:
            continue
        seen.add(el.i)
        fg = parse_color(el.style("color"))
        if not is_mid_grey_text(fg):
            continue
        bg = ctx.effective_bg(el)
        if not bg or bg.approx:
            continue
        if channel_spread(bg) < 40:
            continue
        count += 1
        if len(samples) < 3:
            samples.append(direct[:30])
        if count >= 50:
            break
    return {"count": count, "samples": samples, "triggered": count >= 3}


# ══════════════════════════════════════════════════════════════════════════════
# PIECE 3 — HERO & CARD LAYOUT TELLS (patterns 5, 8, 9, 12, 13, 14, 15, 16, 17, 25)
# ══════════════════════════════════════════════════════════════════════════════


def _accent_stripe(ctx: SlopContext) -> dict[str, Any]:
    stripe_cards = 0
    for el in ctx.visible:
        r = el.box
        if r.w < 100 or r.h < 60:
            continue
        widths = {
            "top": _px(el.styles, "border-top-width"),
            "left": _px(el.styles, "border-left-width"),
            "right": _px(el.styles, "border-right-width"),
            "bottom": _px(el.styles, "border-bottom-width"),
        }
        colors = {
            "top": parse_color(el.style("border-top-color")),
            "left": parse_color(el.style("border-left-color")),
        }
        other_max = max(widths["left"], widths["right"], widths["bottom"])
        top_stripe = (
            widths["top"] >= 3
            and other_max < widths["top"] - 1
            and colors["top"] is not None
            and colors["top"].a > 0.3
        )
        left_stripe = (
            widths["left"] >= 3
            and max(widths["top"], widths["right"], widths["bottom"]) < widths["left"] - 1
            and colors["left"] is not None
            and colors["left"].a > 0.3
        )
        if top_stripe or left_stripe:
            stripe_cards += 1
    return {"stripeCards": stripe_cards, "triggered": stripe_cards >= 2}


def _centered_hero(ctx: SlopContext) -> dict[str, Any]:
    h1 = ctx.h1
    if not h1:
        return {"triggered": False, "h1Found": False}
    cs = h1.styles
    font_size = _px(cs, "font-size")
    centered = cs.get("text-align", "") == "center"
    parent = ctx.parent_of(h1)
    if not centered and parent is not None:
        if parent.style("text-align") == "center":
            centered = True
    if not centered:
        r = h1.box
        pr = parent.box if parent is not None else None
        if pr is not None and pr.w > 0:
            el_cx = r.x + r.w / 2
            pr_cx = pr.x + pr.w / 2
            if abs(el_cx - pr_cx) / pr.w < 0.12:
                centered = True
    big = font_size >= 28
    slop_font = is_slop_font(cs.get("font-family", ""))
    triggered = centered and big and slop_font
    return {"triggered": triggered, "fontSize": font_size, "centered": centered, "slopFont": slop_font, "family": cs.get("font-family", "")}


def _hero_eyebrow_pill(ctx: SlopContext) -> dict[str, Any]:
    h1 = ctx.h1
    if not h1:
        return {"triggered": False}
    h1_rect = h1.box
    pill_keywords = re.compile(
        r"\b(new|beta|now in|introducing|announcing|just shipped|v\d+|launching|coming soon|early access|whats new|just landed|just dropped)\b",
        re.IGNORECASE,
    )
    sparkle_emoji = re.compile(r"✨|🚀|⚡")
    for el in ctx.visible:
        if el.tag not in ("a", "div", "span", "button"):
            continue
        r = el.box
        bottom = r.y + r.h
        if bottom > h1_rect.y or bottom < h1_rect.y - 250:
            continue
        if r.w < 40 or r.w > 500 or r.h > 80:
            continue
        radius = _px(el.styles, "border-radius")
        if not radius or radius < r.h / 3:
            continue
        txt = _full_text(ctx, el)
        if len(txt) < 2 or len(txt) > 80:
            continue
        el_center = r.x + r.w / 2
        h1_center = h1_rect.x + h1_rect.w / 2
        if abs(el_center - h1_center) > h1_rect.w / 2:
            continue
        if pill_keywords.search(txt) or sparkle_emoji.search(txt):
            return {"triggered": True, "text": txt[:60], "radius": radius}
        # Published core 0.5.1 catch-all: a small rounded pill directly above
        # a centered hero is still slop-coded even without a keyword, as long
        # as it carries a background or border.
        if len(txt) < 40 and (
            el.style("background-color") != "rgba(0, 0, 0, 0)"
            or _px(el.styles, "border-top-width") != 0
        ):
            return {"triggered": True, "text": txt[:60], "radius": radius, "weak": True}
    return {"triggered": False}


def _icon_card_grid(ctx: SlopContext) -> dict[str, Any]:
    groups: dict[str, list[Any]] = {}
    for el in ctx.visible:
        parent = ctx.parent_of(el)
        if parent is None:
            continue
        r = el.box
        if not (150 <= r.w <= 600 and 100 <= r.h <= 600):
            continue
        # Reference selector: ':scope > svg, :scope > img, :scope > div > svg,
        # :scope > div > img' — icon must be a direct child or one div deep.
        icon = None
        kids = ctx.snap.children(el.i)
        for k in kids:
            if k.tag in ("svg", "img"):
                icon = k
                break
        if icon is None:
            for k in kids:
                if k.tag == "div":
                    for g in ctx.snap.children(k.i):
                        if g.tag in ("svg", "img"):
                            icon = g
                            break
                if icon is not None:
                    break
        if icon is None:
            continue
        ir = icon.box
        if ir.w > 80 or ir.h > 80:
            continue
        if ir.y > r.y + r.h * 0.4:
            continue
        key = f"{parent.tag}:{round(r.w / 20)}"
        groups.setdefault(key, []).append(el)
    max_group = max((len(v) for v in groups.values()), default=0)
    return {"maxGroupSize": max_group, "triggered": max_group >= 3}


def _numbered_steps(ctx: SlopContext) -> dict[str, Any]:
    number_pat = re.compile(r"^(?:step\s*)?(\d{1,2})(?:[.):]|\s*[—-])?\s*", re.IGNORECASE)
    parents: dict[int, set[int]] = {}
    for el in ctx.visible:
        txt = _full_text(ctx, el)
        m = number_pat.match(txt)
        if not m:
            continue
        n = int(m.group(1))
        if not 1 <= n <= 9:
            continue
        parents.setdefault(el.parent, set()).add(n)
    best_run = 0
    for nums in parents.values():
        if {1, 2, 3}.issubset(nums):
            run = 3 + (1 if 4 in nums else 0) + (1 if 5 in nums else 0)
            best_run = max(best_run, run)
    return {"bestRun": best_run, "triggered": best_run >= 3}


def _stat_banner(ctx: SlopContext) -> dict[str, Any]:
    stat_pat = re.compile(r"^\$?\d+[.,]?\d*\s*[KMB%+]?\+?$")
    candidates: list[dict[str, Any]] = []
    for el in ctx.visible:
        fs = _px(el.styles, "font-size")
        if fs < 28:
            continue
        txt = _full_text(ctx, el)
        if len(txt) > 12 or not stat_pat.match(txt):
            continue
        candidates.append({"el": el, "top": el.box.y, "txt": txt})
    candidates.sort(key=lambda c: c["top"])
    best_cluster = 0
    for i in range(len(candidates)):
        n = 1
        for j in range(i + 1, len(candidates)):
            if abs(candidates[j]["top"] - candidates[i]["top"]) < 80:
                n += 1
        best_cluster = max(best_cluster, n)
    return {"clusterSize": best_cluster, "triggered": best_cluster >= 3}


def _faq_accordion(ctx: SlopContext) -> dict[str, Any]:
    count = 0
    page_height = ctx.doc_height
    for el in ctx.visible:
        if el.tag != "details":
            continue
        abs_top = el.box.y + ctx.scroll_y
        if abs_top < page_height * 0.4:
            continue
        count += 1
    text_faq = False
    for el in ctx.visible:
        if el.tag not in ("h1", "h2", "h3"):
            continue
        t = _full_text(ctx, el).lower()
        if t == "faq" or t == "frequently asked questions" or t.startswith("faq"):
            text_faq = True
            break
    return {
        "detailsCount": count,
        "hasFaqHeading": text_faq,
        "triggered": count >= 3 or (text_faq and count >= 1),
    }


def _gradient_letter_avatars(ctx: SlopContext) -> dict[str, Any]:
    count = 0
    samples: list[dict[str, Any]] = []
    for el in ctx.visible:
        r = el.box
        if not (28 <= r.w <= 96):
            continue
        if abs(r.w - r.h) > 8:
            continue
        radius = _px(el.styles, "border-radius")
        if radius < r.w / 3:
            continue
        bg = el.style("background-image")
        bg_color = el.style("background-color")
        has_gradient = re.search(r"gradient\(", bg)
        has_solid = bg_color not in ("", "rgba(0, 0, 0, 0)", "transparent")
        if not has_gradient and not has_solid:
            continue
        txt = _full_text(ctx, el)
        if not (1 <= len(txt) <= 3):
            continue
        if not re.match(r"^[A-Za-z]{1,3}$", txt):
            continue
        if any(d.tag in ("img", "svg") for d in ctx.snap.descendants(el.i)):
            continue
        count += 1
        if len(samples) < 3:
            samples.append({"initials": txt, "size": round(r.w)})
    return {"count": count, "samples": samples, "triggered": count >= 2}


def _bento_grid(ctx: SlopContext) -> dict[str, Any]:
    best = {"children": 0, "spanVariety": 0, "rounded": 0}
    for el in ctx.visible:
        cs = el.styles
        if cs.get("display", "") != "grid":
            continue
        cols = [c for c in cs.get("grid-template-columns", "").split(" ") if c]
        if len(cols) < 2:
            continue
        kids = ctx.snap.children(el.i)
        if len(kids) < 4:
            continue
        spans: set[int] = set()
        rounded = 0
        sized = 0
        for k in kids:
            kr = k.box
            if kr.w < 80 or kr.h < 60:
                continue
            sized += 1
            gc = k.style("grid-column")
            m = re.search(r"span\s+(\d+)", gc, re.IGNORECASE)
            spans.add(int(m.group(1)) if m else 1)
            if _px(k.styles, "border-radius") >= 12:
                rounded += 1
        if sized < 4:
            continue
        score = {"children": sized, "spanVariety": len(spans), "rounded": rounded}
        if rounded >= 4 and len(spans) >= 2 and sized > best["children"]:
            best = score
    return {
        **best,
        "triggered": best["rounded"] >= 4 and best["spanVariety"] >= 2 and best["children"] >= 5,
    }


def _nested_cards(ctx: SlopContext) -> dict[str, Any]:
    skip = re.compile(r"^(input|select|textarea|img|video|canvas|picture|pre|code|svg|button|a|nav|li)$", re.IGNORECASE)
    chrome = re.compile(r"(dropdown|popover|tooltip|menu|modal|dialog|overlay)")

    def is_card_like(el: SNode) -> bool:
        tag = el.tag
        if skip.match(tag):
            return False
        cs = el.styles
        if cs.get("position", "") in ("absolute", "fixed"):
            return False
        cls = el.cls.lower()
        if chrome.search(cls):
            return False
        if len(_full_text(ctx, el)) < 10:
            return False
        r = el.box
        if r.w < 50 or r.h < 30:
            return False
        has_shadow = bool(cs.get("box-shadow") and cs.get("box-shadow") != "none")
        has_border = (
            _px(cs, "border-top-width") > 0
            or _px(cs, "border-left-width") > 0
            or re.search(r"\bborder\b", cls) is not None
        )
        radius = _px(cs, "border-radius")
        has_radius = radius > 0
        bg = cs.get("background-color", "")
        has_bg = bg not in ("", "rgba(0, 0, 0, 0)", "transparent")
        return (has_shadow or has_border) and (has_radius or has_bg)

    def in_transformed_frame(el: SNode) -> bool:
        cur = ctx.parent_of(el)
        guard = 0
        while cur is not None and guard < 20:
            cs = cur.styles
            if cs.get("transform", "") not in ("", "none") or cs.get("perspective", "") not in ("", "none"):
                return True
            cur = ctx.parent_of(cur)
            guard += 1
        return False

    cards: list[Any] = []
    for el in ctx.visible:
        try:
            if is_card_like(el):
                cards.append(el)
        except Exception:
            continue
    card_ids = {el.i for el in cards}
    nested = 0
    samples: list[str] = []
    for el in cards:
        anc = ctx.parent_of(el)
        has_card_ancestor = False
        guard = 0
        while anc is not None and guard < 30:
            if anc.i in card_ids:
                has_card_ancestor = True
                break
            anc = ctx.parent_of(anc)
            guard += 1
        if not has_card_ancestor:
            continue
        contains_card = any(
            other.i != el.i and other.i in {d.i for d in ctx.snap.descendants(el.i)} and other.i in card_ids
            for other in cards
        )
        if contains_card:
            continue
        if in_transformed_frame(el):
            continue
        nested += 1
        if len(samples) < 3:
            samples.append((el.cls or el.tag).split()[0][:40] if (el.cls or el.tag) else el.tag)
    return {"nested": nested, "samples": samples, "triggered": nested >= 3}


# ══════════════════════════════════════════════════════════════════════════════
# PIECE 4 — GLASS & GLOW EFFECTS (patterns 6, 7, 18, 19)
# ══════════════════════════════════════════════════════════════════════════════


def _glassmorphism(ctx: SlopContext) -> dict[str, Any]:
    glass_count = 0
    for el in ctx.visible:
        cs = el.styles
        filter_val = cs.get("backdrop-filter", "") or cs.get("-webkit-backdrop-filter", "")
        if not re.search(r"blur\(", filter_val):
            continue
        bg = parse_color(cs.get("background-color") or "")
        if bg and 0 < bg.a < 0.8:
            glass_count += 1
    # Published core 0.5.1 fires on a single frosted layer (the >=2 gate is a
    # main-branch refinement not present in the shipped reference).
    return {"glassCount": glass_count, "triggered": glass_count >= 1}


def _colored_glows(ctx: SlopContext) -> dict[str, Any]:
    glow_count = 0
    for el in ctx.visible:
        shadow = el.style("box-shadow")
        if not shadow or shadow == "none" or "rgb" not in shadow:
            continue
        blur_match = re.findall(r"\s(\d+(?:\.\d+)?)px\s", shadow)
        max_blur = max((float(b) for b in blur_match), default=0.0)
        if max_blur < 24:
            continue
        colors = re.findall(r"rgba?\([^)]+\)|#[0-9a-f]{3,8}", shadow)
        for c in colors:
            col = parse_color(c)
            if not col or col.a < 0.1:
                continue
            if is_purple(col):
                glow_count += 1
                break
            mx = max(col.r, col.g, col.b)
            mn = min(col.r, col.g, col.b)
            if mx - mn > 60 and col.a > 0.2:
                glow_count += 1
                break
    return {"glowCount": glow_count, "triggered": glow_count >= 1}


def _aurora_mesh_gradient(ctx: SlopContext) -> dict[str, Any]:
    blobs = 0
    samples: list[dict[str, Any]] = []
    vw, vh = ctx.viewport_w, ctx.viewport_h
    for el in ctx.visible:
        cs = el.styles
        bg_img = cs.get("background-image", "")
        filter_val = cs.get("filter", "")
        blur_m = re.search(r"blur\(([\d.]+)px\)", filter_val)
        blur = float(blur_m.group(1)) if blur_m else 0.0
        is_grad = bool(re.search(r"(radial|conic)-gradient\(", bg_img)) or (
            re.search(r"linear-gradient\(", bg_img) is not None and blur > 0
        )
        if not is_grad:
            continue
        radius = _px(cs, "border-radius")
        r = el.box
        big = r.w >= vw * 0.25 and r.h >= vh * 0.2
        orby = radius >= min(r.w, r.h) * 0.4 or cs.get("border-radius", "") == "50%" or "9999px" in cs.get("border-radius", "")
        positioned = cs.get("position", "") in ("absolute", "fixed")
        if blur >= 24 and big:
            blobs += 1
        elif positioned and orby and big and re.search(r"(radial|conic)-gradient\(", bg_img):
            blobs += 1
        else:
            continue
        if len(samples) < 3:
            samples.append({"blur": round(blur), "radius": cs.get("border-radius", "")})
    return {"blobs": blobs, "samples": samples, "triggered": blobs >= 2}


def _ai_sparkle_badges(ctx: SlopContext) -> dict[str, Any]:
    sparkle_emoji = re.compile(r"[\u2728\u2729\u2734\u2735]|\U0001F31F|\U0001FA84")
    magic_word = re.compile(r"\b(ai|magic|generate|powered by ai|with ai|smart)\b", re.IGNORECASE)
    emoji_hits = 0
    svg_hits = 0.0
    samples: list[str] = []
    for el in ctx.visible:
        txt = _full_text(ctx, el)
        if txt and len(txt) <= 40 and sparkle_emoji.search(txt) and len(ctx.snap.children(el.i)) <= 1:
            emoji_hits += 1
            if len(samples) < 3:
                samples.append(txt[:30])
            continue
        attrs = (el.cls or "").lower()
        if re.search(r"sparkle|sparkles|magic-wand|wand-sparkles", attrs):
            near = ""
            cur = ctx.parent_of(el)
            guard = 0
            while cur is not None and guard < 8:
                if cur.tag in ("button", "a", "label"):
                    near = _full_text(ctx, cur)
                    break
                cur = ctx.parent_of(cur)
                guard += 1
            svg_hits += 1.0 if magic_word.search(near) else 0.5
    total = emoji_hits + svg_hits
    return {
        "emojiHits": emoji_hits,
        "svgHits": svg_hits,
        "samples": samples,
        "triggered": total >= 1 and (emoji_hits >= 1 or svg_hits >= 1),
    }


_register(
    {
        "id": "slop_fonts",
        "label": "AI-default font stack (Inter / Geist / Space Grotesk)",
        "short": "Slop fonts",
        "category": "fonts",
        "weight": 8,
        "fn": _slop_fonts,
    }
)
_register(
    {
        "id": "purple_accent",
        "label": "VibeCode Purple — filled indigo/violet CTAs",
        "short": "Vibe purple",
        "category": "colors",
        "weight": 8,
        "fn": _vibe_purple,
    }
)
_register(
    {
        "id": "gradient_text",
        "label": "Hero gradient text (background-clip:text)",
        "short": "Gradient text",
        "category": "colors",
        "weight": 6,
        "fn": _gradient_text,
    }
)
_register(
    {
        "id": "gradient_backgrounds",
        "label": "Gradient-heavy backgrounds (5+ elements)",
        "short": "Gradient bgs",
        "category": "colors",
        "weight": 4,
        "fn": _gradient_backgrounds,
    }
)
_register(
    {
        "id": "accent_stripe",
        "label": "Colored top/left card borders (the AI em-dash)",
        "short": "Accent stripe",
        "category": "layout",
        "weight": 6,
        "fn": _accent_stripe,
    }
)
_register(
    {
        "id": "glassmorphism",
        "label": "Glassmorphism (backdrop-filter blur on translucent layers)",
        "short": "Glass",
        "category": "css",
        "weight": 4,
        "fn": _glassmorphism,
    }
)
_register(
    {
        "id": "colored_glows",
        "label": "Big colored box-shadow glows (purple/blue/pink)",
        "short": "Colored glows",
        "category": "css",
        "weight": 4,
        "fn": _colored_glows,
    }
)
_register(
    {
        "id": "centered_hero",
        "label": "Centered hero in generic sans (Inter-style)",
        "short": "Centered hero",
        "category": "layout",
        "weight": 4,
        "fn": _centered_hero,
    }
)
_register(
    {
        "id": "hero_eyebrow_pill",
        "label": 'Eyebrow pill above hero ("Now in beta" / "New")',
        "short": "Eyebrow pill",
        "category": "layout",
        "weight": 5,
        "fn": _hero_eyebrow_pill,
    }
)
_register(
    {
        "id": "all_caps_labels",
        "label": "All-caps section labels (text-transform:uppercase)",
        "short": "All-caps",
        "category": "fonts",
        "weight": 3,
        "fn": _all_caps_labels,
    }
)
_register(
    {
        "id": "perma_dark_mode",
        "label": "Perma dark mode + medium-grey body text",
        "short": "Perma dark",
        "category": "colors",
        "weight": 4,
        "fn": _perma_dark_mode,
    }
)
_register(
    {
        "id": "icon_card_grid",
        "label": "Identical feature cards with icon on top",
        "short": "Icon cards",
        "category": "layout",
        "weight": 4,
        "fn": _icon_card_grid,
    }
)
_register(
    {
        "id": "numbered_steps",
        "label": 'Numbered "1 · 2 · 3" step sequences',
        "short": "Numbered steps",
        "category": "layout",
        "weight": 3,
        "fn": _numbered_steps,
    }
)
_register(
    {
        "id": "stat_banner",
        "label": 'Big-number stat banner ("10k+", "99.9%", "$2M+")',
        "short": "Stat banner",
        "category": "layout",
        "weight": 3,
        "fn": _stat_banner,
    }
)
_register(
    {
        "id": "faq_accordion",
        "label": "FAQ accordion in the lower half",
        "short": "FAQ",
        "category": "layout",
        "weight": 2,
        "fn": _faq_accordion,
    }
)
_register(
    {
        "id": "gradient_letter_avatars",
        "label": "Gradient-letter avatars (testimonial slop)",
        "short": "Letter avatars",
        "category": "images",
        "weight": 5,
        "fn": _gradient_letter_avatars,
    }
)
_register(
    {
        "id": "bento_grid",
        "label": "Bento-grid wall — mixed-span rounded card grid",
        "short": "Bento grid",
        "category": "layout",
        "weight": 4,
        "fn": _bento_grid,
    }
)
_register(
    {
        "id": "aurora_mesh_gradient",
        "label": "Aurora / mesh gradient blobs (blurred glowing backdrop)",
        "short": "Aurora blobs",
        "category": "css",
        "weight": 5,
        "fn": _aurora_mesh_gradient,
    }
)
_register(
    {
        "id": "ai_sparkle_badges",
        "label": 'AI-sparkle badges (✨ / Sparkles "magic" tells)',
        "short": "AI sparkles",
        "category": "images",
        "weight": 3,
        "fn": _ai_sparkle_badges,
    }
)
_register(
    {
        "id": "cream_default_bg",
        "label": "Cream / beige default page background",
        "short": "Cream bg",
        "category": "colors",
        "weight": 7,
        "fn": _cream_default_bg,
    }
)
_register(
    {
        "id": "low_contrast_text",
        "label": "Washed-out grey body text (below WCAG AA on a light background)",
        "short": "Low contrast",
        "category": "colors",
        "weight": 7,
        "fn": _low_contrast_text,
    }
)
_register(
    {
        "id": "crushed_tracking",
        "label": "Crushed letter-spacing on display type",
        "short": "Crushed tracking",
        "category": "fonts",
        "weight": 5,
        "fn": _crushed_tracking,
    }
)
_register(
    {
        "id": "gray_on_color",
        "label": "Gray text on a colored background",
        "short": "Gray-on-color",
        "category": "colors",
        "weight": 4,
        "fn": _gray_on_color,
    }
)
_register(
    {
        "id": "oversized_hero_h1",
        "label": "Oversized hero headline (long sentence at display size)",
        "short": "Oversized H1",
        "category": "fonts",
        "weight": 4,
        "fn": _oversized_hero_h1,
    }
)
_register(
    {
        "id": "nested_cards",
        "label": "Cards nested inside cards",
        "short": "Nested cards",
        "category": "layout",
        "weight": 4,
        "fn": _nested_cards,
    }
)
_register(
    {
        "id": "wide_body_tracking",
        "label": "Wide letter-spacing on body text",
        "short": "Wide tracking",
        "category": "fonts",
        "weight": 3,
        "fn": _wide_body_tracking,
    }
)
_register(
    {
        "id": "flat_type_hierarchy",
        "label": "Flat type hierarchy (sizes too close together)",
        "short": "Flat hierarchy",
        "category": "fonts",
        "weight": 3,
        "fn": _flat_type_hierarchy,
    }
)


def run_patterns(ctx: SlopContext) -> list[dict[str, Any]]:
    """Run every pattern; returns rows [{id,label,short,category,weight,triggered,evidence}]."""
    rows: list[dict[str, Any]] = []
    for p in _PATTERNS:
        try:
            evidence = p["fn"](ctx)
        except Exception as exc:  # noqa: BLE001 - one pattern must not kill the rest
            evidence = {"triggered": False, "error": f"{type(exc).__name__}: {exc}"}
        rows.append(
            {
                "id": p["id"],
                "label": p["label"],
                "short": p["short"],
                "category": p["category"],
                "weight": p["weight"],
                "triggered": bool(evidence.get("triggered")),
                "evidence": evidence,
            }
        )
    return rows
