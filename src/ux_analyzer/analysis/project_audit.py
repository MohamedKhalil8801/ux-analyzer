"""Live-page UX audit across application start URLs for experiment reports."""

from __future__ import annotations

import asyncio
import base64
import io
from collections.abc import Sequence
from typing import Any

from ux_analyzer.analysis.accessibility import analyze_accessibility
from ux_analyzer.analysis.geo import analyze_geo
from ux_analyzer.analysis.imagery import analyze_imagery
from ux_analyzer.analysis.meta_semantic import analyze_meta_semantic
from ux_analyzer.analysis.performance import analyze_performance

AUDIT_SCHEMA_VERSION = "ux-audit-v1"
AUDIT_FILENAME = "ux-audit.json"


async def _visual_analyze(url: str) -> list[Any]:
    """Visual UI-fundamental analysis: snapshot + visual pipeline + slop.

    Returns ``[visual_issues, slop_report]`` where ``slop_report`` is the
    full 27-pattern + 9-copy slop scorecard for the page (or None when the
    capture fails). The two share one browser pass so live audits stay cheap.
    """
    try:
        from ux_analyzer.analysis.slop.pipeline import analyze_slop
        from ux_analyzer.analysis.visual.pipeline import analyze_snapshot
        from ux_analyzer.analysis.visual.snapshot import snapshot_from_dict
    except Exception:
        return []
    def _run() -> list[Any]:
        try:
            from PIL import Image, ImageDraw
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch()
                ctx = browser.new_context(
                    viewport={"width": 1280, "height": 800},
                    device_scale_factor=1,
                    user_agent="Mozilla/5.0 SlopDetector/1.0 (+https://github.com/ravidsrk/slop-detect)",
                )
                page = ctx.new_page()
                try:
                    page.goto(url, wait_until="networkidle", timeout=60000)
                    page.wait_for_timeout(2000)
                    try:
                        page.wait_for_function("() => document.fonts.ready.then(() => true)", timeout=5000)
                    except Exception:
                        pass
                    try:
                        page.evaluate("() => { return Promise.all(Array.from(document.images).map(img => img.complete ? Promise.resolve() : new Promise(r => { img.addEventListener('load', () => r(true), {once:true}); img.addEventListener('error', () => r(true), {once:true}); setTimeout(() => r(true), 3000); }))); }")
                    except Exception:
                        pass
                    try:
                        page.evaluate("() => { return new Promise(r => { if (document.readyState === 'complete') r(true); else window.addEventListener('load', () => r(true), {once:true}); setTimeout(() => r(true), 2000); }); }")
                    except Exception:
                        pass
                    page.wait_for_timeout(1000)
                    style_props = ["display","position","flex-direction","justify-content","align-items","gap","row-gap","column-gap","grid-template-columns","font-family","font-size","font-weight","font-style","line-height","letter-spacing","text-transform","text-align","text-decoration-line","color","background-color","background-image","background-clip","backdrop-filter","-webkit-backdrop-filter","filter","margin-top","margin-right","margin-bottom","margin-left","padding-top","padding-right","padding-bottom","padding-left","border-top-width","border-right-width","border-bottom-width","border-left-width","border-top-color","border-right-color","border-bottom-color","border-left-color","border-radius","opacity","width","height","box-shadow","grid-column","transform","perspective"]
                    snapshot_js = """
                    (props) => {
                      const root = document.querySelector('[data-uxa-snapshot-root]') || document.body;
                      const nodes = [];
                      const walk = (el, parentIdx, depth) => {
                        if (nodes.length > 8000) return;
                        const r = el.getBoundingClientRect();
                        const cs = getComputedStyle(el);
                        const idx = nodes.length;
                        const styles = {};
                        for (const p of props) { styles[p] = cs.getPropertyValue(p); }
                        let ownText = '';
                        for (const n of el.childNodes) {
                          if (n.nodeType === Node.TEXT_NODE) ownText += n.textContent;
                        }
                        nodes.push({i: idx, parent: parentIdx, depth, tag: el.tagName.toLowerCase(), cls: (typeof el.className === 'string') ? el.className : '', id: el.id || '', text: ownText.trim().slice(0, 300), box: {x: Math.round(r.x * 100) / 100, y: Math.round(r.y * 100) / 100, w: Math.round(r.width * 100) / 100, h: Math.round(r.height * 100) / 100}, styles});
                        for (const c of el.children) walk(c, idx, depth + 1);
                      };
                      walk(root, -1, 0);
                      const rb = document.querySelector('[data-uxa-snapshot-root], body').getBoundingClientRect();
                      return { rootBox: {x: rb.x, y: rb.y, w: rb.width, h: rb.height}, nodes };
                    }
                    """
                    raw = page.evaluate(snapshot_js, style_props)
                    if "rootBox" not in raw:
                        rb = page.evaluate("() => { const r=document.body.getBoundingClientRect(); return {x:r.x,y:r.y,w:r.width,h:r.height}; }")
                        raw["rootBox"] = rb
                    meta = page.evaluate(
                        "() => {"
                        " function vt(root){ if(!root) return ''; var t = root.innerText != null ? root.innerText : root.textContent; return (t||'').replace(/\\u00AD/g,''); }"
                        " var main = document.querySelector('main, article, [role=\"main\"]') || document.body;"
                        " var clone = main.cloneNode(true);"
                        " var strip = clone.querySelectorAll('nav, footer, header, script, style, noscript, svg, code, pre, [aria-hidden=\"true\"]');"
                        " for (var i=0;i<strip.length;i++){ if (strip[i].parentNode) strip[i].parentNode.removeChild(strip[i]); }"
                        " var text = vt(clone).trim();"
                        " var headings=[]; var hs=clone.querySelectorAll('h1,h2,h3,h4,li,dt');"
                        " for (var j=0;j<hs.length && headings.length<200;j++){ var ht=(hs[j].innerText||hs[j].textContent||'').trim(); if (ht) headings.push(ht.slice(0,200)); }"
                        " var paragraphs=[]; var ps=clone.querySelectorAll('p');"
                        " for (var k=0;k<ps.length && paragraphs.length<200;k++){ var pt=(ps[k].innerText||ps[k].textContent||'').trim(); if (pt) paragraphs.push(pt.slice(0,400)); }"
                        " var words = text ? text.split(/\\s+/).filter(Boolean) : [];"
                        " var centerEl = document.elementFromPoint(window.innerWidth / 2, window.innerHeight / 2);"
                        " return { viewport:{w:window.innerWidth,h:window.innerHeight}, docHeight:document.documentElement.scrollHeight, scrollY:window.scrollY,"
                        "   surface:{ htmlBg:getComputedStyle(document.documentElement).backgroundColor, bodyBg:getComputedStyle(document.body).backgroundColor, centerBg:centerEl ? getComputedStyle(centerEl).backgroundColor : '' },"
                        "   textContext:{ text:text.slice(0,200000), headings:headings, paragraphs:paragraphs, wordCount:words.length } };"
                        " }"
                    )
                    snap = snapshot_from_dict(raw)
                    issues = list(analyze_snapshot(snap))
                    slop_report = None
                    try:
                        slop_report = analyze_slop(
                            snap,
                            viewport_w=int(meta.get("viewport", {}).get("w", 1280)),
                            viewport_h=int(meta.get("viewport", {}).get("h", 800)),
                            doc_height=int(meta.get("docHeight", 0) or 0),
                            scroll_y=int(meta.get("scrollY", 0) or 0),
                            text_context=meta.get("textContext"),
                            surface=meta.get("surface"),
                        )
                    except Exception:
                        slop_report = None
                    try:
                        png_bytes = page.screenshot(full_page=True, type="png")
                        full_img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
                    except Exception:
                        full_img = None
                    for issue in issues:
                        element_nodes = []
                        for ref in getattr(issue, "element_refs", ()):
                            for node in snap.nodes:
                                try:
                                    if snap.selector(node) == ref:
                                        element_nodes.append(node)
                                        break
                                except Exception:
                                    continue
                        selectors: list[str] = []
                        xpaths: list[str] = []
                        boxes: list[dict[str, float]] = []
                        screenshots: list[str] = []
                        for node in element_nodes:
                            try:
                                css = snap.css_selector(node)
                            except Exception:
                                css = snap.selector(node)
                            try:
                                xp = snap.xpath(node)
                            except Exception:
                                xp = ""
                            selectors.append(css)
                            xpaths.append(xp)
                            boxes.append({"x": round(node.box.x, 1), "y": round(node.box.y, 1), "w": round(node.box.w, 1), "h": round(node.box.h, 1)})
                            screenshot_b64: str | None = None
                            try:
                                loc = page.locator(css)
                                try:
                                    loc.wait_for(state="attached", timeout=1500)
                                except Exception:
                                    pass
                                el_png = loc.screenshot(type="png")
                                el_img = Image.open(io.BytesIO(el_png)).convert("RGB")
                                pad = 24
                                # For very large elements, keep padding but allow larger canvas;
                                # we will thumbnail later, so no need to reject
                                padded_w = el_img.width + pad * 2
                                padded_h = el_img.height + pad * 2
                                # Use a light gray canvas to distinguish element bounds
                                padded = Image.new("RGB", (padded_w, padded_h), (240, 241, 242))
                                padded.paste(el_img, (pad, pad))
                                draw = ImageDraw.Draw(padded)
                                for w in range(3):
                                    draw.rectangle([pad - w, pad - w, pad + el_img.width + w, pad + el_img.height + w], outline=(220, 30, 30))
                                draw.rectangle([pad - 1, pad - 1, pad + el_img.width + 1, pad + el_img.height + 1], outline=(255, 255, 255))
                                max_dim = 680
                                if padded.width > max_dim or padded.height > max_dim:
                                    padded.thumbnail((max_dim, max_dim), Image.LANCZOS)
                                buf = io.BytesIO()
                                padded.save(buf, format="JPEG", quality=86, optimize=True)
                                screenshot_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
                            except Exception:
                                if full_img is not None:
                                    try:
                                        is_large = node.box.w > 900 or node.box.h > 500
                                        if is_large:
                                            pad = 32
                                            x0 = max(0, int(node.box.x - pad))
                                            y0 = max(0, int(node.box.y - pad))
                                            x1 = min(full_img.width, x0 + 640)
                                            y1 = min(full_img.height, y0 + 420)
                                            if node.box.h > 600:
                                                cy = int(node.box.y + 80)
                                                y0 = max(0, cy - 180)
                                                y1 = min(full_img.height, y0 + 400)
                                        else:
                                            pad = max(32, int(min(node.box.w, node.box.h) * 0.4))
                                            pad = min(pad, 80)
                                            x0 = max(0, int(node.box.x - pad))
                                            y0 = max(0, int(node.box.y - pad))
                                            x1 = min(full_img.width, int(node.box.x + node.box.w + pad))
                                            y1 = min(full_img.height, int(node.box.y + node.box.h + pad))
                                            if (x1 - x0) < 120:
                                                cx = (x0 + x1) // 2
                                                x0 = max(0, cx - 60)
                                                x1 = min(full_img.width, cx + 60)
                                            if (y1 - y0) < 120:
                                                cy = (y0 + y1) // 2
                                                y0 = max(0, cy - 60)
                                                y1 = min(full_img.height, cy + 60)
                                        if x1 > x0 and y1 > y0:
                                            crop = full_img.crop((x0, y0, x1, y1))
                                            draw = ImageDraw.Draw(crop)
                                            ex0 = int(node.box.x - x0)
                                            ey0 = int(node.box.y - y0)
                                            ex1 = int(node.box.x + node.box.w - x0)
                                            ey1 = int(node.box.y + node.box.h - y0)
                                            ex0 = max(0, min(ex0, crop.width - 1))
                                            ey0 = max(0, min(ey0, crop.height - 1))
                                            ex1 = max(0, min(ex1, crop.width))
                                            ey1 = max(0, min(ey1, crop.height))
                                            is_thin = node.box.w <= 2 or node.box.h <= 2
                                            outline_w = 4 if is_thin else 3
                                            for w in range(outline_w):
                                                draw.rectangle([ex0 - w, ey0 - w, ex1 + w, ey1 + w], outline=(220, 30, 30))
                                            if not is_thin:
                                                draw.rectangle([ex0 - 1, ey0 - 1, ex1 + 1, ey1 + 1], outline=(255, 255, 255))
                                            max_dim = 560
                                            if crop.width > max_dim or crop.height > max_dim:
                                                crop.thumbnail((max_dim, max_dim), Image.LANCZOS)
                                            buf = io.BytesIO()
                                            crop.save(buf, format="JPEG", quality=84, optimize=True)
                                            screenshot_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
                                    except Exception:
                                        screenshot_b64 = None
                            if screenshot_b64:
                                screenshots.append(screenshot_b64)
                        # Combined annotated views grouped by common parent, so
                        # scattered elements never produce a huge mostly-empty
                        # union crop.
                        combined_screenshots: list[str] = []
                        if len(element_nodes) > 1:
                            by_parent: dict[int, list[Any]] = {}
                            for n in element_nodes:
                                by_parent.setdefault(n.parent, []).append(n)
                            for parent_idx, members in by_parent.items():
                                if len(members) < 2 or len(combined_screenshots) >= 3:
                                    continue
                                try:
                                    parent_node = next(
                                        (n for n in snap.nodes if n.i == parent_idx), None
                                    )
                                    if parent_node is None:
                                        continue
                                    parent_css = snap.css_selector(parent_node)
                                    parent_loc = page.locator(parent_css)
                                    try:
                                        parent_loc.wait_for(state="attached", timeout=1500)
                                    except Exception:
                                        pass
                                    png = parent_loc.screenshot(type="png")
                                    parent_img = Image.open(io.BytesIO(png)).convert("RGB")
                                    # Skip enormous containers (near whole-page sections).
                                    if parent_img.width * parent_img.height > 1200000:
                                        continue
                                    draw = ImageDraw.Draw(parent_img)
                                    for node in members:
                                        ex0 = int(node.box.x - parent_node.box.x)
                                        ey0 = int(node.box.y - parent_node.box.y)
                                        ex1 = int(node.box.x + node.box.w - parent_node.box.x)
                                        ey1 = int(node.box.y + node.box.h - parent_node.box.y)
                                        ex0 = max(0, min(ex0, parent_img.width - 1))
                                        ey0 = max(0, min(ey0, parent_img.height - 1))
                                        ex1 = max(0, min(ex1, parent_img.width))
                                        ey1 = max(0, min(ey1, parent_img.height))
                                        for w in range(3):
                                            draw.rectangle([ex0 - w, ey0 - w, ex1 + w, ey1 + w], outline=(220, 30, 30))
                                        draw.rectangle([ex0 - 1, ey0 - 1, ex1 + 1, ey1 + 1], outline=(255, 255, 255))
                                    if parent_img.width > 720 or parent_img.height > 540:
                                        parent_img.thumbnail((720, 540), Image.LANCZOS)
                                    buf = io.BytesIO()
                                    parent_img.save(buf, format="JPEG", quality=82, optimize=True)
                                    combined_screenshots.append(
                                        f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
                                    )
                                except Exception:
                                    continue
                            # Union-crop fallback only when the elements actually
                            # fill the union (no scattered empty space).
                            if not combined_screenshots and full_img is not None:
                                try:
                                    min_x = min(n.box.x for n in element_nodes)
                                    min_y = min(n.box.y for n in element_nodes)
                                    max_x = max(n.box.x + n.box.w for n in element_nodes)
                                    max_y = max(n.box.y + n.box.h for n in element_nodes)
                                    union_w = max_x - min_x
                                    union_h = max_y - min_y
                                    if union_w * union_h > 2500000 or max(union_w, union_h) > 2000:
                                        raise ValueError("union too large, skip combined")
                                    element_area = sum(n.box.w * n.box.h for n in element_nodes)
                                    if element_area / max(union_w * union_h, 1) < 0.2:
                                        raise ValueError("elements too scattered for a useful union")
                                    pad = 32
                                    x0 = max(0, int(min_x - pad))
                                    y0 = max(0, int(min_y - pad))
                                    x1 = min(full_img.width, int(max_x + pad))
                                    y1 = min(full_img.height, int(max_y + pad))
                                    if x1 > x0 and y1 > y0:
                                        crop = full_img.crop((x0, y0, x1, y1))
                                        draw = ImageDraw.Draw(crop)
                                        for node in element_nodes:
                                            ex0 = int(node.box.x - x0)
                                            ey0 = int(node.box.y - y0)
                                            ex1 = int(node.box.x + node.box.w - x0)
                                            ey1 = int(node.box.y + node.box.h - y0)
                                            for w in range(3):
                                                draw.rectangle([ex0 - w, ey0 - w, ex1 + w, ey1 + w], outline=(220, 30, 30))
                                            draw.rectangle([ex0 - 1, ey0 - 1, ex1 + 1, ey1 + 1], outline=(255, 255, 255))
                                        if crop.width > 720 or crop.height > 540:
                                            crop.thumbnail((720, 540), Image.LANCZOS)
                                        buf = io.BytesIO()
                                        crop.save(buf, format="JPEG", quality=76, optimize=True)
                                        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                                        combined_screenshots.append(f"data:image/jpeg;base64,{b64}")
                                except Exception:
                                    pass
                        combined_screenshot = combined_screenshots[0] if combined_screenshots else None
                        try:
                            ev = dict(issue.evidence) if isinstance(issue.evidence, dict) else {}
                            if selectors:
                                ev["element_selectors"] = selectors
                            if xpaths:
                                ev["element_xpaths"] = xpaths
                            if boxes:
                                ev["element_boxes"] = boxes
                            if screenshots:
                                ev["element_screenshots"] = screenshots[:3]
                            if combined_screenshots:
                                ev["combined_screenshots"] = combined_screenshots
                            if combined_screenshot:
                                ev["combined_screenshot"] = combined_screenshot
                            if selectors:
                                ev["element_selector"] = selectors[0]
                            if xpaths:
                                ev["element_xpath"] = xpaths[0]
                            object.__setattr__(issue, "evidence", ev)
                        except Exception:
                            pass
                    return [issues, slop_report]
                finally:
                    browser.close()
        except Exception:
            return []
    try:
        return await asyncio.to_thread(_run)
    except Exception:
        return []


_CATEGORIES = (
    ("GEO", analyze_geo),
    ("meta-semantic", analyze_meta_semantic),
    ("performance", analyze_performance),
    ("accessibility", analyze_accessibility),
    ("imagery", analyze_imagery),
    ("visual", _visual_analyze),
)


async def audit_url(url: str) -> dict[str, Any]:
    results = await asyncio.gather(*(analyze(url) for _, analyze in _CATEGORIES))
    issues: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    slop: dict[str, Any] | None = None
    for (category, _), category_issues in zip(_CATEGORIES, results, strict=True):
        if category == "visual":
            # _visual_analyze returns [issues, slop_report]
            visual_issues = category_issues[0] if category_issues else []
            slop = category_issues[1] if len(category_issues) > 1 else None
            category_issues = visual_issues
        counts[category] = len(category_issues)
        for issue in category_issues:
            issues.append({"category": category, "check_id": issue.check_id, "title": issue.title, "severity": issue.severity, "evidence": issue.evidence})
    report: dict[str, Any] = {"url": url, "counts": counts, "total": len(issues), "issues": issues}
    if slop is not None:
        report["slop"] = slop
    return report


async def audit_urls(urls: Sequence[str]) -> dict[str, Any]:
    unique_urls = tuple(dict.fromkeys(url for url in urls if url))
    url_reports: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for url in unique_urls:
        try:
            url_reports.append(await audit_url(url))
        except Exception as error:
            errors.append({"url": url, "error": f"{type(error).__name__}: {error}"[:512]})
    total = sum(report["total"] for report in url_reports)
    return {"schema_version": AUDIT_SCHEMA_VERSION, "total_issues": total, "urls": url_reports, "errors": errors}


def audit_urls_sync(urls: Sequence[str]) -> dict[str, Any]:
    return asyncio.run(audit_urls(urls))
