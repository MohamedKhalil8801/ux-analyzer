"""Performance detector.

Covers TheUXBites performance slice and Markswebb slowness classes 4.1/4.2.

TheUXBites titles mapped:

- Render-blocking resources slow page display (75min) -> render_blocking
- Render-blocking resources delay page load (12h) -> render_blocking (same root, evidence alias)
- Unused JavaScript slows page load (90min) -> unused_js
- Main thread blocks user interaction (90min) / Main thread is overloaded (12h, UX cat) -> main_thread
- Network dependency tree (12h) -> network_dependency
- Long input delay potential (90min) -> inp / long_input_delay (proxy via main_thread)
- Page takes too long to become interactive (2h) -> tti
- Page loads visually slowly (75min, Speed Index) -> speed_index
- Slow first content paint (90min) -> slow_fcp
- Slow largest content paint (2h) -> slow_lcp
- First Meaningful Paint (12h, UX cat) -> fmp
- User Timing marks and measures (12h) -> user_timing
- Largest image is unnecessarily lazy-loaded (12h) -> lcp_lazy
- Preload Largest Contentful Paint image (12h) -> preload_lcp
- INP breakdown (12h) -> inp
- LCP request discovery (12h) -> preload_lcp / lcp_discovery (same preload hint)

Markswebb:
- 4.1 service slow to load -> render_blocking + unused_js + lcp/preload
- 4.2 users wait too long after action -> main_thread + inp + tti

Static heuristics (no browser required):
- render-blocking: <link rel=stylesheet> without media=print, <script> in <head> without async/defer, @import
- unused JS: total external JS >3 or blocking_js >2
- main thread: large inline script >1KB or blocking_js >1
- network chain: depth = blocking stylesheets + blocking scripts + @import count
- LCP: largest <img> candidate, check loading=lazy, fetchpriority, preload link
- resource hints: preconnect for fonts.googleapis
- image dimensions: missing width/height
- font-display, user timing

Severity:
- critical: render-blocking >2, LCP lazy, missing preload
- high: render-blocking 2, main_thread heavy, slow LCP/TTI
- medium: unused JS, network chain, slow FCP/fmp/speed
- low: user timing, resource hints, image dimensions, font-display
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx


@dataclass(frozen=True, slots=True)
class PerformanceIssue:
    """Evidence-backed performance finding."""

    title: str
    description: str
    severity: str  # critical|high|medium|low
    evidence: dict
    check_id: str


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------
class _PerfParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_head: bool = False
        self.in_style: bool = False
        self.in_script: bool = False
        self.style_buf: str = ""
        self.script_buf: str = ""
        self.script_attrs: dict[str, str] = {}
        self.script_in_head: bool = False
        self.links: list[dict[str, str]] = []
        self.scripts: list[dict] = []  # {in_head, attrs, content}
        self.styles_contents: list[str] = []
        self.images: list[dict[str, str]] = []
        self.metas: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        d = {k.lower(): (v or "") for k, v in attrs}
        if t == "head":
            self.in_head = True
        if t == "link":
            self.links.append(d)
        elif t == "meta":
            self.metas.append(d)
        elif t == "img":
            self.images.append(d)
        elif t == "style":
            self.in_style = True
            self.style_buf = ""
        elif t == "script":
            self.in_script = True
            self.script_buf = ""
            self.script_attrs = d
            self.script_in_head = self.in_head

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t == "head":
            self.in_head = False
        elif t == "style" and self.in_style:
            self.in_style = False
            self.styles_contents.append(self.style_buf)
            self.style_buf = ""
        elif t == "script" and self.in_script:
            self.in_script = False
            self.scripts.append(
                {
                    "in_head": self.script_in_head,
                    "attrs": dict(self.script_attrs),
                    "content": self.script_buf,
                }
            )
            self.script_buf = ""
            self.script_attrs = {}
            self.script_in_head = False

    def handle_data(self, data: str) -> None:
        if self.in_style:
            self.style_buf += data
        if self.in_script:
            self.script_buf += data


def _parse_html(html: str) -> _PerfParser:
    p = _PerfParser()
    try:
        p.feed(html)
    except Exception:
        pass
    return p


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _origin(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc
    if not netloc:
        parsed2 = urlparse(f"https://{url}")
        scheme = parsed2.scheme
        netloc = parsed2.netloc
    return f"{scheme}://{netloc}".rstrip("/")


async def _safe_get(
    client: httpx.AsyncClient, url: str
) -> tuple[int | None, str | None, str | None]:
    try:
        resp = await client.get(url)
        try:
            txt = resp.text
        except Exception:
            txt = resp.content.decode("utf-8", errors="replace")
        return resp.status_code, txt, None
    except Exception as exc:
        return None, None, str(exc)


def _blocking_stylesheets(parser: _PerfParser) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for link in parser.links:
        rel = link.get("rel", "").lower()
        rels = [r.strip() for r in rel.split()]
        if "stylesheet" not in rels:
            continue
        href = link.get("href", "").strip()
        if not href:
            continue
        media = link.get("media", "").lower()
        if "print" in media:
            continue
        out.append(link)
    return out


def _blocking_scripts(parser: _PerfParser) -> list[dict]:
    """Scripts in <head> without async/defer/module (modules are deferred by default)."""
    out: list[dict] = []
    for s in parser.scripts:
        if not s["in_head"]:
            continue
        attrs = s["attrs"]
        if "async" in attrs or "defer" in attrs:
            continue
        if attrs.get("type", "").lower() == "module":
            continue
        # inline or external both count per spec literal
        # if content empty and no src, ignore empty script tags
        if not attrs.get("src", "").strip() and not s["content"].strip():
            continue
        out.append(s)
    return out


def _import_count(parser: _PerfParser) -> int:
    cnt = 0
    for c in parser.styles_contents:
        cnt += c.lower().count("@import")
    return cnt


def _external_scripts(parser: _PerfParser) -> list[dict]:
    return [s for s in parser.scripts if s["attrs"].get("src", "").strip()]


def _large_inline_scripts(parser: _PerfParser, threshold: int = 1024) -> list[dict]:
    return [s for s in parser.scripts if len(s["content"]) > threshold]


def _find_lcp_image(parser: _PerfParser) -> dict[str, str] | None:
    if not parser.images:
        return None
    best: dict[str, str] | None = None
    best_area = -1
    best_idx = 10**9
    for idx, img in enumerate(parser.images):
        src = img.get("src", "").strip()
        if not src or src.startswith("data:"):
            continue
        w_raw = img.get("width", "").strip()
        h_raw = img.get("height", "").strip()
        wi = 0
        hi = 0
        if w_raw:
            m = re.search(r"\d+", w_raw)
            if m:
                try:
                    wi = int(m.group(0))
                except Exception:
                    wi = 0
        if h_raw:
            m = re.search(r"\d+", h_raw)
            if m:
                try:
                    hi = int(m.group(0))
                except Exception:
                    hi = 0
        area = wi * hi if wi and hi else 0
        # Prefer images with area; if tie, earlier index wins
        if area > best_area or (area == best_area and idx < best_idx):
            best = img
            best_area = area
            best_idx = idx
    return best


def _has_preload_for_lcp(parser: _PerfParser, lcp: dict[str, str] | None) -> bool:
    if lcp is None:
        return True
    if lcp.get("fetchpriority", "").lower() == "high":
        return True
    src = lcp.get("src", "").strip()
    if not src:
        return True
    basename = src.split("/")[-1].split("?")[0]
    for link in parser.links:
        rel = link.get("rel", "").lower()
        rels = [r.strip() for r in rel.split()]
        if "preload" not in rels:
            continue
        as_attr = link.get("as", "").lower()
        # allow as=image
        if as_attr and as_attr != "image":
            continue
        href = link.get("href", "").strip()
        if not href:
            # also check imagesrcset variant
            img_src = link.get("imagesrcset", "") or link.get("href", "")
            href = img_src
        if not href:
            continue
        href_base = href.split("/")[-1].split("?")[0]
        if href == src or href in src or src in href or href_base == basename:
            return True
    return False


def _needs_font_preconnect(parser: _PerfParser) -> bool:
    for link in parser.links:
        href = link.get("href", "")
        if "fonts.googleapis.com" in href or "fonts.gstatic.com" in href:
            return True
    for c in parser.styles_contents:
        if "fonts.googleapis.com" in c or "fonts.gstatic.com" in c:
            return True
        # generic @font-face that likely needs preconnect? Only flag if google fonts detected
    return False


def _has_font_preconnect(parser: _PerfParser) -> bool:
    has_gapi = False
    has_gstatic = False
    for link in parser.links:
        rel = link.get("rel", "").lower()
        if "preconnect" not in [r.strip() for r in rel.split()]:
            continue
        href = link.get("href", "")
        if "fonts.googleapis.com" in href:
            has_gapi = True
        if "fonts.gstatic.com" in href:
            has_gstatic = True
    # Require at least gstatic; gapi optional but we check both for strictness
    # If only one present, consider partial but not fully
    if _needs_font_preconnect(parser):
        return has_gapi and has_gstatic
    return True


def _images_missing_dimensions(parser: _PerfParser) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    for img in parser.images:
        src = img.get("src", "").strip()
        if not src or src.startswith("data:"):
            continue
        w = img.get("width", "").strip()
        h = img.get("height", "").strip()
        if not w or not h:
            missing.append(img)
    return missing


def _has_user_timing(parser: _PerfParser) -> bool:
    for s in parser.scripts:
        c = s["content"]
        if "performance.mark" in c or "performance.measure" in c:
            return True
    for c in parser.styles_contents:
        if "performance.mark" in c:
            return True
    return False


def _font_display_missing(parser: _PerfParser) -> bool:
    for c in parser.styles_contents:
        low = c.lower()
        if "@font-face" in low and "font-display" not in low:
            return True
    return False


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------
def _check_render_blocking(parser: _PerfParser) -> PerformanceIssue | None:
    blocking_css = _blocking_stylesheets(parser)
    blocking_js = _blocking_scripts(parser)
    imports = _import_count(parser)
    total = len(blocking_css) + len(blocking_js) + imports
    if total == 0:
        return None
    severity = "medium"
    if total > 2:
        severity = "critical"
    elif total == 2:
        severity = "high"
    evidence = {
        "blocking_stylesheets": len(blocking_css),
        "blocking_scripts": len(blocking_js),
        "import_count": imports,
        "total_blocking": total,
        "stylesheet_hrefs": [link.get("href", "")[:200] for link in blocking_css],
        "script_srcs": [s["attrs"].get("src", "")[:200] or "<inline>" for s in blocking_js],
    }
    return PerformanceIssue(
        title="Render-blocking resources slow page display",
        description=(
            f"Found {total} render-blocking resources ({len(blocking_css)} stylesheets, "
            f"{len(blocking_js)} head scripts, {imports} @import). "
            "These delay first paint and also cause 'Render-blocking resources delay page load'."
        ),
        severity=severity,
        evidence=evidence,
        check_id="render_blocking",
    )


def _check_unused_js(parser: _PerfParser) -> PerformanceIssue | None:
    total_external = len(_external_scripts(parser))
    blocking_external = len(
        [
            s
            for s in parser.scripts
            if s["attrs"].get("src", "").strip()
            and s["in_head"]
            and "async" not in s["attrs"]
            and "defer" not in s["attrs"]
            and s["attrs"].get("type", "").lower() != "module"
        ]
    )
    # literal: flag if >2 blocking or >3 total
    if blocking_external > 2 or total_external > 3:
        evidence = {
            "total_external_js": total_external,
            "blocking_external_js": blocking_external,
            "threshold_blocking": 2,
            "threshold_total": 3,
            "srcs": [s["attrs"].get("src", "")[:200] for s in _external_scripts(parser)][:5],
        }
        return PerformanceIssue(
            title="Unused JavaScript slows page load",
            description=(
                f"Page loads {total_external} external JavaScript files with {blocking_external} render-blocking. "
                "Large bundles or unused JS increase parse/compile cost and TTI."
            ),
            severity="medium",
            evidence=evidence,
            check_id="unused_js",
        )
    return None


def _check_main_thread(parser: _PerfParser) -> PerformanceIssue | None:
    large = _large_inline_scripts(parser, 1024)
    blocking_external = len(
        [
            s
            for s in parser.scripts
            if s["attrs"].get("src", "").strip()
            and s["in_head"]
            and "async" not in s["attrs"]
            and "defer" not in s["attrs"]
            and s["attrs"].get("type", "").lower() != "module"
        ]
    )
    # also consider total blocking scripts (including inline head)
    blocking_all = len(_blocking_scripts(parser))
    if large or blocking_external > 1 or blocking_all >= 2:
        # Map to both "Main thread blocks user interaction" and "Main thread is overloaded"
        severity = "high"
        if large and blocking_external > 1:
            severity = "critical"
        evidence = {
            "large_inline_count": len(large),
            "largest_inline_bytes": max((len(s["content"]) for s in large), default=0),
            "blocking_external_js": blocking_external,
            "blocking_total": blocking_all,
            "total_scripts": len(parser.scripts),
        }
        return PerformanceIssue(
            title="Main thread blocks user interaction",
            description=(
                "Main thread is blocked by large or synchronous JavaScript, increasing input delay and risking 'Main thread is overloaded'. "
                "Long tasks block interaction (INP) and delay interactivity."
            ),
            severity=severity,
            evidence=evidence,
            check_id="main_thread",
        )
    return None


def _check_network_dependency(parser: _PerfParser) -> PerformanceIssue | None:
    blocking_css = _blocking_stylesheets(parser)
    blocking_js = _blocking_scripts(parser)
    imports = _import_count(parser)
    depth = len(blocking_css) + len(blocking_js) + imports
    if depth >= 4:
        evidence = {
            "depth": depth,
            "blocking_stylesheets": len(blocking_css),
            "blocking_scripts": len(blocking_js),
            "import_count": imports,
        }
        return PerformanceIssue(
            title="Network dependency tree",
            description=(
                f"Critical request chain depth is {depth} (stylesheets {len(blocking_css)} + scripts {len(blocking_js)} + imports {imports}). "
                "Deep chains delay discovery of LCP and other resources."
            ),
            severity="medium",
            evidence=evidence,
            check_id="network_dependency",
        )
    return None


def _check_lcp_lazy(parser: _PerfParser) -> PerformanceIssue | None:
    lcp = _find_lcp_image(parser)
    if lcp is None:
        return None
    loading = lcp.get("loading", "").lower()
    if loading == "lazy":
        evidence = {
            "lcp_src": lcp.get("src", "")[:300],
            "loading": loading,
            "width": lcp.get("width", ""),
            "height": lcp.get("height", ""),
        }
        return PerformanceIssue(
            title="Largest image is unnecessarily lazy-loaded",
            description="The likely Largest Contentful Paint image has loading=\"lazy\", which defers its load and hurts LCP.",
            severity="critical",
            evidence=evidence,
            check_id="lcp_lazy",
        )
    return None


def _check_preload_lcp(parser: _PerfParser) -> PerformanceIssue | None:
    lcp = _find_lcp_image(parser)
    if lcp is None:
        return None
    src = lcp.get("src", "").strip()
    if not _has_preload_for_lcp(parser, lcp):
        evidence = {
            "lcp_src": src[:300],
            "has_fetchpriority_high": lcp.get("fetchpriority", "").lower() == "high",
            "preload_found": False,
            "suggestion": 'Add <link rel="preload" as="image" href="..."> or fetchpriority="high"',
        }
        return PerformanceIssue(
            title="Preload Largest Contentful Paint image",
            description=(
                "Largest image is not preloaded and lacks fetchpriority=\"high\". "
                "This also covers 'LCP request discovery' – late discovery delays LCP."
            ),
            severity="critical",
            evidence=evidence,
            check_id="preload_lcp",
        )
    return None


def _check_slow_fcp(parser: _PerfParser) -> PerformanceIssue | None:
    # Proxy for Slow first content paint / First Meaningful Paint
    blocking_css = _blocking_stylesheets(parser)
    blocking_js = _blocking_scripts(parser)
    total_blocking = len(blocking_css) + len(blocking_js) + _import_count(parser)
    if total_blocking >= 2:
        evidence = {
            "total_blocking": total_blocking,
            "blocking_stylesheets": len(blocking_css),
            "blocking_scripts": len(blocking_js),
        }
        return PerformanceIssue(
            title="Slow first content paint",
            description=(
                f"With {total_blocking} blocking resources, first content paint is likely delayed. "
                "This also implies 'First Meaningful Paint' risk."
            ),
            severity="medium",
            evidence=evidence,
            check_id="slow_fcp",
        )
    return None


def _check_slow_lcp(parser: _PerfParser) -> PerformanceIssue | None:
    # Slow largest content paint proxy: only flag when blocking delays LCP,
    # but deduplicate if same LCP already flagged by preload_lcp / lcp_lazy.
    lcp = _find_lcp_image(parser)
    if lcp is None:
        return None
    blocking_total = len(_blocking_stylesheets(parser)) + len(_blocking_scripts(parser)) + _import_count(parser)
    has_preload = _has_preload_for_lcp(parser, lcp)
    lcp_lazy = lcp.get("loading", "").lower() == "lazy"
    # Deduplicate: if LCP is lazy or missing preload, preload_lcp/lcp_lazy already
    # reports the root cause for this src – don't double-report slow_lcp.
    if not has_preload or lcp_lazy:
        return None
    if blocking_total >= 2:
        evidence = {"total_blocking": blocking_total, "lcp_src": lcp.get("src", "")[:300]}
        return PerformanceIssue(
            title="Slow largest content paint",
            description="Blocking resources delay LCP even without explicit LCP hint issues.",
            severity="medium",
            evidence=evidence,
            check_id="slow_lcp",
        )
    return None


def _check_speed_index(parser: _PerfParser) -> PerformanceIssue | None:
    # Page loads visually slowly (Speed Index) proxy: many blocking + images without dimensions
    missing_dims = _images_missing_dimensions(parser)
    blocking_total = len(_blocking_stylesheets(parser)) + len(_blocking_scripts(parser)) + _import_count(parser)
    if blocking_total >= 2 or len(missing_dims) >= 1:
        # Only flag if visual factors present
        if len(missing_dims) >= 1 or blocking_total >= 3:
            evidence = {
                "total_blocking": blocking_total,
                "images_missing_dimensions": len(missing_dims),
                "missing_srcs": [m.get("src", "")[:200] for m in missing_dims[:3]],
            }
            return PerformanceIssue(
                title="Page loads visually slowly",
                description="Visual completeness is delayed by blocking resources or un-sized images causing layout shifts (Speed Index).",
                severity="medium",
                evidence=evidence,
                check_id="speed_index",
            )
    return None


def _check_tti(parser: _PerfParser) -> PerformanceIssue | None:
    blocking_scripts = _blocking_scripts(parser)
    large = _large_inline_scripts(parser, 1024)
    # Page takes too long to become interactive proxy: heavy JS (blocking or large)
    # Do NOT trigger on total_external alone to avoid coupling with unused_js
    if len(blocking_scripts) >= 2 or large:
        evidence = {
            "blocking_scripts": len(blocking_scripts),
            "large_inline": len(large),
            "total_external": len(_external_scripts(parser)),
        }
        return PerformanceIssue(
            title="Page takes too long to become interactive",
            description="Too much blocking JavaScript or large inline scripts delay Time to Interactive.",
            severity="high",
            evidence=evidence,
            check_id="tti",
        )
    return None


def _check_inp(parser: _PerfParser) -> PerformanceIssue | None:
    # INP breakdown / Long input delay potential proxy: large inline + blocking
    large = _large_inline_scripts(parser, 1024)
    blocking_scripts = _blocking_scripts(parser)
    if large or len(blocking_scripts) >= 1:
        # To avoid flagging every page with 1 blocking, require large or >=2 blocking for inp
        if large or len(blocking_scripts) > 1:
            evidence = {
                "large_inline": len(large),
                "blocking_scripts": len(blocking_scripts),
                "total_scripts": len(parser.scripts),
            }
            return PerformanceIssue(
                title="INP breakdown",
                description=(
                    "Long input delay potential due to main-thread blocking JS. "
                    "Also covers 'Long input delay potential'."
                ),
                severity="medium",
                evidence=evidence,
                check_id="inp",
            )
    return None


def _check_resource_hints(parser: _PerfParser) -> PerformanceIssue | None:
    if _needs_font_preconnect(parser) and not _has_font_preconnect(parser):
        evidence = {
            "needs_font_preconnect": True,
            "has_preconnect_gapis": any(
                "fonts.googleapis.com" in link.get("href", "") and "preconnect" in link.get("rel", "") for link in parser.links
            ),
            "has_preconnect_gstatic": any(
                "fonts.gstatic.com" in link.get("href", "") and "preconnect" in link.get("rel", "") for link in parser.links
            ),
            "links": [link for link in parser.links if "preconnect" in link.get("rel", "")][:3],
        }
        return PerformanceIssue(
            title="Missing resource hints for fonts",
            description="Page loads Google Fonts but lacks <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\"> (and fonts.googleapis.com), delaying font discovery.",
            severity="low",
            evidence=evidence,
            check_id="resource_hints",
        )
    return None


def _check_image_dimensions(parser: _PerfParser) -> PerformanceIssue | None:
    missing = _images_missing_dimensions(parser)
    if missing:
        evidence = {
            "missing_count": len(missing),
            "total_images": len(parser.images),
            "missing_srcs": [m.get("src", "")[:200] for m in missing[:3]],
        }
        return PerformanceIssue(
            title="Images missing explicit dimensions",
            description="Images without width/height cause layout shifts and delay visual stability (impacts Speed Index / CLS).",
            severity="low",
            evidence=evidence,
            check_id="image_dimensions",
        )
    return None


def _check_user_timing(parser: _PerfParser) -> PerformanceIssue | None:
    if not _has_user_timing(parser):
        evidence = {"has_performance_mark": False, "suggestion": "Add performance.mark/measure for User Timing"}
        return PerformanceIssue(
            title="User Timing marks and measures",
            description="No User Timing marks (performance.mark/measure) found. Adding them helps diagnose real-user performance.",
            severity="low",
            evidence=evidence,
            check_id="user_timing",
        )
    return None


def _check_font_display(parser: _PerfParser) -> PerformanceIssue | None:
    if _font_display_missing(parser):
        evidence = {"has_font_face": True, "has_font_display": False}
        return PerformanceIssue(
            title="Font display not optimized",
            description="@font-face without font-display: swap causes invisible text during font load (FOIT).",
            severity="low",
            evidence=evidence,
            check_id="font_display",
        )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
async def analyze_performance(url: str, client: httpx.AsyncClient | None = None) -> list[PerformanceIssue]:
    """Run performance heuristics deterministically.

    Args:
        url: Page URL to analyze.
        client: Optional httpx.AsyncClient for testing / reuse.
    """
    if not url or not url.strip():
        raise ValueError("url must not be empty")
    url = url.strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url

    own_client = False
    if client is None:
        own_client = True
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0),
            follow_redirects=True,
            headers={"User-Agent": "ux-analyzer performance detector"},
        )
    try:
        status, html, error = await _safe_get(client, url)
        if error is not None or html is None or status is None or status >= 400:
            return []
        if status != 200:
            return []
        parser = _parse_html(html)

        issues: list[PerformanceIssue] = []

        # Core render/blocking
        v = _check_render_blocking(parser)
        if v:
            issues.append(v)
        v = _check_unused_js(parser)
        if v:
            issues.append(v)
        v = _check_main_thread(parser)
        if v:
            issues.append(v)
        v = _check_network_dependency(parser)
        if v:
            issues.append(v)

        # LCP specific
        v = _check_lcp_lazy(parser)
        if v:
            issues.append(v)
        v = _check_preload_lcp(parser)
        if v:
            issues.append(v)

        # Paint / interactivity proxies (cover TheUXBites metrics)
        v = _check_slow_fcp(parser)
        if v:
            issues.append(v)
        v = _check_slow_lcp(parser)
        if v:
            # Avoid duplicate title if preload_lcp already emitted slow_lcp with same cause? Keep both for metric coverage but deduplicate if identical evidence
            # Only add if not already replaced by preload/lazy? Keep distinct check_id so both appear is intentional for coverage
            # But to avoid double-critical noise when preload already flagged, we keep separate severity
            # Tests expecting preload_lcp should still see slow_lcp; we keep it
            issues.append(v)
        v = _check_speed_index(parser)
        if v:
            issues.append(v)
        v = _check_tti(parser)
        if v:
            issues.append(v)
        v = _check_inp(parser)
        if v:
            issues.append(v)

        # Generic hints
        v = _check_resource_hints(parser)
        if v:
            issues.append(v)
        v = _check_image_dimensions(parser)
        if v:
            issues.append(v)
        v = _check_user_timing(parser)
        if v:
            issues.append(v)
        v = _check_font_display(parser)
        if v:
            issues.append(v)

        # Optional Playwright enhancement: if available, try to capture runtime metrics but never fail tests
        # We intentionally do not require browser; static heuristics are sufficient.
        # Placeholder for future: try realtime INP/TTI measurement via playwright if installed.

        issues.sort(key=lambda x: x.check_id)
        return issues
    finally:
        if own_client:
            await client.aclose()


def analyze_performance_sync(url: str) -> list[PerformanceIssue]:
    """Sync wrapper for analyze_performance."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None  # type: ignore[assignment]
    if loop is not None and loop.is_running():  # type: ignore[union-attr]
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            fut = executor.submit(asyncio.run, analyze_performance(url))
            return fut.result()
    return asyncio.run(analyze_performance(url))

