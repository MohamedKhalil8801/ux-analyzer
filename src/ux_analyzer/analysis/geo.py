"""GEO (AI visibility) detector.

Covers checks seen in live audit via
docs/ux-issue-references/theuxbites_extracted.json:

- robots.txt not found / has errors
- Content is visible without JavaScript
- No JSON-LD structured data found / Structured data is valid
- No XML sitemap found
- No llms.txt file

Meta-tag completeness is intentionally NOT re-checked here: the
meta-semantic detector already covers title, description, viewport,
canonical, and Open Graph individually with actionable per-check
evidence.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx


@dataclass(frozen=True, slots=True)
class GeoIssue:
    """Evidence-backed GEO finding."""

    title: str
    description: str
    severity: str  # critical|high|medium|low
    evidence: dict
    check_id: str


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------
class _GeoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: str = ""
        self._in_title = False
        self._in_script_ld = False
        self._in_script = False
        self._in_style = False
        self._ld_buffer: str = ""
        self.ld_scripts: list[str] = []
        self.metas: list[dict[str, str]] = []
        self.links: list[dict[str, str]] = []
        self.landmarks: set[str] = set()
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        d = {k.lower(): (v or "") for k, v in attrs}
        if t in {"header", "main", "nav", "footer", "aside", "article", "section"}:
            self.landmarks.add(t)
        # ARIA landmark roles also count
        role = d.get("role", "").lower()
        if role in {"banner", "navigation", "main", "contentinfo", "complementary", "region"}:
            self.landmarks.add(f"role:{role}")
        if t == "title":
            self._in_title = True
        elif t == "meta":
            self.metas.append(d)
        elif t == "link":
            self.links.append(d)
        elif t == "script":
            typ = d.get("type", "").lower().strip()
            if typ == "application/ld+json":
                self._in_script_ld = True
                self._ld_buffer = ""
            else:
                self._in_script = True
        elif t == "style":
            self._in_style = True

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t == "title":
            self._in_title = False
        elif t == "script":
            if self._in_script_ld:
                self.ld_scripts.append(self._ld_buffer)
                self._in_script_ld = False
                self._ld_buffer = ""
            elif self._in_script:
                self._in_script = False
        elif t == "style" and self._in_style:
            self._in_style = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        if self._in_script_ld:
            self._ld_buffer += data
            return
        if self._in_script or self._in_style:
            return
        # visible text, strip later
        self._text_parts.append(data)

    def visible_text(self) -> str:
        # Join and normalize whitespace
        raw = " ".join(self._text_parts)
        return " ".join(raw.split())

    def landmark_tags(self) -> set[str]:
        # Return only tag landmarks for the 4 core checks, but keep role info
        return self.landmarks


def _parse_html(html: str) -> _GeoHTMLParser:
    p = _GeoHTMLParser()
    try:
        p.feed(html)
    except Exception:
        # html.parser is lenient, but guard anyway
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
        # url may be without scheme like "example.com"
        # fallback: treat entire url as netloc
        parsed2 = urlparse(f"https://{url}")
        scheme = parsed2.scheme
        netloc = parsed2.netloc
    return f"{scheme}://{netloc}".rstrip("/")


async def _safe_get(
    client: httpx.AsyncClient, url: str
) -> tuple[int | None, str | None, str | None]:
    """Return (status_code, text, error_str)."""
    try:
        resp = await client.get(url)
        # Return body as text even for 404 pages
        try:
            txt = resp.text
        except Exception:
            txt = resp.content.decode("utf-8", errors="replace")
        return resp.status_code, txt, None
    except Exception as exc:  # httpx.RequestError, Timeout, etc.
        return None, None, str(exc)


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------
def _check_robots(status: int | None, body: str | None, error: str | None, url: str) -> GeoIssue | None:
    evidence: dict = {"url": url, "status_code": status}
    if error is not None:
        evidence["error"] = error
        evidence["status"] = "unavailable"
        return None  # graceful: do not flag when network unavailable
    evidence["status_code"] = status
    if status is None or status == 404 or (status is not None and status >= 400):
        evidence["found"] = False
        return GeoIssue(
            title="robots.txt not found",
            description="robots.txt was not found at /robots.txt. AI crawlers and search engines rely on it to discover allowed paths.",
            severity="low",
            evidence=evidence,
            check_id="robots_txt",
        )
    # status 200
    body_str = body or ""
    evidence["body_preview"] = body_str[:1000]
    stripped = body_str.strip()
    if not stripped or len(stripped) < 10:
        evidence["reason"] = "empty or too short"
        return GeoIssue(
            title="robots.txt file has errors",
            description="robots.txt is present but empty or too short to be valid.",
            severity="low",
            evidence=evidence,
            check_id="robots_txt",
        )
    has_ua = bool(re.search(r"(?im)^\s*User-agent\s*:", body_str))
    has_directive = bool(re.search(r"(?im)^\s*(User-agent|Disallow|Allow|Sitemap)\s*:", body_str))
    if not has_ua or not has_directive:
        evidence["reason"] = "missing User-agent or directive"
        return GeoIssue(
            title="robots.txt file has errors",
            description="robots.txt is present but missing a valid User-agent directive or appears malformed.",
            severity="low",
            evidence=evidence,
            check_id="robots_txt",
        )
    evidence["found"] = True
    return None


def _check_sitemap(
    status: int | None,
    body: str | None,
    error: str | None,
    url: str,
    robots_body: str | None,
    robots_status: int | None,
    alt_fetch: list[dict] | None = None,
) -> GeoIssue | None:
    evidence: dict = {"url": url, "status_code": status}
    robots_sitemaps: list[str] = []
    if robots_status == 200 and robots_body:
        for m in re.finditer(r"(?im)^\s*Sitemap:\s*(\S+)", robots_body):
            robots_sitemaps.append(m.group(1).strip())
    if robots_sitemaps:
        evidence["robots_sitemaps"] = robots_sitemaps
    if error is not None:
        evidence["error"] = error
        # if we have alternative sitemaps we already tried elsewhere, but here we just grace
        # Do not flag if network unavailable and no sitemap info
        # However if alt_fetch supplied, include
        if alt_fetch:
            evidence["alt_checks"] = alt_fetch
        # graceful: no false positive
        return None
    # If alt checks already succeeded, caller will have returned None
    # Check primary sitemap
    if status == 200 and body is not None:
        low = body.lower()
        has_xml = "<urlset" in low or "<sitemapindex" in low
        # Some sitemaps are plain text url list, consider found if 200 and non-empty and xml-like or url-like
        if has_xml or (len(body.strip()) > 50 and ("<url" in low or "http" in low)):
            evidence["found"] = True
            return None
        # 200 but empty/invalid still treat as not found? Be lenient: if 200 and body not empty, treat as found to avoid false positive
        if len(body.strip()) > 50:
            evidence["found"] = True
            return None
    # Check fallback robots sitemaps if already fetched via alt_fetch
    if alt_fetch:
        evidence["alt_checks"] = alt_fetch
        for alt in alt_fetch:
            if alt.get("status_code") == 200 and alt.get("body"):
                b = (alt.get("body") or "").lower()
                if "<urlset" in b or "<sitemapindex" in b or len((alt.get("body") or "").strip()) > 50:
                    evidence["found_via_robots"] = alt.get("url")
                    return None
    # Also if robots_sitemaps exist but we haven't fetched yet, caller will fetch.
    # Here we are fallback: if robots_sitemaps present and primary failed, we should not yet flag; caller handles alt fetch.
    # But if we reach here without alt_fetch, include sitemaps and flag only if no sitemap at all
    if status is None:
        return None
    # status is 404 or >=400 and no alt success => flag
    if status == 404 or (status is not None and status >= 400):
        evidence["found"] = False
        return GeoIssue(
            title="No XML sitemap found",
            description="No XML sitemap was found at /sitemap.xml and no Sitemap directive in robots.txt points to a valid sitemap. AI crawlers rely on sitemaps to discover pages.",
            severity="low",
            evidence=evidence,
            check_id="sitemap",
        )
    # fallback generic
    evidence["found"] = False
    return GeoIssue(
        title="No XML sitemap found",
        description="No XML sitemap was found at /sitemap.xml and no Sitemap directive in robots.txt points to a valid sitemap.",
        severity="low",
        evidence=evidence,
        check_id="sitemap",
    )


def _check_llms(status: int | None, body: str | None, error: str | None, url: str) -> GeoIssue | None:
    evidence: dict = {"url": url, "status_code": status}
    if error is not None:
        evidence["error"] = error
        return None
    if status == 200 and body is not None and len(body.strip()) > 10:
        evidence["found"] = True
        return None
    if status == 404 or (status is not None and status >= 400) or (body is not None and len(body.strip()) == 0):
        evidence["found"] = False
        return GeoIssue(
            title="No llms.txt file",
            description="No llms.txt file was found at /llms.txt. This emerging standard helps AI systems understand and cite your site.",
            severity="low",
            evidence=evidence,
            check_id="llms_txt",
        )
    evidence["found"] = False
    return GeoIssue(
        title="No llms.txt file",
        description="No llms.txt file was found at /llms.txt.",
        severity="low",
        evidence=evidence,
        check_id="llms_txt",
    )


def _check_json_ld(parser: _GeoHTMLParser, html: str) -> GeoIssue | None:
    evidence: dict = {"found_count": len(parser.ld_scripts)}
    if not parser.ld_scripts:
        evidence["found"] = False
        return GeoIssue(
            title="No JSON-LD structured data found",
            description="No JSON-LD structured data was found in the page. AI visibility depends on structured data to understand entities and relationships.",
            severity="low",
            evidence=evidence,
            check_id="json_ld",
        )
    # validate each block
    invalid = []
    valid = 0
    for idx, block in enumerate(parser.ld_scripts):
        b = block.strip()
        if not b:
            invalid.append(idx)
            continue
        try:
            data = json.loads(b)
            # must be dict or list
            if isinstance(data, (dict, list)):
                valid += 1
            else:
                invalid.append(idx)
        except Exception as exc:
            invalid.append(idx)
            evidence[f"error_{idx}"] = str(exc)[:500]
    evidence["valid_count"] = valid
    evidence["invalid_indices"] = invalid
    if invalid:
        return GeoIssue(
            title="Structured data is invalid",
            description=f"Found {len(parser.ld_scripts)} JSON-LD block(s) but {len(invalid)} failed to parse as valid JSON.",
            severity="low",
            evidence=evidence,
            check_id="json_ld",
        )
    evidence["found"] = True
    return None


def _check_js_content(parser: _GeoHTMLParser) -> GeoIssue | None:
    text = parser.visible_text()
    length = len(text)
    # Also count words
    words = len(text.split()) if text else 0
    evidence = {"text_length": length, "word_count": words, "threshold": 300, "preview": text[:500]}
    # Portfolio has 5854 chars, many words => pass
    # Heuristic: if raw HTML visible text is >=300 chars, content is SSR-visible
    if length >= 300 and words >= 20:
        return None
    # If html has very little text, flag
    return GeoIssue(
        title="Content not visible without JavaScript",
        description="Primary content is not visible in raw HTML; it may require JavaScript to render, limiting AI crawler visibility.",
        severity="critical",
        evidence=evidence,
        check_id="js_content",
    )


def _check_semantic(parser: _GeoHTMLParser) -> GeoIssue | None:
    # core landmarks
    tag_landmarks = {t for t in parser.landmarks if not t.startswith("role:")}
    required = {"header", "main", "nav", "footer"}
    found = required.intersection(tag_landmarks)
    # also consider ARIA roles mapping: role:banner -> header, role:navigation -> nav, etc.
    role_map = {"role:banner": "header", "role:navigation": "nav", "role:main": "main", "role:contentinfo": "footer"}
    for r, tag in role_map.items():
        if r in parser.landmarks:
            found.add(tag)
    evidence = {
        "found_landmarks": sorted(found),
        "all_landmarks": sorted(tag_landmarks.union({k for k in parser.landmarks if k.startswith("role:")})),
        "required": sorted(required),
    }
    # Need at least 3 of 4 to be considered good (portfolio has 4)
    if len(found) >= 3:
        return None
    missing = sorted(required - found)
    return GeoIssue(
        title="Poor semantic HTML structure",
        description=f"Semantic landmarks are insufficient: found {', '.join(sorted(found)) or 'none'}, missing {', '.join(missing)}. Landmarks help AI understand page structure.",
        severity="medium",
        evidence=evidence,
        check_id="semantic_html",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
async def analyze_geo(url: str, client: httpx.AsyncClient | None = None) -> list[GeoIssue]:
    """Run 7 GEO checks deterministically.

    Args:
        url: Page URL to analyze (e.g. https://example.com/).
        client: Optional httpx.AsyncClient for testing / reuse.
    """
    if not url or not url.strip():
        raise ValueError("url must not be empty")
    url = url.strip()
    # Ensure scheme present
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url

    origin = _origin(url)

    # Prepare client handling
    own_client = False
    if client is None:
        own_client = True
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0),
            follow_redirects=True,
            headers={"User-Agent": "ux-analyzer GEO detector"},
        )

    try:
        # Fetch main page and auxiliary files concurrently
        main_task = asyncio.create_task(_safe_get(client, url))
        robots_url = f"{origin}/robots.txt"
        sitemap_url = f"{origin}/sitemap.xml"
        llms_url = f"{origin}/llms.txt"
        robots_task = asyncio.create_task(_safe_get(client, robots_url))
        sitemap_task = asyncio.create_task(_safe_get(client, sitemap_url))
        llms_task = asyncio.create_task(_safe_get(client, llms_url))

        main_status, main_html, main_error = await main_task
        robots_status, robots_body, robots_error = await robots_task
        sitemap_status, sitemap_body, sitemap_error = await sitemap_task
        llms_status, llms_body, llms_error = await llms_task

        issues: list[GeoIssue] = []

        # robots check
        ri = _check_robots(robots_status, robots_body, robots_error, robots_url)
        if ri:
            issues.append(ri)

        # sitemap: need to handle robots Sitemap directive fallback
        # If primary sitemap missing but robots advertises alternative, fetch it
        alt_fetch: list[dict] = []
        # Only fetch alt if primary failed and we have robots sitemaps
        robots_sitemap_urls: list[str] = []
        if robots_status == 200 and robots_body:
            for m in re.finditer(r"(?im)^\s*Sitemap:\s*(\S+)", robots_body):
                robots_sitemap_urls.append(m.group(1).strip())
        # If primary 404 and we have alts, try them
        need_alt = (sitemap_status is None or sitemap_status >= 400) and robots_sitemap_urls
        if need_alt and sitemap_error is None:  # only if we could fetch robots
            for alt_url in robots_sitemap_urls:
                a_status, a_body, a_err = await _safe_get(client, alt_url)
                alt_fetch.append({"url": alt_url, "status_code": a_status, "body": a_body[:1000] if a_body else None, "error": a_err})
                # early exit if found
                if a_status == 200 and a_body and ("<urlset" in a_body.lower() or "<sitemapindex" in a_body.lower() or len(a_body.strip()) > 50):
                    break

        si = _check_sitemap(sitemap_status, sitemap_body, sitemap_error, sitemap_url, robots_body, robots_status, alt_fetch if robots_sitemap_urls else None)
        if si:
            issues.append(si)

        # llms
        li = _check_llms(llms_status, llms_body, llms_error, llms_url)
        if li:
            issues.append(li)

        # HTML-dependent checks: only if we have HTML
        if main_error is None and main_status == 200 and main_html is not None:
            parser = _parse_html(main_html)

            # json-ld
            ji = _check_json_ld(parser, main_html)
            if ji:
                issues.append(ji)

            # js content
            jsi = _check_js_content(parser)
            if jsi:
                issues.append(jsi)

            # semantic
            semi = _check_semantic(parser)
            if semi:
                issues.append(semi)
        else:
            # main fetch unavailable -> graceful, no HTML checks, but don't crash
            # optionally add evidence-unavailable note? We keep silent to avoid false positives
            pass

        # Sort for determinism by check_id
        issues.sort(key=lambda x: x.check_id)
        return issues
    finally:
        if own_client:
            await client.aclose()


def analyze_geo_sync(url: str) -> list[GeoIssue]:
    """Sync wrapper for analyze_geo."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None  # type: ignore[assignment]
    if loop is not None and loop.is_running():  # type: ignore[union-attr]
        # Called from within running loop: run in new thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            fut = executor.submit(asyncio.run, analyze_geo(url))
            return fut.result()
    return asyncio.run(analyze_geo(url))
