"""Meta & Semantic HTML detector.

Covers document head and structural semantics per
docs/ux-issue-references/theuxbites_extracted.json:

- Meta tags incomplete (4/6) - missing ogImage, canonical
- Page is missing a canonical URL
- <html> element has an [xml:lang] with same base as [lang]
- All heading elements contain content
- Page has a main content area
- HTML5 landmark elements are used
- Page has a skip link or landmark region (via landmarks)
- Plus Baymard/Markswebb implied: title, description, viewport,
  charset, og/twitter, lang, heading hierarchy, duplicate IDs, etc.

No browser required: httpx + html.parser.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx


@dataclass(frozen=True, slots=True)
class MetaSemanticIssue:
    """Evidence-backed meta/semantic finding."""

    title: str
    description: str
    severity: str  # critical|medium|low
    evidence: dict
    check_id: str


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------
class _MetaSemanticParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.titles: list[str] = []
        self._in_title = False
        self._title_buf = ""
        self.metas: list[dict[str, str]] = []
        self.links: list[dict[str, str]] = []
        self.html_attrs: dict[str, str] = {}
        self._seen_html = False
        self.headings: list[dict] = []  # {level:int, text:str}
        self._in_heading: int | None = None
        self._heading_buf = ""
        self.landmarks: set[str] = set()
        self.has_main: bool = False
        self.ids: list[str] = []
        self.charset_found: bool = False
        self._tag_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        d = {k.lower(): (v or "") for k, v in attrs}
        # html attrs
        if t == "html" and not self._seen_html:
            self._seen_html = True
            self.html_attrs = d
        # title
        if t == "title":
            self._in_title = True
            self._title_buf = ""
        # meta
        if t == "meta":
            self.metas.append(d)
            if "charset" in d and d["charset"].strip():
                self.charset_found = True
            # http-equiv content-type charset
            he = d.get("http-equiv", "").lower()
            if he == "content-type" and "charset" in d.get("content", "").lower():
                self.charset_found = True
        # link
        if t == "link":
            self.links.append(d)
        # headings
        if t in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._in_heading = int(t[1])
            self._heading_buf = ""
        # landmarks
        if t in {"header", "nav", "main", "footer", "aside", "article", "section"}:
            self.landmarks.add(t)
            if t == "main":
                self.has_main = True
        role = d.get("role", "").lower()
        if role in {"banner", "navigation", "main", "contentinfo", "complementary", "region"}:
            self.landmarks.add(f"role:{role}")
            if role == "main":
                self.has_main = True
        # also consider explicit <main> already; role main also counts
        # id duplication
        if "id" in d and d["id"].strip():
            self.ids.append(d["id"].strip())

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t == "title" and self._in_title:
            self._in_title = False
            self.titles.append(self._title_buf)
            self._title_buf = ""
        if t in {"h1", "h2", "h3", "h4", "h5", "h6"} and self._in_heading is not None:
            # close heading
            self.headings.append({"level": self._in_heading, "text": self._heading_buf.strip()})
            self._in_heading = None
            self._heading_buf = ""

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_buf += data
        if self._in_heading is not None:
            self._heading_buf += data


def _parse_html(html: str) -> _MetaSemanticParser:
    p = _MetaSemanticParser()
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


async def _safe_get(client: httpx.AsyncClient, url: str) -> tuple[int | None, str | None, str | None]:
    try:
        resp = await client.get(url)
        try:
            txt = resp.text
        except Exception:
            txt = resp.content.decode("utf-8", errors="replace")
        return resp.status_code, txt, None
    except Exception as exc:
        return None, None, str(exc)


def _is_absolute_url(href: str) -> bool:
    href = href.strip()
    if not href:
        return False
    parsed = urlparse(href)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_valid_lang(lang: str) -> bool:
    # BCP47-ish: 2-3 letter primary, optionally - + alphanum 2-8
    # e.g. en, en-US, zh-Hans, ar-EG
    if not lang or not lang.strip():
        return False
    # simple regex, allow underscore variant
    return bool(re.match(r"^[a-zA-Z]{2,3}(?:-[a-zA-Z0-9]{2,8})*$", lang.strip()))


def _lang_base(lang: str) -> str:
    return lang.strip().split("-")[0].split("_")[0].lower() if lang else ""


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------

def _check_title(parser: _MetaSemanticParser) -> list[MetaSemanticIssue]:
    issues: list[MetaSemanticIssue] = []
    titles = parser.titles
    count = len(titles)
    # evidence base
    # handle duplicate first
    if count == 0:
        issues.append(
            MetaSemanticIssue(
                title="Missing <title>",
                description="Document has no <title> element. Page title is required for SEO, bookmarks, and tab identification.",
                severity="critical",
                evidence={"title_count": count, "titles": titles},
                check_id="title_missing",
            )
        )
        return issues
    if count > 1:
        issues.append(
            MetaSemanticIssue(
                title="Duplicate <title> elements",
                description=f"Found {count} <title> elements; only one should exist.",
                severity="medium",
                evidence={"title_count": count, "titles": [t[:200] for t in titles]},
                check_id="title_duplicate",
            )
        )
    # evaluate primary title (first)
    raw = titles[0] if titles else ""
    text = raw.strip()
    length = len(text)
    evidence = {"title_count": count, "title": text[:300], "title_length": length, "all_titles": [t[:200] for t in titles]}
    if not text:
        issues.append(
            MetaSemanticIssue(
                title="Empty <title>",
                description="Title element is present but empty.",
                severity="critical",
                evidence=evidence,
                check_id="title_empty",
            )
        )
        return issues
    if length < 10:
        issues.append(
            MetaSemanticIssue(
                title="Title too short",
                description=f"Title is {length} characters, shorter than recommended minimum 10.",
                severity="medium",
                evidence=evidence,
                check_id="title_too_short",
            )
        )
    elif length > 60:
        issues.append(
            MetaSemanticIssue(
                title="Title too long",
                description=f"Title is {length} characters, exceeding recommended maximum 60 and likely truncated in SERPs.",
                severity="medium",
                evidence=evidence,
                check_id="title_too_long",
            )
        )
    return issues


def _check_description(parser: _MetaSemanticParser) -> list[MetaSemanticIssue]:
    issues: list[MetaSemanticIssue] = []
    # find meta description (name=description)
    found = None
    for m in parser.metas:
        if m.get("name", "").lower() == "description":
            found = m.get("content", "")
            break
    evidence: dict = {"found": found is not None, "content": (found or "")[:400], "length": len((found or "").strip()) if found is not None else 0}
    # Also check duplicate descriptions count
    desc_count = sum(1 for m in parser.metas if m.get("name", "").lower() == "description")
    evidence["count"] = desc_count
    if found is None:
        issues.append(
            MetaSemanticIssue(
                title="Missing meta description",
                description="Page has no meta description. Description influences SERP snippets and share previews.",
                severity="medium",
                evidence=evidence,
                check_id="meta_description_missing",
            )
        )
        return issues
    text = (found or "").strip()
    if not text:
        issues.append(
            MetaSemanticIssue(
                title="Empty meta description",
                description="Meta description is present but empty.",
                severity="medium",
                evidence=evidence,
                check_id="meta_description_empty",
            )
        )
        return issues
    desc_len = len(text)
    if desc_len < 50:
        issues.append(
            MetaSemanticIssue(
                title="Meta description too short",
                description=f"Meta description is {desc_len} characters, shorter than recommended minimum 50.",
                severity="medium",
                evidence=evidence,
                check_id="meta_description_too_short",
            )
        )
    elif desc_len > 300:
        issues.append(
            MetaSemanticIssue(
                title="Meta description too long",
                description=f"Meta description is {desc_len} characters, exceeding recommended maximum 300 and likely truncated.",
                severity="medium",
                evidence=evidence,
                check_id="meta_description_too_long",
            )
        )
    return issues


def _check_charset(parser: _MetaSemanticParser) -> list[MetaSemanticIssue]:
    if parser.charset_found:
        return []
    return [
        MetaSemanticIssue(
            title="Missing charset declaration",
            description="Document has no <meta charset> declaration. Charset should be declared for correct text rendering.",
            severity="medium",
            evidence={"charset_found": False, "meta_count": len(parser.metas)},
            check_id="meta_charset_missing",
        )
    ]


def _check_viewport(parser: _MetaSemanticParser) -> list[MetaSemanticIssue]:
    has = any(m.get("name", "").lower() == "viewport" and m.get("content", "").strip() for m in parser.metas)
    if has:
        return []
    return [
        MetaSemanticIssue(
            title="Missing viewport meta tag",
            description="Page has no meta viewport tag. Without it, mobile rendering and zoom behavior are unpredictable.",
            severity="critical",
            evidence={"viewport_found": False},
            check_id="meta_viewport_missing",
        )
    ]


def _check_canonical(parser: _MetaSemanticParser, url: str) -> list[MetaSemanticIssue]:
    canonicals = []
    for link in parser.links:
        rel = link.get("rel", "").lower()
        rels = [r.strip() for r in rel.split()]
        if "canonical" in rels:
            href = link.get("href", "").strip()
            canonicals.append(href)
    evidence: dict = {"canonical_count": len(canonicals), "hrefs": canonicals[:3]}
    if not canonicals:
        return [
            MetaSemanticIssue(
                title="Page is missing a canonical URL",
                description="No <link rel=\"canonical\"> was found. Canonical URLs prevent duplicate-content issues.",
                severity="medium",
                evidence=evidence,
                check_id="canonical_missing",
            )
        ]
    # check absolute for first canonical
    href = canonicals[0]
    if not _is_absolute_url(href):
        evidence["href"] = href[:500]
        evidence["is_absolute"] = False
        return [
            MetaSemanticIssue(
                title="Canonical URL is not absolute",
                description=f"Canonical href '{href[:80]}' is not an absolute URL. Canonical should be absolute (https://...).",
                severity="medium",
                evidence=evidence,
                check_id="canonical_not_absolute",
            )
        ]
    evidence["is_absolute"] = True
    evidence["href"] = href[:500]
    return []


def _check_og(parser: _MetaSemanticParser) -> list[MetaSemanticIssue]:
    # required og set per task
    required = ["og:title", "og:description", "og:image", "og:url", "og:type"]
    # at least og:image and og:title required -> but we report missing any
    found: dict[str, bool] = {}
    values: dict[str, str] = {}
    for req in required:
        present = False
        val = ""
        for m in parser.metas:
            prop = m.get("property", "").lower().strip()
            name = m.get("name", "").lower().strip()
            content = m.get("content", "").strip()
            if (prop == req.lower() or name == req.lower()) and content:
                present = True
                val = content[:500]
                break
        found[req] = present
        values[req] = val
    missing = [k for k, v in found.items() if not v]
    present_list = [k for k, v in found.items() if v]
    evidence = {"found": present_list, "missing": missing, "values": values, "checks": found, "found_count": len(present_list), "total": len(required)}
    if not missing:
        return []
    # if missing at least og:image or og:title -> medium, else still medium
    # Provide title similar to GEO: Meta tags incomplete etc., but we want explicit OG issue
    # Check if both critical missing
    critical_missing = [k for k in ["og:title", "og:image"] if k in missing]
    if critical_missing:
        return [
            MetaSemanticIssue(
                title=f"Open Graph tags incomplete ({len(present_list)}/{len(required)})",
                description=f"Open Graph tags incomplete: missing {', '.join(missing)}. At least og:title and og:image are required for rich previews.",
                severity="medium",
                evidence=evidence,
                check_id="og_incomplete",
            )
        ]
    # missing non-critical only (e.g., og:url)
    return [
        MetaSemanticIssue(
            title=f"Open Graph tags incomplete ({len(present_list)}/{len(required)})",
            description=f"Open Graph tags incomplete: missing {', '.join(missing)}.",
            severity="medium",
            evidence=evidence,
            check_id="og_incomplete",
        )
    ]


def _check_twitter(parser: _MetaSemanticParser) -> list[MetaSemanticIssue]:
    has_card = False
    for m in parser.metas:
        if m.get("name", "").lower() == "twitter:card" and m.get("content", "").strip():
            has_card = True
            break
        if m.get("property", "").lower() == "twitter:card" and m.get("content", "").strip():
            has_card = True
            break
    if has_card:
        return []
    return [
        MetaSemanticIssue(
            title="Missing twitter:card",
            description="No twitter:card meta tag found. Twitter/X previews fall back to generic unfurling.",
            severity="low",
            evidence={"twitter_card_found": False},
            check_id="twitter_card_missing",
        )
    ]


def _check_lang(parser: _MetaSemanticParser) -> list[MetaSemanticIssue]:
    issues: list[MetaSemanticIssue] = []
    lang = parser.html_attrs.get("lang", "").strip()
    xml_lang = parser.html_attrs.get("xml:lang", "").strip()
    evidence: dict = {"lang": lang, "xml:lang": xml_lang, "html_attrs": dict(parser.html_attrs)}
    if not lang:
        issues.append(
            MetaSemanticIssue(
                title="Missing html lang attribute",
                description="<html> element has no lang attribute. Language declaration is required for accessibility and i18n.",
                severity="critical",
                evidence=evidence,
                check_id="html_lang_missing",
            )
        )
        return issues
    if not _is_valid_lang(lang):
        issues.append(
            MetaSemanticIssue(
                title="Invalid html lang attribute",
                description=f"lang attribute '{lang}' does not appear to be a valid BCP47 language tag.",
                severity="critical",
                evidence=evidence,
                check_id="html_lang_invalid",
            )
        )
    # check xml:lang same base language
    if xml_lang:
        base_lang = _lang_base(lang)
        base_xml = _lang_base(xml_lang)
        evidence["lang_base"] = base_lang
        evidence["xml_lang_base"] = base_xml
        if base_lang and base_xml and base_lang == base_xml:
            issues.append(
                MetaSemanticIssue(
                    title="<html> element has an [xml:lang] attribute with same base language as [lang]",
                    description=f"Both lang='{lang}' and xml:lang='{xml_lang}' share the same base language ('{base_lang}'). The xml:lang is redundant in HTML.",
                    severity="low",
                    evidence=evidence,
                    check_id="html_xml_lang_redundant",
                )
            )
    return issues


def _check_headings(parser: _MetaSemanticParser) -> list[MetaSemanticIssue]:
    issues: list[MetaSemanticIssue] = []
    headings = parser.headings
    levels = [h["level"] for h in headings]
    texts = [h["text"] for h in headings]
    evidence_base: dict = {"heading_count": len(headings), "levels": levels, "texts": [t[:80] for t in texts]}
    # empty headings
    empty_indices = [i for i, t in enumerate(texts) if not t.strip()]
    if empty_indices:
        issues.append(
            MetaSemanticIssue(
                title="All heading elements contain content",
                description=f"Found {len(empty_indices)} empty heading(s) at positions {empty_indices}. Headings must contain text.",
                severity="medium",
                evidence={**evidence_base, "empty_indices": empty_indices},
                check_id="heading_empty",
            )
        )
    h1_count = levels.count(1)
    if h1_count == 0:
        issues.append(
            MetaSemanticIssue(
                title="Missing h1 heading",
                description="Page has no h1 heading. Exactly one h1 should summarize page purpose.",
                severity="critical",
                evidence={**evidence_base, "h1_count": h1_count},
                check_id="h1_missing",
            )
        )
    elif h1_count > 1:
        issues.append(
            MetaSemanticIssue(
                title="Multiple h1 headings",
                description=f"Page has {h1_count} h1 elements; exactly one is recommended for hierarchy clarity.",
                severity="medium",
                evidence={**evidence_base, "h1_count": h1_count},
                check_id="h1_multiple",
            )
        )
    # h1 first heading
    if headings and h1_count > 0:
        first_level = levels[0] if levels else None
        if first_level != 1:
            issues.append(
                MetaSemanticIssue(
                    title="h1 is not the first heading",
                    description=f"First heading is h{first_level}, not h1. h1 should appear before any h2-h6.",
                    severity="medium",
                    evidence={**evidence_base, "first_level": first_level},
                    check_id="h1_not_first",
                )
            )
    # skipped levels
    skipped: list[tuple[int, int]] = []
    prev = None
    for lvl in levels:
        if prev is not None and lvl > prev + 1:
            skipped.append((prev, lvl))
        prev = lvl
    if skipped:
        issues.append(
            MetaSemanticIssue(
                title="Heading levels skipped",
                description=f"Heading hierarchy skips levels: {skipped}. Avoid jumping from h{skipped[0][0]} to h{skipped[0][1]}; maintain sequential order.",
                severity="medium",
                evidence={**evidence_base, "skipped": skipped},
                check_id="heading_skipped",
            )
        )
    return issues


def _check_landmarks(parser: _MetaSemanticParser) -> list[MetaSemanticIssue]:
    issues: list[MetaSemanticIssue] = []
    tag_landmarks = {t for t in parser.landmarks if not t.startswith("role:")}
    # role mapping for checking presence
    found = set()
    required = {"header", "nav", "main", "footer"}
    found = required.intersection(tag_landmarks)
    role_map = {"role:banner": "header", "role:navigation": "nav", "role:main": "main", "role:contentinfo": "footer"}
    for r, tag in role_map.items():
        if r in parser.landmarks:
            found.add(tag)
    evidence = {
        "found_landmarks": sorted(found),
        "all_landmarks": sorted(tag_landmarks.union({k for k in parser.landmarks if k.startswith("role:")})),
        "required": sorted(required),
        "has_main": parser.has_main,
    }
    if not parser.has_main:
        issues.append(
            MetaSemanticIssue(
                title="Page has a main content area",
                description="No <main> landmark found (including role=main). A main landmark is required for navigation and assistive tech.",
                severity="critical",
                evidence=evidence,
                check_id="main_missing",
            )
        )
        # If main missing, also report landmark insufficiency but main missing is primary
        # add generic landmark issue if missing many
        missing = sorted(required - found)
        if missing:
            issues.append(
                MetaSemanticIssue(
                    title="HTML5 landmark elements are used to improve navigation",
                    description=f"Semantic landmarks insufficient: found {', '.join(sorted(found)) or 'none'}, missing {', '.join(missing)}.",
                    severity="medium",
                    evidence=evidence,
                    check_id="landmarks_incomplete",
                )
            )
        return issues
    # main present, check other landmarks completeness – at least warn if missing 2+
    missing = sorted(required - found)
    if len(found) < 3:
        issues.append(
            MetaSemanticIssue(
                title="HTML5 landmark elements are used to improve navigation",
                description=f"Semantic landmarks insufficient: found {', '.join(sorted(found))}, missing {', '.join(missing)}. Landmarks help understanding page structure.",
                severity="medium",
                evidence=evidence,
                check_id="landmarks_incomplete",
            )
        )
    # Also check skip link or landmark region requirement? But landmarks already cover
    return issues


def _check_duplicate_ids(parser: _MetaSemanticParser) -> list[MetaSemanticIssue]:
    counter = Counter(parser.ids)
    dup = {k: v for k, v in counter.items() if v > 1}
    if not dup:
        return []
    return [
        MetaSemanticIssue(
            title="ARIA IDs are unique across the page",
            description=f"Found duplicate IDs: {', '.join(list(dup.keys())[:5])}. IDs must be unique.",
            severity="medium",
            evidence={"duplicate_ids": dup, "total_ids": len(parser.ids), "unique_ids": len(counter)},
            check_id="duplicate_ids",
        )
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
async def analyze_meta_semantic(url: str, client: httpx.AsyncClient | None = None) -> list[MetaSemanticIssue]:
    """Run meta & semantic checks.

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
            headers={"User-Agent": "ux-analyzer meta-semantic detector"},
        )
    try:
        status, html, error = await _safe_get(client, url)
        if error is not None or html is None or status is None or status >= 400:
            # graceful: if fetch failed, return single informational issue? But spec says handle gracefully.
            # Return empty to avoid false positives on network error, similarly to geo.
            # However if 404 we could return empty.
            return []
        if status != 200:
            return []
        parser = _parse_html(html)
        issues: list[MetaSemanticIssue] = []
        issues.extend(_check_title(parser))
        issues.extend(_check_description(parser))
        issues.extend(_check_charset(parser))
        issues.extend(_check_viewport(parser))
        issues.extend(_check_canonical(parser, url))
        issues.extend(_check_og(parser))
        issues.extend(_check_twitter(parser))
        issues.extend(_check_lang(parser))
        issues.extend(_check_headings(parser))
        issues.extend(_check_landmarks(parser))
        issues.extend(_check_duplicate_ids(parser))

        # Sort for determinism
        issues.sort(key=lambda x: x.check_id)
        return issues
    finally:
        if own_client:
            await client.aclose()


def analyze_meta_semantic_sync(url: str) -> list[MetaSemanticIssue]:
    """Sync wrapper for analyze_meta_semantic."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None  # type: ignore[assignment]
    if loop is not None and loop.is_running():  # type: ignore[union-attr]
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            fut = executor.submit(asyncio.run, analyze_meta_semantic(url))
            return fut.result()
    return asyncio.run(analyze_meta_semantic(url))
