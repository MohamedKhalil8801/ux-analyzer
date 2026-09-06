from __future__ import annotations

import httpx
import pytest

from ux_analyzer.analysis.geo import analyze_geo


def _html_full(
    *,
    title: str = "Test Title",
    description: bool = True,
    canonical: bool = True,
    og_image: bool = True,
    og_title: bool = True,
    viewport: bool = True,
    json_ld: str | None = '{"@context":"https://schema.org","@type":"WebSite","name":"Test"}',
    json_ld_invalid: bool = False,
    landmarks: bool = True,
    body_text: str | None = None,
) -> str:
    """Build minimal HTML covering all GEO meta/structured/semantic needs."""
    if body_text is None:
        body_text = " ".join(["lorem ipsum dolor sit amet consectetur adipiscing elit."] * 20)  # ~1200 chars
    head_parts = []
    if title:
        head_parts.append(f"<title>{title}</title>")
    if description:
        head_parts.append('<meta name="description" content="A great site for testing">')
    if viewport:
        head_parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    if og_title:
        head_parts.append('<meta property="og:title" content="OG Title">')
    if og_image:
        head_parts.append('<meta property="og:image" content="https://example.com/og.png">')
    if canonical:
        head_parts.append('<link rel="canonical" href="https://example.com/">')
    if json_ld is not None:
        if json_ld_invalid:
            head_parts.append('<script type="application/ld+json">{invalid json</script>')
        else:
            head_parts.append(f'<script type="application/ld+json">{json_ld}</script>')

    landmark_html = ""
    if landmarks:
        landmark_html = "<header><nav>nav</nav></header><main><p>main</p></main><footer>footer</footer>"
    else:
        landmark_html = "<div>no landmarks</div>"

    return f"""<!doctype html><html><head>{''.join(head_parts)}</head><body>{landmark_html}<div>{body_text}</div></body></html>"""


def _make_client(route_map: dict[str, tuple[int, str]]) -> httpx.AsyncClient:
    """route_map: url substring -> (status, body). Fallback 404. Most specific match wins."""

    def handler(request: httpx.Request) -> httpx.Response:
        from urllib.parse import urlparse

        str(request.url)
        path = request.url.path or "/"
        # Normalize keys to path for exact matching
        items = []
        for k, v in route_map.items():
            if k.startswith("http"):
                kp = urlparse(k).path or "/"
            else:
                kp = k
            items.append((kp, k, v))
        # Sort by key path length descending (specific first)
        items.sort(key=lambda x: len(x[0]), reverse=True)
        for key_path, orig_key, (status, body) in items:
            if key_path == "/":
                if path != "/" and path != "":
                    continue
            elif path != key_path and not path.endswith(key_path):
                # also handle exact path match; for sitemap alt paths need exact
                # fallback to substring check
                if key_path not in path:
                    continue
            headers = {}
            if orig_key.endswith(".xml") or "<urlset" in body.lower() or "<sitemapindex" in body.lower():
                headers["content-type"] = "application/xml"
            elif orig_key.endswith(".txt"):
                headers["content-type"] = "text/plain"
            else:
                headers["content-type"] = "text/html"
            return httpx.Response(status_code=status, text=body, request=request, headers=headers)
        # default 404
        return httpx.Response(status_code=404, text="Not Found", request=request)

    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport, base_url="https://example.com")


# ---------------------------------------------------------------------------
# All present => no issues
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_all_present_no_issues():
    html = _html_full()
    routes = {
        "https://example.com/": (200, html),
        "/robots.txt": (200, "User-agent: *\nAllow: /\nSitemap: https://example.com/sitemap.xml"),
        "/sitemap.xml": (200, '<?xml version="1.0"?><urlset><url><loc>https://example.com/</loc></url></urlset>'),
        "/llms.txt": (200, "# LLMs\nThis is llms file"),
    }
    client = _make_client(routes)
    issues = await analyze_geo("https://example.com/", client=client)
    await client.aclose()
    assert issues == [], f"expected no issues, got {issues}"


# ---------------------------------------------------------------------------
# robots
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_robots_missing():
    html = _html_full()
    routes = {
        "https://example.com/": (200, html),
        "/robots.txt": (404, "Not Found"),
        "/sitemap.xml": (200, "<urlset><url><loc>https://example.com/</loc></url></urlset>"),
        "/llms.txt": (200, "ok"),
    }
    client = _make_client(routes)
    issues = await analyze_geo("https://example.com/", client=client)
    await client.aclose()
    titles = [i.title for i in issues]
    assert "robots.txt not found" in titles
    assert any(i.check_id == "robots_txt" and i.severity == "low" for i in issues)


@pytest.mark.asyncio
async def test_robots_errors_malformed():
    html = _html_full()
    routes = {
        "https://example.com/": (200, html),
        "/robots.txt": (200, "This is not a robots file"),
        "/sitemap.xml": (200, "<urlset></urlset>"),
        "/llms.txt": (200, "ok"),
    }
    client = _make_client(routes)
    issues = await analyze_geo("https://example.com/", client=client)
    await client.aclose()
    assert any(i.title == "robots.txt file has errors" for i in issues)


@pytest.mark.asyncio
async def test_robots_present_no_issue():
    html = _html_full()
    routes = {
        "https://example.com/": (200, html),
        "/robots.txt": (200, "User-agent: *\nDisallow: /private"),
        "/sitemap.xml": (200, "<urlset></urlset>"),
        "/llms.txt": (200, "ok"),
    }
    client = _make_client(routes)
    issues = await analyze_geo("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "robots_txt" for i in issues)


# ---------------------------------------------------------------------------
# sitemap
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_sitemap_missing():
    html = _html_full()
    routes = {
        "https://example.com/": (200, html),
        "/robots.txt": (200, "User-agent: *\nAllow: /"),
        "/sitemap.xml": (404, "Not Found"),
        "/llms.txt": (200, "ok"),
    }
    client = _make_client(routes)
    issues = await analyze_geo("https://example.com/", client=client)
    await client.aclose()
    assert any(i.title == "No XML sitemap found" and i.severity == "low" for i in issues)


@pytest.mark.asyncio
async def test_sitemap_found_via_robots_directive():
    html = _html_full()
    routes = {
        "https://example.com/": (200, html),
        "/robots.txt": (200, "User-agent: *\nAllow: /\nSitemap: https://example.com/custom-sitemap.xml"),
        "/sitemap.xml": (404, "Not Found"),
        "/custom-sitemap.xml": (200, "<urlset><url><loc>https://example.com/</loc></url></urlset>"),
        "/llms.txt": (200, "ok"),
    }
    client = _make_client(routes)
    issues = await analyze_geo("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "sitemap" for i in issues)


@pytest.mark.asyncio
async def test_sitemap_present_no_issue():
    html = _html_full()
    routes = {
        "https://example.com/": (200, html),
        "/robots.txt": (200, "User-agent: *\nAllow: /"),
        "/sitemap.xml": (200, '<sitemapindex><sitemap><loc>https://example.com/s.xml</loc></sitemap></sitemapindex>'),
        "/llms.txt": (200, "ok"),
    }
    client = _make_client(routes)
    issues = await analyze_geo("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "sitemap" for i in issues)


# ---------------------------------------------------------------------------
# llms.txt
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_llms_missing():
    html = _html_full()
    routes = {
        "https://example.com/": (200, html),
        "/robots.txt": (200, "User-agent: *\nAllow: /"),
        "/sitemap.xml": (200, "<urlset></urlset>"),
        "/llms.txt": (404, "Not Found"),
    }
    client = _make_client(routes)
    issues = await analyze_geo("https://example.com/", client=client)
    await client.aclose()
    assert any(i.title == "No llms.txt file" and i.severity == "low" for i in issues)


@pytest.mark.asyncio
async def test_llms_present():
    html = _html_full()
    routes = {
        "https://example.com/": (200, html),
        "/robots.txt": (200, "User-agent: *\nAllow: /"),
        "/sitemap.xml": (200, "<urlset></urlset>"),
        "/llms.txt": (200, "# LLM guide"),
    }
    client = _make_client(routes)
    issues = await analyze_geo("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "llms_txt" for i in issues)


# ---------------------------------------------------------------------------
# json-ld
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_json_ld_missing():
    html = _html_full(json_ld=None)
    routes = {
        "https://example.com/": (200, html),
        "/robots.txt": (200, "User-agent: *\nAllow: /"),
        "/sitemap.xml": (200, "<urlset></urlset>"),
        "/llms.txt": (200, "ok"),
    }
    client = _make_client(routes)
    issues = await analyze_geo("https://example.com/", client=client)
    await client.aclose()
    assert any(i.title == "No JSON-LD structured data found" for i in issues)


@pytest.mark.asyncio
async def test_json_ld_invalid():
    html = _html_full(json_ld_invalid=True)
    routes = {
        "https://example.com/": (200, html),
        "/robots.txt": (200, "User-agent: *\nAllow: /"),
        "/sitemap.xml": (200, "<urlset></urlset>"),
        "/llms.txt": (200, "ok"),
    }
    client = _make_client(routes)
    issues = await analyze_geo("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "json_ld" and "invalid" in i.title.lower() for i in issues)


@pytest.mark.asyncio
async def test_json_ld_valid_no_issue():
    html = _html_full(json_ld='{"@context":"https://schema.org","@type":"Person","name":"Test"}')
    routes = {
        "https://example.com/": (200, html),
        "/robots.txt": (200, "User-agent: *\nAllow: /"),
        "/sitemap.xml": (200, "<urlset></urlset>"),
        "/llms.txt": (200, "ok"),
    }
    client = _make_client(routes)
    issues = await analyze_geo("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "json_ld" for i in issues)


# ---------------------------------------------------------------------------
# meta tags
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_meta_incomplete():
    """GEO no longer emits meta_tags — meta-semantic owns the per-tag
    findings (canonical, og, description, viewport) individually."""
    # missing canonical and og:image would previously yield 4/6
    html = _html_full(canonical=False, og_image=False)
    routes = {
        "https://example.com/": (200, html),
        "/robots.txt": (200, "User-agent: *\nAllow: /"),
        "/sitemap.xml": (200, "<urlset></urlset>"),
        "/llms.txt": (200, "ok"),
    }
    client = _make_client(routes)
    issues = await analyze_geo("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "meta_tags" for i in issues)


@pytest.mark.asyncio
async def test_meta_complete_no_issue():
    html = _html_full()
    routes = {
        "https://example.com/": (200, html),
        "/robots.txt": (200, "User-agent: *\nAllow: /"),
        "/sitemap.xml": (200, "<urlset></urlset>"),
        "/llms.txt": (200, "ok"),
    }
    client = _make_client(routes)
    issues = await analyze_geo("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "meta_tags" for i in issues)


@pytest.mark.asyncio
async def test_meta_tag_completeness_not_rechecked_by_geo():
    """meta_tags is intentionally gone: meta-semantic covers each tag
    individually with actionable evidence (title, description, viewport,
    canonical, og)."""
    html = _html_full(title="")
    routes = {
        "https://example.com/": (200, html),
        "/robots.txt": (200, "User-agent: *\nAllow: /"),
        "/sitemap.xml": (200, "<urlset></urlset>"),
        "/llms.txt": (200, "ok"),
    }
    client = _make_client(routes)
    issues = await analyze_geo("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "meta_tags" for i in issues)


# ---------------------------------------------------------------------------
# js_content
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_js_content_visible_no_issue():
    html = _html_full(body_text=" ".join(["word"] * 200))  # 1000 chars
    routes = {
        "https://example.com/": (200, html),
        "/robots.txt": (200, "User-agent: *\nAllow: /"),
        "/sitemap.xml": (200, "<urlset></urlset>"),
        "/llms.txt": (200, "ok"),
    }
    client = _make_client(routes)
    issues = await analyze_geo("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "js_content" for i in issues)


@pytest.mark.asyncio
async def test_js_content_missing_flagged():
    _html_full(body_text="")  # tiny
    # also landmarks false to isolate
    # keep minimal html with only script placeholder
    html_min = '<html><head><title>t</title><meta name="description" content="d"><meta name="viewport" content="w"><meta property="og:title" content="t"><meta property="og:image" content="x"><link rel="canonical" href="x"><script type="application/ld+json">{"@context":"https://schema.org"}</script></head><body><header></header><main></main><nav></nav><footer></footer><div id="app"></div></body></html>'
    routes = {
        "https://example.com/": (200, html_min),
        "/robots.txt": (200, "User-agent: *\nAllow: /"),
        "/sitemap.xml": (200, "<urlset></urlset>"),
        "/llms.txt": (200, "ok"),
    }
    client = _make_client(routes)
    issues = await analyze_geo("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "js_content" for i in issues)


# ---------------------------------------------------------------------------
# semantic
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_semantic_pass():
    html = _html_full(landmarks=True)
    routes = {
        "https://example.com/": (200, html),
        "/robots.txt": (200, "User-agent: *\nAllow: /"),
        "/sitemap.xml": (200, "<urlset></urlset>"),
        "/llms.txt": (200, "ok"),
    }
    client = _make_client(routes)
    issues = await analyze_geo("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "semantic_html" for i in issues)


@pytest.mark.asyncio
async def test_semantic_missing():
    html = _html_full(landmarks=False, body_text=" ".join(["word"] * 100))
    routes = {
        "https://example.com/": (200, html),
        "/robots.txt": (200, "User-agent: *\nAllow: /"),
        "/sitemap.xml": (200, "<urlset></urlset>"),
        "/llms.txt": (200, "ok"),
    }
    client = _make_client(routes)
    issues = await analyze_geo("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "semantic_html" for i in issues)


# ---------------------------------------------------------------------------
# network errors graceful
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_network_error_graceful():
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/robots.txt" in url:
            raise httpx.ConnectError("mock connect error", request=request)
        if "example.com/" in url and url.rstrip("/") == "https://example.com":
            html = _html_full()
            return httpx.Response(200, text=html, request=request)
        if "/sitemap.xml" in url:
            return httpx.Response(200, text="<urlset></urlset>", request=request)
        if "/llms.txt" in url:
            return httpx.Response(200, text="ok", request=request)
        return httpx.Response(404, text="Not found", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://example.com")
    issues = await analyze_geo("https://example.com/", client=client)
    await client.aclose()
    # should not raise, robots check skipped gracefully
    assert not any(i.check_id == "robots_txt" for i in issues)
    # other checks still work
    assert not any(i.check_id == "sitemap" for i in issues)


@pytest.mark.asyncio
async def test_import_without_side_effects():
    import importlib

    mod = importlib.import_module("ux_analyzer.analysis.geo")
    assert hasattr(mod, "analyze_geo")
    assert hasattr(mod, "analyze_geo_sync")
    assert hasattr(mod, "GeoIssue")


# ---------------------------------------------------------------------------
# portfolio mock reproducing live site: 4 GEO issues, 2 passes
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_portfolio_mock_finds_exactly_4_geo_issues():
    # Replicate https://mohamed-khalil.vercel.app head snippet observed live
    portfolio_head = """
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover" />
    <title>Mohamed Khalil - Mobile, Web & UI/UX</title>
    <meta name="description" content="Mohamed Khalil, sole developer of Muslim Pedia..." />
    <meta name="author" content="Mohamed Khalil" />
    <meta property="og:type" content="website" />
    <meta property="og:title" content="Mohamed Khalil - Mobile, Web & UI/UX" />
    <meta property="og:description" content="Sole developer of Muslim Pedia..." />
    """
    # note: missing canonical and og:image intentionally; the meta-semantic
    # detector owns those findings, GEO no longer re-flags them.
    body = "<header><nav>nav</nav></header><main>" + " ".join(["portfolio content word"] * 300) + "</main><footer>footer</footer>"
    html = f"<!doctype html><html><head>{portfolio_head}</head><body>{body}</body></html>"
    routes = {
        "https://example.com/": (200, html),
        "/robots.txt": (404, "Not Found"),
        "/sitemap.xml": (404, "Not Found"),
        "/llms.txt": (404, "Not Found"),
    }
    client = _make_client(routes)
    issues = await analyze_geo("https://example.com/", client=client)
    await client.aclose()
    titles = sorted([i.title for i in issues])
    # Expect exactly 4 GEO issues; meta_tags was removed as a duplicate
    assert "robots.txt not found" in titles
    assert "No JSON-LD structured data found" in titles
    assert "No XML sitemap found" in titles
    assert "No llms.txt file" in titles
    assert len(issues) == 4, f"expected 4 issues, got {len(issues)}: {titles}"
    # Ensure passing checks NOT flagged
    assert not any(i.check_id == "js_content" for i in issues)
    assert not any(i.check_id == "semantic_html" for i in issues)
    assert not any(i.check_id == "meta_tags" for i in issues)
    # severity recalibration: file-level 404s are low, not critical
    assert all(
        i.severity == "low"
        for i in issues
        if i.check_id in ("robots_txt", "sitemap", "llms_txt", "json_ld")
    )
