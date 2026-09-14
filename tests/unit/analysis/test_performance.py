from __future__ import annotations

import httpx
import pytest

from ux_analyzer.analysis.performance import analyze_performance

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_client(route_map: dict[str, tuple[int, str]]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        from urllib.parse import urlparse

        str(request.url)
        path = request.url.path or "/"
        items = []
        for k, v in route_map.items():
            if k.startswith("http"):
                kp = urlparse(k).path or "/"
            else:
                kp = k
            items.append((kp, k, v))
        items.sort(key=lambda x: len(x[0]), reverse=True)
        for key_path, orig_key, (status, body) in items:
            if key_path == "/":
                if path not in ("/", ""):
                    continue
            elif path != key_path and not path.endswith(key_path):
                if key_path not in path:
                    continue
            headers = {"content-type": "text/html"}
            return httpx.Response(
                status_code=status, text=body, request=request, headers=headers
            )
        return httpx.Response(status_code=404, text="Not Found", request=request)

    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport, base_url="https://example.com")


def _perfect_html() -> str:
    """HTML that should produce zero performance issues (all hints present)."""
    return """<!doctype html><html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Test</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Test&display=swap" rel="stylesheet" media="print" onload="this.media='all'">
<script async src="app.js"></script>
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" fetchpriority="high" alt="hero">
<p>Content</p>
<script>performance.mark('test');</script>
</body></html>"""


# ---------------------------------------------------------------------------
# valid: no issues when all hints present
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_perfect_no_issues():
    html = _perfect_html()
    routes = {"https://example.com/": (200, html)}
    client = _make_client(routes)
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert issues == [], (
        f"expected no issues, got {[(i.check_id, i.title) for i in issues]}"
    )


@pytest.mark.asyncio
async def test_perfect_no_render_blocking():
    html = _perfect_html()
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "render_blocking" for i in issues)


# ---------------------------------------------------------------------------
# render-blocking
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_render_blocking_single_stylesheet():
    html = """<!doctype html><html><head>
<link rel="stylesheet" href="a.css">
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" fetchpriority="high" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "render_blocking" for i in issues)
    rb = [i for i in issues if i.check_id == "render_blocking"][0]
    assert rb.severity in ("medium", "high", "critical")
    assert "slow page display" in rb.title.lower()


@pytest.mark.asyncio
async def test_render_blocking_high_over_three():
    html = """<!doctype html><html><head>
<link rel="stylesheet" href="a.css">
<link rel="stylesheet" href="b.css">
<link rel="stylesheet" href="c.css">
<link rel="stylesheet" href="d.css">
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" fetchpriority="high" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    rb = [i for i in issues if i.check_id == "render_blocking"][0]
    # Render-blocking resources are opportunities, not catastrophes: cap at
    # high even with many blockers; small counts stay medium.
    assert rb.severity == "high"
    assert rb.evidence["total_blocking"] == 4
    assert "type=module" in rb.description


@pytest.mark.asyncio
async def test_render_blocking_medium_for_small_counts():
    html = """<!doctype html><html><head>
<link rel="stylesheet" href="a.css">
<link rel="stylesheet" href="b.css">
<link rel="stylesheet" href="c.css">
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" fetchpriority="high" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    rb = [i for i in issues if i.check_id == "render_blocking"][0]
    assert rb.severity == "medium"
    assert rb.evidence["total_blocking"] == 3


@pytest.mark.asyncio
async def test_render_blocking_media_print_not_counted():
    html = """<!doctype html><html><head>
<link rel="stylesheet" href="a.css" media="print">
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" fetchpriority="high" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "render_blocking" for i in issues)


@pytest.mark.asyncio
async def test_render_blocking_script_in_head():
    html = """<!doctype html><html><head>
<script src="app.js"></script>
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" fetchpriority="high" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert any(
        i.check_id == "render_blocking" and i.evidence["blocking_scripts"] >= 1
        for i in issues
    )


@pytest.mark.asyncio
async def test_render_blocking_async_not_counted():
    html = """<!doctype html><html><head>
<script async src="app.js"></script>
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" fetchpriority="high" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "render_blocking" for i in issues)


@pytest.mark.asyncio
async def test_render_blocking_module_not_counted():
    # <script type="module"> is deferred by spec, should NOT count as blocking
    html = """<!doctype html><html><head>
<script type="module" src="./assets/index-B4yTg1iF.js"></script>
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" fetchpriority="high" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "render_blocking" for i in issues), (
        "module script should be deferred, not blocking"
    )


@pytest.mark.asyncio
async def test_render_blocking_module_case_insensitive():
    html = """<!doctype html><html><head>
<script TYPE="MODULE" src="./assets/app.js"></script>
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" fetchpriority="high" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "render_blocking" for i in issues)


@pytest.mark.asyncio
async def test_render_blocking_inline_module_not_counted():
    html = """<!doctype html><html><head>
<script type="module">console.log('inline module');</script>
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" fetchpriority="high" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "render_blocking" for i in issues)


@pytest.mark.asyncio
async def test_render_blocking_import():
    html = """<!doctype html><html><head>
<style>@import url('a.css'); body{}</style>
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" fetchpriority="high" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert any(
        i.check_id == "render_blocking" and i.evidence["import_count"] == 1
        for i in issues
    )


# ---------------------------------------------------------------------------
# unused JS
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_unused_js_many_files():
    html = """<!doctype html><html><head>
<script async src="a.js"></script>
<script async src="b.js"></script>
<script async src="c.js"></script>
<script async src="d.js"></script>
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" fetchpriority="high" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "unused_js" and i.severity == "medium" for i in issues)
    uj = [i for i in issues if i.check_id == "unused_js"][0]
    assert uj.evidence["total_external_js"] == 4


@pytest.mark.asyncio
async def test_unused_js_blocking_threshold():
    html = """<!doctype html><html><head>
<script src="a.js"></script>
<script src="b.js"></script>
<script src="c.js"></script>
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" fetchpriority="high" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "unused_js" for i in issues)


@pytest.mark.asyncio
async def test_unused_js_not_flagged_when_few():
    html = _perfect_html()
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "unused_js" for i in issues)


# ---------------------------------------------------------------------------
# main thread
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_main_thread_large_inline():
    large = "x" * 2000
    html = f"""<!doctype html><html><head>
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {{font-family:test;src:url(test.woff2);font-display:swap;}}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" fetchpriority="high" alt="hero"><script>performance.mark('x');</script>
<script>{large}</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "main_thread" for i in issues)
    mt = [i for i in issues if i.check_id == "main_thread"][0]
    assert "Main thread" in mt.title


@pytest.mark.asyncio
async def test_main_thread_blocking_scripts():
    html = """<!doctype html><html><head>
<script src="a.js"></script>
<script src="b.js"></script>
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" fetchpriority="high" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "main_thread" for i in issues)


@pytest.mark.asyncio
async def test_main_thread_not_flagged_when_optimized():
    html = _perfect_html()
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "main_thread" for i in issues)
    assert not any(i.check_id == "inp" for i in issues)
    assert not any(i.check_id == "tti" for i in issues)


# ---------------------------------------------------------------------------
# network dependency
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_network_dependency_depth():
    html = """<!doctype html><html><head>
<link rel="stylesheet" href="a.css">
<link rel="stylesheet" href="b.css">
<script src="a.js"></script>
<script src="b.js"></script>
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" fetchpriority="high" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "network_dependency" for i in issues)
    nd = [i for i in issues if i.check_id == "network_dependency"][0]
    assert nd.evidence["depth"] == 4
    assert nd.severity == "medium"


@pytest.mark.asyncio
async def test_network_dependency_not_flagged_shallow():
    html = """<!doctype html><html><head>
<link rel="stylesheet" href="a.css">
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" fetchpriority="high" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "network_dependency" for i in issues)


# ---------------------------------------------------------------------------
# LCP lazy
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_lcp_lazy_flagged():
    html = """<!doctype html><html><head>
<link rel="preload" as="image" href="other.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" loading="lazy" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "lcp_lazy" and i.severity == "critical" for i in issues)


@pytest.mark.asyncio
async def test_lcp_lazy_not_flagged_eager():
    html = """<!doctype html><html><head>
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" loading="eager" fetchpriority="high" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "lcp_lazy" for i in issues)


@pytest.mark.asyncio
async def test_lcp_largest_selection():
    # Two images, largest should be chosen as LCP (second is larger)
    html = """<!doctype html><html><head>
<link rel="preload" as="image" href="large.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="small.jpg" width="100" height="100" loading="eager" alt="small">
<img src="large.jpg" width="1200" height="800" loading="lazy" alt="large"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    # LCP is large.jpg which is lazy -> should flag
    assert any(
        i.check_id == "lcp_lazy" and "large.jpg" in i.evidence["lcp_src"]
        for i in issues
    )


# ---------------------------------------------------------------------------
# preload LCP
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_preload_lcp_missing():
    html = """<!doctype html><html><head>
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "preload_lcp" and i.severity == "critical" for i in issues)
    assert any(
        "Preload Largest" in i.title for i in issues if i.check_id == "preload_lcp"
    )


@pytest.mark.asyncio
async def test_preload_lcp_with_fetchpriority():
    html = """<!doctype html><html><head>
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" fetchpriority="high" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "preload_lcp" for i in issues)


@pytest.mark.asyncio
async def test_preload_lcp_with_link():
    html = """<!doctype html><html><head>
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "preload_lcp" for i in issues)


@pytest.mark.asyncio
async def test_preload_lcp_no_image_no_flag():
    html = """<!doctype html><html><head>
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<p>No images here</p><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id in ("preload_lcp", "lcp_lazy") for i in issues)


@pytest.mark.asyncio
async def test_preload_duplicate_not_double():
    # Same raster image missing preload should only produce preload_lcp, not duplicate slow_lcp
    html = """<!doctype html><html><head>
<link rel="stylesheet" href="a.css">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "preload_lcp" for i in issues)
    assert not any(i.check_id == "slow_lcp" for i in issues), (
        "slow_lcp should be suppressed when preload_lcp already flags same src (deduplication)"
    )


@pytest.mark.asyncio
async def test_preload_lcp_svg_never_a_candidate():
    """Vector graphics are rarely the LCP element; suggesting a preload for a
    below-fold SVG wastes bandwidth (live-verified false positive)."""
    html = """<!doctype html><html><head>
<link rel="stylesheet" href="a.css">
</head><body>
<img src="./digitalkhatt-bismillah.svg" width="1200" height="800" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert not any(
        i.check_id in ("preload_lcp", "lcp_lazy", "slow_lcp") for i in issues
    )


@pytest.mark.asyncio
async def test_slow_lcp_suppressed_when_preload_missing_but_blocking_high():
    html = """<!doctype html><html><head>
<link rel="stylesheet" href="a.css">
<link rel="stylesheet" href="b.css">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    # preload missing should not yield both preload_lcp and slow_lcp
    assert any(i.check_id == "preload_lcp" for i in issues)
    assert not any(i.check_id == "slow_lcp" for i in issues)


# ---------------------------------------------------------------------------
# resource hints / image dimensions / font-display / user timing
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resource_hints_missing():
    html = """<!doctype html><html><head>
<link href="https://fonts.googleapis.com/css2?family=Test&display=swap" rel="stylesheet" media="print">
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" fetchpriority="high" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "resource_hints" and i.severity == "low" for i in issues)


@pytest.mark.asyncio
async def test_resource_hints_present_no_flag():
    html = _perfect_html()
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "resource_hints" for i in issues)


@pytest.mark.asyncio
async def test_image_dimensions_missing():
    html = """<!doctype html><html><head>
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "image_dimensions" for i in issues)


@pytest.mark.asyncio
async def test_image_dimensions_present_no_flag():
    html = _perfect_html()
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "image_dimensions" for i in issues)


@pytest.mark.asyncio
async def test_user_timing_missing():
    html = """<!doctype html><html><head>
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" fetchpriority="high" alt="hero">
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "user_timing" and i.severity == "low" for i in issues)


@pytest.mark.asyncio
async def test_user_timing_present_no_flag():
    html = """<!doctype html><html><head>
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" fetchpriority="high" alt="hero"><script>performance.mark('hero'); performance.measure('m');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "user_timing" for i in issues)


@pytest.mark.asyncio
async def test_font_display_missing():
    html = """<!doctype html><html><head>
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" fetchpriority="high" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "font_display" for i in issues)


@pytest.mark.asyncio
async def test_font_display_present_no_flag():
    html = _perfect_html()
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "font_display" for i in issues)


# ---------------------------------------------------------------------------
# paint / interactivity proxies
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_slow_fcp_and_speed_index_removed_as_duplicates():
    """slow_fcp / speed_index were pure blocking-count proxies duplicating
    render_blocking here and the measured PSI metrics in the report; they
    must no longer be emitted."""
    html = """<!doctype html><html><head>
<link rel="stylesheet" href="a.css">
<link rel="stylesheet" href="b.css">
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="a.jpg" width="800" height="600" alt="a"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id in ("slow_fcp", "speed_index") for i in issues)


@pytest.mark.asyncio
async def test_tti_flagged():
    html = """<!doctype html><html><head>
<script src="a.js"></script>
<script src="b.js"></script>
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" fetchpriority="high" alt="hero"><script>performance.mark('x');</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "tti" for i in issues)


@pytest.mark.asyncio
async def test_inp_flagged():
    large = "y" * 1500
    html = f"""<!doctype html><html><head>
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {{font-family:test;src:url(test.woff2);font-display:swap;}}</style>
</head><body>
<img src="hero.jpg" width="1200" height="800" fetchpriority="high" alt="hero"><script>performance.mark('x');</script>
<script>{large}</script>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "inp" for i in issues)
    assert any("INP" in i.title for i in issues if i.check_id == "inp")


# ---------------------------------------------------------------------------
# portfolio mock (similar to geo live-style mock)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_portfolio_mock_flags_expected():
    # Minimal replica of https://mohamed-khalil.vercel.app head with 2 stylesheets + module script + inline theme script
    # After fix: module is deferred (type="module" => not blocking), so true blocking is 2 CSS + 1 inline = 3, not 4
    html = """<!doctype html><html><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mohamed Khalil</title>
<meta name="description" content="desc">
<script>(function(){try{var s=localStorage.getItem('mk-theme');}catch(e){}})();</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces&display=swap" rel="stylesheet">
<link rel="stylesheet" href="./assets/style.css">
<script type="module" src="./assets/index.js"></script>
<link rel="preload" as="image" href="hero.jpg">
<style>@font-face {font-family:test;src:url(test.woff2);font-display:swap;}</style>
</head><body>
<img src="hero.jpg" width="419" height="119" alt="bismillah"><script>performance.mark('x');</script>
</body></html>"""
    # 2 blocking stylesheets + 1 blocking inline script = total 3, module is deferred so depth 3 (<4 => no network_dependency)
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    ids = [i.check_id for i in issues]
    # Should flag render_blocking due to 2 stylesheets + 1 inline (total 3 => high)
    assert "render_blocking" in ids
    rb = [i for i in issues if i.check_id == "render_blocking"][0]
    assert rb.evidence["total_blocking"] == 3
    assert rb.evidence["blocking_stylesheets"] == 2
    assert rb.evidence["blocking_scripts"] == 1
    assert rb.severity == "medium"
    # network dependency depth 3 should NOT flag (threshold 4)
    assert "network_dependency" not in ids
    # main_thread/tti/inp should not flag with only 1 blocking script
    assert "main_thread" not in ids
    assert "tti" not in ids
    assert "inp" not in ids
    # preload should be present so no preload flag
    assert "preload_lcp" not in ids
    # font hints should not flag because preconnect present
    assert "resource_hints" not in ids
    # removed duplicate proxies
    assert "slow_fcp" not in ids
    assert "speed_index" not in ids


@pytest.mark.asyncio
async def test_network_error_graceful():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("mock", request=request)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.com"
    )
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    assert issues == []


@pytest.mark.asyncio
async def test_url_without_scheme():
    html = _perfect_html()
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("example.com", client=client)
    await client.aclose()
    assert issues == []


@pytest.mark.asyncio
async def test_empty_url_raises():
    with pytest.raises(ValueError):
        await analyze_performance("", client=_make_client({}))


@pytest.mark.asyncio
async def test_import_without_side_effects():
    import importlib

    mod = importlib.import_module("ux_analyzer.analysis.performance")
    assert hasattr(mod, "analyze_performance")
    assert hasattr(mod, "analyze_performance_sync")
    assert hasattr(mod, "PerformanceIssue")


# ---------------------------------------------------------------------------
# offending resources are named in evidence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_main_thread_evidence_names_offending_scripts():
    html = """<!doctype html><html><head>
<script>
(function(){window.requestAnimationFrame(function tick(){requestAnimationFrame(tick)});})();
</script>
<script src="/assets/vendor.js"></script>
<link rel="stylesheet" href="/assets/main.css">
</head><body><p>Content</p></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    mt = [i for i in issues if i.check_id == "main_thread"][0]
    detail = mt.evidence["blocking_script_detail"]
    # external offender is named by URL
    assert any(d.get("src") == "/assets/vendor.js" for d in detail)
    # inline offender carries a greppable preview + markers instead of "<inline>"
    inline = [d for d in detail if d.get("inline")]
    assert inline and "requestAnimationFrame" in inline[0]["preview"]
    assert "requestAnimationFrame" in inline[0]["markers"]
    # description names the offenders too
    assert "/assets/vendor.js" in mt.description
    assert "requestAnimationFrame" in mt.description


@pytest.mark.asyncio
async def test_tti_and_inp_evidence_names_offending_scripts():
    html = """<!doctype html><html><head>
<script>
(function(){window.setInterval(function(){}, 1000);})();
</script>
<script src="/assets/bundle.js"></script>
</head><body><p>Content</p></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    tti = [i for i in issues if i.check_id == "tti"][0]
    assert any(
        d.get("src") == "/assets/bundle.js"
        for d in tti.evidence["blocking_script_detail"]
    )
    assert "/assets/bundle.js" in tti.description
    inp = [i for i in issues if i.check_id == "inp"][0]
    assert any(
        d.get("src") == "/assets/bundle.js"
        for d in inp.evidence["blocking_script_detail"]
    )
    assert "/assets/bundle.js" in inp.description


@pytest.mark.asyncio
async def test_render_blocking_evidence_names_inline_scripts():
    html = """<!doctype html><html><head>
<script>window.__cfg = {mode: "dark"};</script>
<link rel="stylesheet" href="/assets/site.css">
</head><body><p>Content</p></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    rb = [i for i in issues if i.check_id == "render_blocking"][0]
    assert rb.evidence["stylesheet_hrefs"] == ["/assets/site.css"]
    inline = [d for d in rb.evidence["blocking_script_detail"] if d.get("inline")]
    assert inline and '__cfg = {mode: "dark"}' in inline[0]["preview"]
    assert "site.css" in rb.description


@pytest.mark.asyncio
async def test_font_display_evidence_names_families():
    html = """<!doctype html><html><head>
<style>@font-face {font-family:Acme;src:url(acme.woff2)} @font-face {font-family:Brand;src:url(brand.woff2);font-display:swap}</style>
</head><body><p>Content</p></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    fd = [i for i in issues if i.check_id == "font_display"][0]
    assert fd.evidence["font_families_affected"] == ["Acme"]
    assert "Acme" in fd.description


# ---------------------------------------------------------------------------
# data scripts (JSON-LD, import maps, templates) are not executable JS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_json_ld_is_not_blocking_or_large_inline():
    html = """<!doctype html><html><head>
<script type="application/ld+json">{"@context":"https://schema.org","@graph":[{"@type":"Person","@id":"#person","name":"Jane Doe"}]}</script>
<script>window.__cfg = {mode: "dark"};</script>
<link rel="stylesheet" href="/assets/site.css">
</head><body><p>Content</p></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    ids = {i.check_id for i in issues}
    # JSON-LD does not block the main thread, delay TTI, or hurt INP
    assert "main_thread" not in ids
    assert "tti" not in ids
    assert "inp" not in ids
    rb = [i for i in issues if i.check_id == "render_blocking"][0]
    assert rb.evidence["blocking_scripts"] == 1
    detail = rb.evidence["blocking_script_detail"]
    assert len(detail) == 1
    assert detail[0]["inline"] and "__cfg" in detail[0]["preview"]
    assert not any("schema.org" in d.get("preview", "") for d in detail)


@pytest.mark.asyncio
async def test_json_ld_large_block_does_not_flag_main_thread():
    # A giant JSON-LD block used to be miscounted as a large inline script.
    payload = (
        '{"@context":"https://schema.org","@graph":['
        + "".join(f'{{"@type":"Place","name":"Place {i}"}},' for i in range(200))
        + "]}"
    )
    html = f"""<!doctype html><html><head>
<script type="application/ld+json">{payload}</script>
</head><body><p>Content</p></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_performance("https://example.com/", client=client)
    await client.aclose()
    ids = {i.check_id for i in issues}
    assert "main_thread" not in ids
    assert "tti" not in ids
    assert "inp" not in ids
    assert "render_blocking" not in ids
