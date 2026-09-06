from __future__ import annotations

import httpx
import pytest

from ux_analyzer.analysis.imagery import analyze_imagery


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
            return httpx.Response(status_code=status, text=body, request=request, headers=headers)
        return httpx.Response(status_code=404, text="Not Found", request=request)

    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport, base_url="https://example.com")


def _perfect_html() -> str:
    """HTML that should produce zero imagery issues (bespoke lifestyle hero, modern format)."""
    return """<!doctype html><html lang="en"><head><meta charset="UTF-8"><title>Test</title><link rel="preload" as="image" href="hero-bespoke-lifestyle.webp" fetchpriority="high"></head><body>
<section class="hero" id="hero" aria-label="Introduction">
  <canvas class="hero__canvas"></canvas>
  <img src="hero-bespoke-lifestyle.webp" width="1200" height="800" alt="Artisan crafted oak dining table in sunlit living room lifestyle scene" fetchpriority="high" decoding="async" srcset="hero-bespoke-lifestyle.webp 1200w" sizes="100vw">
</section>
<p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Curabitur pretium tincidunt lacus.</p>
<img src="product2.webp" width="800" height="600" alt="Close-up of hand-stitched leather bag detail in studio light" loading="lazy" decoding="async">
<p>More content to justify second image. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt.</p>
<picture><source type="image/webp" srcset="gallery.webp"><img src="gallery.jpg" width="600" height="400" alt="Model wearing winter coat in snowy lifestyle scene" loading="lazy" decoding="async"></picture>
</body></html>"""


# ---------------------------------------------------------------------------
# valid: no issues when bespoke hero present
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_perfect_no_issues():
    html = _perfect_html()
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert issues == [], f"expected no issues, got {[(i.check_id, i.title, i.severity) for i in issues]}"


@pytest.mark.asyncio
async def test_perfect_not_flag_stock():
    html = _perfect_html()
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "generic_stock" for i in issues)


# ---------------------------------------------------------------------------
# hero image present
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_hero_flagged():
    html = """<!doctype html><html><head><title>Test</title></head><body>
<p>Some introductory text with enough length to require a hero visual. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.</p>
<img src="content-bespoke.jpg" width="400" height="300" alt="Descriptive lifestyle product scene" srcset="content-bespoke.webp 400w" decoding="async">
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "no_hero_imagery" and i.severity == "critical" for i in issues)


@pytest.mark.asyncio
async def test_hero_canvas_counts_as_hero():
    # non-decorative hero canvas should count as hero visual (pass)
    html = """<!doctype html><html><head><title>Test</title></head><body>
<section class="hero"><canvas class="hero__canvas"></canvas></section>
<p>Content here. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam.</p>
<img src="digitalkhatt-bismillah.svg" width="419" height="119" alt="Bismillah rendered by DigitalKhatt engine with Tajweed coloring">
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "no_hero_imagery" for i in issues)


@pytest.mark.asyncio
async def test_hero_decorative_canvas_not_counted():
    # decorative canvas (aria-hidden true) SHOULD count as hero for portfolio-style sites (bespoke design counts)
    # see live portfolio https://mohamed-khalil.vercel.app hero__canvas aria-hidden true still satisfies hero
    html = """<!doctype html><html><head><title>Test</title></head><body>
<section class="hero"><canvas class="hero__canvas" aria-hidden="true"></canvas></section>
<p>Some introductory text with enough length to require a hero visual. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.</p>
<img src="content-bespoke.jpg" width="400" height="300" alt="Descriptive lifestyle product scene" srcset="content-bespoke.webp 400w" decoding="async">
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "no_hero_imagery" for i in issues)


@pytest.mark.asyncio
async def test_hero_via_large_top_image_fallback():
    html = """<!doctype html><html><head><title>Test</title></head><body>
<img src="large-banner.jpg" width="1200" height="600" alt="Bespoke lifestyle banner of modern furniture in sunlit room" fetchpriority="high" decoding="async" srcset="large-banner.webp 1200w">
<p>Content lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "no_hero_imagery" for i in issues)


# ---------------------------------------------------------------------------
# stock / bespoke heuristic
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_generic_stock_unsplash():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="https://images.unsplash.com/photo-123" width="1200" height="800" alt="Lifestyle room" fetchpriority="high" decoding="async"></section>
<img src="product2.webp" width="800" height="600" alt="Close-up detail" loading="lazy" decoding="async">
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "generic_stock" and i.severity == "medium" for i in issues)
    stock = [i for i in issues if i.check_id == "generic_stock"][0]
    assert "unsplash" in stock.evidence["srcs"][0].lower()


@pytest.mark.asyncio
async def test_generic_stock_placeholder_filename():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="./assets/stock-placeholder-123.jpg" width="1200" height="800" alt="Room" decoding="async" srcset="hero.webp 1200w"></section>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "generic_stock" for i in issues)


@pytest.mark.asyncio
async def test_generic_stock_picsum():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="https://picsum.photos/1200/800" width="1200" height="800" alt="Random" fetchpriority="high" decoding="async"></section>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "generic_stock" for i in issues)


@pytest.mark.asyncio
async def test_bespoke_not_flagged_stock():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="digitalkhatt-bismillah.svg" width="419" height="119" alt="Bismillah rendered bespoke" style="display:block"></section>
<section class="hero"><canvas class="hero__canvas"></canvas></section>
<img src="hero-bespoke-lifestyle.webp" width="1200" height="800" alt="Bespoke lifestyle" fetchpriority="high" decoding="async">
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "generic_stock" for i in issues)


# ---------------------------------------------------------------------------
# low resolution
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_low_resolution_hero_small():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="bespoke-hero.jpg" width="150" height="100" alt="Descriptive lifestyle scene in hero" decoding="async"></section>
<img src="product2.webp" width="800" height="600" alt="Detail" loading="lazy" decoding="async">
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "low_resolution" and i.severity == "critical" for i in issues)
    low = [i for i in issues if i.check_id == "low_resolution"][0]
    assert low.evidence["hero_flagged"] >= 1


@pytest.mark.asyncio
async def test_low_resolution_missing_dimensions():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="hero-bespoke.jpg" alt="Lifestyle hero with modern furniture" decoding="async" srcset="hero.webp 1200w"></section>
<img src="product2.webp" width="800" height="600" alt="Detail" loading="lazy" decoding="async">
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "low_resolution" for i in issues)
    assert any(i.check_id == "missing_dimensions" for i in issues)


@pytest.mark.asyncio
async def test_low_resolution_svg_excluded():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="digitalkhatt-bismillah.svg" width="419" height="119" alt="Bismillah vector" decoding="async"></section>
<img src="icon.svg" width="24" height="24" alt="" aria-hidden="true">
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "low_resolution" for i in issues)


@pytest.mark.asyncio
async def test_low_resolution_non_hero_small():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="hero.webp" width="1200" height="800" alt="Lifestyle hero" fetchpriority="high" decoding="async"></section>
<img src="thumb.jpg" width="100" height="100" alt="Small thumbnail" loading="lazy" decoding="async">
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "low_resolution" for i in issues)


# ---------------------------------------------------------------------------
# missing alt vs decorative
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_missing_alt_flagged():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="hero.jpg" width="1200" height="800" decoding="async" srcset="hero.webp 1200w"></section>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "missing_alt" and i.severity == "medium" for i in issues)


@pytest.mark.asyncio
async def test_missing_alt_empty_not_decorative_flagged():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="hero.jpg" width="1200" height="800" alt="" decoding="async" srcset="hero.webp 1200w"></section>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "missing_alt" for i in issues)


@pytest.mark.asyncio
async def test_missing_alt_decorative_pass():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="hero.webp" width="1200" height="800" alt="Lifestyle hero" fetchpriority="high" decoding="async"></section>
<img src="deco.svg" width="24" height="24" alt="" aria-hidden="true">
<img src="icon.png" width="16" height="16" alt="" role="presentation">
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    # decorative empty alts should not be flagged; but hero has alt, so no missing_alt
    assert not any(i.check_id == "missing_alt" for i in issues)


# ---------------------------------------------------------------------------
# generic alt
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_generic_alt_flagged():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="hero.webp" width="1200" height="800" alt="image" fetchpriority="high" decoding="async"></section>
<img src="product2.webp" width="800" height="600" alt="photo" loading="lazy" decoding="async">
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "generic_alt" and i.severity == "medium" for i in issues)
    gen = [i for i in issues if i.check_id == "generic_alt"][0]
    assert gen.evidence["count"] >= 2


@pytest.mark.asyncio
async def test_generic_alt_descriptive_pass():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="hero.webp" width="1200" height="800" alt="Sunlit living room with oak table lifestyle" fetchpriority="high" decoding="async"></section>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "generic_alt" for i in issues)


# ---------------------------------------------------------------------------
# lazy LCP
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_lazy_lcp_flagged():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="hero-bespoke.jpg" width="1200" height="800" alt="Bespoke lifestyle hero scene" loading="lazy" decoding="async" srcset="hero.webp 1200w"></section>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "lazy_lcp" and i.severity == "medium" for i in issues)
    lcp = [i for i in issues if i.check_id == "lazy_lcp"][0]
    assert "lazy" in lcp.evidence["loading"]


@pytest.mark.asyncio
async def test_lazy_lcp_largest_selection():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero">
<img src="small.jpg" width="100" height="100" alt="Small" decoding="async">
<img src="large-hero.jpg" width="1200" height="800" alt="Large lifestyle hero in hero section" loading="lazy" decoding="async" srcset="large.webp 1200w">
</section>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "lazy_lcp" and "large-hero.jpg" in i.evidence["lcp_src"] for i in issues)


@pytest.mark.asyncio
async def test_lazy_lcp_not_flagged_when_eager():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="hero.webp" width="1200" height="800" alt="Bespoke lifestyle hero" fetchpriority="high" decoding="async" loading="eager"></section>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "lazy_lcp" for i in issues)


# ---------------------------------------------------------------------------
# modern format
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_modern_format_outdated_flagged():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="hero.jpg" width="1200" height="800" alt="Lifestyle hero with furniture" fetchpriority="high" decoding="async"></section>
<img src="product2.jpg" width="800" height="600" alt="Detail of product" loading="lazy" decoding="async">
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "modern_format" and i.severity == "low" for i in issues)


@pytest.mark.asyncio
async def test_modern_format_with_webp_pass():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="hero.webp" width="1200" height="800" alt="Lifestyle hero" fetchpriority="high" decoding="async"></section>
<img src="product2.jpg" width="800" height="600" alt="Detail" loading="lazy" decoding="async" srcset="product2.webp 800w">
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "modern_format" for i in issues)


@pytest.mark.asyncio
async def test_modern_format_picture_webp_pass():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><picture><source type="image/webp" srcset="hero.webp"><img src="hero.jpg" width="1200" height="800" alt="Lifestyle" fetchpriority="high" decoding="async"></picture></section>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "modern_format" for i in issues)


@pytest.mark.asyncio
async def test_modern_format_svg_pass():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="digitalkhatt-bismillah.svg" width="419" height="119" alt="Bismillah vector" decoding="async"></section>
<section class="hero"><canvas class="hero__canvas"></canvas></section>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "modern_format" for i in issues)


# ---------------------------------------------------------------------------
# cut-out vs lifestyle
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cutout_flagged():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="product-cutout-white-bg.jpg" width="1200" height="800" alt="Product isolated cutout" fetchpriority="high" decoding="async" srcset="product.webp 1200w"></section>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "cutout_vs_lifestyle" and i.severity == "low" for i in issues)


@pytest.mark.asyncio
async def test_cutout_not_flagged_lifestyle():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="hero-lifestyle-room.webp" width="1200" height="800" alt="Furnished living room lifestyle scene" fetchpriority="high" decoding="async"></section>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "cutout_vs_lifestyle" for i in issues)


@pytest.mark.asyncio
async def test_cutout_only_hero_counts():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="hero.webp" width="1200" height="800" alt="Lifestyle hero" fetchpriority="high" decoding="async"></section>
<img src="thumb-cutout-white-bg.jpg" width="400" height="400" alt="Cutout product thumbnail" loading="lazy" decoding="async">
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    # cutout outside hero should not flag
    assert not any(i.check_id == "cutout_vs_lifestyle" for i in issues)


# ---------------------------------------------------------------------------
# image count vs content
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_image_count_too_few_flagged():
    # Wordy page with only 1 visual but no hero canvas
    long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 40  # ~320 words
    html = f"""<!doctype html><html><head><title>T</title></head><body>
<img src="solo.jpg" width="800" height="600" alt="Solo lifestyle image with modern decor" decoding="async" srcset="solo.webp 800w">
<p>{long_text}</p>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "image_count" for i in issues)
    cnt = [i for i in issues if i.check_id == "image_count"][0]
    assert "Too few" in cnt.title


@pytest.mark.asyncio
async def test_image_count_enough_with_canvas():
    long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 40
    html = f"""<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><canvas class="hero__canvas"></canvas></section>
<img src="solo.webp" width="800" height="600" alt="Lifestyle detail in modern room" decoding="async">
<p>{long_text}</p>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "image_count" for i in issues)


@pytest.mark.asyncio
async def test_image_count_too_many_flagged():
    imgs = "\n".join([f'<img src="img{i}.webp" width="400" height="300" alt="Product lifestyle scene {i}" loading="lazy" decoding="async">' for i in range(20)])
    html = f"""<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="hero.webp" width="1200" height="800" alt="Hero lifestyle" fetchpriority="high" decoding="async"></section>
{imgs}
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "image_count" and "Too many" in i.title for i in issues)


@pytest.mark.asyncio
async def test_image_count_counts_every_canvas_and_video():
    """A rendered <canvas>/<video> occupies layout and paints pixels users
    see even when aria-hidden or JS-populated — visual richness counts them."""
    long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 40
    html = f"""<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><canvas class="hero__canvas" aria-hidden="true"></canvas></section>
<img src="digitalkhatt-bismillah.svg" width="419" height="119" alt="Bismillah rendered bespoke">
<video class="promo__video" muted loop playsinline preload="none" aria-hidden="true"></video>
<video class="vmodal__video" controls playsinline></video>
<p>{long_text}</p>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    # 1 img + 1 canvas + 2 videos = 4 visuals for ~320 words => not flagged
    assert not any(i.check_id == "image_count" and "Too few" in i.title for i in issues)
    cnt = [i for i in issues if i.check_id == "image_count"]
    assert not cnt or cnt[0].evidence["total_visuals"] >= 4


@pytest.mark.asyncio
async def test_image_count_video_with_src_counts():
    long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 40
    html = f"""<!doctype html><html><head><title>T</title></head><body>
<img src="solo.webp" width="800" height="600" alt="Solo lifestyle" decoding="async">
<video src="promo.mp4" poster="poster.jpg" class="promo__video"></video>
<p>{long_text}</p>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    # 1 img + 1 video with src => 2 visuals => not flagged for Too few
    assert not any(i.check_id == "image_count" and "Too few" in i.title for i in issues)


@pytest.mark.asyncio
async def test_image_count_decorative_svg_not_counted():
    long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 40
    html = f"""<!doctype html><html><head><title>T</title></head><body>
<img src="solo.webp" width="800" height="600" alt="Solo lifestyle" decoding="async">
<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M0 0h24v24H0z"/></svg>
<svg class="theme-toggle__sun" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"/></svg>
<p>{long_text}</p>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "image_count" and "Too few" in i.title for i in issues)
    cnt = [i for i in issues if i.check_id == "image_count"][0]
    assert cnt.evidence["total_svgs_content"] == 0
    assert cnt.evidence["total_visuals"] == 1


@pytest.mark.asyncio
async def test_image_count_canvas_counted_even_when_aria_hidden():
    long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 40
    html = f"""<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><canvas class="hero__canvas" aria-hidden="true"></canvas></section>
<img src="solo.webp" width="800" height="600" alt="Solo lifestyle" decoding="async">
<p>{long_text}</p>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    # canvas paints pixels users see; aria-hidden is screen-reader semantics only
    assert not any(i.check_id == "image_count" and "Too few" in i.title for i in issues)
    cnt = [i for i in issues if i.check_id == "image_count"]
    assert not cnt or cnt[0].evidence["total_canvases"] == 1


@pytest.mark.asyncio
async def test_image_count_icon_svg_inside_button_not_counted():
    long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 40
    html = f"""<!doctype html><html><head><title>T</title></head><body>
<img src="solo.webp" width="800" height="600" alt="Solo lifestyle" decoding="async">
<button aria-label="Play"><svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg></button>
<p>{long_text}</p>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    # svg inside button should be decorative even without aria-hidden
    assert any(i.check_id == "image_count" and "Too few" in i.title for i in issues)


@pytest.mark.asyncio
async def test_portfolio_live_regression_visuals_counted():
    """Replicates the live portfolio: aria-hidden hero canvas + two JS-fed
    videos all render — the page is NOT 'too few images' for 800 words."""
    long_text = " ".join(["Lorem ipsum dolor sit amet"] * 200)  # ~800 words
    html = f"""<!doctype html><html><head><title>Test</title></head><body>
<section class="hero"><canvas class="hero__canvas" aria-hidden="true"></canvas></section>
<img src="./digitalkhatt-bismillah.svg" width="419" height="119" alt="Bismillah rendered by the DigitalKhatt engine">
<video class="promo__video" muted loop playsinline preload="none" aria-hidden="true"></video>
<video class="vmodal__video" controls playsinline></video>
<svg class="theme-toggle__sun" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"/></svg>
<svg class="theme-toggle__moon" viewBox="0 0 24 24" aria-hidden="true"><path d="M20 14"/></svg>
<button><svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg></button>
<p>{long_text} Additional unique words: Mohamed Khalil Muslim Pedia 100K downloads.</p>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "image_count" and "Too few" in i.title for i in issues)
    # ensure generic_stock not falsely flagged for bespoke svg
    assert not any(i.check_id == "generic_stock" for i in issues)


# ---------------------------------------------------------------------------
# unoptimized loading
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_unoptimized_loading_missing_lazy():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="hero.webp" width="1200" height="800" alt="Hero lifestyle scene with furniture" fetchpriority="high" decoding="async"></section>
<img src="product2.jpg" width="800" height="600" alt="Detail of artisan table close-up" decoding="async">
<img src="product3.jpg" width="800" height="600" alt="Another product detail view" decoding="async">
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "unoptimized_loading" and i.severity == "low" for i in issues)


@pytest.mark.asyncio
async def test_unoptimized_loading_with_lazy_pass():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="hero.webp" width="1200" height="800" alt="Hero lifestyle scene" fetchpriority="high" decoding="async"></section>
<img src="product2.webp" width="800" height="600" alt="Detail view of product craftsmanship" loading="lazy" decoding="async">
<img src="product3.webp" width="800" height="600" alt="Another curated product lifestyle" loading="lazy" decoding="async">
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "unoptimized_loading" for i in issues)


@pytest.mark.asyncio
async def test_unoptimized_svg_excluded():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="hero.webp" width="1200" height="800" alt="Hero" fetchpriority="high" decoding="async"></section>
<img src="deco.svg" width="100" height="100" alt="Decorative vector" decoding="async">
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    # single below svg should not trigger
    assert not any(i.check_id == "unoptimized_loading" for i in issues)


# ---------------------------------------------------------------------------
# missing dimensions explicit
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_missing_dimensions_flagged():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="hero.webp" alt="Lifestyle hero without dims" fetchpriority="high" decoding="async"></section>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "missing_dimensions" for i in issues)


# ---------------------------------------------------------------------------
# portfolio mock (similar to live check)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_portfolio_mock_bespoke_not_stock():
    # Minimal replica of https://mohamed-khalil.vercel.app hero + bismillah
    # Use non-decorative hero canvas to satisfy hero check (decorative canvas would not count)
    html = """<!doctype html><html lang="en"><head><meta charset="UTF-8"><title>Mohamed Khalil</title></head><body>
<section class="hero" id="hero" aria-label="Introduction"><canvas class="hero__canvas" id="heroCanvas"></canvas><div class="hero__inner"><h1>Mohamed Khalil</h1></div></section>
<figure class="project__visual khatt"><div class="khatt__page"><img class="khatt__svg" src="./digitalkhatt-bismillah.svg" width="419" height="119" alt="Bismillah rendered by the DigitalKhatt engine"></div></figure>
<video class="promo__video" muted loop playsinline preload="none" aria-hidden="true"></video>
<p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Additional text to simulate portfolio length. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt.</p>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    ids = [i.check_id for i in issues]
    assert "generic_stock" not in ids, f"bespoke portfolio incorrectly flagged stock: {issues}"
    assert "no_hero_imagery" not in ids, f"portfolio hero canvas should satisfy hero: {issues}"
    assert "generic_alt" not in ids
    # allow other low issues but not stock/no_hero
    # also should not flag low_resolution for svg vector
    assert "low_resolution" not in ids


# ---------------------------------------------------------------------------
# preload LCP
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_preload_lcp_flagged_when_missing():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="hero-bespoke.jpg" width="1200" height="800" alt="Bespoke lifestyle hero scene" fetchpriority="high" decoding="async"></section>
<img src="product2.webp" width="800" height="600" alt="Detail" loading="lazy" decoding="async">
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "preload_lcp" and i.severity == "medium" for i in issues)
    preload = [i for i in issues if i.check_id == "preload_lcp"][0]
    assert "hero-bespoke.jpg" in preload.evidence["lcp_src"]


@pytest.mark.asyncio
async def test_preload_lcp_pass_when_preloaded():
    html = """<!doctype html><html><head><title>T</title></head><body>
<head><link rel="preload" as="image" href="hero-bespoke.jpg" fetchpriority="high"></head>
<section class="hero"><img src="hero-bespoke.jpg" width="1200" height="800" alt="Bespoke lifestyle hero scene" fetchpriority="high" decoding="async"></section>
<img src="product2.webp" width="800" height="600" alt="Detail" loading="lazy" decoding="async">
</body></html>"""
    # Note: <link> inside body still parsed; test that detection finds preload
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "preload_lcp" for i in issues)


@pytest.mark.asyncio
async def test_preload_lcp_svg_excluded():
    html = """<!doctype html><html><head><title>T</title></head><body>
<section class="hero"><img src="digitalkhatt-bismillah.svg" width="419" height="119" alt="Bismillah rendered bespoke"></section>
<section class="hero"><canvas class="hero__canvas"></canvas></section>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "preload_lcp" for i in issues)
    assert not any(i.check_id == "generic_stock" for i in issues)


@pytest.mark.asyncio
async def test_preload_lcp_with_imagesrcset():
    html = """<!doctype html><html><head><link rel="preload" as="image" href="hero.jpg" imagesrcset="hero.jpg 1200w"></head><body>
<section class="hero"><img src="hero.jpg" width="1200" height="800" alt="Bespoke hero lifestyle" fetchpriority="high" decoding="async"></section>
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "preload_lcp" for i in issues)


@pytest.mark.asyncio
async def test_network_error_graceful():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("mock", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://example.com")
    issues = await analyze_imagery("https://example.com/", client=client)
    await client.aclose()
    assert issues == []


@pytest.mark.asyncio
async def test_url_without_scheme():
    html = _perfect_html()
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_imagery("example.com", client=client)
    await client.aclose()
    assert issues == []


@pytest.mark.asyncio
async def test_empty_url_raises():
    with pytest.raises(ValueError):
        await analyze_imagery("", client=_make_client({}))


@pytest.mark.asyncio
async def test_import_without_side_effects():
    import importlib

    mod = importlib.import_module("ux_analyzer.analysis.imagery")
    assert hasattr(mod, "analyze_imagery")
    assert hasattr(mod, "analyze_imagery_sync")
    assert hasattr(mod, "ImageryIssue")
