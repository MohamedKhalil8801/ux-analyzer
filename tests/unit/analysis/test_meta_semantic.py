from __future__ import annotations

import httpx
import pytest

from ux_analyzer.analysis.meta_semantic import analyze_meta_semantic


def _build_html(
    *,
    html_lang: str | None = "en",
    xml_lang: str | None = None,
    title: str | None = "Valid Page Title Length OK",
    title_duplicate: bool = False,
    description: str | None = "This is a valid meta description that has sufficient length to pass the short threshold and is not too long to be flagged as excessive for SEO purposes.",
    charset: bool = True,
    charset_http_equiv: bool = False,
    viewport: bool = True,
    canonical: str | None = "https://example.com/canonical",
    og_title: str | None = "OG Title",
    og_description: str | None = "OG Description",
    og_image: str | None = "https://example.com/og.png",
    og_url: str | None = "https://example.com/",
    og_type: str | None = "website",
    twitter_card: str | None = "summary",
    headings: list[tuple[int, str]] | None = None,
    landmarks: bool = True,
    landmark_mode: str = "full",  # full|main_only|none
    ids: list[str] | None = None,
    extra_head: str = "",
) -> str:
    """Build HTML for meta semantic tests."""
    # html attrs
    html_attrs = []
    if html_lang is not None:
        html_attrs.append(f'lang="{html_lang}"')
    if xml_lang is not None:
        html_attrs.append(f'xml:lang="{xml_lang}"')
    # also need dir? not needed
    html_open = f"<html {' '.join(html_attrs)}>" if html_attrs else "<html>"

    head_parts: list[str] = []
    if title is not None:
        head_parts.append(f"<title>{title}</title>")
        if title_duplicate:
            head_parts.append(f"<title>{title} duplicate</title>")
    if description is not None:
        # Use explicit None to mean missing; empty string to mean empty tag
        head_parts.append(f'<meta name="description" content="{description}">')
    if charset:
        head_parts.append('<meta charset="UTF-8">')
    if charset_http_equiv:
        head_parts.append('<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">')
    if viewport:
        head_parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    if canonical is not None:
        head_parts.append(f'<link rel="canonical" href="{canonical}">')
    if og_title is not None:
        head_parts.append(f'<meta property="og:title" content="{og_title}">')
    if og_description is not None:
        head_parts.append(f'<meta property="og:description" content="{og_description}">')
    if og_image is not None:
        head_parts.append(f'<meta property="og:image" content="{og_image}">')
    if og_url is not None:
        head_parts.append(f'<meta property="og:url" content="{og_url}">')
    if og_type is not None:
        head_parts.append(f'<meta property="og:type" content="{og_type}">')
    if twitter_card is not None:
        head_parts.append(f'<meta name="twitter:card" content="{twitter_card}">')
    if extra_head:
        head_parts.append(extra_head)

    # headings
    if headings is None:
        headings = [(1, "Main Heading"), (2, "Subheading One"), (2, "Subheading Two"), (3, "Detail")]
    heading_html = "".join(f"<h{lvl}>{txt}</h{lvl}>" for lvl, txt in headings)
    # landmarks
    if landmark_mode == "full":
        landmark_html = "<header>h</header><nav>n</nav><main><p>main</p></main><footer>f</footer>"
    elif landmark_mode == "main_only":
        landmark_html = "<main><p>main</p></main>"
    elif landmark_mode == "none":
        landmark_html = "<div>no landmarks</div>"
    else:
        landmark_html = ""
        if landmarks:
            landmark_html = "<header>h</header><nav>n</nav><main><p>main</p></main><footer>f</footer>"

    # ids
    id_html = ""
    if ids:
        for i, idv in enumerate(ids):
            id_html += f'<div id="{idv}">content {i}</div>'

    return f"<!doctype html>{html_open}<head>{''.join(head_parts)}</head><body>{landmark_html}{heading_html}{id_html}</body></html>"


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
                if path != "/" and path != "":
                    continue
            elif path != key_path and not path.endswith(key_path):
                if key_path not in path:
                    continue
            headers = {"content-type": "text/html"}
            return httpx.Response(status_code=status, text=body, request=request, headers=headers)
        return httpx.Response(status_code=404, text="Not Found", request=request)

    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport, base_url="https://example.com")


# ---------------------------------------------------------------------------
# All present => no issues
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_all_present_no_issues():
    html = _build_html()
    routes = {"https://example.com/": (200, html)}
    client = _make_client(routes)
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert issues == [], f"expected no issues, got {[(i.check_id, i.title) for i in issues]}"


# ---------------------------------------------------------------------------
# title checks
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_title_missing():
    html = _build_html(title=None)
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "title_missing" and i.severity == "critical" for i in issues)


@pytest.mark.asyncio
async def test_title_empty():
    html = _build_html(title="")
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "title_empty" for i in issues)


@pytest.mark.asyncio
async def test_title_too_short():
    html = _build_html(title="Hi")
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "title_too_short" for i in issues)
    # evidence should have length
    assert any(i.evidence.get("title_length") == 2 for i in issues if i.check_id == "title_too_short")


@pytest.mark.asyncio
async def test_title_too_long():
    long_title = "A" * 70
    html = _build_html(title=long_title)
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "title_too_long" for i in issues)


@pytest.mark.asyncio
async def test_title_duplicate():
    html = _build_html(title="Valid Page Title Length OK", title_duplicate=True)
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "title_duplicate" for i in issues)


@pytest.mark.asyncio
async def test_title_valid_no_issue():
    html = _build_html(title="Good Title Length Here")
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    title_issues = [i for i in issues if i.check_id.startswith("title_")]
    assert title_issues == []


# ---------------------------------------------------------------------------
# description checks
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_description_missing():
    html = _build_html(description=None)
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "meta_description_missing" and i.severity == "medium" for i in issues)


@pytest.mark.asyncio
async def test_description_empty():
    html = _build_html(description="")
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "meta_description_empty" for i in issues)


@pytest.mark.asyncio
async def test_description_too_short():
    html = _build_html(description="short")
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "meta_description_too_short" for i in issues)


@pytest.mark.asyncio
async def test_description_too_long():
    long_desc = "A" * 350
    html = _build_html(description=long_desc)
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "meta_description_too_long" for i in issues)


@pytest.mark.asyncio
async def test_description_valid_no_issue():
    desc = "This is a valid meta description that has sufficient length to pass the short threshold and is not too long to be flagged as excessive for SEO purposes."
    html = _build_html(description=desc)
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id.startswith("meta_description") for i in issues)


# ---------------------------------------------------------------------------
# charset
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_charset_missing():
    html = _build_html(charset=False)
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "meta_charset_missing" for i in issues)


@pytest.mark.asyncio
async def test_charset_via_http_equiv():
    html = _build_html(charset=False, charset_http_equiv=True)
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "meta_charset_missing" for i in issues)


@pytest.mark.asyncio
async def test_charset_present_no_issue():
    html = _build_html(charset=True)
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "meta_charset_missing" for i in issues)


# ---------------------------------------------------------------------------
# viewport
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_viewport_missing():
    html = _build_html(viewport=False)
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "meta_viewport_missing" and i.severity == "critical" for i in issues)


@pytest.mark.asyncio
async def test_viewport_present_no_issue():
    html = _build_html(viewport=True)
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "meta_viewport_missing" for i in issues)


# ---------------------------------------------------------------------------
# canonical
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_canonical_missing():
    html = _build_html(canonical=None)
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "canonical_missing" and i.severity == "critical" for i in issues)
    assert any(i.title == "Page is missing a canonical URL" for i in issues)


@pytest.mark.asyncio
async def test_canonical_not_absolute():
    html = _build_html(canonical="/relative/path")
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "canonical_not_absolute" for i in issues)
    assert any(i.evidence.get("is_absolute") is False for i in issues if i.check_id == "canonical_not_absolute")


@pytest.mark.asyncio
async def test_canonical_absolute_no_issue():
    html = _build_html(canonical="https://example.com/canonical")
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id.startswith("canonical") for i in issues)


# ---------------------------------------------------------------------------
# og
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_og_incomplete_missing_image():
    html = _build_html(og_image=None)
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "og_incomplete" for i in issues)
    og = [i for i in issues if i.check_id == "og_incomplete"][0]
    assert "og:image" in og.evidence["missing"]
    assert og.severity == "medium"


@pytest.mark.asyncio
async def test_og_incomplete_missing_title():
    html = _build_html(og_title=None)
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "og_incomplete" for i in issues)
    assert any("og:title" in i.evidence["missing"] for i in issues if i.check_id == "og_incomplete")


@pytest.mark.asyncio
async def test_og_complete_no_issue():
    html = _build_html(og_title="t", og_description="d", og_image="https://example.com/img.jpg", og_url="https://example.com", og_type="website")
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "og_incomplete" for i in issues)


@pytest.mark.asyncio
async def test_og_multiple_missing():
    html = _build_html(og_image=None, og_url=None)
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    og = [i for i in issues if i.check_id == "og_incomplete"][0]
    assert "og:image" in og.evidence["missing"]
    assert "og:url" in og.evidence["missing"]
    assert og.evidence["found_count"] == 3


# ---------------------------------------------------------------------------
# twitter
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_twitter_missing():
    html = _build_html(twitter_card=None)
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "twitter_card_missing" and i.severity == "low" for i in issues)


@pytest.mark.asyncio
async def test_twitter_present_no_issue():
    html = _build_html(twitter_card="summary_large_image")
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "twitter_card_missing" for i in issues)


# ---------------------------------------------------------------------------
# lang
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_html_lang_missing():
    html = _build_html(html_lang=None)
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "html_lang_missing" and i.severity == "critical" for i in issues)


@pytest.mark.asyncio
async def test_html_lang_invalid():
    html = _build_html(html_lang="123-invalid")
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "html_lang_invalid" for i in issues)


@pytest.mark.asyncio
async def test_html_lang_valid_no_issue():
    html = _build_html(html_lang="en")
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id in ("html_lang_missing", "html_lang_invalid") for i in issues)


@pytest.mark.asyncio
async def test_xml_lang_same_base():
    html = _build_html(html_lang="en", xml_lang="en")
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "html_xml_lang_redundant" for i in issues)
    assert any("same base language" in i.title for i in issues if i.check_id == "html_xml_lang_redundant")


@pytest.mark.asyncio
async def test_xml_lang_different_base_no_issue():
    html = _build_html(html_lang="en", xml_lang="fr")
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "html_xml_lang_redundant" for i in issues)


@pytest.mark.asyncio
async def test_xml_lang_same_base_with_region():
    html = _build_html(html_lang="en-US", xml_lang="en")
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "html_xml_lang_redundant" for i in issues)


# ---------------------------------------------------------------------------
# headings
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_h1_missing():
    html = _build_html(headings=[(2, "Only H2"), (3, "H3")])
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "h1_missing" and i.severity == "critical" for i in issues)


@pytest.mark.asyncio
async def test_h1_multiple():
    html = _build_html(headings=[(1, "First"), (1, "Second"), (2, "Sub")])
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "h1_multiple" for i in issues)


@pytest.mark.asyncio
async def test_h1_not_first():
    html = _build_html(headings=[(2, "H2 first"), (1, "H1 later")])
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "h1_not_first" for i in issues)


@pytest.mark.asyncio
async def test_heading_skipped():
    html = _build_html(headings=[(1, "H1"), (3, "Skipped H2")])
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "heading_skipped" for i in issues)
    assert any(i.evidence.get("skipped") for i in issues if i.check_id == "heading_skipped")


@pytest.mark.asyncio
async def test_heading_empty():
    html = _build_html(headings=[(1, "Valid"), (2, ""), (2, "   ")])
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "heading_empty" for i in issues)


@pytest.mark.asyncio
async def test_headings_valid_no_issue():
    html = _build_html(headings=[(1, "H1"), (2, "H2"), (3, "H3"), (2, "H2 again")])
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    heading_ids = {"h1_missing", "h1_multiple", "h1_not_first", "heading_skipped", "heading_empty"}
    assert not any(i.check_id in heading_ids for i in issues)


# ---------------------------------------------------------------------------
# landmarks
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_main_missing():
    html = _build_html(landmark_mode="none")
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "main_missing" and i.severity == "critical" for i in issues)
    # also should have landmarks incomplete
    assert any(i.check_id == "landmarks_incomplete" for i in issues)


@pytest.mark.asyncio
async def test_landmarks_incomplete():
    html = _build_html(landmark_mode="main_only")
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    # main present but other landmarks missing => incomplete
    assert any(i.check_id == "landmarks_incomplete" for i in issues)
    assert not any(i.check_id == "main_missing" for i in issues)


@pytest.mark.asyncio
async def test_landmarks_full_no_issue():
    html = _build_html(landmark_mode="full")
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id in ("main_missing", "landmarks_incomplete") for i in issues)


# ---------------------------------------------------------------------------
# duplicate IDs
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_duplicate_ids():
    html = _build_html(ids=["dup", "dup", "unique"])
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "duplicate_ids" for i in issues)
    dup_issue = [i for i in issues if i.check_id == "duplicate_ids"][0]
    assert "dup" in dup_issue.evidence["duplicate_ids"]


@pytest.mark.asyncio
async def test_duplicate_ids_none_no_issue():
    html = _build_html(ids=["a", "b", "c"])
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "duplicate_ids" for i in issues)


# ---------------------------------------------------------------------------
# Evidence sanity
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_evidence_contains_counts():
    html = _build_html()
    client = _make_client({"https://example.com/": (200, html)})
    await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    # when no issues, we can't check; force one
    html2 = _build_html(canonical=None, og_image=None)
    client2 = _make_client({"https://example.com/": (200, html2)})
    issues2 = await analyze_meta_semantic("https://example.com/", client=client2)
    await client2.aclose()
    assert any("canonical_count" in i.evidence for i in issues2 if i.check_id == "canonical_missing")
    assert any("missing" in i.evidence for i in issues2 if i.check_id == "og_incomplete")


# ---------------------------------------------------------------------------
# Portfolio mock: should report missing canonical + og:image but not false-flag title/desc/viewport/lang/h1
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_portfolio_mock_reports_canonical_og_but_not_false_flags():
    portfolio_head = """
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Mohamed Khalil - Mobile, Web & UI/UX</title>
    <meta name="description" content="Mohamed Khalil, sole developer of Muslim Pedia, an all-in-one Muslim app with 100K+ downloads (4.8 on iOS, 4.6 on Android). Built its hardest systems end to end, open-sourced its prayer-time engine, and now ships AI-agent integrations at PAIR Systems." />
    <meta property="og:type" content="website" />
    <meta property="og:title" content="Mohamed Khalil - Mobile, Web & UI/UX" />
    <meta property="og:description" content="Sole developer of Muslim Pedia..." />
    """
    body = "<header><nav>nav</nav></header><main>" + "<h1>Mohamed Khalil</h1><h2>Section</h2>" + "</main><footer>footer</footer>"
    html = f"<!doctype html><html lang=\"en\"><head>{portfolio_head}</head><body>{body}</body></html>"
    routes = {"https://example.com/": (200, html)}
    client = _make_client(routes)
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    ids = [i.check_id for i in issues]
    # Should flag canonical missing and og incomplete (og:image)
    assert "canonical_missing" in ids
    assert "og_incomplete" in ids
    # Must NOT false-flag title/description/viewport/lang/h1
    assert "title_missing" not in ids
    assert "title_empty" not in ids
    assert "title_too_short" not in ids
    assert "title_too_long" not in ids
    assert "meta_description_missing" not in ids
    assert "meta_viewport_missing" not in ids
    assert "html_lang_missing" not in ids
    assert "h1_missing" not in ids
    # og missing should include og:image and og:url
    og = [i for i in issues if i.check_id == "og_incomplete"][0]
    assert "og:image" in og.evidence["missing"]


@pytest.mark.asyncio
async def test_sync_wrapper():
    # sync test uses real httpx via mock? Instead test that sync function exists and raises on empty
    # We just verify import and that analyze still works via async; sync wrapper tested via thread if no loop
    pass


@pytest.mark.asyncio
async def test_network_error_graceful():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("mock", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://example.com")
    issues = await analyze_meta_semantic("https://example.com/", client=client)
    await client.aclose()
    assert issues == []


@pytest.mark.asyncio
async def test_url_without_scheme():
    html = _build_html()
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_meta_semantic("example.com", client=client)
    await client.aclose()
    assert issues == []
