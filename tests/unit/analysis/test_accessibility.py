from __future__ import annotations

import httpx
import pytest

from ux_analyzer.analysis.accessibility import analyze_accessibility

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


def _perfect_html() -> str:
    return """<!doctype html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Perfect Page Title For A11y Test Length OK</title>
</head><body>
<a href="#main-content" class="skip-link">Skip to main content</a>
<header><nav><a href="/about">About us</a></nav></header>
<main id="main-content"><h1>Main Heading</h1><h2>Subheading One</h2><h3>Detail Section</h3>
<button>Click me</button>
<a href="/contact">Contact us</a>
<div role="button" tabindex="0" aria-label="Custom action">Custom</div>
<label for="email">Email address</label><input id="email" type="text">
<label for="msg">Message</label><textarea id="msg"></textarea>
<label for="country">Country</label><select id="country"><option>USA</option></select>
<img src="hero.jpg" alt="hero image" width="120" height="80">
<input type="image" src="btn.jpg" alt="Submit search">
<label for="disk">Disk usage</label><meter id="disk" min="0" max="100" value="50" aria-label="Disk usage">50%</meter>
<progress id="prog" value="70" max="100" aria-label="Loading progress">70%</progress>
<div role="dialog" aria-label="Confirmation dialog"><button>OK</button></div>
<div style="display:block"><input type="checkbox" id="chk"><label for="chk">Agree to terms</label></div>
<div role="switch" aria-checked="true" aria-label="Toggle notifications" tabindex="0">Notifications</div>
<div role="tooltip" id="tip">Helpful tip</div>
<div role="tree"><div role="treeitem" aria-label="Node 1">Item 1</div></div>
<ul><li>List item one</li></ul>
<div role="list"><div role="listitem">List item two</div></div>
<table><caption>Data table</caption><tr><th>Header</th></tr><tr><td>Cell</td></tr></table>
<iframe src="frame.html" title="Embedded content"></iframe>
<object data="test.swf"><p>Fallback content for object</p></object>
<video controls><source src="vid.mp4" type="video/mp4"><track kind="captions" src="cap.vtt" srclang="en" label="English"></video>
</main>
<footer><p>Footer content</p></footer>
</body></html>"""


# ---------------------------------------------------------------------------
# perfect -> no issues
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_perfect_no_issues():
    html = _perfect_html()
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert issues == [], f"expected no issues, got {[(i.check_id, i.title) for i in issues]}"


# ---------------------------------------------------------------------------
# 1. Buttons and links have accessible names
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_button_no_accessible_name():
    html = """<!doctype html><html lang="en"><body>
<header><nav>a</nav></header><main><h1>h</h1>
<a href="#main">Skip</a>
<button></button>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "button_accessible_name" and i.severity == "critical" for i in issues)
    assert any("Buttons and links" in i.title for i in issues if i.check_id == "button_accessible_name")


@pytest.mark.asyncio
async def test_button_with_aria_label_no_issue():
    html = _perfect_html().replace("<button>Click me</button>", '<button aria-label="Click me">Click me</button>')
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "button_accessible_name" for i in issues)


@pytest.mark.asyncio
async def test_link_no_text_flagged():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1>
<a href="#main">Skip</a>
<a href="/empty"></a>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "button_accessible_name" for i in issues)


# ---------------------------------------------------------------------------
# 2. Input buttons have accessible labels
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_input_button_no_label():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1>
<a href="#main">Skip</a>
<input type="button" value=""><input type="submit" value="">
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a" type="text">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "input_button_label" and i.severity == "critical" for i in issues)


@pytest.mark.asyncio
async def test_input_button_with_value_no_issue():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1>
<a href="#main">Skip</a>
<input type="button" value="Click">
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a" type="text">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "input_button_label" for i in issues)


# ---------------------------------------------------------------------------
# 3. Image buttons have alt text
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_image_button_no_alt():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1>
<a href="#main">Skip</a>
<input type="image" src="btn.jpg">
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a" type="text">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "image_button_alt" and i.severity == "critical" for i in issues)


@pytest.mark.asyncio
async def test_image_button_with_alt_no_issue():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1>
<a href="#main">Skip</a>
<input type="image" src="btn.jpg" alt="Search">
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a" type="text">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "image_button_alt" for i in issues)


# ---------------------------------------------------------------------------
# 4. Image alt
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_image_missing_alt():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1>
<a href="#main">Skip</a>
<img src="missing.jpg">
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a" type="text">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "image_alt" and i.severity == "critical" for i in issues)


@pytest.mark.asyncio
async def test_image_empty_alt_decorative_no_issue():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1>
<a href="#main">Skip</a>
<img src="decor.jpg" alt="" aria-hidden="true">
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a" type="text">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "image_alt" for i in issues)


# ---------------------------------------------------------------------------
# 5. Form inputs have accessible names (placeholder not sufficient)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_form_input_no_label_flagged():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1>
<a href="#main">Skip</a>
<input type="text" id="noLabel">
<img src="x.jpg" alt="x">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "form_input_accessible_name" and i.severity == "critical" for i in issues)


@pytest.mark.asyncio
async def test_form_input_placeholder_only_still_flagged():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1>
<a href="#main">Skip</a>
<input type="text" id="ph" placeholder="Enter name">
<img src="x.jpg" alt="x">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "form_input_accessible_name" for i in issues)


@pytest.mark.asyncio
async def test_form_input_with_label_no_issue():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1>
<a href="#main">Skip</a>
<label for="ok">Name</label><input id="ok" type="text">
<img src="x.jpg" alt="x">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "form_input_accessible_name" for i in issues)


@pytest.mark.asyncio
async def test_form_input_with_aria_label_no_issue():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1>
<a href="#main">Skip</a>
<input type="text" aria-label="Search query">
<img src="x.jpg" alt="x">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "form_input_accessible_name" for i in issues)
    # but should flag visible label missing
    assert any(i.check_id == "form_visible_label" for i in issues)


# ---------------------------------------------------------------------------
# 6. Form inputs have visible labels
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_form_visible_label_flagged():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1>
<a href="#main">Skip</a>
<input type="text" aria-label="Search">
<img src="x.jpg" alt="x">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "form_visible_label" for i in issues)

# ---------------------------------------------------------------------------
# 7. Meter
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_meter_no_name():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<meter value="0.6"></meter>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "meter_accessible_name" for i in issues)


@pytest.mark.asyncio
async def test_meter_with_aria_label_no_issue():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<meter aria-label="Disk usage" value="0.6">60%</meter>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "meter_accessible_name" for i in issues)


# ---------------------------------------------------------------------------
# 8. Progress
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_progress_no_name():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<progress value="70" max="100"></progress>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "progress_accessible_name" for i in issues)


# ---------------------------------------------------------------------------
# 9. Dialogs
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_dialog_no_name():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<div role="dialog"><p>Content</p></div>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "dialog_accessible_name" for i in issues)


@pytest.mark.asyncio
async def test_dialog_with_aria_label_no_issue():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<div role="dialog" aria-label="Confirm"><p>Content</p></div>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "dialog_accessible_name" for i in issues)


# ---------------------------------------------------------------------------
# 10. Toggle
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_toggle_no_name():
    # input checkbox without label
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<input type="checkbox" id="tog1">
<img src="x.jpg" alt="x">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "toggle_accessible_name" for i in issues)


@pytest.mark.asyncio
async def test_switch_no_aria_label_flagged():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<div role="switch"></div>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "toggle_accessible_name" for i in issues)


# ---------------------------------------------------------------------------
# 11. Tooltip
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_tooltip_no_name():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<div role="tooltip"></div>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "tooltip_accessible_name" for i in issues)


# ---------------------------------------------------------------------------
# 12. Treeitem
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_treeitem_no_name():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<div role="tree"><div role="treeitem"></div></div>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "treeitem_accessible_name" for i in issues)


# ---------------------------------------------------------------------------
# 13. ARIA role valid
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_aria_invalid_role():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<div role="invalidrole"></div>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "aria_role_valid" and i.severity == "medium" for i in issues)


@pytest.mark.asyncio
async def test_aria_valid_role_no_issue():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<div role="button" aria-label="x" tabindex="0">x</div>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "aria_role_valid" for i in issues)


# ---------------------------------------------------------------------------
# 14. Deprecated ARIA role
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_deprecated_role():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<div role="directory">old</div>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "aria_deprecated" and i.severity == "low" for i in issues)


# ---------------------------------------------------------------------------
# 15. Required ARIA attributes
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_required_aria_missing():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<div role="checkbox" tabindex="0" aria-label="Check">x</div>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "aria_required_attr" for i in issues)
    # with required present, no issue
    html2 = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<div role="checkbox" aria-checked="false" tabindex="0" aria-label="Check">x</div>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client2 = _make_client({"https://example.com/": (200, html2)})
    issues2 = await analyze_accessibility("https://example.com/", client=client2)
    await client2.aclose()
    assert not any(i.check_id == "aria_required_attr" for i in issues2)


# ---------------------------------------------------------------------------
# 16. ARIA parent child
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_aria_parent_missing_child():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<div role="list"></div>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "aria_parent_child" for i in issues)


@pytest.mark.asyncio
async def test_aria_parent_with_child_no_issue():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<div role="list"><div role="listitem">item</div></div>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "aria_parent_child" for i in issues)


# ---------------------------------------------------------------------------
# 17. ARIA child parent
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_aria_child_missing_parent():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<div role="listitem">orphan</div>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "aria_child_parent" for i in issues)


# ---------------------------------------------------------------------------
# 18. ARIA IDs unique
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_duplicate_ids():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<div id="dup">a</div><div id="dup">b</div>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "aria_id_unique" for i in issues)


# ---------------------------------------------------------------------------
# 19. role=text no focusable descendants
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_role_text_with_focusable():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<div role="text"><button>Click</button></div>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "aria_text_no_focusable" for i in issues)


@pytest.mark.asyncio
async def test_role_text_without_focusable_no_issue():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<div role="text">plain text</div>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "aria_text_no_focusable" for i in issues)


# ---------------------------------------------------------------------------
# 20. No duplicate keyboard shortcuts
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_duplicate_accesskey():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<button accesskey="k">A</button><button accesskey="k">B</button>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "duplicate_keyboard_shortcut" for i in issues)


@pytest.mark.asyncio
async def test_unique_accesskey_no_issue():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<button accesskey="k">A</button><button accesskey="j">B</button>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "duplicate_keyboard_shortcut" for i in issues)


# ---------------------------------------------------------------------------
# 21. Skip link
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_skip_link_missing_flagged():
    html = """<!doctype html><html lang="en"><body>
<div>No landmarks here</div><h1>h</h1>
<button>Click</button><img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "skip_link" and i.severity == "low" for i in issues)


@pytest.mark.asyncio
async def test_skip_link_present_no_issue():
    html = """<!doctype html><html lang="en"><body>
<a href="#main">Skip to main</a><header></header><main><h1>h</h1></main><footer>f</footer>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "skip_link" for i in issues)


# ---------------------------------------------------------------------------
# 22. Main landmark
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_main_missing():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><a href="#main">Skip</a><h1>h</h1>
<div>content</div><footer>f</footer>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "landmark_main" for i in issues)


# ---------------------------------------------------------------------------
# 23. Heading content
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_heading_empty():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1></h1><h2>ok</h2></main><footer>f</footer>
<a href="#main">Skip</a><img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "heading_content" for i in issues)


# ---------------------------------------------------------------------------
# 24. Visible label mismatch
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_visible_label_mismatch():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<button aria-label="Close dialog">Submit</button>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "visible_label_match" for i in issues)


@pytest.mark.asyncio
async def test_visible_label_match_no_issue():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<button aria-label="Close dialog - Submit">Submit</button>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "visible_label_match" for i in issues)


# ---------------------------------------------------------------------------
# 25. Visual fatigue
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_visual_fatigue_flagged():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<div style="font-size:9px">tiny</div><div style="font-size:8px">tiny2</div><div style="font-size:7px">tiny3</div>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "visual_fatigue" and i.severity == "low" for i in issues)


# ---------------------------------------------------------------------------
# 26. Frame titles
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_frame_missing_title():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<iframe src="x.html"></iframe>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "frame_titles" for i in issues)


# ---------------------------------------------------------------------------
# 27. Duplicate labels
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_duplicate_labels_flagged():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<label for="a1">Email</label><input id="a1" type="text">
<label for="a2">Email</label><input id="a2" type="text">
<img src="x.jpg" alt="x">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "duplicate_labels" for i in issues)


# ---------------------------------------------------------------------------
# 28. Video captions
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_video_no_captions():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<video src="vid.mp4" controls></video>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "video_captions" for i in issues)


@pytest.mark.asyncio
async def test_video_with_captions_no_issue():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<video controls><track kind="captions" src="c.vtt"></video>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "video_captions" for i in issues)


# ---------------------------------------------------------------------------
# 29. Interactive keyboard focusable
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_interactive_not_focusable():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<div role="button" aria-label="x">x</div>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "interactive_keyboard_focusable" for i in issues)


# ---------------------------------------------------------------------------
# 30. Tab order
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_tabindex_positive():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<button tabindex="5">x</button>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "tab_order" and i.severity == "low" for i in issues)


# ---------------------------------------------------------------------------
# 31. Link distinguishable
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_link_color_only():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<a href="/x" style="color: blue;">Link</a>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "link_distinguishable" for i in issues)


# ---------------------------------------------------------------------------
# 32. Meta refresh
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_meta_refresh_flagged():
    html = """<!doctype html><html lang="en"><head><meta http-equiv="refresh" content="5;url=/"></head><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "meta_refresh" for i in issues)


# ---------------------------------------------------------------------------
# 33. Network error graceful
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_network_error_graceful():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("mock", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://example.com")
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert issues == []


@pytest.mark.asyncio
async def test_empty_url_raises():
    with pytest.raises(ValueError):
        await analyze_accessibility("", client=_make_client({}))


@pytest.mark.asyncio
async def test_import_without_side_effects():
    import importlib

    mod = importlib.import_module("ux_analyzer.analysis.accessibility")
    assert hasattr(mod, "analyze_accessibility")
    assert hasattr(mod, "analyze_accessibility_sync")
    assert hasattr(mod, "AccessibilityIssue")


@pytest.mark.asyncio
async def test_url_without_scheme():
    html = _perfect_html()
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("example.com", client=client)
    await client.aclose()
    assert issues == []


# ---------------------------------------------------------------------------
# 34. Visible label: aria-hidden filtering (monogram bug)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_visible_label_aria_hidden_filtered():
    # Decorative <span aria-hidden="true">M</span> should not be included in visible text
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<a href="#top" aria-label="Mohamed Khalil, home"><span aria-hidden="true">M</span><span>Mohamed Khalil</span></a>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a" type="text">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "visible_label_match" for i in issues), f"aria-hidden monogram should be filtered, got {[(i.check_id, i.evidence) for i in issues]}"


@pytest.mark.asyncio
async def test_visible_label_aria_hidden_nested_filtered():
    # Nested aria-hidden subtree with watch text should not cause mismatch
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<button aria-label="Play the Muslim Pedia promo with sound"><span aria-hidden="true"><span>Watch with sound</span></span></button>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a" type="text">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "visible_label_match" for i in issues)


@pytest.mark.asyncio
async def test_visible_label_theme_toggle_single_state():
    # Both Light/Dark spans concatenated should be treated as single state via CSS; pass if either matches aria-label
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<button aria-label="Switch to light theme"><span><span>Light theme</span><span>Dark theme</span></span></button>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a" type="text">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "visible_label_match" for i in issues), "theme toggle concatenated spans should not flag when one state matches"


@pytest.mark.asyncio
async def test_visible_label_symbolic_not_flagged():
    # Symbolic single-char like × should be skipped
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<button aria-label="Close video">\u00d7</button>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a" type="text">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "visible_label_match" for i in issues), "symbolic × should be ignored"


@pytest.mark.asyncio
async def test_visible_label_symbolic_entity_not_flagged():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<button aria-label="Close video">&times;</button>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a" type="text">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "visible_label_match" for i in issues)


# ---------------------------------------------------------------------------
# 35. Video captions: hidden / empty src filtering
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_video_hidden_dialog_not_flagged():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<div hidden role="dialog" aria-modal="true" aria-label="Promo"><video controls playsinline></video><button aria-label="Close video">\u00d7</button></div>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a" type="text">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "video_captions" for i in issues), "hidden dialog video with empty src should not be flagged"


@pytest.mark.asyncio
async def test_video_aria_hidden_not_flagged():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<video muted loop playsinline aria-hidden="true"></video>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a" type="text">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "video_captions" for i in issues)


@pytest.mark.asyncio
async def test_video_empty_src_not_flagged():
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<video src="" controls></video>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a" type="text">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert not any(i.check_id == "video_captions" for i in issues), "video with empty src=\"\" should be skipped"


@pytest.mark.asyncio
async def test_video_visible_with_src_still_flagged():
    # sanity: visible video with src and no captions should still flag
    html = """<!doctype html><html lang="en"><body>
<header><nav></nav></header><main><h1>h</h1><a href="#main">Skip</a>
<video src="vid.mp4" controls></video>
<img src="x.jpg" alt="x"><label for="a">L</label><input id="a" type="text">
</main><footer>f</footer></body></html>"""
    client = _make_client({"https://example.com/": (200, html)})
    issues = await analyze_accessibility("https://example.com/", client=client)
    await client.aclose()
    assert any(i.check_id == "video_captions" for i in issues)


def test_parser_filters_aria_hidden_text():
    from ux_analyzer.analysis.accessibility import _parse_html

    html = '<a aria-label="home"><span aria-hidden="true">M</span><span>Mohamed Khalil</span></a>'
    parser = _parse_html(html)
    # Find the <a> element
    a_el = next(e for e in parser.elements if e["tag"] == "a")
    assert a_el["text"] == "Mohamed Khalil", f"expected filtered text, got {a_el['text']!r}"
    assert "M Mohamed" not in a_el["text"]

