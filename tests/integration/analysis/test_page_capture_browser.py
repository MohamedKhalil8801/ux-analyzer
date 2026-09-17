"""Live browser capture integration test (Task 3 Step 3).

Serves the bundled fixture app on a loopback port and asserts the persisted
``page-capture.json`` sidecar contract: schema, bounded segments, bounded
inventory, truncation flags.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

import pytest
import uvicorn

from ux_analyzer.analysis.page_capture import (
    MAX_INVENTORY_ENTRIES,
    MAX_SEGMENT_BYTES,
    PAGE_CAPTURE_SCHEMA,
    audit_capture_hook,
    capture_reports_effective_tap_boxes,
    load_page_capture,
    plan_page_segments,
    resolve_redesign_page_list,
)

_fixture_module = pytest.importorskip("fixture_app.app")
fixture_app = _fixture_module.app


@pytest.fixture
def fixture_origin() -> Iterator[str]:
    from tests.e2e.test_demo_benchmark import _free_port

    server = uvicorn.Server(
        uvicorn.Config(
            fixture_app, host="127.0.0.1", port=_free_port(), log_level="error"
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        import time

        import httpx

        origin = f"http://127.0.0.1:{server.config.port}"
        with httpx.Client() as client:
            for _ in range(200):
                try:
                    response = client.get(f"{origin}/app/ready/improved")
                    if response.status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.1)
            else:
                pytest.fail("fixture server did not become ready")
        yield origin
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.mark.e2e
def test_capture_page_persists_bounded_payload(fixture_origin: str) -> None:
    from ux_analyzer.analysis.page_capture import capture_page

    url = f"{fixture_origin}/app/ready/improved"
    payload = capture_page(url)

    assert payload["schema"] == PAGE_CAPTURE_SCHEMA
    assert payload["url"] == url
    assert isinstance(payload["segments"], list) and payload["segments"]
    for segment in payload["segments"]:
        assert segment["height"] <= 2000
        assert len(segment["data_url"]) <= MAX_SEGMENT_BYTES
        assert segment["data_url"].startswith("data:image/jpeg;base64,")
    inventory_lists = ("sections", "headings", "forms", "buttons", "inputs", "links")
    for key in inventory_lists:
        assert isinstance(payload[key], list)
        assert len(payload[key]) <= MAX_INVENTORY_ENTRIES
    assert isinstance(payload["paragraphs"], list)
    assert payload["document_height"] >= payload["captured_height"]
    assert isinstance(payload["truncated"], bool)


@pytest.mark.e2e
def test_capture_page_scroll_mode_steps_segments_by_viewport(
    fixture_origin: str,
) -> None:
    """``capture_page`` defaults to scroll-position capture: segments step
    by the live viewport height and the payload says so explicitly."""

    from ux_analyzer.analysis.page_capture import (
        CAPTURE_MODE_SCROLL,
        capture_page,
    )

    url = f"{fixture_origin}/app/ready/improved"
    payload = capture_page(url)

    assert payload["capture_mode"] == CAPTURE_MODE_SCROLL
    viewport = payload["viewport"]
    assert isinstance(viewport, dict)
    step = int(viewport.get("height", 0))
    assert step > 0
    segments = payload["segments"]
    assert isinstance(segments, list) and segments
    offsets = [int(segment["y_offset"]) for segment in segments]  # type: ignore[union-attr]
    assert offsets[0] == 0
    # Full steps tile the page; only the last segment may be shorter.
    for previous, current in zip(offsets, offsets[1:]):
        assert current - previous == step
    for segment in segments:
        assert int(segment["height"]) <= step  # type: ignore[union-attr]

    assert int(segments[-1]["y_offset"]) + int(segments[-1]["height"]) == int(  # type: ignore[union-attr]
        payload["captured_height"]
    )


@pytest.mark.e2e
def test_capture_page_max_height_truncates_explicitly(fixture_origin: str) -> None:
    from ux_analyzer.analysis.page_capture import capture_page

    url = f"{fixture_origin}/app/ready/improved"
    payload = capture_page(url, max_page_height=800)
    captured = sum(int(s["height"]) for s in payload["segments"])  # type: ignore[union-attr]
    assert captured <= 800
    if payload["document_height"] and int(payload["document_height"]) > 800:  # type: ignore[arg-type]
        assert payload["truncated"] is True


@pytest.mark.e2e
def test_capture_inventory_records_effective_tap_targets(
    fixture_origin: str,
) -> None:
    """Interactive inventory entries carry ``tap_box``: the control itself
    for a standalone button, but the closest tap-reacting ancestor (the
    clickable card) for controls nested inside it -- measured against the
    real ancestor, not the widget's own painted box."""

    from ux_analyzer.analysis.page_capture import capture_page

    url = f"{fixture_origin}/app/ready/improved/tap-targets"
    payload = capture_page(url)
    by_label = {str(e.get("label", "")): e for e in payload["buttons"]}
    solo = by_label.get("Solo button")
    child = by_label.get("Child button")
    assert solo is not None and child is not None

    solo_box = solo["tap_box"]
    # Roughly the button's own size (UA default padding included), and
    # definitely NOT inflated to any enclosing surface.
    assert 60 <= int(solo_box["w"]) <= 140
    assert 20 <= int(solo_box["h"]) <= 60

    # The card (320x240, handler bound via addEventListener) is the child
    # button's effective tap surface.
    child_box = child["tap_box"]
    assert int(child_box["w"]) >= 300
    assert int(child_box["h"]) >= 220
    # ... while the button's own box stays the small widget.
    own = child["box"]
    assert int(own["w"]) <= 100
    assert int(own["h"]) <= 50
    assert capture_reports_effective_tap_boxes(payload)


@pytest.mark.e2e
def test_capture_list_resolution_breadth_with_starts() -> None:
    starts = ("https://127.0.0.1:9000/app/ready/improved",)
    resolved = resolve_redesign_page_list(None, starts, cap=5)
    assert resolved == starts
    empty_plan = plan_page_segments(0)
    assert empty_plan.offsets == ()


@pytest.mark.e2e
def test_audit_pass_persists_bounded_sidecar_document(
    tmp_path: Any, fixture_origin: str
) -> None:
    """One audit pass per page produces the shared page-capture.json sidecar."""

    from ux_analyzer.analysis.project_audit import audit_urls_sync

    url = f"{fixture_origin}/app/ready/improved"
    sink: dict[str, dict[str, object]] = {}
    report = audit_urls_sync(
        (url,), capture_hook=audit_capture_hook(sink)
    )
    assert report["total_issues"] >= 0
    entry = sink[url]
    assert entry["status"] == "captured"
    payload = entry["payload"]
    assert isinstance(payload, dict)
    assert payload["schema"] == PAGE_CAPTURE_SCHEMA
    assert payload["url"] == url
    assert payload["title"]
    segments = payload["segments"]
    assert isinstance(segments, list) and segments
    for segment in segments:
        assert int(segment["height"]) <= 2000
        assert len(str(segment["data_url"])) <= MAX_SEGMENT_BYTES
    headings = payload["headings"]
    assert isinstance(headings, list) and headings  # fixture has real headings
    sections = payload["sections"]
    assert isinstance(sections, list) and sections
    buttons = payload["buttons"]
    assert isinstance(buttons, list) and buttons

    from ux_analyzer.analysis.page_capture import write_page_capture

    document = {
        "schema": PAGE_CAPTURE_SCHEMA,
        "pages": [payload],
    }
    destination = write_page_capture(tmp_path, document)
    assert destination.name == "page-capture.json"
    loaded = load_page_capture(tmp_path)
    assert isinstance(loaded, dict)
    assert loaded.get("schema") == PAGE_CAPTURE_SCHEMA
    pages = loaded.get("pages")
    assert isinstance(pages, list) and len(pages) == 1


@pytest.mark.e2e
def test_capture_includes_description_list_group_titles() -> None:
    """<dt> group titles are persona-visible text and must reach the
    paragraph inventory.

    Regression for the redesign false positive "Group Skills into Clear
    Categories": the live portfolio groups skills inside a
    ``<dl class="skills__grid">`` whose category titles are ``<dt>`` elements.
    The text inventory previously collected only ``p`` and ``li`` nodes, so
    the group titles vanished and the model saw a flat, comma-separated skill
    list -- concluding a grouped list was ungrouped.
    """

    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse
    from starlette.routing import Route

    from tests.e2e.test_demo_benchmark import _free_port
    from ux_analyzer.analysis.page_capture import capture_page

    skills_html = """<!doctype html>
<html><head><meta charset="utf-8"><title>Skills</title></head>
<body>
<h1>Profile</h1>
<section aria-labelledby="skills-h">
  <h2 id="skills-h">Breadth, with depth where it counts.</h2>
  <dl class="skills__grid">
    <dt>Languages &amp; Frameworks</dt>
    <dd><ul><li>Flutter</li><li>Kotlin</li></ul></dd>
    <dt>Advanced Topics</dt>
    <dd><ul><li>Recommender Systems</li><li>Parallel Processing</li></ul></dd>
    <p class="note">A paragraph after the list.</p>
  </dl>
</section>
</body></html>"""

    def skills_page(_request: object) -> HTMLResponse:
        return HTMLResponse(skills_html)

    app = Starlette(routes=[Route("/skills-list", skills_page)])
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=_free_port(), log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        origin = f"http://127.0.0.1:{server.config.port}"
        payload = capture_page(f"{origin}/skills-list")
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    paragraphs = [str(p) for p in payload["paragraphs"]]
    for expected in (
        "Languages & Frameworks",
        "Advanced Topics",
        "Flutter",
        "Kotlin",
        "Recommender Systems",
        "A paragraph after the list.",
    ):
        assert expected in paragraphs, f"missing {expected!r} in {paragraphs}"
    # Group title must precede its chips, matching visual reading order.
    assert paragraphs.index("Languages & Frameworks") < paragraphs.index("Flutter")
