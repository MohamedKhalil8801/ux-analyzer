"""Pure unit tests for redesign page-capture resolution and bounds (Task 3)."""

import pytest

from ux_analyzer.analysis.page_capture import (
    CAPTURE_MODE_FULL_PAGE,
    CAPTURE_MODE_SCROLL,
    DEFAULT_CAPTURE_VIEWPORT,
    DEFAULT_MAX_PAGE_HEIGHT,
    MAX_INVENTORY_ENTRIES,
    MAX_SEGMENT_BYTES,
    MAX_TOTAL_COPY_CHARS,
    PAGE_CAPTURE_FILENAME,
    PAGE_CAPTURE_SCHEMA,
    SEGMENT_HEIGHT_PX,
    audit_capture_hook,
    build_capture_payload,
    build_scroll_capture_payload,
    max_page_height_from_env,
    normalize_capture_url,
    plan_page_segments,
    resolve_redesign_page_list,
)

# ---------------------------------------------------------------------------
# resolve_redesign_page_list
# ---------------------------------------------------------------------------


def _corpus(pages: list[tuple[str, int]], graph: dict[str, list[str]] | None = None):
    return {
        "pages": [{"url": url, "depth": depth} for url, depth in pages],
        "link_graph": graph or {},
    }


def test_resolve_uses_starts_when_corpus_missing_or_empty() -> None:
    starts = ("https://fixture.test/", "https://fixture.test/pricing")
    assert resolve_redesign_page_list(None, starts, cap=10) == starts
    assert resolve_redesign_page_list({"pages": [], "link_graph": {}}, starts, cap=10) == starts


def test_resolve_starts_first_then_bfs_discovery_order() -> None:
    corpus = _corpus(
        [
            ("https://fixture.test/deep", 2),
            ("https://fixture.test/about", 1),
            ("https://fixture.test/pricing", 1),
        ],
        {
            "https://fixture.test/": [
                "https://fixture.test/pricing",
                "https://fixture.test/about",
            ],
            "https://fixture.test/pricing": ["https://fixture.test/deep"],
        },
    )
    starts = ("https://fixture.test/",)
    resolved = resolve_redesign_page_list(corpus, starts, cap=10)
    assert resolved == (
        "https://fixture.test/",
        "https://fixture.test/pricing",
        "https://fixture.test/about",
        "https://fixture.test/deep",
    )


def test_resolve_corpus_beats_starts_semantics() -> None:
    # With a corpus, discovered pages extend the list beyond start URLs.
    corpus = _corpus([("https://fixture.test/discovered", 1)], {
        "https://fixture.test/": ["https://fixture.test/discovered"],
    })
    starts = ("https://fixture.test/",)
    resolved = resolve_redesign_page_list(corpus, starts, cap=10)
    assert resolved == ("https://fixture.test/", "https://fixture.test/discovered")


def test_resolve_dedupes_preserving_first_seen() -> None:
    corpus = _corpus(
        [("https://fixture.test/", 0), ("https://fixture.test/pricing", 1)],
        {"https://fixture.test/": ["https://fixture.test/pricing"]},
    )
    starts = ("https://fixture.test/", "https://fixture.test/pricing")
    resolved = resolve_redesign_page_list(corpus, starts, cap=10)
    assert resolved.count("https://fixture.test/pricing") == 1
    assert resolved[0] == "https://fixture.test/"


def test_resolve_caps_result() -> None:
    pages = [("https://fixture.test/", 0)] + [
        (f"https://fixture.test/page-{i}", 1) for i in range(20)
    ]
    graph = {"https://fixture.test/": [f"https://fixture.test/page-{i}" for i in range(20)]}
    resolved = resolve_redesign_page_list(_corpus(pages, graph), ("https://fixture.test/",), cap=10)
    assert len(resolved) == 10
    assert resolved[0] == "https://fixture.test/"


def test_resolve_depth0_corpus_reaches_only_starts() -> None:
    # Depth 0: corpus contains exactly the start URLs (in a different order);
    # resolution must return the starts, not reorder them.
    corpus = _corpus(
        [
            ("https://fixture.test/pricing", 0),
            ("https://fixture.test/", 0),
        ],
    )
    starts = ("https://fixture.test/", "https://fixture.test/pricing")
    assert resolve_redesign_page_list(corpus, starts, cap=10) == starts


def test_resolve_normalizes_fragments_and_dedupes() -> None:
    corpus = _corpus([("https://fixture.test/pricing#plans", 1)])
    starts = ("https://fixture.test/",)
    resolved = resolve_redesign_page_list(corpus, starts, cap=10)
    assert resolved == ("https://fixture.test/", "https://fixture.test/pricing")


def test_resolve_ignores_offsite_targets() -> None:
    corpus = _corpus([("https://fixture.test/local", 1)], {
        "https://fixture.test/": ["https://other.test/nope", "https://fixture.test/local"],
    })
    resolved = resolve_redesign_page_list(corpus, ("https://fixture.test/",), cap=10)
    assert "https://other.test/nope" not in resolved


def test_normalize_capture_url_strips_fragments() -> None:
    assert normalize_capture_url("https://Fixture.test/a#top") == "https://fixture.test/a"
    with pytest.raises(ValueError):
        normalize_capture_url("not-a-url")


# ---------------------------------------------------------------------------
# plan_page_segments
# ---------------------------------------------------------------------------


def test_segment_plan_covers_document_in_2000px_steps() -> None:
    plan = plan_page_segments(5000)
    assert plan.offsets == ((0, 2000), (2000, 2000), (4000, 1000))
    assert plan.truncated is False


def test_segment_plan_truncates_beyond_cap_and_records_it() -> None:
    plan = plan_page_segments(15000, max_page_height=DEFAULT_MAX_PAGE_HEIGHT)
    assert plan.truncated is True
    covered = sum(height for _, height in plan.offsets)
    assert covered == DEFAULT_MAX_PAGE_HEIGHT
    assert all(height <= SEGMENT_HEIGHT_PX for _, height in plan.offsets)


def test_segment_plan_exact_multiple_and_zero_height() -> None:
    assert plan_page_segments(4000).offsets == ((0, 2000), (2000, 2000))
    empty = plan_page_segments(0)
    assert empty.offsets == ()
    assert empty.truncated is False


def test_bounds_constants_match_plan() -> None:
    assert SEGMENT_HEIGHT_PX == 2000
    assert MAX_SEGMENT_BYTES == 640 * 1024
    assert DEFAULT_MAX_PAGE_HEIGHT == 12_000
    assert MAX_INVENTORY_ENTRIES == 1500
    assert MAX_TOTAL_COPY_CHARS == 200_000
    assert PAGE_CAPTURE_FILENAME == "page-capture.json"
    assert PAGE_CAPTURE_SCHEMA == "page-capture-v3"


def test_max_page_height_env_non_positive_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UXA_REDESIGN_MAX_PAGE_HEIGHT", raising=False)
    assert max_page_height_from_env() == DEFAULT_MAX_PAGE_HEIGHT
    monkeypatch.setenv("UXA_REDESIGN_MAX_PAGE_HEIGHT", "0")
    assert max_page_height_from_env() == DEFAULT_MAX_PAGE_HEIGHT
    monkeypatch.setenv("UXA_REDESIGN_MAX_PAGE_HEIGHT", "bogus")
    assert max_page_height_from_env() == DEFAULT_MAX_PAGE_HEIGHT
    monkeypatch.setenv("UXA_REDESIGN_MAX_PAGE_HEIGHT", "8000")
    assert max_page_height_from_env() == 8000


# ---------------------------------------------------------------------------
# build_capture_payload (shared-pass payload assembly)
# ---------------------------------------------------------------------------


def _png_bytes(width: int = 40, height: int = 3000) -> bytes:
    import io as _io

    from PIL import Image

    buffer = _io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_build_capture_payload_records_viewport_and_timestamp() -> None:
    payload = build_capture_payload(
        "https://fixture.test/",
        _png_bytes(height=1000),
        document_height=1000,
        max_page_height=12_000,
        viewport={"width": 1440, "height": 900},
    )
    assert payload["viewport"] == {"width": 1440, "height": 900}
    assert payload["captured_at"]  # never empty
    defaulted = build_capture_payload(
        "https://fixture.test/",
        _png_bytes(height=1000),
        document_height=1000,
        max_page_height=12_000,
    )
    assert defaulted["viewport"] == DEFAULT_CAPTURE_VIEWPORT
    assert defaulted["captured_at"]


def test_page_viewport_helper_reads_session_viewport() -> None:
    from ux_analyzer.analysis.page_capture import _page_viewport

    class _Page:
        viewport_size = {"width": 1440, "height": 900}

    assert _page_viewport(_Page()) == {"width": 1440, "height": 900}
    assert _page_viewport(object()) is None
    assert _page_viewport(None) is None


def test_build_capture_payload_shapes_document_and_segments() -> None:
    payload = build_capture_payload(
        "https://fixture.test/",
        _png_bytes(),
        document_height=3000,
        title="Fixture",
        max_page_height=12_000,
        inventory={"sections": [{"tag": "header"}], "copyTruncated": True},
    )
    assert payload["schema"] == PAGE_CAPTURE_SCHEMA
    assert payload["url"] == "https://fixture.test/"
    assert payload["title"] == "Fixture"
    assert payload["copy_truncated"] is True
    assert payload["inventory_truncated"] is False
    segments = payload["segments"]
    assert isinstance(segments, list) and len(segments) == 2
    first = segments[0]
    assert first["index"] == 0 and first["y_offset"] == 0
    assert first["height"] == 2000
    assert str(first["data_url"]).startswith("data:image/jpeg;base64,")
    for segment in segments:
        assert len(str(segment["data_url"])) <= MAX_SEGMENT_BYTES
    assert payload["sections"] == [{"tag": "header"}]
    assert payload["paragraphs"] == []


def test_build_capture_payload_records_truncation_beyond_cap() -> None:
    payload = build_capture_payload(
        "https://fixture.test/",
        _png_bytes(height=5000),
        document_height=25_000,
        max_page_height=4_000,
    )
    assert payload["truncated"] is True
    assert payload["document_height"] == 25_000
    assert payload["captured_height"] == 4_000
    covered = sum(int(segment["height"]) for segment in payload["segments"])
    assert covered == 4_000


# ---------------------------------------------------------------------------
# audit_capture_hook (shared browser pass plumbing)
# ---------------------------------------------------------------------------


class _FakePage:
    """Minimal page double: the hook only calls ``evaluate`` for inventory."""

    def __init__(self, inventory: dict[str, object] | None = None) -> None:
        self._inventory = inventory if inventory is not None else {}
        self.calls = 0

    def evaluate(self, *args: object, **kwargs: object) -> dict[str, object]:
        self.calls += 1
        return self._inventory


def _materials(page: object) -> object:
    from ux_analyzer.analysis.project_audit import CaptureMaterials

    return CaptureMaterials(
        url="https://fixture.test/",
        png_bytes=_png_bytes(height=2000),
        title="Shared pass",
        document_height=2000,
        page=page,
    )


def test_build_capture_payload_records_default_capture_mode() -> None:
    payload = build_capture_payload(
        "https://fixture.test/",
        _png_bytes(height=1000),
        document_height=1000,
        max_page_height=12_000,
    )
    assert payload["capture_mode"] == CAPTURE_MODE_FULL_PAGE


def test_build_scroll_capture_payload_tiles_viewport_stepped_segments() -> None:
    payload = build_scroll_capture_payload(
        "https://fixture.test/",
        [_png_bytes(width=40, height=300), _png_bytes(width=40, height=300)],
        document_height=600,
        title="Scroll fixture",
        max_page_height=12_000,
        viewport={"width": 40, "height": 300},
        inventory={"buttons": [{"tag": "button"}]},
    )
    assert payload["schema"] == PAGE_CAPTURE_SCHEMA
    assert payload["capture_mode"] == CAPTURE_MODE_SCROLL
    assert payload["document_height"] == 600
    assert payload["captured_height"] == 600
    assert payload["truncated"] is False
    segments = payload["segments"]
    assert [int(segment["index"]) for segment in segments] == [0, 1]
    assert [int(segment["y_offset"]) for segment in segments] == [0, 300]
    assert [int(segment["height"]) for segment in segments] == [300, 300]
    for segment in segments:
        assert str(segment["data_url"]).startswith("data:image/jpeg;base64,")
    assert payload["buttons"] == [{"tag": "button"}]


def test_scroll_capture_payload_preserves_effective_tap_boxes() -> None:
    """Interactive inventory entries keep their capture-time ``tap_box``
    (the effective tappable surface measured against the closest
    tap-reacting ancestor); payload assembly must not drop it."""

    entry = {
        "kind": "button",
        "tag": "button",
        "label": "Play",
        "box": {"x": 20, "y": 20, "w": 30, "h": 30},
        "tap_box": {"x": 0, "y": 0, "w": 320, "h": 240},
    }
    payload = build_scroll_capture_payload(
        "https://fixture.test/",
        [_png_bytes(width=40, height=300)],
        document_height=300,
        title="Tap fixture",
        max_page_height=12_000,
        viewport={"width": 40, "height": 300},
        inventory={"buttons": [entry], "links": [], "inputs": []},
    )
    assert payload["buttons"] == [entry]


def test_capture_reports_effective_tap_boxes_freshness() -> None:
    """A sidecar page is redesign-fresh only when every interactive control
    carries ``tap_box``; older captures lack it and are re-captured."""

    from ux_analyzer.analysis.page_capture import (
        capture_reports_effective_tap_boxes,
    )

    assert capture_reports_effective_tap_boxes(
        {"buttons": [], "links": [], "inputs": []}
    )
    assert capture_reports_effective_tap_boxes(
        {"buttons": [{"tap_box": {}}], "links": [], "inputs": []}
    )
    assert not capture_reports_effective_tap_boxes(
        {"buttons": [{}], "links": [], "inputs": []}
    )
    assert not capture_reports_effective_tap_boxes(
        {"buttons": [], "links": [{"tap_box": {}}], "inputs": [{}]}
    )


def test_build_scroll_capture_payload_honors_viewport_step_and_truncation() -> None:
    payload = build_scroll_capture_payload(
        "https://fixture.test/",
        [_png_bytes(width=40, height=300), _png_bytes(width=40, height=300)],
        document_height=2000,
        max_page_height=500,
        viewport={"width": 40, "height": 300},
    )
    assert payload["truncated"] is True
    covered = sum(int(segment["height"]) for segment in payload["segments"])
    assert covered == 500
    assert len(payload["segments"]) == 2


def test_audit_capture_hook_builds_payload_from_session_materials() -> None:
    page = _FakePage({"headings": ["H1"], "truncated": True})
    sink: dict[str, dict[str, object]] = {}
    hook = audit_capture_hook(sink, max_page_height=12_000)
    hook("https://fixture.test/", _materials(page))
    entry = sink["https://fixture.test/"]
    assert entry["status"] == "captured"
    payload = entry["payload"]
    assert isinstance(payload, dict)
    assert payload["schema"] == PAGE_CAPTURE_SCHEMA
    assert payload["title"] == "Shared pass"
    assert payload["headings"] == ["H1"]
    assert payload["inventory_truncated"] is True
    assert page.calls == 1  # inventory ran in the shared session


def test_audit_capture_hook_records_error_without_raising() -> None:
    sink: dict[str, dict[str, object]] = {}
    hook = audit_capture_hook(sink)
    hook("https://fixture.test/", object())  # no CaptureMaterials at all
    entry = sink["https://fixture.test/"]
    assert entry["status"] == "error"
    assert "capture materials" in str(entry["reason"])


def test_audit_capture_hook_survives_inventory_failure() -> None:
    class _BrokenPage:
        def evaluate(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("page closed")

    sink: dict[str, dict[str, object]] = {}
    hook = audit_capture_hook(sink, max_page_height=1_000)
    hook("https://fixture.test/", _materials(_BrokenPage()))
    entry = sink["https://fixture.test/"]
    assert entry["status"] == "captured"
    payload = entry["payload"]
    assert isinstance(payload, dict)
    assert payload["sections"] == []  # inventory absent, segments still bounded
    assert isinstance(payload["segments"], list) and payload["segments"]
