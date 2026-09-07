from __future__ import annotations

import hashlib
from typing import Any

import pytest

from ux_analyzer.application.exploration_crawler import (
    ExplorationCrawler,
    PageSettlementPolicy,
    _merge_visible_labels,
)
from ux_analyzer.domain.benchmark import normalize_crawl_url
from ux_analyzer.domain.exploration import ExplorationSpec

# ---------------------------------------------------------------------------
# Fake Page infrastructure (Page abstraction for unit tests)
# ---------------------------------------------------------------------------


class FakeLocator:
    def __init__(self, should_timeout: bool = False) -> None:
        self.should_timeout = should_timeout

    async def wait_for(self, state: str = "detached", timeout: int = 2000) -> None:
        if self.should_timeout:
            raise TimeoutError("locator detach timeout")
        return None


class FakePage:
    """Minimal Page-like fake supporting settlement + extraction contracts."""

    def __init__(
        self,
        url_map: dict[str, dict[str, Any]],
        *,
        networkidle_fail_first: bool = False,
        locator_should_timeout: bool = False,
        reveal_thresholds: tuple[float, ...] = (),
        viewport_height: int = 800,
    ) -> None:
        # url_map keys are normalized URLs; values: {title, headings, links, height_seq}
        self.url_map = url_map
        self.networkidle_fail_first = networkidle_fail_first
        self.locator_should_timeout = locator_should_timeout
        # Scroll-position-triggered reveals: opacity flips from 0 -> 1 when
        # scrollY passes each threshold (simulates GSAP ScrollTrigger).
        self.reveal_thresholds = reveal_thresholds
        self.viewport_height = viewport_height
        self.reveal_events: list[float] = []
        self.current_url: str | None = None
        self.current_data: dict[str, Any] = {}
        self.goto_calls: list[str] = []
        self.wait_for_load_state_calls: list[tuple[str, int | None]] = []
        self.wait_for_timeout_calls: list[int] = []
        self.locator_calls: list[str] = []
        self.evaluate_calls: list[str] = []
        self.networkidle_count = 0
        self.scroll_height_idx = 0
        self.height_seq: list[int] = [1000]
        self.scroll_by_count = 0
        self.scroll_y = 0.0

    def _next_height(self) -> float:
        idx = min(self.scroll_height_idx, len(self.height_seq) - 1)
        val = self.height_seq[idx]
        self.scroll_height_idx += 1
        return val

    def _peek_height(self) -> float:
        idx = min(self.scroll_height_idx, len(self.height_seq) - 1)
        return self.height_seq[idx]

    def _apply_scroll(self, delta: float) -> None:
        sh = self._peek_height()
        max_y = max(0.0, sh - self.viewport_height)
        self.scroll_y = min(max(0.0, self.scroll_y + delta), max_y)
        for threshold in self.reveal_thresholds:
            if threshold not in self.reveal_events and self.scroll_y >= threshold:
                self.reveal_events.append(threshold)

    async def goto(self, url: str, wait_until: str = "domcontentloaded", timeout: int = 15000) -> None:
        self.goto_calls.append(url)
        # Normalize lookup tolerant
        try:
            n = normalize_crawl_url(url)
        except ValueError:
            n = url
        self.current_url = url
        # lookup by normalized, then by raw
        self.current_data = self.url_map.get(n, self.url_map.get(url, {}))
        # reset height seq per navigation
        seq = self.current_data.get("height_seq")
        if isinstance(seq, list) and seq:
            self.height_seq = list(seq)
        else:
            self.height_seq = [1000]
        self.scroll_height_idx = 0
        self.scroll_by_count = 0
        self.scroll_y = 0.0
        self.reveal_events = []

    async def wait_for_load_state(self, state: str, timeout: int | None = None) -> None:
        self.wait_for_load_state_calls.append((state, timeout))
        if state == "networkidle":
            self.networkidle_count += 1
            if self.networkidle_fail_first and self.networkidle_count == 1:
                raise TimeoutError("networkidle timeout (simulated polling)")
        return None

    async def wait_for_timeout(self, ms: int) -> None:
        self.wait_for_timeout_calls.append(ms)
        return None

    def locator(self, selector: str) -> FakeLocator:
        self.locator_calls.append(selector)
        return FakeLocator(should_timeout=self.locator_should_timeout)

    async def evaluate(self, script: str) -> Any:
        self.evaluate_calls.append(script)
        # Combined scroll metrics payload used by the bottom-detection sweep.
        if "window.scrollY" in script and "innerHeight" in script:
            return {
                "y": self.scroll_y,
                "vh": self.viewport_height,
                "sh": self._next_height(),
            }
        if "document.body.scrollHeight" in script:
            return self._next_height()
        if "window.scrollBy" in script:
            self.scroll_by_count += 1
            self._apply_scroll(float(self.viewport_height) * 0.8)
            return None
        if "window.scrollTo" in script:
            self.scroll_y = 0.0
            return None
        if "document.title" in script:
            return self.current_data.get("title", "Fake Title")
        if "Array.from(document.querySelectorAll('a[href]'))" in script:
            return self.current_data.get("links", [])
        if "querySelectorAll('a[href], button" in script:
            # Position-aware visible-label fallback used by the scroll sweep.
            seq = self.current_data.get("scroll_labels_seq") or []
            if seq:
                if self.scroll_y <= 0:
                    return list(seq[0])
                idx = min(max(1, self.scroll_by_count), len(seq) - 1)
                return list(seq[idx])
            return list(self.current_data.get("scroll_labels", []))
        if "querySelectorAll('h1,h2,h3')" in script or "h1,h2,h3" in script:
            return self.current_data.get("headings", [])
        # For extractor EVALUATION_PAYLOAD (contains viewport) – return minimal payload that will fail
        # but crawler's capture fallback will handle via title/headings path; so return something inert.
        if "viewport" in script and "regions" in script:
            return {"viewport": {"width": 1280, "height": 800}, "regions": [], "elements": []}
        return None

    async def title(self) -> str:
        return str(self.current_data.get("title", "Fake Title"))

    async def screenshot(self, type: str = "png") -> bytes:  # type: ignore[override]
        return b"fake-png-bytes"


class SmoothScrollFakePage(FakePage):
    """FakePage whose scrollTo defers the actual position change.

    Simulates smooth-scrolling sites (Lenis / ``scroll-behavior: smooth``):
    ``window.scrollTo(...)`` returns immediately while scrollY keeps gliding
    toward the target over subsequent read ticks. Only once the glide
    completes does a position read report the target (0).
    """

    def __init__(
        self,
        url_map: dict[str, dict[str, Any]],
        *,
        glide_steps: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(url_map, **kwargs)
        self.glide_steps = glide_steps
        self.pending_glide_steps = 0
        self.scroll_top_poll_reads = 0

    async def evaluate(self, script: str) -> Any:
        self.evaluate_calls.append(script)
        if "window.scrollTo" in script:
            # Return immediately; position unchanged — glide starts now.
            self.pending_glide_steps = self.glide_steps
            return self.scroll_y
        if script.strip() == "() => window.scrollY":
            self.scroll_top_poll_reads += 1
            if self.pending_glide_steps > 0:
                self.pending_glide_steps -= 1
                if self.pending_glide_steps == 0:
                    self.scroll_y = 0.0
                else:
                    self.scroll_y = self.scroll_y / 2.0
            return self.scroll_y
        return await super().evaluate(script)


def _factory_for(page: FakePage):
    async def factory():
        return page

    return factory


def _capture_fn_maker(url_map: dict[str, dict[str, Any]]):
    async def capture_fn(page: Any, viewport_id: str) -> dict[str, Any]:
        # Derive title/headings from current_data of page
        data = getattr(page, "current_data", {})
        title = data.get("title", "Untitled")
        headings = tuple(data.get("headings", ()))
        # screenshot digest from fake bytes
        digest = hashlib.sha256(b"fake-png-bytes").hexdigest()
        return {
            "title": title,
            "headings": headings,
            "screenshot_digest": digest,
            "visible_elements": list(data.get("top_labels", [])),
        }

    return capture_fn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_depth_zero_crawls_only_starts() -> None:
    url_map = {
        normalize_crawl_url("https://a.test/"): {
            "title": "Home",
            "headings": ["Welcome"],
            "links": ["https://a.test/about", "https://a.test/contact"],
            "height_seq": [1000],
        }
    }
    page = FakePage(url_map)
    spec = ExplorationSpec(start_urls=("https://a.test/",), depth=0, max_pages=10)
    corpus = await ExplorationCrawler(page=page, capture_fn=_capture_fn_maker(url_map)).crawl(spec)
    assert len(corpus.pages) == 1
    assert corpus.pages[0].depth == 0
    assert corpus.pages[0].url == normalize_crawl_url("https://a.test/")
    # No expansion despite links present
    assert len(corpus.pages) == 1


@pytest.mark.asyncio
async def test_same_origin_drops_cross_origin() -> None:
    url_map = {
        normalize_crawl_url("https://a.test/"): {
            "title": "Home",
            "links": ["https://a.test/about", "https://evil.test/x", "https://a.test/contact"],
        }
    }
    page = FakePage(url_map)
    spec = ExplorationSpec(start_urls=("https://a.test/",), depth=1, max_pages=10)
    corpus = await ExplorationCrawler(page=page, capture_fn=_capture_fn_maker(url_map)).crawl(spec)
    urls = {p.url for p in corpus.pages}
    normals = {p.normalized_url for p in corpus.pages}
    assert "https://evil.test/x" not in urls
    assert normalize_crawl_url("https://evil.test/x") not in normals
    # a.test children should be visited
    assert normalize_crawl_url("https://a.test/about") in normals


@pytest.mark.asyncio
async def test_dedup_removes_fragment_and_sorted_query() -> None:
    # Page A links to two variants that normalize to same URL
    url_map = {
        normalize_crawl_url("https://a.test/"): {
            "title": "Home",
            "links": [
                "https://a.test/p?b=2&a=1#frag",
                "https://a.test/p?a=1&b=2",
                "https://a.test/p?a=1&b=2#other",
            ],
        },
        normalize_crawl_url("https://a.test/p?a=1&b=2"): {
            "title": "P page",
            "links": [],
        },
    }
    page = FakePage(url_map)
    spec = ExplorationSpec(start_urls=("https://a.test/",), depth=1, max_pages=10)
    corpus = await ExplorationCrawler(page=page, capture_fn=_capture_fn_maker(url_map)).crawl(spec)
    # start + one unique child
    assert len(corpus.pages) == 2
    normals = [p.normalized_url for p in corpus.pages]
    assert normals.count(normalize_crawl_url("https://a.test/p?a=1&b=2")) == 1


@pytest.mark.asyncio
async def test_max_pages_caps_bfs() -> None:
    links = [f"https://a.test/page{i}" for i in range(10)]
    url_map: dict[str, dict[str, Any]] = {
        normalize_crawl_url("https://a.test/"): {"title": "Home", "links": links}
    }
    for i in range(10):
        url_map[normalize_crawl_url(f"https://a.test/page{i}")] = {"title": f"P{i}", "links": []}
    page = FakePage(url_map)
    spec = ExplorationSpec(start_urls=("https://a.test/",), depth=5, max_pages=3)
    corpus = await ExplorationCrawler(page=page, capture_fn=_capture_fn_maker(url_map)).crawl(spec)
    assert len(corpus.pages) <= 3
    assert len(corpus.pages) == 3


@pytest.mark.asyncio
async def test_bfs_order_guarantee() -> None:
    # Home links to b,c,d in order; each child has no further links.
    # BFS should visit in link order.
    links = ["https://a.test/b", "https://a.test/c", "https://a.test/d"]
    url_map = {
        normalize_crawl_url("https://a.test/"): {"title": "Home", "links": links},
        normalize_crawl_url("https://a.test/b"): {"title": "B", "links": []},
        normalize_crawl_url("https://a.test/c"): {"title": "C", "links": []},
        normalize_crawl_url("https://a.test/d"): {"title": "D", "links": []},
    }
    page = FakePage(url_map)
    spec = ExplorationSpec(start_urls=("https://a.test/",), depth=1, max_pages=10)
    corpus = await ExplorationCrawler(page=page, capture_fn=_capture_fn_maker(url_map)).crawl(spec)
    # pages[0] is start, subsequent should be b,c,d in order
    assert corpus.pages[0].normalized_url == normalize_crawl_url("https://a.test/")
    assert [p.normalized_url for p in corpus.pages[1:]] == [
        normalize_crawl_url("https://a.test/b"),
        normalize_crawl_url("https://a.test/c"),
        normalize_crawl_url("https://a.test/d"),
    ]


@pytest.mark.asyncio
async def test_cycle_visited_once() -> None:
    # A -> B -> A cycle
    url_map = {
        normalize_crawl_url("https://a.test/a"): {
            "title": "A",
            "links": ["https://a.test/b"],
        },
        normalize_crawl_url("https://a.test/b"): {
            "title": "B",
            "links": ["https://a.test/a"],
        },
    }
    page = FakePage(url_map)
    spec = ExplorationSpec(start_urls=("https://a.test/a",), depth=5, max_pages=10)
    corpus = await ExplorationCrawler(page=page, capture_fn=_capture_fn_maker(url_map)).crawl(spec)
    assert len(corpus.pages) == 2
    assert {p.normalized_url for p in corpus.pages} == {
        normalize_crawl_url("https://a.test/a"),
        normalize_crawl_url("https://a.test/b"),
    }


@pytest.mark.asyncio
async def test_scroll_sweep_collects_below_fold_labels_into_page_evidence() -> None:
    """The scroll sweep records below-fold labels for synthesis evidence."""

    url_map: dict[str, dict[str, Any]] = {
        normalize_crawl_url("https://a.test/"): {
            "title": "Long page",
            "links": [],
            "height_seq": [3000],
            "top_labels": ["Hero A", "Hero B"],
            "scroll_labels_seq": [
                ["Hero A", "Hero B"],
                ["Below-fold X", "Below-fold X2"],
                ["Below-fold Y"],
            ],
        }
    }
    page = FakePage(url_map)
    spec = ExplorationSpec(start_urls=("https://a.test/",), depth=0, max_pages=1)
    policy = PageSettlementPolicy(settle_ms=10000, scroll_wait_ms=10, scroll_networkidle_ms=10)
    crawler = ExplorationCrawler(page=page, capture_fn=_capture_fn_maker(url_map), policy=policy)
    corpus = await crawler.crawl(spec)

    assert len(corpus.pages) == 1
    labels = corpus.pages[0].visible_elements
    assert "Hero A" in labels
    assert "Below-fold X" in labels
    assert "Below-fold Y" in labels
    # Head-most labels come from the top viewport...
    assert labels[0] == "Hero A"
    # ...and below-fold labels are not pushed out by the cap.
    assert labels.index("Below-fold X") < 30


def test_merge_visible_labels_interleaves_scroll_labels() -> None:
    top = tuple(f"top{i}" for i in range(25))
    scroll = tuple(f"scroll{i}" for i in range(10))
    merged = _merge_visible_labels(top, scroll)
    assert len(merged) == 30
    assert merged[:20] == top[:20]
    assert merged[20:] == scroll
    assert "top20" not in merged

    # Sparse scroll keeps room for remaining top labels.
    merged = _merge_visible_labels(top, ("only-below-fold",))
    assert merged[:20] == top[:20]
    assert "only-below-fold" in merged
    assert merged[21] == "top20"

    # Duplicates collapse; cap is respected.
    merged = _merge_visible_labels(("a", "a", "b"), ("b", "c"), cap=3)
    assert merged == ("a", "b", "c")


def test_merge_visible_labels_normalizes_whitespace() -> None:
    merged = _merge_visible_labels(("Hero\u00a0 A",), ("Hero A", "  Below\u00a0fold  "))
    assert merged == ("Hero A", "Below fold")


@pytest.mark.asyncio
async def test_scroll_sweep_triggers_lazy_content() -> None:
    # Simulate height growth: 1000 -> 1500 after first scroll -> stable 2 iters
    url_map = {
        normalize_crawl_url("https://a.test/"): {
            "title": "Lazy",
            "links": [],
            "height_seq": [1000, 1500, 1500, 1500],
        }
    }
    page = FakePage(url_map)
    # Use small settle_ms to bound but still trigger growth
    spec = ExplorationSpec(start_urls=("https://a.test/",), depth=0, max_pages=1, settle_ms=10000)
    policy = PageSettlementPolicy(settle_ms=10000, scroll_wait_ms=10, scroll_networkidle_ms=10)
    crawler = ExplorationCrawler(page=page, capture_fn=_capture_fn_maker(url_map), policy=policy)
    corpus = await crawler.crawl(spec)
    assert len(corpus.pages) == 1
    # scroll sweep should have triggered at least 2 scrollBy evaluates
    scroll_bys = [c for c in page.evaluate_calls if "window.scrollBy" in c]
    assert len(scroll_bys) >= 2
    # height_seq exhaustion indicates growth detection worked (more than 1 height read)
    scroll_heights = [c for c in page.evaluate_calls if "scrollHeight" in c]
    assert len(scroll_heights) >= 3
    # verify instant scrollTo-top reset called
    assert any(
        "window.scrollTo" in c and "behavior: 'instant'" in c
        for c in page.evaluate_calls
    )


@pytest.mark.asyncio
async def test_scroll_sweep_visits_bottom_before_capture_with_scroll_reveals() -> None:
    # Fixed-height page (5000px) whose headings reveal (opacity 0 -> 1)
    # only when scrollY passes thresholds; deepest threshold sits at the
    # bottom of the scroll range (5000 - 800 = 4200). Capture must happen
    # only after ALL reveals fired, i.e. the sweep reached the bottom.
    reveal_thresholds: tuple[float, ...] = (
        600.0, 1200.0, 1800.0, 2400.0, 3000.0, 3600.0, 4200.0,
    )
    url_map: dict[str, dict[str, Any]] = {
        normalize_crawl_url("https://a.test/"): {
            "title": "Reveals",
            "links": [],
            "height_seq": [5000],
        }
    }
    page = FakePage(url_map, reveal_thresholds=reveal_thresholds)

    capture_observations: list[dict[str, Any]] = []

    async def capture(page_arg: Any, viewport_id: str) -> dict[str, Any]:
        capture_observations.append(
            {
                "reveals_fired": len(page_arg.reveal_events),
                "scroll_y_at_capture": page_arg.scroll_y,
            }
        )
        return {"title": "Reveals", "headings": ("H1",), "screenshot_digest": None}

    spec = ExplorationSpec(start_urls=("https://a.test/",), depth=0, max_pages=1)
    policy = PageSettlementPolicy(settle_ms=10000, scroll_wait_ms=10, scroll_networkidle_ms=10)
    crawler = ExplorationCrawler(page=page, capture_fn=capture, policy=policy)
    corpus = await crawler.crawl(spec)

    assert len(corpus.pages) == 1
    # Sweep visited the bottom: every threshold including the deepest
    # (bottom-of-page) one fired.
    assert sorted(page.reveal_events) == sorted(reveal_thresholds)
    # Exactly one capture, and it observed ALL reveals already fired.
    assert len(capture_observations) == 1
    assert capture_observations[0]["reveals_fired"] == len(reveal_thresholds)
    # Scroll was reset to top before capture.
    assert capture_observations[0]["scroll_y_at_capture"] == 0.0


@pytest.mark.asyncio
async def test_capture_waits_for_smooth_scroll_glide_to_settle_at_top() -> None:
    # Smooth-scroll site (Lenis / scroll-behavior:smooth): scrollTo returns
    # immediately while scrollY glides 4200 -> 0 over several ticks. Capture
    # must fire only AFTER scrollY actually reaches 0 within the bounded
    # wait — never mid-glide.
    url_map: dict[str, dict[str, Any]] = {
        normalize_crawl_url("https://a.test/"): {
            "title": "Smooth",
            "links": [],
            "height_seq": [5000],
        }
    }
    page = SmoothScrollFakePage(url_map, glide_steps=4)

    capture_observations: list[dict[str, Any]] = []

    async def capture(page_arg: Any, viewport_id: str) -> dict[str, Any]:
        capture_observations.append({"scroll_y_at_capture": page_arg.scroll_y})
        return {"title": "Smooth", "headings": (), "screenshot_digest": None}

    spec = ExplorationSpec(start_urls=("https://a.test/",), depth=0, max_pages=1)
    policy = PageSettlementPolicy(
        settle_ms=10000,
        scroll_wait_ms=10,
        scroll_networkidle_ms=10,
        scroll_top_timeout_ms=1000,
        scroll_top_poll_interval_ms=50,
    )
    crawler = ExplorationCrawler(page=page, capture_fn=capture, policy=policy)
    corpus = await crawler.crawl(spec)

    assert len(corpus.pages) == 1
    # Exactly one capture, observed only after glide fully settled at top.
    assert len(capture_observations) == 1
    assert capture_observations[0]["scroll_y_at_capture"] == 0.0
    # Glide genuinely deferred the reset: multiple poll reads were needed
    # before scrollY reported 0 (jump evaluate returned a non-zero position).
    assert page.scroll_top_poll_reads >= page.glide_steps
    # Instant-jump script used instead of bare window.scrollTo(0,0).
    assert any(
        "window.scrollTo" in c and "behavior: 'instant'" in c
        for c in page.evaluate_calls
    )


@pytest.mark.asyncio
async def test_networkidle_fallback() -> None:
    # First networkidle fails, should fallback to load and still capture
    url_map = {
        normalize_crawl_url("https://a.test/"): {"title": "Polling", "links": []}
    }
    page = FakePage(url_map, networkidle_fail_first=True)
    spec = ExplorationSpec(start_urls=("https://a.test/",), depth=0, max_pages=1)
    corpus = await ExplorationCrawler(page=page, capture_fn=_capture_fn_maker(url_map)).crawl(spec)
    assert len(corpus.pages) == 1
    # Check that networkidle was attempted then load
    states = [s for s, _ in page.wait_for_load_state_calls]
    assert states.count("networkidle") >= 1
    assert "load" in states


@pytest.mark.asyncio
async def test_link_count_per_page_cap_200() -> None:
    many_links = [f"https://a.test/page{i}" for i in range(300)]
    url_map: dict[str, dict[str, Any]] = {
        normalize_crawl_url("https://a.test/"): {"title": "Home", "links": many_links}
    }
    for i in range(300):
        url_map[normalize_crawl_url(f"https://a.test/page{i}")] = {"title": f"P{i}", "links": []}
    page = FakePage(url_map)
    spec = ExplorationSpec(start_urls=("https://a.test/",), depth=1, max_pages=200)
    corpus = await ExplorationCrawler(page=page, capture_fn=_capture_fn_maker(url_map)).crawl(spec)
    # discovered_links on start page capped at 200
    start_page = next(p for p in corpus.pages if p.normalized_url == normalize_crawl_url("https://a.test/"))
    assert len(start_page.discovered_links) == 200
    # BFS respects cap: max 1 start + 200 children but max_pages 200 so <=200
    assert len(corpus.pages) <= 200
    # Also ensure queue didn't exceed 200 enqueue
    assert len(corpus.pages) == 200 or len(corpus.pages) == 201  # 1 start + 199 or 200 depending on off-by-one


@pytest.mark.asyncio
async def test_progress_detach_bounded_and_scrolled_sweep_bounded() -> None:
    # Locator timeout case should not infinite-loop; settle still bounded
    url_map = {
        normalize_crawl_url("https://a.test/"): {"title": "Progress", "links": [], "height_seq": [1000, 1000]}
    }
    page = FakePage(url_map, locator_should_timeout=True)
    spec = ExplorationSpec(start_urls=("https://a.test/",), depth=0, max_pages=1, settle_ms=500)
    policy = PageSettlementPolicy(settle_ms=500, scroll_wait_ms=10)
    corpus = await ExplorationCrawler(page=page, capture_fn=_capture_fn_maker(url_map), policy=policy).crawl(spec)
    assert len(corpus.pages) == 1
    # locator was attempted
    assert any("[role=progressbar]" in sel for sel in page.locator_calls)


@pytest.mark.asyncio
async def test_settlement_never_infinite_even_with_growing_height() -> None:
    # Height keeps growing each scroll, but settle_ms caps total time
    url_map = {
        normalize_crawl_url("https://a.test/"): {
            "title": "Infinite",
            "links": [],
            # growing each evaluation (simulate infinite scroll not stabilizing)
            "height_seq": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700],
        }
    }
    page = FakePage(url_map)
    spec = ExplorationSpec(start_urls=("https://a.test/",), depth=0, max_pages=1, settle_ms=200)
    policy = PageSettlementPolicy(settle_ms=200, scroll_wait_ms=10, scroll_networkidle_ms=10)
    corpus = await ExplorationCrawler(page=page, capture_fn=_capture_fn_maker(url_map), policy=policy).crawl(spec)
    assert len(corpus.pages) == 1
    # Should not infinite-loop; bounded by time, we expect limited scrolls
    scroll_bys = [c for c in page.evaluate_calls if "window.scrollBy" in c]
    # With 200ms budget and 10ms per iteration, maybe ~10 scrolls, not huge infinite
    assert len(scroll_bys) < 50


@pytest.mark.asyncio
async def test_viewport_capture_uses_settled_dom_via_capture_fn() -> None:
    captured_ids: list[str] = []

    async def capture(page: Any, viewport_id: str) -> dict[str, Any]:
        captured_ids.append(viewport_id)
        return {"title": "Captured", "headings": ("H1",), "screenshot_digest": hashlib.sha256(b"x").hexdigest()}

    url_map = {normalize_crawl_url("https://a.test/"): {"title": "Home", "links": []}}
    page = FakePage(url_map)
    spec = ExplorationSpec(start_urls=("https://a.test/",), depth=0, max_pages=1)
    corpus = await ExplorationCrawler(page=page, capture_fn=capture).crawl(spec)
    assert captured_ids
    assert corpus.pages[0].title == "Captured"
    assert corpus.pages[0].headings == ("H1",)


@pytest.mark.asyncio
async def test_same_origin_with_port_and_case() -> None:
    # Host case and default port handling covered via normalize + same_origin
    url_map = {
        normalize_crawl_url("https://a.test/"): {
            "title": "Home",
            "links": ["https://A.TEST:443/page", "https://a.test:8443/other"],
        },
        normalize_crawl_url("https://a.test/page"): {"title": "Page", "links": []},
    }
    page = FakePage(url_map)
    spec = ExplorationSpec(start_urls=("https://a.test/",), depth=1, max_pages=10)
    corpus = await ExplorationCrawler(page=page, capture_fn=_capture_fn_maker(url_map)).crawl(spec)
    normals = {p.normalized_url for p in corpus.pages}
    # Same origin (case/port collapsed) should be visited
    assert normalize_crawl_url("https://a.test/page") in normals
    # Different port should be dropped
    assert normalize_crawl_url("https://a.test:8443/other") not in normals


@pytest.mark.asyncio
async def test_dedup_via_tracking_params_removed() -> None:
    url_map = {
        normalize_crawl_url("https://a.test/"): {
            "title": "Home",
            "links": [
                "https://a.test/p?utm_source=x&a=1&fbclid=123",
                "https://a.test/p?a=1",
            ],
        },
        normalize_crawl_url("https://a.test/p?a=1"): {"title": "P", "links": []},
    }
    page = FakePage(url_map)
    spec = ExplorationSpec(start_urls=("https://a.test/",), depth=1, max_pages=10)
    corpus = await ExplorationCrawler(page=page, capture_fn=_capture_fn_maker(url_map)).crawl(spec)
    assert len(corpus.pages) == 2


@pytest.mark.asyncio
async def test_bfs_with_multiple_start_urls() -> None:
    url_map = {
        normalize_crawl_url("https://a.test/"): {"title": "A", "links": ["https://a.test/a1"]},
        normalize_crawl_url("https://a.test/docs"): {"title": "Docs", "links": ["https://a.test/d1"]},
        normalize_crawl_url("https://a.test/a1"): {"title": "A1", "links": []},
        normalize_crawl_url("https://a.test/d1"): {"title": "D1", "links": []},
    }
    page = FakePage(url_map)
    spec = ExplorationSpec(start_urls=("https://a.test/", "https://a.test/docs"), depth=1, max_pages=10)
    corpus = await ExplorationCrawler(page=page, capture_fn=_capture_fn_maker(url_map)).crawl(spec)
    assert len(corpus.pages) == 4
    # BFS order: starts first in given order, then their children in order discovered
    assert corpus.pages[0].normalized_url == normalize_crawl_url("https://a.test/")
    assert corpus.pages[1].normalized_url == normalize_crawl_url("https://a.test/docs")
    assert corpus.pages[2].normalized_url == normalize_crawl_url("https://a.test/a1")
    assert corpus.pages[3].normalized_url == normalize_crawl_url("https://a.test/d1")


@pytest.mark.asyncio
async def test_provider_alias_and_page_factory() -> None:
    url_map = {normalize_crawl_url("https://a.test/"): {"title": "Home", "links": []}}
    page = FakePage(url_map)
    spec = ExplorationSpec(start_urls=("https://a.test/",), depth=0, max_pages=1)
    # via provider alias
    corpus = await ExplorationCrawler(provider=page, capture_fn=_capture_fn_maker(url_map)).crawl(spec)
    assert len(corpus.pages) == 1
    # via factory
    corpus2 = await ExplorationCrawler(page_factory=_factory_for(page), capture_fn=_capture_fn_maker(url_map)).crawl(spec)
    assert len(corpus2.pages) == 1


@pytest.mark.asyncio
async def test_corpus_is_immutable_and_digest_computed() -> None:
    url_map = {normalize_crawl_url("https://a.test/"): {"title": "Home", "links": []}}
    page = FakePage(url_map)
    spec = ExplorationSpec(start_urls=("https://a.test/",), depth=0, max_pages=1)
    corpus = await ExplorationCrawler(page=page, capture_fn=_capture_fn_maker(url_map)).crawl(spec)
    assert corpus.corpus_digest
    assert len(corpus.corpus_digest) == 64
    # pages immutable tuple
    assert isinstance(corpus.pages, tuple)
