"""Depth-bounded smart crawler with per-page settlement for exploration mode.

Implements BFS same-origin crawl with deduplication via ``normalize_crawl_url``,
per-page settlement bounded by ``PageSettlementPolicy`` (goto domcontentloaded,
networkidle fallback, progress-detach wait, scroll sweep), link cap 200,
and ``capture_with_diagnostics`` integration.

Design note craving adversarial review (Task 3 spike conclusion):

    No Python library (crawlee[playwright] etc.) fully auto-handles loading
    bars + delayed AJAX + scroll-revealed content without custom
    wait/scroll/mutation logic. ``crawlee`` gives queue/dedup/storage but
    requires custom ``request_handler`` scroll + ``wait_for_load_state`` per
    page and conflicts with ``PlaywrightSessionAdapter`` pool/allowlist.
    Verdict: reuse hardened Playwright loop here; document crawlee as
    optional alternative (``pip install crawlee[playwright]``) in
    ``docs/architecture.md`` — no hard import.

Domain isolation: imports only from ``domain`` (stdlib) and optional
``adapters/web/extractor`` at call time; no model/prompt types.
"""

from __future__ import annotations

import hashlib
import inspect
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, cast

from ux_analyzer.domain.benchmark import normalize_crawl_url, same_origin
from ux_analyzer.domain.exploration import CrawlCorpus, CrawlPage, ExplorationSpec

# ---------------------------------------------------------------------------
# Settlement policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PageSettlementPolicy:
    """Bounded settlement parameters per page."""

    settle_ms: int = 10000
    goto_timeout_ms: int = 15000
    networkidle_timeout_ms: int = 4000
    load_fallback_timeout_ms: int = 2000
    progress_detach_timeout_ms: int = 2000
    scroll_wait_ms: int = 400
    scroll_networkidle_ms: int = 800
    progress_selectors: str = "[role=progressbar], .spinner, [aria-busy=true]"
    # Hard cap on scroll-sweep steps (each step ~0.8 viewport). Bounds the
    # sweep independently of the time budget.
    max_scroll_steps: int = 40
    # Bounded wait for the scroll-to-top reset to actually settle (smooth-
    # scrolling sites glide asynchronously after scrollTo returns).
    scroll_top_timeout_ms: int = 1000
    scroll_top_poll_interval_ms: int = 50


@dataclass(frozen=True, slots=True)
class CrawlFrontier:
    """Serializable crawl position: completed pages plus pending queue.

    Emitted after every completed page via ``crawl(on_frontier=...)`` so the
    caller can persist crash-recovery checkpoints incrementally; also accepted
    back via ``crawl(resume_from=...)`` to continue an interrupted crawl from
    its saved position instead of starting over.
    """

    pages: tuple[CrawlPage, ...]
    link_graph: Mapping[str, tuple[str, ...]]
    visited: tuple[str, ...]
    queued: tuple[tuple[str, int], ...]
    started_at: str


# Tolerance (px) when deciding scrollY + innerHeight has reached
# document.scrollHeight.
_BOTTOM_EPSILON_PX = 2.0

_SCROLL_METRICS_JS = (
    "() => ({"
    "y: window.scrollY,"
    "vh: window.innerHeight,"
    "sh: Math.max("
    "document.documentElement ? document.documentElement.scrollHeight : 0,"
    "document.body ? document.body.scrollHeight : 0)"
    "})"
)

# Force an instant jump to the top. ``behavior: 'instant'`` bypasses CSS
# ``scroll-behavior: smooth``; inline ``auto`` overrides on <html>/<body>
# neutralize smoothing libraries that read computed style. Previous inline
# values are stashed as expandos and restored by _RESTORE_SCROLL_BEHAVIOR_JS.
_SCROLL_TO_TOP_JS = (
    "() => {"
    "try {"
    "const de = document.documentElement;"
    "const b = document.body;"
    "if (de && de.style) {"
    "de.__uxPrevScrollBehavior = de.style.scrollBehavior || '';"
    "de.style.scrollBehavior = 'auto';"
    "}"
    "if (b && b.style) {"
    "b.__uxPrevScrollBehavior = b.style.scrollBehavior || '';"
    "b.style.scrollBehavior = 'auto';"
    "}"
    "window.scrollTo({ top: 0, left: 0, behavior: 'instant' });"
    "} catch (e) {}"
    "return window.scrollY;"
    "}"
)

_READ_SCROLL_Y_JS = "() => window.scrollY"

_RESTORE_SCROLL_BEHAVIOR_JS = (
    "() => {"
    "try {"
    "const de = document.documentElement;"
    "const b = document.body;"
    "if (de && de.style && '__uxPrevScrollBehavior' in de) {"
    "de.style.scrollBehavior = de.__uxPrevScrollBehavior;"
    "delete de.__uxPrevScrollBehavior;"
    "}"
    "if (b && b.style && '__uxPrevScrollBehavior' in b) {"
    "b.style.scrollBehavior = b.__uxPrevScrollBehavior;"
    "delete b.__uxPrevScrollBehavior;"
    "}"
    "} catch (e) {}"
    "return null;"
    "}"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _origin_from_normalized_url(normalized_url: str) -> str:
    from urllib.parse import urlsplit

    parsed = urlsplit(normalized_url)
    host = parsed.hostname.lower() if parsed.hostname else ""  # type: ignore[union-attr]
    port = parsed.port
    if port is None or port == 443:
        return f"https://{host}"
    return f"https://{host}:{port}"


def _is_same_origin_as_any(candidate: str, start_urls: tuple[str, ...]) -> bool:
    for start in start_urls:
        try:
            if same_origin(candidate, start):
                return True
        except ValueError:
            continue
    return False


# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------


class ExplorationCrawler:
    """BFS same-origin crawler with smart per-page settlement.

    Contract: ``crawl(spec) -> CrawlCorpus``.
    Playwright ``Page`` abstraction injected via ``page`` instance or
    ``page_factory`` callable (sync or async) returning a ``Page``-like
    object. ``capture_fn`` may override default
    ``capture_with_diagnostics`` for testing.
    """

    def __init__(
        self,
        *,
        page: Any | None = None,
        page_factory: Callable[[], Any | Awaitable[Any]] | None = None,
        provider: Any | None = None,
        adapter: Any | None = None,
        capture_fn: Callable[[Any, str], Awaitable[Any]] | None = None,
        policy: PageSettlementPolicy | None = None,
    ) -> None:
        # ``provider``/``adapter`` are legacy aliases for ``page``/``page_factory``
        # used by early task spec tests (fake provider pattern).
        resolved_page = (
            page if page is not None else provider if provider is not None else adapter
        )
        resolved_factory = page_factory
        if resolved_page is not None and resolved_factory is not None:
            raise ValueError("provide either page or page_factory, not both")
        self._page = resolved_page
        self._page_factory = resolved_factory
        self._capture_fn = capture_fn
        self._policy = policy if policy is not None else PageSettlementPolicy()

    @property
    def policy(self) -> PageSettlementPolicy:
        return self._policy

    async def crawl(
        self,
        spec: ExplorationSpec,
        *,
        on_frontier: Callable[[CrawlFrontier], Awaitable[None]] | None = None,
        resume_from: CrawlFrontier | None = None,
    ) -> CrawlCorpus:
        # Resolve policy settle override from spec
        effective_policy = self._policy
        if spec.settle_ms != self._policy.settle_ms:
            # Override per-spec settle_ms while keeping other defaults
            effective_policy = replace(self._policy, settle_ms=spec.settle_ms)

        seen: set[str] = set()
        queue: deque[tuple[str, int]] = deque()
        pages: list[CrawlPage] = []
        link_graph: dict[str, tuple[str, ...]] = {}
        started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        if resume_from is not None:
            # Continue an interrupted crawl from its persisted position.
            pages = list(resume_from.pages)
            link_graph = dict(resume_from.link_graph)
            seen = set(resume_from.visited)
            queue.extend(resume_from.queued)
            started_at = resume_from.started_at
        else:
            for raw in spec.start_urls:
                try:
                    n = normalize_crawl_url(raw)
                except ValueError:
                    n = raw
                seen.add(n)
            queue.extend((u, 0) for u in spec.start_urls)

        page = await self._get_or_create_page()

        try:
            while queue and len(pages) < spec.max_pages:
                url, depth = queue.popleft()

                normalized_url: str
                try:
                    normalized_url = normalize_crawl_url(url)
                except ValueError:
                    # Skip un-normalizable URL (e.g., non-https if using strict)
                    # For bench-normalized variant it should succeed for http/https.
                    continue

                origin = _origin_from_normalized_url(normalized_url)
                viewport_id = f"explore-viewport-{len(pages)}-{abs(hash(normalized_url)) % 100000}"

                # --- settlement (bounded, never infinite) ---
                try:
                    await self._settle_page(page, url, effective_policy)
                except Exception:
                    # Bounded try/except: goto failure still attempt to capture
                    # but do not crash whole crawl; record page with empty content.
                    pass

                # --- capture ---
                title: str
                headings: tuple[str, ...]
                screenshot_digest: str | None
                visible_elements: tuple[str, ...] = ()
                try:
                    title, headings, screenshot_digest, visible_elements = await self._capture_page(
                        page, viewport_id
                    )
                except Exception:
                    title = "Untitled"
                    headings = ()
                    screenshot_digest = None
                    visible_elements = ()

                # Guarantee non-empty title for domain validation
                if not title.strip():  # pyright: ignore[reportUnknownMemberType]
                    title = "Untitled"
                else:
                    title = title.strip()  # pyright: ignore[reportUnknownMemberType]
                if not isinstance(headings, tuple):  # pyright: ignore[reportUnnecessaryIsInstance]
                    headings = tuple(headings) if headings else ()  # type: ignore[assignment]  # pyright: ignore[reportUnknownArgumentType,reportUnknownVariableType]
                if not isinstance(visible_elements, tuple):  # pyright: ignore[reportUnnecessaryIsInstance]
                    visible_elements = tuple(visible_elements) if visible_elements else ()  # type: ignore[assignment]

                # --- link extraction (same-origin <a href> absolute links) ---
                raw_links: list[str]
                try:
                    raw_links = await self._extract_links(page)
                except Exception:
                    raw_links = []

                # Filter, normalize, same-origin, cap 200
                filtered: list[str] = []
                seen_this_page: set[str] = set()
                for candidate in raw_links:
                    if not isinstance(candidate, str) or not candidate.strip():  # pyright: ignore[reportUnnecessaryIsInstance,reportUnknownArgumentType,reportUnknownMemberType]
                        continue
                    candidate = candidate.strip()  # pyright: ignore[reportUnknownMemberType]
                    # Normalize (strip fragment, sort query, remove tracking)
                    try:
                        norm = normalize_crawl_url(candidate)
                    except ValueError:
                        continue
                    if not _is_same_origin_as_any(norm, spec.start_urls):
                        continue
                    if norm in seen_this_page:
                        continue
                    seen_this_page.add(norm)
                    filtered.append(norm)
                    if len(filtered) >= 200:
                        break

                discovered_tuple = tuple(filtered)
                link_graph[normalized_url] = discovered_tuple

                # Build domain page
                try:
                    crawl_page = CrawlPage(
                        url=url,
                        normalized_url=normalized_url,
                        origin=origin,
                        depth=depth,
                        title=title,
                        headings=headings,
                        viewport_id=viewport_id,
                        screenshot_digest=screenshot_digest,
                        discovered_links=discovered_tuple,
                        visible_elements=visible_elements,
                    )
                except Exception:
                    # Fallback for edge title/heading validation
                    crawl_page = CrawlPage(
                        url=url,
                        normalized_url=normalized_url,
                        origin=origin,
                        depth=depth,
                        title=title if title else "Untitled",  # pyright: ignore[reportUnknownArgumentType]
                        headings=tuple(h for h in headings if isinstance(h, str)),  # pyright: ignore[reportUnknownVariableType,reportUnnecessaryIsInstance]
                        viewport_id=viewport_id,
                        screenshot_digest=screenshot_digest,
                        discovered_links=discovered_tuple,
                        visible_elements=tuple(x for x in visible_elements if isinstance(x, str)),  # pyright: ignore[reportUnknownVariableType,reportUnnecessaryIsInstance]
                    )
                pages.append(crawl_page)

                # --- BFS enqueue ---
                if depth < spec.depth:
                    for link in filtered:
                        if (
                            link not in seen
                            and len(pages) + len(queue) < spec.max_pages
                        ):
                            seen.add(link)
                            queue.append((link, depth + 1))
                        if len(pages) + len(queue) >= spec.max_pages:
                            break

                # --- incremental progress (checkpointing hook) ---
                if on_frontier is not None:
                    await on_frontier(
                        CrawlFrontier(
                            pages=tuple(pages),
                            link_graph=dict(link_graph),
                            visited=tuple(seen),
                            queued=tuple(queue),
                            started_at=started_at,
                        )
                    )
        finally:
            # Best-effort cleanup if factory created a closable page/browser
            with suppress(Exception):
                closer = getattr(page, "close", None)
                if callable(closer):
                    result = closer()
                    if inspect.isawaitable(result):
                        await result  # type: ignore[no-untyped-call]

        # Publish immutable corpus (digest computed by domain)
        return CrawlCorpus(
            pages=tuple(pages),
            link_graph=link_graph,
            started_at=started_at,
            corpus_digest="",
        )

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    async def _settle_page(
        self, page: Any, url: str, policy: PageSettlementPolicy
    ) -> None:
        # goto domcontentloaded 15s
        await page.goto(
            url, wait_until="domcontentloaded", timeout=policy.goto_timeout_ms
        )

        # networkidle 4s fallback to load 2s
        try:
            await page.wait_for_load_state(
                "networkidle", timeout=policy.networkidle_timeout_ms
            )
        except Exception:
            with suppress(Exception):
                await page.wait_for_load_state(
                    "load", timeout=policy.load_fallback_timeout_ms
                )

        # progress detach 2s cap
        with suppress(Exception):
            locator = page.locator(policy.progress_selectors)
            await locator.wait_for(
                state="detached", timeout=policy.progress_detach_timeout_ms
            )

        # Scroll sweep bounded by settle_ms and max_scroll_steps.
        #
        # Termination: document bottom reached (scrollY + innerHeight >=
        # scrollHeight - epsilon) OR budget/steps exhausted. Height
        # stability alone must NOT end the sweep early: GSAP/ScrollTrigger
        # pages keep a fixed scrollHeight while reveals fire at deep
        # scroll positions, so "stable height" is meaningless far from
        # the bottom. Stability only matters implicitly — if lazy content
        # grows the page, the bottom target moves and the sweep simply
        # continues within its bounds.
        start = time.monotonic()
        settle_budget_s = policy.settle_ms / 1000.0
        steps = 0
        while (
            time.monotonic() - start
        ) < settle_budget_s and steps < policy.max_scroll_steps:
            metrics: dict[str, Any] | None = None
            with suppress(Exception):
                raw_metrics: Any = await page.evaluate(_SCROLL_METRICS_JS)
                if isinstance(raw_metrics, dict):
                    metrics = cast(dict[str, Any], raw_metrics)
            if metrics is not None:
                try:
                    y = float(metrics.get("y") or 0)
                    vh = float(metrics.get("vh") or 0)
                    sh = float(metrics.get("sh") or 0)
                except (TypeError, ValueError):
                    y, vh, sh = 0.0, 0.0, 0.0
                if vh > 0 and sh > 0 and y + vh >= sh - _BOTTOM_EPSILON_PX:
                    break  # document bottom reached
            with suppress(Exception):
                await page.evaluate("window.scrollBy(0, window.innerHeight*0.8)")
            steps += 1
            with suppress(Exception):
                await page.wait_for_timeout(policy.scroll_wait_ms)
            with suppress(Exception):
                await page.wait_for_load_state(
                    "networkidle", timeout=policy.scroll_networkidle_ms
                )

        # Brief wait for reveal transitions triggered near the bottom,
        # then reset to top so capture sees the settled page from the top.
        with suppress(Exception):
            await page.wait_for_timeout(policy.scroll_wait_ms)
        await self._reset_scroll_to_top(page, policy)

    async def _reset_scroll_to_top(
        self, page: Any, policy: PageSettlementPolicy
    ) -> None:
        """Jump to top with instant behavior, then bounded wait until settled.

        A bare ``window.scrollTo(0,0)`` returns immediately on pages with CSS
        ``scroll-behavior: smooth`` or smoothing libraries (e.g. Lenis): the
        subsequent capture then fires mid-glide at an arbitrary scroll offset.
        Force an instant jump (plus temporary ``scroll-behavior: auto`` inline
        overrides), poll ``scrollY`` until it reaches 0 within a bounded
        budget, and restore prior styling afterwards. All page interactions
        are best-effort: a page that cannot be evaluated must not crash the
        crawl.
        """
        start = time.monotonic()
        deadline_s = policy.scroll_top_timeout_ms / 1000.0
        with suppress(Exception):
            await page.evaluate(_SCROLL_TO_TOP_JS)
        while (time.monotonic() - start) < deadline_s:
            raw_y: Any = None
            with suppress(Exception):
                raw_y = await page.evaluate(_READ_SCROLL_Y_JS)
            try:
                y = float(raw_y)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                break  # position unreadable; nothing further to wait for
            if y == 0.0:
                break
            with suppress(Exception):
                await page.wait_for_timeout(policy.scroll_top_poll_interval_ms)
        # Restore prior scroll-behavior styling (best effort).
        with suppress(Exception):
            await page.evaluate(_RESTORE_SCROLL_BEHAVIOR_JS)

    async def _capture_page(
        self, page: Any, viewport_id: str
    ) -> tuple[str, tuple[str, ...], str | None, tuple[str, ...]]:
        # Returns (title, headings, screenshot_digest, visible_elements)
        # visible_elements are up to 30 persona-visible labels/roles for synthesis.
        # If overridden capture_fn supplied, delegate.
        if self._capture_fn is not None:
            result = await self._capture_fn(page, viewport_id)  # type: ignore[no-untyped-call]
            # Normalize various fake shapes
            if isinstance(result, dict):
                title_any: Any = result.get("title", "Untitled")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
                headings_any: Any = result.get("headings", ())  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
                digest_any: Any = result.get("screenshot_digest")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
                visible_any: Any = result.get("visible_elements", ())  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
                if isinstance(headings_any, list):
                    headings_any = tuple(headings_any)  # pyright: ignore[reportUnknownArgumentType,reportUnknownVariableType]
                if isinstance(visible_any, list):
                    visible_any = tuple(str(x) for x in visible_any if isinstance(x, str) and str(x).strip())  # pyright: ignore[reportUnknownVariableType]
                return (
                    str(title_any) if title_any else "Untitled",  # type: ignore[reportUnknownArgumentType]  # pyright: ignore[reportUnknownArgumentType,reportUnknownVariableType]
                    tuple(headings_any) if headings_any else (),  # type: ignore[reportUnknownArgumentType,reportUnknownVariableType]  # pyright: ignore[reportUnknownArgumentType,reportUnknownVariableType]
                    digest_any,  # pyright: ignore[reportUnknownVariableType]
                    tuple(visible_any) if visible_any else (),  # type: ignore[reportUnknownVariableType]
                )  # type: ignore[return-value]
            if hasattr(result, "snapshot") and hasattr(result, "screenshot"):
                # ExtractionResult-like
                title = await self._safe_title(page)
                headings = await self._safe_headings(page)
                ves = await self._safe_visible_elements(page, getattr(result, "snapshot", None))
                screenshot = getattr(result, "screenshot", b"")  # pyright: ignore[reportUnknownArgumentType,reportAny]
                digest = (
                    hashlib.sha256(screenshot).hexdigest()  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType,reportAny]
                    if isinstance(screenshot, (bytes, bytearray)) and screenshot  # pyright: ignore[reportUnnecessaryIsInstance]
                    else None
                )
                # Validate digest is hex; if not, keep None
                return title, headings, digest, ves
            # Fallback: assume tuple
            if isinstance(result, tuple) and len(result) in (3, 4):  # pyright: ignore[reportUnknownArgumentType,reportUnknownVariableType]
                if len(result) == 3:  # type: ignore[reportUnknownVariableType]
                    t, h, d = result  # type: ignore[misc]
                    return t, h, d, ()  # type: ignore[return-value]
                return result  # type: ignore[return-value]
            # Single object with attributes
            title_obj = getattr(result, "title", None) or await self._safe_title(page)  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType,reportAny]
            headings_attr = getattr(result, "headings", None)  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType,reportAny]
            headings_obj = (
                headings_attr if headings_attr else await self._safe_headings(page)
            )  # pyright: ignore[reportUnknownArgumentType]
            digest_obj = getattr(result, "screenshot_digest", None)  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType,reportAny]
            visible_attr = getattr(result, "visible_elements", ())  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType,reportUnknownVariableType,reportAny]
            if isinstance(headings_obj, list):  # pyright: ignore[reportUnknownVariableType]
                headings_obj = tuple(headings_obj)  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType,reportAny]
            if isinstance(visible_attr, list):
                visible_attr = tuple(str(x) for x in visible_attr if isinstance(x, str) and str(x).strip())  # pyright: ignore[reportUnknownVariableType]
            return (
                str(title_obj),
                tuple(headings_obj) if headings_obj else (),  # pyright: ignore[reportUnknownArgumentType,reportUnknownVariableType,reportAny]
                digest_obj,
                tuple(visible_attr) if visible_attr else (),  # type: ignore[reportUnknownVariableType]
            )  # type: ignore[return-value]

        # Default path: try extractor capture_with_diagnostics, fallback to lightweight evaluate
        try:
            from ux_analyzer.adapters.web.extractor import capture_with_diagnostics

            result = await capture_with_diagnostics(page, viewport_id)
            title = await self._safe_title(page)
            headings = await self._safe_headings(page)
            ves2 = await self._safe_visible_elements(page, result.snapshot)
            screenshot = result.screenshot
            digest2: str | None = (
                hashlib.sha256(screenshot).hexdigest()  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType,reportUnknownVariableType,reportAny]
                if isinstance(screenshot, (bytes, bytearray)) and screenshot  # pyright: ignore[reportUnnecessaryIsInstance]
                else None
            )
            return title, headings, digest2, ves2
        except Exception:
            title = await self._safe_title(page)
            headings = await self._safe_headings(page)
            ves_fallback: tuple[str, ...] = ()
            with suppress(Exception):
                ves_fallback = await self._safe_visible_elements(page, None)
            # Best-effort screenshot digest without extractor
            try:
                png = await page.screenshot(type="png")  # type: ignore[call-arg,reportUnknownMemberType,reportUnknownVariableType]
                if isinstance(png, (bytes, bytearray)) and png:  # pyright: ignore[reportUnnecessaryIsInstance]
                    return title, headings, hashlib.sha256(bytes(png)).hexdigest(), ves_fallback  # pyright: ignore[reportUnknownArgumentType]
            except Exception:
                pass
            return title, headings, None, ves_fallback

    async def _safe_title(self, page: Any) -> str:
        try:
            t = await page.title()
            if isinstance(t, str) and t.strip():
                return t.strip()
        except Exception:
            pass
        with suppress(Exception):
            t = await page.evaluate("document.title")
            if isinstance(t, str) and t.strip():
                return t.strip()
        return "Untitled"

    async def _safe_headings(self, page: Any) -> tuple[str, ...]:
        # Try heading extraction via <h1,h2,h3>
        with suppress(Exception):
            raw = await page.evaluate(  # type: ignore[no-untyped-call,reportUnknownMemberType,reportUnknownVariableType]
                "() => Array.from(document.querySelectorAll('h1,h2,h3')).map(e => e.innerText.trim()).filter(Boolean).slice(0,3)"
            )
            if isinstance(raw, list):
                cleaned = tuple(
                    str(x).strip()
                    for x in raw  # pyright: ignore[reportUnknownVariableType]
                    if isinstance(x, str) and str(x).strip()
                )
                if cleaned:
                    return cleaned
        return ()

    async def _safe_visible_elements(
        self, page: Any, snapshot: Any | None = None
    ) -> tuple[str, ...]:
        # Prefer deterministic snapshot when available (already filtered for visibility).
        if snapshot is not None:
            try:
                elements = getattr(snapshot, "elements", None)
                if elements is not None:
                    labels: list[str] = []
                    for el in elements:  # type: ignore[unknownMemberType]
                        vis = getattr(el, "visibility_fraction", 0)
                        label = getattr(el, "label", "") or getattr(el, "rendered_text", "")
                        if not isinstance(label, str) or not label.strip():
                            continue
                        # Keep prominent/visible or actionable controls.
                        try:
                            if float(vis) < 0.05 and not bool(getattr(el, "actionable", False)):
                                continue
                        except Exception:
                            continue
                        # Strip private-ish labels, keep persona-visible wording.
                        lab = " ".join(str(label).split())[:80]
                        if "<" in lab or ">" in lab:
                            continue
                        labels.append(lab)
                        if len(labels) >= 30:
                            break
                    if labels:
                        return tuple(labels)
            except Exception:
                pass
        # Fallback: JS DOM collection of visible, actionable/textual nodes.
        with suppress(Exception):
            raw = await page.evaluate(  # type: ignore[no-untyped-call,reportUnknownMemberType,reportUnknownVariableType]
                """() => {
                  const v = (n) => {
                    const s = getComputedStyle(n);
                    return s.display !== 'none' && s.visibility !== 'hidden'
                      && Number.parseFloat(s.opacity) > 0.05
                      && n.getBoundingClientRect().height > 0;
                  };
                  const nodes = Array.from(document.querySelectorAll('a[href], button, [role=button], [role=link], input, select, textarea, h1, h2, h3, h4, [data-testid]'));
                  const out = [];
                  for (const n of nodes) {
                    if (!v(n)) continue;
                    const t = (n.innerText || n.getAttribute('aria-label') || n.textContent || '').trim().replace(/\\s+/g,' ');
                    if (!t || t.length > 80) continue;
                    if (t.match(/^\\s*(\\{|\\}|<)/)) continue;
                    out.push(t);
                    if (out.length >= 30) break;
                  }
                  if (out.length < 8) {
                    // Fill with headings if actionable set sparse (e.g., portfolio marketing page).
                    for (const h of Array.from(document.querySelectorAll('h1,h2,h3,h4')).slice(0,6)) {
                      const t = (h.innerText || '').trim().replace(/\\s+/g,' ');
                      if (t && !out.includes(t) && t.length <= 80) out.push(t);
                      if (out.length >= 30) break;
                    }
                  }
                  return out;
                }"""
            )
            if isinstance(raw, list):
                cleaned = tuple(
                    str(x).strip()
                    for x in raw  # pyright: ignore[reportUnknownVariableType]
                    if isinstance(x, str) and str(x).strip()
                )
                if cleaned:
                    return cleaned
        return ()

    async def _extract_links(self, page: Any) -> list[str]:
        # Extract same-origin <a href> absolute links via page.evaluate extracting anchors.
        raw = await page.evaluate(  # type: ignore[no-untyped-call,reportUnknownMemberType,reportUnknownVariableType]
            "() => Array.from(document.querySelectorAll('a[href]')).map(a => a.href)"
        )
        if not isinstance(raw, list):
            return []
        links: list[str] = []
        for item in raw:  # pyright: ignore[reportUnknownVariableType]
            if isinstance(item, str) and item.strip():  # pyright: ignore[reportUnknownArgumentType]
                links.append(item.strip())  # pyright: ignore[reportUnknownArgumentType]
                if len(links) >= 200:
                    break
        return links

    async def _get_or_create_page(self) -> Any:
        if self._page is not None:
            return self._page
        if self._page_factory is None:
            raise RuntimeError("ExplorationCrawler requires page or page_factory")
        maybe = self._page_factory()
        if inspect.isawaitable(maybe):
            return await maybe  # type: ignore[no-untyped-call]
        return maybe
