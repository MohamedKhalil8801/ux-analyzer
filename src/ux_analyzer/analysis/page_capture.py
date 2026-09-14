"""Shared crawled-corpus page capture for audit and creative redesign.

Implements the deterministic page list from ADR 0007 (start URLs first, then
BFS discovery order, deduped, capped), the segmented full-page screenshot
with bounded JPEG segments, and the trimmed node/copy inventory persisted as
the versioned ``page-capture.json`` sidecar (schema ``page-capture-v2``).
"""

from __future__ import annotations

import io
import json
import os
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

from ux_analyzer.analysis.project_audit import (
    AUDIT_USER_AGENT,
    encode_region_jpeg,
    settle_page,
    slice_full_page_png,
)

PAGE_CAPTURE_FILENAME = "page-capture.json"
PAGE_CAPTURE_SCHEMA = "page-capture-v2"  # v2: text inventory includes <dt> group titles

SEGMENT_HEIGHT_PX = 2000
MAX_SEGMENT_BYTES = 640 * 1024
DEFAULT_MAX_PAGE_HEIGHT = 12_000
MAX_INVENTORY_ENTRIES = 1500
MAX_TOTAL_COPY_CHARS = 200_000
DEFAULT_MAX_PAGES = 10
DEFAULT_CAPTURE_VIEWPORT: dict[str, int] = {"width": 1280, "height": 800}

# A capture page is produced in one of two ways (self-describing on the
# payload): ``full-page-slice`` cuts one full-page render into ~2000px
# segments, so scroll-dependent UI (sticky headers, scroll-triggered
# sidebars) appears only in its initial state; ``scroll`` shoots the live
# viewport at each scroll offset, so pinned elements repeat in every frame
# and scroll-revealed UI shows up in the segments it actually appears in.
CAPTURE_MODE_SCROLL = "scroll"
CAPTURE_MODE_FULL_PAGE = "full-page-slice"
_SETTLE_BETWEEN_SCROLL_MS = 350
_MAX_URL_LENGTH = 2048

# Init-script installed on every capture page: tags any element that receives
# a tap-ish event listener with ``data-uxa-taps`` so the inventory can
# measure the *effective* tap target -- the control itself, or the closest
# ancestor that reacts to user taps -- instead of the semantic node's painted
# box alone (a small button inside a clickable card is a large target in
# practice, and captions on a whole-card button are not separate targets).
# The document/body-level delegation roots are excluded downstream by the
# inventory's ``effectiveTapBox``, not here.
TAP_INSTRUMENTATION_JS = """
(() => {
  const TYPES = new Set([
    'click', 'dblclick', 'contextmenu',
    'pointerdown', 'pointerup',
    'mousedown', 'mouseup',
    'touchstart', 'touchend', 'touchcancel',
  ]);
  const proto = EventTarget.prototype;
  const original = proto.addEventListener;
  proto.addEventListener = function (type, listener, options) {
    if (TYPES.has(String(type).toLowerCase()) && this instanceof Element) {
      let count = 0;
      try {
        count = parseInt(this.getAttribute('data-uxa-taps') || '0', 10) || 0;
      } catch (_) {
        count = 0;
      }
      this.setAttribute('data-uxa-taps', String(count + 1));
    }
    return original.call(this, type, listener, options);
  };
})();
"""

ENV_MAX_PAGES = "UXA_REDESIGN_MAX_PAGES"
ENV_MAX_PAGE_HEIGHT = "UXA_REDESIGN_MAX_PAGE_HEIGHT"

_DEFAULT_PORTS = {"http": 80, "https": 443}

_INVENTORY_JS = """
(bounds) => {
  const cap = bounds.inventoryCap;
  const copyCap = bounds.copyCap;
  const sections = [];
  const headings = [];
  const forms = [];
  const buttons = [];
  const inputs = [];
  const links = [];
  const paragraphs = [];
  let copyChars = 0;
  let copyTruncated = false;
  let truncated = false;
  // One shared budget across all six node lists: the plan bounds the node
  // inventory at MAX_INVENTORY_ENTRIES entries in total, not per list.
  let used = 0;
  // Effective tap-target geometry: a control's real tappable surface is the
  // control itself, or the closest ancestor that reacts to user taps (cards
  // with click handlers -- found via the data-uxa-taps tag from
  // TAP_INSTRUMENTATION_JS or inline onclick attributes -- and labels
  // wrapping a control). html/body are excluded: handlers there are usually
  // delegation roots, so treating them as the target would inflate every
  // control to the whole page.
  const TAP_ROLES = new Set(['button', 'link', 'menuitem', 'tab', 'checkbox', 'radio', 'switch']);
  const TAP_ATTRIBUTES = ['onclick', 'onpointerdown', 'onpointerup', 'onmousedown', 'onmouseup', 'ontouchstart', 'ontouchend'];
  const isTapReactive = (el) => {
    if (!(el instanceof Element)) return false;
    const tag = (el.tagName || '').toLowerCase();
    if (tag === 'button' || tag === 'a' || tag === 'label') return true;
    const role = el.getAttribute('role');
    if (role && TAP_ROLES.has(role.toLowerCase())) return true;
    if (el.hasAttribute('data-uxa-taps')) return true;
    for (const attr of TAP_ATTRIBUTES) {
      if (el.hasAttribute(attr)) return true;
    }
    return false;
  };
  const effectiveTapBox = (el) => {
    let node = el;
    const parent = node.parentElement;
    if (parent && parent !== document.documentElement && parent !== document.body && isTapReactive(parent)) {
      node = parent;
    }
    const r = node.getBoundingClientRect();
    return { x: Math.round(r.left), y: Math.round(r.top + (window.scrollY || 0)), w: Math.round(r.width), h: Math.round(r.height) };
  };
  const pushNode = (list, kind, el, depth) => {
    if (used >= cap) { truncated = true; return false; }
    const r = el.getBoundingClientRect();
    const absoluteY = r.top + (window.scrollY || 0);
    const text = (el.innerText || el.textContent || '').replace(/\\u00AD/g, '').trim();
    const entry = {
      kind,
      tag: el.tagName.toLowerCase(),
      label: text.slice(0, 200),
      box: { x: Math.round(r.left), y: Math.round(absoluteY), w: Math.round(r.width), h: Math.round(r.height) },
      depth,
    };
    if (kind === 'form') { entry.action = el.getAttribute('action') || ''; entry.method = el.getAttribute('method') || ''; }
    if (kind === 'input') {
      entry.type = el.getAttribute('type') || 'text';
      entry.name = el.getAttribute('name') || '';
      entry.required = !!el.required;
    }
    if (kind === 'link') { entry.href = el.getAttribute('href') || ''; }
    if (kind === 'button' || kind === 'input' || kind === 'link') {
      // Effective tappable surface (self or closest tap-reacting ancestor);
      // the target-size guard validates hit-target claims against this.
      entry.tap_box = effectiveTapBox(el);
    }
    list.push(entry);
    used += 1;
    return true;
  };
  const sectionSelector = 'header, nav, main, section, article, aside, footer, form, [role="banner"], [role="navigation"], [role="main"], [role="complementary"], [role="contentinfo"], [role="search"]';
  document.querySelectorAll(sectionSelector).forEach((el) => {
    const depth = 0;
    const r = el.getBoundingClientRect();
    const absoluteY = r.top + (window.scrollY || 0);
    if (used < cap) {
      const heading = el.querySelector('h1, h2, h3, h4, h5, h6');
      const label = (heading && (heading.innerText || heading.textContent) || el.getAttribute('aria-label') || '').replace(/\\u00AD/g, '').trim();
      sections.push({
        tag: el.tagName.toLowerCase(),
        role: el.getAttribute('role') || '',
        label: label.slice(0, 200),
        box: { x: Math.round(r.left), y: Math.round(absoluteY), w: Math.round(r.width), h: Math.round(r.height) },
        depth,
      });
      used += 1;
    } else { truncated = true; }
  });
  document.querySelectorAll('h1, h2, h3, h4, h5, h6').forEach((el) => {
    pushNode(headings, 'heading', el, 0);
  });
  document.querySelectorAll('form').forEach((el) => pushNode(forms, 'form', el, 0));
  document.querySelectorAll('button, [role="button"], input[type="button"], input[type="submit"]').forEach((el) => {
    pushNode(buttons, 'button', el, 0);
  });
  document.querySelectorAll('input, textarea, select').forEach((el) => {
    pushNode(inputs, 'input', el, 0);
  });
  document.querySelectorAll('a[href]').forEach((el) => {
    pushNode(links, 'link', el, 0);
  });
  const mainLike = document.querySelector('main, article, [role="main"]') || document.body;
  // Text inventory: paragraphs, list items, and description-term titles. A
  // <dl> group title (dt) is persona-visible text like any other; excluding
  // it flattened grouped skill lists (dt category titles vanished, leaving
  // only the li chips) and made automated analysis conclude a grouped list
  // was "a long, comma-separated list without clear grouping". dt has no
  // nested container text (the dd carries the chips), so this cannot
  // duplicate content that p/li already capture.
  mainLike.querySelectorAll('p, li, dt').forEach((el) => {
    if (paragraphs.length >= cap) { truncated = true; return; }
    const text = (el.innerText || el.textContent || '').replace(/\\u00AD/g, '').replace(/\\s+/g, ' ').trim();
    if (!text) return;
    if (copyChars >= copyCap) { copyTruncated = true; return; }
    const remaining = copyCap - copyChars;
    const slice = text.length > remaining ? text.slice(0, remaining) : text;
    if (slice.length < text.length) { copyTruncated = true; }
    copyChars += slice.length;
    paragraphs.push(slice);
  });
  return { sections, headings, forms, buttons, inputs, links, paragraphs, truncated, copyTruncated };
}
"""


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def normalize_capture_url(value: str) -> str:
    """Canonical capture URL: http/https, lowercase host, no fragment."""

    if type(value) is not str or not value or value != value.strip():
        raise ValueError("URL must be a non-empty URL")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise ValueError("URL must have a valid host and port") from error
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("URL must use HTTP or HTTPS")
    if (
        hostname is None
        or not hostname
        or not hostname.isascii()
        or any(char.isspace() for char in hostname)
    ):
        raise ValueError("URL must include a valid host")
    if len(value) > _MAX_URL_LENGTH:
        raise ValueError("URL is too long")
    scheme = parsed.scheme.lower()
    host = hostname.lower()
    default_port = _DEFAULT_PORTS.get(scheme)
    netloc = host if port in (None, default_port) else f"{host}:{port}"
    path = parsed.path or "/"
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{scheme}://{netloc}{path}{query}"


# ---------------------------------------------------------------------------
# Page list resolution (ADR 0007)
# ---------------------------------------------------------------------------


def _site_key(url: str) -> tuple[str, str | int | None]:
    parts = urlsplit(url)
    return (parts.hostname or "", parts.port or _DEFAULT_PORTS.get(parts.scheme))


def _bfs_discovery_order(
    corpus: Mapping[str, object],
    start_urls: Sequence[str],
) -> tuple[str, ...]:
    """Discovery order: BFS over the corpus link graph from the start URLs,
    then any captured corpus page the graph did not reach, in recorded order.
    Same-site targets only; unparseable URLs are skipped.
    """

    graph = corpus.get("link_graph")
    targets_by_page: dict[str, list[str]] = {}
    if isinstance(graph, Mapping):
        for key, targets in cast(Mapping[object, object], graph).items():
            if isinstance(targets, (list, tuple)):
                targets_by_page[str(key)] = [
                    str(item)
                    for item in cast(Sequence[object], targets)
                ]
    captured_pages: list[str] = []
    pages_value = corpus.get("pages")
    if isinstance(pages_value, (list, tuple)):
        for page in cast(Sequence[object], pages_value):
            if isinstance(page, Mapping):
                page_mapping = cast(Mapping[object, object], page)
                url = page_mapping.get("url") or page_mapping.get("normalized_url")
                if isinstance(url, str) and url:
                    captured_pages.append(url)

    ordered: list[str] = []
    seen: set[str] = set()

    def visit(url: str) -> None:
        normalized = normalize_or_none(url)
        if normalized is None or normalized in seen:
            return
        seen.add(normalized)
        ordered.append(normalized)

    start_normalized = normalize_or_none(start_urls[0]) if start_urls else None
    start_site = _site_key(start_normalized) if start_normalized else None
    queue: deque[tuple[str, int]] = deque()
    for start in start_urls:
        normalized = normalize_or_none(start)
        if normalized is None:
            continue
        visit(start)
        queue.append((normalized, 0))
    depth_limit = 5
    while queue:
        current, depth = queue.popleft()
        if depth >= depth_limit:
            continue
        for target in targets_by_page.get(current, ()):
            normalized_target = normalize_or_none(target)
            if normalized_target is None or normalized_target in seen:
                continue
            if start_site is not None and _site_key(normalized_target) != start_site:
                continue
            visit(target)
            queue.append((normalized_target, depth + 1))
    for page_url in captured_pages:
        visit(page_url)
    return tuple(ordered)


def normalize_or_none(url: str) -> str | None:
    try:
        return normalize_capture_url(url)
    except ValueError:
        return None


def resolve_redesign_page_list(
    exploration_corpus: Mapping[str, object] | None,
    application_start_urls: Sequence[str],
    *,
    cap: int,
) -> tuple[str, ...]:
    """One deterministic page list (ADR 0007).

    Start URLs first (declaration order, normalized), then BFS discovery
    order from the crawl corpus link graph (same-site targets only), deduped
    on normalized URL, capped to ``cap`` entries. When the corpus is absent
    or empty the list is exactly the normalized start URLs.
    """

    if type(cap) is not int or cap < 1:
        raise ValueError("cap must be a positive integer")
    if isinstance(application_start_urls, (str, bytes)):
        raise TypeError("application_start_urls must be a sequence of strings")

    resolved: list[str] = []
    seen: set[str] = set()

    def add(url: str) -> None:
        normalized = normalize_or_none(url)
        if normalized is None or normalized in seen:
            return
        seen.add(normalized)
        resolved.append(normalized)

    for url in application_start_urls:
        if len(resolved) >= cap:
            return tuple(resolved)
        add(url)

    if exploration_corpus is not None:
        for url in _bfs_discovery_order(exploration_corpus, application_start_urls):
            if len(resolved) >= cap:
                break
            add(url)

    return tuple(resolved)


# ---------------------------------------------------------------------------
# Segment planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    """Deterministic full-page slicing plan with explicit truncation."""

    offsets: tuple[tuple[int, int], ...]
    truncated: bool
    document_height: int
    captured_height: int


def plan_page_segments(
    document_height: int,
    *,
    max_page_height: int = DEFAULT_MAX_PAGE_HEIGHT,
    segment_height: int = SEGMENT_HEIGHT_PX,
) -> SegmentPlan:
    """Slice [0, min(document_height, max_page_height)) into ~2000px segments."""

    if type(document_height) is not int or document_height < 0:
        raise ValueError("document_height must be a non-negative integer")
    if type(max_page_height) is not int or max_page_height < 1:
        raise ValueError("max_page_height must be a positive integer")
    if type(segment_height) is not int or segment_height < 1:
        raise ValueError("segment_height must be a positive integer")
    captured_height = min(document_height, max_page_height)
    offsets: list[tuple[int, int]] = []
    if captured_height == 0:
        return SegmentPlan((), False, document_height, 0)
    start = 0
    while start < captured_height:
        height = min(segment_height, captured_height - start)
        offsets.append((start, height))
        start += segment_height
    return SegmentPlan(
        tuple(offsets),
        document_height > captured_height,
        document_height,
        captured_height,
    )


# ---------------------------------------------------------------------------
# Shared browser capture (one pass, reused by audit and redesign)
# ---------------------------------------------------------------------------


def _max_page_height_from_env() -> int:
    raw = os.environ.get(ENV_MAX_PAGE_HEIGHT, "")
    if not raw:
        return DEFAULT_MAX_PAGE_HEIGHT
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_PAGE_HEIGHT
    return value if value >= 1 else DEFAULT_MAX_PAGE_HEIGHT


def max_page_height_from_env() -> int:
    """``UXA_REDESIGN_MAX_PAGE_HEIGHT`` capture cap (ADR 0007 default 12000).

    Unset, unparseable, or non-positive values fall back to the default so
    capture callers and the CLI agree on one behavior for the same input.
    """

    return _max_page_height_from_env()


def _page_viewport(page: object) -> dict[str, int] | None:
    """Actual viewport of a Playwright page; ``None`` when unavailable.

    The payload records the viewport the capture was produced in instead of
    hard-coding a value that only matches by coincidence, surfacing the
    coupling between capture and model seeing.
    """

    try:
        viewport = getattr(page, "viewport_size", None)
        if callable(viewport):
            viewport = viewport()
        if isinstance(viewport, dict):
            typed: dict[str, object] = cast(dict[str, object], viewport)
            raw_width = typed.get("width")
            raw_height = typed.get("height")
            if isinstance(raw_width, (int, float)) and isinstance(
                raw_height, (int, float)
            ):
                width = int(raw_width)
                height = int(raw_height)
                if width > 0 and height > 0:
                    return {"width": width, "height": height}
    except Exception:
        return None
    return None


def build_capture_payload(
    url: str,
    png_bytes: bytes,
    *,
    document_height: int,
    title: str = "",
    max_page_height: int | None = None,
    inventory: Mapping[str, object] | None = None,
    viewport: Mapping[str, int] | None = None,
    captured_at: str | None = None,
) -> dict[str, object]:
    """Assemble the versioned ``page-capture`` payload from raw materials.

    Pure apart from PNG slicing: segments are cut deterministically from the
    full-page screenshot; the trimmed inventory (when already collected in
    the audit's session) is passed through bounded. Without an inventory the
    caller must supply one later or the payload stays inventory-free —
    the standalone session always collects one before assembling. ``viewport``
    records the session the capture was produced in; ``captured_at`` defaults
    to the assembly wall-clock time (UTC).
    """

    resolved_height = (
        max_page_height
        if max_page_height is not None
        else max_page_height_from_env()
    )
    plan = plan_page_segments(document_height, max_page_height=resolved_height)
    encoded_segments = slice_full_page_png(png_bytes, offsets=plan.offsets)
    segments: list[dict[str, object]] = [
        {
            "index": index,
            "y_offset": offset,
            "height": height,
            "data_url": encoded_segments[index],
        }
        for index, (offset, height) in enumerate(plan.offsets)
    ]
    return _assemble_payload(
        url,
        segments,
        document_height=plan.document_height,
        captured_height=plan.captured_height,
        truncated=plan.truncated,
        inventory=inventory,
        viewport=viewport,
        title=title,
        captured_at=captured_at,
        capture_mode=CAPTURE_MODE_FULL_PAGE,
    )


def build_scroll_capture_payload(
    url: str,
    shots: Sequence[bytes | bytearray],
    *,
    document_height: int,
    title: str = "",
    max_page_height: int | None = None,
    inventory: Mapping[str, object] | None = None,
    viewport: Mapping[str, int] | None = None,
    captured_at: str | None = None,
) -> dict[str, object]:
    """Assemble a scroll-position ``page-capture`` payload.

    Each ``shots`` entry is one live-viewport PNG taken while the browser
    was scrolled to that segment's offset, so scroll-triggered and pinned
    UI keeps its real state (a sticky header repeats in every frame it is
    visible in). Segments are planned at viewport-height steps so the shots
    tile the captured document contiguously.
    """

    from PIL import Image

    resolved_height = (
        max_page_height
        if max_page_height is not None
        else max_page_height_from_env()
    )
    resolved_viewport = dict(viewport or DEFAULT_CAPTURE_VIEWPORT)
    viewport_height = int(
        resolved_viewport.get("height", DEFAULT_CAPTURE_VIEWPORT["height"])
    )
    plan = plan_page_segments(
        document_height,
        max_page_height=resolved_height,
        segment_height=viewport_height,
    )
    if len(shots) != len(plan.offsets):
        raise ValueError(
            "build_scroll_capture_payload requires exactly one shot per "
            f"planned offset: got {len(shots)} shots, "
            f"{len(plan.offsets)} offsets"
        )
    segments: list[dict[str, object]] = []
    for index, (offset, height) in enumerate(plan.offsets):
        raw = shots[index]
        image = Image.open(io.BytesIO(bytes(raw))).convert("RGB")
        segments.append(
            {
                "index": index,
                "y_offset": offset,
                "height": height,
                "data_url": encode_region_jpeg(image),
            }
        )
    return _assemble_payload(
        url,
        segments,
        document_height=plan.document_height,
        captured_height=plan.captured_height,
        truncated=plan.truncated,
        inventory=inventory,
        viewport=viewport,
        title=title,
        captured_at=captured_at,
        capture_mode=CAPTURE_MODE_SCROLL,
    )


def _assemble_payload(
    url: str,
    segments: Sequence[Mapping[str, object]],
    *,
    document_height: int,
    captured_height: int,
    truncated: bool,
    inventory: Mapping[str, object] | None = None,
    viewport: Mapping[str, int] | None = None,
    title: str = "",
    captured_at: str | None = None,
    capture_mode: str = CAPTURE_MODE_FULL_PAGE,
) -> dict[str, object]:
    """Assemble the versioned payload around already-encoded segments.

    Shared by the full-page-slice and scroll-position capture paths so both
    produce the same ``page-capture`` shape apart from ``capture_mode``.
    """

    inventory_mapping: Mapping[str, object] = inventory or {}
    stamp = datetime.now(UTC).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    return {
        "schema": PAGE_CAPTURE_SCHEMA,
        "url": url,
        "title": title,
        "captured_at": captured_at or stamp,
        "viewport": dict(viewport or DEFAULT_CAPTURE_VIEWPORT),
        "capture_mode": capture_mode,
        "document_height": document_height,
        "captured_height": captured_height,
        "truncated": truncated,
        "copy_truncated": bool(inventory_mapping.get("copyTruncated", False)),
        "inventory_truncated": bool(inventory_mapping.get("truncated", False)),
        "segments": list(segments),
        "sections": inventory_mapping.get("sections", []),
        "headings": inventory_mapping.get("headings", []),
        "forms": inventory_mapping.get("forms", []),
        "buttons": inventory_mapping.get("buttons", []),
        "inputs": inventory_mapping.get("inputs", []),
        "links": inventory_mapping.get("links", []),
        "paragraphs": inventory_mapping.get("paragraphs", []),
    }


def _collect_inventory(page: object) -> dict[str, object]:
    """Run the trimmed node/copy inventory JavaScript on a settled page."""

    evaluate = getattr(page, "evaluate", None)
    if evaluate is None:
        return {}
    try:
        value = evaluate(
            _INVENTORY_JS,
            {
                "inventoryCap": MAX_INVENTORY_ENTRIES,
                "copyCap": MAX_TOTAL_COPY_CHARS,
            },
        )
    except Exception:
        return {}
    if not isinstance(value, dict):
        return {}
    return cast(dict[str, object], value)


def audit_capture_hook(
    sink: dict[str, dict[str, object]],
    *,
    max_page_height: int | None = None,
) -> Callable[[str, object], None]:
    """Hook factory for the shared audit pass (one browser pass per page).

    Pass ``audit_urls_sync(..., capture_hook=audit_capture_hook(sink))``: for
    each audited URL the hook receives the audit's own settled session via
    ``CaptureMaterials`` (ADR 0007), slices the already-taken full-page PNG,
    and collects the inventory from the same live page. Failures are recorded
    in the returned per-URL status mapping and never raised into the audit.
    """

    def hook(url: str, materials: object) -> None:
        png_bytes = getattr(materials, "png_bytes", None)
        if not isinstance(png_bytes, (bytes, bytearray)):
            sink[url] = {"status": "error", "reason": "no capture materials"}
            return
        document_height = getattr(materials, "document_height", 0)
        title = str(getattr(materials, "title", "") or "")
        inventory = _collect_inventory(getattr(materials, "page", None))
        try:
            payload = build_capture_payload(
                url,
                bytes(png_bytes),
                document_height=int(document_height),
                title=title,
                max_page_height=max_page_height,
                inventory=inventory,
                viewport=_page_viewport(getattr(materials, "page", None)),
            )
        except Exception as error:
            sink[url] = {
                "status": "error",
                "reason": f"{type(error).__name__}: {error}"[:512],
            }
            return
        sink[url] = {"status": "captured", "payload": payload}

    return hook


def run_capture_with_own_session(
    url: str, max_page_height: int
) -> dict[str, object]:
    """Standalone fallback: one dedicated browser pass over ``url``.

    Used when no audit session is available (e.g. ``uxa redesign`` running
    against a stale or missing sidecar).
    """

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            from playwright.sync_api import ViewportSize as _ViewportSize

            viewport_spec: _ViewportSize = {
                "width": int(DEFAULT_CAPTURE_VIEWPORT["width"]),
                "height": int(DEFAULT_CAPTURE_VIEWPORT["height"]),
            }
            ctx = browser.new_context(
                viewport=viewport_spec,
                device_scale_factor=1,
                # Same UA as the audit's shared pass so UA-dependent pages
                # render identically in the fallback recapture.
                user_agent=AUDIT_USER_AGENT,
            )
            ctx.add_init_script(TAP_INSTRUMENTATION_JS)
            page = ctx.new_page()
            page.goto(url, wait_until="networkidle", timeout=60000)
            settle_page(page)
            document_height = int(
                page.evaluate("() => document.documentElement.scrollHeight")
            )
            png_bytes = page.screenshot(full_page=True, type="png")
            page.evaluate("() => window.scrollTo(0, 0)")
            inventory = _collect_inventory(page)
            title = str(page.title() or "")
            viewport = _page_viewport(page)
        finally:
            browser.close()
    return build_capture_payload(
        url,
        png_bytes,
        document_height=document_height,
        title=title,
        max_page_height=max_page_height,
        inventory=inventory,
        viewport=viewport,
    )


def run_scroll_capture_with_own_session(
    url: str,
    max_page_height: int,
    *,
    settle_ms: int = _SETTLE_BETWEEN_SCROLL_MS,
) -> dict[str, object]:
    """Standalone scroll-position capture: one dedicated browser pass.

    Shoots the live viewport once per planned scroll offset (viewport-height
    steps, so segments tile the page) instead of slicing one full-page
    render. Pinned/sticky elements repeat in every frame they are visible
    in, and UI that only appears after scrolling shows up in the segments
    where it actually appears -- neither is true of a single full-page
    screenshot, which is why ``uxa redesign`` prefers this mode.
    """

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            from playwright.sync_api import ViewportSize as _ViewportSize

            viewport_spec: _ViewportSize = {
                "width": int(DEFAULT_CAPTURE_VIEWPORT["width"]),
                "height": int(DEFAULT_CAPTURE_VIEWPORT["height"]),
            }
            ctx = browser.new_context(
                viewport=viewport_spec,
                device_scale_factor=1,
                # Same UA as the audit's shared pass so UA-dependent pages
                # render identically in the fallback recapture.
                user_agent=AUDIT_USER_AGENT,
            )
            ctx.add_init_script(TAP_INSTRUMENTATION_JS)
            page = ctx.new_page()
            page.goto(url, wait_until="networkidle", timeout=60000)
            settle_page(page)
            document_height = int(
                page.evaluate("() => document.documentElement.scrollHeight")
            )
            viewport = _page_viewport(page) or dict(DEFAULT_CAPTURE_VIEWPORT)
            viewport_height = int(
                viewport.get("height", DEFAULT_CAPTURE_VIEWPORT["height"])
            )
            plan = plan_page_segments(
                document_height,
                max_page_height=max_page_height,
                segment_height=viewport_height,
            )
            shots: list[bytes] = []
            for offset, _height in plan.offsets:
                page.evaluate(f"() => window.scrollTo(0, {offset})")
                # A short settle lets scroll-reveal animations finish at this
                # position so the frame matches what a reader sees there.
                page.wait_for_timeout(settle_ms)
                shots.append(page.screenshot(type="png"))
            page.evaluate("() => window.scrollTo(0, 0)")
            page.wait_for_timeout(settle_ms)
            inventory = _collect_inventory(page)
            title = str(page.title() or "")
        finally:
            browser.close()
    return build_scroll_capture_payload(
        url,
        shots,
        document_height=document_height,
        title=title,
        max_page_height=max_page_height,
        inventory=inventory,
        viewport=viewport,
    )


def capture_page(
    url: str,
    *,
    max_page_height: int | None = None,
    capture_mode: str = CAPTURE_MODE_SCROLL,
) -> dict[str, object]:
    """Capture one page in a dedicated session (standalone path).

    Defaults to scroll-position capture (``uxa redesign``'s preferred source)
    so pinned and scroll-triggered UI keeps its real state; pass
    ``capture_mode=CAPTURE_MODE_FULL_PAGE`` for the sliced single-render
    payload.
    """

    resolved_height = (
        max_page_height
        if max_page_height is not None
        else max_page_height_from_env()
    )
    if capture_mode == CAPTURE_MODE_SCROLL:
        return run_scroll_capture_with_own_session(url, resolved_height)
    return run_capture_with_own_session(url, resolved_height)


# ---------------------------------------------------------------------------
# Sidecar persistence
# ---------------------------------------------------------------------------


def write_page_capture(output: Path, payload: dict[str, object]) -> Path:
    """Persist ``page-capture.json`` beside experiment artifacts."""

    destination = output / PAGE_CAPTURE_FILENAME
    output.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return destination


def load_page_capture(output: Path) -> dict[str, object] | None:
    """Load the persisted sidecar; ``None`` when absent or unreadable."""

    source = output / PAGE_CAPTURE_FILENAME
    if not source.is_file():
        return None
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    return cast(dict[str, object], value)


def capture_reports_effective_tap_boxes(
    payload: Mapping[str, object],
) -> bool:
    """True when every inventoried interactive control carries ``tap_box``.

    Older sidecars predate effective tap-target measurement (the control
    itself or its closest tap-reacting ancestor); the redesign consumer
    treats such pages as stale so the next run re-captures with the
    tap-reactivity instrumentation installed (ADR 0007).
    """

    for key in ("buttons", "inputs", "links"):
        raw = payload.get(key)
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            continue
        for item in cast(Sequence[object], raw):
            if isinstance(item, Mapping) and "tap_box" not in item:
                return False
    return True


def max_pages_from_env() -> int:
    """``UXA_REDESIGN_MAX_PAGES`` page-list cap (ADR 0007 default 10)."""

    raw = os.environ.get(ENV_MAX_PAGES, "")
    if not raw:
        return DEFAULT_MAX_PAGES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_PAGES
    return value if value >= 1 else DEFAULT_MAX_PAGES


def sidecar_capture_urls(document: Mapping[str, object]) -> tuple[str, ...]:
    """Ordered unique capture URLs from a versioned sidecar document.

    Only schema ``page-capture-v2`` documents with well-formed pages count;
    anything else yields an empty tuple so the caller falls back to capture.
    """

    if document.get("schema") != PAGE_CAPTURE_SCHEMA:
        return ()
    pages = document.get("pages")
    if not isinstance(pages, Sequence) or isinstance(pages, (str, bytes)):
        return ()
    urls: list[str] = []
    seen: set[str] = set()
    for page in cast(Sequence[object], pages):
        if not isinstance(page, Mapping):
            continue
        item = cast(Mapping[object, object], page)
        if item.get("schema") != PAGE_CAPTURE_SCHEMA:
            continue
        segments = item.get("segments")
        if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
            continue
        url = item.get("url")
        if not isinstance(url, str) or not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return tuple(urls)


def sidecar_captures_by_url(document: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Latest capture per URL from a versioned sidecar document.

    Later entries win; pages that fail the same shape checks as
    :func:`sidecar_capture_urls` are dropped so missing pages can be
    re-captured individually.
    """

    captures: dict[str, dict[str, object]] = {}
    if document.get("schema") != PAGE_CAPTURE_SCHEMA:
        return captures
    pages = document.get("pages")
    if not isinstance(pages, Sequence) or isinstance(pages, (str, bytes)):
        return captures
    for page in cast(Sequence[object], pages):
        if not isinstance(page, Mapping):
            continue
        item = cast(Mapping[object, object], page)
        if item.get("schema") != PAGE_CAPTURE_SCHEMA:
            continue
        segments = item.get("segments")
        if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
            continue
        url = item.get("url")
        if not isinstance(url, str) or not url:
            continue
        captures[url] = cast(dict[str, object], dict(item))
    return captures


def fresh_capture_urls(output: Path) -> tuple[str, ...]:
    """Sidecar URLs when the persisted sidecar is present and well-formed."""

    document = load_page_capture(output)
    if document is None:
        return ()
    return sidecar_capture_urls(document)


__all__ = [
    "CAPTURE_MODE_FULL_PAGE",
    "CAPTURE_MODE_SCROLL",
    "DEFAULT_MAX_PAGE_HEIGHT",
    "ENV_MAX_PAGES",
    "ENV_MAX_PAGE_HEIGHT",
    "MAX_INVENTORY_ENTRIES",
    "MAX_SEGMENT_BYTES",
    "MAX_TOTAL_COPY_CHARS",
    "PAGE_CAPTURE_FILENAME",
    "PAGE_CAPTURE_SCHEMA",
    "DEFAULT_MAX_PAGES",
    "SEGMENT_HEIGHT_PX",
    "SegmentPlan",
    "audit_capture_hook",
    "capture_reports_effective_tap_boxes",
    "fresh_capture_urls",
    "TAP_INSTRUMENTATION_JS",
    "max_pages_from_env",
    "max_page_height_from_env",
    "sidecar_capture_urls",
    "sidecar_captures_by_url",
    "build_capture_payload",
    "build_scroll_capture_payload",
    "normalize_capture_url",
    "plan_page_segments",
    "resolve_redesign_page_list",
    "run_capture_with_own_session",
    "run_scroll_capture_with_own_session",
]
