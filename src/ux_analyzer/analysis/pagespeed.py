"""PageSpeed Insights (Lighthouse) integration.

Fetches the real JSON produced by the PageSpeed Insights API
(https://www.googleapis.com/pagespeedonline/v5/runPagespeed) — the same
source that powers pagespeed.web.dev — and derives a bounded, report-safe
structure from it. Every number surfaced here is taken from the API payload;
nothing is invented, smoothed, or estimated locally.

Source of truth (bar)::

    https://www.googleapis.com/pagespeedonline/v5/runPagespeed?url=...&strategy=mobile

Pieces, in dependency order:

1. transport + caching  -> ``fetch_pagespeed`` / ``PagespeedCache``
2. category aggregation -> ``aggregate_categories``
3. audit extraction     -> ``extract_audits``
4. opportunities/savings -> ``extract_opportunities``
5. report assembly      -> ``build_url_report`` / ``pagespeed_report``

Pass/fail convention: an audit is "failed" when the API gives it a score
below 0.9 and a scored display mode; this is the classification pagespeed
web.dev applies in its "Failed audits" list. Audits without a score
(notApplicable / manual / informative) are not failures.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import os
import re
import secrets
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import httpx

PSI_ENDPOINT = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
PAGESPEED_FILENAME = "pagespeed.json"
PAGESPEED_SCHEMA_VERSION = "pagespeed-insights-v1"
CACHE_DIRNAME = "pagespeed-cache"
PSI_CATEGORIES = ("performance", "accessibility", "best-practices", "seo", "pwa")
PSI_STRATEGIES = ("mobile", "desktop")
PSI_PASS_THRESHOLD = 0.9
PAGESPEED_WEB_ROOT = "https://pagespeed.web.dev/analysis"

_CACHE_MAX_BYTES = 8 * 1024 * 1024
_CACHE_KEY_LOCKS: dict[str, threading.Lock] = {}


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Write a cache entry atomically despite concurrent readers.

    On Windows, ``os.replace`` (MoveFileExW with replace-existing) fails
    with WinError 5 whenever another process holds the destination open —
    even when that handle allows delete sharing (verified on Win11 25H2,
    where DeleteFileW succeeds but replace is denied). The fallback
    unlinks the destination first (allowed on share-delete handles) and
    then renames, which targets a nonexistent name and avoids the
    replace-existing semantics entirely.
    """
    def is_share_conflict(error: OSError) -> bool:
        return error.errno in {5, 13, 32} or getattr(error, "winerror", None) in {5, 32}

    delay = 0.02
    for attempt in range(7):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            if not is_share_conflict(error):
                raise
            if os.name == "nt":
                try:
                    destination.unlink()
                except OSError:
                    pass
            if attempt == 6:
                raise
            time.sleep(delay)
            delay *= 2


def _realpath_retry(path: Path) -> str | None:
    """os.path.realpath with a bounded retry for Windows rename races.

    GetFinalPathNameByHandle can transiently fail with access-denied while
    another process is renaming or deleting the path; short retries make
    the containment check race-free in practice.
    """
    delay = 0.01
    for attempt in range(6):
        try:
            return os.path.realpath(path)
        except OSError as error:
            if attempt == 5:
                raise
            time.sleep(delay)
            delay *= 2
    return None
_KEY_PATTERNS = (
    "PSI_API_Key",
    "PSI_API_KEY",
    "GOOGLE_API_KEY",
    "UXA_PSI_API_KEY",
)
_SAFE_KEY = re.compile(r"^[A-Za-z0-9._~+/=-]{10,200}$")
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_RETRY_DELAYS = (5.0, 15.0, 30.0)
_ITEM_KEYS = (
    "url",
    "totalBytes",
    "wastedBytes",
    "wastedMs",
    "responseTime",
    "transferSize",
    "requestCount",
)


class PagespeedApiError(RuntimeError):
    """Raised when the PageSpeed Insights API cannot serve a report."""

    def __init__(
        self,
        reason: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code if type(status_code) is int else None
        self.error_code = error_code


def pagespeed_web_url(url: str) -> str:
    """Deep link that opens (and runs) the same URL on pagespeed.web.dev.

    Saved-report links (``/analysis/<host>/<id>``) carry a server-assigned
    report ID that cannot be reconstructed from the URL alone, so the
    closest constructible link is the ``analysis?url=`` route: it pre-fills
    the analyzer and starts a fresh run for the same URL.
    """
    return f"{PAGESPEED_WEB_ROOT}?url={quote(url.strip(), safe='')}"


# ---------------------------------------------------------------------------
# Piece 1b: saved pagespeed.web.dev report links (server-assigned IDs)
# ---------------------------------------------------------------------------
_SAVED_LINK_RE = re.compile(
    r"^https://pagespeed\.web\.dev/analysis/"
    r"[A-Za-z0-9_-]+/[A-Za-z0-9]+$"
)
_WEB_LINKS_CACHE_FILENAME = "web-links.json"
_WEB_LINKS_MAX_BYTES = 1024 * 1024


def _normalize_saved_link(value: str) -> str | None:
    """Normalize a captured analysis URL to the canonical saved-report link."""
    if not value:
        return None
    parsed = value.split("?", 1)[0]
    if _SAVED_LINK_RE.fullmatch(parsed) is None:
        return None
    return f"{parsed}?form_factor=mobile"


def resolve_pagespeed_web_saved_link(
    url: str,
    *,
    timeout_seconds: float = 240.0,
) -> str | None:
    """Run one pagespeed.web.dev analysis headlessly and capture its ID link.

    The report ID is assigned by Google's backend when an analysis runs in
    the pagespeed.web.dev UI; once the run completes, the report is stored
    server-side and ``/analysis/<slug>/<id>`` shows the SAME saved report
    to anyone. This runs that analysis once (via the embedded page, no API
    key needed) and returns the canonical saved-report link, or ``None``
    when the UI cannot complete a run (no browser, consent wall, timeout).
    """
    from playwright.sync_api import sync_playwright

    if not url or not url.strip():
        return None
    start_link = pagespeed_web_url(url)
    deadline = time.monotonic() + max(1.0, timeout_seconds)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            context = browser.new_context()
            page = context.new_page()
            page.goto(
                start_link,
                wait_until="domcontentloaded",
                timeout=min(60000.0, max(10000.0, timeout_seconds * 1000)),
            )
            try:
                page.click("text=Ok, Got it", timeout=3000)
            except Exception:
                pass
            while time.monotonic() < deadline:
                page.wait_for_timeout(4000)
                saved = _normalize_saved_link(page.url)
                if saved is None:
                    continue
                try:
                    body = page.inner_text("body")
                except Exception:
                    body = ""
                if "Report from" in body:
                    return saved
            return None
        finally:
            browser.close()


# A saved report's numbers come from its own Lighthouse run, which is
# distinct from the API run recorded in pagespeed.json. Lighthouse scores
# vary between runs, so the saved report must never be presented as if it
# showed the recorded API numbers; its own scores are captured and shown
# alongside the link instead.
_SAVED_STRATEGIES = ("mobile", "desktop")

# A capture older than this is re-resolved so the linked report cannot go
# silently stale.
_SAVED_REPORT_TTL_SECONDS = 7 * 24 * 3600.0

# The saved report page embeds one full Lighthouse report per form factor
# as ``article.lh-root`` and hydrates them lazily, so scores are read per
# article once both have rendered; the emulated device text identifies
# which article is mobile and which is desktop.
_SAVED_ARTICLES_JS = """() => {
  const scores = {};
  for (const article of document.querySelectorAll('article.lh-root')) {
    const text = article.textContent || '';
    let strategy = null;
    if (/moto g/i.test(text)) strategy = 'mobile';
    else if (/emulated desktop/i.test(text)) strategy = 'desktop';
    if (strategy === null || strategy in scores) continue;
    const gauge = article.querySelector('.lh-scores-container .lh-gauge__wrapper');
    if (gauge === null) continue;
    const label = (gauge.getAttribute('aria-label') || '') + ' ' + (gauge.textContent || '');
    const match = label.match(/(\\d+)\\s*Performance/i) || label.match(/Performance\\s*(\\d+)/i);
    if (match) scores[strategy] = Number(match[1]);
  }
  return scores;
}"""


def extract_pagespeed_web_saved_scores(
    link: str,
    *,
    timeout_seconds: float = 120.0,
) -> dict[str, int | None]:
    """Read the performance score each saved report renders, per strategy.

    Opens the stored pagespeed.web.dev report (a saved report is served
    from Google's storage, so this never re-runs the analysis) and reads
    the performance gauge of each embedded Lighthouse report, matched to
    its strategy by the emulated device it names. Strategies whose report
    never renders come back as ``None`` so the link can still be shown
    without claiming numbers that were never read.
    """
    from playwright.sync_api import sync_playwright

    def _clean(raw: object) -> dict[str, int | None]:
        scores: dict[str, int | None] = {}
        if isinstance(raw, Mapping):
            pairs = cast(Mapping[object, object], raw)
            for strategy, score in pairs.items():
                if (
                    isinstance(strategy, str)
                    and strategy in _SAVED_STRATEGIES
                    and isinstance(score, int)
                    and not isinstance(score, bool)
                    and 0 <= score <= 100
                ):
                    scores[strategy] = score
        return scores

    deadline = time.monotonic() + max(15.0, timeout_seconds)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_context().new_page()
            try:
                page.goto(
                    link,
                    wait_until="domcontentloaded",
                    timeout=min(60000.0, max(10000.0, timeout_seconds * 1000)),
                )
                try:
                    page.click("text=Ok, Got it", timeout=3000)
                except Exception:
                    pass
                scores: dict[str, int | None] = {}
                while time.monotonic() < deadline:
                    page.wait_for_timeout(2000)
                    scores = _clean(page.evaluate(_SAVED_ARTICLES_JS))
                    if all(s in scores for s in _SAVED_STRATEGIES):
                        return scores
                return scores
            except Exception:  # noqa: BLE001 - unreadable report yields no scores
                return {}
        finally:
            browser.close()


def resolve_pagespeed_web_saved_report(
    url: str,
    *,
    timeout_seconds: float = 240.0,
) -> dict[str, Any] | None:
    """Capture a saved-report link plus the scores that report shows.

    Combines :func:`resolve_pagespeed_web_saved_link` (which runs the
    pagespeed.web.dev analysis once) with
    :func:`extract_pagespeed_web_saved_scores` (which reads the stored
    report's own numbers). Returns ``None`` when no saved report could be
    captured; the ``scores`` mapping is best-effort and may be empty when
    the stored report's gauges could not be read.
    """
    link = resolve_pagespeed_web_saved_link(url, timeout_seconds=timeout_seconds)
    if link is None:
        return None
    try:
        scores = extract_pagespeed_web_saved_scores(link)
    except Exception:  # noqa: BLE001 - the link is captured; scores are optional
        scores = {}
    return {
        "link": link,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scores": scores,
    }


class WebLinksCache:
    """Disk cache of resolved pagespeed.web.dev saved-report captures.

    Keyed by analyzed URL so repeated ``uxa run`` invocations reuse the
    same server-side report instead of triggering a new analysis each time.
    Entries carry the saved link, the capture timestamp, and the scores the
    saved report rendered when it was captured. Legacy entries that predate
    score capture hold a bare link string and are migrated on load; they
    have no capture timestamp, so they expire immediately and are
    re-captured on the next run that allows resolution.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _path(self) -> Path:
        return self.root / CACHE_DIRNAME / _WEB_LINKS_CACHE_FILENAME

    def load_entry(self, url: str) -> dict[str, Any] | None:
        """Return the cached capture for ``url``, migrating legacy entries."""
        try:
            if self._path().stat().st_size > _WEB_LINKS_MAX_BYTES:
                return None
            value = json.loads(self._path().read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError, RecursionError):
            return None
        if not isinstance(value, Mapping):
            return None
        entries = cast(Mapping[object, object], value)
        raw_entry = entries.get(url)
        if isinstance(raw_entry, str):
            return {"link": raw_entry, "captured_at": None, "scores": {}}
        if not isinstance(raw_entry, Mapping):
            return None
        entry = cast(Mapping[str, object], raw_entry)
        link = entry.get("link")
        if not isinstance(link, str) or not link:
            return None
        captured_at = entry.get("captured_at")
        if not isinstance(captured_at, str):
            captured_at = None
        scores: dict[str, int | None] = {}
        raw_scores = entry.get("scores")
        if isinstance(raw_scores, Mapping):
            pairs = cast(Mapping[object, object], raw_scores)
            for strategy, score in pairs.items():
                if isinstance(strategy, str) and isinstance(
                    score, int
                ) and not isinstance(score, bool):
                    scores[strategy] = score
        return {"link": link, "captured_at": captured_at, "scores": scores}

    def load(self, url: str) -> str | None:
        entry = self.load_entry(url)
        if entry is None:
            return None
        return entry["link"]

    def store(self, url: str, saved_link: str) -> None:
        self.store_entry(
            url,
            {
                "link": saved_link,
                "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "scores": {},
            },
        )

    def store_entry(self, url: str, entry: Mapping[str, Any]) -> None:
        path = self._path()
        try:
            if path.is_file():
                value = json.loads(path.read_text(encoding="utf-8"))
            else:
                value = {}
        except (OSError, ValueError, UnicodeError, RecursionError):
            value = {}
        if not isinstance(value, Mapping):
            value = {}
        entries = dict(cast(Mapping[str, object], value))
        entries[url] = {
            "link": entry.get("link"),
            "captured_at": entry.get("captured_at"),
            "scores": dict(entry.get("scores") or {}),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
        try:
            temporary.write_text(
                json.dumps(entries, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(temporary, path)
        except BaseException:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise


def _saved_entry_is_fresh(entry: Mapping[str, Any] | None) -> bool:
    """A capture is reusable only within the saved-report TTL."""
    if entry is None:
        return False
    captured_at = entry.get("captured_at")
    if not isinstance(captured_at, str):
        return False
    try:
        captured = time.strptime(captured_at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    age = time.time() - calendar.timegm(captured)
    return 0 <= age < _SAVED_REPORT_TTL_SECONDS


def enrich_pagespeed_web_links(
    report: Mapping[str, Any],
    *,
    cache_root: Path | str | None = None,
    resolve: bool = True,
    timeout_seconds: float = 240.0,
) -> dict[str, Any]:
    """Attach saved pagespeed.web.dev report links to a derived report.

    For every URL entry the resolved saved link (``/analysis/<slug>/<id>``,
    cached per URL) becomes ``pagespeed_web_url``; the re-run deep link is
    always available as ``pagespeed_web_fresh_url``. Resolution only runs
    once per URL thanks to the cache — subsequent runs reuse the same
    server-side report. When resolution is skipped or fails, the fresh-run
    deep link remains as the fallback.
    """
    urls = report.get("urls")
    if not isinstance(urls, list):
        return dict(report)
    cache = WebLinksCache(cache_root) if cache_root is not None else None
    for idx, url_report in enumerate(urls):
        if not isinstance(url_report, Mapping):
            continue
        entry = dict(url_report)
        url = entry.get("url")
        if not isinstance(url, str) or not url.strip():
            continue
        fresh = pagespeed_web_url(url)
        saved: str | None = None
        if cache is not None:
            saved = cache.load(url)
        if saved is None and resolve:
            try:
                saved = resolve_pagespeed_web_saved_link(
                    url, timeout_seconds=timeout_seconds
                )
            except Exception as error:  # noqa: BLE001 - link must not fail a run
                saved = None
            if saved is not None and cache is not None:
                try:
                    cache.store(url, saved)
                except OSError:
                    pass
        entry["pagespeed_web_fresh_url"] = fresh
        entry["pagespeed_web_url"] = saved or fresh
        entry["pagespeed_web_saved"] = saved is not None
        urls[idx] = entry
    return dict(report)


# ---------------------------------------------------------------------------
# Piece 1: API transport + caching
# ---------------------------------------------------------------------------
def pagespeed_api_key(environ: Mapping[str, str] | None = None) -> str | None:
    """Resolve a Google API key for PageSpeed Insights.

    Checks ``PSI_API_Key`` (and case/underscore variants, plus
    ``GOOGLE_API_KEY``) in the process environment. The CLI already loads
    ``.env`` before commands run, so a key placed there is found.
    """
    env: Mapping[str, str] = os.environ if environ is None else environ
    for name in _KEY_PATTERNS:
        value = env.get(name)
        if not value:
            continue
        value = value.strip()
        if _SAFE_KEY.fullmatch(value) is not None:
            return value
    return None


def _psi_request_url(url: str, strategy: str, key: str | None) -> str:
    pairs: list[tuple[str, str]] = [("url", url), ("strategy", strategy)]
    pairs.extend(("category", category) for category in PSI_CATEGORIES)
    if key:
        pairs.append(("key", key))
    return f"{PSI_ENDPOINT}?{_encode_query(pairs)}"


def _error_code_from_payload(payload: object) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    error = payload.get("error")
    if not isinstance(error, Mapping):
        return None
    error_code = error.get("code")
    message = error.get("message")
    if error_code is not None:
        return f"{error_code}: {message}" if isinstance(message, str) else str(error_code)
    if isinstance(message, str) and message:
        return message[:300]
    return None


async def fetch_pagespeed(
    url: str,
    strategy: str,
    *,
    key: str | None = None,
    client: httpx.AsyncClient | None = None,
    retries: int = 2,
    timeout_seconds: float = 240.0,
) -> dict[str, Any]:
    """Fetch the raw PageSpeed Insights JSON for one URL and strategy.

    Returns the API envelope as parsed JSON (``lighthouseResult``,
    ``loadingExperience``, ...). Raises :class:`PagespeedApiError` when the
    API refuses the request; transient 429/5xx responses are retried with
    backoff. The API key is a query parameter, so it never appears in logs.
    """
    if not url or not url.strip():
        raise ValueError("url must not be empty")
    if strategy not in PSI_STRATEGIES:
        raise ValueError(f"strategy must be one of {', '.join(PSI_STRATEGIES)}")
    if retries < 0:
        raise ValueError("retries must be non-negative")
    own_client = False
    if client is None:
        own_client = True
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds, connect=15.0),
            follow_redirects=True,
            headers={"User-Agent": "ux-analyzer pagespeed collector"},
        )
    params = _psi_request_url(url, strategy, key)
    request_url = params
    try:
        last_error: PagespeedApiError | None = None
        for attempt in range(retries + 1):
            try:
                response = await client.get(request_url)
            except httpx.HTTPError as error:
                last_error = PagespeedApiError(f"network failure: {type(error).__name__}")
                if attempt < retries:
                    await _backoff(attempt)
                continue
            else:
                if response.status_code == 200:
                    try:
                        payload = json.loads(response.content)
                    except (ValueError, UnicodeError, RecursionError) as error:
                        raise PagespeedApiError(
                            f"API returned malformed JSON ({type(error).__name__})"
                        ) from error
                    if not isinstance(payload, Mapping):
                        raise PagespeedApiError("API returned a non-object payload")
                    return dict(cast(Mapping[str, Any], payload))
                error_code = _safe_error_code(response.content)
                if response.status_code in _RETRY_STATUSES and attempt < retries:
                    last_error = PagespeedApiError(
                        f"API rate-limited or unavailable (HTTP {response.status_code})",
                        status_code=response.status_code,
                        error_code=error_code,
                    )
                    await _backoff(attempt)
                    continue
                raise PagespeedApiError(
                    f"API request failed (HTTP {response.status_code})",
                    status_code=response.status_code,
                    error_code=error_code,
                )
        if last_error is not None:
            raise last_error
        raise PagespeedApiError("API request failed after retries")
    finally:
        if own_client:
            await client.aclose()


def _safe_error_code(content: bytes) -> str | None:
    """Best-effort error code from an API error body; never raises."""
    if not content:
        return None
    try:
        payload = json.loads(content)
    except (ValueError, UnicodeError, RecursionError):
        return None
    return _error_code_from_payload(payload)


async def _backoff(attempt: int) -> None:
    delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
    await asyncio_sleep(delay)


async def asyncio_sleep(delay: float) -> None:
    import asyncio

    await asyncio.sleep(delay)


def _encode_query(pairs: Sequence[tuple[str, str]]) -> str:
    return "&".join(
        f"{quote(str(k), safe='')}={quote(str(v), safe='')}" for k, v in pairs
    )


def _read_bytes_shared_delete(path: Path) -> bytes | None:
    """Read a file allowing concurrent replacement on Windows.

    Python's ``open()`` opens without FILE_SHARE_DELETE, so another process
    cannot ``os.replace`` over the file while it is being read (rename fails
    with WinError 5). Reopening the handle with FILE_SHARE_DELETE makes
    cache reads transparent to concurrent writers.
    """
    if os.name != "nt":
        try:
            return path.read_bytes()
        except OSError:
            return None
    import ctypes
    import msvcrt
    from ctypes import wintypes

    create_file = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(path),
        0x80000000,  # GENERIC_READ
        0x1 | 0x2 | 0x4,  # FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE
        None,
        3,  # OPEN_EXISTING
        0x80,  # FILE_ATTRIBUTE_NORMAL
        None,
    )
    if handle is None or int(handle) == ctypes.c_void_p(-1).value:
        return None
    try:
        descriptor = msvcrt.open_osfhandle(int(handle), os.O_RDONLY)
    except OSError:
        _close_windows_handle(handle)
        return None
    try:
        with os.fdopen(descriptor, "rb") as stream:
            return stream.read()
    except OSError:
        return None


def _close_windows_handle(handle: int) -> None:
    import ctypes
    from ctypes import wintypes

    close_handle = ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    try:
        close_handle(int(handle))
    except OSError:
        pass


class PagespeedCache:
    """Disk cache of raw API responses keyed by URL + strategy.

    Cached payloads are byte-identical to what the API returned (raw JSON,
    compact separators), so reruns of ``uxa run`` reuse the previous
    evidence instead of burning quota or producing different numbers.
    Entries are bounded in size and never read or written through symlinks
    or reparse points (junctions), so a hostile entry cannot smuggle file
    content from elsewhere on disk.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _path_for(self, url: str, strategy: str) -> Path:
        digest = hashlib.sha256(
            f"{url}\n{strategy}".encode("utf-8")
        ).hexdigest()
        return self.root / CACHE_DIRNAME / f"{digest}.json"

    @staticmethod
    def _is_link_or_reparse(path: Path) -> bool:
        # A transient stat failure (a file being renamed on Windows) is
        # treated as NOT a reparse point: the realpath containment check
        # in _safe_path is the actual security boundary, and it resolves
        # links through GetFinalPathNameByHandle.
        try:
            if path.is_symlink():
                return True
        except OSError:
            return False
        is_junction = getattr(os.path, "isjunction", None)
        if callable(is_junction):
            try:
                return bool(is_junction(path))
            except OSError:
                return False
        return False

    def _safe_path(self, path: Path) -> bool:
        """True when the resolved target stays inside the cache root.

        Guards against symlinks or junctions at ANY path component (the
        leaf file, any parent directory, or the root itself).
        """
        if self._is_link_or_reparse(self.root):
            return False
        if self._is_link_or_reparse(path):
            return False
        try:
            root_real = _realpath_retry(self.root)
            target_real = _realpath_retry(path)
        except OSError:
            return False
        if root_real is None or target_real is None:
            return False
        root_real = os.path.normcase(root_real)
        target_real = os.path.normcase(target_real)
        if target_real == root_real:
            return False
        return target_real.startswith(root_real.rstrip(os.sep) + os.sep)

    def _safe_path_retry(self, path: Path) -> bool:
        """_safe_path with bounded retries for transient rename races."""
        delay = 0.02
        for attempt in range(6):
            if self._safe_path(path):
                return True
            if attempt == 5:
                return False
            time.sleep(delay)
            delay *= 2
        return False

    def _temporary_path(self, path: Path) -> Path:
        suffix = f"{os.getpid()}.{secrets.token_hex(4)}"
        return path.with_name(f".{path.name}.{suffix}.tmp")

    def load(self, url: str, strategy: str) -> dict[str, Any] | None:
        path = self._path_for(url, strategy)
        if not self._safe_path(path):
            return None
        lock = _CACHE_KEY_LOCKS.setdefault(str(path), threading.Lock())
        with lock:
            try:
                if not path.is_file() or path.stat().st_size > _CACHE_MAX_BYTES:
                    return None
                raw = _read_bytes_shared_delete(path)
                if raw is None:
                    return None
                value = json.loads(raw.decode("utf-8"))
            except (OSError, ValueError, UnicodeError, RecursionError):
                return None
            if not isinstance(value, Mapping):
                return None
            return dict(cast(Mapping[str, Any], value))

    def store(self, url: str, strategy: str, payload: Mapping[str, Any]) -> Path:
        path = self._path_for(url, strategy)
        key = str(path)
        lock = _CACHE_KEY_LOCKS.setdefault(key, threading.Lock())
        with lock:
            if not self._safe_path_retry(path):
                raise OSError("cache target must stay inside the cache root")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._temporary_path(path)
            if not self._safe_path_retry(temporary):
                raise OSError("cache temporary file must stay inside the cache root")
            try:
                encoded = json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
                if len(encoded) > _CACHE_MAX_BYTES:
                    raise OSError("cached payload exceeds size bound")
                temporary.write_bytes(encoded)
                _replace_with_retry(temporary, path)
            except BaseException:
                try:
                    temporary.unlink()
                except OSError:
                    pass
                raise
        return path


# ---------------------------------------------------------------------------
# Piece 2: category score aggregation
# ---------------------------------------------------------------------------
def aggregate_categories(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Aggregate Lighthouse category scores from the API payload.

    Each category yields ``id``, ``title``, ``score`` (0-1 as served),
    ``score_percent`` (the same number as a 0-100 integer), ``display_value``
    (the served human string, e.g. performance totals) and the number of
    ``audit_refs`` the API attached to the category.
    """
    result = _mapping(_mapping(raw).get("lighthouseResult")).get("categories")
    categories = _mapping(result)
    rows: list[dict[str, Any]] = []
    for category_id, raw_category in categories.items():
        if not isinstance(raw_category, Mapping):
            continue
        score = _number(raw_category.get("score"))
        refs = raw_category.get("auditRefs")
        row: dict[str, Any] = {
            "id": _text(raw_category.get("id"), category_id),
            "title": _text(raw_category.get("title"), category_id),
            "score": score,
            "score_percent": round(score * 100) if score is not None else None,
            "display_value": _optional_text(raw_category.get("displayValue")),
            "audit_refs": len(refs) if isinstance(refs, (list, tuple)) else 0,
        }
        rows.append(row)
    return sorted(rows, key=lambda item: str(item["id"]))


# ---------------------------------------------------------------------------
# Piece 3: audit extraction (per-audit pass / fail)
# ---------------------------------------------------------------------------
_SCORED_MODES = frozenset({"binary", "numeric", "metricSavings", "passableAudit"})


def _audit_mode(raw: Mapping[str, Any]) -> str:
    mode = _text(raw.get("scoreDisplayMode"), "numeric")
    return mode if mode in _SCORED_MODES or mode in {
        "notApplicable", "manual", "informative", "error"
    } else "numeric"


def _audit_failed(mode: str, score: float | None) -> bool:
    if mode not in _SCORED_MODES:
        return False
    return score is not None and score < PSI_PASS_THRESHOLD


def _audit_row(audit_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    mode = _audit_mode(raw)
    score = _number(raw.get("score"))
    return {
        "id": audit_id,
        "title": _text(raw.get("title"), audit_id),
        "score": score,
        "score_percent": round(float(score) * 100) if score is not None else None,
        "score_display_mode": mode,
        "display_value": _optional_text(raw.get("displayValue")),
        "description": _optional_text(raw.get("description")),
        "failed": _audit_failed(mode, score),
    }


def extract_audits(raw: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Extract every audit from ``lighthouseResult.audits``.

    Returns one list per classification bucket: ``failed``, ``passed``
    (scored >= 0.9), ``not_applicable``, ``manual``, ``informative`` and
    ``error``. Values are taken verbatim from the API; only the
    failed/passed split is derived, using the pagespeed.web.dev convention
    (score < 0.9 and scored mode = failed).
    """
    audits = _mapping(_mapping(raw).get("lighthouseResult")).get("audits")
    buckets: dict[str, list[dict[str, Any]]] = {
        "failed": [],
        "passed": [],
        "not_applicable": [],
        "manual": [],
        "informative": [],
        "error": [],
    }
    for audit_id, raw_audit in _mapping(audits).items():
        if not isinstance(raw_audit, Mapping):
            continue
        mode = _audit_mode(raw_audit)
        score = _number(raw_audit.get("score"))
        row = _audit_row(audit_id, raw_audit)
        if mode == "error" or (mode in _SCORED_MODES and score is None):
            buckets["error"].append(row)
        elif mode == "notApplicable":
            buckets["not_applicable"].append(row)
        elif mode == "manual":
            buckets["manual"].append(row)
        elif mode == "informative":
            buckets["informative"].append(row)
        elif _audit_failed(mode, score):
            buckets["failed"].append(row)
        else:
            buckets["passed"].append(row)
    for key in buckets:
        buckets[key] = sorted(buckets[key], key=lambda item: str(item["id"]))
    return buckets


# ---------------------------------------------------------------------------
# Piece 4: opportunities and estimated savings
# ---------------------------------------------------------------------------
def _details_items(details: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_items = details.get("items")
    items: list[dict[str, Any]] = []
    if isinstance(raw_items, list):
        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping):
                continue
            item: dict[str, Any] = {}
            for key in _ITEM_KEYS:
                if key in raw_item:
                    value = raw_item[key]
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        item[key] = value
                    elif isinstance(value, str) and key == "url":
                        item[key] = value[:2000]
            if item:
                items.append(item)
    return items[:50]


def extract_opportunities(raw: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Extract opportunities and other savings-bearing audits.

    ``opportunities`` = audits whose ``details.type`` is ``opportunity``;
    ``metric_savings`` = other audits carrying ``overallSavingsMs`` /
    ``overallSavingsBytes`` in their details (Lighthouse ``metricSavings``
    mode). Savings are copied from the API payload unchanged; items list
    the specific URLs and byte/millisecond estimates the API supplied.
    """
    audits = _mapping(_mapping(raw).get("lighthouseResult")).get("audits")
    opportunities: list[dict[str, Any]] = []
    metric_savings: list[dict[str, Any]] = []
    for audit_id, raw_audit in _mapping(audits).items():
        if not isinstance(raw_audit, Mapping):
            continue
        details = raw_audit.get("details")
        if not isinstance(details, Mapping):
            continue
        detail_type = _text(details.get("type"))
        has_savings = (
            "overallSavingsMs" in details or "overallSavingsBytes" in details
        )
        if detail_type != "opportunity" and not has_savings:
            continue
        row: dict[str, Any] = {
            "id": audit_id,
            "title": _text(raw_audit.get("title"), audit_id),
            "score": _number(raw_audit.get("score")),
            "score_display_mode": _audit_mode(raw_audit),
            "display_value": _optional_text(raw_audit.get("displayValue")),
            "savings_ms": _number(details.get("overallSavingsMs")),
            "savings_bytes": _number(details.get("overallSavingsBytes")),
            "items": _details_items(details),
        }
        if detail_type == "opportunity":
            opportunities.append(row)
        else:
            metric_savings.append(row)
    opportunities.sort(key=_savings_sort_key, reverse=True)
    metric_savings.sort(key=_savings_sort_key, reverse=True)
    return {
        "opportunities": opportunities,
        "metric_savings": metric_savings,
    }


def _savings_sort_key(row: Mapping[str, Any]) -> float:
    savings_ms = row.get("savings_ms")
    savings_bytes = row.get("savings_bytes")
    ms = savings_ms if isinstance(savings_ms, (int, float)) else 0
    bytes_value = savings_bytes if isinstance(savings_bytes, (int, float)) else 0
    return float(ms) + float(bytes_value) / 1024.0


# ---------------------------------------------------------------------------
# Field data (CrUX) — part of the API envelope, shown by pagespeed.web.dev
# ---------------------------------------------------------------------------
def extract_field_data(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    """Compact ``loadingExperience`` field data (CrUX) from the payload.

    Only numbers served by the API are included; an empty or absent block
    yields ``None``.
    """
    loading = raw.get("loadingExperience")
    if not isinstance(loading, Mapping):
        return None
    overall = _text(loading.get("overall_category"))
    metrics_raw = loading.get("metrics")
    metrics: dict[str, dict[str, Any]] = {}
    if isinstance(metrics_raw, Mapping):
        for metric_id, metric in metrics_raw.items():
            if not isinstance(metric, Mapping):
                continue
            percentile = _number(metric.get("percentile"))
            category = _text(metric.get("category"))
            row: dict[str, Any] = {}
            if percentile is not None:
                row["percentile"] = percentile
            if category:
                row["category"] = category
            if row:
                metrics[metric_id] = row
    if not overall and not metrics:
        return None
    result: dict[str, Any] = {}
    if overall:
        result["overall_category"] = overall
    if metrics:
        result["metrics"] = metrics
    return result


# ---------------------------------------------------------------------------
# Piece 5: report assembly
# ---------------------------------------------------------------------------
async def build_url_report(
    url: str,
    strategies: Sequence[str] = PSI_STRATEGIES,
    *,
    key: str | None = None,
    cache: PagespeedCache | None = None,
    client: httpx.AsyncClient | None = None,
    use_cache: bool = True,
    store_cache: bool = True,
    timeout_seconds: float = 240.0,
) -> dict[str, Any]:
    """Fetch (or reuse from cache) PSI reports for every strategy of one URL.

    Returns the bounded, report-safe structure consumed by
    :func:`pagespeed_report`; a failed strategy becomes an ``error`` entry
    carrying the API error text so the report can show exactly what
    happened instead of guessing.
    """
    if not url or not url.strip():
        raise ValueError("url must not be empty")
    url = url.strip()
    strategy_entries: dict[str, Any] = {}
    for strategy in strategies:
        payload: dict[str, Any] | None = None
        from_cache = False
        if cache is not None and use_cache:
            payload = cache.load(url, strategy)
            from_cache = payload is not None
        if payload is None:
            try:
                payload = await fetch_pagespeed(
                    url,
                    strategy,
                    key=key,
                    client=client,
                    timeout_seconds=timeout_seconds,
                )
            except (PagespeedApiError, ValueError) as error:
                detail = str(error)
                code = getattr(error, "error_code", None)
                if isinstance(code, str) and code.strip():
                    detail = f"{detail} — {code[:400]}"
                strategy_entries[strategy] = {
                    "status": "error",
                    "error": f"{type(error).__name__}: {detail}"[:900],
                    "error_code": code[:600] if isinstance(code, str) else None,
                }
                continue
            if cache is not None and store_cache:
                try:
                    cache.store(url, strategy, payload)
                except OSError:
                    pass
        lighthouse = _mapping(payload.get("lighthouseResult"))
        audits = extract_audits(payload)
        opportunities = extract_opportunities(payload)
        field_data = extract_field_data(payload)
        entry: dict[str, Any] = {
            "status": "ok",
            "from_cache": from_cache,
            "fetched_at": _optional_text(lighthouse.get("fetchTime")),
            "analysis_timestamp": _optional_text(payload.get("analysisUTCTimestamp")),
            "requested_url": _optional_text(lighthouse.get("requestedUrl")),
            "final_url": _optional_text(lighthouse.get("finalUrl")),
            "main_document_url": _optional_text(lighthouse.get("mainDocumentUrl")),
            "lighthouse_version": _optional_text(lighthouse.get("lighthouseVersion")),
            "user_agent": _optional_text(lighthouse.get("userAgent")),
            "categories": aggregate_categories(payload),
            "audits": {
                "failed": audits["failed"],
                "passed": audits["passed"],
                "not_applicable": audits["not_applicable"],
                "manual": audits["manual"],
                "informative": audits["informative"],
                "error": audits["error"],
                "totals": {
                    "failed": len(audits["failed"]),
                    "passed": len(audits["passed"]),
                    "not_applicable": len(audits["not_applicable"]),
                    "manual": len(audits["manual"]),
                    "informative": len(audits["informative"]),
                    "error": len(audits["error"]),
                },
            },
            "opportunities": opportunities["opportunities"],
            "metric_savings": opportunities["metric_savings"],
        }
        if field_data is not None:
            entry["field_data"] = field_data
        strategy_entries[strategy] = entry
    ok_strategies = [
        strategy
        for strategy, entry in strategy_entries.items()
        if entry.get("status") == "ok"
    ]
    return {
        "url": url,
        "pagespeed_web_url": pagespeed_web_url(url),
        "strategies": strategy_entries,
        "ok_strategies": ok_strategies,
    }


async def pagespeed_report(
    urls: Sequence[str],
    *,
    key: str | None = None,
    cache_root: Path | str | None = None,
    strategies: Sequence[str] = PSI_STRATEGIES,
    use_cache: bool = True,
    store_cache: bool = True,
    timeout_seconds: float = 240.0,
) -> dict[str, Any]:
    """Build the persisted ``pagespeed.json`` report for many URLs.

    URLs are deduplicated and each is fetched with an isolated HTTP client
    so quota and timeout behavior stays predictable. Failures never raise:
    every URL keeps whatever strategies succeeded and records the rest as
    ``error`` entries.
    """
    unique_urls = tuple(dict.fromkeys(url for url in urls if url and url.strip()))
    cache = PagespeedCache(cache_root) if cache_root is not None else None
    url_reports: list[dict[str, Any]] = []
    for url in unique_urls:
        try:
            url_reports.append(
                await build_url_report(
                    url,
                    strategies=strategies,
                    key=key,
                    cache=cache,
                    use_cache=use_cache,
                    store_cache=store_cache,
                    timeout_seconds=timeout_seconds,
                )
            )
        except Exception as error:  # noqa: BLE001 - report must not fail a run
            url_reports.append(
                {
                    "url": url,
                    "strategies": {},
                    "ok_strategies": [],
                    "error": f"{type(error).__name__}: {error}"[:512],
                }
            )
    ok_strategy_count = sum(len(report.get("ok_strategies", ())) for report in url_reports)
    return {
        "schema_version": PAGESPEED_SCHEMA_VERSION,
        "urls": url_reports,
        "url_count": len(url_reports),
        "ok_strategy_count": ok_strategy_count,
    }


def pagespeed_report_sync(
    urls: Sequence[str],
    *,
    key: str | None = None,
    cache_root: Path | str | None = None,
    strategies: Sequence[str] = PSI_STRATEGIES,
    use_cache: bool = True,
    store_cache: bool = True,
    timeout_seconds: float = 240.0,
) -> dict[str, Any]:
    """Sync wrapper for :func:`pagespeed_report` (CLI / test convenience)."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                asyncio.run,
                pagespeed_report(
                    urls,
                    key=key,
                    cache_root=cache_root,
                    strategies=strategies,
                    use_cache=use_cache,
                    store_cache=store_cache,
                    timeout_seconds=timeout_seconds,
                ),
            )
            return future.result()
    return asyncio.run(
        pagespeed_report(
            urls,
            key=key,
            cache_root=cache_root,
            strategies=strategies,
            use_cache=use_cache,
            store_cache=store_cache,
            timeout_seconds=timeout_seconds,
        )
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _text(value: object, default: str = "") -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return default
    return str(value)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = _text(value)
    return text if text else None


def _number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value
