"""Immutable exploration domain contracts (stdlib only)."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import cast
from urllib.parse import urlsplit

from ux_analyzer.domain.benchmark import (
    Budget,
    ScenarioEvaluationTarget,
    VisibleResultVerifierSpec,
    canonicalize_http_origin,
)
from ux_analyzer.domain.benchmark import (
    normalize_crawl_url as _benchmark_normalize_crawl_url,
)
from ux_analyzer.domain.benchmark import (
    same_origin as _benchmark_same_origin,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_non_empty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def normalize_crawl_url(value: str) -> str:
    """Canonicalize a crawl URL: HTTPS only, lowercase host, default-port strip,
    collapse //, strip fragment, sort query, remove tracking params.

    Thin wrapper over benchmark.normalize_crawl_url that enforces HTTPS.
    """

    normalized = _benchmark_normalize_crawl_url(value)
    if not normalized.startswith("https://"):
        raise ValueError("URL must use HTTPS")
    return normalized


# Re-export benchmark same_origin for domain consumers (keeps import path stable).
same_origin = _benchmark_same_origin


def _origin_from_normalized_url(normalized_url: str) -> str:
    parsed = urlsplit(normalized_url)
    host = parsed.hostname.lower() if parsed.hostname else ""  # type: ignore[union-attr]
    port = parsed.port
    if port is None or port == 443:
        return f"https://{host}"
    return f"https://{host}:{port}"


_CORPUS_PAGE_FIELDS = (
    "depth",
    "discovered_links",
    "headings",
    "normalized_url",
    "origin",
    "region_labels",
    "screenshot_digest",
    "title",
    "url",
    "viewport_id",
    "visible_elements",
)


def _corpus_page_payload(page: CrawlPage) -> dict[str, object]:
    """Canonical serialized form of one crawled page (fixed field set)."""

    return {
        "depth": page.depth,
        "discovered_links": list(page.discovered_links),
        "headings": list(page.headings),
        "normalized_url": page.normalized_url,
        "origin": page.origin,
        "region_labels": list(page.region_labels),
        "screenshot_digest": page.screenshot_digest,
        "title": page.title,
        "url": page.url,
        "viewport_id": page.viewport_id,
        "visible_elements": list(page.visible_elements),
    }


def compute_corpus_digest(
    pages: Iterable[object],
    link_graph: Mapping[str, object],
) -> str:
    """Single canonical corpus digest over serialized pages and link graph.

    This is the only corpus identity derivation in the system: domain
    ``CrawlCorpus`` instances and storage-layer serialized corpora both hash
    through this function, so identical content always yields an identical
    digest. Content that is not crawl-corpus shaped (missing ``pages`` fields,
    non-string link-graph targets) raises ``ValueError``; it is never hashed
    with a divergent fallback scheme.
    """

    canonical_graph: dict[str, list[str]] = {}
    for key, graph_targets in link_graph.items():
        if isinstance(graph_targets, (str, bytes)):
            raise ValueError("link_graph value must be a collection of strings")
        entries: list[str] = []
        for item in cast(Iterable[object], graph_targets):
            if not isinstance(item, str):
                raise ValueError("link_graph target must be a string")
            entries.append(item)
        canonical_graph[str(key)] = sorted(entries)

    canonical_pages: list[dict[str, object]] = []
    raw_pages: list[Mapping[str, object]] = []
    for entry in pages:
        if not isinstance(entry, Mapping):
            raise ValueError("corpus pages must serialize to objects")
        raw_pages.append(cast(Mapping[str, object], entry))
    ordered = sorted(raw_pages, key=lambda p: str(p.get("normalized_url", "")))
    for page in ordered:
        missing = [name for name in _CORPUS_PAGE_FIELDS if name not in page]
        if missing:
            raise ValueError(
                "corpus page is not crawl-corpus shaped (missing fields: "
                + ", ".join(missing)
                + ")"
            )
        canonical_pages.append({name: page[name] for name in _CORPUS_PAGE_FIELDS})

    canonical = {"link_graph": canonical_graph, "pages": canonical_pages}
    data = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ExplorationStatus(StrEnum):
    """Lifecycle of an exploration attempt."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    COMPLETED = "completed"


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExplorationSpec:
    """Bounded exploration configuration (immutable)."""

    start_urls: tuple[str, ...]
    depth: int = 2
    max_pages: int = 50
    max_scenarios: int = 8
    settle_ms: int = 10000
    allowed_origins: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # start_urls normalization
        raw_starts = self.start_urls
        if isinstance(raw_starts, (str, bytes)):
            raise TypeError("start_urls must be a collection of strings")
        # allow list input
        starts_iter = tuple(raw_starts)  # type: ignore[arg-type]
        if not starts_iter:
            raise ValueError("start_urls must not be empty")
        normalized: list[str] = []
        for raw in starts_iter:
            if type(raw) is not str:
                raise TypeError("start_url must be a string")
            n = normalize_crawl_url(raw)
            normalized.append(n)
        if len(normalized) != len(set(normalized)):
            raise ValueError("start_urls must be unique after normalization")
        object.__setattr__(self, "start_urls", tuple(normalized))

        # bounds
        if type(self.depth) is not int:
            raise TypeError("depth must be an integer")
        if not 0 <= self.depth <= 5:
            raise ValueError("depth must be between 0 and 5")
        if type(self.max_pages) is not int:
            raise TypeError("max_pages must be an integer")
        if not 1 <= self.max_pages <= 200:
            raise ValueError("max_pages must be between 1 and 200")
        if type(self.max_scenarios) is not int:
            raise TypeError("max_scenarios must be an integer")
        if not 1 <= self.max_scenarios <= 20:
            raise ValueError("max_scenarios must be between 1 and 20")
        if type(self.settle_ms) is not int:
            raise TypeError("settle_ms must be an integer")
        if not 0 <= self.settle_ms <= 15000:
            raise ValueError("settle_ms must be between 0 and 15000")

        # depth 0 => max_pages >= len(start_urls)
        if self.depth == 0 and self.max_pages < len(normalized):
            raise ValueError(
                "max_pages must be >= number of start URLs when depth is 0"
            )

        # allowed_origins
        raw_origins = self.allowed_origins
        if isinstance(raw_origins, (str, bytes)):
            raise TypeError("allowed_origins must be a collection of strings")
        origins_iter = tuple(raw_origins)  # type: ignore[arg-type]
        canonical_origins = tuple(canonicalize_http_origin(o) for o in origins_iter)
        if len(canonical_origins) != len(set(canonical_origins)):
            raise ValueError("allowed origins must be unique")
        object.__setattr__(self, "allowed_origins", canonical_origins)


@dataclass(frozen=True, slots=True)
class CrawlPage:
    """One settled page capture (immutable)."""

    url: str
    normalized_url: str
    origin: str
    depth: int
    title: str
    headings: tuple[str, ...]
    viewport_id: str | None = None
    screenshot_digest: str | None = None
    discovered_links: tuple[str, ...] = ()
    visible_elements: tuple[str, ...] = ()
    region_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # url validation
        if type(self.url) is not str or not self.url.strip():
            raise ValueError("url must not be empty")
        # normalized_url must match normalization of url
        expected_normalized = normalize_crawl_url(self.url)
        if self.normalized_url != expected_normalized:
            raise ValueError("normalized_url must match normalized form of url")
        # origin must match derived origin
        expected_origin = _origin_from_normalized_url(self.normalized_url)
        if self.origin != expected_origin:
            raise ValueError("origin must match origin of normalized_url")
        # depth
        if type(self.depth) is not int:
            raise TypeError("depth must be an integer")
        if self.depth < 0:
            raise ValueError("depth must not be negative")
        # title
        if type(self.title) is not str:
            raise TypeError("title must be a string")
        # headings
        if isinstance(self.headings, (str, bytes)):
            raise TypeError("headings must be a collection of strings")
        headings = tuple(self.headings)  # type: ignore[arg-type]
        for h in headings:
            if type(h) is not str:
                raise TypeError("heading must be a string")
        object.__setattr__(self, "headings", headings)
        # viewport_id
        if self.viewport_id is not None:
            _require_non_empty(self.viewport_id, "viewport_id")
        # screenshot_digest
        if self.screenshot_digest is not None:
            if (
                type(self.screenshot_digest) is not str
                or _SHA256_RE.fullmatch(self.screenshot_digest) is None
            ):
                raise ValueError(
                    "screenshot_digest must be a lowercase SHA-256 hex digest"
                )
        # discovered_links
        if isinstance(self.discovered_links, (str, bytes)):
            raise TypeError("discovered_links must be a collection of strings")
        links = tuple(self.discovered_links)  # type: ignore[arg-type]
        for link in links:
            if type(link) is not str or not link.strip():
                raise ValueError("discovered link must not be empty")
        object.__setattr__(self, "discovered_links", links)
        # visible_elements (persona-visible labels, up to 30, truncated downstream)
        if isinstance(self.visible_elements, (str, bytes)):
            raise TypeError("visible_elements must be a collection of strings")
        vels = tuple(self.visible_elements)  # type: ignore[arg-type]
        for ve in vels:
            if type(ve) is not str or not ve.strip():
                raise ValueError("visible element label must not be empty")
        object.__setattr__(self, "visible_elements", vels)
        # region_labels (captured region names, bounded like headings)
        if isinstance(self.region_labels, (str, bytes)):
            raise TypeError("region_labels must be a collection of strings")
        regions = tuple(self.region_labels)  # type: ignore[arg-type]
        for region in regions:
            if type(region) is not str:
                raise TypeError("region label must be a string")
        object.__setattr__(self, "region_labels", regions)


@dataclass(frozen=True, slots=True)
class CrawlCorpus:
    """Immutable collection of crawled pages with deterministic digest."""

    pages: tuple[CrawlPage, ...]
    link_graph: Mapping[str, tuple[str, ...]] = field(default_factory=dict)  # type: ignore[assignment]
    started_at: str = ""
    corpus_digest: str = ""

    def __post_init__(self) -> None:
        # pages
        if isinstance(self.pages, (str, bytes)):
            raise TypeError("pages must be a collection of CrawlPage")
        pages = tuple(self.pages)  # type: ignore[arg-type]
        for p in pages:
            if not isinstance(p, CrawlPage):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError("pages must contain CrawlPage values")
        # unique normalized_url
        seen: set[str] = set()
        for p in pages:
            if p.normalized_url in seen:
                raise ValueError("crawl corpus pages must have unique normalized_url")
            seen.add(p.normalized_url)
        object.__setattr__(self, "pages", pages)

        # link_graph: MappingProxyType with tuple values
        raw_graph = self.link_graph
        if not isinstance(raw_graph, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("link_graph must be a mapping")
        frozen_graph: dict[str, tuple[str, ...]] = {}
        for key, value in raw_graph.items():
            if type(key) is not str or not key.strip():
                raise ValueError("link_graph key must not be empty")
            if isinstance(value, (str, bytes)):
                raise TypeError("link_graph value must be a collection of strings")
            frozen_graph[key] = tuple(value)  # type: ignore[arg-type]
        object.__setattr__(self, "link_graph", MappingProxyType(frozen_graph))

        # started_at
        if type(self.started_at) is not str:
            raise TypeError("started_at must be a string")
        if self.started_at:
            _require_non_empty(self.started_at, "started_at")

        # corpus_digest
        expected = compute_corpus_digest(
            (_corpus_page_payload(page) for page in pages),
            self.link_graph,
        )
        if self.corpus_digest:
            _require_non_empty(self.corpus_digest, "corpus_digest")
            if _SHA256_RE.fullmatch(self.corpus_digest) is None:
                raise ValueError("corpus_digest must be a lowercase SHA-256 digest")
            if self.corpus_digest != expected:
                raise ValueError(
                    "corpus_digest does not match computed digest of pages"
                )
            object.__setattr__(self, "corpus_digest", self.corpus_digest)
        else:
            object.__setattr__(self, "corpus_digest", expected)


@dataclass(frozen=True, slots=True)
class ScenarioSuggestion:
    """LLM-suggested scenario (live-only, visible-result)."""

    id: str
    name: str
    goal: str
    start_url: str
    verifier: VisibleResultVerifierSpec
    evaluation_target: ScenarioEvaluationTarget
    budget: Budget
    rationale: str
    coverage: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.id, "id")
        _require_non_empty(self.name, "name")
        _require_non_empty(self.goal, "goal")
        _require_non_empty(self.rationale, "rationale")
        # start_url normalized HTTPS
        normalized = normalize_crawl_url(self.start_url)
        object.__setattr__(self, "start_url", normalized)
        # verifier must be visible-result
        if not isinstance(self.verifier, VisibleResultVerifierSpec):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("verifier must be a VisibleResultVerifierSpec")
        if getattr(self.verifier, "type", None) != "visible-result":
            raise ValueError("scenario verifier must be visible-result")
        if not self.verifier.text.strip():
            raise ValueError("verifier text must not be empty")
        # evaluation_target
        if not isinstance(self.evaluation_target, ScenarioEvaluationTarget):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("evaluation_target must be a ScenarioEvaluationTarget")
        # budget
        if not isinstance(self.budget, Budget):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("budget must be a Budget")
        # coverage
        if isinstance(self.coverage, (str, bytes)):
            raise TypeError("coverage must be a collection of strings")
        cov = tuple(self.coverage)  # type: ignore[arg-type]
        for entry in cov:
            _require_non_empty(entry, "coverage entry")
        if len(cov) != len(set(cov)):
            raise ValueError("coverage entries must be unique")
        object.__setattr__(self, "coverage", cov)


__all__ = [
    "CrawlCorpus",
    "CrawlPage",
    "ExplorationSpec",
    "ExplorationStatus",
    "ScenarioSuggestion",
    "compute_corpus_digest",
    "normalize_crawl_url",
    "same_origin",
]
