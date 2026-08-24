from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from enum import StrEnum

import pytest

from ux_analyzer.domain.benchmark import (
    Budget,
    FixtureStateVerifierSpec,
    ScenarioEvaluationTarget,
    VerifierOperator,
    VisibleResultVerifierSpec,
)
from ux_analyzer.domain.exploration import (
    CrawlCorpus,
    CrawlPage,
    ExplorationSpec,
    ExplorationStatus,
    ScenarioSuggestion,
    compute_corpus_digest,
    normalize_crawl_url,
    same_origin,
)


def _valid_target() -> ScenarioEvaluationTarget:
    return ScenarioEvaluationTarget(labels_by_version={"live": "Go"}, role="button")


def _valid_verifier() -> VisibleResultVerifierSpec:
    return VisibleResultVerifierSpec(type="visible-result", text="Success")


def _valid_budget() -> Budget:
    return Budget(max_steps=5, max_observations=3, max_interactions=2)


def _make_page(url: str = "https://example.test/", depth: int = 0) -> CrawlPage:
    normalized = normalize_crawl_url(url)
    # derive origin
    from urllib.parse import urlsplit

    parsed = urlsplit(normalized)
    host = parsed.hostname.lower() if parsed.hostname else ""
    origin = f"https://{host}"
    if parsed.port and parsed.port != 443:
        origin = f"https://{host}:{parsed.port}"
    return CrawlPage(
        url=url,
        normalized_url=normalized,
        origin=origin,
        depth=depth,
        title="Example",
        headings=("Heading One",),
        viewport_id=None,
        screenshot_digest=None,
        discovered_links=(),
    )


def test_exploration_spec_depth_zero_allows_single_start() -> None:
    spec = ExplorationSpec(start_urls=("https://example.test/",), depth=0, max_pages=1)
    assert spec.depth == 0
    assert spec.max_pages == 1


def test_crawl_corpus_is_immutable() -> None:
    page = _make_page()
    corpus = CrawlCorpus(
        pages=(page,),
        link_graph={page.normalized_url: ()},
        started_at="2026-08-23T00:00:00Z",
        corpus_digest="",
    )
    with pytest.raises(FrozenInstanceError):
        corpus.pages = ()  # type: ignore[misc]
    # also link_graph immutability via MappingProxyType
    with pytest.raises(TypeError):
        corpus.link_graph["new"] = ()  # type: ignore[index]


def _digest_of(pages: tuple[CrawlPage, ...]) -> str:
    return CrawlCorpus(pages=pages, link_graph={}).corpus_digest


def test_corpus_digest_is_deterministic_and_distinct_per_pages() -> None:
    page = _make_page()
    assert _digest_of((page,)) == _digest_of((page,))
    other = _make_page(url="https://example.test/other")
    assert _digest_of((page,)) != _digest_of((other,))


def _serialized_page(page: CrawlPage) -> dict[str, object]:
    return {
        "depth": page.depth,
        "discovered_links": list(page.discovered_links),
        "headings": list(page.headings),
        "normalized_url": page.normalized_url,
        "origin": page.origin,
        "screenshot_digest": page.screenshot_digest,
        "title": page.title,
        "url": page.url,
        "viewport_id": page.viewport_id,
        "visible_elements": list(page.visible_elements),
    }


def test_compute_corpus_digest_matches_crawl_corpus_identity() -> None:
    page_a = _make_page()
    page_b = _make_page(url="https://example.test/about")
    corpus = CrawlCorpus(
        pages=(page_b, page_a),
        link_graph={page_a.normalized_url: (page_b.normalized_url,)},
    )
    serialized = [_serialized_page(p) for p in (page_a, page_b)]
    graph = {page_a.normalized_url: [page_b.normalized_url]}
    assert compute_corpus_digest(serialized, graph) == corpus.corpus_digest


def test_compute_corpus_digest_rejects_non_page_shaped_pages() -> None:
    with pytest.raises(ValueError, match="crawl-corpus shaped"):
        compute_corpus_digest([{"title": "incomplete"}], {})


def test_compute_corpus_digest_rejects_non_mapping_page() -> None:
    with pytest.raises(ValueError, match="serialize to objects"):
        compute_corpus_digest(["not-a-page"], {})


def test_compute_corpus_digest_rejects_malformed_link_graph() -> None:
    with pytest.raises(ValueError, match="collection of strings"):
        compute_corpus_digest([], {"https://example.test/": "not-a-collection"})
    with pytest.raises(ValueError, match="string"):
        compute_corpus_digest([], {"https://example.test/": (1, 2)})


def test_corpus_digest_changes_when_visible_elements_change() -> None:
    base = _make_page()
    baseline = _digest_of((base,))
    assert _digest_of((replace(base, visible_elements=("Buy Now",)),)) != baseline
    assert (
        _digest_of((replace(base, visible_elements=("Buy Now", "Sign In")),))
        != baseline
    )
    # order matters
    assert _digest_of(
        (replace(base, visible_elements=("Sign In", "Buy Now")),)
    ) != _digest_of((replace(base, visible_elements=("Buy Now", "Sign In")),))


def test_corpus_digest_changes_when_any_page_field_changes() -> None:
    base = _make_page()
    baseline = _digest_of((base,))
    variants: list[CrawlPage] = [
        replace(
            base,
            url="https://example.test/other",
            normalized_url=normalize_crawl_url("https://example.test/other"),
        ),
        replace(
            base,
            normalized_url=normalize_crawl_url("https://example.test/?a=1"),
            url="https://example.test/?a=1",
        ),
        replace(base, depth=1),
        replace(base, title="Other Title"),
        replace(base, headings=("Other Heading",)),
        replace(base, viewport_id="vp-1"),
        replace(base, screenshot_digest="a" * 64),
        replace(base, discovered_links=("https://example.test/link",)),
        replace(base, visible_elements=("Checkout",)),
    ]
    for variant in variants:
        assert variant != base
        assert _digest_of((variant,)) != baseline


def test_exploration_spec_rejects_duplicate_normalized_starts() -> None:
    # https://example.test and https://example.test/ normalize to same? Also case and query sort
    with pytest.raises(ValueError, match="unique"):
        ExplorationSpec(
            start_urls=("https://example.test/", "https://EXAMPLE.test/"),
            depth=1,
            max_pages=10,
        )
    with pytest.raises(ValueError, match="unique"):
        ExplorationSpec(
            start_urls=(
                "https://example.test/?b=2&a=1",
                "https://example.test/?a=1&b=2",
            ),
            depth=1,
            max_pages=10,
        )


def test_exploration_spec_depth_zero_requires_max_pages_coverage() -> None:
    with pytest.raises(ValueError, match="max_pages"):
        ExplorationSpec(
            start_urls=("https://a.test/", "https://a.test/about"),
            depth=0,
            max_pages=1,
        )


def test_exploration_spec_rejects_non_https() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        ExplorationSpec(start_urls=("http://example.test/",), depth=0, max_pages=1)
    with pytest.raises(ValueError, match="HTTPS"):
        ExplorationSpec(start_urls=("ftp://example.test/",), depth=0, max_pages=1)


def test_exploration_spec_max_scenarios_bounds() -> None:
    with pytest.raises(ValueError, match="max_scenarios"):
        ExplorationSpec(
            start_urls=("https://example.test/",),
            depth=1,
            max_pages=10,
            max_scenarios=0,
        )
    with pytest.raises(ValueError, match="max_scenarios"):
        ExplorationSpec(
            start_urls=("https://example.test/",),
            depth=1,
            max_pages=10,
            max_scenarios=21,
        )
    # valid edges
    low = ExplorationSpec(
        start_urls=("https://example.test/",), depth=1, max_pages=10, max_scenarios=1
    )
    high = ExplorationSpec(
        start_urls=("https://example.test/",), depth=1, max_pages=10, max_scenarios=20
    )
    assert low.max_scenarios == 1
    assert high.max_scenarios == 20


def test_scenario_suggestion_requires_visible_result_verifier() -> None:
    target = _valid_target()
    budget = _valid_budget()
    visible = _valid_verifier()
    # valid
    sugg = ScenarioSuggestion(
        id="s1",
        name="Find pricing",
        goal="Locate pricing information",
        start_url="https://example.test/",
        verifier=visible,
        evaluation_target=target,
        budget=budget,
        rationale="Covers pricing discovery",
        coverage=("pricing",),
    )
    assert sugg.verifier.type == "visible-result"
    # invalid fixture-state should be rejected
    fixture_verifier = FixtureStateVerifierSpec(
        type="fixture-state",
        resource="r",
        field="f",
        operator=VerifierOperator.EQUALS,
        expected_fixture_key="k",
    )
    with pytest.raises((ValueError, TypeError)):
        ScenarioSuggestion(
            id="s2",
            name="Bad",
            goal="Bad goal",
            start_url="https://example.test/",
            verifier=fixture_verifier,  # type: ignore[arg-type]
            evaluation_target=target,
            budget=budget,
            rationale="bad",
            coverage=("x",),
        )


def test_normalize_crawl_url_strips_fragment_and_sorts_query_and_removes_tracking() -> (
    None
):
    # fragment stripped
    assert (
        normalize_crawl_url("https://example.test/page#section")
        == "https://example.test/page"
    )
    # query sorted
    assert (
        normalize_crawl_url("https://example.test/p?b=2&a=1")
        == "https://example.test/p?a=1&b=2"
    )
    # tracking removed utm_* and fbclid/gclid
    assert (
        normalize_crawl_url(
            "https://example.test/p?utm_source=x&a=1&fbclid=123&gclid=abc"
        )
        == "https://example.test/p?a=1"
    )
    # combined: fragment + tracking + unsorted
    assert (
        normalize_crawl_url("https://example.test/p?b=2&utm_medium=email&a=1#frag")
        == "https://example.test/p?a=1&b=2"
    )


def test_normalize_crawl_url_lowercases_host_and_removes_default_port_and_collapses_slashes() -> (
    None
):
    assert normalize_crawl_url("https://EXAMPLE.test/") == "https://example.test/"
    assert normalize_crawl_url("https://example.test:443/") == "https://example.test/"
    assert (
        normalize_crawl_url("https://example.test:8443/")
        == "https://example.test:8443/"
    )
    assert (
        normalize_crawl_url("https://example.test//a///b") == "https://example.test/a/b"
    )


def test_same_origin_detection() -> None:
    assert same_origin("https://example.test/a", "https://example.test/b") is True
    assert same_origin("https://example.test:443/a", "https://example.test/b") is True
    assert same_origin("https://EXAMPLE.test/a", "https://example.test/b") is True
    assert same_origin("https://example.test/a", "https://other.test/b") is False
    assert same_origin("https://example.test:8443/a", "https://example.test/b") is False


def test_exploration_spec_validates_allowed_origins_unique() -> None:
    with pytest.raises(ValueError, match="unique"):
        ExplorationSpec(
            start_urls=("https://example.test/",),
            depth=1,
            max_pages=10,
            allowed_origins=("https://example.test", "https://example.test/"),
        )


def test_crawl_corpus_digest_is_deterministic() -> None:
    page = _make_page("https://example.test/")
    corpus1 = CrawlCorpus(
        pages=(page,),
        link_graph={},
        started_at="2026-08-23T00:00:00Z",
        corpus_digest="",
    )
    corpus2 = CrawlCorpus(
        pages=(page,),
        link_graph={},
        started_at="2026-08-23T00:00:00Z",
        corpus_digest="",
    )
    assert corpus1.corpus_digest == corpus2.corpus_digest
    assert len(corpus1.corpus_digest) == 64


def test_corpus_digest_changes_when_link_graph_changes() -> None:
    page = _make_page()
    other = _make_page(url="https://example.test/other")
    flat = CrawlCorpus(pages=(page,), link_graph={}).corpus_digest
    linked = CrawlCorpus(
        pages=(page,),
        link_graph={page.normalized_url: (other.normalized_url,)},
    ).corpus_digest
    assert flat != linked
    # contradictory navigation must not collide
    reversed_graph = CrawlCorpus(
        pages=(page,),
        link_graph={other.normalized_url: (page.normalized_url,)},
    ).corpus_digest
    assert linked != reversed_graph


def test_corpus_digest_is_canonical_over_link_graph_ordering() -> None:
    page = _make_page()
    a = "https://example.test/a"
    b = "https://example.test/b"
    first = CrawlCorpus(pages=(page,), link_graph={page.normalized_url: (a, b)})
    second = CrawlCorpus(pages=(page,), link_graph={page.normalized_url: (b, a)})
    assert first.corpus_digest == second.corpus_digest
    # insertion order of keys does not matter either
    third = CrawlCorpus(
        pages=(page,),
        link_graph={
            b: (page.normalized_url,),
            a: (page.normalized_url,),
        },
    )
    fourth = CrawlCorpus(
        pages=(page,),
        link_graph={
            a: (page.normalized_url,),
            b: (page.normalized_url,),
        },
    )
    assert third.corpus_digest == fourth.corpus_digest


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        (None, True),
        ("a" * 64, True),
        ("0123456789abcdef" * 4, True),
        ("", False),
        ("A" * 64, False),
        ("g" * 64, False),
        ("a" * 63, False),
        ("a" * 65, False),
        ("deadbeef", False),
        (123, False),
    ],
)
def test_screenshot_digest_validation(value: object, valid: bool) -> None:
    page = _make_page()
    if valid:
        replaced = replace(page, screenshot_digest=value)  # type: ignore[arg-type]
        assert replaced.screenshot_digest == value
    else:
        with pytest.raises(ValueError, match="screenshot_digest"):
            replace(page, screenshot_digest=value)  # type: ignore[arg-type]


def test_exploration_status_is_strenum() -> None:
    assert isinstance(ExplorationStatus, type) and issubclass(
        ExplorationStatus, StrEnum
    )
    assert ExplorationStatus.PENDING == "pending"
    assert isinstance(ExplorationStatus.PENDING, str)


def test_exploration_status_enum_values() -> None:
    assert (
        ExplorationStatus("pending") is ExplorationStatus.PENDING
        or ExplorationStatus.PENDING.value == "pending"
    )
    # ensure at least unavailable exists
    assert hasattr(ExplorationStatus, "UNAVAILABLE")
