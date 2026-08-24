from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import pytest

from ux_analyzer.domain.benchmark import (
    Budget,
    ScenarioEvaluationTarget,
    VisibleResultVerifierSpec,
)
from ux_analyzer.domain.exploration import (
    CrawlCorpus,
    CrawlPage,
    ScenarioSuggestion,
    normalize_crawl_url,
)
from ux_analyzer.storage.exploration_artifacts import (
    ExplorationArtifactError,
    ExplorationArtifactStore,
    ExplorationAttemptExistsError,
    exploration_digest,
    recompute_corpus_digest,
)


def _page(url: str, depth: int = 0) -> CrawlPage:
    norm = normalize_crawl_url(url)
    # derive origin
    from urllib.parse import urlsplit

    parsed = urlsplit(norm)
    host = parsed.hostname.lower() if parsed.hostname else ""
    port = parsed.port
    if port is None or port == 443:
        origin = f"https://{host}"
    else:
        origin = f"https://{host}:{port}"
    return CrawlPage(
        url=url,
        normalized_url=norm,
        origin=origin,
        depth=depth,
        title=f"Title for {url}",
        headings=("Heading One", "Heading Two"),
        viewport_id="viewport-1",
        screenshot_digest="a" * 64,
        discovered_links=(),
        visible_elements=("Button Label", "Link Label"),
    )


def _corpus(urls: tuple[str, ...] = ("https://example.test/",)) -> CrawlCorpus:
    pages = tuple(_page(u, depth=0) for u in urls)
    return CrawlCorpus(pages=pages, link_graph={}, started_at="2026-08-23T00:00:00Z")


def _budget() -> Budget:
    return Budget(
        max_steps=10,
        max_observations=10,
        max_interactions=5,
        timeout_seconds=30,
        max_model_calls=10,
    )


def _evaluation_target() -> ScenarioEvaluationTarget:
    return ScenarioEvaluationTarget(
        labels_by_version={"live": "Result Label"}, role="button"
    )


def _verifier(text: str = "Success visible") -> VisibleResultVerifierSpec:
    return VisibleResultVerifierSpec(type="visible-result", text=text)


def _suggestion(id: str, goal: str, start_url: str) -> ScenarioSuggestion:
    return ScenarioSuggestion(
        id=id,
        name=f"Name {id}",
        goal=goal,
        start_url=start_url,
        verifier=_verifier(),
        evaluation_target=_evaluation_target(),
        budget=_budget(),
        rationale="Rationale for scenario",
        coverage=("coverage-a",),
    )


def _sample_suggestions(corpus: CrawlCorpus) -> tuple[ScenarioSuggestion, ...]:
    url = corpus.pages[0].url
    return (
        _suggestion("s1", "Find the invite control", url),
        _suggestion("s2", "Complete onboarding", url),
    )


def _store(tmp_path: Path) -> ExplorationArtifactStore:
    return ExplorationArtifactStore(tmp_path)


def test_write_is_atomic_and_immutable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    corpus = _corpus()
    suggestions = _sample_suggestions(corpus)
    curated = suggestions

    a = store.write_attempt(corpus, suggestions, curated)
    b = store.write_attempt(corpus, suggestions, curated)

    assert a != b
    assert a.is_dir()
    assert b.is_dir()
    # attempt IDs stay distinct via timestamp prefix and/or sequence increment
    assert a.name != b.name
    # second write did not overwrite first
    assert (a / "corpus.json").exists()
    assert (b / "corpus.json").exists()
    # explicit duplicate attempt_id should raise FileExistsError
    with pytest.raises(FileExistsError):
        store.write_attempt(corpus, suggestions, curated, attempt_id=a.name)

    # index should contain both
    idx = store.index  # property returns dict-like
    # handle both callable and dict
    idx_data = idx() if callable(idx) else idx
    attempts = cast(
        list[dict[str, object]], cast(dict[str, object], idx_data)["attempts"]
    )
    ids = {cast(str, rec["attempt_id"]) for rec in attempts}
    assert a.name in ids
    assert b.name in ids


def test_checksum_mismatch_detected(tmp_path: Path) -> None:
    store = _store(tmp_path)

    def _rewrite_json(path: Path, mutate: Callable[[object], None]) -> None:
        data = json.loads(path.read_bytes())
        mutate(data)
        path.write_bytes(
            (
                json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                + "\n"
            ).encode("ascii")
        )

    corpus = _corpus()
    suggestions = _sample_suggestions(corpus)
    a = store.write_attempt(corpus, suggestions, suggestions)
    original_corpus_bytes = (a / "corpus.json").read_bytes()

    def _tamper_page_title(payload: object) -> None:
        assert isinstance(payload, dict)
        pages = cast(list[dict[str, object]], payload["pages"])
        pages[0]["title"] = "Tampered Title"

    # canonical JSON whose bytes no longer match the manifest digests
    _rewrite_json(a / "corpus.json", _tamper_page_title)
    with pytest.raises(ExplorationArtifactError):
        store.load_attempt(a.name)

    # restore the verified bytes so later writes see an intact store
    (a / "corpus.json").write_bytes(original_corpus_bytes)

    corpus2 = _corpus(("https://example.test/other",))
    sugg2 = _sample_suggestions(corpus2)
    b = store.write_attempt(corpus2, sugg2, sugg2)
    original_suggestions_bytes = (b / "suggestions.json").read_bytes()

    def _tamper_goal(payload: object) -> None:
        assert isinstance(payload, list)
        first = cast(dict[str, object], payload[0])
        first["goal"] = "tampered goal"

    _rewrite_json(b / "suggestions.json", _tamper_goal)
    with pytest.raises(ExplorationArtifactError):
        store.load_attempt(b.name)

    # A tampered bundle must block publishing loudly: the index rebuild
    # refuses to launder corruption into a clean-looking index and names the
    # offending attempt. Evidence stays in place, untouched.
    corpus3 = _corpus(("https://example.test/third",))
    sugg3 = _sample_suggestions(corpus3)
    with pytest.raises(ExplorationArtifactError, match="rebuild refused") as info:
        store.write_attempt(corpus3, sugg3, sugg3)
    assert b.name in str(info.value)
    # tampered evidence was neither deleted nor moved aside
    assert b.exists()
    on_disk = {
        child.name for child in (tmp_path / "exploration" / "attempts").iterdir()
    }
    assert not any(name.startswith(".quarantine-") for name in on_disk)
    # listing now fails too: every read path verifies payload digests
    with pytest.raises(ExplorationArtifactError, match="suggestions digest mismatch"):
        store.get_index()

    # repairing the bundle restores all read paths without touching the index
    (b / "suggestions.json").write_bytes(original_suggestions_bytes)
    loaded = store.load_attempt(b.name)
    assert loaded["attempt_id"] == b.name
    idx_data = store.index() if callable(store.index) else store.index
    ids = {rec["attempt_id"] for rec in idx_data["attempts"]}  # type: ignore[union-attr]
    assert {a.name, b.name} <= ids


def test_traversal_rejection(tmp_path: Path) -> None:
    store = _store(tmp_path)
    corpus = _corpus()
    suggestions = _sample_suggestions(corpus)
    # write one valid attempt first
    a = store.write_attempt(corpus, suggestions, suggestions)
    assert a.exists()

    # attempt to load with traversal
    for bad_id in (
        "../evil",
        "foo/bar",
        "..\\evil",
        "evil/../traversal",
        "/absolute",
        "attempt/../../etc/passwd",
    ):
        with pytest.raises((ExplorationArtifactError, ValueError)):
            store.load_attempt(bad_id)
        # also write with traversal should be rejected
        with pytest.raises((ExplorationArtifactError, ValueError)):
            store.write_attempt(corpus, suggestions, suggestions, attempt_id=bad_id)

    # also check that attempting to use path traversal via attempt_id containing slash is rejected as malformed
    with pytest.raises(ExplorationArtifactError):
        store.load_attempt("20260810T120000Z-abc123def456-1/../../evil")


def test_malformed_id_rejection(tmp_path: Path) -> None:
    store = _store(tmp_path)
    corpus = _corpus()
    suggestions = _sample_suggestions(corpus)

    valid = store.write_attempt(corpus, suggestions, suggestions)
    assert valid.exists()

    malformed_ids = [
        "attempt-1",
        "20260810T120000Z-not-a-digest-1",
        "20260810T120000Z-deadbeefdead",  # missing seq
        "20260810T120000Z-deadbeefdead-0",  # seq 0 invalid
        "20260810T120000Z-deadbeefdead-01",  # leading zero? pattern requires [1-9][0-9]*, so 01 invalid? but we treat as malformed
        "not-a-timestamp-deadbeefdead-1",
        "",
        "2026-13-40T999999Z-deadbeefdead-1",
    ]
    for bad in malformed_ids:
        with pytest.raises((ExplorationArtifactError, ValueError)):
            store.load_attempt(bad)
        with pytest.raises((ExplorationArtifactError, ValueError)):
            store.write_attempt(corpus, suggestions, suggestions, attempt_id=bad)


def test_digest_stability(tmp_path: Path) -> None:
    store = _store(tmp_path)
    corpus = _corpus()
    # digest via store method
    d1 = store.digest(corpus)
    d2 = store.digest(corpus)
    assert d1 == d2
    assert len(d1) == 64
    # digest via module helper
    d3 = exploration_digest(corpus)
    assert d1 == d3

    # canonical JSON stability: same logical corpus with different in-memory order should give same digest
    # Create two corpora with same pages but constructed separately
    corpus_a = _corpus(("https://example.test/", "https://example.test/about"))
    corpus_b = _corpus(("https://example.test/about", "https://example.test/"))
    # Note: CrawlCorpus sorts pages by normalized_url for digest, so both should have same digest despite input order
    # Our store digest uses canonical bytes of the dataclass serialization which may not sort pages, but exploration_digest via corpus_digest field should be stable
    # Check that CrawlCorpus digest is stable
    assert corpus_a.corpus_digest == corpus_b.corpus_digest

    # Also check that suggestions digest stable
    sugg = _sample_suggestions(corpus)
    s1 = store.digest(sugg)
    s2 = store.digest(sugg)
    assert s1 == s2

    # Different corpus should give different digest
    corpus_diff = _corpus(("https://example.test/different",))
    d_diff = store.digest(corpus_diff)
    assert d_diff != d1


def test_index_and_manifest_structure(tmp_path: Path) -> None:
    store = _store(tmp_path)
    corpus = _corpus()
    suggestions = _sample_suggestions(corpus)
    curated = suggestions[:1]
    a = store.write_attempt(
        corpus,
        suggestions,
        curated,
        spec={"depth": 2},
        model="test-model",
        prompt_version="exploration-synthesis-v1",
    )
    # check manifest
    manifest_path = a / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_bytes())
    assert "digests" in manifest
    assert "spec" in manifest
    assert "model" in manifest
    assert "prompt_version" in manifest
    assert "created_at" in manifest
    assert manifest["model"] == "test-model"
    assert manifest["prompt_version"] == "exploration-synthesis-v1"
    # check corpus etc.
    for fname in [
        "corpus.json",
        "suggestions.json",
        "curated.json",
        "project.fragment.yaml",
        "manifest.json",
    ]:
        assert (a / fname).is_file()
    # check corpus.json is canonical JSON
    corpus_bytes = (a / "corpus.json").read_bytes()
    corpus_val = json.loads(corpus_bytes)
    # canonical check: re-serialize should equal
    canonical = (
        json.dumps(corpus_val, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    assert corpus_bytes == canonical

    # index
    idx_data = store.index() if callable(store.index) else store.index
    if callable(idx_data):
        idx_data = idx_data()
    idx_mapping = cast(dict[str, object], idx_data)
    index_records = cast(list[dict[str, object]], idx_mapping["attempts"])
    assert idx_mapping["schema_version"] == "exploration-index-v1"
    assert any(rec["attempt_id"] == a.name for rec in index_records)
    rec = next(r for r in index_records if r["attempt_id"] == a.name)
    assert rec["corpus_digest"] == manifest["corpus_digest"]
    assert rec["page_count"] == len(corpus.pages)
    assert rec["scenario_count"] == len(curated)


def test_not_overwritable_explicit(tmp_path: Path) -> None:
    store = _store(tmp_path)
    corpus = _corpus()
    sugg = _sample_suggestions(corpus)
    a = store.write_attempt(corpus, sugg, sugg)
    # Try to write again with same attempt_id explicitly -> FileExistsError
    with pytest.raises(FileExistsError):
        store.write_attempt(corpus, sugg, sugg, attempt_id=a.name)
    # Ensure original intact
    assert (a / "corpus.json").exists()
    loaded = store.load_attempt(a.name)
    assert loaded["attempt_id"] == a.name


def test_load_attempt_traversal_and_malformed_combined(tmp_path: Path) -> None:
    store = _store(tmp_path)
    corpus = _corpus()
    sugg = _sample_suggestions(corpus)
    store.write_attempt(corpus, sugg, sugg)
    # Test path traversal via load_attempt with encoded traversal
    with pytest.raises((ExplorationArtifactError, ValueError)):
        store.load_attempt("../20260810T120000Z-deadbeefdead-1")
    with pytest.raises((ExplorationArtifactError, ValueError)):
        store.load_attempt("20260810T120000Z-deadbeefdead-1/evil")


# ---------------------------------------------------------------------------
# Canonical digest unification: domain == storage, no fallback hashing
# ---------------------------------------------------------------------------


def _page_mapping(page: CrawlPage) -> dict[str, object]:
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


def _corpus_mapping(
    corpus: CrawlCorpus, *, corpus_digest: str | None = None
) -> dict[str, object]:
    payload: dict[str, object] = {
        "pages": [_page_mapping(page) for page in corpus.pages],
        "link_graph": {k: list(v) for k, v in corpus.link_graph.items()},
        "started_at": corpus.started_at,
    }
    if corpus_digest is not None:
        payload["corpus_digest"] = corpus_digest
    return payload


def _published_attempt_ids(root: Path) -> set[str]:
    attempts_root = root / "exploration" / "attempts"
    if not attempts_root.exists():
        return set()
    return {
        child.name
        for child in attempts_root.iterdir()
        if not child.name.startswith(".")
    }


def test_storage_recompute_matches_domain_identity_for_same_content(
    tmp_path: Path,
) -> None:
    corpus = _corpus(("https://example.test/a", "https://example.test/b"))
    mapping = _corpus_mapping(corpus)
    assert recompute_corpus_digest(mapping) == corpus.corpus_digest
    # page order in the serialized mapping must not matter
    reversed_mapping = _corpus_mapping(corpus)
    pages = cast(list[object], reversed_mapping["pages"])
    pages.reverse()
    assert recompute_corpus_digest(reversed_mapping) == corpus.corpus_digest
    # a CrawlCorpus object written through the store carries the same identity
    object_root = tmp_path / "object-write"
    store = _store(object_root)
    attempt = store.write_attempt(corpus, (), ())
    manifest = json.loads((attempt / "manifest.json").read_bytes())
    assert manifest["corpus_digest"] == corpus.corpus_digest


def test_declared_digest_mismatch_refused_by_hmac_gate(tmp_path: Path) -> None:
    store = _store(tmp_path)
    corpus = _corpus()
    sugg = _sample_suggestions(corpus)
    forged = _corpus_mapping(corpus, corpus_digest="0" * 64)
    with pytest.raises(ExplorationArtifactError, match="does not match"):
        store.write_attempt(forged, sugg, sugg)
    assert _published_attempt_ids(tmp_path) == set()


def test_matching_declared_digest_publishes_with_domain_identity(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    corpus = _corpus()
    sugg = _sample_suggestions(corpus)
    declared = _corpus_mapping(corpus, corpus_digest=corpus.corpus_digest)
    attempt = store.write_attempt(declared, sugg, sugg)
    manifest = json.loads((attempt / "manifest.json").read_bytes())
    assert manifest["corpus_digest"] == corpus.corpus_digest
    loaded = store.load_attempt(attempt.name)
    assert loaded["attempt_id"] == attempt.name


_NON_PAGE_PAYLOADS: list[dict[str, object]] = [
    {"foo": "bar"},
    {"pages": "not-a-list"},
    {"pages": [{"title": "incomplete page"}]},
    {
        "pages": [
            {
                "depth": 0,
                "discovered_links": [],
                "headings": [],
                "normalized_url": "https://example.test/",
                "origin": "https://example.test",
                "screenshot_digest": None,
                "title": "Example",
                "url": "https://example.test/",
                "viewport_id": None,
                "visible_elements": [],
            }
        ],
        "link_graph": {"https://example.test/": "not-a-collection"},
    },
]


@pytest.mark.parametrize("payload", _NON_PAGE_PAYLOADS)
def test_non_page_shaped_corpus_rejected_not_hashed_differently(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    store = _store(tmp_path)
    with pytest.raises(ExplorationArtifactError):
        store.write_attempt(payload, (), ())
    assert _published_attempt_ids(tmp_path) == set()


def test_unknown_status_rejected_loudly(tmp_path: Path) -> None:
    store = _store(tmp_path)
    corpus = _corpus()
    sugg = _sample_suggestions(corpus)
    with pytest.raises(ExplorationArtifactError, match="unknown exploration status"):
        store.write_attempt(corpus, sugg, sugg, status="bogus-status")
    assert _published_attempt_ids(tmp_path) == set()


def test_write_attempt_rejects_unknown_kwargs(tmp_path: Path) -> None:
    store = _store(tmp_path)
    corpus = _corpus()
    sugg = _sample_suggestions(corpus)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        store.write_attempt(corpus, sugg, sugg, totally_unknown_kwarg=1)


def test_write_attempt_alias_kwargs_still_supported(tmp_path: Path) -> None:
    store = _store(tmp_path)
    corpus = _corpus()
    attempt = store.write_attempt(corpus, (), (), model="m-1", status="failed")
    manifest = json.loads((attempt / "manifest.json").read_bytes())
    assert manifest["model"] == "m-1"
    assert manifest["status"] == "failed"


def test_write_requires_explicit_curated_no_auto_accept(tmp_path: Path) -> None:
    """The store never invents a curated set; callers decide auto-accept."""

    store = _store(tmp_path)
    corpus = _corpus()
    sugg = _sample_suggestions(corpus)

    with pytest.raises(ExplorationArtifactError, match="curated is required"):
        store.write_attempt(corpus)
    with pytest.raises(ExplorationArtifactError, match="curated is required"):
        store.write_attempt(corpus, sugg)
    assert _published_attempt_ids(tmp_path) == set()

    # an explicit empty curated set is honored, not replaced by suggestions
    attempt = store.write_attempt(corpus, sugg, ())
    loaded = store.load_attempt(attempt.name)
    assert loaded["curated"] == []
    assert len(cast(list[object], loaded["suggestions"])) == 2
    manifest = json.loads((attempt / "manifest.json").read_bytes())
    assert manifest["scenario_count"] == 0


def test_write_rejects_coerced_persona_and_spec_payloads(tmp_path: Path) -> None:
    """Typed inputs only: no str(persona_set) or {"value": ...} invention."""

    store = _store(tmp_path)
    corpus = _corpus()

    with pytest.raises(ExplorationArtifactError, match="persona_set"):
        store.write_attempt(corpus, (), (), persona_set="legacy-string-persona")
    with pytest.raises(ExplorationArtifactError, match="persona_set"):
        store.write_attempt(corpus, (), (), persona_set=42)
    with pytest.raises(ExplorationArtifactError, match="spec"):
        store.write_attempt(corpus, (), (), spec="raw spec text")
    assert _published_attempt_ids(tmp_path) == set()

    attempt = store.write_attempt(
        corpus,
        (),
        (),
        persona_set=({"id": "p1", "name": "Primary"},),
        spec={"depth": 2},
    )
    manifest = json.loads((attempt / "manifest.json").read_bytes())
    assert manifest["persona_set"] == [{"id": "p1", "name": "Primary"}]
    assert manifest["spec"] == {"depth": 2}


def test_fragment_never_mints_placeholder_scenario_ids(tmp_path: Path) -> None:
    """Curated entries without ids are refused, not given UUID stand-ins."""

    store = _store(tmp_path)
    corpus = _corpus()

    with pytest.raises(ExplorationArtifactError):
        store.write_attempt(corpus, (), ({"goal": "no id here"},))
    with pytest.raises(ExplorationArtifactError):
        store.write_attempt(corpus, (), ("raw string scenario",))
    assert _published_attempt_ids(tmp_path) == set()


def test_duplicate_publish_raises_documented_exists_error(tmp_path: Path) -> None:
    """Attempt-exists refusals are ExplorationArtifactError + FileExistsError."""

    store = _store(tmp_path)
    corpus = _corpus()
    sugg = _sample_suggestions(corpus)
    a = store.write_attempt(corpus, sugg, sugg)
    with pytest.raises(ExplorationAttemptExistsError) as info:
        store.write_attempt(corpus, sugg, sugg, attempt_id=a.name)
    assert isinstance(info.value, FileExistsError)
    assert isinstance(info.value, ExplorationArtifactError)


# ---------------------------------------------------------------------------
# Concurrency: publication locking must serialize competing writers
# ---------------------------------------------------------------------------


_CONCURRENT_WRITERS = 8
_PUBLISH_TIMEOUT_SECONDS = 120


def test_concurrent_publishers_distinct_corpora_all_survive(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    corpora = [
        _corpus((f"https://example.test/writer-{index}/",))
        for index in range(_CONCURRENT_WRITERS)
    ]
    stores = [ExplorationArtifactStore(root) for _ in range(_CONCURRENT_WRITERS)]

    def publish(index: int) -> Path:
        corpus = corpora[index]
        suggestions = _sample_suggestions(corpus)
        return stores[index].write_attempt(corpus, suggestions, suggestions)

    with ThreadPoolExecutor(max_workers=_CONCURRENT_WRITERS) as executor:
        published = list(
            executor.map(
                publish, range(_CONCURRENT_WRITERS), timeout=_PUBLISH_TIMEOUT_SECONDS
            )
        )

    names = {path.name for path in published}
    # zero lost attempts, zero ID collisions
    assert len(published) == _CONCURRENT_WRITERS
    assert len(names) == _CONCURRENT_WRITERS
    for path in published:
        assert path.is_dir()
        assert (path / "corpus.json").is_file()
    attempts_root = root / "exploration" / "attempts"
    assert {child.name for child in attempts_root.iterdir()} == names

    final_store = ExplorationArtifactStore(root)
    final_records = cast(
        list[dict[str, object]],
        cast(dict[str, object], final_store.index())["attempts"],
    )
    ids = {cast(str, rec["attempt_id"]) for rec in final_records}
    assert ids == names
    # every surviving bundle validates end-to-end after the race
    for name in names:
        loaded = final_store.load_attempt(name)
        assert loaded["attempt_id"] == name


def test_concurrent_same_attempt_id_publish_exactly_one_winner(
    tmp_path: Path,
) -> None:
    root = tmp_path / "shared"
    seed_store = ExplorationArtifactStore(root)
    corpus = _corpus(("https://example.test/race/",))
    suggestions = _sample_suggestions(corpus)
    seed = seed_store.write_attempt(corpus, suggestions, suggestions)
    seed_manifest = json.loads((seed / "manifest.json").read_bytes())
    created_token, digest_prefix, sequence_text = seed.name.rsplit("-", 2)
    # next free sequence for the same (timestamp token, corpus digest) pair;
    # every racer targets this exact ID simultaneously
    race_id = f"{created_token}-{digest_prefix}-{int(sequence_text) + 1}"

    winners: list[str] = []
    losers: list[Exception] = []

    def publish(_: int) -> None:
        store = ExplorationArtifactStore(root)
        try:
            attempt = store.write_attempt(
                corpus,
                suggestions,
                suggestions,
                attempt_id=race_id,
                created_at=cast(str, seed_manifest["created_at"]),
            )
        except Exception as error:  # noqa: BLE001 - classified below
            losers.append(error)
        else:
            winners.append(attempt.name)

    with ThreadPoolExecutor(max_workers=_CONCURRENT_WRITERS) as executor:
        list(executor.map(publish, range(_CONCURRENT_WRITERS), timeout=_PUBLISH_TIMEOUT_SECONDS))

    # exactly one winner; every loser refused with FileExistsError
    assert winners == [race_id]
    assert len(losers) == _CONCURRENT_WRITERS - 1
    assert all(isinstance(error, FileExistsError) for error in losers)

    # no partial or staging directories survive the race
    attempts_root = root / "exploration" / "attempts"
    on_disk = {child.name for child in attempts_root.iterdir()}
    assert on_disk == {seed.name, race_id}
    assert (attempts_root / race_id / "corpus.json").is_file()
    loaded = ExplorationArtifactStore(root).load_attempt(race_id)
    assert loaded["attempt_id"] == race_id


# ---------------------------------------------------------------------------
# Fix pins: verified-byte reads, schema gate, quarantine, read transactions
# ---------------------------------------------------------------------------


def _rewrite_json(path: Path, mutate: Callable[[object], None]) -> None:
    data = json.loads(path.read_bytes())
    mutate(data)
    path.write_bytes(
        (
            json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            + "\n"
        ).encode("ascii")
    )


def test_load_attempt_returns_exactly_verified_bytes(tmp_path: Path) -> None:
    """load_attempt must parse the same bytes whose digests it verified."""

    store = _store(tmp_path)
    corpus = _corpus()
    sugg = _sample_suggestions(corpus)
    a = store.write_attempt(corpus, sugg, sugg)

    reads: list[str] = []

    class _CountingStore(ExplorationArtifactStore):
        def _read_json_object(
            self, path: Path, label: str
        ) -> tuple[dict[str, object] | list[object], bytes]:
            reads.append(path.name)
            return super()._read_json_object(path, label)

    counting = _CountingStore(tmp_path)
    loaded = counting.load_attempt(a.name)

    # exactly one JSON read per bundle file plus the trusted index record;
    # no second unverified re-read of any bundle file. The index read feeds
    # the mandatory record-vs-manifest cross-check before returning.
    assert sorted(reads) == [
        "corpus.json",
        "curated.json",
        "index.json",
        "manifest.json",
        "suggestions.json",
    ]

    manifest_bytes = (a / "manifest.json").read_bytes()
    corpus_bytes = (a / "corpus.json").read_bytes()
    suggestions_bytes = (a / "suggestions.json").read_bytes()
    curated_bytes = (a / "curated.json").read_bytes()
    assert loaded["manifest"] == json.loads(manifest_bytes)
    assert loaded["corpus"] == json.loads(corpus_bytes)
    assert loaded["suggestions"] == json.loads(suggestions_bytes)
    assert loaded["curated"] == json.loads(curated_bytes)
    assert isinstance(loaded["fragment"], dict)


def test_unsupported_manifest_schema_version_rejected_on_read(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    corpus = _corpus()
    sugg = _sample_suggestions(corpus)
    a = store.write_attempt(corpus, sugg, sugg)
    original_manifest_bytes = (a / "manifest.json").read_bytes()

    def _bump_schema(payload: object) -> None:
        assert isinstance(payload, dict)
        payload["schema_version"] = "exploration-artifact-v0"

    _rewrite_json(a / "manifest.json", _bump_schema)
    with pytest.raises(ExplorationArtifactError, match="unsupported.*schema version"):
        store.load_attempt(a.name)

    # restore verified bytes so the later publish sees an intact store
    (a / "manifest.json").write_bytes(original_manifest_bytes)

    # unknown future version is equally refused on read
    def _future_schema(payload: object) -> None:
        assert isinstance(payload, dict)
        payload["schema_version"] = "exploration-artifact-v999"

    future_corpus = _corpus(("https://example.test/future/",))
    future_sugg = _sample_suggestions(future_corpus)
    b = store.write_attempt(future_corpus, future_sugg, future_sugg)
    _rewrite_json(b / "manifest.json", _future_schema)
    with pytest.raises(ExplorationArtifactError, match="unsupported.*schema version"):
        store.load_attempt(b.name)


def test_rebuild_surfaces_corruption_instead_of_laundering(
    tmp_path: Path,
) -> None:
    """Index rebuild must refuse to publish a sanitized-clean index."""

    store = _store(tmp_path)
    corpus = _corpus()
    sugg = _sample_suggestions(corpus)
    a = store.write_attempt(corpus, sugg, sugg)
    original_corpus_bytes = (a / "corpus.json").read_bytes()

    def _tamper_goal(payload: object) -> None:
        assert isinstance(payload, list)
        first = cast(dict[str, object], payload[0])
        first["goal"] = "tampered goal"

    _rewrite_json(a / "suggestions.json", _tamper_goal)

    with pytest.raises(ExplorationArtifactError, match="rebuild refused") as info:
        store.write_attempt(
            _corpus(("https://example.test/rebuild/",)),
            _sample_suggestions(_corpus(("https://example.test/rebuild/",))),
            _sample_suggestions(_corpus(("https://example.test/rebuild/",))),
        )
    assert a.name in str(info.value)

    attempts_root = tmp_path / "exploration" / "attempts"
    on_disk = {child.name for child in attempts_root.iterdir()}
    # evidence preserved byte-for-byte in place; no quarantine sidecars
    assert a.name in on_disk
    assert (a / "corpus.json").read_bytes() == original_corpus_bytes
    assert not any(name.startswith(".quarantine-") for name in on_disk)

    # the published index was never rewritten to drop the corrupt record
    index_payload = _read_index_payload(tmp_path)
    assert a.name in {
        cast(str, rec["attempt_id"])
        for rec in cast(list[dict[str, object]], index_payload["attempts"])
    }
    # and listing refuses the corrupt bundle loudly
    with pytest.raises(ExplorationArtifactError, match="suggestions digest mismatch"):
        store.get_index()


def test_torn_read_retries_until_snapshot_is_stable(tmp_path: Path) -> None:
    """A mid-read mutation forces an optimistic retry instead of torn output."""

    store = _store(tmp_path)
    corpus = _corpus()
    sugg = _sample_suggestions(corpus)
    a = store.write_attempt(corpus, sugg, sugg)

    # Writers keep .publication.lock forever once created; drop it so the
    # read takes the optimistic (lockless) path where retries live.
    lock_path = tmp_path / "exploration" / ".publication.lock"
    assert lock_path.exists()
    lock_path.unlink()

    mutated = {"done": False}
    clone_name = "-".join((*a.name.rsplit("-", 2)[:2], "99"))

    class _MutatingMidReadStore(ExplorationArtifactStore):
        def _optimistic_read_snapshot(
            self, lock_path: Path
        ) -> tuple[bool, tuple[str, ...], str | None]:
            snapshot = super()._optimistic_read_snapshot(lock_path)
            if not mutated["done"]:
                mutated["done"] = True
                # mutate between the before/after snapshots by cloning a
                # valid attempt under a fresh unused sequence
                src = tmp_path / "exploration" / "attempts" / a.name
                dst = tmp_path / "exploration" / "attempts" / clone_name
                shutil.copytree(src, dst)
            return snapshot

    racing = _MutatingMidReadStore(tmp_path)
    loaded = racing.load_attempt(a.name)
    assert mutated["done"]
    assert loaded["attempt_id"] == a.name
    assert loaded["corpus"] == json.loads((a / "corpus.json").read_bytes())

    # clone landed mid-read; index still validates against surviving bundles
    attempts_root = tmp_path / "exploration" / "attempts"
    assert (attempts_root / clone_name / "manifest.json").is_file()
    ids = {
        cast(str, rec["attempt_id"])
        for rec in cast(list[dict[str, object]], racing.index()["attempts"])
    }
    assert a.name in ids


# ---------------------------------------------------------------------------
# Index-record verification on read paths (byte-level trust)
# ---------------------------------------------------------------------------


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _index_path(root: Path) -> Path:
    return root / "exploration" / "index.json"


def _read_index_payload(root: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(_index_path(root).read_bytes()))


def _write_index_payload(root: Path, payload: object) -> None:
    _index_path(root).write_bytes(_canonical_json_bytes(payload))


def _record_for(payload: dict[str, object], attempt_id: str) -> dict[str, object]:
    records = cast(list[dict[str, object]], payload["attempts"])
    return next(r for r in records if r["attempt_id"] == attempt_id)


def test_tampered_bundle_with_intact_index_record_rejected(tmp_path: Path) -> None:
    """Self-consistent bundles must still match the bytes their index pins."""

    store = _store(tmp_path)
    corpus = _corpus()
    sugg = _sample_suggestions(corpus)
    a = store.write_attempt(corpus, sugg, sugg)

    # Forge an internally consistent bundle: tamper the corpus payload and
    # re-sign the manifest digests over the tampered bytes. Every
    # manifest-level check passes; only the index-record cross-check can
    # detect that the published manifest bytes were replaced.
    forged_corpus = json.loads((a / "corpus.json").read_bytes())
    pages = cast(list[dict[str, object]], forged_corpus["pages"])
    pages[0]["title"] = "Forged Title"
    corpus_bytes = _canonical_json_bytes(forged_corpus)
    (a / "corpus.json").write_bytes(corpus_bytes)

    manifest = json.loads((a / "manifest.json").read_bytes())
    digests = cast(dict[str, str], manifest["digests"])
    digests["corpus"] = hashlib.sha256(corpus_bytes).hexdigest()
    (a / "manifest.json").write_bytes(_canonical_json_bytes(manifest))

    # the index was never touched: it still pins the original manifest bytes
    with pytest.raises(ExplorationArtifactError, match="manifest digest mismatch"):
        store.load_attempt(a.name)
    with pytest.raises(ExplorationArtifactError, match="manifest digest mismatch"):
        store.get_index()


def test_listing_detects_payload_corruption_without_resign(tmp_path: Path) -> None:
    """Listing verifies manifest digests against actual bytes like load."""

    store = _store(tmp_path)
    corpus = _corpus()
    sugg = _sample_suggestions(corpus)
    a = store.write_attempt(corpus, sugg, sugg)
    original_corpus_bytes = (a / "corpus.json").read_bytes()

    # corrupt the corpus payload only: manifest and index are untouched
    def _tamper_title(payload: object) -> None:
        assert isinstance(payload, dict)
        pages = cast(list[dict[str, object]], payload["pages"])
        pages[0]["title"] = "Silently Corrupted"

    _rewrite_json(a / "corpus.json", _tamper_title)
    with pytest.raises(ExplorationArtifactError, match="corpus digest mismatch"):
        store.get_index()
    with pytest.raises(ExplorationArtifactError, match="corpus digest mismatch"):
        store.index()

    # fragment payloads are equally verified on listing
    (a / "corpus.json").write_bytes(original_corpus_bytes)
    frag_corpus = _corpus(("https://example.test/frag/",))
    frag_sugg = _sample_suggestions(frag_corpus)
    b = store.write_attempt(frag_corpus, frag_sugg, frag_sugg)
    fragment_path = b / "project.fragment.yaml"
    fragment_path.write_bytes(fragment_path.read_bytes() + b"# trailing comment\n")
    with pytest.raises(ExplorationArtifactError, match="fragment digest mismatch"):
        store.get_index()


def test_semantically_invalid_curated_with_consistent_digests_rejected(
    tmp_path: Path,
) -> None:
    """Digest-consistent payloads must still be semantically sane."""

    store = _store(tmp_path)
    corpus = _corpus()
    sugg = _sample_suggestions(corpus)

    # publishing duplicate-id content is refused outright
    with pytest.raises(ExplorationArtifactError, match="duplicate scenario id"):
        store.write_attempt(
            _corpus(("https://example.test/semantics/",)),
            (sugg[0], sugg[0]),
            (sugg[0], sugg[0]),
        )

    a = store.write_attempt(corpus, sugg, sugg)

    # Forge a fully consistent state around a semantically broken curated
    # list: duplicate scenario id, re-signed manifest, re-pinned index.
    curated = json.loads((a / "curated.json").read_bytes())
    curated.append(json.loads(json.dumps(curated[0])))
    curated_bytes = _canonical_json_bytes(curated)
    (a / "curated.json").write_bytes(curated_bytes)

    manifest = json.loads((a / "manifest.json").read_bytes())
    digests = cast(dict[str, str], manifest["digests"])
    digests["curated"] = hashlib.sha256(curated_bytes).hexdigest()
    manifest_bytes = _canonical_json_bytes(manifest)
    (a / "manifest.json").write_bytes(manifest_bytes)

    payload = _read_index_payload(tmp_path)
    _record_for(payload, a.name)["manifest_digest"] = hashlib.sha256(
        manifest_bytes
    ).hexdigest()
    _write_index_payload(tmp_path, payload)

    with pytest.raises(ExplorationArtifactError, match="duplicate scenario id"):
        store.load_attempt(a.name)


def test_legacy_index_schema_version_still_loads(tmp_path: Path) -> None:
    """Legacy indexes predate byte-level pinning and remain readable."""

    store = _store(tmp_path)
    corpus = _corpus()
    sugg = _sample_suggestions(corpus)
    a = store.write_attempt(corpus, sugg, sugg)

    payload = _read_index_payload(tmp_path)
    payload["schema_version"] = "exploration-index-v0"
    for record in cast(list[dict[str, object]], payload["attempts"]):
        record.pop("manifest_digest", None)
    _write_index_payload(tmp_path, payload)

    loaded = store.load_attempt(a.name)
    assert loaded["attempt_id"] == a.name
    listing = store.get_index()
    assert listing["schema_version"] == "exploration-index-v0"
    ids = {
        cast(str, rec["attempt_id"])
        for rec in cast(list[dict[str, object]], listing["attempts"])
    }
    assert a.name in ids


def test_legacy_index_with_present_but_wrong_manifest_digest_rejected(
    tmp_path: Path,
) -> None:
    """Legacy records that do carry a pin get it verified like any other."""

    store = _store(tmp_path)
    corpus = _corpus()
    sugg = _sample_suggestions(corpus)
    a = store.write_attempt(corpus, sugg, sugg)

    payload = _read_index_payload(tmp_path)
    payload["schema_version"] = "exploration-index-v0"
    _record_for(payload, a.name)["manifest_digest"] = "0" * 64
    _write_index_payload(tmp_path, payload)

    with pytest.raises(ExplorationArtifactError, match="manifest digest mismatch"):
        store.load_attempt(a.name)


def test_unknown_future_index_schema_version_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    corpus = _corpus()
    sugg = _sample_suggestions(corpus)
    a = store.write_attempt(corpus, sugg, sugg)

    payload = _read_index_payload(tmp_path)
    payload["schema_version"] = "exploration-index-v999"
    _write_index_payload(tmp_path, payload)

    with pytest.raises(
        ExplorationArtifactError, match="unsupported exploration index schema"
    ):
        store.load_attempt(a.name)
    with pytest.raises(
        ExplorationArtifactError, match="unsupported exploration index schema"
    ):
        store.get_index()


def test_current_index_record_requires_manifest_digest(tmp_path: Path) -> None:
    store = _store(tmp_path)
    corpus = _corpus()
    sugg = _sample_suggestions(corpus)
    a = store.write_attempt(corpus, sugg, sugg)

    payload = _read_index_payload(tmp_path)
    assert payload["schema_version"] == "exploration-index-v1"
    _record_for(payload, a.name).pop("manifest_digest", None)
    _write_index_payload(tmp_path, payload)

    with pytest.raises(ExplorationArtifactError, match="missing manifest_digest"):
        store.load_attempt(a.name)
    with pytest.raises(ExplorationArtifactError, match="missing manifest_digest"):
        store.get_index()
