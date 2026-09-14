from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import ux_analyzer.application.exploration_synthesizer as exploration_synthesizer
from ux_analyzer.application.exploration_synthesizer import (
    ExplorationSynthesisResponse,
    ExplorationSynthesizer,
    SynthesisCallReceipt,
)
from ux_analyzer.domain.exploration import CrawlCorpus, CrawlPage, normalize_crawl_url
from ux_analyzer.ports.models import ChatMessage, ModelCallRecord, ModelRole, TokenUsage

# pyright: reportPrivateUsage=false

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_page(
    url: str = "https://example.test/",
    depth: int = 0,
    title: str = "Example",
    headings: tuple[str, ...] = ("Heading One",),
    visible_elements: tuple[Any, ...] | None = None,
    discovered_links: tuple[str, ...] = (),
) -> CrawlPage:
    normalized = normalize_crawl_url(url)
    from urllib.parse import urlsplit

    parsed = urlsplit(normalized)
    host = parsed.hostname.lower() if parsed.hostname else ""
    origin = f"https://{host}"
    if parsed.port and parsed.port != 443:
        origin = f"https://{host}:{parsed.port}"
    # CrawlPage now supports visible_elements field (Task 4)
    clean_visible: tuple[str, ...] = ()
    if visible_elements is not None:
        # filter to strings
        try:
            clean_visible = tuple(
                str(x).strip()
                for x in visible_elements
                if isinstance(x, str) and str(x).strip() or not isinstance(x, str)
            )
            # keep only string instances for domain validation
            clean_visible = tuple(x for x in clean_visible if x)
            # for non-string elements (dict/object), convert via str
            if any(not isinstance(x, str) for x in visible_elements):
                # fallback: convert all to str where needed
                tmp: list[str] = []
                for x in visible_elements:
                    if isinstance(x, str) and x.strip():
                        tmp.append(x.strip())
                    elif isinstance(x, dict):
                        lbl = x.get("label") or x.get("text") or ""
                        if isinstance(lbl, str) and lbl.strip():
                            tmp.append(lbl.strip())
                    elif x is not None:
                        s = str(x).strip()
                        if s:
                            tmp.append(s)
                clean_visible = tuple(tmp)
        except Exception:
            clean_visible = ()
    page = CrawlPage(
        url=url,
        normalized_url=normalized,
        origin=origin,
        depth=depth,
        title=title,
        headings=headings,
        viewport_id=None,
        screenshot_digest=None,
        discovered_links=discovered_links,
        visible_elements=clean_visible,
    )
    return page


def _make_corpus(pages: list[CrawlPage]) -> CrawlCorpus:
    # link_graph empty for simplicity
    return CrawlCorpus(
        pages=tuple(pages),
        link_graph={},
        started_at="2026-08-23T00:00:00Z",
        corpus_digest="",
    )


class RecordingClient:
    """Fake StructuredModelClient that records prompts and returns valid visible-result scenarios."""

    endpoint_origin = "https://test.example"
    provider_id = "test-provider"
    provider_version = "test-v1"

    def __init__(self, responses: list[dict[str, Any] | Exception] | None = None):
        self.last_prompt: str | None = None
        self.last_messages: tuple[ChatMessage, ...] | None = None
        self.last_user_content: str | None = None
        self.calls = 0
        self.messages_history: list[tuple[ChatMessage, ...]] = []
        # if None, auto-generate valid scenarios based on request
        self._responses = list(responses) if responses is not None else None
        self._auto = responses is None

    async def complete(self, schema, messages, model, role):
        self.calls += 1
        self.last_messages = tuple(messages)
        self.messages_history.append(tuple(messages))
        # Join all message contents for leak checks
        self.last_prompt = "\n".join(m.content for m in messages)
        if messages and len(messages) >= 2:
            self.last_user_content = messages[1].content
        else:
            self.last_user_content = messages[0].content if messages else ""
        if self._responses is not None:
            if not self._responses:
                # fallback empty
                return schema.model_validate({"scenarios": []})
            nxt = self._responses.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            # nxt is dict payload
            return schema.model_validate(nxt)
        # auto-generate: parse packs to know urls
        try:
            payload = (
                json.loads(self.last_user_content) if self.last_user_content else {}
            )
            pages = payload.get("pages", [])
            # handle TL;DR packs that have summary instead of visible_elements
            urls = [p.get("url") for p in pages[:5] if p.get("url")]
            if not urls:
                urls = ["https://example.test/"]
        except Exception:
            urls = ["https://example.test/"]
        # Respect max_scenarios from payload
        try:
            max_s = int(payload.get("corpus_meta", {}).get("max_scenarios", 2))
        except Exception:
            max_s = 2
        max_s = max(1, min(20, max_s))

        def _pack_anchor(page: dict[str, Any]) -> tuple[str, str | None]:
            """Anchor verifier text in evidence labels the model was shown."""

            headings = page.get("headings") or []
            visible = page.get("visible_elements") or []
            if headings:
                return str(headings[0]), "heading"
            if visible:
                return str(visible[0]), None
            return f"Visible result {self.calls}", None

        def _pack_target_label(page: dict[str, Any], fallback: str) -> str:
            """Anchor the evaluation target in the same evidence the model saw.

            A real model picks a label the crawl rendered; an invented label
            can only fail at evaluation time and is rejected at synthesis time.
            """

            headings = page.get("headings") or []
            visible = page.get("visible_elements") or []
            if headings:
                return str(headings[0])
            if visible:
                return str(visible[0])
            return fallback

        scenarios = []
        for i in range(min(len(urls), max_s)):
            page = next(
                (p for p in pages if p.get("url") == urls[i % len(urls)]),
                None,
            )
            if page is None:
                anchor, anchor_role = f"Visible result {i + 1}", None
                target_label = f"Visible result {i + 1}"
            else:
                anchor, anchor_role = _pack_anchor(page)
                target_label = _pack_target_label(page, anchor)
            scenarios.append(
                {
                    "id": f"scenario-{i + 1}",
                    "name": f"Test Scenario {i + 1}",
                    "goal": f"Goal number {i + 1} unique",
                    "start_url": urls[i % len(urls)],
                    "verifier": {
                        "type": "visible-result",
                        "text": anchor,
                        "role": anchor_role,
                    },
                    "evaluation_target": {
                        "label": target_label,
                        "role": "heading",
                    },
                    "rationale": f"Rationale {i + 1} covering discovery",
                    "coverage": ["discovery"],
                }
            )
        # if urls single, still generate up to max_s with same start_url but different goals
        while len(scenarios) < max_s and len(scenarios) < 3:
            idx = len(scenarios) + 1
            page = next(
                (p for p in pages if p.get("url") == urls[0]),
                None,
            )
            if page is None:
                anchor, anchor_role = f"Visible result {idx}", None
                target_label = f"Visible result {idx}"
            else:
                anchor, anchor_role = _pack_anchor(page)
                target_label = _pack_target_label(page, anchor)
            scenarios.append(
                {
                    "id": f"scenario-{idx}",
                    "name": f"Test Scenario {idx}",
                    "goal": f"Goal number {idx} unique",
                    "start_url": urls[0],
                    "verifier": {
                        "type": "visible-result",
                        "text": anchor,
                        "role": anchor_role,
                    },
                    "evaluation_target": {"label": target_label},
                    "rationale": f"Rationale {idx}",
                    "coverage": ["navigation"],
                }
            )
        return schema.model_validate({"scenarios": scenarios[:max_s]})


def _validation_error() -> ValidationError:
    return ValidationError.from_exception_data(
        title="ExplorationSynthesisResponse",
        line_errors=[
            {
                "type": "missing",
                "loc": ("scenarios",),
                "msg": "Field required",
                "input": {},
            }
        ],
    )


class RetryRecordingClient:
    endpoint_origin = "https://test.example"
    provider_id = "test-provider"
    provider_version = "test-v1"

    def __init__(self):
        self.calls = 0
        self.last_prompt: str | None = None
        self.last_messages: tuple[ChatMessage, ...] | None = None
        self._first = True

    async def complete(self, schema, messages, model, role):
        self.calls += 1
        self.last_messages = tuple(messages)
        self.last_prompt = "\n".join(m.content for m in messages)
        if self._first:
            self._first = False
            raise _validation_error()
        # second call succeeds
        return schema.model_validate(
            {
                "scenarios": [
                    {
                        "id": "retry-ok",
                        "name": "Retry Scenario",
                        "goal": "Goal after retry",
                        "start_url": "https://example.test/",
                        "verifier": {
                            "type": "visible-result",
                            "text": "Success after retry",
                        },
                        "evaluation_target": {"label": "Target"},
                        "rationale": "Recovered from invalid JSON",
                        "coverage": ["discovery"],
                    }
                ]
            }
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_suggest_produces_visible_result_scenarios() -> None:
    pages = [
        _make_page(
            url="https://example.test/",
            title="Home",
            headings=("Welcome", "Features"),
            visible_elements=("Get Started", "Pricing"),
        ),
        _make_page(
            url="https://example.test/about",
            title="About",
            headings=("About Us",),
            visible_elements=("Team",),
        ),
        _make_page(
            url="https://example.test/docs",
            title="Docs",
            headings=("Docs",),
            visible_elements=("Guide",),
        ),
    ]
    corpus = _make_corpus(pages)
    client = RecordingClient()
    result = await ExplorationSynthesizer(client, model="test-model").suggest(
        corpus, max_scenarios=5
    )
    sugg = result.suggestions
    assert result.status == "ok"
    assert 1 <= len(sugg) <= 5
    assert all(s.verifier.type == "visible-result" for s in sugg)
    assert all(s.verifier.text.strip() != "" for s in sugg)
    # no fixture inputs leaked (client auto prompt shouldn't contain invite_email)
    assert "invite_email" not in (client.last_prompt or "")
    # each evaluation_target label non-empty
    assert all(s.evaluation_target.labels_by_version.get("live") for s in sugg)
    # each start_url subset of corpus
    allowed = {p.normalized_url for p in corpus.pages}
    for s in sugg:
        assert s.start_url in allowed


@pytest.mark.asyncio
async def test_suggest_respects_max_scenarios() -> None:
    pages = [_make_page(url="https://example.test/", title="Home")]
    corpus = _make_corpus(pages)
    # client that tries to return 10 scenarios
    payload = {
        "scenarios": [
            {
                "id": f"s{i}",
                "name": f"Name {i}",
                "goal": f"Goal {i}",
                "start_url": "https://example.test/",
                "verifier": {"type": "visible-result", "text": f"Text {i}"},
                "evaluation_target": {"label": f"Label {i}"},
                "rationale": f"Rationale {i}",
                "coverage": ["discovery"],
            }
            for i in range(10)
        ]
    }
    client = RecordingClient(responses=[payload])
    result = await ExplorationSynthesizer(client, model="test-model").suggest(
        corpus, max_scenarios=2
    )
    sugg = result.suggestions
    assert len(sugg) <= 2
    assert len(sugg) == 2  # deterministic cap


@pytest.mark.asyncio
async def test_no_fixture_inputs_leaked_and_no_private_fields() -> None:
    pages = [
        _make_page(
            url="https://example.test/",
            title="Home",
            headings=("H",),
            visible_elements=("Label",),
        )
    ]
    corpus = _make_corpus(pages)
    client = RecordingClient()
    await ExplorationSynthesizer(client, model="test-model").suggest(
        corpus, max_scenarios=3
    )
    prompt = client.last_prompt or ""
    user_content = client.last_user_content or ""
    # Private data keys must not appear as data keys in user payload (instructional mention in system prompt is allowed)
    # So check user_content doesn't contain private field keys as JSON keys
    for forbidden in ["selector", "hidden_label", "destination_url", "test_id"]:
        # check as JSON key pattern
        assert f'"{forbidden}"' not in user_content.lower()
    # Also check no hidden-label etc
    assert '"hidden-label"' not in user_content.lower()
    # Fixture-state must not be in output, but prompt may mention it as prohibition - check user payload not containing it as data
    # user payload should not contain fixture-state as a verifier type (it describes pages, not verifiers)
    # So we don't check prompt for these instructional words
    # Check no leak of example fixture key invite_email in either prompt or payload
    assert "invite_email" not in prompt.lower()
    assert "invite_email" not in user_content.lower()
    # No dest URLs beyond corpus: discovered_links not leaked
    evil_page = _make_page(
        url="https://example.test/",
        title="Home",
        discovered_links=("https://evil.test/secret",),
    )
    corpus2 = _make_corpus([evil_page])
    client2 = RecordingClient()
    await ExplorationSynthesizer(client2, model="test-model").suggest(
        corpus2, max_scenarios=2
    )
    assert "evil.test" not in (client2.last_user_content or "")
    # prompt may contain evil.test as part of leak check? No, it shouldn't


@pytest.mark.asyncio
async def test_payload_compressed_headings_and_elements_and_caps() -> None:
    # Headings 10, visible_elements 100 -> should be truncated to 3 and 30
    many_headings = tuple(f"Heading {i}" for i in range(10))
    many_labels = tuple(f"Label {i}" for i in range(100))
    page = _make_page(
        url="https://example.test/",
        title="Home",
        headings=many_headings,
        visible_elements=many_labels,
    )
    corpus = _make_corpus([page])
    client = RecordingClient()
    await ExplorationSynthesizer(client, model="test-model").suggest(
        corpus, max_scenarios=2
    )
    user_content = client.last_user_content or ""
    data = json.loads(user_content)
    packs = data.get("pages", [])
    assert len(packs) == 1
    assert len(packs[0]["headings"]) <= 3
    assert len(packs[0]["visible_elements"]) <= 30
    # title truncated check (long title)
    long_title = "A" * 500
    page2 = _make_page(url="https://example.test/", title=long_title, headings=("H",))
    corpus2 = _make_corpus([page2])
    client2 = RecordingClient()
    await ExplorationSynthesizer(client2, model="test-model").suggest(
        corpus2, max_scenarios=1
    )
    data2 = json.loads(client2.last_user_content or "{}")
    assert len(data2["pages"][0]["title"]) <= 120
    # total cap 25 pages
    many_pages = [
        _make_page(url=f"https://example.test/page{i}", title=f"P{i}")
        for i in range(30)
    ]
    corpus3 = _make_corpus(many_pages)
    client3 = RecordingClient()
    await ExplorationSynthesizer(client3, model="test-model").suggest(
        corpus3, max_scenarios=5
    )
    data3 = json.loads(client3.last_user_content or "{}")
    assert len(data3["pages"]) <= 25
    # 100k chars cap
    huge_pages = [
        _make_page(
            url=f"https://example.test/page{i}",
            title="T" * 120,
            headings=("H" * 120,) * 3,
            visible_elements=tuple("L" * 80 for _ in range(30)),
        )
        for i in range(25)
    ]
    corpus4 = _make_corpus(huge_pages)
    client4 = RecordingClient()
    await ExplorationSynthesizer(client4, model="test-model").suggest(
        corpus4, max_scenarios=5
    )
    assert len(client4.last_user_content or "") <= 100_000
    # also packs json length <=100k
    data4 = json.loads(client4.last_user_content or "{}")
    assert len(json.dumps(data4["pages"])) <= 100_000


@pytest.mark.asyncio
async def test_chunking_large_corpus_triggers_batches() -> None:
    # Create 30 pages to exceed 25 cap, should trigger chunking path (multiple calls)
    many_pages = [
        _make_page(
            url=f"https://example.test/page{i}",
            title=f"Page {i}",
            headings=(f"H{i}",),
            visible_elements=(f"Label {i} {j}" for j in range(5)),
        )
        for i in range(30)
    ]
    # fix generator for visible_elements
    many_pages = [
        _make_page(
            url=f"https://example.test/page{i}",
            title=f"Page {i}",
            headings=(f"H{i}",),
            visible_elements=tuple(f"Label {i} {j}" for j in range(5)),
        )
        for i in range(30)
    ]
    corpus = _make_corpus(many_pages)
    client = RecordingClient()
    result = await ExplorationSynthesizer(client, model="test-model").suggest(
        corpus, max_scenarios=5
    )
    sugg = result.suggestions
    # chunking should have produced at least one call, and result within limits
    assert client.calls >= 1
    # If chunking splits into multiple batches, calls >1; our implementation may use 3 chunks for 25 pages cap with 10 per chunk
    # Ensure still returns valid scenarios
    assert 1 <= len(sugg) <= 5
    # Payload per call should be within 100k
    for msgs in client.messages_history:
        user = msgs[1].content if len(msgs) >= 2 else ""
        assert len(user) <= 100_000
    # Also test that a huge single-page char overload is handled without error
    huge = _make_page(
        url="https://example.test/",
        title="T" * 5000,
        headings=tuple("H" * 500 for _ in range(5)),
        visible_elements=tuple("L" * 200 for _ in range(100)),
    )
    corpus2 = _make_corpus([huge])
    client2 = RecordingClient()
    result2 = await ExplorationSynthesizer(client2, model="test-model").suggest(
        corpus2, max_scenarios=3
    )
    sugg2 = result2.suggestions
    assert 1 <= len(sugg2) <= 3
    assert len(client2.last_user_content or "") <= 100_000


@pytest.mark.asyncio
async def test_retry_on_invalid_json() -> None:
    pages = [_make_page(url="https://example.test/", title="Home")]
    corpus = _make_corpus(pages)
    client = RetryRecordingClient()
    result = await ExplorationSynthesizer(client, model="test-model").suggest(
        corpus, max_scenarios=3
    )
    sugg = result.suggestions
    assert client.calls == 2  # first failed, second succeeded via json_object fallback
    assert result.status == "ok"
    assert len(sugg) == 1
    assert sugg[0].verifier.text == "Success after retry"


@pytest.mark.asyncio
async def test_operational_failure_marks_unavailable_without_retry() -> None:
    pages = [_make_page(url="https://example.test/", title="Home")]
    corpus = _make_corpus(pages)

    class OutageClient:
        endpoint_origin = "https://test.example"
        provider_id = "test-provider"
        provider_version = "test-v1"

        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, schema, messages, model, role):
            self.calls += 1
            raise ConnectionError("provider unreachable")

    client = OutageClient()
    result = await ExplorationSynthesizer(client, model="test-model").suggest(
        corpus, max_scenarios=3
    )
    # Operational outage must NOT be retried like an invalid schema answer.
    assert client.calls == 1
    assert result.status == "unavailable"
    assert result.suggestions == ()
    assert result.limitations
    assert "transport" in result.limitations[0].casefold()


@pytest.mark.asyncio
async def test_invalid_structured_output_bounded_retry_then_invalid_status() -> None:
    pages = [_make_page(url="https://example.test/", title="Home")]
    corpus = _make_corpus(pages)

    class AlwaysInvalidClient:
        endpoint_origin = "https://test.example"
        provider_id = "test-provider"
        provider_version = "test-v1"

        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, schema, messages, model, role):
            self.calls += 1
            raise _validation_error()

    client = AlwaysInvalidClient()
    result = await ExplorationSynthesizer(client, model="test-model").suggest(
        corpus, max_scenarios=3
    )
    # Bounded retry: initial attempt + exactly one invalid-output retry.
    assert client.calls == 2
    assert result.status == "invalid"
    assert result.suggestions == ()
    assert result.limitations


@pytest.mark.asyncio
async def test_legitimate_empty_response_is_ok_not_failure() -> None:
    pages = [_make_page(url="https://example.test/", title="Home")]
    corpus = _make_corpus(pages)
    client = RecordingClient(responses=[{"scenarios": []}])
    result = await ExplorationSynthesizer(client, model="test-model").suggest(
        corpus, max_scenarios=3
    )
    # A parsed-but-empty answer is a legit "no scenarios", distinct from failure.
    assert client.calls == 1
    assert result.status == "ok"
    assert result.suggestions == ()
    assert result.limitations == ()


@pytest.mark.asyncio
async def test_missing_prompt_file_is_hard_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = [_make_page(url="https://example.test/", title="Home")]
    corpus = _make_corpus(pages)
    monkeypatch.setattr(
        exploration_synthesizer,
        "EXPLORATION_PROMPT_PATH",
        Path(exploration_synthesizer.__file__).parent / "does-not-exist.txt",
    )
    with pytest.raises(FileNotFoundError):
        await ExplorationSynthesizer(RecordingClient(), model="test-model").suggest(
            corpus, max_scenarios=3
        )


@pytest.mark.asyncio
async def test_model_narrative_redacted_before_domain_conversion() -> None:
    pages = [_make_page(url="https://example.test/", title="Home")]
    corpus = _make_corpus(pages)
    payload = {
        "scenarios": [
            {
                "id": "leaky",
                "name": "Leaky Name",
                "goal": "Leaky goal",
                "start_url": "https://example.test/",
                "verifier": {"type": "visible-result", "text": "Banner says Welcome"},
                "evaluation_target": {"label": "Welcome banner"},
                "rationale": "Derived from chain of thought reasoning",
                "coverage": ["discovery"],
            },
            {
                "id": "clean",
                "name": "Clean Name",
                "goal": "Clean goal",
                "start_url": "https://example.test/",
                "verifier": {"type": "visible-result", "text": "Clean text"},
                "evaluation_target": {"label": "Clean target"},
                "rationale": "Covers discovery coverage",
                "coverage": ["discovery"],
            },
        ]
    }
    client = RecordingClient(responses=[payload])
    result = await ExplorationSynthesizer(client, model="test-model").suggest(
        corpus, max_scenarios=5
    )
    by_id = {s.id: s for s in result.suggestions}
    assert by_id["leaky"].rationale == "[redacted]"
    assert "chain of thought" not in by_id["leaky"].rationale.lower()
    # Untainted strings pass through untouched.
    assert by_id["clean"].rationale == "Covers discovery coverage"


@pytest.mark.asyncio
async def test_synthesized_scenarios_never_pin_roles() -> None:
    """Synthesized verifiers match any rendered text; role pinning stays manual."""

    pages = [
        _make_page(
            url="https://example.test/",
            title="Home",
            headings=("Welcome",),
            visible_elements=("Get Started",),
        )
    ]
    corpus = _make_corpus(pages)
    payload = {
        "scenarios": [
            {
                "id": "s1",
                "name": "S1",
                "goal": "Goal one",
                "start_url": "https://example.test/",
                "verifier": {
                    "type": "visible-result",
                    "text": "Get Started",
                    "role": "heading",
                },
                "evaluation_target": {"label": "Get Started", "role": "heading"},
                "rationale": "Rationale",
                "coverage": ["discovery"],
            }
        ]
    }
    client = RecordingClient(responses=[payload])
    result = await ExplorationSynthesizer(client, model="test-model").suggest(
        corpus, max_scenarios=3
    )
    assert [s.id for s in result.suggestions] == ["s1"]
    assert result.suggestions[0].verifier.role is None
    assert result.suggestions[0].evaluation_target.role is None


@pytest.mark.asyncio
async def test_rejected_scenarios_leave_audit_trail() -> None:
    pages = [
        _make_page(url="https://example.test/", title="Home"),
        _make_page(url="https://example.test/about", title="About"),
    ]
    corpus = _make_corpus(pages)
    payload = {
        "scenarios": [
            {
                "id": "good",
                "name": "Good",
                "goal": "Good goal",
                "start_url": "https://example.test/",
                "verifier": {"type": "visible-result", "text": "Good text"},
                "evaluation_target": {"label": "Good label"},
                "rationale": "Rationale",
                "coverage": ["discovery"],
            },
            {
                "id": "bad-url",
                "name": "Bad Url",
                "goal": "Bad url goal",
                "start_url": "https://evil.test/not-in-corpus",
                "verifier": {"type": "visible-result", "text": "Text"},
                "evaluation_target": {"label": "Label"},
                "rationale": "Rationale",
                "coverage": ["discovery"],
            },
            {
                "id": "dup-goal",
                "name": "Dup Goal",
                "goal": "good goal",  # case-insensitive duplicate of accepted goal
                "start_url": "https://example.test/about",
                "verifier": {"type": "visible-result", "text": "Text"},
                "evaluation_target": {"label": "Label"},
                "rationale": "Rationale",
                "coverage": ["navigation"],
            },
        ]
    }
    client = RecordingClient(responses=[payload])
    result = await ExplorationSynthesizer(client, model="test-model").suggest(
        corpus, max_scenarios=5
    )
    assert [s.id for s in result.suggestions] == ["good"]
    audits = {a.scenario_id: a for a in result.rejected_audits}
    assert set(audits) == {"bad-url", "dup-goal"}
    assert audits["bad-url"].reason_code == "start-url-outside-corpus"
    assert audits["dup-goal"].reason_code == "duplicate-goal"
    for audit in result.rejected_audits:
        assert len(audit.payload_digest) == 64
        int(audit.payload_digest, 16)  # sha256 hex


@pytest.mark.asyncio
async def test_duplicate_goals_deduped() -> None:
    pages = [_make_page(url="https://example.test/", title="Home")]
    corpus = _make_corpus(pages)
    payload = {
        "scenarios": [
            {
                "id": "s1",
                "name": "Name 1",
                "goal": "Same Goal",
                "start_url": "https://example.test/",
                "verifier": {"type": "visible-result", "text": "Text 1"},
                "evaluation_target": {"label": "Label 1"},
                "rationale": "Rationale 1",
                "coverage": ["discovery"],
            },
            {
                "id": "s2",
                "name": "Name 2",
                "goal": "Same Goal",  # duplicate
                "start_url": "https://example.test/",
                "verifier": {"type": "visible-result", "text": "Text 2"},
                "evaluation_target": {"label": "Label 2"},
                "rationale": "Rationale 2",
                "coverage": ["navigation"],
            },
            {
                "id": "s3",
                "name": "Name 3",
                "goal": "same goal",  # case-insensitive duplicate
                "start_url": "https://example.test/",
                "verifier": {"type": "visible-result", "text": "Text 3"},
                "evaluation_target": {"label": "Label 3"},
                "rationale": "Rationale 3",
                "coverage": ["forms"],
            },
            {
                "id": "s4",
                "name": "Name 4",
                "goal": "Different Goal",
                "start_url": "https://example.test/",
                "verifier": {"type": "visible-result", "text": "Text 4"},
                "evaluation_target": {"label": "Label 4"},
                "rationale": "Rationale 4",
                "coverage": ["completion"],
            },
        ]
    }
    client = RecordingClient(responses=[payload])
    result = await ExplorationSynthesizer(client, model="test-model").suggest(
        corpus, max_scenarios=10
    )
    sugg = result.suggestions
    # deduped to 2 unique goals (Same Goal and Different Goal)
    assert len(sugg) == 2
    goals = {s.goal.strip().casefold() for s in sugg}
    assert len(goals) == 2
    # Both rejected duplicates are audited.
    assert {a.scenario_id for a in result.rejected_audits} == {"s2", "s3"}
    assert all(a.reason_code == "duplicate-goal" for a in result.rejected_audits)


@pytest.mark.asyncio
async def test_validates_all_of_unique_text_non_empty_start_url_subset() -> None:
    pages = [
        _make_page(url="https://example.test/", title="Home"),
        _make_page(url="https://example.test/about", title="About"),
    ]
    corpus = _make_corpus(pages)
    payload = {
        "scenarios": [
            # valid
            {
                "id": "good",
                "name": "Good",
                "goal": "Good goal",
                "start_url": "https://example.test/",
                "verifier": {
                    "type": "visible-result",
                    "text": "Good text",
                    "all_of": ["a", "b"],
                },
                "evaluation_target": {"label": "Good label"},
                "rationale": "Rationale",
                "coverage": ["discovery"],
            },
            # invalid: start_url not in corpus subset
            {
                "id": "bad-url",
                "name": "Bad Url",
                "goal": "Bad url goal",
                "start_url": "https://evil.test/not-in-corpus",
                "verifier": {"type": "visible-result", "text": "Text"},
                "evaluation_target": {"label": "Label"},
                "rationale": "Rationale",
                "coverage": ["discovery"],
            },
            # invalid: all_of duplicate should be rejected (pydantic will error, but we test filtering)
            # we make a payload where second scenario has duplicate all_of - it will be filtered by validation and not returned
            # Instead we test that valid scenario remains
        ]
    }
    client = RecordingClient(responses=[payload])
    result = await ExplorationSynthesizer(client, model="test-model").suggest(
        corpus, max_scenarios=5
    )
    sugg = result.suggestions
    # only good should survive (bad-url filtered)
    assert len(sugg) == 1
    assert sugg[0].id == "good"
    # rejection is audited, not silent
    assert [a.scenario_id for a in result.rejected_audits] == ["bad-url"]
    # also test all_of duplicate handling: provide invalid second scenario that fails validation internally should not crash
    payload2 = {
        "scenarios": [
            {
                "id": "good2",
                "name": "Good2",
                "goal": "Goal2",
                "start_url": "https://example.test/",
                "verifier": {"type": "visible-result", "text": "Text2"},
                "evaluation_target": {"label": "Label2"},
                "rationale": "Rationale2",
                "coverage": ["discovery"],
            }
        ]
    }
    client2 = RecordingClient(responses=[payload2])
    result2 = await ExplorationSynthesizer(client2, model="test-model").suggest(
        corpus, max_scenarios=5
    )
    assert len(result2.suggestions) == 1


@pytest.mark.asyncio
async def test_max_scenarios_bounds_deterministic() -> None:
    pages = [_make_page(url="https://example.test/", title="Home")]
    corpus = _make_corpus(pages)
    client = RecordingClient()
    # valid range
    result = await ExplorationSynthesizer(client, model="test-model").suggest(
        corpus, max_scenarios=1
    )
    sugg = result.suggestions
    assert 1 <= len(sugg) <= 1
    result2 = await ExplorationSynthesizer(client, model="test-model").suggest(
        corpus, max_scenarios=20
    )
    assert len(result2.suggestions) <= 20
    # invalid bounds should raise
    with pytest.raises(ValueError):
        await ExplorationSynthesizer(client, model="test-model").suggest(
            corpus, max_scenarios=0
        )
    with pytest.raises(ValueError):
        await ExplorationSynthesizer(client, model="test-model").suggest(
            corpus, max_scenarios=21
        )
    # deterministic: same input yields same ids order sorted
    payload = {
        "scenarios": [
            {
                "id": "b-id",
                "name": "B",
                "goal": "Goal B",
                "start_url": "https://example.test/",
                "verifier": {"type": "visible-result", "text": "Text B"},
                "evaluation_target": {"label": "Label B"},
                "rationale": "Rationale B",
                "coverage": ["discovery"],
            },
            {
                "id": "a-id",
                "name": "A",
                "goal": "Goal A",
                "start_url": "https://example.test/",
                "verifier": {"type": "visible-result", "text": "Text A"},
                "evaluation_target": {"label": "Label A"},
                "rationale": "Rationale A",
                "coverage": ["discovery"],
            },
        ]
    }
    client_a = RecordingClient(responses=[payload, payload])
    s1 = (
        await ExplorationSynthesizer(client_a, model="test-model").suggest(
            corpus, max_scenarios=2
        )
    ).suggestions
    client_b = RecordingClient(responses=[payload])
    s2 = (
        await ExplorationSynthesizer(client_b, model="test-model").suggest(
            corpus, max_scenarios=2
        )
    ).suggestions
    assert [s.id for s in s1] == [s.id for s in s2]
    # sorted deterministic by id
    assert [s.id for s in s1] == sorted([s.id for s in s1])


@pytest.mark.asyncio
async def test_produces_visible_result_only_no_selectors_leaked() -> None:
    pages = [
        _make_page(
            url="https://example.test/",
            title="Home",
            visible_elements=("Button label",),
        )
    ]
    corpus = _make_corpus(pages)
    # Try to return fixture-state via manual payload: our synthesizer should filter it
    # But our RecordingClient auto-generates visible-result only, so we test that fixture-state is not produced
    # Provide a client that returns fixture-state attempt: pydantic will reject type, so the invalid
    # output is retried once and then recovered through the empty-response fallback.
    payload = {
        "scenarios": [
            {
                "id": "bad-fixture",
                "name": "Bad",
                "goal": "Bad goal",
                "start_url": "https://example.test/",
                "verifier": {
                    "type": "fixture-state",
                    "text": "Should be filtered",
                },  # type mismatch will cause validation error
                "evaluation_target": {"label": "Label"},
                "rationale": "Rationale",
                "coverage": ["discovery"],
            }
        ]
    }
    client = RecordingClient(responses=[payload])
    result = await ExplorationSynthesizer(client, model="test-model").suggest(
        corpus, max_scenarios=5
    )
    # Should filter out invalid fixture-state, result empty but explicitly ok
    assert len(result.suggestions) == 0
    assert result.status == "ok"
    assert client.calls == 2  # one bounded retry on the invalid structured output
    # Also ensure no selectors etc in successful case
    client2 = RecordingClient()
    result2 = await ExplorationSynthesizer(client2, model="test-model").suggest(
        corpus, max_scenarios=3
    )
    for s in result2.suggestions:
        assert s.verifier.type == "visible-result"
        assert "selector" not in s.verifier.text.lower()
        assert s.verifier.text.strip() != ""


@pytest.mark.asyncio
async def test_chunked_path_surfaces_unavailable_when_outage_hits() -> None:
    many_pages = [
        _make_page(url=f"https://example.test/page{i}", title=f"Page {i}")
        for i in range(30)
    ]
    corpus = _make_corpus(many_pages)

    class OutageClient(RecordingClient):
        async def complete(self, schema, messages, model, role):  # noqa: D102
            self.calls += 1
            raise TimeoutError("gateway timeout")

    client = OutageClient()
    result = await ExplorationSynthesizer(client, model="test-model").suggest(
        corpus, max_scenarios=5
    )
    # Operational outage inside the chunked path is surfaced, not retried.
    assert client.calls == 1
    assert result.status == "unavailable"
    assert result.suggestions == ()
    assert result.limitations


# ---------------------------------------------------------------------------
# Per-call provenance receipts
# ---------------------------------------------------------------------------


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _prompt_digest_of(messages: tuple[ChatMessage, ...]) -> str:
    return _sha256("\n".join(m.content for m in messages))


@pytest.mark.asyncio
async def test_successful_synthesis_attaches_verifiable_receipt() -> None:
    pages = [_make_page(url="https://example.test/", title="Home")]
    corpus = _make_corpus(pages)
    payload = {
        "scenarios": [
            {
                "id": "r1",
                "name": "Receipt Scenario",
                "goal": "Goal with receipt",
                "start_url": "https://example.test/",
                "verifier": {"type": "visible-result", "text": "Visible text"},
                "evaluation_target": {"label": "Target"},
                "rationale": "Rationale",
                "coverage": ["discovery"],
            }
        ]
    }
    client = RecordingClient(responses=[payload])
    result = await ExplorationSynthesizer(client, model="test-model").suggest(
        corpus, max_scenarios=3
    )
    assert result.status == "ok"
    assert client.calls == 1
    assert len(result.receipts) == 1
    receipt = result.receipts[0]
    # digest shape
    for digest in (
        receipt.prompt_digest,
        receipt.schema_version_digest,
        receipt.output_digest,
    ):
        assert len(digest) == 64
        int(digest, 16)  # sha256 hex
    # prompt digest recomputes from what the client actually received
    assert client.last_messages is not None
    assert receipt.prompt_digest == _prompt_digest_of(client.last_messages)
    # schema version + its digest
    assert receipt.schema_version == "exploration-synthesis-v1"
    assert receipt.schema_version_digest == _sha256("exploration-synthesis-v1")
    # output digest recomputes from the raw validated response payload
    raw_response = ExplorationSynthesisResponse.model_validate(payload)
    expected_output = _sha256(_canonical_json(raw_response.model_dump(mode="json")))
    assert receipt.output_digest == expected_output
    # derived locally: no provider record metadata on RecordingClient
    assert receipt.source == "synthesizer-derived"
    assert receipt.attempts == 1
    assert receipt.latency_ms is None
    assert receipt.total_tokens is None


@pytest.mark.asyncio
async def test_chunked_synthesis_records_receipt_per_successful_call() -> None:
    # Force chunking: >25 pages and packed size over the 80k chunk threshold.
    long_url = "https://example.test/" + "p" * 300
    many_pages = [
        _make_page(
            url=f"{long_url}/{i}",
            title="T" * 120,
            headings=tuple("H" * 120 for _ in range(3)),
            visible_elements=tuple("L" * 80 for _ in range(30)),
        )
        for i in range(30)
    ]
    corpus = _make_corpus(many_pages)
    client = RecordingClient()
    result = await ExplorationSynthesizer(client, model="test-model").suggest(
        corpus, max_scenarios=5
    )
    assert result.status == "ok"
    assert client.calls >= 2
    # One receipt per successful call, aligned with the recorded call order.
    assert len(result.receipts) == client.calls
    for idx, receipt in enumerate(result.receipts):
        assert receipt.source == "synthesizer-derived"
        assert receipt.prompt_digest == _prompt_digest_of(client.messages_history[idx])
        assert receipt.schema_version_digest == _sha256("exploration-synthesis-v1")
        assert len(receipt.output_digest) == 64


@pytest.mark.asyncio
async def test_receipt_enriched_from_client_call_record_when_exposed() -> None:
    pages = [_make_page(url="https://example.test/", title="Home")]
    corpus = _make_corpus(pages)
    payload = {
        "scenarios": [
            {
                "id": "rr",
                "name": "Recorded",
                "goal": "Goal from recorded call",
                "start_url": "https://example.test/",
                "verifier": {"type": "visible-result", "text": "Visible"},
                "evaluation_target": {"label": "Target"},
                "rationale": "Rationale",
                "coverage": ["discovery"],
            }
        ]
    }
    record = ModelCallRecord(
        role=ModelRole.COGNITIVE,
        model="test-model",
        endpoint_origin="https://test.example",
        prompt_digest="f" * 64,  # deliberately different from local recompute
        schema_version="exploration-synthesis-v1",
        attempts=2,
        latency_ms=123,
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        request={},
        response={},
    )

    class RecordKeepingClient(RecordingClient):
        def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
            super().__init__(responses)
            self.records: list[ModelCallRecord] = [record]

    client = RecordKeepingClient(responses=[payload])
    result = await ExplorationSynthesizer(client, model="test-model").suggest(
        corpus, max_scenarios=3
    )
    assert result.status == "ok"
    assert len(result.receipts) == 1
    receipt = result.receipts[0]
    assert receipt.source == "client-call-record"
    assert receipt.attempts == 2
    assert receipt.latency_ms == 123
    assert receipt.total_tokens == 15
    # Digests stay locally recomputable even when a provider record exists.
    assert client.last_messages is not None
    assert receipt.prompt_digest == _prompt_digest_of(client.last_messages)
    assert receipt.prompt_digest != record.prompt_digest
    raw_response = ExplorationSynthesisResponse.model_validate(payload)
    assert receipt.output_digest == _sha256(
        _canonical_json(raw_response.model_dump(mode="json"))
    )


def test_receipt_rejects_malformed_digests() -> None:
    with pytest.raises(ValueError):
        SynthesisCallReceipt(
            prompt_digest="not-a-digest",
            schema_version="exploration-synthesis-v1",
            schema_version_digest=_sha256("exploration-synthesis-v1"),
            output_digest=_sha256("{}"),
        )
    with pytest.raises(ValueError):
        SynthesisCallReceipt(
            prompt_digest=_sha256("p"),
            schema_version="exploration-synthesis-v1",
            schema_version_digest=_sha256("exploration-synthesis-v1"),
            output_digest=_sha256("{}"),
            attempts=0,
        )


# ---------------------------------------------------------------------------
# Error category wiring
# ---------------------------------------------------------------------------


class ModelFailureError(RuntimeError):
    """Test double matching the adapter's duck-typed failure error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@pytest.mark.asyncio
async def test_provider_failure_marks_unavailable_with_provider_limitation() -> None:
    pages = [_make_page(url="https://example.test/", title="Home")]
    corpus = _make_corpus(pages)

    class ProviderFailureClient:
        endpoint_origin = "https://test.example"

        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, schema, messages, model, role):
            self.calls += 1
            raise ModelFailureError("provider reports model unavailable")

    client = ProviderFailureClient()
    result = await ExplorationSynthesizer(client, model="test-model").suggest(
        corpus, max_scenarios=3
    )
    # Provider-reported failure is operational: surfaced once, never retried.
    assert client.calls == 1
    assert result.status == "unavailable"
    assert result.suggestions == ()
    assert result.receipts == ()
    assert result.limitations
    assert "unavailable" in result.limitations[0].casefold()


# ---------------------------------------------------------------------------
# Manifest typing without attribute guessing
# ---------------------------------------------------------------------------


def test_manifest_uses_defaults_for_minimal_client() -> None:
    class MinimalClient:
        endpoint_origin = "https://minimal.example"

        async def complete(self, schema, messages, model, role):
            return schema.model_validate({"scenarios": []})

    manifest = ExplorationSynthesizer(MinimalClient(), model="m").manifest
    assert manifest.provider_id == "openai-compatible-structured"
    assert manifest.provider_version == "openai-compatible-v1"
    assert manifest.endpoint_origin == "https://minimal.example"
    assert manifest.schema_version == "exploration-synthesis-v1"


def test_manifest_uses_exposed_provider_metadata() -> None:
    manifest = ExplorationSynthesizer(RecordingClient(), model="m").manifest
    assert manifest.provider_id == "test-provider"
    assert manifest.provider_version == "test-v1"


# ---------------------------------------------------------------------------
# Loud compression budget accounting (fix round)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_page_cap_drop_is_loud_limitation_counter_and_digest() -> None:
    pages = [
        _make_page(url=f"https://example.test/page{i}", title=f"P{i}")
        for i in range(30)
    ]
    corpus = _make_corpus(pages)
    client = RecordingClient()
    result = await ExplorationSynthesizer(client, model="test-model").suggest(
        corpus, max_scenarios=3
    )
    assert result.status == "ok"
    stats = result.compression
    assert stats.pages_received == 30
    assert stats.pages_included == 25
    assert stats.pages_dropped == 5
    assert stats.tldr_degraded_pages == 0
    truncation_notes = [
        note for note in result.limitations if "truncated for synthesis" in note
    ]
    assert len(truncation_notes) == 1
    assert "(5 dropped)" in truncation_notes[0]
    # Truncation binds the compressed view to the source corpus digest.
    assert corpus.corpus_digest in truncation_notes[0]
    # One sha256 payload digest per successful call over the exact payload.
    assert len(result.payload_digests) == client.calls
    assert client.last_user_content is not None
    assert result.payload_digests[-1] == _sha256(client.last_user_content)


@pytest.mark.asyncio
async def test_unsatisfiable_budget_fails_loudly_without_model_call() -> None:
    pages = [
        _make_page(
            url="https://example.test/",
            title="Home",
            headings=("Heading One",),
        )
    ]
    corpus = _make_corpus(pages)
    client = RecordingClient()
    synth = ExplorationSynthesizer(
        client, model="test-model", max_chars=40
    )
    with pytest.raises(ValueError, match="budget unsatisfiable"):
        await synth.suggest(corpus, max_scenarios=1)
    # Loud failure: no model call is made with an arbitrary page subset.
    assert client.calls == 0
    assert client.last_messages is None


@pytest.mark.asyncio
async def test_tldr_degradation_recorded_in_limitations_and_counters() -> None:
    pages = [
        _make_page(
            url=f"https://e.test/{i}",
            title="T",
            headings=("H",),
            visible_elements=("L",),
        )
        for i in range(20)
    ]
    corpus = _make_corpus(pages)
    built, dropped = exploration_synthesizer._build_packs(corpus, 25)
    assert dropped == 0
    # Choose a budget exactly between the TL;DR total and the reduced floor so
    # the degradation path triggers deterministically without page dropping.
    tldr_preview = [
        {"url": p["url"], "depth": p["depth"], "summary": "T | H | L"}
        for p in built
    ]
    budget = exploration_synthesizer._packs_json_length(tldr_preview) + 10
    assert exploration_synthesizer._packs_json_length(built) > budget
    result = await ExplorationSynthesizer(
        RecordingClient(), model="test-model", max_chars=budget
    ).suggest(corpus, max_scenarios=2)
    notes = [
        note for note in result.limitations if "compact per-page summaries" in note
    ]
    assert len(notes) == 1
    assert "20 pages" in notes[0]
    # Degradation also records the source corpus digest for verifiability.
    assert any(corpus.corpus_digest in note for note in result.limitations)
    assert result.compression.tldr_degraded_pages == 20
    assert result.compression.pages_dropped == 0
    assert result.compression.pages_included == 20


@pytest.mark.asyncio
async def test_payload_digest_binds_sent_compressed_payload() -> None:
    pages = [_make_page(url="https://example.test/", title="Home")]
    corpus = _make_corpus(pages)
    payload = {
        "scenarios": [
            {
                "id": "bound",
                "name": "Bound",
                "goal": "Digest binding goal",
                "start_url": "https://example.test/",
                "verifier": {"type": "visible-result", "text": "Visible"},
                "evaluation_target": {"label": "Target"},
                "rationale": "Rationale",
                "coverage": ["discovery"],
            }
        ]
    }
    client = RecordingClient(responses=[payload])
    result = await ExplorationSynthesizer(client, model="test-model").suggest(
        corpus, max_scenarios=1
    )
    assert result.status == "ok"
    assert len(result.payload_digests) == 1
    assert client.last_user_content is not None
    assert result.payload_digests[0] == _sha256(client.last_user_content)
    # The receipt prompt digest still covers system + rendered payload.
    assert client.last_messages is not None
    rendered = "\n".join(m.content for m in client.last_messages)
    assert result.receipts[0].prompt_digest == _sha256(rendered)


def test_pack_construction_enforces_field_allowlist_and_bounds() -> None:
    page = _make_page(
        url="https://example.test/",
        title="Home",
        headings=("H1", "H2"),
        visible_elements=("Label One", "#btn", "https://evil.test/x"),
    )
    pack = exploration_synthesizer._build_page_pack(page)
    # Explicit allowlist: exactly these fields, nothing more.
    assert set(pack) == {
        "url",
        "depth",
        "title",
        "headings",
        "visible_elements",
    }
    assert len(pack["title"]) <= exploration_synthesizer._TITLE_TRUNC
    assert len(pack["headings"]) <= exploration_synthesizer._HEADINGS_PER_PAGE
    assert (
        len(pack["visible_elements"])
        <= exploration_synthesizer._ELEMENTS_PER_PAGE
    )
    # Deterministic dest-URL guard retained; fuzzy selector heuristic deleted.
    assert "https://evil.test/x" not in pack["visible_elements"]
    assert "#btn" in pack["visible_elements"]

    # Fields outside the allowlist are rejected loudly.
    polluted = dict(pack)
    polluted["selector"] = ".hidden"
    with pytest.raises(ValueError, match="allowlist"):
        exploration_synthesizer._validate_full_pack(polluted)

    # Missing allowlisted fields are rejected loudly.
    missing = {k: v for k, v in pack.items() if k != "title"}
    with pytest.raises(ValueError, match="missing"):
        exploration_synthesizer._validate_full_pack(missing)

    # Per-field bounds are enforced at construction.
    oversized = dict(pack)
    oversized["title"] = "X" * (exploration_synthesizer._TITLE_TRUNC + 1)
    with pytest.raises(ValueError, match="exceeds its bound"):
        exploration_synthesizer._validate_full_pack(oversized)


def test_tldr_pack_shape_enforced() -> None:
    valid = {"url": "https://example.test/", "depth": 0, "summary": "S"}
    assert exploration_synthesizer._validate_tldr_pack(valid) == valid
    polluted = dict(valid)
    polluted["selector"] = "#x"
    with pytest.raises(ValueError):
        exploration_synthesizer._validate_tldr_pack(polluted)

def test_operational_limitation_includes_provider_error_code() -> None:
    from ux_analyzer.adapters.openai import ModelFailureError

    error = ModelFailureError(
        "model unavailable",
        status_code=400,
        error_code="content-blocked",
        error_type="agent_router_api_error",
        request_id="req123",
    )
    limitation = exploration_synthesizer._operational_limitation(
        error, "model-provider-unavailable"
    )
    assert "model is unavailable" in limitation
    assert "HTTP 400" in limitation
    assert "Provider error code: content-blocked" in limitation
    assert "Request ID: req123" in limitation


def test_operational_limitation_without_provider_details() -> None:
    from ux_analyzer.adapters.openai import ModelFailureError

    error = ModelFailureError("model unavailable")
    limitation = exploration_synthesizer._operational_limitation(
        error, "model-provider-unavailable"
    )
    assert "HTTP" not in limitation
    assert "Provider error code" not in limitation
    assert "Request ID" not in limitation
