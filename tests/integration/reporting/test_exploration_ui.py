from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from playwright.async_api import async_playwright

import ux_analyzer
from ux_analyzer.adapters.exploration_server import (
    ExplorationReviewServer,
    _loopback_bind_host,
    create_exploration_app,
    curate_auto_accept,
)
from ux_analyzer.domain.benchmark import (
    Budget,
    Persona,
    ScenarioEvaluationTarget,
    VisibleResultVerifierSpec,
)
from ux_analyzer.domain.exploration import (
    CrawlCorpus,
    CrawlPage,
    ScenarioSuggestion,
    normalize_crawl_url,
)


def _page(url: str, depth: int = 0) -> CrawlPage:
    norm = normalize_crawl_url(url)
    from urllib.parse import urlsplit

    parsed = urlsplit(norm)
    host = parsed.hostname.lower() if parsed.hostname else ""
    origin = f"https://{host}"
    if parsed.port and parsed.port != 443:
        origin = f"https://{host}:{parsed.port}"
    return CrawlPage(
        url=url,
        normalized_url=norm,
        origin=origin,
        depth=depth,
        title=f"Title {url}",
        headings=("Heading One",),
        viewport_id="vp-1",
        screenshot_digest="a" * 64,
        discovered_links=(),
        visible_elements=("Button",),
    )


def _corpus(urls=("https://example.test/", "https://example.test/about")) -> CrawlCorpus:
    pages = tuple(_page(u, depth=i) for i, u in enumerate(urls))
    return CrawlCorpus(pages=pages, link_graph={}, started_at="2026-08-23T00:00:00Z")


def _budget() -> Budget:
    return Budget(max_steps=10, max_observations=5, max_interactions=5, timeout_seconds=30, max_model_calls=10)


def _eval() -> ScenarioEvaluationTarget:
    return ScenarioEvaluationTarget(labels_by_version={"live": "Go"}, role="button")


def _verifier(text: str = "Success visible") -> VisibleResultVerifierSpec:
    return VisibleResultVerifierSpec(type="visible-result", text=text)


def _suggestion(id: str, goal: str, start_url: str) -> ScenarioSuggestion:
    return ScenarioSuggestion(
        id=id,
        name=f"Name {id}",
        goal=goal,
        start_url=start_url,
        verifier=_verifier(),
        evaluation_target=_eval(),
        budget=_budget(),
        rationale="Rationale",
        coverage=("coverage-a",),
    )


def _personas():
    return (
        Persona(
            id="p1",
            name="Persona One",
            working_memory_capacity=4,
            initial_confidence=0.6,
            initial_frustration=0.2,
            abandonment_threshold=0.8,
            attention_temperature=1.0,
        ),
        Persona(
            id="p2",
            name="Persona Two",
            working_memory_capacity=5,
            initial_confidence=0.7,
            initial_frustration=0.1,
            abandonment_threshold=0.9,
            attention_temperature=0.9,
        ),
    )


def _app():
    corpus = _corpus()
    sugg = (
        _suggestion("s1", "Goal one", "https://example.test/"),
        _suggestion("s2", "Goal two", "https://example.test/about"),
    )
    existing = _personas()[:1]
    suggested = _personas()[1:]
    app = create_exploration_app(corpus, sugg, existing_personas=existing, suggested_personas=suggested, auto_accept_flag=False)
    return app, corpus, sugg, existing, suggested


def _with_signature(app: Any, payload: Mapping[str, object]) -> dict[str, object]:
    """Chain-of-custody: echo the app's signed suggestions payload."""
    sig = getattr(app.state, "suggestions_signature", None)
    assert isinstance(sig, str) and len(sig) == 64
    return {**payload, "suggestions_signature": sig}


def test_ui_renders_suggestions_and_accepts_all() -> None:
    app, corpus, sugg, _, _ = _app()
    client = TestClient(app)
    # GET suggestions
    r = client.get("/__explore/api/suggestions")
    assert r.status_code == 200
    data = r.json()
    assert "suggestions" in data
    assert len(data["suggestions"]) == 2
    assert "corpus_summary" in data
    assert data["corpus_summary"]["pages_count"] == 2
    assert "personas" in data
    assert "existing" in data["personas"]
    assert "suggested" in data["personas"]
    # GET static SPA
    r2 = client.get("/__explore/")
    assert r2.status_code == 200
    html = r2.text
    assert "Suggested Scenarios" in html
    assert "Accept All" in html
    assert "Save & Continue" in html
    assert "Crawl Summary" in html
    # POST accept all (auto-accept same logic)
    payload = {"accepted_ids": ["s1", "s2"], "edited": [], "added": [], "persona_selection": {"mode": "existing", "persona_ids": ["p1"]}}
    r3 = client.post("/__explore/api/curate", json=_with_signature(app, payload))
    assert r3.status_code == 200, r3.text
    j = r3.json()
    assert j["status"] == "ok"
    assert j["curated_count"] == 2
    assert "fragment_yaml" in j
    assert "s1" in j["fragment_yaml"]
    assert "s2" in j["fragment_yaml"]
    assert "exploration-run" in j["fragment_yaml"]


def test_edit_persists() -> None:
    app, _, _, _, _ = _app()
    client = TestClient(app)
    # edit s1 goal and verifier
    edited = {
        "id": "s1",
        "name": "Name s1 edited",
        "goal": "Edited goal one",
        "start_url": "https://example.test/",
        "verifier": {"type": "visible-result", "text": "Edited success", "role": "heading", "all_of": []},
        "evaluation_target": {"label": "Edited target", "role": "button"},
        "budget": {"max_steps": 30, "max_observations": 12, "max_interactions": 8, "timeout_seconds": 120, "max_model_calls": 32},
        "rationale": "Edited",
        "coverage": ["edited"],
    }
    payload = {"accepted_ids": ["s2"], "edited": [edited], "added": [], "persona_selection": {"mode": "existing", "persona_ids": ["p1"]}}
    r = client.post("/__explore/api/curate", json=_with_signature(app, payload))
    assert r.status_code == 200, r.text
    j = r.json()
    assert len(j["curated"]) == 2
    # edited goal should persist
    edited_cur = next(x for x in j["curated"] if x["id"] == "s1")
    assert edited_cur["goal"] == "Edited goal one"
    assert edited_cur["verifier"]["text"] == "Edited success"
    assert edited_cur["evaluation_target"]["label"] == "Edited target"
    assert "Edited goal one" in j["fragment_yaml"]
    assert "Edited success" in j["fragment_yaml"]


def test_add_custom() -> None:
    app, _, _, _, _ = _app()
    client = TestClient(app)
    custom = {
        "id": "custom-1",
        "name": "Custom scenario",
        "goal": "Custom goal for new flow",
        "start_url": "https://example.test/",
        "verifier": {"type": "visible-result", "text": "Custom success", "role": "button", "all_of": []},
        "evaluation_target": {"label": "Custom target", "role": "button"},
        "budget": {"max_steps": 20, "max_observations": 10, "max_interactions": 5, "timeout_seconds": 60, "max_model_calls": 20},
        "rationale": "Custom rationale",
        "coverage": ["custom"],
    }
    payload = {"accepted_ids": ["s1", "s2"], "edited": [], "added": [custom], "persona_selection": {"mode": "existing", "persona_ids": ["p1"]}}
    r = client.post("/__explore/api/curate", json=_with_signature(app, payload))
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["curated_count"] == 3
    ids = {x["id"] for x in j["curated"]}
    assert "custom-1" in ids
    assert "Custom goal for new flow" in j["fragment_yaml"]
    assert "custom-1" in j["fragment_yaml"]


def test_create_custom_persona() -> None:
    app, _, _, _, _ = _app()
    client = TestClient(app)
    persona_selection = {
        "mode": "custom",
        "custom_persona": {
            "id": "custom-persona",
            "name": "Custom Persona",
            "working_memory_capacity": 6,
            "initial_confidence": 0.5,
            "initial_frustration": 0.3,
            "abandonment_threshold": 0.85,
            "attention_temperature": 1.2,
        },
    }
    payload = {"accepted_ids": ["s1"], "edited": [], "added": [], "persona_selection": persona_selection}
    r = client.post("/__explore/api/curate", json=_with_signature(app, payload))
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["persona_selection"]["mode"] == "custom"
    assert "custom-persona" in j["fragment_yaml"]
    # check sliders persisted? fragment should contain persona
    assert "Custom Persona" in j["fragment_yaml"] or "custom-persona" in j["fragment_yaml"]
    # also GET suggestions still renders persona panel
    r2 = client.get("/__explore/api/suggestions")
    assert r2.status_code == 200
    data = r2.json()
    assert "personas" in data


def test_validation_error_on_empty_verifier() -> None:
    app, _, _, _, _ = _app()
    client = TestClient(app)
    # edited with empty verifier text
    bad_edited = {
        "id": "s1",
        "name": "Bad",
        "goal": "Goal one",
        "start_url": "https://example.test/",
        "verifier": {"type": "visible-result", "text": "", "role": None, "all_of": []},
        "evaluation_target": {"label": "Target"},
        "budget": {"max_steps": 20, "max_observations": 12, "max_interactions": 8, "timeout_seconds": 120, "max_model_calls": 32},
        "rationale": "Bad",
        "coverage": [],
    }
    payload = {"accepted_ids": [], "edited": [bad_edited], "added": [], "persona_selection": {"mode": "existing", "persona_ids": ["p1"]}}
    r = client.post("/__explore/api/curate", json=_with_signature(app, payload))
    assert r.status_code == 422
    assert "verifier" in r.text.lower() or "empty" in r.text.lower()

    # also empty goal
    bad_edited2 = {
        "id": "s2",
        "name": "Bad2",
        "goal": "   ",
        "start_url": "https://example.test/",
        "verifier": {"type": "visible-result", "text": "OK", "role": None, "all_of": []},
        "evaluation_target": {"label": "Target"},
        "budget": {"max_steps": 20, "max_observations": 12, "max_interactions": 8, "timeout_seconds": 120, "max_model_calls": 32},
        "rationale": "Bad",
        "coverage": [],
    }
    r2 = client.post("/__explore/api/curate", json=_with_signature(app, {"accepted_ids": [], "edited": [bad_edited2], "added": [], "persona_selection": {"mode": "existing", "persona_ids": ["p1"]}}))
    assert r2.status_code == 422
    assert "goal" in r2.text.lower()

    # duplicate id
    dup_added = {
        "id": "s1",
        "name": "Dup",
        "goal": "Dup goal",
        "start_url": "https://example.test/",
        "verifier": {"type": "visible-result", "text": "OK", "role": None, "all_of": []},
        "evaluation_target": {"label": "Target"},
        "budget": {"max_steps": 20, "max_observations": 12, "max_interactions": 8, "timeout_seconds": 120, "max_model_calls": 32},
        "rationale": "Dup",
        "coverage": [],
    }
    r3 = client.post("/__explore/api/curate", json=_with_signature(app, {"accepted_ids": ["s1"], "edited": [], "added": [dup_added], "persona_selection": {"mode": "existing", "persona_ids": ["p1"]}}))
    assert r3.status_code == 422
    assert "duplicate" in r3.text.lower()


def test_auto_accept_same_logic_bypass() -> None:
    corpus = _corpus()
    sugg = (
        _suggestion("s1", "Goal one", "https://example.test/"),
        _suggestion("s2", "Goal two", "https://example.test/about"),
    )
    # auto-accept via helper
    result = curate_auto_accept(corpus, sugg, persona_selection={"mode": "existing", "persona_ids": ["p1"]})
    assert result["curated_count"] == 2
    assert result["auto_accept_flag"] is True
    assert "s1" in result["fragment_yaml"]
    # same validation via server endpoint accepting all
    app = create_exploration_app(corpus, sugg, auto_accept_flag=True)
    client = TestClient(app)
    r = client.post("/__explore/api/curate", json=_with_signature(app, {"accepted_ids": ["s1", "s2"], "edited": [], "added": [], "persona_selection": {"mode": "existing", "persona_ids": ["p1"]}, "auto_accept_flag": True}))
    assert r.status_code == 200
    j = r.json()
    # curated sets should be identical (ids and goals)
    auto_ids = {x["id"] for x in result["curated"]}
    manual_ids = {x["id"] for x in j["curated"]}
    assert auto_ids == manual_ids
    # fragment yaml should be equivalent (deterministic sort_keys, so compare parsed yaml)
    import yaml

    yaml_auto = yaml.safe_load(result["fragment_yaml"])
    yaml_manual = yaml.safe_load(j["fragment_yaml"])
    assert yaml_auto["scenarios"] == yaml_manual["scenarios"]


def test_loopback_bind_host() -> None:
    assert _loopback_bind_host("127.0.0.1") == "127.0.0.1"
    assert _loopback_bind_host("localhost") == "localhost"
    assert _loopback_bind_host("::1") == "::1"
    with pytest.raises(ValueError):
        _loopback_bind_host("0.0.0.0")
    with pytest.raises(ValueError):
        _loopback_bind_host("example.test")


def test_offline_static_assets_contain_no_external_urls() -> None:
    app, _, _, _, _ = _app()
    client = TestClient(app)
    assets = {
        "/__explore/explore.css": client.get("/__explore/explore.css"),
        "/__explore/explore.js": client.get("/__explore/explore.js"),
        "/__explore/": client.get("/__explore/"),
    }
    external_url = re.compile(r"https?://")
    for path, response in assets.items():
        assert response.status_code == 200, path
        match = external_url.search(response.text)
        assert match is None, f"external URL {match.group(0)!r} found in {path}"
    assert "color-scheme" in assets["/__explore/explore.css"].text
    assert "fetch" in assets["/__explore/explore.js"].text
    html = assets["/__explore/"].text
    assert "Suggested Scenarios" in html
    assert 'id="validation-error"' in html
    assert "aria-live" in html


def test_suggestions_response_is_canonical_without_aliases() -> None:
    app, _, _, _, _ = _app()
    client = TestClient(app)
    r = client.get("/__explore/api/suggestions")
    assert r.status_code == 200
    data = r.json()
    # canonical snake_case keys only: no compatibility aliases
    assert "corpus_summary" in data
    assert "auto_accept_flag" in data
    assert "corpus" not in data
    assert "auto_accept" not in data
    for suggestion in data["suggestions"]:
        assert set(suggestion) <= {
            "id",
            "name",
            "goal",
            "start_url",
            "verifier",
            "evaluation_target",
            "budget",
            "rationale",
            "coverage",
        }


def test_curate_rejects_blank_evaluation_target_label() -> None:
    app, _, _, _, _ = _app()
    client = TestClient(app)
    bad_edited = {
        "id": "s1",
        "name": "Bad target",
        "goal": "Goal one",
        "start_url": "https://example.test/",
        "verifier": {"type": "visible-result", "text": "OK", "role": None, "all_of": []},
        "evaluation_target": {"label": "   ", "role": None},
        "budget": {"max_steps": 20, "max_observations": 12, "max_interactions": 8, "timeout_seconds": 120, "max_model_calls": 32},
        "rationale": "Bad",
        "coverage": [],
    }
    r = client.post(
        "/__explore/api/curate",
        json=_with_signature(app, {"accepted_ids": [], "edited": [bad_edited], "added": [], "persona_selection": {"mode": "existing", "persona_ids": ["p1"]}}),
    )
    assert r.status_code == 422, r.text
    assert "evaluation_target" in r.text.lower()


def test_curate_rejects_unknown_evaluation_target_version_id() -> None:
    app, _, _, _, _ = _app()
    client = TestClient(app)
    bad_added = {
        "id": "custom-1",
        "name": "Custom scenario",
        "goal": "Custom goal",
        "start_url": "https://example.test/",
        "verifier": {"type": "visible-result", "text": "OK", "role": None, "all_of": []},
        "evaluation_target": {"labels_by_version": {"prod-v9": "Target"}, "role": None},
        "budget": {"max_steps": 20, "max_observations": 12, "max_interactions": 8, "timeout_seconds": 120, "max_model_calls": 32},
        "rationale": "Custom",
        "coverage": [],
    }
    r = client.post(
        "/__explore/api/curate",
        json=_with_signature(app, {"accepted_ids": ["s1"], "edited": [], "added": [bad_added], "persona_selection": {"mode": "existing", "persona_ids": ["p1"]}}),
    )
    assert r.status_code == 422, r.text
    assert "unknown application version id" in r.text.lower()
    assert "prod-v9" in r.text


def test_curate_strict_payload_rejects_aliases_and_unknown_keys() -> None:
    app, _, _, _, _ = _app()
    client = TestClient(app)
    base = {"accepted_ids": ["s1"], "edited": [], "added": [], "persona_selection": {"mode": "existing", "persona_ids": ["p1"]}}
    # camelCase alias must be rejected, not silently accepted
    r = client.post("/__explore/api/curate", json=_with_signature(app, {**base, "acceptedIds": ["s2"]}))
    assert r.status_code == 422, r.text
    assert "acceptedIds" in r.text
    # unknown keys must be rejected
    r2 = client.post("/__explore/api/curate", json=_with_signature(app, {**base, "mystery_key": 1}))
    assert r2.status_code == 422, r2.text
    assert "mystery_key" in r2.text


def test_invalid_persona_input_returns_422() -> None:
    app, _, _, _, _ = _app()
    client = TestClient(app)
    invalid_persona_selection = {
        "mode": "custom",
        "custom_persona": {
            "id": "bad-persona",
            "name": "",
            "working_memory_capacity": 0,
            "initial_confidence": 0.5,
            "initial_frustration": 0.3,
            "abandonment_threshold": 0.85,
            "attention_temperature": -1,
        },
    }
    r = client.post(
        "/__explore/api/curate",
        json=_with_signature(app, {"accepted_ids": ["s1"], "edited": [], "added": [], "persona_selection": invalid_persona_selection}),
    )
    assert r.status_code == 422, r.text
    assert "working_memory_capacity" in r.text


# ---------------------------------------------------------------------------
# Review UI regression tests (critic round 2)
# ---------------------------------------------------------------------------

REPORTING_STATIC_DIR = (
    Path(ux_analyzer.__file__).resolve().parent / "reporting" / "static"
)
EXPLORE_JS_PATH = REPORTING_STATIC_DIR / "explore.js"


def _suggestion_payload(id: str, start_url: str) -> dict[str, object]:
    """Canonical scenario dict exactly as the suggestions API serializes it."""
    return {
        "id": id,
        "name": f"Name {id}",
        "goal": f"Goal {id}",
        "start_url": start_url,
        "verifier": {"type": "visible-result", "text": "Success visible", "role": None, "all_of": []},
        "evaluation_target": {"label": "Go", "role": "button"},
        "budget": {"max_steps": 10, "max_observations": 5, "max_interactions": 5, "timeout_seconds": 30, "max_model_calls": 10},
        "rationale": "Rationale",
        "coverage": ["coverage-a"],
    }


def _ui_state_after_custom_add() -> dict[str, object]:
    """Mirror of the JS `state` object after a user edits s1, adds a custom
    scenario via the form (which sets accepted[custom-1]=true), and then also
    edits the added custom scenario — i.e. exactly what buildCuratePayload
    must consume without ever emitting an id twice."""
    return {
        "accepted": {"s1": True, "s2": True, "custom-1": True},
        "edits": {
            "s1": {**_suggestion_payload("s1", "https://example.test/"), "goal": "Edited goal one"},
            "custom-1": {**_suggestion_payload("custom-1", "https://example.test/about"), "goal": "Edited custom goal"},
        },
        "added": [_suggestion_payload("custom-1", "https://example.test/about")],
        "personaSelection": {"mode": "existing", "persona_ids": ["p1"]},
        "autoAccept": False,
    }


async def test_curate_payload_builder_js_contract_end_to_end() -> None:
    """Execute the REAL buildCuratePayload from explore.js in Chromium and
    POST its exact output to the curate API.

    Regression: the old builder put a custom-scenario id in BOTH accepted_ids
    and added, so the server _claim guard raised a guaranteed 422 on every
    custom-scenario save. The server must still reject duplicated ids loudly.
    """
    js_source = EXPLORE_JS_PATH.read_text(encoding="utf-8")
    app, *_ = _app()
    # the builder must echo the chain-of-custody signature from UI state
    ui_state = {
        **_ui_state_after_custom_add(),
        "suggestionsSignature": app.state.suggestions_signature,
    }
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("about:blank")
        await page.add_script_tag(content=js_source)
        payload = await page.evaluate(
            "(s) => window.buildCuratePayload(s)", ui_state
        )
        await browser.close()

    assert payload["suggestions_signature"] == app.state.suggestions_signature
    accepted_ids = payload["accepted_ids"]
    edited_ids = [entry["id"] for entry in payload["edited"]]
    added_ids = [entry["id"] for entry in payload["added"]]
    claimed = [*accepted_ids, *edited_ids, *added_ids]
    assert len(claimed) == len(set(claimed)), f"id claimed twice: {claimed}"
    # custom scenarios appear ONLY in added; edits replace their originals
    assert added_ids == ["custom-1"]
    assert "custom-1" not in accepted_ids
    assert edited_ids == ["s1"]
    assert accepted_ids == ["s2"]
    # an edit of an added scenario folds back into added, once
    assert payload["added"][0]["goal"] == "Edited custom goal"

    app, *_ = _app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        r = await ac.post("/__explore/api/curate", json=payload)
        assert r.status_code == 200, r.text
        assert r.json()["curated_count"] == 3
        # negative control: a shape where one id is claimed twice (here s1 in
        # both accepted_ids and edited) is rejected loudly by _claim
        bad = {**payload, "accepted_ids": [*payload["accepted_ids"], "s1"]}
        r2 = await ac.post("/__explore/api/curate", json=bad)
    assert r2.status_code == 422
    assert "duplicate scenario id" in r2.text.lower()


async def test_persona_selection_payload_matches_selected_mode() -> None:
    """Execute the REAL buildCuratePayload in Chromium and pin persona_selection
    normalization for every mode.

    Regression: touching the custom-persona form (or an older localStorage
    blob) left custom_persona on the state even after switching back to the
    existing/suggested radio, so Save & Continue sent a payload the server
    rejects with "custom_persona is only allowed with mode 'custom'".
    """
    js_source = EXPLORE_JS_PATH.read_text(encoding="utf-8")
    base_state = _ui_state_after_custom_add()
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("about:blank")
        await page.add_script_tag(content=js_source)

        async def build(persona_selection: dict[str, object]) -> dict[str, object]:
            state = {**base_state, "personaSelection": persona_selection}
            payload = await page.evaluate(
                "(s) => window.buildCuratePayload(s)", state
            )
            return payload["persona_selection"]

        existing = await build(
            {
                "mode": "existing",
                "persona_ids": ["p1"],
                "custom_persona": {"id": "custom-persona", "name": "Leftover"},
            }
        )
        suggested = await build({"mode": "suggested", "persona_ids": ["p2"]})
        custom = await build(
            {
                "mode": "custom",
                "custom_persona": {"id": "custom-persona", "name": "Mine"},
                "persona_ids": ["p1"],
            }
        )
        default_mode = await build({})
        await browser.close()

    # stale custom_persona is stripped outside custom mode
    assert existing == {"mode": "existing", "persona_ids": ["p1"]}
    assert suggested == {"mode": "suggested", "persona_ids": ["p2"]}
    # custom mode carries ONLY the persona; persona_ids are dropped
    assert custom == {
        "mode": "custom",
        "custom_persona": {"id": "custom-persona", "name": "Mine"},
    }
    # missing selection normalizes to existing with empty ids
    assert default_mode == {"mode": "existing", "persona_ids": []}

    # end-to-end: the previously-failing shape now saves through the API
    app, *_ = _app()
    ui_state = {
        **_ui_state_after_custom_add(),
        "personaSelection": {
            "mode": "existing",
            "persona_ids": ["p1"],
            "custom_persona": {"id": "custom-persona", "name": "Leftover"},
        },
        "suggestionsSignature": app.state.suggestions_signature,
    }
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("about:blank")
        await page.add_script_tag(content=EXPLORE_JS_PATH.read_text(encoding="utf-8"))
        payload = await page.evaluate(
            "(s) => window.buildCuratePayload(s)", ui_state
        )
        await browser.close()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        r = await ac.post("/__explore/api/curate", json=payload)
    assert r.status_code == 200, r.text


def test_review_page_single_source_with_unified_a11y_and_no_inline_styles() -> None:
    app, *_ = _app()
    client = TestClient(app)
    r = client.get("/__explore/")
    assert r.status_code == 200
    html = r.text
    # no inline-style leakage in the served page
    assert 'style="' not in html and "style='" not in html
    # unified alert markup on the validation error region
    m = re.search(r'<span id="validation-error"[^>]*>', html)
    assert m, html
    tag = m.group(0)
    assert 'role="alert"' in tag
    assert 'aria-live="assertive"' in tag
    # exactly ONE aria-live region exists in markup and at runtime; JS may
    # retune that single element in place (success -> status/polite,
    # error -> alert/assertive) but never adds another region — the runtime
    # probes below assert [aria-live] count stays 1 through render and save.
    assert html.count("aria-live") == 1, html.count("aria-live")
    assert html.count('role="alert"') == 1
    js = EXPLORE_JS_PATH.read_text(encoding="utf-8")
    # retuning targets the existing region by id; nothing else may speak
    assert 'getElementById("validation-error")' in js
    # primary actions start disabled until suggestions load (honest degradation)
    m_save = re.search(r'<button id="save-continue"[^>]*>', html)
    assert m_save and "disabled" in m_save.group(0)
    # chain-of-custody: signed suggestion set embedded in the page
    assert 'name="uxa-suggestions-signature"' in html
    # reset control present (curation persistence is clearable)
    assert 'id="reset-curation"' in html
    # the divergent static copy is gone; template is the only source
    assert not (REPORTING_STATIC_DIR / "explore.html").exists()


async def test_review_ui_browser_keyboard_persistence_reset_and_save() -> None:
    corpus = _corpus()
    sugg = (
        _suggestion("s1", "Goal one", "https://example.test/"),
        _suggestion("s2", "Goal two", "https://example.test/about"),
    )
    existing, suggested = _personas()[0:1], _personas()[1:]
    server = ExplorationReviewServer(
        corpus,
        sugg,
        existing_personas=existing,
        suggested_personas=suggested,
        no_browser=True,
    )
    await server.start_in_background()
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(server.url)
            await page.wait_for_selector(".suggestion-card")
            cards = page.locator(".suggestion-card")
            assert await cards.count() == 2

            # roving tabindex: exactly one tabbable card at rest
            tabs = await page.eval_on_selector_all(
                ".suggestion-card", "els => els.map(e => e.tabIndex)"
            )
            assert tabs == [0, -1]

            # Arrow/Home/End navigation across cards
            active_id_expr = "document.activeElement.dataset.id"
            await cards.nth(0).focus()
            await page.keyboard.press("ArrowDown")
            assert await page.evaluate(active_id_expr) == "s2"
            await page.keyboard.press("ArrowUp")
            assert await page.evaluate(active_id_expr) == "s1"
            await page.keyboard.press("End")
            assert await page.evaluate(active_id_expr) == "s2"
            await page.keyboard.press("Home")
            assert await page.evaluate(active_id_expr) == "s1"

            # Enter toggles acceptance on the focused card
            await page.keyboard.press("End")
            await page.keyboard.press("Enter")
            checked_expr = "els => els.map(e => e.checked)"
            checked = await page.eval_on_selector_all(
                ".suggestion-checkbox", checked_expr
            )
            assert checked == [True, False]

            # curation state survives reload via localStorage
            await page.reload()
            await page.wait_for_selector(".suggestion-card")
            checked = await page.eval_on_selector_all(
                ".suggestion-checkbox", checked_expr
            )
            assert checked == [True, False]

            # Reset clears persisted state back to defaults
            await page.click("#reset-curation")
            await page.wait_for_load_state("load")
            await page.wait_for_selector(".suggestion-card")
            checked = await page.eval_on_selector_all(
                ".suggestion-checkbox", checked_expr
            )
            assert checked == [True, True]

            # add a custom scenario through the real form and save end-to-end:
            # must NOT hit the historical duplicate-id 422
            await page.click("#add-custom-btn")
            await page.fill("#custom-id", "custom-ui")
            await page.fill("#custom-goal", "Custom goal from UI")
            await page.select_option("#custom-url", "https://example.test/")
            await page.fill("#custom-verifier", "Visible success text")
            await page.fill("#custom-target", "Custom target")
            await page.click("#custom-save")
            assert await cards.count() == 3
            # single counting source: displayed count must equal the curated
            # set size (accepted originals + added), never double-count the
            # added scenario via its accepted flag
            count_text = await page.text_content("#scenario-count")
            assert count_text is not None and count_text.strip() == "3 scenarios selected"
            await page.click("#save-continue")
            banner = page.locator("#success-banner")
            await banner.wait_for(state="visible")
            text = await banner.text_content()
            assert text is not None and "Curated 3" in text
            # success speaks through the SAME single live region, retuned to
            # polite status semantics (errors use alert/assertive)
            live = await page.evaluate(
                """() => {
                  const regions = document.querySelectorAll('[aria-live]');
                  const r = document.getElementById('validation-error');
                  return {
                    regions: regions.length,
                    role: r.getAttribute('role'),
                    politeness: r.getAttribute('aria-live'),
                    text: r.textContent,
                  };
                }"""
            )
            assert live["regions"] == 1
            assert live["role"] == "status"
            assert live["politeness"] == "polite"
            assert "Curated 3" in (live["text"] or "") and "successfully" in (live["text"] or "")
            await browser.close()

        # the curated result reached the server intact
        curated_result = server.app.state.curated_result
        assert curated_result is not None
        assert curated_result["curated_count"] == 3
        curated_ids = {item["id"] for item in curated_result["curated"]}
        assert curated_ids == {"s1", "s2", "custom-ui"}
    finally:
        await server.stop()


# ---------------------------------------------------------------------------
# Review UI regression tests (critic round 3)
# ---------------------------------------------------------------------------


def test_page_embeds_signature_matching_canonical_suggestions_payload() -> None:
    """Chain-of-custody: the served page embeds sha256(canonical JSON) of the
    exact suggestions payload the API serves."""
    import hashlib
    import json

    app, *_ = _app()
    client = TestClient(app)
    html = client.get("/__explore/").text
    m = re.search(
        r'<meta name="uxa-suggestions-signature" content="([0-9a-f]{64})">', html
    )
    assert m, html
    data = client.get("/__explore/api/suggestions").json()
    canonical = json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == m.group(1)
    assert m.group(1) == app.state.suggestions_signature


def test_curate_rejects_missing_or_stale_suggestions_signature() -> None:
    app, _, _, _, _ = _app()
    client = TestClient(app)
    base = {
        "accepted_ids": ["s1"],
        "edited": [],
        "added": [],
        "persona_selection": {"mode": "existing", "persona_ids": ["p1"]},
    }
    # missing signature -> rejected before any curation logic runs
    r = client.post("/__explore/api/curate", json=base)
    assert r.status_code == 422, r.text
    assert "suggestions_signature" in r.text
    # stale/tampered signature -> 422 mismatch
    stale = {**base, "suggestions_signature": "0" * 64}
    r2 = client.post("/__explore/api/curate", json=stale)
    assert r2.status_code == 422, r2.text
    assert "mismatch" in r2.text.lower()
    # control: the correct echo is accepted
    ok = client.post("/__explore/api/curate", json=_with_signature(app, base))
    assert ok.status_code == 200, ok.text


async def _probe_badge(auto_accept: bool) -> dict[str, object]:
    corpus = _corpus()
    sugg = (_suggestion("s1", "Goal one", "https://example.test/"),)
    server = ExplorationReviewServer(
        corpus, sugg, auto_accept=auto_accept, no_browser=True
    )
    await server.start_in_background()
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(server.url)
            await page.wait_for_selector(".suggestion-card")
            info = await page.evaluate(
                """() => {
                  const b = document.getElementById('auto-accept-badge');
                  const cs = getComputedStyle(b);
                  return {
                    display: cs.display,
                    hiddenClass: b.classList.contains('hidden'),
                    styleAttr: b.getAttribute('style'),
                    liveRegions: document.querySelectorAll('[aria-live]').length,
                  };
                }"""
            )
            await browser.close()
    finally:
        await server.stop()
    return info


async def test_auto_accept_badge_visibility_via_class_toggle_only() -> None:
    """Regression: badge used inline style.display which lost to
    `.hidden{display:none !important}` — it could never become visible.
    Visibility must now come from class toggling alone."""
    on = await _probe_badge(auto_accept=True)
    assert on["styleAttr"] is None  # no inline style writes anywhere
    assert on["hiddenClass"] is False
    assert on["display"] != "none"
    off = await _probe_badge(auto_accept=False)
    assert off["hiddenClass"] is True
    assert off["display"] == "none"
    # single aria-live region even after dynamic cards are rendered
    assert on["liveRegions"] == 1


async def test_suggestions_load_failure_degrades_honestly_with_retry() -> None:
    """Regression: a failed load left dead panels + one error string. Now an
    explicit error panel explains retry/abort and Save & Continue stays
    disabled until the load succeeds."""
    corpus = _corpus()
    sugg = (_suggestion("s1", "Goal one", "https://example.test/"),)
    server = ExplorationReviewServer(corpus, sugg, no_browser=True)
    await server.start_in_background()
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()

            async def _abort(route: Any) -> None:
                await route.abort()

            await page.route("**/__explore/api/suggestions", _abort)
            await page.goto(server.url)
            await page.wait_for_selector("#suggestions-error")
            panel_text = (await page.text_content("#suggestions-error")) or ""
            assert "retry" in panel_text.lower()
            assert "abort" in panel_text.lower()
            assert not panel_text.strip().endswith(":")
            assert await page.is_disabled("#save-continue")
            assert await page.is_disabled("#accept-all")
            assert await page.locator(".suggestion-card").count() == 0
            assert await page.locator(".load-error").count() == 1

            # recovery path: retry loads the set and re-enables actions
            await page.unroute("**/__explore/api/suggestions")
            await page.click("#retry-load")
            await page.wait_for_selector(".suggestion-card")
            assert not await page.is_disabled("#save-continue")
            assert not await page.is_disabled("#accept-all")
            assert await page.locator("#suggestions-error").count() == 0
            await browser.close()
    finally:
        await server.stop()


# ---------------------------------------------------------------------------
# Review UI regression tests (critic round 4)
# ---------------------------------------------------------------------------


def _curation_blob(signature: str, **overrides: Any) -> str:
    """A persisted curation blob as explore.js stores it (signature-pinned)."""
    blob: dict[str, object] = {
        "signature": signature,
        "accepted": {"s1": False},
        "edits": {},
        "added": [],
    }
    blob.update(overrides)
    return json.dumps(blob)


_SET_SEED_JS = "(v) => localStorage.setItem('uxa-explore-curation', v)"


async def test_stored_curation_ignored_on_signature_mismatch() -> None:
    """A curation blob signed for a different suggestion set must never
    resurface: it is ignored wholesale, evicted from storage, and the
    discard is announced through the single live region."""
    server = ExplorationReviewServer(
        _corpus(),
        (
            _suggestion("s1", "Goal one", "https://example.test/"),
            _suggestion("s2", "Goal two", "https://example.test/about"),
        ),
        no_browser=True,
    )
    await server.start_in_background()
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(server.url)
            await page.wait_for_selector(".suggestion-card")
            # blob pinned to a different (stale) suggestion set: unaccepts s1
            await page.evaluate(_SET_SEED_JS, _curation_blob("f" * 64))
            await page.reload()
            await page.wait_for_selector(".suggestion-card")
            checked = await page.eval_on_selector_all(
                ".suggestion-checkbox", "els => els.map(e => e.checked)"
            )
            # stored unaccept of s1 was NOT re-applied: defaults win
            assert checked == [True, True]
            live_text = await page.text_content("#validation-error")
            assert live_text is not None and "ignored" in live_text.lower()
            # the mismatched blob was evicted from storage
            stored = await page.evaluate("localStorage.getItem('uxa-explore-curation')")
            assert stored is None
            await browser.close()
    finally:
        await server.stop()


async def test_stale_curation_entries_dropped_and_save_succeeds() -> None:
    """Restored curation referencing ids the server no longer offers, or a
    start_url unknown to the corpus, is dropped with a count announcement;
    a save immediately afterwards succeeds instead of detonating as an
    unfixable 422 mid-flow."""
    corpus = _corpus(("https://example.test/", "https://example.test/about"))
    sugg = (
        _suggestion("s1", "Goal one", "https://example.test/"),
        _suggestion("s2", "Goal two", "https://example.test/about"),
    )
    server = ExplorationReviewServer(corpus, sugg, no_browser=True)
    await server.start_in_background()
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(server.url)
            await page.wait_for_selector(".suggestion-card")
            seed = {
                "signature": server.app.state.suggestions_signature,
                "accepted": {"s1": False, "ghost-id": True},
                "edits": {"s9-gone": _suggestion_payload("s9-gone", "https://example.test/")},
                "added": [
                    _suggestion_payload("custom-ok", "https://example.test/"),
                    _suggestion_payload("custom-gone", "https://gone.example.test/"),
                ],
            }
            await page.evaluate(_SET_SEED_JS, json.dumps(seed))
            await page.reload()
            await page.wait_for_selector(".suggestion-card")
            cards = page.locator(".suggestion-card")
            # ghost accepted key + unknown edit id + unknown start_url added
            # entry are all dropped; only custom-ok survives -> 3 cards
            assert await cards.count() == 3
            checked = await page.eval_on_selector_all(
                ".suggestion-checkbox", "els => els.map(e => e.checked)"
            )
            assert checked == [False, True, True]
            live_text = await page.text_content("#validation-error")
            assert live_text is not None and "3 stale entries discarded" in live_text
            # displayed count equals curated set size (s2 + custom-ok),
            # proving no double count after restore
            count_text = await page.text_content("#scenario-count")
            assert count_text is not None and count_text.strip() == "2 scenarios selected"
            # save right after restore succeeds end-to-end — no stale ids sent
            await page.click("#save-continue")
            banner = page.locator("#success-banner")
            await banner.wait_for(state="visible")
            text = await banner.text_content()
            assert text is not None and "Curated 2" in text
            await browser.close()

        curated_result = server.app.state.curated_result
        assert curated_result is not None
        assert curated_result["curated_count"] == 2
        curated_ids = {item["id"] for item in curated_result["curated"]}
        assert curated_ids == {"s2", "custom-ok"}
    finally:
        await server.stop()
