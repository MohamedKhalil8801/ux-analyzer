# Exploration Mode with Depth-Bounded Smart Crawler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` or `executing-plans` to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add `uxa explore` exploration workflow: same-origin BFS crawl from start URLs up to configurable depth (0 = starts only, up to `max_pages`), smart page settlement (networkidle + scroll sweep + mutation), cognitive-model synthesis of up to 20 covering scenarios, and local web UI for accept/edit/add with auto-accept bypass. Accepted scenarios feed existing run pipeline unchanged.

**Architecture:** New exploration phase sits before classic benchmark execution. It reuses `PlaywrightSessionAdapter` + `BrowserAllowedOrigins` + deterministic extraction pipeline. Crawl produces immutable `CrawlCorpus` artifact with per-page snapshots. `ExplorationSynthesizer` (cognitive role) proposes `visible-result` scenarios. `ExplorationReviewServer` (local FastAPI) provides human-in-loop curation. `ExplorationArtifactStore` persists immutable attempt. On accept, a generated project fragment is merged into a runnable experiment and handed to `ExperimentRunner`. No `uxa run` contract break.

**Tech Stack:** Python 3.12, Playwright, FastAPI (local review UI), Pydantic v2, Typer, Jinja2 vanilla JS/CSS, existing OpenAI-compatible / Codex transport, pytest.

**Research Finding (2026-08-23) on dynamic pages:**

- No Python crawling library fully auto-handles loading bars + delayed AJAX + scroll-revealed content without custom logic. Recommended handling is explicit: `wait_for_load_state('networkidle')` + progress-detach wait + incremental `window.scrollBy` loop comparing `document.body.scrollHeight` stability + `wait_for_selector` / mutation wait.
- `crawlee[playwright]` Python (`PlaywrightCrawler(max_crawl_depth, max_requests_per_crawl, enqueue_links: {strategy: same-origin})`) provides queue/dedup/concurrency but still needs custom scroll/networkidle handler per request. It adds heavy dep and alternate storage/pool that conflicts with `PlaywrightSessionAdapter`. Verdict: do **not** adopt crawlee as default; reuse hardened Playwright loop, keep crawlee as documented alternative spike (Task 3 note). Sources: `crawlee.dev/python/api/class/PlaywrightCrawler` (`max_crawl_depth`, `max_requests_per_crawl`, `enqueue_links`), Apify guides on infinite scroll (`page.mouse.wheel`/`scrollIntoViewIfNeeded` + `waitForResponse`/`expect.poll`), and `@crawlee/playwright` `infiniteScroll` options (`maxScrollHeight`, `waitForSecs`, `scrollDownAndUp`, `buttonSelector`).

## Global Constraints

- Existing `project.yaml` and `uxa run/ablate/report/synthesize` stay valid; default exploration disabled.
- `uxa explore` is separate command; it does not mutate existing CLI flags.
- Crawl frontier is same-origin only; cross-origin document follow blocked. Resource allowlist stays exact-origin via `BrowserAllowedOrigins`.
- Per-page settlement bounded (default 10s); crawl depth 0–5, max_pages 1–200 (default 50), max_scenarios 1–20 (default 8) configurable.
- Exploration scenarios are live-only: `visible-result` verifier only; `fixture-state` rejected. `fixture_inputs` must be empty when referencing live versions.
- No private data (selectors, hidden labels, dest URLs, fixture keys, test ids) enters model prompts or review UI payload beyond redacted allowlist.
- Crawled pages compressed: url, depth, title, headings, persona-visible element labels/roles/bounds, screenshot hash ref (bounded). Full DOM not sent to LLM.
- Local review UI is offline, no external requests; `--auto-accept` must produce same curated set as manual accept-all.
- All new artifacts immutable, checksummed, atomically published.

## File Map

- Create `src/ux_analyzer/domain/exploration.py`: `ExplorationSpec`, `CrawlPage`, `CrawlCorpus`, `ScenarioSuggestion`, `ExplorationStatus`.
- Create `src/ux_analyzer/application/exploration_crawler.py`: `ExplorationCrawler`, `PageSettlementPolicy`, BFS, dedup, same-origin filter.
- Create `src/ux_analyzer/application/exploration_synthesizer.py`: `ExplorationSynthesizer` with cognitive prompt/schema `exploration-synthesis-v1`.
- Create `src/ux_analyzer/storage/exploration_artifacts.py`: `ExplorationArtifactStore`, immutable layout + digests.
- Create `src/ux_analyzer/application/exploration_review.py`: `ExplorationReviewService`, persona resolution (existing / suggested / custom).
- Create `src/ux_analyzer/adapters/exploration_server.py` or `src/ux_analyzer/reporting/exploration_ui.py`: FastAPI local review server + static SPA.
- Modify `src/ux_analyzer/config/models.py`, `src/ux_analyzer/config/loader.py`: exploration config, live-only validation, URL normalization helpers.
- Modify `src/ux_analyzer/cli.py`: `uxa explore` command, `--starting-url`, `--depth`, `--max-pages`, `--max-scenarios`, `--auto-accept`, `--review-port`, `--output`.
- Modify `src/ux_analyzer/domain/benchmark.py`: URL canonicalization + same-origin helper.
- Modify `src/ux_analyzer/ports/observation.py` / `adapters/web/session.py` if needed for settle timing.
- Docs: `docs/adr/0005-*`, `docs/glossary.md`, `docs/architecture.md`, `docs/run-bundle-format.md`, `README.md`.

---

### Task 1: Define Exploration Domain Contracts

**Files:**
- Create: `src/ux_analyzer/domain/exploration.py`
- Test: `tests/unit/domain/test_exploration.py`

**Interfaces:**
- Produces: `ExplorationSpec(start_urls, depth, max_pages, max_scenarios, settle_ms, allowed_origins)`, `CrawlPage(url, depth, normalized_url, origin, title, headings, viewport_id, screenshot_digest, discovered_links)`, `CrawlCorpus(pages, link_graph, started_at, corpus_digest)`, `ScenarioSuggestion(id, goal, start_url, verifier, evaluation_target, budget, rationale, coverage)`, `ExplorationStatus`.

- [ ] **Step 1: Write failing domain invariants**

```python
def test_exploration_spec_depth_zero_allows_single_start():
    spec = ExplorationSpec(start_urls=("https://example.test/",), depth=0, max_pages=1)
    assert spec.depth == 0

def test_crawl_corpus_is_immutable():
    corpus = CrawlCorpus(pages=(page_a,), link_graph={...})
    with pytest.raises((AttributeError, TypeError)):
        corpus.pages = ()
```

Also cover: duplicate start rejection, depth 0 => max_pages >= len(starts), invalid HTTPS rejection, max_scenarios 1–20, budget positivity, verifier type must be visible-result, persona id validity.

- [ ] **Step 2: Run failing**

Run: `rtk uv run pytest tests/unit/domain/test_exploration.py -q`
Expected: collection fails, module missing.

- [ ] **Step 3: Implement frozen dataclasses**

`@dataclass(frozen=True, slots=True)` with `MappingProxyType` for maps, tuple normalization, `canonicalize_https_url`/`canonicalize_http_origin` reuse, SHA-256 digest helper.

- [ ] **Step 4: Run passing**

Run: `rtk uv run pytest tests/unit/domain/test_exploration.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/ux_analyzer/domain/exploration.py tests/unit/domain/test_exploration.py
rtk git commit -m "feat: define exploration domain contracts"
```

### Task 2: Configuration and URL Normalization

**Files:**
- Modify: `src/ux_analyzer/config/models.py`, `src/ux_analyzer/config/loader.py`, `src/ux_analyzer/domain/benchmark.py`
- Test: `tests/unit/config/test_exploration_config.py`

**Interfaces:**
- Consumes: Task 1 domain.
- Produces: `ExplorationConfigModel`, loader validation, `normalize_url(url)->str`, `same_origin(a,b)->bool`.

- [ ] **Step 1: Write failing config tests**

```python
def test_exploration_accepts_multiple_starts_depth_zero(tmp_path):
    proj = _base_project()
    proj["exploration"] = {"start_urls": ["https://a.test/", "https://a.test/about"], "depth": 0, "max_pages": 2}
    loaded = load_project(_write(tmp_path, proj))
    assert loaded.exploration.max_pages == 2

def test_exploration_rejects_cross_origin_without_allowlist():
    proj = _base_project()
    proj["exploration"] = {"start_urls": ["https://a.test/"], "depth": 1, "max_pages": 10}
    # enqueue link https://evil.test/ must be dropped by crawler, not config error
    assert True

def test_exploration_max_scenarios_bound():
    proj = _bad({"max_scenarios": 30})
    with pytest.raises(ProjectConfigError):
        load_project(proj)
```

Also: `depth` 0–5, `max_pages` 1–200, `settle_ms` 0–15000, duplicate normalized starts rejected, query-sort normalization (`?b=2&a=1` → `?a=1&b=2`), fragment stripped, tracking param filter optional.

- [ ] **Step 2: Run RED**

`rtk uv run pytest tests/unit/config/test_exploration_config.py -q` → fails.

- [ ] **Step 3: Implement models + helpers**

Add `ExplorationModel(_ConfigModel)` with `Literal` bounds, reuse `canonicalize_https_url` plus new `normalize_crawl_url`. In `loader._validate_references`, if exploration present, validate start origins are HTTPS and unique after normalization. In `domain/benchmark.py`, add `normalize_crawl_url` (stdlib `urllib.parse`): strip fragment, sort query, remove `utm_*`/`fbclid`, lowercase host, remove default ports, collapse slashes, handle trailing slash consistency.

- [ ] **Step 4: Run GREEN**

`rtk uv run pytest tests/unit/config/test_exploration_config.py tests/unit/domain/test_exploration.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/ux_analyzer/config/models.py src/ux_analyzer/config/loader.py src/ux_analyzer/domain/benchmark.py tests/unit/config/test_exploration_config.py
rtk git commit -m "feat: add exploration config and URL normalization"
```

### Task 3: Depth-Bounded Smart Crawler

**Files:**
- Create: `src/ux_analyzer/application/exploration_crawler.py`
- Modify: `src/ux_analyzer/adapters/web/extractor.py` (settle helper), `src/ux_analyzer/ports/observation.py`
- Test: `tests/unit/application/test_exploration_crawler.py`, `tests/integration/web/test_exploration_crawl.py`

**Interfaces:**
- Consumes: `ExplorationSpec`, `ObservationProvider`, existing extraction.
- Produces: `ExplorationCrawler.crawl(spec) -> CrawlCorpus`, `PageSettlementPolicy`.

- [ ] **Step 1: Write failing crawler behavior tests**

```python
@pytest.mark.asyncio
async def test_depth_zero_crawls_only_starts(fake_provider):
    spec = ExplorationSpec(start_urls=("https://a.test/",), depth=0, max_pages=10)
    corpus = await ExplorationCrawler(provider=fake_provider).crawl(spec)
    assert len(corpus.pages) == 1
    assert corpus.pages[0].depth == 0

@pytest.mark.asyncio
async def test_same_origin_drops_cross_origin():
    # start https://a.test/ page links to https://evil.test/x → ignored
    assert "https://evil.test/x" not in {p.url for p in corpus.pages}

@pytest.mark.asyncio
async def test_dedup_removes_fragment_and_sorted_query():
    # https://a.test/p?b=2&a=1#frag and https://a.test/p?a=1&b=2 → one page
    assert len(corpus.pages) == 2  # start + one unique child

@pytest.mark.asyncio
async def test_max_pages_caps_bfs():
    spec = ExplorationSpec(start_urls=("https://a.test/",), depth=5, max_pages=3)
    assert len((await crawler.crawl(spec)).pages) <= 3
```

Also: BFS order guarantee, cycle `A→B→A` visited once, viewport capture uses settled DOM, scroll sweep triggers lazy content (fake provider simulates height growth), networkidle fallback on polling page, budget timeout returns partial corpus with `status=partial`.

Spike note (doc test, not implementation): evaluate `crawlee[playwright]` alternative; record in code comment why custom loop chosen. Keep `pip install crawlee[playwright]` as optional alternative documented in `docs/architecture.md`—no code import unless spike passes.

- [ ] **Step 2: RED**

`rtk uv run pytest tests/unit/application/test_exploration_crawler.py -q` → fails.

- [ ] **Step 3: Implement BFS + settle**

Pseudo:

```python
from collections import deque
seen: set[str] = set(normalize(u) for u in spec.start_urls)
queue: deque[tuple[str,int]] = deque((u,0) for u in spec.start_urls)
pages: list[CrawlPage] = []
while queue and len(pages) < spec.max_pages:
  url, depth = queue.popleft()
  snapshot = await settle_and_capture(url)  # networkidle + loading-detach + scroll sweep
  discovered = extract_same_origin_links(snapshot)  # <a href> absolute, filter same_origin
  pages.append(page_for(url, depth, snapshot))
  if depth < spec.depth:
    for link in discovered:
      n = normalize(link)
      if n not in seen and same_origin(link, spec.start_origins) and len(pages)+len(queue) < spec.max_pages:
        seen.add(n); queue.append((link, depth+1))
```

`settle_and_capture`:

```python
async def settle_and_capture(page, url, settle_ms=10000):
  await page.goto(url, wait_until='domcontentloaded', timeout=15000)
  try: await page.wait_for_load_state('networkidle', timeout=4000)
  except: await page.wait_for_load_state('load', timeout=2000)
  # progress detach
  with suppress(TimeoutError): await page.locator('[role=progressbar], .spinner, [aria-busy=true]').wait_for(state='detached', timeout=2000)
  # scroll sweep
  start = time.monotonic(); prev_h = await page.evaluate('document.body.scrollHeight')
  stable = 0
  while time.monotonic()-start < settle_ms/1000 and stable < 2:
    await page.evaluate('window.scrollBy(0, window.innerHeight*0.8)')
    await page.wait_for_timeout(400)
    try: await page.wait_for_load_state('networkidle', timeout=800)
    except: pass
    h = await page.evaluate('document.body.scrollHeight')
    stable = stable+1 if h==prev_h else 0
    prev_h = h
  await page.evaluate('window.scrollTo(0,0)')
  return await capture_snapshot_with_diagnostics(page, viewport_id)
```

Bounded, no infinite.

- [ ] **Step 4: GREEN**

`rtk uv run pytest tests/unit/application/test_exploration_crawler.py tests/integration/web/test_exploration_crawl.py -q` → PASS (use fake provider for unit, real Playwright with local fixture for integration).

- [ ] **Step 5: Commit**

```bash
rtk git add src/ux_analyzer/application/exploration_crawler.py src/ux_analyzer/adapters/web/extractor.py tests/unit/application/test_exploration_crawler.py tests/integration/web/test_exploration_crawl.py
rtk git commit -m "feat: add depth-bounded smart crawler with settlement"
```

### Task 4: Cognitive Scenario Synthesis

**Files:**
- Create: `src/ux_analyzer/application/exploration_synthesizer.py`, `src/ux_analyzer/providers/exploration_prompts.py`
- Test: `tests/unit/application/test_exploration_synthesizer.py`, `tests/integration/application/test_exploration_synthesizer.py`

**Interfaces:**
- Consumes: `CrawlCorpus`, `StructuredModelClient`, `UXA_COGNITIVE_MODEL`.
- Produces: `ExplorationSynthesizer.suggest(corpus, max_scenarios) -> tuple[ScenarioSuggestion, ...]`.

- [ ] **Step 1: Failing synthesis tests**

```python
@pytest.mark.asyncio
async def test_suggest_produces_visible_result_scenarios(recording_client):
    sugg = await ExplorationSynthesizer(recording_client, model="test-model").suggest(corpus_with_3_pages, max_scenarios=5)
    assert 1 <= len(sugg) <= 5
    assert all(s.verifier.type == "visible-result" for s in sugg)
    assert all(not s.verifier.text.strip()=="" for s in sugg)
    # no fixture inputs leaked
    assert "invite_email" not in recording_client.last_prompt

@pytest.mark.asyncio
async def test_suggest_respects_max_scenarios():
    sugg = await synthesizer.suggest(large_corpus, max_scenarios=20)
    assert len(sugg) <= 20
```

Also: prompt excludes selectors/hidden labels/dest URLs, payload is per-page compressed evidence (url, title, headings, visible elements), chunking when corpus > 80k tokens, retry on invalid JSON via `json_object` fallback, duplicate goals deduped.

- [ ] **Step 2: RED**

`rtk uv run pytest tests/unit/application/test_exploration_synthesizer.py -q` → fails.

- [ ] **Step 3: Implement**

- New prompt `exploration-synthesis-v1.txt`: describes UX tech lead, gives crawl summary format, asks for covering scenarios (discovery, task completion, navigation, forms), requires `visible-result` verifier with concrete text that is rendered in viewport, requires `evaluation_target` label/role, forbids fixture-state.
- Schema `exploration-synthesis-v1`: `{scenarios: [{id, name, goal, start_url, verifier: {type:"visible-result", text, role?, all_of?}, evaluation_target, rationale, coverage}]}`
- `ExplorationSynthesizer` builds per-page evidence packs: truncate headings to 3 per page, element labels to top 30 per page by prominence, total cap 25 pages or 100k chars; if overflow, summarize pages in batches via cheap map-reduce (first pass per-page TL;DR, second pass synthesis).
- Validate `all_of` unique, text non-empty, `start_url` subset of corpus URLs.

- [ ] **Step 4: GREEN**

`rtk uv run pytest tests/unit/application/test_exploration_synthesizer.py tests/integration/application/test_exploration_synthesizer.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/ux_analyzer/application/exploration_synthesizer.py src/ux_analyzer/providers/exploration_prompts.py tests/unit/application/test_exploration_synthesizer.py
rtk git commit -m "feat: add cognitive scenario synthesis for exploration"
```

### Task 5: Immutable Exploration Artifact Store

**Files:**
- Create: `src/ux_analyzer/storage/exploration_artifacts.py`
- Test: `tests/integration/storage/test_exploration_artifacts.py`

**Interfaces:**
- Produces: `ExplorationArtifactStore.write_attempt(corpus, suggestions, curated, persona_set) -> Path`, `load_attempt`, `index.json`.

Layout:

```text
<output>/exploration/
  index.json  # {schema_version, attempts:[{attempt_id, created_at, corpus_digest, status}]}
  attempts/<attempt-id>/
    corpus.json
    suggestions.json
    curated.json
    project.fragment.yaml
    manifest.json  # digests, spec, model, prompt_version
```

`attempt_id = <utc Z>-<first12(corpus_digest)>-<seq>` (same collision-safe scheme as synthesis). Atomic via staging rename. Digest = SHA-256 of canonical JSON + corpus pages.

- [ ] **Step 1: Failing artifact tests**

```python
def test_write_is_atomic_and_immutable(store):
    a = store.write_attempt(corpus, suggestions, curated)
    b = store.write_attempt(corpus, suggestions, curated)  # new seq, not overwrite
    assert a != b
    with pytest.raises(FileExistsError):
        store.write_attempt(corpus, suggestions, curated, attempt_id=a.name)
```

Also: checksum mismatch detection, path traversal rejection, malformed id rejection.

- [ ] **Step 2: RED** → missing module.
- [ ] **Step 3: Implement** (reuse `secure_create_exclusive_file`, `secure_replace_exclusive_file` from `storage/run_bundle.py`).
- [ ] **Step 4: GREEN**
- [ ] **Step 5: Commit**

### Task 6: Local Web UI for Human-in-Loop Curation

**Files:**
- Create: `src/ux_analyzer/adapters/exploration_server.py`
- Create: `src/ux_analyzer/reporting/static/explore.html`, `explore.js`, `explore.css` (or `src/ux_analyzer/reporting/templates/explore.html.j2`)
- Test: `tests/integration/reporting/test_exploration_ui.py`

**Interfaces:**
- Produces: `ExplorationReviewServer(port, corpus, suggestions, existing_personas) -> curated tuple + persona set` plus REST-ish endpoints (all same-origin, no CORS):
  - `GET /__explore/api/suggestions` → {suggestions, corpus_summary, personas:{existing, suggested}}
  - `POST /__explore/api/curate` → {accepted_ids, edited[], added[], persona_selection, auto_accept_flag}
  - `GET /__explore/` → static SPA

SPA features:
- Left: crawl summary (pages count, depth, urls)
- Center: suggestion cards (goal, start_url, verifier text, rationale, coverage tags). Each card: checkbox accept, expand edit (name, goal, verifier text/role, evaluation_target, budget sliders), duplicate, delete, "Add custom scenario" form.
- Right: persona panel: radio choose existing personas, or model-suggested personas (if synthesizer emits `suggested_personas`), or "Create custom persona" (capacity, confidence, frustration, temperature sliders with live defaults).
- Top bar: `--auto-accept` badge, count, Save & Continue button.
- Validation inline: empty goal/verifier → red; duplicate scenario id → error.
- On Save, POST curated set; server validates (visible-result only, non-empty verifier, known start_url) and returns `project.fragment.yaml` preview; then shuts down gracefully.

- [ ] **Step 1: Failing UI tests (Playwright)**

```python
def test_ui_renders_suggestions_and_accepts_all(page, exploration_server):
    page.goto(exploration_server.url)
    expect(page.get_by_text("Suggested Scenarios")).to_be_visible()
    page.get_by_role("button", name="Accept All").click()
    page.get_by_role("button", name="Save & Continue").click()
    expect(page.get_by_text("Curated")).to_be_visible()
```

Also: edit scenario goal persists, add custom scenario, create custom persona, validation error on empty verifier.

- [ ] **Step 2: RED**
- [ ] **Step 3: Implement FastAPI server**

Reuse `uvicorn` runner, bind to loopback only (`_loopback_bind_host` helper). Serve static via `StaticFiles`. API validates with Pydantic, writes curated.json atomically, signals `asyncio.Event` to unblock CLI.

Design: server runs in same event loop as `uxa explore` command; CLI opens browser via `webbrowser.open` optionally (`--no-browser` disables). Timeout for human action is unbounded (operator must Ctrl-C to abort).

- [ ] **Step 4: GREEN** (Playwright e2e with fake corpus).
- [ ] **Step 5: Commit**

### Task 7: CLI `uxa explore` Wiring

**Files:**
- Modify: `src/ux_analyzer/cli.py`, `src/ux_analyzer/config/loader.py`, `src/ux_analyzer/application/experiment.py`
- Test: `tests/integration/cli/test_explore_command.py`

**Interfaces:**
- Produces: `uxa explore --starting-url URL [--starting-url URL ...] --depth N --max-pages M --max-scenarios K [--auto-accept] [--review-port PORT] [--output DIR] [--project PROJECT]`

Spec:

```bash
uxa explore benchmarks/demo/project.yaml \
  --starting-url https://example.test/ \
  --starting-url https://example.test/docs \
  --depth 2 --max-pages 50 --max-scenarios 10 \
  --output .uxa-output --auto-accept
```

Without `--auto-accept`, launches review UI then waits for curation. With it, skips UI, uses all suggestions.

Behavior:

1. Load base project (to reuse personas/providers) if `--project` given; else bootstrap minimal `BenchmarkProject` with one `Application(live)` per unique start origin.
2. Build `ExplorationSpec` from flags (flag wins over YAML exploration section).
3. Run `ExplorationCrawler` → `CrawlCorpus`.
4. Run `ExplorationSynthesizer` → suggestions (+ optional persona suggestions from same LLM call or second prompt).
5. If `--auto-accept`: curated = suggestions.
   Else: start `ExplorationReviewServer`, block until POST curated, validate.
6. Persist via `ExplorationArtifactStore`.
7. Materialize `project.fragment.yaml` or full `project.yaml` in output with new `scenarios` + `experiments` (one experiment `exploration-run` covering all curated scenarios × selected personas × default policies). Also keep original scenarios if base project provided (merge strategy: append).
8. Optionally `--run` flag immediately invokes `ExperimentRunner` on generated experiment (or print next step: `uxa run <generated> --experiment exploration-run`).

- [ ] **Step 1: Failing CLI tests**

```python
def test_explore_depth_zero_auto_accept(tmp_path):
    result = runner.invoke(app, ["explore", str(demo_project), "--starting-url", "https://a.test/", "--depth", "0", "--max-pages", "1", "--auto-accept", "--output", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / "exploration" / "index.json").exists()

def test_explore_rejects_invalid_depth():
    result = runner.invoke(app, ["explore", str(demo_project), "--depth", "10"])
    assert result.exit_code != 0
```

Also: duplicate start rejection, max_pages < starts rejection, requires model env unless --dry-run, exploration artifact digest stability.

- [ ] **Step 2: RED**
- [ ] **Step 3: Implement** (Typer command, flag validation, async orchestration via `asyncio.run`).
- [ ] **Step 4: GREEN** (dry-run + fake model).
- [ ] **Step 5: Commit**

### Task 8: End-to-End Pipeline & Dry-Run Support

**Files:**
- Modify: `src/ux_analyzer/cli.py` (dry-run preview for explore), `docs/run-bundle-format.md`, `README.md`, `docs/architecture.md`
- Test: `tests/e2e/test_exploration_flow.py`

**Interfaces:**
- Covers: `uxa explore --dry-run` prints crawl matrix (pages estimate) + synthesis token estimate; `uxa explore --auto-accept --run` does full crawl→suggest→store→run→report without UI.

- [ ] **Step 1: Failing e2e**

```python
@pytest.mark.asyncio
async def test_e2e_explore_then_run(tmp_path, fake_crawler, fake_synthesizer):
    # dry-run
    r = runner.invoke(app, ["explore", str(demo), "--starting-url", "https://a.test/", "--depth", "1", "--auto-accept", "--dry-run"])
    assert "explore matrix" in r.stdout
    # real with fakes
    r = runner.invoke(app, ["explore", str(demo), "--starting-url", "https://a.test/", "--depth", "1", "--auto-accept", "--output", str(tmp_path)])
    assert r.exit_code == 0
    # generated project is runnable
    r2 = runner.invoke(app, ["run", str(tmp_path/"exploration"/"generated.yaml"), "--experiment", "exploration-run", "--dry-run"])
    assert r2.exit_code == 0
```

Also: `uxa report` offline after exploration-run, exploration artifact appears in report index.

- [ ] **Step 2: RED**
- [ ] **Step 3: Implement dry-run + --run forwarding**
- [ ] **Step 4: GREEN** (with fixture live site + fake model).
- [ ] **Step 5: Document** (`docs/run-bundle-format.md` new exploration layout, `README` commands table, `architecture.md` new flow diagram).
- [ ] **Step 6: Commit**

### Task 9: Final Verification and Review Gates

**Files:** Modify only files implicated by failures.

- [ ] **Step 1: Unit + integration focused**

```bash
rtk uv run pytest tests/unit/domain/test_exploration.py tests/unit/config/test_exploration_config.py tests/unit/application/test_exploration_crawler.py tests/unit/application/test_exploration_synthesizer.py -q
rtk uv run pytest tests/integration/web/test_exploration_crawl.py tests/integration/storage/test_exploration_artifacts.py tests/integration/cli/test_explore_command.py tests/integration/reporting/test_exploration_ui.py -q
```

- [ ] **Step 2: Non-live regression**

```bash
rtk uv run pytest -m "not live" -q
rtk uv run ruff check .
rtk uv run ruff format --check .
rtk uv run pyright
rtk git diff --check
```

- [ ] **Step 3: Manual smoke with real site (opt-in)**

```bash
uv run uxa explore benchmarks/demo/project.yaml --starting-url https://example.test --depth 1 --max-pages 5 --max-scenarios 4 --output .uxa-output --auto-accept --dry-run
```

- [ ] **Step 4: Review dispatched** (security: allowlist, SSRF, path traversal; correctness: BFS + dedup; accessibility: UI labels).

- [ ] **Step 5: Commit fixes**

```bash
rtk git add -u
rtk git commit -m "fix: close exploration verification gaps"
```

## Acceptance Criteria

- `uxa explore --starting-url URL --depth N` crawls same-origin BFS to N hops, dedup via normalized URL, caps at `max_pages`, respects per-page settle budget and captures scroll-revealed content.
- Dynamic pages with loading spinner + lazy scroll are captured after settlement (verified with fixture lazy page).
- Cognitive synthesis produces 1–20 `visible-result` scenarios (configurable), each with goal, start_url in corpus, non-empty verifier text, and budgets; no private fields leaked.
- Local web UI shows suggestions, allows accept/edit/add/custom persona, validates inline, publishes immutable artifact on Save.
- `--auto-accept` produces audit-identical curated set without UI.
- Exploration artifact is immutable, checksummed, indexed, and not overwritable.
- Accepted scenarios generate runnable `project.yaml` that `uxa run --dry-run` accepts and `uxa run` executes via existing pipeline.
- All pre-existing non-live tests and strict checks pass; new code respects clean-architecture boundaries (domain stdlib-only, no browser/model types in domain).

## Open Decisions Resolved

- Separate `uxa explore` (not flag on `uxa run`) — per operator answer.
- Same-origin crawl only — per answer.
- Strict BFS + maxPages — per answer.
- Dynamic pages: no turnkey lib; hardened Playwright loop chosen; crawlee documented as alternative — per research.
- Local web UI + `--auto-accept` — per answer.
- Cognitive model, up to 20 configurable — per answer.
- Persona chooser (existing / suggested / custom) in UI — per answer.
- Immutable artifact — per answer.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Crawl explosion via calendar/faceted links | Normalized dedup + max_pages + depth cap + link count per page cap (e.g., 200) |
| Polling site prevents networkidle | Fallback to `load` + bounded 4s timeout |
| LLM token overflow (50 pages) | Per-page compression + chunked summarization |
| UI port collision | `--review-port` + random free port fallback |
| Operator abandons UI | Ctrl-C aborts; artifact stays `pending`, no partial write |
| SSRF via starting-url | Validate HTTPS only, loopback allowed only for tests, BrowserAllowedOrigins enforced |

## Rollout

- Phase 1: domain + crawler (Tasks 1–3)
- Phase 2: synthesis + artifact store (4–5)
- Phase 3: UI + CLI (6–7)
- Phase 4: e2e + docs (8–9)
- No breaking change; exploration is additive.

