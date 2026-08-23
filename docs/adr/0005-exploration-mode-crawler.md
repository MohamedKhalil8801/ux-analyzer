# ADR 0005: Exploration Mode with Depth-Bounded Crawler and Human-in-Loop Scenario Synthesis

- Status: Proposed
- Date: 2026-08-23
- Scope: Live-site discovery, depth-bounded crawling, dynamic-page settling, LLM scenario suggestion, and local review UI
- Decision owner: UX Analyzer maintainers

## Decision Summary

Add `uxa explore` as separate operator workflow for live targets. Operator supplies one or more starting HTTPS URLs, crawl depth, and max pages. Crawler does BFS same-origin crawl with deduplication. Per-page settling uses adaptive strategy: `networkidle` + incremental scroll sweep + mutation stabilization, bounded by configurable settle budget (default 10s). No turnkey external library fully auto-handles dynamic pages; recommendation is hardened Playwright loop reusing existing `PlaywrightSessionAdapter` + `NetworkPolicy`, not introducing `crawlee` as hard dependency (evaluate as optional later). Crawl produces immutable exploration artifact. Cognitive model (`UXA_COGNITIVE_MODEL`) analyzes crawl corpus and proposes up to N scenarios (configurable, max 20). Local web UI lets operator accept, edit, add, or define personas; `--auto-accept` bypasses UI. Accepted scenarios materialize as `project.yaml` fragment and feed existing `RunAgent`/`ExperimentRunner` pipeline unchanged. Verification for exploration scenarios is `visible-result` only; `fixture-state` forbidden.

## Context

Classic benchmark requires pre-authored `scenarios` with known start_state and fixture verifier. Live-site users lack this. Requirement is: start from URLs + depth → auto-discover site structure → LLM suggests covering scenarios → human curates → run analysis as today. Dynamic pages require scroll-triggered and delayed content.

Research 2026-08-23:

- No Python library auto-handles loading bars + scroll-reveal + lazy-load without custom wait/scroll logic.
- `crawlee[s:playwright]` (Python) provides `PlaywrightCrawler(max_crawl_depth, max_requests_per_crawl, enqueue_links)` with dedup, queue, auto-scaling, but still requires custom `request_handler` scroll + `waitForSelector`/`wait_for_load_state('networkidle')` per page. It adds storage, session pool, and Apify coupling. See `crawlee.dev/python/api/class/PlaywrightCrawler` (max_crawl_depth, max_requests_per_crawl) and guides recommending explicit scroll loops + `expect.poll`/`waitForResponse`.
- Alternative independent stack: direct Playwright `page.wait_for_load_state('networkidle')` + `locator.scrollIntoViewIfNeeded()` loop bounded by `maxScrollHeight/waitForSecs/scrollDownAndUp` (ported from `@crawlee/playwright` `infiniteScroll`) plus JS `window.scrollBy` + height stabilization + mutation observer.

Decision keeps crawler inside current modular monolith boundary.

## Options Considered

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| Depend on `crawlee[playwright]` | Queue, dedup, concurrency, retries for free | Extra dep (~20k LOC), own storage, hidden browser pool conflicts with existing adapter, still needs custom scroll logic | Deferred — evaluate as Task 3 spike, not default |
| Custom `ExplorationCrawler` on `PlaywrightSessionAdapter` | Reuses existing allowlist, extraction, trace, settle semantics; smaller surface | Must reimplement BFS/dedup/budget | **Chosen** |
| Pure HTTP (requests/BeautifulSoup) | Fast | Fails on JS-rendered, scroll-revealed content | Rejected |
| Scrapy+Splash | Large-scale HTTP | Headless rendering weaker than Playwright for modern SPAs | Rejected |

## Dynamic Page Strategy (Chosen)

Per-page settle budget `page_settle_ms` (operator-configurable, default 8000-10000):

1. `goto` with `wait_until: domcontentloaded` then `page.wait_for_load_state('networkidle', timeout=4000)` (tolerate polling sites: fallback to `load`).
2. Wait for visible progress indicators (`[role=progressbar]`, `.loading`, `.spinner`) to detach, with 2s cap.
3. Incremental scroll sweep: `window.scrollBy(0, viewportHeight*0.8)` in loop, after each scroll `wait_for_timeout(400)` + `wait_for_load_state('networkidle', timeout=800)` + recompute `document.body.scrollHeight`; stop when height stable for 2 iterations or `maxScrollHeight` reached or budget exhausted. Use `scrollIntoViewIfNeeded` for IntersectionObserver targets when available.
4. Mutation stabilization: `page.wait_for_function('prev => document.body.innerHTML.length === prev', initialLen, timeout=1000)` opportunistic.
5. Capture final `ViewportSnapshot` + screenshot aligned (reuse `capture_with_diagnostics`).

Bounded; never infinite.

## Crawl Boundary

- Same-origin only: discovered link must share origin with its parent start origin. Origins derived from starting URLs. No cross-origin follow. `allowed_origins` still explicit for sub-resources, but crawl frontier restricted to start origins.
- Depth: BFS. Depth 0 = starts only. Depth N = up to N hops. `max_crawl_depth` semantics match crawlee.
- Dedup: normalize URL (lowercase host, strip fragment, sort query keys, remove tracking params, collapse `//`, respect canonical). Seen set is `set[str]`.
- Caps: `depth` 0-5 (default 2), `max_pages` 1-200 (default 50). Fail-closed on limit.
- Respect `robots.txt` optionally (default off for UX audit, flag `--respect-robots`).

## Scenario Synthesis

- Input: per-page compressed evidence: url, depth, title, headings, visible element labels/roles, screenshot hash refs (bounded). Not full DOM to fit context.
- Provider: `StructuredCognitiveAgent` with new prompt `exploration-synthesis-v1` and schema `exploration-synthesis-v1` (goal, verifier text/role, evaluation_target, rationale, coverage tags). Uses cognitive model; token budget capped at ~120k input via chunking + summarization.
- Output: up to `max_suggested_scenarios` (1-20, default 8) with `goal`, `verifier: visible-result`, `start_url`, `evaluation_target`, `budget`, `safeguards: [live-*]`.
- No fixture inputs, no fixture-state verifier.

## Review UI

- Local FastAPI server (reuse fixture_appinfra) at `http://127.0.0.1:<port>/__explore` serving static SPA. No external requests.
- Operations: list suggestions, accept toggle, edit JSON, add custom scenario, manage personas (choose existing, model-suggested, or custom with `working_memory_capacity`, `attention_temperature` etc.), `--auto-accept` skips server and accepts all.
- Persist: immutable exploration artifact under `<output>/exploration/<attempt-id>/`.

## Consequences

- `ApplicationVersionKind.LIVE` remains; exploration start URLs map to live versions.
- Existing `uxa run` / project YAML unchanged; exploration is pre-phase producing scenarios.
- Security: crawl respects `BrowserAllowedOrigins` origin allowlist for resources; frontier blocked for foreign origins. Host-side model calls remain separate.
- Deterministic fallback: crawl or synthesis failure leaves artifact with `status: unavailable` and operator can retry.

## Alternatives Deferred

- Crawlee integration spike if operator wants Apify-scale features.
- `UXA_EXPLORE_MODEL` separate model ID (currently reuse cognitive).
- `respect_robots` default on.

## Review Trigger

Accept after:

1. BFS depth + dedup + same-origin tests pass for depth 0/1/2 and cycle case.
2. Scroll/networkidle settle captures lazy content in fixture live site test.
3. Exploration artifact immutability + digest checks.
4. UI accept/edit/add persona flow e2e passes offline.
5. Synthesis produces valid `visible-result` scenarios within token budget without leaking private fields.
6. `--auto-accept` produces identical `project.yaml` as manual accept-all.
7. Non-live regression still passes.
