# Scenario discovery

`uxa explore` finds the tasks worth measuring. It crawls a live site, asks a
model to propose scenarios from what is actually on the pages, lets you curate
them, and writes out a project you can run immediately.

It sits before the normal run pipeline. It never modifies the
`uxa run` / `report` / `synthesize` contracts — it only produces inputs for them.

```text
starting URLs
  -> ExplorationSpec        (flags win over the YAML exploration section)
  -> ExplorationCrawler     (same-origin BFS, per-page settlement)
  -> CrawlCorpus            (immutable, digested)
  -> ExplorationSynthesizer (cognitive model, compressed per-page evidence)
  -> ScenarioSuggestion set (visible-result verifiers only)
  -> curation               (local review UI, or --auto-accept)
  -> immutable attempt + index under <output>/exploration/
  -> generated project.yaml with experiment `exploration-run`
  -> uxa run <project.yaml> --experiment exploration-run
```

## Starting URLs and the crawl boundary

Starting URLs come from `--starting-url` (repeatable) or, when no flag is
given, from `exploration.start_urls` in the project YAML. If neither supplies
one, the command exits with:

```text
starting-url is required (provide --starting-url or set
exploration.start_urls in project)
```

**HTTPS is required.** A non-HTTPS starting URL is rejected. URLs are
canonicalized before anything else happens: the host is lowercased, a default
port is elided, `//` is collapsed, the fragment is stripped, query parameters
are sorted, and tracking parameters are removed. Two starting URLs that
normalize to the same value are a load error.

### Bounds

| Setting | Flag | YAML field | Default | Range |
| --- | --- | --- | --- | --- |
| Crawl depth | `--depth` | `exploration.depth` | `2` | 0–5 |
| Page cap | `--max-pages` | `exploration.max_pages` | `50` | 1–200 |
| Scenario cap | `--max-scenarios` | `exploration.max_scenarios` | `8` | 1–20 |
| Page settle time | — | `exploration.settle_ms` | `10000` ms | 0–15000 ms |
| Robots handling | — | `exploration.respect_robots` | `false` | — |

Flags win over YAML. The effective values are validated twice — once for the
flags you passed and once for the merged result — so a bad combination fails
before any browser starts. Errors are explicit:

```text
depth must be between 0 and 5
max_pages must be between 1 and 200
max_scenarios must be between 1 and 20
max_pages must be >= number of start URLs when depth is 0
starting-url must be unique after normalization (duplicate detected)
```

There is no `--settle-ms` and no `--respect-robots` flag; those two are YAML-only.

### The page settlement policy

A page is not captured the moment `DOMContentLoaded` fires. Settlement, in
order:

1. navigate with `domcontentloaded` and a 15-second timeout
2. wait for `networkidle` for up to 4 seconds, falling back to `load` for 2
   seconds
3. wait up to 2 seconds for the configured progress indicators to detach
4. run a scroll sweep — up to 40 steps, bounded by `settle_ms` — so lazily
   revealed content actually renders, recording the labels each scroll position
   reveals so synthesis can anchor on below-fold content too
5. jump back to the top instantly and wait for that to settle
6. capture

A `goto` failure does not abort the crawl; the page is still captured, just
without scroll-sweep labels. The crawler prints a `crawl matrix:` line with the
resolved start count, depth, page cap, and scenario cap before it begins.

### Where the workspace goes

`--output` defaults to `.uxa-output/explore/<site-slug>/`, where the slug is the
first starting URL's hostname, lowercased with non-alphanumeric characters
collapsed to hyphens.

## How the corpus is built

`ExplorationCrawler` is a breadth-first, same-origin crawler.

- It starts from the normalized starting URLs.
- It enqueues only same-origin `<a href>` targets. Out-of-origin links are
  dropped during normalization, and each page keeps at most 200 discovered
  same-origin links.
- Deduplication is by normalized URL, so `/dashboard`, `/dashboard/`, and
  `/dashboard?utm_source=x` are one page.
- The frontier is depth-bounded: at `--depth 0` only the starting URLs are
  captured; each level deeper adds the links found at the previous level.
- The page cap counts captured pages plus queued targets, so a wide frontier
  stops the crawl before the cap is exceeded.

Each captured page is recorded as a `CrawlPage` with its URL, normalized URL,
origin, depth, title, headings, region labels, up to 30 persona-visible element
labels (whitespace-collapsed and truncated to 80 characters each), discovered
links, viewport id, and screenshot digest.

### What is and is not in a corpus page

The corpus is a redacted projection, not a DOM dump. It holds:

- normalized URL, origin, depth, title
- headings
- region labels
- persona-visible element labels
- discovered links
- viewport id and screenshot digest

It does **not** hold raw DOM, CSS selectors, test IDs, hidden labels,
accessibility-only names, destination URLs, or any private application state.
This matters because the corpus is fed to a model.

The corpus carries a single SHA-256 `corpus_digest` computed over its canonical
serialization: pages sorted by normalized URL, a fixed field set per page, and
the link graph with sorted targets. Identical content always yields an identical
digest, and content that is not crawl-corpus shaped is an error rather than
being hashed with a fallback scheme. The digest is printed after the crawl as:

```text
crawl corpus: 12 pages digest=3f9a1c8e0b21
```

## How candidate scenarios are synthesized

`ExplorationSynthesizer` calls the cognitive model with compressed per-page
evidence packs. It does not send the model a page dump.

**Per-page pack fields** — an explicit allowlist of exactly these keys:

| Field | Bound |
| --- | --- |
| `url` | normalized |
| `depth` | integer |
| `title` | truncated to 120 characters |
| `headings` | at most 3, each truncated to 120 characters |
| `visible_elements` | at most 30 persona-visible labels, each truncated to 80 characters |

**Whole-request bounds:**

| Bound | Value |
| --- | --- |
| Pages fed to synthesis | 25 |
| Total pack characters per request | 100,000 |
| Characters per call when chunking | 80,000 |
| Scenarios per run | 1–20, from `--max-scenarios` |

The 25-page bound is a hard cap on what the synthesizer sees, independent of
`--max-pages`. If the crawl captured more than 25 pages, only the first 25 in
corpus order are used and the result carries an explicit limitation naming the
pages received, the pages included, the pages dropped, and the source corpus
digest. A crawler configured above 25 pages therefore costs crawl time the
synthesizer does not read.

If the packs exceed the character budget, the synthesizer compresses before it
sends: first by cutting visible elements per page (20, then 10, then 5), then
headings, then the title. If that still does not fit, it switches to compact
per-page summaries. If a request still overflows, the packs are split into
chunks, each chunk is synthesized separately, and the results are merged,
deduplicated, and capped globally. Chunking stops early once twice
`max_scenarios` candidates have been collected.

Compression is never silent. Page-cap drops, summary degradation, and budget
hits each append a limitation string and increment a counter on the result. A
budget that even per-page summaries cannot satisfy raises an error instead of
shipping an arbitrary subset.

**Validation applied to every proposal before it becomes a suggestion:**

- the verifier must be `visible-result` with non-empty text
- the `start_url` must be a URL that is actually in the corpus
- `all_of` entries must be non-empty and unique
- the goal must be non-empty, and duplicate goals are deduplicated
- the evaluation target label and region must appear in the corpus
- at most `--max-scenarios` survive, chosen by sorting on id for determinism

**Failure handling.** Operational transport failure marks the result
`unavailable` and is not retried. Invalid structured output is retried at most
once, then marked `invalid`. Every rejected proposal gets a sanitized audit
record with a reason code and a SHA-256 digest of the payload it attempted.
Model-authored narrative strings are redacted before conversion into domain
objects. A model call never fabricates a suggestion to keep the pipeline going.

A valid outcome of zero suggestions produces a warning, not an error:

```text
warning: synthesis produced 0 scenarios — the crawl had 4 page(s) with sparse
visible labels; try --depth 1 for broader coverage or add a custom scenario in
the review UI.
```

## The review UI

Unless you pass `--auto-accept`, the command starts a local review server on
`127.0.0.1` and opens your browser at:

```text
http://127.0.0.1:<port>/__explore/
```

`--review-port` picks the port; the default is the first free port. Use
`--no-browser` to start the server without opening a browser. The server serves
no external requests, exposes no CORS headers, and shuts down as soon as you
save.

### Three panels

**Crawl Summary** — page count, maximum depth reached, every crawled URL, page
titles, crawl start time, corpus digest, and the link graph. Below it sit two
routes to the same result: an expandable **Add custom scenario form** and an
**Add custom scenario** button. The form takes ID, name, goal, start URL (chosen
from the crawled pages), verifier text, and evaluation target label.
**Add custom scenario** adds it to the curated set immediately; the ID must be
unique.

**Suggested Scenarios** — one card per suggestion showing its goal, start URL,
verifier text, rationale, and coverage tags, with:

- an accept checkbox, labeled `Accept <id>`
- **Edit** — an inline panel with Name, Goal, Verifier text, Verifier role,
  Evaluation target label, Evaluation target role, four budget sliders (Max
  steps 1–50, Max observations 1–30, Max interactions 1–30, Max model calls
  1–64), a Timeout seconds field (10–600, empty meaning progress-based only),
  and **Save edit**
- **Duplicate** — a copy with a fresh id
- **Delete**

Every card is keyboard reachable: Enter or Space on a focused card toggles
acceptance.

**Persona Selection** — choose among personas already defined in the base
project, or define a custom persona inline with fields for ID and name plus
sliders for working-memory capacity (1–10), initial confidence, initial
frustration, attention temperature (0.1–2.0), and abandonment threshold. The
selection decides which personas the generated experiment uses; a custom persona
is written into the generated project. The payload also has a `suggested` mode,
but the current synthesizer does not propose personas, so that list is empty
unless you supply one.

### Top-bar operations

| Control | What it does |
| --- | --- |
| **Accept All** | Accepts every suggestion, original and added. |
| **Reset** | Clears the saved curation and reloads the page. |
| **Save & Continue** | Validates, previews the generated YAML, persists the attempt, and shuts the server down. |
| Scenario count badge | Live count of selected scenarios. |
| Auto-accept badge | Visible when the server was started in auto-accept mode. |

**Accept All** and **Save & Continue** are disabled until the suggestion set has
loaded. **Save & Continue** is blocked while any validation message is active.
The live validation region reports:

```text
Goal must not be empty
Verifier text must not be empty
Evaluation target label must not be empty
Duplicate scenario id: <id>
```

After a successful save the page shows a `project.fragment.yaml` preview, the
curated count, and a confirmation banner.

### Validation the UI applies

Curation is validated server-side, not just in the browser. An invalid
submission returns HTTP 422 with the specific reasons joined by `; `:

- `unknown suggestion id: <id>`
- `duplicate scenario id: <id>`
- `curated entries must be objects`
- any field that fails the canonical scenario schema, reported as
  `scenario <id>: <message>`
- `scenario <id>: start_url not in corpus: <url>` — the start URL must be a page
  the crawl actually reached
- `scenario <id>: start_url invalid: <detail>`

### Chain of custody

The served suggestion set is signed with a `suggestions_signature` embedded in
the page. A curation payload whose signature does not match the currently served
set is rejected before any curation logic runs:

```text
suggestions_signature mismatch: the suggestion set is stale or was tampered
with; reload the review page to get the current signed payload
```

That is what stops a stale browser tab from submitting against a different
suggestion set.

## Auto-accept

`--auto-accept` skips the server entirely. The curated set equals the suggested
set, and the same validation runs — it is not a bypass. It is the right choice
for a scripted or CI run:

```powershell
uv run uxa explore --starting-url https://example.com --auto-accept
```

If the curated set ends up empty, the command exits with:

```text
curated set is empty; nothing to materialize or run (accept at least one
scenario in the review UI or via --auto-accept)
```

## Resume and checkpointing

The crawl is the expensive part, so progress is checkpointed. After every
completed page the frontier is persisted; after the crawl, the corpus is saved;
after synthesis, the suggestions are saved. The checkpoint lives at:

```text
<output>/exploration/checkpoint/
```

Its `state.json` and `payload.json` record a settings fingerprint derived from
the starting URLs, depth, page cap, scenario cap, settle time, and allowed
origins. `state.json` carries the schema version, a `pending` status, the
fingerprint, whether suggestions exist, whether the crawl finished, and an
update timestamp. `payload.json` carries the corpus, the suggestions, and — while
the crawl is in flight — the frontier's queue and visited set. The payload is
written before the state, so a crash mid-write leaves no valid checkpoint.
Checkpoint writes are best-effort: a disk failure never kills a healthy crawl.

Without `--resume`, a matching checkpoint is used automatically:

```text
resume: reusing checkpointed crawl corpus (12 pages); crawl skipped
resume: reusing checkpointed suggestions (5 scenarios); synthesis skipped
```

An interrupted crawl resumes from its saved frontier:

```text
crawling: resuming interrupted crawl (7 pages done, 11 queued)
```

With `--resume` but a checkpoint whose fingerprint does not match the current
settings, the command fails loudly rather than mixing two crawls:

```text
resume requested but the existing checkpoint does not match current
exploration settings (starting URLs, depth, max-pages, max-scenarios, or
settle-ms)
```

Once curation, persistence, and materialization finish, the checkpoint is
deleted. If you interrupt during the crawl with Ctrl+C:

```text
explore interrupted; crawl checkpoint retained for resume (rerun the same
command to continue)
```

and the command exits 130.

## What gets written where

The attempt is published first, then the runnable project files.

### The immutable attempt

```text
<output>/exploration/
  index.json
  attempts/
    <attempt-id>/
      corpus.json
      suggestions.json
      curated.json
      project.fragment.yaml
      manifest.json
  checkpoint/                     # while a crawl is in flight
```

Attempt ids use the collision-safe form
`<created-at-utc>-<first12(corpus_digest)>-<sequence>`, where the timestamp is
path-safe and ends in `Z`, the digest component is twelve lowercase hex
characters, and the sequence is a positive integer. An existing attempt
directory is never overwritten; a rerun publishes the next sequence.

| File | What it holds |
| --- | --- |
| `corpus.json` | The canonical serialized `CrawlCorpus`. |
| `suggestions.json` | The synthesizer's proposals, as received. |
| `curated.json` | The accepted/edited set. Identical to `suggestions.json` under `--auto-accept`. |
| `project.fragment.yaml` | YAML preview of the scenarios and experiment this attempt would produce. |
| `manifest.json` | Schema version, attempt id and creation time, corpus digest, SHA-256 digests of the other four files, the resolved `ExplorationSpec`, model id, prompt version, status, page count, scenario count. |

All JSON files use one canonical rule: ASCII, sorted keys, compact separators,
one terminating newline. A digest mismatch makes an attempt unreadable rather
than silently trusted.

Attempt status is one of `pending`, `running`, `succeeded`, `partial`, `failed`,
`unavailable`, `completed`. Publication is atomic: files are written into a
same-parent hidden staging directory, fsynced, and the complete directory is
renamed into `attempts/`. The index is written through a temporary file and
atomically replaced under an inter-process publication lock.

If synthesis fails outright, the command persists a real attempt with status
`unavailable` and a `limitation` field reading
`scenario synthesis unavailable: <reason>`, then exits — it never fabricates
suggestions:

```text
synthesis failed: <ErrorType>: <message>
```

`index.json` carries `schema_version: exploration-index-v1` and an `attempts`
list; each record has `attempt_id`, `created_at`, `status`, `corpus_digest`,
`page_count`, `scenario_count`, and `manifest_digest`. The index references only
attempt directories that exist on disk. After publishing, the command re-reads
the index and fails if its own attempt is not listed.

### The generated project files

```text
<output>/project.yaml               # full runnable project
<output>/generated.yaml             # same content, convenience copy
<output>/project.fragment.yaml      # scenarios + experiment only
<output>/exploration/project.yaml   # full project
<output>/exploration/generated.yaml # full project
<output>/exploration/project.fragment.yaml
```

With no base project, `explore` bootstraps a minimal project: one `live`
application per unique starting origin, with the start origin plus
`https://fonts.googleapis.com` and `https://fonts.gstatic.com` on each version's
`allowed_origins` (so the run is not safety-blocked on the first external
stylesheet), and a `default-explorer` persona:

```yaml
id: default-explorer
name: Default Explorer
working_memory_capacity: 4
initial_confidence: 0.55
initial_frustration: 0.10
abandonment_threshold: 0.75
attention_temperature: 1.0
```

With a base project, the generated `project.yaml` is the base project with the
curated scenarios and any custom personas appended. Existing personas and
applications are reused; a colliding application id gets an `-explore` suffix
rather than overwriting anything.

Each curated scenario is materialized with:

- the curated id, name, goal, and `visible-result` verifier (plus `all_of`)
- `application_version_ids` resolved to the live version whose start URL shares
  the scenario's origin
- `evaluation_target.labels_by_version` covering the resolved version and the
  `live` kind, so the file validates
- `budget` with the curated values, or the materialization defaults of 20 steps,
  18 observations, 8 interactions, 32 model calls, and a 90-second
  `stall_timeout_seconds`; `timeout_seconds` is `null`, so the generated
  scenario runs progress-based
- `start_state: dashboard`, empty `fixture_inputs` and `safeguards`
- `expected_evidence: [target-discovery, verified-completion]`
- `viewport: {width: 1280, height: 800}`
- `eligible_persona_ids` from your persona selection, or `["default-explorer"]`

The experiment is always:

```yaml
- id: exploration-run
  name: Exploration Run
  scenario_ids: [<curated scenario ids>]
  application_version_ids: [<resolved live version ids>]
  persona_ids: [<selected persona ids>]
  policies: [full-list]
  run_count: 1
```

`full-list` is the only policy an exploration-generated experiment uses. Change
it in the generated YAML if you want to compare attention policies; see
[configuration](configuration.md#experiments).

## Running the result

The command prints the exact next step:

```text
next step: uxa run <output>/project.yaml --experiment exploration-run
```

Or run it in one go with `--run`, which executes the generated experiment into
`<output>/exploration-run-output`:

```powershell
uv run uxa explore --starting-url https://example.com --run
```

## Previewing before spending anything

`--dry-run` prints the crawl matrix, a page-count upper bound, a rough synthesis
token estimate, a settings fingerprint, and the resolved output directory —
without launching a browser or a model. Both estimates are labeled as estimates
derived from configuration, not measurements:

```text
explore matrix:
 starts: 1
  - https://example.com
 depth: 2
 max_pages: 50
 max_scenarios: 8
 settle_ms: 10000
 estimated_pages: 9 (upper-bound estimate)
 synthesis token estimate: ~7700 tokens (rough estimate, not a measurement)
 config fingerprint: 7c1e0a4b93d2 (settings hash, not a corpus digest)
 output: .uxa-output\explore\example
```

`estimated_pages` is `min(max_pages, max(1, starts * (depth + 1) * 3))`. The
token estimate is `estimated_pages * 800 + 500`.

## Endpoint reference

The review server exposes these routes on `127.0.0.1`. They are useful for
scripting a non-browser review pass.

| Route | Method | Returns |
| --- | --- | --- |
| `/__explore/` (or `/__explore`) | GET | The review page. |
| `/__explore/explore.js`, `/__explore/explore.css` | GET | Static assets. `/__explore/static/...` also resolves. |
| `/__explore/api/suggestions` | GET | `suggestions`, `corpus_summary`, `personas: {existing, suggested}`, `auto_accept_flag`. |
| `/__explore/api/corpus` | GET | `corpus_summary` and `personas`. |
| `/__explore/api/curate` | POST | Submits curation. Returns `status`, `curated`, `curated_count`, `fragment_yaml`, `persona_selection`, `auto_accept_flag`, `corpus_summary`, then shuts the server down. |

A curation request body carries:

| Field | Type | Meaning |
| --- | --- | --- |
| `suggestions_signature` | string | Required. The signature of the suggestion set the page was rendered from. |
| `accepted_ids` | list of strings | Suggestion ids kept as-is. |
| `edited` | list of scenarios | Edited originals. |
| `added` | list of scenarios | Hand-written scenarios. |
| `persona_selection` | object or absent | `mode` is `existing`, `suggested`, or `custom` (default `existing`), plus `persona_ids` or a `custom_persona` object. |
| `auto_accept_flag` | boolean or null | Optional override of the server's auto-accept state. |

Every id must appear exactly once across `accepted_ids`, `edited`, and `added`.
A missing or stale `suggestions_signature`, an unknown suggestion id, or a
duplicate id gets HTTP 422 with the reasons joined.
