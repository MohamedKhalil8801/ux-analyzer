# Architecture

This is contributor reference material. For how to use the tool, start with
[getting started](getting-started.md) and [commands](commands.md).

## Current Boundary

The system is a Python 3.12 modular monolith. Chromium and the bundled FastAPI
fixture are current platform implementations. Domain and application policy do
not depend on Playwright, FastAPI, DOM selectors, CSS selectors, HTTPX, or an
OpenAI SDK.

Logical dependency direction points inward toward domain contracts:

```text
CLI / composition root
        |
        +--> configuration boundary --> immutable domain configuration
        |
        +--> application use cases --> domain + ports
        |                              ^
        |                              |
        +--> adapters / storage / reporting implement outer capabilities
        |
        +--> bundled fixture app (controlled target, separate process)
```

The composition root in `ux_analyzer.cli` wires concrete adapters. The
application layer coordinates ports. The domain layer owns invariants and
state transitions. Outer adapters translate external types at boundaries.

## Module Ownership

| Module | Owns | External dependency rule |
| --- | --- | --- |
| `src/ux_analyzer/domain` | Benchmark, snapshot, attention, run, evidence vocabulary and invariants | Standard library only; no browser, model, filesystem, or transport types. |
| `src/ux_analyzer/config` | YAML validation, reference checks, canonical config digest, domain conversion | Pydantic and PyYAML stay at configuration boundary. |
| `src/ux_analyzer/ports` | Observation, verification, model, and artifact contracts | Contracts use platform-neutral/domain types. |
| `src/ux_analyzer/application` | Run orchestration, action validation, state updates, matrix expansion, evaluation, report use case | `run_agent.py` consumes ports; it does not import Playwright, HTTPX, filesystem, or OpenAI adapter classes. |
| `src/ux_analyzer/providers` | Heuristic prominence, staged saliency prominence, seeded attention, memory, scent roles, cognitive role, finding rules | Provider implementations expose domain/application-facing behavior and model ports. Learned failures preserve explicit heuristic fallback. |
| `src/ux_analyzer/adapters/saliency` | Foveacast preprocessing, ONNX Runtime sessions, CPU/DirectML selection, model timing metadata | Runtime and model files stay outside domain/application policy. DirectML is optional and never silently replaces an explicit failure. |
| `src/ux_analyzer/saliency` | Versioned model manifest, content-addressed artifact install, checksum and runtime status | Model downloads occur only through explicit CLI commands. |
| `src/ux_analyzer/adapters/web` | Playwright sessions, fail-closed network policy, extraction, fixture verification | Browser and HTTP details stop at adapter boundary. |
| `src/ux_analyzer/adapters/openai.py` | OpenAI-compatible structured HTTP client | API key and HTTPX stay here; role records are sanitized before persistence/logging. |
| `src/ux_analyzer/storage` | Staging, append-only timeline, artifacts, checksums, atomic publication | Filesystem details stay outside application policy. |
| `src/ux_analyzer/reporting` | Sanitized static HTML replay | Private keys are removed before report rendering. |
| `fixture_app` | Controlled local SaaS target and private in-memory state API | No persistence, outbound communication, or production integration. |

`application.experiment` contains the finite policy selector for the four
current policies. This is known variation, not a plugin registry.

## Runtime Flow

One `RunAgent.execute` call performs:

1. Create a run bundle in `.staging`.
2. Start isolated browser session with fixed viewport and test account.
3. Reset fixture state with scenario fixture inputs.
4. Capture and normalize current viewport.
5. Resolve configured prominence provider. Heuristic scoring stays deterministic;
   explicit Foveacast opt-in predicts 1s, 3s, and 7s maps, aggregates element
   evidence, selects the current search-stage mixture, and uses experiment-scoped
   cache entries.
6. On learned model, cache, or aggregation failure, record sanitized fallback
   evidence and continue with heuristic prominence; invalid learned samples are
   excluded from comparison scorecards.
7. Optionally call coarse scent provider. Sample bounded progressive observation
   using run seed, or preserve the complete persona-safe list for unrestricted
   list policies.
8. Call cognitive role with persona-visible observation and memory.
9. Validate action against noticed/remembered/actionable/stale-snapshot rules.
10. Resolve typed input through scenario fixture values, then execute platform action.
11. Record result, apply deterministic state updates, and recapture after state change.
12. Verify independently after state-changing actions and at terminal decision.
13. Record terminal event, provider/model manifests, sanitized role call records,
    config digest, and artifact checksums.
14. Calculate per-run metrics/findings before publishing the immutable bundle.
15. Finalize bundle atomically or leave crash marker and return no success result.

The experiment runner expands stable `RunSpec` values and executes runs with
bounded concurrency. Default worker count is one. Each run gets isolated agent
state and fixture session; partial experiment results retain per-run failures.
After execution, CLI aggregates evaluable cells, runs exact paired-seed directional
gates, writes `experiment.json` with safe runner/evaluator failure records, and
renders `<output>/report.html` from finalized, staging, and available experiment
evidence. Report projections keep private execution fields out.

## Exploration Flow

`uxa explore` is a separate discovery command that feeds the existing run
pipeline. It sits before classic benchmark execution and never mutates the
`uxa run/ablate/report/synthesize` contracts:

```text
uxa explore [PROJECT] --starting-url URL ...
    -> load base project (personas/providers) or bootstrap live applications
    -> ExplorationSpec (flags win over YAML; depth 0-5, max_pages 1-200, scenarios 1-20)
    -> ExplorationCrawler (same-origin BFS, dedup via normalized URLs,
       networkidle + scroll-sweep page settlement)  ->  CrawlCorpus
    -> ExplorationSynthesizer (cognitive role, compressed per-page evidence)
       -> ScenarioSuggestion set (visible-result only)
    -> human curation: local review UI  |  --auto-accept accepts all suggestions
    -> ExplorationArtifactStore (immutable checksummed attempt + index)
    -> generated project.yaml / project.fragment.yaml (experiment exploration-run)
    -> ExperimentRunner pipeline unchanged  (uxa run --experiment exploration-run)
```

Boundary rules match the rest of the system. `domain/exploration.py` stays
stdlib-only and holds `ExplorationSpec`, `CrawlPage`, `CrawlCorpus`,
`ScenarioSuggestion`, and digest helpers. Playwright settlement lives in the
crawler application module over observation adapters; same-origin enforcement
reuses URL canonicalization plus the browser network allowlist. Synthesis sends
only redacted per-page evidence (titles, headings, visible labels/bounds) to the
model; selectors, hidden DOM facts, destination URLs, and fixture keys never
enter prompts or review-UI payloads.

`--dry-run` prints the crawl matrix estimate (starts, depth, page cap,
scenario cap, estimated pages, synthesis token estimate) without launching a
browser or model client. The local review UI is loopback-only FastAPI with no
external requests; `--auto-accept` produces the same curated set as manual
accept-all. Artifacts are immutable, canonical-JSON checksummed, atomically
published attempts indexed under `<output>/exploration/index.json`
(see [output formats](output-formats.md)). The generated project merges
base scenarios when a base project is provided (append strategy), so accepted
exploration scenarios run through the unchanged runner, evaluation, and report
path.

## Current implementation choices

- Web platform: Playwright Chromium only.
- Extraction: deterministic DOM/layout/rendered visibility facts.
- Prominence default: `heuristic-prominence-v1`, inspectable feature
  contributions. Foveacast is an explicit opt-in provider pending real focused
  gate review; it does not change the default automatically.
- Attention: `progressive-attention-v4`, region-first seeded softmax sampling,
  default batch size two, and bounded recovery observations after no progress.
- Unrestricted baselines: `full-list` and `prominence-ranked-list` expose every
  visible persona-safe element in one complete observation.
- Scent: optional structured coarse and full roles for
  `progressive-prominence-scent`.
- Cognitive action: structured model output limited to listed element IDs and
  finite actions.
- Verification: `fixture-state`, `visible-result`, or `colour-change` typed
  contract, independent of the cognitive claim.
- Replay: static HTML, inline CSS/JavaScript, embedded image data when present.

Foveacast saliency contracts, registry, adapter, aggregation, stage selection,
cache, fallback, and replay paths are implemented as model-dependent
extensions. The pinned CPU known-screenshot gate now produces stable finite
outputs for all three durations after symbolic input dimensions were handled.
No real-model focused promotion, human calibration, production API, or
issue-tracker adapter is part of current evidence. DirectML remains an optional
Windows/AMD path; this environment reported no DirectML provider. Since then
live focused heuristic-versus-Foveacast comparisons have completed with valid
cells and warm CPU inference latency has been measured, but no externally
configured latency/RSS budget and no DirectML hardware result exist. The
conditional promotion status is recorded in
[ADR 0001](adr/0001-provisional-saliency-provider.md).

## Saliency evidence boundary

`ViewportSnapshot` and `ElementSnapshot` remain deterministic interface facts.
Saliency maps, element aggregates, attention profiles, operational prominence,
provider identity, model checksums, execution provider, timings, cache state,
and fallback reasons are model-estimate or operational evidence. The stage
selector uses immediate, early, and eventual estimates without wall-clock
simulation; all three maps are eagerly cached for one screenshot. Cache scope is
one experiment output, never global.

The report may show ranked elements, aggregation components, duration tabs, and
heatmaps. It must keep numeric saliency outside cognitive prompts and obey
source-image redaction. Missing source overlays produce heatmap-only replay.
The persisted evidence contract is in [output formats](output-formats.md).

## Evidence-room report synthesis boundary

Report synthesis starts only after run bundles and the experiment result are
finalized. The application builds a redacted evidence room containing frozen
expectations, deterministic run facts, replay events, metrics, geometry, and
allowlisted screenshot or heatmap artifacts. It excludes run-agent chat,
selectors, hidden DOM facts, private reasoning, raw model responses, and prior
report prose.

The post-run flow is:

```text
finalized runs
    -> evidence-room manifest and frozen expectations
    -> report analyst retrieves candidate evidence
    -> evidence auditor checks refs and counterevidence
    -> pattern reviewer checks recurrence and cross-surface scope
    -> adjudicator resolves objections and judges publication/severity
    -> deterministic validation and final verification
    -> immutable synthesis attempt
    -> offline report renderer and playback workspace
```

The four report roles use fresh isolated contexts. They may request only
known evidence IDs, and every requested entry is resolved, bounded, and
revalidated before it can support a finding. Retrieval defaults to three rounds,
a 32-entry per-role request cap (16 entries per resolution batch), and 16 MiB of
cumulative attachments. Role output is advisory until the deterministic
publication contract accepts it.

Frozen expectations define desired outcomes, invariants, acceptable alternatives,
reference paths, effort bounds, and warning signals. A path deviation is not
automatically an issue: a valid alternate route remains acceptable when the
outcome and invariants hold. A finding needs evidence of user impact or task
harm, not merely distance from one reference path.

Severity is an evidence judgment over task importance, user impact, frequency,
recoverability, accessibility impact, scope, recurrence, leverage, and
counterevidence. Broad scope or recurrence can inform the judgment but cannot
multiply severity by itself. The UX principle pack supplies interpretive
questions and labels; principles are not evidence, cannot establish severity,
and cannot replace recorded observations or verification.

Each synthesis attempt is immutable and records its corpus, expectation and
principle digests, role manifests, retrieval log, objections, candidate and
rejected findings, final findings, limitations, status, and fallback metadata.
Regeneration creates a new attempt rather than rewriting an old one. Outcomes
are `accepted`, `no-issues`, `rejected`, or `unavailable`. Missing transport or
invalid model output preserves deterministic findings and remains visible as a
bounded fallback state.

Every visible finding links to independently verifiable evidence such as a
recorded event, viewport or element snapshot, metric, replay sequence,
screenshot, or heatmap artifact. The renderer resolves those links locally and
opens the corresponding playback workspace; `uxa report` never calls a model.

## Creative redesign boundary (model estimates)

The creative redesign pipeline is a second, deliberately separate post-run
flow. It never touches the run evidence corpus: it reads only the shared
crawled-corpus page capture (ADR 0007 — one browser pass per page shared by
the audit and the redesign, persisted as the versioned `page-capture.json`
sidecar) and the published redesign principle pack. Its output is a set of
design proposals that are model estimates, never verified claims:

```text
crawled-corpus page capture (shared sidecar)
    -> redesign proposer: per-page proposals + page understanding (inference)
    -> redesign critic/merger: kills/merges with reasons, consistency notes
    -> deterministic schema/guardrail validation
    -> immutable redesign attempt (<output>/redesign/<attempt-id>/)
    -> offline report Redesign tab (impact x effort stamped as model estimates)
```

The two roles run in fresh isolated contexts sharing only the capture digest
and principle pack. Deliberate-choice checks are required for grouping,
unification, simplification, and relocation proposals. The pipeline is gated
by `UXA_REDESIGN_ENABLED` (auto mode) or the `uxa redesign` command; failures
are best-effort and never fail the experiment. The report renders proposals on
a dedicated tab that labels impact and effort as model estimates (Flag-pair
ink) and separates them visually and textually from evidence-grounded findings.
See [redesign](redesign.md) for the user-facing contract.
