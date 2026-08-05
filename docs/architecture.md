# Architecture

## Current Boundary

POC is Python 3.12 modular monolith. Chromium and the bundled FastAPI fixture
are current platform implementations. Domain and application policy do not
depend on Playwright, FastAPI, DOM selectors, CSS selectors, HTTPX, or an
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
current policies. This is known POC variation, not a plugin registry.

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

## Current POC Choices

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
- Verification: fixture-state or visible-result typed contract, independent of
  cognitive claim.
- Replay: static HTML, inline CSS/JavaScript, embedded image data when present.

Foveacast saliency contracts, registry, adapter, aggregation, stage selection,
cache, fallback, and replay paths are implemented as model-dependent
extensions. The pinned CPU known-screenshot gate now produces stable finite
outputs for all three durations after symbolic input dimensions were handled.
No real-model focused promotion, human calibration, production API, or
issue-tracker adapter is part of current evidence. DirectML remains an optional
Windows/AMD path; this environment reported no DirectML provider. No real
focused comparison, latency/RSS budget, or paired completion result exists.
Deferred contracts are recorded in [roadmap](roadmap.md).

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
Intentional changes from the original implementation plan are recorded in
[POC plan versus current implementation](poc-plan-vs-current.md).
