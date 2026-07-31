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
| `src/ux_analyzer/providers` | Heuristic prominence, seeded attention, memory, scent roles, cognitive role, finding rules | Provider implementations expose domain/application-facing behavior and model ports. |
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
5. Score visible elements with heuristic prominence.
6. Optionally call coarse scent provider.
7. Sample bounded progressive observation using run seed, or preserve the complete
   persona-safe list for unrestricted list policies.
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
- Prominence: `heuristic-prominence-v1`, inspectable feature contributions.
- Attention: `progressive-attention-v1`, region-first seeded softmax sampling,
  one to three newly revealed elements.
- Unrestricted baselines: `full-list` and `prominence-ranked-list` expose every
  visible persona-safe element in one complete observation.
- Scent: optional structured coarse and full roles for
  `progressive-prominence-scent`.
- Cognitive action: structured model output limited to listed element IDs and
  finite actions.
- Verification: fixture-state or visible-result typed contract, independent of
  cognitive claim.
- Replay: static HTML, inline CSS/JavaScript, embedded image data when present.

No pretrained saliency, expectation generator, desktop provider, mobile
provider, human calibration, production API, or issue-tracker adapter is part of
current POC. Deferred contracts are recorded in [roadmap](roadmap.md).
