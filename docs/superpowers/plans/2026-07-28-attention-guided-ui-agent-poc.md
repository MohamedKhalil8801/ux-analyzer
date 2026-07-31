# Attention-Guided UI Agent Benchmarkable POC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task.

**Goal:** Build a reproducible Python proof of concept that compares unrestricted element access with progressive attention-guided UX evaluation across two controlled SaaS workflows and produces evidence-backed JSON and HTML reports.

**Architecture:** Use a Python 3.12 modular monolith. Platform-neutral domain and application modules define benchmark, interface snapshot, attention, run, and evaluation concepts; ports isolate browser automation, model calls, verification, and artifact persistence; a Playwright web adapter and bundled FastAPI fixture provide first end-to-end implementation. Configuration is versioned YAML validated into typed models, while each execution produces an immutable run bundle.

**Tech Stack:** Python 3.12, uv, Pydantic 2, PyYAML, Typer, Playwright Chromium, FastAPI, Uvicorn, HTTPX, Jinja2, pytest, pytest-asyncio, Ruff, Pyright.

## Global Constraints

- Initial platform is Chromium web automation, but domain and application code must not import Playwright, FastAPI, DOM, CSS selector, or browser-specific types.
- Only bundled fixture origins may be automated. Browser traffic must fail closed for every other origin; host-side model traffic uses the separately configured OpenAI-compatible endpoint.
- Model configuration comes from `UXA_LLM_BASE_URL`, `UXA_LLM_API_KEY`, `UXA_SCENT_MODEL`, and `UXA_COGNITIVE_MODEL`.
- Scent and cognitive roles use separate prompts, schemas, model IDs, manifests, retry records, and usage records even when endpoint and model match.
- Cognitive agent never receives selectors, test IDs, hidden labels, destination URLs, internal verifier state, numeric prominence scores, or numeric scent scores.
- Scenario fixtures supply all typed form values and redaction rules. Agent chooses actions but never invents credentials, email addresses, or 2FA secrets.
- Every stochastic choice uses recorded run seed. Memory decay is deterministic; attention sampling is stochastic.
- Every run terminates with one closed outcome: `verified-success`, `agent-abandoned`, `budget-exhausted`, `timed-out`, `provider-failure`, `model-failure`, `safety-blocked`, or `internal-error`.
- Official success always comes from independent verifier. Agent claim is recorded separately.
- Run bundles are append-only during execution and immutable after finalization.
- POC uses deterministic web extraction and heuristic prominence. Pretrained saliency and expectation generation remain declared extension boundaries, not implementations.
- Default benchmark compares `full-list` against `progressive-prominence-scent`. Ablations run `prominence-ranked-list` and `progressive-prominence` separately.
- Default experiment cell count is 10 seeded runs. Worker count is configurable and defaults to 1.
- Reports distinguish deterministic facts, model-dependent estimates, and unsupported human claims. Unsupported human claims must never become findings.

## Domain Model

```text
BenchmarkProject
  owns Applications, Scenarios, Personas, ExperimentDefinitions

Application
  owns immutable ApplicationVersions

Scenario
  defines goal, start state, fixture inputs, budgets, verifier spec,
  safeguards, eligible personas, and expected evidence

ExperimentDefinition
  selects scenarios x application versions x personas x policies x seeds

Experiment
  expands definition into RunSpecs and aggregates RunResults

Run
  owns one seed, one scenario, one app version, one persona, one policy,
  one provider manifest set, ordered RunEvents, and one terminal outcome

ViewportSnapshot
  owns immutable ElementSnapshots, RegionSnapshots, graph edges,
  screenshot artifact, and private execution references

AttentionState
  owns noticed IDs, inspected IDs, focus region, budgets, memory,
  confidence, frustration, failed candidates, and current subgoal

ProgressiveObservation
  exposes only persona-visible snapshots and remembered observations

Finding
  references evidence IDs and carries evidence class, category, severity,
  reproducibility, limitations, and optional generated explanation
```

Key invariants:

- `ApplicationVersion` changes interface presentation; `Scenario` goal and verifier stay fixed across defective and improved versions.
- Element IDs identify one immutable viewport snapshot. Cross-snapshot continuity uses optional lineage links, never selectors as identity.
- Private execution reference can only execute against viewport snapshot that created it.
- Every observation references elements already present in captured viewport and contains 1-3 newly revealed elements plus optional region context.
- Full post-notice scent can only be calculated after element becomes noticed. Coarse pre-notice scent uses glance-level visible cues only.
- An interaction may target only currently remembered, noticed, actionable element.
- Finalized run contains terminal event, verification result, model/provider manifests, configuration digest, and artifact checksums.

## File Map

```text
pyproject.toml                         toolchain and package metadata
src/ux_analyzer/cli.py                 Typer command surface
src/ux_analyzer/config/models.py       YAML-facing validated definitions
src/ux_analyzer/config/loader.py       load, resolve, and hash project config
src/ux_analyzer/domain/benchmark.py    project, app, scenario, persona, experiment
src/ux_analyzer/domain/interface.py    viewport, element, region, graph, visibility
src/ux_analyzer/domain/attention.py    prominence, scent, observation, memory, actions
src/ux_analyzer/domain/run.py          run spec, events, status, result, manifests
src/ux_analyzer/domain/findings.py     metrics, evidence classes, findings
src/ux_analyzer/ports/observation.py   platform observation/execution/session port
src/ux_analyzer/ports/models.py        scent and cognitive model ports
src/ux_analyzer/ports/verification.py  reset and verifier ports
src/ux_analyzer/ports/artifacts.py     immutable run bundle port
src/ux_analyzer/adapters/web/          Playwright provider and extraction pipeline
src/ux_analyzer/adapters/openai.py     OpenAI-compatible structured model adapter
src/ux_analyzer/providers/             heuristic prominence, scent, policies, memory
src/ux_analyzer/application/           run, experiment, evaluation, report use cases
src/ux_analyzer/storage/run_bundle.py  filesystem artifact implementation
src/ux_analyzer/reporting/             static HTML replay renderer and templates
fixture_app/                            bundled controlled SaaS application
benchmarks/demo/                        YAML project, scenarios, personas, experiments
tests/unit/                             pure domain/provider tests
tests/integration/                      adapter and bundle contract tests
tests/e2e/                              fixture browser and benchmark tests
```

## Task 1: Bootstrap Python Project and Quality Gates

**Files:**
- Create: `pyproject.toml`
- Create: `src/ux_analyzer/__init__.py`
- Create: `src/ux_analyzer/cli.py`
- Create: `tests/unit/test_cli.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces console command `uxa` backed by `ux_analyzer.cli:app`.
- Establishes Python 3.12, Ruff, Pyright, pytest, and asyncio conventions used by all later tasks.

- [ ] **Step 1: Add failing CLI smoke test**

```python
from typer.testing import CliRunner

from ux_analyzer.cli import app


def test_version_command() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "uxa 0.1.0"
```

- [ ] **Step 2: Run test and verify import failure**

Run: `rtk uv run pytest tests/unit/test_cli.py -q`

Expected: FAIL because package and command do not exist.

- [ ] **Step 3: Add package metadata and minimal Typer command**

Declare runtime dependencies `pydantic>=2.8`, `PyYAML>=6.0`, `typer>=0.12`, `playwright>=1.46`, `fastapi>=0.112`, `uvicorn>=0.30`, `httpx>=0.27`, and `jinja2>=3.1`. Declare dev dependencies `pytest`, `pytest-asyncio`, `ruff`, and `pyright`. Configure `src` package layout, strict Pyright for `src/ux_analyzer`, Ruff formatting/linting, and pytest asyncio mode.

- [ ] **Step 4: Run quality gates**

Run: `rtk uv sync`

Run: `rtk uv run pytest tests/unit/test_cli.py -q`

Run: `rtk uv run ruff check .`

Run: `rtk uv run pyright`

Expected: all pass.

- [ ] **Step 5: Commit foundation**

```bash
rtk git add pyproject.toml src/ux_analyzer tests/unit/test_cli.py .gitignore
rtk git commit -m "chore: bootstrap ux analyzer"
```

## Task 2: Define Benchmark Configuration and Domain Contracts

**Files:**
- Create: `src/ux_analyzer/config/models.py`
- Create: `src/ux_analyzer/config/loader.py`
- Create: `src/ux_analyzer/domain/benchmark.py`
- Create: `tests/unit/config/test_loader.py`
- Create: `tests/fixtures/config/minimal-project.yaml`

**Interfaces:**
- Produces `load_project(path: Path) -> LoadedProject`.
- Produces immutable `BenchmarkProject`, `Application`, `ApplicationVersion`, `Scenario`, `Persona`, `ExperimentDefinition`, `Budget`, `VerifierSpec`, and `FixtureInputs`.
- `LoadedProject` includes canonical SHA-256 configuration digest.

- [ ] **Step 1: Write validation tests**

Cover valid project loading, duplicate IDs, unknown references, missing defective/improved versions, unsupported verifier type, invalid persona parameter range, zero run count, and digest stability when YAML key order changes.

```python
def test_scenario_references_existing_application_versions(tmp_path: Path) -> None:
    path = write_project(tmp_path, scenario_versions=["missing"])
    with pytest.raises(ProjectConfigError, match="unknown application version"):
        load_project(path)
```

- [ ] **Step 2: Run tests and verify failures**

Run: `rtk uv run pytest tests/unit/config/test_loader.py -q`

- [ ] **Step 3: Implement typed configuration and cross-reference validation**

Use Pydantic only at YAML boundary. Convert validated models into frozen domain dataclasses and enums. Canonicalize loaded data through sorted JSON before hashing. Supported verifier specs for POC:

```python
VerifierSpec = FixtureStateVerifierSpec | VisibleResultVerifierSpec
```

`FixtureStateVerifierSpec` contains `resource`, `field`, `operator`, and typed `expected_fixture_key`. `VisibleResultVerifierSpec` contains persona-visible `text` and optional role.

- [ ] **Step 4: Verify config tests and type checks**

Run: `rtk uv run pytest tests/unit/config/test_loader.py -q`

Run: `rtk uv run pyright src/ux_analyzer/config src/ux_analyzer/domain/benchmark.py`

- [ ] **Step 5: Commit config model**

```bash
rtk git add src/ux_analyzer/config src/ux_analyzer/domain/benchmark.py tests/unit/config tests/fixtures/config
rtk git commit -m "feat: define benchmark configuration"
```

## Task 3: Model Interface, Attention, Run, and Finding State

**Files:**
- Create: `src/ux_analyzer/domain/interface.py`
- Create: `src/ux_analyzer/domain/attention.py`
- Create: `src/ux_analyzer/domain/run.py`
- Create: `src/ux_analyzer/domain/findings.py`
- Create: `tests/unit/domain/test_invariants.py`

**Interfaces:**
- Produces closed enums and frozen dataclasses for all runtime state.
- Produces `RunState.apply(event: RunEvent) -> RunState` as sole lifecycle transition API.
- Produces `PersonaVisibleElement.from_snapshot(...)` that excludes private fields by construction.

- [ ] **Step 1: Write invariant and transition tests**

Test invalid bounds, visibility fractions outside `[0, 1]`, duplicate snapshot element IDs, execution against stale viewport, interaction with unnoticed element, observation batch larger than configured maximum, post-notice scent before notice, event after terminal outcome, and missing verification on finalized run.

```python
def test_private_execution_reference_is_not_serialized_to_agent() -> None:
    visible = PersonaVisibleElement.from_snapshot(element_snapshot())
    assert "execution_reference" not in visible.model_dump()
    assert "provider_id" not in visible.model_dump()
```

- [ ] **Step 2: Run tests and verify failures**

Run: `rtk uv run pytest tests/unit/domain/test_invariants.py -q`

- [ ] **Step 3: Implement domain state and reducer**

Use discriminated unions for `AttentionAction`, `RunEvent`, and `RunOutcome`. Keep `PrivateExecutionReference` in interface snapshot private representation. Define evidence classes as `deterministic-fact`, `model-estimate`, and `unsupported-human-claim`; reject findings using unsupported class.

- [ ] **Step 4: Run domain tests**

Run: `rtk uv run pytest tests/unit/domain -q`

Run: `rtk uv run pyright src/ux_analyzer/domain`

- [ ] **Step 5: Commit domain model**

```bash
rtk git add src/ux_analyzer/domain tests/unit/domain
rtk git commit -m "feat: model benchmark run state"
```

## Task 4: Implement Immutable Run Bundles

**Files:**
- Create: `src/ux_analyzer/ports/artifacts.py`
- Create: `src/ux_analyzer/storage/run_bundle.py`
- Create: `tests/integration/storage/test_run_bundle.py`

**Interfaces:**
- Produces `RunBundleWriter.start`, `append_event`, `write_artifact`, `finalize`, and `abort`.
- Final bundle contains `manifest.json`, `timeline.jsonl`, `artifacts/`, `checksums.sha256`, and `result.json`.

- [ ] **Step 1: Write bundle contract tests**

Test atomic staging directory, monotonic event sequence, redaction before write, checksum generation, refusal to mutate finalized bundle, crash recovery marker, and manifest inclusion of seed, config digest, endpoint origin without API key, model IDs, prompt versions, package version, and provider versions.

- [ ] **Step 2: Run tests and verify failures**

Run: `rtk uv run pytest tests/integration/storage/test_run_bundle.py -q`

- [ ] **Step 3: Implement filesystem adapter**

Write into `<output>/.staging/<run-id>` and atomically rename to `<output>/runs/<run-id>` on finalization. Append JSONL with flush after every event. Redact configured exact values and keys before serialization. Store screenshot and trace files by content hash to avoid duplicate copies inside one experiment.

- [ ] **Step 4: Verify bundle contract**

Run: `rtk uv run pytest tests/integration/storage/test_run_bundle.py -q`

- [ ] **Step 5: Commit storage**

```bash
rtk git add src/ux_analyzer/ports/artifacts.py src/ux_analyzer/storage tests/integration/storage
rtk git commit -m "feat: persist immutable run bundles"
```

## Task 5: Build Controlled SaaS Fixture

**Files:**
- Create: `fixture_app/app.py`
- Create: `fixture_app/state.py`
- Create: `fixture_app/templates/layout.html`
- Create: `fixture_app/templates/dashboard.html`
- Create: `fixture_app/templates/settings.html`
- Create: `fixture_app/static/app.css`
- Create: `fixture_app/static/app.js`
- Create: `tests/e2e/fixture/test_fixture_contract.py`

**Interfaces:**
- Produces local fixture UI routes under `/app/{session_id}/{version}`.
- Produces private control API: `POST /__control/reset`, `GET /__control/state/{session_id}`, and `DELETE /__control/session/{session_id}`.
- Supports versions `defective` and `improved`, and scenarios `invite-teammate` and `enable-2fa`.

- [ ] **Step 1: Write fixture API and browser behavior tests**

Verify isolated reset state, fake invitation creation, fake 2FA enablement, no outbound communication, defective and improved variants sharing same state contract, and scenario completion visible in both UI and private state.

- [ ] **Step 2: Run tests and verify failures**

Run: `rtk uv run pytest tests/e2e/fixture/test_fixture_contract.py -q`

- [ ] **Step 3: Implement fixture behavior and planted defects**

Invite defective version: visually strong `Share` opens public-link dialog; teammate management is under low-prominence `Members` item in account menu. Improved version exposes clear `Invite teammate` action in team region and retains same verifier state.

2FA defective version: security entry has ambiguous shield icon and weak label inside deeper settings hierarchy; enable action gives weak confirmation. Improved version exposes labeled `Security` navigation, direct `Two-factor authentication` section, and explicit confirmation. Use fake setup code from scenario fixture.

- [ ] **Step 4: Verify fixture tests**

Run: `rtk uv run pytest tests/e2e/fixture/test_fixture_contract.py -q`

- [ ] **Step 5: Commit fixture**

```bash
rtk git add fixture_app tests/e2e/fixture
rtk git commit -m "feat: add controlled ux benchmark fixture"
```

## Task 6: Define Platform Ports and Playwright Session Safety

**Files:**
- Create: `src/ux_analyzer/ports/observation.py`
- Create: `src/ux_analyzer/ports/verification.py`
- Create: `src/ux_analyzer/adapters/web/session.py`
- Create: `src/ux_analyzer/adapters/web/network_policy.py`
- Create: `tests/integration/web/test_session_safety.py`

**Interfaces:**
- Produces `ObservationProvider` protocol with `start_session`, `capture`, `execute`, `reset`, and `end_session`.
- Produces platform-neutral `PlatformAction` union.
- Produces `BrowserAllowedOrigins` that admits bundled fixture origins only.

- [ ] **Step 1: Write safety tests**

Verify blocked navigation, blocked popup, blocked form submission to foreign origin, blocked fetch/XHR/image to foreign origin, permitted fixture assets, permitted model adapter traffic outside browser, test-account-only session creation, and cleanup after provider failure.

- [ ] **Step 2: Run tests and verify failures**

Run: `rtk uv run pytest tests/integration/web/test_session_safety.py -q`

- [ ] **Step 3: Implement Playwright session adapter**

Route every browser request through allowlist handler. Treat redirects to foreign origins as `SafetyBlocked`. Use isolated browser context per run, fixed viewport from scenario config, disabled downloads, denied geolocation/clipboard/notifications, and retained Playwright trace.

- [ ] **Step 4: Verify safety tests**

Run: `rtk uv run pytest tests/integration/web/test_session_safety.py -q`

- [ ] **Step 5: Commit platform boundary**

```bash
rtk git add src/ux_analyzer/ports src/ux_analyzer/adapters/web/session.py src/ux_analyzer/adapters/web/network_policy.py tests/integration/web/test_session_safety.py
rtk git commit -m "feat: add safe web observation sessions"
```

## Task 7: Extract Rendered Elements, Visibility, and Regions

**Files:**
- Create: `src/ux_analyzer/adapters/web/extractor.py`
- Create: `src/ux_analyzer/adapters/web/visibility.py`
- Create: `src/ux_analyzer/adapters/web/grouping.py`
- Create: `tests/integration/web/test_extraction.py`
- Create: `tests/fixtures/pages/extraction-cases.html`

**Interfaces:**
- Produces `capture(page, viewport_id) -> ViewportSnapshot`.
- Produces persona-visible text separately from private selector and node references.
- Produces basic region graph using landmarks, visual containment, headings, forms, lists, dialogs, and proximity.

- [ ] **Step 1: Write extraction cases**

Cover buttons, links, inputs, labels, tabs, menus, icon-only controls, hidden tabs, clipped content, zero-size nodes, opacity, disabled state, sticky overlay, modal blocking, partial viewport intersection, nested scroll container, repeated cards, and below-fold exclusion.

- [ ] **Step 2: Run tests and verify failures**

Run: `rtk uv run pytest tests/integration/web/test_extraction.py -q`

- [ ] **Step 3: Implement one browser evaluation payload plus Python normalization**

Collect rendered DOM facts in one `page.evaluate` call, then normalize and validate in Python. Use `elementsFromPoint` sampling plus center/corner probes for practical occlusion fraction. Compute screenshot-derived local contrast from cropped pixels only after DOM geometry is known. Never expose attributes `id`, `data-testid`, `href`, handler names, or raw selectors to persona-visible output.

- [ ] **Step 4: Verify extraction tests and leakage assertions**

Run: `rtk uv run pytest tests/integration/web/test_extraction.py -q`

- [ ] **Step 5: Commit perception extraction**

```bash
rtk git add src/ux_analyzer/adapters/web tests/integration/web/test_extraction.py tests/fixtures/pages
rtk git commit -m "feat: extract visible interface snapshots"
```

## Task 8: Implement Heuristic Prominence and Two-Level Attention Policy

**Files:**
- Create: `src/ux_analyzer/providers/prominence.py`
- Create: `src/ux_analyzer/providers/attention_policy.py`
- Create: `tests/unit/providers/test_prominence.py`
- Create: `tests/unit/providers/test_attention_policy.py`

**Interfaces:**
- Produces `HeuristicProminenceProvider.score(snapshot, config) -> tuple[ProminenceResult, ...]`.
- Produces `ProgressiveAttentionPolicy.next_observation(state, snapshot, scores, coarse_scent, rng) -> ObservationSelection`.
- Produces feature contribution map for every score.

- [ ] **Step 1: Write scoring and seeded-sampling tests**

Verify area, center distance, reading order, typography, contrast, isolation, actionability, motion, competition, occlusion penalties, probability normalization, region-first selection, sparse-page element fallback, 1-3 element batch limit, novelty reduction, failed-candidate penalty, fixed-seed repeatability, and differing paths across seeds.

- [ ] **Step 2: Run tests and verify failures**

Run: `rtk uv run pytest tests/unit/providers/test_prominence.py tests/unit/providers/test_attention_policy.py -q`

- [ ] **Step 3: Implement interpretable baseline**

Keep weights in versioned YAML config. Normalize each feature before weighted combination, preserve raw and normalized values, then use seeded softmax sampling with configurable temperature. Region score aggregates visible child probability mass and region priors without double-counting child selection.

- [ ] **Step 4: Verify provider tests**

Run: `rtk uv run pytest tests/unit/providers -q`

- [ ] **Step 5: Commit attention baseline**

```bash
rtk git add src/ux_analyzer/providers tests/unit/providers
rtk git commit -m "feat: add heuristic attention policy"
```

## Task 9: Add OpenAI-Compatible Structured Model Roles

**Files:**
- Create: `src/ux_analyzer/ports/models.py`
- Create: `src/ux_analyzer/adapters/openai.py`
- Create: `src/ux_analyzer/providers/scent.py`
- Create: `src/ux_analyzer/providers/cognitive.py`
- Create: `src/ux_analyzer/prompts/scent-coarse-v1.txt`
- Create: `src/ux_analyzer/prompts/scent-full-v1.txt`
- Create: `src/ux_analyzer/prompts/cognitive-v1.txt`
- Create: `tests/unit/providers/test_model_roles.py`
- Create: `tests/integration/models/test_openai_compatible.py`

**Interfaces:**
- Produces `CoarseScentEvaluator`, `FullScentEvaluator`, and `CognitiveAgent` protocols.
- Produces `OpenAICompatibleStructuredClient.complete(schema, messages, model, role) -> T`.
- Records endpoint origin, model, prompt digest, schema version, attempts, latency, token usage, and sanitized request/response.

- [ ] **Step 1: Write role separation and validation tests**

Verify env loading, different model per role, no API key in logs, coarse prompt receiving glance cues only, full prompt rejecting unnoticed elements, cognitive prompt excluding numeric scores/private fields, JSON schema validation, bounded retry count, retry event recording, and terminal model failure after exhaustion.

- [ ] **Step 2: Run tests and verify failures**

Run: `rtk uv run pytest tests/unit/providers/test_model_roles.py tests/integration/models/test_openai_compatible.py -q`

- [ ] **Step 3: Implement HTTP adapter and role providers**

Use OpenAI-compatible `/chat/completions` with JSON schema response format when supported. On providers lacking strict schema mode, parse one JSON object and validate locally. Retry only transport errors, rate limits, server errors, and invalid structured outputs according to configured bounded policy; never retry safety rejection or authentication failure.

- [ ] **Step 4: Verify fake-server integration tests**

Run: `rtk uv run pytest tests/integration/models/test_openai_compatible.py -q`

- [ ] **Step 5: Commit model adapters**

```bash
rtk git add src/ux_analyzer/ports/models.py src/ux_analyzer/adapters/openai.py src/ux_analyzer/providers src/ux_analyzer/prompts tests/unit/providers/test_model_roles.py tests/integration/models
rtk git commit -m "feat: add structured llm roles"
```

## Task 10: Implement Memory, Budgets, and Action Validation

**Files:**
- Create: `src/ux_analyzer/providers/memory.py`
- Create: `src/ux_analyzer/application/state_updates.py`
- Create: `src/ux_analyzer/application/action_validation.py`
- Create: `tests/unit/application/test_state_updates.py`
- Create: `tests/unit/application/test_action_validation.py`

**Interfaces:**
- Produces deterministic `MemoryPolicy.update` and `MemoryPolicy.recall`.
- Produces `apply_observation`, `apply_interaction_result`, and `apply_failure` state transitions.
- Produces `validate_action(decision, state, snapshot) -> ValidatedAction`.

- [ ] **Step 1: Write behavior tests**

Test working-memory capacity, episodic retention, deterministic decay, important failure retention, confidence/frustration clamping, inspection/scroll/wrong-action budget consumption, remembered target requirement, actionable requirement, stale snapshot rejection, scenario fixture lookup for type actions, and abandonment when threshold is crossed.

- [ ] **Step 2: Run tests and verify failures**

Run: `rtk uv run pytest tests/unit/application/test_state_updates.py tests/unit/application/test_action_validation.py -q`

- [ ] **Step 3: Implement explicit state updates**

Keep all persona parameter effects in versioned formulas. Do not ask model to calculate budgets, confidence, frustration, or memory decay. Model proposes qualitative state update; application clamps and applies configured deterministic rule.

- [ ] **Step 4: Verify application state tests**

Run: `rtk uv run pytest tests/unit/application -q`

- [ ] **Step 5: Commit behavior state**

```bash
rtk git add src/ux_analyzer/providers/memory.py src/ux_analyzer/application tests/unit/application
rtk git commit -m "feat: enforce simulated user state"
```

## Task 11: Orchestrate One Complete Run and Independent Verification

**Files:**
- Create: `src/ux_analyzer/application/run_agent.py`
- Create: `src/ux_analyzer/adapters/web/verifier.py`
- Create: `tests/integration/application/test_run_agent.py`

**Interfaces:**
- Produces `RunAgent.execute(spec: RunSpec) -> RunResult`.
- Uses ports only; no direct Playwright, HTTPX, filesystem, or OpenAI imports in application module.
- Executes capture, scoring, progressive observation, decision, validation, interaction, recapture, verification, and finalization loop.

- [ ] **Step 1: Write orchestration tests with fakes**

Cover verified success, claimed-but-unverified success, agent abandonment, budget exhaustion, timeout, provider failure, model failure, safety block, internal error, scroll producing new viewport, back navigation, wait action, wrong interaction, stale action rejection, and cleanup/finalization for every terminal path.

- [ ] **Step 2: Run tests and verify failures**

Run: `rtk uv run pytest tests/integration/application/test_run_agent.py -q`

- [ ] **Step 3: Implement run use case and typed verifier adapter**

Evaluate official verifier after state-changing actions and once at terminal decision. Fixture-state verifier queries private fixture control API; visible-result verifier captures fresh viewport and checks persona-visible role/text. Record agent claim independently. Write terminal event before bundle finalization.

- [ ] **Step 4: Verify orchestration tests**

Run: `rtk uv run pytest tests/integration/application/test_run_agent.py -q`

- [ ] **Step 5: Commit run orchestration**

```bash
rtk git add src/ux_analyzer/application/run_agent.py src/ux_analyzer/adapters/web/verifier.py tests/integration/application
rtk git commit -m "feat: execute attention-guided runs"
```

## Task 12: Expand Experiments and Policy Ablations

**Files:**
- Create: `src/ux_analyzer/application/experiment.py`
- Create: `src/ux_analyzer/providers/full_list_policy.py`
- Create: `src/ux_analyzer/providers/ranked_list_policy.py`
- Create: `tests/unit/application/test_experiment.py`
- Create: `tests/integration/application/test_experiment_runner.py`

**Interfaces:**
- Produces `expand_experiment(definition) -> tuple[RunSpec, ...]` in stable order.
- Produces `ExperimentRunner.run(specs, workers=1) -> ExperimentResult`.
- Supports policies `full-list`, `prominence-ranked-list`, `progressive-prominence`, and `progressive-prominence-scent`.

- [ ] **Step 1: Write matrix and execution tests**

Verify core-pair default, ablation selection, 10 default seeds per cell, explicit seed override, no duplicate run IDs, stable run IDs from semantic inputs, serial default, bounded concurrency, per-run isolation, partial experiment result after failures, and cancellation cleanup.

- [ ] **Step 2: Run tests and verify failures**

Run: `rtk uv run pytest tests/unit/application/test_experiment.py tests/integration/application/test_experiment_runner.py -q`

- [ ] **Step 3: Implement policies and runner**

`full-list` reveals all visible persona-safe elements in one observation. `prominence-ranked-list` reveals same full list sorted by heuristic score without numeric scores. Progressive variants use normal batch limits; prominence-only bypasses scent contribution. Preserve identical scenario, persona, seed, fixture state, and model configuration across compared policies.

- [ ] **Step 4: Verify experiment tests**

Run: `rtk uv run pytest tests/unit/application/test_experiment.py tests/integration/application/test_experiment_runner.py -q`

- [ ] **Step 5: Commit experiments**

```bash
rtk git add src/ux_analyzer/application/experiment.py src/ux_analyzer/providers tests/unit/application/test_experiment.py tests/integration/application/test_experiment_runner.py
rtk git commit -m "feat: run benchmark policy experiments"
```

## Task 13: Calculate Metrics, Findings, and Comparison Gates

**Files:**
- Create: `src/ux_analyzer/application/evaluation.py`
- Create: `src/ux_analyzer/providers/finding_rules.py`
- Create: `tests/unit/application/test_evaluation.py`
- Create: `tests/unit/providers/test_finding_rules.py`

**Interfaces:**
- Produces per-run metrics, per-cell aggregates, typed findings, and `VariantComparison`.
- Produces directional gate result based on fixed paired seeds.

- [ ] **Step 1: Write metric and evidence-rule tests**

Cover target discovery rank, inspected elements/regions, scrolls, wrong actions, backtracks, completion, abandonment, target scent, strongest competing scent, discovery cost, median and interval summaries, reproducibility, evidence references, and unsupported-claim rejection.

Finding rules must include weak target prominence, weak scent, strong misleading alternative, unexpected hierarchy, ambiguous icon/label, excessive depth, target below fold, missing feedback, wrong-action burden, and poor recovery.

- [ ] **Step 2: Run tests and verify failures**

Run: `rtk uv run pytest tests/unit/application/test_evaluation.py tests/unit/providers/test_finding_rules.py -q`

- [ ] **Step 3: Implement formulas and gate**

Version discovery-cost formula in config and preserve component values:

```text
inspection_cost
+ region_cost
+ scroll_cost
+ wrong_action_cost
+ backtrack_cost
+ uncertainty_cost
+ abandonment_penalty
```

Improved version passes when paired-seed median discovery cost decreases, median wrong-action/backtrack burden does not increase, and verified completion rate does not regress. Report each component; never collapse evidence into one unexplained score.

- [ ] **Step 4: Verify evaluation tests**

Run: `rtk uv run pytest tests/unit/application/test_evaluation.py tests/unit/providers/test_finding_rules.py -q`

- [ ] **Step 5: Commit evaluation**

```bash
rtk git add src/ux_analyzer/application/evaluation.py src/ux_analyzer/providers/finding_rules.py tests/unit/application/test_evaluation.py tests/unit/providers/test_finding_rules.py
rtk git commit -m "feat: evaluate ux benchmark evidence"
```

## Task 14: Generate Self-Contained HTML Replay

**Files:**
- Create: `src/ux_analyzer/application/report.py`
- Create: `src/ux_analyzer/reporting/renderer.py`
- Create: `src/ux_analyzer/reporting/templates/experiment.html.j2`
- Create: `src/ux_analyzer/reporting/static/report.css`
- Create: `src/ux_analyzer/reporting/static/report.js`
- Create: `tests/integration/reporting/test_renderer.py`

**Interfaces:**
- Produces `render_experiment_report(bundle_root, output_path) -> Path`.
- Report opens from filesystem without server and embeds only sanitized evidence.

- [ ] **Step 1: Write report tests**

Verify inline or relative bundled assets, no external requests, overview comparison table, evidence-class separation, run filters, timeline navigation, screenshot overlay bounds, noticed/inspected states, feature contributions, scent records, decisions, actions, verification, memory, model manifests, limitations, and escaped untrusted UI text.

- [ ] **Step 2: Run tests and verify failures**

Run: `rtk uv run pytest tests/integration/reporting/test_renderer.py -q`

- [ ] **Step 3: Implement static renderer**

Generate experiment index plus one self-contained run page when single-file size would exceed configured threshold. Use HTML/CSS/vanilla JavaScript only. Overlay coordinates scale from recorded viewport dimensions. Unsupported human claims appear only in limitations block, never scorecards.

- [ ] **Step 4: Verify renderer and browser smoke test**

Run: `rtk uv run pytest tests/integration/reporting/test_renderer.py -q`

- [ ] **Step 5: Commit replay**

```bash
rtk git add src/ux_analyzer/application/report.py src/ux_analyzer/reporting tests/integration/reporting
rtk git commit -m "feat: render ux experiment replay"
```

## Task 15: Add Demo Benchmark Definitions and CLI Workflows

**Files:**
- Create: `benchmarks/demo/project.yaml`
- Create: `benchmarks/demo/scenarios/invite-teammate.yaml`
- Create: `benchmarks/demo/scenarios/enable-2fa.yaml`
- Create: `benchmarks/demo/personas/first-time-nontechnical.yaml`
- Create: `benchmarks/demo/personas/impatient.yaml`
- Create: `benchmarks/demo/experiments/core-pair.yaml`
- Create: `benchmarks/demo/experiments/ablations.yaml`
- Modify: `src/ux_analyzer/cli.py`
- Create: `tests/integration/cli/test_commands.py`

**Interfaces:**
- Produces commands `uxa validate`, `uxa fixture serve`, `uxa run`, `uxa ablate`, `uxa report`, and `uxa inspect-run`.
- `uxa run benchmarks/demo/project.yaml --experiment core-pair --output .uxa-output` executes default benchmark.

- [ ] **Step 1: Write CLI tests**

Verify validation errors are actionable, env variables are checked without printing secrets, dry-run prints expanded matrix and estimated model calls, core command selects two policies, ablate command selects optional policies, worker default is one, run count override works, report regenerates from bundles, and inspect-run prints terminal outcome plus artifact paths.

- [ ] **Step 2: Run tests and verify failures**

Run: `rtk uv run pytest tests/integration/cli/test_commands.py -q`

- [ ] **Step 3: Implement YAML definitions and CLI orchestration**

Use fixture inputs `invite_email` and `totp_code`, both marked sensitive for exact-value redaction. Personas use explicit numeric capacities and thresholds. Keep expectation provider field absent; loader defaults it to disabled while retaining optional schema slot.

- [ ] **Step 4: Verify CLI workflows**

Run: `rtk uv run pytest tests/integration/cli/test_commands.py -q`

Run: `rtk uv run uxa validate benchmarks/demo/project.yaml`

- [ ] **Step 5: Commit benchmark surface**

```bash
rtk git add benchmarks/demo src/ux_analyzer/cli.py tests/integration/cli
rtk git commit -m "feat: add demo benchmark commands"
```

## Task 16: Add End-to-End Acceptance and Live Compatibility Suites

**Files:**
- Create: `tests/e2e/test_demo_benchmark.py`
- Create: `tests/e2e/test_private_data_leakage.py`
- Create: `tests/live/test_openai_endpoint.py`
- Create: `tests/recordings/`
- Create: `docs/testing.md`

**Interfaces:**
- Produces deterministic CI acceptance using fake/recorded model responses.
- Produces opt-in live suite gated by `UXA_RUN_LIVE_TESTS=1` and required model env variables.

- [ ] **Step 1: Write end-to-end acceptance tests**

Run bundled fixture, execute both scenarios across defective/improved versions and both personas with reduced CI seeds, then assert all concept acceptance criteria relevant to benchmarkable POC: visible extraction, inspectable prominence, no below-fold leakage, progressive observations, supported actions, private-data isolation, independent verification, full timeline recording, stochastic seeded variation, lower improved discovery cost, causal findings, simulated labels, provider manifests, and evidence classes.

- [ ] **Step 2: Add explicit leakage scan**

Scan every model request and persona-visible observation for fixture selectors, `data-testid`, private control API paths, raw destination URLs, fixture-state keys, configured sensitive values, and API key. Fail with event ID and redacted field path.

- [ ] **Step 3: Add live endpoint compatibility test**

Make one coarse scent, one full scent, and one cognitive structured call. Assert schema validity and manifest capture. Skip unless opt-in flag is set; never record live response fixtures automatically.

- [ ] **Step 4: Run complete verification**

Run: `rtk uv run playwright install chromium`

Run: `rtk uv run pytest -m "not live" -q`

Run: `rtk uv run ruff format --check .`

Run: `rtk uv run ruff check .`

Run: `rtk uv run pyright`

Expected: all pass. Live suite remains skipped unless explicitly enabled.

- [ ] **Step 5: Commit acceptance suite**

```bash
rtk git add tests/e2e tests/live tests/recordings docs/testing.md
rtk git commit -m "test: verify benchmarkable poc"
```

## Task 17: Document Operation, Limits, and Extension Contracts

**Files:**
- Create: `README.md`
- Create: `docs/architecture.md`
- Create: `docs/domain-model.md`
- Create: `docs/run-bundle-format.md`
- Create: `docs/model-provider.md`
- Create: `docs/security.md`
- Create: `docs/roadmap.md`

**Interfaces:**
- Documents exact commands, configuration fields, domain vocabulary, trust boundaries, artifact format, evidence classes, and deferred provider contracts.

- [ ] **Step 1: Document quick start and environment contract**

Include `uv sync`, Playwright install, fixture launch, project validation, dry-run matrix, benchmark execution, report opening, and live endpoint test. State env names exactly and show placeholder values only.

- [ ] **Step 2: Document architecture and domain model**

Show inward dependency direction, module ownership, run state machine, entity relationships, element snapshot identity, private/persona-visible split, scenario versus application-version semantics, experiment matrix, and verifier independence.

- [ ] **Step 3: Document security and scientific limits**

State fixture-only automation, blocked external browser traffic, fake communications, credential handling, redaction, artifact retention, model data exposure, non-human-calibrated metrics, and prohibited claims.

- [ ] **Step 4: Document deferred roadmap boundaries**

Describe pretrained saliency provider, frozen expectation provider, desktop/mobile adapters, human calibration, statistical calibration, production API, and issue-tracker integration. Do not include implementation placeholders in current POC tasks.

- [ ] **Step 5: Verify documentation and final diff**

Run: `rtk uv run uxa validate benchmarks/demo/project.yaml`

Run: `rtk uv run pytest -m "not live" -q`

Run: `rtk uv run ruff check .`

Run: `rtk uv run pyright`

Run: `rtk git diff --check`

- [ ] **Step 6: Commit documentation**

```bash
rtk git add README.md docs
rtk git commit -m "docs: describe attention benchmark poc"
```

## Completion Gate

POC is complete only when all conditions hold:

- Both scenarios run against defective and improved fixture versions for both personas.
- Core-pair experiment expands to paired seeds for full-list and progressive prominence+scent policies.
- Ablation command supports prominence-ranked list and progressive prominence-only policies.
- Every browser run blocks non-allowlisted origins and uses isolated fixture state.
- Every model request passes automated leakage scan.
- Every run bundle finalizes with checksums, manifests, timeline, verification result, and terminal outcome.
- HTML replay opens without server and explains every observation, score contribution, action, result, and finding.
- Improved variants pass directional multi-metric gate under recorded/fake CI responses.
- Live endpoint suite validates configured OpenAI-compatible endpoint when explicitly enabled.
- `pytest -m "not live"`, Ruff, Pyright, config validation, and `git diff --check` pass.
- Documentation explicitly labels outputs as simulated and blocks real-user completion or satisfaction claims.

## Execution Order and Review Checkpoints

1. Tasks 1-4 establish typed core and persistence. Review domain vocabulary and invariants before adapter work.
2. Tasks 5-7 establish controlled environment and deterministic perception. Review safety and leakage tests before model integration.
3. Tasks 8-11 establish attention, cognition, memory, and one complete run. Review event timeline and terminal semantics before experiment scaling.
4. Tasks 12-14 establish comparisons, findings, and replay. Review metric formulas and evidence references before CLI stabilization.
5. Tasks 15-17 establish operator workflow, acceptance suite, and documentation. Run complete verification before declaring POC complete.
