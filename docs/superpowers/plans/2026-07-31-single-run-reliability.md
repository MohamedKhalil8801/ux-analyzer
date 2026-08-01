# Single-Run Benchmark Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute exactly one selected `RunSpec`, terminate every model-driven run with bounded and classified behavior, and publish a filesystem-openable report for both success and failure.

**Architecture:** Keep the existing experiment runner and real adapter composition. Add a semantic single-run resolver that produces a one-spec `_ResolvedMatrix`, enrich the existing per-run context with bounded counters and safe cognitive state, and classify role validation failures at the `RunAgent` boundary. Extend bundle manifests and report projections with selected semantic inputs and new terminal diagnostic events.

**Tech Stack:** Python 3.12, Typer, Pydantic 2, Playwright, HTTPX, Jinja2, pytest, pytest-asyncio.

## Global Constraints

- The normal `uxa run` matrix remains available and is not changed to silently select one cell.
- `uxa run-one` must print `run specs: 1` and use the real Playwright, OpenAI-compatible, and verifier adapters.
- Browser/model transport timeouts remain finite; the benchmark wall timeout remains optional.
- Every terminal path writes a terminal event, verification result, result JSON, checksums, and a finalized bundle.
- Cognitive inputs must exclude selectors, test IDs, private fixture values, verifier state, and numeric prominence/scent scores.

### Task 1: Add Semantic Single-Run Resolution

**Files:**
- Modify: `src/ux_analyzer/cli.py`
- Modify: `src/ux_analyzer/application/experiment.py`
- Test: `tests/unit/application/test_experiment.py`
- Test: `tests/integration/cli/test_commands.py`

- [ ] Write failing tests for a resolver selecting `invite-teammate`, `improved`, `first-time-nontechnical`, `progressive-prominence-scent`, seed `0`, and for the CLI dry-run output containing exactly `run specs: 1`.
- [ ] Run the focused tests and observe failure because `run-one` and semantic filtering do not exist.
- [ ] Implement `_resolve_single_run` by resolving IDs against the loaded project, constructing a one-cell `ExperimentDefinition`, calling `expand_experiment`, and rejecting missing/ineligible IDs.
- [ ] Add `uxa run-one PROJECT --scenario --version --persona --policy --seed --fixture-origin --output` and route it through `_execute_matrix` and `_complete_experiment`.
- [ ] Run the focused tests and the existing CLI tests.

### Task 2: Add Run Budgets and Repetition Guards

**Files:**
- Modify: `src/ux_analyzer/config/models.py`
- Modify: `src/ux_analyzer/config/loader.py`
- Modify: `src/ux_analyzer/domain/benchmark.py`
- Modify: `src/ux_analyzer/application/run_agent.py`
- Test: `tests/integration/application/test_run_agent.py`
- Test: `tests/unit/config/test_loader.py`

- [ ] Write failing tests for a repeated successful `type-fixture` action, max model-call exhaustion, and a no-progress sequence producing a terminal event plus finalized artifacts.
- [ ] Run those tests and observe failure because the current loop only checks step and observation budgets.
- [ ] Add a default positive `max_model_calls` budget field, per-run model-call counter, repeated-action fingerprint counter, completed fixture-key set, and bounded no-progress counter.
- [ ] Emit `repeated-action-detected`, `repeated-fixture-input`, or `no-progress-detected` events and return `budget-exhausted` or `agent-abandoned` with an actionable terminal reason.
- [ ] Run focused run-agent tests and storage/report integrity tests.

### Task 3: Normalize Model Failures at the Run Boundary

**Files:**
- Modify: `src/ux_analyzer/providers/scent.py`
- Modify: `src/ux_analyzer/providers/cognitive.py`
- Modify: `src/ux_analyzer/application/run_agent.py`
- Modify: `src/ux_analyzer/adapters/openai.py`
- Test: `tests/unit/providers/test_model_roles.py`
- Test: `tests/integration/application/test_run_agent.py`

- [ ] Write failing tests for an unknown coarse-scent ID and a cognitive `inspect` action without `element_id`.
- [ ] Run the tests and observe raw `ValueError`/Pydantic validation becoming `internal-error`.
- [ ] Add a sanitized `ModelResponseValidationError` carrying role, reason, and provider response summary; raise it for unknown IDs and incomplete actions.
- [ ] Catch it around each role call in `RunAgent`, append a `model-failure` timeline event, and finalize with `model-failure` and a clear reason.
- [ ] Preserve the OpenAI client’s sanitized request/response record, attempts, prompt/schema versions, and latency.
- [ ] Run focused model-role and run-agent tests.

### Task 4: Enrich Cognitive State Without Private Data

**Files:**
- Modify: `src/ux_analyzer/domain/attention.py`
- Modify: `src/ux_analyzer/providers/cognitive.py`
- Modify: `src/ux_analyzer/application/run_agent.py`
- Modify: `src/ux_analyzer/prompts/cognitive-v1.txt`
- Test: `tests/unit/providers/test_model_roles.py`
- Test: `tests/e2e/test_private_data_leakage.py`

- [ ] Write failing tests asserting cognitive payloads include previous action/result, viewport ID, completed fixture keys, available controls, and persona parameters while excluding selectors, test IDs, private values, verifier fields, and numeric scores.
- [ ] Run the tests and observe missing context fields.
- [ ] Add safe optional context fields to observations and serialize them through `CognitiveObservation`; derive completed fixture keys from successful actions and expose only qualitative persona parameters.
- [ ] Update the cognitive prompt to prefer a different control after successful fixture input and to abandon clearly when no safe progress is possible.
- [ ] Run model-role and leakage tests.

### Task 5: Persist Semantic Manifest and Report Diagnostics

**Files:**
- Modify: `src/ux_analyzer/ports/artifacts.py`
- Modify: `src/ux_analyzer/storage/run_bundle.py`
- Modify: `src/ux_analyzer/reporting/renderer.py`
- Modify: `src/ux_analyzer/reporting/static/report.js`
- Test: `tests/integration/storage/test_run_bundle.py`
- Test: `tests/integration/reporting/test_renderer.py`

- [ ] Write failing tests asserting the manifest contains scenario/version/persona/policy/seed, and report data contains final outcome, failure reason, model details, repeated-action diagnostics, and timeline events.
- [ ] Run the tests and observe those fields absent or not projected.
- [ ] Add backward-compatible semantic fields to `BundleManifest`, write them from `RunSpec`, and project `model-failure`/repetition events through the report allowlist.
- [ ] Keep HTML autoescaping, static filesystem operation, and private-data filtering intact.
- [ ] Run storage and report tests, then regenerate a report from a finalized failed fixture bundle.

### Task 6: Full Verification and One Live Run

**Files:**
- No production files unless verification exposes a focused defect.
- Preserve generated artifact: `reports/single-run/report.html`.

- [ ] Run `rtk uv run pytest tests/unit tests/integration -q`.
- [ ] Run `rtk uv run pytest -m "not live and not e2e" -q`, Ruff, Pyright, and `git diff --check`.
- [ ] Start the fixture server and execute exactly one `uxa run-one` command with the requested semantic inputs.
- [ ] Confirm one finalized bundle, `experiment.json`, `report.html`, terminal outcome, verifier result, model-call count/latency, and no private values in report artifacts.
- [ ] Run the opt-in live role compatibility test when the configured `.env` permits it; report any external blocker precisely.
