# Focused Benchmark Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make long experiments resumable and methodologically useful, terminate semantic action cycles early, separate simulated-user effort from benchmark operating cost, and validate the changes with a balanced four-cell live benchmark.

**Architecture:** Add a pure bounded cycle detector to semantic progress handling and feed it recaptured safe UI states from `RunAgent`. Add an atomic experiment checkpoint that records selected, finalized, failed, interrupted, and pending run IDs; use finalized bundle validation to skip completed runs on resume. Keep deterministic attention policies to one seed while retaining repeated seeded trials only for stochastic progressive policies. Derive user-effort and analysis-cost summaries independently in the report.

**Tech Stack:** Python 3.12, dataclasses, Typer, Playwright, pytest, Jinja2, Ruff, Pyright.

## Global Constraints

- Preserve finalized run-bundle immutability, checksums, redaction, and filesystem-openable reports.
- Cycle detection may use only persona-safe semantic action/state data; never selectors, test IDs, execution references, fixture values, verifier state, or numeric prominence/scent scores.
- Cycle lengths 2 through 4 terminate as `agent-abandoned` with `repeated-action-cycle`; identical-action and repeated-fixture guards remain available.
- Resume is enabled explicitly with `--resume`, skips only integrity-valid finalized bundles whose run IDs belong to the selected matrix, and atomically checkpoints after each run result or execution failure.
- Deterministic policies (`full-list`, `prominence-ranked-list`) execute only the first configured seed; stochastic progressive policies retain every configured seed.
- User effort and analysis cost remain separate. User effort contains simulated actions, observations, discovery cost, and estimated task time. Analysis cost contains model calls, attempts, latency, and tokens; monetary cost is shown only when explicit pricing exists.
- The focused live experiment contains exactly four cells: invite-teammate, first-time-nontechnical, defective/improved, full-list/progressive-prominence-scent, seed 0.
- Do not run the 160-cell matrix during this implementation.

---

### Task 1: Detect Repeated Semantic Action Cycles

**Files:**
- Modify: `src/ux_analyzer/application/progress.py`
- Modify: `src/ux_analyzer/application/run_agent.py`
- Modify: `src/ux_analyzer/reporting/renderer.py`
- Modify: `src/ux_analyzer/reporting/static/report.js`
- Test: `tests/unit/application/test_progress.py`
- Test: `tests/integration/application/test_run_agent.py`
- Test: `tests/integration/reporting/test_renderer.py`

**Interfaces:**
- Produces a bounded safe transition signature and `repeated_cycle_length(history, min_length=2, max_length=4) -> int | None`.
- `RunAgent` emits `repeated-action-cycle` with safe `cycle_length` and terminates `agent-abandoned` after the second complete repetition.

- [ ] **Step 1: Write failing pure tests** for period-2, period-3, period-4, insufficient history, and non-equivalent semantic states.
- [ ] **Step 2: Run `rtk uv run pytest tests/unit/application/test_progress.py -q`** and confirm failure because cycle detection does not exist.
- [ ] **Step 3: Implement the bounded suffix-cycle detector** using action fingerprints plus before/after semantic snapshot signatures and safe URL origin/path state.
- [ ] **Step 4: Write failing RunAgent/report tests** proving Share/Close-style cycles finalize early, appear in the timeline/report, and do not leak private data.
- [ ] **Step 5: Run the focused integration tests** and confirm the expected pre-implementation failures.
- [ ] **Step 6: Integrate cycle history and terminal events** without weakening existing no-progress, fixture, verification, or budget behavior.
- [ ] **Step 7: Run focused tests, Ruff, and Pyright** for changed modules.

### Task 2: Add Atomic Experiment Checkpoints And Resume

**Files:**
- Create: `src/ux_analyzer/application/checkpoint.py`
- Modify: `src/ux_analyzer/application/experiment.py`
- Modify: `src/ux_analyzer/cli.py`
- Test: `tests/unit/application/test_checkpoint.py`
- Test: `tests/integration/application/test_experiment_runner.py`
- Test: `tests/integration/cli/test_commands.py`

**Interfaces:**
- Produces `ExperimentCheckpointStore` with atomic initialize/update/load operations at `experiment-progress.json`.
- `ExperimentRunner.run` accepts an async-compatible completion callback invoked after each finalized result or sanitized execution failure.
- `uxa run --resume` validates selected run IDs against finalized bundles, skips valid completed runs, marks interrupted staging runs, and reports pending/resumed counts.

- [ ] **Step 1: Write failing checkpoint tests** for atomic initialization, per-run update, malformed checkpoint rejection, and selected-matrix mismatch.
- [ ] **Step 2: Run checkpoint tests** and confirm failure because the store does not exist.
- [ ] **Step 3: Implement the checkpoint store** with same-directory temporary replace and redacted semantic status records.
- [ ] **Step 4: Write failing runner/CLI tests** for callback ordering, resume skip, invalid finalized bundle rerun, and interrupted staging reporting.
- [ ] **Step 5: Run focused runner/CLI tests** and confirm behavioral failures.
- [ ] **Step 6: Add callback and `--resume` orchestration**, preserving normal non-resume collision behavior and complete report generation.
- [ ] **Step 7: Run focused tests, Ruff, and Pyright** for changed modules.

### Task 3: Remove Deterministic Seed Duplication And Add Focused Matrix

**Files:**
- Modify: `src/ux_analyzer/application/experiment.py`
- Modify: `src/ux_analyzer/cli.py`
- Modify: `src/ux_analyzer/domain/benchmark.py`
- Modify: `src/ux_analyzer/config/loader.py`
- Modify: `benchmarks/demo/project.yaml`
- Create: `benchmarks/demo/experiments/focused-validation.yaml`
- Test: `tests/unit/application/test_experiment.py`
- Test: `tests/unit/config/test_loader.py`
- Test: `tests/integration/cli/test_commands.py`

**Interfaces:**
- Produces policy-specific seed expansion: deterministic policies use the first configured seed; progressive policies use all seeds.
- Adds experiment ID `focused-validation`, expanding to exactly four balanced RunSpecs.

- [ ] **Step 1: Write failing expansion tests** proving ten configured repetitions produce one deterministic cell and ten progressive cells with stable run IDs.
- [ ] **Step 2: Run expansion tests** and confirm the current Cartesian expansion fails the expectation.
- [ ] **Step 3: Implement explicit policy stochasticity metadata and policy-specific seed expansion** with CLI disclosure of suppressed deterministic repetitions.
- [ ] **Step 4: Write failing configuration/CLI tests** for the exact four focused cells.
- [ ] **Step 5: Add and load `focused-validation`**, then run focused tests until green.
- [ ] **Step 6: Run Ruff and Pyright** for changed modules.

### Task 4: Separate User Effort From Analysis Cost

**Files:**
- Modify: `src/ux_analyzer/reporting/renderer.py`
- Modify: `src/ux_analyzer/reporting/templates/experiment.html.j2`
- Modify: `src/ux_analyzer/reporting/static/report.js`
- Modify: `src/ux_analyzer/reporting/static/report.css`
- Test: `tests/integration/reporting/test_renderer.py`

**Interfaces:**
- Produces per-run and aggregate `user_effort` values: actions, observations, discovery cost, and estimated task seconds using a versioned, disclosed deterministic formula.
- Produces per-run and aggregate `analysis_cost` values: calls, attempts, latency, prompt/completion/total tokens, and optional monetary estimate.

- [ ] **Step 1: Write failing report tests** for independent effort/cost columns, exact hand-derived totals, unavailable monetary pricing, and private-data exclusion.
- [ ] **Step 2: Run renderer tests** and confirm missing summaries.
- [ ] **Step 3: Implement report projections and visible formula/version disclosure** without folding analysis latency or token cost into simulated UX effort.
- [ ] **Step 4: Update responsive report markup/styles and playback details** for the new values.
- [ ] **Step 5: Run all renderer tests, Ruff, and Pyright** for changed modules.

### Task 5: Verify Locally And Run Four Live Cells

**Files:**
- Modify: `docs/testing.md`
- Preserve: `reports/focused-validation/`

**Interfaces:**
- Consumes the `focused-validation` experiment and the existing fixture/OpenAI-compatible adapters.
- Produces finalized bundles, `experiment-progress.json`, `experiment.json`, and `report.html` for four selected cells.

- [ ] **Step 1: Run `rtk uv run pytest tests/unit tests/integration -q`**.
- [ ] **Step 2: Run `rtk uv run pytest -m "not live and not e2e" -q`**.
- [ ] **Step 3: Run `rtk uv run ruff check .`, `rtk uv run pyright`, and `rtk git diff --check`**.
- [ ] **Step 4: Verify the fixture server or start it on `127.0.0.1:8000`**.
- [ ] **Step 5: Run `rtk uv run uxa run benchmarks/demo/project.yaml --experiment focused-validation --workers 4 --fixture-origin http://127.0.0.1:8000 --output reports/focused-validation`**.
- [ ] **Step 6: Regenerate and inspect `reports/focused-validation/report.html`**, confirming four trusted cells, intelligible outcomes, early cycle detection where applicable, improved-versus-defective direction, and separated cost measures.
- [ ] **Step 7: Run live role compatibility** with `tests/live/test_openai_endpoint.py` when `.env` enables live tests.
- [ ] **Step 8: Update testing documentation** with focused and resume commands and record observed limitations without overstating validity.
