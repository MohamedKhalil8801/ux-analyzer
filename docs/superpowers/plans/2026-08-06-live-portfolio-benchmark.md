# Live Portfolio Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a small, human-constrained FoveaCast-versus-heuristic benchmark against Mohamed Khalil's animated public portfolio.

**Architecture:** Add an explicit `live` application version with a configured HTTPS start URL and resource-origin allowlist while preserving fixture-derived URLs and private control APIs for existing benchmarks. Public targets use visible-result verification, no fixture reset/state endpoints, and configurable post-navigation/action settling so saliency screenshots and extracted geometry represent the same settled viewport. Portfolio scenarios use progressive prominence without semantic scent to isolate visual-attention provider differences.

**Tech Stack:** Python 3.12, Pydantic, Typer, Playwright, pytest, YAML, FoveaCast ONNX runtime.

## Global Constraints

- Benchmark contains exactly three user tasks, within requested 1-4 range.
- Browser begins at `https://mohamed-khalil.vercel.app` for every run.
- Agent receives no selectors, hidden labels, destination URLs, saliency values, verifier state, direct URL jumps, or private fixture state.
- Comparison crosses only `heuristic` and `foveacast` under `progressive-prominence`; semantic scent is disabled.
- Verification uses persona-visible text currently inside viewport.
- Public target access remains exact-origin allowlisted; unrelated external links remain blocked.
- Existing fixture benchmark behavior and configuration remain compatible.
- Existing unrelated dirty-worktree changes remain untouched.

---

### Task 1: Live Target Configuration

**Files:**
- Modify: `src/ux_analyzer/config/models.py`
- Modify: `src/ux_analyzer/config/loader.py`
- Modify: `src/ux_analyzer/domain/benchmark.py`
- Test: `tests/unit/config/test_loader.py`

**Interfaces:**
- Produces: `ApplicationVersionKind.LIVE`.
- Produces: `ApplicationVersion.start_url: str | None`, `allowed_origins: tuple[str, ...]`, `navigation_settle_ms: int`, and `action_settle_ms: int`.
- Preserves: fixture `defective`/`improved` version pairing and legacy defaults.

- [x] **Step 1: Write failing loader tests**

Add tests proving a single `live` version loads with an HTTPS URL, explicit resource origins, and settle timings; live versions reject missing URLs; fixture applications still reject missing defective/improved partners.

```python
def test_loads_single_live_application_version(tmp_path: Path) -> None:
    project = _read_project()
    project["applications"][0]["versions"] = [{
        "id": "portfolio-live",
        "kind": "live",
        "label": "Production",
        "start_url": "https://portfolio.example/work",
        "allowed_origins": ["https://fonts.example"],
        "navigation_settle_ms": 2000,
        "action_settle_ms": 900,
    }]
    project["scenarios"][0]["application_version_ids"] = ["portfolio-live"]
    project["scenarios"][0]["evaluation_target"]["labels_by_version"] = {
        "live": "Work"
    }
    project["experiments"][0]["application_version_ids"] = ["portfolio-live"]

    version = load_project(_write_project(tmp_path, project)).project.applications[0].versions[0]

    assert version.kind is ApplicationVersionKind.LIVE
    assert version.start_url == "https://portfolio.example/work"
    assert version.allowed_origins == ("https://fonts.example",)
    assert version.navigation_settle_ms == 2000
    assert version.action_settle_ms == 900
```

- [x] **Step 2: Run tests and confirm RED**

Run: `uv run pytest tests/unit/config/test_loader.py -q`

Expected: failure because `live` and URL/timing fields are unsupported.

- [x] **Step 3: Implement minimal typed configuration**

Add Pydantic fields with non-negative timing validation, map them into immutable domain values, and update application invariants: all-live applications may contain live versions; controlled applications still require defective and improved versions.

- [x] **Step 4: Run focused tests and confirm GREEN** (loader suite passes)

Run: `uv run pytest tests/unit/config/test_loader.py -q`

Expected: all loader tests pass.

### Task 2: Safe Public Browser Sessions

**Files:**
- Modify: `src/ux_analyzer/adapters/web/network_policy.py`
- Modify: `src/ux_analyzer/adapters/web/session.py`
- Modify: `src/ux_analyzer/adapters/web/extractor.py`
- Modify: `src/ux_analyzer/ports/observation.py`
- Modify: `src/ux_analyzer/cli.py`
- Test: `tests/integration/web/test_session_safety.py`
- Test: `tests/integration/cli/test_commands.py`

**Interfaces:**
- Produces: exact HTTP(S) origin allowlisting for explicitly configured public targets while retaining `fixture_only` host checks.
- Produces: public observation sessions that reload browser state but never call `/__control/*`.
- Produces: `ObservationSessionConfig.navigation_settle_ms` and `action_settle_ms`.
- Produces: extractor screenshot bytes aligned with extracted geometry and used by FoveaCast.

- [x] **Step 1: Write failing public-session tests**

Cover explicit public-origin normalization, blocked unlisted origins, live start URL selection, no control HTTP calls for live runs, and configured waits after navigation/scroll.

```python
def test_session_config_uses_live_start_url(tmp_path: Path) -> None:
    spec = _live_run_spec()
    config = cli._session_config(spec, output=tmp_path, fixture_origin="http://127.0.0.1:8000")
    assert config.start_url == "https://portfolio.example/work"
    assert config.navigation_settle_ms == 2000
    assert config.action_settle_ms == 900
```

- [x] **Step 2: Run tests and confirm RED**

Run: `uv run pytest tests/integration/web/test_session_safety.py tests/integration/cli/test_commands.py -q`

Expected: failures from remote-origin rejection and missing live session behavior.

- [x] **Step 3: Implement exact-origin live access and settling**

Keep fixture methods fail-closed. Build matrix browser allowlist from fixture origin plus exact origins declared by selected live versions. Make fixture control lifecycle conditional on non-live versions. Apply configured waits after document navigation and executable actions before recapture.

- [x] **Step 4: Align screenshot and geometry capture**

Return screenshot bytes from `capture_with_diagnostics`; replace adapter's preliminary screenshot with extractor screenshot when creating `ObservationCapture`, ensuring saliency map and element bounds describe one capture point.

- [x] **Step 5: Run focused tests and confirm GREEN** (focused web/CLI suites pass)

Run: `uv run pytest tests/integration/web/test_session_safety.py tests/integration/cli/test_commands.py -q`

Expected: all focused tests pass.

### Task 3: Portfolio Benchmark Definition

**Files:**
- Create: `benchmarks/portfolio/project.yaml`
- Modify: `docs/testing.md`
- Test: `tests/unit/config/test_loader.py`

**Interfaces:**
- Produces: experiment `portfolio-foveacast-vs-heuristic` with six cells: three scenarios by two prominence providers.
- Scenarios: current work, open-source scale, contact availability.

- [x] **Step 1: Write failing benchmark-shape test**

```python
def test_portfolio_benchmark_is_small_visual_only_matrix() -> None:
    path = Path(__file__).parents[3] / "benchmarks" / "portfolio" / "project.yaml"
    loaded = load_project(path)
    experiment = loaded.project.experiments[0]
    assert len(loaded.project.scenarios) == 3
    assert experiment.prominence_provider_ids == ("heuristic", "foveacast")
    assert experiment.policies == (ExperimentPolicy.PROGRESSIVE_PROMINENCE,)
    assert len(expand_experiment(ExperimentContext(
        definition=experiment,
        project=loaded.project,
        config_digest=loaded.config_digest,
    ))) == 6
```

- [x] **Step 2: Run test and confirm RED** (missing benchmark file failure recorded)

Run: `uv run pytest tests/unit/config/test_loader.py::test_portfolio_benchmark_is_small_visual_only_matrix -q`

Expected: file-not-found failure.

- [x] **Step 3: Add exact scenario config** (live target, null timeout, 60/30/24/40, attention 3/1)

Use one `live` version. Configure target plus Google Fonts origins, `navigation_settle_ms: 2400`, `action_settle_ms: 2800`, `1440x900` viewport, visible-result verifiers, `timeout_seconds: null`, and bounded budgets of 60 steps, 30 observations, 24 interactions, and 40 model calls. Live audit measured counters settling around 2400 ms after entering viewport, so action settling must remain 2800 ms rather than 1200 ms. Wall-clock model latency is reported operating cost and must not become a UX failure. Configure attention `batch_size: 3` and `cross_region_exploration: 1`; keep progressive-prominence without scent calls or target oracle:

```yaml
- Find Mohamed's current employer, role, and primary web frontend technologies.
- Find stated city and country coverage for Mohamed's open-source prayer-time project.
- Find Mohamed's public email address and whether he is open to remote frontend work.
```

Evaluation targets are top-navigation links `PAIR Systems`, `Open Prayer Times`, and `Contact`. Primary verifier texts are `Frontend Engineer`, `48,000+`, and `m.khalil.bus@gmail.com`; `all_of` strings are respectively `PAIR Systems`, `React`, `TypeScript`; `cities calibrated`, `130+`, `countries`; and `Open to mobile and web roles`, `open to remote`. Omit verifier roles so live extracted result roles do not invalidate checks.

- [x] **Step 4: Document validation and execution commands** (PowerShell Codex command)

Document:

```text
$env:UXA_LLM_MODE = "codex"
$env:UXA_LLM_COGNITIVE_REASONING_EFFORT = "low"
$env:UXA_LLM_TIMEOUT_SECONDS = "180"
$env:UXA_LLM_MAX_CONCURRENT_CALLS = "1"
uv run uxa validate benchmarks/portfolio/project.yaml
uv run uxa run benchmarks/portfolio/project.yaml --experiment portfolio-foveacast-vs-heuristic --workers 1 --output reports/portfolio-foveacast-vs-heuristic-final3
```

- [x] **Step 5: Run focused test and confirm GREEN** (6-spec dry-run, loader tests pass)

Run: `uv run pytest tests/unit/config/test_loader.py::test_portfolio_benchmark_is_small_visual_only_matrix -q`

Expected: one passing test and six expanded cells.

### Task 4: Execute, Diagnose, and Review

**Files:**
- Generated ignored evidence: `reports/portfolio-foveacast-vs-heuristic/`
- Modify only files implicated by reproduced non-UX failures.

**Interfaces:**
- Consumes: six-cell benchmark from Task 3.
- Produces: comparison-valid experiment report and evidence-based provider comparison.

- [ ] **Step 1: Run deterministic verification**

Run:

```text
uv run pytest -m "not live" -q
uv run ruff format --check .
uv run ruff check .
uv run pyright
git diff --check
```

- [ ] **Step 2: Check model readiness and benchmark configuration**

Run:

```text
uv run uxa models status foveacast-v0.2.0 --provider cpu
uv run uxa validate benchmarks/portfolio/project.yaml
uv run uxa run benchmarks/portfolio/project.yaml --experiment portfolio-foveacast-vs-heuristic --workers 1 --output reports/portfolio-foveacast-vs-heuristic --dry-run
```

- [ ] **Step 3: Execute real comparison**

Run the six cells serially. Do not accept FoveaCast fallback, comparison-invalid samples, leaked private state, or automation failures as UX failures.

- [ ] **Step 4: Diagnose every failed or invalid cell**

For each failure, inspect run timeline, screenshots, saliency metadata, blocked requests, model records, and terminal verification. Add a regression test before fixing code. Re-run only affected cells, then full six-cell experiment with `--resume` where integrity permits.

- [ ] **Step 5: Dispatch harsh independent reviewers**

One reviewer checks plan/spec and human-capability boundaries; one checks code quality/security; one checks benchmark evidence and whether conclusions overclaim. Fix every Critical/Important finding and re-review until clean.

- [ ] **Step 6: Perform final verification and summarize evidence**

Report per-provider completion, observations, interactions, discovery cost, target rank/prominence, inference cost, invalid/fallback counts, and actual portfolio UX findings. Distinguish simulated-agent evidence from real-user claims.
