# Prominence Provider Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible benchmark that compares heuristic, FoveaCast, and calibrated late-fusion prominence and produces a decision report.

**Architecture:** Add a focused benchmark module that consumes labeled cases, scores existing provider outputs, calibrates a one-parameter late fusion on a calibration split, and evaluates all strategies on holdout cases. Keep corpus construction and learned-model execution in a benchmark script so production provider behavior remains unchanged.

**Tech Stack:** Python 3.12, NumPy, Pillow, ONNX Runtime, pytest, JSON, Markdown.

## Global Constraints

- Primary objective is human-labeled ranking fidelity; simulated-agent outcomes are secondary.
- Never count FoveaCast fallback output as learned output.
- Calibration and holdout cases must be disjoint.
- Preserve raw provider scores, component evidence, timing, and provenance.
- Do not change production provider behavior while benchmarking.

---

### Task 1: Ranking Metrics and Fusion

**Files:**
- Create: `src/ux_analyzer/benchmarking/prominence_comparison.py`
- Create: `src/ux_analyzer/benchmarking/__init__.py`
- Test: `tests/unit/benchmarking/test_prominence_comparison.py`

**Interfaces:**
- Produces: `RankingMetrics`, `score_ranking(labels, probabilities)`, `fuse_probabilities(heuristic, foveacast, weight)`, and `select_fusion_weight(cases)`.

- [ ] Write tests for perfect, reversed, tied, and partial rankings, plus fusion endpoint identity and holdout-safe coefficient selection.
- [ ] Run `rtk uv run pytest tests/unit/benchmarking/test_prominence_comparison.py -q` and confirm failure because the module is missing.
- [ ] Implement deterministic metrics and grid-search fusion for weights `0.0` through `1.0` in steps of `0.05`.
- [ ] Re-run the focused tests and confirm they pass.

### Task 2: Labeled Visual Corpus

**Files:**
- Create: `benchmarks/prominence/cases.json`
- Create: `benchmarks/prominence/assets/*.png`
- Create: `tests/unit/benchmarking/test_prominence_corpus.py`

**Interfaces:**
- Produces: labeled calibration/holdout cases with screenshot path, viewport, element bounds/roles/labels/diagnostics, and integer relevance grades from 0 to 3.

- [ ] Write a schema/coverage test requiring both splits, all defined challenge tags, at least 24 cases, unique IDs, valid bounds, and at least one positive target per case.
- [ ] Run the corpus test and confirm failure because the corpus is missing.
- [ ] Generate deterministic raster screenshots and matching element metadata for the challenge matrix.
- [ ] Re-run the corpus test and confirm it passes.

### Task 3: Benchmark Runner and Report

**Files:**
- Create: `scripts/benchmark_prominence.py`
- Modify: `pyproject.toml`
- Test: `tests/integration/benchmarking/test_prominence_benchmark.py`

**Interfaces:**
- Consumes: `benchmarks/prominence/cases.json`, `HeuristicProminenceProvider`, `FoveacastSaliencyProvider`, `aggregate_saliency`, and ranking metrics.
- Produces: `summary.json` and `decision.md` under a selected output directory.

- [ ] Write an integration test using deterministic provider outputs that asserts split isolation, per-case provenance, aggregate metrics, selected weight, and recommendation fields.
- [ ] Run the integration test and confirm failure because the runner is missing.
- [ ] Implement the runner, strict learned-output validation, timing, JSON serialization, and Markdown rendering.
- [ ] Re-run the integration test and confirm it passes.

### Task 4: Real Benchmark Execution

**Files:**
- Generated: `reports/prominence-comprehensive/summary.json`
- Generated: `reports/prominence-comprehensive/decision.md`
- Modify: `docs/testing.md`

**Interfaces:**
- Produces: a completed real-model comparison and a reproducible command.

- [ ] Verify FoveaCast model readiness and corpus validity.
- [ ] Run `rtk uv run python scripts/benchmark_prominence.py --cases benchmarks/prominence/cases.json --output reports/prominence-comprehensive`.
- [ ] Inspect every invalid/fallback case and fix benchmark defects with regression tests.
- [ ] Document the execution and interpretation command in `docs/testing.md`.
- [ ] Run focused tests, non-live tests, Ruff, Pyright, and `git diff --check`.
- [ ] Write the final architecture recommendation from the generated evidence without overstating synthetic labels as real-user data.

