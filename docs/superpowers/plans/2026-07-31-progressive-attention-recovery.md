# Progressive Attention Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve progressive attention coverage and replace lifetime repeated-action counting with bounded, meaningful no-progress recovery.

**Architecture:** Extend the existing attention-policy call with a recovery level while keeping normal seeded sampling intact. RunAgent derives meaningful progress from safe recaptured snapshot signatures, resets consecutive repetition after progress, and asks the attention policy for deterministic bounded recovery observations after stalls.

**Tech Stack:** Python 3.12, dataclasses, Typer/Pydantic configuration, pytest, Ruff, Pyright.

## Global Constraints

- Cognitive input must still exclude selectors, test IDs, execution references, fixture values, verifier state, and numeric prominence/scent scores.
- Normal progressive observations remain seeded and reproducible.
- Recovery reveals at most three elements and never performs an action.
- Full-list and ranked-list behavior remain unchanged.
- Every terminal path still finalizes the run bundle and report.

---

### Task 1: Calibrate Progressive Attention And Add Recovery Selection

**Files:**
- Modify: `src/ux_analyzer/providers/attention_policy.py`
- Modify: `src/ux_analyzer/providers/full_list_policy.py`
- Modify: `src/ux_analyzer/providers/ranked_list_policy.py`
- Modify: `src/ux_analyzer/config/models.py`
- Modify: `src/ux_analyzer/config/loader.py`
- Modify: `src/ux_analyzer/cli.py`
- Modify: `benchmarks/demo/project.yaml`
- Test: `tests/unit/providers/test_attention_policy.py`
- Test: `tests/unit/config/test_loader.py`

**Interfaces:**
- Produces `next_observation(..., *, recovery_level: int = 0)` for all attention policies.
- Progressive recovery uses existing candidate logits and returns
  `selection_mode="recovery-ranked"`.

- [x] Add failing tests for default batch size 2, seeded reproducibility, and bounded high-scent recovery.
- [x] Run the focused tests and confirm failure from the current batch size/signature.
- [x] Add immutable lineage-based recovery while preserving the normal seeded draw.
- [x] Set configuration and demo defaults to batch size 2 and calibrate demo scent weight.
- [x] Run provider/config tests and Ruff/Pyright for changed modules.

### Task 2: Detect Meaningful Progress And Reset Consecutive Repetition

**Files:**
- Create: `src/ux_analyzer/application/progress.py`
- Modify: `src/ux_analyzer/application/run_agent.py`
- Modify: `src/ux_analyzer/reporting/renderer.py`
- Modify: `src/ux_analyzer/reporting/static/report.js`
- Test: `tests/unit/application/test_progress.py`
- Test: `tests/integration/application/test_run_agent.py`
- Test: `tests/integration/reporting/test_renderer.py`

**Interfaces:**
- Produces `snapshot_progress_signature(snapshot) -> tuple[tuple[object, ...], ...]`.
- RunAgent passes `recovery_level=context.no_progress_count` to attention selection.
- Emits `no-progress-recovery` for recoverable stalls and
  `no-progress-detected` before terminal abandonment.

- [x] Add failing pure tests proving bounds-only changes are not progress and lineage/actionability changes are progress.
- [x] Add failing integration tests proving fixture completion resets earlier scroll repetition and repeated semantic stalls finalize as agent-abandoned.
- [x] Implement safe semantic snapshot signatures and post-recapture progress evaluation.
- [x] Replace lifetime action counts with consecutive no-progress fingerprint state.
- [x] Project and render recovery events and forced selections safely.
- [ ] Run all required verification commands and one exact live run.
