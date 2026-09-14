# Creative Redesign Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Creative Redesign pipeline and report tab: a capable model reads whole persisted page captures, infers page intent, section relationships, and audience, then publishes schema-validated, cross-page-consistent design proposals — grouping, whitespace, unification, radical redesign, copy, relocation, simplification, new sections, and accessibility — as clearly-labeled model estimates in a new report.html tab.

**Architecture:** Keep run execution unchanged. At experiment completion (or via a standalone command), one shared browser pass per page over a deterministic crawled-corpus page list persists a versioned `page-capture.json` sidecar; the existing live audit consumes the same pass. A two-role isolated model pipeline (Proposer per page, then one Critic/Merger across pages) turns captures into Design Proposals with impact×effort ratings and a required deliberate-choice guardrail, validated deterministically and persisted as immutable attempts. `uxa report` renders the newest valid attempt in a stamped Redesign tab and stays model-free.

**Tech Stack:** Python 3.12, dataclasses and Pydantic v2, Typer, Jinja2, vanilla JavaScript/CSS, existing OpenAI-compatible and Codex structured transports, pytest, Playwright.

## Global Constraints

- `uxa report` is offline and must never initiate a model call.
- Design Proposals are model estimates (Report flag/amber ink in the report). They never cite Evidence References, never enter the run evidence corpus, and never render as UX findings.
- The redesign pipeline is separate from the four-role report synthesis pipeline; failure in one never invalidates the other or the deterministic report.
- Page scope is one deterministic list: exploration corpus URLs when present, else application start URLs; start URLs first, then BFS discovery order, deduped, capped by `UXA_REDESIGN_MAX_PAGES` (default 10). Audit and redesign share this list and one browser capture pass per page.
- Full-page screenshots are sliced into bounded segments (~2000px, individual JPEG encode ≤640KB per segment) covering up to `UXA_REDESIGN_MAX_PAGE_HEIGHT` (default 12000px); truncation beyond the cap is recorded, never silent, and the node/copy inventory covers the remainder.
- Grouping/unification/simplification/move proposals require a non-empty `deliberate_choice_check` naming the potentially-intentional pattern and why the proposal still stands.
- Principles come from a versioned, static Redesign Principle Pack (Gestalt, Nielsen, lawsofux.com sources, WCAG 2.2 anchors, copy/tone) in original standalone language; principles may name or explain, never prove.
- New roles `redesign-proposer` and `redesign-critic-merger` join the ModelRole enum; fresh context per role; same prompt/contract through OpenAI-compatible and Codex transports (Codex best-effort, not live-verified).
- `UXA_REDESIGN_MODEL` with fallback to `UXA_REPORT_MODEL` (both must support structured output + image input); no secrets in YAML, source, or docs.
- Feature gates: `UXA_REDESIGN_ENABLED` (default off; dotenv-loaded, explicit environment wins), `UXA_REPORT_SYNTHESIS_ENABLED` overrides `evaluation.report_synthesis.enabled` when set; unset env never changes current behavior.
- Attempts are immutable (`redesign/<attempt-id>/`, digest-checked); regeneration creates a new attempt; deterministic validation gates publication; failures and rejections are recorded, not omitted.
- Best-effort discipline: capture/redesign failures never fail an otherwise healthy run; they land in the attempt status and the stamped tab placeholder.
- Reports remain standalone/offline; the tab renders everything from persisted attempt payloads.

## File Map

- Create `src/ux_analyzer/domain/redesign.py`: page capture references, Design Proposal, category/impact/effort enums, deliberate_choice_check, RedesignAttempt, RedesignAttemptStatus.
- Create `src/ux_analyzer/providers/redesign_principles.py`: versioned static Redesign Principle Pack (original language; Gestalt/Nielsen/WCAG/copy anchors; lawsofux.com as named source doctrine).
- Create `src/ux_analyzer/providers/redesign.py`: `redesign-proposer` and `redesign-critic-merger` role prompts + structured schemas (Pydantic), mirroring report_synthesis role classes.
- Create `src/ux_analyzer/analysis/page_capture.py`: capture list resolution (corpus-else-starts, ordering, cap), one shared browser pass per page (reuse settle/scroll logic from project_audit), segmented screenshot + trimmed node inventory + copy inventory; `PAGE_CAPTURE_FILENAME = "page-capture.json"`, schema `page-capture-v1`.
- Create `src/ux_analyzer/application/redesign.py`: capture→prompt payload builder, per-page proposer loop, critic/merger call, cross-page consistency repair, deterministic validation (schema, refs resolve to capture sections, deliberate-choice presence, bounded counts), publication gate.
- Create `src/ux_analyzer/storage/redesign_artifacts.py`: immutable attempt store (`redesign/<attempt-id>/index.json`, `payload.json`, digest), newest-valid-attempt selection.
- Modify `src/ux_analyzer/analysis/project_audit.py`: emit capture list + shared browser pass hook; audit stays backward-compatible.
- Modify `src/ux_analyzer/cli.py`: `uxa redesign` command (`--pages`, `--audience`, `--output`), auto-run flag after experiments, `UXA_REDESIGN_ENABLED`/`UXA_REPORT_SYNTHESIS_ENABLED` env gates, audit URL resolution via shared page list.
- Modify `src/ux_analyzer/ports/models.py` and `src/ux_analyzer/adapters/openai.py`: two new roles, image attachments (segment list).
- Modify `src/ux_analyzer/reporting/renderer.py`: load newest valid redesign attempt (bounded, schema-checked), expose tab context; absent-attempt placeholder data.
- Modify `src/ux_analyzer/reporting/templates/experiment.html.j2`, `static/report.css`, `static/report.js`: Redesign tab (cards, impact×effort chips, category/page filters, stamped placeholder), rail entry.
- Modify `.env.example`, `README.md`, `docs/model-provider.md`, `docs/architecture.md`: operator contracts.
- Test files per task below under `tests/unit/`, `tests/integration/`, `tests/e2e/`.

---

### Task 1: Design Proposal Domain Contracts

**Files:**
- Create: `src/ux_analyzer/domain/redesign.py`
- Test: `tests/unit/domain/test_redesign.py`

**Interfaces:**
- Produces: `DesignCategory` (grouping|whitespace|unification|radical-redesign|copy|relocation|simplification|new-section|accessibility), `Impact` (low|medium|high), `Effort` (small|medium|large), `SectionReference` (url, section_label, box, summary), `DeliberateChoiceCheck`, `DesignProposal` (proposal_id, page_url, category, title, observation, rationale, principle_ids, change, impact, effort, section_refs, also_affects, deliberate_choice_check|None), `RedesignAttemptStatus` (accepted|no-proposals|unavailable|rejected), `RedesignAttempt`.
- Consumes: nothing from the evidence corpus — deliberately independent (ADR 0007).

- [x] **Step 1: Write failing invariant tests** — proposal requires non-empty title/observation/change/rationale; `deliberate_choice_check` required exactly for grouping/unification/simplification/relocation categories; section_refs non-empty and URL-consistent; impact/effort enum-validated; attempt status enumerates all four values; immutability (frozen dataclasses).
- [x] **Step 2: Run** `uv run pytest tests/unit/domain/test_redesign.py -q` — expect import failure.
- [x] **Step 3: Implement** frozen dataclasses with `__post_init__` validation matching the repo's exploration.py style.
- [x] **Step 4: Run** full unit file; verify green.

### Task 2: Redesign Principle Pack

**Files:**
- Create: `src/ux_analyzer/providers/redesign_principles.py`
- Test: `tests/unit/providers/test_redesign_principles.py`

**Interfaces:**
- Produces: `RedesignPrinciple(id, name, statement, source)`; `redesign_principle_pack()` returning versioned tuple (ids like `gestalt-proximity`, `gestalt-similarity`, `nieland-consistency` → `nielsen-consistency`, `wcag-contrast-1.4.3`, `wcag-target-size-2.5.8`, `wcag-labels-3.3.2`, `wcag-reflow-1.4.10`, `copy-tone-fit`, `hierarchy-f-pattern`, `progressive-disclosure`, …) with pack version string.
- Constraint: original standalone language; lawsofux.com/Nielsen/WCAG cited as sources, never quoted at length; principles are interpretive only.

- [x] **Step 1: Failing test** — pack loads, ids unique, version present, no principle id collides with UX Principle Pack ids, statements non-empty.
- [x] **Step 2: Implement** static tuple + version constant (no I/O).
- [x] **Step 3: Green** + `uv run ruff check src/ux_analyzer/providers/redesign_principles.py`.

### Task 3: Page Capture Sidecar

**Files:**
- Create: `src/ux_analyzer/analysis/page_capture.py`
- Modify: `src/ux_analyzer/analysis/project_audit.py`
- Test: `tests/unit/analysis/test_page_capture.py`, `tests/integration/analysis/test_page_capture_browser.py`

**Interfaces:**
- Produces: `resolve_redesign_page_list(exploration_corpus, application_versions, cap) -> tuple[str, ...]` (starts first, BFS order, dedup, cap); `capture_page(url) -> PageCapture` (segmented screenshots, node inventory: sections/headings/forms/buttons/inputs/links with box+styles+text; copy inventory: paragraphs/headings; truncation flag); `PAGE_CAPTURE_SCHEMA = "page-capture-v1"`.
- Shares one browser pass with the audit: extract refactor so `_visual_analyze`'s session/settle/screenshot logic is reusable; audit output unchanged.
- Bounds: ≤640KB per segment JPEG; segment height ~2000px; `UXA_REDESIGN_MAX_PAGE_HEIGHT` env (default 12000); node inventory ≤1500 entries; copy text ≤200k chars.

- [x] **Step 1: Failing tests** for list resolution (corpus beats starts; ordering; dedup; cap; depth-0 edge) and payload bounds (segment count math, truncation flag).
- [x] **Step 2: Implement** resolution (pure, no browser) — green unit tests.
- [x] **Step 3: Implement** browser capture reusing existing settle logic (scroll-warm, fonts ready, networkidle); fixture_app live test asserts `page-capture.json` exists, schema matches, segments ≤ cap, inventory counts bounded.
- [x] **Step 4: Wire** audit pass to produce capture in the same session; assert existing audit tests unchanged and green.

### Task 4: Model Roles — Proposer and Critic/Merger

**Files:**
- Create: `src/ux_analyzer/providers/redesign.py`
- Modify: `src/ux_analyzer/ports/models.py`, `src/ux_analyzer/adapters/openai.py`
- Test: `tests/unit/providers/test_redesign_roles.py`, `tests/integration/models/test_redesign_transport.py`

**Interfaces:**
- Produces: `RedesignProposer` (role `redesign-proposer`, prompt version `redesign-proposer-v1`) with `analyze(page_payload, audience, principles)` → `ProposerResponse` (page_understanding: intent/audience-inference/section-relationships; proposals: list[DesignProposal-shaped Pydantic]); `RedesignCriticMerger` (role `redesign-critic-merger`, `redesign-critic-merger-v1`) with `review(consolidated, page_payloads_digests, principles)` → `CriticResponse` (final_proposals, killed: list[{proposal_id, reason}], consistency_notes).
- Prompts encode: whole-page holistic reading; nine proposal families; deliberate-choice guardrail wording; audience inference must be stated as inference; principles referenced by id only; no evidence-ID claims; JSON-schema output; segment images attached in order with y-offsets.
- Transport: same chat contract through OpenAI adapter (image segments as attachments) and Codex path (text-first, segments referenced by digest when transport cannot attach — recorded as a limitation).

- [x] **Step 1: Failing tests** — role enum contains both values; schema rejects proposals violating Task 1 invariants; deliberate-choice missing → schema error; prompt versions frozen strings; transport smoke test with stubbed HTTP (API) and stubbed codex exec.
- [x] **Step 2: Implement** roles mirroring `_ReportRole` structure from `providers/report_synthesis.py`.
- [x] **Step 3: Green** both unit and stubbed-transport integration tests.

### Task 5: Redesign Application Pipeline + Immutable Attempts

**Files:**
- Create: `src/ux_analyzer/application/redesign.py`, `src/ux_analyzer/storage/redesign_artifacts.py`
- Test: `tests/unit/application/test_redesign.py`, `tests/integration/storage/test_redesign_artifacts.py`, `tests/integration/application/test_redesign_pipeline.py`

**Interfaces:**
- Produces: `run_redesign_pass(captures, audience, settings) -> RedesignAttempt`; deterministic validation before publication: all section_refs resolve into the persisted capture for that URL; bounded counts (proposals ≤ 40 total, ≤ 12/page); deliberate-choice presence per category; principle ids all known; killed proposals removed with reasons preserved.
- Storage: `redesign/<attempt-id>/index.json` + `payload.json` + SHA-256 digest; `newest_valid_attempt(root)` selection; corruption → status `unavailable` recorded, never silent.
- Failure semantics: capture missing → attempt `unavailable` with reason; transport failure → `unavailable`; validation rejection → `rejected` with machine-readable reasons. Failures never raise into the caller's run.

- [x] **Step 1: Failing tests** — happy path with fake proposer/critic doubles; contradiction repair (critic output overrides per-page proposals); cap enforcement; validation rejects unknown principle id / dangling section ref / missing deliberate check; attempt immutability (second run new id, first untouched); newest-valid selection with a corrupt newer attempt.
- [x] **Step 2: Implement** pipeline + storage.
- [x] **Step 3: Green.**

### Task 6: CLI — `uxa redesign`, gates, audit unification

**Files:**
- Modify: `src/ux_analyzer/cli.py`
- Test: `tests/integration/cli/test_redesign_command.py`, `tests/unit/analysis/test_project_audit_pages.py`

**Interfaces:**
- `uxa redesign OUTPUT [--pages URL ...] [--audience TEXT] [--max-pages N]`: loads persisted `page-capture.json` when fresh, else captures; runs pass; writes attempt; prints status line mirroring audit/pagespeed echoes.
- Auto mode: after `_complete_experiment`, when `UXA_REDESIGN_ENABLED` truthy → resolve page list (shared with audit), one shared capture pass, audit + capture artifacts, then redesign attempt; failure path mirrors `_write_ux_audit` best-effort warnings.
- `UXA_REPORT_SYNTHESIS_ENABLED` overrides YAML synthesis gate when set; unset → current behavior; dotenv loads via existing `load_environment_file` (explicit env wins).
- Audit URL resolution switches to the shared page list when an exploration artifact exists (Page findings may then cover more URLs than start URLs; renderer requires no change — it renders whatever `ux_audit.urls` contains).

- [x] **Step 1: Failing CLI tests** — command absent-artifact path; enabled-gate on/off; env override of synthesis YAML; audit page list uses corpus when present (stub crawler); `--pages` override; deterministic ordering printed in dry-run. *(coverage in `tests/integration/cli/test_redesign_command.py`, incl. page-list unification)*
- [x] **Step 2: Implement** command + gates + audit resolution refactor.
- [x] **Step 3: Green** + existing CLI tests unbroken (`uv run pytest tests/integration/cli -q`).

### Task 7: Report Tab

**Files:**
- Modify: `src/ux_analyzer/reporting/renderer.py`, `src/ux_analyzer/reporting/templates/experiment.html.j2`, `src/ux_analyzer/reporting/static/report.css`, `src/ux_analyzer/reporting/static/report.js`
- Test: `tests/integration/reporting/test_renderer_redesign.py`

**Interfaces:**
- Renderer: `load_redesign_attempt(root)` bounded (≤8MB) schema-checked load of newest valid attempt; tab context `{attempt_status, audience_inference, proposals[], consistency_notes, pack_version, unavailable_reason}`; absent attempt → placeholder context with reason class.
- Template: new rail tab `Redesign` (always rendered); section `#view-redesign` in Lab Observation Log language — proposals as exhibit-card sheets, impact×effort stamp chips (Flag pair ink — model estimates), category filters and page filter (chips, report.js), deliberate-choice note rendered as an annotation row, per-page understanding intro (intent + inferred audience labeled as model inference), consistency notes block; unavailable/no-proposals states as stamped placeholder rows with reason.
- CSS: reuse stamp chip patterns, Folder panels, Sheet Shadow; no new colors; respects `prefers-reduced-motion`.
- JS: filter interactions only; no network, no model calls.

- [x] **Step 1: Failing renderer tests** — accepted attempt renders proposals ordered impact→effort; unavailable renders stamped placeholder with reason; rejected renders reasons; truncated-capture limitation surfaces; report byte-size threshold still respected. *(in `tests/integration/reporting/test_renderer_redesign.py`, plus corrupt/oversized-attempt and missing-attempt placeholders)*
- [x] **Step 2: Implement** renderer context + template + filters.
- [x] **Step 3: Green**; run full reporting suite (`uv run pytest tests/integration/reporting -q`).

### Task 8: Docs, Env, and Verification Sweep

**Files:**
- Modify: `.env.example`, `README.md`, `docs/model-provider.md`, `docs/architecture.md`
- Test: full non-live suite

- [x] **Step 1: Docs** — env var table entries (`UXA_REDESIGN_ENABLED`, `UXA_REDESIGN_MODEL`, `UXA_REDESIGN_MAX_PAGES`, `UXA_REDESIGN_MAX_PAGE_HEIGHT`, `UXA_REPORT_SYNTHESIS_ENABLED`), command reference row for `uxa redesign`, architecture note on the two-pipeline separation and model-estimate doctrine, model-provider contract section for the two new roles.
- [x] **Step 2: Full sweep** — `uv run uxa validate benchmarks/demo/project.yaml` green; `uv run pytest -m "not live" -q` green per-directory (unit 1297, integration 850, e2e 40) with only the two pre-existing e2e failures (verified identical on HEAD via stash-compare: `test_production_cli_report_is_interactive_and_causal` `#analysis-summary` visibility, `test_every_published_finding_opens_independently_verifiable_playback` Verify-evidence click); `uv run ruff check .` green after a mechanical one-line import-sort fix in `scripts/slop_bench/parity_sweep.py` (pre-existing ruff-version drift in an untouched file); `uv run pyright` cannot complete bare on this machine (>20 min third-party import resolution — earlier plans always ran it scoped to explicit paths, and the only plan that used it bare has an unchecked step), so it was verified scoped: modified files show an identical 256-error baseline on HEAD vs working tree (no new errors); `git diff --check` clean.
- [x] **Step 3: Manual smoke** — served `fixture_app` + a stub model server (scratch tooling in `/tmp`, no repo changes): feature-off `uxa run` renders the Redesign tab with `data-redesign-state="missing"` + "Not generated" stamp + placeholder; direct `uxa redesign <output> --pages <fixture-url>` with the stub produced an **accepted** attempt (proposal, impact×effort chip, per-page understanding, category/page filters, consistency notes, pack version all rendered in report.html). Auto-mode after the fixture experiment is a no-op because fixture application versions have no `start_url` (pre-existing on HEAD — the shared page list resolves only for `kind: live` apps; the gate/pipeline wiring is covered by `test_redesign_command.py`).

**Sweep findings (this session):**
- Fixed a real pipeline gap: `run_redesign_pass` called the proposer without image attachments, so the model never saw the persisted segment screenshots (ADR 0007 / model-provider contract promise "images attached in order with y-offsets"). `application/redesign.py` now materializes capture segments into ordered bounded JPEG `ModelAttachment`s (temp scratch dir per pass, cleaned up; digest-checked by the transport) and passes them to `proposer.analyze`. Covered by a new unit test; transport wiring confirmed on the smoke's wire log (`text,img` parts).
- The prior session's model-provider/synthesis failure signal (byte-boundary + manifest/preflight tests failing in a broad run but passing isolated) did **not** reproduce in two full clean non-live runs this session; classified as test-ordering/env flakiness, not a regression.

**Post-review follow-up (live demo re-litigation):**
- Scroll-position capture is now a first-class mode: `page_capture` shoots the viewport at each scroll offset (viewport-height steps, per-frame settle) instead of slicing one full-page render; `uxa redesign` defaults to it, payloads self-describe `capture_mode`, and ADR 0007 documents both modes. Verified live: 15 segments, and the `is-stuck` header styling appears in frames ≥1 — the model can no longer claim the nav is not persistent (mk-02 premise is false), and scroll-triggered UI shows where it actually appears.
- Deterministic target-size guard in `_validate_final_proposals`: `category: accessibility` hit-target claims (target-tap vocabulary) are rejected when the referenced control's **effective tappable surface** (`tap_box`, see below) is already ≥44px — kills the "Watch with sound" whole-video-card false positive (320×569 in inventory) and the repeated "sitewide tap targets" claims that reference 590×50/711×123 controls. Contrast-type accessibility proposals are not gated.
- The guard measures the **effective surface a finger actually hits**, not the semantic node's painted box: capture installs a listener-tagging init script (`data-uxa-taps` on any element receiving a click-ish listener, injected on standalone captures and the shared audit/report session), and every `button`/`input`/`link` entry carries `tap_box` = the control itself or its closest tap-reacting ancestor (a card whose handler wraps a small child button, a `<label>` wrapping a checkbox); `html`/`body`/`document` delegation roots are excluded. Referrals resolve to the best-matching control (label match wins, else intersection-over-union). Sidecars without `tap_box` are re-captured. The target vocabulary gate now also requires a size judgment, so contrast write-ups that merely mention a tap target pass through.
- Model refs are normalized at the gate (`w`/`h` aliases → `width`/`height`); rejected attempts now persist the proposals that did validate alongside the rejection reasons (observability when a weak model mixes one malformed item into an otherwise valid set).
- Live re-run against the demo site (scroll capture + deepseek-v4.1-flash:free via tokenharbor.ai): the three disputed proposals (sticky-nav persistence, demo-CTA target, work-gallery label) did **not** recur; new findings surface instead (eyebrow contrast, the visually-hidden "Recognition and education" heading, hero role/availability copy, duplicated tech chips), with the pipeline rejecting only the free model's imprecise tap-target claims.

### Milestone 2 (separate work package, same plan): unify impact×effort across UX findings

- Extend synthesis finding schema with `effort` estimate from the adjudicator; deterministic fallback findings emit `effort: unspecified`; renderer renders impact×effort chips for findings (severity remains the impact axis); fix export mirrors the new field (ADR 0006 lockstep); migrate finding tests. Ship only after Milestone 1 is stable.

### Explicitly deferred

- Visual mockup generation (HTML previews / edited screenshots).
- Mobile/tablet viewport and keyboard-order capture.
- Fix Export integration for design proposals.
- Critic-driven per-page second retrieval rounds.
