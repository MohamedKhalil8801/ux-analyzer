# Report Review Fixes Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every defect found while reviewing `reports/exploration-generated`, so a regenerated report has no false success badges, no swapped hierarchy evidence, no invented scenario names, no unlabelled controls reaching the persona, and no silently-dropped aggregate rows.

**Architecture:** Four independent fixes across four layers, each pinned by an already-written red test. The persona-visibility fix changes only *what the observation layer carries*, never what the model is told about code-only affordances; the exploratory-verification fix adds an evidence path that uses only what a sighted user can see. The synthesis fix adds a fail-closed name check at the corpus boundary. The hierarchy fix corrects role assignment and ancestry-aware guards. The report fix makes rendered text agree with the computed status class.

**Tech Stack:** Python 3.12, dataclasses + Pydantic v2, Jinja2, pytest, Playwright, Typer.

## Global Constraints

- **No code-only hints to the persona.** The persona must see exactly what a sighted user sees. Do NOT inject `aria-label`, `data-*`, `title`, `placeholder`, `alt`, or DOM selectors into any prompt-facing representation. Element identity must come from perceived role, position, size, and icon appearance.
- **Fail closed, never fail silently.** A name that cannot be verified against recorded evidence must be rejected and audited, not shipped. A run whose evaluation fails must never render as success.
- **Do not weaken existing guards to make a test pass.** If a red test conflicts with a deliberate guard, fix the test's premise, not the guard.
- **Keep `uxa report` offline and model-free.**
- Every change must keep `tests/unit tests/integration` green apart from the pre-existing failures unrelated to this work.

---

## Existing red tests (already written, currently failing)

`tests/unit/test_report_review_regressions.py`
- `test_icon_only_control_keeps_a_label_when_rendered_text_is_empty_string`
- `test_icon_only_control_distinguishes_absent_text_from_empty_text`
- `test_synthesized_scenario_region_label_must_exist_in_corpus`
- `test_synthesized_scenario_labels_must_exist_in_corpus`
- `test_curated_scenario_with_unsupported_region_is_reported`
- `test_inverted_emphasis_names_the_larger_element_as_primary`
- `test_inverted_emphasis_does_not_fire_on_display_title_and_eyebrow`

`tests/integration/reporting/test_report_review_rendering.py`
- `test_evaluation_failed_run_is_not_labelled_verified_success`
- `test_aggregate_scenario_count_matches_comparison_table`

Two guard tests already pass and must stay passing:
`test_evaluation_failed_run_reports_its_failure_reason`, `test_failed_run_count_matches_experiment_records`.

---

### Task 1: Icon-only controls must reach the persona with a perceivable identity

**Files:**
- Modify: `src/ux_analyzer/domain/interface.py` (`PersonaVisibleElement.from_snapshot`)
- Modify: `src/ux_analyzer/domain/attention.py` (`ProgressiveObservation.from_snapshot`)
- Modify: `src/ux_analyzer/adapters/web/extractor.py` (JS `label` expression + `_rendered_text`)
- Test: `tests/unit/test_report_review_regressions.py`

**Root cause:** `from_snapshot` uses `rendered_text if rendered_text is not None else label`. For an icon-only button `rendered_text == ""`, so the real label is discarded. Additionally `ProgressiveObservation.from_snapshot` *drops* any element whose `rendered_text` is empty or whitespace — so an icon-only control disappears from the progressive observation entirely. The JS already computes `label = rendered || accessible || placeholder || role || 'control'` but stores `renderedText: rendered`, collapsing an icon-only control to `""`.

**Design — satisfiable without leaking code-only hints:**
A sighted user recognises an icon-only button by its *role* plus *where it sits* plus *what the icon looks like*. Encode exactly that, and deliberately exclude the author-supplied accessible name:

1. In the JS, keep `label = rendered || accessible || ...` for the private `ElementSnapshot.label` (used by evaluation and audit, which may legitimately know the author's name).
2. Add an explicit private field on the raw element recording whether text was genuinely rendered, so Python can tell "no text rendered" from "text rendered as empty".
3. In `from_snapshot`, compute the persona label with a documented precedence that never returns empty for an actionable control:
   - rendered text (when non-empty) — what a sighted user reads;
   - else, for an icon-bearing actionable control, a role-and-position identity string (e.g. `button in header, icon only, 44×44 at top-right`) — what a sighted user perceives;
   - else the snapshot label as a last resort.
4. Drop the `rendered_text` empty-string filter in `ProgressiveObservation.from_snapshot` so icon-only controls are *visible to the persona at all*. Fix the filter to test actionable-ness rather than text presence.

**Explicitly out of scope:** deciding whether the icon *means* "toggle theme". That is the model's job and belongs to Task 2.

- [ ] **Step 1:** Add a failing test asserting an icon-only actionable control appears in `ProgressiveObservation.from_snapshot` output.
- [ ] **Step 2:** Run the unit file; confirm the new test fails for the right reason.
- [ ] **Step 3:** Thread the "text genuinely rendered" signal from JS through `_rendered_text` / the raw-element dataclass into `ElementSnapshot`.
- [ ] **Step 4:** Implement the precedence chain in `from_snapshot`; assert non-empty for actionable controls and keep it *identical* for `""` and `None`.
- [ ] **Step 5:** Relax the empty-text filter in `ProgressiveObservation.from_snapshot`.
- [ ] **Step 6:** Grep the codebase for any other `rendered_text is not None` / `.strip()` gate that silently drops elements, and fix each.
- [ ] **Step 7:** Run `tests/unit/test_report_review_regressions.py tests/unit/domain tests/unit/adapters -q`.

---

### Task 2: Exploratory theme-verification must use perceivable evidence, not self-claims

**Files:**
- Modify: `src/ux_analyzer/adapters/web/verifier.py`
- Modify: `src/ux_analyzer/application/run_agent.py` (claim/verification gating)
- Test: `tests/unit/adapters/web/test_verifier_exploratory.py` (new)

**Root cause:** In the reviewed run the agent clicked the theme toggle, the verifier correctly returned `verified: false` ("persona-visible result not found"), the model nonetheless recorded success off `target_engaged`, and the run then looped until the budget died. The verifier only matches *text*. A theme change produces no new text, so a text-only verifier can never confirm it.

**Design:** Add a verifier capability that recognises **perceivable state change** rather than text, with strict bounds:

- Compare the verification capture against the run's baseline capture on signals a sighted user could genuinely notice: CSS custom-property / computed background and foreground colour of the document root and body, and the dominant background colour of the viewport.
- Require a *material* delta on at least one signal (a documented threshold, not "any change"), so hover flicker, focus rings and anti-aliasing cannot pass it.
- The scenario must opt in — this changes verifier semantics, so it needs an explicit spec field (e.g. `type: "visible-state-change"`) and must not alter behaviour for existing `visible-result` scenarios.
- Record the observed before/after values in `VerificationResult` evidence, so a reviewer can audit why it passed.

- [ ] **Step 1:** Write failing tests: (a) a material root/body colour change verifies; (b) a hover-only or sub-threshold change does not; (c) an unchanged capture does not; (d) evidence carries before/after values.
- [ ] **Step 2:** Run; confirm red for the right reasons.
- [ ] **Step 3:** Implement the capture of perceivable signals in the web operator (document root + body computed colours, dominant viewport background), without exposing any code-only attribute to the persona.
- [ ] **Step 4:** Implement `visible-state-change` verification with the threshold and evidence payload.
- [ ] **Step 5:** Gate `agent_claimed_success` / `target_engaged`-derived success so a claim cannot override a negative verification. Add a red test first: a run whose verifier says `verified: false` must never terminate `verified-success`.
- [ ] **Step 6:** Add a termination guard so a repeated identical failed interaction cannot burn the whole budget: after N consecutive failed verifications on the same element, terminate with a *distinct* terminal state instead of looping to budget exhaustion. Add a red test first.
- [ ] **Step 7:** Run `tests/unit/adapters/web tests/unit/application tests/integration/application/test_run_agent.py -q`.

---

### Task 3: Synthesized scenario names must be verified against recorded evidence

**Files:**
- Modify: `src/ux_analyzer/application/exploration_synthesizer.py`
- Modify: `src/ux_analyzer/domain/exploration.py` (`CrawlPage` needs region evidence)
- Modify: `src/ux_analyzer/cli.py` (`_explore_curated_to_full_scenario`, review-UI boundary)
- Test: `tests/unit/test_report_review_regressions.py`, `tests/integration/cli/test_explore_command.py`

**Root cause:** `_verifier_anchor_supported` validates only the verifier text. Nothing validates `evaluation_target.label` or `evaluation_target.region_label`, so `region_label: "Main work showcase"` — a region that never existed — shipped and failed the run at evaluation time.

**Design:** Validate both names at the corpus boundary, symmetrically with the existing verifier check:

- `_evaluation_target_label_supported(label, corpus)` — the label must appear (normalized substring) in a captured page's `visible_elements` or `headings`.
- `_evaluation_target_region_supported(region_label, corpus)` — the region label must match a region recorded during the crawl. `CrawlPage` currently records none, so extend the crawl to persist region labels per page (bounded and truncated like headings), then validate against them.
- Both checks fail closed: when a corpus has no recorded evidence at all, match the existing posture of `_verifier_anchor_supported` (pass, since absence of evidence is not disproof) — but when evidence *exists* and the name is absent, reject with a new audit reason code.
- Reject, do not silently repair: the curated scenario must be dropped from `valid_schemas` with a sanitized audit record naming the offending name kind.
- Surface rejections in the explore command output and in the review UI so a human sees "region 'Main work showcase' was not found in the crawl" *before* running anything.
- Tighten the weak `test_curated_scenario_with_unsupported_region_is_reported` test to assert on real curation output.

- [ ] **Step 1:** Extend `CrawlPage` with bounded region evidence; update crawl persistence and its tests.
- [ ] **Step 2:** Implement `_evaluation_target_label_supported` and `_evaluation_target_region_supported`.
- [ ] **Step 3:** Wire both into `_validate_and_convert` with distinct audit reason codes.
- [ ] **Step 4:** Rewrite `test_curated_scenario_with_unsupported_region_is_reported` to assert on actual curation output.
- [ ] **Step 5:** Add an integration test that a scenario with an invented region is rejected and reported end-to-end.
- [ ] **Step 6:** Run `tests/unit/test_report_review_regressions.py tests/integration/cli/test_explore_command.py -q`.

---

### Task 4: inverted-emphasis must name its elements correctly and respect ancestry

**Files:**
- Modify: `src/ux_analyzer/analysis/visual/hierarchy.py` (`_inverted_emphasis_issues`)
- Test: `tests/unit/test_report_review_regressions.py`, `tests/unit/analysis/visual/test_hierarchy.py`

**Root cause:** `primary` is chosen by DOM order while `secondary` is the later *larger* node, and both names are emitted unchanged — so `primary_font_px=13.44 / secondary_font_px=153.6` labels the hero headline "secondary". The heading guard tests only the flagged node's own tag, so `h1 > span` evades it.

**Design:**
- Rename internally to what the algorithm actually means: a **larger** node (the newer, bigger one) and a **promoted/dominant** node. Either (a) re-derive the emitted semantics so `primary` is genuinely the larger element and rename the issue accordingly, or (b) keep both roles but **swap the emitted values** so the evidence never claims the larger element is secondary. Pick one, document it in the docstring, and make the issue title/description match the chosen vocabulary.
- Make the heading guard **ancestry-aware**: walk up from the flagged node and skip when any ancestor is a heading, mirroring `_is_in_nav_landmark`'s existing ancestor walk.
- Re-verify against the real page: the hero (`p.hero__eyebrow` 13.44px + `h1 > span` 153.6px) must produce **no** issue.

- [ ] **Step 1:** Add a failing unit test for the ancestry-aware guard at `tests/unit/analysis/visual/test_hierarchy.py`.
- [ ] **Step 2:** Make the heading guard ancestry-aware; confirm both red tests now pass.
- [ ] **Step 3:** Fix the emitted semantics so the reported roles cannot contradict the sizes; assert `primary_font_px > secondary_font_px` holds universally, or rename the fields.
- [ ] **Step 4:** Audit any other check in `hierarchy.py` that infers role from DOM order alone; document or fix.
- [ ] **Step 5:** Run `tests/unit/analysis/visual -q`.

---

### Task 5: Report rendering must agree with itself

**Files:**
- Modify: `src/ux_analyzer/reporting/templates/experiment.html.j2`
- Modify: `src/ux_analyzer/reporting/renderer.py`
- Test: `tests/integration/reporting/test_report_review_rendering.py`

**Root cause (a):** The pill renders `{{ row.outcome }}` raw, so a run with `status_class == "status-evaluation"` shows the text `verified-success`. Class and text disagree.

**Design (a):** Derive the pill text from the same derivation as the class. Introduce a single `status_label` computed alongside `status_class` (an evaluation failure reads as an evaluation failure; an untrusted run reads as untrusted) and render that.

**Root cause (b):** `_comparison_rows` filters on `run["trusted"] and run["comparison_valid"] and run["metrics"]`, so a run that is listed in the comparison table can be absent from the aggregate table with no explanation.

**Design (b):** When a run appears in the comparison table but is excluded from aggregation, record why and render an explicit excluded-runs note under the aggregate table naming each excluded scenario and its reason. Never drop a row silently. (An invalid sample legitimately should not be aggregated — the defect is the silence, not the exclusion.)

- [ ] **Step 1:** Add a failing test for the pill text matching a human-readable label for `status-evaluation`.
- [ ] **Step 2:** Implement `status_label` in the renderer; render it in the template.
- [ ] **Step 3:** Add a failing test that an excluded comparison run is named in an explanatory note.
- [ ] **Step 4:** Implement exclusion-reason collection and the template note.
- [ ] **Step 5:** Run `tests/integration/reporting -q`.

---

### Task 6: Regenerate the report and re-review for regressions

**Files:**
- Regenerate: `reports/exploration-generated/`
- Write: `reports/exploration-generated-review-v2.md`

- [ ] **Step 1:** Run the full fast gate: `uv run pytest tests/unit tests/integration -q`; record all failures and confirm each is either newly-fixed or pre-existing-and-unrelated.
- [ ] **Step 2:** Regenerate the report from the same artifacts (`uv run uxa report reports/exploration-generated`) and confirm the rendering fixes appear.
- [ ] **Step 3:** Re-run the exploration so the synthesis fixes take effect, and confirm the invented region label is rejected rather than shipped.
- [ ] **Step 4:** Re-audit the regenerated report against its artifacts, checking every original finding and hunting for regressions introduced by these changes.
- [ ] **Step 5:** Write `reports/exploration-generated-review-v2.md` with a per-finding verdict (fixed / not fixed / new).
