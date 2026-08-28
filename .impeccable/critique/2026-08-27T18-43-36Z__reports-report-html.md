---
target: reports\report.html
total_score: 22
max_score: 40
na_heuristics: 
p0_count: 2
p1_count: 2
timestamp: 2026-08-27T18-43-36Z
slug: reports-report-html
---
# Critique: `reports/report.html`

Method: dual-agent (A: ses_fbba62fb5ffe · B: ses_fbba5f3e3ffe)

## Design Specificity Verdict

The core replay/findings surface is unmistakably authored — the Calibration Notebook system (`.evidence-ref` mono chips, tricolor `.evidence-chip` trio, hatch-textured `.run-row.status-untrusted`, dark `.viewport-stage` with Signal overlays) could not be transplanted to a generic SaaS dashboard. But the page splits into two products: the polished instrument (Analysis → Playback workspace), and two bolt-on audit sections (`#ux-audit`, `#pagespeed`) that speak a different dialect — raw `<pre>` JSON dumps, editorial verdicts ("Crafted, not generated. This page has a point of view."), no evidence-class coloring. The bolt-ons are category-interchangeable scanner output pasted into a precision instrument. And 6,650 of 10,290 lines sit after `</html>` — no authored system permits that.

Deterministic scan: 64 findings, all in the inline `<style>`. Drift concentrates exactly where the dialect splits: the px-based slop-scorecard block (30px/14px/13.5px/12px — nothing in the ramp or DESIGN.md), undocumented .74/.75/.76rem sizes in PSI/audit styles, off-palette literal `#a62e27`, live fallback `var(--bg-alt, rgba(127,127,127,.08))` where `--bg-alt` is undefined. Two `var(--muted, #667)` hits are false positives (fallback dormant). Detector caveat: ran DEGRADED (regex fallback — htmlparser2/css-tree missing), so custom properties and contrast weren't evaluated; 64 is an undercount.

No browser automation tool in session; CLI scan only — no user-visible overlay.

## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | Outcome pills carry no status class — "verified-success" and "model-failure" render identical slate ink |
| 2 | Match System / Real World | 2 | Jargon: "invalid structured output; model-failure", `target-prominence=0.0459842`, `0.3999999999999999`; mojibake "ΓÇö"/"┬╖"; "Show 1 recorded signals" |
| 3 | User Control and Freedom | 3 | Full transport + URL state restore, but overlay selection irreversible; no deselect, no dismissing sticky chip |
| 4 | Consistency and Standards | 1 | Two duplicated `#pagespeed` sections after `</html>` with duplicate IDs; uncolored pills violate Tricolor Evidence Rule; two different "Open replay" behaviors |
| 5 | Error Prevention | 3 | Strong misuse-prevention: disclaimers, "What this evidence cannot show", hatch for untrusted |
| 6 | Recognition Rather Than Recall | 2 | Three vocabularies to memorize; no severity legend anywhere |
| 7 | Flexibility and Efficiency | 2 | Deep links + scenario filter, but no playback keyboard shortcuts, no skip link, no print stylesheet |
| 8 | Aesthetic and Minimalist Design | 2 | Discipline diluted by JSON dumps, 6,650-line duplicated appendix, empty pattern list rendering 40 blank lines |
| 9 | Error Recovery | 2 | Failed runs honestly displayed, reasons untranslated; "Check first" is a dead end with no link onward |
| 10 | Help and Documentation | 2 | Method notes good; no severity scale definition, no evidence-class legend, no "how to verify" walkthrough |
| **Total** | | **22/40** | **Acceptable — significant improvements needed** |

Cognitive load: HIGH (5/8 checklist failures). No single focus; chunking fails (22 ungrouped audit issues, 20-column table); no grouping/severity sort among 6 Criticals; no on-page legend; 5-link nav + 5-control toolbar push decision limits.

## Overall Impression

The instrument inside is genuinely good — the evidence-reference loop actually works (shareable URLs, focus-moving verification, aria-current sync). But the page buries its one real finding behind a collapsed `<details>` in engine prose, rewards verification with the unexplained float `0.4` in a 20-column table, then closes with two copies of a 3,300-line Lighthouse wall after `</html>`. The generator's assembly step is visibly broken, and bolt-on audit sections break the report's own color doctrine. Biggest opportunity: make recorded behavioral evidence the spine of the page; demote imported scanner output to an honest appendix.

## What's Working

1. Evidence honesty is rendered, not claimed — `.evidence-chip` trio (deterministic-fact 9 / model-estimate 15 / unsupported-human-claim 0), lede disclaimer, "What this evidence cannot show".
2. The verification loop is technically real — `.evidence-ref` buttons carry `data-evidence-target`, resolve run/event/element, highlight `[data-viewing-evidence="true"]`, move focus, encode in URL.
3. Progressive disclosure with discipline — `<details>` for verification/aggregates/PSI; 44px targets, sticky headers, prefers-reduced-motion honored.

## Priority Issues

1. **[P0] Document structurally corrupted — two duplicated `#pagespeed` sections after `</html>`** (lines 3636–10286; duplicate `id="pagespeed-title"`). ~65% of file orphaned; landmarks end at playback; duplicate IDs break `aria-labelledby`; generator assembly broken. Fix: emit `#pagespeed` once per URL inside `<main>`, unique ids, dedupe, build test asserting single `</html>`. Command: `/impeccable harden`.
2. **[P0] Outcome pills violate the Tricolor Evidence Rule** — bare `<span class="status-pill">` with no status class; "verified-success" and "model-failure" visually identical. Fix: emit `status-pill status-success`/`status-error` matching `.run-status-banner`. Command: `/impeccable polish`.
3. **[P1] Only recorded finding buried, collapsed, machine prose** — inside collapsed `<details class="fallback-details">` ("Show 1 recorded signals"); "Recorded cause" unparsable dump; 22 static audit issues get priority above it. Fix: expand details when only content; sort/group audit issues critical-first by category; plain-sentence cause + `<details>` for internals; add severity/evidence-class legend. Command: `/impeccable clarify` + `/impeccable layout`.
4. **[P1] Verification lands on unexplained number** — "View comparison" → `discovery-cost` = "0.4" in 20-column table; `#aggregate-table` leaks `0.3999999999999999`. Fix: focused plain-language metric panel or header definition tooltip; round floats. Command: `/impeccable clarify`.
5. **[P2] Bolt-on sections drift + corruption** — mojibake "ΓÇÖ"/"┬╖"; editorial verdict voice; px type scale; off-palette `#a62e27`; undefined `--bg-alt` fallback (detector confirms); empty pattern list renders ~40 blank lines. Fix: fix generator encoding (UTF-8/`&mdash;`); neutral result statement; normalize slop/PSI styles to rem ramp + palette; collapse empty lists. Command: `/impeccable polish` + `/impeccable clarify`.

## Persona Red Flags

**Alex (Power User):** Loves URL state and deep links. Fails: no playback keyboard shortcuts; no sort on 20-column table; `.evidence-nav` can't reach `#ux-audit`; Ctrl+F poisoned by duplicated appendix; hover-select on `.viewport-overlay` hijacks pointer with no undo.

**Sam (Accessibility-Dependent):** Good bones (3px teal focus, role="status", output readout, aria-pressed, reduced-motion). Fails: no skip link in 10k-line document; duplicated `id="pagespeed-title"`; audit/pagespeed outside `<main>`; tabbing through 20 overlays silently re-renders pane; `#current-event-card` has no aria-live; `th` lack `scope="col"`; `.timeline-summary` .65rem below comfortable minimum.

**Margaret (Non-Technical Stakeholder, from PRODUCT.md):** Walk-through failure. h1 lacks app/date; status pill "Recorded evidence available" in warning colors (good or bad?); only real finding hidden behind click in machine prose; 22 issues incl. 6 "Critical" where "No hero/bespoke imagery" outranks real defects, raw JSON `{"found": false}` as evidence; "Check first" dead end; verification reward = `0.4` in 20-column table; ends after `</html>` in two Lighthouse walls. Cannot complete "understand top findings + verify one claim."

## Minor Observations

- Pause glyph is "Ⅱ" — Roman numeral two, not a pause symbol.
- "56884 ms" raw — "57 s" reads faster (latency columns generally).
- `.evidence-nav` label "Recorded signals" doesn't match section h2.
- Two "Open replay" patterns behave differently (`data-open-run` re-renders; `.tested-scope-detail` anchors reload).
- `element_screenshots` base64 duplicated inside JSON `<pre>` blocks — significant bloat.
- Dead code: `.provider-comparison-*` JS/CSS with no matching DOM.
- PSI "Detail" links external — footnote against "offline" claim.
- No print stylesheet.
- Document `<title>` lacks app name and date.

## Questions to Consider

- If synthesis is missing, is a 22-item severity-labeled audit wall the honest alternative — or does labeling scanner output "Critical" manufacture conclusions the instrument didn't earn?
- What is "Critical" for? "No JSON-LD" and "Text fails WCAG contrast" wear the same label — should the report say what the scale measures?
- Who can actually verify "weak-target-prominence" from a screenshot plus the float `0.0459842` — and if the answer is "nobody who receives this file," why is verify the primary CTA?
- The tool exists to produce recorded behavioral evidence — why does that evidence get one collapsed `<details>` while imported Lighthouse output gets 6,650 lines and the final word?
