---
version: 1
slug: "ux-analyzer-reporting-templates-experiment-html-j2"
primary_target: "src/ux_analyzer/reporting/templates/experiment.html.j2"
related_targets: []
---

# Surface brief: experiment report (reports/report.html)

## Scope and mode

The generated static experiment report (`report.html`), built by `src/ux_analyzer/reporting/renderer.py` + `templates/experiment.html.j2` + `static/report.css` (+ `report.js` / `report-index.js`). Read mode with an Operate-style entry: the visitor scans a dashboard, then drills into the detail view that owns their question.

## Audience and job

Non-technical stakeholders receiving a shared file (primary), UX/design teams reviewing run evidence (secondary). Job: read the at-a-glance cards, open the matching view, verify claims via exhibit tags — offline, shareable deep links.

## Information architecture (current)

Single self-contained file, hash-routed views inside one DOM. Five views, no redundancy:

- `#view-overview` (default) — session masthead (roundel + title, no lede), then the **Index sheet** (`#dash-index`): status stamp, synthesis assessment line, fallback notice when applicable, and four folder-tab cards (Recorded signals, Page findings, Performance, Evidence desk). The former Analysis summary section was folded into this sheet; the summary grid is gone — cards carry the numbers.
- `#view-findings`, `#view-audit`, `#view-performance` — one concern each.
- `#view-evidence` — the merged evidence desk: Tested scope roster → section nav → comparison table → saliency gate → provider methods → replay workspace. (Replay was merged into this view; no separate replay tab.)
- Sticky `.view-rail` (5 tabs) switches views synchronously on click; `routeView` in both JS files resolves `#view-*`, anchors, and `?run=` deep links; popstate/evidence-restore is guarded so hash navigation cannot hijack the active view.

## Chosen direction

"The Lab Observation Log" (user-locked, seed fcdbd2ef). The dashboard is the case-file index: manila folder-tab cards on ruled paper, stamps for counts and states, exhibit tags for evidence.

## Constraints

- Fully self-contained single HTML file; no external requests; Public Sans embedded as base64 woff2.
- All functionality preserved: replay workspace JS contract (IDs/classes), comparison tables, deep-link URL state, synthesis statuses, split index/run-pages mode.
- Tests: 133 renderer tests green; e2e flows activate the owning view via its rail tab before interacting with it.

## Unresolved decisions

- Print stylesheet still owed (paper-native world).
- Mojibake + editorial verdict strings live in `ux-audit.json` data, not the builder.
- The explore review UI shares tokens only; its layout is unchanged.
