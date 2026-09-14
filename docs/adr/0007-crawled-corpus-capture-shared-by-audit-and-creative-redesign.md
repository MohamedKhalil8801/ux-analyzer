# Crawled-Corpus Capture Shared by Audit and Creative Redesign

Status: Accepted
Date: 2026-09-11

The Page findings audit historically analyzed only application start URLs while
`uxa explore` crawled up to 200 pages whose corpus (headings, visible labels,
links, screenshot digests only) fed scenario synthesis and nothing else. We
unified both surfaces on one deterministic page list — start URLs first, then
BFS discovery order, deduped, capped via `UXA_REDESIGN_MAX_PAGES` (default 10)
— captured in one shared browser pass per page that persists a versioned
`page-capture.json` sidecar (segmented full-page screenshots, trimmed node
inventory, copy inventory) consumed by the audit, the Redesign tab, and future
consumers.

Creative redesign is a separate pipeline from report synthesis, not an
extension of it: two isolated roles (Proposer per page, then one
Critic/Merger across pages) produce Design Proposals — schema-validated,
page/section-referenced suggestions carrying impact×effort ratings and a
required `deliberate_choice_check` — published as immutable attempts gated by
deterministic validation with a mirrored Redesign Attempt Status. Design
proposals are always model estimates: they are deliberately outside the run
evidence corpus and never cite Evidence References, because their value is
creative interpretation of persisted page captures, not verified task
behavior. Interpretation is bounded by a versioned Redesign Principle Pack
(Gestalt, Nielsen, lawsofux.com, WCAG 2.2 anchors, copy/tone) which may name
and explain but never prove. The same prompt/contract runs over both the
OpenAI-compatible and Codex transports (Codex best-effort, not yet live-
verified); the feature is env-gated (`UXA_REDESIGN_ENABLED`, default off,
dotenv-loaded) with `UXA_REPORT_SYNTHESIS_ENABLED` added for symmetry, and
`uxa report` remains model-free.

## Capture Modes

A page capture is produced in one of two self-describing modes
(`capture_mode` on the payload):

- `full-page-slice` — one full-page render cut into ~2000px JPEG segments
  (the audit's shared pass). Scroll-dependent UI (sticky headers,
  scroll-triggered sidebars, scroll-revealed sections) appears only in its
  initial, unscrolled state.
- `scroll` — the live viewport is shot once per planned scroll offset
  (viewport-height steps, so segments tile the page contiguously); the
  standalone redesign capture defaults to this mode for the same reason the
  demo site's issues were missed: a pinned header repeats in every frame it
  is visible in, and scroll-triggered UI shows up in the segments where it
  actually appears. Each frame settles briefly after scrolling so
  scroll-reveal animations finish before the shot.

Consumers must not claim scroll-dependent UI is missing from a
`full-page-slice` capture, and must treat a header that appears in every
frame of a `scroll` capture as persistent by construction.

## Effective Tap Targets

Every `button`, `input`, and `link` inventory entry carries two boxes:

- `box` — the control's own painted bounds.
- `tap_box` — its *effective tappable surface*: the control itself, or the
  closest ancestor that reacts to user taps (a card div whose click
  handler wraps a small child button, a `<label>` wrapping a checkbox).

To record which ancestors actually react, a capture init script wraps
`EventTarget.prototype.addEventListener` and tags every element that
receives a tap-ish listener (`data-uxa-taps`); the inventory also accepts
semantic controls (`a[href]`, `button`, `[role=...]`) and inline `on*`
attributes as tap-reacting. `html`/`body`/`document` are excluded: a
handler there is usually a delegation root, and treating it as the target
would inflate every control to the whole page. Older sidecars without
`tap_box` on their interactive entries are stale for the redesign consumer
and are re-captured with the instrumentation installed.

The target-size guard validates hit-target (WCAG 2.5.8-like) proposals
against `tap_box`, never against the widget's own painted box: a small
play glyph on a whole-card button is a large target, and a contrast
proposal that merely mentions a tap target in passing is not routed into
the size machinery.

## Considered Options

- Strict evidence-ID grounding for proposals (rejected: forbids exactly the
  creative leaps the feature exists for; conflicts with the model-estimate
  honesty doctrine in a different way — by pretending creative judgment is
  run evidence).
- Folding redesign into the four-role synthesis pipeline (rejected: couples a
  creative pass to findings adjudication, ~2x cost, muddies both).
- Re-capturing pages at redesign time (rejected: duplicate crawling, worse
  reproducibility, standalone command needs live network).
- Impact×effort for redesign only (rejected: two prioritization dialects in
  one report; unified matrix is phased — proposals first, findings next).

## Consequences

- Audit scope changes semantics when an exploration artifact exists: Page
  findings may cover more URLs than start URLs; tests and docs updated.
- Report size budget grows with persisted captures; bounded by the capture
  caps (segment count, `UXA_REDESIGN_MAX_PAGE_HEIGHT` default 12000px,
  per-segment JPEG encoding). Scroll capture emits one segment per
  viewport-height step; full-page slice emits ~2000px segments — the total
  captured pixels are comparable.
- The model sees no more than the persisted capture: no live site access, no
  DOM beyond the trimmed inventory, consistent with the offline report
  doctrine.
- Unifying findings on impact×effort later must migrate synthesis schemas,
  deterministic fallback findings, the report, and the Fix Export in
  lockstep.
