# Design proposals

`uxa redesign` reads captured pages and produces a set of **design proposals**:
model-generated suggestions for improving a page's structure, visual design,
copy, or accessibility.

A design proposal is a **model estimate**. It carries no verification status, it
does not cite run evidence, and it is never a UX finding. The report keeps
proposals on their own view, separately labeled and visually distinct from
evidence-backed findings. Impact and effort are model estimates too, not
measured values.

The pipeline is deliberately independent of report synthesis. It never reads
the run evidence corpus, and report synthesis never reads redesign output.

```text
shared page-capture.json  ──▶  proposer (per page)  ──▶  critic/merger (cross page)
                                                              │
                                              deterministic validation gate
                                                              │
                                    immutable attempt under <output>/redesign/
                                                              │
                                                    report: Design proposals
```

## Run it

```powershell
uv run uxa redesign reports\exploration-generated
```

Or have it run automatically after a completed experiment:

```powershell
$env:UXA_REDESIGN_ENABLED = "1"
uv run uxa run .uxa-output\explore\example\project.yaml --experiment exploration-run
```

The automatic pass is best-effort: it warns and continues on any failure and
never fails the experiment. If the model environment is incomplete it prints:

```text
warning: redesign enabled but model environment incomplete; skipping
```

## Which pages get captured

There is one shared browser capture pass per page, persisted as
`page-capture.json` beside the experiment output. Both the live-page audit and
the redesign read it, so a page is never fetched twice for the same output
directory.

The page list is resolved deterministically:

1. the application start URLs, in declaration order, normalized
2. then the breadth-first discovery order from the newest finalized exploration
   attempt's crawl corpus (status `succeeded`, `partial`, or `completed`),
   same-site targets only
3. deduplicated on normalized URL
4. truncated to the page cap

With no exploration corpus the list is exactly the start URLs. When you pass
`--pages`, that list replaces resolution entirely (and the cap still applies).
The cap is `--max-pages`, falling back to `UXA_REDESIGN_MAX_PAGES`, falling back
to 10.

`uxa redesign --dry-run` resolves and prints the list without contacting a
model. Useful for confirming coverage before spending anything.

A page already present and fresh in the sidecar is reused without a browser.
Pages absent from the sidecar — and pages captured before effective tap-target
measurement existed — get a dedicated capture pass, and the sidecar is updated
so the next run stays on the shared single-pass path.

## The capture pipeline

Each captured page becomes a `page-capture-v3` document:

| Field | Meaning |
| --- | --- |
| `url`, `title`, `captured_at` | Identity and timestamp. |
| `viewport` | The viewport the capture was produced in (default 1280×800). |
| `capture_mode` | `full-page-slice` or `scroll`. |
| `document_height` | Full page height in CSS px. |
| `captured_height` | How much of it was actually captured. |
| `truncated` | `true` when `document_height` exceeds `captured_height`. |
| `copy_truncated`, `inventory_truncated` | Whether the text or node inventory hit its own bounds. |
| `segments` | Ordered image segments, each a JPEG data URL plus its `y_offset` and `height`. |
| `sections`, `headings`, `forms`, `buttons`, `inputs`, `links`, `paragraphs` | A trimmed node and copy inventory, with measured tap boxes on interactive entries. |

The full-page path slices `[0, min(document_height, max_page_height))` into
2000-pixel segments, so a tall page is a stack of images rather than one
unreadable image.

### Capture limits, and what they hide

| Variable | Default | Effect |
| --- | --- | --- |
| `UXA_REDESIGN_MAX_PAGES` | `10` | Page-list cap for the shared capture pass. |
| `UXA_REDESIGN_MAX_PAGE_HEIGHT` | `12000` | Per-page capture height ceiling in CSS px. |

Unset, unparseable, or non-positive values fall back to the defaults.

**Content below the capture limit is never captured and is never shown to the
model.** A page taller than the cap produces segments only up to the cap, and
the rest of the page does not exist for the redesign roles — not in an image,
not in the inventory, not in the model's input. Any conclusion the model draws
is therefore a statement about the captured region only.

This is never silent. When any captured page was truncated, the report's Design
proposals view states the cap explicitly:

> Model estimates cover only the first 12000px of each page; content below the
> capture limit was not captured and was not shown to the model.

If you are auditing a long page and the proposals feel narrow, raise
`UXA_REDESIGN_MAX_PAGE_HEIGHT` and re-run `uxa redesign`.

## The two model roles

Both roles run in fresh, isolated message tuples. They share only the published
principle pack and the operator audience string. The proposer sees the page
capture; the critic sees the proposer's candidates plus a digest per page
payload, never the raw pages. Neither role ever sees the run evidence corpus,
another role's reasoning, or existing report prose.

### Proposer (per page)

Prompt version `redesign-proposer-v1`. Runs once per captured page, in sorted
URL order. Receives:

- the page's text-only capture view: URL, title, viewport, geometry, truncation
  flags, and the trimmed node and copy inventory
- the segment list as ordered metadata (index, y-offset, height, digest)
- the segment pixels as `image/jpeg` attachments, in y-offset order — the bytes
  travel once, not re-embedded as data URLs in the JSON
- the published principle pack
- your `--audience` string, if you supplied one

Returns, per page: a page understanding (intent, inferred audience, section
relationships) and a set of proposals. The audience is always labeled an
inference, never asserted. The page URL in the result is the capture key, not
whatever the model echoed back.

### Critic/merger (cross page)

Prompt version `redesign-critic-merger-v1`. Runs once, after every proposer
pass. Receives the consolidated per-page candidates, the digests of every page
payload, the principle pack, and the audience. It:

- kills or merges conflicting proposals, recording the reason for each kill
- emits cross-page consistency notes
- returns the final proposal set, which deterministically overrides the
  per-page proposals

### The principle pack

Version `redesign-principles-2026-09`, 18 principles, immutable and static.
Principles are interpretive lenses: they help name or explain an
evidence-backed-looking observation, and they are not evidence. A principle can
never establish that a problem exists or how severe it is.

Proposals must cite principle ids from this pack, and only from this pack.

| Group | Principle ids |
| --- | --- |
| Gestalt | `gestalt-proximity`, `gestalt-similarity`, `gestalt-common-region` |
| Usability heuristics | `nielsen-consistency`, `nielsen-recognition`, `nielsen-aesthetic-minimalist` |
| Accessibility anchors | `wcag-contrast-1.4.3`, `wcag-target-size-2.5.8`, `wcag-labels-3.3.2`, `wcag-reflow-1.4.10`, `wcag-heading-order-1.3.1` |
| Copy and tone | `copy-tone-fit`, `copy-clarity-over-cleverness` |
| Hierarchy and scan | `hierarchy-f-pattern`, `progressive-disclosure`, `whitespace-breathing-room`, `visual-hierarchy-scale`, `structural-reuse-patterns` |

## What a proposal contains

| Field | Meaning |
| --- | --- |
| `proposal_id` | Unique within the attempt. |
| `page_url` | The captured page. |
| `category` | One of nine families (below). |
| `title` | Short name. |
| `observation` | What is on the page now. |
| `rationale` | Why the observation is a problem, in terms of principles. |
| `change` | The concrete change. |
| `principle_ids` | One or more pack ids. Unique. |
| `impact` | `low`, `medium`, or `high` — a model estimate. |
| `effort` | `small`, `medium`, or `large` — a model estimate. |
| `section_refs` | One or more references into the capture: URL, section label, box (`x`, `y`, `width`, `height`), summary. Every ref must target the proposal's own page. |
| `also_affects` | Optional extra pages or areas. |
| `deliberate_choice_check` | Required for four categories; see below. |

The nine categories: `grouping`, `whitespace`, `unification`, `radical-redesign`,
`copy`, `relocation`, `simplification`, `new-section`, `accessibility`.

### The deliberate-choice guard

`grouping`, `unification`, `simplification`, and `relocation` proposals are
rejected unless they carry a `deliberate_choice_check`: the potentially
intentional pattern, plus why the proposal stands anyway. Conversely the field
must be absent in the other five categories. Intentional design should not be
reported as an accident.

## Deterministic validation

The critic's output is not published as-is. Every surviving proposal passes a
validation gate, and a failure is recorded with its reason rather than repaired.

| Check | Failure reason recorded |
| --- | --- |
| `principle_ids` is a list | `<id>: principle_ids must be a list` |
| Every principle id exists in the pack | `<id>: unknown principle ids: [...]` |
| A persisted capture exists for `page_url` | `<id>: no persisted capture for <url>` |
| `section_refs` is a list | `<id>: section_refs must be a list` |
| Every section reference resolves | `<id>: dangling section reference on <url>` |
| A hit-target claim does not contradict a control already ≥ 44 px | `<id>: <target-size reason>` |
| Schema invariants hold | `<id>: <ValueError message>` |
| At most 12 valid proposals per page | `<url>: N proposals exceed the per-page bound of 12` |
| At most 40 valid proposals in total | `N proposals exceed the total bound of 40` |

A section reference resolves when its label appears in the capture's inventory
labels or its box overlaps an inventoried element box, and its vertical extent
stays within the captured document height (with a 2 px tolerance). A ref that
overlaps a real control but extends past the bottom of what was captured is a
dangling ref.

The 44 px rule is a guard against a specific failure: a model claiming a whole
video card, or a large card wrapper, needs a bigger tap target when the capture
already measured the interactive control inside it at or above the WCAG minimum.
Only proposals whose own text asks for a larger tappable surface are gated;
color-contrast and other accessibility proposals pass through.

Only successfully validated proposals count toward the per-page and total
bounds, so a malformed entry can never push a page's valid set over the limit.

Surviving proposals are published in a deterministic order: impact
(`high` → `medium` → `low`), then effort (`small` → `medium` → `large`), then
`proposal_id`. The same ordering drives the report's default proposal order.

## Attempt statuses

| Status | Meaning |
| --- | --- |
| `accepted` | Proposals passed the gate and are published. |
| `no-proposals` | The pass completed and no proposal survived. A valid result, not a failure. |
| `rejected` | At least one validation failure. The reasons are recorded, and the proposals that did pass are kept on the record. |
| `unavailable` | The pass could not complete: no captures, a proposer or critic failure, an unusable role response, or a page payload over the JSON budget. The reason is recorded. |

A `rejected` attempt is not an empty one. Each surviving item is fully
validated, so a weak model mixing one malformed proposal into an otherwise good
set still leaves the good proposals readable. Read `rejection_reasons` next to
`proposals` when you see this status — the dropped items are named, not
silently missing.

An `unavailable` reason is a real, named failure. The ones you will see:

| Reason | What it means |
| --- | --- |
| `no page captures available` | Nothing was captured. |
| `proposer failed for <url>: ...` | The per-page role failed on one page. |
| `proposer returned an unusable response for <url>` | The response was structurally wrong. |
| `critic/merger failed: ...` | The cross-page role failed. |
| `critic/merger returned an unusable response` | The response was structurally wrong. |
| `page payload for <url> exceeds the redesign JSON budget; segment pixels must ride as image attachments` | A page's text payload was over 1,000,000 bytes. |

The report prefers content over recency: it shows the newest `accepted` attempt
and falls back to the newest valid attempt only when nothing was ever accepted,
so a later outage does not erase an earlier good set. Attempts that are corrupt
or whose `payload.json` does not match the `payload_sha256` in `index.json` are
skipped with a note, and the next older attempt is tried.

## The immutable attempt layout

```text
<output>/redesign/
  redesign-20260912T101500Z-3f9a1c8e/
    index.json
    payload.json
    model-calls.json
```

Attempt ids are `redesign-<YYYYMMDDTHHMMSSZ>-<first 8 hex of a uuid4>`, which
sorts chronologically. A directory that already exists is never overwritten.

| File | Schema | Contents |
| --- | --- | --- |
| `index.json` | `redesign-index-v1` | `attempt_id`, `status`, `created_at`, `payload_sha256`. |
| `payload.json` | `redesign-attempt-payload-v1` | `captures_digest` binding the proposal set to the exact capture it read, `pack_version`, `proposals`, `killed`, `page_understanding`, `consistency_notes`, `rejection_reasons`, `audience`, `status`, `unavailable_reason`, `created_at`. |
| `model-calls.json` | `redesign-model-calls-v1` | Sanitized transport audit: role, model, endpoint origin, schema version, attempts, latency, queue wait, response mode, token usage, retries, failure reason and provider diagnostics. |

`model-calls.json` never stores request or response payloads, so it stays safe
to keep next to a report. A failed attempt is still published with its reason
and still gets its transport records, so a terminal model failure stays
debuggable.

Publication failures are warnings, not errors:

```text
warning: redesign attempt not persisted: <reason>
warning: could not persist redesign model-calls.json: <reason>
```

## Reading the report view

The report's **Design proposals** view states the framing before it lists
anything: proposals are model estimates, they are not run-evidence findings,
and impact and effort are model estimates rather than measured values. When
capture was truncated, the cap disclosure appears directly underneath.

When the attempt is `accepted`, the view shows the per-page understanding
(intent, inferred audience, section relationships), then the proposal cards
with filters by category and by page. Each card shows observation, rationale,
change, cited principle ids, and — where the capture carried a verified
locator — the recorded sections with a copy-locator affordance. Cards for the
four deliberate-choice categories carry the deliberate-choice note, and cards
with cross-page effects list them.

The other three attempt statuses get their own labeled header and a placeholder
instead of cards, so an empty or rejected attempt is never mistaken for a page
with nothing to improve:

| Attempt status | Header shown | Placeholder |
| --- | --- | --- |
| `accepted` | Model estimates | The proposal cards. |
| `no-proposals` | No proposals | `The redesign pass completed but the model proposed no changes.` |
| `unavailable` | Unavailable | The recorded `unavailable_reason`, or a generic message. |
| `rejected` | Rejected | `The redesign pass was rejected during validation.` plus the recorded rejection reasons. |

Consistency notes survive the `no-proposals`, `unavailable`, and `rejected`
states and still render. A `rejected` attempt's surviving proposals stay in
`payload.json` but are not shown as cards; the rejection reasons are what the
report surfaces. Read `payload.json` when you want the full set.

## Troubleshooting this pipeline

| Symptom | See |
| --- | --- |
| `no page captures available: provide --pages or run an experiment or exploration first` | Run an experiment or `uxa explore` first, or pass `--pages`. |
| `model environment error: missing model environment variables: ...` | A required transport variable is unset. |
| `model environment error: redesign_model or report_model is required for redesign roles` | Neither `UXA_REDESIGN_MODEL` nor its `UXA_REPORT_MODEL` fallback is set. |
| `redesign failed: capture unavailable: <type>: <message>` | The page could not be captured. Check the URL, and that the origin is reachable. |
| Attempt `unavailable` with a proposer failure | The per-page role call failed. Read `model-calls.json` in that attempt directory. |
| Attempt `rejected` | Read `rejection_reasons` in `payload.json`; surviving proposals are still in `proposals`. |
| Proposals seem to ignore content lower on the page | Raise `UXA_REDESIGN_MAX_PAGE_HEIGHT`. |

More general failure modes are covered in [troubleshooting](troubleshooting.md).
