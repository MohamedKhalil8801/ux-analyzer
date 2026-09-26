# Troubleshooting

Concrete failure modes, what they actually mean, and what to do about them.
Error text below is quoted from the tool.

## Inspecting a run first

Before anything else:

```powershell
uv run uxa inspect-run reports\<id>\runs\<run-id>
```

```text
run: run-0f1c9a2b4e7d...
outcome: verification-failed
verified: False
claimed: True
artifacts:
- artifacts/6b1f...png
- ...
```

`outcome` is the closed terminal outcome, `verified` is the independent
verifier's answer, and `claimed` is what the model said. A mismatch between
`verified` and `claimed` is a real finding about the interface's feedback, not a
tool bug.

A non-existent path exits with `run bundle does not exist: <path>`.

Then read `result.json` and `timeline.jsonl` in the same directory. Every claim
in the report resolves to something in those two files.

## Model configuration

### `model environment error: missing model environment variables: ...`

The named variables are not set, or are set to an empty string. The list tells
you exactly which. In `api` mode the required names are `UXA_LLM_BASE_URL`,
`UXA_LLM_API_KEY`, `UXA_SCENT_MODEL`, and `UXA_COGNITIVE_MODEL`, plus
`UXA_REPORT_MODEL` when report synthesis is enabled. In `codex` mode the base URL
and key are not required.

Check without printing values:

```powershell
uv run uxa validate <project> --check-env
```

### `UXA_LLM_MODE must be one of: api, codex`

`UXA_LLM_MODE` is set to something else. Use `api` or `codex`.

### `UXA_LLM_BASE_URL must be an HTTP(S) URL`

The base URL has no scheme, an unsupported scheme, or no host. It must start
with `http://` or `https://`. The adapter appends `/chat/completions` itself, so
point it at the API root, e.g. `https://provider.example/v1`.

### `UXA_LLM_BASE_URL must not contain credentials`

The URL embeds a username or password. Move the credential to
`UXA_LLM_API_KEY`; the base URL must be credential-free.

### `timeout_seconds must be greater than zero` / `max_concurrent_calls must be greater than zero`

`UXA_LLM_TIMEOUT_SECONDS` or `UXA_LLM_MAX_CONCURRENT_CALLS` is set to zero or a
negative number. To remove the model-call timeout entirely, set the timeout to
the literal string `none` (or `off`, or `unlimited`).

### `<name> must be one of: none, minimal, low, medium, high, xhigh, max`

A reasoning-effort variable holds an unrecognized value. The name in the message
identifies which one — `scent_reasoning_effort`, `cognitive_reasoning_effort`,
or `report_reasoning_effort`. Valid values are `none`, `minimal`, `low`,
`medium`, `high`, `xhigh`, `max`.

### `timeout_seconds must be numeric` / `max_concurrent_calls must be numeric`

`UXA_LLM_TIMEOUT_SECONDS` or `UXA_LLM_MAX_CONCURRENT_CALLS` is set to something
that is not a number. `UXA_LLM_TIMEOUT_SECONDS` also accepts the literal strings
`none`, `off`, and `unlimited` to disable the model-call timeout.

### Report synthesis is enabled but no report model is set

`uxa synthesize` and `uxa validate --check-env` both require `UXA_REPORT_MODEL`
when the gate is on, and fail with:

```text
model environment error: missing model environment variables: UXA_REPORT_MODEL
```

`uxa run` behaves differently on purpose: synthesis is best-effort, so the runs
still finalize, the deterministic report still renders, and you get:

```text
warning: report synthesis unavailable; deterministic findings retained
(ModelConfigurationError)
```

Fix it by setting `UXA_REPORT_MODEL`, or turn the gate off for this run with
`uxa run --no-synthesis`, or set
`evaluation.report_synthesis.enabled: false` in the project.

### `model environment error: redesign_model or report_model is required for redesign roles`

`uxa redesign` has no separate required-name check, so it fails at the first
model call instead. Set `UXA_REDESIGN_MODEL`, or `UXA_REPORT_MODEL` as its
fallback.

### `codex` mode fails immediately

Codex mode shells out to the `codex` executable. It must be installed, on
`PATH`, and already logged in. Codex mode never reads or prints account
credentials, so a login problem has to be fixed in the CLI itself.

## Playwright or Chromium is missing

Symptom: the command fails the moment it tries to open a browser, with the CLI
prefix `run failed:` (for `uxa run`) or a Playwright error surfacing from
`uxa slop`, `uxa explore`, or `uxa pagespeed --web-ui`. `uv run` installs the
Playwright library, but not the browser build it drives.

```powershell
uv run playwright install chromium
```

Confirm before a long run:

```powershell
uv run uxa slop https://example.com
```

If that prints a score, the browser works.

## Live targets safety-block on every page

```text
warning: live targets reference non-allowlisted origins (https://cdn.example,
https://fonts.example); these requests will safety-block. Re-run interactively
or pass --yes / --allow-origin.
```

Your target loads resources from origins that are not on the allowlist, and the
browser network policy fails closed. A run that needs those resources ends with
outcome `safety-blocked`.

Re-run in a terminal so the tool can ask you once, or grant the origins up
front:

```powershell
uv run uxa run <project> --experiment <id> --allow-origin https://cdn.example --allow-origin https://fonts.example
```

or accept every referenced origin:

```powershell
uv run uxa run <project> --experiment <id> --yes
```

Prefer listing the origins explicitly in the project's
`allowed_origins` when they are part of the target, so the choice is recorded in
the config digest rather than in a shell flag.

## The run exits 1 even though the report was written

This is intentional. `uxa run`, `uxa run-one`, and `uxa ablate` exit 1 when the
experiment had any runner failure, evaluator failure, or invalid UX sample. The
report and `experiment.json` are still written, and the run summary says which:

```text
finalized runs: 4; execution failures: 0
UX samples: 3 valid; 1 invalid
outcomes: agent-abandoned=1, verified-success=3
```

`experiment.json` carries the same information in `failures` and `invalid_runs`.
Read those keys rather than re-running.

## A run is marked invalid

An invalid UX sample is one whose evidence cannot support a UX claim. It is
excluded from comparisons and aggregations, and it is listed in
`experiment.json` under `invalid_runs` with a `reason`. Invalidity is never
quiet, and it is never converted into a product claim.

The reasons you will see:

| `reason` prefix | Meaning | What to do |
| --- | --- | --- |
| `evaluation-failure: evaluation evidence unavailable: target label 'X' role 'Y' region 'Z' not found in recorded snapshots` | The scenario's `evaluation_target` never matched anything the run actually recorded. | Fix the scenario, not the product. Correct the label, role, or region in `evaluation_target` so it matches the real control. |
| `saliency-fallback: ...` | The learned prominence provider, its cache, or aggregation failed and the run continued on the heuristic fallback. | Only relevant if you are using a learned provider. Install the runtime and model (`uxa models install`), or accept the run as heuristic-only evidence. |
| `budget-exhausted: ...`, `timed-out: ...`, `verification-failed: ...`, `model-failure: ...`, `safety-blocked: ...`, `provider-failure: ...`, `internal-error: ...` | The run ended in a terminal outcome that carries no usable UX signal. | Fix the cause, then re-run the cell. |

A `budget-exhausted` run is **not** automatically invalid. It stays a valid
sample when the run ended because a human-scale budget ran out, with the reason
set to one of: `attention budget exhausted`, `step budget exhausted`,
`observation budget exhausted`, `interaction budget exhausted`, `action budget
exhausted`, `human attention budget exhausted`, or `human action budget
exhausted`. Those describe a persona hitting a real limit, which is a result.

### The evaluation-target case in detail

This is the most common structural problem, and it is a scenario defect, not a
product bug. The scenario declares the control the task is about:

```yaml
evaluation_target:
  labels_by_version:
    live: Invite teammate
  role: button
```

The evaluator resolves that label, role, and region against the run's recorded
snapshots only. If no recorded element matches — because the label is wrong, the
role is wrong, the control is in a region the run never captured, or the control
simply does not exist on any reachable page — evaluation raises
`EvaluationEvidenceUnavailable`, the run is marked
`ux_sample_valid: false`, and the whole experiment exits 1.

The fix is to correct the scenario. Change the label to the text the page
actually renders, drop the `role` if the control is not that role, or drop the
`region_label` if the control is not in that region. Widening the crawl or the
budget does not help if the label is wrong.

`uxa explore` already guards against this for generated scenarios: a proposal
whose verifier text, evaluation target label, or region never appears in the
crawl corpus is rejected during synthesis with a reason code such as
`verifier-anchor-unavailable`, `evaluation-target-label-unavailable`, or
`evaluation-target-region-unavailable`. A hand-written scenario has no such
guard, which is why this failure is worth checking for first.

## Report synthesis: unavailable versus rejected

Four statuses, and the difference matters.

| Status | Meaning |
| --- | --- |
| `accepted` | Findings passed validation and were published. |
| `no-issues` | The pass completed and established no issue. A valid result. |
| `unavailable` | The pass could not run or could not complete. |
| `rejected` | The pass ran, but the publication contract refused the output. |

### `unavailable` causes

- A role's model is not configured. The recorded limitation is
  `Report synthesis is unavailable because role configuration is incomplete.`
- The model transport failed. The attempt records the failure; the report keeps
  its deterministic findings.
- A role produced no usable output, or returned an unusable response.
- The report model does not accept image input while the evidence corpus
  contains screenshots or heatmaps.

During `uxa run`, an unavailable synthesis is a warning, not a failure:

```text
warning: report synthesis unavailable; deterministic findings retained
(<ErrorType>)
```

The experiment still finalizes, and the report still renders from deterministic
findings and replay. `uxa synthesize` on its own exits non-zero:

```text
synthesis failed: <ErrorType>: <message>
```

### `rejected` causes

- A role's completion receipts are incomplete, or do not cover all four roles:
  `Publication validation requires validated completion receipts for all four
  synthesis roles.`
- Duplicate finding ids: `Publication validation rejected duplicate finding IDs.`
- A candidate was published without clearing its blocking objections.
- A candidate's model was returned but not every candidate was established as a
  valid benign result.
- One of the four roles produced structurally invalid output that a bounded
  retry could not fix.

A rejected attempt is retained and readable. It cannot become the selected
report synthesis, so the report falls back to deterministic findings.

### Where to look

```powershell
Get-Content reports\<id>\synthesis\index.json
```

`index.json` lists every attempt with its id, creation time, status, corpus
digest, synthesis digest, and corpus-manifest digest, plus the
`accepted_attempt_id` pointer. Read `synthesis.json` in the attempt directory
for the `limitations` array and the role manifests — that is where the specific
reason lives.

Regenerate a fresh attempt from the same finalized runs at any time:

```powershell
uv run uxa synthesize <project> --experiment <id> --output reports\<id>
```

Attempts are immutable, so this publishes a new attempt rather than rewriting
the old one.

## Design-proposal attempt rejected

A rejected redesign attempt means at least one proposal failed deterministic
validation. Read the two fields side by side in `payload.json`:

```powershell
Get-Content reports\<id>\redesign\<attempt-id>\payload.json |
  ConvertFrom-Json | Select-Object status, rejection_reasons,
    @{n='proposals';e={$_.proposals.Count}}
```

**A single dropped proposal does not mean the others are missing from the
record.** A rejected attempt keeps every proposal that passed the gate, fully
validated, in `proposals`. Only the failures are in `rejection_reasons`. So this
is a normal, healthy result:

```text
status: rejected
proposals: 7
rejection_reasons:
  - p-04: unknown principle ids: ['wcag-focus-visible-2.4.7']
  - p-09: dangling section reference on https://example.com/pricing
```

Two of nine proposals were dropped for named reasons; seven are published and
readable in `payload.json`.

Be aware of what the report does and does not do with that. A `rejected` attempt
is rendered as a placeholder reading `The redesign pass was rejected during
validation.` plus the rejection reasons — the seven surviving proposals are not
drawn as cards. Only an `accepted` attempt gets cards. So if you want the seven
in the report, the practical move is to fix what the model got wrong (tighten
the audience hint with `--audience`, or restrict `--pages` to pages whose
captures are complete) and re-run, rather than trying to make the rejected set
render.

The reason strings you will see:

| Reason | What it means |
| --- | --- |
| `<id>: unknown principle ids: [...]` | The model cited a principle outside the pack. The pack has 18 fixed ids; see [redesign](redesign.md#the-principle-pack). |
| `<id>: dangling section reference on <url>` | The proposal pointed at a section that is not in the persisted capture — usually content below the capture limit, or a renamed section. |
| `<id>: no persisted capture for <url>` | The proposal named a page that was not captured. |
| `<id>: section_refs must be a list` / `principle_ids must be a list` | The model returned a structurally wrong shape. |
| `<id>: <ValueError message>` | A schema invariant failed, e.g. a missing `deliberate_choice_check` on a `grouping` proposal, or a section ref targeting a different page. |
| `<url>: N proposals exceed the per-page bound of 12` | One page produced too many valid proposals. |
| `N proposals exceed the total bound of 40` | The whole attempt produced too many. |
| `<id>: <target-size reason>` | An accessibility proposal asked for a bigger tap target on a control the capture already measured at or above 44 px. |

## Page capture is truncated

The report's Design proposals view says so explicitly:

> Model estimates cover only the first 12000px of each page; content below the
> capture limit was not captured and was not shown to the model.

**That content does not exist for the model.** It was not captured, so it was
not in the image segments, not in the node inventory, and not in the model input.
Any proposal about it would have been unfounded.

Raise the cap and re-run:

```powershell
$env:UXA_REDESIGN_MAX_PAGE_HEIGHT = "24000"
uv run uxa redesign reports\<id>
```

Unset, unparseable, and non-positive values all fall back to 12000. The
corresponding fields in `page-capture.json` are `document_height`,
`captured_height`, and `truncated`.

The same cap governs the shared page list: `UXA_REDESIGN_MAX_PAGES` (default
`10`) bounds how many pages the audit and redesign capture between them. If a
page you expected is simply missing, check the cap before the height.

## `uxa redesign` cannot find pages

```text
no page captures available: provide --pages or run an experiment or
exploration first
```

Nothing has been captured into that output directory yet. Run an experiment or
`uxa explore` into the same `--output`, or name the pages yourself:

```powershell
uv run uxa redesign reports\<id> --pages https://example.com/pricing
uv run uxa redesign reports\<id> --dry-run
```

`--dry-run` shows the resolved page list without contacting a model, which is
the fastest way to see whether resolution or the cap is the problem.

```text
redesign failed: capture unavailable: <ErrorType>: <message>
```

means the page list resolved but a page could not be captured. Check the URL and
that the origin is reachable.

## `uxa explore` problems

### `starting-url is required (provide --starting-url or set exploration.start_urls in project)`

Neither the flag nor the YAML supplied one. HTTPS is required; an `http://`
starting URL is rejected with `invalid starting-url '<url>': URL must use HTTPS`.

### `max_pages must be >= number of start URLs when depth is 0`

At depth 0 the crawler captures only the starting URLs, so the page cap cannot be
lower than the number of them. Raise `--max-pages` or lower `--depth`.

### `synthesis failed: <ErrorType>: <message>`

The synthesizer could not complete. A real `unavailable` attempt is persisted
with the reason before the command exits, so the crawl evidence is not lost. On a
rerun with `--resume`, the checkpointed corpus is reused and only synthesis is
retried.

### `synthesis produced 0 scenarios — the crawl had N page(s) with sparse visible labels`

Not an error. The pages had too little visible text to anchor a task on. Try
`--depth 1` or a deeper crawl, lower `--starting-url` to a content-rich page, or
add a scenario by hand in the review UI.

Also check your cap: the synthesizer only reads the first 25 corpus pages, so a
very wide crawl past that point adds crawl time without adding candidates.

### `curated set is empty; nothing to materialize or run`

You saved with nothing selected. Accept at least one scenario in the review UI,
or use `--auto-accept`.

### `review UI did not return curated set (aborted)`

The review server shut down without a curation submission — usually a closed tab
or Ctrl+C. Re-run; the checkpointed corpus is reused.

### `suggestions_signature mismatch: the suggestion set is stale or was tampered with`

A browser tab from an earlier session submitted against a different suggestion
set. Reload the review page and curate again. This check is deliberate; do not
work around it.

### `resume requested but the existing checkpoint does not match current exploration settings`

You changed starting URLs, depth, page cap, scenario cap, or settle time while a
checkpoint from a previous run exists. Start a fresh workspace with a new
`--output`, or rerun with the original settings.

### `exploration attempt already exists: ...`

An attempt directory with that id is already on disk. Attempt directories are
immutable and never overwritten, so this should not happen in normal use; if it
does, the output directory has inconsistent state and you should inspect
`<output>/exploration/index.json`.

## `uxa run` problems

### `checkpoint error: experiment checkpoint already exists; use --resume or a new output`

`experiment-progress.json` is already in the output directory from a previous
attempt. Either resume or use a fresh `--output`:

```powershell
uv run uxa run <project> --experiment <id> --resume
```

### `unknown experiment '<id>'; available experiments: <list>`

The `--experiment` default is `core-pair`. Use one of the ids your project
defines, or pass `--experiment`.

### `cannot expand experiment '<id>': <detail>`

The experiment's axes did not expand into a valid matrix — for example a scenario
that lists no eligible version for a selected version. Fix the experiment or the
scenario.

### `unsupported policy '<value>'; choose one of: ...`

`--policy` on `uxa ablate` got a value outside the four allowed policies.

### `run-count must be greater than zero` / `workers must be greater than zero`

Both are validated before anything starts.

### The report says some runs failed

By design. Runner failures, evaluator failures, and staging `crash.marker`
records all appear with safe identity, stage, terminal state, reason, and
whatever timeline evidence exists. They are never dropped. Check
`experiment.json` → `failures` for the list.

## Report rendering problems

### `report failed: bundle root does not exist: <path>`

The path you gave is not a directory. Pass the experiment output directory, its
`runs/` directory, or a single run bundle.

### `report failed: no run or failure evidence found under <path>`

The directory exists but has no run bundles and no `experiment.json` failures to
render. Check that the run actually got as far as finalizing, and that you are
pointing at the right output directory.

### The Page checks and Page speed views are empty

They refetch `ux-audit.json` and `pagespeed.json` at page load, and browsers
block that fetch on `file://` URLs. Serve the directory instead:

```powershell
uv run uxa report reports\<id> --serve --port 8000
```

The server binds `127.0.0.1` only. If the sidecars were never written, look for
the run-time warnings — the audit and PageSpeed passes are best-effort and warn
rather than fail, so a missing `pagespeed.json` usually means the API call
failed or `UXA_SKIP_PAGESPEED` was set.

### The report split into many pages

Reports larger than 4,000,000 encoded bytes become an index plus one page per
run under `<output-stem>-runs/`. That is the intended behavior for a large
experiment, not a failure. Set `--output` to choose the stem.

### A run is missing from the report

Check the run actually finalized under `runs/`, then check `experiment.json` →
`failures`. A run whose bundle is still in `.staging/` did not finalize. If it is
there, re-run with `--resume` to archive it and complete the cell.

### `uxa export` says `export package already exists: <dir>`

The package directory must not exist. Use a fresh `--out`.

### `uxa export` says `nothing to export: select issues or pass --all`

Non-interactive selection needs `--all` or at least one `--finding`. In an
interactive terminal you get a selection UI instead and the flags are ignored —
space toggles, arrow keys move, Enter continues.

### `uxa export` says `unknown finding id(s) [...]`

Finding ids are namespaced by source: `<run-id>:<category>` for deterministic
rule findings, `audit:<check-id>`, `slop:<page-slug>`,
`pagespeed:<strategy>:<audit-id>`, `redesign:<proposal-id>`, and the
adjudicator's own id for report-synthesis findings. The error lists the valid
ids for that report.

### `uxa export` says `nothing to export: the report contains no issues`

The report has no findings. That is either a correct "no issues" result or a run
that never got far enough to produce any. Check `experiment.json` for failures
first.

### `uxa export` warns about unavailable evidence

```text
warning: issue 'audit:tap-target-size' has evidence unavailable; it is exported
with 'evidence unavailable' markers.
```

The export still succeeds. The issue's markdown marks the missing references
rather than dropping the issue. This usually means the referenced artifact or
replay position is not resolvable in the current bundle.

## Trace bundles are too large

`traces/<run-id>.zip` includes screencast frames, which is what makes replay
video work. Turn them off to shrink traces substantially — snapshots and network
records are kept either way:

```powershell
$env:UXA_TRACE_SCREENCAST = "0"
```

## Live endpoint tests

Tests marked `live` stay skipped unless you opt in, and the CLI deliberately
removes `UXA_RUN_LIVE_TESTS` from the environment after loading `.env`, so a
line in `.env` cannot enable them:

```powershell
$env:UXA_RUN_LIVE_TESTS = "1"
uv run pytest -m live -q
```

Treat this as external data transfer: it makes real calls to your configured
provider.

## Getting more detail

| You want | Look at |
| --- | --- |
| One run's outcome and artifacts | `uv run uxa inspect-run reports\<id>\runs\<run-id>` |
| One run's event stream | `runs/<run-id>/timeline.jsonl` |
| One run's final state and metrics | `runs/<run-id>/result.json` |
| What was proven reproducible | `runs/<run-id>/manifest.json` and `checksums.sha256` |
| Experiment-level failures and invalid samples | `experiment.json` → `failures`, `invalid_runs` |
| Why synthesis failed | `synthesis/<attempt-id>/synthesis.json` → `limitations` |
| Why a design proposal was dropped | `redesign/<attempt-id>/payload.json` → `rejection_reasons` |
| What the crawl actually saw | `exploration/attempts/<id>/corpus.json` |
| What reached the model, sanitized | `runs/<run-id>/timeline.jsonl`, events of kind `model-call-recorded` |
