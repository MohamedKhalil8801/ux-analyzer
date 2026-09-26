# Output formats

Everything `uxa` writes lands under an output directory — by default
`reports/<project-id>` for runs, `.uxa-output/explore/<site-slug>` for
exploration. This document is the map of that directory.

## The shape of an experiment output

```text
<output>/
  experiment.json                 # the experiment summary
  experiment-progress.json        # resume state
  report.html                     # the rendered report
  report-runs/                    # only when the report is split
    <run-id>.html
  pagespeed-cache/                # raw PageSpeed Insights responses
  runs/
    <run-id>/
      manifest.json
      timeline.jsonl
      result.json
      checksums.sha256
      artifacts/
        <sha256>
  saliency/                   # only for saliency-backed runs
    <artifact-namespace>/
      1s.npz  3s.npz  7s.npz
      1s-heatmap.png  3s-heatmap.png  7s-heatmap.png
      profiles.json  metadata.json
  traces/
    <run-id>.zip
  .staging/                       # in-flight bundles only
  .interrupted/                   # archived interrupted staging evidence
  .quarantine/                    # invalidated finalized bundles
  synthesis/                      # report-synthesis attempts
  exploration/                    # exploration attempts and checkpoints
  redesign/                       # design-proposal attempts
  page-capture.json               # shared page capture
  ux-audit.json                   # live-page audit
  pagespeed.json                  # PageSpeed Insights
```

Not every file is always present. The three JSON sidecars and the design
proposals are best-effort: a failure to write one emits a warning and never
fails the run. Synthesis appears only when report synthesis was enabled or
explicitly invoked. Exploration appears only when you ran `uxa explore` into
this output.

## Run bundles

### Lifecycle

A run writes to `<output>/.staging/<run-id>/`. While that directory is active it
contains an `.active` marker.

On success, finalization writes the terminal files, checksums every published
file, and atomically renames the directory to:

```text
<output>/runs/<run-id>/
```

On an abort, the timeline is closed, `crash.marker` is written, `.active` is
removed, and the staging evidence is left in place for recovery. An interrupted
bundle keeps its evidence; the next `--resume` archives it under
`.interrupted/<run-id>/` and re-runs that cell.

A finalized bundle cannot be mutated. Anything you or a tool change inside
`runs/<run-id>/` after finalization breaks its checksums.

### `manifest.json`

Reproducibility metadata for one run:

| Field | Meaning |
| --- | --- |
| `run_id` | The run's identity hash. |
| `seed` | Attention seed. |
| `model_trial` | External model-replication trial. Independent of the seed. |
| `config_digest` | The project configuration digest this run was expanded from. |
| `endpoint_origin` | Origin only, e.g. `https://provider.example`. Never the full URL, never credentials. |
| `model_ids` | Per-role model ids (`scent`, `cognitive`, and others when used). |
| `prompt_versions` | Per-role prompt and schema versions. |
| `package_version` | The `uxa` version that produced the run. |
| `provider_versions`, `provider_manifests` | Provider identity, role, optional model id, endpoint origin, version. |

An API key is never a manifest field. A URL containing credentials is rejected
outright. A legacy manifest with no `model_trial` reads as model trial `0`.

### `timeline.jsonl`

One JSON object per line, in order. The writer assigns monotonic one-based
`sequence` values itself and flushes and `fsync`s every event, so an interrupted
run still has a complete prefix.

Typed lifecycle events:

```text
run-started
viewport-captured
observation-recorded
action-proposed
action-executed
verification-recorded
agent-claim-contradicted
run-terminated
```

The application also records provider and application events, including
`coarse-scent-recorded`, `full-scent-recorded`, `agent-claim`,
`attention-exhausted`, `action-rejected`, `prominence-recorded`,
`attention-selection-recorded`, `decision-recorded`, and `model-call-recorded`.
Model-call records carry the sanitized request and response, retries, latency,
attempts, and token usage. The last lifecycle event is always `run-terminated`.

The terminal event carries `outcome`, `verification`, `provider_manifests`,
`configuration_digest`, and `artifact_checksums`.

### `result.json`

The serialized run result: run id, terminal outcome, the independent
verification result, the agent's own claim, final state, replay evidence,
per-run metrics, findings, and the bundle path when known.

The agent's claim is stored separately from verification and never changes it.
That separation is what makes false-success measurement possible.

### `artifacts/`

Raw content stored by SHA-256. Repeated identical content within one run reuses
one path. The in-memory artifact reference keeps the human name, path, digest,
and size; the stored path is the digest.

### `checksums.sha256`

One line per file:

```text
<sha256>  <relative-path>
```

Every published file in a finalized bundle is covered — `manifest.json`,
`timeline.jsonl`, `result.json`, everything under `artifacts/` and
`saliency/`. `checksums.sha256` itself and `.active` are excluded.

Verify a bundle yourself:

```powershell
Get-Content reports\<id>\runs\<run-id>\checksums.sha256 | ForEach-Object {
  $parts = $_ -split '  ', 2
  $actual = (Get-FileHash "reports\<id>\runs\<run-id>\$($parts[1])" -Algorithm SHA256).Hash.ToLower()
  if ($actual -ne $parts[0]) { "MISMATCH: $($parts[1])" }
}
```

An aborted staging directory keeps `crash.marker` and never receives a
finalized checksum file.

## Terminal outcomes

Every finalized run has exactly one terminal outcome. The set is closed:

| Outcome | Meaning |
| --- | --- |
| `verified-success` | The independent verifier confirmed the task. |
| `agent-abandoned` | The simulated persona gave up. |
| `budget-exhausted` | A budget ran out while the run was still working. |
| `verification-failed` | Verification kept failing, so the run stopped. Deliberately distinct from `budget-exhausted`. |
| `timed-out` | A deadline elapsed. |
| `provider-failure` | The target application or browser failed. |
| `model-failure` | The model transport failed. |
| `safety-blocked` | The browser network policy blocked a request the run needed. |
| `internal-error` | An unexpected internal failure. |

Check any one of them with:

```powershell
uv run uxa inspect-run reports\<id>\runs\<run-id>
```

## The experiment summary

### `experiment.json`

Written once per experiment, atomically. Six top-level keys:

| Key | Contents |
| --- | --- |
| `run_metrics` | Per-run metrics for every evaluable cell. |
| `cell_aggregates` | Aggregates grouped by prominence provider. |
| `variant_comparisons` | Paired variant comparisons across the experiment. |
| `findings` | Findings per run id. |
| `failures` | Runner failures and evaluator failures, with safe identity, stage, terminal state, and reason. |
| `invalid_runs` | Runs whose evidence cannot support a UX claim, with a sanitized reason. |

`failures` and `invalid_runs` exist so a problem is visible rather than absent.
Nothing is silently omitted.

### `experiment-progress.json`

The experiment resume checkpoint, rewritten atomically (temp file plus
same-directory replace) after every finalized result and every recorded runner
failure, so an interrupted long experiment is resumable. It carries a schema
version, the `selected_run_ids`, a `statuses` map, the `finalized_run_ids`,
`failed_run_ids`, `interrupted_run_ids`, and `pending_run_ids` lists, and a
`failure_types` map from run id to the sanitized error type. Failure values are
error type names, not messages.

If a checkpoint already exists and you start the same experiment without
`--resume`, the run stops with:

```text
checkpoint error: experiment checkpoint already exists; use --resume or a new
output
```

With `--resume`, the tool validates each selected finalized bundle, skips the
ones that pass, quarantines the ones that do not, archives interrupted staging
evidence, and rebuilds the aggregates from both the skipped and the newly
completed runs — so a resumed report is complete, not partial.

## Sidecars

These sit beside the report and are read live by it. See
[redesign](redesign.md) for the shared capture pipeline.

### `page-capture.json`

Schema `page-capture-v3`. The shared browser capture of the resolved page list:
per page, the URL, title, viewport, capture mode, document and captured height,
truncation flags, ordered image segments, and a trimmed node and copy
inventory. It is what the live-page audit and the design-proposal pass both
read, so each page is captured once. Detail in
[redesign § the capture pipeline](redesign.md#the-capture-pipeline).

### `ux-audit.json`

Schema `ux-audit-v1`. The deterministic live-page audit. It holds the issue
total, a per-URL report, and bounded error entries. A failure to audit records
an entry here rather than failing the run.

### `pagespeed.json`

The derived PageSpeed Insights report for the experiment's start URLs: one
analysis per URL per strategy, its category scores, audit pass/fail totals,
and opportunities with estimated savings. The API's raw responses are cached
under `pagespeed-cache/`.

When `UXA_PAGESPEED_WEB_LINKS` is enabled, the report additionally carries a
saved pagespeed.web.dev link and that report's own scores, kept separate from
the API run's because it is a second, independent Lighthouse analysis.

`UXA_SKIP_PAGESPEED` skips this file entirely.

### `traces/`

One `<run-id>.zip` Playwright trace per run: snapshots, network records, and —
unless `UXA_TRACE_SCREENCAST` is turned off — the screencast frames the report
replay uses. Traces are the largest files in an output directory and can
contain real page content. Trace publication is not best-effort: a trace
failure propagates rather than being swallowed, and partially written bytes are
discarded.

## Report synthesis

`uxa synthesize` publishes immutable attempts at experiment scope, beside the
run bundles. It never adds files to, rewrites, or changes checksums in a run
bundle.

```text
<output>/synthesis/
  index.json
  attempts/
    <attempt-id>/
      synthesis.json
      corpus-manifest.json
```

Attempt ids are `<created-at-utc>-<first 12 hex of the corpus digest>-<sequence>`.
An existing attempt directory is never overwritten; a rerun publishes the next
sequence.

| File | Contents |
| --- | --- |
| `synthesis.json` | Schema `synthesis-artifact-v3`, plus attempt identity and creation time; the corpus, expectation, principle-pack, prompt, and schema digests; model and role manifests; retrieval log; usage; candidate findings; objections; rejected findings; final findings; status; limitations; and whether deterministic fallback is available. |
| `corpus-manifest.json` | The canonical `EvidenceCorpus` document: every piece of persisted evidence the synthesis was allowed to see, as run, viewport, element, event, metric, replay, and artifact references. |
| `index.json` | `schema_version: synthesis-index-v1`, an `accepted_attempt_id` (or `null`), and an `attempts` list. |

All three use the same canonical JSON rule: ASCII, sorted keys, compact
separators, one terminating newline. The prompt and schema digests are SHA-256
values of their persisted version identifiers.

Only an `accepted` or `no-issues` attempt can be the selected report synthesis.
`rejected` and `unavailable` attempts stay on disk and stay readable, but the
report falls back to deterministic run findings.

`uxa report` is offline. It reads the selected attempt and never calls a model,
so the report regenerates with no network and no credentials. A missing,
rejected, unavailable, or invalid synthesis simply falls back — that is a
normal state, not an error.

## Exploration

```text
<output>/exploration/
  index.json
  attempts/
    <attempt-id>/
      corpus.json
      suggestions.json
      curated.json
      project.fragment.yaml
      manifest.json
  checkpoint/
    state.json
    payload.json
  project.yaml
  generated.yaml
  project.fragment.yaml
```

Schema `exploration-artifact-v1` for the attempt, `exploration-index-v1` for the
index. Full detail in [exploration](exploration.md#what-gets-written-where).

## Design proposals

```text
<output>/redesign/
  redesign-<YYYYMMDDTHHMMSSZ>-<8 hex>/
    index.json
    payload.json
    model-calls.json
```

Detail in [redesign](redesign.md#the-immutable-attempt-layout).

## The report itself

`report.html` is a single self-contained file: inline CSS and JavaScript, and
image data embedded where present. No external request is required to open it,
and none is made.

`uxa report` scans, in order: the bundle root itself if it looks like a run
bundle; every directory under `runs/`; and every directory under `.staging/` that
still has a `crash.marker` or an `.active` marker. So a crashed or still-active
run appears in the report too, marked `crashed` or `untrusted` and explicitly
excluded from scorecards, with the crash reason shown. That is deliberate: a
problem is recorded, not omitted.

One file is written when the encoded report is at most 4,000,000 bytes.
Larger reports become an index plus one page per run under
`<output-stem>-runs/`. The index links out to the per-run pages; the per-run
pages link back.

**Page checks** and **Page speed** refetch `ux-audit.json` and `pagespeed.json`
at page load. Browsers block that fetch on `file://` URLs, so use
`uxa report --serve` when you want those two views live. The server binds
`127.0.0.1` only.

Report projections are explicitly allowlisted. Selectors, test IDs, hidden
labels, destination URLs, provider ids, execution references, handler names,
passwords, tokens, and API keys are stripped before rendering, and all text is
HTML-escaped.

## Fix export package

`uxa export` writes a self-contained markdown package for an external fixing
agent.

```text
<package>/
  INDEX.md
  manifest.json
  issues/
    <finding-id>.md
  assets/
    <evidence-id>.png
```

| Path | Contents |
| --- | --- |
| `INDEX.md` | The issue list, synthesis status, tool version, and export time. |
| `manifest.json` | `schema_version: fix-export-v1`, export time, tool version, source report path, synthesis attempt id and status, whether deterministic fallback was used, and per issue its `finding_id`, sanitized `filename`, severity, and assigned skill set, plus the SHA-256 of every copied asset. |
| `issues/<finding-id>.md` | One issue: the problem, its cause and impact, embedded evidence with links into `../assets/`, fix options, skill references, and the fixer workflow. |
| `assets/` | Copied evidence bytes, named by evidence id. |

Every asset is verified against the SHA-256 recorded in the bundle before it is
copied; a mismatch aborts the export with
`artifact for evidence '<id>' does not match its recorded sha256 in the bundle`.
Evidence that cannot be resolved is marked `evidence unavailable` in the issue
rather than dropped, and the CLI warns which finding it was.

Issue file names are the finding id sanitized for the filesystem, so they are not
guaranteed to be a verbatim id. `manifest.json` records the mapping.

The package directory must not already exist — `export package already exists:
<dir>` is an error, so pick a fresh `--out` or remove the old one.

## Redaction in the artifacts

Before any of these JSON documents are written, the redaction policy replaces:

- every configured exact value — in CLI runs, every scenario input marked
  `sensitive: true` — with `[REDACTED]`
- every value under a sensitive mapping key with `[REDACTED]`

Text fields additionally pass through pattern redaction covering credential
URLs, bearer tokens, sensitive assignments, fixture secrets, and filesystem
paths.

Raw bundles are still sensitive. Bundle serialization can retain
provider-private execution references and other run-state fields when they are
not configured as redaction keys. Restrict access to the output directory and
delete retained traces and screenshots when you no longer need them. See
[security](security.md).
