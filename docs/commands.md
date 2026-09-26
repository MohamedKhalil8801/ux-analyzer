# Command reference

Every `uxa` command, with its real options and defaults. Run
`uv run uxa <command> --help` for the authoritative list.

```text
version      Print package version.
validate     Validate one benchmark project configuration.
run          Expand and execute one benchmark experiment.
run-one      Execute exactly one semantically selected benchmark run.
ablate       Execute selected attention policy ablations.
report       Regenerate static report from finalized run bundles.
pagespeed    Fetch the complete PageSpeed Insights report for any URL.
slop         Score any page against the 27-rule AI-design-slop fingerprint.
synthesize   Run report synthesis for finalized experiment evidence.
explore      Discover scenarios via smart crawl + cognitive synthesis + human curation.
inspect-run  Print terminal outcome and artifact paths for one run bundle.
export       Export selected issues as an LLM-optimized fix package.
redesign     Propose creative redesign improvements from persisted page captures.
fixture      Serve the bundled controlled fixture application.
models       Manage local saliency-model releases.
```

## Shared conventions

**Output directories.** `run`, `run-one`, `ablate`, and `synthesize` default
`--output` to `reports/<project-id>`. `explore` defaults to
`.uxa-output/explore/<site-slug>/`, where the slug comes from the hostname of
the first starting URL. `export` defaults to
`.uxa-output/fix-export-<YYYYMMDD-HHMMSS>/`.

**Model settings.** Every command that talks to a model resolves settings from
the environment (or `.env`) before it starts. A missing variable produces
`model environment error: missing model environment variables: <names>` and
exit code 1. `--check-env` checks names without printing values. A dry run
tolerates missing settings unless `--check-env` is also given.

**Exit codes.** `run`, `run-one`, and `ablate` exit 1 when the experiment had
any runner failure, evaluator failure, or invalid UX sample, even though the
report was still written.

---

## `uxa version`

Prints `uxa <version>`.

```powershell
uv run uxa version
```

## `uxa validate`

Validates one project YAML: reference integrity, field constraints, and the
canonical configuration digest. Prints the project id, scenario/persona/experiment
counts, and the SHA-256 config digest. The digest is part of every `RunSpec`,
so changing any referenced field changes every run id.

### Arguments and options

| Option | Default | Meaning |
| --- | --- | --- |
| `PROJECT` (required) | — | Path to the project YAML. |
| `--check-env` | off | Also verify the required model environment names are present. Never prints values. |

```powershell
uv run uxa validate benchmarks\demo\project.yaml
uv run uxa validate benchmarks\demo\project.yaml --check-env
```

## `uxa run`

Expands one experiment into its run matrix and executes it. Every cell gets an
isolated browser context and a fresh run bundle.

### Arguments and options

| Option | Default | Meaning |
| --- | --- | --- |
| `PROJECT` (required) | — | Path to the project YAML. |
| `--experiment` | `core-pair` | Experiment id to expand. |
| `--output` | `reports/<project-id>` | Output directory. |
| `--workers` | `1` | Concurrent runs. |
| `--run-count` | experiment's own value | Override the number of repeats; also clears any explicit `seeds`. |
| `--dry-run` | off | Print the expanded matrix and model-call estimate. No browser, no model, no writes. |
| `--check-env` | off | Verify model environment names before expanding. |
| `--fixture-origin` | `http://127.0.0.1:8000` | Origin of the bundled fixture target. |
| `--resume` | off | Skip finalized bundles that pass integrity and terminal-structure validation, and rebuild aggregates from skipped plus newly finished runs. |
| `--no-synthesis` | off | Execute runs without the automatic post-run synthesis attempt. |
| `--profile-output` | unset | Directory for per-run stage-timing JSON files. |
| `--allow-origin` | none | Grant one extra browser resource origin to every live application version. Repeatable. Skips the interactive prompt. |
| `--yes` | off | Accept every referenced non-allowlisted origin without prompting. |

```powershell
uv run uxa run benchmarks\demo\project.yaml --experiment core-pair --workers 4
uv run uxa run benchmarks\demo\project.yaml --experiment core-pair --dry-run
uv run uxa run benchmarks\demo\project.yaml --experiment core-pair --resume
```

Before a live run starts, the tool looks for origins your target references
that are not on the allowlist. Interactively you are asked once. In a
non-interactive session nothing is granted and the run is told:

```text
warning: live targets reference non-allowlisted origins (...); these requests
will safety-block. Re-run interactively or pass --yes / --allow-origin.
```

## `uxa run-one`

Executes exactly one matrix cell. Useful for reproducing a single row of a
larger experiment.

### Arguments and options

| Option | Default | Meaning |
| --- | --- | --- |
| `PROJECT` (required) | — | Path to the project YAML. |
| `--scenario` (required) | — | Scenario id. |
| `--version` (required) | — | Application version id. |
| `--persona` (required) | — | Persona id. |
| `--policy` (required) | — | One of `full-list`, `prominence-ranked-list`, `progressive-prominence`, `progressive-prominence-scent`. |
| `--seed` (required) | — | Attention seed. |
| `--prominence-provider` / `--prominence-provider-id` | `heuristic` | Prominence provider id. |
| `--output` | `reports/<project-id>` | Output directory. |
| `--fixture-origin` | `http://127.0.0.1:8000` | Origin of the bundled fixture target. |
| `--dry-run` | off | Print the resolved spec and stop. |
| `--check-env` | off | Verify model environment names. |
| `--profile-output` | unset | Directory for this run's stage-timing JSON. |

```powershell
uv run uxa run-one benchmarks\demo\project.yaml `
  --scenario invite-teammate `
  --version fixture-app-defective `
  --persona first-time-nontechnical `
  --policy progressive-prominence-scent `
  --seed 0
```

## `uxa ablate`

Runs the same machinery as `uxa run` but restricted to a set of attention
policies. This is the secondary benchmarking path: it answers "how much of the
result comes from the attention policy" rather than "what is wrong with this
interface". Attention-policy ablation is not part of the user-facing analysis
workflow; use `uxa run` for that.

### Arguments and options

Same as `uxa run`, with two differences:

| Option | Default | Meaning |
| --- | --- | --- |
| `--experiment` | `ablations` | Experiment id to expand. |
| `--policy` | all policies in the experiment | Restrict the matrix to these policies. Repeatable. |

```powershell
uv run uxa ablate benchmarks\demo\project.yaml --experiment ablations --workers 4
uv run uxa ablate benchmarks\demo\project.yaml --policy full-list --policy progressive-prominence-scent --dry-run
```

## `uxa report`

Renders a static HTML report from a bundle root. It is entirely offline: it
reads stored bundles and immutable synthesis artifacts and never calls a model.
The report can therefore be regenerated without network access or credentials.

It accepts a bundle root containing one run bundle, a `runs/` directory of
finalized bundles, or an experiment directory with `experiment.json` plus
whatever partial evidence exists. It also picks up anything still in
`.staging/` that has a `crash.marker` or an `.active` marker. Finalized runs,
evaluator failures, runner failures, and crashed bundles are all included; failed
and crashed rows keep safe identity, stage, terminal state, reason, and
available timeline evidence rather than being dropped, and are marked untrusted
so they stay out of scorecards.

### Arguments and options

| Option | Default | Meaning |
| --- | --- | --- |
| `BUNDLE_ROOT` (required) | — | Bundle root, `runs/` directory, or experiment directory. |
| `--output` | `<bundle-root>/report.html` | Rendered report path. |
| `--serve` | off | Serve the rendered directory over loopback HTTP after rendering. |
| `--port` | ephemeral | Port for `--serve`. Range 1–65535. |
| `--browser` / `--no-browser` | `--browser` | Open the served report in your browser. |

One HTML file is emitted when the encoded report is at most 4,000,000 bytes.
Larger reports produce an index plus one page per run under
`<output-stem>-runs/`.

```powershell
uv run uxa report reports\exploration-generated
uv run uxa report reports\exploration-generated --output reports\final.html
uv run uxa report reports\exploration-generated --serve --port 8000
```

Use `--serve` when you want the **Page checks** and **Page speed** views live;
those views refetch their JSON sidecars at page load, which browsers block on
`file://` URLs. The server binds `127.0.0.1` only.

## `uxa pagespeed`

Fetches the complete PageSpeed Insights report for a URL, using the same API
behind pagespeed.web.dev. It reports Lighthouse category scores, per-audit
pass/fail results, and opportunity savings.

By default it also runs one headless pagespeed.web.dev analysis per URL to
capture the stable saved-report link. That is a second, independent Lighthouse
run, so its scores are labeled and shown separately from the API run's. The
capture is cached per URL for 7 days.

### Arguments and options

| Option | Default | Meaning |
| --- | --- | --- |
| `URL` (required) | — | URL to analyze. |
| `--strategy` | `mobile,desktop` | Lighthouse strategy. Repeatable. |
| `--json` | off | Emit the derived report JSON instead of the summary. |
| `--cache-root` | unset | Directory for the raw-response disk cache. |
| `--no-cache` | off | Ignore and do not update the disk cache. |
| `--web-ui` / `--no-web-ui` | `--web-ui` | Run one pagespeed.web.dev analysis per URL to capture the saved-report link. |

Set `UXA_SKIP_PAGESPEED=1` to disable the web-UI capture. The API key is
resolved from the first usable value among `PSI_API_Key`, `PSI_API_KEY`,
`GOOGLE_API_KEY`, and `UXA_PSI_API_KEY`; a value is only usable if it is 10–200
characters drawn from `A–Z a–z 0–9 . _ ~ + / = -`. Without a key, the shared
unauthenticated quota is frequently exhausted.

```powershell
uv run uxa pagespeed https://example.com
uv run uxa pagespeed https://example.com --strategy desktop --json
uv run uxa pagespeed https://example.com --no-web-ui
```

## `uxa slop`

Scores a page against the 27-rule AI-design-slop fingerprint. It renders the
page in a headless browser, extracts a computed-style snapshot plus page text,
then reports a 0–100 score, a tier (`Clean` / `Mild` / `Heavy`), a letter grade,
and every triggered pattern with its evidence.

Tier thresholds: design axis is `Clean` 0–9, `Mild` 10–27, `Heavy` 28 and above.
With `--copy`, the copy axis adds 9 more patterns, scored `Clean` 0–7, `Mild`
8–19, `Heavy` 20 and above; pages with fewer than 40 words are reported as too
thin to judge. When both axes are scored, the unified score is the higher axis
plus 6 per additional non-clean axis.

### Arguments and options

| Option | Default | Meaning |
| --- | --- | --- |
| `SOURCE` (required) | — | URL or local HTML file. |
| `--json` | off | Emit JSON instead of pretty output. |
| `--copy` | off | Also score the copy axis. |

Set `UXA_SLOP_DISABLE_JS` to any non-empty value — including `0` — to pin the
DOM (JavaScript disabled) for deterministic comparisons against a frozen local
file. Unset it to re-enable JavaScript.

```powershell
uv run uxa slop https://example.com
uv run uxa slop .\page.html --copy
uv run uxa slop https://example.com --json
```

## `uxa synthesize`

Builds one new immutable synthesis attempt from finalized run evidence, without
re-running anything. Use it when you want a fresh reading of an experiment whose
runs are already complete.

### Arguments and options

| Option | Default | Meaning |
| --- | --- | --- |
| `PROJECT` (required) | — | Path to the project YAML. |
| `--experiment` (required) | — | Experiment id whose finalized evidence to synthesize. |
| `--output` | `reports/<project-id>` | Experiment output directory holding the run bundles. |

The command reads `<output>/experiment.json` first, so the experiment must have
been run already. It then writes the attempt under `<output>/synthesis/` and
re-renders `report.html`.

Attempts have one of four statuses: `accepted`, `no-issues`, `rejected`,
`unavailable`. See [troubleshooting](troubleshooting.md#synthesis-attempt-is-unavailable)
for what each means.

```powershell
uv run uxa synthesize .uxa-output\explore\example\project.yaml --experiment exploration-run
```

## `uxa explore`

Crawls a site, synthesizes candidate scenarios, and curates them. Covered in
full in [exploration](exploration.md).

### Arguments and options

| Option | Default | Meaning |
| --- | --- | --- |
| `PROJECT` (positional, optional) | — | Base project YAML to reuse personas and providers from. |
| `--project` | — | Same thing, as a flag instead of a positional. |
| `--starting-url` | from YAML `exploration.start_urls` | Starting URL. Repeatable. HTTPS required. Flags win over YAML. |
| `--depth` | `2` | Crawl depth, 0–5. |
| `--max-pages` | `50` | Page cap, 1–200. |
| `--max-scenarios` | `8` | Scenario cap, 1–20. |
| `--auto-accept` | off | Skip the review UI and accept every suggestion. |
| `--review-port` | ephemeral free port | Review UI port, 1–65535. |
| `--output` | `.uxa-output/explore/<site-slug>` | Output directory. |
| `--run` | off | Immediately run the generated `exploration-run` experiment into `<output>/exploration-run-output`. |
| `--dry-run` | off | Print the crawl matrix, page estimate, token estimate, and a settings fingerprint. No browser, no model. |
| `--no-browser` | off | Do not auto-open your browser on the review UI. |
| `--resume` | off | Reuse a checkpointed crawl corpus and suggestions from an interrupted run. |

```powershell
uv run uxa explore --starting-url https://example.com
uv run uxa explore --starting-url https://example.com --depth 3 --max-pages 80 --auto-accept
uv run uxa explore --starting-url https://example.com --dry-run
```

## `uxa inspect-run`

Prints the terminal outcome, verification result, agent claim, and artifact
paths for one finalized run bundle. This is the first thing to reach for when a
run did not behave as you expected.

### Arguments and options

| Option | Default | Meaning |
| --- | --- | --- |
| `RUN_PATH` (required) | — | Path to one run bundle directory, i.e. `runs/<run-id>`. |

```powershell
uv run uxa inspect-run reports\exploration-generated\runs\run-0f1c...
```

Output looks like:

```text
run: run-0f1c...
outcome: verified-success
verified: True
claimed: True
artifacts:
- artifacts/2b1f...png
- ...
```

A non-existent path exits with `run bundle does not exist: <path>`.

## `uxa export`

Writes a self-contained markdown fix package for an external fixing agent: one
issue file per selected finding, an index, a manifest, and copied evidence
assets. See [output formats](output-formats.md#fix-export-package).

**In an interactive terminal, `uxa export` opens a selection UI** — space
toggles an issue, arrow keys move, Enter continues — and the flags below are
ignored. The flags are what apply in a non-interactive session (piped stdio, CI).

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--report` (required) | — | Experiment output directory containing `report.html`. |
| `--out` | `.uxa-output/fix-export-<timestamp>` | Package directory. Must not already exist. |
| `--all` | off | Export every issue in the report. |
| `--finding` | none | Export one issue by finding id. Repeatable. |
| `--exclude` | none | Drop one issue by finding id. Repeatable. |
| `--skill-set` | none | Skill set applied to every issue. The first value wins. |
| `--issue-skill` | none | Per-issue override, formatted `FINDING_ID=SET`. Repeatable. |
| `--skill-sets` | `UXA_SKILL_SETS` or the platform config dir | Path to the skill-sets file. |
| `--skills-note` | none | Free-text note embedded in the package. |
| `--notes` | none | Reproduction notes file, embedded verbatim. Must exist. |

```powershell
uv run uxa export --report reports\exploration-generated --all
uv run uxa export --report reports\exploration-generated --finding "audit:tap-target-size" --out .uxa-output\fix-one
```

Finding ids are namespaced by source, and the namespace is part of the id:
deterministic rule findings use `<run-id>:<category>`, live-page audit findings
use `audit:<check-id>`, slop findings use `slop:<page-slug>`, PageSpeed findings
use `pagespeed:<strategy>:<audit-id>`, design proposals use
`redesign:<proposal-id>`, and report-synthesis findings carry the id the
adjudicator assigned. Read the report to get the exact ids; the CLI lists the
valid ones when you pass an unknown one:

```text
unknown finding id(s) ['UX-001']; valid: audit:tap-target-size, slop:example
```

Non-interactive selection requires `--all` or at least one `--finding`:

```text
nothing to export: select issues or pass --all
```

A report with no issues at all is also refused:

```text
nothing to export: the report contains no issues
```

If a selected issue references evidence that cannot be resolved, the package is
still written and the CLI warns:

```text
warning: issue 'audit:tap-target-size' has evidence unavailable; it is exported
with 'evidence unavailable' markers.
```

## `uxa redesign`

Generates design proposals from a persisted or fresh page capture. Covered in
full in [redesign](redesign.md).

### Arguments and options

| Option | Default | Meaning |
| --- | --- | --- |
| `OUTPUT` (required) | — | Experiment output directory. |
| `EXTRA_PAGES...` (optional positional) | none | Bare page URLs appended to `--pages`. |
| `--pages` | resolved list | Explicit page URLs. Repeatable. Overrides the resolved list. |
| `--audience` | empty | Optional operator context. The model still infers and states its own audience. |
| `--max-pages` | `UXA_REDESIGN_MAX_PAGES`, else `10` | Page-list cap. |
| `--dry-run` | off | Resolve and print the page list without running models. |

```powershell
uv run uxa redesign reports\exploration-generated
uv run uxa redesign reports\exploration-generated --pages https://example.com/pricing --audience "trial signup funnel"
uv run uxa redesign reports\exploration-generated --dry-run
```

With no captures and no `--pages`, the command exits with:

```text
no page captures available: provide --pages or run an experiment or
exploration first
```

Set `UXA_REDESIGN_ENABLED=1` to run the pass automatically after a completed
experiment instead of invoking it by hand. It is off by default and never fails
the experiment.

## `uxa fixture serve`

Serves the bundled controlled fixture application. It is a local target with
in-memory state and no outbound communication, useful for exercising a project
without pointing at anything real.

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Bind host. Non-loopback values are rejected. |
| `--port` | `8000` | Bind port. |

```powershell
uv run uxa fixture serve --host 127.0.0.1 --port 8000
```

Then, in a second shell:

```powershell
uv run uxa validate benchmarks\demo\project.yaml
uv run uxa run benchmarks\demo\project.yaml --experiment core-pair --workers 1
```

## `uxa models`

Manages local saliency-model releases. Foveacast inference never downloads a
model or a runtime; artifacts are installed explicitly through these commands.
The default prominence provider is `heuristic`, so none of this is needed
unless you are explicitly comparing prominence providers.

### `uxa models install`

| Option | Default | Meaning |
| --- | --- | --- |
| `MODEL_ID` | `foveacast-v0.2.0` | Release to download and verify. |
| `--precision` | `fp16` | Artifact precision. |

```powershell
uv sync --extra saliency-cpu
uv run uxa models install foveacast-v0.2.0 --precision fp16
```

On Windows you can use the DirectML runtime instead, but install exactly one of
the two extras — both provide the same `onnxruntime` import:

```powershell
uv sync --extra saliency-directml
```

### `uxa models status`

Inspects the runtime and any locally installed artifacts. Never downloads.

| Option | Default | Meaning |
| --- | --- | --- |
| `MODEL_ID` | `foveacast-v0.2.0` | Release to inspect. |
| `--provider` | `cpu` | Execution provider: `cpu`, `directml`, or `auto`. |

```powershell
uv run uxa models status foveacast-v0.2.0 --provider cpu
```

A missing runtime reports `runtime missing` with a diagnostic telling you to
install the `saliency-cpu` or `saliency-directml` extra.

### `uxa models remove`

| Option | Default | Meaning |
| --- | --- | --- |
| `MODEL_ID` | `foveacast-v0.2.0` | Release to remove. |
| `--precision` | `fp16` | Artifact precision. |

```powershell
uv run uxa models remove foveacast-v0.2.0 --precision fp16
```

`UXA_MODEL_HOME` relocates the model store root when the release lives
elsewhere.

## Experiment vocabulary

`--policy` and `policies:` accept exactly four values:

| Policy | What the agent sees per observation |
| --- | --- |
| `full-list` | Every visible persona-safe element at once. |
| `prominence-ranked-list` | Every visible persona-safe element, ranked by prominence. |
| `progressive-prominence` | A bounded batch of new elements, chosen by prominence. |
| `progressive-prominence-scent` | The same, with a scent call ranking candidate relevance to the goal. |

Run ids are SHA-256 hashes over experiment id, scenario id, application version
id, persona id, policy, attention seed, and config digest, plus the model trial
and prominence provider when they differ from their defaults.
