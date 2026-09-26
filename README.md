# uxa — Recorded-interaction UX analysis

`uxa` drives a real browser through a task, records what the user actually saw
and did, and turns that record into a verifiable UX report.

The pipeline is **evidence-first**. A finding is publishable only when it points
at something the run captured — an event, a metric, a viewport, an element, a
screenshot, a heatmap, or a replay position. The model proposes; deterministic
code decides. Nothing is asserted that the record cannot support.

```text
crawl → scenarios → runs → recorded evidence → UX report → fix package
                                               ↘ redesign proposals
```

## What it does

| Stage | Command | Output |
| --- | --- | --- |
| Discover scenarios from a live site | `uxa explore` | Crawl corpus + scenario suggestions + review UI |
| Execute recorded runs | `uxa run` | Immutable run bundles with UI-state snapshots |
| Synthesize the UX report | `uxa synthesize` | Evidence-backed findings, or a valid "no issues" |
| Propose design changes | `uxa redesign` | Model-estimate design proposals (not findings) |
| Audit page performance | `uxa pagespeed` | Lighthouse categories and per-audit results |
| Audit AI-design slop | `uxa slop` | 0–100 score over a 27-rule fingerprint |
| Hand fixes to an agent | `uxa export` | Self-contained markdown fix package |

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync
uv run playwright install chromium
```

### Analyze a live site end to end

This is the main path. It needs an HTTPS target and model configuration.

```powershell
$env:UXA_LLM_BASE_URL  = "https://<provider-host>/v1"
$env:UXA_LLM_API_KEY   = "<api-key>"
$env:UXA_SCENT_MODEL   = "<model-id>"
$env:UXA_COGNITIVE_MODEL = "<model-id>"
$env:UXA_REPORT_MODEL  = "<model-id>"

uv run uxa explore --starting-url https://example.com
```

`explore` crawls the site, synthesizes candidate scenarios, and opens a review
UI. Accept or edit the scenarios, then run them:

```powershell
uv run uxa run .uxa-output\explore\example\project.yaml --experiment exploration-run
```

Open `reports/exploration-generated/report.html` in any browser. It is static
HTML — no server needed. `uxa report` only reads recorded bundles and never
calls a model.

### Self-contained demo (no live target)

The demo ships a local fixture app and needs no network target.

```powershell
uv run uxa fixture serve --host 127.0.0.1 --port 8000
```

In a second shell:

```powershell
uv run uxa validate benchmarks/demo/project.yaml
uv run uxa run benchmarks/demo/project.yaml --experiment core-pair --output .uxa-output --workers 1
```

## Commands

Run `uxa <command> --help` for the full option list.

| Command | Purpose |
| --- | --- |
| `uxa version` | Print the package version. |
| `uxa validate PROJECT` | Validate one project YAML and print its SHA-256 config digest. |
| `uxa validate PROJECT --check-env` | Also check required model environment names, without printing values. |
| `uxa fixture serve` | Serve the bundled demo fixture app. Default `127.0.0.1:8000`. |
| `uxa models install\|status\|remove` | Manage local saliency-model releases. |
| `uxa explore [PROJECT]` | Crawl a site, synthesize scenarios, curate them in a review UI, and write a runnable project. |
| `uxa explore [PROJECT] --auto-accept` | Skip the review UI and accept every suggestion. |
| `uxa run PROJECT --experiment ID` | Execute one experiment. |
| `uxa run-one PROJECT --scenario --version --persona --policy --seed` | Execute exactly one matrix cell. |
| `uxa run PROJECT --dry-run` | Print the expanded matrix and model-call estimate. No browser, no model. |
| `uxa run PROJECT --no-synthesis` | Execute runs without the automatic synthesis attempt. |
| `uxa ablate PROJECT --policy POLICY` | Execute attention-policy ablations. Repeat `--policy`. |
| `uxa synthesize PROJECT --experiment ID --output DIR` | Build a new immutable synthesis attempt from finalized evidence, without rerunning runs. |
| `uxa redesign OUTPUT` | Generate model-estimate design proposals from a persisted or fresh page capture. |
| `uxa report BUNDLE_ROOT --output FILE` | Render finalized bundles into static HTML. |
| `uxa pagespeed URL` | Fetch the full PageSpeed Insights report for a URL. |
| `uxa slop SOURCE` | Score a URL or local HTML file against the 27-rule slop fingerprint. |
| `uxa export --report DIR` | Export selected findings as a fix package for an external agent. |
| `uxa inspect-run RUN_DIR` | Print terminal outcome, verification, and artifact paths for one run. |

Attention policies: `full-list`, `prominence-ranked-list`,
`progressive-prominence`, `progressive-prominence-scent`.

Common run options: `--workers`, `--run-count`, `--resume`, `--check-env`,
`--fixture-origin`. `--resume` skips only finalized bundles that pass integrity
and terminal-structure validation.

## Configuration

`uxa` loads one combined YAML project. Root fields:

```yaml
id: project-id
name: Project name
applications: []
scenarios: []
personas: []
experiments: []
```

- **Applications** carry `id`, `name`, and `versions`. Each version needs `id`,
  `kind` (`defective`, `improved`, or `live`), and `label`; a `live` version also
  needs `start_url`. A non-live application needs at least one `defective` and
  one `improved` version.
- **Scenarios** carry `id`, `name`, `goal`, `application_version_ids`,
  `start_state`, `fixture_inputs`, `budget`, `verifier`, `safeguards`,
  `eligible_persona_ids`, `expected_evidence`, `evaluation_target`, and an
  optional `viewport`. Budgets require positive `max_steps`,
  `max_observations`, and `max_interactions`; `timeout_seconds` is optional and
  `null` means no overall deadline.
- **Verifiers** are `fixture-state` (`resource`, `field`, `operator`,
  `expected_fixture_key`; operators `equals`, `not-equals`, `contains`,
  `truthy`, `falsy`), `visible-result` (`text`, optional `role` and `all_of`),
  or `colour-change` (`threshold`, default `32.0`).
- **Personas** carry `id`, `name`, positive `working_memory_capacity`, bounded
  `initial_confidence`, `initial_frustration`, and `abandonment_threshold`, plus
  positive `attention_temperature`.
- **Experiments** carry `id`, `name`, `scenario_ids`, `application_version_ids`,
  `persona_ids`, `policies`, optional unique `seeds`, and positive `run_count`.
  Empty `seeds` means `0..run_count-1`. `model_trials` and
  `prominence_provider_ids` have defaults.

Optional root sections: `providers` (prominence weights, progressive-attention
formula weights, `providers.expectation` for versioned frozen-expectation
documents, and optional `providers.saliency`), `evaluation` (discovery-cost,
finding-rule, state-update formulas, and `evaluation.report_synthesis` for the
post-run synthesis pipeline), and `exploration` (the crawl boundary for
`uxa explore`).

The loader validates references and canonicalizes sorted JSON before hashing, so
the config digest is part of every `RunSpec` and bundle manifest.

## Environment variables

Never put real keys in YAML, source, or documentation.

### Model transport

| Variable | Purpose |
| --- | --- |
| `UXA_LLM_MODE` | `api` (default) or `codex`. |
| `UXA_LLM_BASE_URL` | HTTP(S) base URL without credentials. Requests go to `<base-url>/chat/completions`. |
| `UXA_LLM_API_KEY` | API key for `api` mode. |
| `UXA_LLM_TIMEOUT_SECONDS` | Per-request transport timeout. |
| `UXA_LLM_MAX_CONCURRENT_CALLS` | Client-side concurrency cap. |
| `UXA_LLM_REQUEST_MAX_BYTES` | Refuse oversized requests before transport. |
| `UXA_LLM_SESSION_ID` | Optional provider session identifier. |
| `UXA_SCENT_MODEL` | Model id for information-scent roles. |
| `UXA_COGNITIVE_MODEL` | Model id for the cognitive agent. |
| `UXA_REPORT_MODEL` | Model id for report synthesis. Required by `uxa synthesize` and by `validate --check-env` when synthesis is enabled; optional during `uxa run`, which degrades to deterministic findings. |
| `UXA_REDESIGN_MODEL` | Model id for `uxa redesign`. Falls back to `UXA_REPORT_MODEL`. |
| `UXA_LLM_SCENT_REASONING_EFFORT` | Reasoning effort for scent calls. |
| `UXA_LLM_COGNITIVE_REASONING_EFFORT` | Reasoning effort for cognitive calls. |
| `UXA_LLM_REPORT_REASONING_EFFORT` | Reasoning effort for report calls. |

Scent and cognitive model ids may be equal; the roles stay separate. Use a
structured-output, vision-capable model for `UXA_REPORT_MODEL` when the evidence
corpus contains screenshots or heatmaps.

In `codex` mode, the `codex` CLI must already be installed, logged in, and on
`PATH`. That mode never reads or prints credentials.

### Pipeline toggles and capture limits

| Variable | Default | Purpose |
| --- | --- | --- |
| `UXA_REPORT_SYNTHESIS_ENABLED` | from YAML | Non-empty and not `0`/`false`/`no`/`off` enables; those four disable; unset keeps project behavior. |
| `UXA_REDESIGN_ENABLED` | off | Run the redesign pass automatically after a completed experiment. |
| `UXA_REDESIGN_MAX_PAGES` | 10 | Page-capture list bound for redesign. |
| `UXA_REDESIGN_MAX_PAGE_HEIGHT` | 12000 | Per-page capture height cap in px. Unset, unparseable, or non-positive falls back to 12000. |
| `UXA_TRACE_SCREENCAST` | on | Record trace screencast frames (replay video) during runs. Set to a falsey value to shrink trace archives. |
| `UXA_MODEL_HOME` | platform default | Root for local saliency-model releases. |
| `UXA_SKILL_SETS` | platform default | Directory of fix-export skill sets. |

### Reporting and audits

| Variable | Default | Purpose |
| --- | --- | --- |
| `PSI_API_Key` / `PSI_API_KEY` / `GOOGLE_API_KEY` / `UXA_PSI_API_KEY` | unset | PageSpeed Insights key. Checked in that order. |
| `UXA_SKIP_PAGESPEED` | unset | Skips the PageSpeed Insights pass during `uxa run`. |
| `UXA_PAGESPEED_WEB_LINKS` | unset | Enables saved-report link capture during `uxa run`. |
| `UXA_SLOP_DISABLE_JS` | unset | Any non-empty value, including `0`, disables JavaScript in `uxa slop`. |
| `UXA_RUN_LIVE_TESTS` | unset | Enables opt-in live endpoint tests. Must be set in the real process environment; a line in `.env` is ignored. |

## Evidence and honesty rules

These are the guarantees the pipeline is built around.

- **Findings are evidence-backed.** Every claim resolves to a persisted
  reference. Unsupported claims are rejected before publication, not softened.
- **A model estimate is labeled as one.** Redesign proposals and heuristic
  prominence carry no verification status and are never presented as findings.
- **"No issues" is a valid result.** If the evidence establishes no harm, the
  report says so instead of inventing one.
- **A rejected proposal is dropped, and it is named.** Validation failures are
  recorded with the reason, never silently repaired.
- **Silent truncation is disclosed.** When page capture is capped, the report
  states how much content was not captured.
- **Simulated evidence stays labeled.** Benchmark output is synthetic and is not
  a measurement of real-user completion, satisfaction, emotion, accessibility
  behavior, or product demand.
- **A broken scenario is a scenario defect.** A verifier that cannot resolve on
  any reachable page invalidates that run's evidence; it is never converted into
  a product claim.

## Documentation

Start with [Getting started](docs/getting-started.md) for a first analysis, then
[Commands](docs/commands.md) for the full reference.

| Document | Contents |
| --- | --- |
| [Getting started](docs/getting-started.md) | Install, model setup, and a complete first analysis. |
| [Commands](docs/commands.md) | Every command, its options, and a worked example. |
| [Configuration](docs/configuration.md) | Project YAML schema and every environment variable. |
| [Exploration](docs/exploration.md) | The scenario-discovery workflow. |
| [Redesign](docs/redesign.md) | Design proposals, the two model roles, and capture limits. |
| [Output formats](docs/output-formats.md) | Run bundle layout, sidecars, and how to read them. |
| [Security](docs/security.md) | What reaches the model provider, redaction, and key handling. |
| [Troubleshooting](docs/troubleshooting.md) | Real failure modes and what to do about them. |
| [Glossary](docs/glossary.md) | Full catalog of domain terms. |
| [Context and language](CONTEXT.md) | Enforced vocabulary and language constraints for agents. |
| [Model provider](docs/model-provider.md) | Transport modes and the [synthesis contract](docs/model-provider.md#report-synthesis-contract). |

Contributor documentation — [architecture](docs/architecture.md),
[testing](docs/testing.md), and the [architecture decision records](docs/adr/) —
is engineering reference material, not user documentation.

## Verification

```powershell
uv run uxa validate benchmarks/demo/project.yaml
uv run pytest -m "not live" -q
uv run ruff check .
uv run pyright
git diff --check
```

Live endpoint tests are opt-in via `UXA_RUN_LIVE_TESTS=1` and are never part of
the non-live suite.

## License

Apache License 2.0. See [LICENSE](LICENSE).
