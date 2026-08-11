# Attention-Guided UI Agent

Python 3.12 proof of concept for benchmarkable synthetic UI discovery runs.
It compares unrestricted element access with progressive attention-guided
observation across two controlled SaaS workflows:

- `invite-teammate`
- `enable-2fa`

Current POC uses Chromium through Playwright, deterministic rendered-element
extraction, heuristic prominence, seeded progressive attention, structured
OpenAI-compatible model calls, and an independent fixture verifier. Outputs
are simulated benchmark evidence. They are not measurements of real-user
completion, satisfaction, emotion, accessibility behavior, or product demand.

## Quick Start

Run from repository root.

```powershell
uv sync
uv run playwright install chromium
uv run uxa fixture serve --host 127.0.0.1 --port 8000
```

Keep fixture server running. In second shell:

```powershell
uv run uxa validate benchmarks/demo/project.yaml
uv run uxa run benchmarks/demo/project.yaml --experiment core-pair --output .uxa-output --dry-run
```

Execution needs model configuration. Use placeholder values while configuring
your environment; never place real keys in YAML, source, or documentation.

```powershell
$env:UXA_LLM_BASE_URL = "https://<provider-host>/v1"
$env:UXA_LLM_API_KEY = "<api-key>"
$env:UXA_SCENT_MODEL = "<scent-model-id>"
$env:UXA_COGNITIVE_MODEL = "<cognitive-model-id>"
$env:UXA_REPORT_MODEL = "<report-model-id>"
```

`UXA_LLM_BASE_URL` must be an HTTP(S) URL without credentials. Adapter posts to
`<base-url>/chat/completions`. Scent and cognitive model IDs may be equal, but
roles remain separate. `UXA_REPORT_MODEL` is required for live report synthesis
when the project enables it; use a structured-output vision-capable model when
the evidence corpus contains screenshots or heatmaps.

Run core benchmark:

```powershell
uv run uxa run benchmarks/demo/project.yaml --experiment core-pair --output .uxa-output --workers 1
```

Completed or partial execution writes immutable run bundles when available,
`.uxa-output/experiment.json`, and `.uxa-output/report.html`. Summary contains
per-run metrics/findings, per-cell aggregates, exact paired-seed directional
gates, and safe failure records. Report includes failed runs and staging crash
markers instead of omitting them.

Run policy ablations:

```powershell
uv run uxa ablate benchmarks/demo/project.yaml --experiment ablations --output .uxa-output --workers 1
uv run uxa ablate benchmarks/demo/project.yaml --experiment ablations --policy prominence-ranked-list --policy progressive-prominence --output .uxa-output --dry-run
```

Regenerate filesystem-openable replay and inspect one run:

```powershell
uv run uxa report .uxa-output --output report.html
uv run uxa inspect-run .uxa-output/runs/<run-id>
```

The report is static HTML. Open `report.html` directly; no report server is
required. `uxa report` only reads recorded bundles and never calls a model.

When `evaluation.report_synthesis.enabled: true`, `uxa run` automatically
attempts synthesis after finalized runs. Use `--no-synthesis` to skip that
attempt while keeping deterministic findings and the offline report. Run
`uxa synthesize PROJECT --experiment ID --output DIR` to create a new immutable
synthesis attempt from finalized evidence without rerunning the experiment.
Missing model configuration, transport failure, or invalid role output produces
an explicit unavailable or rejected attempt and keeps the deterministic report
available.

Run opt-in live endpoint compatibility test. This sends one coarse-scent, one
full-scent, and one cognitive request. Without the flag it skips.

```powershell
$env:UXA_RUN_LIVE_TESTS = "1"
uv run pytest tests/live/test_openai_endpoint.py -m live -q
```

## Commands

| Command | Purpose |
| --- | --- |
| `uxa version` | Print package version. |
| `uxa validate PROJECT` | Validate one combined project YAML and print SHA-256 config digest. |
| `uxa validate PROJECT --check-env` | Validate project and required model environment names without printing values. |
| `uxa fixture serve --host HOST --port PORT` | Serve bundled FastAPI fixture. Defaults: `127.0.0.1:8000`. |
| `uxa run PROJECT --experiment ID --output DIR` | Execute selected experiment. Defaults: `core-pair`, `.uxa-output`, one worker. |
| `uxa run-one PROJECT --scenario ID --version ID --persona ID --policy ID --seed N --output DIR` | Execute exactly one semantic cell. |
| `uxa run PROJECT --dry-run` | Print expanded matrix and estimated model calls without browser or model execution. |
| `uxa run PROJECT --no-synthesis` | Execute runs without the automatic report-synthesis attempt. |
| `uxa ablate PROJECT --experiment ID --policy POLICY` | Execute selected ablation policies. Repeat `--policy`; default experiment is `ablations`. |
| `uxa synthesize PROJECT --experiment ID --output DIR` | Synthesize a new immutable report attempt from finalized evidence. |
| `uxa report BUNDLE_ROOT --output FILE` | Render finalized bundles into static HTML. |
| `uxa inspect-run RUN_DIR` | Print terminal outcome, verification, claim, and artifact paths. |

Supported policies:
`full-list`, `prominence-ranked-list`, `progressive-prominence`, and
`progressive-prominence-scent`.

Useful run options are `--workers`, `--run-count`, `--resume`, `--dry-run`,
`--check-env`, and `--fixture-origin`. Fixture origin must be an HTTP(S) origin
without path, query, fragment, or credentials. `--resume` skips only selected
finalized bundles that pass integrity and terminal-structure validation.

## Configuration

`uxa` loads one combined YAML project. Current root fields:

```yaml
id: project-id
name: Project name
applications: []
scenarios: []
personas: []
experiments: []
```

Applications contain `id`, `name`, and `versions`. Every application must
provide versions with `id`, `kind` (`defective` or `improved`), and `label`.

Scenarios contain `id`, `name`, `goal`, `application_version_ids`, `start_state`,
`fixture_inputs`, `budget`, `verifier`, `safeguards`,
`eligible_persona_ids`, and `expected_evidence`. Fixture inputs have `value` and
optional `sensitive: true`. Budgets require positive `max_steps`,
`max_observations`, and `max_interactions`. `timeout_seconds` is optional; null
means no overall run deadline. Finite values must be positive. Model, verifier,
fixture HTTP, and browser operations keep separate bounded safety timeouts.

Supported verifiers:

- `fixture-state`: `resource`, `field`, `operator`, and
  `expected_fixture_key`; operators are `equals`, `not-equals`, `contains`,
  `truthy`, and `falsy`.
- `visible-result`: `text` and optional `role`.

Personas contain `id`, `name`, positive `working_memory_capacity`, bounded
`initial_confidence`, `initial_frustration`, and `abandonment_threshold`, plus
positive `attention_temperature`.

Root `providers` config versions prominence weights/temperature and progressive
attention formula weights. `providers.expectation` enables versioned frozen
expectation documents; existing user files omit it and remain disabled by
default. Root `evaluation` config versions discovery-cost, finding-rule, and
state-update formulas. `evaluation.report_synthesis.enabled` enables the
post-run four-role report synthesis pipeline and its bounded retrieval,
adjudication, and verification settings. Persona attention temperature and
abandonment threshold override corresponding per-run policy values.

Experiments contain `id`, `name`, `scenario_ids`, `application_version_ids`,
`persona_ids`, `policies`, optional unique `seeds`, and positive `run_count`.
When `seeds` is empty, loader uses `0..run_count-1`. `--run-count N` replaces
that seed set with `0..N-1`.

Loader validates references and canonicalizes sorted JSON before computing the
configuration SHA-256 digest. The digest is part of every `RunSpec` and bundle
manifest.

## Demo Matrix

`core-pair` currently expands:

```text
8 full-list cells x 1 seed
+ 8 progressive-prominence-scent cells x 10 seeds
= 88 run specs
```

Its policies are `full-list` and `progressive-prominence-scent`. The default
dry-run estimate is 248 model calls for one attention cycle across the matrix:
one cognitive call per `full-list` run and three role calls per scent-guided
run (coarse scent, full scent, cognitive). The CLI also reports the maximum
logical model-call budget separately.

Deterministic list policies use only the first configured seed. Progressive
policies retain every configured seed because their attention selection is
seeded. Provider-side model sampling is not currently controlled by that seed.

`ablations` selects `prominence-ranked-list` and `progressive-prominence`.
Compared cells keep scenario, persona, seed, configuration, fixture state, and
model configuration aligned. Variant comparison pairs exact seeds and applies
the directional gate described in [evaluation docs](docs/domain-model.md).

## LLM modes

`UXA_LLM_MODE=api` is default. Choose transport with this mode matrix:

| Mode | Required configuration | Transport |
| --- | --- | --- |
| `api` | `UXA_LLM_MODE`, `UXA_LLM_BASE_URL`, `UXA_LLM_API_KEY`, `UXA_SCENT_MODEL`, `UXA_COGNITIVE_MODEL`; `UXA_REPORT_MODEL` when synthesis is enabled | OpenAI-compatible HTTP transport |
| `codex` | `UXA_LLM_MODE`, `UXA_SCENT_MODEL`, `UXA_COGNITIVE_MODEL`, logged-in Codex CLI; report model when synthesis is enabled | `codex exec` subprocess transport |

Codex mode is opt-in. Codex must already be installed, logged in, and available
as `codex` on `PATH`. Account mode does not read or print credentials.

## Documentation

- [Architecture](docs/architecture.md)
- [Domain model](docs/domain-model.md)
- [Run bundle format](docs/run-bundle-format.md)
- [Model provider](docs/model-provider.md)
- [Report synthesis and provider boundary](docs/model-provider.md#report-synthesis-contract)
- [Security](docs/security.md)
- [Roadmap and deferred contracts](docs/roadmap.md)
- [Original POC plan versus current implementation](docs/poc-plan-vs-current.md)
- [Testing](docs/testing.md)
- [Glossary](docs/glossary.md)

## Verification

Non-live verification commands:

```powershell
uv run uxa validate benchmarks/demo/project.yaml
uv run pytest -m "not live" -q
uv run ruff check .
uv run pyright
git diff --check
```

Live tests are opt-in and must never be treated as required non-live CI.
