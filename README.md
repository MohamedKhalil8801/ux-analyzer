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
```

`UXA_LLM_BASE_URL` must be an HTTP(S) URL without credentials. Adapter posts to
`<base-url>/chat/completions`. Scent and cognitive model IDs may be equal, but
roles remain separate.

Run core benchmark:

```powershell
uv run uxa run benchmarks/demo/project.yaml --experiment core-pair --output .uxa-output --workers 1
```

Completed execution writes immutable run bundles, `.uxa-output/experiment.json`,
and `.uxa-output/report.html`. Summary contains per-run metrics/findings,
per-cell aggregates, and exact paired-seed directional gates.

Run policy ablations:

```powershell
uv run uxa ablate benchmarks/demo/project.yaml --experiment ablations --output .uxa-output --workers 1
uv run uxa ablate benchmarks/demo/project.yaml --experiment ablations --policy prominence-ranked-list --policy progressive-prominence --output .uxa-output --dry-run
```

Regenerate filesystem-openable replay and inspect one finalized run:

```powershell
uv run uxa report .uxa-output --output report.html
uv run uxa inspect-run .uxa-output/runs/<run-id>
```

The report is static HTML. Open `report.html` directly; no report server is
required.

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
| `uxa run PROJECT --dry-run` | Print expanded matrix and estimated model calls without browser or model execution. |
| `uxa ablate PROJECT --experiment ID --policy POLICY` | Execute selected ablation policies. Repeat `--policy`; default experiment is `ablations`. |
| `uxa report BUNDLE_ROOT --output FILE` | Render finalized bundles into static HTML. |
| `uxa inspect-run RUN_DIR` | Print terminal outcome, verification, claim, and artifact paths. |

Supported policies:
`full-list`, `prominence-ranked-list`, `progressive-prominence`, and
`progressive-prominence-scent`.

Useful run options are `--workers`, `--run-count`, `--dry-run`, `--check-env`,
and `--fixture-origin`. Fixture origin must be an HTTP(S) origin without path,
query, fragment, or credentials.

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
`max_observations`, `max_interactions`, and `timeout_seconds`.

Supported verifiers:

- `fixture-state`: `resource`, `field`, `operator`, and
  `expected_fixture_key`; operators are `equals`, `not-equals`, `contains`,
  `truthy`, and `falsy`.
- `visible-result`: `text` and optional `role`.

Personas contain `id`, `name`, positive `working_memory_capacity`, bounded
`initial_confidence`, `initial_frustration`, and `abandonment_threshold`, plus
positive `attention_temperature`.

Root `providers` config versions prominence weights/temperature and progressive
attention formula weights. Root `evaluation` config versions discovery-cost,
finding-rule, and state-update formulas. Persona attention temperature and
abandonment threshold override corresponding per-run policy values.

Experiments contain `id`, `name`, `scenario_ids`, `application_version_ids`,
`persona_ids`, `policies`, optional unique `seeds`, and positive `run_count`.
When `seeds` is empty, loader uses `0..run_count-1`. `--run-count N` replaces
that seed set with `0..N-1`.

Loader validates references and canonicalizes sorted JSON before computing the
configuration SHA-256 digest. The digest is part of every `RunSpec` and bundle
manifest.

## Demo Matrix

`core-pair` expands:

```text
2 scenarios x 2 application versions x 2 personas
x 2 policies x 10 seeds = 160 run specs
```

Its policies are `full-list` and `progressive-prominence-scent`. The default
dry-run estimate is 320 model calls: one cognitive call per `full-list` run and
three role calls per scent-guided run (coarse scent, full scent, cognitive).

`ablations` selects `prominence-ranked-list` and `progressive-prominence`.
Compared cells keep scenario, persona, seed, configuration, fixture state, and
model configuration aligned. Variant comparison pairs exact seeds and applies
the directional gate described in [evaluation docs](docs/domain-model.md).

## Documentation

- [Architecture](docs/architecture.md)
- [Domain model](docs/domain-model.md)
- [Run bundle format](docs/run-bundle-format.md)
- [Model provider](docs/model-provider.md)
- [Security](docs/security.md)
- [Roadmap and deferred contracts](docs/roadmap.md)
- [Testing](docs/testing.md)

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
