# Getting started

`uxa` drives a real Chromium browser through user tasks, records what a simulated
persona actually saw and did, verifies the outcome independently of the model's
own claim, and turns that record into a static HTML report you can open in a
browser and check by hand.

## What the tool does

- Crawls a live site, proposes task scenarios from what is actually on the
  pages, and lets you curate them in a local review UI
  ([`uxa explore`](exploration.md)).
- Executes each curated scenario against the live site in a real browser,
  revealing the interface progressively to a simulated persona rather than
  handing the model the whole DOM
  ([`uxa run`](commands.md#uxa-run)).
- Records every viewport, observation, action, model call, and verification
  result into an immutable run bundle, then renders a self-contained
  `report.html` ([output formats](output-formats.md)).
- Runs an evidence-grounded report-synthesis pass that turns the recorded
  evidence into plain-language findings, or into a valid "no issues" result
  ([`uxa synthesize`](commands.md#uxa-synthesize)).
- Proposes design changes as clearly labeled model estimates, separate from
  findings ([redesign](redesign.md)).
- Audits page performance and AI-design slop for any URL
  ([`uxa pagespeed`](commands.md#uxa-pagespeed),
  [`uxa slop`](commands.md#uxa-slop)).

## What the tool does not claim

`uxa` runs a simulation, not a user study. Its output is benchmark evidence
produced by a configured model, under a configured persona, policy, and
prominence provider.

- Completion, satisfaction, emotion, and abandonment numbers describe the
  simulation, not real people.
- Design proposals from [`uxa redesign`](redesign.md) are model estimates. They
  carry no verification status and are never presented as findings.
- When page capture is capped, content below the cap is not captured and not
  shown to the model. The report discloses this.
- When a scenario itself is broken — a verifier anchor that cannot resolve on
  any reachable page, a self-contradictory goal — the run is marked invalid
  rather than converted into a product claim.
- "No issues" is a valid outcome. The tool does not manufacture a finding to
  fill an empty report.

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync
uv run playwright install chromium
```

`uv sync` installs the package and its dependencies. `playwright install
chromium` downloads the browser build the tool drives; without it, `uxa run`,
`uxa explore`, `uxa slop`, and `uxa pagespeed --web-ui` all fail when they try
to launch a browser. See [troubleshooting](troubleshooting.md#playwright-or-chromium-is-missing).

Confirm the CLI is wired up:

```powershell
uv run uxa version
```

## Configure a model provider

`uxa` needs a model for three separate job families:

- **scent** — how strongly a noticed control suggests it leads to the goal
- **cognitive** — the reasoning agent that picks the next action
- **report** — the four report-synthesis roles
- **redesign** — the two design-proposal roles; falls back to the report model

### API mode (default)

```powershell
$env:UXA_LLM_MODE          = "api"
$env:UXA_LLM_BASE_URL      = "https://<provider-host>/v1"
$env:UXA_LLM_API_KEY       = "<api-key>"
$env:UXA_SCENT_MODEL       = "<scent-model-id>"
$env:UXA_COGNITIVE_MODEL   = "<cognitive-model-id>"
$env:UXA_REPORT_MODEL      = "<report-model-id>"
```

Requests go to `<UXA_LLM_BASE_URL>/chat/completions` with the API key in an
`Authorization: Bearer` header. The base URL must be HTTP(S) and must not embed
credentials; a URL with a username or password is rejected.

### Model environment variables

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `UXA_LLM_MODE` | no | `api` | Transport: `api` or `codex`. |
| `UXA_LLM_BASE_URL` | `api` mode | — | OpenAI-compatible base URL, no credentials. |
| `UXA_LLM_API_KEY` | `api` mode | — | Bearer token for the endpoint. |
| `UXA_SCENT_MODEL` | yes | — | Model id for the two scent roles. |
| `UXA_COGNITIVE_MODEL` | yes | — | Model id for the cognitive agent. |
| `UXA_REPORT_MODEL` | when report synthesis is enabled | — | Model id for the four report roles. |
| `UXA_REDESIGN_MODEL` | no | `UXA_REPORT_MODEL` | Model id for the two redesign roles. |
| `UXA_LLM_TIMEOUT_SECONDS` | no | `30` | Per-request timeout. `none`, `off`, or `unlimited` disables it. |
| `UXA_LLM_MAX_CONCURRENT_CALLS` | no | `2` | Client-side model-call concurrency cap. |
| `UXA_LLM_REQUEST_MAX_BYTES` | no | `750000` | Serialized request ceiling; values below `100000` are clamped up. |
| `UXA_LLM_SESSION_ID` | no | unset | Provider session identifier sent with each call. |
| `UXA_LLM_SCENT_REASONING_EFFORT` | no | unset | Scent reasoning effort. |
| `UXA_LLM_COGNITIVE_REASONING_EFFORT` | no | unset | Cognitive reasoning effort. |
| `UXA_LLM_REPORT_REASONING_EFFORT` | no | unset | Report and redesign reasoning effort. |

Reasoning effort accepts one of `none`, `minimal`, `low`, `medium`, `high`,
`xhigh`, `max`.

The scent and cognitive model ids may be the same value; the roles stay
separate and keep their own prompts, schemas, and manifests either way. Use a
structured-output, vision-capable model for `UXA_REPORT_MODEL` whenever the
evidence corpus contains screenshots or heatmaps.

### A `.env` file works too

Values are read from the process environment first, then from a `.env` file in
the current working directory. Existing environment values are never
overridden. [`.env.example`](../.env.example) lists the same names with
placeholders.

### Keep keys out of files you commit

API keys belong in the environment or a secret manager. Never put a real key in
a project YAML, in source, in a test, or in documentation — the loader hashes
the whole project file into the config digest, so a key placed there is also
written into every run manifest. [Security](security.md) has the full handling
rules.

### Codex mode

`UXA_LLM_MODE=codex` swaps the HTTP client for the locally installed `codex`
CLI. It is an alternative transport for people who already have a Codex
subscription, not a different analysis mode.

```powershell
$env:UXA_LLM_MODE        = "codex"
$env:UXA_SCENT_MODEL     = "<scent-model-id>"
$env:UXA_COGNITIVE_MODEL = "<cognitive-model-id>"
$env:UXA_REPORT_MODEL    = "<report-model-id>"
```

Prerequisites:

- the `codex` executable is installed and on `PATH`
- it is already logged in
- `UXA_SCENT_MODEL` and `UXA_COGNITIVE_MODEL` are still set, plus
  `UXA_REPORT_MODEL` when report synthesis is on

`UXA_LLM_BASE_URL` and `UXA_LLM_API_KEY` are not read in codex mode. The tool
never reads or prints Codex account credentials; authentication comes entirely
from the logged-in CLI. Reasoning-effort variables are forwarded to the CLI as
`codex exec -c model_reasoning_effort=<value>`, so they affect real execution
rather than being silently dropped.

Check the configuration without printing any value:

```powershell
uv run uxa validate path\to\project.yaml --check-env
```

## First analysis of a live site, end to end

Live analysis needs an HTTPS target. Starting URLs are rejected if they are not
HTTPS.

### 1. Crawl and curate scenarios

```powershell
uv run uxa explore --starting-url https://example.com --depth 2
```

This crawls same-origin pages from the starting URL, asks the cognitive model
to propose candidate scenarios from the visible labels it found, and opens a
local review UI at `http://127.0.0.1:<port>/__explore/`.

In the review UI you can accept, edit, duplicate, delete, or add scenarios; pick
which personas the experiment uses; and preview the generated YAML before
saving. When you click **Save & Continue**, the curated set is validated,
persisted, and written out as a runnable project.

Skip the UI entirely with `--auto-accept`; the curated set is then identical to
the suggested set. Use `--dry-run` first if you just want to see the crawl
matrix and a token estimate without launching a browser or a model.

By default the workspace is `.uxa-output/explore/<site-slug>/`. It contains a
runnable `project.yaml` at its root.

### 2. Run the curated scenarios

```powershell
uv run uxa run .uxa-output\explore\example\project.yaml --experiment exploration-run --workers 2
```

Each run drives a real browser session, records evidence, and finalizes an
immutable run bundle under `reports/exploration-generated/runs/`. If the browser
is blocked loading a third-party font or script origin, the tool asks once which
origins to allow. In a non-interactive shell nothing is granted implicitly and
the run safety-blocks instead; pass `--allow-origin https://cdn.example` (repeatable)
or `--yes` to grant them up front.

Interrupted experiments resume without redoing finished work:

```powershell
uv run uxa run .uxa-output\explore\example\project.yaml --experiment exploration-run --workers 2 --resume
```

`--resume` skips only finalized bundles that pass checksum and terminal-structure
validation; anything interrupted or half-written is marked and re-run.

### 3. Open the report

```powershell
uv run uxa report reports\exploration-generated
```

That renders `reports/exploration-generated/report.html`. Open it directly in a
browser — it is a self-contained static file with inline CSS, JavaScript, and
embedded images. No server and no network are required.

The **Page checks** and **Page speed** views refetch `ux-audit.json` and
`pagespeed.json` at page load, which browsers block on `file://` URLs. If you
want those two views live, serve the directory instead:

```powershell
uv run uxa report reports\exploration-generated --serve
```

The server binds `127.0.0.1` only.

## Optional: model-estimate design proposals

```powershell
uv run uxa redesign reports\exploration-generated
```

This reads the persisted `page-capture.json` from the run, sends it to the
proposer and critic/merger roles, and publishes an immutable attempt under
`reports/exploration-generated/redesign/`. The report's **Design proposals**
view then lists them, labeled as model estimates. Details in
[redesign](redesign.md).

## Optional: hand the fixes to an agent

```powershell
uv run uxa export --report reports\exploration-generated --all
```

This writes a self-contained markdown fix package — issues, embedded evidence,
fix options, skill references, and the fixer workflow — under
`.uxa-output/fix-export-<timestamp>/`.

## Next steps

| Document | Contents |
| --- | --- |
| [Commands](commands.md) | Every `uxa` command, its options, and a worked example. |
| [Configuration](configuration.md) | Project YAML schema and every environment variable. |
| [Exploration](exploration.md) | The scenario-discovery workflow in detail. |
| [Redesign](redesign.md) | The design-proposal pipeline and its limits. |
| [Output formats](output-formats.md) | Run bundle layout, sidecars, and how to read them. |
| [Security](security.md) | What reaches the model provider and how to handle keys. |
| [Troubleshooting](troubleshooting.md) | Concrete failure modes and what to do about them. |
| [Glossary](glossary.md) | Vocabulary used throughout. |
