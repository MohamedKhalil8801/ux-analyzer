# Model Provider

## Environment Contract

`UXA_LLM_MODE` selects LLM transport and defaults to `api`:

| Mode | Required configuration | Transport |
| --- | --- | --- |
| `api` | mode, base URL, API key, scent model, cognitive model; report model when synthesis is enabled | OpenAI-compatible HTTP |
| `codex` | mode, scent model, cognitive model, logged-in Codex CLI; report model when synthesis is enabled | `codex exec` subprocess |

API mode uses these environment names:

```text
UXA_LLM_MODE          mode selector; defaults to api
UXA_LLM_BASE_URL       HTTP(S) base URL without credentials
UXA_LLM_API_KEY        secret sent as Bearer authorization header
UXA_SCENT_MODEL        model ID for coarse and full scent roles
UXA_COGNITIVE_MODEL    model ID for cognitive role
UXA_REPORT_MODEL       model ID for the four report-synthesis roles
UXA_REDESIGN_MODEL     optional model ID for the two redesign roles; falls
                       back to UXA_REPORT_MODEL when unset
UXA_LLM_TIMEOUT_SECONDS
                       positive model-call timeout in seconds; defaults to 30.
                       none, off, or unlimited disables the model-call timeout
UXA_LLM_MAX_CONCURRENT_CALLS
                       client-side model-call concurrency cap; defaults to 2
UXA_LLM_REQUEST_MAX_BYTES
                       serialized request byte ceiling; defaults to 750000
                       (clamped to at least 100000)
UXA_LLM_SESSION_ID     optional provider session identifier sent with each call
UXA_LLM_REPORT_REASONING_EFFORT
                       optional report effort: none, minimal, low, medium,
                       high, xhigh, or max
```

Use placeholders in configuration examples:

```powershell
$env:UXA_LLM_MODE = "api"
$env:UXA_LLM_BASE_URL = "https://<provider-host>/v1"
$env:UXA_LLM_API_KEY = "<api-key>"
$env:UXA_SCENT_MODEL = "<scent-model-id>"
$env:UXA_COGNITIVE_MODEL = "<cognitive-model-id>"
$env:UXA_REPORT_MODEL = "<report-model-id>"
$env:UXA_REDESIGN_MODEL = "<redesign-model-id>"
```

Codex mode is opt-in. Set `UXA_LLM_MODE=codex`, keep the two model variables,
and install and log in to Codex before running. The `codex` executable must be
available on `PATH`. Account mode uses the logged-in CLI and does not read or
print account credentials.

In API mode, the adapter posts to `<base-url>/chat/completions` with JSON. It
records only endpoint origin, never URL credentials or API key.
`uxa validate --check-env` checks presence without printing values. Non-dry
`uxa run` requires the settings for selected mode. Dry-run matrix expansion can
run without them unless `--check-env` is also supplied.

## Report Synthesis Contract

Report synthesis is a post-run, evidence-grounded use case. `UXA_REPORT_MODEL`
selects the model used by the analyst, evidence auditor, pattern reviewer, and
adjudicator. The model must support the repository's structured JSON schemas.
It must also accept `image/png` and `image/jpeg` inputs when the evidence room
contains screenshot or heatmap attachments. A model without the required vision
input support makes that synthesis attempt unavailable; it does not turn an
image estimate into a deterministic fact.

The four roles receive fresh, isolated message tuples. They share only the
redacted evidence-room manifest, frozen expectations, UX principles as
interpretive context, and explicitly retrieved allowlisted evidence. They do
not receive run-agent chat, hidden DOM facts, selectors, private reasoning,
raw prior role responses, or existing report prose. The analyst proposes
candidates, the evidence auditor checks references and counterevidence, the
pattern reviewer checks cross-surface and recurrence claims, and the
adjudicator decides publication and severity.

Retrieval is bounded. Defaults are three retrieval rounds, a 32-entry per-role
request cap resolved in 16-entry batches, and 16 MiB of cumulative
attachments. Project settings
can bound retrieval rounds from 1 to 5, adjudication revisions from 0 to 2,
and final verification passes from 1 to 2. Evidence references remain tied to
recorded run, viewport, element, event, metric, replay, and artifact identity.

Enabled projects attempt synthesis automatically after `uxa run` finalizes its
runs. `uxa run ... --no-synthesis` skips the attempt and still renders the
deterministic report. `uxa synthesize PROJECT --experiment ID --output DIR`
creates a new immutable synthesis attempt from finalized bundles. Missing
configuration, unavailable transport, schema failure, rejected candidates, or
unresolved objections are recorded as `unavailable` or `rejected`; deterministic
findings and replay remain available through fallback behavior. A successful
attempt is `accepted` or `no-issues`.

`uxa report` is intentionally offline. It reads stored bundles and immutable
synthesis artifacts, resolves only independently verifiable evidence links, and
never calls a model. The report can therefore be regenerated without network
access or credentials.

## Creative Redesign Contract

The creative redesign pipeline (ADR 0007) is deliberately separate from report
synthesis. It never reads the run evidence corpus and produces no verified
claims: its output is a set of design proposals that are model estimates about
captured pages. `UXA_REDESIGN_MODEL` selects the model for the two isolated
roles; when unset, both roles use `UXA_REPORT_MODEL`, so one vision-capable
structured-output model can serve everything.

- **Redesign proposer** receives the segmented page captures (images attached
  in order with y-offsets), each page's trimmed DOM/copy inventory, and the
  published redesign principle pack. It returns proposals with a page URL,
  category, observation, rationale, concrete change, principle IDs, impact,
  effort, section references into the capture, and a per-page understanding
  (intent and audience, always labeled an inference).
- **Redesign critic/merger** receives the proposer's candidates plus the same
  pack and captures digest. It kills or merges conflicting proposals with
  recorded reasons, adds cross-page consistency notes, and emits the final
  proposal set. Deliberate-choice checks are mandatory for grouping,
  unification, simplification, and relocation proposals so intentional design
  is not flagged as an accident.

Both roles use fresh, isolated message tuples and the same transport behavior
as the report roles (structured tool-call responses in API mode, evidence-file
serialization under Codex). Deterministic validation enforces the proposal
schema, principle-ID membership, unique IDs, and section references that resolve
into the capture; violations make the attempt `rejected` or `unavailable` and
are recorded with reasons. Attempts are immutable under
`<output>/redesign/<attempt-id>/` and carry a `captures_digest` binding the
proposal set to the exact capture it read; the attempt files and schemas are in
[output formats](output-formats.md). Impact and effort are model
estimates stamped as such in the report; they are never presented as measured
facts.

## Saliency Execution Providers

Foveacast runs three model sessions in fixed `1s`, `3s`, `7s` order. CPU is
available on every supported platform. DirectML is optional and Windows-only;
install exactly one runtime extra because both packages provide the
`onnxruntime` import:

```powershell
uv sync --extra saliency-cpu
# or, on Windows:
uv sync --extra saliency-directml
```

The saliency execution preference is one of `cpu`, `directml`, or `auto`:

| Preference | Behavior |
| --- | --- |
| `cpu` | Use `CPUExecutionProvider`. |
| `directml` | Require Windows and an available `DmlExecutionProvider`; initialization failure stops inference. |
| `auto` | Try DirectML on Windows when available, then fall back to CPU with a recorded reason. |

Foveacast never downloads models or runtimes during inference. Model artifacts
must be installed explicitly through model management. DirectML sessions use
sequential execution and disable ONNX Runtime memory-pattern optimization;
the three sessions are never run concurrently.

The adapter computes requested provider preference, actual provider, fallback
reason, DirectML adapter/device ID when exposed by ONNX Runtime, session options,
and cold-load timing. Current persisted saliency evidence is narrower: each
duration records actual execution provider, model/version/checksum, input/output
and geometry metadata, preprocessing version, per-duration inference timing,
cache state, and warnings; cache and bundle metadata also retain aggregate
provenance and artifact checksums. Requested preference, adapter/device ID,
session options, and cold-load timing are currently unavailable/not persisted.
They are review triggers, not fields this document infers. CPU behavior and
deterministic interface snapshots remain unchanged.

## Saliency validation status

Status: historical Task 14 record, superseded by the 2026-09-07 review addendum
in
[`docs/adr/0001-provisional-saliency-provider.md`](adr/0001-provisional-saliency-provider.md).

Foveacast model management is explicit. Inference never downloads a model or
runtime. Task 14 ran:

```text
rtk uv run uxa models install foveacast-v0.2.0 --precision fp16
rtk uv run uxa models status foveacast-v0.2.0
```

Install returned success and downloaded all six pinned model/parity artifacts.
Initial Task 14 status returned `runtime missing` with diagnostic `install
saliency-cpu or saliency-directml extra`. The autonomous dependency attempt
`rtk uv sync --extra saliency-cpu` returned success and installed
`onnxruntime==1.28.0`; subsequent CPU status returned `ready`, while DirectML
status returned `unsupported provider` with available providers
`AzureExecutionProvider, CPUExecutionProvider`. The complete artifact table,
SHA-256 values, release tag, and license chain are recorded in
[`docs/adr/0001-provisional-saliency-provider.md`](adr/0001-provisional-saliency-provider.md).

The real CPU known-screenshot test now produces stable finite maps for all three
durations after the adapter accepted symbolic spatial dimensions. This proves
adapter/runtime behavior for one pinned input only. It does not establish model
quality, sequential warm latency budget, peak RSS, output parity
budget, or real cache measurement. Fake runtime tests verify orchestration and
provider-selection contracts only.

The focused real experiment has since run: ADR 0001's 2026-09-07 addendum
records three live focused comparison rounds against a real portfolio site, the
final one with 8/8 finalized runs, 8/8 valid UX samples, 8/8 verified success,
and zero saliency fallbacks. DirectML is still unavailable, no real latency/RSS
budget has been supplied, and promotion remains gated.
`uxa validate --check-env` separately reported configured LLM
settings; no real endpoint result is claimed. The deterministic Task 13 fake
path passed and remains separate from real promotion evidence. Foveacast
fallback preserves normal UX execution through
`heuristic-prominence-v1`, records a sanitized reason, and invalidates the
learned comparison sample. A fallback is not a valid learned result.

Numeric maps and scores remain model-estimate evidence. They are not sent to
the cognitive role. Source overlays remain subject to existing redaction; when
redacted, reports retain heatmap-only artifacts and geometry/ranking evidence.
No human calibration or real-user claim follows synthetic or model output.

Current default remains `heuristic-prominence-v1`. Foveacast is explicit
opt-in pending the conditional review in the ADR. DirectML AMD parity and
stability remain a separate optional hardware gate; no unavailable adapter or
device result is inferred.

DirectML parity and stability coverage is hardware-marked and skips when
Windows, `onnxruntime-directml`, or local model files are unavailable. It never
downloads artifacts. It resolves artifacts through `ModelRegistry`, so install
them first with `uxa models install foveacast-v0.2.0 --precision fp16`; point
`UXA_MODEL_HOME` at a non-default model root when the local release lives
elsewhere.

## Three Separate Roles

| Role | Prompt/schema | Input boundary | Output |
| --- | --- | --- | --- |
| `coarse-scent` | `scent-coarse-v1.txt`, `scent-coarse-v1` | Goal plus visible element ID, role, label, region label, actionability. Glance-level only. | Scores in `[0, 1]` for listed elements. |
| `full-scent` | `scent-full-v1.txt`, `scent-full-v1` | Goal plus same visible meaning and disabled state for already noticed elements in current viewport. | Scores in `[0, 1]` for noticed elements only. |
| `cognitive` | `cognitive-v3.txt`, `cognitive-v1` | Goal plus newly revealed and remembered sighted-visible elements and optional rendered region label. | One typed action: inspect, interact, type-fixture, scroll, wait, back, complete, or abandon. |

Roles remain separate even when endpoint and model IDs match. Each has its own
prompt version, schema version, `ModelRole`, manifest, model call record, retry
records, and usage record. The current `progressive-prominence-scent` policy
uses all three. Other policies use cognitive calls without scent calls.

Production creates one audit client per run, even when HTTP transport is shared.
Each sanitized role call is appended to `timeline.jsonl` as
`model-call-recorded`; role-specific prompt/schema manifests are stored in the
bundle manifest and terminal run state.

## Forbidden Exposure

No role receives selectors, test IDs, hidden labels, destination URLs, provider
IDs, execution references, internal verifier state, numeric prominence scores,
or numeric scent scores. The cognitive role also does not receive bounds or
visibility fractions. It chooses only IDs present in its supplied observation.

Sighted-mode element labels use pixel-visible rendered text only. CSS-clipped,
offscreen, transparent, zero-font-size, and screen-reader-only text is excluded.
For input, select, and textarea controls, associated `label` text is included
only when that label is visibly painted in the viewport. Accessibility names
remain private semantic facts and are not substituted into cognitive labels.
Model-facing region labels use a visible heading or a generic rendered kind such
as `Navigation`; aria-only region names stay private.

The cognitive model cannot decide official success. For visible-result tasks it
must explicitly propose `complete` after observing enough visible evidence.
Only that action invokes independent verification; ordinary successful actions
and terminal verification cannot promote the outcome. A failed completion check
is recorded and the bounded run continues or ends non-successfully. Application
validation enforces current viewport, noticed and remembered targets,
actionability, disabled state, budgets, and fixture-key resolution before
execution. `complete` has no element ID and consumes one step without a browser
interaction. Fixture-state tasks retain automatic after-action verification.

Typed input action text is a scenario fixture key, not a secret invented by the
model. `action_validation` replaces the key with the scenario-configured value
at the application boundary. Sensitive values are redacted from persisted model
records and bundle JSON.

## Structured HTTP Contract

The client sends:

```text
POST <UXA_LLM_BASE_URL>/chat/completions
Authorization: Bearer <UXA_LLM_API_KEY>
Content-Type: application/json
```

Strict mode requests JSON Schema response format. If provider returns a 400
that identifies unsupported `response_format`, `json_schema`, `strict`, or
`unsupported` behavior, client retries the logical call in `json_object` mode
and validates the returned JSON locally with Pydantic.

Redesign roles (ADR 0007) share the report-role transport behavior: they start
in tool-call mode, degrade tool-call → strict → json-object when a response
fails local validation (not only on coded 400s), reuse the tolerant report
JSON recovery parser in degraded modes, and re-anchor the output contract with
an explicit final user message.

Accepted structured responses must contain one JSON object matching the role
schema. Unknown element IDs, duplicate score IDs, and full-scent scores for
unnoticed elements are rejected by role providers.

## Retry and Failure Rules

Default `RetryPolicy` is:

```text
max_attempts = 3
base_delay_seconds = 0.25
max_delay_seconds = 2.0
multiplier = 2.0
```

Retries are bounded and recorded for transport errors, rate limits, server
errors, and invalid structured output. Safety rejection and authentication
failure are not retried. Exhaustion raises model failure and leaves a sanitized
record with role, model, prompt digest, schema version, attempts, latency, token
usage, request, response, and retry events.

HTTP rate-limit retries honor the provider's `Retry-After` header. Model calls
share a concurrency limiter controlled by `UXA_LLM_MAX_CONCURRENT_CALLS`
(default `2`); experiment `--workers` controls run concurrency, not unrestricted
model request concurrency.

In Codex mode, role-specific reasoning settings are forwarded as
`codex exec -c model_reasoning_effort=<value>`. This overrides the user's Codex
profile for each structured call, so `UXA_LLM_SCENT_REASONING_EFFORT` and
`UXA_LLM_COGNITIVE_REASONING_EFFORT` reflect actual execution.

Request/response records pass through recursive sanitization. It removes known
secret keys, bearer values, API key, and configured exact fixture values. Logs
record role, model, endpoint origin, and attempt count only.

## Live Compatibility Test

Live calls are opt-in. API mode requires `UXA_RUN_LIVE_TESTS=1` plus its mode,
base URL, API key, and two model variables. Codex mode requires the same live
gate, `UXA_LLM_MODE=codex`, two model variables, and an installed, logged-in
Codex CLI. Codex mode does not read or print account credentials.

API mode runs:

```powershell
uv run pytest tests/live/test_openai_endpoint.py -m live -q
```

Both modes use this test's same structured-role assertions: one coarse-scent,
one full-scent, and one cognitive call; structured outputs, role order, endpoint
origin, and non-empty sanitized records. It does not record live response
fixtures automatically.
