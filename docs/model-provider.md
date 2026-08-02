# Model Provider

## Environment Contract

OpenAI-compatible settings come from exactly these required names:

```text
UXA_LLM_BASE_URL       HTTP(S) base URL without credentials
UXA_LLM_API_KEY        secret sent as Bearer authorization header
UXA_SCENT_MODEL        model ID for coarse and full scent roles
UXA_COGNITIVE_MODEL    model ID for cognitive role
```

Use placeholders in configuration examples:

```powershell
$env:UXA_LLM_BASE_URL = "https://<provider-host>/v1"
$env:UXA_LLM_API_KEY = "<api-key>"
$env:UXA_SCENT_MODEL = "<scent-model-id>"
$env:UXA_COGNITIVE_MODEL = "<cognitive-model-id>"
```

The adapter posts to `<base-url>/chat/completions` with JSON. It records only
endpoint origin, never URL credentials or API key. `uxa validate --check-env`
checks presence without printing values. Non-dry `uxa run` requires all four
settings. Dry-run matrix expansion can run without them unless `--check-env` is
also supplied.

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

Inference metadata records requested and actual provider, fallback reason,
DirectML adapter/device ID when exposed by ONNX Runtime, and session options.
These fields belong to the model-evidence manifest; CPU behavior and
deterministic interface snapshots remain unchanged.

DirectML parity and stability coverage is hardware-marked and skips when
Windows, `onnxruntime-directml`, or local model files are unavailable. It never
downloads artifacts. Set `UXA_FOVEACAST_MODEL_1S`, `UXA_FOVEACAST_MODEL_3S`,
and `UXA_FOVEACAST_MODEL_7S` to existing files before running the marked test.

## Three Separate Roles

| Role | Prompt/schema | Input boundary | Output |
| --- | --- | --- | --- |
| `coarse-scent` | `scent-coarse-v1.txt`, `scent-coarse-v1` | Goal plus visible element ID, role, label, region label, actionability. Glance-level only. | Scores in `[0, 1]` for listed elements. |
| `full-scent` | `scent-full-v1.txt`, `scent-full-v1` | Goal plus same visible meaning and disabled state for already noticed elements in current viewport. | Scores in `[0, 1]` for noticed elements only. |
| `cognitive` | `cognitive-v1.txt`, `cognitive-v1` | Goal plus newly revealed and remembered persona-visible elements and optional region label. | One typed action: inspect, interact, scroll, wait, back, or abandon. |

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

The cognitive model cannot decide official success. It can propose an action and
reason string. Application validation enforces current viewport, noticed and
remembered target, actionability, disabled state, budgets, and fixture-key
resolution before execution.

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

Request/response records pass through recursive sanitization. It removes known
secret keys, bearer values, API key, and configured exact fixture values. Logs
record role, model, endpoint origin, and attempt count only.

## Live Compatibility Test

Live calls are opt-in. Set `UXA_RUN_LIVE_TESTS=1` plus the four required model
variables, then run:

```powershell
uv run pytest tests/live/test_openai_endpoint.py -m live -q
```

The test makes one coarse-scent, one full-scent, and one cognitive call; checks
structured outputs, role order, endpoint origin, and non-empty sanitized records.
It does not record live response fixtures automatically.
