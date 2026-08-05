# Codex Subscription LLM Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let every model-backed UX Analyzer command switch from API-key HTTP transport to a logged-in Codex subscription account without changing task-specific model selection or structured-call behavior.

**Architecture:** Keep `OpenAICompatibleSettings` and `StructuredModelClient` as the configuration and application boundaries. Add a validated `UXA_LLM_MODE` field, retain the current HTTP client for `api` mode, and add a small Codex subprocess transport that shares record, retry, redaction, schema-validation, and failure mechanics. Select the client once at the existing CLI execution factory so normal runs and live tests use the same mode.

**Tech Stack:** Python 3.12, existing `asyncio`, `tempfile`, `json`, `Pydantic`, `pytest`, `httpx`, Typer, and the installed Codex CLI. No runtime dependency additions.

## Global Constraints

- Preserve `UXA_LLM_MODE=api` as the default.
- Preserve API mode's existing `UXA_LLM_BASE_URL`, `UXA_LLM_API_KEY`, `UXA_SCENT_MODEL`, and `UXA_COGNITIVE_MODEL` contract.
- In `codex` mode, require only existing model variables; do not add model environment names.
- Use literal `codex` from `PATH`; do not inspect executable locations or Codex account files.
- Use one ephemeral, read-only `codex exec` process per logical model call.
- Keep role prompts, role model mapping, JSON Schemas, local Pydantic validation, retries, redaction, records, and failure semantics.
- Record sanitized provider origin `codex-cli` for Codex calls.
- Keep live tests opt-in through `UXA_RUN_LIVE_TESTS=1`.
- Do not add browser automation, API-key fallback, account-token handling, or new provider SDKs.

---

## File Map

- Modify `src/ux_analyzer/adapters/openai.py`: mode-aware settings, shared structured-call support, and `CodexStructuredClient`; retain existing HTTP payload behavior.
- Modify `src/ux_analyzer/cli.py`: select the client through one factory, accept the shared client protocol, and print mode-specific readiness output.
- Modify `tests/unit/test_environment.py`: mode validation and mode-specific required variables.
- Modify `tests/integration/models/test_openai_compatible.py`: mocked Codex command, schema output, retry, record, and failure coverage.
- Modify `tests/integration/cli/test_commands.py`: Codex readiness output and normal API compatibility.
- Modify `tests/live/test_openai_endpoint.py`: select API or Codex client while preserving existing role assertions.
- Modify `.env.example`: document `UXA_LLM_MODE=api`.
- Modify `README.md`, `docs/model-provider.md`, and `docs/testing.md`: document mode selection and commands after implementation.
- Do not modify `ports.models.StructuredModelClient`; both transports already fit its `complete` contract.

## Interfaces

Add these exact settings and factory contracts without renaming existing public classes:

```python
@dataclass(frozen=True, slots=True, repr=False)
class OpenAICompatibleSettings:
    base_url: str
    api_key: str = field(repr=False)
    scent_model: str
    cognitive_model: str
    mode: Literal["api", "codex"] = "api"
    scent_reasoning_effort: str | None = None
    cognitive_reasoning_effort: str | None = None
    timeout_seconds: float = 30.0
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    redaction_values: tuple[str, ...] = ()

    @property
    def endpoint_origin(self) -> str:
        return "codex-cli" if self.mode == "codex" else _endpoint_origin(self.base_url)


def create_structured_model_client(
    settings: OpenAICompatibleSettings,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> StructuredModelClient:
    if settings.mode == "codex":
        return CodexStructuredClient(settings)
    return OpenAICompatibleStructuredClient(settings, http_client=http_client)
```

`create_structured_model_client` returns `OpenAICompatibleStructuredClient` for
`api` and `CodexStructuredClient` for `codex`. `CodexStructuredClient` exposes
the same `endpoint_origin`, `records`, `retry_events`, `manifest`, and
`complete` behavior used by current orchestration.

## Task 1: Add Mode-Aware Settings

**Files:**
- Modify: `src/ux_analyzer/adapters/openai.py:141-283`
- Modify: `tests/unit/test_environment.py:17-82`
- Modify: `.env.example:1-9`

**Interfaces:**
- Consumes: environment values loaded by existing `load_environment_file`.
- Produces: validated `OpenAICompatibleSettings.mode` and `endpoint_origin == "codex-cli"` in Codex mode.

- [ ] **Step 1: Write failing settings tests**

Add tests equivalent to:

```python
def test_codex_mode_does_not_require_endpoint_credentials() -> None:
    settings = OpenAICompatibleSettings.from_env(
        {
            "UXA_LLM_MODE": "codex",
            "UXA_SCENT_MODEL": "gpt-scent",
            "UXA_COGNITIVE_MODEL": "gpt-cognitive",
        },
        dotenv_path=Path("missing-test.env"),
    )

    assert settings.mode == "codex"
    assert settings.scent_model == "gpt-scent"
    assert settings.cognitive_model == "gpt-cognitive"
    assert settings.endpoint_origin == "codex-cli"


def test_unknown_llm_mode_is_rejected() -> None:
    with pytest.raises(ModelConfigurationError, match="UXA_LLM_MODE"):
        OpenAICompatibleSettings.from_env(
            {
                "UXA_LLM_MODE": "browser",
                "UXA_SCENT_MODEL": "gpt-scent",
                "UXA_COGNITIVE_MODEL": "gpt-cognitive",
            },
            dotenv_path=Path("missing-test.env"),
        )
```

Keep existing API-mode tests and add `UXA_LLM_MODE=api` to the exact
`.env.example` expectation.

- [ ] **Step 2: Run settings tests and verify failure**

Run: `rtk uv run pytest tests/unit/test_environment.py -q`

Expected: FAIL because `UXA_LLM_MODE` is not parsed and Codex mode still
requires endpoint credentials.

- [ ] **Step 3: Implement the smallest settings change**

Add `mode` with default `"api"`. Normalize it with `strip().lower()` and
reject values other than `api` and `codex`. Keep both model variables required
in both modes. Require and validate base URL plus API key only in `api` mode.
In Codex mode, set the internal endpoint origin to the fixed sanitized value
`codex-cli`; do not read or validate a base URL.

Update `from_env` and `model_validate` so missing API-only values do not cause a
`KeyError` in Codex mode. Preserve existing API validation messages and URL
normalization.

Add this line to `.env.example` before the existing model variables:

```text
UXA_LLM_MODE=api
```

- [ ] **Step 4: Run settings tests and verify pass**

Run: `rtk uv run pytest tests/unit/test_environment.py -q`

Expected: PASS, including existing API dotenv precedence and placeholder
checks.

## Task 2: Add Codex Structured Transport

**Files:**
- Modify: `src/ux_analyzer/adapters/openai.py:421-748`
- Modify: `tests/integration/models/test_openai_compatible.py:1-337`

**Interfaces:**
- Consumes: `OpenAICompatibleSettings`, `ChatMessage`, role schema, model ID, and `ModelRole`.
- Produces: `CodexStructuredClient.complete(...)` returning the validated schema instance and recording the same `ModelCallRecord` contract.

- [ ] **Step 1: Add failing mocked Codex tests**

Add a fake `asyncio.create_subprocess_exec` that captures arguments, writes a
valid JSON object to the path passed after `--output-last-message`, and returns
a process with `returncode == 0` and an async `communicate` method. Test that a
Codex client call:

```python
result = await client.complete(
    CoarseScentResponse,
    (ChatMessage(role="user", content="Find invite"),),
    model="gpt-scent",
    role=ModelRole.COARSE_SCENT,
)
```

returns the validated response, passes `gpt-scent` to `--model`, passes a JSON
Schema file to `--output-schema`, passes an output file to
`--output-last-message`, includes `--ephemeral` and `--sandbox read-only`, and
records `endpoint_origin == "codex-cli"`.

Add one test where the fake process returns a non-zero exit code and assert
`ModelFailureError` after the configured attempt bound. Add one test where the
output file contains invalid JSON and assert the existing
`invalid-structured-output` retry event.

- [ ] **Step 2: Run mocked Codex tests and verify failure**

Run: `rtk uv run pytest tests/integration/models/test_openai_compatible.py -q`

Expected: FAIL because no Codex client or command path exists.

- [ ] **Step 3: Extract only reusable call mechanics**

Move only the existing manifest, record, retry-event, delay, and sleep helpers
needed by both transports into a private support class or private helper set in
`openai.py`. Do not refactor the HTTP request payload, response classification,
or existing API fallback path.

Keep `OpenAICompatibleStructuredClient` behavior unchanged. Add
`CodexStructuredClient` using the shared helpers and the existing settings.

- [ ] **Step 4: Implement one Codex subprocess attempt**

For each attempt, create a temporary directory and write
`schema.model_json_schema()` to a schema file. Serialize existing messages in
order, preserving each role, and send the resulting prompt through stdin to:

```text
codex exec --ephemeral --sandbox read-only --model <model> \
  --output-schema <schema-file> --output-last-message <response-file> -
```

Use `asyncio.create_subprocess_exec` with captured stdout and stderr. Apply the
existing settings timeout to `process.communicate`. Read only the final output
file, parse one JSON object, and validate it with the supplied Pydantic schema.
Do not include Codex stdout or stderr in successful records.

- [ ] **Step 5: Reuse existing retry and record behavior**

Retry process errors, timeout errors, and invalid structured output through the
existing `RetryPolicy`. Record sanitized request metadata containing role,
model, messages, and schema version. Record sanitized response metadata or a
short process failure category. Raise `ModelFailureError` after the final
attempt. Use `codex-cli` for manifest provider ID, provider version, and
endpoint origin; keep API provider metadata unchanged.

- [ ] **Step 6: Add the client factory**

Implement:

```python
def create_structured_model_client(
    settings: OpenAICompatibleSettings,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> StructuredModelClient:
    if settings.mode == "codex":
        return CodexStructuredClient(settings)
    return OpenAICompatibleStructuredClient(settings, http_client=http_client)
```

- [ ] **Step 7: Run model tests and verify pass**

Run: `rtk uv run pytest tests/integration/models/test_openai_compatible.py -q`

Expected: PASS, including all existing API transport tests and new Codex
subprocess tests.

## Task 3: Route All LLM Callers Through Mode Selection

**Files:**
- Modify: `src/ux_analyzer/cli.py:20-25,711-733,1072-1149`
- Modify: `tests/integration/cli/test_commands.py:180-240`

**Interfaces:**
- Consumes: `create_structured_model_client` and `StructuredModelClient`.
- Produces: normal `uxa run`, `uxa run-one`, and `uxa ablate` execution using the selected transport.

- [ ] **Step 1: Add failing CLI selection tests**

Set `UXA_LLM_MODE=codex`, remove base URL and API key, keep the two model
variables, invoke `uxa validate ... --check-env`, and assert output contains
`mode: codex`, `scent and cognitive models configured`, and no `API key present`.
Keep the existing API-mode test asserting endpoint origin and API-key presence
without printing the key value.

- [ ] **Step 2: Run CLI tests and verify failure**

Run: `rtk uv run pytest tests/integration/cli/test_commands.py -q`

Expected: FAIL because CLI readiness assumes API mode and execution constructs
`OpenAICompatibleStructuredClient` directly.

- [ ] **Step 3: Select the client at the existing execution factory**

Replace direct construction inside `_execute_matrix` with
`create_structured_model_client(settings, http_client=client_http)`. Change
`_build_agent` and any helper parameter from `OpenAICompatibleStructuredClient`
to `StructuredModelClient`. Keep settings redaction and all run orchestration
unchanged.

- [ ] **Step 4: Make readiness output mode-aware**

Keep API output in its current shape. For Codex mode print the mode and model
configuration without claiming an API key is present. Do not invoke `codex`
from `--check-env`; the agreed contract is to call it when an actual model
call occurs and let a failed call raise the existing model failure.

- [ ] **Step 5: Run CLI tests and verify pass**

Run: `rtk uv run pytest tests/integration/cli/test_commands.py -q`

Expected: PASS for API mode, Codex readiness, redaction, and existing command
behavior.

## Task 4: Update Opt-In Live Coverage

**Files:**
- Modify: `tests/live/test_openai_endpoint.py:39-93`

**Interfaces:**
- Consumes: `OpenAICompatibleSettings` and `create_structured_model_client`.
- Produces: one live test that verifies all three structured roles through either selected transport.

- [ ] **Step 1: Add mode-aware live setup**

Keep the existing `UXA_RUN_LIVE_TESTS=1` gate. Load settings through
`OpenAICompatibleSettings.from_env()`. Require base URL and API key only when
mode is `api`; require the two model variables in either mode. Build the client
with `create_structured_model_client` and keep the existing coarse, full, and
cognitive assertions.

- [ ] **Step 2: Preserve API live behavior**

Run: `rtk uv run pytest tests/live/test_openai_endpoint.py -m live -q`

Expected without live configuration: SKIP, as before. With API mode and valid
API configuration: the same three-role compatibility test runs.

- [ ] **Step 3: Document Codex live command**

Add this command to the live-test documentation without including credentials:

```powershell
$env:UXA_LLM_MODE = "codex"
$env:UXA_SCENT_MODEL = "<scent-model-id>"
$env:UXA_COGNITIVE_MODEL = "<cognitive-model-id>"
$env:UXA_RUN_LIVE_TESTS = "1"
uv run pytest tests/live/test_openai_endpoint.py -m live -q
```

## Task 5: Document and Verify the Feature

**Files:**
- Modify: `README.md:182-205`
- Modify: `docs/model-provider.md:5-27,191-202`
- Modify: `docs/testing.md:24-37`

**Interfaces:**
- Consumes: accepted ADR `docs/adr/0002-codex-subscription-llm-mode.md` and glossary `docs/glossary.md`.
- Produces: operator documentation that distinguishes API and Codex modes and keeps credentials out of examples.

- [ ] **Step 1: Update configuration documentation**

Document the mode matrix:

| Mode | Required configuration | Transport |
| --- | --- | --- |
| `api` | mode, base URL, API key, scent model, cognitive model | OpenAI-compatible HTTP |
| `codex` | mode, scent model, cognitive model, logged-in Codex CLI | `codex exec` subprocess |

State that `api` is the default, Codex must already be installed and logged
in, and account mode does not read or print credentials.

- [ ] **Step 2: Update testing documentation**

Keep the non-live command unchanged. Document both live setup paths and state
that Codex mode is opt-in and uses the same structured-role assertions.

- [ ] **Step 3: Add the glossary link**

Add `docs/glossary.md` to the README documentation list.

- [ ] **Step 4: Run focused verification**

Run:

```text
rtk uv run pytest tests/unit/test_environment.py tests/integration/models/test_openai_compatible.py tests/integration/cli/test_commands.py tests/live/test_openai_endpoint.py -m "not live" -q
rtk uv run ruff check .
rtk uv run pyright
rtk git diff --check
```

Expected: all focused non-live tests pass, lint and type checking pass, and
the diff has no whitespace errors. Run the live command separately only when a
logged-in Codex account or API credentials are intentionally available.

## Self-Review Checklist

- [x] API-key mode remains default and its required variables remain unchanged.
- [x] Codex mode is available to normal execution paths, not only the live test.
- [x] Task model selection uses existing scent and cognitive variables.
- [x] Structured output, retries, records, redaction, and failure semantics are reused.
- [x] No executable discovery, account-file reading, browser automation, or new runtime dependency is introduced.
- [x] Live calls remain opt-in and documentation contains placeholder values only.
- [x] All named interfaces and file paths match current repository symbols.
