# ADR 0002: Codex Subscription LLM Mode

- Status: Accepted
- Date: 2026-08-05
- Scope: LLM authentication and transport selection
- Decision owner: UX Analyzer maintainers

## Decision Summary

Keep existing OpenAI-compatible API-key execution. Add an opt-in Codex mode for
users authenticated through a logged-in Codex subscription account.

Select mode with `UXA_LLM_MODE`:

```text
UXA_LLM_MODE=api    # default; requires base URL and API key
UXA_LLM_MODE=codex  # uses logged-in Codex CLI; no API key or base URL
```

Both modes use existing task-specific model variables:

```text
UXA_SCENT_MODEL
UXA_COGNITIVE_MODEL
```

Codex mode passes the model selected for each role to one ephemeral
`codex exec` subprocess per logical model call. It uses read-only execution,
the role's existing JSON Schema, and local Pydantic validation.

## Context

Current LLM execution assumes an OpenAI-compatible HTTP endpoint. Every live
test and normal model-backed run requires `UXA_LLM_BASE_URL`,
`UXA_LLM_API_KEY`, `UXA_SCENT_MODEL`, and `UXA_COGNITIVE_MODEL`.

Some users have a regular subscription account authenticated by Codex CLI but
do not have API credentials or an API endpoint. The application already owns
role prompts, schemas, retries, redaction, model records, and failure
contracts. Account mode should change transport and authentication only.

## Chosen Design

1. Add `UXA_LLM_MODE` with `api` as the default.
2. Require both existing model variables in either mode.
3. Require base URL and API key only in `api` mode.
4. Keep `OpenAICompatibleSettings` as the configuration boundary to avoid a
   broad rename. Its mode controls validation and client selection.
5. Reuse the existing `StructuredModelClient` contract, role prompts, schemas,
   retry policy, local validation, model records, redaction, and failure type.
6. Run `codex exec` directly. Do not discover the executable, read account
   files, or add an executable-path setting. A failed subprocess becomes the
   existing model failure.
7. Use `--ephemeral`, `--sandbox read-only`, `--model`, `--output-schema`, and
   `--output-last-message`. Send the serialized role messages through stdin.
8. Record `codex-cli` as sanitized endpoint origin and provider identity. Never
   persist Codex authentication data, stdout diagnostics, or stderr secrets.

## Alternatives Rejected

### Browser or ChatGPT web automation

Rejected. It adds session-cookie handling, browser state, fragile UI coupling,
and credential exposure without improving model-call contracts.

### New model environment variables for Codex

Rejected. Model choice follows task role, not authentication mode. Existing
`UXA_SCENT_MODEL` and `UXA_COGNITIVE_MODEL` remain the only model variables.

### Automatic Codex executable discovery

Rejected. Codex is an explicit user prerequisite. Calling `codex` directly is
the smallest contract and gives a direct failure when the prerequisite is not
met.

### Separate production-only account path

Rejected. Account selection must work anywhere the application creates an LLM
client, including normal `uxa run`, not only live tests.

## Consequences

Positive:

- Subscription users can run model-backed commands without an API key.
- API-key behavior remains the default and keeps its current contract.
- Role-specific model selection remains unchanged.
- Existing audit and validation behavior remains consistent across transports.

Costs:

- Codex mode starts one process per logical model call.
- Codex CLI must be installed, logged in, and available as `codex` on `PATH`.
- Codex usage and model availability follow the user's subscription and Codex
  configuration rather than API billing/configuration.

## Verification Contract

- Unit tests validate mode-specific environment requirements.
- Mocked model tests validate command arguments, schema-file handling, role
  model selection, structured output, retries, and process failures.
- Existing API-compatible tests remain unchanged except for shared client
  selection coverage.
- Live tests remain opt-in with `UXA_RUN_LIVE_TESTS=1`; Codex live mode adds
  `UXA_LLM_MODE=codex` and uses the existing model variables.
