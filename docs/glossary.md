# Glossary

## API Mode

LLM mode that sends structured requests to the configured OpenAI-compatible
HTTP endpoint using `UXA_LLM_BASE_URL` and `UXA_LLM_API_KEY`.

## Codex Mode

LLM mode that invokes the locally installed `codex` CLI. Authentication comes
from the user's already logged-in Codex subscription account.

## Task Model

Model selected for one application role. Scent roles use `UXA_SCENT_MODEL`;
the cognitive role uses `UXA_COGNITIVE_MODEL`. Authentication mode does not
change this mapping.

## Structured Model Call

One logical request for a coarse-scent, full-scent, or cognitive role. The
request has role messages and a Pydantic-backed JSON Schema; the response is
validated before use.

## Account Authentication

Authentication supplied by a logged-in local tool rather than an API key
provided to UX Analyzer. In this plan, account authentication means Codex CLI
authentication only.
