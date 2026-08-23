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

## Exploration Mode

Operator workflow that starts from one or more start URLs and a crawl depth, crawls live targets, asks the cognitive model to propose covering scenarios, lets the operator curate them in a local review UI (or auto-accept), then runs the standard benchmark pipeline. Verification for exploration scenarios is `visible-result` only.

## Crawl Depth

BFS link-hop distance from any start URL. Depth 0 means start URLs only. Depth 1 adds direct same-origin links from starts, and so on. Depth is bounded (0–5) and combined with `max_pages`.

## Crawl Corpus

Immutable, redacted collection of crawled pages: normalized URL, depth, title, headings, visible elements, screenshot reference, discovered same-origin links, and settle metadata. Stored under `<output>/exploration/<attempt-id>/`.

## Scenario Suggestion

Structured proposal produced by the cognitive model from the crawl corpus: `goal`, `start_url`, `verifier` (`visible-result`), `evaluation_target`, `budget`, and coverage rationale. Up to 20, configurable. Operator may accept, edit, or add.

## Exploration Review UI

Local, offline browser UI served by `uxa explore` for reviewing suggested scenarios and personas. Operations: toggle accept, edit JSON, add custom scenario, choose existing / suggested / custom persona. `--auto-accept` skips the UI.

## Same-Origin Crawl

Frontier rule that enqueued links must share origin with a start URL origin. Resource origins (`allowed_origins`) remain separate for sub-resources; crawl itself never follows cross-origin documents.

## Page Settlement

Per-page readiness procedure before capture: wait for `networkidle` (with fallback), dismiss loading indicators, perform bounded incremental scroll sweep, and observe mutation stabilization, all within `page_settle_ms`. Ensures scroll-revealed and delayed content is visible.

## Exploration Artifact

Immutable experiment-scoped directory `exploration/<attempt-id>/` containing `corpus.json`, `suggestions.json`, `curated.json`, `project.fragment.yaml`, and digests. Attempt ID is `<utc>-<first12(corpus_digest)>-<seq>`.

## Auto-Accept

Flag `--auto-accept` on `uxa explore` that accepts all suggested scenarios without launching the review UI. Curated set equals suggested set; still persisted as immutable artifact.
