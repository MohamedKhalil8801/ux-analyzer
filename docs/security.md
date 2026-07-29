# Security and Limits

## Trust Boundaries

| Boundary | Trusted side | Untrusted or restricted side | Control |
| --- | --- | --- | --- |
| Browser to target | Playwright session policy | Every browser request, redirect, popup, form submission, image, fetch, or XHR | Exact bundled fixture-origin allowlist; fail closed. |
| Fixture state | Local FastAPI fixture process | Scenario input and browser actions | Test account IDs, isolated in-memory sessions, private control routes, no outbound communication. |
| Application to model | Role providers and application validator | Model output, endpoint response, provider behavior | Structured schema validation, bounded retries, action validation, independent verifier. |
| Model request to provider | Host-side OpenAI-compatible client | External configured endpoint | API key in header only; endpoint origin in manifests; request/response sanitization. |
| Run process to artifacts | Bundle writer | Event values, fixture values, provider responses | Redaction before JSON write, content hashes, atomic publication, immutable final bundle. |
| Raw bundle to report | Renderer | Serialized run data and UI text | Private-key filtering, HTML autoescape, safe JSON encoding, no external report requests. |

## Browser and Fixture Safety

Only bundled fixture origins may be automated. Normalized allowed hosts are
`127.0.0.1`, `localhost`, `::1`, and `fixture.test`, using HTTP(S) and the exact
configured port. Configured fixture origin cannot contain credentials, path,
query, or fragment.

Every browser request is routed through the allowlist. Foreign origins are
blocked, including redirects and popups. `about:`, `blob:`, and `data:` schemes
are treated as non-network browser resources. Host-side model traffic is
separate from browser traffic and is allowed only because it is an adapter HTTP
call, not a browser navigation.

Each run uses an isolated browser context, fixed `1280x800` fixture viewport,
disabled downloads, blocked service workers, denied geolocation, denied
clipboard access, denied notifications, and a retained Playwright trace. A
session requires a `TestAccountId` beginning with `test-`.

The fixture state store is process-local, in memory, thread-safe, and has no
persistence or network side effects. Invite and two-factor flows use fake
values and fake state transitions. They must not be pointed at production
systems, real accounts, real credentials, payment flows, or real communication
endpoints.

## Secrets and Redaction

Scenario `fixture_inputs` are the only source of typed values. Mark sensitive
inputs with `sensitive: true`. Demo marks `invite_email` and `totp_code`
sensitive. The application resolves typed-action fixture keys; model does not
invent credentials, email addresses, or two-factor secrets.

Bundle `RedactionPolicy` removes configured exact values and configured mapping
keys before writing `manifest.json`, `timeline.jsonl`, `result.json`, and
`crash.marker`. In the CLI execution path, scenario values marked sensitive
become exact redaction values. Model log sanitization also removes API-key-like
keys, bearer tokens, API key, and configured exact fixture values. Manifest
stores endpoint origin, model IDs, prompt versions, package version, and
provider versions, not API key.

Raw output is still sensitive. Bundle serialization can retain provider-private
execution references or other run-state fields when they are not configured as
redaction keys. Restrict filesystem permissions, avoid sharing `.uxa-output`,
and delete retained traces/screenshots when no longer needed. Public persona
observations and static reports use an explicit allowlist and strip selectors,
test IDs, hidden labels, destination URLs, provider IDs, execution references,
passwords, tokens, and API keys.

## Model Exposure

Model inputs are limited by role:

- Coarse scent sees glance-level visible cues only.
- Full scent sees only already noticed visible controls.
- Cognitive sees newly revealed and remembered persona-visible controls.

No role sees DOM selectors, test IDs, hidden labels, raw URLs, provider handles,
private control API paths, fixture-state keys, numeric prominence, numeric scent,
or verifier state. Model requests are audited in tests for forbidden values and
field paths.

External model providers can receive goal text, visible labels, role names,
region labels, actionability/disabled state where role permits, and model
prompt content. Do not run this POC with sensitive target content unless data
handling, endpoint retention, access, and jurisdiction are acceptable.

## Evidence and Claim Limits

`deterministic-fact` covers directly recorded interface and execution facts.
`model-estimate` covers outputs depending on heuristic, scent, persona, memory,
policy, or model configuration. `unsupported-human-claim` covers real-user
outcomes that synthetic runs cannot establish.

Unsupported human claims are rejected from findings and scorecards. Reports
state that output is simulated benchmark evidence, not real-user completion or
satisfaction. Discovery cost, notice probability, scent, abandonment, and
persona paths describe the configured simulation only. They are not population
probabilities, clinical measures, accessibility conformance, or predictions of
conversion or product-market fit.

## Operator Rules

1. Use only bundled fixture origins and disposable test accounts.
2. Use placeholder model values in docs and examples; keep API keys in environment variables or a secret manager.
3. Do not commit `.uxa-output`, traces, screenshots, or live responses containing sensitive data.
4. Review raw bundle retention before distributing reports or artifacts.
5. Treat live endpoint tests as opt-in external data transfer.
6. Keep official success tied to independent verification, never model claims.
