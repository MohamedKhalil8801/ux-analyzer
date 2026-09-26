# Security and privacy

`uxa` drives a real browser against a real site, sends parts of what it sees to
a model provider, and writes screenshots, traces, and page text to disk. This
document says exactly what leaves your machine, what never does, and how to
handle keys.

## What is sent to the model provider

Only what each role needs, and nothing more. The run agent has three separate
roles with separate inputs:

| Role | What it receives |
| --- | --- |
| Coarse scent | The goal, plus glance-level facts for candidate elements: element id, semantic role, visible label, region label, whether the element is actionable. |
| Full scent | The same visible meaning plus disabled state, and only for elements already noticed in the current viewport. |
| Cognitive | The goal, the newly revealed and remembered persona-visible elements (id, role, label, actionability, disabled state, region label), and bounded persona state. |

Deliberately **never** sent, in any role:

- CSS selectors or test IDs
- hidden labels or accessibility-only names (`aria-label`, `aria-labelledby`,
  offscreen label text)
- destination URLs, raw URLs, or private control API paths
- provider ids, execution references, or provider handles
- fixture-state keys or fixture values
- numeric prominence scores or numeric scent scores
- verifier state, and the model cannot influence official success
- the bounds, selectors, visibility fractions, or geometry of an element — the
  cognitive role does not even receive bounds
- run-agent chat from other surfaces, private reasoning, raw prior role
  responses, or existing report prose

Visible labels come from pixel-visible rendered text only. CSS-clipped,
offscreen, transparent, zero-font-size, and screen-reader-only text is excluded
from model-facing labels. Where an input, select, or textarea has an associated
`<label>`, that label text is included only when the label is itself visibly
painted in the viewport.

The report-synthesis roles receive a redacted evidence room: frozen
expectations, deterministic run facts, replay events, metrics, geometry, and
allowlisted screenshot or heatmap artifacts. They do not receive the run agent's
conversation, hidden DOM facts, selectors, private reasoning, or prior report
prose.

The two design-proposal roles receive the shared page capture: the page's text
inventory, node inventory, and the capture's image segments. DOM locators
(`selector`, `xpath`) exist in the persisted capture for the report's
copy-locator affordance and are stripped from the model-facing view.

## What is never sent

- **API keys.** The key is read from the environment and placed in an
  `Authorization` header. It is never written to a manifest, a bundle, a log
  line, or a model request body.
- **URL credentials.** A base URL containing a username or password is
  rejected before any request is made. Manifests and provider records store the
  **origin** only, never the full URL.
- **Model credentials in codex mode.** In codex mode the tool never reads or
  prints Codex account credentials; authentication comes entirely from the
  logged-in CLI.
- **Private browser state.** Selectors, test IDs, hidden labels, destination
  URLs, execution references, handler names, passwords, tokens, and provider
  ids are stripped from the rendered report.

## The bundled fixture app

`uxa fixture serve` starts a local, controlled target. It is not a real product
and must never be pointed at production.

- State is process-local and in memory, guarded by a lock. There is no
  persistence and no outbound communication.
- Fixture values are fake: the default `invite_email` is `person@example.com`
  and the default `totp_code` is `123456`. Invite and two-factor flows perform
  fake state transitions only.
- Fixture browser sessions are restricted to `127.0.0.1`, `localhost`, `::1`,
  and `fixture.test`, over HTTP(S), on the exact configured port. A configured
  fixture origin cannot contain credentials, a path, a query, or a fragment.
- Sessions use disposable test accounts. The account id must begin with
  `test-`; anything else is rejected before the browser starts.

Do not point the fixture at real accounts, real credentials, payment flows, or
real communication endpoints. Use it only as a local target.

## Browser network policy

Every browser request is checked against a per-session allowlist before it goes
out. The policy is fail-closed: a request to an origin that is not allowed is
aborted, including redirects and popups. `about:`, `blob:`, and `data:` are
treated as non-network browser resources, not as navigations.

A **live** session carries its own exact normalized HTTP(S) origins: the start
origin plus explicitly configured resource origins. Origins are never unioned
across experiment cells, so a live session cannot reach the fixture origin or
another live target's origin.

Before a live run starts, the tool looks for origins your target references that
are not on the allowlist and asks you once. In a non-interactive session
nothing is granted implicitly:

```text
warning: live targets reference non-allowlisted origins (...); these requests
will safety-block. Re-run interactively or pass --yes / --allow-origin.
```

`--allow-origin` and `--yes` are session-only grants. They do not modify the
project YAML and do not change the config digest.

Host-side model traffic is separate from browser traffic. The model request is
an adapter HTTP call from the host process, not a browser navigation, and it is
allowed for exactly that reason.

### Session hardening

Each run gets its own isolated browser context with:

- the scenario's viewport (default 1280×800)
- downloads disabled
- browser permissions granted: none
- service workers blocked
- geolocation denied
- clipboard read and write denied
- notifications denied
- a retained Playwright trace

Traces contain screenshots, DOM snapshots, and network records of pages you
pointed the tool at. Treat them as sensitive and delete them when you no longer
need them.

## Redaction

Redaction happens **before** any bundle JSON is written, not at render time.

- Every scenario input marked `sensitive: true` becomes an exact redaction
  value, and its key becomes a redacted mapping key. In the CLI execution path
  this covers every scenario in the project.
- Configured exact values and configured mapping keys are replaced with the
  literal string `[REDACTED]` in `manifest.json`, `timeline.jsonl`,
  `result.json`, and `crash.marker`.
- Text fields additionally pass through pattern redaction covering:
  - credential-bearing URLs (`scheme://user:pass@host`)
  - `Bearer` tokens
  - assignments to sensitive keys (`password`, `token`, `api_key`, `secret`,
    `credential`, `authorization`, `access_token`, `x_api_key`, and any
    configured key)
  - fixture secret markers
  - Windows and POSIX filesystem paths
  - bracketed segments, which is how CSS attribute selectors are caught
- Model request and response records are sanitized recursively through the
  same rules, so a secret echoed back by a model does not land in a bundle.
- Binary artifacts are handled by type rather than by pattern: PNG screenshots
  whose pixels came from a redacted source are blanked, zip archives are
  sanitized entry by entry, and other byte payloads have configured exact values
  replaced with the literal bytes of `[REDACTED]`.

The report applies a second, independent allowlist projection. It starts from
the fields the report needs and drops everything else, so a field that was never
intended for the report cannot appear in it.

## What still lands on disk

Redaction is not a substitute for access control. A raw run bundle is a
diagnostic artifact, and bundle serialization can retain provider-private
execution references and other run-state fields that are not configured as
redaction keys.

On disk you will find:

| Artifact | Sensitivity |
| --- | --- |
| `runs/<id>/artifacts/` | Screenshots and raw page content, subject to redaction. |
| `runs/<id>/timeline.jsonl` | Every observation, action, and sanitized model call, with visible page text. |
| `traces/<id>.zip` | Full page screenshots, DOM snapshots, and network records. |
| `page-capture.json` | Page images and the trimmed node/copy inventory. |
| `report.html` | Public projection of the above; private fields removed. |
| `exploration/` | Crawled page labels, headings, and screenshot digests. |

Guard the output directory accordingly. Do not commit `.uxa-output`,
`reports/`, traces, or screenshots from a live target.

## Served UIs are loopback-only

`uxa` serves two local UIs, and both bind `127.0.0.1` only:

- the report server behind `uxa report --serve`
- the exploration review UI behind `uxa explore`

The report server uses Python's standard-library `http.server`. It serves the
rendered report directory over loopback until you interrupt it. When no
`--port` is given it binds an ephemeral port. The exploration review server
validates its bind host before starting and refuses anything outside the
loopback set:

```text
fixture host must be loopback-only: use 127.0.0.1, localhost, or ::1
```

The same check guards `uxa fixture serve --host`. The review server exposes no
CORS headers and makes no external request.

Nothing in either UI is authenticated, because nothing in either UI is
reachable off the machine. Do not port-forward or reverse-proxy them.

The exploration review UI does carry a chain-of-custody check: the served
suggestion set is signed, and a curation payload whose signature does not match
the currently served set is rejected before any curation logic runs. A stale
browser tab cannot submit against a different suggestion set.

## Handling keys

1. **Keep keys in the environment or a secret manager.** Export them into the
   shell for the session, or rely on a `.env` file that is itself
   git-ignored.
2. **Never put a key in a project YAML.** The config digest hashes the entire
   project file, so a key there is written into every run manifest and every
   bundle derived from it.
3. **Never put a key in source, tests, or documentation.** Use placeholders such
   as `<api-key>` and `<provider-host>`.
4. **Use `.env`, not your shell profile, for anything shared.** `.env` is read
   from the current working directory and never overrides a value already in the
   process environment.
5. **Check configuration without leaking it.** `uv run uxa validate PROJECT
   --check-env` reports which required names are present and never prints their
   values. In codex mode the confirmation line reads `(mode: codex; scent and
   cognitive models configured)`; in API mode it reports only the endpoint
   origin and that a key is present.
6. **Scope the key.** Use a key restricted to the models and the region you
   actually need. The tool never sends a key anywhere except the configured
   endpoint's authorization header.
7. **Check the provider's retention and jurisdiction before pointing `uxa` at
   sensitive target content.** A model provider receives goal text, visible
   labels, semantic roles, region labels, and page images. If your data cannot
   go to that provider, do not run against that target.
8. **Rotate after a suspected exposure.** Check the manifest and timeline files
   of any run you shared before rotating; redaction is pattern-based and cannot
   catch every shape of secret.

## Operator checklist

- [ ] Bundled fixture origins only for fixture sessions; live origins only when
      you have explicitly configured them.
- [ ] Disposable test accounts for anything you point at.
- [ ] Keys in the environment or a secret manager, never in YAML or docs.
- [ ] `UXA_REPORT_MODEL` is vision-capable if your evidence contains images —
      an unsuitable model makes the synthesis attempt unavailable rather than
      degrading an image estimate into a fact.
- [ ] Output directory access restricted; traces and screenshots deleted on a
      schedule.
- [ ] Provider data handling, endpoint retention, access, and jurisdiction
      reviewed before running against sensitive content.
- [ ] Live endpoint tests treated as opt-in external data transfer.
- [ ] Official success tied to independent verification, never to a model claim.
