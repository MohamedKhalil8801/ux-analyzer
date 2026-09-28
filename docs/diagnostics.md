# Diagnostics

Every synthesis role can fail for a reason the persisted artifact does not
describe. Before this module existed, an attempt recorded *that* a response was
rejected and the sanitized reason, but never *which field* failed or whether the
same text had failed in an earlier attempt. Diagnosing a live run then meant
guessing.

`ux_analyzer.observability.diagnostics` records a structured event for every
role call, validation rejection, and attempt outcome.

## Contract

Every event carries:

| Field | Meaning |
| --- | --- |
| `timestamp` | UTC, ISO-8601, millisecond resolution |
| `event` | `role.call.started`, `role.call.responded`, `role.call.failed`, `response.rejected`, `attempt.finished` |
| `layer` | `provider` or `application` |
| `role`, `stage`, `round` | Which role, phase, and retrieval round |
| `reason_code` | Stable machine token, e.g. `forbidden-narrative`, `bounded-output` |
| `reason` | Sanitized reason from the codebase's own fixed vocabulary |
| `field` | Path to the offending field, e.g. `limitations[3]` |
| `field_hash` | SHA-256, first 16 hex chars, of the offending text |
| `field_length` | Character length of the offending text |
| `extras` | Structured context: counts, indices, schema name |

## The text itself is never stored

`text=` is accepted only to be fingerprinted. It is never logged, returned,
persisted, or attached to an exception. It is the only parameter that ever
receives model-authored content.

Every other value is truncated to 200 characters rather than filtered for
wording. Those values are counts, indices, schema names, and reason codes drawn
from this codebase's own vocabulary; filtering them for banned phrases would
destroy the record that makes a failure diagnosable, which is the exact failure
this module was written to fix.

The fingerprint is a correlation key, not a security boundary. Identical
rejected text produces an identical hash, which is how you tell a recurring
failure from a one-off.

## What the model is told when it is rejected

A rejected role is retried once with correction context instead of an identical
prompt. That context carries everything available about the failure, and the
boundary still holds: identifiers are echoed, prose is not.

| Key | Content |
| --- | --- |
| `previous_validation_failure` | Bounded headline drawn from an allowlist of our own phrases |
| `reason_code` | Stable machine token, e.g. `undelivered-evidence-id` |
| `issues[]` | Per-issue detail, each with `code`, `detail`, and where applicable `field`, `constraint`, `identifiers`, `response_diagnostics` |

Detail is recovered per failure class:

- **Role validation** — the reason plus the identifiers it interpolated, so
  `unknown evidence ID: event:run-a:99` arrives naming `event:run-a:99`
  instead of `unknown evidence ID`
- **Schema validation** — the failing field path, the constraint type, and the
  validator message, taken from the pydantic error the provider chains
- **Transport shape failure** — the adapter's own diagnostics: stage, response
  mode, finish reason, content length
- **Application contract** — the rejection sentence itself, so
  `repeated evidence request` no longer arrives as
  `evidence, schema, or publication contract failed`

Only token-shaped values are echoed: `_feedback_identifier` admits at most 120
characters from `[A-Za-z0-9_.:/@+-]` with no whitespace, so free text cannot
pass. A rejection carrying a prose phrase yields no identifiers at all.

### Identifiers reach the model; they do not reach the artifact

Echoing a bounded identifier the model itself just emitted is what lets it
correct the exact reference. Persisting it is a different act, and the run
bundle is not the place to store provider content. The persisted
`retrieval_log` therefore records only the bounded headline and the stable
code, which is what keeps the artifact free of model-supplied text while the
model still gets what it needs to fix itself.

## Reading a failed run
`synthesize` writes `<output>/diagnostics.jsonl`, one JSON object per line, and
prints the path. A rejected attempt is diagnosable without the model:

```powershell
Get-Content reports\exploration-generated\diagnostics.jsonl |
  Where-Object { $_ -match 'response.rejected' } |
  ForEach-Object { $_ | ConvertFrom-Json } |
  Format-Table timestamp, role, reason, field, field_hash
```

To group a failure that has recurred across attempts, count `field_hash`
values. A single hash appearing in every attempt is the same text failing
consistently; many distinct hashes are a wandering failure.

## Configuration

| Setting | Effect |
| --- | --- |
| `UXA_DIAGNOSTIC_LOG=<path>` | Append JSON lines to `<path>`. Set by `synthesize` automatically. |
| `UXA_DIAGNOSTIC_STDERR=1` | Mirror every event to standard error. |

An in-memory ring buffer (2048 events) always records, so
`diagnostics.recent()`, `diagnostics.drain()`, and `diagnostics.reset()` work
without any sink configured. `reset()` is for tests.
