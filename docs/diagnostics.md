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
persisted, or attached to an exception. `reason` and every string in `extras`
are passed through `redact_forbidden_narrative` as defense in depth, so a future
caller that interpolates model text cannot leak it through another field.

The fingerprint is a correlation key, not a security boundary. It is only safe
to keep because the text it summarizes is already excluded from the corpus
boundary by `FORBIDDEN_NARRATIVE_MARKERS`. Identical rejected text produces an
identical hash, which is how you tell a recurring failure from a one-off.

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
