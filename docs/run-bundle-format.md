# Run Bundle Format

## Lifecycle

Each run writes to:

```text
<output>/.staging/<run-id>/
```

Finalization writes terminal files, checksums every published file except the
checksum file itself, and atomically renames the staging directory to:

```text
<output>/runs/<run-id>/
```

An active staging bundle has `.active`. `abort(reason)` closes the timeline,
writes `crash.marker`, removes `.active`, and leaves staging evidence for
recovery. A finalized bundle cannot be mutated.

## Final Tree

```text
<output>/
  experiment-progress.json
  experiment.json
  report.html
  runs/
    <run-id>/
      manifest.json
      timeline.jsonl
      artifacts/
        <sha256>
      saliency/
        <artifact-namespace>/
          1s.npz
          3s.npz
          7s.npz
          1s-heatmap.png
          3s-heatmap.png
          7s-heatmap.png
          profiles.json
          metadata.json
      result.json
      checksums.sha256
```

`artifacts/<sha256>` stores raw content by SHA-256. Repeated identical content
inside one run reuses one path. Artifact names are retained in the in-memory
`ArtifactReference` (`name`, path, digest, size); stored path is the digest.

## `manifest.json`

Manifest is reproducibility metadata. Current fields:

```json
{
  "run_id": "run-...",
  "seed": 0,
  "model_trial": 0,
  "config_digest": "sha256...",
  "endpoint_origin": "https://provider.example",
  "model_ids": {
    "scent": "<scent-model-id>",
    "cognitive": "<cognitive-model-id>"
  },
  "prompt_versions": {
    "coarse-scent": "scent-coarse-v1",
    "full-scent": "scent-full-v1",
    "cognitive": "cognitive-v1"
  },
  "package_version": "0.1.0",
  "provider_versions": {},
  "provider_manifests": []
}
```

`seed` identifies deterministic attention behavior. `model_trial` identifies
external model-trial replication and remains independent from attention seed.
New manifests persist both fields in run identity metadata. Legacy manifests
without `model_trial` are compatible and read as model trial `0`.

`endpoint_origin` and provider-manifest endpoint values are origin-only. URL
credentials are rejected and API keys are never manifest fields. Final run
state and terminal event also carry role-specific provider manifests, including
provider ID, role, optional model ID, endpoint origin, and version.

## `timeline.jsonl`

One JSON object per line. Writer assigns monotonic one-based `sequence` values,
ignoring any caller-provided sequence, and flushes plus `fsync`s every event.

Typed lifecycle events are:

```text
run-started
viewport-captured
observation-recorded
action-proposed
action-executed
verification-recorded
run-terminated
```

The current application also records provider/application events such as
`coarse-scent-recorded`, `full-scent-recorded`, `agent-claim`,
`action-rejected`, `prominence-recorded`, `attention-selection-recorded`,
`decision-recorded`, and `model-call-recorded`. Model call records contain
sanitized request/response, retries, latency, attempts, and token usage. The last
lifecycle terminal event is `run-terminated`.

Learned prominence uses typed allowlisted events. `saliency-inference-recorded`
or `saliency-cache-hit` records viewport/artifact namespace, provider identity,
three model checksums, execution provider, preprocessing version, precision,
cache key/state, duration timings, artifact references, and bounded warnings.
`saliency-profiles-recorded` links the same profile artifacts to search stage;
`prominence-recorded` links operational prominence to its profile event through
`source_event_id`, and records active provider, stage, selected duration mixture,
and selected element IDs without numeric saliency. `saliency-fallback-recorded`
records learned-provider failure, fallback provider, stage, and sanitized reason.
`ProminenceEvidence` retains profile event ID and operational event ID as
separate typed fields; legacy `source_event_id` aliases operational event ID.

Typical terminal event fields include `outcome`, `verification`,
`provider_manifests`, `configuration_digest`, and `artifact_checksums`.

## `result.json`

Contains serialized `RunResult`, including run ID, terminal outcome,
independent verification, agent claim, final state, replay evidence, per-run
metrics, findings, and bundle path when available. JSON serialization converts
domain dataclasses, enums, paths, bytes, and mappings at the storage boundary.

Metrics and findings identify prominence provider and active search stage. Learned
target prominence keeps immediate (`1s`), early (`3s`), and eventual (`7s`)
profile evidence while scoring target prominence from active operational stage
only. Missing active-stage target evidence is unavailable, not borrowed from
another stage. Deterministic interface/action/geometry/verifier facts remain
`deterministic-fact`; saliency, prominence, scent, policy, and cost estimates
remain `model-estimate`.

Fallback learned runs are operationally allowed to finish, but are not valid
Foveacast comparison samples. `ux_sample_valid=false`, `comparison_valid=false`,
and a sanitized `ux_sample_invalid_reason` persist through resume, evaluation,
scorecard aggregation, and report grouping. Heuristic fallback evidence remains
visible as diagnostic evidence. Learned samples with missing, malformed, or
unlinked saliency replay artifacts are likewise excluded from comparison output.

Experiment execution also writes `<output>/experiment.json` with per-run metrics,
cell aggregates, paired variant comparisons, directional gates, findings, and
partial failures. This file feeds the comparison/gate sections in the HTML report.

Long experiment execution writes `<output>/experiment-progress.json`
atomically after each finalized result or sanitized runner failure. Explicit
`--resume` validates selected finalized bundles, skips trusted run IDs, marks
interrupted staging evidence, and rebuilds experiment aggregates from both
skipped and newly completed selected runs.

## `checksums.sha256`

Each line is:

```text
<sha256>  <relative-path>
```

For finalized bundles it covers manifest, timeline, result, and files under
`artifacts/`, but excludes `checksums.sha256` and `.active`. Aborted staging
directories retain `crash.marker` for recovery and do not receive a finalized
checksum file. Verify each published digest against bytes at its relative path.

Saliency artifacts are checksum-covered generated evidence. Native `.npz` files
contain normalized maps and geometry; grayscale `*-heatmap.png` files are
heatmap-only and contain no source screenshot pixels; `profiles.json` contains
element profiles and aggregation components; `metadata.json` contains cache key,
provider/model/version, checksums, execution provider, geometry, warnings, and
per-duration inference timing. Cache status is separate from runtime timing:
cache hits still record provider/model identity and timing/artifact references.

Reports may show a sanitized source screenshot overlay only when artifact policy
permits it. Redaction never gets bypassed. When source pixels are blanked or
unavailable, replay states `Overlay unavailable due redaction` and retains
heatmap-only image, element geometry, ranked elements, and aggregation detail.
Replay parser applies bounded JSON, timeline, image, and profile/aggregate limits;
oversized or malformed evidence is unavailable rather than rendered.

## Redaction and Retention

`RedactionPolicy` redacts configured exact values and configured mapping keys
before manifest, timeline, result, and crash-marker JSON writes. Demo sensitive
fixture values are `invite_email` and `totp_code`; their exact values are
passed to redaction. The replacement string is `[REDACTED]`.

Raw bundles remain diagnostic artifacts, not public model payloads. Private
execution references and provider-only fields may be present in serialized run
state unless a configured redaction key removes them. Protect output directory
access and retention accordingly. The HTML renderer creates public projections
and removes private keys such as selectors, test IDs, destination URLs,
execution references, provider IDs, hidden labels, passwords, tokens, and API
keys.

## Report Input

`uxa report` reads a bundle root containing either one run bundle, a `runs/`
directory of finalized bundles, or an experiment directory with `experiment.json`
and partial evidence. It includes finalized runs, evaluator failures, runner
failures, and staging `crash.marker` records. Failed rows retain safe run/spec
identity, stage, terminal state, reason, and available timeline evidence; they
are never silently omitted. It emits one self-contained HTML file when the
encoded report is at most 2,000,000 bytes by default. Larger reports produce an
index plus `<output-stem>-runs/<run-id>.html` pages. No external requests are
required to open the report.
