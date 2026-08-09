# ADR 0004: Versioned Offline Synthesis Artifacts

- Status: Accepted
- Date: 2026-08-09
- Scope: Synthesis persistence, regeneration, fallback, and report rendering
- Decision owner: UX Analyzer maintainers

## Decision Summary

Persist every report-synthesis attempt as an immutable, checksummed,
experiment-level artifact. Normal analysis runs synthesis automatically when
configured. Existing projects keep their disabled default for compatibility;
new project templates enable Frozen Expectations and report synthesis. An
explicit `--no-synthesis` option disables the automatic step.

Each attempt records the evidence-corpus digest, expectation and UX-principle
pack digests, model and role manifests, prompt and schema versions, retrieval
log, token and latency usage, candidate findings, objections, rejected
findings, final findings, limitations, and acceptance status. Regeneration
creates a new attempt and never overwrites an earlier artifact.

## Rendering Boundary

`uxa report` is offline and never silently calls a model. It renders the latest
accepted synthesis selected by the synthesis index. When synthesis is missing,
unavailable, rejected, or fails, the report renders deterministic findings and
an explicit synthesis status. A valid no-issue synthesis states that no
supported issues were established in the tested scenarios; it never claims the
entire product is issue-free.

The report is conclusions-first: overall assessment, prioritized root-cause
findings, and what to fix first precede experiment tables, heatmaps, replay,
metrics, and audit details. Every visible claim links to resolvable evidence,
including viewport, element, event, metric, heatmap, screenshot, and replay
references where applicable.

## Consequences

- Reports remain reproducible and inspectable without network or model access.
- Synthesis failure does not fail completed experiment execution or destroy the
  final deliverable.
- Accepted and rejected attempts remain available for audit and comparison.
- Report size thresholds and split-report behavior must account for synthesis
  data without weakening existing artifact validation or redaction.
