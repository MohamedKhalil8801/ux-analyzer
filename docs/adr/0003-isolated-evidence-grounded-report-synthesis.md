# ADR 0003: Isolated Evidence-Grounded Report Synthesis

- Status: Accepted
- Date: 2026-08-09
- Scope: Frozen expectations, final UX diagnosis, and model-review boundaries
- Decision owner: UX Analyzer maintainers

## Decision Summary

Add report synthesis as an experiment-level, post-run pipeline. It consumes an
immutable redacted evidence corpus and a versioned Frozen Expectation selected
by application version, scenario, and persona. It must not receive run-agent
chat history, prior prompts, raw model responses, private reasoning, cognitive
prose, or existing finding prose.

The pipeline uses four fresh-context roles: `ux-analyst`, `evidence-auditor`,
`pattern-reviewer`, and `report-adjudicator`. The same configured capable model
may serve every role initially, but each call has an independent context and a
role-specific structured contract. Publication requires deterministic
reference validation and no unresolved blocking objection.

## Evidence Boundary

The corpus includes frozen expectations; scenario, persona, and goal data;
sanitized UI structure and geometry; screenshots; Foveacast maps and heatmaps;
prominence and scent estimates labeled with provider provenance; executed
actions and replay positions; verifier outcomes; deterministic metrics;
limitations; and counterevidence.

UX principles come from a versioned, static, image-free local pack. They are
interpretive lenses, not evidence. A principle may help explain a finding but
cannot establish an issue or severity without observed evidence.

## Expectation Semantics

A Frozen Expectation describes desired outcomes, required invariants,
acceptable alternatives, optional reference paths, effort bounds, and warning
signals. A reference path is not the only correct path. The synthesizer judges
whether a material deviation indicates confusion, excessive effort, failure,
or a valid alternative.

## Review and Severity

Review consensus means no unresolved blocking objection, not literal
unanimity. Reviewers classify objections as blocking, material, or editorial.
The adjudicator must resolve blocking objections with evidence or remove or
downgrade the disputed finding.

Severity is a semantic model judgment justified by task importance, user
impact, affected users, frequency, recoverability, accessibility consequences,
scope, recurrence, fix leverage, evidence strength, and counterevidence.
Breadth and recurrence are indicators, not automatic severity multipliers.

## Consequences

- Existing deterministic per-run findings remain the fallback deliverable.
- The synthesis pipeline is outside `RunAgent`; one failed synthesis cannot
  invalidate or erase completed run evidence.
- Model-assisted semantic review complements, but never replaces, deterministic
  schema, checksum, reference, and publication validation.
- Application-owned retrieval provides bounded evidence access consistently
  across Codex and OpenAI-compatible transports.
