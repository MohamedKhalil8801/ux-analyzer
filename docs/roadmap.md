# Roadmap and Deferred Contracts

Current POC is complete only within its declared boundary: Chromium web
automation, bundled fixture, deterministic extraction, heuristic prominence,
seeded attention, optional structured model roles, independent verification,
immutable bundles, and static replay. Items below are deferred extensions, not
current implementation placeholders.

## Active Next Phase: Foveacast Saliency Integration

Planning basis approved on 2026-08-01. Detailed implementation sequence is in
[`docs/superpowers/plans/2026-08-01-foveacast-saliency-integration.md`](superpowers/plans/2026-08-01-foveacast-saliency-integration.md).

This phase closes the current post-reliability POC baseline, then adds
Foveacast v0.2.0 FP16 ONNX models for 1-second, 3-second, and 7-second visual
attention. Learned outputs remain model-dependent evidence separate from
deterministic element snapshots. Existing heuristic prominence remains
available as baseline and runtime fallback.

Approved boundaries:

- Web and bundled fixture only.
- CPU inference required; optional DirectML acceleration on Windows.
- Exact experiment-scoped screenshot cache.
- Immediate, early, and eventual element attention remain separate.
- Search stage selects operational prominence without exposing scores to the
  cognitive model.
- Focused heuristic-versus-Foveacast comparison precedes any provisional
  default change.
- Full learned-provider matrix is deferred beyond this phase.
- SUM is deferred because its Linux/Mamba stack is not a justified target for
  current Windows/AMD environment.
- Frozen expectations become a separate later phase.

Provider promotion is conditional. An ADR must preserve exact model checksums,
execution provider, focused comparison evidence, failures, limitations, and
review trigger whether Foveacast is promoted or heuristic remains default.

## Pretrained Saliency Provider

Contract: a versioned provider accepts a normalized viewport/screenshot and
returns per-element saliency evidence plus feature provenance. Provider manifest
must record source, exact version or commit, weights checksum, license, runtime,
hardware requirements, and configuration. It must be comparable with
`heuristic-prominence-v1` through the same evaluation cells and must not be
treated as ground truth.

Current planned provider is Foveacast v0.2.0 FP16 with separate 1s, 3s, and 7s
models. Initial comparison uses existing heuristic provider and evaluates a
simple hybrid only when focused evidence shows complementary errors. SUM and
full UEyes metric reproduction are not part of active phase.

Required comparison: heuristic, learned, and any hybrid provider use identical
scenario, version, persona, policy, seed, fixture state, and model settings.
Raw saliency and element aggregation remain visible in evidence.

## Frozen Expectation Provider

Contract: generate three to five likely task paths, labels, control types, and
feedback expectations before interface exposure. Persist immutable expectation
artifact and provider manifest before first capture. Runtime may compare actual
navigation with expectation, but reports must keep expectation conformity,
visual discoverability, task efficiency, and learnability as separate metrics.
Current project loader retains no enabled expectation provider field.

## Desktop and Mobile Observation Providers

Contract: implement platform-neutral `ObservationProvider` lifecycle:
`start_session`, `capture`, `execute`, `reset`, and `end_session`. Adapter must
map platform-native nodes, screenshots, bounds, visibility, and actions into
domain snapshots without leaking native handles inward. Each platform needs its
own allowlist, test-account policy, cleanup, trace/artifact rules, and leakage
tests. Desktop and mobile are not current targets.

## Human Calibration

Contract: collect first-noticed element, fixation or attention order when
available, click sequence, scrolls, backtracks, completion, abandonment, and
expectation data from real participants. Keep calibration and held-out
validation data separate. Report population, task, platform, accessibility, and
demographic scope. Do not turn persona YAML values into population claims.

## Statistical Calibration

Contract: after human data demonstrates a measurable gap, fit the simplest
interpretable mapping from synthetic metrics to observed human outcomes. Record
training data identity, held-out evaluation, calibration version, uncertainty,
and drift checks. Preserve raw simulated metrics beside calibrated estimates.
Calibration cannot retroactively make an unvalidated POC metric a human fact.

## Production API

Contract: define authenticated tenancy, authorization, quotas, job lifecycle,
provider egress policy, model secret handling, artifact access, retention,
deletion, audit logs, rate limits, cancellation, and isolation before exposing
benchmark execution as a service. API transport types must stay outside domain
policy. Current CLI and local filesystem bundle are not production service
contracts.

## Issue-Tracker Integration

Contract: publish only evidence-backed findings with evidence IDs, source run
links, reproducibility, limitations, severity policy, destination mapping,
deduplication, idempotency, approval workflow, and rollback/update behavior.
Human claims must remain blocked. Current POC generates findings and reports but
does not create or update external tickets.

## Promotion Criteria

An extension should enter implementation only after a concrete measured need,
versioned contract, security review, fixture or recorded tests, evidence-class
handling, and reproducibility plan are approved. The current baseline remains
available for every ablation and regression comparison.
