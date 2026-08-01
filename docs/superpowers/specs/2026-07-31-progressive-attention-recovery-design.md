# Progressive Attention Recovery Design

## Goal

Make progressive attention reliably discover strong visible task controls without
turning it into a full-list policy, and terminate only consecutive no-progress
behavior rather than lifetime repetition.

## Attention Selection

- Change the progressive attention default and demo configuration to
  `batch_size: 2`.
- Preserve seeded region-first softmax sampling during normal observations.
- Keep the cognitive boundary unchanged: only newly revealed and remembered
  persona-safe elements are sent to the cognitive model.
- Track continuously eligible, unseen controls only when they are actionable,
  enabled, have coarse scent at or above the configured threshold, and have a
  unique provider-stable lineage.
- Perform the normal seeded region-first draw before recovery. After two prior
  misses, force one overdue control on its third eligible observation while
  retaining one normally sampled element. Mixed-region recovery clears the
  single-region context.
- Persist bounded recovery counters in immutable attention policy state, not in
  cognitive memory. Missing, ambiguous, stale, or provider-changed lineages reset
  recovery rather than conflating controls.
- Keep `recovery_level` on the shared policy interface for run-stall diagnostics;
  full-list and ranked-list policies accept and ignore it.
- Recovery never clicks a control, exposes selectors, or reads verifier-private
  state.

## Meaningful Progress

Run orchestration evaluates progress after recapture. An action makes meaningful
progress when it succeeds and at least one of these safe conditions holds:

- navigation occurred;
- a fixture input was newly completed;
- the visible snapshot signature changed by element lineage, label, role,
  actionable state, disabled state, or visibility;
- independent verification succeeded.

Bounds and scroll offsets are excluded. A scroll that merely moves the page while
showing the same semantic controls is no progress.

## Recovery And Termination

- Recoverable no-progress actions emit `no-progress-recovery` and set the
  recovery level for the next observation.
- A meaningful-progress action resets recovery and repetition state.
- Repetition counts only identical consecutive actions following no progress.
  Actions are fingerprinted by kind, direction, target lineage, and fixture key.
- Three identical consecutive no-progress actions, or five consecutive stalled
  actions of any kind, emit `no-progress-detected` and terminate as
  `agent-abandoned` with a clear reason.
- Repeating a successfully completed fixture input remains an immediate bounded
  failure with its existing diagnostic.

## Reporting

The report exposes `no-progress-recovery`, including count, previous action,
and safe reason. Existing terminal and integrity reporting remains unchanged.

## Tests

- Default and demo batch size are 2.
- Normal seeded selection remains reproducible.
- Recovery forces a continuously eligible high-scent unseen actionable control
  on its third eligible observation while retaining one stochastic reveal.
- A successful fixture input resets repetition accumulated before it.
- Scroll-only changes with the same semantic snapshot activate recovery.
- Three consecutive no-progress actions finalize as `agent-abandoned`.
- Recovery events appear in the static report without private data.
