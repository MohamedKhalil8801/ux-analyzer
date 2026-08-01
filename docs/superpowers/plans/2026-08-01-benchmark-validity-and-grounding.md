# Benchmark Validity And Cognitive Grounding Plan

## Goal

Keep synthetic UX metrics faithful by separating simulated user outcomes from
benchmark failures, resolving evaluation targets per application version, and
preventing cognitive actions from relying on unsupported interface guesses.

## Constraints

- Do not guarantee target discovery or task success.
- Keep prominence, attention, scent, cognition, and verification as separate
  inspectable layers.
- Keep heuristic, learned-saliency, and hybrid prominence providers
  interchangeable.
- Invalid benchmark executions must finalize, remain reportable, and be excluded
  from UX aggregates.
- Preserve existing bundle privacy and immutability guarantees.

## Tasks

- [x] Add per-version evaluation roles and correct demo target labels/roles.
- [x] Persist explicit UX-sample validity on every finalized result.
- [x] Skip UX evaluation for model, provider, safety, timeout, and internal failures.
- [x] Add invalid-run records and valid/invalid/outcome counts to experiment output.
- [x] Classify sanitized HTTP transport failures more precisely.
- [x] Ground cognitive decisions in revealed and remembered evidence without making
      the simulated user optimal.
- [x] Update CLI and report wording to distinguish execution, UX, and evaluation
      status.
- [x] Reserve one multi-element attention slot for cross-region exploration.
- [x] Use compact request-local scent IDs and map validated responses back to the
      snapshot.
- [x] Allow forgotten elements to be revealed again with a novelty penalty.
- [x] Run focused tests, all non-live tests, Ruff, Pyright, and diff checks.
- [x] Run one exact live cell after local verification; result was
      `verified-success` with a valid UX sample.
