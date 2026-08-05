# Original POC Plan Versus Current Implementation

This document compares the original implementation plan in
[`docs/superpowers/plans/2026-07-28-attention-guided-ui-agent-poc.md`](superpowers/plans/2026-07-28-attention-guided-ui-agent-poc.md)
with the current implementation. It records intentional extensions,
methodological changes, and acceptance work that remains outstanding. The
original plan remains historical and is not edited to match later decisions.

## Summary

The core POC architecture remains as planned: Python 3.12 modular monolith,
platform-neutral domain and application layers, Playwright Chromium, bundled
FastAPI fixture, heuristic prominence, progressive persona-safe exposure,
structured OpenAI-compatible roles, independent verification, immutable run
bundles, and filesystem-openable HTML reports.

The implementation has since been changed to make real runs terminate and
produce interpretable evidence before scaling the full matrix. The largest
changes are exact single-run execution, checkpoint/resume, additional terminal
guards, progressive-attention recovery, separate attention seed and external
model-trial identity, and separate UX-effort and benchmark-cost reporting.
Task 14 also added a conditional Foveacast provider decision record without
changing the heuristic default because real model and operational gate evidence
was unavailable.

## Execution And Reliability Changes

| Area | Original POC plan | Current implementation | Reason and effect |
| --- | --- | --- | --- |
| Exact run selection | `uxa run` and `uxa ablate` expand experiment matrices. | `uxa run-one` resolves exactly one scenario/version/persona/policy/seed cell and prints `run specs: 1`. | Allows one real cell to be debugged without silently launching the other matrix cells. |
| Focused validation | Core and ablation experiments only. | `focused-validation` contains four invite-teammate cells: defective/improved crossed with full-list/progressive-scent for one persona and seed. | Provides a balanced smoke comparison before expensive full execution. It is not statistical validation. |
| Resume | Experiment runner returned partial results, but no persisted resume contract was planned. | `uxa run --resume` uses atomic `experiment-progress.json`, validates finalized bundles, skips only trusted selected runs, archives interrupted staging evidence, and aggregates skipped plus new results. | Long runs can be resumed without losing finalized work or producing an incomplete summary. |
| Overall timeout | `timed-out` was a required outcome and timeout behavior was tested. | Scenario `timeout_seconds` is optional and the demo sets it to `null`; CLI reports `overall run timeout: none`. HTTP, model, verifier, and browser operations retain finite safety timeouts. | A single wall deadline previously misclassified slow but active runs. Non-time resource guards now bound the benchmark loop. |
| Budgets | Steps, observations, interactions, and timeout. | Adds `max_model_calls` and explicit counters for repeated fixture input, identical no-progress actions, semantic cycles, and exhausted visible attention. | Prevents unbounded model spending and repetitive behavior while preserving an actionable terminal reason. |
| Semantic cycles | Only adjacent wrong-action/budget behavior was planned. | Safe action/result-state suffixes detect cycles of length 2-4. Failed actions break the history. Repeated cycles finalize as `agent-abandoned`. | Alternating loops such as Share/Close no longer consume the full budget. |

## Attention And Cognitive Changes

| Area | Original POC plan | Current implementation | Reason and effect |
| --- | --- | --- | --- |
| Progressive batch | Configurable one-to-three element batch selected through seeded region-first attention. | Demo default is batch size 2. Normal selection remains seeded; bounded recovery can reveal up to three high-value unobserved elements after no progress. | Reduces single-element tunnel vision without turning progressive exposure into a full-list policy. |
| Recovery | Novelty and failed-candidate penalties were planned, but no explicit recovery level existed. | `RunAgent` derives safe semantic progress from recaptured UI state and passes a bounded recovery level to the attention policy. Recovery and terminal no-progress events are persisted. | A failed interaction changes the next exposure opportunity instead of repeating the same scan indefinitely. |
| Cognitive context | Cognitive role received a progressive observation and memory. | Cognitive v2 also receives previous action/result, viewport ID, remembered elements, completed fixture keys, fixture-completion state, available controls, and persona behavior parameters. | Gives the model enough safe state to avoid repeating completed input and to choose submit/confirm controls. |
| Model response failures | Structured schema validation and terminal model failure were planned. | Unknown scent element IDs and incomplete/unknown cognitive actions are normalized into sanitized `model-failure` events and outcomes. | Provider validation failures no longer become opaque Pydantic-only `internal-error` results. |
| Attention exhaustion | Not separately classified. | When all visible elements have been examined without supported progress, the run records `attention-exhausted` and finalizes as `agent-abandoned`. | Exhausted search is treated as UX evidence, not a software crash. |

The cognitive role still does not receive selectors, test IDs, execution
references, private fixture values, verifier state, destination URLs, or
numeric prominence/scent scores.

## Experiment And Evaluation Changes

| Area | Original POC plan | Current implementation | Reason and effect |
| --- | --- | --- | --- |
| Core and ablation matrices | Historical ten-seed expansion across every semantic cell. | `core-pair` and `ablation` each contain 8 specs. Attention seed and external model trial remain separate axes. | Current matrices state exact semantic specs without conflating local attention randomness with external model variation. |
| Baseline replication | No separate external model-trial axis. | Four semantic cells x two model trials = 8 specs, with attention seed held fixed. | Measures external model variation without changing scenario, version, persona, policy, or attention seed. |
| Historical matrix | Old 88-run result combined deterministic-list suppression with ten progressive seeds. | Retained as historical evidence only; it is not current matrix semantics. | Prevents old run counts from being read as current experiment definitions. |
| Directional gate | Improved must lower paired median discovery cost, not increase wrong actions/backtracks, and not regress completion. | For a pair where defective fails and improved verifies, completion improvement dominates the failed run's artificially cheap effort. Effort gates still apply to pairs where both variants complete, and completion regression always fails. | Prevents early abandonment from being scored as easier than successful completion. |
| Effort reporting | Discovery cost and its components were the main effort measure. | Report separates simulated user effort (actions, observations, discovery cost, `simulated-task-time-v1`) from analysis cost (model calls, attempts, latency, tokens). | Slow model inference is benchmark operating cost, not simulated user task time. |
| Time estimate | No task-time formula. | `simulated-task-time-v1` uses `1.35 seconds * observations + 1.1 seconds * actions`. | Provides a directly comparable deterministic proxy. It is not measured or calibrated human time. |
| Report integrity | Report included finalized, failed, and staging evidence with redaction. | Finalized bundle validation additionally requires full checksum coverage, valid JSON, matching run IDs, terminal outcome/event structure, and complete selected-matrix aggregation after resume. | Prevents malformed or incomplete bundles from being treated as trusted UX samples. |

## Reproducibility Deviation

The original global constraint says every stochastic choice uses the recorded
run seed. This is true for local attention sampling and deterministic memory
updates. It is not currently guaranteed for the external OpenAI-compatible
provider: requests do not pass a provider seed, and provider-side model
sampling may vary between otherwise identical runs.

Consequences:

- A benchmark seed identifies the local attention path, not every source of
  model nondeterminism.
- Suppressing repeated full-list seeds removes duplicate local attention
  trials, but does not prove the cognitive model is deterministic.
- Statistical claims require explicit provider sampling controls or repeated
  model trials recorded separately from attention seeds.

## Validation Status Difference

The original completion gate calls for the complete two-scenario,
two-version, two-persona, paired-seed core benchmark. That full live benchmark
has not been rerun after the reliability changes. Recent live validation used
the four-cell `focused-validation` experiment plus exact `run-one` retries.
This verifies execution and failure classification, but it does not satisfy the
original full-matrix statistical acceptance condition.

## Saliency Integration Status

Foveacast is no longer only a deferred contract. The current implementation has
an explicit provider boundary, pinned six-artifact registry, CPU/DirectML
selection, three-duration maps, element aggregation, search-stage selection,
experiment-scoped cache, replay artifacts, redaction, and heuristic fallback.
`ElementSnapshot` remains deterministic; saliency maps, profiles, operational
prominence, timings, provider identity, and cache state remain model-dependent
evidence.

Task 13 deterministic fake acceptance passed:

```text
rtk uv run pytest tests/e2e/test_saliency_benchmark.py -m "not live and not directml" -q
13 passed, 1 deselected
```

It exercised all eight focused cells through production orchestration and
evaluation, but its synthetic maps and operational measurements cannot promote
a provider or support human claims. Task 14 explicitly installed all six
pinned artifacts. Initial status reported `runtime missing`; an autonomous
`uv sync --extra saliency-cpu` then installed the CPU runtime. The real CPU
known-screenshot gate reached session validation but failed before valid
inference because the pinned model exposed a symbolic input shape while the
adapter requires fixed positive integers. DirectML status reported no available
provider. The exact real eight-cell run with `--resume` was not executed, so no
real paired completion, rank, alignment, latency, RSS, parity, or fallback
result exists. LLM environment validation separately reported configured
settings, but no real endpoint call is claimed.

The conditional ADR retains `heuristic-prominence-v1` as default and keeps
Foveacast explicit opt-in. Hybrid is not justified without concrete reviewed
complementary errors. SUM, full saliency matrix, full UEyes reproduction,
frozen expectations, human calibration, and statistical calibration remain
deferred. Six inherited Task 12 CLI tests fail around resume, legacy bundle
compatibility, completion aggregation, and provider comparison grouping; they
remain visible and are part of the next review trigger.

## Unchanged Deferred Boundaries

The following remain deferred or out of current saliency decision scope:

- Additional pretrained saliency challengers such as UMSI++ and SUM.
- Frozen expected-path generation before interface exposure.
- Desktop and mobile observation providers.
- Human and statistical calibration.
- Production API and issue-tracker integrations.

Expected paths are therefore not currently generated before exposure. When
implemented, they must be persisted before first capture and reported
separately from visual discoverability, task efficiency, and learnability.

## Maintenance Rule

Update this document when a later implementation changes the original POC's
matrix semantics, model exposure boundary, terminal outcomes, evaluation gate,
or deferred-provider status. Operational details that do not change those
contracts belong in the architecture, testing, or run-bundle documentation.
