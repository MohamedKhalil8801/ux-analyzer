# Domain Model

## Vocabulary

| Term | Meaning |
| --- | --- |
| `BenchmarkProject` | Immutable project owning applications, scenarios, personas, and experiment definitions. |
| `Application` | Target product identity with immutable presentation versions. |
| `ApplicationVersion` | One presentation variant. POC requires `defective` and `improved` kinds. |
| `Scenario` | Fixed goal, start state, fixture inputs, budgets, safeguards, eligible personas, and verifier contract. |
| `Persona` | Explicit simulation parameters: working-memory capacity, confidence, frustration, abandonment threshold, and attention temperature. |
| `ExperimentDefinition` | Cartesian-product selection of scenarios, versions, personas, policies, and seeds. |
| `RunSpec` | One immutable assignment of scenario, application version, persona, policy, seed, and config digest. |
| `Run` | Runtime state for one `RunSpec`, ordered events, snapshots, verification, manifests, and terminal outcome. |
| `ViewportSnapshot` | Immutable capture identity plus element snapshots, regions, graph edges, screenshot reference, and private provider references. |
| `ElementSnapshot` | One rendered element in one viewport, including private execution metadata. |
| `PersonaVisibleElement` | Safe projection of an element sent to persona/model logic. |
| `AttentionState` | Noticed and inspected IDs, focus region, budgets, bounded memory, confidence, frustration, failed candidates, subgoal, and current viewport. |
| `ProgressiveObservation` | One bounded public reveal with one to three new elements, remembered elements, and optional region context. |
| `CompleteObservation` | One complete visible persona-safe list used only by `full-list` and `prominence-ranked-list`. |
| `Finding` | Evidence-backed category/severity/reproducibility record. Unsupported human claims cannot become findings. |

Relationship:

```text
BenchmarkProject
  +-- Applications
  |     +-- ApplicationVersions: defective / improved
  +-- Scenarios
  +-- Personas
  +-- ExperimentDefinitions
          |
          +--> deterministic RunSpecs
                    |
                    +--> Run
                           +-- ViewportSnapshots
                           +-- AttentionState
                           +-- Ordered RunEvents
                           +-- VerificationResult
                           +-- ProviderManifests
                           +-- ArtifactChecksums
```

`ApplicationVersion` changes presentation only. Scenario goal and verifier stay
fixed while defective and improved versions are compared. Persona parameters
are experimental assumptions, not measurements of a real population.

## Snapshot Identity

`ViewportSnapshot.id` identifies one immutable capture. `ElementSnapshot.id`
identifies one element inside that capture. Same-looking elements across later
captures do not share identity automatically. Optional `lineage_id` can express
continuity; selectors are never identity.

An interaction's `PrivateExecutionReference` contains provider ID, viewport ID,
and provider token. It is valid only when all of these match the exact captured
element and current viewport. A stale viewport or different provider is rejected.

## Private and Persona-Visible Data

Private `ElementSnapshot` may contain:
`provider_id`, `execution_reference`, `selector`, `test_id`, `hidden_label`, and
`destination_url`.

`PersonaVisibleElement.model_dump()` contains only:

```text
id, role, label, bounds, visibility_fraction,
actionable, disabled, region_id
```

The cognitive request is narrower still. Each cognitive element contains
`element_id`, `role`, `label`, `actionable`, `disabled`, and `region_label`.
It does not receive bounds, selectors, test IDs, hidden labels, destination
URLs, provider IDs, execution references, numeric prominence, or numeric scent.
The coarse scent role receives glance-level element ID, role, label, region
label, and actionability. Full scent receives the same visible meaning plus
disabled state, and only for noticed elements.

## Attention and Action Rules

Current domain actions are:

- `notice-elements`
- `inspect-element`
- `interact-with-element`
- `scroll` up/down
- `wait`
- `back`
- `abandon` with reason

An interaction or inspection must target a noticed and remembered element in the
current viewport. Interactions additionally require actionable, non-disabled
state and a remaining interaction budget. Typed input is not invented by the
model: action validation resolves a fixture key through scenario inputs.

Progressive observations reveal one to three new elements. Complete list policies
preserve every visible persona-safe element, including lists larger than three.
All revealed IDs and optional region IDs must exist in the captured viewport.
Coarse scent may guide pre-notice sampling. Full scent requires the element to be
noticed first.

## Run State Machine

```text
created
   | RunStarted
   v
running
   | ViewportCaptured / ObservationRecorded /
   | ActionProposed / ActionExecuted / VerificationRecorded
   v
running
   | RunTerminated(outcome, verification, manifests,
   |               config_digest, artifact_checksums)
   v
finalized
```

`RunState.apply` is the sole lifecycle transition API. A run cannot start twice,
cannot record observations without a current viewport, cannot use a stale
viewport, and cannot accept events after finalization. Finalization requires an
independent verification result, provider manifests, matching configuration
digest, and at least one artifact checksum.

Terminal outcomes are closed:

`verified-success`, `agent-abandoned`, `budget-exhausted`, `timed-out`,
`provider-failure`, `model-failure`, `safety-blocked`, and `internal-error`.

Agent success claims are recorded independently. A claim never changes official
verification. This permits false-success measurement.

## Experiments and Verifier Independence

Current `core-pair` matrix:

```text
8 full-list cells x 1 seed
+ 8 progressive-prominence-scent cells x 10 seeds
= 88 runs
```

Policies are `full-list` and `progressive-prominence-scent`. `ablations` uses
`prominence-ranked-list` and `progressive-prominence`. Deterministic list
policies use only the first configured seed; progressive policies retain the
configured seed set.

Run IDs are SHA-256 hashes of experiment ID, scenario ID, application version
ID, persona ID, policy, seed, and config digest. Matrix order is deterministic.
Compared variant cells must share scenario, persona, policy, config digest, and
the exact seed set; only application version changes. The directional gate is:

1. Verified completion rate does not regress.
2. For pairs where both variants complete, improved paired-seed median
   discovery cost decreases.
3. For those jointly completed pairs, median wrong-action burden does not
   increase.
4. For those jointly completed pairs, median backtrack burden does not
   increase.

When defective does not complete and improved verifies for a paired seed, the
completion improvement dominates the failed run's lower effort. Otherwise an
early abandonment could be incorrectly scored as easier than completion.

`WebVerifier` uses the typed scenario verifier. `fixture-state` reads private
fixture state for the current session. `visible-result` captures a fresh public
snapshot and checks text and optional role. Neither verifier accepts the agent's
claim as official success.

## Evidence Classes

| Class | Meaning | Examples |
| --- | --- | --- |
| `deterministic-fact` | Directly measured or recorded by fixture/browser/run state. | Visibility, fold position, action count, scrolls, verification result, visible feedback. |
| `model-estimate` | Depends on prominence, scent, persona, memory, policy, or model provider. | Notice order, scent, confidence, frustration, abandonment, discovery cost. |
| `unsupported-human-claim` | Not inferable from synthetic runs alone. | Real-user completion, satisfaction, emotion, population validity, product demand. |

Findings and scorecards reject `unsupported-human-claim`. Reports keep supported
evidence separate and place excluded claims in limitations.
