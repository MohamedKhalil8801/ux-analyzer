# Domain Model

## Vocabulary

| Term | Meaning |
| --- | --- |
| `BenchmarkProject` | Immutable project owning applications, scenarios, personas, and experiment definitions. |
| `Application` | Target product identity with immutable presentation versions. |
| `ApplicationVersion` | One presentation variant. POC requires `defective` and `improved` kinds. |
| `Scenario` | Fixed goal, start state, fixture inputs, budgets, safeguards, eligible personas, and verifier contract. |
| `Persona` | Explicit simulation parameters: working-memory capacity, confidence, frustration, abandonment threshold, and attention temperature. |
| `ExperimentDefinition` | Cartesian-product selection of scenarios, versions, personas, policies, prominence providers, attention seeds, and external model trials. |
| `RunSpec` | One immutable assignment of scenario, application version, persona, policy, prominence provider, attention seed, model trial, and config digest. |
| `Run` | Runtime state for one `RunSpec`, ordered events, snapshots, verification, manifests, and terminal outcome. |
| `ViewportSnapshot` | Immutable capture identity plus element snapshots, regions, graph edges, screenshot reference, and private provider references. |
| `ElementSnapshot` | One rendered element in one viewport, including private execution metadata. |
| `PersonaVisibleElement` | Safe projection of an element sent to persona/model logic. |
| `AttentionState` | Noticed and inspected IDs, focus region, budgets, bounded memory, confidence, frustration, failed candidates, subgoal, and current viewport. |
| `ProgressiveObservation` | One bounded public reveal with one to three new elements, remembered elements, and optional region context. |
| `CompleteObservation` | One complete visible persona-safe list used only by `full-list` and `prominence-ranked-list`. |
| `SaliencyPrediction` | One model-dependent normalized saliency plane with duration, model identity, geometry, provider, and timing provenance. |
| `ElementSaliencyAggregate` | Model-dependent evidence derived by sampling one saliency plane inside viewport-clipped element bounds. |
| `ElementAttentionProfile` | Immediate, early, eventual, and optional general element estimates plus aggregate provenance. |
| `OperationalProminence` | Search-stage selection from attention profiles, with provider and stage provenance. |
| `SearchStage` | `initial`, `exploration`, or `persistent` selection context; duration labels are not wall-clock timers. |
| `SaliencyCache` | Experiment-scoped, checksum-covered cache keyed by screenshot, geometry, model, preprocessing, precision, and actual execution provider. |
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
                           +-- SaliencyPrediction / ElementAttentionProfile
                           +-- OperationalProminence
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

## Saliency evidence boundary

`ViewportSnapshot` and `ElementSnapshot` remain deterministic interface facts and
are not modified by model output. `SaliencyPrediction`,
`ElementSaliencyAggregate`, `ElementAttentionProfile`, and
`OperationalProminence` are model-estimate evidence. Provider manifests retain
model version, checksums, preprocessing, execution provider, timing, cache, and
fallback provenance.

The configured stage mixtures are:

```text
initial:     immediate 1.00
exploration: early     1.00
persistent:  early     0.25, eventual 0.75
```

All three duration maps are inferred and cached together for one exact
screenshot, so changing search stage does not infer again. Cache entries stay
under the experiment output. If runtime, cache, or aggregation fails, the
operational provider returns explicit heuristic fallback, records the reason,
and the learned comparison sample is invalid.

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

Current controlled matrices:

```text
core-pair: 8 specs
ablation: 8 specs
baseline replication: 4 semantic cells x 2 model trials = 8 specs
saliency-focused-validation: 2 scenarios x 2 versions x 2 prominence providers = 8 specs
```

Attention `seed` and external `model_trial` are separate axes. Core and ablation
matrices use one model trial per semantic spec. Baseline replication keeps one
attention seed and varies two external model trials across four semantic cells.
The former 88-run matrix with ten seeds is historical evidence, not current
matrix semantics.

Run IDs are SHA-256 hashes of experiment ID, scenario ID, application version
ID, persona ID, policy, attention seed, and config digest. Non-default
`model_trial` and non-default `prominence_provider_id` are added conditionally.
Default `model_trial=0` and heuristic prominence therefore retain legacy run-ID
encoding for migration and resume compatibility. This is an intentional
deviation from the plan's unconditional axis encoding; manifests and reports
still record those axes.

Comparison pairing fixes all axes except axis under comparison. Version pairs
keep prominence provider fixed and vary only application version. Provider pairs
keep application version fixed and vary only prominence provider. Model-trial
pairs keep both version and provider fixed and vary only model trial. Scenario,
persona, policy, config digest, attention seed, and every other non-compared axis
remain fixed. The directional gate is:

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
