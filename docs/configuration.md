# Configuration

`uxa` takes two kinds of configuration:

- a **project YAML** that defines what to analyze and how
- **environment variables** that define the model transport, pipeline toggles,
  and capture limits

---

# Part 1 — Project YAML

One file, validated on load. Unknown keys are rejected everywhere. Reference
integrity is checked too: unknown scenario/persona/version ids, duplicate ids,
and unresolved fixture keys are all load errors. The loader then canonicalizes
the document and hashes it, and that SHA-256 **config digest** becomes part of
every `RunSpec` and bundle manifest — so changing any referenced field changes
every run id.

Run `uv run uxa validate PROJECT` to check a file and print its digest.

## Root fields

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `id` | string, non-empty | yes | Project id. Also the default output folder name (`reports/<id>`). |
| `name` | string, non-empty | yes | Human label. |
| `applications` | list, at least 1 | yes | See [applications](#applications). |
| `scenarios` | list, at least 1 | yes | See [scenarios](#scenarios). |
| `personas` | list, at least 1 | yes | See [personas](#personas). |
| `experiments` | list, at least 1 | yes | See [experiments](#experiments). |
| `providers` | mapping | no | Defaults to an all-defaults block. See [providers](#providers). |
| `evaluation` | mapping | no | Defaults to an all-defaults block. See [evaluation](#evaluation). |
| `exploration` | mapping or absent | no | Crawl boundary for [`uxa explore`](exploration.md). See [exploration](#exploration). |

## `applications`

An application is a target product identity with one or more presentation
versions.

```yaml
applications:
  - id: fixture-app
    name: Controlled SaaS fixture
    versions:
      - id: fixture-app-defective
        kind: defective
        label: Defective
      - id: fixture-app-improved
        kind: improved
        label: Improved
```

### Version fields

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `id` | string, non-empty | yes | — | Unique across the whole project. |
| `kind` | `defective` \| `improved` \| `live` | yes | — | Presentation variant. |
| `label` | string, non-empty | yes | — | Shown in reports. |
| `start_url` | HTTPS URL or absent | `live` only | absent | Canonicalized on load. Required when `kind: live`. |
| `allowed_origins` | list of HTTP(S) origins | no | `[]` | Browser resource allowlist. Canonicalized; must be unique. |
| `navigation_settle_ms` | integer ≥ 0 | no | `0` | Wait after a navigation before capturing. |
| `action_settle_ms` | integer ≥ 0 | no | `0` | Wait after an in-page action before capturing. |

**Both `defective` and `improved` are required.** Unless every version of an
application is `live`, the application must carry at least one version of each
kind. A/B comparison needs both sides.

`kind: live` versions are what [`uxa explore`](exploration.md) generates for a
real site. Their sessions are strictly origin-scoped: a live session can reach
its own start origin and configured resource origins and nothing else.

## `scenarios`

A scenario is one task: a goal, a start state, a budget, a verifier, and the
people and versions allowed to attempt it.

```yaml
scenarios:
  - id: invite-teammate
    name: Invite teammate
    goal: Invite the supplied address to the workspace
    application_version_ids: [fixture-app-defective, fixture-app-improved]
    start_state: dashboard
    fixture_inputs:
      invite_email:
        value: demo-invite@example.test
        sensitive: true
    budget:
      max_steps: 24
      max_observations: 12
      max_interactions: 10
      timeout_seconds: null
    verifier:
      type: fixture-state
      resource: workspace
      field: invited_email
      operator: equals
      expected_fixture_key: invite_email
    safeguards: [fixture-only, test-account-only]
    eligible_persona_ids: [first-time-nontechnical, impatient]
    expected_evidence: [target-discovery, verified-completion]
    evaluation_target:
      labels_by_version:
        defective: Share
        improved: Invite teammate
      roles_by_version:
        defective: button
        improved: link
    viewport:
      width: 1280
      height: 800
```

### Scenario fields

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `id` | string, non-empty | yes | — | Unique in the project. |
| `name` | string, non-empty | yes | — | Shown in reports. |
| `goal` | string, non-empty | yes | — | The user's task, in user language. Sent to the model. |
| `application_version_ids` | list, at least 1 | yes | — | Must exist. |
| `start_state` | string, non-empty | yes | — | Named entry point for the target. |
| `fixture_inputs` | mapping of key → input | no | `{}` | Typed values the model may reference but never invent. Forbidden on live versions. |
| `budget` | mapping | yes | — | See [budget](#scenario-budget). |
| `verifier` | mapping | yes | — | See [verifier](#scenario-verifier). |
| `safeguards` | list of strings | no | `[]` | Free-form operator labels. |
| `eligible_persona_ids` | list, at least 1 | yes | — | Must exist. |
| `expected_evidence` | list of strings | no | `[]` | Free-form evidence labels the scenario is expected to produce. |
| `evaluation_target` | mapping | yes | — | The control the task is really about. See [evaluation target](#scenario-evaluation-target). |
| `viewport` | mapping | no | `{width: 1280, height: 800}` | Both values must be > 0. |

### Scenario `fixture_inputs`

```yaml
fixture_inputs:
  invite_email:
    value: demo-invite@example.test
    sensitive: true
  invite_role:
    value: member
    sensitive: false
```

| Field | Type | Required | Default |
| --- | --- | --- | --- |
| `value` | string | yes | — |
| `sensitive` | boolean | no | `false` |

Values marked `sensitive: true` become exact redaction values. Their text is
replaced with `[REDACTED]` in every persisted bundle, timeline, result, crash
marker, and model-call record, and the key is redacted wherever it appears as a
mapping key. See [security](security.md).

When the model types into a field it sends a **fixture key**, not a literal.
The application substitutes the configured value at the action boundary, so a
credential is never written by the model.

### Scenario `budget`

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `max_steps` | integer > 0 | yes | — | Total decisions the run may make. |
| `max_observations` | integer > 0 | yes | — | Bounded reveals. |
| `max_interactions` | integer > 0 | yes | — | Real browser interactions. |
| `max_model_calls` | integer > 0 | no | `64` | Ceiling on model calls. |
| `timeout_seconds` | number > 0 or `null` | no | `null` | Wall-clock deadline. `null` means progress-based termination only. |
| `stall_timeout_seconds` | number > 0 or `null` | no | `null` | No-progress cutoff. |

Set `timeout_seconds: null` when model latency is the dominant cost and you do
not want slow model calls to be recorded as a UX failure. Use
`stall_timeout_seconds` to stop a run that has stopped making progress.

### Scenario `verifier`

There are exactly three verifier kinds. The verifier is the authority on task
success; the model's own claim is recorded separately and never promotes an
outcome.

#### `fixture-state`

Reads private application state for the current session. Only valid against a
non-live target.

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `type` | `fixture-state` | yes | — |
| `resource` | string, non-empty | yes | Which state resource to read. |
| `field` | string, non-empty | yes | Field on that resource. |
| `operator` | `equals` \| `not-equals` \| `contains` \| `truthy` \| `falsy` | yes | Comparison. |
| `expected_fixture_key` | string, non-empty | yes | Must name a key in `fixture_inputs`. |

#### `visible-result`

Captures a fresh public snapshot and checks the rendered result. Used for live
targets, where private state is not available. Only pixel-visible rendered text
inside the current viewport counts; CSS-clipped, offscreen, transparent,
zero-font-size, and screen-reader-only text is not visible evidence.

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `type` | `visible-result` | yes | — | — |
| `text` | string, non-empty | yes | — | The result the user should be able to see. |
| `role` | string, non-empty or absent | no | absent | Narrow the match to one semantic role. |
| `all_of` | list of strings | no | `[]` | Every string must be visible. Entries must be non-empty and unique. |

For a `visible-result` scenario, only an explicit `complete` action by the
cognitive agent invokes verification. `complete` takes no element id and
consumes one step without a browser interaction. Ordinary successful actions
and terminal failure or abandonment never upgrade a run to verified success.

#### `colour-change`

Compares a recapture after the action against a recapture taken before it, and
fails when nothing visible changed beyond a threshold.

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `type` | `colour-change` | yes | — | — |
| `threshold` | number > 0 | no | `32.0` | Allowed per-pixel difference. |

### Scenario `evaluation_target`

The control the task is really about, so the report can attribute an outcome to
a specific interface element.

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `labels_by_version` | mapping of version id or kind → label | yes (non-empty) | — | Must cover every referenced version, either by version id or by `kind` (`defective`, `improved`, `live`). |
| `role` | string, non-empty or absent | no | absent | Optional shared semantic role. |
| `roles_by_version` | mapping of version id or kind → role | no | `{}` | Per-version roles, when the roles differ between variants. |
| `region_label` | string, non-empty or absent | no | absent | Optional containing region. |

Using the `kind` name as the key applies that label to every version of that
kind, which is what you want when the label is the same on both sides.

### Live-version constraints

If a scenario references any `live` application version:

- `fixture_inputs` must be empty
- `verifier` must not be `fixture-state`

A live scenario that violates either is a load error, because neither
mechanism exists against a real site.

## `personas`

A persona is an explicit set of simulation parameters, not a description. Its
values are experimental assumptions about the configuration, not measurements
of a population.

| Field | Type | Required | Constraint |
| --- | --- | --- | --- |
| `id` | string, non-empty | yes | Unique. |
| `name` | string, non-empty | yes | — |
| `working_memory_capacity` | integer | yes | 1–100 |
| `initial_confidence` | number | yes | 0–1 |
| `initial_frustration` | number | yes | 0–1 |
| `abandonment_threshold` | number | yes | 0–1 |
| `attention_temperature` | number | yes | > 0 |

```yaml
personas:
  - id: first-time-nontechnical
    name: First-time nontechnical user
    working_memory_capacity: 4
    initial_confidence: 0.55
    initial_frustration: 0.10
    abandonment_threshold: 0.72
    attention_temperature: 1.15
```

`working_memory_capacity` bounds how many observed elements the persona keeps
without degradation. `attention_temperature` scales how sharply the attention
policy concentrates on the highest-prominence candidates. `abandonment_threshold`
is the frustration level at which the persona gives up.

## `experiments`

An experiment selects a Cartesian product of scenarios, application versions,
personas, policies, seeds, model trials, and prominence providers.

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `id` | string, non-empty | yes | — | Unique. |
| `name` | string, non-empty | yes | — | — |
| `scenario_ids` | list, at least 1 | yes | — | Must exist. |
| `application_version_ids` | list, at least 1 | yes | — | Must exist. |
| `persona_ids` | list, at least 1 | yes | — | Must exist. |
| `policies` | list, at least 1 | yes | — | `full-list`, `prominence-ranked-list`, `progressive-prominence`, `progressive-prominence-scent`. |
| `seeds` | list of integers | no | `[]` | Empty means `0..run_count-1`. Must be unique. |
| `model_trials` | list of integers, at least 1 | no | `[0]` | External model replication axis, independent of the attention seed. |
| `prominence_provider_ids` | list of strings, at least 1 | no | `["heuristic"]` | Must be unique and resolvable. |
| `run_count` | integer > 0 | yes | — | Repeats per cell. |

```yaml
experiments:
  - id: core-pair
    name: Core unrestricted versus progressive attention pair
    scenario_ids: [invite-teammate, enable-2fa]
    application_version_ids: [fixture-app-defective, fixture-app-improved]
    persona_ids: [first-time-nontechnical]
    policies: [full-list, progressive-prominence-scent]
    model_trials: [0]
    run_count: 1
```

`--run-count` on the command line overrides `run_count` and clears any explicit
`seeds`.

### The four attention policies

| Policy | What the agent sees per observation |
| --- | --- |
| `full-list` | Every visible persona-safe element at once. |
| `prominence-ranked-list` | Every visible persona-safe element, ranked by prominence. |
| `progressive-prominence` | A bounded batch of new elements, chosen by prominence. |
| `progressive-prominence-scent` | The same, with a scent call ranking relevance to the goal. |

The first two are unrestricted baselines. The last two progressively reveal
interface information, which is what makes discovery cost and information scent
measurable.

## `providers`

Optional. Every field below has a default, so an empty `providers:` block is
valid.

### `providers.prominence`

| Field | Type | Default |
| --- | --- | --- |
| `version` | string | `heuristic-prominence-v1` |
| `weights` | mapping of feature name → number | `{}` |
| `temperature` | number > 0 | `1.0` |

`weights` is an open feature-name mapping. The shipped demo sets `area`,
`center_distance`, `reading_order`, `typography`, `contrast`, `isolation`,
`actionability`, `motion`, `competition`, and `occlusion`; negative weights are
penalties.

### `providers.attention`

| Field | Type | Default | Range |
| --- | --- | --- | --- |
| `version` | string | `progressive-attention-v4` | — |
| `batch_size` | integer | `2` | 1–3 |
| `cross_region_exploration` | integer | `1` | 0–2 |
| `prominence_weight` | number ≥ 0 | `1.0` | — |
| `coarse_scent_weight` | number ≥ 0 | `0.5` | — |
| `novelty_penalty` | number | `0.25` | 0–1 |
| `failure_penalty` | number | `0.5` | 0–1 |
| `recovery_scent_threshold` | number | `0.9` | 0–1 |
| `recovery_after_misses` | integer ≥ 1 | `2` | — |

`recovery_after_misses` is how many fruitless attempts must happen before the
policy starts offering elements it previously skipped, which is how a
low-prominence but goal-relevant control gets a second chance.

### `providers.expectation`

Versioned frozen-expectation documents: a baseline of what a persona should
notice, understand, and achieve, fixed before the run.

| Field | Type | Default |
| --- | --- | --- |
| `enabled` | boolean | `false` |
| `provider_id` | `frozen-expectation-v1` | `frozen-expectation-v1` |
| `documents` | list | `[]` |

`enabled: true` with no documents is a load error.

Each document:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `id` | string, non-empty | yes | — |
| `schema_version` | `frozen-expectation-v1` | yes | — |
| `application_version_id` | string, non-empty | yes | Must exist and be eligible for the scenario. |
| `scenario_id` | string, non-empty | yes | Must exist. |
| `persona_id` | string, non-empty | yes | Must exist, or the wildcard `"*"`. |
| `desired_outcomes` | list, at least 1 | yes | What should be achievable. |
| `required_invariants` | list of strings | no | What must stay true. |
| `acceptable_alternatives` | list of strings | no | Routes that are equally valid. |
| `reference_paths` | list of label paths | no | Known-good routes, e.g. `[dashboard, invite]`. |
| `effort_bounds` | mapping of name → number | no | e.g. `max_steps`, `max_wrong_actions`. |
| `warning_signals` | list of strings | no | What to look for as a warning. |

The triple (application version, scenario, persona) must be unique across
documents. A document is a baseline for comparison, not a UX judgment:
deviating from one reference path is not automatically a problem when
`acceptable_alternatives` covers the route the user actually took and the
desired outcomes and invariants still hold.

### `providers.saliency`

Optional; only relevant when you explicitly compare prominence providers. The
default prominence provider is `heuristic`, and no model is downloaded unless
you run `uxa models install`.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `model_set` | list of strings, at least 1 | `["foveacast-v0.2.0"]` | Must be non-empty and unique. |
| `precision` | `fp16` | `fp16` | — |
| `execution_provider_preference` | `auto` \| `cpu` \| `directml` | `auto` | Alias: `execution_preference`. |
| `cache` | mapping | see below | Experiment-scoped. |
| `aggregation` | mapping | see below | Saliency-plane to element aggregation. |
| `stage_selector` | mapping | see below | Search-stage duration mixture. |
| `fallback` | mapping | see below | Operational fallback provider. |

`cache`: `enabled` (default `true`), `scope` (default and only value
`experiment`).

`aggregation`: `version` (`element-saliency-aggregation-v1`),
`density_weight` (`0.60`), `robust_peak_weight` (`0.25`), `mass_share_weight`
(`0.15`), `temperature` (`1.0`), `meaningful_score_threshold` (`1e-9`),
`semantic_roles` (`button`, `checkbox`, `input`, `link`, `menu`, `tab`,
`text`), `structural_roles` (`other`).

`stage_selector`: `version` (`saliency-stage-selector-v1`), `temperature`
(`1.0`), `mixtures` (alias `stage_mixtures`), default:

```yaml
mixtures:
  initial:     { "1s": 1.0 }
  exploration: { "3s": 1.0 }
  persistent:  { "3s": 0.25, "7s": 0.75 }
```

`fallback`: `enabled` (default `true`), `provider_id` (default `heuristic`).

When the learned provider, its cache, or aggregation fails, the run continues
on the operational fallback, records a sanitized reason, and the sample is
marked invalid for comparison. A fallback run is not a valid learned result.

## `evaluation`

Optional. Every field has a default.

### `evaluation.discovery_cost`

Weights for the composite search-cost metric. Raw per-run counts stay visible
beside the composite.

| Field | Type | Default | Minimum |
| --- | --- | --- | --- |
| `version` | string | `discovery-cost-v1` | — |
| `inspection_cost` | number | `1.0` | 0 |
| `region_cost` | number | `1.0` | 0 |
| `scroll_cost` | number | `1.0` | 0 |
| `wrong_action_cost` | number | `1.0` | 0 |
| `backtrack_cost` | number | `1.0` | 0 |
| `uncertainty_cost` | number | `1.0` | 0 |
| `abandonment_penalty` | number | `1.0` | 0 |

### `evaluation.findings`

Thresholds for the deterministic finding rules. These produce
deterministic-fact-backed findings; they are not the model-authored findings
from report synthesis.

| Field | Type | Default | Range |
| --- | --- | --- | --- |
| `version` | string | `finding-rules-v1` | — |
| `weak_target_prominence_below` | number | `0.25` | 0–1 |
| `weak_scent_below` | number | `0.30` | 0–1 |
| `misleading_scent_margin` | number | `0.20` | 0–1 |
| `excessive_navigation_depth_at_least` | integer | `4` | ≥ 1 |
| `wrong_action_count_at_least` | integer | `2` | ≥ 1 |

### `evaluation.state_updates`

Deterministic deltas applied to persona confidence and frustration after each
action.

| Field | Type | Default |
| --- | --- | --- |
| `version` | string | `state-updates-v1` |
| `success_confidence_delta` | number | `0.05` |
| `success_frustration_delta` | number | `-0.1` |
| `failure_confidence_delta` | number | `-0.1` |
| `failure_frustration_delta` | number | `0.2` |

### `evaluation.report_synthesis`

| Field | Type | Default | Range |
| --- | --- | --- | --- |
| `enabled` | boolean | `false` | — |
| `max_retrieval_rounds` | integer | `3` | 1–5 |
| `max_adjudication_revisions` | integer | `1` | 0–2 |
| `max_final_verifications` | integer | `1` | 1–2 |

When enabled, `uxa run` attempts report synthesis automatically after the runs
finalize. `uxa run --no-synthesis` skips the attempt and still renders the
deterministic report. `UXA_REPORT_SYNTHESIS_ENABLED` overrides the YAML gate
when set to a non-empty value.

`uxa synthesize` requires `UXA_REPORT_MODEL`. `uxa run` does not: it attempts
synthesis best-effort and, with no report model configured, keeps the
deterministic findings and prints a warning. The report model must support the
structured JSON schemas plus image input whenever the evidence corpus contains
screenshots or heatmaps.

## `exploration`

Optional. Supplies the crawl boundary for [`uxa explore`](exploration.md).
Command-line flags win over these values.

```yaml
exploration:
  start_urls:
    - https://example.com
  depth: 2
  max_pages: 50
  max_scenarios: 8
  settle_ms: 10000
  respect_robots: false
```

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `start_urls` | list, at least 1 | yes | — | HTTPS only. Normalized and required to be unique. Alias: none. |
| `depth` | integer | no | `2` | 0–5. |
| `max_pages` | integer | no | `50` | 1–200. Must be ≥ the number of start URLs when `depth` is 0. |
| `max_scenarios` | integer | no | `8` | 1–20. |
| `settle_ms` | integer | no | `10000` | 0–15000. Alias: `page_settle_ms`. |
| `respect_robots` | boolean | no | `false` | Opt-in robots handling. |

There is no `--respect-robots` flag; the only way to change it is this YAML
field. `settle_ms` bounds how long the crawler waits for each page to settle
before capturing it, including its scroll sweep.

## Complete annotated example

```yaml
id: my-product
name: My product walkthrough

applications:
  - id: my-app
    name: My product
    versions:
      # A non-live application needs both a defective and an improved variant.
      - id: my-app-defective
        kind: defective
        label: Before
        navigation_settle_ms: 1200
      - id: my-app-improved
        kind: improved
        label: After
        navigation_settle_ms: 1200
      # A live version targets a real site. start_url is required and
      # allowed_origins is the browser resource allowlist for its sessions.
      - id: my-app-live
        kind: live
        label: Production
        start_url: https://app.example.com/dashboard
        allowed_origins:
          - https://app.example.com
          - https://fonts.googleapis.com
          - https://fonts.gstatic.com
        action_settle_ms: 600

scenarios:
  - id: invite-teammate
    name: Invite a teammate
    goal: Invite the supplied address to the workspace as a member
    application_version_ids: [my-app-defective, my-app-improved]
    start_state: dashboard
    fixture_inputs:
      invite_email:
        value: demo-invite@example.test
        sensitive: true     # redacted as [REDACTED] everywhere it is persisted
      invite_role:
        value: member
    budget:
      max_steps: 24
      max_observations: 12
      max_interactions: 10
      max_model_calls: 64
      timeout_seconds: null       # no wall-clock deadline
      stall_timeout_seconds: 90   # but no progress for 90s ends the run
    verifier:
      type: fixture-state
      resource: workspace
      field: invited_email
      operator: equals
      expected_fixture_key: invite_email   # must name a fixture_inputs key
    safeguards: [fixture-only, test-account-only]
    eligible_persona_ids: [first-time-nontechnical, impatient]
    expected_evidence: [target-discovery, information-scent, verified-completion]
    evaluation_target:
      # Key by version id, or by kind when the label is the same on both sides.
      labels_by_version:
        defective: Share
        improved: Invite teammate
      role: button
    viewport:
      width: 1280
      height: 800

  - id: find-invoice
    name: Find the latest invoice
    goal: Open the most recent invoice for this account
    application_version_ids: [my-app-live]
    start_state: dashboard
    # No fixture_inputs and no fixture-state verifier on a live version.
    budget:
      max_steps: 20
      max_observations: 12
      max_interactions: 8
      timeout_seconds: 600
    verifier:
      type: visible-result
      text: Amount due
      role: table
      all_of: [Invoice, Amount due]
    eligible_persona_ids: [first-time-nontechnical]
    evaluation_target:
      labels_by_version:
        live: Invoices        # `live` as a key applies to every live version
    viewport:
      width: 1280
      height: 800

personas:
  - id: first-time-nontechnical
    name: First-time nontechnical user
    working_memory_capacity: 4
    initial_confidence: 0.55
    initial_frustration: 0.10
    abandonment_threshold: 0.72
    attention_temperature: 1.15
  - id: impatient
    name: Impatient user
    working_memory_capacity: 3
    initial_confidence: 0.60
    initial_frustration: 0.18
    abandonment_threshold: 0.55
    attention_temperature: 0.85

providers:
  prominence:
    version: heuristic-prominence-v1
    temperature: 1.0
    weights:
      area: 0.16
      center_distance: 0.14
      reading_order: 0.10
      typography: 0.12
      contrast: 0.12
      isolation: 0.10
      actionability: 0.14
      motion: 0.05
      competition: -0.08
      occlusion: -0.09
  attention:
    version: progressive-attention-v4
    batch_size: 2
    cross_region_exploration: 1
    prominence_weight: 1.0
    coarse_scent_weight: 1.0
    novelty_penalty: 0.25
    failure_penalty: 0.5
    recovery_scent_threshold: 0.9
    recovery_after_misses: 2
  expectation:
    enabled: true
    provider_id: frozen-expectation-v1
    documents:
      - id: invite-teammate-improved-v1
        schema_version: frozen-expectation-v1
        application_version_id: my-app-improved
        scenario_id: invite-teammate
        persona_id: "*"          # wildcard: applies to every persona
        desired_outcomes:
          - The supplied address becomes a workspace member.
        required_invariants:
          - The invitee remains a member of the workspace.
        acceptable_alternatives:
          - Use the workspace team settings entry.
        reference_paths:
          - [dashboard, invite-teammate]
          - [dashboard, settings, team, invite]
        effort_bounds:
          max_steps: 24
          max_wrong_actions: 1
          max_backtracks: 1
        warning_signals:
          - The invitation control uses a broad or ambiguous action label.
  # saliency: only needed for explicit prominence-provider comparison.

evaluation:
  discovery_cost:
    version: discovery-cost-v1
    inspection_cost: 1.0
    region_cost: 1.0
    scroll_cost: 1.0
    wrong_action_cost: 1.0
    backtrack_cost: 1.0
    uncertainty_cost: 1.0
    abandonment_penalty: 1.0
  findings:
    version: finding-rules-v1
    weak_target_prominence_below: 0.25
    weak_scent_below: 0.30
    misleading_scent_margin: 0.20
    excessive_navigation_depth_at_least: 4
    wrong_action_count_at_least: 2
  state_updates:
    version: state-updates-v1
    success_confidence_delta: 0.05
    success_frustration_delta: -0.10
    failure_confidence_delta: -0.10
    failure_frustration_delta: 0.20
  report_synthesis:
    enabled: true                # requires UXA_REPORT_MODEL
    max_retrieval_rounds: 3
    max_adjudication_revisions: 1
    max_final_verifications: 1

exploration:
  start_urls:
    - https://app.example.com
  depth: 2
  max_pages: 50
  max_scenarios: 8
  settle_ms: 10000
  respect_robots: false

experiments:
  - id: core-pair
    name: Unrestricted versus progressive attention
    scenario_ids: [invite-teammate]
    application_version_ids: [my-app-defective, my-app-improved]
    persona_ids: [first-time-nontechnical]
    policies: [full-list, progressive-prominence-scent]
    seeds: [0]                # omit for 0..run_count-1
    model_trials: [0]
    prominence_provider_ids: [heuristic]
    run_count: 1
  - id: live-check
    name: Live production check
    scenario_ids: [find-invoice]
    application_version_ids: [my-app-live]
    persona_ids: [first-time-nontechnical]
    policies: [progressive-prominence-scent]
    run_count: 1
```

---

# Part 2 — Environment variables

Values come from the process environment first, then from a `.env` file in the
current working directory. Existing environment values are never overridden.

A variable set to the empty string counts as unset for every flag below. That
matters because a `VAR=` line copied out of [`.env.example`](../.env.example)
must not silently flip behavior.

**Never put a real key in a project YAML, in source, in a test, or in
documentation.** The config digest hashes the whole project file, so a key in
YAML propagates into every run manifest.

## Model transport

| Variable | Default | If unset or empty |
| --- | --- | --- |
| `UXA_LLM_MODE` | `api` | API mode. Any value other than `api` or `codex` is rejected with `UXA_LLM_MODE must be one of: api, codex`. |
| `UXA_LLM_BASE_URL` | — | **Required in `api` mode.** Unset is a `missing model environment variables: UXA_LLM_BASE_URL` error. Must be HTTP(S); a URL with credentials is rejected. |
| `UXA_LLM_API_KEY` | — | **Required in `api` mode.** Sent as a bearer header only. Never persisted. |
| `UXA_LLM_TIMEOUT_SECONDS` | `30` | 30-second per-request timeout. `none`, `off`, or `unlimited` disables it entirely. `0` or negative is rejected. |
| `UXA_LLM_MAX_CONCURRENT_CALLS` | `2` | Client-side cap of 2 concurrent model calls, independent of `--workers`. |
| `UXA_LLM_REQUEST_MAX_BYTES` | `750000` | Serialized request byte ceiling. Unparseable values fall back to 750000; values below `100000` are clamped up to `100000`. A request over the ceiling is refused before transport. |
| `UXA_LLM_SESSION_ID` | unset | No provider session identifier is sent. |
| `UXA_SCENT_MODEL` | — | **Required.** Missing is a `missing model environment variables: UXA_SCENT_MODEL` error. |
| `UXA_COGNITIVE_MODEL` | — | **Required.** |
| `UXA_REPORT_MODEL` | — | Required by `uxa validate --check-env` and by `uxa synthesize` when report synthesis is enabled. During `uxa run` it is optional: if the gate is on and it is unset, the run still finalizes and prints `warning: report synthesis unavailable; deterministic findings retained (ModelConfigurationError)`. Also the fallback for `UXA_REDESIGN_MODEL`. |
| `UXA_REDESIGN_MODEL` | `UXA_REPORT_MODEL` | `uxa redesign` uses the report model. |
| `UXA_LLM_SCENT_REASONING_EFFORT` | unset | Provider default effort. One of `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`. |
| `UXA_LLM_COGNITIVE_REASONING_EFFORT` | unset | Provider default effort. |
| `UXA_LLM_REPORT_REASONING_EFFORT` | unset | Provider default effort. Also applied to the redesign roles. |

In `codex` mode `UXA_LLM_BASE_URL` and `UXA_LLM_API_KEY` are neither read nor
required; the `codex` CLI on `PATH` supplies authentication. The reasoning-effort
variables are forwarded to the CLI as `codex exec -c
model_reasoning_effort=<value>`.

## Pipeline toggles

| Variable | Default | If unset or empty |
| --- | --- | --- |
| `UXA_REPORT_SYNTHESIS_ENABLED` | project YAML decides | The project's `evaluation.report_synthesis.enabled` applies. Any non-empty value other than `0`, `false`, `no`, or `off` enables synthesis regardless of YAML; those four disable it. |
| `UXA_REDESIGN_ENABLED` | off | No redesign pass runs after `uxa run`. Set to `1`, `true`, `yes`, or `on` (case-insensitive) to run one automatically. An incomplete model environment produces `warning: redesign enabled but model environment incomplete; skipping` and the run continues. |
| `UXA_TRACE_SCREENCAST` | on | Trace archives include screencast frames, which is what produces replay video. Set to a falsey value to shrink trace archives substantially; snapshots and network records are kept either way. |
| `UXA_RUN_LIVE_TESTS` | off | Live endpoint tests stay skipped. The CLI removes this variable from the environment right after loading `.env`, unless it was already set in the real process environment — a `UXA_RUN_LIVE_TESTS=1` line in `.env` is deliberately ignored so it can never influence a production run. |

## Capture limits

| Variable | Default | If unset, unparseable, or below 1 |
| --- | --- | --- |
| `UXA_REDESIGN_MAX_PAGES` | `10` | Falls back to 10. Bounds the shared page list used by the live-page audit and the redesign pass. |
| `UXA_REDESIGN_MAX_PAGE_HEIGHT` | `12000` | Falls back to 12000. Per-page capture height ceiling in CSS px. Content below the ceiling is **never captured and never shown to the model**; the report discloses the limitation. See [truncated capture](troubleshooting.md#page-capture-is-truncated). |

## Local storage locations

| Variable | Default | If unset |
| --- | --- | --- |
| `UXA_MODEL_HOME` | the platform's per-user data directory for `ux-analyzer` | Model releases live in the platform default location. |
| `UXA_SKILL_SETS` | the platform's per-user config directory for `ux-analyzer` | Fix-export skill sets are read from the platform default path. A missing file means no skill sets are configured, which is not an error. |

## Reporting and audits

| Variable | Default | If unset |
| --- | --- | --- |
| `UXA_PSI_API_KEY`, `PSI_API_Key`, `PSI_API_KEY`, `GOOGLE_API_KEY` | unset | PageSpeed Insights runs unauthenticated against the shared quota, which is frequently exhausted. Names are checked in exactly that order and the first usable value wins; a value is only usable if it is 10–200 characters drawn from `A–Z a–z 0–9 . _ ~ + / = -`. |
| `UXA_SKIP_PAGESPEED` | unset | `uxa run` performs its PageSpeed Insights pass. Set to a truthy value to skip it. Any value other than empty, `0`, `false`, `False` counts as a skip. |
| `UXA_PAGESPEED_WEB_LINKS` | unset | No saved-report link capture during `uxa run`. Set to `1`, `true`, or `True` to enable it. It triggers a second, independent Lighthouse analysis whose scores are labeled separately. |
| `UXA_SLOP_DISABLE_JS` | unset | `uxa slop` renders the page with JavaScript enabled. **Any non-empty value** — including `0` — disables JavaScript, which is what pins the DOM for deterministic comparison against a frozen local file. To re-enable it, unset the variable or set it to the empty string. |

## Quick reference

```powershell
$env:UXA_LLM_MODE = "api"
$env:UXA_LLM_BASE_URL = "https://<provider-host>/v1"
$env:UXA_LLM_API_KEY = "<api-key>"
$env:UXA_SCENT_MODEL = "<scent-model-id>"
$env:UXA_COGNITIVE_MODEL = "<cognitive-model-id>"
$env:UXA_REPORT_MODEL = "<report-model-id>"
$env:UXA_REDESIGN_MODEL = "<redesign-model-id>"     # optional
$env:UXA_PSI_API_KEY = "<google-api-key>"            # optional
```
