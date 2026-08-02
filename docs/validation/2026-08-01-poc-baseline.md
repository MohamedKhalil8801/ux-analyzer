# POC Baseline Validation

Status: dry-run only. No live OpenAI-compatible endpoint was called.

## Reproducibility

- Task-start git SHA: `6557ad88eb6305e4655781a25520606227c09f00`
- Project: `attention-guided-demo`
- Config digest: `a585597657168693c5de91f5132b3dc518f4458854399df7772954581f5b4d8a`
- Attention seed: one configured seed for core and ablations (`run_count: 1`); baseline uses seed `0`.
- Model trials: core and ablations use `[0]`; baseline uses `[0, 1]`.
- Live model IDs, completion, failures, cost, artifact roots, and model-trial variation: not available from dry-run.

## Reduced Matrix

Runtime budget override keeps both scenarios, both application versions, and each policy comparison while removing `impatient` from core and ablation matrices.

| Experiment | Scenarios | Versions | Persona | Policies | Seeds | Model trials | Runs |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| `core-pair` | `invite-teammate`, `enable-2fa` | defective, improved | `first-time-nontechnical` | `full-list`, `progressive-prominence-scent` | 1 | `[0]` | 8 |
| `ablations` | `invite-teammate`, `enable-2fa` | defective, improved | `first-time-nontechnical` | `prominence-ranked-list`, `progressive-prominence` | 1 | `[0]` | 8 |
| `baseline-model-trials` | `invite-teammate` | defective, improved | `first-time-nontechnical` | `full-list`, `progressive-prominence-scent` | `[0]` | `[0, 1]` | 8 |

`baseline-model-trials` has two model trials per semantic version/policy cell. It does not use five trials.

## Dry-Run Evidence

Command:

```text
rtk uv run uxa validate benchmarks/demo/project.yaml
```

Result:

```text
valid project: attention-guided-demo (2 scenarios, 2 personas, 4 experiments)
config digest: a585597657168693c5de91f5132b3dc518f4458854399df7772954581f5b4d8a
```

Command:

```text
rtk uv run uxa run benchmarks/demo/project.yaml --experiment core-pair --dry-run
```

Result: exit `0`; `run specs: 8`; `configured seeds: 1`; `deterministic seed repetitions suppressed: 0`; `maximum logical model calls: 512`; `overall run timeout: none`. Matrix contains both scenarios, both versions, `first-time-nontechnical`, and one run for each of two policies.

Command:

```text
rtk uv run uxa run benchmarks/demo/project.yaml --experiment ablations --dry-run
```

Result: exit `0`; `run specs: 8`; `configured seeds: 1`; `deterministic seed repetitions suppressed: 0`; `maximum logical model calls: 512`; `overall run timeout: none`. Matrix contains both scenarios, both versions, `first-time-nontechnical`, and one run for each of two ablation policies.

Command:

```text
rtk uv run uxa run benchmarks/demo/project.yaml --experiment baseline-model-trials --dry-run
```

Result: exit `0`; `run specs: 8`; `configured seeds: 1`; `maximum logical model calls: 512`; `deterministic seed repetitions suppressed: 0`; `overall run timeout: none`. Matrix contains two runs for each version under each of `full-list` and `progressive-prominence-scent`. Model trials are counted as separate accounting cells.

## Live Execution Deferred

Live baseline execution is explicitly deferred. User runtime-budget override requires lightweight local validation, and user forbids live OpenAI endpoint execution for this batch. Live runs would require configured endpoint credentials, external model calls, cost, and longer runtime. No live summaries, reports, completion results, failure counts, or model-trial variation are claimed here.
