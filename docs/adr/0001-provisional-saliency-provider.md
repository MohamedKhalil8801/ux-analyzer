# ADR 0001: Provisional Saliency Provider

- Status: Proposed
- Date: 2026-08-05
- Scope: Foveacast saliency integration and focused provider comparison
- Decision owner: UX Analyzer maintainers

## Decision summary

Keep `heuristic-prominence-v1` as default provider. Keep `foveacast`
(`foveacast-v0.2.0-fp16`) as explicit opt-in only. Do not enable a hybrid
provider and do not change default configuration from this ADR.

This ADR remains Proposed because real focused promotion gate was not executed
or passed. It can become Accepted only after review trigger below completes with
real model output and persisted operational evidence.

## Context and baseline

Current branch at review start was `94ab617` (`test: add focused saliency
validation`). Earlier POC baseline in
`docs/validation/2026-08-01-poc-baseline.md` is dry-run only and explicitly
contains no live OpenAI-compatible endpoint result. Its baseline provider is
heuristic prominence; attention seed and external model trial remain separate
axes.

Current configuration validation passed:

```text
rtk uv run uxa validate benchmarks/demo/project.yaml
valid project: attention-guided-demo (2 scenarios, 2 personas, 5 experiments)
config digest: 4c7ba37a227fcbf76eee0e3feb830b738cd3447d2e7d977d3e27a21dde514c5d
```

Task 13 deterministic fake acceptance exercised all eight focused cells through
production orchestration, persisted bundles, evaluation, and report rendering.
It is orchestration evidence only. It is not human calibration, real-model
evidence, or promotion evidence.

## Approved model artifacts

Source is GitHub release `khawkins98/foveacast-training` tag `v0.2.0`. The
explicit install command returned exit code 0 and downloaded all six artifacts.
The registry verifies each artifact SHA-256 before atomic publication. At the
initial Task 14 check, runtime status was `runtime missing`. The autonomous
dependency attempt was then made:

```text
rtk uv sync --extra saliency-cpu
Resolved 48 packages in 1ms
Downloaded onnxruntime
Installed flatbuffers==25.12.19, onnxruntime==1.28.0, protobuf==7.35.1
exit code: 0
```

The command also emitted a non-fatal hardlink warning and fell back to copying
files. Afterward, CPU status returned `ready`. DirectML status returned
`unsupported provider`, with available providers
`AzureExecutionProvider, CPUExecutionProvider`.

| Duration | Artifact | SHA-256 |
| --- | --- | --- |
| 1s | `foveacast-v3-1s-fp16.onnx` | `4b9fdc2734e36c612a120ab7b0050ae276723160ccc625c6f554e906dd6345d5` |
| 3s | `foveacast-v3-3s-fp16.onnx` | `842a23f97908d146b8749e05f6b220bdb495eae76c75cc7252550825585ef76e` |
| 7s | `foveacast-v3-7s-fp16.onnx` | `cf66388dc6fe5db4712c77cf3929d04380ed73ac963677b8c9de2b6380fb51e0` |
| 1s | `foveacast-v3-1s-fp16.parity.json` | `2572f301f49bf49ec38bcff57fa1282380726e79f1ce4c9a5cd56f9f4894944b` |
| 3s | `foveacast-v3-3s-fp16.parity.json` | `3ccbd9da648baaf7c0ddca9fce1e56d92ce71f94de2f2751b2dd2af41c276a20` |
| 7s | `foveacast-v3-7s-fp16.parity.json` | `16cb94c448a6dc510e7d44d31147634750bb3d1801cfa55daf0bcf83e1d97cca` |

License chain:

- Foveacast training code: MIT, attribution `khawkins98/foveacast-training`.
- MSI-Net architecture: MIT, attribution `MSI-Net authors`.
- UEyes dataset-derived weights: CC BY 4.0, attribution `Jiang et al. 2023`.
- ONNX Runtime: MIT, attribution `ONNX Runtime authors`.

Required explicit command and result:

```text
rtk uv run uxa models install foveacast-v0.2.0 --precision fp16
foveacast-v0.2.0: downloaded
artifacts: foveacast-v3-1s-fp16.onnx, foveacast-v3-3s-fp16.onnx, foveacast-v3-7s-fp16.onnx, foveacast-v3-1s-fp16.parity.json, foveacast-v3-3s-fp16.parity.json, foveacast-v3-7s-fp16.parity.json
license attribution: Foveacast training code: MIT (khawkins98/foveacast-training); MSI-Net architecture: MIT (MSI-Net authors); UEyes dataset-derived weights: CC BY 4.0 (Jiang et al. 2023); ONNX Runtime: MIT (ONNX Runtime authors); Use of UEyes dataset-derived weights requires attribution to Jiang et al. 2023.
```

No inference path downloads artifacts or runtimes automatically.

The initial plan status command also returned:

```text
rtk uv run uxa models status foveacast-v0.2.0
foveacast-v0.2.0: runtime missing
diagnostic: install saliency-cpu or saliency-directml extra
license attribution: Foveacast training code: MIT (khawkins98/foveacast-training); MSI-Net architecture: MIT (MSI-Net authors); UEyes dataset-derived weights: CC BY 4.0 (Jiang et al. 2023); ONNX Runtime: MIT (ONNX Runtime authors); Use of UEyes dataset-derived weights requires attribution to Jiang et al. 2023.
```

The LLM credential check was separate from saliency runtime readiness:

```text
rtk uv run uxa validate benchmarks/demo/project.yaml --check-env
valid project: attention-guided-demo (2 scenarios, 2 personas, 5 experiments)
model environment: configured (API key present; scent and cognitive models configured)
```

No real endpoint call is claimed. Credential availability did not supply model
or focused-comparison evidence.

## Focused comparison matrix

The required matrix is exactly eight cells:

```text
2 scenarios: invite-teammate, enable-2fa
x 2 versions: fixture-app-defective, fixture-app-improved
x 1 persona: first-time-nontechnical
x 2 providers: heuristic, foveacast
x 1 policy: progressive-prominence-scent
x seed 0
x model trial 0
= 8 cells
```

The intended real command was the saliency-focused experiment with explicit
Foveacast provider cells and `--resume`:

```text
rtk uv run uxa run benchmarks/demo/project.yaml --experiment saliency-focused-validation --workers 4 --fixture-origin http://127.0.0.1:8000 --output reports/saliency-focused-validation --resume
```

It was not executed. CPU adapter validation failed before valid learned output
and DirectML was unavailable, so a real eight-cell run could not produce
learned output. No LLM endpoint result, completion result, paired rank result,
or focused report from real Foveacast cells is claimed here. The CLI has no
provider-selection option; a future gate must set explicit CPU or DirectML
configuration and record requested versus actual provider rather than relying
on `auto`.

## Evidence collected

### Deterministic fake acceptance

```text
rtk uv run pytest tests/e2e/test_saliency_benchmark.py -m "not live and not directml" -q
13 passed, 1 deselected in 29.80s
```

This passed fake-map checks for three duration maps, element profiles, stage
selection without reinference, experiment-scoped cache miss/hit behavior,
artifacts, reports, provider identity, fallback invalidation, and redaction.
Fake operational values are explicitly synthetic and cannot satisfy promotion.

CPU adapter fake suite also passed:

```text
rtk uv run pytest tests/integration/saliency/test_foveacast_cpu.py -m "not live" -q
13 passed, 1 deselected in 1.77s
```

These tests verify preprocessing, session reuse, fixed duration order, output
validation, and metadata contracts with a fake runtime. They do not measure
real model behavior.

### Real CPU evidence

```text
rtk uv run pytest tests/integration/saliency/test_foveacast_cpu.py -m live -rs -q
1 failed, 13 deselected in 0.95s
ValueError: input shape must be fixed positive integers
```

The real CPU test reached pinned model session validation but failed before valid
inference because the model exposed a symbolic input shape and the adapter
requires fixed positive integers. Actual valid maps, model-load completion,
adapter/device ID, cold load, sequential 1s/3s/7s warm latency, peak RSS, output
parity, map alignment, ranked elements, target/distractor behavior, and real
cache reuse are unavailable.

### DirectML evidence

The fake provider-selection suite passed, but no hardware claim follows:

```text
rtk uv run pytest tests/integration/saliency/test_foveacast_directml.py -q
9 passed, 1 skipped in 1.14s

rtk uv run pytest tests/integration/saliency/test_foveacast_directml.py -m directml -rs -q
SKIPPED: pinned Foveacast DirectML artifacts/runtime unavailable: unsupported provider; available providers: AzureExecutionProvider, CPUExecutionProvider
1 skipped, 9 deselected
```

Status was also checked separately for CPU and DirectML:

```text
initial Task 14 check:
rtk uv run uxa models status foveacast-v0.2.0 --provider cpu
foveacast-v0.2.0: runtime missing
diagnostic: install saliency-cpu or saliency-directml extra

after uv sync:
rtk uv run uxa models status foveacast-v0.2.0 --provider cpu
foveacast-v0.2.0: ready

rtk uv run uxa models status foveacast-v0.2.0 --provider directml
foveacast-v0.2.0: unsupported provider
diagnostic: available providers: AzureExecutionProvider, CPUExecutionProvider
```

No AMD adapter, device ID, DirectML model load, parity, stability, latency, or
RSS result is available. DirectML is not claimed on this evidence.

### Persisted evidence boundary

Persisted saliency artifacts record actual execution provider per duration,
model/version/checksum, input/output and geometry metadata, preprocessing
version, per-duration inference timing, cache state, warnings, profile and
aggregate evidence, and checksum-covered artifact paths. The adapter's
requested provider preference, adapter/device ID, session options, and cold-load
timing are not currently persisted. They remain unavailable review fields; no
ADR claim fills them in.

### Operational measurements

Task 13 fake sampler records synthetic operational values to validate data flow.
It does not establish an acceptable budget. This review has no real latency or
RSS measurement and no externally configured latency/RSS budget. Operational
acceptability is unassessed, not passed.

### Paired results and fallback

No real heuristic/Foveacast paired results were produced. The fake failure path
proved that a learned failure returns explicit heuristic fallback and marks the
learned comparison sample invalid. The real focused fallback count is not
available because the real matrix did not run. A fallback-substituted cell
cannot promote Foveacast.

### Redaction and replay limits

Fake report tests confirmed no fixture secret, selector, execution reference, or
numeric saliency score entered the report or cognitive payload. Production
reports keep heatmap-only replay when source overlays are redacted. Numeric
saliency remains model-estimate evidence and never enters the cognitive model.
No source overlay policy was weakened for this review.

## Hybrid decision

Hybrid is not justified. No real paired errors were reviewed, so heuristic and
Foveacast complementarity is unproven. Do not enable the dormant normalized
probability blend or expand comparison to hybrid cells.

## Rejected alternatives and deferred scope

SUM remains outside this phase as a plan-scoped non-goal/deferred boundary, not
as a quality or platform claim. Full UEyes metric reproduction, human
calibration, full learned-provider matrix, and frozen expectations remain
deferred. Synthetic acceptance is never treated as human truth.

## Carry-forward findings and impact

Six inherited Task 12 CLI tests still fail and remain visible:

```text
rtk uv run pytest tests/integration/cli/test_commands.py -q
6 failed, 28 passed
```

The final requested non-live, non-DirectML suite preserved the same six
failures:

```text
rtk uv run pytest -m "not live and not directml" -q
6 failed, 628 passed, 4 deselected in 136.62s (0:02:16)
```

Failures:

- `test_resume_matrix_skips_only_valid_finalized_runs`
- `test_resume_default_config_reuses_legacy_manifest_without_model_trial`
- `test_resume_completion_evaluates_all_selected_finalized_bundles[0]`
- `test_resume_completion_evaluates_all_selected_finalized_bundles[1]`
- `test_resume_completion_keeps_model_trials_separate_for_variant_comparisons`
- `test_resume_report_groups_variant_comparisons_by_prominence_provider`

These failures affect confidence in resume, direct/legacy bundle compatibility,
completion aggregation, provider identity consistency, and provider comparison
reporting. They are not fixed or hidden by this ADR. Existing fallback validity
checks, report redaction/trust checks, and resource/format validation remain
required; they do not substitute for missing real gate evidence.

Final repository checks also recorded a pre-existing format gap:

```text
rtk uv run ruff format --check .
nonzero exit; unformatted files reported

rtk uv run ruff check .
All checks passed!

rtk git diff --check
exit code: 0
```

This docs-only task did not normalize unrelated Python files. The format gap is
not evidence for or against Foveacast quality, but remains a release concern.

## Decision and default

| Item | Decision |
| --- | --- |
| Default provider | `heuristic-prominence-v1` |
| Learned provider | `foveacast`, explicit opt-in only |
| Hybrid | Not justified; disabled |
| CPU | Required execution path; runtime installed, real adapter gate blocked before valid inference |
| DirectML | Optional Windows path; no hardware evidence |
| Fallback | Operational failures continue with heuristic and invalidate learned comparison |
| ADR status | Proposed; not Accepted |

This is a conservative conditional decision. It does not promote Foveacast and
does not alter provider defaults.

## Review trigger for acceptance or change

Reopen this ADR when all of these are available and reviewed:

1. `saliency-cpu` is installed and `uxa models status foveacast-v0.2.0 --provider cpu` reports ready for all six pinned artifacts.
2. CPU known-screenshot evidence loads and runs all three models, records actual provider, model checksums, sequential warm latency, peak RSS, finite output, alignment, and repeatability, and resolves the current symbolic-input-shape adapter failure. Review must separately resolve requested provider preference, adapter/device ID, session options, and cold-load timing, which are not currently persisted, or explicitly accept those gaps before changing this ADR.
3. The exact eight-cell focused experiment runs with explicit Foveacast cells, `--resume`, complete LLM settings, no fallback substitution, and persisted report/evaluation output.
4. All eight comparison cells are valid; all four Foveacast cells have learned output; completion does not regress; one preregistered prominence defect improves; no material unexplained paired regression exists.
5. A real latency/RSS budget is supplied and all operational measurements pass it. Synthetic measurements cannot satisfy this item.
6. The six inherited CLI failures and their direct/legacy bundle, provider identity, resume, completion, and comparison impacts are resolved or explicitly accepted by a separate review.
7. DirectML AMD parity/stability is reviewed separately when the optional runtime and hardware exist; unavailable hardware remains explicitly unavailable.

Only after that review may status become Accepted and a later ADR revision
consider provisional default promotion. Human calibration and real-user claims
still require separate evidence.
