# Foveacast Saliency Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task.

**Goal:** Add versioned Foveacast 1-second, 3-second, and 7-second saliency predictions to existing UX Analyzer web benchmark, convert maps into inspectable element-level attention profiles, and conditionally promote learned prominence after focused evidence.

**Architecture:** Preserve current modular monolith and deterministic interface snapshots. Add separate model-dependent saliency domain records, a formal screenshot-to-saliency port, an in-process ONNX Runtime adapter, content-addressed experiment cache, pixel-to-element aggregation, and a search-stage adapter that still emits existing `ProminenceResult` input consumed by attention policy. Existing LLM, action, verifier, memory, and browser behavior remain unchanged except for receiving selected operational prominence through current contract.

**Tech Stack:** Python 3.12, NumPy, Pillow, platformdirs, ONNX Runtime CPU, optional ONNX Runtime DirectML on Windows, existing Typer/Playwright/Pydantic/pytest/Ruff/Pyright stack.

## Scope

- Close current POC validation gap before saliency promotion work.
- Separate local attention seed from external model trial identity.
- Add Foveacast v0.2.0 FP16 model registration, installation, verification, and removal.
- Support 1s immediate prominence, 3s early discoverability, and 7s eventual noticeability.
- Run all three duration models sequentially for each exact new screenshot and cache results inside experiment output.
- Aggregate native maps into per-element model evidence without modifying deterministic `ElementSnapshot`.
- Select operational prominence by search stage and adapt it into existing attention-policy input.
- Support CPU everywhere and optional DirectML acceleration on Windows/AMD hardware.
- Compare heuristic and Foveacast on eight focused UX cells. Evaluate simple hybrid only if base providers show complementary failures.
- Conditionally record provisional default-provider decision in ADR.

## Non-Goals

- Frozen expectation provider.
- SUM, UMSI++, UniAR, ScanDiff, ST-DiffEye, TPP-Gaze, or another challenger model.
- New model training, ONNX conversion, quantization, or UEyes dataset reproduction.
- Full human-saliency metric benchmark.
- Desktop/mobile observation providers.
- Screenshot-only UI element detection.
- Full learned-provider core matrix in this phase.
- Changes to cognitive prompts or exposure of numeric saliency values to LLM.
- Global cross-project screenshot cache.
- Weakening existing screenshot or secret-redaction policy.

## Approved Provider Artifacts

Pin GitHub release `khawkins98/foveacast-training` tag `v0.2.0`.

| Duration | Artifact | SHA-256 |
|---|---|---|
| 1s | `foveacast-v3-1s-fp16.onnx` | `4b9fdc2734e36c612a120ab7b0050ae276723160ccc625c6f554e906dd6345d5` |
| 3s | `foveacast-v3-3s-fp16.onnx` | `842a23f97908d146b8749e05f6b220bdb495eae76c75cc7252550825585ef76e` |
| 7s | `foveacast-v3-7s-fp16.onnx` | `cf66388dc6fe5db4712c77cf3929d04380ed73ac963677b8c9de2b6380fb51e0` |

Also download and checksum release parity JSON files:

| Duration | Artifact | SHA-256 |
|---|---|---|
| 1s | `foveacast-v3-1s-fp16.parity.json` | `2572f301f49bf49ec38bcff57fa1282380726e79f1ce4c9a5cd56f9f4894944b` |
| 3s | `foveacast-v3-3s-fp16.parity.json` | `3ccbd9da648baaf7c0ddca9fce1e56d92ce71f94de2f2751b2dd2af41c276a20` |
| 7s | `foveacast-v3-7s-fp16.parity.json` | `16cb94c448a6dc510e7d44d31147634750bb3d1801cfa55daf0bcf83e1d97cca` |

License chain recorded in model manifest:

- Foveacast training code: MIT.
- MSI-Net architecture: MIT.
- UEyes dataset-derived weights: CC BY 4.0 attribution to Jiang et al. 2023.
- ONNX Runtime: MIT.

## Domain Model

```python
class AttentionDuration(StrEnum):
    ONE_SECOND = "1s"
    THREE_SECONDS = "3s"
    SEVEN_SECONDS = "7s"
    GENERAL = "general"


class AttentionEstimateKind(StrEnum):
    PREDICTED = "predicted"
    DERIVED = "derived"
    FALLBACK = "fallback"
    UNAVAILABLE = "unavailable"


class SearchStage(StrEnum):
    INITIAL = "initial"
    EXPLORATION = "exploration"
    PERSISTENT = "persistent"


@dataclass(frozen=True, slots=True)
class ElementAttentionProfile:
    viewport_id: str
    element_id: str
    immediate: AttentionEstimate | None
    early: AttentionEstimate | None
    eventual: AttentionEstimate | None
    general: AttentionEstimate | None
    aggregates: tuple[ElementSaliencyAggregate, ...]
```

Evidence boundaries:

- `ViewportSnapshot` and `ElementSnapshot` remain deterministic interface facts.
- `SaliencyPrediction`, `ElementSaliencyAggregate`, and `ElementAttentionProfile` are model-estimate evidence.
- `OperationalProminence` is derived model-estimate evidence selected for one search stage.
- `ProminenceResult` remains compatibility input for current attention policy; payload gains provider/stage provenance outside LLM-visible schema.
- Cognitive agent still receives observation order and persona-visible element content only.

## Search-Stage Defaults

```yaml
stage_mixtures:
  initial:
    immediate: 1.0
  exploration:
    early: 1.0
  persistent:
    early: 0.25
    eventual: 0.75
```

- New materially different viewport starts at `initial`.
- After first observation on viewport, stage becomes `exploration`.
- Stage becomes `persistent` when existing no-progress recovery level reaches configured threshold.
- Stage changes never trigger new inference because all three maps are eagerly cached.
- Duration labels are attention stages, not simulated wall-clock timers.

## Initial Element Aggregation Formula

For each duration map and visible element:

```text
density = mean saliency inside viewport-clipped bounds
robust_peak = 95th percentile saliency inside clipped bounds
mass_share = element saliency sum / visible viewport saliency sum
visibility_adjustment = visibility_fraction * (1 - occlusion_fraction)

raw_element_attention =
    0.60 * density
  + 0.25 * robust_peak
  + 0.15 * sqrt(mass_share)

adjusted_element_attention =
  raw_element_attention * visibility_adjustment
```

Rules:

- Coefficients are versioned configuration, not human-calibrated science.
- Preserve density, p95, raw mass, mass share, clipped area, visibility, occlusion, raw score, and adjusted score.
- Aggregate every element for evidence.
- Runtime candidate adapter suppresses structural containers when they contain meaningful scored children.
- Normalize adjusted operational candidates into probabilities using existing configured temperature.
- Hybrid, if justified later, blends normalized learned and heuristic probabilities through versioned `learned_weight`; default candidate value is `0.70`.

## File Map

```text
pyproject.toml
src/ux_analyzer/domain/saliency.py
src/ux_analyzer/ports/saliency.py
src/ux_analyzer/adapters/saliency/foveacast.py
src/ux_analyzer/adapters/saliency/__init__.py
src/ux_analyzer/saliency/model_registry.py
src/ux_analyzer/saliency/manifests/foveacast-v0.2.0.yaml
src/ux_analyzer/providers/saliency_aggregation.py
src/ux_analyzer/providers/saliency_prominence.py
src/ux_analyzer/storage/saliency_cache.py
src/ux_analyzer/application/saliency.py
src/ux_analyzer/application/run_agent.py
src/ux_analyzer/application/experiment.py
src/ux_analyzer/application/evaluation.py
src/ux_analyzer/config/models.py
src/ux_analyzer/config/loader.py
src/ux_analyzer/domain/benchmark.py
src/ux_analyzer/domain/run.py
src/ux_analyzer/ports/artifacts.py
src/ux_analyzer/reporting/renderer.py
src/ux_analyzer/cli.py
benchmarks/demo/project.yaml
benchmarks/demo/experiments/baseline-model-trials.yaml
benchmarks/demo/experiments/saliency-focused-validation.yaml
docs/adr/0001-provisional-saliency-provider.md
```

## Task 1: Add External Model-Trial Identity

**Files:**
- Modify: `src/ux_analyzer/config/models.py`
- Modify: `src/ux_analyzer/config/loader.py`
- Modify: `src/ux_analyzer/domain/benchmark.py`
- Modify: `src/ux_analyzer/domain/run.py`
- Modify: `src/ux_analyzer/application/experiment.py`
- Modify: `src/ux_analyzer/ports/artifacts.py`
- Modify: `src/ux_analyzer/application/evaluation.py`
- Modify: `src/ux_analyzer/reporting/renderer.py`
- Test: `tests/unit/config/test_loader.py`
- Test: `tests/unit/application/test_experiment.py`
- Test: `tests/unit/application/test_evaluation.py`
- Test: `tests/integration/storage/test_run_bundle.py`

**Interfaces:**
- Add `ExperimentDefinition.model_trials: tuple[int, ...] = (0,)`.
- Add `RunSpec.model_trial: int = 0`.
- Include model trial in run ID, manifest, report grouping, and reproducibility labels.
- Keep attention `seed` separate and unchanged.

- [ ] **Step 1: Write failing compatibility and expansion tests**

```python
def test_missing_model_trials_defaults_to_zero() -> None:
    loaded = load_project(PROJECT_WITHOUT_MODEL_TRIALS)
    assert loaded.project.experiments[0].model_trials == (0,)


def test_model_trials_expand_independently_from_attention_seeds() -> None:
    specs = expand_experiment(context(model_trials=(0, 1, 2), seeds=(7,)))
    assert {(spec.seed, spec.model_trial) for spec in specs} == {
        (7, 0),
        (7, 1),
        (7, 2),
    }
```

- [ ] **Step 2: Run focused tests and verify failures**

Run: `rtk uv run pytest tests/unit/config/test_loader.py tests/unit/application/test_experiment.py tests/unit/application/test_evaluation.py tests/integration/storage/test_run_bundle.py -q`

- [ ] **Step 3: Implement additive schema and stable identity**

Update deterministic run ID payload with `model_trial`. Keep old YAML valid by defaulting to one trial. Update evaluation wording so model-dependent results are identified by both attention seed and model trial, never seed alone.

- [ ] **Step 4: Verify compatibility**

Run: `rtk uv run pytest tests/unit/config/test_loader.py tests/unit/application/test_experiment.py tests/unit/application/test_evaluation.py tests/integration/storage/test_run_bundle.py -q`

Run: `rtk uv run pyright src/ux_analyzer/config src/ux_analyzer/domain src/ux_analyzer/application/experiment.py`

- [ ] **Step 5: Commit model-trial identity**

```bash
rtk git add src/ux_analyzer tests/unit tests/integration/storage
rtk git commit -m "feat: separate model trial identity"
```

## Task 2: Close Current POC Baseline Evidence

**Files:**
- Create: `benchmarks/demo/experiments/baseline-model-trials.yaml`
- Modify: `benchmarks/demo/project.yaml`
- Create after execution: `docs/validation/2026-08-poc-baseline.md`

**Interfaces:**
- Full `core-pair` and required ablation matrix use `model_trials: [0]`.
- Focused replication experiment uses fixed attention seed and `model_trials: [0, 1, 2, 3, 4]`.

- [ ] **Step 1: Add and validate baseline experiment definitions**

Focused replication must include current four invite cells and preserve exact scenario/version/persona/policy semantics. It changes model trial only.

- [ ] **Step 2: Dry-run matrices**

Run: `rtk uv run uxa validate benchmarks/demo/project.yaml`

Run: `rtk uv run uxa run benchmarks/demo/project.yaml --experiment core-pair --dry-run`

Run: `rtk uv run uxa run benchmarks/demo/project.yaml --experiment baseline-model-trials --dry-run`

Expected: full current matrix uses model trial 0; focused replication prints five model trials per semantic cell.

- [ ] **Step 3: Execute current baseline before saliency default changes**

Run full matrix with resume enabled and configured OpenAI-compatible endpoint. Then run focused model-trial replication into separate output root. Preserve generated summaries and reports unchanged.

- [ ] **Step 4: Write baseline validation note**

Record command, git SHA, config digest, selected model IDs, attention seeds, model trials, run counts, failures, completion, cost, variation across model trials, and artifact roots. This note is factual baseline evidence, not saliency comparison.

- [ ] **Step 5: Commit definitions and validation note**

```bash
rtk git add benchmarks/demo docs/validation/2026-08-poc-baseline.md
rtk git commit -m "test: record post-reliability baseline"
```

## Task 3: Define Saliency Domain and Port Contracts

**Files:**
- Create: `src/ux_analyzer/domain/saliency.py`
- Create: `src/ux_analyzer/ports/saliency.py`
- Test: `tests/unit/domain/test_saliency.py`

**Interfaces:**
- `SaliencyProvider.predict(request) -> SaliencyPredictionSet`.
- `SaliencyPlane` carries normalized float32 bytes plus dimensions without importing NumPy into domain.
- `ElementAttentionProfile` owns optional duration/general estimates and aggregate provenance.

- [ ] **Step 1: Write failing invariants**

Test finite values, normalized ranges, exact plane byte length, duplicate durations, viewport mismatch, duplicate element profiles, predicted/derived/fallback source requirements, and unavailable estimates carrying no score.

```python
def test_saliency_plane_requires_float32_shape_size() -> None:
    with pytest.raises(ValueError, match="plane byte length"):
        SaliencyPlane(width=2, height=2, values=b"short")
```

- [ ] **Step 2: Run tests and verify failures**

Run: `rtk uv run pytest tests/unit/domain/test_saliency.py -q`

- [ ] **Step 3: Implement closed types**

Define request metadata for screenshot hash, screenshot dimensions, DPR, zoom, requested durations, model set, precision, and execution-provider preference. Define prediction metadata for provider/model/version/checksum/input/output dimensions/preprocessing/inference duration/execution provider/warnings/cache state.

- [ ] **Step 4: Verify domain tests and static types**

Run: `rtk uv run pytest tests/unit/domain/test_saliency.py -q`

Run: `rtk uv run pyright src/ux_analyzer/domain/saliency.py src/ux_analyzer/ports/saliency.py`

- [ ] **Step 5: Commit contracts**

```bash
rtk git add src/ux_analyzer/domain/saliency.py src/ux_analyzer/ports/saliency.py tests/unit/domain/test_saliency.py
rtk git commit -m "feat: define saliency provider contracts"
```

## Task 4: Add Versioned Model Registry and CLI Management

**Files:**
- Modify: `pyproject.toml`
- Create: `src/ux_analyzer/saliency/model_registry.py`
- Create: `src/ux_analyzer/saliency/manifests/foveacast-v0.2.0.yaml`
- Modify: `src/ux_analyzer/cli.py`
- Test: `tests/unit/saliency/test_model_registry.py`
- Test: `tests/integration/cli/test_commands.py`

**Interfaces:**
- Add core dependencies `numpy`, `pillow`, and `platformdirs`.
- Add mutually exclusive extras `saliency-cpu` and `saliency-directml`.
- Add commands `uxa models install`, `uxa models status`, and `uxa models remove`.
- Honor `UXA_MODEL_HOME`; otherwise use platformdirs user data directory.

- [ ] **Step 1: Write registry and CLI tests**

Use local HTTP fixture, never GitHub, in tests. Verify exact URLs/checksums, partial-download cleanup, atomic install, checksum rejection, parity JSON retention, attribution output, status diagnostics, idempotent install, and explicit removal.

- [ ] **Step 2: Run tests and verify failures**

Run: `rtk uv run pytest tests/unit/saliency/test_model_registry.py tests/integration/cli/test_commands.py -q`

- [ ] **Step 3: Implement pinned manifest and model store**

Install into content-addressed path such as:

```text
<model-home>/foveacast/v0.2.0/fp16/<sha256>/<artifact>
```

Runs never auto-download. `models status` must distinguish missing runtime package, missing artifact, checksum mismatch, unsupported execution provider, and ready state.

- [ ] **Step 4: Verify commands**

Run: `rtk uv run pytest tests/unit/saliency/test_model_registry.py tests/integration/cli/test_commands.py -q`

Run: `rtk uv run uxa models status foveacast-v0.2.0`

- [ ] **Step 5: Commit registry**

```bash
rtk git add pyproject.toml src/ux_analyzer/saliency src/ux_analyzer/cli.py tests/unit/saliency tests/integration/cli
rtk git commit -m "feat: manage saliency model artifacts"
```

## Task 5: Implement Foveacast CPU Inference and Geometry

**Files:**
- Create: `src/ux_analyzer/adapters/saliency/__init__.py`
- Create: `src/ux_analyzer/adapters/saliency/foveacast.py`
- Test: `tests/unit/adapters/saliency/test_foveacast_preprocess.py`
- Test: `tests/integration/saliency/test_foveacast_cpu.py`
- Add small fixtures: `tests/fixtures/saliency/`

**Interfaces:**
- `FoveacastSaliencyProvider` reuses one session per duration.
- Input contract: RGB, 0-255 float32, NCHW, aspect-preserving fit to 240x320, constant pad value 126.
- Output contract: native 240x320 normalized float map and exact inverse geometry metadata.

- [ ] **Step 1: Write preprocessing golden tests**

Cover landscape, portrait, square, 1x1, odd padding, RGBA input, and invalid image. Assert pixel-center bilinear resize, channel order, value range, NCHW layout, scale, pad offsets, and no ImageNet normalization.

- [ ] **Step 2: Write postprocessing geometry tests**

Use synthetic delta and stripe maps. Crop padded output to content region before mapping into original screenshot geometry. Assert hotspot returns to original source coordinate within one screenshot pixel after round trip.

- [ ] **Step 3: Implement CPU adapter and session reuse**

Load all three sessions once, validate input/output names and fixed shapes, execute sequentially, reject NaN/Inf/constant malformed output, and record cold-load plus per-duration warm latency. Normalize only according to released model/postprocessing contract.

- [ ] **Step 4: Verify CPU inference**

Run: `rtk uv sync --extra saliency-cpu`

Run: `rtk uv run pytest tests/unit/adapters/saliency/test_foveacast_preprocess.py tests/integration/saliency/test_foveacast_cpu.py -q`

- [ ] **Step 5: Commit CPU adapter**

```bash
rtk git add src/ux_analyzer/adapters/saliency tests/unit/adapters/saliency tests/integration/saliency tests/fixtures/saliency
rtk git commit -m "feat: run foveacast saliency inference"
```

## Task 6: Add Optional DirectML Execution

**Files:**
- Modify: `src/ux_analyzer/adapters/saliency/foveacast.py`
- Test: `tests/integration/saliency/test_foveacast_directml.py`
- Modify: `docs/model-provider.md`

**Interfaces:**
- Execution preference: `auto | cpu | directml`.
- `auto` chooses DirectML when installed and available on Windows, otherwise CPU.
- Manifest records requested provider, actual provider, fallback reason, adapter/device ID, and session options.

- [ ] **Step 1: Write provider-selection tests with fake ORT module**

Verify CPU, DirectML, auto preference, unavailable DirectML fallback, explicit DirectML failure, and actual provider recording.

- [ ] **Step 2: Configure DirectML sessions correctly**

Set sequential execution and disable memory-pattern optimization. Do not call same session concurrently. Use three sessions sequentially, one per duration.

- [ ] **Step 3: Add opt-in hardware test**

Mark test `directml`. On Windows AMD machine, compare CPU and DirectML maps for all durations using max absolute difference at most `0.002` and rank correlation at least `0.999`. Verify repeated DirectML output stability, cache equivalence, and CPU fallback.

- [ ] **Step 4: Run CPU and local DirectML verification**

Run CPU suite in CPU-extra environment. Run DirectML test in separate DirectML-extra environment because `onnxruntime` packages expose same import.

- [ ] **Step 5: Commit DirectML support**

```bash
rtk git add src/ux_analyzer/adapters/saliency/foveacast.py tests/integration/saliency/test_foveacast_directml.py docs/model-provider.md
rtk git commit -m "feat: support directml saliency inference"
```

## Task 7: Aggregate Saliency Maps into Element Attention Profiles

**Files:**
- Create: `src/ux_analyzer/providers/saliency_aggregation.py`
- Test: `tests/unit/providers/test_saliency_aggregation.py`

**Interfaces:**
- `aggregate_saliency(snapshot, predictions, config) -> tuple[ElementAttentionProfile, ...]`.
- Config version `element-saliency-aggregation-v1` contains formula weights and candidate rules.

- [ ] **Step 1: Write coordinate and formula tests**

Cover viewport clipping, DPR/zoom metadata, partially visible elements, occlusion fraction, offscreen rejection, nested elements, identical bounds, zero saliency, large containers, tiny peaks, and finite normalization.

```python
def test_large_container_does_not_win_from_area_alone() -> None:
    profiles = aggregate_saliency(snapshot_with_card_and_button(), predictions(), config())
    assert score(profiles, "button") > score(profiles, "background-card")
```

- [ ] **Step 2: Run tests and verify failures**

Run: `rtk uv run pytest tests/unit/providers/test_saliency_aggregation.py -q`

- [ ] **Step 3: Implement native-map sampling and evidence**

Map screenshot bounds through inverse preprocessing geometry into native map coordinates. Use clipped bounds and existing visibility/occlusion fractions; do not create pixel-perfect element masks. Preserve raw values and formula contributions for each duration.

- [ ] **Step 4: Implement semantic-leaf candidate classification**

Aggregate every element, but mark structural container as non-operational when it contains meaningful scored children and is not itself actionable or a semantic target. Keep regions and containers visible in report evidence.

- [ ] **Step 5: Verify aggregation**

Run: `rtk uv run pytest tests/unit/providers/test_saliency_aggregation.py -q`

Run: `rtk uv run pyright src/ux_analyzer/providers/saliency_aggregation.py`

- [ ] **Step 6: Commit aggregation**

```bash
rtk git add src/ux_analyzer/providers/saliency_aggregation.py tests/unit/providers/test_saliency_aggregation.py
rtk git commit -m "feat: aggregate saliency by ui element"
```

## Task 8: Add Experiment-Scoped Inference Cache and Artifacts

**Files:**
- Create: `src/ux_analyzer/storage/saliency_cache.py`
- Modify: `src/ux_analyzer/ports/artifacts.py`
- Modify: `src/ux_analyzer/storage/run_bundle.py`
- Test: `tests/integration/storage/test_saliency_cache.py`
- Test: `tests/integration/storage/test_run_bundle.py`

**Interfaces:**
- Cache key includes screenshot SHA-256, screenshot dimensions, DPR, zoom, three model checksums, preprocessing version, precision, and actual execution provider.
- Cache root lives under experiment output and is never global.
- Cache entry contains native numeric maps, metadata JSON, and checksums.

- [ ] **Step 1: Write cache tests**

Verify exact hits, pixel-change misses, geometry-change misses, model/preprocess/provider misses, atomic writes, interrupted-entry rejection, checksum validation, resume reuse, and no API keys or fixture secrets in metadata.

- [ ] **Step 2: Define artifact outputs**

Persist per viewport:

```text
saliency/<viewport-id>/1s.npz
saliency/<viewport-id>/3s.npz
saliency/<viewport-id>/7s.npz
saliency/<viewport-id>/1s-heatmap.png
saliency/<viewport-id>/3s-heatmap.png
saliency/<viewport-id>/7s-heatmap.png
saliency/<viewport-id>/profiles.json
saliency/<viewport-id>/metadata.json
```

Native `.npz` files store float32 maps and geometry. Heatmap-only PNGs contain no source pixels. Source-image overlays are written only when artifact policy permits sanitized source pixels; otherwise record `overlay-redacted` warning and rely on heatmap plus element geometry.

- [ ] **Step 3: Implement cache and bundle linking**

Cache hit must still append run event and materialize or reference checksum-covered evidence in finalized bundle. A cache hit may not bypass provider/model manifest recording.

- [ ] **Step 4: Verify cache and artifact integrity**

Run: `rtk uv run pytest tests/integration/storage/test_saliency_cache.py tests/integration/storage/test_run_bundle.py -q`

- [ ] **Step 5: Commit cache**

```bash
rtk git add src/ux_analyzer/storage src/ux_analyzer/ports/artifacts.py tests/integration/storage
rtk git commit -m "feat: cache saliency inference evidence"
```

## Task 9: Select Stage Attention and Adapt Existing Prominence Contract

**Files:**
- Create: `src/ux_analyzer/providers/saliency_prominence.py`
- Create: `src/ux_analyzer/application/saliency.py`
- Modify: `src/ux_analyzer/providers/prominence.py`
- Test: `tests/unit/providers/test_saliency_prominence.py`

**Interfaces:**
- `AttentionStageSelector.select(profiles, stage) -> tuple[ProminenceResult, ...]`.
- `FoveacastProminenceProvider.score(capture, snapshot, stage, artifacts) -> ProminenceBatch`.
- Existing `ProgressiveAttentionPolicy.next_observation` signature remains unchanged.

- [ ] **Step 1: Write stage-selection tests**

Verify initial uses immediate, exploration uses early, persistent uses 25/75 early/eventual mix, all probabilities normalize, stage changes do not infer again, missing duration uses explicit fallback provenance, and LLM payload contains no numeric scores.

- [ ] **Step 2: Write heuristic-fallback tests**

When model/runtime/cache/aggregation fails, learned provider returns explicit fallback batch from existing `HeuristicProminenceProvider`; failure reason and unavailable learned profile remain preserved.

- [ ] **Step 3: Implement provider composition**

Do not change heuristic formula. Add formal application-facing prominence protocol outside `run_agent.py`. Learned adapter composes model provider, cache, aggregator, stage selector, and artifact writer.

- [ ] **Step 4: Add dormant simple hybrid adapter**

Implement versioned blend over normalized probabilities but keep it unselected unless focused comparison review explicitly approves evaluation. Preserve learned and heuristic components separately.

- [ ] **Step 5: Verify adapters**

Run: `rtk uv run pytest tests/unit/providers/test_saliency_prominence.py tests/unit/providers/test_prominence.py tests/unit/providers/test_attention_policy.py -q`

- [ ] **Step 6: Commit stage adapter**

```bash
rtk git add src/ux_analyzer/providers src/ux_analyzer/application/saliency.py tests/unit/providers
rtk git commit -m "feat: select staged saliency prominence"
```

## Task 10: Add Prominence Provider as Experiment Axis

**Files:**
- Modify: `src/ux_analyzer/config/models.py`
- Modify: `src/ux_analyzer/config/loader.py`
- Modify: `src/ux_analyzer/domain/benchmark.py`
- Modify: `src/ux_analyzer/domain/run.py`
- Modify: `src/ux_analyzer/application/experiment.py`
- Modify: `src/ux_analyzer/application/checkpoint.py`
- Modify: `src/ux_analyzer/application/progress.py`
- Modify: `src/ux_analyzer/ports/artifacts.py`
- Modify: `src/ux_analyzer/cli.py`
- Test: `tests/unit/config/test_loader.py`
- Test: `tests/unit/application/test_experiment.py`
- Test: `tests/unit/application/test_checkpoint.py`
- Test: `tests/integration/cli/test_commands.py`

**Interfaces:**
- Add `ExperimentDefinition.prominence_provider_ids: tuple[str, ...] = ("heuristic",)`.
- Add `RunSpec.prominence_provider_id: str = "heuristic"`.
- Include provider ID in run identity, matrix print, checkpoint selected set, manifest, resume trust checks, and report cell key.

- [ ] **Step 1: Write backward-compatibility and matrix tests**

Existing YAML without provider axis expands exactly as before using heuristic. New eight-cell experiment expands `2 scenarios x 2 versions x 1 persona x 2 providers x 1 policy x 1 seed x 1 model trial`.

- [ ] **Step 2: Run tests and verify failures**

Run: `rtk uv run pytest tests/unit/config/test_loader.py tests/unit/application/test_experiment.py tests/unit/application/test_checkpoint.py tests/integration/cli/test_commands.py -q`

- [ ] **Step 3: Implement provider definitions**

Keep current `providers.prominence` block as heuristic configuration. Add `providers.saliency` for Foveacast model set, execution preference, cache, aggregation, stage selector, and fallback. Resolve IDs through provider registry; reject unknown IDs before execution.

- [ ] **Step 4: Verify dry-run output and resume identity**

Run: `rtk uv run pytest tests/unit/config/test_loader.py tests/unit/application/test_experiment.py tests/unit/application/test_checkpoint.py tests/integration/cli/test_commands.py -q`

Run: `rtk uv run uxa validate benchmarks/demo/project.yaml`

- [ ] **Step 5: Commit experiment axis**

```bash
rtk git add src/ux_analyzer benchmarks/demo tests/unit tests/integration/cli
rtk git commit -m "feat: vary prominence providers by experiment"
```

## Task 11: Integrate Saliency into RunAgent and Evidence Timeline

**Files:**
- Modify: `src/ux_analyzer/application/run_agent.py`
- Modify: `src/ux_analyzer/domain/run.py`
- Modify: `src/ux_analyzer/cli.py`
- Test: `tests/integration/application/test_run_agent.py`
- Test: `tests/e2e/test_private_data_leakage.py`

**Interfaces:**
- `_capture` returns current screenshot bytes/hash and snapshot for immediate scoring without persisting raw bytes in domain state.
- RunAgent derives search stage from viewport identity, observation history, and existing no-progress count.
- Saliency events precede operational `prominence-recorded` event.

- [ ] **Step 1: Write run-loop tests**

Cover first capture initial stage, second observation exploration stage, recovery threshold persistent stage, scroll/modal recapture reset to initial, cache hit on repeated exact screenshot, stage change without reinference, model failure fallback, and comparison evidence marked invalid when fallback occurred.

- [ ] **Step 2: Define timeline events**

Append structured events:

```text
saliency-inference-recorded
saliency-cache-hit
saliency-profiles-recorded
saliency-fallback-recorded
prominence-recorded
```

Events include viewport ID, provider/model checksums, execution provider, durations, timings, cache key, artifact IDs, search stage, selected mixture, and sanitized warning/error. Do not include raw map values inline in JSONL.

- [ ] **Step 3: Implement capture-to-prominence flow**

Call saliency provider after screenshot capture and element extraction, before coarse scent and attention selection. Keep screenshot bytes only as local application data until inference/artifact write completes.

- [ ] **Step 4: Extend leakage tests**

Assert saliency events, manifests, model requests, observations, and reports contain no selectors, test IDs, execution references, hidden values, API keys, or sensitive fixture values. Numeric saliency never enters cognitive request.

- [ ] **Step 5: Verify run integration**

Run: `rtk uv run pytest tests/integration/application/test_run_agent.py tests/e2e/test_private_data_leakage.py -q`

- [ ] **Step 6: Commit runtime integration**

```bash
rtk git add src/ux_analyzer/application/run_agent.py src/ux_analyzer/domain/run.py src/ux_analyzer/cli.py tests/integration/application/test_run_agent.py tests/e2e/test_private_data_leakage.py
rtk git commit -m "feat: record learned saliency during runs"
```

## Task 12: Extend Evaluation, Findings, and Replay

**Files:**
- Modify: `src/ux_analyzer/application/evaluation.py`
- Modify: `src/ux_analyzer/providers/finding_rules.py`
- Modify: `src/ux_analyzer/reporting/renderer.py`
- Modify: `src/ux_analyzer/reporting/static/report.css`
- Modify: `src/ux_analyzer/reporting/static/report.js`
- Modify: `docs/run-bundle-format.md`
- Test: `tests/unit/application/test_evaluation.py`
- Test: `tests/unit/providers/test_finding_rules.py`
- Test: `tests/integration/reporting/test_renderer.py`

**Interfaces:**
- Metrics and findings record prominence provider ID and active search stage.
- Target prominence uses active operational score while retaining immediate/early/eventual source evidence.
- Fallback learned runs are not scored as valid Foveacast samples.

- [ ] **Step 1: Write evaluation tests**

Verify target profiles, active-stage target score, provider grouping, fallback invalidation, cache/runtime cost separation, provider comparison cell keys, and deterministic facts versus model estimates.

- [ ] **Step 2: Update finding evidence language**

Replace hard-coded phrase `Heuristic prominence estimate for target.` with provider-aware wording. Findings cite profile event, operational prominence event, duration/stage, provider model/version, and limitations.

- [ ] **Step 3: Add focused replay views**

Add duration tabs, heatmap-only image or sanitized overlay, ranked element table, aggregation component detail, search-stage timeline, selected mixture, cache status, execution provider, model checksums, inference timing, and fallback warnings. Do not redesign report shell.

- [ ] **Step 4: Verify report escaping and redaction behavior**

If source screenshot is blanked by current artifact policy, report must state overlay unavailable due redaction and still show heatmap-only artifact plus element geometry/table. Never bypass sanitizer by file extension or custom writer path.

- [ ] **Step 5: Run tests**

Run: `rtk uv run pytest tests/unit/application/test_evaluation.py tests/unit/providers/test_finding_rules.py tests/integration/reporting/test_renderer.py -q`

- [ ] **Step 6: Commit evidence UI**

```bash
rtk git add src/ux_analyzer/application/evaluation.py src/ux_analyzer/providers/finding_rules.py src/ux_analyzer/reporting docs/run-bundle-format.md tests/unit tests/integration/reporting
rtk git commit -m "feat: report saliency evidence"
```

## Task 13: Add Focused Saliency Validation and Acceptance Suite

**Files:**
- Create: `benchmarks/demo/experiments/saliency-focused-validation.yaml`
- Modify: `benchmarks/demo/project.yaml`
- Create: `tests/e2e/test_saliency_benchmark.py`
- Modify: `tests/e2e/test_demo_benchmark.py`
- Modify: `docs/testing.md`

**Interfaces:**
- Focused matrix contains eight cells:

```text
2 scenarios
x 2 application versions
x 1 first-time nontechnical persona
x 2 prominence providers (heuristic, foveacast)
x 1 progressive-prominence-scent policy
x seed 0
x model trial 0
= 8 runs
```

- [ ] **Step 1: Add deterministic fake-saliency E2E acceptance**

Use small deterministic fake maps to verify all orchestration, artifacts, reports, stage transitions, provider axis, fallback, cache, and evaluation without downloading ONNX models.

- [ ] **Step 2: Add real-model opt-in test**

Gate with installed Foveacast model status. Run both scenarios on focused states, confirm all three finite maps, element profiles attached, target coordinates aligned, semantic container suppression, and replay artifacts valid.

- [ ] **Step 3: Define focused promotion gate**

Foveacast may pass when:

- all eight runs produce valid learned evidence without alignment/leakage failures;
- verified completion does not regress against paired heuristic cells;
- inference latency and peak memory remain operationally acceptable and are reported;
- at least one preregistered planted prominence defect shows better target discovery rank, target operational prominence, misleading-competitor margin, or resulting discovery path;
- no other paired cell shows material unexplained regression;
- fallback did not substitute for learned output in compared cells.

No full saliency matrix is required in this phase.

- [ ] **Step 4: Run automated verification**

Run: `rtk uv run pytest -m "not live and not directml" -q`

Run: `rtk uv run ruff format --check .`

Run: `rtk uv run ruff check .`

Run: `rtk uv run pyright`

- [ ] **Step 5: Commit focused validation**

```bash
rtk git add benchmarks/demo tests/e2e docs/testing.md
rtk git commit -m "test: add focused saliency validation"
```

## Task 14: Execute Focused Comparison and Record Conditional ADR

**Files:**
- Create: `docs/adr/0001-provisional-saliency-provider.md`
- Modify: `docs/model-provider.md`
- Modify: `docs/architecture.md`
- Modify: `docs/domain-model.md`
- Modify: `docs/roadmap.md`
- Modify: `docs/poc-plan-vs-current.md`

**Interfaces:**
- ADR status is `Accepted` only after focused gate review.
- ADR outcome is conditional:
  - promote `foveacast-v0.2.0-fp16` provisionally, or
  - retain `heuristic-prominence-v1` default and keep Foveacast opt-in.

- [ ] **Step 1: Install and verify model artifacts**

Run: `rtk uv run uxa models install foveacast-v0.2.0 --precision fp16`

Run: `rtk uv run uxa models status foveacast-v0.2.0`

Expected: all six model/parity artifacts match pinned checksums and attribution is printed.

- [ ] **Step 2: Run CPU and DirectML smoke evidence**

Run CPU known-screenshot suite. In separate DirectML environment, run local AMD hardware parity/stability suite. Record actual execution provider, adapter ID, model load time, sequential 1s/3s/7s warm latency, peak RSS, and output parity.

- [ ] **Step 3: Execute eight-cell focused experiment**

Run saliency-focused validation with explicit Foveacast execution provider and resume enabled. Generate evaluation summary and report. Review map alignment, top-ranked elements, target/distractor behavior, cache reuse, and fallback count.

- [ ] **Step 4: Decide whether hybrid is justified**

Only when heuristic and Foveacast make complementary errors on reviewed cells, enable configured blend and rerun same focused cells. Otherwise record hybrid as not justified and do not expand comparison.

- [ ] **Step 5: Write ADR from evidence**

Record baseline, exact artifacts/checksums, execution providers, focused matrix, paired results, latency/memory, cache/fallbacks, redaction limitations, rejected SUM rationale, hybrid decision, chosen default, and trigger for future review. Do not claim human calibration.

- [ ] **Step 6: Update architecture and roadmap status**

Document provider boundary, model-estimate evidence, stage selector, cache scope, DirectML optional path, operational fallback, and deferred full matrix/frozen expectations.

- [ ] **Step 7: Run final verification**

Run: `rtk uv run pytest -m "not live and not directml" -q`

Run: `rtk uv run ruff format --check .`

Run: `rtk uv run ruff check .`

Run: `rtk uv run pyright`

Run: `rtk git diff --check`

- [ ] **Step 8: Commit decision documentation**

```bash
rtk git add docs
rtk git commit -m "docs: record saliency provider decision"
```

## Acceptance Criteria

- Existing benchmark YAML remains valid and defaults to heuristic prominence.
- Attention seed and model trial are separate in run IDs, manifests, reports, resume, and evaluation.
- Current POC full matrix executes once after reliability changes, with focused model-trial replication recorded.
- Foveacast v0.2.0 FP16 artifacts install only through explicit command and pass pinned checksums.
- CPU inference works for all three durations. When DirectML is installed and available, local AMD hardware passes parity/stability tests; `auto` falls back explicitly only when DirectML is unavailable or initialization fails.
- `ElementSnapshot` remains deterministic and unchanged by model output.
- Every learned viewport has typed profiles for immediate, early, and eventual attention with full provenance.
- Native maps, heatmaps, element aggregates, geometry, timing, provider, cache, and warnings are checksum-covered evidence.
- Exact repeated screenshots reuse experiment-scoped cache; changed pixels/geometry/model/runtime config miss cache.
- Existing attention policy consumes stage-selected `ProminenceResult` without numeric saliency reaching cognitive model.
- Structural containers remain reportable but do not dominate operational element selection when meaningful children exist.
- Learned inference failures continue normal UX run through explicit heuristic fallback and invalidate learned comparison cell.
- Eight-cell focused comparison completes with no leakage or alignment failures.
- Conditional ADR selects provisional default from evidence; no automatic provider change.
- SUM, full UEyes benchmark, full saliency matrix, and frozen expectations remain deferred.

## Risks and Smallest Resolving Experiments

| Risk | Smallest experiment |
|---|---|
| Foveacast preprocessing geometry differs from consumer app or training pipeline | Synthetic hotspot round-trip plus known screenshot parity before runtime integration |
| FP16 ONNX unsupported by DirectML operator set | Load and infer one duration, then all three, before adding DirectML acceptance |
| DirectML numerical output shifts rankings | CPU versus DirectML map tolerance and rank-correlation test on fixed screenshots |
| Three eager models cause excessive latency or memory | Sequential cold/warm timing and RSS measurement on Ryzen 9 7950X3D and RX 7800 XT |
| Large containers dominate aggregation | Synthetic card/button fixture and semantic-leaf candidate tests |
| Screenshot changes defeat cache reuse | Pair identical fixture captures and one-pixel/config changes against cache-key tests |
| Current PNG redaction removes useful overlays | Heatmap-only artifact plus geometry/ranking report; never weaken redaction for convenience |
| Learned prominence worsens task behavior | Eight paired focused cells before default promotion |
| Heuristic and learned signals are complementary | Review paired profile/rank/path errors; run hybrid only if complementary error is concrete |
| Model artifacts change upstream | Pin tag, URLs, SHA-256, parity files, attribution, and local content-addressed model store |

## Execution Checkpoints

1. Tasks 1-2: approve reproducibility baseline before saliency code.
2. Tasks 3-6: approve model contracts, registry, CPU geometry, and DirectML parity before element scoring.
3. Tasks 7-9: approve aggregation formula, candidate suppression, cache, and stage selector before RunAgent integration.
4. Tasks 10-12: approve config compatibility, timeline evidence, evaluation, redaction, and report changes before live focused comparison.
5. Tasks 13-14: execute focused gate, decide hybrid need, and record conditional ADR. Stop phase after decision; do not expand benchmark automatically.
