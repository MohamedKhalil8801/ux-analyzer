# Prominence Provider Benchmark

## Result

The 24-case CPU run recommends **keep both providers as separate evidence channels**. It does not support replacing the heuristic with FoveaCast, replacing FoveaCast with the heuristic, or averaging their scores into a single hybrid rank.

Use the heuristic for the agent's deterministic prominence ordering and fallback path. Run FoveaCast once per settled screenshot when the model is available, retain its timed saliency profiles and provenance, and give the later LLM both views. Do not blend the distributions by default.

The decision is provisional. The corpus uses controlled screenshots and human-authored relevance labels, not eye tracking or real-user task studies.

## Evidence Freshness

The repository contains older portfolio bundles under `reports/portfolio-*`.
Those bundles are excluded from the numeric comparison. They record config
digests and provider IDs, but they do not record the repository SHA, working
tree state, benchmark-source fingerprint, or model artifact checksums. The
latest historical `final6` bundle was written at 20:24 on August 8, seven
minutes before commit `e2f3d77` (`feat: harden live benchmark execution`). That
commit changed live extraction, orchestration, provider selection, and report
integrity, so the old metrics cannot be assumed to describe the current code.

The August 1 POC document is also explicitly dry-run evidence and contains no
live endpoint result. The current decision uses only the fresh 24-case run
below. Its output records the commit, dirty-worktree flag, corpus SHA-256,
benchmark-source SHA-256, and all six pinned FoveaCast artifact checksums in
`reports/prominence-comprehensive/summary.json`.

The current run used commit `e2f3d77df5b4c47eaab3dcd32e11334f4e3f225a` with a
dirty working tree because the benchmark and provenance changes are
uncommitted. Rerun the benchmark after committing those changes before using
the result as a release baseline.

## What The Heuristic Does

`HeuristicProminenceProvider.score()` computes ten raw features for every captured element:

| Feature | Raw value |
| --- | --- |
| `area` | `bounds.width * bounds.height` |
| `center_distance` | Euclidean distance from element center to viewport center |
| `reading_order` | Element index in the snapshot |
| `typography` | Role strength (`button .8`, `text .75`, `link/tab .7`, `input .65`, `checkbox .6`, `menu .55`, `other .4`) plus up to `.2` for label length |
| `contrast` | `local_contrast`, or `.5` when unavailable |
| `isolation` | `1 / (1 + near_count)` from `near` graph edges |
| `actionability` | `1` when actionable and enabled, otherwise `0` |
| `motion` | Always `0` in the current implementation |
| `competition` | `near_count` |
| `occlusion` | Explicit occlusion, or `1 - visibility_fraction` |

For each feature, the provider min-max normalizes values within the current snapshot. `center_distance` and `reading_order` reverse the normalized value because smaller values should rank higher. A constant feature gets `.5` for every element.

With the default `heuristic-prominence-v1` weights, each element's raw score is:

```text
score(e) = .16 area + .14 center_distance + .10 reading_order
         + .12 typography + .12 contrast + .10 isolation
         + .14 actionability + .05 motion - .08 competition - .09 occlusion
```

The terms above mean weighted normalized features, not the raw measurements. The provider then applies temperature-1 softmax:

```text
p(e) = exp(score(e) - max_score) / sum(exp(score(i) - max_score))
```

It stores the raw values, normalized values, weighted feature contributions, raw score, and probability. The current provider copies that probability into both `first_notice_probability` and `notice_within_budget_probability`; it does not model time or gaze directly.

## What FoveaCast Does

FoveaCast receives the screenshot rather than DOM semantics. The adapter resizes and pads the image into the pinned `320 x 240` model input, runs three ONNX models for `1s`, `3s`, and `7s`, and records geometry, checksums, preprocessing, execution provider, and timing.

For each duration and element, `aggregate_saliency()` samples the element rectangle from the saliency plane and computes:

```text
density      = mean(pixel values in the element region)
robust_peak  = 95th percentile of that region
mass_share   = element region mass / visible screenshot mass
raw          = .60 density + .25 robust_peak + .15 sqrt(mass_share)
adjusted     = raw * visibility_fraction
```

The provider softmaxes adjusted candidate scores per duration. It suppresses non-actionable structural containers that contain meaningful semantic children. The staged selector uses `1s` for initial search, `3s` for exploration, and `.25 * 3s + .75 * 7s` for persistent attention. Each result retains duration-specific saliency aggregates and prediction provenance.

That gives FoveaCast visual and temporal evidence. It does not know an element's role, label, actionability, reading order, or task goal unless those affect the pixels.

## Benchmark Matrix

The corpus contains 12 calibration and 12 holdout cases. Tags cover single targets, competing calls to action, dense navigation, forms, headings, low contrast, occlusion, center bias, decorative containers, responsive layouts, semantic-versus-visual conflict, and reading order. The runner excludes structural `other` containers from the common ranking candidate set because FoveaCast intentionally suppresses them.

Fusion searched FoveaCast weights from `0.00` to `1.00` in `.05` steps on calibration only. The selected weight was `0.00`, so the hybrid matched the heuristic on holdout rather than improving it.

| Split | Provider | Top-1 | Hit@3 | MRR | NDCG@3 | Pairwise | Mean scoring ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Calibration | Heuristic | 0.083 | 0.667 | 0.444 | 0.575 | 0.350 | 0.174 |
| Calibration | FoveaCast | 0.083 | 0.333 | 0.361 | 0.465 | 0.217 | 863.622 |
| Calibration | Hybrid (`w=0.00`) | 0.083 | 0.667 | 0.444 | 0.575 | 0.350 | 863.796 |
| Holdout | Heuristic | 0.083 | 0.750 | 0.451 | 0.593 | 0.367 | 0.164 |
| Holdout | FoveaCast | 0.167 | 0.500 | 0.403 | 0.520 | 0.267 | 891.885 |
| Holdout | Hybrid (`w=0.00`) | 0.083 | 0.750 | 0.451 | 0.593 | 0.367 | 892.049 |

FoveaCast won holdout top-1, but the heuristic won hit@3, MRR, NDCG@3, and pairwise ordering. Both providers missed most top targets. In this run FoveaCast cost about 892 ms per case versus 0.16 ms for the heuristic, roughly 5,600 times slower on this Windows CPU run. Timing is operational context, not a ranking metric, and will vary with model-cache state and machine load.

Case-level patterns explain the split. FoveaCast ranked the heading and semantic search input first on both layout variants. The heuristic ranked primary actions, forms, and navigation more reliably. Both providers struggled with the low-contrast and occluded targets. A separate opt-in repository acceptance run produced valid finite maps and placed each 1s, 3s, and 7s hotspot inside the authored target bounds. The earlier 3/3 failure measured distance from the center of a large flat target rather than alignment with the target itself; the corrected bounds check passes with a 1 px tolerance.

## Decision

Keep both, with distinct roles:

- **Heuristic in the control path:** deterministic ordering, explicit feature contributions, negligible latency, and fail-closed fallback.
- **FoveaCast in the evidence path:** visual heatmaps, three time horizons, and model provenance for the report-writing LLM when inference succeeds.
- **No score blending yet:** the calibrated linear fusion selected the heuristic endpoint and added FoveaCast cost without improving holdout metrics.
- **Benchmark both during provider changes:** keep this 24-case corpus as a regression gate and add real-user or eye-tracking labels before changing the primary provider.

Command used:

```text
rtk uv run python scripts/benchmark_prominence.py --cases benchmarks/prominence/cases.json --output reports/prominence-comprehensive
```

Machine-readable evidence is in `reports/prominence-comprehensive/summary.json`; the generated decision report is `reports/prominence-comprehensive/decision.md`.
