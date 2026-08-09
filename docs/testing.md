# Testing

Run deterministic checks with:

```text
uv sync
uv run playwright install chromium
uv run pytest -m "not live" -q
uv run ruff format --check .
uv run ruff check .
uv run pyright
git diff --check
```

Task 16 acceptance starts local fixture and model servers, drives reduced
production `uxa run`, opens generated report through `file://` with Playwright,
blocks external report requests, and exercises element hover/focus/click. It
uses one fixed CI seed and a reduced improved 2FA matrix. Run bundles,
timelines, checksums, independent verification, provider manifests, exact
prominence contributions, causal findings, and process cards are checked.
Browser extraction and fixture-only network safety remain covered by
`tests/integration/web` and install Chromium before running those suites.

Live API endpoint checks stay skipped unless `UXA_RUN_LIVE_TESTS=1` and API-mode
variables exist:

```text
UXA_LLM_MODE=api
UXA_LLM_BASE_URL=https://<provider-host>/v1
UXA_LLM_API_KEY=<api-key>
UXA_SCENT_MODEL=<scent-model-id>
UXA_COGNITIVE_MODEL=<cognitive-model-id>
UXA_RUN_LIVE_TESTS=1
uv run pytest -m live -q
```

Credential-free Codex live coverage is opt-in and uses an installed, logged-in
Codex CLI:

```powershell
$env:UXA_LLM_MODE = "codex"
$env:UXA_SCENT_MODEL = "<scent-model-id>"
$env:UXA_COGNITIVE_MODEL = "<cognitive-model-id>"
$env:UXA_RUN_LIVE_TESTS = "1"
uv run pytest tests/live/test_openai_endpoint.py -m live -q
```

Both paths use same structured-role assertions: one coarse-scent, one full-scent,
and one cognitive call, with structured outputs, role order, endpoint origin,
and sanitized records checked. Codex mode does not read or print credentials.

Live responses are checked in memory and never recorded automatically. Fixture
inputs are scenario-owned; sensitive values are redacted from bundles and
leakage scans cover model requests plus persona-visible observations.

Use the balanced focused experiment before running the larger benchmark matrix:

```text
uv run uxa fixture serve --host 127.0.0.1 --port 8000
uv run uxa run benchmarks/demo/project.yaml --experiment focused-validation --workers 4 --fixture-origin http://127.0.0.1:8000 --output reports/focused-validation
```

`focused-validation` contains exactly four invite-teammate cells: defective and
improved versions crossed with full-list and progressive-prominence-scent for
the first-time-nontechnical persona at seed 0. The run writes
`experiment-progress.json` after each completed cell. If execution is
interrupted, rerun the same selection with `--resume`; only integrity-valid
finalized bundles are skipped:

```text
uv run uxa run benchmarks/demo/project.yaml --experiment focused-validation --workers 4 --fixture-origin http://127.0.0.1:8000 --output reports/focused-validation --resume
```

## Live portfolio benchmark

Validate and run six serial cells against public portfolio. Use fresh output
directory for each Codex comparison:

```powershell
$env:UXA_LLM_MODE = "codex"
$env:UXA_LLM_COGNITIVE_REASONING_EFFORT = "low"
$env:UXA_LLM_TIMEOUT_SECONDS = "180"
$env:UXA_LLM_MAX_CONCURRENT_CALLS = "1"
uv run uxa validate benchmarks/portfolio/project.yaml
uv run uxa run benchmarks/portfolio/project.yaml --experiment portfolio-foveacast-vs-heuristic --workers 1 --output reports/portfolio-foveacast-vs-heuristic-final3
```

This is a focused one-trial provider comparison and smoke benchmark: three live
scenarios crossed with `heuristic` and `foveacast` under `progressive-prominence`,
seed 0, and model trial 0. It has no fixture inputs, uses visible-result
verification, and disables saliency fallback. Browser viewport is 1440x900.
Navigation settles for 2400 ms; live audit showed counters settling around 2400 ms
after entering viewport, so action settling is 2800 ms. Do not treat results as a
statistical comparison, generalize beyond this target, or make real-user claims.
Each scenario sets `timeout_seconds: null`: wall-clock model latency is reported
as operating cost and must not become a UX failure. Human budgets stay bounded at
60 steps, 30 observations, 24 interactions, and 40 model calls. Attention uses
`batch_size: 3` and `cross_region_exploration: 1`; `progressive-prominence` makes
no scent calls and receives no target oracle. The Codex environment variables and
`--workers 1` are infrastructure settings, not UX behavior.

Run separate saliency-focused validation after deterministic checks:

```text
uv run uxa validate benchmarks/demo/project.yaml
uv run pytest tests/e2e/test_saliency_benchmark.py -m "not live and not directml" -q
uv run uxa run benchmarks/demo/project.yaml --experiment saliency-focused-validation --workers 4 --fixture-origin http://127.0.0.1:8000 --output reports/saliency-focused-validation --resume
```

CLI run is real production execution, not default fake acceptance. Before
using it, install optional CPU runtime and pinned model explicitly, then check
readiness without downloading during execution:

```text
uv sync --extra saliency-cpu
uv run uxa models install foveacast-v0.2.0 --precision fp16
uv run uxa models status foveacast-v0.2.0 --provider cpu
UXA_LLM_BASE_URL=https://provider.example/v1
UXA_LLM_API_KEY=<secret>
UXA_SCENT_MODEL=<scent-model>
UXA_COGNITIVE_MODEL=<cognitive-model>
```

Use `UXA_RUN_REAL_SALIENCY_TESTS=1` only for opt-in real-model pytest
acceptance. Default fake pytest path needs no ONNX runtime, model artifact,
LLM endpoint, credential, or network. CLI focused comparison needs ready model,
complete LLM settings, `--resume`, and no fallback substitution; any
Foveacast fallback disqualifies promotion.

`saliency-focused-validation` does not redefine legacy `focused-validation`. Its
exact eight-cell matrix is:

```text
invite-teammate, enable-2fa
x fixture-app-defective, fixture-app-improved
x first-time-nontechnical
x heuristic, foveacast
x progressive-prominence-scent
x seed 0
x model trial 0
= 8 cells
```

Default acceptance path uses small deterministic fake saliency maps. It needs no
ONNX model, model download, external endpoint, credentials, or network. It checks
three duration maps, element profiles, cache miss/hit reuse, stage transitions
without reinference, artifacts, replay/report rendering, provider identity,
fallback invalidation, cognitive-prompt redaction, and paired gate inputs.
It drives all eight cells through `ExperimentRunner`, persisted run bundles,
evaluation, and report rendering. Its injected sampler writes typed operational
evidence with `measured` status, latency, peak RSS bytes, scope, sampler,
platform, model checksums, and explicit `synthetic: true`; values validate
orchestration only, not promotion. Promotion requires persisted OS high-water
measurements; unavailable or synthetic operational evidence fails gate.

Real-model acceptance is opt-in only. It is also marked `live`, requires
`UXA_RUN_REAL_SALIENCY_TESTS=1`, and checks `uxa models status` through the local
registry before running. Missing runtime, missing or mismatched artifacts, and
unsupported CPU conditions skip test; test never downloads models:

```text
UXA_RUN_REAL_SALIENCY_TESTS=1 uv run pytest tests/e2e/test_saliency_benchmark.py -m live -q
```

Opt-in path uses focused fixture screenshots, all three finite duration maps,
target geometry alignment, immediate/early/eventual profiles, semantic-container
suppression, experiment-scoped cache miss/hit replay, checksum-covered native
maps and heatmaps. OS high-water RSS measurement uses Windows
`GetProcessMemoryInfo` or Unix `resource.getrusage`; it never uses
`tracemalloc`. Missing measurement is recorded as `unavailable` and remains
unassessed. Typed operational evidence is persisted before report and gate
evaluation, then loaded by both.

DirectML hardware parity remains separately marked `directml`; it is not part of
default saliency acceptance. Live LLM tests remain skipped unless their existing
environment gate is enabled.

Focused promotion requires all eight cells comparison-valid with zero alignment
or leakage failures, all four Foveacast cells learned-valid with no fallback
substitution, verified completion no worse than paired heuristic, finite reported
inference latency and peak memory, one preregistered planted prominence defect
improving rank/operational prominence/misleading-competitor margin/discovery path,
and no material unexplained paired regression. Missing operational metrics or any
fallback invalidates promotion. Operational acceptability also needs an explicit
externally configured latency/RSS budget; repository has no such budget yet, so
gate rejects promotion as `operational budget is unassessed` rather than inventing
a threshold. Full saliency matrix remains deferred.

Run identity digest scopes to selected experiment. Adding
`saliency-focused-validation` does not change legacy experiment run IDs;
changing selected experiment configuration does. Additive-config identity is
covered by `tests/unit/config/test_loader.py` before resuming old reports.

Saliency evidence stays outside cognitive prompts: numeric maps, selectors, test
IDs, execution references, hidden values, credentials, and sensitive fixture
values are not public payload. Reports retain heatmap-only replay when source
overlay is redacted and state redaction explicitly.

The report separates simulated user effort from analysis cost. Estimated task
time is a deterministic proxy derived from observations and actions, not model
latency or a measured human completion time. Model calls, latency, attempts,
and tokens describe benchmark operating cost separately. Monetary cost remains
unavailable until explicit provider pricing is configured.

Outputs are simulated benchmark evidence. They do not support claims about
real-user completion, satisfaction, or human behavior.

## Comprehensive prominence provider benchmark

Run the controlled 24-case comparison before changing prominence providers:

```text
rtk uv run python scripts/generate_prominence_corpus.py
rtk uv run python scripts/benchmark_prominence.py --cases benchmarks/prominence/cases.json --output reports/prominence-comprehensive
```

The corpus splits 12 calibration cases from 12 holdout cases. The runner compares
heuristic, FoveaCast, and a calibrated late-fusion candidate, rejects FoveaCast
fallback samples, and writes `summary.json` plus `decision.md`. See
`docs/validation/2026-08-09-prominence-provider-benchmark.md` for the formulas,
metric table, and current recommendation. The labels are controlled human-authored
synthetic evidence; they are not eye tracking or real-user measurements.
