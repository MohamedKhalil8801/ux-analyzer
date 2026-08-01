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

Live endpoint checks stay skipped unless `UXA_RUN_LIVE_TESTS=1` and all four
variables exist:

```text
UXA_LLM_BASE_URL=https://provider.example/v1
UXA_LLM_API_KEY=<secret>
UXA_SCENT_MODEL=<scent-model>
UXA_COGNITIVE_MODEL=<cognitive-model>
uv run pytest -m live -q
```

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

The report separates simulated user effort from analysis cost. Estimated task
time is a deterministic proxy derived from observations and actions, not model
latency or a measured human completion time. Model calls, latency, attempts,
and tokens describe benchmark operating cost separately. Monetary cost remains
unavailable until explicit provider pricing is configured.

Outputs are simulated benchmark evidence. They do not support claims about
real-user completion, satisfaction, or human behavior.
