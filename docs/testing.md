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

Outputs are simulated benchmark evidence. They do not support claims about
real-user completion, satisfaction, or human behavior.
