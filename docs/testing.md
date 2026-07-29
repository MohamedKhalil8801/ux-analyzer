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

Task 16 acceptance calls bundled `fixture_app` through its ASGI contract and
uses recorded extracted snapshots for deterministic run orchestration. It uses
one fixed CI seed per matrix cell and recorded structured scent responses from
`tests/recordings/ci-model-responses.json`. Run bundles, timelines, checksums,
independent verification, provider manifests, and evidence classes are
checked. Browser extraction and fixture-only network safety remain covered by
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
