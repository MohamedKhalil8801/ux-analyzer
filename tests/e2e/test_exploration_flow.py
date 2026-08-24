"""End-to-end exploration flow: dry-run preview, auto-accept explore, run, report."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

import ux_analyzer.cli as cli
from tests.integration.reporting.test_renderer import _write_run
from ux_analyzer.cli import app
from ux_analyzer.domain.benchmark import (
    Budget,
    ScenarioEvaluationTarget,
    VisibleResultVerifierSpec,
)
from ux_analyzer.domain.exploration import (
    CrawlCorpus,
    CrawlPage,
    ScenarioSuggestion,
    normalize_crawl_url,
)

DEMO_PROJECT = Path(__file__).parents[2] / "benchmarks" / "demo" / "project.yaml"
runner = CliRunner()

MODEL_ENV = {
    "UXA_LLM_MODE": "api",
    "UXA_LLM_BASE_URL": "https://llm.example.test/v1",
    "UXA_LLM_API_KEY": "test-key",
    "UXA_SCENT_MODEL": "test-scent-model",
    "UXA_COGNITIVE_MODEL": "test-cognitive-model",
}

_MODEL_ENV_NAMES = frozenset(MODEL_ENV) | {"UXA_REPORT_MODEL"}


def _origin(normalized_url: str) -> str:
    from urllib.parse import urlsplit

    parsed = urlsplit(normalized_url)
    host = parsed.hostname.lower()
    port = parsed.port
    if port is None or port == 443:
        return f"https://{host}"
    return f"https://{host}:{port}"


def _fake_corpus(start_urls: tuple[str, ...], max_pages: int) -> CrawlCorpus:
    pages: list[CrawlPage] = []
    for index, url in enumerate(start_urls):
        if len(pages) >= max_pages:
            break
        normalized = normalize_crawl_url(url)
        pages.append(
            CrawlPage(
                url=url,
                normalized_url=normalized,
                origin=_origin(normalized),
                depth=0,
                title=f"Exploration Page {index + 1}",
                headings=("Documentation",),
                visible_elements=("Get Started", "Docs"),
            )
        )
    return CrawlCorpus(
        pages=tuple(pages),
        link_graph={},
        started_at="2026-01-01T00:00:00Z",
    )


def _fake_suggestions(
    corpus: CrawlCorpus, max_scenarios: int, settings: Any
) -> tuple[ScenarioSuggestion, ...]:
    budget = Budget(
        max_steps=20,
        max_observations=12,
        max_interactions=8,
        timeout_seconds=120,
        max_model_calls=32,
    )
    suggestions: list[ScenarioSuggestion] = []
    for index, page in enumerate(corpus.pages[:max_scenarios]):
        label = f"Exploration Page {index + 1}"
        suggestions.append(
            ScenarioSuggestion(
                id=f"explore-scenario-{index + 1}",
                name=label,
                goal=f"Explore {page.normalized_url} until {label} is visible",
                start_url=page.normalized_url,
                verifier=VisibleResultVerifierSpec(type="visible-result", text=label),
                evaluation_target=ScenarioEvaluationTarget(
                    labels_by_version={"live": label}
                ),
                budget=budget,
                rationale="Deterministic offline synthesis",
                coverage=("discovery",),
            )
        )
    return tuple(suggestions)


@pytest.fixture
def fake_exploration(monkeypatch):
    for name in _MODEL_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name, value in MODEL_ENV.items():
        monkeypatch.setenv(name, value)

    async def fake_crawler(spec, policy, *, on_frontier=None, resume_from=None):
        return _fake_corpus(spec.start_urls, spec.max_pages)

    async def fake_synthesizer(corpus, max_scenarios, settings):
        return _fake_suggestions(corpus, max_scenarios, settings)

    monkeypatch.setattr(cli, "_explore_run_crawler", fake_crawler)
    monkeypatch.setattr(cli, "_explore_run_synthesizer", fake_synthesizer)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _explore_args(output: Path) -> list[str]:
    return [
        "explore",
        str(DEMO_PROJECT),
        "--starting-url",
        "https://a.test/",
        "--depth",
        "0",
        "--max-pages",
        "1",
        "--max-scenarios",
        "3",
        "--auto-accept",
        "--output",
        str(output),
    ]


@pytest.mark.e2e
def test_explore_dry_run_prints_matrix_without_browser_or_model(
    tmp_path, monkeypatch
) -> None:
    for name in _MODEL_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)

    async def forbidden_crawler(spec, policy, **kwargs):
        raise AssertionError("dry-run must not launch a browser or crawl")

    async def forbidden_synthesizer(corpus, max_scenarios, settings):
        raise AssertionError("dry-run must not call the synthesis model")

    monkeypatch.setattr(cli, "_explore_run_crawler", forbidden_crawler)
    monkeypatch.setattr(cli, "_explore_run_synthesizer", forbidden_synthesizer)

    result = runner.invoke(
        app,
        [
            "explore",
            str(DEMO_PROJECT),
            "--starting-url",
            "https://a.test/",
            "--starting-url",
            "https://a.test/docs",
            "--depth",
            "1",
            "--max-pages",
            "10",
            "--max-scenarios",
            "5",
            "--auto-accept",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "explore matrix" in result.output
    assert "starts: 2" in result.output
    assert "depth: 1" in result.output
    assert "max_pages: 10" in result.output
    assert "max_scenarios: 5" in result.output
    assert "estimated_pages:" in result.output
    assert "synthesis token estimate" in result.output
    estimate = int(re.search(r"estimated_pages:\s*(\d+)", result.output).group(1))  # type: ignore[union-attr]
    assert 1 <= estimate <= 10


@pytest.mark.e2e
def test_explore_auto_accept_produces_exploration_artifacts(
    tmp_path, fake_exploration
) -> None:
    output = tmp_path / "out"

    result = runner.invoke(app, _explore_args(output))

    assert result.exit_code == 0, result.output
    index_path = output / "exploration" / "index.json"
    assert index_path.exists()
    index = _read_json(index_path)
    assert index["schema_version"] == "exploration-index-v1"
    assert len(index["attempts"]) == 1

    record = index["attempts"][0]
    attempt_dir = output / "exploration" / "attempts" / record["attempt_id"]
    assert attempt_dir.is_dir()
    for name in (
        "corpus.json",
        "suggestions.json",
        "curated.json",
        "project.fragment.yaml",
        "manifest.json",
    ):
        assert (attempt_dir / name).is_file(), name

    corpus = _read_json(attempt_dir / "corpus.json")
    assert len(corpus["pages"]) == 1
    suggestions = _read_json(attempt_dir / "suggestions.json")
    curated = _read_json(attempt_dir / "curated.json")
    assert len(curated) == len(suggestions) == 1

    fragment = yaml.safe_load(
        (attempt_dir / "project.fragment.yaml").read_text(encoding="utf-8")
    )
    assert fragment["experiments"][0]["id"] == "exploration-run"


@pytest.mark.e2e
def test_generated_project_is_runnable_via_dry_run(tmp_path, fake_exploration) -> None:
    output = tmp_path / "out"

    result = runner.invoke(app, _explore_args(output))
    assert result.exit_code == 0, result.output

    generated = output / "project.yaml"
    assert generated.exists()
    validate_result = runner.invoke(app, ["validate", str(generated)])
    assert validate_result.exit_code == 0, validate_result.output

    run_result = runner.invoke(
        app,
        [
            "run",
            str(generated),
            "--experiment",
            "exploration-run",
            "--output",
            str(tmp_path / "run-out"),
            "--dry-run",
        ],
    )
    assert run_result.exit_code == 0, run_result.output
    assert "experiment: exploration-run" in run_result.output
    specs_match = re.search(r"run specs: ([1-9]\d*)", run_result.output)
    assert specs_match is not None, run_result.output

    alt_result = runner.invoke(
        app,
        [
            "run",
            str(output / "exploration" / "generated.yaml"),
            "--experiment",
            "exploration-run",
            "--output",
            str(tmp_path / "alt-run-out"),
            "--dry-run",
        ],
    )
    assert alt_result.exit_code == 0, alt_result.output


@pytest.mark.e2e
def test_report_works_offline_after_exploration_run(tmp_path, fake_exploration) -> None:
    output = tmp_path / "out"

    result = runner.invoke(app, _explore_args(output))
    assert result.exit_code == 0, result.output

    # Reuse existing report fixtures: one finalized run bundle stands in for an
    # executed exploration-run without needing a live browser or model.
    run_output = tmp_path / "run-out"
    _write_run(run_output, "exploration-run-1", version="improved", discovery_cost=4)

    report_result = runner.invoke(
        app,
        [
            "report",
            str(run_output),
            "--output",
            str(run_output / "report.html"),
        ],
    )
    assert report_result.exit_code == 0, report_result.output

    html = (run_output / "report.html").read_text(encoding="utf-8")
    assert "Invite" in html
    # The report is self-contained: no external resource requests.
    assert '<script src="http' not in html
    assert 'src="http' not in html


@pytest.mark.e2e
def test_exploration_artifact_appears_in_output_index(
    tmp_path, fake_exploration
) -> None:
    output = tmp_path / "out"

    first = runner.invoke(app, _explore_args(output))
    assert first.exit_code == 0, first.output

    index = _read_json(output / "exploration" / "index.json")
    assert len(index["attempts"]) == 1
    record = index["attempts"][0]
    assert record["status"] == "succeeded"
    assert re.fullmatch(r"[0-9a-f]{64}", record["corpus_digest"])
    assert record["attempt_id"] == (
        f"{record['attempt_id'].rsplit('-', 2)[0]}-"
        f"{record['corpus_digest'][:12]}-"
        f"{record['attempt_id'].rsplit('-', 2)[2]}"
    )
    assert record["page_count"] == 1
    assert record["scenario_count"] == 1
    manifest = _read_json(
        output / "exploration" / "attempts" / record["attempt_id"] / "manifest.json"
    )
    assert manifest["corpus_digest"] == record["corpus_digest"]

    # A second exploration publishes a new immutable attempt instead of overwriting.
    second = runner.invoke(app, _explore_args(output))
    assert second.exit_code == 0, second.output

    updated_index = _read_json(output / "exploration" / "index.json")
    assert len(updated_index["attempts"]) == 2
    attempt_ids = [item["attempt_id"] for item in updated_index["attempts"]]
    assert len(set(attempt_ids)) == 2
    for attempt_id in attempt_ids:
        assert (output / "exploration" / "attempts" / attempt_id).is_dir()
