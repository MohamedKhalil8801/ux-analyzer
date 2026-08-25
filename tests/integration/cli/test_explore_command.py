"""Integration tests for the `uxa explore` command wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from typer.testing import CliRunner

import ux_analyzer.cli as cli
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

DEMO_PROJECT = Path(__file__).parents[3] / "benchmarks" / "demo" / "project.yaml"
runner = CliRunner()

MODEL_ENV = {
    "UXA_LLM_MODE": "api",
    "UXA_LLM_BASE_URL": "https://llm.example.test/v1",
    "UXA_LLM_API_KEY": "test-key",
    "UXA_SCENT_MODEL": "test-scent-model",
    "UXA_COGNITIVE_MODEL": "test-cognitive-model",
}


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
                origin=urlsplit_origin(normalized),
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


def urlsplit_origin(normalized_url: str) -> str:
    from urllib.parse import urlsplit

    parsed = urlsplit(normalized_url)
    host = parsed.hostname.lower()
    port = parsed.port
    if port is None or port == 443:
        return f"https://{host}"
    return f"https://{host}:{port}"


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


def _install_fakes(monkeypatch) -> None:
    async def fake_crawler(spec, policy, *, on_frontier=None, resume_from=None):
        return _fake_corpus(spec.start_urls, spec.max_pages)

    async def fake_synthesizer(corpus, max_scenarios, settings):
        return _fake_suggestions(corpus, max_scenarios, settings)

    monkeypatch.setattr(cli, "_explore_run_crawler", fake_crawler)
    monkeypatch.setattr(cli, "_explore_run_synthesizer", fake_synthesizer)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_explore_depth_zero_auto_accept(tmp_path, monkeypatch) -> None:
    for name, value in MODEL_ENV.items():
        monkeypatch.setenv(name, value)
    _install_fakes(monkeypatch)
    output = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "explore",
            str(DEMO_PROJECT),
            "--starting-url",
            "https://a.test/",
            "--depth",
            "0",
            "--max-pages",
            "1",
            "--max-scenarios",
            "10",
            "--auto-accept",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    index_path = output / "exploration" / "index.json"
    assert index_path.exists()
    index = _read_json(index_path)
    assert len(index["attempts"]) == 1

    attempt_dir = (
        output / "exploration" / "attempts" / index["attempts"][0]["attempt_id"]
    )
    assert (attempt_dir / "corpus.json").exists()
    assert (attempt_dir / "suggestions.json").exists()
    assert (attempt_dir / "curated.json").exists()
    assert (attempt_dir / "project.fragment.yaml").exists()
    assert (attempt_dir / "manifest.json").exists()

    suggestions = _read_json(attempt_dir / "suggestions.json")
    curated = _read_json(attempt_dir / "curated.json")
    assert len(suggestions) >= 1
    assert len(curated) == len(suggestions)

    fragment_path = output / "exploration" / "project.fragment.yaml"
    assert fragment_path.exists()
    fragment = yaml.safe_load(fragment_path.read_text(encoding="utf-8"))
    assert fragment["experiments"][0]["id"] == "exploration-run"

    # Generated full project merges base scenarios (append) and is runnable.
    generated = output / "project.yaml"
    assert generated.exists()
    merged = yaml.safe_load(generated.read_text(encoding="utf-8"))
    base = yaml.safe_load(DEMO_PROJECT.read_text(encoding="utf-8"))
    assert len(merged["scenarios"]) == len(base["scenarios"]) + len(curated)

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
    assert "next step: uxa run" in result.output


def test_explore_bootstrap_project_is_runnable(tmp_path, monkeypatch) -> None:
    for name, value in MODEL_ENV.items():
        monkeypatch.setenv(name, value)
    _install_fakes(monkeypatch)
    output = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "explore",
            "--starting-url",
            "https://a.test/",
            "--starting-url",
            "https://b.test/",
            "--depth",
            "0",
            "--max-pages",
            "2",
            "--auto-accept",
            "--output",
            str(output),
        ],
    )
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


def test_explore_bootstrap_preserves_resource_origins(tmp_path, monkeypatch) -> None:
    for name, value in MODEL_ENV.items():
        monkeypatch.setenv(name, value)
    _install_fakes(monkeypatch)
    output = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "explore",
            "--starting-url",
            "https://a.test/",
            "--depth",
            "0",
            "--max-pages",
            "1",
            "--max-scenarios",
            "1",
            "--auto-accept",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    generated = yaml.safe_load((output / "project.yaml").read_text(encoding="utf-8"))
    version = generated["applications"][0]["versions"][0]
    allowed = set(version["allowed_origins"])
    # Start origin must stay first-class and the crawl-time CDN allowlist
    # must survive materialization; otherwise `uxa run` safety-blocks on the
    # first Google Fonts subresource.
    assert "https://a.test" in allowed
    assert "https://fonts.googleapis.com" in allowed
    assert "https://fonts.gstatic.com" in allowed


def test_explore_rejects_invalid_depth(monkeypatch) -> None:
    result = runner.invoke(
        app,
        [
            "explore",
            str(DEMO_PROJECT),
            "--starting-url",
            "https://a.test/",
            "--depth",
            "10",
        ],
    )
    assert result.exit_code != 0
    assert "depth must be between 0 and 5" in result.output


def test_explore_rejects_duplicate_start_urls(monkeypatch) -> None:
    result = runner.invoke(
        app,
        [
            "explore",
            str(DEMO_PROJECT),
            "--starting-url",
            "https://a.test/",
            "--starting-url",
            "https://a.test/#section",
        ],
    )
    assert result.exit_code != 0
    assert "unique after normalization" in result.output


def test_explore_rejects_max_pages_below_starts_at_depth_zero(monkeypatch) -> None:
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
            "0",
            "--max-pages",
            "1",
            "--auto-accept",
        ],
    )
    assert result.exit_code != 0
    assert "max_pages must be >= number of start URLs when depth is 0" in result.output


def test_explore_requires_model_env_unless_dry_run(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    for name in (
        *MODEL_ENV.keys(),
        "UXA_SCENT_MODEL",
        "UXA_COGNITIVE_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    without_env = runner.invoke(
        app,
        [
            "explore",
            str(DEMO_PROJECT),
            "--starting-url",
            "https://a.test/",
            "--depth",
            "0",
            "--auto-accept",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert without_env.exit_code != 0
    assert "missing model environment variables" in without_env.output

    dry_run = runner.invoke(
        app,
        [
            "explore",
            str(DEMO_PROJECT),
            "--starting-url",
            "https://a.test/",
            "--depth",
            "0",
            "--auto-accept",
            "--dry-run",
        ],
    )
    assert dry_run.exit_code == 0, dry_run.output
    assert "explore matrix" in dry_run.output
    assert "synthesis token estimate" in dry_run.output


def test_explore_artifact_digest_stability(tmp_path, monkeypatch) -> None:
    for name, value in MODEL_ENV.items():
        monkeypatch.setenv(name, value)
    _install_fakes(monkeypatch)

    def _invoke(output: Path):
        return runner.invoke(
            app,
            [
                "explore",
                str(DEMO_PROJECT),
                "--starting-url",
                "https://a.test/",
                "--depth",
                "0",
                "--max-pages",
                "1",
                "--auto-accept",
                "--output",
                str(output),
            ],
        )

    first_output = tmp_path / "one"
    second_output = tmp_path / "two"
    first = _invoke(first_output)
    second = _invoke(second_output)
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output

    def _attempt(output: Path) -> dict[str, Any]:
        index = _read_json(output / "exploration" / "index.json")
        attempt_id = index["attempts"][0]["attempt_id"]
        manifest = _read_json(
            output / "exploration" / "attempts" / attempt_id / "manifest.json"
        )
        return {"index_record": index["attempts"][0], "manifest": manifest}

    first_attempt = _attempt(first_output)
    second_attempt = _attempt(second_output)

    # Page-based corpus digest is identical across independent runs.
    assert (
        first_attempt["index_record"]["corpus_digest"]
        == second_attempt["index_record"]["corpus_digest"]
    )
    assert (
        first_attempt["manifest"]["corpus_digest"]
        == second_attempt["manifest"]["corpus_digest"]
    )
    first_digests = first_attempt["manifest"]["digests"]
    second_digests = second_attempt["manifest"]["digests"]
    assert first_digests["corpus"] == second_digests["corpus"]
    assert first_digests["suggestions"] == second_digests["suggestions"]
    assert first_digests["curated"] == second_digests["curated"]

    # Each run publishes its own immutable attempt directory.
    for output in (first_output, second_output):
        index = _read_json(output / "exploration" / "index.json")
        attempt_id = index["attempts"][0]["attempt_id"]
        assert (output / "exploration" / "attempts" / attempt_id).is_dir()


def _explore_success_args(output: Path) -> list[str]:
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
        "--output",
        str(output),
    ]


def _explore_auto_accept_args(output: Path) -> list[str]:
    return [
        *_explore_success_args(output)[:10],
        "--auto-accept",
        "--output",
        str(output),
    ]


def test_explore_crawler_failure_exits_nonzero_without_succeeded_artifacts(
    tmp_path, monkeypatch
) -> None:
    for name, value in MODEL_ENV.items():
        monkeypatch.setenv(name, value)

    async def broken_crawler(spec: Any, policy: Any, **kwargs: Any) -> Any:
        raise RuntimeError("browser crashed")

    monkeypatch.setattr(cli, "_explore_run_crawler", broken_crawler)
    output = tmp_path / "out"

    result = runner.invoke(app, _explore_auto_accept_args(output))

    assert result.exit_code != 0
    assert "crawl failed" in result.output
    assert "RuntimeError" in result.output
    # No artifacts may claim success from a failed crawl.
    exploration_root = output / "exploration"
    if (exploration_root / "index.json").exists():
        index = _read_json(exploration_root / "index.json")
        assert all(record.get("status") != "succeeded" for record in index["attempts"])
    assert not (exploration_root / "checkpoint").exists()


def test_explore_synthesizer_failure_persists_unavailable_attempt_and_exits_nonzero(
    tmp_path, monkeypatch
) -> None:
    for name, value in MODEL_ENV.items():
        monkeypatch.setenv(name, value)
    _install_fakes(monkeypatch)

    async def broken_synthesizer(corpus: Any, max_scenarios: int, settings: Any) -> Any:
        raise RuntimeError("model endpoint down")

    monkeypatch.setattr(cli, "_explore_run_synthesizer", broken_synthesizer)
    output = tmp_path / "out"

    result = runner.invoke(app, _explore_auto_accept_args(output))

    assert result.exit_code != 0
    assert "synthesis failed" in result.output

    index = _read_json(output / "exploration" / "index.json")
    assert len(index["attempts"]) == 1
    record = index["attempts"][0]
    assert record["status"] == "unavailable"
    manifest = _read_json(
        output / "exploration" / "attempts" / record["attempt_id"] / "manifest.json"
    )
    assert manifest["status"] == "unavailable"
    assert "scenario synthesis unavailable" in str(manifest.get("limitation"))
    assert "RuntimeError" in str(manifest.get("limitation"))
    suggestions = _read_json(
        output / "exploration" / "attempts" / record["attempt_id"] / "suggestions.json"
    )
    assert suggestions == []
    # The real crawl evidence is retained; no fabricated suggestions exist.
    corpus = _read_json(
        output / "exploration" / "attempts" / record["attempt_id"] / "corpus.json"
    )
    assert len(corpus["pages"]) >= 1


def test_explore_empty_curated_set_exits_nonzero(tmp_path, monkeypatch) -> None:
    for name, value in MODEL_ENV.items():
        monkeypatch.setenv(name, value)

    async def empty_synthesizer(corpus: Any, max_scenarios: int, settings: Any) -> Any:
        return ()

    async def fake_crawler(spec: Any, policy: Any, **kwargs: Any) -> Any:
        return _fake_corpus(spec.start_urls, spec.max_pages)

    monkeypatch.setattr(cli, "_explore_run_crawler", fake_crawler)
    monkeypatch.setattr(cli, "_explore_run_synthesizer", empty_synthesizer)
    output = tmp_path / "out"

    result = runner.invoke(app, _explore_auto_accept_args(output))

    assert result.exit_code != 0
    assert result.exit_code == 1
    assert "curated set is empty" in result.output
    assert not (output / "exploration" / "index.json").exists()
    assert not (output / "project.yaml").exists()


def test_explore_resume_skips_crawl_after_interrupted_review(
    tmp_path, monkeypatch
) -> None:
    for name, value in MODEL_ENV.items():
        monkeypatch.setenv(name, value)
    calls = {"crawler": 0}

    async def counting_crawler(spec: Any, policy: Any, **kwargs: Any) -> Any:
        calls["crawler"] += 1
        return _fake_corpus(spec.start_urls, spec.max_pages)

    async def fake_synthesizer(corpus: Any, max_scenarios: int, settings: Any) -> Any:
        return _fake_suggestions(corpus, max_scenarios, settings)

    monkeypatch.setattr(cli, "_explore_run_crawler", counting_crawler)
    monkeypatch.setattr(cli, "_explore_run_synthesizer", fake_synthesizer)

    review_calls = {"n": 0}

    async def interrupted_then_curated(self: Any) -> Any:
        review_calls["n"] += 1
        if review_calls["n"] == 1:
            # Simulate Ctrl-C during human review.
            raise KeyboardInterrupt
        from ux_analyzer.adapters.exploration_server import _suggestion_to_dict

        return {
            "curated": [_suggestion_to_dict(s) for s in self.suggestions],
        }

    monkeypatch.setattr(
        cli.ExplorationReviewServer, "serve_forever", interrupted_then_curated
    )

    output = tmp_path / "out"
    args = _explore_success_args(output)

    first = runner.invoke(app, args)
    assert first.exit_code != 0
    assert "interrupted" in first.output

    checkpoint_dir = output / "exploration" / "checkpoint"
    state = _read_json(checkpoint_dir / "state.json")
    assert state["status"] == "pending"
    assert state["has_suggestions"] is True
    payload = _read_json(checkpoint_dir / "payload.json")
    assert len(payload["corpus"]["pages"]) >= 1
    assert len(payload["suggestions"]) >= 1
    # No attempt was published before curation completed.
    assert not (output / "exploration" / "index.json").exists()

    second = runner.invoke(app, args)
    assert second.exit_code == 0, second.output
    # Crawl and synthesis each ran exactly once across both invocations.
    assert calls["crawler"] == 1
    assert review_calls["n"] == 2
    assert "crawl skipped" in second.output
    assert "synthesis skipped" in second.output

    index = _read_json(output / "exploration" / "index.json")
    assert len(index["attempts"]) == 1
    assert index["attempts"][0]["status"] == "succeeded"
    # Checkpoint cleared after successful completion.
    assert not checkpoint_dir.exists()


def test_explore_resume_with_mismatched_fingerprint_exits_nonzero(
    tmp_path, monkeypatch
) -> None:
    for name, value in MODEL_ENV.items():
        monkeypatch.setenv(name, value)
    calls = {"crawler": 0}

    async def counting_crawler(spec: Any, policy: Any, **kwargs: Any) -> Any:
        calls["crawler"] += 1
        return _fake_corpus(spec.start_urls, spec.max_pages)

    async def fake_synthesizer(corpus: Any, max_scenarios: int, settings: Any) -> Any:
        return _fake_suggestions(corpus, max_scenarios, settings)

    monkeypatch.setattr(cli, "_explore_run_crawler", counting_crawler)
    monkeypatch.setattr(cli, "_explore_run_synthesizer", fake_synthesizer)

    review_calls = {"n": 0}

    async def interrupted_then_curated(self: Any) -> Any:
        review_calls["n"] += 1
        if review_calls["n"] == 1:
            raise KeyboardInterrupt
        from ux_analyzer.adapters.exploration_server import _suggestion_to_dict

        return {
            "curated": [_suggestion_to_dict(s) for s in self.suggestions],
        }

    monkeypatch.setattr(
        cli.ExplorationReviewServer, "serve_forever", interrupted_then_curated
    )

    output = tmp_path / "out"

    def _args(max_pages: str) -> list[str]:
        return [
            "explore",
            str(DEMO_PROJECT),
            "--starting-url",
            "https://a.test/",
            "--depth",
            "0",
            "--max-pages",
            max_pages,
            "--max-scenarios",
            "3",
            "--output",
            str(output),
        ]

    first = runner.invoke(app, _args("1"))
    assert first.exit_code != 0
    assert (output / "exploration" / "checkpoint" / "state.json").exists()
    crawler_calls_after_first_run = calls["crawler"]

    second = runner.invoke(app, [*_args("2"), "--resume"])
    assert second.exit_code != 0
    assert "does not match" in second.output
    assert "resume requested but the existing checkpoint does not match" in (
        second.output
    )
    # The mismatched resume must be refused before any crawl work happens.
    assert calls["crawler"] == crawler_calls_after_first_run


def test_explore_interrupted_mid_crawl_retains_pages_and_resume_completes(
    tmp_path, monkeypatch
) -> None:
    for name, value in MODEL_ENV.items():
        monkeypatch.setenv(name, value)
    from ux_analyzer.application.exploration_crawler import CrawlFrontier

    calls = {"crawler": 0}
    docs_url = normalize_crawl_url("https://a.test/docs")

    def _page(url: str, depth: int, title: str) -> CrawlPage:
        normalized = normalize_crawl_url(url)
        return CrawlPage(
            url=url,
            normalized_url=normalized,
            origin=urlsplit_origin(normalized),
            depth=depth,
            title=title,
            headings=("Documentation",),
        )

    async def interrupting_then_recovering_crawler(
        spec: Any, policy: Any, *, on_frontier=None, resume_from=None
    ) -> Any:
        calls["crawler"] += 1
        if calls["crawler"] == 1:
            # Complete the first page (emitting a per-page checkpoint),
            # then simulate Ctrl-C before the second page.
            home = _page("https://a.test/", 0, "Home")
            if on_frontier is not None:
                await on_frontier(
                    CrawlFrontier(
                        pages=(home,),
                        link_graph={home.normalized_url: (docs_url,)},
                        visited=(home.normalized_url, docs_url),
                        queued=((docs_url, 1),),
                        started_at="2026-01-01T00:00:00Z",
                    )
                )
            raise KeyboardInterrupt
        docs = _page(docs_url, 1, "Docs")
        pages = tuple(resume_from.pages) + (docs,) if resume_from else (docs,)
        graph = dict(resume_from.link_graph) if resume_from else {}
        graph[docs.normalized_url] = ()
        return CrawlCorpus(
            pages=pages,
            link_graph=graph,
            started_at=(
                resume_from.started_at if resume_from else "2026-01-01T00:00:00Z"
            ),
        )

    async def fake_synthesizer(corpus: Any, max_scenarios: int, settings: Any) -> Any:
        return _fake_suggestions(corpus, max_scenarios, settings)

    monkeypatch.setattr(
        cli, "_explore_run_crawler", interrupting_then_recovering_crawler
    )
    monkeypatch.setattr(cli, "_explore_run_synthesizer", fake_synthesizer)

    review_calls = {"n": 0}

    async def auto_curated(self: Any) -> Any:
        review_calls["n"] += 1
        from ux_analyzer.adapters.exploration_server import _suggestion_to_dict

        return {
            "curated": [_suggestion_to_dict(s) for s in self.suggestions],
        }

    monkeypatch.setattr(cli.ExplorationReviewServer, "serve_forever", auto_curated)

    output = tmp_path / "out"
    args = [
        "explore",
        str(DEMO_PROJECT),
        "--starting-url",
        "https://a.test/",
        "--depth",
        "1",
        "--max-pages",
        "2",
        "--max-scenarios",
        "3",
        "--output",
        str(output),
    ]

    first = runner.invoke(app, [*args, "--auto-accept"])
    assert first.exit_code != 0
    assert "interrupted" in first.output

    # Per-page checkpoint retains the completed page and the pending queue.
    checkpoint_dir = output / "exploration" / "checkpoint"
    state = _read_json(checkpoint_dir / "state.json")
    assert state["status"] == "pending"
    assert state["crawl_complete"] is False
    payload = _read_json(checkpoint_dir / "payload.json")
    assert len(payload["corpus"]["pages"]) == 1
    assert payload["frontier"]["queued"] == [[docs_url, 1]]
    assert sorted(payload["frontier"]["visited"]) == sorted(
        [normalize_crawl_url("https://a.test/"), docs_url]
    )

    second = runner.invoke(app, args)
    assert second.exit_code == 0, second.output
    assert "resuming interrupted crawl" in second.output
    assert "crawl skipped" not in second.output
    assert calls["crawler"] == 2

    index = _read_json(output / "exploration" / "index.json")
    record = index["attempts"][0]
    assert record["status"] == "succeeded"
    assert record["page_count"] == 2
    corpus = _read_json(
        output / "exploration" / "attempts" / record["attempt_id"] / "corpus.json"
    )
    assert len(corpus["pages"]) == 2
    # Checkpoint cleared after successful completion.
    assert not checkpoint_dir.exists()
