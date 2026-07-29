from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

import ux_analyzer.cli as cli
from ux_analyzer.adapters.openai import OpenAICompatibleSettings
from ux_analyzer.application.experiment import (
    ExperimentContext,
    ExperimentResult,
    expand_experiment,
)
from ux_analyzer.cli import app
from ux_analyzer.config.loader import load_project
from ux_analyzer.domain.attention import AttentionState
from ux_analyzer.domain.benchmark import Budget
from ux_analyzer.domain.interface import BoundingBox, ElementSnapshot, ViewportSnapshot
from ux_analyzer.ports.observation import (
    SessionHandle,
    ViewportSize,
)
from ux_analyzer.ports.observation import TestAccountId as AccountId
from ux_analyzer.providers.full_list_policy import FullListPolicy

DEMO_PROJECT = Path(__file__).parents[3] / "benchmarks" / "demo" / "project.yaml"
runner = CliRunner()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_run(root: Path) -> Path:
    run = root / "runs" / "run-demo"
    (run / "artifacts").mkdir(parents=True)
    (run / "artifacts" / "trace.zip").write_bytes(b"trace")
    _write_json(
        run / "manifest.json",
        {
            "run_id": "run-demo",
            "seed": 3,
            "config_digest": "digest",
            "endpoint_origin": "https://llm.example.test",
            "model_ids": {"cognitive": "model"},
            "prompt_versions": {"cognitive": "cognitive-v1"},
            "provider_versions": {"fixture": "v1"},
        },
    )
    (run / "timeline.jsonl").write_text(
        json.dumps(
            {
                "sequence": 1,
                "kind": "run-terminated",
                "outcome": {"kind": "verified-success"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        run / "result.json",
        {
            "run_id": "run-demo",
            "agent_claimed_success": True,
            "verification": {"verified": True},
        },
    )
    return run


def test_validate_reports_actionable_config_error(tmp_path: Path) -> None:
    project = yaml.safe_load(DEMO_PROJECT.read_text(encoding="utf-8"))
    assert isinstance(project, dict)
    project["scenarios"][0]["application_version_ids"] = ["missing-version"]
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")

    result = runner.invoke(app, ["validate", str(invalid)])

    assert result.exit_code == 1
    assert "validation error" in result.stdout
    assert "unknown application version" in result.stdout
    assert "missing-version" in result.stdout


def test_env_check_never_prints_secret_values(monkeypatch) -> None:
    secret = "super-secret-api-key"
    monkeypatch.setenv("UXA_LLM_API_KEY", secret)
    monkeypatch.delenv("UXA_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("UXA_SCENT_MODEL", raising=False)
    monkeypatch.delenv("UXA_COGNITIVE_MODEL", raising=False)

    result = runner.invoke(
        app,
        ["validate", str(DEMO_PROJECT), "--check-env"],
    )

    assert result.exit_code == 1
    assert "missing model environment variables" in result.stdout
    assert secret not in result.stdout
    assert "UXA_LLM_API_KEY" not in result.stdout


def test_run_dry_run_prints_matrix_model_calls_and_serial_default(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("UXA_LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("UXA_LLM_API_KEY", "super-secret-api-key")
    monkeypatch.setenv("UXA_SCENT_MODEL", "scent-model")
    monkeypatch.setenv("UXA_COGNITIVE_MODEL", "cognitive-model")

    result = runner.invoke(
        app,
        [
            "run",
            str(DEMO_PROJECT),
            "--experiment",
            "core-pair",
            "--output",
            str(tmp_path),
            "--dry-run",
            "--check-env",
        ],
    )

    assert result.exit_code == 0
    assert "policies: full-list, progressive-prominence-scent" in result.stdout
    assert "workers: 1" in result.stdout
    assert "run specs: 160" in result.stdout
    assert "estimated model calls: 320" in result.stdout
    assert "super-secret-api-key" not in result.stdout


def test_ablate_selects_optional_policies_and_run_count_override(
    tmp_path: Path,
) -> None:
    result = runner.invoke(
        app,
        [
            "ablate",
            str(DEMO_PROJECT),
            "--experiment",
            "ablations",
            "--run-count",
            "2",
            "--output",
            str(tmp_path),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "policies: prominence-ranked-list, progressive-prominence" in result.stdout
    assert "seeds per cell: 2" in result.stdout
    assert "run specs: 32" in result.stdout


def test_production_policy_adapter_preserves_complete_list_observation() -> None:
    snapshot = ViewportSnapshot(
        id="viewport-complete",
        elements=tuple(
            ElementSnapshot(
                id=f"item-{index}",
                role="button",
                label=f"Item {index}",
                bounds=BoundingBox(x=index * 20, y=0, width=10, height=10),
                visibility_fraction=1,
                actionable=True,
            )
            for index in range(5)
        ),
    )
    state = AttentionState.initial(Budget(10, 10, 5, 10), confidence=0.5, frustration=0)

    selection = cli._AttentionPolicyAdapter(FullListPolicy()).next_observation(
        state,
        snapshot,
        (),
        (),
        object(),
    )

    assert selection.selected_ids == tuple(f"item-{index}" for index in range(5))
    assert len(selection.observation.newly_revealed_elements) == 5


def test_production_agent_uses_project_and_persona_runtime_configuration(
    tmp_path: Path,
) -> None:
    loaded = load_project(DEMO_PROJECT)
    definition = next(
        item for item in loaded.project.experiments if item.id == "core-pair"
    )
    spec = next(
        item
        for item in expand_experiment(
            ExperimentContext(definition, loaded.project, loaded.config_digest)
        )
        if item.policy.value == "progressive-prominence-scent"
    )

    class FakeClient:
        endpoint_origin = "https://llm.example.test"
        records: tuple[object, ...] = ()

    settings = OpenAICompatibleSettings(
        base_url="https://llm.example.test/v1",
        api_key="secret",
        scent_model="scent-model",
        cognitive_model="cognitive-model",
    )
    client = FakeClient()
    agent = cli._build_agent(
        spec,
        adapter=object(),
        client=client,
        output=tmp_path,
        fixture_origin="http://fixture.test",
        settings=settings,
        runtime=loaded.runtime,
    )

    assert agent.prominence_provider.config.version == "heuristic-prominence-v1"
    assert agent.attention_policy._policy.config.temperature == pytest.approx(
        spec.persona.attention_temperature
    )
    assert agent.state_update_config.version == "state-updates-v1"
    assert agent.state_update_config.abandonment_threshold == pytest.approx(
        spec.persona.abandonment_threshold
    )
    assert agent.model_record_source is client
    assert agent.result_evaluator is not None


def test_report_regenerates_from_finalized_bundles(tmp_path: Path) -> None:
def test_session_config_uses_scenario_viewport(tmp_path: Path) -> None:
    loaded = load_project(DEMO_PROJECT)
    definition = next(
        item for item in loaded.project.experiments if item.id == "core-pair"
    )
    spec = next(
        iter(
            expand_experiment(
                ExperimentContext(definition, loaded.project, loaded.config_digest)
            )
        )
    )
    spec = replace(
        spec,
        scenario=replace(spec.scenario, viewport_width=900, viewport_height=700),
    )

    config = cli._session_config(
        spec,
        output=tmp_path,
        fixture_origin="http://fixture.test",
    )

    assert config.viewport == ViewportSize(width=900, height=700)


def test_full_list_bundle_manifest_omits_unused_scent_roles(tmp_path: Path) -> None:
    loaded = load_project(DEMO_PROJECT)
    definition = next(
        item for item in loaded.project.experiments if item.id == "core-pair"
    )
    spec = next(
        item
        for item in expand_experiment(
            ExperimentContext(definition, loaded.project, loaded.config_digest)
        )
        if item.policy.value == "full-list"
    )
    settings = OpenAICompatibleSettings(
        base_url="https://llm.example.test/v1",
        api_key="secret",
        scent_model="scent-model",
        cognitive_model="cognitive-model",
    )

    writer = cli._BundleFactory(tmp_path, settings, loaded.runtime).start(spec)
    manifest = json.loads((writer.staging_path / "manifest.json").read_text())
    writer.abort("test complete")

    assert manifest["model_ids"] == {"cognitive": "cognitive-model"}
    assert manifest["prompt_versions"] == {"cognitive": "cognitive-v1"}
    assert [item["role"] for item in manifest["provider_manifests"]] == ["cognitive"]


@pytest.mark.asyncio
async def test_fixture_provider_resets_state_before_reload_and_deletes_on_end(
    monkeypatch, tmp_path: Path
) -> None:
    events: list[str] = []

    class FakeAdapter:
        async def reset(self, session: SessionHandle) -> None:
            del session
            events.append("browser-reset")

        async def end_session(self, session: SessionHandle) -> None:
            del session
            events.append("browser-end")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> None:
            del args

        async def post(self, url: str, json: object) -> FakeResponse:
            del url, json
            events.append("fixture-reset")
            return FakeResponse()

        async def delete(self, url: str) -> FakeResponse:
            del url
            events.append("fixture-delete")
            return FakeResponse()

    monkeypatch.setattr(cli.httpx, "AsyncClient", lambda **kwargs: FakeClient())
    provider = cli._FixtureObservationProvider(
        FakeAdapter(),  # type: ignore[arg-type]
        "http://fixture.test",
        {"totp_code": "246810"},
    )
    session = SessionHandle(
        session_id="run-1",
        test_account_id=AccountId("test-run-1"),
        viewport=ViewportSize(900, 700),
        trace_path=tmp_path / "trace.zip",
        blocked_events=[],
    )

    await provider.reset(session)
    await provider.end_session(session)

    assert events == [
        "fixture-reset",
        "browser-reset",
        "browser-end",
        "fixture-delete",
    ]


    _write_run(tmp_path)
    output = tmp_path / "report.html"

    result = runner.invoke(
        app,
        ["report", str(tmp_path), "--output", str(output)],
    )

    assert result.exit_code == 0
    assert output.is_file()
    assert "report generated" in result.stdout


def test_inspect_run_prints_terminal_outcome_and_artifacts(tmp_path: Path) -> None:
    run = _write_run(tmp_path)

    result = runner.invoke(app, ["inspect-run", str(run)])

    assert result.exit_code == 0
    assert "verified-success" in result.stdout
    assert "artifacts/trace.zip" in result.stdout


def test_fixture_serve_delegates_to_uvicorn(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(target: str, **kwargs: object) -> None:
        captured["target"] = target
        captured.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)

    result = runner.invoke(
        app,
        ["fixture", "serve", "--host", "127.0.0.1", "--port", "8765"],
    )

    assert result.exit_code == 0
    assert captured == {
        "target": "fixture_app.app:app",
        "host": "127.0.0.1",
        "port": 8765,
    }


def test_production_run_completes_evaluation_summary_and_report(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("UXA_LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("UXA_LLM_API_KEY", "super-secret-api-key")
    monkeypatch.setenv("UXA_SCENT_MODEL", "scent-model")
    monkeypatch.setenv("UXA_COGNITIVE_MODEL", "cognitive-model")
    completed: dict[str, object] = {}

    async def fake_execute_matrix(*args: object, **kwargs: object) -> ExperimentResult:
        del args, kwargs
        return ExperimentResult(specs=(), results=(), failures=())

    def fake_complete(
        result: ExperimentResult, *, output: Path, runtime: object
    ) -> tuple[Path, Path]:
        completed.update(result=result, output=output, runtime=runtime)
        summary = output / "experiment.json"
        report = output / "report.html"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text("{}", encoding="utf-8")
        report.write_text("<html></html>", encoding="utf-8")
        return summary, report

    monkeypatch.setattr(cli, "_execute_matrix", fake_execute_matrix)
    monkeypatch.setattr(cli, "_complete_experiment", fake_complete, raising=False)

    result = runner.invoke(
        app,
        [
            "run",
            str(DEMO_PROJECT),
            "--experiment",
            "core-pair",
            "--run-count",
            "1",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert completed["output"] == tmp_path
    assert completed["runtime"] is not None
    assert "evaluation summary:" in result.stdout
    assert "report generated:" in result.stdout
