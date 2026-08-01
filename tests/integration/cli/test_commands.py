from __future__ import annotations

import hashlib
import json
import subprocess
import sys
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
    ExperimentFailure,
    ExperimentResult,
    expand_experiment,
)
from ux_analyzer.cli import app
from ux_analyzer.config.loader import load_project
from ux_analyzer.domain.attention import AttentionState
from ux_analyzer.domain.benchmark import Budget, ExperimentPolicy
from ux_analyzer.domain.interface import BoundingBox, ElementSnapshot, ViewportSnapshot
from ux_analyzer.ports.observation import (
    SessionHandle,
    ViewportSize,
)
from ux_analyzer.ports.observation import TestAccountId as AccountId
from ux_analyzer.providers.attention_policy import ProgressiveAttentionPolicy
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


def _write_resumable_metrics_bundle(
    root: Path,
    spec,
    *,
    cost: float,
    include_model_trial: bool = True,
) -> None:
    run = root / "runs" / spec.run_id
    run.mkdir(parents=True)
    metrics = {
        "run_id": spec.run_id,
        "seed": spec.seed,
        "scenario_id": spec.scenario.id,
        "application_version_id": spec.application_version.id,
        "persona_id": spec.persona.id,
        "policy": spec.policy.value,
        "target": {"element_id": "target"},
        "target_discovery_rank": 1,
        "inspected_elements": 1,
        "inspected_regions": 1,
        "scrolls": 0,
        "wrong_actions": 0,
        "backtracks": 0,
        "verified_completion": True,
        "claimed_completion": True,
        "false_success": False,
        "abandoned": False,
        "discovery_cost": {
            "inspection_cost": cost,
            "region_cost": 0,
            "scroll_cost": 0,
            "wrong_action_cost": 0,
            "backtrack_cost": 0,
            "uncertainty_cost": 0,
            "abandonment_penalty": 0,
        },
    }
    if include_model_trial:
        metrics["model_trial"] = spec.model_trial
    manifest = {"run_id": spec.run_id, "seed": spec.seed}
    if include_model_trial:
        manifest["model_trial"] = spec.model_trial
    contents = {
        "manifest.json": json.dumps(manifest).encode(),
        "timeline.jsonl": (
            json.dumps(
                {
                    "kind": "run-terminated",
                    "outcome": {"kind": "verified-success"},
                }
            ).encode()
            + b"\n"
        ),
        "result.json": json.dumps(
            {
                "run_id": spec.run_id,
                "outcome": {"kind": "verified-success"},
                "verification": {"verified": True},
                "agent_claimed_success": True,
                "metrics": metrics,
                "findings": [],
            }
        ).encode(),
    }
    for name, content in contents.items():
        (run / name).write_bytes(content)
    (run / "checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256(content).hexdigest()}  {name}\n"
            for name, content in contents.items()
        )
    )


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


def test_env_check_never_prints_secret_values(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
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
    assert "run specs: 88" in result.stdout
    assert "model calls for one attention cycle: 248" in result.stdout
    assert "maximum logical model calls: 5632" in result.stdout
    assert "deterministic seed repetitions suppressed: 72" in result.stdout
    assert "overall run timeout: none" in result.stdout
    assert "super-secret-api-key" not in result.stdout


def test_single_run_resolver_selects_one_stable_semantic_spec() -> None:
    first = cli._resolve_single_run(
        DEMO_PROJECT,
        scenario_id="invite-teammate",
        version_id="fixture-app-improved",
        persona_id="first-time-nontechnical",
        policy="progressive-prominence-scent",
        seed=0,
    )
    repeat = cli._resolve_single_run(
        DEMO_PROJECT,
        scenario_id="invite-teammate",
        version_id="fixture-app-improved",
        persona_id="first-time-nontechnical",
        policy="progressive-prominence-scent",
        seed=0,
    )

    assert len(first.specs) == 1
    spec = first.specs[0]
    assert spec.scenario.id == "invite-teammate"
    assert spec.application_version.id == "fixture-app-improved"
    assert spec.persona.id == "first-time-nontechnical"
    assert spec.policy.value == "progressive-prominence-scent"
    assert spec.seed == 0
    assert spec.run_id == repeat.specs[0].run_id


def test_run_one_dry_run_never_expands_other_matrix_cells(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run-one",
            str(DEMO_PROJECT),
            "--scenario",
            "invite-teammate",
            "--version",
            "fixture-app-improved",
            "--persona",
            "first-time-nontechnical",
            "--policy",
            "progressive-prominence-scent",
            "--seed",
            "0",
            "--output",
            str(tmp_path),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "run specs: 1" in result.stdout
    assert (
        "invite-teammate/fixture-app-improved/first-time-nontechnical/" in result.stdout
    )
    assert "enable-2fa/" not in result.stdout
    assert "fixture-app-defective/" not in result.stdout


def test_resume_matrix_skips_only_valid_finalized_runs(tmp_path: Path) -> None:
    matrix = cli._resolve_matrix_or_exit(
        DEMO_PROJECT, "core-pair", run_count=1, policies=()
    )
    completed = matrix.specs[0]
    run = tmp_path / "runs" / completed.run_id
    run.mkdir(parents=True)
    contents = {
        "manifest.json": json.dumps({"run_id": completed.run_id}).encode(),
        "timeline.jsonl": b'{"kind":"run-terminated","outcome":{"kind":"verified-success"}}\n',
        "result.json": json.dumps(
            {
                "run_id": completed.run_id,
                "outcome": {"kind": "verified-success"},
            }
        ).encode(),
    }
    for name, content in contents.items():
        (run / name).write_bytes(content)
    (run / "checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256(content).hexdigest()}  {name}\n"
            for name, content in contents.items()
        )
    )

    resumed, store = cli._prepare_resumed_matrix(matrix, tmp_path)

    assert completed.run_id not in {spec.run_id for spec in resumed.specs}
    assert len(resumed.specs) == len(matrix.specs) - 1
    assert store.state.finalized_run_ids == (completed.run_id,)


def test_resume_default_config_reuses_legacy_manifest_without_model_trial(
    tmp_path: Path,
) -> None:
    matrix = cli._resolve_matrix_or_exit(
        DEMO_PROJECT,
        "core-pair",
        run_count=1,
        policies=(),
    )
    spec = matrix.specs[0]
    legacy_payload = {
        "application_version_id": spec.application_version.id,
        "config_digest": spec.config_digest,
        "experiment_id": "core-pair",
        "persona_id": spec.persona.id,
        "policy": spec.policy.value,
        "scenario_id": spec.scenario.id,
        "seed": spec.seed,
    }
    canonical = json.dumps(
        legacy_payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    legacy_run_id = f"run-{hashlib.sha256(canonical).hexdigest()}"
    legacy_spec = replace(spec, run_id=legacy_run_id)
    _write_resumable_metrics_bundle(
        tmp_path,
        legacy_spec,
        cost=1,
        include_model_trial=False,
    )

    resumed, store = cli._prepare_resumed_matrix(matrix, tmp_path)

    assert spec.model_trial == 0
    assert spec.run_id == legacy_run_id
    assert spec.run_id not in {item.run_id for item in resumed.specs}
    assert store.state.finalized_run_ids == (legacy_run_id,)


@pytest.mark.parametrize("new_result_count", (0, 1))
def test_resume_completion_evaluates_all_selected_finalized_bundles(
    tmp_path: Path, new_result_count: int
) -> None:
    matrix = cli._resolve_matrix_or_exit(
        DEMO_PROJECT, "core-pair", run_count=1, policies=()
    )
    selected = matrix.specs[:2]
    for index, spec in enumerate(selected):
        _write_resumable_metrics_bundle(tmp_path, spec, cost=float(index + 1))
    returned = ()
    if new_result_count:
        from pydantic import TypeAdapter

        from ux_analyzer.application.evaluation import RunMetrics
        from ux_analyzer.application.run_agent import RunResult
        from ux_analyzer.domain.run import VerifiedSuccess
        from ux_analyzer.ports.verification import VerificationResult

        persisted = json.loads(
            (tmp_path / "runs" / selected[-1].run_id / "result.json").read_text()
        )
        returned = (
            RunResult(
                run_id=selected[-1].run_id,
                outcome=VerifiedSuccess(),
                verification=VerificationResult(verified=True),
                agent_claimed_success=True,
                state=None,  # type: ignore[arg-type]
                metrics=TypeAdapter(RunMetrics).validate_python(persisted["metrics"]),
            ),
        )

    summary_path, _ = cli._complete_experiment(
        ExperimentResult(specs=selected, results=returned, failures=()),
        output=tmp_path,
        runtime=matrix.loaded.runtime,
        selected_specs=selected,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert {item["run_id"] for item in summary["run_metrics"]} == {
        spec.run_id for spec in selected
    }


def test_resume_completion_keeps_model_trials_separate_for_variant_comparisons(
    tmp_path: Path,
) -> None:
    project = yaml.safe_load(DEMO_PROJECT.read_text(encoding="utf-8"))
    assert isinstance(project, dict)
    experiment = next(
        item for item in project["experiments"] if item["id"] == "focused-validation"
    )
    experiment["model_trials"] = [0, 1]
    project_path = tmp_path / "project.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")

    matrix = cli._resolve_matrix_or_exit(
        project_path,
        "focused-validation",
        run_count=1,
        policies=(),
    )
    assert {spec.model_trial for spec in matrix.specs} == {0, 1}
    for index, spec in enumerate(matrix.specs):
        _write_resumable_metrics_bundle(tmp_path, spec, cost=float(index + 1))

    resumed, store = cli._prepare_resumed_matrix(matrix, tmp_path)
    assert resumed.specs == ()
    assert set(store.state.finalized_run_ids) == {spec.run_id for spec in matrix.specs}

    summary_path, _ = cli._complete_experiment(
        ExperimentResult(specs=matrix.specs, results=(), failures=()),
        output=tmp_path,
        runtime=matrix.loaded.runtime,
        selected_specs=matrix.specs,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    comparisons = summary["variant_comparisons"]
    assert len(comparisons) == 4
    assert {comparison["paired_model_trials"][0] for comparison in comparisons} == {
        0,
        1,
    }
    assert {comparison["baseline"]["model_trial"] for comparison in comparisons} == {
        0,
        1,
    }


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
    assert "configured seeds: 2" in result.stdout
    assert "run specs: 24" in result.stdout


def test_focused_validation_dry_run_expands_exactly_four_balanced_cells(
    tmp_path: Path,
) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            str(DEMO_PROJECT),
            "--experiment",
            "focused-validation",
            "--output",
            str(tmp_path),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "run specs: 4" in result.stdout
    assert result.stdout.count("/full-list: 1 runs") == 2
    assert result.stdout.count("/progressive-prominence-scent: 1 runs") == 2


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


def test_cli_composition_distinguishes_progressive_ablation() -> None:
    prominence_only = cli._attention_policy_for(ExperimentPolicy.PROGRESSIVE_PROMINENCE)
    prominence_with_scent = cli._attention_policy_for(
        ExperimentPolicy.PROGRESSIVE_PROMINENCE_SCENT
    )

    assert isinstance(prominence_only, ProgressiveAttentionPolicy)
    assert isinstance(prominence_with_scent, ProgressiveAttentionPolicy)
    assert prominence_only.config.coarse_scent_weight == 0
    assert prominence_with_scent.config.coarse_scent_weight > 0


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
    assert agent.cognitive_agent.fixture_keys == tuple(
        sorted(spec.scenario.fixture_inputs.values)
    )
    assert agent.model_record_source is client
    assert agent.result_evaluator is not None


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
    tmp_path: Path,
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

    client = FakeClient()
    provider = cli._FixtureObservationProvider(
        FakeAdapter(),  # type: ignore[arg-type]
        "http://fixture.test",
        {"totp_code": "246810"},
        client,  # type: ignore[arg-type]
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


def test_report_regenerates_from_finalized_bundles(tmp_path: Path) -> None:
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


def test_installed_fixture_app_imports_outside_repository(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import fixture_app.app"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_fixture_serve_rejects_external_bind_host(monkeypatch) -> None:
    def unexpected_run(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"uvicorn must not start: {args!r} {kwargs!r}")

    monkeypatch.setattr("uvicorn.run", unexpected_run)

    result = runner.invoke(app, ["fixture", "serve", "--host", "0.0.0.0"])

    assert result.exit_code == 1
    assert "loopback" in result.stdout


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
        result: ExperimentResult,
        *,
        output: Path,
        runtime: object,
        selected_specs: object,
    ) -> tuple[Path, Path]:
        completed.update(
            result=result,
            output=output,
            runtime=runtime,
            selected_specs=selected_specs,
        )
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
    assert completed["selected_specs"]
    assert "evaluation summary:" in result.stdout
    assert "report generated:" in result.stdout


def test_complete_experiment_always_reports_all_failed_specs(tmp_path: Path) -> None:
    loaded = load_project(DEMO_PROJECT)
    definition = next(
        item for item in loaded.project.experiments if item.id == "core-pair"
    )
    spec = expand_experiment(
        ExperimentContext(definition, loaded.project, loaded.config_digest)
    )[0]
    failure = ExperimentFailure(
        run_id=spec.run_id,
        error_type="ProviderFailure",
        message="browser capture failed",
        spec=spec,
    )

    summary_path, report_path = cli._complete_experiment(
        ExperimentResult(specs=(spec,), results=(), failures=(failure,)),
        output=tmp_path,
        runtime=loaded.runtime,
    )

    assert report_path is not None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    html = report_path.read_text(encoding="utf-8")
    assert summary["failures"] == [
        {
            "application_version_id": spec.application_version.id,
            "error_type": "ProviderFailure",
            "model_trial": spec.model_trial,
            "persona_id": spec.persona.id,
            "policy": spec.policy.value,
            "reason": "browser capture failed",
            "run_id": spec.run_id,
            "scenario_id": spec.scenario.id,
            "seed": spec.seed,
            "stage": "execution",
            "terminal_state": "failed",
        }
    ]
    assert report_path.is_file()
    assert "browser capture failed" in html
    assert next(iter(spec.scenario.fixture_inputs.values.values())) not in html
