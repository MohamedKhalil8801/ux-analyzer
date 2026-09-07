from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

import ux_analyzer.cli as cli
from ux_analyzer.adapters.openai import CodexStructuredClient, OpenAICompatibleSettings
from ux_analyzer.application.experiment import (
    ExperimentContext,
    ExperimentFailure,
    ExperimentResult,
    expand_experiment,
)
from ux_analyzer.cli import app
from ux_analyzer.config.loader import load_project
from ux_analyzer.domain.attention import AttentionState
from ux_analyzer.domain.benchmark import Budget, ExperimentPolicy, FixtureInputs
from ux_analyzer.domain.interface import BoundingBox, ElementSnapshot, ViewportSnapshot
from ux_analyzer.domain.synthesis import (
    CANONICAL_SYNTHESIS_ROLES,
    SynthesisAttempt,
    SynthesisRoleReceipt,
    SynthesisStatus,
)
from ux_analyzer.ports.models import ModelCallRecord, TokenUsage
from ux_analyzer.ports.observation import (
    ObservationCapture,
    SessionHandle,
    ViewportSize,
)
from ux_analyzer.ports.observation import TestAccountId as AccountId
from ux_analyzer.providers.attention_policy import ProgressiveAttentionPolicy
from ux_analyzer.providers.full_list_policy import FullListPolicy

DEMO_PROJECT = Path(__file__).parents[3] / "benchmarks" / "demo" / "project.yaml"
runner = CliRunner()


def _live_run_spec(tmp_path: Path):
    source = Path(__file__).parents[2] / "fixtures" / "config" / "minimal-project.yaml"
    project = yaml.safe_load(source.read_text(encoding="utf-8"))
    assert isinstance(project, dict)
    project["applications"][0]["versions"] = [
        {
            "id": "portfolio-live",
            "kind": "live",
            "label": "Production",
            "start_url": "https://portfolio.example/work",
            "allowed_origins": ["https://fonts.example"],
            "navigation_settle_ms": 2000,
            "action_settle_ms": 900,
        }
    ]
    project["scenarios"][0]["application_version_ids"] = ["portfolio-live"]
    project["scenarios"][0]["fixture_inputs"] = {}
    project["scenarios"][0]["verifier"] = {
        "type": "visible-result",
        "text": "Frontend Engineer",
    }
    project["scenarios"][0]["evaluation_target"]["labels_by_version"] = {"live": "Work"}
    project["experiments"][0]["application_version_ids"] = ["portfolio-live"]
    project_path = tmp_path / "live-project.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    loaded = load_project(project_path)
    definition = loaded.project.experiments[0]
    specs = expand_experiment(
        ExperimentContext(definition, loaded.project, loaded.config_digest)
    )
    assert len(specs) == 1
    return specs[0], loaded


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
    manifest_prominence_provider_id: str | None = None,
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
        "prominence_provider_id": spec.prominence_provider_id,
        "comparison_valid": True,
        "prominence_fallback": False,
        "prominence_fallback_reason": None,
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
    manifest["prominence_provider_id"] = (
        spec.prominence_provider_id
        if manifest_prominence_provider_id is None
        else manifest_prominence_provider_id
    )
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
                "ux_sample_valid": True,
                "ux_sample_invalid_reason": None,
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


def _write_synthesis_finalized_bundle(root: Path, spec: Any) -> None:
    _write_resumable_metrics_bundle(root, spec, cost=1.0)
    identity = {
        "run_id": spec.run_id,
        "seed": spec.seed,
        "model_trial": spec.model_trial,
        "policy": spec.policy.value,
        "prominence_provider_id": spec.prominence_provider_id,
        "config_digest": spec.config_digest,
        "scenario_id": spec.scenario.id,
        "application_version_id": spec.application_version.id,
        "persona_id": spec.persona.id,
    }
    raw_spec = {
        **identity,
        "scenario": {
            "id": spec.scenario.id,
            "name": spec.scenario.name,
            "goal": spec.scenario.goal,
        },
        "application_version": {
            "id": spec.application_version.id,
            "label": spec.application_version.label,
        },
        "persona": {"id": spec.persona.id, "name": spec.persona.name},
    }
    run = root / "runs" / spec.run_id
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(identity)
    _write_json(run / "manifest.json", manifest)
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    result["state"] = {"spec": raw_spec}
    result["metrics"].update(identity)
    _write_json(run / "result.json", result)
    _write_json(root / "experiment.json", {"run_metrics": [identity]})
    (run / "checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(run).as_posix()}\n"
            for path in sorted(run.rglob("*"))
            if path.is_file() and path.name != "checksums.sha256"
        ),
        encoding="utf-8",
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


def test_env_check_reports_api_mode_without_secret_value(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    secret = "super-secret-api-key"
    monkeypatch.setenv("UXA_LLM_MODE", "api")
    monkeypatch.setenv("UXA_LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("UXA_LLM_API_KEY", secret)
    monkeypatch.setenv("UXA_SCENT_MODEL", "scent-model")
    monkeypatch.setenv("UXA_COGNITIVE_MODEL", "cognitive-model")
    monkeypatch.setenv("UXA_REPORT_MODEL", "report-model")
    monkeypatch.setenv("UXA_LLM_TIMEOUT_SECONDS", "30")

    result = runner.invoke(
        app,
        ["validate", str(DEMO_PROJECT), "--check-env"],
    )

    assert result.exit_code == 0, result.stdout
    assert "endpoint origin: https://llm.example.test" in result.stdout
    assert "API key present" in result.stdout
    assert secret not in result.stdout


def test_env_check_reports_codex_mode_without_api_key_claim(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UXA_LLM_MODE", "codex")
    monkeypatch.delenv("UXA_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("UXA_LLM_API_KEY", raising=False)
    monkeypatch.setenv("UXA_SCENT_MODEL", "scent-model")
    monkeypatch.setenv("UXA_COGNITIVE_MODEL", "cognitive-model")
    monkeypatch.setenv("UXA_REPORT_MODEL", "report-model")

    result = runner.invoke(
        app,
        ["validate", str(DEMO_PROJECT), "--check-env"],
    )

    assert result.exit_code == 0, result.stdout
    assert "mode: codex" in result.stdout
    assert "scent and cognitive models configured" in result.stdout
    assert "API key present" not in result.stdout


def test_validate_check_env_requires_report_model_for_configured_synthesis(
    monkeypatch, tmp_path: Path
) -> None:
    project = yaml.safe_load(DEMO_PROJECT.read_text(encoding="utf-8"))
    assert isinstance(project, dict)
    project.setdefault("evaluation", {})["report_synthesis"] = {"enabled": True}
    project_path = tmp_path / "configured-project.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UXA_LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("UXA_LLM_API_KEY", "super-secret-api-key")
    monkeypatch.setenv("UXA_SCENT_MODEL", "scent-model")
    monkeypatch.setenv("UXA_COGNITIVE_MODEL", "cognitive-model")
    monkeypatch.setenv("UXA_LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.delenv("UXA_REPORT_MODEL", raising=False)

    result = runner.invoke(app, ["validate", str(project_path), "--check-env"])

    assert result.exit_code == 1
    assert "UXA_REPORT_MODEL" in result.stdout
    assert "super-secret-api-key" not in result.stdout


def test_run_check_env_requires_report_model_unless_synthesis_is_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    project = yaml.safe_load(DEMO_PROJECT.read_text(encoding="utf-8"))
    assert isinstance(project, dict)
    project.setdefault("evaluation", {})["report_synthesis"] = {"enabled": True}
    project_path = tmp_path / "configured-project.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UXA_LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("UXA_LLM_API_KEY", "super-secret-api-key")
    monkeypatch.setenv("UXA_SCENT_MODEL", "scent-model")
    monkeypatch.setenv("UXA_COGNITIVE_MODEL", "cognitive-model")
    monkeypatch.setenv("UXA_LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.delenv("UXA_REPORT_MODEL", raising=False)

    checked = runner.invoke(
        app,
        [
            "run",
            str(project_path),
            "--experiment",
            "core-pair",
            "--dry-run",
            "--check-env",
        ],
    )
    disabled = runner.invoke(
        app,
        [
            "run",
            str(project_path),
            "--experiment",
            "core-pair",
            "--dry-run",
            "--check-env",
            "--no-synthesis",
        ],
    )

    assert checked.exit_code == 1
    assert "UXA_REPORT_MODEL" in checked.stdout
    assert disabled.exit_code == 0, disabled.stdout


def test_run_dry_run_prints_matrix_model_calls_and_serial_default(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("UXA_LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("UXA_LLM_API_KEY", "super-secret-api-key")
    monkeypatch.setenv("UXA_SCENT_MODEL", "scent-model")
    monkeypatch.setenv("UXA_COGNITIVE_MODEL", "cognitive-model")
    monkeypatch.setenv("UXA_LLM_TIMEOUT_SECONDS", "30")

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
    assert "run specs: 8" in result.stdout
    assert "model calls for one attention cycle: 16" in result.stdout
    assert "maximum logical model calls: 512" in result.stdout
    assert "deterministic seed repetitions suppressed: 0" in result.stdout
    assert "overall run timeout: none" in result.stdout
    assert "super-secret-api-key" not in result.stdout


def test_baseline_model_trials_dry_run_counts_trials_without_negative_suppression(
    tmp_path: Path,
) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            str(DEMO_PROJECT),
            "--experiment",
            "baseline-model-trials",
            "--output",
            str(tmp_path),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "run specs: 8" in result.stdout
    assert "deterministic seed repetitions suppressed: 0" in result.stdout
    assert result.stdout.count("/full-list: 2 runs") == 2
    assert result.stdout.count("/progressive-prominence-scent: 2 runs") == 2


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
        from ux_analyzer.domain.run import RunState, VerifiedSuccess
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
                state=RunState.initial(selected[-1]),
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


def test_resume_completion_excludes_finalized_bundle_with_wrong_provider(
    tmp_path: Path,
) -> None:
    project = yaml.safe_load(DEMO_PROJECT.read_text(encoding="utf-8"))
    assert isinstance(project, dict)
    experiment = next(
        item for item in project["experiments"] if item["id"] == "focused-validation"
    )
    experiment["prominence_provider_ids"] = ["heuristic", "foveacast"]
    project_path = tmp_path / "provider-axis.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    matrix = cli._resolve_matrix_or_exit(
        project_path, "focused-validation", run_count=1, policies=()
    )
    foveacast_spec = next(
        spec for spec in matrix.specs if spec.prominence_provider_id == "foveacast"
    )
    _write_resumable_metrics_bundle(
        tmp_path,
        foveacast_spec,
        cost=1,
        manifest_prominence_provider_id="heuristic",
    )

    summary_path, _ = cli._complete_experiment(
        ExperimentResult(specs=(foveacast_spec,), results=(), failures=()),
        output=tmp_path,
        runtime=matrix.loaded.runtime,
        selected_specs=(foveacast_spec,),
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["run_metrics"] == []
    assert summary["variant_comparisons"] == []


@pytest.mark.parametrize(
    ("manifest_provider_id", "returned_provider_id"),
    (("heuristic", "foveacast"), ("foveacast", "heuristic")),
)
def test_resume_completion_rejects_mismatched_manifest_or_returned_provider(
    tmp_path: Path,
    manifest_provider_id: str,
    returned_provider_id: str,
) -> None:
    from pydantic import TypeAdapter

    from ux_analyzer.application.evaluation import RunMetrics
    from ux_analyzer.application.run_agent import RunResult
    from ux_analyzer.domain.run import RunState, VerifiedSuccess
    from ux_analyzer.ports.verification import VerificationResult

    project = yaml.safe_load(DEMO_PROJECT.read_text(encoding="utf-8"))
    assert isinstance(project, dict)
    experiment = next(
        item for item in project["experiments"] if item["id"] == "focused-validation"
    )
    experiment["prominence_provider_ids"] = ["heuristic", "foveacast"]
    project_path = tmp_path / "provider-axis.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    matrix = cli._resolve_matrix_or_exit(
        project_path, "focused-validation", run_count=1, policies=()
    )
    foveacast_spec = next(
        spec for spec in matrix.specs if spec.prominence_provider_id == "foveacast"
    )
    _write_resumable_metrics_bundle(
        tmp_path,
        foveacast_spec,
        cost=1,
        manifest_prominence_provider_id=manifest_provider_id,
    )
    persisted = json.loads(
        (tmp_path / "runs" / foveacast_spec.run_id / "result.json").read_text()
    )
    returned_spec = replace(foveacast_spec, prominence_provider_id=returned_provider_id)
    returned = RunResult(
        run_id=foveacast_spec.run_id,
        outcome=VerifiedSuccess(),
        verification=VerificationResult(verified=True),
        agent_claimed_success=True,
        state=RunState.initial(returned_spec),
        metrics=TypeAdapter(RunMetrics).validate_python(persisted["metrics"]),
    )

    summary_path, _ = cli._complete_experiment(
        ExperimentResult(specs=(foveacast_spec,), results=(returned,), failures=()),
        output=tmp_path,
        runtime=matrix.loaded.runtime,
        selected_specs=(foveacast_spec,),
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["run_metrics"] == []
    assert summary["variant_comparisons"] == []


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
    assert "run specs: 12" in result.stdout
    assert "deterministic seed repetitions suppressed: 4" in result.stdout


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


def test_provider_axis_dry_run_expands_eight_cells_and_prints_provider(
    tmp_path: Path,
) -> None:
    project = yaml.safe_load(DEMO_PROJECT.read_text(encoding="utf-8"))
    assert isinstance(project, dict)
    experiment = next(
        item for item in project["experiments"] if item["id"] == "focused-validation"
    )
    experiment["prominence_provider_ids"] = ["heuristic", "foveacast"]
    project_path = tmp_path / "provider-axis.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "run",
            str(project_path),
            "--experiment",
            "focused-validation",
            "--output",
            str(tmp_path / "output"),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "prominence providers: heuristic, foveacast" in result.stdout
    assert "run specs: 8" in result.stdout
    assert "prominence-provider=foveacast" in result.stdout


@pytest.mark.parametrize(
    ("provider_id", "expected_provider_version"),
    (
        ("heuristic", "heuristic-project-v2"),
        ("foveacast", "foveacast-prominence-v1"),
    ),
)
def test_bundle_manifest_records_active_provider_and_prompt_provenance(
    tmp_path: Path,
    provider_id: str,
    expected_provider_version: str,
) -> None:
    project = yaml.safe_load(DEMO_PROJECT.read_text(encoding="utf-8"))
    assert isinstance(project, dict)
    project["providers"]["prominence"]["version"] = "heuristic-project-v2"
    experiment = next(
        item for item in project["experiments"] if item["id"] == "focused-validation"
    )
    experiment["prominence_provider_ids"] = [provider_id]
    project_path = tmp_path / "provider-axis.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    loaded = load_project(project_path)
    definition = next(
        item for item in loaded.project.experiments if item.id == "focused-validation"
    )
    spec = next(
        item
        for item in expand_experiment(
            ExperimentContext(definition, loaded.project, loaded.config_digest)
        )
        if item.prominence_provider_id == provider_id
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

    assert manifest["prominence_provider_id"] == provider_id
    assert manifest["endpoint_origin"] == "https://llm.example.test"
    assert manifest["provider_versions"]["models"] == "openai-compatible-v1"
    assert manifest["provider_versions"]["prominence"] == expected_provider_version
    assert manifest["prompt_versions"]["cognitive"] == "cognitive-v3"
    cognitive_manifest = next(
        item for item in manifest["provider_manifests"] if item["role"] == "cognitive"
    )
    assert cognitive_manifest["prompt_version"] == "cognitive-v3"
    assert cognitive_manifest["schema_version"] == "cognitive-v1"
    assert all(
        item["provider_id"] == "openai-compatible-structured"
        and item["version"] == "openai-compatible-v1"
        and item["endpoint_origin"] == "https://llm.example.test"
        for item in manifest["provider_manifests"]
        if item["role"] != "prominence"
    )


def test_codex_bundle_manifest_uses_selected_client_provider_metadata(
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
    settings = OpenAICompatibleSettings(
        base_url="",
        api_key="",
        scent_model="scent-model",
        cognitive_model="cognitive-model",
        mode="codex",
    )
    client = CodexStructuredClient(settings)

    writer = cli._BundleFactory(
        tmp_path, settings, loaded.runtime, client=client
    ).start(spec)
    manifest = json.loads((writer.staging_path / "manifest.json").read_text())
    writer.abort("test complete")

    assert manifest["endpoint_origin"] == "codex-cli"
    assert manifest["provider_versions"]["models"] == "codex-cli"
    assert {
        (item["role"], item["provider_id"], item["version"], item["endpoint_origin"])
        for item in manifest["provider_manifests"]
    } == {
        ("cognitive", "codex-cli", "codex-cli", "codex-cli"),
        ("coarse-scent", "codex-cli", "codex-cli", "codex-cli"),
        ("full-scent", "codex-cli", "codex-cli", "codex-cli"),
    }


def test_resume_trust_requires_matching_prominence_provider_id(tmp_path: Path) -> None:
    project = yaml.safe_load(DEMO_PROJECT.read_text(encoding="utf-8"))
    assert isinstance(project, dict)
    experiment = next(
        item for item in project["experiments"] if item["id"] == "focused-validation"
    )
    experiment["prominence_provider_ids"] = ["heuristic", "foveacast"]
    project_path = tmp_path / "provider-axis.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    matrix = cli._resolve_matrix_or_exit(
        project_path, "focused-validation", run_count=1, policies=()
    )
    foveacast_spec = next(
        spec for spec in matrix.specs if spec.prominence_provider_id == "foveacast"
    )
    _write_resumable_metrics_bundle(
        tmp_path,
        foveacast_spec,
        cost=1,
        manifest_prominence_provider_id="heuristic",
    )

    resumed, store = cli._prepare_resumed_matrix(matrix, tmp_path)

    assert foveacast_spec.run_id in {spec.run_id for spec in resumed.specs}
    assert foveacast_spec.run_id not in store.state.finalized_run_ids


def test_resume_report_groups_variant_comparisons_by_prominence_provider(
    tmp_path: Path,
) -> None:
    project = yaml.safe_load(DEMO_PROJECT.read_text(encoding="utf-8"))
    assert isinstance(project, dict)
    experiment = next(
        item for item in project["experiments"] if item["id"] == "focused-validation"
    )
    experiment["prominence_provider_ids"] = ["heuristic", "foveacast"]
    project_path = tmp_path / "provider-axis.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    matrix = cli._resolve_matrix_or_exit(
        project_path, "focused-validation", run_count=1, policies=()
    )
    for index, spec in enumerate(matrix.specs):
        _write_resumable_metrics_bundle(tmp_path, spec, cost=float(index + 1))

    summary_path, _ = cli._complete_experiment(
        ExperimentResult(specs=matrix.specs, results=(), failures=()),
        output=tmp_path,
        runtime=matrix.loaded.runtime,
        selected_specs=matrix.specs,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    comparisons = summary["variant_comparisons"]
    assert len(comparisons) == 4
    assert {item["prominence_provider_id"] for item in comparisons} == {
        "heuristic",
        "foveacast",
    }
    assert len(summary["cell_aggregates"]) == 8
    assert {item["prominence_provider_id"] for item in summary["cell_aggregates"]} == {
        "heuristic",
        "foveacast",
    }


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


def test_session_config_uses_live_start_url_and_settle_timings(tmp_path: Path) -> None:
    spec, _ = _live_run_spec(tmp_path)

    config = cli._session_config(
        spec,
        output=tmp_path,
        fixture_origin="http://127.0.0.1:8000",
    )

    assert config.start_url == "https://portfolio.example/work"
    assert config.navigation_origins == ("https://portfolio.example",)
    assert config.resource_origins == ("https://fonts.example",)
    assert config.navigation_settle_ms == 2000
    assert config.action_settle_ms == 900


def test_live_agent_uses_visible_verification_without_fixture_state_client(
    tmp_path: Path,
) -> None:
    spec, loaded = _live_run_spec(tmp_path)

    class FakeClient:
        endpoint_origin = "https://llm.example.test"
        records: tuple[object, ...] = ()

    settings = OpenAICompatibleSettings(
        base_url="https://llm.example.test/v1",
        api_key="secret",
        scent_model="scent-model",
        cognitive_model="cognitive-model",
    )
    agent = cli._build_agent(
        spec,
        adapter=object(),
        client=FakeClient(),
        output=tmp_path,
        fixture_origin="http://127.0.0.1:8000",
        settings=settings,
        runtime=loaded.runtime,
    )

    assert agent.verifier.spec.type == "visible-result"
    assert agent.verifier._fixture_state_client is None
    assert agent.observation_provider._fixture_control_enabled is False


def test_live_agent_drops_fixture_values_and_keys_defensively(tmp_path: Path) -> None:
    spec, loaded = _live_run_spec(tmp_path)
    spec = replace(
        spec,
        scenario=replace(
            spec.scenario,
            fixture_inputs=FixtureInputs(values={"unexpected": "secret"}),
        ),
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
    agent = cli._build_agent(
        spec,
        adapter=object(),
        client=FakeClient(),
        output=tmp_path,
        fixture_origin="http://127.0.0.1:8000",
        settings=settings,
        runtime=loaded.runtime,
    )

    assert agent.observation_provider._fixture_inputs == {}
    assert agent.cognitive_agent.fixture_keys == ()


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
    assert manifest["prompt_versions"] == {"cognitive": "cognitive-v3"}
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


@pytest.mark.asyncio
async def test_fixture_provider_extracts_against_current_page_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted: list[dict[str, object]] = []
    page = object()

    class FakeAdapter:
        async def capture(self, session: SessionHandle) -> ObservationCapture:
            return ObservationCapture(
                session_id=session.session_id,
                viewport_id="viewport-1",
                url="http://fixture.test/app",
                title="Fixture",
                viewport=session.viewport,
                screenshot=b"capture-artifact",
            )

        def page_for_testing(self, session: SessionHandle) -> object:
            del session
            return page

    async def fake_capture_snapshot_with_diagnostics(
        extraction_page: object,
        viewport_id: str,
        **kwargs: object,
    ) -> SimpleNamespace:
        extracted.append(kwargs)
        assert extraction_page is page
        return SimpleNamespace(
            snapshot=ViewportSnapshot(id=viewport_id, elements=()),
            screenshot=b"extractor-artifact",
        )

    monkeypatch.setattr(
        cli,
        "capture_snapshot_with_diagnostics",
        fake_capture_snapshot_with_diagnostics,
    )
    provider = cli._FixtureObservationProvider(
        FakeAdapter(),  # type: ignore[arg-type]
        "http://fixture.test",
        {},
    )
    session = SessionHandle(
        session_id="run-1",
        test_account_id=AccountId("test-run-1"),
        viewport=ViewportSize(900, 700),
        trace_path=Path("trace.zip"),
        blocked_events=[],
    )

    capture = await provider.capture(session)

    assert extracted == [{}]
    assert capture.snapshot is not None
    assert capture.snapshot.id == "viewport-1"
    assert capture.screenshot == b"extractor-artifact"


@pytest.mark.asyncio
async def test_live_provider_reloads_browser_without_fixture_control_calls(
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

    class ForbiddenClient:
        async def post(self, url: str, json: object) -> None:
            raise AssertionError(f"unexpected fixture POST: {url} {json}")

        async def delete(self, url: str) -> None:
            raise AssertionError(f"unexpected fixture DELETE: {url}")

    provider = cli._FixtureObservationProvider(
        FakeAdapter(),  # type: ignore[arg-type]
        "http://127.0.0.1:8000",
        {},
        ForbiddenClient(),  # type: ignore[arg-type]
        fixture_control_enabled=False,
    )
    session = SessionHandle(
        session_id="live-run",
        test_account_id=AccountId("test-live-run"),
        viewport=ViewportSize(900, 700),
        trace_path=tmp_path / "trace.zip",
        blocked_events=[],
    )

    await provider.reset(session)
    await provider.end_session(session)

    assert events == ["browser-reset", "browser-end"]


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


def test_completion_exposes_summary_and_report_render_phases(
    tmp_path: Path,
) -> None:
    summary = cli._write_experiment_summary(
        ExperimentResult(specs=(), results=(), failures=()),
        output=tmp_path,
        selected_specs=(),
    )

    assert summary == tmp_path / "experiment.json"
    assert summary.is_file()


def test_persist_synthesis_attempt_retries_sequence_collision(
    monkeypatch, tmp_path: Path
) -> None:
    attempt = SynthesisAttempt(
        attempt_id="placeholder",
        status=SynthesisStatus.NO_ISSUES,
    )
    corpus = object()
    ids = iter(("attempt-1", "attempt-2"))
    writes: list[str] = []

    class FakeStore:
        def __init__(self, output: Path) -> None:
            assert output == tmp_path

        def write_attempt(self, value: SynthesisAttempt, received: object) -> Path:
            assert received is corpus
            writes.append(value.attempt_id)
            if len(writes) == 1:
                raise cli.SynthesisArtifactError(
                    "attempt sequence already exists for creation token"
                )
            return tmp_path / value.attempt_id

    monkeypatch.setattr(cli, "SynthesisArtifactStore", FakeStore)
    monkeypatch.setattr(cli, "_synthesis_corpus", lambda **kwargs: corpus)
    monkeypatch.setattr(cli, "_storage_attempt_id", lambda attempt, store: next(ids))

    result = cli._persist_synthesis_attempt(
        attempt,
        result=ExperimentResult(specs=(), results=(), failures=()),
        output=tmp_path,
        loaded=SimpleNamespace(),
    )

    assert result == tmp_path / "attempt-2"
    assert writes == ["attempt-1", "attempt-2"]


def test_storage_attempt_id_uses_global_same_second_sequence() -> None:
    first = SynthesisAttempt(
        attempt_id="20260810T120000Z-aaaaaaaaaaaa-1",
        status=SynthesisStatus.NO_ISSUES,
    )
    second = SynthesisAttempt(
        attempt_id="placeholder",
        status=SynthesisStatus.NO_ISSUES,
        corpus_digest="b" * 64,
        created_at="2026-08-10T12:00:00+00:00",
    )
    store = SimpleNamespace(attempts=(first,))

    assert cli._storage_attempt_id(second, store) == (
        "2026-08-10T120000Z-bbbbbbbbbbbb-2"
    )


def test_configured_run_synthesizes_after_summary_before_render(
    monkeypatch, tmp_path: Path
) -> None:
    project = yaml.safe_load(DEMO_PROJECT.read_text(encoding="utf-8"))
    assert isinstance(project, dict)
    project.setdefault("evaluation", {})["report_synthesis"] = {"enabled": True}
    project_path = tmp_path / "configured-project.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")

    monkeypatch.setenv("UXA_LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("UXA_LLM_API_KEY", "super-secret-api-key")
    monkeypatch.setenv("UXA_SCENT_MODEL", "scent-model")
    monkeypatch.setenv("UXA_COGNITIVE_MODEL", "cognitive-model")
    monkeypatch.setenv("UXA_LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("UXA_REPORT_MODEL", "report-model")
    calls: list[str] = []

    async def fake_execute_matrix(*args: object, **kwargs: object) -> ExperimentResult:
        del args, kwargs
        return ExperimentResult(specs=(), results=(), failures=())

    def fake_write_summary(
        result: ExperimentResult,
        *,
        output: Path,
        selected_specs: object,
    ) -> Path:
        del result, selected_specs
        calls.append("summary")
        summary = output / "experiment.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text("{}", encoding="utf-8")
        return summary

    async def fake_synthesis(**kwargs: object) -> object:
        del kwargs
        calls.append("synthesis")
        return object()

    def fake_persist(*args: object, **kwargs: object) -> None:
        del args, kwargs
        calls.append("persist")

    def fake_render(*, output: Path) -> Path:
        calls.append("render")
        report = output / "report.html"
        report.write_text("<html></html>", encoding="utf-8")
        return report

    monkeypatch.setattr(cli, "_execute_matrix", fake_execute_matrix)
    monkeypatch.setattr(cli, "_write_experiment_summary", fake_write_summary)
    monkeypatch.setattr(cli, "_run_report_synthesis", fake_synthesis)
    monkeypatch.setattr(cli, "_persist_synthesis_attempt", fake_persist)
    monkeypatch.setattr(cli, "_render_completed_report", fake_render)

    result = runner.invoke(
        app,
        [
            "run",
            str(project_path),
            "--experiment",
            "core-pair",
            "--run-count",
            "1",
            "--output",
            str(tmp_path / "output"),
        ],
    )

    assert result.exit_code == 0
    assert calls == ["summary", "synthesis", "persist", "render"]


@pytest.mark.parametrize("pending_count", (1, 0))
def test_resume_synthesis_uses_all_finalized_selected_specs(
    monkeypatch,
    tmp_path: Path,
    pending_count: int,
) -> None:
    project = yaml.safe_load(DEMO_PROJECT.read_text(encoding="utf-8"))
    assert isinstance(project, dict)
    project.setdefault("evaluation", {})["report_synthesis"] = {"enabled": True}
    project_path = tmp_path / f"resume-project-{pending_count}.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    selected_matrix = cli._resolve_matrix_or_exit(
        project_path,
        "core-pair",
        run_count=1,
        policies=(),
    )
    all_finalized = ExperimentResult(
        specs=selected_matrix.specs,
        results=(),
        failures=(),
    )
    captured: dict[str, object] = {}

    monkeypatch.setenv("UXA_LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("UXA_LLM_API_KEY", "super-secret-api-key")
    monkeypatch.setenv("UXA_SCENT_MODEL", "scent-model")
    monkeypatch.setenv("UXA_COGNITIVE_MODEL", "cognitive-model")
    monkeypatch.setenv("UXA_LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("UXA_REPORT_MODEL", "report-model")

    def fake_prepare(matrix: object, output: Path) -> tuple[object, object]:
        del output
        assert matrix == selected_matrix
        pending_matrix = replace(matrix, specs=matrix.specs[:pending_count])
        captured["pending_specs"] = pending_matrix.specs
        checkpoint = SimpleNamespace(
            state=SimpleNamespace(
                finalized_run_ids=(),
                interrupted_run_ids=(),
                pending_run_ids=tuple(spec.run_id for spec in pending_matrix.specs),
            )
        )
        return pending_matrix, checkpoint

    async def fake_execute_matrix(*args: object, **kwargs: object) -> ExperimentResult:
        del kwargs
        pending_matrix = args[0]
        return ExperimentResult(specs=pending_matrix.specs, results=(), failures=())

    def fake_finalized(matrix: object, output: Path) -> ExperimentResult:
        del output
        captured["finalized_matrix"] = matrix
        return all_finalized

    def fake_write_summary(
        result: ExperimentResult,
        *,
        output: Path,
        selected_specs: object,
    ) -> Path:
        del result, selected_specs
        summary = output / "experiment.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text("{}", encoding="utf-8")
        return summary

    async def fake_synthesis(**kwargs: object) -> object:
        captured["synthesis_result"] = kwargs["result"]
        return object()

    def fake_persist(*args: object, **kwargs: object) -> None:
        del args
        captured["persist_result"] = kwargs["result"]

    def fake_render(*, output: Path) -> Path:
        report = output / "report.html"
        report.write_text("<html></html>", encoding="utf-8")
        return report

    monkeypatch.setattr(cli, "_prepare_resumed_matrix", fake_prepare)
    monkeypatch.setattr(cli, "_execute_matrix", fake_execute_matrix)
    monkeypatch.setattr(cli, "_finalized_experiment_result", fake_finalized)
    monkeypatch.setattr(cli, "_write_experiment_summary", fake_write_summary)
    monkeypatch.setattr(cli, "_run_report_synthesis", fake_synthesis)
    monkeypatch.setattr(cli, "_persist_synthesis_attempt", fake_persist)
    monkeypatch.setattr(cli, "_render_completed_report", fake_render)

    result = runner.invoke(
        app,
        [
            "run",
            str(project_path),
            "--experiment",
            "core-pair",
            "--run-count",
            "1",
            "--output",
            str(tmp_path / "output"),
            "--resume",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["finalized_matrix"] == selected_matrix
    assert captured["synthesis_result"] is all_finalized
    assert captured["persist_result"] is all_finalized


def test_configured_synthesis_bounds_reach_service_without_clamping(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project = yaml.safe_load(DEMO_PROJECT.read_text(encoding="utf-8"))
    assert isinstance(project, dict)
    project.setdefault("evaluation", {})["report_synthesis"] = {
        "enabled": True,
        "max_retrieval_rounds": 5,
        "max_adjudication_revisions": 2,
        "max_final_verifications": 2,
    }
    project_path = tmp_path / "configured-project.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    loaded = load_project(project_path)
    output = tmp_path / "output"
    output.mkdir()
    settings = OpenAICompatibleSettings(
        base_url="https://llm.example.test/v1",
        api_key="secret",
        scent_model="scent-model",
        cognitive_model="cognitive-model",
        report_model="report-model",
    )
    captured: dict[str, object] = {}

    class RecordingService:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        async def synthesize(self, corpus: object) -> object:
            del corpus
            return object()

    monkeypatch.setattr(
        cli, "create_structured_model_client", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(cli, "ReportSynthesisService", RecordingService)
    monkeypatch.setattr(cli, "_synthesis_corpus", lambda **kwargs: object())

    asyncio.run(
        cli._run_report_synthesis(
            result=ExperimentResult(specs=(), results=(), failures=()),
            output=output,
            loaded=loaded,
            settings=settings,
        )
    )

    assert captured["max_retrieval_rounds"] == 5
    assert captured["max_adjudication_revisions"] == 2
    assert captured["max_final_verifications"] == 2


def test_no_synthesis_skips_report_model_and_synthesis(
    monkeypatch, tmp_path: Path
) -> None:
    project = yaml.safe_load(DEMO_PROJECT.read_text(encoding="utf-8"))
    assert isinstance(project, dict)
    project.setdefault("evaluation", {})["report_synthesis"] = {"enabled": True}
    project_path = tmp_path / "configured-project.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")

    monkeypatch.setenv("UXA_LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("UXA_LLM_API_KEY", "super-secret-api-key")
    monkeypatch.setenv("UXA_SCENT_MODEL", "scent-model")
    monkeypatch.setenv("UXA_COGNITIVE_MODEL", "cognitive-model")
    monkeypatch.setenv("UXA_LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.delenv("UXA_REPORT_MODEL", raising=False)
    calls: list[str] = []

    async def fake_execute_matrix(*args: object, **kwargs: object) -> ExperimentResult:
        del args, kwargs
        return ExperimentResult(specs=(), results=(), failures=())

    def fake_write_summary(
        result: ExperimentResult,
        *,
        output: Path,
        selected_specs: object,
    ) -> Path:
        del result, selected_specs
        calls.append("summary")
        summary = output / "experiment.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text("{}", encoding="utf-8")
        return summary

    async def unexpected_synthesis(**kwargs: object) -> object:
        del kwargs
        raise AssertionError("synthesis must be disabled")

    def fake_render(*, output: Path) -> Path:
        calls.append("render")
        report = output / "report.html"
        report.write_text("<html></html>", encoding="utf-8")
        return report

    monkeypatch.setattr(cli, "_execute_matrix", fake_execute_matrix)
    monkeypatch.setattr(cli, "_write_experiment_summary", fake_write_summary)
    monkeypatch.setattr(cli, "_run_report_synthesis", unexpected_synthesis)
    monkeypatch.setattr(cli, "_render_completed_report", fake_render)

    result = runner.invoke(
        app,
        [
            "run",
            str(project_path),
            "--experiment",
            "core-pair",
            "--run-count",
            "1",
            "--output",
            str(tmp_path / "output"),
            "--no-synthesis",
        ],
    )

    assert result.exit_code == 0
    assert calls == ["summary", "render"]


def test_synthesis_missing_report_model_keeps_completed_run_successful(
    monkeypatch, tmp_path: Path
) -> None:
    project = yaml.safe_load(DEMO_PROJECT.read_text(encoding="utf-8"))
    assert isinstance(project, dict)
    project.setdefault("evaluation", {})["report_synthesis"] = {"enabled": True}
    project_path = tmp_path / "configured-project.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UXA_LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("UXA_LLM_API_KEY", "super-secret-api-key")
    monkeypatch.setenv("UXA_SCENT_MODEL", "scent-model")
    monkeypatch.setenv("UXA_COGNITIVE_MODEL", "cognitive-model")
    monkeypatch.setenv("UXA_LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.delenv("UXA_REPORT_MODEL", raising=False)
    calls: list[str] = []

    async def fake_execute_matrix(*args: object, **kwargs: object) -> ExperimentResult:
        del args, kwargs
        return ExperimentResult(specs=(), results=(), failures=())

    def fake_write_summary(
        result: ExperimentResult,
        *,
        output: Path,
        selected_specs: object,
    ) -> Path:
        del result, selected_specs
        calls.append("summary")
        summary = output / "experiment.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text("{}", encoding="utf-8")
        return summary

    persist_unavailable = cli._persist_unavailable_synthesis

    def record_unavailable(**kwargs: object) -> None:
        calls.append("unavailable")
        persist_unavailable(**kwargs)

    def fake_render(*, output: Path) -> Path:
        calls.append("render")
        report = output / "report.html"
        report.write_text("<html></html>", encoding="utf-8")
        return report

    monkeypatch.setattr(cli, "_execute_matrix", fake_execute_matrix)
    monkeypatch.setattr(cli, "_write_experiment_summary", fake_write_summary)
    monkeypatch.setattr(cli, "_persist_unavailable_synthesis", record_unavailable)
    monkeypatch.setattr(cli, "_render_completed_report", fake_render)

    result = runner.invoke(
        app,
        [
            "run",
            str(project_path),
            "--experiment",
            "core-pair",
            "--run-count",
            "1",
            "--output",
            str(tmp_path / "output"),
        ],
    )

    assert result.exit_code == 0
    assert calls == ["summary", "unavailable", "render"]
    assert "warning: report synthesis unavailable" in result.stdout
    attempts = list((tmp_path / "output" / "synthesis" / "attempts").iterdir())
    assert len(attempts) == 1
    synthesis = json.loads((attempts[0] / "synthesis.json").read_text())
    assert synthesis["status"] == "unavailable"


def test_synthesis_failure_does_not_change_run_exit_code(
    monkeypatch, tmp_path: Path
) -> None:
    project = yaml.safe_load(DEMO_PROJECT.read_text(encoding="utf-8"))
    assert isinstance(project, dict)
    project.setdefault("evaluation", {})["report_synthesis"] = {"enabled": True}
    project_path = tmp_path / "configured-project.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")

    monkeypatch.setenv("UXA_LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("UXA_LLM_API_KEY", "super-secret-api-key")
    monkeypatch.setenv("UXA_SCENT_MODEL", "scent-model")
    monkeypatch.setenv("UXA_COGNITIVE_MODEL", "cognitive-model")
    monkeypatch.setenv("UXA_REPORT_MODEL", "report-model")
    monkeypatch.setenv("UXA_LLM_TIMEOUT_SECONDS", "30")
    calls: list[str] = []

    async def fake_execute_matrix(*args: object, **kwargs: object) -> ExperimentResult:
        del args, kwargs
        return ExperimentResult(specs=(), results=(), failures=())

    def fake_write_summary(
        result: ExperimentResult,
        *,
        output: Path,
        selected_specs: object,
    ) -> Path:
        del result, selected_specs
        calls.append("summary")
        summary = output / "experiment.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text("{}", encoding="utf-8")
        return summary

    async def failing_synthesis(**kwargs: object) -> object:
        del kwargs
        calls.append("synthesis")
        raise RuntimeError("provider unavailable")

    def fake_persist_unavailable(**kwargs: object) -> None:
        del kwargs
        calls.append("unavailable")

    def fake_render(*, output: Path) -> Path:
        calls.append("render")
        report = output / "report.html"
        report.write_text("<html></html>", encoding="utf-8")
        return report

    monkeypatch.setattr(cli, "_execute_matrix", fake_execute_matrix)
    monkeypatch.setattr(cli, "_write_experiment_summary", fake_write_summary)
    monkeypatch.setattr(cli, "_run_report_synthesis", failing_synthesis)
    monkeypatch.setattr(cli, "_persist_unavailable_synthesis", fake_persist_unavailable)
    monkeypatch.setattr(cli, "_render_completed_report", fake_render)

    result = runner.invoke(
        app,
        [
            "run",
            str(project_path),
            "--experiment",
            "core-pair",
            "--run-count",
            "1",
            "--output",
            str(tmp_path / "output"),
        ],
    )

    assert result.exit_code == 0
    assert calls == ["summary", "synthesis", "unavailable", "render"]
    assert "warning: report synthesis unavailable" in result.stdout


def test_report_command_does_not_call_synthesis(monkeypatch, tmp_path: Path) -> None:
    _write_run(tmp_path)

    def unexpected_synthesis(**kwargs: object) -> object:
        del kwargs
        raise AssertionError("report must remain model-free")

    monkeypatch.setattr(cli, "_run_report_synthesis", unexpected_synthesis)
    output = tmp_path / "report.html"

    result = runner.invoke(
        app,
        ["report", str(tmp_path), "--output", str(output)],
    )

    assert result.exit_code == 0
    assert output.is_file()


def test_synthesize_uses_only_finalized_output_specs(
    monkeypatch, tmp_path: Path
) -> None:
    loaded = load_project(DEMO_PROJECT)
    finalized = SimpleNamespace(
        run_id="run-finalized", prominence_provider_id="heuristic"
    )
    pending = SimpleNamespace(run_id="run-pending", prominence_provider_id="heuristic")
    matrix = SimpleNamespace(loaded=loaded, specs=(finalized, pending))

    monkeypatch.setattr(
        cli,
        "finalized_bundle_is_valid",
        lambda output, run_id, *, expected_prominence_provider_id: (
            run_id == "run-finalized"
        ),
    )

    result = cli._finalized_experiment_result(matrix, tmp_path)

    assert result.specs == (finalized,)
    assert result.results[0].run_id == "run-finalized"
    assert result.results[0].bundle_path == tmp_path / "runs" / "run-finalized"


def test_finalized_synthesis_reference_is_absolute_for_relative_output(
    monkeypatch, tmp_path: Path
) -> None:
    loaded = load_project(DEMO_PROJECT)
    finalized = SimpleNamespace(
        run_id="run-finalized", prominence_provider_id="heuristic"
    )
    matrix = SimpleNamespace(loaded=loaded, specs=(finalized,))
    monkeypatch.chdir(tmp_path.parent)
    relative_output = Path(tmp_path.name)

    monkeypatch.setattr(
        cli,
        "finalized_bundle_is_valid",
        lambda output, run_id, *, expected_prominence_provider_id: (
            run_id == "run-finalized" and expected_prominence_provider_id == "heuristic"
        ),
    )

    result = cli._finalized_experiment_result(matrix, relative_output)

    assert result.results[0].bundle_path == tmp_path / "runs" / "run-finalized"


def test_synthesize_relative_output_reaches_report_roles_and_persists_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    project = yaml.safe_load(DEMO_PROJECT.read_text(encoding="utf-8"))
    assert isinstance(project, dict)
    experiment = next(
        item for item in project["experiments"] if item["id"] == "focused-validation"
    )
    experiment["id"] = "cli-relative-synthesis"
    experiment["scenario_ids"] = ["invite-teammate"]
    experiment["application_version_ids"] = ["fixture-app-improved"]
    experiment["policies"] = ["full-list"]
    project["experiments"] = [experiment]
    project_path = tmp_path / "configured-project.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")

    matrix = cli._resolve_matrix_or_exit(
        project_path,
        "cli-relative-synthesis",
        run_count=None,
        policies=(),
    )
    assert len(matrix.specs) == 1
    output_root = tmp_path / "relative-synthesis-output"
    _write_synthesis_finalized_bundle(output_root, matrix.specs[0])

    monkeypatch.setenv("UXA_LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("UXA_LLM_API_KEY", "super-secret-api-key")
    monkeypatch.setenv("UXA_SCENT_MODEL", "scent-model")
    monkeypatch.setenv("UXA_COGNITIVE_MODEL", "cognitive-model")
    monkeypatch.setenv("UXA_REPORT_MODEL", "report-model")
    monkeypatch.setenv("UXA_LLM_TIMEOUT_SECONDS", "30")

    class FakeReportClient:
        provider_id = "fake-report-client"
        provider_version = "fake-report-v1"
        endpoint_origin = "https://llm.example.test"

        def __init__(self) -> None:
            self.calls: list[tuple[Any, str, tuple[Any, ...]]] = []
            self.records: list[ModelCallRecord] = []

        async def complete(
            self,
            schema: type[Any],
            messages: Sequence[Any],
            model: str,
            role: Any,
        ) -> object:
            self.calls.append((role, model, tuple(messages)))
            payload: dict[str, object] = {
                "complete": True,
                "evidence_requests": [],
            }
            for field_name in (
                "candidate_findings",
                "objections",
                "final_findings",
                "objection_resolutions",
            ):
                if field_name in schema.model_fields:
                    payload[field_name] = []
            response = schema.model_validate(payload)
            self.records.append(
                ModelCallRecord(
                    role=role,
                    model=model,
                    endpoint_origin=self.endpoint_origin,
                    prompt_digest=hashlib.sha256(
                        repr(tuple(messages)).encode("utf-8")
                    ).hexdigest(),
                    schema_version=schema.schema_version,
                    attempts=1,
                    latency_ms=0,
                    token_usage=TokenUsage(0, 0, 0),
                    request={},
                    response={},
                )
            )
            return response

    client = FakeReportClient()
    monkeypatch.setattr(
        cli,
        "create_structured_model_client",
        lambda *args, **kwargs: client,
    )
    captured: dict[str, object] = {}
    original_corpus = cli._synthesis_corpus

    def capture_corpus(*, result: Any, output: Path, loaded: Any) -> Any:
        captured["bundle_path"] = getattr(result.results[0], "bundle_path")
        return original_corpus(result=result, output=output, loaded=loaded)

    monkeypatch.setattr(cli, "_synthesis_corpus", capture_corpus)
    monkeypatch.chdir(tmp_path.parent)
    relative_output = Path(tmp_path.name) / output_root.name

    result = runner.invoke(
        app,
        [
            "synthesize",
            str(project_path),
            "--experiment",
            "cli-relative-synthesis",
            "--output",
            str(relative_output),
        ],
    )

    assert result.exit_code == 0, result.stdout
    bundle_path = captured["bundle_path"]
    assert isinstance(bundle_path, Path)
    assert bundle_path == output_root / "runs" / matrix.specs[0].run_id
    assert bundle_path.is_absolute()
    assert [role.value for role, _, _ in client.calls] == [
        "report-analyst",
        "report-evidence-auditor",
        "report-pattern-reviewer",
        "report-adjudicator",
    ]
    assert {model for _, model, _ in client.calls} == {"report-model"}
    attempts = list((output_root / "synthesis" / "attempts").iterdir())
    assert len(attempts) == 1
    synthesis = json.loads((attempts[0] / "synthesis.json").read_text(encoding="utf-8"))
    assert synthesis["status"] == "no-issues"
    assert (output_root / "report.html").is_file()


def test_synthesize_regeneration_retains_immutable_attempts(
    monkeypatch, tmp_path: Path
) -> None:
    loaded = load_project(DEMO_PROJECT)
    monkeypatch.setattr(
        cli,
        "_resolve_matrix_or_exit",
        lambda *args, **kwargs: SimpleNamespace(loaded=loaded, specs=()),
    )
    monkeypatch.setattr(
        cli,
        "_finalized_experiment_result",
        lambda *args, **kwargs: ExperimentResult(specs=(), results=(), failures=()),
    )
    monkeypatch.setenv("UXA_LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("UXA_LLM_API_KEY", "super-secret-api-key")
    monkeypatch.setenv("UXA_SCENT_MODEL", "scent-model")
    monkeypatch.setenv("UXA_COGNITIVE_MODEL", "cognitive-model")
    monkeypatch.setenv("UXA_REPORT_MODEL", "report-model")
    monkeypatch.setenv("UXA_LLM_TIMEOUT_SECONDS", "30")
    output = tmp_path / "output"
    output.mkdir()
    (output / "experiment.json").write_text("{}", encoding="utf-8")

    async def fake_synthesis(**kwargs: object):
        result = kwargs["result"]
        loaded_project = kwargs["loaded"]
        corpus = cli._synthesis_corpus(
            result=result,
            output=output,
            loaded=loaded_project,
        )
        attempt = await cli.ReportSynthesisService().synthesize(corpus)
        receipts = tuple(
            SynthesisRoleReceipt(
                role=role,
                provider_id="fixture-provider",
                model_id="fixture-model",
                prompt_digest=hashlib.sha256(f"{role}:prompt".encode()).hexdigest(),
                schema_digest=hashlib.sha256(f"{role}:schema".encode()).hexdigest(),
                output_digest=hashlib.sha256(f"{role}:output".encode()).hexdigest(),
            )
            for role in CANONICAL_SYNTHESIS_ROLES
        )
        return replace(
            attempt,
            status=SynthesisStatus.NO_ISSUES,
            role_receipts=receipts,
        )

    def fake_render(*, output: Path) -> Path:
        report = output / "report.html"
        report.write_text("<html></html>", encoding="utf-8")
        return report

    monkeypatch.setattr(cli, "_run_report_synthesis", fake_synthesis)
    monkeypatch.setattr(cli, "_render_completed_report", fake_render)

    for _ in range(2):
        result = runner.invoke(
            app,
            [
                "synthesize",
                str(DEMO_PROJECT),
                "--experiment",
                "core-pair",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0

    attempts = list((output / "synthesis" / "attempts").iterdir())
    assert len(attempts) == 2
    assert attempts[0].name != attempts[1].name
    index = json.loads((output / "synthesis" / "index.json").read_text())
    assert index["accepted_attempt_id"] in {attempt.name for attempt in attempts}


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
    monkeypatch.setenv("UXA_LLM_TIMEOUT_SECONDS", "30")

    async def unexpected_synthesis(**kwargs: object) -> object:
        del kwargs
        raise AssertionError("legacy project must keep synthesis disabled")

    monkeypatch.setattr(cli, "_run_report_synthesis", unexpected_synthesis)
    completed: dict[str, object] = {}

    async def fake_execute_matrix(*args: object, **kwargs: object) -> ExperimentResult:
        del args, kwargs
        return ExperimentResult(specs=(), results=(), failures=())

    def fake_write_summary(
        result: ExperimentResult,
        *,
        output: Path,
        selected_specs: object,
    ) -> Path:
        completed.update(
            result=result,
            output=output,
            selected_specs=selected_specs,
        )
        summary = output / "experiment.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text("{}", encoding="utf-8")
        return summary

    def fake_render(*, output: Path) -> Path:
        report = output / "report.html"
        report.write_text("<html></html>", encoding="utf-8")
        return report

    monkeypatch.setattr(cli, "_execute_matrix", fake_execute_matrix)
    monkeypatch.setattr(cli, "_write_experiment_summary", fake_write_summary)
    monkeypatch.setattr(cli, "_render_completed_report", fake_render)

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
            "prominence_provider_id": spec.prominence_provider_id,
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
