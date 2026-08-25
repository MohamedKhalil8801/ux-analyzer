"""Regression tests: the main `uxa run` path must persist the live-page audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

import ux_analyzer.cli as cli

runner = CliRunner()

MODEL_ENV = {
    "UXA_LLM_MODE": "api",
    "UXA_LLM_BASE_URL": "https://llm.example.test/v1",
    "UXA_LLM_API_KEY": "test-key",
    "UXA_SCENT_MODEL": "test-scent-model",
    "UXA_COGNITIVE_MODEL": "test-cognitive-model",
}


def _project_file(tmp_path: Path) -> Path:
    project = {
        "id": "audit-regression",
        "name": "Audit Regression",
        "applications": [
            {
                "id": "app",
                "name": "App",
                "versions": [
                    {
                        "id": "app-defective",
                        "kind": "defective",
                        "label": "Defective",
                        "start_url": "https://site.test/",
                        "allowed_origins": ["https://site.test"],
                    },
                    {
                        "id": "app-improved",
                        "kind": "improved",
                        "label": "Improved",
                        "start_url": "https://site.test/",
                        "allowed_origins": ["https://site.test"],
                    },
                ],
            }
        ],
        "scenarios": [
            {
                "id": "scenario-1",
                "name": "Scenario 1",
                "goal": "Reach the target control.",
                "application_version_ids": ["app-defective"],
                "start_state": "dashboard",
                "fixture_inputs": {},
                "budget": {
                    "max_steps": 5,
                    "max_observations": 5,
                    "max_interactions": 5,
                },
                "verifier": {"type": "visible-result", "text": "Target"},
                "safeguards": [],
                "eligible_persona_ids": ["persona-1"],
                "expected_evidence": ["target-discovery"],
                "evaluation_target": {"labels_by_version": {"defective": "Target"}},
            }
        ],
        "personas": [
            {
                "id": "persona-1",
                "name": "Persona One",
                "working_memory_capacity": 4,
                "initial_confidence": 0.5,
                "initial_frustration": 0.1,
                "abandonment_threshold": 0.8,
                "attention_temperature": 1.0,
            }
        ],
        "experiments": [
            {
                "id": "exp-1",
                "name": "Experiment One",
                "scenario_ids": ["scenario-1"],
                "application_version_ids": ["app-defective"],
                "persona_ids": ["persona-1"],
                "policies": ["full-list"],
                "run_count": 1,
            }
        ],
        "providers": {},
        "evaluation": {"report_synthesis": {"enabled": False}},
    }
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump(project), encoding="utf-8")
    return path


@dataclass(frozen=True)
class _Version:
    id: str = "app-defective"
    start_url: str | None = "https://site.test/"


@dataclass(frozen=True)
class _Spec:
    application_version: _Version = field(default_factory=_Version)


@dataclass(frozen=True)
class _State:
    spec: _Spec = field(default_factory=_Spec)


@dataclass(frozen=True)
class _RunResult:
    state: _State = field(default_factory=_State)
    run_id: str = "run-1"


def _fake_experiment_result() -> SimpleNamespace:
    return SimpleNamespace(results=(_RunResult(),), failures=())


def test_run_path_persists_ux_audit_and_feeds_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in MODEL_ENV.items():
        monkeypatch.setenv(name, value)
    project = _project_file(tmp_path)
    output = tmp_path / "out"
    calls: dict[str, Any] = {}

    def fake_negotiate(loaded: Any, *, allow_origin: Any, yes: bool) -> dict[str, Any]:
        return {}

    async def fake_execute_matrix(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return _fake_experiment_result()

    def fake_summary(result: Any, *, output: Path, selected_specs: Any) -> Path:
        calls["summary"] = output
        written = output / "experiment.json"
        written.parent.mkdir(parents=True, exist_ok=True)
        written.write_text("{}", encoding="utf-8")
        return written

    def fake_audit(*, output: Path, results: Any) -> Path | None:
        calls["audit_results"] = tuple(results)
        payload = {
            "schema_version": "ux-audit-v1",
            "total_issues": 1,
            "urls": [
                {
                    "url": "https://site.test/",
                    "total": 1,
                    "issues": [
                        {
                            "category": "GEO",
                            "check_id": "robots_txt",
                            "title": "robots.txt not found",
                            "severity": "critical",
                            "evidence": {"status_code": 404},
                        }
                    ],
                }
            ],
            "errors": [],
        }
        import json

        output.mkdir(parents=True, exist_ok=True)
        target = output / "ux-audit.json"
        target.write_text(json.dumps(payload), encoding="utf-8")
        return target

    def fake_render(*, output: Path) -> Path:
        import json

        run_dir = output / "runs" / "run-1"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(
            json.dumps({"run_id": "run-1"}), encoding="utf-8"
        )
        (run_dir / "timeline.jsonl").write_text("", encoding="utf-8")
        (run_dir / "result.json").write_text("{}", encoding="utf-8")
        from ux_analyzer.reporting.renderer import render_experiment_report

        return render_experiment_report(output, output / "report.html")

    monkeypatch.setattr(cli, "_negotiate_live_origin_gaps", fake_negotiate)
    monkeypatch.setattr(cli, "_execute_matrix", fake_execute_matrix)
    monkeypatch.setattr(cli, "_write_experiment_summary", fake_summary)
    monkeypatch.setattr(cli, "_write_ux_audit", fake_audit)
    monkeypatch.setattr(cli, "_render_completed_report", fake_render)

    result = runner.invoke(
        cli.app,
        [
            "run",
            str(project),
            "--experiment",
            "exp-1",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    # audit received the run results (start URLs extracted from specs)
    assert calls["audit_results"][0].state.spec.application_version.start_url == (
        "https://site.test/"
    )
    # audit file persisted beside bundles and rendered into the report
    assert (output / "ux-audit.json").exists()
    html = (output / "report.html").read_text(encoding="utf-8")
    assert 'id="ux-audit"' in html
    assert "robots.txt not found" in html
