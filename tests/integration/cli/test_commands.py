from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from typer.testing import CliRunner

from ux_analyzer.cli import app

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
