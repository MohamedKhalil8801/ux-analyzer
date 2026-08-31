"""Integration tests for the non-interactive export command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import ux_analyzer.cli as cli_module
from ux_analyzer.cli import app


def _view(tmp_path: Path) -> dict[str, object]:
    return {
        "bundle_root": tmp_path,
        "synthesis_status": "completed",
        "using_fallback": False,
        "attempt_id": "attempt-1",
        "findings": [
            {
                "finding_id": "run-1:visual-hierarchy",
                "title": "Visual hierarchy",
                "issue": "Primary action is visually muted.",
                "impact": "Users miss the main call to action.",
                "root_cause": "Button uses body-text weight.",
                "fixes": ("Raise button contrast",),
                "severity": "high",
                "confidence": 0.9,
                "evidence_refs": [
                    {"evidence_id": "ev-1", "kind": "element", "run_id": "run-1"}
                ],
                "evidence_targets": [
                    {"kind": "element", "run_id": "run-1", "available": True}
                ],
                "affected_surfaces": ("settings",),
                "principles": ("Visual Hierarchy",),
                "counterevidence": [],
                "limitations": (),
                "reviewer_state": "approved",
                "evidence_class": "deterministic-fact",
                "reproducibility": "seeded",
                "severity_justification": "blocks the main task",
                "reviewer_notes": (),
            }
        ],
        "limitations": [],
    }


def test_export_non_tty_all_writes_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli_module, "load_report_findings", lambda _report: _view(tmp_path)
    )
    monkeypatch.setenv("UXA_SKILL_SETS", str(tmp_path / "sets.yaml"))
    (tmp_path / "sets.yaml").write_text(
        "sets:\n  frontend-fix:\n    skills: [tdd]\n    default: true\n",
        encoding="utf-8",
    )
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    package = tmp_path / "package"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "export",
            "--report",
            str(report_dir),
            "--out",
            str(package),
            "--all",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (package / "INDEX.md").is_file()
    manifest = json.loads((package / "manifest.json").read_text("utf-8"))
    assert manifest["schema_version"] == "fix-export-v1"
    assert manifest["issues"][0]["finding_id"] == "run-1:visual-hierarchy"


def test_export_without_selection_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli_module, "load_report_findings", lambda _report: _view(tmp_path)
    )
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["export", "--report", str(report_dir), "--out", str(tmp_path / "p")],
    )

    assert result.exit_code != 0
    assert "nothing to export" in result.output


def test_export_no_findings_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = _view(tmp_path)
    empty["findings"] = []
    monkeypatch.setattr(
        cli_module, "load_report_findings", lambda _report: empty
    )
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "export",
            "--report",
            str(report_dir),
            "--out",
            str(tmp_path / "p"),
            "--all",
        ],
    )

    assert result.exit_code != 0
    assert "nothing to export" in result.output


def test_export_malformed_issue_skill_flag_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli_module, "load_report_findings", lambda _report: _view(tmp_path)
    )
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "export",
            "--report",
            str(report_dir),
            "--out",
            str(tmp_path / "p"),
            "--all",
            "--issue-skill",
            "no-equals-sign",
        ],
    )

    assert result.exit_code != 0
