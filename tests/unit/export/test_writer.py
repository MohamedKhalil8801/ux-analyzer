"""Unit tests for the export package writer."""

from __future__ import annotations

from pathlib import Path

import pytest

from ux_analyzer.export.catalog import ArtifactFile, EvidenceRefView, IssueView
from ux_analyzer.export.render import ExportContext
from ux_analyzer.export.skills import SkillSet
from ux_analyzer.export.writer import ExportError, ExportResult, write_export


def _issue(artifact: ArtifactFile | None) -> IssueView:
    return IssueView(
        finding_id="run-1:spacing",
        filename="run-1-spacing.md",
        title="Spacing",
        group="reviewed",
        severity="medium",
        issue_text="Uneven card padding.",
        impact="Layout looks broken.",
        root_cause="Hard-coded margins.",
        fixes=("Use spacing tokens",),
        affected_surfaces=("dashboard",),
        principles=(),
        limitations=(),
        reviewer_notes=(),
        severity_justification="Cosmetic",
        evidence_class="deterministic-fact",
        reproducibility="seeded",
        confidence=None,
        evidence=(
            EvidenceRefView("ev-1", "screenshot", "run-1", artifact is not None),
        ),
        artifacts=((artifact,) if artifact else ()),
        source="reviewed",
    )


def _context(issue: IssueView) -> ExportContext:
    return ExportContext(
        report_path=Path("D:/reports"),
        exported_at="2026-08-31T10:00:00Z",
        tool_version="0.1.0",
        synthesis_status="completed",
        using_fallback=False,
        attempt_id="attempt-1",
        issues=(issue,),
        assignments={"run-1:spacing": "frontend-fix"},
        skill_sets=(SkillSet("frontend-fix", ("tdd",), True),),
        skills_note=None,
        reproduction_notes=None,
    )


def test_write_export_copies_assets_and_writes_files(tmp_path: Path) -> None:
    png = tmp_path / "shot.png"
    png.write_bytes(b"\x89PNG fake")
    issue = _issue(ArtifactFile("ev-1", png, None))
    package = tmp_path / "package"

    result = write_export(package, _context(issue))

    assert isinstance(result, ExportResult)
    assert result.issue_count == 1
    assert result.asset_count == 1
    assert (package / "INDEX.md").is_file()
    assert (package / "issues" / "run-1-spacing.md").is_file()
    copied = package / "assets" / "ev-1.png"
    assert copied.read_bytes() == b"\x89PNG fake"
    assert (package / "manifest.json").is_file()
    index_text = (package / "INDEX.md").read_text(encoding="utf-8")
    assert "FIX-REPORT.md template" in index_text
    assert not (package / "FIX-REPORT.md").exists()


def test_missing_artifact_becomes_marker_not_error(tmp_path: Path) -> None:
    issue = _issue(ArtifactFile("ev-1", tmp_path / "gone.png", None))
    package = tmp_path / "package"

    result = write_export(package, _context(issue))

    assert result.asset_count == 0
    md = (package / "issues" / "run-1-spacing.md").read_text(encoding="utf-8")
    assert "Evidence unavailable" in md


def test_sha256_mismatch_is_an_error(tmp_path: Path) -> None:
    png = tmp_path / "shot.png"
    png.write_bytes(b"tampered")
    issue = _issue(ArtifactFile("ev-1", png, "0" * 64))

    with pytest.raises(ExportError, match="ev-1"):
        write_export(tmp_path / "package", _context(issue))


def test_existing_package_is_never_overwritten(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()

    with pytest.raises(ExportError, match="already exists"):
        write_export(package, _context(_issue(None)))
