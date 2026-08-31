"""Unit tests for pure export renderers."""

from __future__ import annotations

from pathlib import Path

from ux_analyzer.export.catalog import ArtifactFile, EvidenceRefView, IssueView
from ux_analyzer.export.render import ExportContext, render_index, render_issue
from ux_analyzer.export.skills import SkillSet


def _issue(**overrides: object) -> IssueView:
    base = dict(
        finding_id="run-1:visual-hierarchy",
        filename="run-1-visual-hierarchy.md",
        title="Visual hierarchy",
        group="visual-hierarchy",
        severity="high",
        issue_text="Primary action is visually muted.",
        impact="Users miss the main call to action.",
        root_cause="Button uses the same weight as body text.",
        fixes=("Raise button contrast", "Use the primary button style"),
        affected_surfaces=("settings",),
        principles=("Visual Hierarchy",),
        limitations=("One viewport tested",),
        reviewer_notes=("Reviewer: consider mobile",),
        severity_justification="Blocks the primary task",
        evidence_class="deterministic-fact",
        reproducibility="seeded",
        confidence=0.92,
        evidence=(
            EvidenceRefView(
                "ev-1",
                "screenshot",
                "run-1",
                True,
                {
                    "Element selectors": "#submit-cta",
                    "Element xpaths": "/html/body/main/div[2]/button",
                },
            ),
            EvidenceRefView("ev-2", "element", "run-1", False, {}),
        ),
        artifacts=(
            ArtifactFile("ev-1", Path("bundle/runs/run-1/shot.png"), "abc"),
        ),
        source="reviewed",
    )
    base.update(overrides)
    return IssueView(**base)  # type: ignore[arg-type]


def test_render_issue_lists_selectors_fixes_and_markers() -> None:
    md = render_issue(_issue(), {"ev-1": "assets/ev-1.png"})

    assert "run-1:visual-hierarchy" in md
    assert "### Evidence" in md
    assert "- `ev-1` (screenshot, run `run-1`)" in md
    assert "- **SELECTOR:** `#submit-cta`" in md
    assert "- **XPATH:** `/html/body/main/div[2]/button`" in md
    assert "![screenshot ev-1](assets/ev-1.png)" in md
    assert "**Evidence unavailable:** `ev-2` (element)" in md
    assert "Option A: Raise button contrast" in md
    assert "Option B: Use the primary button style" in md
    assert "invent a better solution" in md
    assert "interpretive lenses, not evidence" in md
    assert "Blocks the primary task" in md
    assert "One viewport tested" in md


def test_render_index_contains_protocol_table_and_skills() -> None:
    context = ExportContext(
        report_path=Path("D:/reports"),
        exported_at="2026-08-31T10:00:00Z",
        tool_version="0.1.0",
        synthesis_status="completed",
        using_fallback=False,
        attempt_id="attempt-1",
        issues=(_issue(),),
        assignments={"run-1:visual-hierarchy": "frontend-fix"},
        skill_sets=(SkillSet("frontend-fix", ("tdd", "impeccable"), True),),
        skills_note="Prefer TDD throughout.",
        reproduction_notes="npm install && npm run dev",
    )
    md = render_index(context)

    assert "# Fix Export" in md
    assert "npm install && npm run dev" in md
    assert "tdd, impeccable" in md
    assert "Prefer TDD throughout." in md
    assert "run-1-visual-hierarchy.md" in md
    assert "| `run-1:visual-hierarchy` |" in md
    assert "## Fixer Workflow" in md
    assert "5 rounds maximum" in md
    assert "## FIX-REPORT.md template" in md
    assert "evidence unavailable" in md.lower()


def test_render_manifest_shape() -> None:
    from ux_analyzer.export.render import render_manifest

    context = ExportContext(
        report_path=Path("D:/reports"),
        exported_at="2026-08-31T10:00:00Z",
        tool_version="0.1.0",
        synthesis_status="completed",
        using_fallback=False,
        attempt_id="attempt-1",
        issues=(_issue(),),
        assignments={"run-1:visual-hierarchy": "frontend-fix"},
        skill_sets=(),
        skills_note=None,
        reproduction_notes=None,
    )
    manifest = render_manifest(context, {"assets/ev-1.png": "abc"})

    assert manifest["schema_version"] == "fix-export-v1"
    assert manifest["exported_at"] == "2026-08-31T10:00:00Z"
    assert manifest["report_path"] == "D:/reports"
    assert manifest["issues"][0]["filename"] == "run-1-visual-hierarchy.md"
    assert manifest["assets"] == {"assets/ev-1.png": "abc"}
