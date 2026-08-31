"""Unit tests for the UI-agnostic selection flow."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from ux_analyzer.export.catalog import (
    EvidenceRefView,
    IssueCatalog,
    IssueGroup,
    IssueView,
)
from ux_analyzer.export.flow import SelectionResult, run_selection_flow
from ux_analyzer.export.skills import SkillSet


def _issue(fid: str, resolved: bool = True) -> IssueView:
    return IssueView(
        finding_id=fid,
        filename=f"{fid}.md",
        title=fid,
        group="reviewed",
        severity="high",
        issue_text="x",
        impact="y",
        root_cause="z",
        fixes=("fix",),
        affected_surfaces=(),
        principles=(),
        limitations=(),
        reviewer_notes=(),
        severity_justification="",
        evidence_class="deterministic-fact",
        reproducibility="seeded",
        confidence=None,
        evidence=(EvidenceRefView("ev", "element", "run-1", resolved),),
        source="reviewed",
    )


class FakeUI:
    def __init__(
        self,
        picks: dict[str, tuple[str, ...]],
        skill_choice: str | None,
        overrides: dict[str, str] | None = None,
        confirm: bool = True,
    ) -> None:
        self.picks = picks
        self.skill_choice = skill_choice
        self.overrides = overrides or {}
        self.confirms = confirm
        self.summaries: list[str] = []

    def select_group(
        self, group_name: str, items: Sequence[tuple[str, str, bool]]
    ) -> tuple[str, ...]:
        return self.picks.get(group_name, ())

    def choose_skill_set(self, sets: Sequence[SkillSet]) -> str | None:
        return self.skill_choice

    def override_per_issue(
        self,
        issues: Sequence[IssueView],
        sets: Sequence[SkillSet],
        default_name: str | None,
    ) -> dict[str, str]:
        return self.overrides

    def confirm(self, summary: str) -> bool:
        self.summaries.append(summary)
        return self.confirms


def _catalog(*issues: IssueView) -> IssueCatalog:
    return IssueCatalog(
        groups=(IssueGroup("reviewed", tuple(issues)),),
        issues=tuple(issues),
    )


def test_flow_preselects_only_fully_resolved_issues() -> None:
    catalog = _catalog(_issue("good"), _issue("bad", resolved=False))
    ui = FakeUI({"reviewed": ("good", "bad")}, None)

    result = run_selection_flow(catalog, (), ui)

    assert result is not None
    assert [i.finding_id for i in result.issues] == ["good", "bad"]
    assert "evidence unavailable" in ui.summaries[0]


def test_flow_returns_none_when_nothing_selected() -> None:
    catalog = _catalog(_issue("good"))
    assert run_selection_flow(catalog, (), FakeUI({"reviewed": ()}, None)) is None


def test_flow_cancelled_when_not_confirmed() -> None:
    catalog = _catalog(_issue("good"))
    ui = FakeUI({"reviewed": ("good",)}, "frontend-fix", confirm=False)

    assert (
        run_selection_flow(
            catalog, (SkillSet("frontend-fix", (), True),), ui
        )
        is None
    )


def test_flow_resolves_assignments() -> None:
    catalog = _catalog(_issue("good"))
    ui = FakeUI(
        {"reviewed": ("good",)},
        "frontend-fix",
        overrides={"good": "design-only"},
    )

    result = run_selection_flow(
        catalog,
        (SkillSet("frontend-fix", (), True), SkillSet("design-only", ())),
        ui,
    )

    assert result == SelectionResult(
        issues=result.issues,
        default_skill_set="frontend-fix",
        per_issue_skills={"good": "design-only"},
    )


def test_flow_unknown_override_set_raises() -> None:
    catalog = _catalog(_issue("good"))
    ui = FakeUI(
        {"reviewed": ("good",)}, None, overrides={"good": "nope"}
    )

    with pytest.raises(ValueError, match="nope"):
        run_selection_flow(catalog, (SkillSet("frontend-fix", ()),), ui)
