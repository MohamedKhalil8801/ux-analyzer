"""UI-agnostic selection flow for fix exports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ux_analyzer.export.catalog import IssueCatalog, IssueView
from ux_analyzer.export.skills import SkillSet, resolve_assignments


@dataclass(frozen=True, slots=True)
class SelectionResult:
    issues: tuple[IssueView, ...]
    default_skill_set: str | None
    per_issue_skills: dict[str, str] = field(default_factory=dict[str, str])


class SelectionUI(Protocol):
    """Terminal UI contract; implemented by prompt-toolkit in interactive.py."""

    def select_group(
        self, group_name: str, items: Sequence[tuple[str, str, bool]]
    ) -> tuple[str, ...]: ...

    def choose_skill_set(self, sets: Sequence[SkillSet]) -> str | None: ...

    def override_per_issue(
        self,
        issues: Sequence[IssueView],
        sets: Sequence[SkillSet],
        default_name: str | None,
    ) -> dict[str, str]: ...

    def confirm(self, summary: str) -> bool: ...


def run_selection_flow(
    catalog: IssueCatalog,
    sets: Sequence[SkillSet],
    ui: SelectionUI,
) -> SelectionResult | None:
    chosen: list[IssueView] = []
    for group in catalog.groups:
        items: list[tuple[str, str, bool]] = []
        for issue in group.issues:
            # A rejected candidate is listed like any other, so the reader can
            # see it and decide, but it starts unchecked. Same rule as evidence
            # that could not be resolved: the claim is visible and the choice to
            # act on it is deliberate.
            notes: list[str] = []
            if not issue.published:
                notes.append("rejected during review")
            if issue.has_unresolved_evidence:
                notes.append("evidence unavailable")
            items.append(
                (
                    issue.finding_id,
                    issue.title
                    + "".join(f"  [warning: {note}]" for note in notes),
                    issue.published and not issue.has_unresolved_evidence,
                )
            )
        for finding_id in ui.select_group(group.name, items):
            issue = catalog.find(finding_id)
            if issue is not None:
                chosen.append(issue)
    if not chosen:
        return None
    default_name = ui.choose_skill_set(sets) if sets else None
    per_issue = (
        ui.override_per_issue(chosen, sets, default_name) if sets else {}
    )
    resolve_assignments(sets, default_name, per_issue)
    unresolved = sum(1 for issue in chosen if issue.has_unresolved_evidence)
    rejected = sum(1 for issue in chosen if not issue.published)
    summary = (
        f"Export {len(chosen)} issue(s)"
        + (f" ({unresolved} with evidence unavailable)" if unresolved else "")
        + (f" ({rejected} rejected during review)" if rejected else "")
        + f"; skills: default '{default_name or '(none)'}' with "
        f"{len(per_issue)} per-issue override(s)."
    )
    if not ui.confirm(summary):
        return None
    return SelectionResult(tuple(chosen), default_name, dict(per_issue))
