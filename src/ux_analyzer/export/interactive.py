"""Prompt-toolkit implementation of the export selection UI."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from prompt_toolkit.shortcuts import (
    checkboxlist_dialog,
    confirm_dialog,
    radiolist_dialog,
)

from ux_analyzer.export.catalog import IssueCatalog, IssueView
from ux_analyzer.export.flow import SelectionResult, run_selection_flow
from ux_analyzer.export.skills import SkillSet


class PromptToolkitUI:
    """Checkbox/radio dialogs; Space toggles, Enter continues."""

    def select_group(
        self, group_name: str, items: Sequence[tuple[str, str, bool]]
    ) -> tuple[str, ...]:
        values = [(finding_id, label) for finding_id, label, _ in items]
        defaults = tuple(
            finding_id for finding_id, _, preselected in items if preselected
        )
        result = cast(
            "tuple[str, ...] | None",
            checkboxlist_dialog(
                title=f"Select issues: {group_name}",
                text="Space toggles, Enter continues to the next type.",
                values=values,
                default_values=defaults,
            ).run(),
        )
        return tuple(result or ())

    def choose_skill_set(self, sets: Sequence[SkillSet]) -> str | None:
        default = next((s.name for s in sets if s.is_default), None)
        values: list[tuple[str | None, str]] = [(None, "(no skills)")]
        values += [(s.name, f"{s.name} ({len(s.skills)} skill(s))") for s in sets]
        return cast(
            "str | None",
            radiolist_dialog(
                title="Skill set for all issues",
                text="One set applies to every issue; adjust per issue next.",
                values=values,
                default=default,
            ).run(),
        )

    def override_per_issue(
        self,
        issues: Sequence[IssueView],
        sets: Sequence[SkillSet],
        default_name: str | None,
    ) -> dict[str, str]:
        overrides: dict[str, str] = {}
        for issue in issues:
            values: list[tuple[str | None, str]] = [
                (None, f"(use default: {default_name or 'none'})")
            ]
            values += [(s.name, s.name) for s in sets]
            choice = cast(
                "str | None",
                radiolist_dialog(
                    title=f"Skill set for: {issue.finding_id}",
                    values=values,
                    default=None,
                ).run(),
            )
            if choice is not None:
                overrides[issue.finding_id] = choice
        return overrides

    def confirm(self, summary: str) -> bool:
        return bool(
            cast(
                "bool | None",
                confirm_dialog(title="Write export package", text=summary).run(),
            )
        )


def run_interactive_export(
    catalog: IssueCatalog,
    sets: tuple[SkillSet, ...],
    ui: PromptToolkitUI | None = None,
) -> SelectionResult | None:
    return run_selection_flow(catalog, sets, ui or PromptToolkitUI())
