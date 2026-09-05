"""Prompt-toolkit implementation of the export selection UI.

Keyboard contract: Space toggles, Enter continues (submits), arrow keys
move between options, Tab reaches Ok/Cancel, Esc cancels.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.key_binding.bindings.focus import focus_next, focus_previous
from prompt_toolkit.key_binding.defaults import load_key_bindings
from prompt_toolkit.key_binding.key_bindings import KeyBindings, merge_key_bindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout.containers import HSplit
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.shortcuts import yes_no_dialog
from prompt_toolkit.widgets import Button, CheckboxList, Dialog, Label, RadioList

from ux_analyzer.export.catalog import IssueCatalog, IssueView
from ux_analyzer.export.flow import SelectionResult, run_selection_flow
from ux_analyzer.export.skills import SkillSet


class _SpaceToggleList(CheckboxList[str]):
    """Checkbox list without the built-in Enter-to-toggle binding."""

    def __init__(
        self,
        values: Sequence[tuple[str, str]],
        default_values: Sequence[str] | None = None,
    ) -> None:
        super().__init__(values=values, default_values=default_values)
        kb = KeyBindings()

        def _up(event: KeyPressEvent) -> None:
            self._selected_index = max(0, self._selected_index - 1)  # type: ignore[attr-defined]

        def _down(event: KeyPressEvent) -> None:
            self._selected_index = min(  # type: ignore[attr-defined]
                len(self.values) - 1, self._selected_index + 1
            )

        def _pageup(event: KeyPressEvent) -> None:
            window = event.app.layout.current_window
            if window.render_info:
                self._selected_index = max(  # type: ignore[attr-defined]
                    0,
                    self._selected_index  # type: ignore[attr-defined]
                    - len(window.render_info.displayed_lines),
                )

        def _pagedown(event: KeyPressEvent) -> None:
            window = event.app.layout.current_window
            if window.render_info:
                self._selected_index = min(  # type: ignore[attr-defined]
                    len(self.values) - 1,
                    self._selected_index  # type: ignore[attr-defined]
                    + len(window.render_info.displayed_lines),
                )

        def _toggle(event: KeyPressEvent) -> None:
            self._handle_enter()  # type: ignore[attr-defined]

        kb.add("up")(_up)
        kb.add("k")(_up)
        kb.add("down")(_down)
        kb.add("j")(_down)
        kb.add("pageup")(_pageup)
        kb.add("pagedown")(_pagedown)
        kb.add(" ")(_toggle)
        self.control.key_bindings = kb


class _EnterSelectsRadioList(RadioList[str | None]):
    """Radio list where Enter picks the highlighted value and continues."""

    def __init__(
        self,
        values: Sequence[tuple[str | None, str]],
        default: str | None = None,
    ) -> None:
        super().__init__(values=values, default=default)
        kb = KeyBindings()

        def _up(event: KeyPressEvent) -> None:
            self._selected_index = max(0, self._selected_index - 1)  # type: ignore[attr-defined]

        def _down(event: KeyPressEvent) -> None:
            self._selected_index = min(  # type: ignore[attr-defined]
                len(self.values) - 1, self._selected_index + 1
            )

        def _select(event: KeyPressEvent) -> None:
            self._handle_enter()  # type: ignore[attr-defined]
            event.app.exit(result=self.current_value)

        kb.add("up")(_up)
        kb.add("k")(_up)
        kb.add("down")(_down)
        kb.add("j")(_down)
        kb.add("enter")(_select)
        self.control.key_bindings = kb


def _cancel() -> None:
    get_app().exit()


def _base_app(
    dialog: Dialog,
    extra_bindings: KeyBindings,
    *,
    input: Any = None,
    output: Any = None,
) -> Application[Any]:
    kb = KeyBindings()

    def _tab(event: KeyPressEvent) -> None:
        focus_next(event)

    def _shift_tab(event: KeyPressEvent) -> None:
        focus_previous(event)

    def _escape(event: KeyPressEvent) -> None:
        event.app.exit()

    kb.add("tab")(_tab)
    kb.add("s-tab")(_shift_tab)
    kb.add("escape")(_escape)

    return Application(
        layout=Layout(dialog),
        key_bindings=merge_key_bindings([load_key_bindings(), kb, extra_bindings]),
        mouse_support=True,
        full_screen=True,
        input=input,
        output=output,
    )


def _checkbox_app(
    title: str,
    text: str,
    values: Sequence[tuple[str, str]],
    default_values: Sequence[str],
    *,
    input: Any = None,
    output: Any = None,
) -> Application[list[str] | None]:
    cb_list = _SpaceToggleList(values=values, default_values=default_values)

    def accept() -> None:
        get_app().exit(result=list(cb_list.current_values))

    dialog = Dialog(
        title=title,
        body=HSplit(
            [Label(text=text, dont_extend_height=True), cb_list],
            padding=1,
        ),
        buttons=[
            Button(text="Ok", handler=accept),
            Button(text="Cancel", handler=_cancel),
        ],
        with_background=True,
    )
    kb = KeyBindings()

    def _accept(event: KeyPressEvent) -> None:
        accept()

    kb.add("enter")(_accept)

    app: Application[list[str] | None] = _base_app(
        dialog, kb, input=input, output=output
    )
    return app


def _radio_app(
    title: str,
    text: str,
    values: Sequence[tuple[str | None, str]],
    default: str | None,
    *,
    input: Any = None,
    output: Any = None,
) -> Application[str | None]:
    radio_list = _EnterSelectsRadioList(values=values, default=default)

    def accept() -> None:
        get_app().exit(result=radio_list.current_value)

    dialog = Dialog(
        title=title,
        body=HSplit(
            [Label(text=text, dont_extend_height=True), radio_list],
            padding=1,
        ),
        buttons=[
            Button(text="Ok", handler=accept),
            Button(text="Cancel", handler=_cancel),
        ],
        with_background=True,
    )
    app: Application[str | None] = _base_app(
        dialog, KeyBindings(), input=input, output=output
    )
    return app


def _run(app: Application[Any]) -> Any:
    try:
        return app.run()
    except (KeyboardInterrupt, EOFError):
        return None


class PromptToolkitUI:
    """Checkbox/radio dialogs; Space toggles, Enter continues."""

    def select_group(
        self, group_name: str, items: Sequence[tuple[str, str, bool]]
    ) -> tuple[str, ...]:
        values = [(finding_id, label) for finding_id, label, _ in items]
        if not values:
            return ()
        defaults = tuple(
            finding_id for finding_id, _, preselected in items if preselected
        )
        result = _run(
            _checkbox_app(
                title=f"Select issues: {group_name}",
                text=(
                    "Space toggles selection. Enter continues to the next "
                    "type. Esc cancels the export."
                ),
                values=values,
                default_values=defaults,
            )
        )
        return tuple(result or ())

    def choose_skill_set(self, sets: Sequence[SkillSet]) -> str | None:
        default = next((s.name for s in sets if s.is_default), None)
        values: list[tuple[str | None, str]] = [(None, "(no skills)")]
        values += [(s.name, f"{s.name} ({len(s.skills)} skill(s))") for s in sets]
        return _run(
            _radio_app(
                title="Skill set for all issues",
                text=(
                    "One set applies to every issue; adjust per issue next. "
                    "Enter picks the highlighted set and continues."
                ),
                values=values,
                default=default,
            )
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
            choice = _run(
                _radio_app(
                    title=f"Skill set for: {issue.finding_id}",
                    text="Enter picks the highlighted set and continues.",
                    values=values,
                    default=None,
                )
            )
            if choice is not None:
                overrides[issue.finding_id] = choice
        return overrides

    def confirm(self, summary: str) -> bool:
        return bool(_run(yes_no_dialog(title="Write export package", text=summary)))


def run_interactive_export(
    catalog: IssueCatalog,
    sets: tuple[SkillSet, ...],
    ui: PromptToolkitUI | None = None,
) -> SelectionResult | None:
    return run_selection_flow(catalog, sets, ui or PromptToolkitUI())
