"""Scripted-TUI tests for the prompt-toolkit export selection screens."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Any

from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from ux_analyzer.export.interactive import _checkbox_app, _radio_app


def _run_with_timeout(
    app_factory: Callable[[Any], Any],
    keys: str,
    timeout: float = 10.0,
) -> Any:
    results: queue.Queue[Any] = queue.Queue()

    def target() -> None:
        with create_pipe_input() as pipe:
            pipe.send_text(keys)
            app = app_factory(pipe)
            results.put(app.run())

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    try:
        return results.get(timeout=timeout)
    except queue.Empty:
        raise AssertionError(
            "prompt-toolkit dialog did not exit; Enter/Esc binding is broken"
        ) from None


def _checkbox_app_factory(
    **kwargs: Any,
) -> Callable[[Any], Any]:
    def build(pipe: Any) -> Any:
        return _checkbox_app(
            input=pipe, output=DummyOutput(), **kwargs
        )

    return build


def _radio_app_factory(**kwargs: Any) -> Callable[[Any], Any]:
    def build(pipe: Any) -> Any:
        return _radio_app(input=pipe, output=DummyOutput(), **kwargs)

    return build


def test_space_toggles_and_enter_submits() -> None:
    result = _run_with_timeout(
        _checkbox_app_factory(
            title="Select issues",
            text="Space toggles selection. Enter continues.",
            values=[("a", "Issue A"), ("b", "Issue B")],
            default_values=(),
        ),
        " \r",
    )

    assert result == ["a"]


def test_arrow_moves_then_space_toggles_second_item() -> None:
    result = _run_with_timeout(
        _checkbox_app_factory(
            title="Select issues",
            text="Space toggles selection. Enter continues.",
            values=[("a", "Issue A"), ("b", "Issue B")],
            default_values=(),
        ),
        "\x1b[B \r",
    )

    assert result == ["b"]


def test_preselected_items_stay_selected_without_keystrokes() -> None:
    result = _run_with_timeout(
        _checkbox_app_factory(
            title="Select issues",
            text="Space toggles selection. Enter continues.",
            values=[("a", "Issue A"), ("b", "Issue B")],
            default_values=("a",),
        ),
        "\r",
    )

    assert result == ["a"]


def test_space_does_not_toggle_when_enter_only_submits() -> None:
    result = _run_with_timeout(
        _checkbox_app_factory(
            title="Select issues",
            text="Space toggles selection. Enter continues.",
            values=[("a", "Issue A")],
            default_values=("a",),
        ),
        "\x1b[B \r",
    )

    assert result == []


def test_radio_enter_picks_highlighted_value_and_continues() -> None:
    result = _run_with_timeout(
        _radio_app_factory(
            title="Skill set",
            text="Enter picks the highlighted set and continues.",
            values=[(None, "(no skills)"), ("tdd", "tdd")],
            default=None,
        ),
        "\x1b[B\r",
    )

    assert result == "tdd"


def test_radio_enter_on_first_value_picks_none() -> None:
    result = _run_with_timeout(
        _radio_app_factory(
            title="Skill set",
            text="Enter picks the highlighted set and continues.",
            values=[(None, "(no skills)"), ("tdd", "tdd")],
            default=None,
        ),
        "\r",
    )

    assert result is None


def test_escape_cancels_checkbox_dialog() -> None:
    result = _run_with_timeout(
        _checkbox_app_factory(
            title="Select issues",
            text="Space toggles selection. Enter continues.",
            values=[("a", "Issue A")],
            default_values=("a",),
        ),
        "\x1b",
    )

    assert result is None


def test_escape_cancels_radio_dialog() -> None:
    result = _run_with_timeout(
        _radio_app_factory(
            title="Skill set",
            text="Enter picks the highlighted set and continues.",
            values=[("tdd", "tdd")],
            default=None,
        ),
        "\x1b",
    )

    assert result is None
