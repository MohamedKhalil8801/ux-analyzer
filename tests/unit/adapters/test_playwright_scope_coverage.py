"""Every async Playwright lifetime must own its unretrieved-task reports.

The project suppresses exactly one class of event-loop report: a task whose
coroutine belongs to Playwright failing with a Playwright error, reported as
"Task exception was never retrieved". That happens when ``page.close()``,
``context.close()``, or ``browser.close()`` shuts the target while a Playwright
call is still in flight. The suppression is deliberately narrow, so it only
works where it is applied - and a browser lifetime that forgets it prints a
traceback after a run that actually succeeded.

That is not hypothetical: the exploration crawler launched a browser without the
scope and printed a ``TargetClosedError`` teardown traceback after five
successful walkthrough runs. This test encodes the invariant so the next async
launch site cannot reintroduce it.
"""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src" / "ux_analyzer"
QUIET_SCOPE = "playwright_task_quiet_scope"


def _async_playwright_sites() -> list[tuple[Path, int, str]]:
    """Return every ``async_playwright()`` call with its enclosing function."""

    sites: list[tuple[Path, int, str]] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        # utf-8-sig, because a couple of files in this tree still carry a BOM
        # and ast.parse rejects U+FEFF.
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # It is reached as a bare Name after
            # ``from playwright.async_api import async_playwright``.
            called = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else None
            )
            if called != "async_playwright":
                continue
            # async_playwright is also named in prose inside the scope's own
            # docstring; only a real call expression counts.
            owner = parents.get(node)
            while owner is not None and not isinstance(
                owner, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                owner = parents.get(owner)
            source = (
                ast.unparse(owner) if owner is not None else ast.unparse(node)
            )
            sites.append((path, node.lineno, source))
    return sites


def test_the_inventory_is_not_empty() -> None:
    """Guard the guard: an empty result would make the test below vacuous."""

    assert _async_playwright_sites(), "no async Playwright call sites were found"


def test_every_async_playwright_lifetime_is_wrapped_in_the_quiet_scope() -> None:
    unwrapped = [
        f"{path.relative_to(SOURCE_ROOT)}:{line}"
        for path, line, source in _async_playwright_sites()
        if QUIET_SCOPE not in source
    ]

    assert not unwrapped, (
        "async Playwright used outside "
        f"{QUIET_SCOPE}() at: {', '.join(unwrapped)}. Wrap the whole browser "
        "lifetime, outside async_playwright(), so teardown tasks are drained "
        "and their exceptions retrieved."
    )
