from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

import ux_analyzer.cli as cli
from ux_analyzer.analysis.referenced_origins import (
    referenced_origins,
    referenced_origins_sync,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# referenced_origins parser
# ---------------------------------------------------------------------------
def _client(html: str, content_type: str = "text/html") -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=html,
            headers={"content-type": content_type},
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_referenced_origins_extracts_foreign_origins() -> None:
    html = """<!doctype html><html><head>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=X">
    <link rel="preconnect" href="https://fonts.gstatic.com">
    <script src="/assets/app.js"></script>
    </head><body>
    <img src="https://cdn.example.org/pic.webp" srcset="https://cdn.example.org/pic2x.webp 2x">
    <img src="data:image/png;base64,AAAA">
    <iframe src="about:blank"></iframe>
    </body></html>"""
    client = _client(html)
    try:
        origins = await referenced_origins("https://site.test/", client=client)
    finally:
        await client.aclose()
    # anchors are navigation targets, not load-time subresources
    assert "https://other.example.net" not in origins
    assert origins == frozenset(
        {
            "https://fonts.googleapis.com",
            "https://fonts.gstatic.com",
            "https://cdn.example.org",
        }
    )


@pytest.mark.asyncio
async def test_referenced_origins_ignores_non_html_and_errors() -> None:
    client = _client("{}", content_type="application/json")
    try:
        assert await referenced_origins("https://site.test/", client=client) == frozenset()
    finally:
        await client.aclose()

    def failing(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    broken = httpx.AsyncClient(transport=httpx.MockTransport(failing))
    try:
        assert await referenced_origins("https://down.test/", client=broken) == frozenset()
    finally:
        await broken.aclose()


def test_referenced_origins_sync_outside_loop() -> None:
    assert referenced_origins_sync("not-a-url") == frozenset()


# ---------------------------------------------------------------------------
# CLI negotiation helpers
# ---------------------------------------------------------------------------
def _loaded_with_live_version(
    tmp_path: Any, *, start_url: str, allowed: list[str]
) -> Any:
    version = SimpleNamespace(
        id="app-live",
        kind=cli.ApplicationVersionKind.LIVE,
        start_url=start_url,
        allowed_origins=tuple(allowed),
    )
    application = SimpleNamespace(versions=(version,))
    return SimpleNamespace(project=SimpleNamespace(applications=(application,)))


def test_collect_origin_gaps_reports_missing_only(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = _loaded_with_live_version(
        tmp_path,
        start_url="https://site.test/",
        allowed=["https://fonts.googleapis.com"],
    )
    monkeypatch.setattr(
        cli,
        "_referenced_origins_sync_cli",
        lambda url: frozenset(
            {"https://site.test", "https://fonts.googleapis.com", "https://cdn.test"}
        ),
    )
    gaps = cli._collect_origin_gaps(loaded)
    assert gaps == {"app-live": frozenset({"https://cdn.test"})}


def test_select_extra_origins_auto_yes(tmp_path: Any) -> None:
    grants = cli._select_extra_origins(
        {"v1": frozenset({"https://a.test", "https://b.test"})},
        auto_yes=True,
    )
    assert grants == {"v1": {"https://a.test", "https://b.test"}}


def test_select_extra_origins_interactive_selection() -> None:
    answers = iter(["s", "2"])
    grants = cli._select_extra_origins(
        {"v1": frozenset({"https://a.test", "https://b.test"})},
        auto_yes=False,
        prompt=lambda _: next(answers),
        echo=lambda *_: None,
    )
    assert grants == {"v1": {"https://b.test"}}


def test_select_extra_origins_rejects_then_none() -> None:
    answers = iter(["banana", "n"])
    grants = cli._select_extra_origins(
        {"v1": frozenset({"https://a.test"})},
        auto_yes=False,
        prompt=lambda _: next(answers),
        echo=lambda *_: None,
    )
    assert grants == {}


def test_negotiate_non_interactive_grants_nothing(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _loaded_with_live_version(
        tmp_path,
        start_url="https://site.test/",
        allowed=[],
    )
    monkeypatch.setattr(
        cli,
        "_referenced_origins_sync_cli",
        lambda url: frozenset({"https://cdn.test"}),
    )
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    grants = cli._negotiate_live_origin_gaps(loaded, allow_origin=(), yes=False)
    assert grants == {}


def test_negotiate_yes_grants_all(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = _loaded_with_live_version(
        tmp_path,
        start_url="https://site.test/",
        allowed=[],
    )
    monkeypatch.setattr(
        cli,
        "_referenced_origins_sync_cli",
        lambda url: frozenset({"https://cdn.test", "https://fonts.test"}),
    )
    grants = cli._negotiate_live_origin_gaps(
        loaded, allow_origin=("https://fonts.test",), yes=True
    )
    # flag-granted origin applies to every live version and --yes absorbs the
    # remaining referenced gaps without prompting.
    assert grants == {"app-live": {"https://fonts.test", "https://cdn.test"}}


def test_negotiate_allow_origin_flag_applies_without_gap(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = _loaded_with_live_version(
        tmp_path,
        start_url="https://site.test/",
        allowed=[],
    )
    monkeypatch.setattr(
        cli,
        "_referenced_origins_sync_cli",
        lambda url: frozenset({"https://site.test"}),
    )
    grants = cli._negotiate_live_origin_gaps(
        loaded, allow_origin=("https://extra.test",), yes=False
    )
    assert grants == {"app-live": {"https://extra.test"}}


# ---------------------------------------------------------------------------
# spec patching
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Version:
    id: str
    allowed_origins: tuple[str, ...]


@dataclass(frozen=True)
class _Spec:
    application_version: _Version


def test_with_extra_resource_origins_patches_matching_specs() -> None:
    specs = (
        _Spec(_Version("v1", ("https://a.test",))),
        _Spec(_Version("v2", ("https://a.test",))),
    )
    patched = cli._with_extra_resource_origins(
        specs, {"v1": ["https://b.test"]}  # type: ignore[arg-type]
    )
    assert patched[0].application_version.allowed_origins == (
        "https://a.test",
        "https://b.test",
    )
    assert patched[1].application_version.allowed_origins == ("https://a.test",)
    # originals untouched (immutability preserved via replace semantics)
    assert specs[0].application_version.allowed_origins == ("https://a.test",)


# ---------------------------------------------------------------------------
# command surface
# ---------------------------------------------------------------------------
def test_run_command_passes_allow_origin_flags(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_command(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(cli, "_run_experiment_command", fake_command)
    result = runner.invoke(
        cli.app,
        [
            "run",
            "project.yaml",
            "--allow-origin",
            "https://cdn.test",
            "--yes",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["allow_origin"] == ["https://cdn.test"]
    assert captured["yes"] is True


