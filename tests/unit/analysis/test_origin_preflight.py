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
