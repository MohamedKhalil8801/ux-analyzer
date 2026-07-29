from __future__ import annotations

# ruff: noqa: E402
import asyncio
import socket
import sys
from pathlib import Path

import httpx
import pytest_asyncio
import uvicorn

REPOSITORY_ROOT = Path(__file__).parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from fixture_app.app import app as fixture_app


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@pytest_asyncio.fixture
async def fixture_origin() -> str:
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            fixture_app,
            host="127.0.0.1",
            port=port,
            log_level="error",
        )
    )
    task = asyncio.create_task(server.serve())
    origin = f"http://127.0.0.1:{port}"
    async with httpx.AsyncClient() as client:
        for _ in range(100):
            try:
                if (await client.get(f"{origin}/app/ready/improved")).status_code < 500:
                    break
            except httpx.ConnectError:
                await asyncio.sleep(0.01)
        else:
            raise RuntimeError("fixture server did not start")
    try:
        yield origin
    finally:
        server.should_exit = True
        await task
