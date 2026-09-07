"""Hermetic unit-test environment: UXA_ settings never leak across tests."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _restore_uxa_environment() -> Iterator[None]:
    """Snapshot and restore all UXA_* environment variables per test.

    Loading a developer .env (for example through
    ``OpenAICompatibleSettings.from_env``) mutates ``os.environ`` through
    ``load_dotenv``; without isolation, ambient values such as
    ``UXA_LLM_REQUEST_MAX_BYTES`` change transport-budget behavior for every
    later test in the process. This fixture restores the environment after
    each unit test so the suite stays deterministic regardless of local
    configuration.
    """

    saved = {
        name: os.environ[name] for name in list(os.environ) if name.startswith("UXA_")
    }
    try:
        yield
    finally:
        for name in list(os.environ):
            if name.startswith("UXA_") and name not in saved:
                del os.environ[name]
        for name, value in saved.items():
            os.environ[name] = value
