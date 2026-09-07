"""Hermetic integration-test environment: UXA_ settings never leak across tests."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _restore_uxa_environment() -> Iterator[None]:
    """Snapshot and restore all UXA_* environment variables per test.

    Loading a developer .env mutates ``os.environ`` through ``load_dotenv``
    (see ``load_environment_file``), which would otherwise leak values such
    as ``UXA_LLM_REQUEST_MAX_BYTES`` into later byte-budget-sensitive tests.
    Restoring per test keeps transport ceilings and model configuration
    deterministic regardless of local configuration files.
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
