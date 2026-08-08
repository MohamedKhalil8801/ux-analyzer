import os

import pytest
from typer.testing import CliRunner

from ux_analyzer.cli import app


def test_version_command() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "uxa 0.1.0"


def test_cli_does_not_enable_live_tests_from_dotenv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UXA_RUN_LIVE_TESTS", raising=False)

    result = CliRunner().invoke(app, ["version"])

    assert result.exit_code == 0
    assert "UXA_RUN_LIVE_TESTS" not in os.environ
