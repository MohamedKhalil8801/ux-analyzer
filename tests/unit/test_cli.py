from typer.testing import CliRunner

from ux_analyzer.cli import app


def test_version_command() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "uxa 0.1.0"
