import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest
from typer.testing import CliRunner

import ux_analyzer.cli as cli
from ux_analyzer.cli import app
from ux_analyzer.ports.artifacts import BundleStateError


def _symlink_or_skip(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as error:
        pytest.skip(f"symlink race fixture unavailable: {error}")


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


def test_atomic_experiment_json_ignores_predictable_temporary_symlink(
    tmp_path: Path,
) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("outside", encoding="ascii")
    predictable = tmp_path / ".experiment.json.tmp"
    _symlink_or_skip(predictable, victim)

    summary = tmp_path / "experiment.json"
    cli._atomic_write_experiment_json(summary, {"status": "complete"})

    assert victim.read_text(encoding="ascii") == "outside"
    assert predictable.is_symlink()
    assert json.loads(summary.read_text(encoding="utf-8")) == {"status": "complete"}


def test_atomic_experiment_json_rejects_destination_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    summary = tmp_path / "experiment.json"
    summary.write_text("previous\n", encoding="ascii")
    displaced = tmp_path / "experiment.previous.json"
    victim = tmp_path / "victim.txt"
    victim.write_text("outside", encoding="ascii")
    secure_replace = cli.secure_replace

    def swap_destination(source: Path, destination: Path, *args: object, **kwargs: object):
        destination.rename(displaced)
        _symlink_or_skip(destination, victim)
        return secure_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(cli, "secure_replace", swap_destination)

    with pytest.raises(BundleStateError, match="symlink|reparse|link"):
        cli._atomic_write_experiment_json(summary, {"status": "complete"})

    assert victim.read_text(encoding="ascii") == "outside"
    assert displaced.read_text(encoding="ascii") == "previous\n"


def test_atomic_experiment_json_rejects_parent_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "experiment.json"
    victim.write_text("outside", encoding="ascii")
    backup = tmp_path / "output-original"
    secure_replace = cli.secure_replace

    def swap_parent(source: Path, destination: Path, *args: object, **kwargs: object):
        output.rename(backup)
        _symlink_or_skip(output, outside, directory=True)
        try:
            return secure_replace(source, destination, *args, **kwargs)
        finally:
            output.unlink()
            backup.rename(output)

    monkeypatch.setattr(cli, "secure_replace", swap_parent)

    with pytest.raises(BundleStateError, match="symlink|reparse|path|containment"):
        cli._atomic_write_experiment_json(
            output / "experiment.json", {"status": "complete"}
        )

    assert victim.read_text(encoding="ascii") == "outside"
    assert not tuple(output.glob(".experiment.*.tmp"))


def test_atomic_experiment_json_rejects_real_parent_swap_before_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "experiment.json"
    victim.write_text("outside", encoding="ascii")
    backup = tmp_path / "output-original"
    secure_open = cli.secure_open_file_descriptor

    @contextmanager
    def swap_parent(path: Path, *args: object, **kwargs: object):
        output.rename(backup)
        outside.rename(output)
        with secure_open(path, *args, **kwargs) as opened:
            yield opened

    monkeypatch.setattr(cli, "secure_open_file_descriptor", swap_parent)

    try:
        with pytest.raises(BundleStateError, match="identity changed"):
            cli._atomic_write_experiment_json(
                output / "experiment.json", {"status": "complete"}
            )
    finally:
        output.rename(outside)
        backup.rename(output)

    assert victim.read_text(encoding="ascii") == "outside"


def test_atomic_experiment_json_concurrent_writes_are_collision_safe(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "experiment.json"
    values = tuple({"writer": index, "payload": "x" * 1024} for index in range(8))

    with ThreadPoolExecutor(max_workers=len(values)) as executor:
        futures = [
            executor.submit(cli._atomic_write_experiment_json, summary, value)
            for value in values
        ]
        for future in futures:
            future.result()

    assert json.loads(summary.read_text(encoding="utf-8")) in values
    assert not tuple(tmp_path.glob(".experiment.*.tmp"))


def test_atomic_experiment_json_is_bounded_and_preserves_existing_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    summary = tmp_path / "experiment.json"
    summary.write_text("previous\n", encoding="ascii")
    monkeypatch.setattr(cli, "_MAX_EXPERIMENT_JSON_BYTES", 16)

    with pytest.raises(ValueError, match="experiment summary.*exceeds"):
        cli._atomic_write_experiment_json(summary, {"payload": "too large"})

    assert summary.read_text(encoding="ascii") == "previous\n"
    assert not tuple(tmp_path.glob(".experiment.*.tmp"))


def test_atomic_experiment_json_canonically_replaces_regular_file(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "experiment.json"
    summary.write_text("previous\n", encoding="ascii")

    cli._atomic_write_experiment_json(
        summary, {"z": "\N{LATIN SMALL LETTER E WITH ACUTE}", "a": [2, 1]}
    )

    assert summary.read_bytes() == b'{"a":[2,1],"z":"\\u00e9"}\n'
