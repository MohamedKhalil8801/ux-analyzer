import json
import os
import threading
import time
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
    secure_replace = cli.secure_replace_exclusive_file

    def swap_destination(
        parent: object,
        source: object,
        destination_name: str,
        *args: object,
        **kwargs: object,
    ):
        destination = tmp_path / destination_name
        destination.rename(displaced)
        _symlink_or_skip(destination, victim)
        return secure_replace(
            parent, source, destination_name, *args, **kwargs
        )

    monkeypatch.setattr(cli, "secure_replace_exclusive_file", swap_destination)

    with pytest.raises(BundleStateError, match="symlink|reparse|link"):
        cli._atomic_write_experiment_json(summary, {"status": "complete"})

    assert victim.read_text(encoding="ascii") == "outside"
    assert displaced.read_text(encoding="ascii") == "previous\n"
    assert not tuple(tmp_path.glob(".experiment.*.tmp"))


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
    secure_create = cli.secure_create_exclusive_file

    @contextmanager
    def swap_parent(parent: object, *args: object, **kwargs: object):
        output.rename(backup)
        _symlink_or_skip(output, outside, directory=True)
        try:
            with secure_create(parent, *args, **kwargs) as temporary:
                yield temporary
        finally:
            output.unlink()
            backup.rename(output)

    monkeypatch.setattr(cli, "secure_create_exclusive_file", swap_parent)

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
    secure_open = cli.secure_open_directory

    @contextmanager
    def swap_parent(path: Path, *args: object, **kwargs: object):
        with secure_open(path, *args, **kwargs) as opened:
            output.rename(backup)
            outside.rename(output)
            try:
                yield opened
            finally:
                output.rename(outside)
                backup.rename(output)

    monkeypatch.setattr(cli, "secure_open_directory", swap_parent)

    with pytest.raises(BundleStateError, match="identity changed"):
        cli._atomic_write_experiment_json(
            output / "experiment.json", {"status": "complete"}
        )

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


@pytest.mark.parametrize(
    "payload",
    [
        "\x00" * 11,
        "\x01\x02\x03\x04" * 3,
        '"' * 32,
        "\\" * 32,
        "\N{GRINNING FACE}" * 6,
    ],
    ids=["nul", "control", "quote", "backslash", "unicode"],
)
def test_atomic_experiment_json_bounds_escaped_strings_without_encoder_tokens(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: str,
) -> None:
    def forbidden_iterencode(*args: object, **kwargs: object):
        del args, kwargs
        raise AssertionError("JSONEncoder materialized an escaped string token")

    monkeypatch.setattr(cli, "_MAX_EXPERIMENT_JSON_BYTES", 64)
    monkeypatch.setattr(json.JSONEncoder, "iterencode", forbidden_iterencode)

    with pytest.raises(ValueError, match="experiment summary.*exceeds"):
        cli._atomic_write_experiment_json(
            tmp_path / "experiment.json", {"payload": payload}
        )

    assert not tuple(tmp_path.glob(".experiment.*.tmp"))


def test_atomic_experiment_json_rejects_projected_eight_mib_nul_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = "\x00" * (cli._MAX_EXPERIMENT_JSON_BYTES // 6 + 1)

    def forbidden_iterencode(*args: object, **kwargs: object):
        del args, kwargs
        raise AssertionError("JSONEncoder materialized an escaped string token")

    monkeypatch.setattr(json.JSONEncoder, "iterencode", forbidden_iterencode)

    with pytest.raises(ValueError, match="experiment summary.*exceeds"):
        cli._atomic_write_experiment_json(
            tmp_path / "experiment.json", {"payload": payload}
        )

    assert not tuple(tmp_path.glob(".experiment.*.tmp"))


def test_atomic_experiment_json_matches_canonical_json_at_exact_bound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    value = {
        "z": (
            "\x00\b\f\n\r\t\"\\\x7f\ud800"
            "\N{GRINNING FACE}\N{LATIN SMALL LETTER E WITH ACUTE}"
        ),
        "scalars": [
            None,
            True,
            False,
            0,
            -17,
            1.25,
            -0.0,
            float("nan"),
            float("inf"),
            float("-inf"),
        ],
        "nested": {"b": ["plain", "\x1f"], "\x00a": {}},
    }
    expected = (
        json.dumps(
            cli._json_data(value),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    summary = tmp_path / "experiment.json"
    monkeypatch.setattr(cli, "_MAX_EXPERIMENT_JSON_BYTES", len(expected))

    cli._atomic_write_experiment_json(summary, value)

    assert summary.read_bytes() == expected

    monkeypatch.setattr(cli, "_MAX_EXPERIMENT_JSON_BYTES", len(expected) - 1)
    with pytest.raises(ValueError, match="experiment summary.*exceeds"):
        cli._atomic_write_experiment_json(summary, value)
    assert summary.read_bytes() == expected


def test_atomic_experiment_json_accounts_streaming_under_publication_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    canonical_chunks = cli._canonical_experiment_json_chunks
    accounting_lock = threading.Lock()
    active = 0
    maximum_active = 0

    def tracked_chunks(value: object, *, max_bytes: int) -> tuple[bytes, ...]:
        nonlocal active, maximum_active
        with accounting_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.02)
            return canonical_chunks(value, max_bytes=max_bytes)
        finally:
            with accounting_lock:
                active -= 1

    monkeypatch.setattr(cli, "_canonical_experiment_json_chunks", tracked_chunks)
    summaries = tuple(tmp_path / f"experiment-{index}.json" for index in range(6))

    with ThreadPoolExecutor(max_workers=len(summaries)) as executor:
        futures = [
            executor.submit(
                cli._atomic_write_experiment_json, path, {"writer": index}
            )
            for index, path in enumerate(summaries)
        ]
        for future in futures:
            future.result()

    assert maximum_active == 1


def test_atomic_experiment_json_canonically_replaces_regular_file(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "experiment.json"
    summary.write_text("previous\n", encoding="ascii")

    cli._atomic_write_experiment_json(
        summary, {"z": "\N{LATIN SMALL LETTER E WITH ACUTE}", "a": [2, 1]}
    )

    assert summary.read_bytes() == b'{"a":[2,1],"z":"\\u00e9"}\n'
