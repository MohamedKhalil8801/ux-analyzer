"""Diagnostics must make a failure diagnosable without reopening the boundary.

The contract under test is narrow and important: an operator reading the log
after a failed run must be able to tell *which* field failed, *why*, and
*whether the same text failed before* - while the offending text itself never
appears anywhere in the record.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ux_analyzer.observability import diagnostics

_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}\+00:00$")


@pytest.fixture(autouse=True)
def _clean_diagnostics():
    diagnostics.reset()
    yield
    diagnostics.reset()


def test_every_event_carries_a_utc_millisecond_timestamp():
    event = diagnostics.record("role.call.started", role="report-analyst")

    assert _TIMESTAMP.match(event["timestamp"]), event["timestamp"]


def test_timestamps_are_ordered_within_a_run():
    first = diagnostics.record("a")["timestamp"]
    second = diagnostics.record("b")["timestamp"]

    assert first <= second


def test_rejected_text_is_fingerprinted_and_never_stored():
    secret = "The prior finding asserted the filter was broken"

    event = diagnostics.record(
        "response.rejected",
        reason="response limitation contains forbidden narrative",
        field_path="limitations[3]",
        text=secret,
    )

    assert event["field"] == "limitations[3]"
    assert event["field_length"] == len(secret)
    assert event["field_hash"] == diagnostics.fingerprint(secret)
    assert secret not in json.dumps(event)
    assert "prior finding" not in json.dumps(event)


def test_identical_text_fingerprints_identically_across_events():
    text = "no decision rationale was recorded for this click"

    first = diagnostics.record("response.rejected", text=text)
    second = diagnostics.record("response.rejected", text=text)

    assert first["field_hash"] == second["field_hash"]


def test_different_text_fingerprints_differently():
    assert diagnostics.fingerprint("alpha") != diagnostics.fingerprint("beta")


def test_reasons_are_redacted_even_if_a_caller_interpolates_model_text():
    event = diagnostics.record(
        "response.rejected",
        reason="failed: the prior agent concluded otherwise",
    )

    # Redaction is whole-string, not per-marker: a partially redacted string
    # would still disclose everything except the phrase we happened to match.
    assert event["reason"] == "[redacted]"


def test_string_extras_are_redacted():
    event = diagnostics.record(
        "role.call.failed",
        note="consulted the system prompt for guidance",
        count=3,
        ok=True,
    )

    assert event["extras"]["note"] == "[redacted]"
    assert event["extras"]["count"] == 3
    assert event["extras"]["ok"] is True


def test_events_without_text_omit_the_fingerprint_fields():
    event = diagnostics.record("role.call.started", role="report-analyst")

    assert "field_hash" not in event
    assert "field_length" not in event


def test_file_sink_appends_one_json_object_per_line(tmp_path: Path):
    log = tmp_path / "nested" / "diagnostics.jsonl"
    diagnostics.configure(path=log)
    diagnostics.record("role.call.started", role="report-analyst")
    diagnostics.record("response.rejected", reason="bounded-output", text="x")

    lines = log.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    for line in lines:
        payload = json.loads(line)
        assert _TIMESTAMP.match(payload["timestamp"])
    assert json.loads(lines[1])["reason"] == "bounded-output"


def test_file_sink_appends_rather_than_truncates(tmp_path: Path):
    log = tmp_path / "diagnostics.jsonl"
    diagnostics.configure(path=log)
    diagnostics.record("first")
    diagnostics.configure(path=log)
    diagnostics.record("second")

    assert len(log.read_text(encoding="utf-8").splitlines()) == 2


def test_recent_returns_events_oldest_first():
    diagnostics.record("first")
    diagnostics.record("second")

    assert [e["event"] for e in diagnostics.recent()] == ["first", "second"]


def test_recent_limit_returns_the_tail():
    for name in ("a", "b", "c"):
        diagnostics.record(name)

    assert [e["event"] for e in diagnostics.recent(limit=2)] == ["b", "c"]


def test_drain_clears_the_buffer():
    diagnostics.record("a")

    assert len(diagnostics.drain()) == 1
    assert diagnostics.recent() == ()


def test_ring_buffer_is_bounded():
    for index in range(diagnostics._RING_LIMIT + 50):
        diagnostics.record("noise", index=index)

    assert len(diagnostics.recent()) == diagnostics._RING_LIMIT


def test_configure_closes_the_previous_file_handle(tmp_path: Path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    diagnostics.configure(path=first)
    diagnostics.record("a")
    diagnostics.configure(path=second)
    diagnostics.record("b")

    assert len(first.read_text(encoding="utf-8").splitlines()) == 1
    assert len(second.read_text(encoding="utf-8").splitlines()) == 1


def test_environment_variable_configures_the_file_sink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    log = tmp_path / "from-env.jsonl"
    monkeypatch.setenv("UXA_DIAGNOSTIC_LOG", str(log))
    diagnostics.configure()

    diagnostics.record("role.call.started")

    assert len(log.read_text(encoding="utf-8").splitlines()) == 1


def test_recording_never_raises_on_an_unserializable_extra():
    event = diagnostics.record("role.call.failed", detail=object())

    assert "detail" in event["extras"]
